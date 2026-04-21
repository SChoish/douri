from __future__ import annotations

from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCValue, Identity, MLP, default_init


class DeterministicGCActor(nn.Module):
    hidden_dims: tuple[int, ...]
    action_dim: int
    layer_norm: bool = True
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None

    @nn.compact
    def __call__(self, observations, goals=None, goal_encoded=False):
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        hidden = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)(inputs)
        actions = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))(hidden)
        return jnp.tanh(actions)


class GOUBPhase2PolicyAgent(flax.struct.PyTreeNode):
    """Frozen GOUB phase1 proposal distillation + offline RL phase2."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, 1.0 - expectile)
        return weight * (diff**2)

    def _critic(self, observations, goals, actions, *, params=None, target=False):
        name = 'target_critic' if target else 'critic'
        return self.network.select(name)(observations, goals, actions, params=params)

    def _min_q(self, observations, goals, actions, *, params=None, target=False):
        q1, q2 = self._critic(observations, goals, actions, params=params, target=target)
        return jnp.minimum(q1, q2), q1, q2

    def _value(self, observations, goals, *, params=None):
        return self.network.select('value')(observations, goals, params=params)

    def _actor_dist(self, observations, goals, *, params=None, temperature=1.0):
        return self.network.select('actor')(observations, goals, params=params, temperature=temperature)

    def _actor_actions(
        self,
        observations,
        goals,
        *,
        params=None,
        seed=None,
        temperature=1.0,
        deterministic=None,
    ):
        rl_algo = str(self.config['rl_algo'])
        if deterministic is None:
            deterministic = (rl_algo != 'iql') or float(temperature) <= 0.0

        if rl_algo == 'iql':
            dist = self._actor_dist(observations, goals, params=params, temperature=max(float(temperature), 1e-6))
            actions = dist.mode() if deterministic else dist.sample(seed=seed)
        else:
            actions = self.network.select('actor')(observations, goals, params=params)
        return jnp.clip(actions, -1.0, 1.0)

    def _repeat_for_candidates(self, observations, goals, candidate_actions):
        batch_size, num_candidates, _ = candidate_actions.shape
        flat_obs = jnp.repeat(observations[:, None, :], num_candidates, axis=1).reshape(-1, observations.shape[-1])
        flat_goals = jnp.repeat(goals[:, None, :], num_candidates, axis=1).reshape(-1, goals.shape[-1])
        flat_actions = candidate_actions.reshape(-1, candidate_actions.shape[-1])
        return batch_size, num_candidates, flat_obs, flat_goals, flat_actions

    def _proposal_scores(self, observations, goals, candidate_actions, *, params=None):
        batch_size, num_candidates, flat_obs, flat_goals, flat_actions = self._repeat_for_candidates(
            observations, goals, candidate_actions
        )
        min_q, _, _ = self._min_q(flat_obs, flat_goals, flat_actions, params=params, target=False)
        return min_q.reshape(batch_size, num_candidates)

    def _distill_actor_loss(self, batch, teacher_info, grad_params):
        observations = batch['observations']
        goals = batch['value_goals']
        candidate_actions = teacher_info['candidate_actions']

        scores = self._proposal_scores(observations, goals, candidate_actions, params=grad_params)
        scores = jax.lax.stop_gradient(scores)
        distill_mode = str(self.config['distill_mode'])
        distill_loss = str(self.config['distill_loss'])
        if distill_mode not in ('argmax', 'softmax'):
            raise ValueError(f'Unsupported distill_mode={distill_mode!r}')
        if distill_loss not in ('mse', 'nll'):
            raise ValueError(f'Unsupported distill_loss={distill_loss!r}')

        if distill_mode == 'softmax':
            if str(self.config['rl_algo']) == 'iql':
                baseline = self._value(observations, goals)
                teacher_scores = scores - baseline[:, None]
            else:
                teacher_scores = scores
            weights = jax.nn.softmax(float(self.config['distill_beta']) * teacher_scores, axis=1)
            weights = jax.lax.stop_gradient(weights)
        else:
            teacher_indices = jnp.argmax(scores, axis=1)
            weights = jax.nn.one_hot(teacher_indices, scores.shape[1], dtype=jnp.float32)
            weights = jax.lax.stop_gradient(weights)

        teacher_action = jnp.sum(candidate_actions * weights[..., None], axis=1)
        teacher_action = jax.lax.stop_gradient(teacher_action)

        rl_algo = str(self.config['rl_algo'])
        if distill_loss == 'nll':
            if rl_algo != 'iql':
                raise ValueError("distill_loss='nll' is only supported for rl_algo='iql'.")
            batch_size, num_candidates, flat_obs, flat_goals, flat_actions = self._repeat_for_candidates(
                observations, goals, candidate_actions
            )
            flat_dist = self._actor_dist(flat_obs, flat_goals, params=grad_params, temperature=1.0)
            flat_log_prob = flat_dist.log_prob(flat_actions).reshape(batch_size, num_candidates)
            actor_loss = -(weights * flat_log_prob).sum(axis=1).mean()
            pred_actions = jnp.clip(self._actor_dist(observations, goals, params=grad_params).mode(), -1.0, 1.0)
            bc_metric = jnp.sum(weights * flat_log_prob, axis=1).mean()
        else:
            pred_actions = self._actor_actions(observations, goals, params=grad_params, deterministic=True)
            actor_loss = jnp.mean((pred_actions - teacher_action) ** 2)
            bc_metric = -actor_loss

        centered = candidate_actions - candidate_actions.mean(axis=1, keepdims=True)
        diversity = jnp.linalg.norm(centered, axis=-1).mean()
        q_max = scores.max(axis=1).mean()
        q_mean = scores.mean(axis=1).mean()

        return actor_loss, {
            'distill/actor_loss': actor_loss,
            'distill/bc_proxy': bc_metric,
            'distill/teacher_action_mse': jnp.mean((pred_actions - teacher_action) ** 2),
            'proposal/q_max': q_max,
            'proposal/q_mean': q_mean,
            'proposal/q_gap': q_max - q_mean,
            'proposal/a_mean_norm': jnp.linalg.norm(teacher_info['a_mean'], axis=-1).mean(),
            'proposal/action_diversity': diversity,
            'proposal/planned_next_norm': jnp.linalg.norm(teacher_info['planned_next_obs'], axis=-1).mean(),
            'proposal/subgoal_norm': jnp.linalg.norm(teacher_info['subgoals'], axis=-1).mean(),
            'proposal/candidate_count': jnp.asarray(candidate_actions.shape[1], dtype=jnp.float32),
        }

    def _iql_critic_loss(self, batch, grad_params):
        next_v = self._value(batch['next_observations'], batch['value_goals'])
        target_q = batch['rewards'] + float(self.config['discount']) * batch['masks'] * next_v
        q1, q2 = self._critic(batch['observations'], batch['value_goals'], batch['actions'], params=grad_params)
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()
        return critic_loss, {
            'critic/loss': critic_loss,
            'critic/q1_mean': q1.mean(),
            'critic/q2_mean': q2.mean(),
            'critic/target_mean': target_q.mean(),
        }

    def _iql_value_loss(self, batch, grad_params):
        q, _, _ = self._min_q(batch['observations'], batch['value_goals'], batch['actions'], target=True)
        v = self._value(batch['observations'], batch['value_goals'], params=grad_params)
        adv = q - v
        value_loss = self.expectile_loss(adv, adv, float(self.config['expectile'])).mean()
        return value_loss, {
            'value/loss': value_loss,
            'value/v_mean': v.mean(),
            'value/adv_mean': adv.mean(),
            'value/adv_max': adv.max(),
        }

    def _iql_actor_loss(self, batch, grad_params):
        q, _, _ = self._min_q(batch['observations'], batch['value_goals'], batch['actions'], target=True)
        v = self._value(batch['observations'], batch['value_goals'])
        adv = q - v
        exp_a = jnp.exp(adv * float(self.config['iql_temperature']))
        exp_a = jnp.minimum(exp_a, 100.0)
        dist = self._actor_dist(batch['observations'], batch['value_goals'], params=grad_params, temperature=1.0)
        log_prob = dist.log_prob(batch['actions'])
        actor_loss = -(exp_a * log_prob).mean()
        return actor_loss, {
            'actor/loss': actor_loss,
            'actor/adv_mean': adv.mean(),
            'actor/bc_log_prob': log_prob.mean(),
            'actor/weight_mean': exp_a.mean(),
            'actor/weight_max': exp_a.max(),
            'actor/mse': jnp.mean((jnp.clip(dist.mode(), -1.0, 1.0) - batch['actions']) ** 2),
            'actor/std': jnp.mean(dist.scale_diag),
        }

    def _td3_critic_loss(self, batch, grad_params, rng):
        noise = jax.random.normal(rng, batch['actions'].shape, dtype=jnp.float32)
        noise = noise * float(self.config['td3_target_policy_noise'])
        noise = jnp.clip(
            noise,
            -float(self.config['td3_target_noise_clip']),
            float(self.config['td3_target_noise_clip']),
        )
        next_actions = self._actor_actions(batch['next_observations'], batch['value_goals'])
        next_actions = jnp.clip(next_actions + noise, -1.0, 1.0)
        target_q, _, _ = self._min_q(batch['next_observations'], batch['value_goals'], next_actions, target=True)
        target = batch['rewards'] + float(self.config['discount']) * batch['masks'] * target_q
        q1, q2 = self._critic(batch['observations'], batch['value_goals'], batch['actions'], params=grad_params)
        critic_loss = ((q1 - target) ** 2 + (q2 - target) ** 2).mean()
        return critic_loss, {
            'critic/loss': critic_loss,
            'critic/q1_mean': q1.mean(),
            'critic/q2_mean': q2.mean(),
            'critic/target_mean': target.mean(),
        }

    def _td3_actor_loss(self, batch, grad_params):
        actions_pi = self._actor_actions(batch['observations'], batch['value_goals'], params=grad_params)
        q_pi, q1_pi, _ = self._min_q(batch['observations'], batch['value_goals'], actions_pi)
        q_abs = jnp.maximum(jnp.mean(jnp.abs(q_pi)), 1e-6)
        lambda_coef = float(self.config['td3bc_alpha']) / q_abs
        bc_loss = jnp.mean((actions_pi - batch['actions']) ** 2)
        actor_loss = bc_loss - lambda_coef * q1_pi.mean()
        return actor_loss, {
            'actor/loss': actor_loss,
            'actor/bc_loss': bc_loss,
            'actor/q_pi_mean': q_pi.mean(),
            'actor/lambda': lambda_coef,
            'actor/mse': bc_loss,
        }

    @jax.jit
    def update_distill(self, batch, teacher_info):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            info = {}
            rl_algo = str(self.config['rl_algo'])
            if rl_algo == 'iql':
                critic_loss, critic_info = self._iql_critic_loss(batch, grad_params)
                value_loss, value_info = self._iql_value_loss(batch, grad_params)
                total_loss = critic_loss + value_loss
                info.update(critic_info)
                info.update(value_info)
            elif rl_algo == 'td3bc':
                critic_loss, critic_info = self._td3_critic_loss(batch, grad_params, rng)
                total_loss = critic_loss
                info.update(critic_info)
            else:
                raise ValueError(f'Unsupported rl_algo={rl_algo!r}')

            distill_actor_loss, distill_info = self._distill_actor_loss(batch, teacher_info, grad_params)
            total_loss = total_loss + float(self.config['distill_actor_weight']) * distill_actor_loss
            info.update(distill_info)
            info['phase2/total_loss'] = total_loss
            info['phase2/stage_id'] = jnp.asarray(0.0, dtype=jnp.float32)
            return total_loss, info

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self._target_update(new_network, 'critic')
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update_finetune(self, batch):
        new_rng, rng = jax.random.split(self.rng)
        actor_update = jnp.asarray(True)
        if str(self.config['rl_algo']) == 'td3bc':
            delay = max(1, int(self.config['td3_policy_delay']))
            actor_update = jnp.asarray(((self.network.step - 1) % delay) == 0)

        def loss_fn(grad_params):
            info = {}
            rl_algo = str(self.config['rl_algo'])
            if rl_algo == 'iql':
                critic_loss, critic_info = self._iql_critic_loss(batch, grad_params)
                value_loss, value_info = self._iql_value_loss(batch, grad_params)
                actor_loss, actor_info = self._iql_actor_loss(batch, grad_params)
                total_loss = critic_loss + value_loss + actor_loss
                info.update(critic_info)
                info.update(value_info)
                info.update(actor_info)
            elif rl_algo == 'td3bc':
                critic_loss, critic_info = self._td3_critic_loss(batch, grad_params, rng)
                total_loss = critic_loss
                info.update(critic_info)

                def _actor_on(_):
                    return self._td3_actor_loss(batch, grad_params)

                def _actor_off(_):
                    zero = jnp.array(0.0, dtype=jnp.float32)
                    return zero, {
                        'actor/loss': zero,
                        'actor/bc_loss': zero,
                        'actor/q_pi_mean': zero,
                        'actor/lambda': zero,
                        'actor/mse': zero,
                    }

                actor_loss, actor_info = jax.lax.cond(actor_update, _actor_on, _actor_off, operand=None)
                total_loss = total_loss + actor_loss
                info.update(actor_info)
            else:
                raise ValueError(f'Unsupported rl_algo={rl_algo!r}')

            info['actor/update_applied'] = actor_update.astype(jnp.float32)
            info['phase2/total_loss'] = total_loss
            info['phase2/stage_id'] = jnp.asarray(1.0, dtype=jnp.float32)
            return total_loss, info

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self._target_update(new_network, 'critic')
        return self.replace(network=new_network, rng=new_rng), info

    def _target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * float(self.config['tau']) + tp * (1.0 - float(self.config['tau'])),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    def sample_actions(self, observations, goals, seed=None, temperature=1.0):
        """Action sampling for rollout/eval.

        Intentionally **not** ``@jax.jit``: ``utils.evaluation.evaluate`` may pass
        ``temperature`` as a traced/array-like value for TD3-style agents, and a
        jit boundary would force Python control flow on ``temperature``.
        """
        observations = jnp.asarray(observations, dtype=jnp.float32)
        goals = jnp.asarray(goals, dtype=jnp.float32)
        rl_algo = str(self.config['rl_algo'])
        if rl_algo == 'iql':
            temp0 = float(jnp.asarray(temperature).reshape(()))
            if temp0 <= 0.0:
                actions = self._actor_actions(observations, goals, deterministic=True)
            else:
                actions = self._actor_actions(
                    observations,
                    goals,
                    deterministic=False,
                    seed=seed,
                    temperature=float(temp0),
                )
        else:
            actions = self._actor_actions(observations, goals, deterministic=True)
        return jnp.clip(actions, -1.0, 1.0)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rl_algo = str(config['rl_algo'])
        if rl_algo not in ('iql', 'td3bc'):
            raise ValueError(f"agent.rl_algo must be 'iql' or 'td3bc', got {rl_algo!r}")
        if rl_algo == 'td3bc' and str(config['distill_loss']) == 'nll':
            raise ValueError("TD3+BC phase2 does not support distill_loss='nll'; use 'mse'.")

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        action_dim = ex_actions.shape[-1]

        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            actor_encoder_def = GCEncoder(state_encoder=encoder_module(), goal_encoder=encoder_module())
            critic_encoder_def = GCEncoder(state_encoder=encoder_module(), goal_encoder=encoder_module())
            target_critic_encoder_def = GCEncoder(state_encoder=encoder_module(), goal_encoder=encoder_module())
            value_encoder_def = GCEncoder(state_encoder=encoder_module(), goal_encoder=encoder_module())
        else:
            actor_encoder_def = GCEncoder(state_encoder=Identity(), goal_encoder=Identity())
            critic_encoder_def = GCEncoder(state_encoder=Identity(), goal_encoder=Identity())
            target_critic_encoder_def = GCEncoder(state_encoder=Identity(), goal_encoder=Identity())
            value_encoder_def = GCEncoder(state_encoder=Identity(), goal_encoder=Identity())

        if rl_algo == 'iql':
            actor_def = GCActor(
                hidden_dims=tuple(config['actor_hidden_dims']),
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=bool(config['const_std']),
                gc_encoder=actor_encoder_def,
            )
        else:
            actor_def = DeterministicGCActor(
                hidden_dims=tuple(config['actor_hidden_dims']),
                action_dim=action_dim,
                layer_norm=bool(config['layer_norm']),
                gc_encoder=actor_encoder_def,
            )

        critic_def = GCValue(
            hidden_dims=tuple(config['critic_hidden_dims']),
            layer_norm=bool(config['layer_norm']),
            ensemble=True,
            gc_encoder=critic_encoder_def,
        )
        target_critic_def = GCValue(
            hidden_dims=tuple(config['critic_hidden_dims']),
            layer_norm=bool(config['layer_norm']),
            ensemble=True,
            gc_encoder=target_critic_encoder_def,
        )

        network_info = {
            'actor': (actor_def, (ex_observations, ex_goals)),
            'critic': (critic_def, (ex_observations, ex_goals, ex_actions)),
            'target_critic': (target_critic_def, (ex_observations, ex_goals, ex_actions)),
        }
        if rl_algo == 'iql':
            value_def = GCValue(
                hidden_dims=tuple(config['value_hidden_dims']),
                layer_norm=bool(config['layer_norm']),
                ensemble=False,
                gc_encoder=value_encoder_def,
            )
            network_info['value'] = (value_def, (ex_observations, ex_goals))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_params = network_def.init(init_rng, **network_args)['params']
        network_params['modules_target_critic'] = network_params['modules_critic']
        network_tx = cls._build_optimizer(network_params, config)
        network = TrainState.create(network_def, network_params, tx=network_tx)
        return cls(rng=rng, network=network, config=flax.core.FrozenDict(**config))

    @staticmethod
    def _build_optimizer(network_params, config):
        label_map = {}
        for key, value in network_params.items():
            if key == 'modules_actor':
                label = 'actor'
            elif key == 'modules_critic':
                label = 'critic'
            elif key == 'modules_value':
                label = 'value'
            else:
                label = 'frozen'
            label_map[key] = jax.tree_util.tree_map(lambda _: label, value)

        txs = {
            'actor': optax.adam(float(config['actor_lr'])),
            'critic': optax.adam(float(config['critic_lr'])),
            'value': optax.adam(float(config['value_lr'])),
            'frozen': optax.set_to_zero(),
        }
        return optax.multi_transform(txs, label_map)


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='goub_phase2_policy',
            rl_algo='iql',
            batch_size=1024,
            actor_hidden_dims=(512, 512, 512),
            critic_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            expectile=0.7,
            iql_temperature=3.0,
            td3bc_alpha=2.5,
            td3_policy_delay=2,
            td3_target_policy_noise=0.2,
            td3_target_noise_clip=0.5,
            actor_lr=3e-4,
            critic_lr=3e-4,
            value_lr=3e-4,
            num_action_samples=16,
            action_noise_std=0.05,
            include_mean_action=True,
            include_dataset_action=True,
            planner_noise_scale=0.0,
            num_planner_samples=25,
            distill_mode='argmax',
            distill_beta=3.0,
            distill_loss='mse',
            distill_actor_weight=1.0,
            dataset_class='HGCDataset',
            subgoal_steps=25,
            value_p_curgoal=0.2,
            value_p_trajgoal=0.5,
            value_p_randomgoal=0.3,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=True,
            p_aug=0.0,
            frame_stack=ml_collections.config_dict.placeholder(int),
            encoder=ml_collections.config_dict.placeholder(str),
            const_std=True,
            discrete=False,
        )
    )
