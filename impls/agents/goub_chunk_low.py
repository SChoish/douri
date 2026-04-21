from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import GCActor, GCValue, Identity, LengthNormalize, MLP


class GOUBChunkLowAgent(flax.struct.PyTreeNode):
    """Chunked low-level controller trained on real offline trajectory segments."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def _actor_observations(self, observations, local_plan_context):
        return jnp.concatenate([observations, local_plan_context], axis=-1)

    def value_loss(self, batch, grad_params):
        (next_v1_t, next_v2_t) = self.network.select('target_value')(batch['next_observations'], batch['value_goals'])
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

        (v1_t, v2_t) = self.network.select('target_value')(batch['observations'], batch['value_goals'])
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
        (v1, v2) = self.network.select('value')(batch['observations'], batch['value_goals'], params=grad_params)

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2
        v = (v1 + v2) / 2

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def chunk_actor_loss(self, batch, grad_params):
        observations = batch['observations']
        future_observations = batch['chunk_future_observations']
        batch_size, horizon, obs_dim = future_observations.shape

        prev_observations = jnp.concatenate([observations[:, None, :], future_observations[:, :-1, :]], axis=1)
        goals = batch['value_goals']
        repeated_goals = jnp.repeat(goals[:, None, :], horizon, axis=1)

        flat_prev = prev_observations.reshape(batch_size * horizon, obs_dim)
        flat_future = future_observations.reshape(batch_size * horizon, obs_dim)
        flat_goals = repeated_goals.reshape(batch_size * horizon, goals.shape[-1])

        prev_v1, prev_v2 = self.network.select('value')(flat_prev, flat_goals)
        next_v1, next_v2 = self.network.select('value')(flat_future, flat_goals)
        prev_v = ((prev_v1 + prev_v2) / 2).reshape(batch_size, horizon)
        next_v = ((next_v1 + next_v2) / 2).reshape(batch_size, horizon)

        adv_steps = next_v - prev_v
        eta = float(self.config['chunk_adv_eta'])
        discounts = jnp.power(
            jnp.full((horizon,), eta, dtype=jnp.float32),
            jnp.arange(horizon, dtype=jnp.float32),
        )
        chunk_adv = (adv_steps * discounts[None, :]).sum(axis=1)

        beta = float(self.config.get('chunk_actor_beta', self.config.get('low_alpha', 3.0)))
        exp_a = jnp.exp(chunk_adv * beta)
        exp_a = jnp.minimum(exp_a, 100.0)

        goal_reps = self.network.select('goal_rep')(
            jnp.concatenate([observations, goals], axis=-1),
            params=grad_params,
        )
        actor_observations = self._actor_observations(observations, batch['local_plan_context'])
        dist = self.network.select('chunk_actor')(
            actor_observations,
            goal_reps,
            goal_encoded=True,
            params=grad_params,
        )
        action_chunks = batch['action_chunks']
        log_prob = dist.log_prob(action_chunks)
        actor_loss = -(exp_a * log_prob).mean()

        action_dim = action_chunks.shape[-1] // horizon
        chunk_mode = dist.mode().reshape(batch_size, horizon, action_dim)
        target_chunks = action_chunks.reshape(batch_size, horizon, action_dim)

        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': chunk_adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'weight_mean': exp_a.mean(),
            'weight_max': exp_a.max(),
            'weight_min': exp_a.min(),
            'mse': jnp.mean((chunk_mode - target_chunks) ** 2),
            'std': jnp.mean(dist.scale_diag),
            'chunk_action_norm': jnp.linalg.norm(target_chunks, axis=-1).mean(),
            'first_action_norm': jnp.linalg.norm(target_chunks[:, 0], axis=-1).mean(),
            'planner_context_norm': jnp.linalg.norm(batch['local_plan_context'], axis=-1).mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        chunk_actor_loss, chunk_actor_info = self.chunk_actor_loss(batch, grad_params)
        for k, v in chunk_actor_info.items():
            info[f'chunk_actor/{k}'] = v

        loss = value_loss + chunk_actor_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')
        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_action_chunk(
        self,
        observations,
        local_plan_context,
        goals,
        seed=None,
        temperature=1.0,
        deterministic=True,
    ):
        squeeze = observations.ndim == 1
        if squeeze:
            observations = observations[None]
            local_plan_context = local_plan_context[None]
            goals = goals[None]

        goal_reps = self.network.select('goal_rep')(jnp.concatenate([observations, goals], axis=-1))
        actor_observations = self._actor_observations(observations, local_plan_context)
        dist = self.network.select('chunk_actor')(
            actor_observations,
            goal_reps,
            goal_encoded=True,
            temperature=temperature,
        )
        action_chunks = dist.mode() if deterministic else dist.sample(seed=seed)
        action_chunks = jnp.clip(action_chunks, -1, 1)

        horizon = int(self.config['chunk_policy_horizon'])
        action_dim = action_chunks.shape[-1] // horizon
        action_chunks = action_chunks.reshape(action_chunks.shape[0], horizon, action_dim)
        if squeeze:
            action_chunks = action_chunks[0]
        return action_chunks

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        ex_local_plan_context,
        config,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        action_dim = ex_actions.shape[-1]
        actor_obs = jnp.concatenate([ex_observations, ex_local_plan_context], axis=-1)

        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            goal_rep_seq = [encoder_module()]
        else:
            goal_rep_seq = []
        goal_rep_seq.append(
            MLP(
                hidden_dims=(*config['value_hidden_dims'], config['rep_dim']),
                activate_final=False,
                layer_norm=config['layer_norm'],
            )
        )
        goal_rep_seq.append(LengthNormalize())
        goal_rep_def = nn.Sequential(goal_rep_seq)

        if config['encoder'] is not None:
            value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            target_value_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
            chunk_actor_encoder_def = GCEncoder(state_encoder=encoder_module(), concat_encoder=goal_rep_def)
        else:
            value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            target_value_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)
            chunk_actor_encoder_def = GCEncoder(state_encoder=Identity(), concat_encoder=goal_rep_def)

        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=value_encoder_def,
        )
        target_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=target_value_encoder_def,
        )
        chunk_actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            state_dependent_std=False,
            const_std=config['const_std'],
            gc_encoder=chunk_actor_encoder_def,
        )

        network_info = dict(
            goal_rep=(goal_rep_def, (jnp.concatenate([ex_observations, ex_goals], axis=-1))),
            value=(value_def, (ex_observations, ex_goals)),
            target_value=(target_value_def, (ex_observations, ex_goals)),
            chunk_actor=(chunk_actor_def, (actor_obs, ex_goals)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)
        params = network.params
        params['modules_target_value'] = params['modules_value']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='goub_chunk_low',
            lr=3e-4,
            batch_size=1024,
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            expectile=0.7,
            low_alpha=3.0,
            chunk_actor_beta=3.0,
            chunk_adv_eta=0.9,
            subgoal_steps=25,
            chunk_context_horizon=8,
            chunk_policy_horizon=4,
            chunk_commit_length=1,
            low_goal_slice=(0, 1),
            chunk_use_relative_context=True,
            low_level_mode='chunk_actor',
            planner_frozen=True,
            rep_dim=10,
            const_std=True,
            discrete=False,
            encoder=ml_collections.config_dict.placeholder(str),
            dataset_class='ChunkHGCDataset',
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
        )
    )
    return config
