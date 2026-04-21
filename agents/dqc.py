"""DQC-style chunk critic + distilled partial critic + flow behavior policy."""

from __future__ import annotations

from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, GCValue, MLP


def _safe_logit(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    x = jnp.clip(x, eps, 1.0 - eps)
    return jnp.log(x) - jnp.log1p(-x)


class ScalarValueNet(nn.Module):
    """Scalar-logit value network V(s, g)."""

    hidden_dims: Sequence[int]
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations: jnp.ndarray, goals: jnp.ndarray | None = None) -> jnp.ndarray:
        xs = [observations]
        if goals is not None:
            xs.append(goals)
        x = jnp.concatenate(xs, axis=-1)
        return MLP((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)(x).squeeze(-1)


class BinaryChunkCritic(nn.Module):
    """Binary-logit chunk/action critic with configurable ensemble size."""

    hidden_dims: Sequence[int]
    num_qs: int
    layer_norm: bool = True

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray | None = None,
        actions_flat: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        xs = [observations]
        if goals is not None:
            xs.append(goals)
        if actions_flat is not None:
            xs.append(actions_flat)
        x = jnp.concatenate(xs, axis=-1)
        h = MLP(tuple(self.hidden_dims), activate_final=True, layer_norm=self.layer_norm)(x)
        logits = [nn.Dense(1, name=f'q_head_{i}')(h).squeeze(-1) for i in range(int(self.num_qs))]
        return jnp.stack(logits, axis=0)


class DQCAgent(flax.struct.PyTreeNode):
    """DQC value-learning and best-of-N policy extraction."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _agg_q(self, qs: jnp.ndarray) -> jnp.ndarray:
        q_agg = str(self.config['q_agg']).lower()
        if q_agg == 'mean':
            return jnp.mean(qs, axis=0)
        if q_agg == 'min':
            return jnp.min(qs, axis=0)
        raise ValueError(f"q_agg must be 'mean' or 'min', got {q_agg!r}")

    def _valid_mask(self, batch: dict) -> jnp.ndarray:
        valids = batch.get('valids', None)
        if valids is None:
            return jnp.ones((batch['observations'].shape[0],), dtype=jnp.float32)
        valids = jnp.asarray(valids, dtype=jnp.float32)
        if valids.ndim == 1:
            return valids
        return valids.reshape(valids.shape[0], -1)[:, -1]

    def _weighted_mean(self, values: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
        denom = jnp.maximum(jnp.sum(weights), 1e-6)
        return jnp.sum(values * weights) / denom

    def _chunk_critic_loss(self, batch: dict, grad_params: dict) -> tuple[jnp.ndarray, dict]:
        goals = batch.get('value_goals', None)
        backup_h = jnp.asarray(batch['high_value_backup_horizon'], dtype=jnp.float32)
        next_v_logit = self.network.select('value')(
            batch['high_value_next_observations'],
            goals,
            params=grad_params,
        )
        next_v = jax.nn.sigmoid(next_v_logit)
        target_v = jnp.asarray(batch['high_value_rewards'], dtype=jnp.float32) + jnp.power(
            float(self.config['discount']), backup_h
        ) * jnp.asarray(batch['high_value_masks'], dtype=jnp.float32) * next_v
        target_v = jnp.clip(target_v, 0.0, 1.0)

        q_logits = self.network.select('chunk_critic')(
            batch['observations'],
            goals,
            batch['high_value_action_chunks'],
            params=grad_params,
        )
        bce = optax.sigmoid_binary_cross_entropy(q_logits, target_v[None, :])
        loss = jnp.mean(bce)
        q = jax.nn.sigmoid(q_logits)
        return loss, {
            'chunk_critic/critic_loss': loss,
            'chunk_critic/q_mean': q.mean(),
            'chunk_critic/q_max': q.max(),
            'chunk_critic/q_min': q.min(),
            'chunk_critic/q_logit_mean': q_logits.mean(),
            'chunk_critic/q_logit_max': q_logits.max(),
            'chunk_critic/q_logit_min': q_logits.min(),
        }

    def _partial_critic_loss(self, batch: dict, grad_params: dict) -> tuple[jnp.ndarray, dict]:
        goals = batch.get('value_goals', None)
        valid_mask = self._valid_mask(batch)

        if bool(self.config['use_chunk_critic']):
            target_logits = self.network.select('chunk_critic')(
                batch['observations'],
                goals,
                batch['high_value_action_chunks'],
            )
            target_v = self._agg_q(jax.nn.sigmoid(target_logits))
        else:
            backup_h = jnp.asarray(batch['high_value_backup_horizon'], dtype=jnp.float32)
            next_v_logit = self.network.select('value')(
                batch['high_value_next_observations'],
                goals,
                params=grad_params,
            )
            next_v = jax.nn.sigmoid(next_v_logit)
            target_v = jnp.asarray(batch['high_value_rewards'], dtype=jnp.float32) + jnp.power(
                float(self.config['discount']), backup_h
            ) * jnp.asarray(batch['high_value_masks'], dtype=jnp.float32) * next_v
        target_v = jnp.clip(jax.lax.stop_gradient(target_v), 0.0, 1.0)

        q_part_logits = self.network.select('action_critic')(
            batch['observations'],
            goals,
            batch['partial_action_chunks'],
            params=grad_params,
        )
        q_part = jax.nn.sigmoid(q_part_logits)
        q_part_agg = self._agg_q(q_part)
        kappa_d = float(self.config['kappa_d'])
        weight_d = jnp.where(target_v >= q_part_agg, kappa_d, 1.0 - kappa_d)
        method = str(self.config['distill_method']).lower()
        if method == 'expectile':
            per = jnp.mean(optax.sigmoid_binary_cross_entropy(q_part_logits, target_v[None, :]), axis=0)
        elif method == 'quantile':
            target_v_logit = _safe_logit(target_v)
            per = jnp.mean(jnp.abs(q_part_logits - target_v_logit[None, :]), axis=0)
        else:
            raise ValueError(f"distill_method must be 'expectile' or 'quantile', got {method!r}")

        loss = self._weighted_mean(weight_d * per, valid_mask)
        return loss, {
            'action_critic/distill_loss': loss,
            'action_critic/q_part_mean': q_part_agg.mean(),
            'action_critic/target_v_mean': target_v.mean(),
            'action_critic/weight_d_mean': weight_d.mean(),
        }

    def _value_loss(self, batch: dict, grad_params: dict) -> tuple[jnp.ndarray, dict]:
        goals = batch.get('value_goals', None)
        valid_mask = self._valid_mask(batch)

        ex_qs = self.network.select('target_action_critic')(
            batch['observations'],
            goals,
            batch['partial_action_chunks'],
        )
        ex_qs = jax.nn.sigmoid(ex_qs)
        ex_q = self._agg_q(ex_qs)

        v_logit = self.network.select('value')(batch['observations'], goals, params=grad_params)
        v = jax.nn.sigmoid(v_logit)

        kappa_b = float(self.config['kappa_b'])
        weight_b = jnp.where(ex_q >= v, kappa_b, 1.0 - kappa_b)
        backup_type = str(self.config['implicit_backup_type']).lower()
        if backup_type == 'expectile':
            per = optax.sigmoid_binary_cross_entropy(v_logit, jax.lax.stop_gradient(ex_q))
        elif backup_type == 'quantile':
            ex_q_logit = _safe_logit(jax.lax.stop_gradient(ex_q))
            per = jnp.abs(v_logit - ex_q_logit)
        else:
            raise ValueError(f"implicit_backup_type must be 'expectile' or 'quantile', got {backup_type!r}")
        loss = self._weighted_mean(weight_b * per, valid_mask)
        adv = ex_q - v
        return loss, {
            'action_critic/value_loss': loss,
            'action_critic/adv': adv.mean(),
            'action_critic/v_mean': v.mean(),
            'action_critic/v_max': v.max(),
            'action_critic/v_min': v.min(),
            'action_critic/weight_b_mean': weight_b.mean(),
        }

    def _actor_flow_loss(self, batch: dict, grad_params: dict, rng: jax.Array) -> tuple[jnp.ndarray, dict]:
        goals = batch.get('value_goals', None)
        valid_mask = self._valid_mask(batch)
        x1 = batch['partial_action_chunks']
        x0 = jax.random.normal(rng, x1.shape, dtype=jnp.float32)
        t = jax.random.uniform(rng, (x1.shape[0], 1), minval=0.0, maxval=1.0)
        xt = (1.0 - t) * x0 + t * x1
        vel = x1 - x0
        pred = self.network.select('actor_bc')(batch['observations'], goals, xt, t, params=grad_params)
        per = jnp.mean((pred - vel) ** 2, axis=-1)
        loss = self._weighted_mean(per, valid_mask)
        return loss, {
            'actor_bc/loss': loss,
            'actor_bc/pred_norm': jnp.linalg.norm(pred, axis=-1).mean(),
            'actor_bc/target_norm': jnp.linalg.norm(vel, axis=-1).mean(),
        }

    @jax.jit
    def total_loss(self, batch: dict, grad_params: dict, rng: jax.Array):
        batch = jax.tree_util.tree_map(lambda x: jnp.asarray(x), batch)
        info = {}
        total = jnp.asarray(0.0, dtype=jnp.float32)

        if bool(self.config['use_chunk_critic']):
            chunk_loss, chunk_info = self._chunk_critic_loss(batch, grad_params)
            total = total + chunk_loss
            info.update(chunk_info)
        else:
            info['chunk_critic/critic_loss'] = jnp.asarray(0.0, dtype=jnp.float32)

        part_loss, part_info = self._partial_critic_loss(batch, grad_params)
        value_loss, value_info = self._value_loss(batch, grad_params)
        actor_loss, actor_info = self._actor_flow_loss(batch, grad_params, rng)

        total = total + part_loss + value_loss + actor_loss
        info.update(part_info)
        info.update(value_info)
        info.update(actor_info)
        info['dqc/total_loss'] = total
        return total, info

    @staticmethod
    def _ema_target_action_critic(network: TrainState, tau: float) -> TrainState:
        updated = dict(network.params)
        updated['modules_target_action_critic'] = jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1.0 - tau),
            updated['modules_action_critic'],
            updated['modules_target_action_critic'],
        )
        return network.replace(params=updated)

    @jax.jit
    def update(self, batch: dict):
        new_rng, loss_rng = jax.random.split(self.rng)

        def loss_fn(params):
            return self.total_loss(batch, params, rng=loss_rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        new_network = self._ema_target_action_critic(new_network, float(self.config['tau']))
        return self.replace(rng=new_rng, network=new_network), info

    def compute_flow_actions(self, observations, goals, noises):
        flow_steps = int(self.config['flow_steps'])
        if flow_steps < 1:
            raise ValueError('flow_steps must be >= 1')
        bsz, n, flat_dim = noises.shape
        obs_dim = observations.shape[-1]
        obs_rep = jnp.repeat(observations[:, None, :], n, axis=1).reshape(bsz * n, obs_dim)
        if goals is not None:
            goal_dim = goals.shape[-1]
            goal_rep = jnp.repeat(goals[:, None, :], n, axis=1).reshape(bsz * n, goal_dim)
        else:
            goal_rep = None
        actions = noises.reshape(bsz * n, flat_dim)
        dt = 1.0 / float(flow_steps)
        for i in range(flow_steps):
            t = jnp.full((bsz * n, 1), float(i) * dt, dtype=jnp.float32)
            vel = self.network.select('actor_bc')(obs_rep, goal_rep, actions, t)
            actions = jnp.clip(actions + vel * dt, -1.0, 1.0)
        return actions.reshape(bsz, n, flat_dim)

    def apply_bfn(self, observations, goals=None, seed=None, best_of_n=None):
        if seed is None:
            seed = self.rng
        bsz = observations.shape[0]
        n = int(best_of_n if best_of_n is not None else self.config['best_of_n'])
        flat_dim = int(self.config['policy_chunk_size']) * int(self.config['action_dim'])
        noises = jax.random.normal(seed, (bsz, n, flat_dim), dtype=jnp.float32)
        candidates = self.compute_flow_actions(observations, goals, noises)  # [B, N, P]

        obs_rep = jnp.repeat(observations[:, None, :], n, axis=1).reshape(bsz * n, observations.shape[-1])
        if goals is not None:
            goals_rep = jnp.repeat(goals[:, None, :], n, axis=1).reshape(bsz * n, goals.shape[-1])
        else:
            goals_rep = None
        cand_flat = candidates.reshape(bsz * n, flat_dim)
        logits = self.network.select('action_critic')(obs_rep, goals_rep, cand_flat)
        qs = jax.nn.sigmoid(logits).reshape(logits.shape[0], bsz, n)
        scores = self._agg_q(qs)
        best_idx = jnp.argmax(scores, axis=1)
        chosen = candidates[jnp.arange(bsz), best_idx]
        return chosen

    def sample_actions(self, observations, goals=None, seed=None, best_of_n=None):
        observations = jnp.asarray(observations, dtype=jnp.float32)
        squeeze = observations.ndim == 1
        if squeeze:
            observations = observations[None]
            goals = None if goals is None else jnp.asarray(goals, dtype=jnp.float32)[None]
        else:
            goals = None if goals is None else jnp.asarray(goals, dtype=jnp.float32)

        if seed is None:
            seed = self.rng
        chosen_flat = self.apply_bfn(observations, goals=goals, seed=seed, best_of_n=best_of_n)
        ha = int(self.config['policy_chunk_size'])
        ad = int(self.config['action_dim'])
        chosen_chunk = chosen_flat.reshape(chosen_flat.shape[0], ha, ad)
        if squeeze:
            chosen_chunk = chosen_chunk[0]
        return chosen_chunk

    def sample_first_action(self, observations, goals=None, seed=None, best_of_n=None):
        chunk = self.sample_actions(observations, goals=goals, seed=seed, best_of_n=best_of_n)
        return chunk[0] if chunk.ndim == 2 else chunk[:, 0]

    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations: np.ndarray,
        ex_full_action_chunks: np.ndarray,
        ex_partial_action_chunks: np.ndarray,
        config: dict,
    ):
        rng = jax.random.PRNGKey(int(seed))
        rng, init_rng = jax.random.split(rng)

        ex_obs = jnp.asarray(ex_observations, dtype=jnp.float32)
        ex_full = jnp.asarray(ex_full_action_chunks, dtype=jnp.float32)
        ex_part = jnp.asarray(ex_partial_action_chunks, dtype=jnp.float32)
        ex_goal = ex_obs
        ex_t = jnp.zeros((ex_obs.shape[0], 1), dtype=jnp.float32)

        value_def = ScalarValueNet(
            hidden_dims=tuple(config['value_hidden_dims']),
            layer_norm=bool(config['layer_norm']),
        )
        chunk_critic_def = BinaryChunkCritic(
            hidden_dims=tuple(config['value_hidden_dims']),
            num_qs=int(config['num_qs']),
            layer_norm=bool(config['layer_norm']),
        )
        action_critic_def = BinaryChunkCritic(
            hidden_dims=tuple(config['value_hidden_dims']),
            num_qs=int(config['num_qs']),
            layer_norm=bool(config['layer_norm']),
        )
        target_action_critic_def = BinaryChunkCritic(
            hidden_dims=tuple(config['value_hidden_dims']),
            num_qs=int(config['num_qs']),
            layer_norm=bool(config['layer_norm']),
        )
        actor_bc_def = ActorVectorField(
            hidden_dims=tuple(config['actor_hidden_dims']),
            action_dim=int(config['policy_chunk_size']) * int(config['action_dim']),
            layer_norm=bool(config['actor_layer_norm']),
        )

        network_info = {
            'chunk_critic': (chunk_critic_def, (ex_obs, ex_goal, ex_full)),
            'action_critic': (action_critic_def, (ex_obs, ex_goal, ex_part)),
            'target_action_critic': (target_action_critic_def, (ex_obs, ex_goal, ex_part)),
            'value': (value_def, (ex_obs, ex_goal)),
            'actor_bc': (actor_bc_def, (ex_obs, ex_goal, ex_part, ex_t)),
        }
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_params = network_def.init(init_rng, **network_args)['params']
        network_params['modules_target_action_critic'] = network_params['modules_action_critic']
        network_tx = optax.adam(float(config['lr']))
        network = TrainState.create(network_def, network_params, tx=network_tx)
        return cls(rng=rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='dqc',
            lr=3e-4,
            batch_size=256,
            discount=0.99,
            tau=0.005,
            num_qs=2,
            q_agg='min',
            flow_steps=16,
            best_of_n=64,
            use_chunk_critic=True,
            backup_horizon=8,
            policy_chunk_size=4,
            distill_method='expectile',
            kappa_d=0.7,
            implicit_backup_type='expectile',
            kappa_b=0.7,
            actor_hidden_dims=(256, 256, 256),
            value_hidden_dims=(256, 256, 256),
            layer_norm=True,
            actor_layer_norm=True,
            dataset_class='DQCActionSeqDataset',
            frame_stack=None,
            p_aug=0.0,
            action_dim=2,
            value_p_curgoal=0.2,
            value_p_trajgoal=0.5,
            value_p_randomgoal=0.3,
            value_geom_sample=True,
            gc_negative=True,
        )
    )
