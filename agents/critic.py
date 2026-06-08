"""Critic agent (chunk critic + partial critic + scalar value).

Single-file critic module. Networks, agent, helpers, and default config live here.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.goal_representation import assert_phi_goal_obs_indices, goal_representation, normalize_phi_goal_obs_indices
from utils.networks import MLP


_VALID_CRITIC_TYPES = ('dqc', 'iql', 'trl')
_TRL_ALIASES = (
    'trl',
    'chunk_trl',
    'direct_chunk_trl',
    'state_transitive',
    'transitive_v_local_q',
    'transitivechunkrl',
)


def _canonicalize_critic_config(config: dict) -> tuple[str, str, bool]:
    """Normalize critic mode aliases in-place and return mode flags."""
    critic_type = str(config.get('critic_type', 'dqc')).lower()
    algorithm = str(config.get('algorithm', '')).lower()
    if critic_type in _TRL_ALIASES or algorithm in _TRL_ALIASES:
        critic_type = 'trl'
    if critic_type not in _VALID_CRITIC_TYPES:
        raise ValueError(
            f"critic_type must be one of {', '.join(repr(x) for x in _VALID_CRITIC_TYPES)}, got {critic_type!r}"
        )
    is_trl = critic_type == 'trl'
    if is_trl:
        config['critic_type'] = 'trl'
        config['algorithm'] = 'trl'
        config['use_chunk_critic'] = False
        if config.get('subgoal_value_bonus_type', None) in (None, ''):
            config['subgoal_value_bonus_type'] = 'transitive_product'
        config['subgoal_value_log_eps'] = float(config.get('subgoal_value_log_eps', 1e-6))
        if config.get('subgoal_value_ratio_eps', None) is None:
            config['subgoal_value_ratio_eps'] = 1e-3
        config['subgoal_value_ratio_clip'] = float(config.get('subgoal_value_ratio_clip', 5.0))
    elif critic_type == 'iql' and bool(config.get('use_chunk_critic', False)):
        config['use_chunk_critic'] = False
    else:
        config['critic_type'] = critic_type
    if config.get('subgoal_value_bonus_type', None) in (None, ''):
        config['subgoal_value_bonus_type'] = 'single_value'
    if config.get('subgoal_value_ratio_eps', None) is None:
        config['subgoal_value_ratio_eps'] = 1e-6
    return str(config['critic_type']), str(config.get('algorithm', algorithm)), is_trl


def _safe_logit(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    x = jnp.clip(x, eps, 1.0 - eps)
    return jnp.log(x) - jnp.log1p(-x)


def _bce_expectile_loss(logits: jnp.ndarray, targets: jnp.ndarray, tau: float) -> jnp.ndarray:
    probs = jax.nn.sigmoid(logits)
    weight = jnp.where(targets >= probs, float(tau), 1.0 - float(tau))
    return weight * optax.sigmoid_binary_cross_entropy(logits, targets)


def _is_trl_type(critic_type: str, algorithm: str = '') -> bool:
    critic_type = str(critic_type).lower()
    algorithm = str(algorithm).lower()
    return critic_type in _TRL_ALIASES or algorithm in _TRL_ALIASES


class ScalarValueNet(nn.Module):
    hidden_dims: Sequence[int]
    layer_norm: bool = True
    goal_representation: str = 'full'
    phi_goal_obs_indices: tuple[int, ...] = ()
    env_name: str = ''

    @nn.compact
    def __call__(self, observations: jnp.ndarray, goals: jnp.ndarray | None = None) -> jnp.ndarray:
        xs = [observations]
        if goals is not None:
            xs.append(
                goal_representation(
                    goals,
                    self.goal_representation,
                    self.phi_goal_obs_indices,
                    env_name=self.env_name,
                )
            )
        x = jnp.concatenate(xs, axis=-1)
        return MLP((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)(x).squeeze(-1)


class BinaryChunkCritic(nn.Module):
    hidden_dims: Sequence[int]
    num_qs: int
    layer_norm: bool = True
    goal_representation: str = 'full'
    phi_goal_obs_indices: tuple[int, ...] = ()
    env_name: str = ''

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray | None = None,
        actions_flat: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        xs = [observations]
        if goals is not None:
            xs.append(
                goal_representation(
                    goals,
                    self.goal_representation,
                    self.phi_goal_obs_indices,
                    env_name=self.env_name,
                )
            )
        if actions_flat is not None:
            actions_flat = jnp.asarray(actions_flat)
            if actions_flat.ndim > 2:
                actions_flat = actions_flat.reshape((actions_flat.shape[0], -1))
            xs.append(actions_flat)
        x = jnp.concatenate(xs, axis=-1)
        h = MLP(tuple(self.hidden_dims), activate_final=True, layer_norm=self.layer_norm)(x)
        logits = [nn.Dense(1, name=f'q_head_{i}')(h).squeeze(-1) for i in range(int(self.num_qs))]
        return jnp.stack(logits, axis=0)


class CriticAgent(flax.struct.PyTreeNode):
    """Chunk critic + action critic + scalar value stack."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def aggregate_ensemble_q(self, qs: jnp.ndarray) -> jnp.ndarray:
        q_agg = str(self.config['q_agg']).lower()
        if q_agg == 'mean':
            return jnp.mean(qs, axis=0)
        if q_agg == 'min':
            return jnp.min(qs, axis=0)
        raise ValueError(f"q_agg must be 'mean' or 'min', got {q_agg!r}")

    def _critic_type(self) -> str:
        return str(self.config.get('critic_type', 'dqc')).lower()

    def _is_trl(self) -> bool:
        return _is_trl_type(
            self._critic_type(),
            str(self.config.get('algorithm', '')),
        )

    def _has_chunk_critic(self) -> bool:
        """True iff ``critic_type='dqc'`` and ``use_chunk_critic`` is enabled."""
        return (
            (not self._is_trl())
            and self._critic_type() == 'dqc'
            and bool(self.config.get('use_chunk_critic', False))
        )

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

    def chunk_critic_loss(self, batch: dict, grad_params: dict) -> tuple[jnp.ndarray, dict]:
        goals = batch.get('value_goals', None)
        full_chunk_horizon = jnp.asarray(batch['full_chunk_horizon'], dtype=jnp.float32)
        next_v_logit = self.network.select('value')(batch['full_chunk_next_observations'], goals, params=grad_params)
        next_v = jax.nn.sigmoid(next_v_logit)
        target_v = jnp.asarray(batch['full_chunk_rewards'], dtype=jnp.float32) + jnp.power(
            float(self.config['discount']), full_chunk_horizon
        ) * jnp.asarray(batch['full_chunk_masks'], dtype=jnp.float32) * next_v
        target_v = jnp.clip(target_v, 0.0, 1.0)

        q_logits = self.network.select('chunk_critic')(
            batch['observations'], goals, batch['full_chunk_actions'], params=grad_params
        )
        loss = jnp.mean(optax.sigmoid_binary_cross_entropy(q_logits, target_v[None, :]))
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

    def partial_critic_loss(self, batch: dict, grad_params: dict) -> tuple[jnp.ndarray, dict]:
        goals = batch.get('value_goals', None)
        valid_mask = self._valid_mask(batch)
        critic_type = self._critic_type()

        if self._has_chunk_critic():
            target_logits = self.network.select('target_chunk_critic')(
                batch['observations'], goals, batch['full_chunk_actions']
            )
            target_v = self.aggregate_ensemble_q(jax.nn.sigmoid(target_logits))
        else:
            if critic_type == 'iql':
                horizon_arr = jnp.asarray(batch['action_chunk_horizon_per_sample'], dtype=jnp.float32)
                next_obs = batch['action_chunk_next_observations']
                rewards = batch['action_chunk_rewards']
                masks = batch['action_chunk_masks']
            else:
                horizon_arr = jnp.asarray(batch['full_chunk_horizon'], dtype=jnp.float32)
                next_obs = batch['full_chunk_next_observations']
                rewards = batch['full_chunk_rewards']
                masks = batch['full_chunk_masks']
            next_v_logit = self.network.select('value')(next_obs, goals, params=grad_params)
            next_v = jax.nn.sigmoid(next_v_logit)
            target_v = jnp.asarray(rewards, dtype=jnp.float32) + jnp.power(
                float(self.config['discount']), horizon_arr
            ) * jnp.asarray(masks, dtype=jnp.float32) * next_v
        target_v = jnp.clip(jax.lax.stop_gradient(target_v), 0.0, 1.0)

        q_logits = self.network.select('action_critic')(
            batch['observations'], goals, batch['action_chunk_actions'], params=grad_params
        )
        q_part = jax.nn.sigmoid(q_logits)
        q_part_agg = self.aggregate_ensemble_q(q_part)
        weight_d = jnp.where(target_v >= q_part_agg, float(self.config['kappa_d']), 1.0 - float(self.config['kappa_d']))

        method = str(self.config['distill_method']).lower()
        if method == 'expectile':
            per = jnp.mean(optax.sigmoid_binary_cross_entropy(q_logits, target_v[None, :]), axis=0)
        elif method == 'quantile':
            per = jnp.mean(jnp.abs(q_logits - _safe_logit(target_v)[None, :]), axis=0)
        else:
            raise ValueError(f"distill_method must be 'expectile' or 'quantile', got {method!r}")
        loss = self._weighted_mean(weight_d * per, valid_mask)
        return loss, {
            'action_critic/distill_loss': loss,
            'action_critic/q_part_mean': q_part_agg.mean(),
            'action_critic/target_v_mean': target_v.mean(),
            'action_critic/weight_d_mean': weight_d.mean(),
        }

    def value_loss(self, batch: dict, grad_params: dict) -> tuple[jnp.ndarray, dict]:
        goals = batch.get('value_goals', None)
        valid_mask = self._valid_mask(batch)
        ex_qs = self.network.select('target_action_critic')(batch['observations'], goals, batch['action_chunk_actions'])
        ex_q = self.aggregate_ensemble_q(jax.nn.sigmoid(ex_qs))
        v_logit = self.network.select('value')(batch['observations'], goals, params=grad_params)
        v = jax.nn.sigmoid(v_logit)
        weight_b = jnp.where(ex_q >= v, float(self.config['kappa_b']), 1.0 - float(self.config['kappa_b']))
        btype = str(self.config['implicit_backup_type']).lower()
        if btype == 'expectile':
            per = optax.sigmoid_binary_cross_entropy(v_logit, jax.lax.stop_gradient(ex_q))
        elif btype == 'quantile':
            per = jnp.abs(v_logit - _safe_logit(jax.lax.stop_gradient(ex_q)))
        else:
            raise ValueError(f"implicit_backup_type must be 'expectile' or 'quantile', got {btype!r}")
        loss = self._weighted_mean(weight_b * per, valid_mask)
        adv = ex_q - v
        return loss, {
            'action_critic/value_loss': loss,
            'action_critic/adv': adv.mean(),
            'action_critic/v_mean': v.mean(),
            'action_critic/v_max': v.max(),
            'action_critic/v_min': v.min(),
        }

    def trl_loss(self, batch: dict, grad_params: dict) -> tuple[jnp.ndarray, dict]:
        """State-pair transitive V plus local subgoal-conditioned action Q."""
        valid_mask = self._valid_mask(batch)
        eps = float(self.config.get('q_value_eps', 1e-6))
        discount = float(self.config['discount'])
        tau_v = float(self.config.get('tau_v', 0.9))
        lambda_v_self = float(self.config.get('lambda_v_self', 1.0))
        lambda_v_base = float(self.config.get('lambda_v_base', 1.0))
        lambda_v_tri = float(self.config.get('lambda_v_tri', 1.0))
        lambda_q_local = float(self.config.get('lambda_q_local', 1.0))
        value_base_horizon = float(self.config.get('value_base_horizon', 5.0))

        observations = batch['observations']
        goals = batch['value_goals']
        split_obs = batch['trans_v_split_observations']

        v_self_logits = self.network.select('value')(
            observations, observations, params=grad_params,
        )
        self_target = jnp.ones((observations.shape[0],), dtype=jnp.float32)
        loss_v_self_per = optax.sigmoid_binary_cross_entropy(v_self_logits, self_target)
        loss_v_self = self._weighted_mean(loss_v_self_per, valid_mask)
        v_self = jax.nn.sigmoid(v_self_logits)

        v_base_logits = self.network.select('value')(
            observations, batch['value_base_goals'], params=grad_params,
        )
        v_base = jax.nn.sigmoid(v_base_logits)
        base_target = jnp.clip(
            jnp.power(discount, jnp.asarray(batch['value_base_offsets'], dtype=jnp.float32)),
            eps,
            1.0,
        )
        loss_v_base_per = optax.sigmoid_binary_cross_entropy(v_base_logits, base_target)
        loss_v_base = self._weighted_mean(loss_v_base_per, valid_mask)

        v_tri_logits = self.network.select('value')(observations, goals, params=grad_params)
        v_tri = jax.nn.sigmoid(v_tri_logits)
        target_left_logits = self.network.select('target_value')(observations, batch['trans_v_left_goals'])
        target_right_logits = self.network.select('target_value')(split_obs, batch['trans_v_right_goals'])
        target_v_left = jnp.clip(jax.nn.sigmoid(target_left_logits), eps, 1.0)
        target_v_right = jnp.clip(jax.nn.sigmoid(target_right_logits), eps, 1.0)

        value_offsets = jnp.asarray(batch.get('value_offsets', jnp.ones_like(valid_mask)), dtype=jnp.float32)
        split_offsets = jnp.asarray(
            batch.get('trans_v_split_offsets', value_offsets),
            dtype=jnp.float32,
        )
        left_offsets = split_offsets
        right_offsets = value_offsets - split_offsets
        h_base = jnp.asarray(value_base_horizon, dtype=jnp.float32)
        exact_left = jnp.clip(jnp.power(discount, left_offsets), eps, 1.0)
        exact_right = jnp.clip(jnp.power(discount, right_offsets), eps, 1.0)
        target_v_left = jnp.where(left_offsets <= h_base, exact_left, target_v_left)
        target_v_right = jnp.where(right_offsets <= h_base, exact_right, target_v_right)

        target_v_tri = jax.lax.stop_gradient(jnp.clip(target_v_left * target_v_right, eps, 1.0))
        tri_valid = jnp.asarray(batch['trans_v_valid_mask'], dtype=jnp.float32) * valid_mask
        loss_v_tri_per = _bce_expectile_loss(v_tri_logits, target_v_tri, tau_v)

        if bool(self.config.get('value_transitive_reweight', True)):
            power = float(self.config.get('value_distance_weight_power', 1.0))
            clip_min = float(self.config.get('value_distance_weight_clip_min', 0.05))
            clip_max = float(self.config.get('value_distance_weight_clip_max', 1.0))
            v_for_weight = jax.lax.stop_gradient(jnp.clip(v_tri, eps, 1.0))
            d_est = jnp.log(v_for_weight) / jnp.log(jnp.asarray(discount, dtype=jnp.float32))
            d_est = jnp.maximum(d_est, 0.0)
            dist_w = 1.0 / jnp.power(1.0 + d_est, power)
            dist_w = jnp.clip(dist_w, clip_min, clip_max)
            dist_w = jax.lax.stop_gradient(dist_w)
            tri_weights = tri_valid * dist_w
        else:
            dist_w = jnp.ones_like(tri_valid)
            tri_weights = tri_valid
        loss_v_tri = self._weighted_mean(loss_v_tri_per, tri_weights)

        q_goals = batch.get('q_goals', goals)
        q_logits = self.network.select('action_critic')(
            observations, q_goals, batch['action_chunk_actions'], params=grad_params,
        )
        q_pred = jnp.clip(jax.nn.sigmoid(q_logits), eps, 1.0)
        q_offsets = jnp.asarray(batch.get('q_goal_offsets', batch['value_offsets']), dtype=jnp.float32)
        h = jnp.asarray(float(self.config['action_chunk_horizon']), dtype=jnp.float32)
        target_v_logits = self.network.select('target_value')(
            batch['action_chunk_next_observations'], q_goals,
        )
        target_v_next = jnp.clip(jax.nn.sigmoid(target_v_logits), eps, 1.0)
        reached = (q_offsets >= 1.0) & (q_offsets <= h)
        reached_target = jnp.power(discount, q_offsets)
        bootstrap_target = jnp.power(discount, h) * target_v_next
        target_q = jax.lax.stop_gradient(jnp.clip(jnp.where(reached, reached_target, bootstrap_target), eps, 1.0))
        loss_q_per = jnp.mean(optax.sigmoid_binary_cross_entropy(q_logits, target_q[None, :]), axis=0)
        loss_q = self._weighted_mean(loss_q_per, valid_mask)

        total = (
            lambda_v_self * loss_v_self
            + lambda_v_base * loss_v_base
            + lambda_v_tri * loss_v_tri
            + lambda_q_local * loss_q
        )
        q_agg = self.aggregate_ensemble_q(q_pred)
        return total, {
            'loss/total': total,
            'value/loss': lambda_v_self * loss_v_self + lambda_v_base * loss_v_base + lambda_v_tri * loss_v_tri,
            'value/self_loss': loss_v_self,
            'value/base_loss': loss_v_base,
            'value/tri_loss': loss_v_tri,
            'value/self_pred_mean': v_self.mean(),
            'value/base_pred_mean': v_base.mean(),
            'value/base_target_mean': base_target.mean(),
            'value/tri_pred_mean': v_tri.mean(),
            'value/tri_target_mean': target_v_tri.mean(),
            'value/trans_valid_fraction': jnp.mean(tri_valid),
            'value/value_offset_mean': jnp.mean(value_offsets),
            'value/split_offset_mean': jnp.mean(split_offsets),
            'value/tri_distance_weight_mean': dist_w.mean(),
            'local_q/loss': loss_q,
            'local_q/pred_mean': q_agg.mean(),
            'local_q/target_mean': target_q.mean(),
            'local_q/target_v_mean': target_v_next.mean(),
            'action_critic/trl_loss': total,
            'action_critic/distill_loss': loss_q,
            'action_critic/value_loss': lambda_v_self * loss_v_self + lambda_v_base * loss_v_base + lambda_v_tri * loss_v_tri,
            'action_critic/q_part_mean': q_agg.mean(),
            'action_critic/target_v_mean': target_q.mean(),
            'action_critic/adv': (target_q - q_agg).mean(),
            'action_critic/v_mean': v_tri.mean(),
            'action_critic/v_max': v_tri.max(),
            'action_critic/v_min': v_tri.min(),
        }

    def _flatten_action_candidates(self, action_chunk_actions: jnp.ndarray) -> tuple[jnp.ndarray, int]:
        actions = jnp.asarray(action_chunk_actions, dtype=jnp.float32)
        if actions.ndim == 4:
            bsz, num_candidates = actions.shape[:2]
            return actions.reshape(bsz, num_candidates, -1), num_candidates
        if actions.ndim == 3:
            bsz, num_candidates = actions.shape[0], actions.shape[1]
            return actions, int(num_candidates)
        if actions.ndim == 2:
            return actions[:, None, :], 1
        raise ValueError(f'action_chunk_actions must be rank-2/3/4, got shape={actions.shape}')

    @partial(jax.jit, static_argnames=('use_partial_critic',))
    def score_action_chunks(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray | None,
        action_chunk_actions: jnp.ndarray,
        network_params: dict | None = None,
        use_partial_critic: bool | None = None,
    ) -> jnp.ndarray:
        actions, num_candidates = self._flatten_action_candidates(action_chunk_actions)
        obs = jnp.asarray(observations, dtype=jnp.float32)
        obs_rep = jnp.repeat(obs[:, None, :], num_candidates, axis=1).reshape(obs.shape[0] * num_candidates, -1)
        if goals is not None:
            goals = jnp.asarray(goals, dtype=jnp.float32)
            if goals.ndim == 3:
                if goals.shape[1] != num_candidates:
                    raise ValueError(
                        f'score_action_chunks: per-candidate goals shape {goals.shape} does not '
                        f'match num_candidates={num_candidates}.'
                    )
                goal_rep = goals.reshape(goals.shape[0] * num_candidates, -1)
            elif goals.ndim == 2:
                goal_rep = jnp.repeat(goals[:, None, :], num_candidates, axis=1).reshape(
                    goals.shape[0] * num_candidates, -1
                )
            else:
                raise ValueError(f'score_action_chunks: goals must be rank-2/3, got shape={goals.shape}')
        else:
            goal_rep = None
        flat_actions = actions.reshape(obs.shape[0] * num_candidates, -1)
        partial_dim = int(self.config['action_chunk_horizon']) * int(self.config['action_dim'])
        full_dim = int(self.config['full_chunk_horizon']) * int(self.config['action_dim'])
        critic_type = self._critic_type()
        if use_partial_critic is None:
            if critic_type == 'iql' or self._is_trl():
                use_partial_critic = True
            elif flat_actions.shape[-1] == partial_dim and partial_dim != full_dim:
                use_partial_critic = True
            elif flat_actions.shape[-1] == full_dim and partial_dim != full_dim:
                use_partial_critic = False
            else:
                use_partial_critic = True

        force_partial = critic_type == 'iql' or self._is_trl()
        if force_partial or bool(use_partial_critic) or not bool(self.config['use_chunk_critic']):
            logits = self.network.select('action_critic')(obs_rep, goal_rep, flat_actions, params=network_params)
        else:
            logits = self.network.select('chunk_critic')(obs_rep, goal_rep, flat_actions, params=network_params)
        qs = jax.nn.sigmoid(logits).reshape(logits.shape[0], obs.shape[0], num_candidates)
        return self.aggregate_ensemble_q(qs).reshape(obs.shape[0], num_candidates)

    @partial(jax.jit, static_argnames=())
    def score_transitive_subgoals(
        self,
        observations: jnp.ndarray,
        subgoals: jnp.ndarray,
        goals: jnp.ndarray,
        network_params: dict | None = None,
    ) -> jnp.ndarray:
        eps = jnp.asarray(float(self.config.get('subgoal_value_ratio_eps', 1e-6)), dtype=jnp.float32)
        obs = jnp.asarray(observations, dtype=jnp.float32)
        z = jnp.asarray(subgoals, dtype=jnp.float32)
        g = jnp.asarray(goals, dtype=jnp.float32)
        num_candidates = z.shape[1] if z.ndim == 3 else 1
        obs_rep = jnp.repeat(obs[:, None, :], num_candidates, axis=1).reshape(obs.shape[0] * num_candidates, -1)
        if z.ndim == 3:
            z_flat = z.reshape(z.shape[0] * num_candidates, -1)
        else:
            z_flat = z
        if g.ndim == 3:
            g_flat = g.reshape(g.shape[0] * num_candidates, -1)
        else:
            g_flat = jnp.repeat(g[:, None, :], num_candidates, axis=1).reshape(obs.shape[0] * num_candidates, -1)

        v_s_z = jax.nn.sigmoid(self.network.select('value')(obs_rep, z_flat, params=network_params))
        v_z_g = jax.nn.sigmoid(self.network.select('value')(z_flat, g_flat, params=network_params))
        v_s_g = jax.nn.sigmoid(self.network.select('value')(obs_rep, g_flat, params=network_params))
        ratio = (v_s_z * v_z_g) / (v_s_g + eps)
        return ratio.reshape(obs.shape[0], num_candidates)

    @jax.jit
    def total_loss(self, batch: dict, grad_params: dict, rng=None):
        batch = jax.tree_util.tree_map(lambda x: jnp.asarray(x), batch)
        info = {}
        total = jnp.asarray(0.0, dtype=jnp.float32)
        if self._is_trl():
            tl, ti = self.trl_loss(batch, grad_params)
            info['chunk_critic/critic_loss'] = jnp.asarray(0.0, dtype=jnp.float32)
            info.update(ti)
            info['total_loss'] = tl
            return tl, info
        if self._has_chunk_critic():
            cl, ci = self.chunk_critic_loss(batch, grad_params)
            total = total + cl
            info.update(ci)
        else:
            info['chunk_critic/critic_loss'] = jnp.asarray(0.0, dtype=jnp.float32)
        pl, pi = self.partial_critic_loss(batch, grad_params)
        vl, vi = self.value_loss(batch, grad_params)
        total = total + pl + vl
        info.update(pi)
        info.update(vi)
        info['total_loss'] = total
        return total, info

    def _ema_target_critics(self, network: TrainState, tau: float) -> TrainState:
        updated = dict(network.params)
        if self._has_chunk_critic():
            updated['modules_target_chunk_critic'] = jax.tree_util.tree_map(
                lambda p, tp: p * tau + tp * (1.0 - tau),
                updated['modules_chunk_critic'],
                updated['modules_target_chunk_critic'],
            )
        updated['modules_target_action_critic'] = jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1.0 - tau),
            updated['modules_action_critic'],
            updated['modules_target_action_critic'],
        )
        if self._is_trl():
            updated['modules_target_value'] = jax.tree_util.tree_map(
                lambda p, tp: p * tau + tp * (1.0 - tau),
                updated['modules_value'],
                updated['modules_target_value'],
            )
        return network.replace(params=updated)

    def _validate_trl_batch(self, batch: dict) -> None:
        if not self._is_trl():
            return
        required = (
            'observations',
            'value_goals',
            'action_chunk_actions',
            'action_chunk_next_observations',
            'value_offsets',
            'value_base_goals',
            'value_base_offsets',
            'trans_v_split_observations',
            'trans_v_left_goals',
            'trans_v_right_goals',
            'trans_v_valid_mask',
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"trl batch missing required keys: {missing}")

    def update(self, batch: dict):
        self._validate_trl_batch(batch)
        return self._update_impl(batch)

    @jax.jit
    def _update_impl(self, batch: dict):
        new_rng, loss_rng = jax.random.split(self.rng)

        def loss_fn(params):
            return self.total_loss(batch, params, rng=loss_rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        new_network = self._ema_target_critics(
            new_network,
            float(self.config.get('target_tau', self.config['tau'])),
        )
        return self.replace(rng=new_rng, network=new_network), info

    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations: np.ndarray,
        ex_full_chunk_actions: np.ndarray | None,
        ex_action_chunk_actions: np.ndarray,
        config: dict,
        ex_goals: np.ndarray | None = None,
    ):
        config = dict(config)
        rng = jax.random.PRNGKey(int(seed))
        rng, network_init_rng = jax.random.split(rng)
        ex_obs = jnp.asarray(ex_observations, dtype=jnp.float32)
        ex_part = jnp.asarray(ex_action_chunk_actions, dtype=jnp.float32)
        ex_goal = jnp.asarray(ex_observations if ex_goals is None else ex_goals, dtype=jnp.float32)

        hdims = tuple(config['value_hidden_dims'])
        ln = bool(config['layer_norm'])
        nq = int(config['num_qs'])
        goal_rep = str(config.get('goal_representation', 'full')).lower()
        phi_idxs = normalize_phi_goal_obs_indices(config.get('phi_goal_obs_indices', ()))
        env_name = str(config.get('env_name', ''))
        assert_phi_goal_obs_indices(
            int(ex_obs.shape[-1]),
            goal_rep,
            phi_idxs,
            where='CriticAgent.create (critic goal_representation)',
            env_name=env_name,
        )
        critic_type, _algorithm, is_trl = _canonicalize_critic_config(config)

        value_def = ScalarValueNet(
            hdims,
            layer_norm=ln,
            goal_representation=goal_rep,
            phi_goal_obs_indices=phi_idxs,
            env_name=env_name,
        )
        action_critic_def = BinaryChunkCritic(
            hdims, nq, ln, goal_representation=goal_rep, phi_goal_obs_indices=phi_idxs, env_name=env_name
        )
        target_action_critic_def = BinaryChunkCritic(
            hdims, nq, ln, goal_representation=goal_rep, phi_goal_obs_indices=phi_idxs, env_name=env_name
        )

        network_info = {
            'action_critic': (action_critic_def, (ex_obs, ex_goal, ex_part)),
            'target_action_critic': (target_action_critic_def, (ex_obs, ex_goal, ex_part)),
        }
        if not is_trl:
            network_info['value'] = (value_def, (ex_obs, ex_goal))
        else:
            target_value_def = ScalarValueNet(
                hdims,
                layer_norm=ln,
                goal_representation=goal_rep,
                phi_goal_obs_indices=phi_idxs,
                env_name=env_name,
            )
            network_info['value'] = (value_def, (ex_obs, ex_goal))
            network_info['target_value'] = (target_value_def, (ex_obs, ex_goal))

        if critic_type == 'dqc':
            if ex_full_chunk_actions is None:
                raise ValueError(
                    "critic_type='dqc' requires ex_full_chunk_actions for chunk_critic init."
                )
            ex_full = jnp.asarray(ex_full_chunk_actions, dtype=jnp.float32)
            chunk_critic_def = BinaryChunkCritic(
                hdims, nq, ln, goal_representation=goal_rep, phi_goal_obs_indices=phi_idxs, env_name=env_name
            )
            target_chunk_critic_def = BinaryChunkCritic(
                hdims, nq, ln, goal_representation=goal_rep, phi_goal_obs_indices=phi_idxs, env_name=env_name
            )
            network_info['chunk_critic'] = (chunk_critic_def, (ex_obs, ex_goal, ex_full))
            network_info['target_chunk_critic'] = (target_chunk_critic_def, (ex_obs, ex_goal, ex_full))

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_params = network_def.init(network_init_rng, **network_args)['params']
        if critic_type == 'dqc':
            network_params['modules_target_chunk_critic'] = network_params['modules_chunk_critic']
        network_params['modules_target_action_critic'] = network_params['modules_action_critic']
        if is_trl:
            network_params['modules_target_value'] = network_params['modules_value']
        network = TrainState.create(network_def, network_params, tx=optax.adam(float(config['lr'])))
        cfg_out = dict(config)
        cfg_out['phi_goal_obs_indices'] = phi_idxs
        return cls(rng=rng, network=network, config=flax.core.FrozenDict(**cfg_out))


def validate_config(critic_config, actor_config=None) -> None:
    _canonicalize_critic_config(critic_config)

    action_chunk_horizon = int(critic_config.get('action_chunk_horizon', 0))
    full_chunk_horizon = int(critic_config.get('full_chunk_horizon', 0))
    if action_chunk_horizon < 1:
        raise ValueError('action_chunk_horizon must be >= 1.')
    if full_chunk_horizon < action_chunk_horizon:
        raise ValueError(
            f'full_chunk_horizon must be >= action_chunk_horizon, '
            f'got full_chunk_horizon={full_chunk_horizon}, action_chunk_horizon={action_chunk_horizon}.'
        )
    if actor_config is None:
        return
    if int(actor_config.get('actor_chunk_horizon', 0)) < 1:
        raise ValueError('actor_chunk_horizon must be >= 1.')


def extract_critic_primary_score(info: dict) -> float:
    if 'chunk_critic/q_mean' in info:
        return float(info['chunk_critic/q_mean'])
    return float(info['action_critic/q_part_mean'])


def get_config():
    return ml_collections.ConfigDict(
        dict(
            lr=3e-4,
            batch_size=256,
            tau=0.005,
            target_tau=0.005,
            layer_norm=True,
            frame_stack=None,
            p_aug=0.0,
            q_agg='mean',
            goal_representation='full',
            phi_goal_obs_indices=(),
            env_name='antmaze-medium-navigate-v0',
            full_chunk_horizon=25,
            action_chunk_horizon=5,
            clip_chunk_to_goal=True,
            value_hidden_dims=(512, 512, 512),
            discount=0.995,
            num_qs=2,
            algorithm='dqc',
            critic_type='dqc',
            use_chunk_critic=True,
            tau_v=0.7,
            lambda_v_base=1.0,
            lambda_v_tri=1.0,
            lambda_v_self=1.0,
            value_base_horizon=5,
            value_transitive_reweight=True,
            value_distance_weight_power=1.0,
            value_distance_weight_clip_min=0.05,
            value_distance_weight_clip_max=1.0,
            lambda_q_local=1.0,
            q_target_from_value=True,
            subgoal_value_bonus_type=None,
            subgoal_value_log_eps=1e-6,
            subgoal_value_ratio_eps=None,
            subgoal_value_ratio_clip=5.0,
            rescore_single_candidate=False,
            q_value_eps=1e-6,
            distill_method='expectile',
            kappa_d=0.7,
            implicit_backup_type='quantile',
            kappa_b=0.7,
            action_dim=2,
            value_p_curgoal=0.0,
            value_p_trajgoal=1.0,
            value_p_randomgoal=0.0,
            value_geom_sample=True,
            max_goal_steps=None,
            max_goal_steps_from_env=False,
            gc_negative=False,
        )
    )


__all__ = [
    'BinaryChunkCritic',
    'ScalarValueNet',
    'CriticAgent',
    'validate_config',
    'extract_critic_primary_score',
    'get_config',
    '_is_trl_type',
    '_canonicalize_critic_config',
]
