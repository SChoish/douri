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


def _safe_logit(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    x = jnp.clip(x, eps, 1.0 - eps)
    return jnp.log(x) - jnp.log1p(-x)


def _expectile_loss(diff: jnp.ndarray, tau: float) -> jnp.ndarray:
    weight = jnp.where(diff > 0.0, float(tau), 1.0 - float(tau))
    return weight * jnp.square(diff)


def _is_direct_chunk_trl_type(critic_type: str, algorithm: str = '') -> bool:
    critic_type = str(critic_type).lower()
    algorithm = str(algorithm).lower()
    return critic_type in ('trl', 'chunk_trl', 'direct_chunk_trl') or algorithm in (
        'chunk_trl',
        'direct_chunk_trl',
        'transitivechunkrl',
    )


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

    def _is_direct_chunk_trl(self) -> bool:
        return _is_direct_chunk_trl_type(
            self._critic_type(),
            str(self.config.get('algorithm', '')),
        )

    def _has_chunk_critic(self) -> bool:
        """True iff ``critic_type='dqc'`` and ``use_chunk_critic`` is enabled.

        IQL/direct chunk TRL modes never instantiate the separate DQC
        chunk_critic, so gated paths must skip missing modules.
        """
        return self._critic_type() == 'dqc' and bool(self.config.get('use_chunk_critic', False))

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
        """Train the partial ``action_critic`` head.

        - DQC w/ chunk_critic: target = ``aggregate(target_chunk_critic(s, g, a_full))``.
          (chunk-to-action distillation; horizon ``H_full``.)
        - DQC w/o chunk_critic: target = ``r_full + gamma^{H_full} * mask * V(s_{t+H_full})``.
        - IQL: target = ``r_action + gamma^{H_action} * mask * V(s_{t+H_action})``.
          ``value`` gradient flows here because IQL Q is regressed against the *current* V
          rather than a separate target V; use ``stop_gradient`` only on the V output if you
          want vanilla IQL semantics — current code keeps the existing pattern shared with
          the DQC bootstrap branch (gradient through V) for symmetry.
        """
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

    def direct_chunk_trl_loss(self, batch: dict, grad_params: dict) -> tuple[jnp.ndarray, dict]:
        """Direct chunk-level TRL.

        ``action_critic`` is the primary ``Q_H(s, A_H, g)`` head.  ``value`` is
        only an in-sample expectile readout of the target chunk Q.
        """
        valid_mask = self._valid_mask(batch)
        eps = float(self.config.get('q_value_eps', 1e-6))
        discount = float(self.config['discount'])
        tau_q = float(self.config.get('tau_q', 0.7))
        tau_v = float(self.config.get('tau_v', 0.9))
        lambda_q_base = float(self.config.get('lambda_q_base', 1.0))
        lambda_q_tri = float(self.config.get('lambda_q_tri', 1.0))
        lambda_v = float(self.config.get('lambda_v', 1.0))
        use_v_in_q_target = bool(self.config.get('use_v_in_q_target', False))

        goals = batch['value_goals']
        actions = batch['action_chunk_actions']
        split_obs = batch['trl_split_observations']
        split_goals = batch['trl_split_goals']
        split_actions = batch['trl_split_action_chunk_actions']

        q_base_logits = self.network.select('action_critic')(
            batch['observations'], batch['trl_base_goals'], actions, params=grad_params,
        )
        q_base = jnp.clip(jax.nn.sigmoid(q_base_logits), eps, 1.0)
        base_target = jnp.power(discount, jnp.asarray(batch['trl_base_offsets'], dtype=jnp.float32))
        loss_q_base_per = jnp.mean(jnp.square(q_base - base_target[None, :]), axis=0)
        loss_q_base = self._weighted_mean(loss_q_base_per, valid_mask)

        q_tri_logits = self.network.select('action_critic')(
            batch['observations'], goals, actions, params=grad_params,
        )
        q_tri = jnp.clip(jax.nn.sigmoid(q_tri_logits), eps, 1.0)
        q_left_logits = self.network.select('target_action_critic')(
            batch['observations'], split_goals, actions,
        )
        q_left = jnp.clip(jax.nn.sigmoid(q_left_logits), eps, 1.0)
        if use_v_in_q_target:
            v_right_logits = self.network.select('target_value')(split_obs, goals)
            q_right = jnp.clip(jax.nn.sigmoid(v_right_logits)[None, :], eps, 1.0)
        else:
            q_right_logits = self.network.select('target_action_critic')(
                split_obs, goals, split_actions,
            )
            q_right = jnp.clip(jax.nn.sigmoid(q_right_logits), eps, 1.0)
        target_q_tri = jnp.clip(jax.lax.stop_gradient(q_left * q_right), eps, 1.0)
        tri_valid = jnp.asarray(batch['trl_valid_mask'], dtype=jnp.float32) * valid_mask
        loss_q_tri_per = jnp.mean(_expectile_loss(target_q_tri - q_tri, tau_q), axis=0)
        loss_q_tri = self._weighted_mean(loss_q_tri_per, tri_valid)

        v_logit = self.network.select('value')(batch['observations'], goals, params=grad_params)
        v = jnp.clip(jax.nn.sigmoid(v_logit), eps, 1.0)
        target_v_logits = self.network.select('target_action_critic')(
            batch['observations'], goals, actions,
        )
        target_v = self.aggregate_ensemble_q(jnp.clip(jax.nn.sigmoid(target_v_logits), eps, 1.0))
        target_v = jax.lax.stop_gradient(target_v)
        loss_v_per = _expectile_loss(target_v - v, tau_v)
        loss_v = self._weighted_mean(loss_v_per, valid_mask)

        total = lambda_q_base * loss_q_base + lambda_q_tri * loss_q_tri + lambda_v * loss_v
        q_base_agg = self.aggregate_ensemble_q(q_base)
        q_tri_agg = self.aggregate_ensemble_q(q_tri)
        target_tri_agg = self.aggregate_ensemble_q(target_q_tri)
        return total, {
            'loss/total': total,
            'loss/q_base': loss_q_base,
            'loss/q_tri': loss_q_tri,
            'loss/v': loss_v,
            'q/base_pred_mean': q_base_agg.mean(),
            'q/base_target_mean': base_target.mean(),
            'q/tri_pred_mean': q_tri_agg.mean(),
            'q/tri_target_mean': target_tri_agg.mean(),
            'v/pred_mean': v.mean(),
            'v/target_mean': target_v.mean(),
            'sampler/valid_tri_fraction': jnp.mean(jnp.asarray(batch['trl_valid_mask'], dtype=jnp.float32)),
            # Backward-compatible keys used by existing logging/extraction paths.
            'action_critic/trl_loss': total,
            'action_critic/distill_loss': loss_q_tri,
            'action_critic/value_loss': loss_v,
            'action_critic/q_part_mean': q_tri_agg.mean(),
            'action_critic/target_v_mean': target_tri_agg.mean(),
            'action_critic/adv': (target_v - v).mean(),
            'action_critic/v_mean': v.mean(),
            'action_critic/v_max': v.max(),
            'action_critic/v_min': v.min(),
        }

    def _flatten_action_candidates(self, action_chunk_actions: jnp.ndarray) -> tuple[jnp.ndarray, int]:
        actions = jnp.asarray(action_chunk_actions, dtype=jnp.float32)
        if actions.ndim == 4:
            bsz, num_candidates = actions.shape[:2]
            return actions.reshape(bsz, num_candidates, -1), num_candidates
        if actions.ndim == 3:
            bsz = actions.shape[0]
            return actions.reshape(bsz, 1, -1), 1
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
                # Per-candidate goals [B, N, D] -> directly flatten.
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
            if critic_type == 'iql' or self._is_direct_chunk_trl():
                use_partial_critic = True
            elif flat_actions.shape[-1] == partial_dim and partial_dim != full_dim:
                use_partial_critic = True
            elif flat_actions.shape[-1] == full_dim and partial_dim != full_dim:
                use_partial_critic = False
            else:
                use_partial_critic = True

        # IQL/direct chunk TRL: never call DQC chunk_critic (module not initialized).
        force_partial = critic_type == 'iql' or self._is_direct_chunk_trl()
        if force_partial or bool(use_partial_critic) or not bool(self.config['use_chunk_critic']):
            logits = self.network.select('action_critic')(obs_rep, goal_rep, flat_actions, params=network_params)
        else:
            logits = self.network.select('chunk_critic')(obs_rep, goal_rep, flat_actions, params=network_params)
        qs = jax.nn.sigmoid(logits).reshape(logits.shape[0], obs.shape[0], num_candidates)
        return self.aggregate_ensemble_q(qs).reshape(obs.shape[0], num_candidates)

    @jax.jit
    def total_loss(self, batch: dict, grad_params: dict, rng=None):
        batch = jax.tree_util.tree_map(lambda x: jnp.asarray(x), batch)
        info = {}
        total = jnp.asarray(0.0, dtype=jnp.float32)
        if self._is_direct_chunk_trl():
            tl, ti = self.direct_chunk_trl_loss(batch, grad_params)
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
        """EMA-update target critics. Skips ``target_chunk_critic`` when not initialized."""
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
        if self._is_direct_chunk_trl():
            updated['modules_target_value'] = jax.tree_util.tree_map(
                lambda p, tp: p * tau + tp * (1.0 - tau),
                updated['modules_value'],
                updated['modules_target_value'],
            )
        return network.replace(params=updated)

    @jax.jit
    def update(self, batch: dict):
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
        critic_type = str(config.get('critic_type', 'dqc')).lower()
        algorithm = str(config.get('algorithm', '')).lower()
        if critic_type not in ('dqc', 'iql', 'trl', 'chunk_trl', 'direct_chunk_trl'):
            raise ValueError(
                f"critic_type must be one of 'dqc', 'iql', 'trl', 'chunk_trl', or "
                f"'direct_chunk_trl', got {critic_type!r}"
            )
        is_trl = _is_direct_chunk_trl_type(critic_type, algorithm)

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
    critic_type = str(critic_config.get('critic_type', 'dqc')).lower()
    algorithm = str(critic_config.get('algorithm', '')).lower()
    if algorithm in ('chunk_trl', 'direct_chunk_trl', 'transitivechunkrl'):
        critic_type = 'direct_chunk_trl'
    if critic_type not in ('dqc', 'iql', 'trl', 'chunk_trl', 'direct_chunk_trl'):
        raise ValueError(
            f"critic_type must be one of 'dqc', 'iql', 'trl', 'chunk_trl', or "
            f"'direct_chunk_trl', got {critic_type!r}"
        )
    critic_config['critic_type'] = critic_type
    if _is_direct_chunk_trl_type(critic_type, algorithm):
        critic_config['algorithm'] = 'direct_chunk_trl'

    action_chunk_horizon = int(critic_config.get('action_chunk_horizon', 0))
    full_chunk_horizon = int(critic_config.get('full_chunk_horizon', 0))
    if action_chunk_horizon < 1:
        raise ValueError('action_chunk_horizon must be >= 1.')
    if full_chunk_horizon < action_chunk_horizon:
        raise ValueError(
            f'full_chunk_horizon must be >= action_chunk_horizon, '
            f'got full_chunk_horizon={full_chunk_horizon}, action_chunk_horizon={action_chunk_horizon}.'
        )
    if (critic_type == 'iql' or _is_direct_chunk_trl_type(critic_type, algorithm)) and bool(
        critic_config.get('use_chunk_critic', False)
    ):
        # IQL/direct chunk TRL never train DQC chunk_critic; force the flag off.
        critic_config['use_chunk_critic'] = False
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
            # Goal input to value/Q nets. 'full' preserves raw goal concat.
            # 'phi' uses ManipSpace cube channels when inferred, else maze (x,y).
            # When omitted, ``main`` / ``eval_checkpoint`` fill indices from ``env_name``
            # and observation shape after the env is constructed.
            goal_representation='phi',
            phi_goal_obs_indices=(),
            # Default used when constructing agents directly in tests/tools.
            # ``main.py`` / ``eval_checkpoint.py`` overwrite this from the run env.
            env_name='antmaze-medium-navigate-v0',
            full_chunk_horizon=25,
            action_chunk_horizon=5,
            # Match dynamics' default clip_path_to_goal semantics for critic backups:
            # if the sampled value goal lies within the chunk window, use the goal
            # state as next_obs, shorten backup_horizon to steps-to-goal, and set
            # mask=0 so Q terminates at the goal.
            clip_chunk_to_goal=True,
            value_hidden_dims=(512, 512, 512),
            discount=0.995,
            num_qs=2,
            # 'dqc' (default): chunk_critic + partial action_critic + value (current behavior).
            # 'iql':           action_critic + value only; Q backup uses V at s_{t+H_action}.
            # 'direct_chunk_trl'/'chunk_trl'/'trl':
            #                  direct chunk-level TRL with Q_H as the primary
            #                  learned object and V as an expectile readout.
            algorithm='dqc',
            critic_type='dqc',
            use_chunk_critic=True,
            tau_q=0.7,
            tau_v=0.9,
            lambda_q_base=1.0,
            lambda_q_tri=1.0,
            lambda_v=1.0,
            use_v_in_q_target=False,
            q_value_eps=1e-6,
            distill_method='expectile',
            kappa_d=0.7,
            implicit_backup_type='quantile',
            kappa_b=0.7,
            action_dim=2,
            value_p_curgoal=0.2,
            value_p_trajgoal=0.5,
            value_p_randomgoal=0.3,
            value_geom_sample=False,
            # Optional cap for same-trajectory sampled value goals. None/<=0
            # preserves terminal-only clipping.
            max_goal_steps=None,
            max_goal_steps_from_env=True,
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
]
