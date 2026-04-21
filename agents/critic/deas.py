from __future__ import annotations

from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.hl_gauss import (
    cross_entropy_loss_on_scalar,
    hl_gauss_atom_centers,
    transform_from_probs,
)

from .common import DistributionalCriticEnsemble, DistributionalValueNet


class DEASSeqCriticAgent(flax.struct.PyTreeNode):
    """Detached DEAS value + distributional action-chunk critic."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _z(self) -> jnp.ndarray:
        return hl_gauss_atom_centers(
            float(self.config['v_min']),
            float(self.config['v_max']),
            int(self.config['num_atoms']),
        )

    def _sigma(self) -> float:
        return float(self.config['sigma'])

    def _expectile(self) -> float:
        return float(self.config['expectile'])

    def _actions_flat(self, actions: jnp.ndarray) -> jnp.ndarray:
        b, ell, a = actions.shape
        return jnp.reshape(actions, (b, ell * a))

    def aggregate_ensemble_q(self, qs: jnp.ndarray) -> jnp.ndarray:
        q_agg = str(self.config.get('q_agg', 'min')).lower()
        if q_agg == 'mean':
            return jnp.mean(qs, axis=0)
        if q_agg == 'min':
            return jnp.min(qs, axis=0)
        raise ValueError(f'Unknown q_agg: {q_agg!r} (expected min or mean).')

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

    def score_action_chunks(
        self,
        observations: jnp.ndarray,
        goals: jnp.ndarray | None,
        action_chunk_actions: jnp.ndarray,
        network_params: dict | None = None,
        use_partial_critic: bool | None = None,
    ) -> jnp.ndarray:
        # NOTE: this scorer is not goal-conditioned in this implementation.
        # Joint DEAS mode is intended to be critic-only; SPI actor training is blocked at config validation.
        del goals, use_partial_critic
        actions, num_candidates = self._flatten_action_candidates(action_chunk_actions)
        obs = jnp.asarray(observations, dtype=jnp.float32)
        obs_rep = jnp.repeat(obs[:, None, :], num_candidates, axis=1).reshape(obs.shape[0] * num_candidates, -1)
        flat_actions = actions.reshape(obs.shape[0] * num_candidates, -1)
        logits = self.network.select('critic')(obs_rep, flat_actions, params=network_params)
        probs = jax.nn.softmax(logits, axis=-1)
        qs = transform_from_probs(probs, z_centers=self._z()).reshape(logits.shape[0], obs.shape[0], num_candidates)
        return self.aggregate_ensemble_q(qs).reshape(obs.shape[0], num_candidates)

    def value_loss(self, batch: dict, grad_params: dict) -> tuple[jnp.ndarray, dict]:
        z = self._z()
        obs = batch['observations']
        actions_flat = self._actions_flat(batch['actions'])
        tq_logits = self.network.select('target_critic')(obs, actions_flat, params=None)
        tq_logits = jax.lax.stop_gradient(tq_logits)
        tq_probs = jax.nn.softmax(tq_logits, axis=-1)
        qs = transform_from_probs(tq_probs, z_centers=z)

        q_agg = str(self.config.get('q_agg', 'min')).lower()
        b = obs.shape[0]
        if q_agg == 'min':
            min_idx = jnp.argmin(qs, axis=0)
            q = qs[min_idx, jnp.arange(b)]
            q_prob = tq_probs[min_idx, jnp.arange(b)]
        elif q_agg == 'mean':
            q = jnp.mean(qs, axis=0)
            q_prob = jnp.mean(tq_probs, axis=0)
        else:
            raise ValueError(f'Unknown q_agg: {q_agg!r} (expected min or mean).')

        v_logits = self.network.select('value')(obs, params=grad_params)
        v_prob = jax.nn.softmax(v_logits, axis=-1)
        v = transform_from_probs(v_prob, z_centers=z)

        weight = jnp.where(q >= v, self._expectile(), 1.0 - self._expectile())
        log_sv = jax.nn.log_softmax(v_logits, axis=-1)
        ce = -jnp.sum(q_prob * log_sv, axis=-1)
        loss = jnp.mean(weight * ce)
        return loss, {
            'value/value_loss': loss,
            'value/v_mean': jnp.mean(v),
            'value/q_teacher_mean': jnp.mean(q),
            'value/weight_mean': jnp.mean(weight),
        }

    def critic_loss(self, batch: dict, grad_params: dict) -> tuple[jnp.ndarray, dict]:
        z = self._z()
        obs = batch['observations']
        nxt = batch['next_observations']
        actions_flat = self._actions_flat(batch['actions'])

        next_v_logits = self.network.select('value')(nxt, params=grad_params)
        next_v_prob = jax.nn.softmax(next_v_logits, axis=-1)
        next_v = transform_from_probs(next_v_prob, z_centers=z)
        next_v = jax.lax.stop_gradient(next_v)

        target_v = batch['chunk_return'] + batch['bootstrap_discount'] * batch['masks'] * next_v
        q_logits = self.network.select('critic')(obs, actions_flat, params=grad_params)
        ce_per = [
            cross_entropy_loss_on_scalar(q_logits[qi], target_v, z_centers=z, sigma=self._sigma())
            for qi in range(int(q_logits.shape[0]))
        ]
        loss = jnp.mean(jnp.stack(ce_per, axis=0))

        q_probs = jax.nn.softmax(q_logits, axis=-1)
        qs = transform_from_probs(q_probs, z_centers=z)
        return loss, {
            'critic/critic_loss': loss,
            'critic/q_mean': jnp.mean(qs),
            'critic/q_min': jnp.min(qs),
            'critic/q_max': jnp.max(qs),
            'critic/batch_rewards_mean': jnp.mean(batch['step_rewards']),
            'critic/target_v_mean': jnp.mean(target_v),
        }

    @jax.jit
    def total_loss(self, batch: dict, grad_params: dict, rng=None):
        batch = jax.tree_util.tree_map(lambda x: jnp.asarray(x), batch)
        vl, vi = self.value_loss(batch, grad_params)
        cl, ci = self.critic_loss(batch, grad_params)
        info = {}
        info.update(vi)
        info.update(ci)
        return vl + cl, info

    @staticmethod
    def _ema_target_critic(network: TrainState, tau: float) -> TrainState:
        ud = dict(network.params)
        ud['modules_target_critic'] = jax.tree_util.tree_map(
            lambda po, tp: po * tau + tp * (1.0 - tau),
            ud['modules_critic'],
            ud['modules_target_critic'],
        )
        return network.replace(params=ud)

    @jax.jit
    def update(self, batch: dict):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(p):
            return self.total_loss(batch, p, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        new_network = self._ema_target_critic(new_network, float(self.config['tau']))
        return self.replace(network=new_network, rng=new_rng), info

    @classmethod
    def create(
        cls,
        seed: int,
        ex_observations: np.ndarray,
        ex_actions_seq: np.ndarray,
        config: dict,
        ex_goals: np.ndarray | None = None,
    ):
        rng = jax.random.PRNGKey(int(seed))
        rng, init_rng = jax.random.split(rng)
        ex_obs = jnp.asarray(ex_observations, dtype=jnp.float32)
        ex_act = jnp.asarray(ex_actions_seq, dtype=jnp.float32)
        ex_flat = jnp.reshape(ex_act, (ex_act.shape[0], -1))

        value_def = DistributionalValueNet(
            hidden_dims=tuple(config['value_hidden_dims']),
            num_atoms=int(config['num_atoms']),
            layer_norm=bool(config['layer_norm']),
        )
        critic_def = DistributionalCriticEnsemble(
            hidden_dims=tuple(config['critic_hidden_dims']),
            num_atoms=int(config['num_atoms']),
            num_q=int(config['num_critic_ensembles']),
            layer_norm=bool(config['layer_norm']),
        )
        target_critic_def = DistributionalCriticEnsemble(
            hidden_dims=tuple(config['critic_hidden_dims']),
            num_atoms=int(config['num_atoms']),
            num_q=int(config['num_critic_ensembles']),
            layer_norm=bool(config['layer_norm']),
        )
        network_info = {
            'value': (value_def, (ex_obs,)),
            'critic': (critic_def, (ex_obs, ex_flat)),
            'target_critic': (target_critic_def, (ex_obs, ex_flat)),
        }
        network_def = ModuleDict({k: v[0] for k, v in network_info.items()})
        network_args = {k: v[1] for k, v in network_info.items()}
        network_tx = optax.adam(learning_rate=float(config['lr']))
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)
        ud = flax.core.unfreeze(network.params)
        ud['modules_target_critic'] = jax.tree_util.tree_map(lambda x: jnp.array(x), ud['modules_critic'])
        network = network.replace(params=ud, opt_state=network_tx.init(ud))
        return cls(rng=rng, network=network, config=flax.core.FrozenDict(**config))
