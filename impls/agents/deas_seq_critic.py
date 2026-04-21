"""DEAS-style detached distributional critic + value (HL-Gauss), action-sequence Q.

Tensor shapes (continuous control)
----------------------------------
* ``observations``: ``[B, D]``
* ``actions``: ``[B, L, A]`` with ``L = critic_action_sequence``
* ``actions_flat``: ``[B, L * A]`` — concatenated chunk fed to the critic MLP
* ``next_observations``: ``[B, D]`` — ``s_{t + nstep_options * L}``
* ``chunk_return``, ``bootstrap_discount``, ``masks``: ``[B]``
* Critic logits (per ensemble head): ``[num_q, B, num_atoms]``
* Value logits: ``[B, num_atoms]``

Training uses only offline batch actions for the value teacher (target critic); no
actor sampling enters value or critic targets.
"""

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
from utils.hl_gauss import (
    cross_entropy_loss_on_scalar,
    hl_gauss_atom_centers,
    transform_from_probs,
    transform_to_probs,
)
from utils.networks import MLP


class DistributionalValueNet(nn.Module):
    """``V(s)`` → distributional logits ``[B, num_atoms]``."""

    hidden_dims: Sequence[int]
    num_atoms: int
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        h = MLP((*self.hidden_dims, self.num_atoms), activate_final=False, layer_norm=self.layer_norm)(observations)
        return h


class DistributionalCriticEnsemble(nn.Module):
    """``Q(s, flat_a)`` with ``num_q`` heads; returns ``[num_q, B, num_atoms]``."""

    hidden_dims: Sequence[int]
    num_atoms: int
    num_q: int
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions_flat: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions_flat], axis=-1)
        h = MLP(tuple(self.hidden_dims), activate_final=True, layer_norm=self.layer_norm)(x)
        outs = []
        for i in range(int(self.num_q)):
            outs.append(nn.Dense(self.num_atoms, name=f'q_head_{i}')(h))
        return jnp.stack(outs, axis=0)


class DEASSeqCriticAgent(flax.struct.PyTreeNode):
    """Detached DEAS value + distributional action-chunk critic (HL-Gauss)."""

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
        """``[B, L, A] -> [B, L*A]``."""
        b, ell, a = actions.shape
        return jnp.reshape(actions, (b, ell * a))

    def value_loss(self, batch: dict, grad_params: dict) -> tuple[jnp.ndarray, dict]:
        """Detached DEAS value loss (teacher = target critic on dataset chunk only)."""
        z = self._z()
        sigma = self._sigma()
        e = self._expectile()

        obs = batch['observations']
        actions_flat = self._actions_flat(batch['actions'])

        tq_logits = self.network.select('target_critic')(obs, actions_flat, params=None)
        tq_logits = jax.lax.stop_gradient(tq_logits)
        tq_probs = jax.nn.softmax(tq_logits, axis=-1)
        qs = transform_from_probs(tq_probs, z_centers=z)  # [Q, B]

        q_agg = str(self.config.get('q_agg', 'min')).lower()
        b = obs.shape[0]
        if q_agg == 'min':
            min_idx = jnp.argmin(qs, axis=0)  # [B]
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

        weight = jnp.where(q >= v, e, 1.0 - e)
        log_sv = jax.nn.log_softmax(v_logits, axis=-1)
        ce = -jnp.sum(q_prob * log_sv, axis=-1)
        loss = jnp.mean(weight * ce)

        info = {
            'value_loss': loss,
            'v_mean': jnp.mean(v),
            'q_teacher_mean': jnp.mean(q),
            'weight_mean': jnp.mean(weight),
        }
        return loss, info

    def critic_loss(self, batch: dict, grad_params: dict) -> tuple[jnp.ndarray, dict]:
        """Distributional Bellman backup on scalars with HL-Gauss CE (ensemble mean loss)."""
        z = self._z()
        sigma = self._sigma()

        obs = batch['observations']
        nxt = batch['next_observations']
        actions_flat = self._actions_flat(batch['actions'])

        next_v_logits = self.network.select('value')(nxt, params=grad_params)
        next_v_prob = jax.nn.softmax(next_v_logits, axis=-1)
        next_v = transform_from_probs(next_v_prob, z_centers=z)
        next_v = jax.lax.stop_gradient(next_v)

        chunk_ret = batch['chunk_return']
        disc = batch['bootstrap_discount']
        m = batch['masks']
        target_v = chunk_ret + disc * m * next_v

        q_logits = self.network.select('critic')(obs, actions_flat, params=grad_params)
        num_q = int(q_logits.shape[0])
        ce_per = []
        for qi in range(num_q):
            ce_per.append(
                cross_entropy_loss_on_scalar(q_logits[qi], target_v, z_centers=z, sigma=sigma)
            )
        ce_stack = jnp.stack(ce_per, axis=0)  # [Q, B]
        loss = jnp.mean(ce_stack)

        q_probs = jax.nn.softmax(q_logits, axis=-1)
        qs = transform_from_probs(q_probs, z_centers=z)
        q_mean = jnp.mean(qs)
        q_min = jnp.min(qs)
        q_max = jnp.max(qs)
        step_rew = batch['step_rewards']
        batch_rewards_mean = jnp.mean(step_rew)
        target_v_mean = jnp.mean(target_v)

        info = {
            'critic_loss': loss,
            'q_mean': q_mean,
            'q_min': q_min,
            'q_max': q_max,
            'batch_rewards_mean': batch_rewards_mean,
            'target_v_mean': target_v_mean,
        }
        return loss, info

    @jax.jit
    def total_loss(self, batch: dict, grad_params: dict, rng=None):
        batch = jax.tree_util.tree_map(lambda x: jnp.asarray(x), batch)
        info = {}
        vl, vi = self.value_loss(batch, grad_params)
        for k, v in vi.items():
            info[f'value/{k}'] = v
        cl, ci = self.critic_loss(batch, grad_params)
        for k, v in ci.items():
            info[f'critic/{k}'] = v
        return vl + cl, info

    @staticmethod
    def _ema_target_critic(network: TrainState, tau: float) -> TrainState:
        """EMA-update ``target_critic`` from ``critic`` (functional params update)."""
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
    def create(cls, seed: int, ex_observations: np.ndarray, ex_actions_seq: np.ndarray, config: dict):
        """Create agent.

        Args:
            ex_observations: ``[1, D]``
            ex_actions_seq: ``[1, L, A]`` example action chunk.
        """
        rng = jax.random.PRNGKey(int(seed))
        rng, init_rng = jax.random.split(rng)

        if ex_actions_seq.ndim != 3:
            raise ValueError(f'ex_actions_seq must be [1, L, A], got shape {ex_actions_seq.shape}.')

        ex_obs = jnp.asarray(ex_observations, dtype=jnp.float32)
        ex_act = jnp.asarray(ex_actions_seq, dtype=jnp.float32)
        ex_flat = jnp.reshape(ex_act, (ex_act.shape[0], -1))

        num_atoms = int(config['num_atoms'])
        num_q = int(config['num_critic_ensembles'])
        if num_q < 2:
            raise ValueError('num_critic_ensembles must be >= 2.')

        value_def = DistributionalValueNet(
            hidden_dims=tuple(config['value_hidden_dims']),
            num_atoms=num_atoms,
            layer_norm=bool(config['layer_norm']),
        )
        critic_def = DistributionalCriticEnsemble(
            hidden_dims=tuple(config['critic_hidden_dims']),
            num_atoms=num_atoms,
            num_q=num_q,
            layer_norm=bool(config['layer_norm']),
        )
        target_critic_def = DistributionalCriticEnsemble(
            hidden_dims=tuple(config['critic_hidden_dims']),
            num_atoms=num_atoms,
            num_q=num_q,
            layer_norm=bool(config['layer_norm']),
        )

        network_info = dict(
            value=(value_def, (ex_obs,)),
            critic=(critic_def, (ex_obs, ex_flat)),
            target_critic=(target_critic_def, (ex_obs, ex_flat)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=float(config['lr']))
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        # Plain dict params so ``jax.grad`` / optax see matching PyTree types (not ``FrozenDict`` vs ``dict``).
        ud = flax.core.unfreeze(network.params)
        ud['modules_target_critic'] = jax.tree_util.tree_map(lambda x: jnp.array(x), ud['modules_critic'])
        opt_state = network_tx.init(ud)
        network = network.replace(params=ud, opt_state=opt_state)

        return cls(rng=rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='deas_seq_critic',
            lr=3e-4,
            batch_size=256,
            value_hidden_dims=(256, 256, 256),
            critic_hidden_dims=(256, 256, 256),
            layer_norm=True,
            tau=0.005,
            expectile=0.7,
            num_atoms=51,
            v_min=-200.0,
            v_max=0.0,
            sigma=1.0,
            q_agg='min',
            num_critic_ensembles=2,
            gamma1=0.99,
            gamma2=0.99,
            critic_action_sequence=4,
            actor_action_sequence=4,
            nstep_options=1,
            dataset_class='DEASActionSeqDataset',
            frame_stack=None,
            p_aug=0.0,
        )
    )
