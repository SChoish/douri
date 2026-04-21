from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from utils.networks import MLP


def _safe_logit(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    x = jnp.clip(x, eps, 1.0 - eps)
    return jnp.log(x) - jnp.log1p(-x)


class ScalarValueNet(nn.Module):
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


class DistributionalValueNet(nn.Module):
    hidden_dims: Sequence[int]
    num_atoms: int
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        return MLP((*self.hidden_dims, self.num_atoms), activate_final=False, layer_norm=self.layer_norm)(observations)


class DistributionalCriticEnsemble(nn.Module):
    hidden_dims: Sequence[int]
    num_atoms: int
    num_q: int
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations: jnp.ndarray, actions_flat: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([observations, actions_flat], axis=-1)
        h = MLP(tuple(self.hidden_dims), activate_final=True, layer_norm=self.layer_norm)(x)
        outs = [nn.Dense(self.num_atoms, name=f'q_head_{i}')(h) for i in range(int(self.num_q))]
        return jnp.stack(outs, axis=0)
