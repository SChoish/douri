from __future__ import annotations

from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from utils.flax_utils import TrainState, nonpytree_field
from utils.networks import MLP


SPI_CONDITIONED_CHOICES = ('subgoal', 'goal')
# Backward-compat alias for the old key name (`spi_goal_conditioning`); kept so
# already-saved checkpoints / external imports do not break.
SPI_GOAL_CONDITIONING_CHOICES = SPI_CONDITIONED_CHOICES


def validate_spi_conditioned(value: str) -> str:
    v = str(value).strip().lower()
    if v not in SPI_CONDITIONED_CHOICES:
        raise ValueError(
            f"actor.spi_conditioned must be one of {SPI_CONDITIONED_CHOICES}, got {value!r}."
        )
    return v


# Legacy name preserved for any external callers; same semantics.
validate_spi_goal_conditioning = validate_spi_conditioned


class DeterministicChunkActor(nn.Module):
    """Deterministic actor that outputs a flattened action chunk."""

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations: jnp.ndarray, goals: jnp.ndarray | None = None) -> jnp.ndarray:
        xs = [observations]
        if goals is not None:
            xs.append(goals)
        x = jnp.concatenate(xs, axis=-1)
        out = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)(x)
        return jnp.clip(out, -1.0, 1.0)


class JointActorAgent(flax.struct.PyTreeNode):
    """Joint actor trained against an external critic scorer."""

    rng: Any
    actor: Any
    config: Any = nonpytree_field()

    def _goals(self, batch: dict) -> jnp.ndarray | None:
        # ``spi_goals`` carries whatever conditioning vector the joint loop chose for
        # this update (predicted subgoal or global goal); see ``config.spi_conditioned``
        # and ``main._build_actor_batch_from_goub``. Both π and the critic scorer in
        # ``actor_loss`` share this same vector so Q stays consistent with π.
        goals = batch.get('spi_goals', None)
        if goals is None:
            goals = batch.get('value_goals', None)
        return None if goals is None else jnp.asarray(goals, dtype=jnp.float32)

    def _chunk_dim(self) -> int:
        return int(self.config['actor_chunk_horizon']) * int(self.config['action_dim'])

    def _dim_mask(self, batch: dict, chunk_dim: int) -> jnp.ndarray:
        valids = batch.get('valids', None)
        if valids is None:
            return jnp.ones((batch['observations'].shape[0], chunk_dim), dtype=jnp.float32)
        valids = jnp.asarray(valids, dtype=jnp.float32)
        if valids.ndim == 1:
            return jnp.repeat(valids[:, None], chunk_dim, axis=1)
        steps = valids.shape[-1]
        if chunk_dim % steps != 0:
            return jnp.ones((batch['observations'].shape[0], chunk_dim), dtype=jnp.float32)
        rep = chunk_dim // steps
        return jnp.repeat(valids, rep, axis=-1)

    def _proposal_chunks(self, batch: dict) -> jnp.ndarray:
        external = batch.get('proposal_partial_chunks', None)
        if external is None:
            raise ValueError('Joint actor update requires proposal_partial_chunks in the batch.')
        external = jnp.asarray(external, dtype=jnp.float32)
        if external.ndim == 4:
            external = external.reshape(external.shape[0], external.shape[1], -1)
        if external.ndim != 3:
            raise ValueError(f'proposal_partial_chunks must be rank-3/4, got shape={external.shape}')
        chunk_dim = self._chunk_dim()
        if external.shape[-1] != chunk_dim:
            raise ValueError(f'proposal_partial_chunks last dim must be {chunk_dim}, got {external.shape[-1]}.')
        return jax.lax.stop_gradient(external)

    def actor_loss(self, batch: dict, actor_params: dict, critic_agent: Any) -> tuple[jnp.ndarray, dict]:
        proposal_chunks = self._proposal_chunks(batch)
        proposal_scores = batch.get('proposal_scores', None)
        if proposal_scores is None:
            raise ValueError(
                'SPI actor path requires proposal_scores precomputed from the current critic snapshot. '
                'Rescore proposals in the joint training loop before actor update.'
            )
        proposal_scores = jax.lax.stop_gradient(jnp.asarray(proposal_scores, dtype=jnp.float32))
        if proposal_scores.ndim != 2:
            raise ValueError(f'proposal_scores must be rank-2 [B, K], got shape={proposal_scores.shape}')
        if proposal_scores.shape[:2] != proposal_chunks.shape[:2]:
            raise ValueError(
                'proposal_scores must align with proposal_partial_chunks, '
                f'got scores={proposal_scores.shape} proposals={proposal_chunks.shape}.'
            )

        goals = self._goals(batch)
        rho = jax.nn.softmax(float(self.config['spi_beta']) * proposal_scores, axis=1)
        rho = jax.lax.stop_gradient(rho)

        actor_chunk = self.actor(batch['observations'], goals, params=actor_params)
        actor_q = critic_agent.score_action_chunks(
            batch['observations'],
            goals,
            actor_chunk,
            network_params=critic_agent.network.params,
            use_partial_critic=True,
        )[:, 0]

        dim_mask = self._dim_mask(batch, proposal_chunks.shape[-1])
        diff = (actor_chunk[:, None, :] - proposal_chunks) * dim_mask[:, None, :]
        sqdist = jnp.sum(diff**2, axis=-1)
        prox = jnp.sum(rho * sqdist, axis=1)
        # Scale critic Q by batch-mean |Q| so the SPI term is ``-Q / (mean|Q| + eps)``.
        q_eps = jnp.asarray(float(self.config.get('spi_q_norm_eps', 1e-6)), dtype=jnp.float32)
        q_scale = jax.lax.stop_gradient(jnp.mean(jnp.abs(actor_q)) + q_eps)
        actor_q_scaled = actor_q / q_scale
        actor_loss = jnp.mean(-actor_q_scaled + prox / (2.0 * float(self.config['spi_tau'])))

        rho_eps = 1e-8
        rho_entropy = -jnp.sum(rho * jnp.log(rho + rho_eps), axis=1).mean()
        return actor_loss, {
            'spi_actor/actor_loss': actor_loss,
            'spi_actor/q_mean': actor_q.mean(),
            'spi_actor/q_max': actor_q.max(),
            'spi_actor/q_min': actor_q.min(),
            'spi_actor/prox_mean': prox.mean(),
            'spi_actor/prox_max': prox.max(),
            'spi_actor/prox_min': prox.min(),
            'spi_actor/rho_entropy': rho_entropy,
            'spi_actor/rho_max_mean': jnp.max(rho, axis=1).mean(),
        }

    @jax.jit
    def update(self, batch: dict, critic_agent: Any):
        new_rng, _ = jax.random.split(self.rng)
        batch = jax.tree_util.tree_map(lambda x: jnp.asarray(x), batch)

        def loss_fn(actor_params):
            return self.actor_loss(batch, actor_params=actor_params, critic_agent=critic_agent)

        new_actor, info = self.actor.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(rng=new_rng, actor=new_actor), info

    def sample_actions(self, observations, goals=None):
        observations = jnp.asarray(observations, dtype=jnp.float32)
        squeeze = observations.ndim == 1
        if squeeze:
            observations = observations[None]
            goals = None if goals is None else jnp.asarray(goals, dtype=jnp.float32)[None]
        elif goals is not None:
            goals = jnp.asarray(goals, dtype=jnp.float32)

        chunk = self.actor(observations, goals)
        horizon = int(self.config['actor_chunk_horizon'])
        action_dim = int(self.config['action_dim'])
        chunk = chunk.reshape(chunk.shape[0], horizon, action_dim)
        if squeeze:
            chunk = chunk[0]
        return chunk

    @classmethod
    def create(cls, seed: int, ex_observations: np.ndarray, config: dict, ex_goals: np.ndarray | None = None):
        rng = jax.random.PRNGKey(int(seed))
        rng, init_rng = jax.random.split(rng)
        ex_obs = jnp.asarray(ex_observations, dtype=jnp.float32)
        ex_goal = None if ex_goals is None else jnp.asarray(ex_goals, dtype=jnp.float32)

        config = dict(config)
        # Accept the legacy key ``spi_goal_conditioning`` from older checkpoints / configs
        # and migrate it to the canonical ``spi_conditioned`` so the rest of the agent only
        # has to look at one key. If both are set, the canonical name wins.
        if 'spi_conditioned' not in config and 'spi_goal_conditioning' in config:
            config['spi_conditioned'] = config['spi_goal_conditioning']
        config['spi_conditioned'] = validate_spi_conditioned(
            config.get('spi_conditioned', 'subgoal')
        )
        config.pop('spi_goal_conditioning', None)

        actor_def = DeterministicChunkActor(
            hidden_dims=(512, 512, 512),
            action_dim=int(config['actor_chunk_horizon']) * int(config['action_dim']),
            layer_norm=bool(config['spi_actor_layer_norm']),
        )
        actor_params = actor_def.init(init_rng, ex_obs, ex_goal)['params']
        actor = TrainState.create(actor_def, actor_params, tx=optax.adam(float(config['lr'])))
        return cls(rng=rng, actor=actor, config=flax.core.FrozenDict(**config))


SPI_CONDITIONED_CHOICES = ('subgoal', 'goal')


def get_actor_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='joint_actor',
            lr=3e-4,
            spi_tau=0.5,
            spi_beta=10.0,
            spi_actor_layer_norm=True,
            spi_q_norm_eps=1e-6,
            # Conditioning vector for both ``π(s, g)`` and ``Q(s, g, a)`` in the SPI loss.
            #   'subgoal' (default): use GOUB ``predict_subgoal(s, g_global)``
            #     → matches training-time subgoal teacher; π/Q see local subgoal.
            #   'goal'   : use the (global) ``high_actor_goals`` directly
            #     → π/Q both see the final goal; SPI prox still pulls toward GOUB
            #       proposal chunks (which were planned to subgoals).
            # Set in ``main._build_actor_batch_from_goub`` via ``spi_goals``.
            # Legacy alias accepted from saved checkpoints: ``spi_goal_conditioning``.
            spi_conditioned='subgoal',
            actor_chunk_horizon=ml_collections.config_dict.placeholder(int),
            action_dim=2,
        )
    )
