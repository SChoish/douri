"""Inference-time subgoal filters."""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from agents.critic import ScalarValueNet
from utils.goal_representation import manip_cube_pos_indices, normalize_phi_goal_obs_indices


SubgoalFilterFn = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def _as_row(x: np.ndarray | jnp.ndarray) -> jnp.ndarray:
    arr = jnp.asarray(x, dtype=jnp.float32)
    return arr.reshape(1, -1) if arr.ndim == 1 else arr


def _manip_cube_pos_indices(obs_dim: int) -> tuple[int, ...]:
    """Return compact ManipSpace cube position channels for a single frame.

    OGBench ManipSpace compact obs starts with UR5 arm/gripper channels, then
    repeats ``cube_pos(3), cube_quat(4), cos_yaw, sin_yaw`` for each cube. When
    value filtering rejects a subgoal, we only copy these cube position channels
    from the final goal, leaving arm/gripper and cube orientation untouched.
    """

    return manip_cube_pos_indices(obs_dim)


def _filter_replacement_target(
    subgoal_b: jnp.ndarray, goal_b: jnp.ndarray, phi_goal_obs_indices: tuple[int, ...],
) -> jnp.ndarray:
    idxs = _manip_cube_pos_indices(int(subgoal_b.shape[-1]))
    if not idxs:
        idxs = phi_goal_obs_indices
        if not idxs:
            raise ValueError(
                'Subgoal filter replacement needs ManipSpace cube layout or '
                'critic_agent.phi_goal_obs_indices for non-ManipSpace observations.'
            )
    target = subgoal_b
    idx_arr = jnp.asarray(idxs, dtype=jnp.int32)
    return target.at[:, idx_arr].set(goal_b[:, idx_arr])


def make_value_subgoal_filter_from_params(
    critic_config: Any,
    critic_value_params: Any | None,
    reachability_threshold: float = 0.5,
) -> SubgoalFilterFn | None:
    """Return a filter that replaces bad subgoal components with the final goal.

    The filter compares sigmoid-transformed scalar values under the critic
    value head. If ``V(subgoal, g) <= V(s, g)`` and ``V(s, subgoal) > R``, only
    the goal representation ``phi`` channels listed in
    ``critic_agent.phi_goal_obs_indices`` are copied from ``g`` (ManipSpace:
    inferred cube position channels). Other channels in the predicted subgoal
    are kept unchanged.
    """

    if critic_value_params is None:
        return None
    phi_idxs = normalize_phi_goal_obs_indices(critic_config.get('phi_goal_obs_indices', ()))
    value_def = ScalarValueNet(
        tuple(int(x) for x in critic_config['value_hidden_dims']),
        layer_norm=bool(critic_config.get('layer_norm', True)),
        goal_representation=str(critic_config.get('goal_representation', 'full')),
        phi_goal_obs_indices=phi_idxs,
    )

    @jax.jit
    def _filter(obs: jnp.ndarray, subgoal: jnp.ndarray, goal: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        obs_b = _as_row(obs)
        sg_b = _as_row(subgoal)
        goal_b = _as_row(goal)
        cur_to_goal_v = jax.nn.sigmoid(value_def.apply({'params': critic_value_params}, obs_b, goal_b))
        sg_to_goal_v = jax.nn.sigmoid(value_def.apply({'params': critic_value_params}, sg_b, goal_b))
        obs_to_sg_v = jax.nn.sigmoid(value_def.apply({'params': critic_value_params}, obs_b, sg_b))
        replacement = _filter_replacement_target(sg_b, goal_b, phi_idxs)
        should_filter = (sg_to_goal_v <= cur_to_goal_v) & (obs_to_sg_v > float(reachability_threshold))
        filtered = jnp.where(should_filter.reshape((-1, 1)), replacement, sg_b).reshape(subgoal.shape)
        return filtered, should_filter

    stats = {'filtered': 0, 'total': 0}

    def _reset_stats() -> None:
        stats['filtered'] = 0
        stats['total'] = 0

    def _get_stats() -> dict[str, int]:
        return dict(stats)

    def _call(obs: np.ndarray, subgoal: np.ndarray, goal: np.ndarray) -> np.ndarray:
        filtered, should_filter = _filter(
            jnp.asarray(obs, dtype=jnp.float32),
            jnp.asarray(subgoal, dtype=jnp.float32),
            jnp.asarray(goal, dtype=jnp.float32),
        )
        should_filter_np = np.asarray(jax.device_get(should_filter), dtype=bool).reshape(-1)
        stats['filtered'] += int(np.count_nonzero(should_filter_np))
        stats['total'] += int(should_filter_np.size)
        return np.asarray(jax.device_get(filtered), dtype=np.float32).reshape(np.asarray(subgoal).shape)

    _call.reset_stats = _reset_stats  # type: ignore[attr-defined]
    _call.get_stats = _get_stats  # type: ignore[attr-defined]
    return _call


def make_value_subgoal_filter_from_critic_agent(
    critic_agent: Any | None,
    reachability_threshold: float = 0.5,
) -> SubgoalFilterFn | None:
    """Build the same value filter from a loaded ``CriticAgent``."""

    if critic_agent is None:
        return None
    return make_value_subgoal_filter_from_params(
        critic_agent.config,
        critic_agent.network.params.get('modules_value', None),
        reachability_threshold=reachability_threshold,
    )
