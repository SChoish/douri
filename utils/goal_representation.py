"""Goal representation helpers shared by goal-conditioned networks."""

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp

_MANIP_ARM_JOINT_DIM = 6
_MANIP_HEAD_DIM = 2 * _MANIP_ARM_JOINT_DIM + 3 + 1 + 1 + 1 + 1
_MANIP_CUBE_STRIDE = 3 + 4 + 1 + 1


def manip_cube_pos_indices(obs_dim: int) -> tuple[int, ...]:
    """Return compact ManipSpace cube-position channels for one observation frame."""

    dim = int(obs_dim)
    rem = dim - _MANIP_HEAD_DIM
    if rem < _MANIP_CUBE_STRIDE or rem % _MANIP_CUBE_STRIDE != 0:
        return ()
    idxs: list[int] = []
    for start in range(_MANIP_HEAD_DIM, dim, _MANIP_CUBE_STRIDE):
        idxs.extend((start, start + 1, start + 2))
    return tuple(idxs)


def normalize_phi_goal_obs_indices(raw: object) -> tuple[int, ...]:
    """Parse YAML / CLI values into a tuple of non-negative ints (may be empty)."""

    if raw is None:
        return ()
    if isinstance(raw, (list, tuple)):
        return tuple(int(x) for x in raw)
    raise TypeError(f'phi_goal_obs_indices must be a list/tuple of ints, got {type(raw).__name__}')


def assert_phi_goal_obs_indices(
    obs_dim: int,
    mode: str,
    phi_goal_obs_indices: Sequence[int] | tuple[int, ...],
    *,
    where: str,
) -> None:
    """Require explicit phi indices for non-ManipSpace observations when mode uses phi."""

    mode_l = str(mode).lower()
    if mode_l in ('full', 'raw', 'none', ''):
        return
    if mode_l not in ('phi', 'auto', 'goal_phi'):
        return
    dim = int(obs_dim)
    if manip_cube_pos_indices(dim):
        return
    idxs = tuple(int(x) for x in phi_goal_obs_indices)
    if not idxs:
        raise ValueError(
            f'{where}: goal_representation={mode!r} with obs_dim={dim} requires '
            'critic_agent.phi_goal_obs_indices (e.g. [0, 1] for planar x,y in the '
            'goal observation). Implicit [:2] slicing is disabled.'
        )
    for i in idxs:
        if i < 0 or i >= dim:
            raise ValueError(
                f'{where}: phi_goal_obs_indices={idxs!r} out of range for obs_dim={dim}.'
            )


def goal_representation(
    goals: jnp.ndarray | None,
    mode: str,
    phi_goal_obs_indices: Sequence[int] | tuple[int, ...] = (),
) -> jnp.ndarray | None:
    """Map a full goal state to the configured goal representation.

    ``full`` keeps historical behavior. ``phi`` / ``auto`` / ``goal_phi``:
    ManipSpace compact observations use inferred cube-position channels;
    otherwise ``phi_goal_obs_indices`` must list observation indices (e.g.
    ``(0, 1)`` for maze-style x,y). There is no implicit ``goals[..., :2]`` fallback.
    """

    if goals is None:
        return None
    mode_l = str(mode).lower()
    if mode_l in ('full', 'raw', 'none', ''):
        return goals
    if mode_l not in ('phi', 'auto', 'goal_phi'):
        raise ValueError(
            f"Unknown goal_representation={mode!r}; expected 'full' or 'phi'."
        )
    obs_dim = int(goals.shape[-1])
    idxs = manip_cube_pos_indices(obs_dim)
    if not idxs:
        idxs = tuple(int(x) for x in phi_goal_obs_indices)
        if not idxs:
            raise ValueError(
                f'goal_representation={mode_l!r} requires critic_agent.phi_goal_obs_indices for '
                f'obs_dim={obs_dim} (non-ManipSpace).'
            )
        for i in idxs:
            if i < 0 or i >= obs_dim:
                raise ValueError(f'phi_goal_obs_indices={idxs!r} out of range for obs_dim={obs_dim}.')
    take = jnp.asarray(idxs, dtype=jnp.int32)
    return jnp.take(goals, take, axis=-1)
