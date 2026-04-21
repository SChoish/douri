"""Environment / maze helpers for GOUB rollout scripts (state sync, xy snap, navigator setup)."""

from __future__ import annotations

from collections.abc import Callable

import os

import numpy as np


def configure_mujoco_gl(mujoco_gl: str) -> None:
    """Set ``MUJOCO_GL`` before creating MuJoCo-backed envs (headless ``rgb_array`` rollouts).

    If ``mujoco_gl`` is non-empty, it must be one of ``egl``, ``osmesa``, ``glfw``.
    If empty and ``DISPLAY`` is unset, default to ``egl`` so ``env.render()`` does not require X11/GLFW.
    """
    s = (mujoco_gl or '').strip().lower()
    if s:
        if s not in ('egl', 'osmesa', 'glfw'):
            raise ValueError(f'Invalid mujoco_gl={mujoco_gl!r} (use egl, osmesa, glfw, or empty for auto)')
        os.environ['MUJOCO_GL'] = s
    elif not (os.environ.get('DISPLAY') or '').strip():
        os.environ.setdefault('MUJOCO_GL', 'egl')

from utils.maze_navigator import MazeNavigatorMap


def sync_env_state_from_obs_vector(env, obs: np.ndarray, goal_obs: np.ndarray) -> np.ndarray:
    """Set MuJoCo state from flat observation (qpos‖qvel) and maze goal xy from ``goal_obs[:2]``."""
    u = env.unwrapped
    ob = np.asarray(obs, dtype=np.float64).reshape(-1)
    nq, nv = int(u.model.nq), int(u.model.nv)
    need = nq + nv
    if ob.shape[0] < need:
        raise ValueError(f'Observation dim {ob.shape[0]} < nq+nv={need} (cannot replay physics).')
    u.set_state(ob[:nq].copy(), ob[nq:need].copy())
    if hasattr(u, 'set_goal'):
        g_xy = np.asarray(goal_obs[:2], dtype=np.float64).reshape(2)
        u.set_goal(goal_xy=g_xy)
    return np.asarray(u.get_ob(), dtype=np.float32)


def sync_env_state_from_obs_vector_aligned(env, obs: np.ndarray, goal_obs: np.ndarray) -> np.ndarray:
    """Update physics like :func:`sync_env_state_from_obs_vector` and return an obs matching ``env``'s space.

    :class:`utils.env_utils.FrameStackWrapper` keeps a deque of base observations; mutating the base env without
    going through ``reset`` / ``step`` leaves stale frames. When that wrapper is detected, refill the deque with
    the current base ``get_ob()`` repeated ``num_stack`` times (same protocol as ``FrameStackWrapper.reset``).
    """
    sync_env_state_from_obs_vector(env, obs, goal_obs)
    if hasattr(env, 'frames') and hasattr(env, 'num_stack') and hasattr(env, 'get_observation'):
        ob0 = np.asarray(env.unwrapped.get_ob(), dtype=np.float32).reshape(-1)
        env.frames.clear()
        for _ in range(int(env.num_stack)):
            env.frames.append(ob0)
        return np.asarray(env.get_observation(), dtype=np.float32).reshape(-1)
    return np.asarray(env.unwrapped.get_ob(), dtype=np.float32).reshape(-1)


def make_xy_clamper(
    goal_obs: np.ndarray,
    navigator: MazeNavigatorMap | None,
    clamp_dim0: int,
    clamp_dim1: int,
    navigator_clamp_mode: str,
    navigator_edge_inset: float,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a function that optionally snaps ``(clamp_dim0, clamp_dim1)`` xy using ``navigator``."""
    if navigator is None:

        def _identity(vec: np.ndarray) -> np.ndarray:
            return vec

        return _identity

    g_np = np.asarray(goal_obs, dtype=np.float32)

    def _clamp(vec: np.ndarray) -> np.ndarray:
        kw = {'mode': navigator_clamp_mode, 'edge_inset': float(navigator_edge_inset)}
        if navigator_clamp_mode == 'oracle':
            kw['goal_obs'] = g_np
        return navigator.clamp_obs_xy(vec, clamp_dim0, clamp_dim1, **kw)

    return _clamp


def load_maze_navigator_snap(maze_type: str, env_name: str) -> MazeNavigatorMap:
    """Build a :class:`MazeNavigatorMap` when ``--navigator snap`` is enabled."""
    mt = maze_type.strip()
    if mt:
        return MazeNavigatorMap.from_maze_type_embedded(mt)
    try:
        return MazeNavigatorMap.from_env_name(env_name)
    except ValueError as ex:
        raise ValueError(
            f'Could not infer maze type from env_name={env_name!r} ({ex}). '
            'Pass --maze_type= one of arena|medium|large|giant|teleport.'
        ) from ex


def format_maze_navigator_log(
    navigator: MazeNavigatorMap,
    navigator_clamp: str,
    navigator_edge_inset: float,
) -> str:
    ei = float(navigator_edge_inset)
    box_half = 0.5 * float(navigator.maze_unit) * max(0.0, min(1.0, 1.0 - ei))
    return (
        f'Navigator snap enabled (source={navigator.source}, maze_type={navigator.maze_type}, '
        f'clamp={navigator_clamp}, edge_inset={ei}, '
        f'box_half={box_half:.3f}, free cells={len(navigator.free_xy)})'
    )


def env_render_rgb_u8(env) -> np.ndarray | None:
    """Return a single RGB uint8 frame from ``env.render()``, or None if unavailable."""
    try:
        fr = env.render()
    except Exception:
        return None
    if fr is None:
        return None
    x = np.asarray(fr)
    if x.ndim != 3 or x.shape[-1] < 3:
        return None
    x = x[..., :3]
    if x.dtype != np.uint8:
        x = np.clip(x, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(x)
