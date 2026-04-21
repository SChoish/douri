"""Real-env rollout: chunked GOUB planner + low-level chunk action policy."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from agents.goub_phase1 import GOUBPhase1Agent
from utils.goub_rollout_env import (
    env_render_rgb_u8,
    make_xy_clamper,
    sync_env_state_from_obs_vector_aligned,
)
from utils.maze_navigator import MazeNavigatorMap


def _goal_distance(s: np.ndarray, g: np.ndarray, stop_dims: tuple[int, ...] | None) -> float:
    if stop_dims:
        idx = np.asarray(stop_dims, dtype=np.int32)
        return float(np.linalg.norm(s[idx] - g[idx]))
    return float(np.linalg.norm(s - g))


def _build_plan_context_from_trajectory(
    trajectory: np.ndarray,
    current_state: np.ndarray,
    context_horizon: int,
    low_goal_slice: tuple[int, ...],
    use_relative_context: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build planner-friendly local context from the first planned future states."""
    current_state = np.asarray(current_state, dtype=np.float32).reshape(-1)
    trajectory = np.asarray(trajectory, dtype=np.float32)
    if trajectory.ndim != 2:
        raise ValueError(f'Expected trajectory shape (T, D), got {trajectory.shape}')
    if trajectory.shape[0] < context_horizon + 1:
        raise ValueError(
            f'Plan trajectory too short for context_horizon={context_horizon}: got {trajectory.shape[0]} states'
        )

    idx = np.asarray(low_goal_slice, dtype=np.int32)
    planned_states = trajectory[1 : context_horizon + 1]
    deltas = planned_states[:, idx]
    if use_relative_context:
        deltas = deltas - current_state[idx][None, :]
    return deltas.reshape(-1), deltas


def _summarize_chunk_debug(debug: dict[str, np.ndarray], dataset_action_norm_mean: float | None = None) -> None:
    if not debug:
        return
    planner_context_norms = np.asarray(debug.get('planner_context_norms', []), dtype=np.float32)
    chunk_action_norms = np.asarray(debug.get('chunk_action_norms', []), dtype=np.float32)
    first_action_norms = np.asarray(debug.get('first_action_norms', []), dtype=np.float32)
    local_plan_delta_norms = np.asarray(debug.get('local_plan_delta_norms', []), dtype=np.float32)

    if planner_context_norms.size > 0:
        print(
            'planner_context_norm'
            f' mean={planner_context_norms.mean():.4f} max={planner_context_norms.max():.4f}'
        )
    if chunk_action_norms.size > 0:
        print(f'chunk_action_norm mean={chunk_action_norms.mean():.4f} max={chunk_action_norms.max():.4f}')
    if first_action_norms.size > 0:
        msg = f'first_action_norm mean={first_action_norms.mean():.4f} max={first_action_norms.max():.4f}'
        if dataset_action_norm_mean is not None and dataset_action_norm_mean > 0:
            msg += f'  ratio_to_dataset_mean={first_action_norms.mean() / dataset_action_norm_mean:.4f}'
        print(msg)
    if local_plan_delta_norms.size > 0:
        mean_delta_norms = local_plan_delta_norms.mean(axis=0)
        for i, v in enumerate(mean_delta_norms, start=1):
            print(f'local_plan_delta_norm[{i}] mean={float(v):.4f}')


def rollout_chunked_bridge_chunk_actor(
    env,
    agent: GOUBPhase1Agent,
    chunk_agent,
    s0: np.ndarray,
    s_g: np.ndarray,
    max_chunks: int,
    goal_tol: float = 0.0,
    goal_stop_dims: tuple[int, ...] | None = None,
    navigator: MazeNavigatorMap | None = None,
    clamp_dim0: int = 0,
    clamp_dim1: int = 1,
    navigator_clamp_mode: str = 'ij',
    navigator_edge_inset: float = 0.08,
    *,
    action_low: np.ndarray,
    action_high: np.ndarray,
    deterministic: bool = True,
    temperature: float = 1.0,
    record_env_rgb: bool = True,
) -> tuple[np.ndarray, np.ndarray, int, bool, np.ndarray | None, np.ndarray | None, dict[str, np.ndarray]]:
    """Chunked GOUB bridge + chunk actor rollout in the real environment."""
    g_np = np.asarray(s_g, dtype=np.float32)
    low = np.asarray(action_low, dtype=np.float32).reshape(-1)
    high = np.asarray(action_high, dtype=np.float32).reshape(-1)
    xy_clamper = make_xy_clamper(
        g_np, navigator, clamp_dim0, clamp_dim1, navigator_clamp_mode, navigator_edge_inset
    )

    context_horizon = int(chunk_agent.config['chunk_context_horizon'])
    policy_horizon = int(chunk_agent.config['chunk_policy_horizon'])
    commit_length = int(chunk_agent.config['chunk_commit_length'])
    low_goal_slice = tuple(int(x) for x in chunk_agent.config['low_goal_slice'])
    use_relative_context = bool(chunk_agent.config.get('chunk_use_relative_context', True))

    cur = sync_env_state_from_obs_vector_aligned(env, s0, s_g)
    d = int(cur.shape[-1])
    states: list[np.ndarray] = [cur.copy()]
    hats_list: list[np.ndarray] = []
    rgb_frames: list[np.ndarray] = []
    frame_plan_trajs: list[np.ndarray] = []
    debug: dict[str, list] = {
        'planner_context_norms': [],
        'chunk_action_norms': [],
        'first_action_norms': [],
        'local_plan_delta_norms': [],
    }

    def _pack_frame_plan_trajs() -> np.ndarray:
        if frame_plan_trajs:
            return np.stack(frame_plan_trajs, axis=0)
        return np.zeros((len(states), 0, d), dtype=np.float32)

    def _maybe_record() -> None:
        if not record_env_rgb:
            return
        fr = env_render_rgb_u8(env)
        if fr is not None:
            rgb_frames.append(fr)

    _maybe_record()

    use_tol = goal_tol is not None and float(goal_tol) > 0.0
    if use_tol and _goal_distance(cur, g_np, goal_stop_dims) <= float(goal_tol):
        hats = np.zeros((0, d), dtype=np.float32)
        env_rgb = np.stack(rgb_frames, axis=0) if rgb_frames else None
        return np.stack(states, axis=0), hats, 0, True, env_rgb, _pack_frame_plan_trajs(), {
            k: np.asarray(v, dtype=np.float32) for k, v in debug.items()
        }

    for chunk_i in range(max_chunks):
        s_np = np.asarray(states[-1], dtype=np.float32).reshape(-1)
        s = jnp.asarray(xy_clamper(s_np), dtype=jnp.float32)
        g = jnp.asarray(s_g, dtype=jnp.float32)
        hat = agent.predict_subgoal(s, g)
        hat_np = xy_clamper(np.asarray(jax.device_get(hat), dtype=np.float32).reshape(-1))
        hat = jnp.asarray(hat_np, dtype=jnp.float32)

        out = agent.plan(s, hat)
        chunk_traj = np.asarray(jax.device_get(out['trajectory']), dtype=np.float32)
        if chunk_traj.shape[0] < 2:
            break
        chunk_traj = np.stack([xy_clamper(chunk_traj[i]) for i in range(chunk_traj.shape[0])])
        if chunk_traj.shape[0] < context_horizon + 1:
            raise ValueError(
                f'Planner trajectory length {chunk_traj.shape[0]} is shorter than context horizon {context_horizon + 1}'
            )
        plan_seg = np.asarray(chunk_traj[1:], dtype=np.float32)
        if len(frame_plan_trajs) < len(states):
            frame_plan_trajs.append(plan_seg.copy())

        local_plan_context, local_plan_deltas = _build_plan_context_from_trajectory(
            chunk_traj,
            s_np,
            context_horizon=context_horizon,
            low_goal_slice=low_goal_slice,
            use_relative_context=use_relative_context,
        )
        debug['planner_context_norms'].append(float(np.linalg.norm(local_plan_context)))
        debug['local_plan_delta_norms'].append(np.linalg.norm(local_plan_deltas, axis=-1).astype(np.float32))

        action_chunk = chunk_agent.sample_action_chunk(
            jnp.asarray(s_np, dtype=jnp.float32),
            jnp.asarray(local_plan_context, dtype=jnp.float32),
            g,
            seed=(jax.random.PRNGKey(chunk_i) if not deterministic else None),
            deterministic=deterministic,
            temperature=float(temperature),
        )
        actions = np.asarray(jax.device_get(action_chunk), dtype=np.float32).reshape(policy_horizon, -1)
        debug['chunk_action_norms'].append(float(np.linalg.norm(actions)))
        debug['first_action_norms'].append(float(np.linalg.norm(actions[0])))

        n_exec = min(int(actions.shape[0]), max(1, commit_length))
        for i in range(n_exec):
            hats_list.append(hat_np.copy())
            a = np.clip(actions[i], low, high)
            ob, _r, terminated, truncated, _info = env.step(a)
            ob_f = np.asarray(ob, dtype=np.float32).reshape(-1)
            states.append(ob_f)
            frame_plan_trajs.append(plan_seg.copy())
            _maybe_record()
            if use_tol and _goal_distance(ob_f, g_np, goal_stop_dims) <= float(goal_tol):
                hats = np.stack(hats_list, axis=0) if hats_list else np.zeros((0, d), dtype=np.float32)
                env_rgb = np.stack(rgb_frames, axis=0) if rgb_frames else None
                return np.stack(states, axis=0), hats, chunk_i + 1, True, env_rgb, _pack_frame_plan_trajs(), {
                    k: np.asarray(v, dtype=np.float32) for k, v in debug.items()
                }
            if terminated or truncated:
                hats = np.stack(hats_list, axis=0) if hats_list else np.zeros((0, d), dtype=np.float32)
                env_rgb = np.stack(rgb_frames, axis=0) if rgb_frames else None
                return np.stack(states, axis=0), hats, chunk_i + 1, False, env_rgb, _pack_frame_plan_trajs(), {
                    k: np.asarray(v, dtype=np.float32) for k, v in debug.items()
                }

    hats = np.stack(hats_list, axis=0) if hats_list else np.zeros((0, d), dtype=np.float32)
    env_rgb = np.stack(rgb_frames, axis=0) if rgb_frames else None
    return np.stack(states, axis=0), hats, max_chunks, False, env_rgb, _pack_frame_plan_trajs(), {
        k: np.asarray(v, dtype=np.float32) for k, v in debug.items()
    }
