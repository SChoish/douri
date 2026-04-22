#!/usr/bin/env python3
"""Real-env rollout: GOUB ``predict_subgoal`` + joint chunk actor (same path as ``main._evaluate_env_tasks``).

Loads ``checkpoints/goub/params_<epoch>.pkl`` and ``checkpoints/actor/params_<epoch>.pkl`` from a joint run.

Example::

    MUJOCO_GL=egl python rollout_actor_goub.py \\
        --run_dir=runs/... \\
        --checkpoint_epoch=1000 \\
        --task_id=1 \\
        --max_chunks=300 \\
        --navigator=snap \\
        --out_mp4=runs/.../actor_rollout.mp4
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import flax
import jax.numpy as jnp
import numpy as np

from agents.critic.actor import JointActorAgent, get_actor_config
from agents.goub_dynamics import GOUBDynamicsAgent
from rollout_subgoal_goub import (
    _get_trajectory,
    _list_checkpoint_suffixes,
    _load_checkpoint_pkl,
    _load_run_flags,
    _resolve_goub_checkpoint_dir,
)
from utils.datasets import Dataset
from utils.env_utils import make_env_and_datasets
from utils.flax_utils import merge_checkpoint_state_dict
from utils.goub_rollout_env import (
    configure_mujoco_gl,
    env_render_rgb_u8,
    format_maze_navigator_log,
    load_maze_navigator_snap,
    sync_env_state_from_obs_vector_aligned,
)
from utils.goub_rollout_plot import (
    axis_limits,
    maze_navigator_for_xy_plot,
    overlay_rgb_frames_obs2d_panel,
    write_rgb_array_mp4,
)
from utils.maze_navigator import MazeNavigatorMap


def _resolve_actor_checkpoint_dir(run_dir: Path) -> Path:
    d = run_dir / 'checkpoints' / 'actor'
    if not d.is_dir():
        raise FileNotFoundError(f'Missing actor checkpoints directory: {d}')
    if not _list_checkpoint_suffixes(d):
        raise FileNotFoundError(f'No params_*.pkl under {d}')
    return d


def _load_actor_config_from_flags(flags_path: Path) -> dict:
    with open(flags_path, 'r', encoding='utf-8') as f:
        root = json.load(f)
    act = root.get('actor')
    if not isinstance(act, dict):
        raise KeyError(f'{flags_path} must contain an "actor" object (joint run flags).')
    if not bool(act.get('use_spi_actor', False)):
        raise ValueError('This run has use_spi_actor=false; no joint actor to rollout.')
    base = get_actor_config()
    for k, v in act.items():
        base[k] = v
    return dict(base)


def _load_actor_pkl(agent: JointActorAgent, pkl_path: Path) -> JointActorAgent:
    with open(pkl_path, 'rb') as f:
        load_dict = pickle.load(f)
    template = flax.serialization.to_state_dict(agent)
    merged = merge_checkpoint_state_dict(template, load_dict['agent'])
    return flax.serialization.from_state_dict(agent, merged)


def _align_action_to_env(a: np.ndarray, env_dim: int) -> np.ndarray:
    """Pad or truncate actor output to ``env_dim`` (handles mis-saved ``action_dim`` in flags)."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    if a.shape[-1] == env_dim:
        return a
    if a.shape[-1] < env_dim:
        out = np.zeros((env_dim,), dtype=np.float32)
        out[: int(a.shape[-1])] = a
        return out
    return a[:env_dim].copy()


def rollout_goub_actor_env(
    env,
    goub_agent: GOUBDynamicsAgent,
    actor_agent: JointActorAgent,
    s0: np.ndarray,
    s_g: np.ndarray,
    max_chunks: int,
    goal_tol: float,
    goal_stop_dims: tuple[int, ...] | None,
    *,
    low: np.ndarray,
    high: np.ndarray,
    actor_horizon: int,
    env_action_dim: int,
    record_env_rgb: bool = True,
) -> tuple[np.ndarray, np.ndarray, int, bool, np.ndarray | None]:
    cur = sync_env_state_from_obs_vector_aligned(env, s0, s_g)
    goal = np.asarray(s_g, dtype=np.float32).reshape(-1)
    states: list[np.ndarray] = [cur.copy()]
    hats_list: list[np.ndarray] = []
    rgb_frames: list[np.ndarray] = []

    def _maybe_record() -> None:
        if not record_env_rgb:
            return
        fr = env_render_rgb_u8(env)
        if fr is not None:
            rgb_frames.append(fr)

    _maybe_record()

    from rollout_subgoal_goub import _goal_within_tol

    if _goal_within_tol(cur, goal, goal_stop_dims, float(goal_tol)):
        return np.stack(states, axis=0), np.zeros((0, cur.shape[-1]), dtype=np.float32), 0, True, (
            np.stack(rgb_frames, axis=0) if rgb_frames else None
        )

    reached = False
    terminated = False
    truncated = False
    n_chunks_used = 0
    for _ in range(max(1, int(max_chunks))):
        obs = states[-1]
        if _goal_within_tol(obs, goal, goal_stop_dims, float(goal_tol)):
            reached = True
            break
        pred = np.asarray(goub_agent.predict_subgoal(obs, goal), dtype=np.float32).reshape(-1)
        chunk = np.asarray(actor_agent.sample_actions(obs, pred), dtype=np.float32).reshape(actor_horizon, -1)
        chunk_done = False
        for _i in range(int(chunk.shape[0])):
            a = _align_action_to_env(chunk[_i], env_action_dim)
            a = np.clip(a, low, high)
            ob, _r, term, trunc, info = env.step(a)
            obs = np.asarray(ob, dtype=np.float32).reshape(-1)
            states.append(obs.copy())
            hats_list.append(pred.copy())
            _maybe_record()
            succ_flag = bool(info.get('success', False)) if isinstance(info, dict) else False
            reached = succ_flag or _goal_within_tol(obs, goal, goal_stop_dims, float(goal_tol))
            terminated = bool(term)
            truncated = bool(trunc)
            if reached or terminated or truncated:
                chunk_done = True
                break
        n_chunks_used += 1
        if chunk_done:
            break

    hats = np.stack(hats_list, axis=0) if hats_list else np.zeros((0, states[0].shape[-1]), dtype=np.float32)
    env_rgb = np.stack(rgb_frames, axis=0) if rgb_frames else None
    return np.stack(states, axis=0), hats, n_chunks_used, reached, env_rgb


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run_dir', type=str, required=True)
    p.add_argument(
        '--checkpoint_epoch',
        type=int,
        default=1000,
        help='Suffix for both goub/params_<n>.pkl and actor/params_<n>.pkl (nearest match if missing).',
    )
    p.add_argument('--goub_epoch', type=int, default=-1, help='Override GOUB checkpoint suffix (-1 = same as --checkpoint_epoch).')
    p.add_argument('--actor_epoch', type=int, default=-1, help='Override actor checkpoint suffix (-1 = same as --checkpoint_epoch).')
    p.add_argument('--traj_idx', type=int, default=0, help='Offline episode index when --task_id=0.')
    p.add_argument('--task_id', type=int, default=0, help='OGBench task id [1..num_tasks]; 0 uses offline traj start/goal.')
    p.add_argument('--max_chunks', type=int, default=1000, help='Max replan rounds (same sense as training eval).')
    p.add_argument('--goal_tol', type=float, default=0.5)
    p.add_argument('--goal_stop', type=str, choices=('plot', 'full'), default='plot')
    p.add_argument('--plot_dim0', type=int, default=0)
    p.add_argument('--plot_dim1', type=int, default=1)
    p.add_argument('--navigator', type=str, choices=('none', 'snap'), default='none')
    p.add_argument('--navigator_clamp', type=str, choices=('ij', 'oracle', 'union', 'center'), default='ij')
    p.add_argument('--navigator_edge_inset', type=float, default=0.08)
    p.add_argument('--maze_type', type=str, default='')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_path', type=str, default='rollout_actor_goub.png')
    p.add_argument('--out_mp4', type=str, default='')
    p.add_argument('--fps', type=float, default=60.0)
    p.add_argument('--no_mp4', action='store_true')
    p.add_argument('--mujoco_gl', type=str, default='', metavar='BACKEND')
    p.add_argument(
        '--value_heatmap',
        action='store_true',
        help='Overlay DQC scalar value V(s, goal) on the right XY panel (joint checkpoints/critic/).',
    )
    p.add_argument('--value_grid_n', type=int, default=56)
    p.add_argument('--critic_epoch', type=int, default=-1, help='Critic checkpoint suffix; -1 = GOUB epoch used.')
    args = p.parse_args()

    try:
        configure_mujoco_gl(str(args.mujoco_gl))
    except ValueError as e:
        p.error(str(e))

    run_dir = Path(args.run_dir).resolve()
    flags_path = run_dir / 'flags.json'
    goub_ckpt_dir = _resolve_goub_checkpoint_dir(run_dir)
    actor_ckpt_dir = _resolve_actor_checkpoint_dir(run_dir)

    def _pick_epoch(requested: int, suffixes: list[int]) -> int:
        if requested < 0:
            return suffixes[-1]
        if requested not in suffixes:
            nearest = min(suffixes, key=lambda x: abs(x - requested))
            print(f'Warning: checkpoint {requested} not found; using {nearest}')
            return nearest
        return requested

    goub_suf = _list_checkpoint_suffixes(goub_ckpt_dir)
    actor_suf = _list_checkpoint_suffixes(actor_ckpt_dir)
    base_ep = int(args.checkpoint_epoch)
    goub_ep = int(args.goub_epoch) if int(args.goub_epoch) >= 0 else base_ep
    actor_ep = int(args.actor_epoch) if int(args.actor_epoch) >= 0 else base_ep
    goub_ep = _pick_epoch(goub_ep, goub_suf)
    actor_ep = _pick_epoch(actor_ep, actor_suf)

    goub_cfg, env_name = _load_run_flags(run_dir)
    actor_cfg = _load_actor_config_from_flags(flags_path)

    navigator: MazeNavigatorMap | None = None
    if args.navigator == 'snap':
        navigator = load_maze_navigator_snap(args.maze_type, env_name)
        print(format_maze_navigator_log(navigator, str(args.navigator_clamp), float(args.navigator_edge_inset)))

    env, train_raw, _ = make_env_and_datasets(
        env_name, frame_stack=goub_cfg.get('frame_stack'), render_mode='rgb_array',
    )
    tid = int(args.task_id)
    if tid != 0:
        u = env.unwrapped
        n_tasks = int(getattr(u, 'num_tasks', 5))
        if not (1 <= tid <= n_tasks):
            p.error(f'--task_id must be in [1, {n_tasks}] for {env_name!r} (got {tid})')
        ob, info = env.reset(options=dict(task_id=tid, render_goal=False))
        if 'goal' not in info:
            raise RuntimeError(f'{env_name!r} reset did not set info["goal"].')
        s0 = np.asarray(ob, dtype=np.float32).reshape(-1)
        s_g = np.asarray(info['goal'], dtype=np.float32).reshape(-1)
        traj = np.stack([s0, s_g], axis=0)
        print(f'OGBench eval reset: task_id={tid}  obs_dim={s0.shape[-1]}')
    else:
        dataset = Dataset.create(**train_raw)
        traj = _get_trajectory(dataset, int(args.traj_idx))
        s0 = np.asarray(traj[0], dtype=np.float32).reshape(-1)
        s_g = np.asarray(traj[-1], dtype=np.float32).reshape(-1)

    tol = float(args.goal_tol)
    if tol > 0 and args.goal_stop == 'plot':
        stop_dims: tuple[int, ...] | None = (int(args.plot_dim0), int(args.plot_dim1))
    elif tol > 0 and args.goal_stop == 'full':
        stop_dims = None
    else:
        stop_dims = None

    ex = jnp.zeros((1, s0.shape[-1]), dtype=jnp.float32)
    act_dim = int(np.prod(env.action_space.shape))
    saved_actor_dim = int(actor_cfg.get('action_dim', act_dim))
    actor_cfg['action_dim'] = act_dim
    ex_act = jnp.zeros((1, act_dim), dtype=jnp.float32)
    goub_agent = GOUBDynamicsAgent.create(int(args.seed), ex, goub_cfg, ex_actions=ex_act)
    goub_pkl = goub_ckpt_dir / f'params_{goub_ep}.pkl'
    goub_agent = _load_checkpoint_pkl(goub_agent, goub_pkl)
    print(f'Loaded GOUB {goub_pkl}')

    ex_goal = jnp.asarray(s_g.reshape(1, -1), dtype=jnp.float32)
    actor_agent = JointActorAgent.create(int(args.seed), ex, actor_cfg, ex_goals=ex_goal)
    actor_pkl = actor_ckpt_dir / f'params_{actor_ep}.pkl'
    actor_agent = _load_actor_pkl(actor_agent, actor_pkl)
    print(f'Loaded actor {actor_pkl}')

    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    env_action_dim = int(low.shape[-1])
    actor_horizon = int(actor_cfg['actor_chunk_horizon'])
    if saved_actor_dim != env_action_dim:
        print(
            f'Corrected flags actor action_dim={saved_actor_dim} to env action dim={env_action_dim} '
            'before loading the actor checkpoint.'
        )

    roll, hats, n_chunks, reached, env_frames = rollout_goub_actor_env(
        env,
        goub_agent,
        actor_agent,
        s0,
        s_g,
        int(args.max_chunks),
        goal_tol=tol,
        goal_stop_dims=stop_dims,
        low=low,
        high=high,
        actor_horizon=actor_horizon,
        env_action_dim=env_action_dim,
        record_env_rgb=True,
    )
    n_trans = max(0, int(roll.shape[0]) - 1)
    print(
        f'Actor rollout: {n_chunks} replan rounds, actor_horizon={actor_horizon} → '
        f'{roll.shape[0]} obs ({n_trans} transitions), goal_reached={reached}'
    )

    d0, d1 = int(args.plot_dim0), int(args.plot_dim1)
    plot_nav = maze_navigator_for_xy_plot(navigator, env_name, d0, d1)

    heat_mesh = None
    heat_vmin = heat_vmax = None
    if bool(args.value_heatmap):
        from utils.rollout_value_field import dqc_value_mesh_for_xy, load_dqc_critic_joint_run

        ce = int(args.critic_epoch) if int(args.critic_epoch) >= 0 else int(goub_ep)
        critic_agent = load_dqc_critic_joint_run(
            run_dir,
            ce,
            env,
            train_raw,
            seed=int(args.seed),
        )
        print(f'Loaded critic for value heatmap (epoch suffix {ce})')
        xlim, ylim = axis_limits(traj, roll, hats, d0, d1, s_g, s0, navigator=plot_nav, seg=None)
        tpl = np.asarray(roll[0], dtype=np.float32).reshape(-1)
        XX, YY, ZZ, heat_vmin, heat_vmax = dqc_value_mesh_for_xy(
            critic_agent,
            tpl,
            np.asarray(s_g, dtype=np.float32).reshape(-1),
            int(d0),
            int(d1),
            xlim,
            ylim,
            grid_n=int(args.value_grid_n),
        )
        heat_mesh = (XX, YY, ZZ)

    if (not args.no_mp4) and env_frames is not None and env_frames.size > 0:
        mp4_out = Path(args.out_mp4.strip()) if str(args.out_mp4).strip() else Path(args.out_path).with_suffix('.mp4')
        mp4_out.parent.mkdir(parents=True, exist_ok=True)
        try:
            frames = overlay_rgb_frames_obs2d_panel(
                env_frames,
                traj,
                roll,
                hats,
                None,
                s0,
                s_g,
                d0,
                d1,
                plot_nav,
                env_name=env_name,
                chunk_hat_stride=actor_horizon,
                value_heatmap=heat_mesh,
                value_heatmap_vmin=heat_vmin,
                value_heatmap_vmax=heat_vmax,
            )
            write_rgb_array_mp4(frames, mp4_out, float(args.fps))
            print(f'Wrote MP4 {mp4_out.resolve()}')
        except Exception as e:
            print(f'Warning: MP4 failed ({e!r}). pip install imageio imageio-ffmpeg pillow')
    elif not args.no_mp4:
        print('Warning: no RGB frames; check render_mode')


if __name__ == '__main__':
    main()

