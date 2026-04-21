#!/usr/bin/env python3
"""실제 환경에서 GOUB Phase1 플래너 + 청크 저레벨 액터 롤아웃.

예전에는 ``rollout_subgoal_goub.py`` 에 통합돼 있던 모드가 있었습니다. 상태 궤적만 보려면 ``rollout_subgoal_goub.py``, IDM 환경은 ``rollout_idm_goub.py`` 를 쓰면 됩니다.
상태 공간 롤아웃 스크립트를 단순화했으므로 청크 액터는 이 파일을 사용하세요.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from agents.goub_chunk_low import GOUBChunkLowAgent, get_config as get_chunk_config
from agents.goub_phase1 import GOUBPhase1Agent, get_config as get_phase1_config
from rollout_subgoal_goub import _get_trajectory, _goal_distance, _list_checkpoint_suffixes, _load_checkpoint_pkl
from utils.datasets import Dataset
from utils.env_utils import make_env_and_datasets
from utils.goub_rollout_chunk_actor import _summarize_chunk_debug, rollout_chunked_bridge_chunk_actor
from utils.goub_rollout_env import configure_mujoco_gl, format_maze_navigator_log, load_maze_navigator_snap
from utils.goub_rollout_plot import (
    maze_navigator_for_xy_plot,
    overlay_rgb_frames_obs2d_panel,
    write_rgb_array_mp4,
)


def _load_flags_merged(run_dir: Path, config_factory) -> tuple[ConfigDict, str]:
    flags_path = run_dir / 'flags.json'
    if not flags_path.is_file():
        raise FileNotFoundError(f'Missing flags.json under {run_dir}')
    with open(flags_path, 'r', encoding='utf-8') as f:
        flags = json.load(f)
    env_name = flags.get('env_name')
    if not env_name:
        raise KeyError('flags.json must contain env_name')
    cfg = config_factory()
    for k, v in (flags.get('agent') or {}).items():
        cfg[k] = v
    return cfg, env_name


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run_dir', type=str, required=True, help='Phase1 학습 런 (flags.json + checkpoints/).')
    p.add_argument('--chunk_run_dir', type=str, required=True, help='청크 저레벨 액터 학습 런 디렉터리.')
    p.add_argument('--checkpoint_epoch', type=int, default=-1, help='Phase1 params_<n>.pkl 접미사. -1이면 최댓값.')
    p.add_argument('--chunk_checkpoint_epoch', type=int, default=-1, help='청크 액터 체크포인트. -1이면 최댓값.')
    p.add_argument('--traj_idx', type=int, default=0)
    p.add_argument('--max_steps', type=int, default=1000, help='최대 재플랜(청크) 수.')
    p.add_argument('--goal_tol', type=float, default=0.5)
    p.add_argument('--goal_stop', type=str, choices=('plot', 'full'), default='plot')
    p.add_argument('--plot_dim0', type=int, default=0)
    p.add_argument('--plot_dim1', type=int, default=1)
    p.add_argument('--out_mp4', type=str, default='', help='비어 있으면 rollout_chunk_actor_goub.mp4')
    p.add_argument('--fps', type=float, default=60.0)
    p.add_argument('--no_mp4', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--navigator', type=str, choices=('none', 'snap'), default='none')
    p.add_argument('--navigator_clamp', type=str, choices=('ij', 'oracle', 'union', 'center'), default='ij')
    p.add_argument('--navigator_edge_inset', type=float, default=0.08)
    p.add_argument('--maze_type', type=str, default='')
    p.add_argument('--clamp_dim0', type=int, default=-1)
    p.add_argument('--clamp_dim1', type=int, default=-1)
    p.add_argument('--chunk_temperature', type=float, default=1.0)
    p.add_argument('--chunk_stochastic', action='store_true')
    p.add_argument(
        '--mujoco_gl',
        type=str,
        default='',
        metavar='BACKEND',
        help='MuJoCo GL backend (egl|osmesa|glfw). Empty = auto: no DISPLAY → egl.',
    )
    args = p.parse_args()

    try:
        configure_mujoco_gl(str(args.mujoco_gl))
    except ValueError as e:
        p.error(str(e))

    run_dir = Path(args.run_dir).resolve()
    ckpt_dir = run_dir / 'checkpoints'
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f'No checkpoints/ under {run_dir}')
    suffixes = _list_checkpoint_suffixes(ckpt_dir)
    if not suffixes:
        raise FileNotFoundError(f'No params_*.pkl in {ckpt_dir}')
    ckpt_epoch = int(args.checkpoint_epoch)
    if ckpt_epoch < 0:
        ckpt_epoch = suffixes[-1]
    if ckpt_epoch not in suffixes:
        nearest = min(suffixes, key=lambda x: abs(x - ckpt_epoch))
        print(f'Warning: checkpoint suffix {ckpt_epoch} not found; using nearest {nearest}')
        ckpt_epoch = nearest

    cfg, env_name = _load_flags_merged(run_dir, get_phase1_config)
    navigator = None
    if args.navigator == 'snap':
        navigator = load_maze_navigator_snap(args.maze_type, env_name)
        print(format_maze_navigator_log(navigator, str(args.navigator_clamp), float(args.navigator_edge_inset)))

    _, train_raw, _ = make_env_and_datasets(env_name, frame_stack=cfg.get('frame_stack'))
    dataset = Dataset.create(**train_raw)
    dataset_action_norm_mean = float(
        np.linalg.norm(np.asarray(dataset['actions'], dtype=np.float32), axis=-1).mean()
    )

    traj = _get_trajectory(dataset, args.traj_idx)
    s0 = traj[0]
    s_g = traj[-1]
    tol = float(args.goal_tol)
    if tol > 0 and args.goal_stop == 'plot':
        stop_dims: tuple[int, ...] | None = (args.plot_dim0, args.plot_dim1)
    elif tol > 0 and args.goal_stop == 'full':
        stop_dims = None
    else:
        stop_dims = None

    ex = jnp.zeros((1, s0.shape[-1]), dtype=jnp.float32)
    act_dim = int(np.asarray(dataset['actions']).shape[-1])
    ex_act = jnp.zeros((1, act_dim), dtype=jnp.float32)
    agent = GOUBPhase1Agent.create(args.seed, ex, cfg, ex_actions=ex_act)
    pkl_path = ckpt_dir / f'params_{ckpt_epoch}.pkl'
    if not pkl_path.is_file():
        raise FileNotFoundError(f'Missing checkpoint: {pkl_path}')
    agent = _load_checkpoint_pkl(agent, pkl_path)
    goub_N = int(agent.config['goub_N'])
    print(f'Loaded Phase1 {pkl_path}  goub_N={goub_N}')

    c0 = args.clamp_dim0 if args.clamp_dim0 >= 0 else args.plot_dim0
    c1 = args.clamp_dim1 if args.clamp_dim1 >= 0 else args.plot_dim1
    nav_kw = dict(
        navigator=navigator,
        clamp_dim0=c0,
        clamp_dim1=c1,
        navigator_clamp_mode=str(args.navigator_clamp),
        navigator_edge_inset=float(args.navigator_edge_inset),
    )

    chunk_run_dir = Path(str(args.chunk_run_dir).strip()).resolve()
    chunk_ckpt_dir = chunk_run_dir / 'checkpoints'
    if not chunk_ckpt_dir.is_dir():
        raise FileNotFoundError(f'No chunk checkpoints/ under {chunk_run_dir}')
    chunk_suffixes = _list_checkpoint_suffixes(chunk_ckpt_dir)
    if not chunk_suffixes:
        raise FileNotFoundError(f'No params_*.pkl in {chunk_ckpt_dir}')
    chunk_epoch = int(args.chunk_checkpoint_epoch)
    if chunk_epoch < 0:
        chunk_epoch = chunk_suffixes[-1]
    if chunk_epoch not in chunk_suffixes:
        nearest = min(chunk_suffixes, key=lambda x: abs(x - chunk_epoch))
        print(f'Warning: chunk checkpoint suffix {chunk_epoch} not found; using nearest {nearest}')
        chunk_epoch = nearest
    chunk_ckpt_path = chunk_ckpt_dir / f'params_{chunk_epoch}.pkl'
    chunk_cfg, chunk_env_name = _load_flags_merged(chunk_run_dir, get_chunk_config)
    if str(chunk_env_name) != str(env_name):
        raise ValueError(
            f'Chunk actor env_name={chunk_env_name!r} does not match planner env_name={env_name!r}.'
        )
    example_context = np.zeros(
        (1, int(chunk_cfg['chunk_context_horizon']) * len(chunk_cfg['low_goal_slice'])),
        dtype=np.float32,
    )
    example_obs = np.zeros((1, s0.shape[-1]), dtype=np.float32)
    action_dim = int(np.asarray(dataset['actions']).shape[-1])
    example_chunk_actions = np.zeros(
        (1, int(chunk_cfg['chunk_policy_horizon']) * action_dim),
        dtype=np.float32,
    )
    chunk_agent = GOUBChunkLowAgent.create(
        args.seed,
        example_obs,
        example_chunk_actions,
        example_context,
        chunk_cfg,
    )
    chunk_agent = _load_checkpoint_pkl(chunk_agent, chunk_ckpt_path)
    env, _, _ = make_env_and_datasets(env_name, frame_stack=cfg.get('frame_stack'), render_mode='rgb_array')
    env_action_low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    env_action_high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    print(
        f'Loaded chunk actor {chunk_ckpt_path}  '
        f'H_Q={int(chunk_cfg["chunk_context_horizon"])}  '
        f'H_pi={int(chunk_cfg["chunk_policy_horizon"])}  '
        f'C={int(chunk_cfg["chunk_commit_length"])}  '
        f'low_goal_slice={tuple(int(x) for x in chunk_cfg["low_goal_slice"])}'
    )

    max_steps_eff = int(args.max_steps)
    roll, hats, n_planner_steps, reached, env_frames, frame_plan_trajs, chunk_debug = (
        rollout_chunked_bridge_chunk_actor(
            env,
            agent,
            chunk_agent,
            s0,
            s_g,
            max_steps_eff,
            goal_tol=tol,
            goal_stop_dims=stop_dims,
            **nav_kw,
            action_low=env_action_low,
            action_high=env_action_high,
            deterministic=not bool(args.chunk_stochastic),
            temperature=float(args.chunk_temperature),
        )
    )
    n_trans = max(0, int(roll.shape[0]) - 1)
    final_dist = _goal_distance(
        np.asarray(roll[-1], dtype=np.float32),
        np.asarray(s_g, dtype=np.float32),
        stop_dims,
    )
    total_states = roll.shape[0]
    commit_length = int(chunk_agent.config['chunk_commit_length'])
    print(
        f'Chunked GOUB + chunk_actor (env): {n_planner_steps} replans × up to {commit_length} env steps/replan '
        f'(H_Q={int(chunk_agent.config["chunk_context_horizon"])}, '
        f'H_pi={int(chunk_agent.config["chunk_policy_horizon"])}, '
        f'C={commit_length}) → '
        f'{total_states} obs ({total_states - 1} transitions). '
        f'Chunk checkpoint: {chunk_ckpt_path}'
    )
    _summarize_chunk_debug(chunk_debug, dataset_action_norm_mean)

    d0, d1 = args.plot_dim0, args.plot_dim1
    plot_nav = maze_navigator_for_xy_plot(navigator, env_name, d0, d1)
    mp4_env = Path(args.out_mp4.strip()) if str(args.out_mp4).strip() else Path('rollout_chunk_actor_goub.mp4')
    mp4_env.parent.mkdir(parents=True, exist_ok=True)
    if (not args.no_mp4) and env_frames is not None and env_frames.size > 0:
        try:
            _pf = min(int(chunk_agent.config['chunk_commit_length']), int(chunk_agent.config['chunk_policy_horizon']))
            _frames = overlay_rgb_frames_obs2d_panel(
                env_frames,
                traj,
                roll,
                hats,
                frame_plan_trajs,
                s0,
                s_g,
                d0,
                d1,
                plot_nav,
                env_name=env_name,
                chunk_hat_stride=_pf,
            )
            write_rgb_array_mp4(_frames, mp4_env, float(args.fps))
            print(
                f'Wrote env rollout MP4 {mp4_env.resolve()}  '
                f'frames={env_frames.shape[0]}  {env_frames.shape[1]}×{env_frames.shape[2]}'
            )
        except Exception as e:
            print(f'Warning: env MP4 failed ({e!r}). Install: pip install imageio imageio-ffmpeg')
    elif not args.no_mp4:
        print('Warning: no RGB frames captured (env.render).')

    print(
        f'Episode length={len(traj)}  planner_steps={n_planner_steps}  max_steps={max_steps_eff}  '
        f'transitions={n_trans}  goal_reached={reached}  final_dist={final_dist:.4f}  '
        f'goal_tol={tol}  goal_stop={args.goal_stop}  checkpoint_suffix={ckpt_epoch}'
    )
    print(f'dataset_mean_action_norm={dataset_action_norm_mean:.4f}')
    if tol > 0:
        if reached:
            print(f'Goal reached: {n_trans} transition(s), tol={tol:g} ({args.goal_stop})')
        else:
            print(f'Not reached: final goal distance={final_dist:.4f} (tol={tol:g}, {args.goal_stop})')
    else:
        print(
            f'Planner steps: {n_planner_steps} (budget max {max_steps_eff}), '
            f'goal_tol=0 (no early stop)'
        )


if __name__ == '__main__':
    main()
