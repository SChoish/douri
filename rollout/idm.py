#!/usr/bin/env python3
"""Real-environment rollout: GOUB planner + inverse dynamics (IDM).

This script is **separate** from ``rollout/subgoal.py`` (pure state-space trajectory plots).
Each replan: ``predict_subgoal`` → ``plan`` or stochastic ``sample_plan`` (see
``--planner_noise_scale``) → IDM actions for up to ``--action_chunk_horizon`` env steps
(capped at ``goub_N``), then repeat.

Headless RGB: if ``DISPLAY`` is unset, ``MUJOCO_GL`` defaults to ``egl`` (see ``--mujoco_gl``).

Example::

    cd <douri repo root>
    python rollout/idm.py \\
        --run_dir=runs/... \\
        --checkpoint_epoch=1000 \\
        --max_steps=1000 \\
        --out_path=idm_rollout.png

``--idm_checkpoint`` is optional: when omitted, weights are read from ``idm_net`` inside the GOUB
``params_<epoch>.pkl``. Use a standalone IDM pickle only for older GOUB runs without ``idm_net``.

OGBench-style evaluation goals: pass ``--task_id`` in ``[1, num_tasks]`` (typically 5). The script
calls ``env.reset(options=dict(task_id=..., render_goal=False))`` and uses the returned observation
as ``s0`` and ``info['goal']`` as ``s_g``.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from agents.goub_dynamics import GOUBDynamicsAgent
from utils.env_utils import make_env_and_datasets
from utils.run_io import (
    goal_within_tol,
    list_checkpoint_suffixes,
    load_checkpoint_pkl,
    load_run_flags,
    pick_epoch,
    resolve_goub_checkpoint_dir,
)
from rollout.env import (
    configure_mujoco_gl,
    env_render_rgb_u8,
    format_maze_navigator_log,
    load_maze_navigator_snap,
    make_xy_clamper,
    sync_env_state_from_obs_vector_aligned,
)
from rollout.plot import (
    axis_limits,
    maze_navigator_for_xy_plot,
    overlay_rgb_frames_obs2d_panel,
    write_rgb_array_mp4,
)
from utils.inverse_dynamics import InverseDynamicsMLP
from rollout.maze_navigator import MazeNavigatorMap


def rollout_goub_idm_env(
    env,
    agent: GOUBDynamicsAgent,
    idm_model: InverseDynamicsMLP,
    idm_params,
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
    action_chunk_horizon: int = 5,
    record_env_rgb: bool = True,
    planner_noise_scale: float = 0.0,
    planner_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, int, bool, np.ndarray | None, np.ndarray]:
    """Chunked GOUB bridge + inverse dynamics in the real environment.

    Note: ``navigator`` / ``xy_clamper`` are used only for visualization outputs
    (subgoal markers and planned state-space trajectories). The actual rollout
    policy runs on the raw environment observations / GOUB predictions so this
    path stays aligned with training-time evaluation in ``main.py``.
    """
    g_np = np.asarray(s_g, dtype=np.float32)
    low = np.asarray(action_low, dtype=np.float32).reshape(-1)
    high = np.asarray(action_high, dtype=np.float32).reshape(-1)
    xy_clamper = make_xy_clamper(
        g_np, navigator, clamp_dim0, clamp_dim1, navigator_clamp_mode, navigator_edge_inset
    )

    cur = sync_env_state_from_obs_vector_aligned(env, s0, s_g)
    d = int(cur.shape[-1])
    states: list[np.ndarray] = [cur.copy()]
    hats_list: list[np.ndarray] = []
    rgb_frames: list[np.ndarray] = []
    frame_plan_trajs: list[np.ndarray] = []

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

    if goal_within_tol(cur, g_np, goal_stop_dims, float(goal_tol)):
        hats = np.zeros((0, d), dtype=np.float32)
        env_rgb = np.stack(rgb_frames, axis=0) if rgb_frames else None
        return np.stack(states, axis=0), hats, 0, True, env_rgb, _pack_frame_plan_trajs()

    @jax.jit
    def _idm_actions(p, o_stack: jnp.ndarray, on_stack: jnp.ndarray) -> jnp.ndarray:
        return idm_model.apply({'params': p}, o_stack, on_stack)

    plan_rng = jax.random.PRNGKey(int(planner_seed))
    use_stoch_plan = float(planner_noise_scale) > 0.0

    for chunk_i in range(max_chunks):
        s_np = np.asarray(states[-1], dtype=np.float32).reshape(-1)
        s = jnp.asarray(s_np, dtype=jnp.float32)
        g = jnp.asarray(s_g, dtype=jnp.float32)
        hat = agent.infer_subgoal(s, g)
        hat_np = np.asarray(jax.device_get(hat), dtype=np.float32).reshape(-1)
        hat_plot_np = xy_clamper(hat_np.copy())

        if use_stoch_plan:
            plan_rng, sk = jax.random.split(plan_rng)
            out = agent.sample_plan(s, hat, sk, noise_scale=float(planner_noise_scale))
        else:
            out = agent.plan(s, hat)
        chunk_traj_raw = np.asarray(jax.device_get(out['trajectory']), dtype=np.float32)
        if chunk_traj_raw.shape[0] < 2:
            break
        chunk_traj_plot = np.stack([xy_clamper(chunk_traj_raw[i].copy()) for i in range(chunk_traj_raw.shape[0])])
        plan_seg = np.asarray(chunk_traj_plot[1:], dtype=np.float32)
        if len(frame_plan_trajs) < len(states):
            frame_plan_trajs.append(plan_seg.copy())
        o_prev = jnp.asarray(chunk_traj_raw[:-1], dtype=jnp.float32)
        o_next = jnp.asarray(chunk_traj_raw[1:], dtype=jnp.float32)
        actions = np.asarray(jax.device_get(_idm_actions(idm_params, o_prev, o_next)), dtype=np.float32)

        n_exec = min(int(actions.shape[0]), max(1, int(action_chunk_horizon)))
        for i in range(n_exec):
            # One hat row per env transition (same as ``rollout.subgoal``) so
            # ``overlay_rgb_frames_obs2d_panel(..., chunk_hat_stride=inv_dyn_freq)`` stays aligned.
            hats_list.append(hat_plot_np.copy())
            a = np.clip(actions[i], low, high)
            ob, _r, terminated, truncated, _info = env.step(a)
            ob_f = np.asarray(ob, dtype=np.float32).reshape(-1)
            states.append(ob_f)
            frame_plan_trajs.append(plan_seg.copy())
            _maybe_record()
            if goal_within_tol(ob_f, g_np, goal_stop_dims, float(goal_tol)):
                hats = np.stack(hats_list, axis=0) if hats_list else np.zeros((0, d), dtype=np.float32)
                env_rgb = np.stack(rgb_frames, axis=0) if rgb_frames else None
                return np.stack(states, axis=0), hats, chunk_i + 1, True, env_rgb, _pack_frame_plan_trajs()
            if terminated or truncated:
                hats = np.stack(hats_list, axis=0) if hats_list else np.zeros((0, d), dtype=np.float32)
                env_rgb = np.stack(rgb_frames, axis=0) if rgb_frames else None
                return np.stack(states, axis=0), hats, chunk_i + 1, False, env_rgb, _pack_frame_plan_trajs()

    hats = np.stack(hats_list, axis=0) if hats_list else np.zeros((0, d), dtype=np.float32)
    env_rgb = np.stack(rgb_frames, axis=0) if rgb_frames else None
    return np.stack(states, axis=0), hats, max_chunks, False, env_rgb, _pack_frame_plan_trajs()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run_dir', type=str, required=True)
    p.add_argument(
        '--checkpoint_epoch',
        type=int,
        default=1000,
        help='GOUB params_<n>.pkl suffix (default 1000). If missing, nearest available is used.',
    )
    p.add_argument(
        '--task_id',
        type=int,
        required=True,
        help='OGBench eval task id in [1, num_tasks] via env.reset(options=dict(task_id=...)).',
    )
    p.add_argument('--max_steps', type=int, default=1000, help='Max replan chunks (default 1000).')
    p.add_argument(
        '--idm_checkpoint',
        type=str,
        default='',
        help='Standalone inverse dynamics params_*.pkl. Empty → use idm_net inside GOUB checkpoint.',
    )
    p.add_argument('--action_chunk_horizon', type=int, default=5)
    p.add_argument(
        '--planner_noise_scale',
        type=float,
        default=0.0,
        help='If > 0, replan uses stochastic sample_plan(s, hat, rng, noise_scale=...) instead of plan().',
    )
    p.add_argument(
        '--planner_seed',
        type=int,
        default=-1,
        help='JAX PRNG seed for sample_plan splits; -1 means use --seed.',
    )
    p.add_argument('--mujoco_gl', type=str, default='', metavar='BACKEND')
    p.add_argument('--goal_tol', type=float, default=0.5)
    p.add_argument('--goal_stop', type=str, choices=('plot', 'full'), default='plot')
    p.add_argument('--plot_dim0', type=int, default=0)
    p.add_argument('--plot_dim1', type=int, default=1)
    p.add_argument('--navigator', type=str, choices=('none', 'snap'), default='none')
    p.add_argument('--navigator_clamp', type=str, choices=('ij', 'oracle', 'union', 'center'), default='ij')
    p.add_argument('--navigator_edge_inset', type=float, default=0.08)
    p.add_argument('--maze_type', type=str, default='')
    p.add_argument('--clamp_dim0', type=int, default=-1)
    p.add_argument('--clamp_dim1', type=int, default=-1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out_path', type=str, default='rollout_idm.png')
    p.add_argument('--out_mp4', type=str, default='')
    p.add_argument('--fps', type=float, default=60.0)
    p.add_argument('--no_mp4', action='store_true')
    p.add_argument(
        '--value_heatmap',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Overlay DQC scalar value V(s, goal) on the right XY panel (requires joint checkpoints/critic/).',
    )
    p.add_argument('--value_grid_n', type=int, default=56, help='Square grid resolution for value heatmap.')
    p.add_argument(
        '--critic_epoch',
        type=int,
        default=-1,
        help='Critic params_<n>.pkl suffix; -1 = same suffix as GOUB --checkpoint_epoch.',
    )
    args = p.parse_args()

    try:
        configure_mujoco_gl(str(args.mujoco_gl))
    except ValueError as e:
        p.error(str(e))

    if int(args.action_chunk_horizon) < 1:
        p.error('--action_chunk_horizon must be >= 1')
    planner_seed = int(args.seed) if int(args.planner_seed) < 0 else int(args.planner_seed)

    run_dir = Path(args.run_dir).resolve()
    ckpt_dir = resolve_goub_checkpoint_dir(run_dir)
    ckpt_epoch = pick_epoch(int(args.checkpoint_epoch), list_checkpoint_suffixes(ckpt_dir))

    cfg, env_name = load_run_flags(run_dir)
    navigator: MazeNavigatorMap | None = None
    if args.navigator == 'snap':
        navigator = load_maze_navigator_snap(args.maze_type, env_name)
        print(format_maze_navigator_log(navigator, str(args.navigator_clamp), float(args.navigator_edge_inset)))

    env, train_raw, _ = make_env_and_datasets(
        env_name, frame_stack=cfg.get('frame_stack'), render_mode='rgb_array',
    )
    tid = int(args.task_id)
    u = env.unwrapped
    n_tasks = int(getattr(u, 'num_tasks', 5))
    if not (1 <= tid <= n_tasks):
        p.error(f'--task_id must be in [1, {n_tasks}] for {env_name!r} (got {tid})')
    ob, info = env.reset(options=dict(task_id=tid, render_goal=False))
    if 'goal' not in info:
        raise RuntimeError(
            f'{env_name!r} reset(task_id=...) did not set info["goal"]; cannot run IDM rollout in task mode.'
        )
    s0 = np.asarray(ob, dtype=np.float32).reshape(-1)
    s_g = np.asarray(info['goal'], dtype=np.float32).reshape(-1)
    traj = np.stack([s0, s_g], axis=0)
    print(f'OGBench eval reset: task_id={tid}  obs_dim={s0.shape[-1]}  goal_dim={s_g.shape[-1]}')
    tol = float(args.goal_tol)
    if tol > 0 and args.goal_stop == 'plot':
        stop_dims = (args.plot_dim0, args.plot_dim1)
    elif tol > 0 and args.goal_stop == 'full':
        stop_dims = None
    else:
        stop_dims = None

    ex = jnp.zeros((1, s0.shape[-1]), dtype=jnp.float32)
    act_dim = int(np.prod(env.action_space.shape))
    ex_act = jnp.zeros((1, act_dim), dtype=jnp.float32)
    agent = GOUBDynamicsAgent.create(args.seed, ex, cfg, ex_actions=ex_act)
    pkl_path = ckpt_dir / f'params_{ckpt_epoch}.pkl'
    agent = load_checkpoint_pkl(agent, pkl_path)
    goub_N = int(agent.config['goub_N'])
    print(
        f'Loaded GOUB {pkl_path}  goub_N={goub_N}  '
        f'planner_noise_scale={float(args.planner_noise_scale)}  planner_seed={planner_seed}'
    )

    idm_ck_stripped = str(args.idm_checkpoint).strip()
    if not idm_ck_stripped:
        ptree = agent.network.params
        # Flax ``nn.scan`` / combined modules may prefix keys with ``modules_``.
        if 'idm_net' in ptree:
            idm_params = ptree['idm_net']
        elif 'modules_idm_net' in ptree:
            idm_params = ptree['modules_idm_net']
        else:
            raise FileNotFoundError(
                'GOUB checkpoint has no idm_net / modules_idm_net '
                '(train with embedded IDM or pass --idm_checkpoint).'
            )
        idm_model = InverseDynamicsMLP(
            obs_dim=int(s0.shape[-1]),
            action_dim=int(agent.config['idm_action_dim']),
            hidden_dims=tuple(int(x) for x in agent.config['idm_hidden_dims']),
        )
        print('Using idm_net from GOUB checkpoint')
    else:
        idm_ckpt_path = Path(idm_ck_stripped).resolve()
        if not idm_ckpt_path.is_file():
            raise FileNotFoundError(f'IDM checkpoint not found: {idm_ckpt_path}')
        with open(idm_ckpt_path, 'rb') as f:
            idm_ck = pickle.load(f)
        idm_model = InverseDynamicsMLP(
            obs_dim=int(idm_ck['obs_dim']),
            action_dim=int(idm_ck['action_dim']),
            hidden_dims=tuple(idm_ck['hidden_dims']),
        )
        idm_params = idm_ck['params']
        if int(idm_ck['obs_dim']) != int(s0.shape[-1]):
            raise ValueError(f'IDM obs_dim mismatch vs trajectory obs dim={int(s0.shape[-1])}')
        print(f'Loaded standalone IDM {idm_ckpt_path} (epoch={idm_ck.get("epoch")})')
    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)

    c0 = args.clamp_dim0 if args.clamp_dim0 >= 0 else args.plot_dim0
    c1 = args.clamp_dim1 if args.clamp_dim1 >= 0 else args.plot_dim1
    nav_kw = dict(
        navigator=navigator,
        clamp_dim0=c0,
        clamp_dim1=c1,
        navigator_clamp_mode=str(args.navigator_clamp),
        navigator_edge_inset=float(args.navigator_edge_inset),
    )

    from rollout.value_field import dqc_value_mesh_for_xy, load_dqc_critic_joint_run

    ce = int(args.critic_epoch) if int(args.critic_epoch) >= 0 else int(ckpt_epoch)
    critic_agent = load_dqc_critic_joint_run(
        run_dir,
        ce,
        env,
        train_raw,
        seed=int(args.seed),
    )
    critic_value_params = critic_agent.network.params.get('modules_value', None)
    print(f'Loaded critic for IDM inference/value heatmap (epoch suffix {ce})')

    roll, hats, n_chunks, reached, env_frames, frame_plan_trajs = rollout_goub_idm_env(
        env,
        agent,
        idm_model,
        idm_params,
        s0,
        s_g,
        int(args.max_steps),
        goal_tol=tol,
        goal_stop_dims=stop_dims,
        **nav_kw,
        action_low=low,
        action_high=high,
        action_chunk_horizon=int(args.action_chunk_horizon),
        planner_noise_scale=float(args.planner_noise_scale),
        planner_seed=planner_seed,
    )
    n_trans = max(0, int(roll.shape[0]) - 1)
    action_chunk_horizon = int(args.action_chunk_horizon)
    steps_per_replan = min(action_chunk_horizon, goub_N)
    print(
        f'IDM rollout: {n_chunks} replans × up to {steps_per_replan} env steps/replan → '
        f'{roll.shape[0]} obs ({n_trans} transitions), goal_reached={reached}'
    )

    d0, d1 = args.plot_dim0, args.plot_dim1
    plot_nav = maze_navigator_for_xy_plot(navigator, env_name, d0, d1)

    heat_mesh = None
    heat_vmin = heat_vmax = None
    if bool(args.value_heatmap):
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
            _pf = min(action_chunk_horizon, goub_N)
            frames = overlay_rgb_frames_obs2d_panel(
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
                value_heatmap=heat_mesh,
                value_heatmap_vmin=heat_vmin,
                value_heatmap_vmax=heat_vmax,
            )
            write_rgb_array_mp4(frames, mp4_out, float(args.fps))
            print(f'Wrote MP4 {mp4_out.resolve()}')
        except Exception as e:
            print(f'Warning: MP4 failed ({e!r}). pip install imageio imageio-ffmpeg')
    elif not args.no_mp4:
        print('Warning: no RGB frames; check render_mode')


if __name__ == '__main__':
    main()
