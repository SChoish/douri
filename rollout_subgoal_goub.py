#!/usr/bin/env python3
"""State-space rollout only: subgoal net + GOUB ``plan`` trajectory vs one offline episode.

Pipeline (no environment dynamics — open-loop in observation space). **Ant maze:** the planner
does not know about walls; by default ``--navigator snap`` projects each virtual ``(x,y)`` into a
walkable cell so rollouts match the grid (use ``--navigator none`` for the raw network trajectory).
When ``--goal_tol > 0`` and the snap rollout never satisfies that goal tolerance, the script retries
once with ``navigator`` disabled so a difficult maze layout can still be visualized (xy may cross walls).

1. Pick one episode from the compact offline dataset (``terminals`` boundaries).
2. Fix a high-level goal ``s_g`` (default: last state of that episode, i.e. terminal).
3. Starting from ``s_0`` of the episode, repeat up to ``max_steps`` (optionally **early-stop**
   when distance to ``s_g`` is at most ``--goal_tol`` (``<=``, inclusive); see ``--goal_stop``).

   Each **chunk**: ``predict_subgoal(s, s_g)`` once → one ``plan`` / ``sample_plan`` → append
   ``trajectory[1:K+1]`` (``K`` from ``--state_chunk_horizon``; ``0`` → ``goub_N``), clamped, capped by ``N``.
   ``K=1`` uses only ``next_step`` (``trajectory[1]``) per chunk.

       (repeat)  hat = G(s, g);  traj = plan(s, hat);  walk traj[1], …, traj[K]

4. Plot dataset vs rollout in 2D; writes **PNG** and optional **matplotlib MP4** at ``--fps``.

**Real env + IDM** rollouts live in ``rollout_idm_goub.py``. **Chunk low-level actor** rollouts:
``rollout_chunk_actor_goub.py``.

``loss_sub_mean`` in training logs is the batch-mean of ``phase1/loss_subgoal``,
i.e. MSE between ``subgoal_net(s, g)`` and teacher ``high_actor_targets``,
averaged over an epoch (``training/phase1/loss_subgoal_epoch_mean``).

Example::

    cd <douri repo root>
    python rollout_subgoal_goub.py \\
        --run_dir=runs/20260416_235557_seed0_antmaze-large-navigate-v0 \\
        --checkpoint_epoch=1000 \\
        --traj_idx=0 \\
        --max_steps=1000 \\
        --out_path=rollout_plot.png \\
        --fps=60

Writes ``rollout_plot.png`` and ``rollout_plot.mp4`` (unless ``--no_mp4``).

``--segment_compare``: 오프라인 창 ``true = traj[t:t+k]`` 와 비교. ``segment_mode=bridge``는 단일 브리지
재샘플 정렬; ``iterative``는 위 청크 롤아웃(``trajectory[1:K+1]``)으로 ``k-1`` 전이까지 생성.
``--max_steps``는 ``segment_compare``일 때 쓰이지 않는다.
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
import matplotlib.pyplot as plt
import numpy as np
from ml_collections import ConfigDict

from agents.goub_phase1 import GOUBPhase1Agent, get_config
from utils.datasets import Dataset
from utils.env_utils import make_env_and_datasets
from utils.flax_utils import merge_checkpoint_state_dict
from utils.goub_rollout_env import format_maze_navigator_log, load_maze_navigator_snap, make_xy_clamper
from utils.goub_rollout_plot import (
    axis_limits,
    draw_dataset_background,
    maze_navigator_for_xy_plot,
    plot_maze_cell_tiles,
    write_rollout_mp4,
)
from utils.maze_navigator import MazeNavigatorMap


def _load_checkpoint_pkl(agent, pkl_path: Path):
    with open(pkl_path, 'rb') as f:
        load_dict = pickle.load(f)
    template = flax.serialization.to_state_dict(agent)
    merged = merge_checkpoint_state_dict(template, load_dict['agent'])
    return flax.serialization.from_state_dict(agent, merged)


def _list_checkpoint_suffixes(checkpoints_dir: Path) -> list[int]:
    """Integers ``n`` such that ``params_<n>.pkl`` exists (training epoch, or legacy global_step)."""
    out = []
    for p in checkpoints_dir.glob('params_*.pkl'):
        m = re.search(r'params_(\d+)\.pkl$', p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def _load_run_flags(run_dir: Path) -> tuple[ConfigDict, str]:
    """Return ``(merged agent config, env_name)`` from ``flags.json``."""
    flags_path = run_dir / 'flags.json'
    if not flags_path.is_file():
        raise FileNotFoundError(f'Missing flags.json under {run_dir}')
    with open(flags_path, 'r', encoding='utf-8') as f:
        flags = json.load(f)
    env_name = flags.get('env_name')
    if not env_name:
        raise KeyError('flags.json must contain env_name')
    cfg = get_config()
    for k, v in (flags.get('agent') or {}).items():
        cfg[k] = v
    return cfg, env_name


def _episode_slices(terminals: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end) inclusive indices for each episode."""
    terminals = np.asarray(terminals).reshape(-1)
    ends = np.nonzero(terminals > 0)[0]
    if len(ends) == 0:
        raise ValueError('No terminal flags found; cannot split episodes.')
    starts = np.concatenate([[0], ends[:-1] + 1])
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def _get_trajectory(dataset: Dataset, traj_idx: int) -> np.ndarray:
    obs = np.asarray(dataset['observations'])
    terms = np.asarray(dataset['terminals'])
    slices = _episode_slices(terms)
    if traj_idx < 0 or traj_idx >= len(slices):
        raise IndexError(f'traj_idx={traj_idx} out of range [0, {len(slices) - 1}]')
    s, e = slices[traj_idx]
    return obs[s : e + 1].copy()


def _sample_segment_start_k(traj_len: int, k: int, rng: np.random.Generator) -> int:
    """Uniform random ``t`` such that ``traj[t:t+k]`` (k contiguous states) fits in the episode."""
    if k < 2:
        raise ValueError('k must be >= 2 for segment compare')
    if traj_len < k:
        raise ValueError(f'Episode length {traj_len} < k={k}')
    return int(rng.integers(0, traj_len - k + 1))


def _segment_alignment_errors(
    roll: np.ndarray, seg: np.ndarray, plot_dims: tuple[int, int]
) -> dict[str, float]:
    """Per-step L2 vs dataset segment (full obs and x,y plot dims)."""
    if roll.shape[0] != seg.shape[0]:
        raise ValueError(f'roll length {roll.shape[0]} != seg length {seg.shape[0]}')
    diff = roll.astype(np.float64) - seg.astype(np.float64)
    l2_full = np.linalg.norm(diff, axis=1)
    d0, d1 = plot_dims
    l2_xy = np.linalg.norm(diff[:, [d0, d1]], axis=1)
    return {
        'mean_l2_full': float(l2_full.mean()),
        'mean_l2_xy': float(l2_xy.mean()),
        'max_l2_full': float(l2_full.max()),
        'max_l2_xy': float(l2_xy.max()),
        'final_l2_full': float(l2_full[-1]),
        'final_l2_xy': float(l2_xy[-1]),
    }


def _goal_distance(s: np.ndarray, g: np.ndarray, stop_dims: tuple[int, ...] | None) -> float:
    if stop_dims:
        idx = np.asarray(stop_dims, dtype=np.int32)
        return float(np.linalg.norm(s[idx] - g[idx]))
    return float(np.linalg.norm(s - g))


def _goal_within_tol(
    s: np.ndarray, g: np.ndarray, stop_dims: tuple[int, ...] | None, goal_tol: float
) -> bool:
    """``goal_tol > 0``일 때만 사용. 거리가 ``goal_tol`` 이하(포함, ``<=``)이면 도달로 본다."""
    if goal_tol is None or float(goal_tol) <= 0.0:
        return False
    return _goal_distance(s, g, stop_dims) <= float(goal_tol)


def bridge_trajectory(
    agent: GOUBPhase1Agent,
    s_start: np.ndarray,
    s_end: np.ndarray,
    k: int,
) -> np.ndarray:
    """Single bridge pass from s_start to s_end, resampled to k evenly-spaced points.

    ``plan(s_start, s_end)`` produces the full reverse chain ``[x_N, x_{N-1}, ..., x_0]``
    (N+1 states, where x_N = s_start and x_0 ≈ s_end).  This function linearly
    resamples that trajectory to ``k`` points so it can be aligned with an offline
    segment ``traj[t:t+k]`` of the same length.

    Returns:
        Array of shape ``(k, D)`` — h-transform bridge path from s_start to s_end.
    """
    s_start_j = jnp.asarray(s_start, dtype=jnp.float32)
    s_end_j = jnp.asarray(s_end, dtype=jnp.float32)
    out = agent.plan(s_start_j, s_end_j)
    # trajectory: (N+1, D) — x_N (=s_start), x_{N-1}, ..., x_0 (≈s_end)
    traj_bridge = np.asarray(jax.device_get(out['trajectory']), dtype=np.float32)
    n_bridge = traj_bridge.shape[0]  # N+1
    if n_bridge == k:
        return traj_bridge
    # linear resample along the time axis
    src_t = np.linspace(0.0, 1.0, n_bridge)
    dst_t = np.linspace(0.0, 1.0, k)
    d = traj_bridge.shape[-1]
    resampled = np.zeros((k, d), dtype=np.float32)
    for dim in range(d):
        resampled[:, dim] = np.interp(dst_t, src_t, traj_bridge[:, dim])
    return resampled


def rollout_subgoal_goub(
    agent: GOUBPhase1Agent,
    s0: np.ndarray,
    s_g: np.ndarray,
    max_steps: int,
    goal_tol: float = 0.0,
    goal_stop_dims: tuple[int, ...] | None = None,
    navigator: MazeNavigatorMap | None = None,
    clamp_dim0: int = 0,
    clamp_dim1: int = 1,
    navigator_clamp_mode: str = 'ij',
    navigator_edge_inset: float = 0.08,
    *,
    stochastic: bool = False,
    plan_key: jnp.ndarray | None = None,
    sample_noise_scale: float = 1.0,
    state_chunk_horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Roll out GOUB + subgoal in state space.

    Returns:
        ``roll``: shape ``(T+1, D)`` with ``roll[0]=s0`` and ``T`` planner steps
        (``T <= max_steps``).
        ``hats``: shape ``(T, D)`` — ``hats[i]`` is the bridge endpoint ``hat`` used for the transition
        ``roll[i] → roll[i+1]`` (constant for ``state_chunk_horizon`` consecutive steps).
        ``n_steps``: number of appended states along rollouts (``T``), i.e. transition count.
        ``reached``: True if rollout stopped early because goal distance
        (subset of dims if ``goal_stop_dims``) is at most ``goal_tol`` (``<=``).
        If ``goal_tol <= 0``, never early-stops and ``reached`` is False.

    If ``navigator`` is set, clamps ``(clamp_dim0, clamp_dim1)`` using
    ``ogbench.locomaze.maze``-aligned modes (``ij`` / ``oracle`` / ``union`` / ``center``).

    Args:
        state_chunk_horizon: After each ``predict_subgoal``, call ``plan`` (or ``sample_plan``) **once**
            and walk the first ``K`` states along its reverse trajectory — i.e. append
            ``trajectory[1], …, trajectory[K]`` (1-based slice ``trajectory[1:K+1]``), capped by ``goub_N``.
            Use ``1`` for legacy behaviour (only ``next_step`` / ``trajectory[1]`` per chunk, then new
            subgoal).
    """
    g_np = np.asarray(s_g, dtype=np.float32)
    xy_clamper = make_xy_clamper(
        g_np, navigator, clamp_dim0, clamp_dim1, navigator_clamp_mode, navigator_edge_inset
    )
    s0f = xy_clamper(np.asarray(s0, dtype=np.float32))
    d = int(s0f.shape[-1])
    states: list[np.ndarray] = [s0f]
    hats_list: list[np.ndarray] = []
    s = jnp.asarray(s0f, dtype=jnp.float32)
    g = jnp.asarray(s_g, dtype=jnp.float32)
    if stochastic and plan_key is None:
        raise ValueError('stochastic=True requires plan_key (JAX PRNGKey).')

    sch = max(1, int(state_chunk_horizon))

    if _goal_within_tol(s0f, g_np, goal_stop_dims, float(goal_tol)):
        roll = np.stack(states, axis=0)
        hats = np.zeros((0, d), dtype=np.float32)
        return roll, hats, 0, True

    key = plan_key
    total = 0
    while total < max_steps:
        hat = agent.predict_subgoal(s, g)
        hat_np = xy_clamper(np.asarray(jax.device_get(hat), dtype=np.float32).reshape(-1))
        hat = jnp.asarray(hat_np, dtype=jnp.float32)

        if stochastic:
            assert key is not None
            key, sk = jax.random.split(key)
            out = agent.sample_plan(s, hat, sk, noise_scale=float(sample_noise_scale))
        else:
            out = agent.plan(s, hat)
        chunk_traj = np.asarray(jax.device_get(out['trajectory']), dtype=np.float32)
        if chunk_traj.ndim == 3:
            chunk_traj = chunk_traj[0]
        n_plus = int(chunk_traj.shape[0])
        n_bridge = n_plus - 1  # number of new states available along plan (x_{N-1}..x_0)
        if n_bridge < 1:
            break
        k_req = min(sch, n_bridge, max_steps - total)
        if k_req < 1:
            break
        for j in range(k_req):
            sn = xy_clamper(np.asarray(chunk_traj[j + 1], dtype=np.float32))
            hats_list.append(hat_np.copy())
            states.append(sn)
            total += 1
            s = jnp.asarray(sn, dtype=jnp.float32)
            if _goal_within_tol(sn, g_np, goal_stop_dims, float(goal_tol)):
                roll = np.stack(states, axis=0)
                hats = np.stack(hats_list, axis=0) if hats_list else np.zeros((0, d), dtype=np.float32)
                return roll, hats, total, True

    roll = np.stack(states, axis=0)
    hats = np.stack(hats_list, axis=0) if hats_list else np.zeros((0, d), dtype=np.float32)
    return roll, hats, total, False


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run_dir', type=str, required=True, help='Training run folder (contains flags.json + checkpoints/).')
    p.add_argument(
        '--checkpoint_epoch',
        type=int,
        default=1000,
        help='Integer suffix of params_<n>.pkl (default 1000). If missing, nearest available checkpoint is used.',
    )
    p.add_argument(
        '--checkpoint_step',
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument('--traj_idx', type=int, default=0, help='Which offline episode index to compare against.')
    p.add_argument(
        '--state_chunk_horizon',
        type=int,
        default=0,
        metavar='K',
        help=(
            'Per chunk: append first K states from plan trajectory (trajectory[1:K+1]), one plan() call. '
            'K=0 (default) uses goub_N from checkpoint. Capped by N. K=1 = next_step only per chunk.'
        ),
    )
    p.add_argument(
        '--max_steps',
        type=int,
        default=1000,
        help='Max appended plan states (transitions), unless --segment_compare (then k-1). Default: 1000.',
    )
    p.add_argument(
        '--goal_tol',
        type=float,
        default=0.5,
        help='Early-stop when goal distance <= this (default 0.5, inclusive). 0 = run full max_steps.',
    )
    p.add_argument(
        '--goal_stop',
        type=str,
        choices=('plot', 'full'),
        default='plot',
        help='plot: L2 only on (plot_dim0, plot_dim1). full: L2 on full observation.',
    )
    p.add_argument('--plot_dim0', type=int, default=0, help='Observation index for x-axis.')
    p.add_argument('--plot_dim1', type=int, default=1, help='Observation index for y-axis.')
    p.add_argument('--out_path', type=str, default='rollout_subgoal_goub.png', help='Output PNG path.')
    p.add_argument('--out_mp4', type=str, default='', help='Output MP4 path (default: same stem as --out_path).')
    p.add_argument('--fps', type=float, default=60.0, help='MP4 frame rate.')
    p.add_argument('--no_mp4', action='store_true', help='Skip MP4; write PNG only.')
    p.add_argument(
        '--navigator',
        type=str,
        choices=('none', 'snap'),
        default='snap',
        help=(
            'snap (default): after each planned state, snap (clamp_dim0, clamp_dim1) into the maze free region '
            '(see --navigator_clamp) so open-loop rollouts do not cut through walls. '
            'none: raw plan states (can leave walkable cells).'
        ),
    )
    p.add_argument(
        '--navigator_clamp',
        type=str,
        choices=('ij', 'oracle', 'union', 'center'),
        default='ij',
        help='ij: OGBench grid snap (xy_to_ij/ij_to_xy). oracle: one-step BFS subgoal. union: corridor box union. center: nearest free center.',
    )
    p.add_argument(
        '--navigator_edge_inset',
        type=float,
        default=0.08,
        help='Shrink each free-cell box by this fraction of maze_unit (wall margin).',
    )
    p.add_argument(
        '--maze_type',
        type=str,
        default='',
        help='Optional: force embedded maze layout (arena|medium|large|giant|teleport). If empty, inferred from env_name.',
    )
    p.add_argument(
        '--clamp_dim0',
        type=int,
        default=-1,
        help='Observation index for maze world x (-1 = same as --plot_dim0).',
    )
    p.add_argument(
        '--clamp_dim1',
        type=int,
        default=-1,
        help='Observation index for maze world y (-1 = same as --plot_dim1).',
    )
    p.add_argument('--seed', type=int, default=0, help='Agent create / JAX seed.')
    p.add_argument(
        '--sample_seeds',
        type=int,
        nargs='*',
        default=None,
        metavar='SEED',
        help=(
            'If non-empty: use stochastic sample_plan (not deterministic plan) with jax.random.PRNGKey(SEED) '
            'per rollout; write <out_stem>_seed<SEED>.png for each. Skips MP4 when more than one seed.'
        ),
    )
    p.add_argument(
        '--sample_noise_scale',
        type=float,
        default=1.0,
        help='noise_scale passed to sample_plan when --sample_seeds is set.',
    )
    p.add_argument(
        '--sample_overlay',
        action='store_true',
        help='With --sample_seeds: draw every rollout on one axes (distinct colors), write <out_stem>_overlay.png only.',
    )
    p.add_argument(
        '--segment_compare',
        action='store_true',
        help='Sample offline window traj[t:t+k] and compare vs generated path. See --segment_mode.',
    )
    p.add_argument(
        '--segment_mode',
        type=str,
        choices=('iterative', 'bridge'),
        default='bridge',
        help=(
            'bridge (default): single plan(s_t, s_{t+k-1}) → reverse chain resampled to k states. '
            'iterative: same chunked state rollout as the default script (trajectory[1:K+1] per plan) for k states.'
        ),
    )
    p.add_argument(
        '--segment_k',
        type=int,
        default=26,
        help='Max window length in states (k). true=traj[t:t+k]. With --segment_fixed_len, use exactly k states (only t random); else draw k uniformly from 2..min(segment_k, len(episode)).',
    )
    p.add_argument(
        '--segment_len',
        type=int,
        default=None,
        help='Deprecated: former "transitions" count; mapped to --segment_k = segment_len + 1 (overrides --segment_k).',
    )
    p.add_argument(
        '--segment_fixed_len',
        action='store_true',
        help='Only randomize t; use exactly --segment_k states (traj[t:t+k]).',
    )
    p.add_argument(
        '--segment_seed',
        type=int,
        default=None,
        help='RNG seed for sampling segment (default: same as --seed).',
    )
    args = p.parse_args()

    if args.segment_len is not None:
        print(
            'Warning: --segment_len is deprecated (was transition count); '
            f'setting --segment_k = {int(args.segment_len) + 1} (states in traj[t:t+k]).'
        )
        args.segment_k = int(args.segment_len) + 1

    if args.segment_compare and args.sample_seeds:
        p.error('--sample_seeds cannot be used with --segment_compare')
    if int(args.state_chunk_horizon) < 0:
        p.error('--state_chunk_horizon must be >= 0 (0 = use goub_N)')
    if args.sample_overlay and not args.sample_seeds:
        p.error('--sample_overlay requires --sample_seeds')

    if args.checkpoint_step is not None and args.checkpoint_epoch >= 0:
        p.error('Use only one of --checkpoint_epoch and --checkpoint_step (--checkpoint_step is deprecated).')
    if args.checkpoint_step is not None:
        print('Warning: --checkpoint_step is deprecated; use --checkpoint_epoch (same suffix in params_<n>.pkl).')
        ckpt_epoch = int(args.checkpoint_step)
    else:
        ckpt_epoch = int(args.checkpoint_epoch)

    run_dir = Path(args.run_dir).resolve()
    ckpt_dir = run_dir / 'checkpoints'
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f'No checkpoints/ under {run_dir}')

    suffixes = _list_checkpoint_suffixes(ckpt_dir)
    if not suffixes:
        raise FileNotFoundError(f'No params_*.pkl in {ckpt_dir}')
    if ckpt_epoch < 0:
        ckpt_epoch = suffixes[-1]
    if ckpt_epoch not in suffixes:
        nearest = min(suffixes, key=lambda x: abs(x - ckpt_epoch))
        print(f'Warning: checkpoint suffix {ckpt_epoch} not found; using nearest {nearest}')
        ckpt_epoch = nearest

    cfg, env_name = _load_run_flags(run_dir)

    navigator: MazeNavigatorMap | None = None
    if args.navigator == 'snap':
        try:
            navigator = load_maze_navigator_snap(args.maze_type, env_name)
        except ValueError as ex:
            p.error(str(ex))
        print(format_maze_navigator_log(navigator, str(args.navigator_clamp), float(args.navigator_edge_inset)))

    _, train_raw, _ = make_env_and_datasets(env_name, frame_stack=cfg.get('frame_stack'))
    dataset = Dataset.create(**train_raw)

    traj = _get_trajectory(dataset, args.traj_idx)
    data_len = int(len(traj))
    segment_t: int | None = None
    seg: np.ndarray | None = None
    max_steps_eff = int(args.max_steps)

    seg_mode: str = 'bridge'
    if args.segment_compare:
        seg_mode = str(args.segment_mode)
        seg_seed = int(args.segment_seed) if args.segment_seed is not None else int(args.seed)
        rng = np.random.default_rng(seg_seed)
        k_max = min(int(args.segment_k), len(traj))
        if k_max < 2:
            p.error(f'Episode length {len(traj)} too short for segment compare (need k >= 2 states).')
        if args.segment_fixed_len:
            k = int(args.segment_k)
            if len(traj) < k:
                p.error(f'Episode length {len(traj)} < segment_k={k}')
            segment_t = _sample_segment_start_k(len(traj), k, rng)
        else:
            k = int(rng.integers(2, k_max + 1))
            segment_t = int(rng.integers(0, len(traj) - k + 1))
        seg = traj[segment_t : segment_t + k].copy()
        s0 = seg[0]
        s_g = traj[-1]
        max_steps_eff = k - 1
        tol = 0.0
        stop_dims: tuple[int, ...] | None = None
        print(
            f'Segment compare: mode={seg_mode} | true=traj[t:t+k] | '
            f't={segment_t}, k={k} states (indices {segment_t}..{segment_t + k - 1}), '
            f'segment_seed={seg_seed}, fixed_k={args.segment_fixed_len}'
        )
    else:
        s0 = traj[0]
        s_g = traj[-1]
        tol = float(args.goal_tol)
        if tol > 0 and args.goal_stop == 'plot':
            stop_dims = (args.plot_dim0, args.plot_dim1)
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
    print(f'Loaded {pkl_path}')

    goub_N = int(agent.config['goub_N'])
    raw_chunk_h = int(args.state_chunk_horizon)
    iter_state_chunk_h = goub_N if raw_chunk_h <= 0 else raw_chunk_h
    if iter_state_chunk_h < 1:
        p.error('--state_chunk_horizon resolved to < 1 (use >= 1, or 0 for goub_N)')
    c0 = args.clamp_dim0 if args.clamp_dim0 >= 0 else args.plot_dim0
    c1 = args.clamp_dim1 if args.clamp_dim1 >= 0 else args.plot_dim1

    nav_kw = dict(
        navigator=navigator,
        clamp_dim0=c0,
        clamp_dim1=c1,
        navigator_clamp_mode=str(args.navigator_clamp),
        navigator_edge_inset=float(args.navigator_edge_inset),
    )

    seeds_to_run: list[int | None] = [None] if not args.sample_seeds else list(args.sample_seeds)
    rollouts: list[tuple[np.ndarray, np.ndarray, int, bool, int | None, bool]] = []

    if args.segment_compare and seg_mode == 'bridge':
        assert seg is not None
        roll = bridge_trajectory(agent, seg[0], seg[-1], k)
        hats = np.zeros((0, roll.shape[-1]), dtype=np.float32)
        n_planner_steps = 0
        reached = False
        print(
            f'Bridge: plan(s_t, s_{{t+k-1}}) → reverse chain (N+1={goub_N + 1} states), '
            f'resampled to k={k} points for alignment.'
        )
        rollouts.append((roll, hats, n_planner_steps, reached, None, False))
    else:
        for plan_seed in seeds_to_run:
            skw: dict = {'sample_noise_scale': float(args.sample_noise_scale)}
            if plan_seed is not None:
                skw['stochastic'] = True
                skw['plan_key'] = jax.random.PRNGKey(int(plan_seed))
            else:
                skw['stochastic'] = False
                skw['plan_key'] = None
            roll, hats, n_planner_steps, reached = rollout_subgoal_goub(
                agent,
                s0,
                s_g,
                max_steps_eff,
                goal_tol=tol,
                goal_stop_dims=stop_dims,
                **nav_kw,
                **skw,
                state_chunk_horizon=iter_state_chunk_h,
            )
            if navigator is not None and tol > 0 and not reached:
                print(
                    'State rollout: goal not reached with wall projection (--navigator snap); '
                    'retrying once without navigator (raw xy, may cross walls).'
                )
                nav_kw_raw = {**nav_kw, 'navigator': None}
                roll, hats, n_planner_steps, reached = rollout_subgoal_goub(
                    agent,
                    s0,
                    s_g,
                    max_steps_eff,
                    goal_tol=tol,
                    goal_stop_dims=stop_dims,
                    **nav_kw_raw,
                    **skw,
                    state_chunk_horizon=iter_state_chunk_h,
                )
            if plan_seed is not None:
                print(
                    f'State rollout: K={iter_state_chunk_h} (raw --state_chunk_horizon={raw_chunk_h}), '
                    f'sample_plan, seed={plan_seed}'
                )
            else:
                print(
                    f'State rollout: K={iter_state_chunk_h} (raw --state_chunk_horizon={raw_chunk_h}); '
                    f'each chunk: predict_subgoal → plan → trajectory[1:K+1]'
                )
            rollouts.append((roll, hats, n_planner_steps, reached, plan_seed, bool(skw['stochastic'])))

    d0, d1 = args.plot_dim0, args.plot_dim1
    plot_nav = maze_navigator_for_xy_plot(navigator, env_name, d0, d1)
    if plot_nav is not None and navigator is None and d0 == 0 and d1 == 1:
        print(
            f'Maze plot tiles: auto from env_name (source={plot_nav.source}, maze_type={plot_nav.maze_type}); '
            'rollout xy not clamped unless --navigator snap.'
        )

    if args.sample_overlay:
        roll_union = np.concatenate([r[0] for r in rollouts], axis=0)
        hat_pieces = [r[1] for r in rollouts if r[1].size > 0]
        hats_union = (
            np.concatenate(hat_pieces, axis=0)
            if hat_pieces
            else np.zeros((0, rollouts[0][0].shape[-1]), dtype=np.float32)
        )
        xlim, ylim = axis_limits(traj, roll_union, hats_union, d0, d1, s_g, s0, navigator=plot_nav, seg=seg)
        fig, ax = plt.subplots(figsize=(8, 6.5))
        plot_maze_cell_tiles(ax, plot_nav, d0, d1)
        draw_dataset_background(ax, traj, d0, d1)
        colors = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        summary_lines = [
            f'sample_plan overlay  noise_scale={args.sample_noise_scale:g}',
            f'state rollout  goub_N={goub_N}  K={iter_state_chunk_h}',
        ]
        for idx, (roll, _hats, n_ps, reached, plan_seed, _stoch) in enumerate(rollouts):
            c = colors[idx % len(colors)]
            fd = _goal_distance(
                np.asarray(roll[-1], dtype=np.float32),
                np.asarray(s_g, dtype=np.float32),
                stop_dims,
            )
            n_tr = max(0, int(roll.shape[0]) - 1)
            ax.plot(
                roll[:, d0],
                roll[:, d1],
                '-',
                color=c,
                lw=2.0,
                zorder=5,
                label=f'seed={plan_seed}  trans={n_tr}  reached={reached}  ‖end−g‖={fd:.2f}',
            )
            ax.scatter(
                roll[:, d0],
                roll[:, d1],
                c=c,
                s=22,
                zorder=6,
                alpha=0.5,
                edgecolors='0.2',
                linewidths=0.2,
            )
            if roll.shape[0] > 0:
                end = roll[-1]
                ax.scatter(
                    [end[d0]],
                    [end[d1]],
                    c=c,
                    s=80,
                    marker='X',
                    zorder=7,
                    edgecolors='k',
                    linewidths=0.35,
                )
            summary_lines.append(f'seed {plan_seed}: planner_steps={n_ps}  trans={n_tr}  reached={reached}  final_dist={fd:.4f}')
        ax.scatter(
            [s_g[d0]],
            [s_g[d1]],
            c='limegreen',
            s=95,
            marker='*',
            zorder=9,
            edgecolors='k',
            linewidths=0.4,
            label='$s_g$',
        )
        ax.scatter(
            [s0[d0]],
            [s0[d1]],
            c='black',
            s=70,
            marker='P',
            zorder=9,
            edgecolors='k',
            linewidths=0.35,
            label='$s_0$',
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(f'obs dim {d0}')
        ax.set_ylabel(f'obs dim {d1}')
        ax.set_title(
            f'epoch={ckpt_epoch}  traj={args.traj_idx}  data_len={data_len}  state rollout  overlay  tol={tol:g}',
            fontsize=10,
        )
        ax.text(
            0.02,
            0.98,
            '\n'.join(summary_lines),
            transform=ax.transAxes,
            va='top',
            ha='left',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.88),
        )
        ax.legend(loc='lower right', fontsize=7)
        if plot_nav is not None and d0 == 0 and d1 == 1:
            ax.grid(False)
        else:
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_ov = Path(args.out_path).with_name(f'{Path(args.out_path).stem}_overlay.png')
        out_ov.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_ov, dpi=150)
        plt.close(fig)
        print(f'Wrote overlay PNG {out_ov.resolve()}')
    else:
        for roll, hats, n_planner_steps, reached, plan_seed, stochastic_run in rollouts:
            n_trans = max(0, int(roll.shape[0]) - 1)
            final_dist = _goal_distance(
                np.asarray(roll[-1], dtype=np.float32),
                np.asarray(s_g, dtype=np.float32),
                stop_dims,
            )
            seg_metrics: dict[str, float] | None = None
            if args.segment_compare and seg is not None:
                if roll.shape[0] != seg.shape[0]:
                    print(
                        f'Warning: generated length {roll.shape[0]} != true segment length {seg.shape[0]}; '
                        'truncating to min length for metrics.'
                    )
                    m = min(roll.shape[0], seg.shape[0])
                    roll_p = roll[:m]
                    seg_p = seg[:m]
                else:
                    roll_p, seg_p = roll, seg
                seg_metrics = _segment_alignment_errors(roll_p, seg_p, (d0, d1))
                mode_label = 'h-transform bridge' if seg_mode == 'bridge' else 'state rollout (chunked plan)'
                print(
                    f'true vs {mode_label} — mean L2 (full obs): {seg_metrics["mean_l2_full"]:.4f}, '
                    f'mean L2 (xy): {seg_metrics["mean_l2_xy"]:.4f}, '
                    f'final L2 (full): {seg_metrics["final_l2_full"]:.4f}'
                )

            fig, ax = plt.subplots(figsize=(7, 6))
            plot_maze_cell_tiles(ax, plot_nav, d0, d1)
            draw_dataset_background(ax, traj, d0, d1)
            if args.segment_compare and seg is not None:
                ax.plot(seg[:, d0], seg[:, d1], '-', color='C0', lw=2.1, zorder=5, label='true (offline)')
                ax.scatter(seg[:, d0], seg[:, d1], c='C0', s=28, zorder=6, alpha=0.85, edgecolors='0.2', linewidths=0.2)
            if args.segment_compare and seg_mode == 'bridge':
                gen_label = 'h-transform bridge'
            else:
                gen_label = 'estimated traj'
                if stochastic_run:
                    gen_label += ' (sample_plan)'
            ax.plot(roll[:, d0], roll[:, d1], '-', color='C1', lw=2.2, zorder=5, alpha=0.96, label=gen_label)
            ax.scatter(
                roll[:, d0],
                roll[:, d1],
                c='C1',
                s=36,
                zorder=6,
                marker='s',
                alpha=0.85,
                edgecolors='0.12',
                linewidths=0.32,
            )
            if hats.size > 0:
                ax.scatter(
                    hats[:, d0],
                    hats[:, d1],
                    c='darkviolet',
                    s=58,
                    marker='D',
                    zorder=7,
                    alpha=0.96,
                    edgecolors='0.1',
                    linewidths=0.42,
                    label='subgoal est',
                )
            if roll.shape[0] > 0:
                end = roll[-1]
                ax.scatter(
                    [end[d0]],
                    [end[d1]],
                    c='darkorange',
                    s=70,
                    marker='X',
                    zorder=6,
                    label='rollout end',
                )
            ax.scatter([s_g[d0]], [s_g[d1]], c='green', s=80, marker='*', zorder=5, label='goal $s_g$ (terminal)')
            ax.scatter([s0[d0]], [s0[d1]], c='black', s=60, marker='P', zorder=5, label='$s_0$ (rollout start)')
            ax.set_xlabel(f'obs dim {d0}')
            ax.set_ylabel(f'obs dim {d1}')
            if args.segment_compare and segment_t is not None and seg is not None:
                _k = int(seg.shape[0])
                if seg_mode == 'bridge':
                    title = (
                        f'epoch={ckpt_epoch}  traj={args.traj_idx}  data_len={data_len}  true vs h-transform bridge  '
                        f'k={_k}, t={segment_t}..{segment_t + _k - 1}  (N={goub_N}→resample→{_k})'
                    )
                else:
                    title = (
                        f'epoch={ckpt_epoch}  traj={args.traj_idx}  data_len={data_len}  true vs state rollout  '
                        f'k={_k}, t={segment_t}..{segment_t + _k - 1}, '
                        f'K={iter_state_chunk_h} from plan traj. / chunk'
                    )
            elif tol > 0:
                if reached:
                    reach_txt = f'goal in {n_trans} trans. (tol≤{tol:g})'
                else:
                    reach_txt = f'not reached (dist={final_dist:.3f})'
                metric = args.goal_stop
                title = (
                    f'epoch={ckpt_epoch}  traj={args.traj_idx}  data_len={data_len}  '
                    f'planner steps={n_planner_steps}/{max_steps_eff} ({reach_txt}, {metric}, tol={tol:g})'
                    + (f'  sample_plan seed={plan_seed}' if stochastic_run and plan_seed is not None else '')
                )
            else:
                title = (
                    f'epoch={ckpt_epoch}  traj={args.traj_idx}  data_len={data_len}  planner steps={n_planner_steps} (no early stop)'
                    + (f'  sample_plan seed={plan_seed}' if stochastic_run and plan_seed is not None else '')
                )
            ax.set_title(title, fontsize=10)
            if args.segment_compare:
                _k2 = int(seg.shape[0]) if seg is not None else 0
                if seg_mode == 'bridge':
                    text_lines = [
                        f'true = traj[t:t+k]  (k={_k2} states)\n'
                        f'bridge: 1× plan() (N={goub_N}) → {_k2} pts'
                    ]
                else:
                    text_lines = [
                        f'true = traj[t:t+k]  (k={_k2} states)\n'
                        f'iterative: {n_planner_steps} plan-states (k-1={_k2 - 1})\n'
                        f'K={iter_state_chunk_h} states from trajectory[1:K+1] / chunk'
                    ]
            else:
                text_lines = [
                    f'Planner steps: {n_planner_steps}\n(budget: {max_steps_eff})\n'
                    f'K={iter_state_chunk_h} (trajectory[1:K+1] per chunk)\n'
                    f'dataset len: {data_len}'
                ]
                if tol > 0:
                    if reached:
                        text_lines.append(f'Goal reached in {n_trans} trans. (tol≤{tol:g})')
                    else:
                        text_lines.append(f'Not reached — dist at end {final_dist:.3f} (tol={tol:g})')
            if stochastic_run and plan_seed is not None:
                text_lines.append(f'sample_plan  noise_scale={args.sample_noise_scale:g}  seed={plan_seed}')
            if seg_metrics is not None:
                text_lines.append(
                    f"mean ‖true−GOUB‖₂ (full): {seg_metrics['mean_l2_full']:.3f}\n"
                    f"mean ‖true−GOUB‖₂ (xy): {seg_metrics['mean_l2_xy']:.3f}"
                )
            ax.text(
                0.02,
                0.98,
                '\n'.join(text_lines),
                transform=ax.transAxes,
                va='top',
                ha='left',
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85),
            )
            ax.legend(loc='lower right', fontsize=8, framealpha=0.93, borderaxespad=0.4)
            xlim, ylim = axis_limits(traj, roll, hats, d0, d1, s_g, s0, navigator=plot_nav, seg=seg)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_aspect('equal', adjustable='box')
            if plot_nav is not None and d0 == 0 and d1 == 1:
                ax.grid(False)
            else:
                ax.grid(True, alpha=0.3)
            fig.tight_layout(pad=0.9)
            out = Path(args.out_path)
            if plan_seed is not None:
                out = out.with_name(f'{out.stem}_seed{plan_seed}{out.suffix}')
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f'Wrote PNG {out.resolve()}')
    
            if (not args.no_mp4) and len(rollouts) == 1:
                mp4_out = Path(args.out_mp4.strip()) if args.out_mp4.strip() else out.with_suffix('.mp4')
                title_prefix = f'epoch={ckpt_epoch} traj={args.traj_idx} data_len={data_len}'
                if args.segment_compare and segment_t is not None:
                    title_prefix += f' seg_t={segment_t}'
                try:
                    write_rollout_mp4(
                        traj,
                        roll,
                        hats,
                        s0,
                        s_g,
                        d0,
                        d1,
                        mp4_out,
                        float(args.fps),
                        title_prefix,
                        navigator=plot_nav,
                        seg=seg,
                        chunk_hat_stride=int(iter_state_chunk_h),
                    )
                    print(f'Wrote MP4 {mp4_out.resolve()}')
                except Exception as e:
                    print(f'Warning: MP4 export failed ({e!r}). Ensure ffmpeg is installed and on PATH.')
            print(
                f'Episode length={len(traj)}  planner_steps={n_planner_steps}  max_steps={max_steps_eff}  '
                f'transitions={n_trans}  goal_reached={reached}  final_dist={final_dist:.4f}  '
                f'goal_tol={tol}  goal_stop={args.goal_stop}  checkpoint_suffix={ckpt_epoch}'
            )
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
