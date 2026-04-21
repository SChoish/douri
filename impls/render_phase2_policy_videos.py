#!/usr/bin/env python3
"""Load a saved GOUB phase-2 policy checkpoint and render one MP4 per OGBench ``task_id``.

Example::

    cd impls
    python render_phase2_policy_videos.py \\
        --run_dir=runs/20260419_174012_goub_phase2_policy_seed0_antmaze-medium-navigate-v0 \\
        --checkpoint_step=489000
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import jax
import numpy as np
from ml_collections import ConfigDict

from agents.goub_phase2_policy import GOUBPhase2PolicyAgent
from utils.datasets import HGCDataset
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent
from utils.goub_rollout_env import configure_mujoco_gl
from utils.goub_rollout_plot import write_rgb_array_mp4


def _impl_dir() -> Path:
    return Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run_dir', type=str, required=True, help='Phase-2 run folder (contains flags.json, checkpoints/).')
    p.add_argument(
        '--checkpoint_step',
        type=int,
        required=True,
        help='Training step suffix, e.g. 300000 for checkpoints/params_300000.pkl',
    )
    p.add_argument('--task_ids', type=str, default='1,2,3,4,5', help='Comma-separated task_id values (OGBench eval).')
    p.add_argument('--video_frame_skip', type=int, default=3)
    p.add_argument('--fps', type=float, default=20.0)
    p.add_argument('--mujoco_gl', type=str, default='', metavar='BACKEND')
    p.add_argument('--eval_on_cpu', type=int, default=1, choices=(0, 1))
    args = p.parse_args()

    try:
        configure_mujoco_gl(str(args.mujoco_gl))
    except ValueError as e:
        p.error(str(e))

    impl = _impl_dir()
    run_dir = Path(args.run_dir).expanduser()
    if not run_dir.is_absolute():
        run_dir = (impl / run_dir).resolve()
    else:
        run_dir = run_dir.resolve()

    flags_path = run_dir / 'flags.json'
    if not flags_path.is_file():
        raise FileNotFoundError(f'Missing flags.json: {flags_path}')
    with open(flags_path, 'r', encoding='utf-8') as f:
        flags = json.load(f)

    env_name = str(flags['env_name'])
    seed = int(flags['seed'])
    config = ConfigDict(flags['agent'])

    ckpt = run_dir / 'checkpoints' / f'params_{int(args.checkpoint_step)}.pkl'
    if not ckpt.is_file():
        raise FileNotFoundError(f'Missing checkpoint: {ckpt}')

    task_ids = [int(x.strip()) for x in str(args.task_ids).split(',') if x.strip()]

    os.chdir(impl)
    env, train_dataset, _ = make_env_and_datasets(
        env_name,
        frame_stack=config['frame_stack'],
        render_mode='rgb_array',
    )
    train_hgc = HGCDataset(train_dataset, config)

    np.random.seed(seed)
    random_state = __import__('random')
    random_state.seed(seed)

    example_batch = train_hgc.sample(1)
    agent = GOUBPhase2PolicyAgent.create(
        seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )
    # ``restore_agent`` expects a glob matching exactly one directory, then loads
    # ``<that_dir>/params_<step>.pkl`` (see ``utils.flax_utils.restore_agent``).
    agent = restore_agent(agent, str(run_dir / 'checkpoints'), int(args.checkpoint_step))

    if int(args.eval_on_cpu):
        agent = jax.device_put(agent, device=jax.devices('cpu')[0])

    eval_temperature = float(flags.get('eval_temperature', 0.0))
    eval_gaussian = flags.get('eval_gaussian', None)
    if eval_gaussian is not None:
        eval_gaussian = float(eval_gaussian)

    for task_id in task_ids:
        print(f'--- task_id={task_id} (checkpoint_step={int(args.checkpoint_step)}) ---')
        _stats, _trajs, renders = evaluate(
            agent=agent,
            env=env,
            task_id=task_id,
            config=config,
            num_eval_episodes=0,
            num_video_episodes=1,
            video_frame_skip=int(args.video_frame_skip),
            eval_temperature=eval_temperature,
            eval_gaussian=eval_gaussian,
            disable_tqdm=True,
        )
        if not renders:
            print(f'Warning: no render buffer for task_id={task_id}')
            continue
        frames = np.asarray(renders[0], dtype=np.uint8)
        if frames.dtype != np.uint8:
            frames = np.clip(frames, 0.0, 255.0).astype(np.uint8)
        out_mp4 = run_dir / f'phase2_policy_render_step{int(args.checkpoint_step)}_task{task_id}.mp4'
        write_rgb_array_mp4(frames, out_mp4, float(args.fps))
        print(f'Wrote {out_mp4}')


if __name__ == '__main__':
    main()
