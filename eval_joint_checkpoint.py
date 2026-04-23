#!/usr/bin/env python3
"""Load a joint run checkpoint and run the same env eval as training (``main._evaluate_env_tasks``).

Reads ``flags.json`` + ``checkpoints/{goub,critic,actor}/params_<epoch>.pkl``.

Example::

    MUJOCO_GL=egl python eval_joint_checkpoint.py \\
        --run_dir=runs/20260422_015908_joint_dqc_seed0_antmaze-medium-navigate-v0 \\
        --epoch=1000

IDM env-eval uses ``--idm_action_chunk_horizon`` (default **5**) for ``_idm_action_chunk`` only; the
saved critic YAML may still use a larger ``action_chunk_horizon`` for training.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from agents.critic import get_config as get_critic_config, validate_joint_mode
from agents.actor import get_actor_config
from agents.goub_dynamics import GOUBDynamicsAgent, get_dynamics_config
from main import (
    _create_actor_agent,
    _create_critic_agent,
    _evaluate_env_tasks,
    _extract_critic_value_params,
    _intersect_valid_starts,
    _make_critic_dataset,
    _require_matching_frame_stack,
    _sample_shared_idxs,
    _update_config,
)
from utils.datasets import Dataset, PathHGCDataset
from utils.env_utils import make_env_and_datasets
from utils.run_io import (
    list_checkpoint_suffixes,
    load_checkpoint_pkl,
    parse_int_list,
    pick_epoch,
    resolve_actor_checkpoint_dir,
    resolve_critic_checkpoint_dir,
    resolve_goub_checkpoint_dir,
)


def _build_configs(root: dict, fg: dict) -> tuple[Any, Any, Any]:
    joint_h = int(fg['joint_horizon'])
    goub_config = _update_config(get_dynamics_config(), root['goub'])
    critic_config = _update_config(get_critic_config(), root['critic_agent'])
    # Saved ``flags.json`` holds the full merged actor dict (not only SPI keys).
    actor_config = _update_config(get_actor_config(), root['actor'])
    goub_config['goub_N'] = joint_h
    goub_config['subgoal_steps'] = joint_h
    critic_config['full_chunk_horizon'] = joint_h
    actor_config['actor_chunk_horizon'] = int(critic_config['action_chunk_horizon'])
    validate_joint_mode(critic_config, actor_config)
    bs = int(fg['batch_size'])
    goub_config['batch_size'] = bs
    critic_config['batch_size'] = bs
    actor_config['batch_size'] = bs
    _require_matching_frame_stack(goub_config, critic_config)
    return goub_config, critic_config, actor_config


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run_dir', type=str, required=True)
    p.add_argument('--epoch', type=int, default=1000, help='Checkpoint suffix for goub/critic/actor.')
    p.add_argument('--seed', type=int, default=-1, help='Agent RNG seed; -1 uses flags.json flags.seed.')
    p.add_argument('--eval_task_ids', type=str, default='', help='Override e.g. "1,2,3,4,5" (empty = flags).')
    p.add_argument('--eval_episodes_per_task', type=int, default=-1, help='-1 = use flags.')
    p.add_argument('--eval_max_chunks', type=int, default=-1, help='-1 = use flags.')
    p.add_argument('--eval_goal_tol', type=float, default=-1.0, help='<0 = use flags.')
    p.add_argument('--eval_goal_dims', type=str, default='', help='Override e.g. "0,1" (empty = flags).')
    p.add_argument(
        '--idm_action_chunk_horizon',
        type=int,
        default=5,
        help='Env-eval IDM: env steps per replan (passed as critic_config action_chunk_horizon only for _evaluate_env_tasks).',
    )
    p.add_argument('--mujoco_gl', type=str, default='', metavar='BACKEND')
    args = p.parse_args()

    if str(args.mujoco_gl).strip():
        from rollout.env import configure_mujoco_gl

        configure_mujoco_gl(str(args.mujoco_gl))

    run_dir = Path(args.run_dir).resolve()
    flags_path = run_dir / 'flags.json'
    if not flags_path.is_file():
        raise FileNotFoundError(flags_path)
    with open(flags_path, 'r', encoding='utf-8') as f:
        root = json.load(f)
    fg = root['flags']
    seed = int(fg['seed']) if int(args.seed) < 0 else int(args.seed)

    goub_config, critic_config, actor_config = _build_configs(root, fg)
    env_name = fg['env_name']
    env, train_plain, _ = make_env_and_datasets(env_name, frame_stack=critic_config['frame_stack'])
    action_dim = int(np.asarray(env.action_space.shape).prod())
    critic_config['action_dim'] = action_dim
    actor_config['action_dim'] = action_dim

    goub_dataset = PathHGCDataset(Dataset.create(**train_plain), goub_config)
    critic_dataset = _make_critic_dataset(train_plain, critic_config)
    common = _intersect_valid_starts(goub_dataset, critic_dataset)
    bs = int(goub_config['batch_size'])
    ex_idxs = _sample_shared_idxs(common, bs)
    ex_goub = goub_dataset.sample(len(ex_idxs), idxs=ex_idxs)
    ex_critic = critic_dataset.sample(len(ex_idxs), idxs=ex_idxs)

    ex = jnp.asarray(ex_goub['observations'], dtype=jnp.float32)
    ex_act = jnp.asarray(ex_goub['actions'], dtype=jnp.float32)
    goub_agent = GOUBDynamicsAgent.create(seed, ex, goub_config, ex_actions=ex_act)
    critic_agent = _create_critic_agent(seed, ex_critic, critic_config)
    actor_agent = _create_actor_agent(seed, ex_goub, actor_config)

    goub_dir = resolve_goub_checkpoint_dir(run_dir)
    ep = pick_epoch(int(args.epoch), list_checkpoint_suffixes(goub_dir))
    goub_pkl = goub_dir / f'params_{ep}.pkl'
    critic_pkl = resolve_critic_checkpoint_dir(run_dir) / f'params_{ep}.pkl'
    goub_agent = load_checkpoint_pkl(goub_agent, goub_pkl)
    critic_agent = load_checkpoint_pkl(critic_agent, critic_pkl)
    actor_dir = resolve_actor_checkpoint_dir(run_dir, required=True)
    actor_pkl = actor_dir / f'params_{ep}.pkl'
    if not actor_pkl.is_file():
        raise FileNotFoundError(actor_pkl)
    actor_agent = load_checkpoint_pkl(actor_agent, actor_pkl)

    task_ids = parse_int_list(args.eval_task_ids) if str(args.eval_task_ids).strip() else parse_int_list(
        str(fg.get('eval_task_ids', '1'))
    )
    ep_task = int(fg['eval_episodes_per_task']) if int(args.eval_episodes_per_task) < 0 else int(args.eval_episodes_per_task)
    max_chunks = int(fg['eval_max_chunks']) if int(args.eval_max_chunks) < 0 else int(args.eval_max_chunks)
    goal_tol = float(fg['eval_goal_tol']) if float(args.eval_goal_tol) < 0 else float(args.eval_goal_tol)
    goal_dims_text = str(args.eval_goal_dims).strip() or str(fg.get('eval_goal_dims', '0,1'))
    goal_dims = parse_int_list(goal_dims_text) or None

    critic_eval = copy.deepcopy(critic_config)
    idm_h = int(args.idm_action_chunk_horizon)
    if idm_h < 1:
        p.error('--idm_action_chunk_horizon must be >= 1')
    critic_eval['action_chunk_horizon'] = idm_h

    print(f'Loaded epoch={ep} from {run_dir}')
    print(
        f'eval task_ids={task_ids} episodes_per_task={ep_task} max_chunks={max_chunks} '
        f'goal_tol={goal_tol} goal_dims={goal_dims}  idm_action_chunk_horizon={idm_h} '
        f'(training critic had {int(critic_config["action_chunk_horizon"])})'
    )

    metrics = _evaluate_env_tasks(
        env,
        goub_agent,
        actor_agent,
        actor_config,
        critic_eval,
        critic_value_params=_extract_critic_value_params(critic_agent),
        task_ids=task_ids,
        episodes_per_task=ep_task,
        max_chunks=max_chunks,
        goal_tol=goal_tol,
        goal_dims=goal_dims,
    )
    print('--- IDM ---')
    print(f"eval_idm/success_rate_mean={metrics.get('eval_idm/success_rate_mean', float('nan')):.4f}")
    for tid in task_ids:
        k = f'eval_idm/task_{tid}/success_rate'
        if k in metrics:
            print(f'  {k}={metrics[k]:.4f}')
    print('--- Actor ---')
    print(f"eval/success_rate_mean={metrics.get('eval/success_rate_mean', float('nan')):.4f}")
    for tid in task_ids:
        k = f'eval/task_{tid}/success_rate'
        if k in metrics:
            print(f'  {k}={metrics[k]:.4f}')


if __name__ == '__main__':
    main()
