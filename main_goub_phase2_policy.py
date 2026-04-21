"""GOUB Phase-2 policy training on top of frozen GOUB phase1 checkpoints.

This entrypoint is additive to the cleaned repo:

* Phase1 planner + IDM stay frozen and external to the trainable phase2 agent.
* A new goal-conditioned phase2 actor/critic(/value) is trained with `value_goals`.
* Training can run:
  1. proposal distillation from frozen GOUB+IDM action proposals, then
  2. optional offline RL fine-tuning with either IQL or TD3+BC.

Run layout
----------
`runs/<timestamp>_goub_phase2_policy_seed<seed>_<env>/`
  * `config_used.yaml`
  * `flags.json`
  * `train.csv`
  * `eval.csv` (when evaluation is enabled)
  * `run.log`
  * `checkpoints/params_<step>.pkl`

Resume
------
``--resume_run_dir=<existing run>`` + ``--resume_step=<N>`` loads ``checkpoints/params_<N>.pkl``,
truncates ``train.csv`` / ``eval.csv`` rows with ``step > N``, then trains from ``N+1`` to
``train_steps`` (append logs). Update ``--train_steps`` / distill+finetune so they still sum.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import random
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import jax
import numpy as np
import tqdm
import wandb
import yaml
from absl import app, flags
from ml_collections import ConfigDict, config_flags

from agents.goub_phase2_policy import GOUBPhase2PolicyAgent
from utils.datasets import HGCDataset
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent
from utils.goub_phase2_utils import FrozenGOUBProposalGenerator, load_frozen_goub_bundle
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb

FLAGS = flags.FLAGS


def _impl_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _default_yaml_path():
    return os.path.join(_impl_dir(), 'config', 'goub_phase2_policy_antmaze.yaml')


def _sanitize_token(s: str) -> str:
    s = re.sub(r'[^\w.\-]+', '_', s)
    return s[:120] if len(s) > 120 else s


flags.DEFINE_string(
    'run_config',
    '',
    'YAML training config. Empty -> use config/goub_phase2_policy_antmaze.yaml if it exists.',
)
flags.DEFINE_string(
    'runs_root',
    '',
    'Directory for timestamped run folders. Empty -> <impl>/runs',
)
flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.')

flags.DEFINE_integer('train_steps', 1000000, 'Total phase2 training steps.')
flags.DEFINE_integer('distill_steps', 300000, 'Proposal-distillation steps at the beginning.')
flags.DEFINE_integer('finetune_steps', 700000, 'Offline RL fine-tuning steps after distillation.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('eval_interval', 100000, 'Evaluation interval; <=0 disables evaluation.')
flags.DEFINE_integer('save_interval', 100000, 'Checkpoint interval.')

flags.DEFINE_string('phase1_run_dir', '', 'Frozen phase1 run directory containing flags.json and checkpoints/.')
flags.DEFINE_integer(
    'phase1_checkpoint_epoch',
    -1,
    'Frozen phase1 params_<n>.pkl suffix. -1 picks the latest available checkpoint.',
)
flags.DEFINE_string(
    'idm_checkpoint',
    '',
    'Optional standalone IDM params_*.pkl. Empty -> use embedded idm_net from the phase1 checkpoint.',
)

flags.DEFINE_integer('eval_tasks', None, 'Number of eval tasks (None for all).')
flags.DEFINE_integer('eval_episodes', 20, 'Episodes per eval task.')
flags.DEFINE_float('eval_temperature', 0.0, 'Actor temperature for evaluation.')
flags.DEFINE_float('eval_gaussian', None, 'Optional extra Gaussian action noise during evaluation.')
flags.DEFINE_integer('video_episodes', 0, 'Rendered episodes per task.')
flags.DEFINE_integer('video_frame_skip', 3, 'Frame skip for rendered videos.')
flags.DEFINE_integer('eval_on_cpu', 1, 'Whether to move the eval agent to CPU.')

flags.DEFINE_boolean('use_wandb', False, 'Log to Weights & Biases.')
flags.DEFINE_boolean('use_tqdm', False, 'Show tqdm progress bars during training.')

flags.DEFINE_string(
    'resume_run_dir',
    '',
    'Existing phase-2 run directory to continue (append train/eval logs, reuse checkpoints/).',
)
flags.DEFINE_integer(
    'resume_step',
    0,
    'Load checkpoints/params_<resume_step>.pkl and continue training from resume_step+1.',
)

config_flags.DEFINE_config_file('agent', 'agents/goub_phase2_policy.py', lock_config=False)


def _load_yaml(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _argv_sets_flag(flag_name: str) -> bool:
    dashed = flag_name.replace('_', '-')
    for arg in sys.argv[1:]:
        if arg.startswith(f'--{flag_name}=') or arg.startswith(f'--{dashed}='):
            return True
        if arg in (f'--{flag_name}', f'--{dashed}'):
            return True
    return False


def _argv_sets_agent_key(key: str) -> bool:
    """True if ``sys.argv`` sets ``FLAGS.agent.<key>`` via ``--agent.<key>=...``."""
    key_snake = str(key)
    key_dash = key_snake.replace('_', '-')
    for arg in sys.argv[1:]:
        if arg.startswith(f'--agent.{key_snake}=') or arg.startswith(f'--agent.{key_dash}='):
            return True
        if arg in (f'--agent.{key_snake}', f'--agent.{key_dash}'):
            return True
    return False


def _apply_yaml_to_flags(data: dict) -> None:
    agent_updates = data.pop('agent', None)
    if agent_updates is not None and not isinstance(agent_updates, dict):
        raise ValueError('YAML key "agent" must be a mapping.')

    for key, value in data.items():
        if not hasattr(FLAGS, key):
            raise ValueError(f'Unknown YAML top-level key: {key!r}')
        if _argv_sets_flag(key):
            continue
        setattr(FLAGS, key, value)

    if agent_updates:
        for k, v in agent_updates.items():
            if _argv_sets_agent_key(k):
                continue
            FLAGS.agent[k] = v


def _setup_file_logger(run_dir: str, *, append: bool = False) -> logging.Logger:
    log_path = os.path.join(run_dir, 'run.log')
    logger = logging.getLogger('goub_phase2_policy')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding='utf-8', mode='a' if append else 'w')
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def _detect_rl_algo_from_phase2_checkpoint(ckpt_path: str) -> str:
    """Return ``'iql'`` if the pickle contains a ``modules_value`` subtree, else ``'td3bc'``."""
    with open(ckpt_path, 'rb') as f:
        load_dict = pickle.load(f)

    def _contains_modules_value(obj) -> bool:
        if isinstance(obj, dict):
            if 'modules_value' in obj:
                return True
            return any(_contains_modules_value(v) for v in obj.values())
        return False

    agent_sd = load_dict.get('agent')
    if agent_sd is None:
        raise KeyError(f'Checkpoint missing agent state: {ckpt_path}')
    return 'iql' if _contains_modules_value(agent_sd) else 'td3bc'


def _truncate_csv_rows_after_step(csv_path: str, max_step: int) -> None:
    """Keep header and data rows whose last column ``step`` is <= ``max_step``."""
    if not os.path.isfile(csv_path):
        return
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if len(lines) <= 1:
        return
    header, body = lines[0], lines[1:]
    kept = [header]
    for line in body:
        parts = line.rstrip('\n').split(',')
        if not parts:
            continue
        try:
            step_val = int(float(parts[-1]))
        except ValueError:
            kept.append(line if line.endswith('\n') else line + '\n')
            continue
        if step_val <= int(max_step):
            kept.append(line if line.endswith('\n') else line + '\n')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.writelines(kept)


def _resolve_total_steps() -> int:
    total = int(FLAGS.distill_steps) + int(FLAGS.finetune_steps)
    if total <= 0:
        raise ValueError('distill_steps + finetune_steps must be > 0')
    if int(FLAGS.train_steps) != total:
        raise ValueError(
            f'train_steps ({int(FLAGS.train_steps)}) must equal distill_steps + finetune_steps ({total}).'
        )
    return total


def _run_evaluation(agent, env, config, step: int, eval_logger: CsvLogger | None):
    if int(FLAGS.eval_interval) <= 0:
        return
    if FLAGS.eval_on_cpu:
        eval_agent = jax.device_put(agent, device=jax.devices('cpu')[0])
    else:
        eval_agent = agent

    task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
    num_tasks = FLAGS.eval_tasks if FLAGS.eval_tasks is not None else len(task_infos)
    eval_metrics = {}
    overall_metrics = defaultdict(list)
    task_iter = range(1, num_tasks + 1)
    if FLAGS.use_tqdm:
        task_iter = tqdm.trange(1, num_tasks + 1, desc='eval_tasks', leave=False)

    for task_id in task_iter:
        task_name = task_infos[task_id - 1]['task_name']
        eval_info, _trajs, _renders = evaluate(
            agent=eval_agent,
            env=env,
            task_id=task_id,
            config=config,
            num_eval_episodes=FLAGS.eval_episodes,
            num_video_episodes=FLAGS.video_episodes,
            video_frame_skip=FLAGS.video_frame_skip,
            eval_temperature=FLAGS.eval_temperature,
            eval_gaussian=FLAGS.eval_gaussian,
            disable_tqdm=not FLAGS.use_tqdm,
        )
        metric_names = ['success']
        eval_metrics.update({f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names})
        for k, v in eval_info.items():
            if k in metric_names:
                overall_metrics[k].append(v)

    for k, v in overall_metrics.items():
        eval_metrics[f'evaluation/overall_{k}'] = np.mean(v)

    if FLAGS.use_wandb and eval_metrics:
        wandb.log(eval_metrics, step=step)
    if eval_logger is not None and eval_metrics:
        eval_logger.log(eval_metrics, step=step)


def main(_):
    cfg_path = FLAGS.run_config.strip() or _default_yaml_path()
    if os.path.isfile(cfg_path):
        _apply_yaml_to_flags(_load_yaml(cfg_path))
    elif FLAGS.run_config.strip():
        raise FileNotFoundError(f'run_config YAML not found: {cfg_path}')

    resume_dir = str(FLAGS.resume_run_dir).strip()
    resume_step = int(FLAGS.resume_step)
    if resume_dir and resume_step <= 0:
        raise ValueError('--resume_step must be > 0 when --resume_run_dir is set.')
    if resume_step > 0 and not resume_dir:
        raise ValueError('--resume_run_dir is required when --resume_step > 0.')
    if not resume_dir:
        resume_step = 0

    append_logs = False
    if resume_dir:
        run_dir = resume_dir
        if not os.path.isabs(run_dir):
            run_dir = os.path.join(_impl_dir(), run_dir)
        run_dir = os.path.abspath(run_dir)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f'resume_run_dir not found: {run_dir}')
        ckpt_dir = os.path.join(run_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        _truncate_csv_rows_after_step(os.path.join(run_dir, 'train.csv'), resume_step)
        if int(FLAGS.eval_interval) > 0:
            _truncate_csv_rows_after_step(os.path.join(run_dir, 'eval.csv'), resume_step)
        # Replace agent config so the Flax template matches the checkpoint (YAML may be IQL vs saved TD3+BC).
        with open(os.path.join(run_dir, 'flags.json'), 'r', encoding='utf-8') as f:
            saved_flags = json.load(f)
        saved_agent = saved_flags.get('agent') or {}
        FLAGS.agent = ConfigDict(saved_agent)
        ckpt_path = os.path.join(ckpt_dir, f'params_{int(resume_step)}.pkl')
        detected = _detect_rl_algo_from_phase2_checkpoint(ckpt_path)
        if str(FLAGS.agent.get('rl_algo', '')) != detected:
            FLAGS.agent['rl_algo'] = detected
        append_logs = True
    else:
        ts = time.strftime('%Y%m%d_%H%M%S')
        env_tok = _sanitize_token(FLAGS.env_name)
        run_folder = f'{ts}_{FLAGS.agent["agent_name"]}_seed{FLAGS.seed}_{env_tok}'
        runs_root = FLAGS.runs_root.strip() or os.path.join(_impl_dir(), 'runs')
        run_dir = os.path.join(runs_root, run_folder)
        ckpt_dir = os.path.join(run_dir, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        if os.path.isfile(cfg_path):
            shutil.copy2(cfg_path, os.path.join(run_dir, 'config_used.yaml'))

    config = FLAGS.agent
    if config['agent_name'] != 'goub_phase2_policy':
        raise ValueError(
            f"This entrypoint is GOUB phase2 policy only, but got agent_name={config['agent_name']!r}."
        )
    if config['dataset_class'] != 'HGCDataset':
        raise ValueError(
            f"GOUB phase2 expects dataset_class='HGCDataset', but got {config['dataset_class']!r}."
        )
    if not str(FLAGS.phase1_run_dir).strip():
        raise ValueError('phase1_run_dir must be set for GOUB phase2 training.')

    total_steps = _resolve_total_steps()

    exp_name = get_exp_name(FLAGS.seed, env_name=FLAGS.env_name, agent_name=config['agent_name'])
    if FLAGS.use_wandb:
        setup_wandb(project='OGBench-GOUB', group=FLAGS.run_group, name=exp_name)
        project = wandb.run.project
    else:
        project = 'OGBench-GOUB'

    with open(os.path.join(run_dir, 'flags.json'), 'w', encoding='utf-8') as f:
        json.dump(get_flag_dict(), f, indent=2)

    run_logger = _setup_file_logger(run_dir, append=append_logs)
    run_logger.info('run_dir=%s', run_dir)
    run_logger.info('project=%s exp_name=%s', project, exp_name)
    run_logger.info(
        'phase2 schedule: distill_steps=%d finetune_steps=%d total_steps=%d rl_algo=%s',
        int(FLAGS.distill_steps),
        int(FLAGS.finetune_steps),
        total_steps,
        str(config['rl_algo']),
    )

    env, train_raw, _ = make_env_and_datasets(
        FLAGS.env_name,
        frame_stack=config['frame_stack'],
    )
    train_dataset = HGCDataset(train_raw, config)

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = train_dataset.sample(1)
    agent = GOUBPhase2PolicyAgent.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['actions'],
        config,
    )
    if int(resume_step) > 0:
        agent = restore_agent(agent, str(ckpt_dir), int(resume_step))
        run_logger.info('restored checkpoint resume_step=%d', int(resume_step))

    frozen_bundle = load_frozen_goub_bundle(
        phase1_run_dir=FLAGS.phase1_run_dir,
        phase1_checkpoint_epoch=int(FLAGS.phase1_checkpoint_epoch),
        example_observations=np.asarray(example_batch['observations'], dtype=np.float32),
        example_actions=np.asarray(example_batch['actions'], dtype=np.float32),
        seed=int(FLAGS.seed),
        idm_checkpoint=str(FLAGS.idm_checkpoint),
    )
    run_logger.info(
        'loaded frozen phase1: agent_name=%s env_name=%s checkpoint=%s',
        frozen_bundle.phase1_agent_name,
        frozen_bundle.env_name,
        str(frozen_bundle.checkpoint_path),
    )
    if frozen_bundle.env_name != str(FLAGS.env_name):
        raise ValueError(
            f'phase1 env_name {frozen_bundle.env_name!r} does not match phase2 env_name {str(FLAGS.env_name)!r}'
        )

    action_low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    action_high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    proposal_generator = FrozenGOUBProposalGenerator(
        frozen_bundle,
        action_low=action_low,
        action_high=action_high,
        num_action_samples=int(config['num_action_samples']),
        action_noise_std=float(config['action_noise_std']),
        include_mean_action=bool(config['include_mean_action']),
        include_dataset_action=bool(config['include_dataset_action']),
        planner_noise_scale=float(config['planner_noise_scale']),
        num_planner_samples=int(config['num_planner_samples']),
    )

    train_logger = CsvLogger(os.path.join(run_dir, 'train.csv'), resume=append_logs)
    eval_logger = (
        CsvLogger(os.path.join(run_dir, 'eval.csv'), resume=append_logs) if int(FLAGS.eval_interval) > 0 else None
    )
    first_time = time.time()
    last_time = time.time()
    train_start = int(resume_step) + 1 if int(resume_step) > 0 else 1
    if train_start > total_steps:
        raise ValueError(f'Nothing to train: resume_step+1={train_start} > train_steps={total_steps}')
    train_iter = range(train_start, total_steps + 1)
    if FLAGS.use_tqdm:
        train_iter = tqdm.tqdm(train_iter, smoothing=0.1, dynamic_ncols=True)

    for step in train_iter:
        batch = train_dataset.sample(config['batch_size'])
        if step <= int(FLAGS.distill_steps):
            teacher_info = proposal_generator.build(batch, seed=int(FLAGS.seed) + step)
            agent, update_info = agent.update_distill(batch, teacher_info)
            stage_name = 'distill'
        else:
            agent, update_info = agent.update_finetune(batch)
            stage_name = 'finetune'

        if step % int(FLAGS.log_interval) == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            train_metrics['training/stage_is_distill'] = float(step <= int(FLAGS.distill_steps))
            train_metrics['time/step_time'] = (time.time() - last_time) / int(FLAGS.log_interval)
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            if FLAGS.use_wandb:
                wandb.log(train_metrics, step=step)
            train_logger.log(train_metrics, step=step)
            run_logger.info(
                'step=%d stage=%s total_loss=%.6f',
                step,
                stage_name,
                float(update_info['phase2/total_loss']),
            )

        if int(FLAGS.eval_interval) > 0 and (step == 1 or step % int(FLAGS.eval_interval) == 0):
            _run_evaluation(agent, env, config, step=step, eval_logger=eval_logger)

        if step % int(FLAGS.save_interval) == 0:
            save_agent(agent, ckpt_dir, step)
            run_logger.info('saved checkpoint step=%d', step)

    if total_steps % int(FLAGS.save_interval) != 0:
        save_agent(agent, ckpt_dir, total_steps)
        run_logger.info('saved final checkpoint step=%d', total_steps)

    train_logger.close()
    if eval_logger is not None:
        eval_logger.close()


if __name__ == '__main__':
    app.run(main)
