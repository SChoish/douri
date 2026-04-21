"""Train unified critic stacks (DEAS or DQC, critic-only, offline).

Example::

    cd /path/to/douri
    export PYTHONPATH=.
    python main_critic.py --env_name=antmaze-medium-navigate-v0 --train_epochs=2
"""

import json
import logging
import math
import os
import re
import shutil
import sys
import time

import numpy as np
import tqdm
import wandb
import yaml
from absl import app, flags
from ml_collections import config_flags

from agents.critic import get_critic_class
from utils.datasets import Dataset
from utils.deas_sequence_dataset import DEASActionSeqDataset
from utils.dqc_sequence_dataset import DQCActionSeqDataset
from utils.env_utils import make_env_and_datasets
from utils.flax_utils import save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb

FLAGS = flags.FLAGS


def _impl_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _default_yaml_path():
    return os.path.join(_impl_dir(), 'config', 'critic_antmaze.yaml')


def _sanitize_token(s: str) -> str:
    s = re.sub(r'[^\w.\-]+', '_', s)
    return s[:120] if len(s) > 120 else s


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
            FLAGS.agent[k] = v


flags.DEFINE_string('run_config', '', 'YAML config; empty uses config/critic_antmaze.yaml.')
flags.DEFINE_string('runs_root', '', 'Run root; default <repo>/runs.')
flags.DEFINE_string('run_group', 'Debug', 'W&B group.')
flags.DEFINE_integer('seed', 0, 'Seed.')
flags.DEFINE_string(
    'env_name',
    'antmaze-medium-navigate-v0',
    'OGBench env / dataset name (YAML may override).',
)
flags.DEFINE_integer('train_epochs', 10, 'Training epochs.')
flags.DEFINE_integer('log_every_n_epochs', 1, 'Log interval (epochs).')
flags.DEFINE_integer('save_every_n_epochs', 10, 'Checkpoint interval.')
flags.DEFINE_boolean('use_wandb', False, 'W&B.')
flags.DEFINE_boolean('use_tqdm', False, 'tqdm over epochs.')
flags.DEFINE_enum('critic', 'deas', ['deas', 'dqc'], 'Critic stack to use.')

config_flags.DEFINE_config_file('agent', 'agents/critic/__init__.py', lock_config=False)


def _steps_per_epoch(dataset_size: int, batch_size: int) -> int:
    return max(1, math.ceil(dataset_size / batch_size))


def _setup_file_logger(run_dir: str) -> logging.Logger:
    log_path = os.path.join(run_dir, 'run.log')
    logger = logging.getLogger('critic')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def main(_):
    impl = _impl_dir()
    cfg_path = FLAGS.run_config.strip() or _default_yaml_path()
    if os.path.isfile(cfg_path):
        _apply_yaml_to_flags(_load_yaml(cfg_path))
    elif FLAGS.run_config.strip():
        raise FileNotFoundError(f'run_config YAML not found: {cfg_path}')

    config = FLAGS.agent
    critic_name = str(FLAGS.critic).lower()
    critic_cls = get_critic_class(critic_name)

    ts = time.strftime('%Y%m%d_%H%M%S')
    env_tok = _sanitize_token(FLAGS.env_name)
    run_folder = f'{ts}_{critic_name}_critic_seed{FLAGS.seed}_{env_tok}'
    runs_root = FLAGS.runs_root.strip() or os.path.join(impl, 'runs')
    run_dir = os.path.join(runs_root, run_folder)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    if os.path.isfile(cfg_path):
        shutil.copy2(cfg_path, os.path.join(run_dir, 'config_used.yaml'))

    exp_name = get_exp_name(FLAGS.seed, env_name=FLAGS.env_name, agent_name=f'{critic_name}_critic')
    if FLAGS.use_wandb:
        setup_wandb(project='OGBench-Critic', group=FLAGS.run_group, name=exp_name)
    with open(os.path.join(run_dir, 'flags.json'), 'w', encoding='utf-8') as f:
        json.dump(get_flag_dict(), f, indent=2)

    run_logger = _setup_file_logger(run_dir)
    run_logger.info('run_dir=%s', run_dir)
    run_logger.info('critic=%s', critic_name)

    _, train_ds_plain, _ = make_env_and_datasets(FLAGS.env_name, frame_stack=config['frame_stack'])
    if critic_name == 'deas':
        train_dataset = DEASActionSeqDataset(Dataset.create(**train_ds_plain), config)
    else:
        train_dataset = DQCActionSeqDataset(Dataset.create(**train_ds_plain), config)

    np.random.seed(FLAGS.seed)
    ex = train_dataset.sample(1)
    if critic_name == 'deas':
        agent = critic_cls.create(
            FLAGS.seed,
            ex['observations'],
            ex['actions'],
            config,
        )
    else:
        agent = critic_cls.create(
            FLAGS.seed,
            ex['observations'],
            ex['full_chunk_actions'],
            ex['action_chunk_actions'],
            config,
        )

    batch_size = int(config['batch_size'])
    spe = _steps_per_epoch(train_dataset.size, batch_size)
    train_logger = CsvLogger(os.path.join(run_dir, 'train.csv'))
    first_time = time.time()
    last_log = time.time()

    epoch_iter = range(1, FLAGS.train_epochs + 1)
    if FLAGS.use_tqdm:
        epoch_iter = tqdm.tqdm(epoch_iter, smoothing=0.1, dynamic_ncols=True)

    for epoch in epoch_iter:
        losses_v, losses_c = [], []
        last_info = None
        for _ in range(spe):
            batch = train_dataset.sample(batch_size)
            agent, info = agent.update(batch)
            last_info = info
            if critic_name == 'deas':
                losses_v.append(float(last_info['value/value_loss']))
                losses_c.append(float(last_info['critic/critic_loss']))
            else:
                losses_v.append(float(last_info['action_critic/value_loss']))
                losses_c.append(float(last_info['chunk_critic/critic_loss']))

        gstep = epoch * spe
        if epoch % FLAGS.log_every_n_epochs == 0 and last_info is not None:
            metrics = {f'train/{k}': float(v) for k, v in last_info.items()}
            metrics['train/value_loss_epoch_mean'] = float(np.mean(losses_v))
            metrics['train/critic_loss_epoch_mean'] = float(np.mean(losses_c))
            metrics['train/epoch'] = float(epoch)
            metrics['time/wall_sec'] = time.time() - last_log
            metrics['time/total_sec'] = time.time() - first_time
            last_log = time.time()
            if FLAGS.use_wandb:
                wandb.log(metrics, step=gstep)
            train_logger.log(metrics, step=gstep)
            run_logger.info('epoch=%d %s', epoch, ' '.join(f'{k}={v:.4g}' for k, v in list(metrics.items())[:8]))

        if epoch % FLAGS.save_every_n_epochs == 0:
            save_agent(agent, ckpt_dir, epoch)

    train_logger.close()
    run_logger.info('done run_dir=%s', run_dir)


if __name__ == '__main__':
    app.run(main)
