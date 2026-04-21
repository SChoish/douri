"""Train DQC-style chunk/value/flow module (offline)."""

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

from agents.dqc import DQCAgent
from utils.dqc_sequence_dataset import DQCActionSeqDataset
from utils.env_utils import make_env_and_datasets
from utils.flax_utils import save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb

FLAGS = flags.FLAGS


def _impl_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _default_yaml_path():
    return os.path.join(_impl_dir(), 'config', 'dqc_antmaze.yaml')


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


flags.DEFINE_string('run_config', '', 'YAML config; empty uses config/dqc_antmaze.yaml.')
flags.DEFINE_string('runs_root', '', 'Run root; default <repo>/runs.')
flags.DEFINE_string('run_group', 'Debug', 'W&B group.')
flags.DEFINE_integer('seed', 0, 'Seed.')
flags.DEFINE_string('env_name', 'antmaze-medium-navigate-v0', 'OGBench env / dataset name.')
flags.DEFINE_integer('train_epochs', 10, 'Training epochs.')
flags.DEFINE_integer('log_every_n_epochs', 1, 'Log interval (epochs).')
flags.DEFINE_integer('save_every_n_epochs', 10, 'Checkpoint interval.')
flags.DEFINE_boolean('use_wandb', False, 'W&B.')
flags.DEFINE_boolean('use_tqdm', False, 'tqdm over epochs.')

config_flags.DEFINE_config_file('agent', 'agents/dqc.py', lock_config=False)


def _steps_per_epoch(dataset_size: int, batch_size: int) -> int:
    return max(1, math.ceil(dataset_size / batch_size))


def _setup_file_logger(run_dir: str) -> logging.Logger:
    log_path = os.path.join(run_dir, 'run.log')
    logger = logging.getLogger('dqc')
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
    if config['agent_name'] != 'dqc':
        raise ValueError(f"Expected agent_name='dqc', got {config['agent_name']!r}")
    if config['dataset_class'] != 'DQCActionSeqDataset':
        raise ValueError(f"Expected dataset_class='DQCActionSeqDataset', got {config['dataset_class']!r}")

    ts = time.strftime('%Y%m%d_%H%M%S')
    env_tok = _sanitize_token(FLAGS.env_name)
    run_folder = f'{ts}_dqc_seed{FLAGS.seed}_{env_tok}'
    runs_root = FLAGS.runs_root.strip() or os.path.join(impl, 'runs')
    run_dir = os.path.join(runs_root, run_folder)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    if os.path.isfile(cfg_path):
        shutil.copy2(cfg_path, os.path.join(run_dir, 'config_used.yaml'))

    exp_name = get_exp_name(FLAGS.seed, env_name=FLAGS.env_name, agent_name=config['agent_name'])
    if FLAGS.use_wandb:
        setup_wandb(project='OGBench-DQC', group=FLAGS.run_group, name=exp_name)
        project = wandb.run.project
    else:
        project = 'OGBench-DQC'

    with open(os.path.join(run_dir, 'flags.json'), 'w', encoding='utf-8') as f:
        json.dump(get_flag_dict(), f, indent=2)

    run_logger = _setup_file_logger(run_dir)
    run_logger.info('run_dir=%s project=%s', run_dir, project)

    env, train_plain, _ = make_env_and_datasets(FLAGS.env_name, frame_stack=config['frame_stack'])
    config['action_dim'] = int(np.asarray(env.action_space.shape).prod())
    train_dataset = DQCActionSeqDataset(train_plain, config)

    np.random.seed(FLAGS.seed)
    ex = train_dataset.sample(1)
    agent = DQCAgent.create(
        FLAGS.seed,
        ex['observations'],
        ex['high_value_action_chunks'],
        ex['partial_action_chunks'],
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
        last_info = None
        for _ in range(spe):
            batch = train_dataset.sample(batch_size)
            agent, info = agent.update(batch)
            last_info = info

        gstep = epoch * spe
        if epoch % FLAGS.log_every_n_epochs == 0 and last_info is not None:
            metrics = {f'train/{k}': float(v) for k, v in last_info.items()}
            metrics['train/epoch'] = float(epoch)
            metrics['time/wall_sec'] = time.time() - last_log
            metrics['time/total_sec'] = time.time() - first_time
            last_log = time.time()
            if FLAGS.use_wandb:
                wandb.log(metrics, step=gstep)
            train_logger.log(metrics, step=gstep)
            run_logger.info('epoch=%d dqc_total_loss=%.6f', epoch, float(last_info['dqc/total_loss']))

        if epoch % FLAGS.save_every_n_epochs == 0:
            save_agent(agent, ckpt_dir, epoch)

    train_logger.close()
    run_logger.info('done run_dir=%s', run_dir)


if __name__ == '__main__':
    app.run(main)
