"""GOUB-inspired Phase-1 training (YAML config + timestamped runs).

Layout
------
* Default YAML: ``<impl>/config/goub_phase1_antmaze.yaml`` (override with ``--run_config``).
* Each run: ``<runs_root>/<timestamp>_seed<seed>_<env>/``
    * ``config_used.yaml`` — copy of the loaded YAML
    * ``flags.json`` — resolved absl + agent flags
    * ``train.csv`` — metrics (CsvLogger; default log every **10** epochs)
    * ``run.log`` — human-readable epoch log (same cadence)
    * ``checkpoints/params_<epoch>.pkl`` — periodic + final saves (epoch index, not optimizer step); includes inverse-dynamics head.

**Resume:** ``--resume_pkl=.../params_<n>.pkl`` + ``--resume_start_epoch=<n>`` (last completed epoch in that run)
+ ``--train_epochs=<target>`` trains epochs ``n+1 .. target`` into a **new** timestamped ``run_dir``.

One **epoch** = ``ceil(dataset_size / batch_size)`` gradient updates.
"""

import json
import logging
import math
import os
import pickle
import random
import re
import shutil
import sys
import time

import flax
import numpy as np
import tqdm
import wandb
import yaml
from absl import app, flags
from ml_collections import config_flags

from agents.goub_phase1 import GOUBPhase1Agent
from utils.datasets import HGCDataset
from utils.env_utils import make_env_and_datasets
from utils.flax_utils import merge_checkpoint_state_dict, save_agent
from utils.inverse_dynamics_train import parse_hidden_dims
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb

FLAGS = flags.FLAGS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def _impl_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _default_yaml_path():
    return os.path.join(_impl_dir(), 'config', 'goub_phase1_antmaze.yaml')


def _sanitize_token(s: str) -> str:
    s = re.sub(r'[^\w.\-]+', '_', s)
    return s[:120] if len(s) > 120 else s


# ---------------------------------------------------------------------------
# absl flags
# ---------------------------------------------------------------------------

flags.DEFINE_string(
    'run_config',
    '',
    'YAML training config. Empty → use impls/config/goub_phase1_antmaze.yaml if it exists.',
)
flags.DEFINE_string(
    'runs_root',
    '',
    'Directory for timestamped run folders. Empty → <impl>/runs',
)

flags.DEFINE_string('run_group', 'Debug', 'Run group (W&B only).')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.')

flags.DEFINE_integer('train_epochs', 500, 'Last epoch index to train (1..train_epochs), or total target when resuming.')
flags.DEFINE_string(
    'resume_pkl',
    '',
    'If set, load agent weights from this params_*.pkl before training.',
)
flags.DEFINE_integer(
    'resume_start_epoch',
    -1,
    'When resuming: last fully completed epoch in that checkpoint (next train epoch is +1). Required if resume_pkl is set.',
)
flags.DEFINE_integer(
    'log_every_n_epochs',
    10,
    'Log train/val metrics every this many epochs.',
)
flags.DEFINE_integer(
    'save_every_n_epochs',
    100,
    'Save checkpoint every this many epochs.',
)

flags.DEFINE_boolean('use_wandb', False, 'Log to Weights & Biases.')
flags.DEFINE_boolean('use_tqdm', False, 'Show tqdm progress bar over epochs.')

config_flags.DEFINE_config_file('agent', 'agents/goub_phase1.py', lock_config=False)


def _steps_per_epoch(dataset_size: int, batch_size: int) -> int:
    return max(1, math.ceil(dataset_size / batch_size))


def _load_yaml(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _argv_sets_flag(flag_name: str) -> bool:
    """True if ``sys.argv`` explicitly sets this absl flag (YAML must not override)."""
    dashed = flag_name.replace('_', '-')
    for arg in sys.argv[1:]:
        if arg.startswith(f'--{flag_name}=') or arg.startswith(f'--{dashed}='):
            return True
        if arg in (f'--{flag_name}', f'--{dashed}'):
            return True
    return False


def _apply_yaml_to_flags(data: dict) -> None:
    """Merge YAML dict into absl FLAGS and agent ConfigDict.

    Top-level keys already set on the command line are skipped so CLI wins
    over YAML.  (``agent:`` nested keys always come from YAML when loaded.)
    """
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


def _setup_file_logger(run_dir: str) -> logging.Logger:
    log_path = os.path.join(run_dir, 'run.log')
    logger = logging.getLogger('goub_phase1')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def main(_):
    impl = _impl_dir()

    # --- YAML ---
    cfg_path = FLAGS.run_config.strip() or _default_yaml_path()
    if os.path.isfile(cfg_path):
        _apply_yaml_to_flags(_load_yaml(cfg_path))
    elif FLAGS.run_config.strip():
        raise FileNotFoundError(f'run_config YAML not found: {cfg_path}')

    config = FLAGS.agent
    _ih = config.get('idm_hidden_dims', (512, 512, 512))
    if isinstance(_ih, str):
        _ih = parse_hidden_dims(_ih)
    else:
        _ih = tuple(int(x) for x in _ih)
    config['idm_hidden_dims'] = _ih

    resume_pkl = FLAGS.resume_pkl.strip()
    if resume_pkl:
        if FLAGS.resume_start_epoch < 0:
            raise ValueError('resume_pkl is set but resume_start_epoch is unset (<0). Set resume_start_epoch to the checkpoint epoch.')
        if FLAGS.train_epochs <= FLAGS.resume_start_epoch:
            raise ValueError(
                f'train_epochs ({FLAGS.train_epochs}) must be > resume_start_epoch ({FLAGS.resume_start_epoch}) when resuming.'
            )

    # --- run directory ---
    ts = time.strftime('%Y%m%d_%H%M%S')
    env_tok = _sanitize_token(FLAGS.env_name)
    run_folder = f'{ts}_seed{FLAGS.seed}_{env_tok}'
    runs_root = FLAGS.runs_root.strip() or os.path.join(impl, 'runs')
    run_dir = os.path.join(runs_root, run_folder)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    if os.path.isfile(cfg_path):
        shutil.copy2(cfg_path, os.path.join(run_dir, 'config_used.yaml'))

    exp_name = get_exp_name(FLAGS.seed, env_name=FLAGS.env_name, agent_name=config['agent_name'])

    if FLAGS.use_wandb:
        setup_wandb(project='OGBench-GOUB', group=FLAGS.run_group, name=exp_name)
        project = wandb.run.project
    else:
        project = 'OGBench-GOUB'

    with open(os.path.join(run_dir, 'flags.json'), 'w', encoding='utf-8') as f:
        json.dump(get_flag_dict(), f, indent=2)

    run_logger = _setup_file_logger(run_dir)
    run_logger.info('run_dir=%s', run_dir)
    run_logger.info('project=%s exp_name=%s', project, exp_name)
    if os.path.isfile(cfg_path):
        run_logger.info('loaded YAML: %s', os.path.abspath(cfg_path))
    if resume_pkl:
        run_logger.info(
            'resume from pkl=%s resume_start_epoch=%d train_epochs=%d (will run epochs %d..%d)',
            os.path.abspath(resume_pkl),
            FLAGS.resume_start_epoch,
            FLAGS.train_epochs,
            FLAGS.resume_start_epoch + 1,
            FLAGS.train_epochs,
        )

    env, train_ds_plain, val_ds_plain = make_env_and_datasets(
        FLAGS.env_name, frame_stack=config['frame_stack'],
    )

    train_dataset = HGCDataset(train_ds_plain, config)
    val_dataset = HGCDataset(val_ds_plain, config) if val_ds_plain is not None else None

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = train_dataset.sample(1)
    ex_act = np.asarray(example_batch['actions'], dtype=np.float32)
    agent = GOUBPhase1Agent.create(FLAGS.seed, example_batch['observations'], config, ex_actions=ex_act)

    if resume_pkl:
        if not os.path.isfile(resume_pkl):
            raise FileNotFoundError(f'resume_pkl not found: {resume_pkl}')
        with open(resume_pkl, 'rb') as f:
            load_dict = pickle.load(f)
        template_sd = flax.serialization.to_state_dict(agent)
        merged = merge_checkpoint_state_dict(template_sd, load_dict['agent'])
        agent = flax.serialization.from_state_dict(agent, merged)
        run_logger.info('loaded weights from %s', os.path.abspath(resume_pkl))

    batch_size = int(config['batch_size'])
    spe = _steps_per_epoch(train_dataset.size, batch_size)
    epoch_begin = 1
    epoch_end_inclusive = FLAGS.train_epochs
    if resume_pkl:
        epoch_begin = FLAGS.resume_start_epoch + 1
        epoch_end_inclusive = FLAGS.train_epochs
    run_logger.info(
        'dataset_size=%d batch_size=%d steps_per_epoch=%d epoch_range=%d..%d',
        train_dataset.size,
        batch_size,
        spe,
        epoch_begin,
        epoch_end_inclusive,
    )

    train_logger = CsvLogger(os.path.join(run_dir, 'train.csv'))
    first_time = time.time()
    last_log_time = time.time()

    epoch_iter = range(epoch_begin, epoch_end_inclusive + 1)
    if FLAGS.use_tqdm:
        epoch_iter = tqdm.tqdm(epoch_iter, smoothing=0.1, dynamic_ncols=True)

    last_saved_epoch = FLAGS.resume_start_epoch if resume_pkl else 0

    for epoch in epoch_iter:
        losses = []
        losses_goub = []
        losses_sub = []
        losses_idm = []
        last_info = None

        for _ in range(spe):
            batch = train_dataset.sample(batch_size)
            agent, update_info = agent.update(batch)
            last_info = update_info
            losses.append(float(update_info['phase1/loss']))
            losses_goub.append(float(update_info['phase1/loss_goub']))
            losses_sub.append(float(update_info['phase1/loss_subgoal']))
            losses_idm.append(float(update_info['phase1/loss_idm']))

        global_step = epoch * spe

        if epoch % FLAGS.log_every_n_epochs == 0 and last_info is not None:
            metrics = {f'training/{k}': float(v) for k, v in last_info.items()}
            metrics['training/phase1/loss_epoch_mean'] = float(np.mean(losses))
            metrics['training/phase1/loss_goub_epoch_mean'] = float(np.mean(losses_goub))
            metrics['training/phase1/loss_subgoal_epoch_mean'] = float(np.mean(losses_sub))
            metrics['training/phase1/loss_idm_epoch_mean'] = float(np.mean(losses_idm))
            metrics['training/epoch'] = float(epoch)
            metrics['training/steps_per_epoch'] = float(spe)
            metrics['training/global_step'] = float(global_step)

            if val_dataset is not None:
                val_batch = val_dataset.sample(batch_size)
                _, val_info = agent.total_loss(
                    val_batch,
                    grad_params=agent.network.params,
                    rng=agent.rng,
                )
                metrics.update({f'validation/{k}': float(v) for k, v in val_info.items()})

            metrics['time/epoch_wall_sec'] = time.time() - last_log_time
            metrics['time/total_time'] = time.time() - first_time
            last_log_time = time.time()

            if FLAGS.use_wandb:
                wandb.log(metrics, step=global_step)
            train_logger.log(metrics, step=global_step)

            run_logger.info(
                'epoch=%d global_step=%d '
                'loss_mean=%.6f loss_goub_mean=%.6f loss_sub_mean=%.6f loss_idm_mean=%.6f | last: '
                'loss=%.6f loss_goub=%.6f loss_sub=%.6f loss_idm=%.6f eps_norm=%.6f mu_true=%.6f mu_pred=%.6f xN-1_norm=%.6f',
                epoch,
                global_step,
                metrics['training/phase1/loss_epoch_mean'],
                metrics['training/phase1/loss_goub_epoch_mean'],
                metrics['training/phase1/loss_subgoal_epoch_mean'],
                metrics['training/phase1/loss_idm_epoch_mean'],
                metrics.get('training/phase1/loss', float('nan')),
                metrics.get('training/phase1/loss_goub', float('nan')),
                metrics.get('training/phase1/loss_subgoal', float('nan')),
                metrics.get('training/phase1/loss_idm', float('nan')),
                metrics.get('training/phase1/eps_norm', float('nan')),
                metrics.get('training/phase1/mu_true_norm', float('nan')),
                metrics.get('training/phase1/mu_pred_norm', float('nan')),
                metrics.get('training/phase1/xN_minus_1_norm', float('nan')),
            )

        if epoch % FLAGS.save_every_n_epochs == 0:
            save_agent(agent, ckpt_dir, epoch)
            last_saved_epoch = epoch
            run_logger.info('saved checkpoint epoch=%d global_step=%d', epoch, global_step)

    final_step = epoch_end_inclusive * spe
    if last_saved_epoch != epoch_end_inclusive:
        save_agent(agent, ckpt_dir, epoch_end_inclusive)
        run_logger.info('saved final checkpoint epoch=%d global_step=%d', epoch_end_inclusive, final_step)

    train_logger.close()
    run_logger.info('training finished. run_dir=%s', run_dir)


if __name__ == '__main__':
    app.run(main)
