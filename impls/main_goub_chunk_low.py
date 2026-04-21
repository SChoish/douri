import json
import os
import random
import time

import numpy as np
import tqdm
import wandb
from absl import app, flags
from ml_collections import config_flags

from agents.goub_chunk_low import GOUBChunkLowAgent
from utils.datasets import ChunkHGCDataset
from utils.env_utils import make_env_and_datasets
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'Debug', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Restore path.')
flags.DEFINE_integer('restore_epoch', None, 'Restore epoch.')

flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')

flags.DEFINE_boolean('use_wandb', True, 'Log metrics to Weights & Biases.')
flags.DEFINE_boolean('use_tqdm', False, 'Show tqdm progress bars during training.')

config_flags.DEFINE_config_file('agent', 'agents/goub_chunk_low.py', lock_config=False)


def main(_):
    config = FLAGS.agent

    if config['agent_name'] != 'goub_chunk_low':
        raise ValueError(f"This entrypoint is GOUB chunk-low only, but got agent_name={config['agent_name']!r}.")
    if config['dataset_class'] != 'ChunkHGCDataset':
        raise ValueError(
            f"GOUB chunk-low expects dataset_class='ChunkHGCDataset', but got {config['dataset_class']!r}."
        )

    exp_name = get_exp_name(
        FLAGS.seed,
        env_name=FLAGS.env_name,
        agent_name=config['agent_name'],
    )

    if FLAGS.use_wandb:
        setup_wandb(project='OGBench', group=FLAGS.run_group, name=exp_name)
        project = wandb.run.project
    else:
        project = 'OGBench'

    FLAGS.save_dir = os.path.join(FLAGS.save_dir, project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    flag_dict = get_flag_dict()
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w', encoding='utf-8') as f:
        json.dump(flag_dict, f)

    _env, train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name,
        frame_stack=config['frame_stack'],
    )
    train_dataset = ChunkHGCDataset(train_dataset, config)
    if val_dataset is not None:
        val_dataset = ChunkHGCDataset(val_dataset, config)

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    example_batch = train_dataset.sample(1)
    agent = GOUBChunkLowAgent.create(
        FLAGS.seed,
        example_batch['observations'],
        example_batch['action_chunks'],
        example_batch['local_plan_context'],
        config,
    )

    if FLAGS.restore_path is not None:
        agent = restore_agent(agent, FLAGS.restore_path, FLAGS.restore_epoch)

    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    first_time = time.time()
    last_time = time.time()
    train_iter = range(1, FLAGS.train_steps + 1)
    if FLAGS.use_tqdm:
        train_iter = tqdm.tqdm(train_iter, smoothing=0.1, dynamic_ncols=True)

    dataset_action_norm = float(np.linalg.norm(np.asarray(train_dataset.dataset['actions']), axis=-1).mean())

    for i in train_iter:
        batch = train_dataset.sample(config['batch_size'])
        agent, update_info = agent.update(batch)

        if i % FLAGS.log_interval == 0:
            train_metrics = {f'training/{k}': v for k, v in update_info.items()}
            train_metrics['training/data_action_norm_mean'] = dataset_action_norm
            if val_dataset is not None:
                val_batch = val_dataset.sample(config['batch_size'])
                _, val_info = agent.total_loss(val_batch, grad_params=None)
                train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
            train_metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            train_metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            if FLAGS.use_wandb:
                wandb.log(train_metrics, step=i)
            train_logger.log(train_metrics, step=i)

        if i % FLAGS.save_interval == 0:
            save_agent(agent, FLAGS.save_dir, i)

    if FLAGS.train_steps % FLAGS.save_interval != 0:
        save_agent(agent, FLAGS.save_dir, FLAGS.train_steps)

    train_logger.close()


if __name__ == '__main__':
    app.run(main)
