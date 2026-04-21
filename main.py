"""Joint offline training for GOUB + DQC critic + SPI actor."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import sys
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
import wandb
import yaml
from absl import app, flags

from agents.critic import (
    extract_actor_loss,
    extract_critic_primary_score,
    extract_critic_total_loss,
    extract_value_loss,
    get_config as get_critic_config,
    get_critic_class,
    normalize_critic_name,
    validate_joint_mode,
)
from agents.critic.actor import JointActorAgent, get_actor_config
from agents.goub_dynamics import GOUBDynamicsAgent, get_dynamics_config
from utils.datasets import Dataset, PathHGCDataset
from utils.deas_sequence_dataset import DEASActionSeqDataset
from utils.dqc_sequence_dataset import DQCActionSeqDataset
from utils.env_utils import make_env_and_datasets
from utils.flax_utils import save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb

FLAGS = flags.FLAGS


def _impl_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _default_yaml_path():
    return os.path.join(_impl_dir(), 'config', 'joint_train_antmaze.yaml')


def _sanitize_token(s: str) -> str:
    s = re.sub(r'[^\w.\-]+', '_', s)
    return s[:120] if len(s) > 120 else s


flags.DEFINE_string('run_config', '', 'YAML config; empty uses config/joint_train_antmaze.yaml.')
flags.DEFINE_string('runs_root', '', 'Run root; default <repo>/runs.')
flags.DEFINE_string('run_group', 'Debug', 'W&B group.')
flags.DEFINE_integer('seed', 0, 'Seed.')
flags.DEFINE_string('env_name', 'antmaze-medium-navigate-v0', 'OGBench env / dataset name.')
flags.DEFINE_integer('train_epochs', 10, 'Training epochs.')
flags.DEFINE_integer('log_every_n_epochs', 1, 'Log interval (epochs).')
flags.DEFINE_integer('save_every_n_epochs', 10, 'Checkpoint interval.')
flags.DEFINE_boolean('use_wandb', False, 'W&B.')
flags.DEFINE_boolean('use_tqdm', False, 'tqdm over epochs.')
flags.DEFINE_enum('critic', 'dqc', ['deas', 'dqc'], 'External critic stack.')
flags.DEFINE_integer('shared_batch_size', 0, 'Override all module batch sizes when > 0.')
flags.DEFINE_integer('plan_candidates', 8, 'Number of GOUB candidate plans scored by the critic.')
flags.DEFINE_integer('proposal_topk', 4, 'How many critic-ranked GOUB proposals to pass to the actor.')
flags.DEFINE_float('plan_noise_scale', 1.0, 'Noise scale used for stochastic GOUB plan sampling.')

_SPI_ACTOR_KEYS = {
    'use_spi_actor',
    'spi_tau',
    'spi_beta',
    'spi_num_samples',
    'spi_candidate_source',
    'spi_use_partial_critic',
    'spi_actor_hidden_dims',
    'spi_actor_layer_norm',
    'spi_eval_use_actor',
    'spi_dist_normalize_by_dim',
    'spi_warmstart_steps',
}

_DQC_ONLY_CONFIG_KEYS = {
    'use_chunk_critic',
    'distill_method',
    'kappa_d',
    'implicit_backup_type',
    'kappa_b',
}


def _steps_per_epoch(dataset_size: int, batch_size: int) -> int:
    return max(1, math.ceil(dataset_size / batch_size))


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


def _apply_yaml_to_flags(data: dict) -> tuple[dict, dict, dict]:
    goub_updates = data.pop('goub', None)
    critic_updates = data.pop('critic_agent', None)
    actor_updates = data.pop('actor', None)
    for name, updates in [('goub', goub_updates), ('critic_agent', critic_updates), ('actor', actor_updates)]:
        if updates is not None and not isinstance(updates, dict):
            raise ValueError(f'YAML key "{name}" must be a mapping.')

    for key, value in data.items():
        if not hasattr(FLAGS, key):
            raise ValueError(f'Unknown YAML top-level key: {key!r}')
        if _argv_sets_flag(key):
            continue
        setattr(FLAGS, key, value)
    return goub_updates or {}, critic_updates or {}, actor_updates or {}


def _setup_file_logger(run_dir: str) -> logging.Logger:
    log_path = os.path.join(run_dir, 'run.log')
    logger = logging.getLogger('joint')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    logger.propagate = False
    return logger


def _update_config(config: Any, updates: dict) -> Any:
    for key, value in updates.items():
        config[key] = value
    return config


def _require_matching_frame_stack(goub_config: Any, critic_config: Any) -> None:
    frame_stacks = {
        'goub': goub_config.get('frame_stack', None),
        'critic': critic_config.get('frame_stack', None),
    }
    if len({str(v) for v in frame_stacks.values()}) != 1:
        raise ValueError(f'Joint training requires matching frame_stack across modules, got {frame_stacks}.')


def _make_critic_dataset(train_plain: dict, critic_name: str, critic_config: Any):
    dataset = Dataset.create(**train_plain)
    if critic_name == 'deas':
        return DEASActionSeqDataset(dataset, critic_config)
    return DQCActionSeqDataset(dataset, critic_config)


def _intersect_valid_starts(goub_dataset: PathHGCDataset, critic_dataset: Any) -> np.ndarray:
    common = np.intersect1d(goub_dataset.path_valid_idxs, critic_dataset.valid_starts, assume_unique=False)
    common = np.asarray(common, dtype=np.int64)
    if len(common) == 0:
        raise ValueError('No shared valid starts across GOUB and critic datasets.')
    return common


def _sample_shared_idxs(common_valid_starts: np.ndarray, batch_size: int) -> np.ndarray:
    picked = np.random.randint(len(common_valid_starts), size=batch_size)
    return common_valid_starts[picked]


def _idm_actions_from_trajectories(goub_agent: GOUBDynamicsAgent, trajectories: np.ndarray, horizon: int) -> np.ndarray:
    if trajectories.shape[1] <= horizon:
        raise ValueError(
            f'GOUB trajectory length {trajectories.shape[1]} is too short for horizon={horizon}. '
            'Increase goub_N / subgoal_steps or reduce chunk horizons.'
        )
    prev_states = trajectories[:, :horizon, :]
    next_states = trajectories[:, 1 : horizon + 1, :]
    flat_prev = prev_states.reshape(-1, prev_states.shape[-1])
    flat_next = next_states.reshape(-1, next_states.shape[-1])
    pred = goub_agent.network.select('idm_net')(jnp.asarray(flat_prev), jnp.asarray(flat_next))
    return np.asarray(pred, dtype=np.float32).reshape(trajectories.shape[0], horizon, -1)


def _rank_candidate_actions(
    candidate_actions: np.ndarray,
    scores: np.ndarray,
    keep_topk: int,
) -> tuple[np.ndarray, np.ndarray]:
    keep_topk = max(1, min(int(keep_topk), candidate_actions.shape[1]))
    order = np.argsort(-scores, axis=1)[:, :keep_topk]
    gather_idx = order[:, :, None, None]
    gathered = np.take_along_axis(candidate_actions, gather_idx, axis=1)
    gathered_scores = np.take_along_axis(scores, order, axis=1)
    return np.asarray(gathered, dtype=np.float32), np.asarray(gathered_scores, dtype=np.float32)


def _build_actor_batch_from_goub(
    goub_agent: GOUBDynamicsAgent,
    critic_agent: Any,
    goub_batch: dict,
    actor_config: Any,
) -> tuple[GOUBDynamicsAgent, dict, dict]:
    obs = np.asarray(goub_batch['observations'], dtype=np.float32)
    high_goals = np.asarray(goub_batch['high_actor_goals'], dtype=np.float32)
    predicted_subgoals = np.asarray(goub_agent.predict_subgoal(obs, high_goals), dtype=np.float32)

    plan_candidates = max(1, int(FLAGS.plan_candidates))
    proposal_horizon = int(actor_config['actor_chunk_horizon'])

    det_plan = np.asarray(goub_agent.plan(obs, predicted_subgoals)['trajectory'], dtype=np.float32)
    trajectories = [det_plan]

    plan_rng = goub_agent.rng
    for _ in range(plan_candidates - 1):
        plan_rng, sample_rng = jax.random.split(plan_rng)
        sampled = goub_agent.sample_plan(
            obs,
            predicted_subgoals,
            sample_rng,
            noise_scale=float(FLAGS.plan_noise_scale),
        )
        trajectories.append(np.asarray(sampled['trajectory'], dtype=np.float32))
    goub_agent = goub_agent.replace(rng=plan_rng)

    candidate_trajectories = np.stack(trajectories, axis=1)  # [B, N, T, D]
    flat_trajectories = candidate_trajectories.reshape(-1, candidate_trajectories.shape[2], candidate_trajectories.shape[3])
    candidate_actions = _idm_actions_from_trajectories(goub_agent, flat_trajectories, proposal_horizon)
    candidate_actions = candidate_actions.reshape(
        candidate_trajectories.shape[0],
        candidate_trajectories.shape[1],
        proposal_horizon,
        -1,
    )  # [B, N, ha, A]
    candidate_scores = np.asarray(
        critic_agent.score_action_chunks(
            obs,
            predicted_subgoals,
            candidate_actions,
            network_params=critic_agent.network.params,
            use_partial_critic=bool(actor_config.get('spi_use_partial_critic', True)),
        ),
        dtype=np.float32,
    )
    actor_batch = {
        'observations': obs,
        'spi_goals': predicted_subgoals,
        # Candidate action chunks generated from GOUB proposals; rescored after critic update.
        # Shape: [B, N, ha, A]
        'candidate_partial_chunks': candidate_actions,
        'valids': np.ones((obs.shape[0], proposal_horizon), dtype=np.float32),
    }

    coupling_info = {
        'coupling/predicted_subgoal_norm': float(np.linalg.norm(predicted_subgoals, axis=-1).mean()),
        'coupling/critic_score_mean': float(candidate_scores.mean()),
        'coupling/critic_score_max': float(candidate_scores.max()),
        'coupling/critic_score_min': float(candidate_scores.min()),
        'coupling/proposal_count': float(candidate_actions.shape[1]),
    }
    return goub_agent, actor_batch, coupling_info


def _rescore_actor_batch_for_update(actor_batch: dict, critic_agent: Any, actor_config: Any) -> dict:
    obs = np.asarray(actor_batch['observations'], dtype=np.float32)
    goals = np.asarray(actor_batch['spi_goals'], dtype=np.float32)
    candidates = np.asarray(actor_batch['candidate_partial_chunks'], dtype=np.float32)  # [B, N, ha, A]
    rescored = np.asarray(
        critic_agent.score_action_chunks(
            obs,
            goals,
            candidates,
            network_params=critic_agent.network.params,
            use_partial_critic=bool(actor_config.get('spi_use_partial_critic', True)),
        ),
        dtype=np.float32,
    )
    proposal_chunks, proposal_scores = _rank_candidate_actions(candidates, rescored, keep_topk=int(FLAGS.proposal_topk))
    return {
        'observations': obs,
        'spi_goals': goals,
        # Shape: [B, K, ha, A]
        'proposal_partial_chunks': proposal_chunks,
        # Shape: [B, K]
        'proposal_scores': proposal_scores,
        'valids': np.asarray(actor_batch['valids'], dtype=np.float32),
    }


def _build_joint_batches_deas(
    goub_agent: GOUBDynamicsAgent,
    critic_agent: Any,
    goub_batch: dict,
    critic_batch: dict,
    actor_config: Any,
) -> tuple[GOUBDynamicsAgent, dict, dict | None, dict]:
    if not bool(actor_config.get('use_spi_actor', False)):
        obs = np.asarray(goub_batch['observations'], dtype=np.float32)
        high_goals = np.asarray(goub_batch['high_actor_goals'], dtype=np.float32)
        predicted_subgoals = np.asarray(goub_agent.predict_subgoal(obs, high_goals), dtype=np.float32)
        coupling_info = {
            'coupling/predicted_subgoal_norm': float(np.linalg.norm(predicted_subgoals, axis=-1).mean()),
            'coupling/critic_score_mean': float('nan'),
            'coupling/critic_score_max': float('nan'),
            'coupling/critic_score_min': float('nan'),
            'coupling/proposal_count': 0.0,
        }
        return goub_agent, critic_batch, None, coupling_info
    goub_agent, actor_batch, coupling_info = _build_actor_batch_from_goub(goub_agent, critic_agent, goub_batch, actor_config)
    return goub_agent, critic_batch, actor_batch, coupling_info


def _build_joint_batches_dqc(
    goub_agent: GOUBDynamicsAgent,
    critic_agent: Any,
    goub_batch: dict,
    critic_batch: dict,
    actor_config: Any,
) -> tuple[GOUBDynamicsAgent, dict, dict | None, dict]:
    if not bool(actor_config.get('use_spi_actor', False)):
        obs = np.asarray(goub_batch['observations'], dtype=np.float32)
        high_goals = np.asarray(goub_batch['high_actor_goals'], dtype=np.float32)
        predicted_subgoals = np.asarray(goub_agent.predict_subgoal(obs, high_goals), dtype=np.float32)
        coupling_info = {
            'coupling/predicted_subgoal_norm': float(np.linalg.norm(predicted_subgoals, axis=-1).mean()),
            'coupling/critic_score_mean': float('nan'),
            'coupling/critic_score_max': float('nan'),
            'coupling/critic_score_min': float('nan'),
            'coupling/proposal_count': 0.0,
        }
        return goub_agent, critic_batch, None, coupling_info
    goub_agent, actor_batch, coupling_info = _build_actor_batch_from_goub(goub_agent, critic_agent, goub_batch, actor_config)
    return goub_agent, critic_batch, actor_batch, coupling_info


def _merge_actor_updates(actor_config: Any, actor_updates: dict) -> Any:
    ignored = sorted(k for k in actor_updates.keys() if k not in _SPI_ACTOR_KEYS)
    if ignored:
        logging.warning('Ignoring deprecated non-SPI actor keys in joint mode: %s', ', '.join(ignored))
    for key in _SPI_ACTOR_KEYS:
        if key in actor_updates:
            actor_config[key] = actor_updates[key]
    return actor_config


def _warn_ignored_deas_fields(critic_updates: dict, actor_updates: dict) -> None:
    ignored = sorted(set(critic_updates.keys()) & _DQC_ONLY_CONFIG_KEYS)
    if ignored:
        logging.warning('Ignoring DQC-only settings for critic=deas: %s', ', '.join(ignored))


def _prepare_joint_configs(goub_updates: dict, critic_updates: dict, actor_updates: dict):
    goub_config = _update_config(get_dynamics_config(), goub_updates)
    critic_config = _update_config(get_critic_config(), critic_updates)
    actor_config = _merge_actor_updates(get_actor_config(), actor_updates)
    critic_name = normalize_critic_name(FLAGS.critic)
    critic_config['critic'] = critic_name
    actor_config['critic'] = critic_name
    deas_spi_requested = False
    if critic_name == 'dqc':
        actor_config['actor_chunk_horizon'] = (
            int(critic_config['action_chunk_horizon'])
            if bool(actor_config.get('spi_use_partial_critic', True))
            else int(critic_config['full_chunk_horizon'])
        )
    else:
        deas_spi_requested = bool(actor_config.get('use_spi_actor', False))
        if deas_spi_requested:
            logging.warning('DEAS joint mode ignores SPI actor request and runs critic-only.')
        actor_config['use_spi_actor'] = False
        actor_config['spi_use_partial_critic'] = False
        actor_config['actor_chunk_horizon'] = int(critic_config['full_chunk_horizon'])
        _warn_ignored_deas_fields(critic_updates, actor_updates)
    validate_joint_mode(
        critic_name,
        critic_config,
        actor_config,
        plan_candidates=int(FLAGS.plan_candidates),
        proposal_topk=int(FLAGS.proposal_topk),
        deas_spi_requested=deas_spi_requested,
    )
    if FLAGS.shared_batch_size > 0:
        shared_batch = int(FLAGS.shared_batch_size)
        goub_config['batch_size'] = shared_batch
        critic_config['batch_size'] = shared_batch
        actor_config['batch_size'] = shared_batch
    batch_sizes = {
        'goub': int(goub_config['batch_size']),
        'critic': int(critic_config['batch_size']),
        'actor': int(actor_config.get('batch_size', critic_config['batch_size'])),
    }
    if len(set(batch_sizes.values())) != 1:
        raise ValueError(f'Joint training requires matching batch_size across modules, got {batch_sizes}.')
    _require_matching_frame_stack(goub_config, critic_config)
    return critic_name, goub_config, critic_config, actor_config


def _create_critic_agent(critic_name: str, critic_cls, seed: int, ex: dict, critic_config):
    if critic_name == 'deas':
        return critic_cls.create(
            seed,
            ex['observations'],
            ex['actions'],
            critic_config,
            ex_goals=ex.get('value_goals'),
        )
    return critic_cls.create(
        seed,
        ex['observations'],
        ex['full_chunk_actions'],
        ex['action_chunk_actions'],
        critic_config,
        ex_goals=ex.get('value_goals'),
    )


def _create_actor_agent(seed: int, ex_goub: dict, actor_config):
    if not bool(actor_config.get('use_spi_actor', False)):
        return None
    return JointActorAgent.create(
        seed,
        ex_goub['observations'],
        actor_config,
        ex_goals=ex_goub.get('high_actor_targets'),
    )


def main(_):
    impl = _impl_dir()
    cfg_path = FLAGS.run_config.strip() or _default_yaml_path()
    goub_updates, critic_updates, actor_updates = {}, {}, {}
    if os.path.isfile(cfg_path):
        goub_updates, critic_updates, actor_updates = _apply_yaml_to_flags(_load_yaml(cfg_path))
    elif FLAGS.run_config.strip():
        raise FileNotFoundError(f'run_config YAML not found: {cfg_path}')

    critic_name = normalize_critic_name(FLAGS.critic)
    critic_cls = get_critic_class(critic_name)
    critic_name, goub_config, critic_config, actor_config = _prepare_joint_configs(
        goub_updates,
        critic_updates,
        actor_updates,
    )

    ts = time.strftime('%Y%m%d_%H%M%S')
    env_tok = _sanitize_token(FLAGS.env_name)
    run_folder = f'{ts}_joint_{critic_name}_seed{FLAGS.seed}_{env_tok}'
    runs_root = FLAGS.runs_root.strip() or os.path.join(impl, 'runs')
    run_dir = os.path.join(runs_root, run_folder)
    ckpt_root = os.path.join(run_dir, 'checkpoints')
    goub_ckpt_dir = os.path.join(ckpt_root, 'goub')
    critic_ckpt_dir = os.path.join(ckpt_root, 'critic')
    actor_ckpt_dir = os.path.join(ckpt_root, 'actor')
    os.makedirs(goub_ckpt_dir, exist_ok=True)
    os.makedirs(critic_ckpt_dir, exist_ok=True)
    if bool(actor_config.get('use_spi_actor', False)):
        os.makedirs(actor_ckpt_dir, exist_ok=True)
    if os.path.isfile(cfg_path):
        shutil.copy2(cfg_path, os.path.join(run_dir, 'config_used.yaml'))

    exp_name = get_exp_name(FLAGS.seed, env_name=FLAGS.env_name, agent_name=f'joint_{critic_name}')
    if FLAGS.use_wandb:
        setup_wandb(project='OGBench-Joint', group=FLAGS.run_group, name=exp_name)

    with open(os.path.join(run_dir, 'flags.json'), 'w', encoding='utf-8') as f:
        json.dump(
            dict(
                flags=get_flag_dict(),
                goub=goub_config.to_dict(),
                critic_agent=critic_config.to_dict(),
                actor=actor_config.to_dict(),
            ),
            f,
            indent=2,
        )

    run_logger = _setup_file_logger(run_dir)
    run_logger.info('run_dir=%s critic=%s', run_dir, critic_name)

    env, train_plain, _ = make_env_and_datasets(FLAGS.env_name, frame_stack=critic_config['frame_stack'])
    action_dim = int(np.asarray(env.action_space.shape).prod())
    critic_config['action_dim'] = action_dim
    actor_config['action_dim'] = action_dim

    goub_dataset = PathHGCDataset(Dataset.create(**train_plain), goub_config)
    critic_dataset = _make_critic_dataset(train_plain, critic_name, critic_config)
    common_valid_starts = _intersect_valid_starts(goub_dataset, critic_dataset)

    if bool(actor_config.get('use_spi_actor', False)) and int(goub_config['goub_N']) < int(actor_config['actor_chunk_horizon']):
        raise ValueError(
            f'goub_N={int(goub_config["goub_N"])} must be >= actor_chunk_horizon={int(actor_config["actor_chunk_horizon"])} '
            'for critic-ranked GOUB proposals.'
        )

    np.random.seed(FLAGS.seed)
    ex_idxs = _sample_shared_idxs(common_valid_starts, int(goub_config['batch_size']))
    ex_goub = goub_dataset.sample(len(ex_idxs), idxs=ex_idxs)
    ex_critic = critic_dataset.sample(len(ex_idxs), idxs=ex_idxs)

    goub_agent = GOUBDynamicsAgent.create(
        FLAGS.seed,
        ex_goub['observations'],
        goub_config,
        ex_actions=ex_goub['actions'],
    )
    critic_agent = _create_critic_agent(critic_name, critic_cls, FLAGS.seed, ex_critic, critic_config)
    actor_agent = _create_actor_agent(FLAGS.seed, ex_goub, actor_config)

    batch_size = int(goub_config['batch_size'])
    spe = _steps_per_epoch(len(common_valid_starts), batch_size)
    run_logger.info(
        'shared_valid_starts=%d batch_size=%d steps_per_epoch=%d goub_h=%d critic_h=%d actor_h=%d actor_enabled=%s',
        len(common_valid_starts),
        batch_size,
        spe,
        int(goub_config['subgoal_steps']),
        int(critic_config.get('full_chunk_horizon', 0)),
        int(actor_config.get('actor_chunk_horizon', 0)),
        bool(actor_config.get('use_spi_actor', False)),
    )

    train_logger = CsvLogger(os.path.join(run_dir, 'train.csv'))
    first_time = time.time()
    last_log = time.time()

    epoch_iter = range(1, FLAGS.train_epochs + 1)
    if FLAGS.use_tqdm:
        epoch_iter = tqdm.tqdm(epoch_iter, smoothing=0.1, dynamic_ncols=True)

    for epoch in epoch_iter:
        goub_losses = []
        critic_losses = []
        value_losses = []
        actor_losses = []
        actor_enabled = bool(actor_agent is not None)
        critic_scores = []
        last_goub_info = None
        last_critic_info = None
        last_actor_info = None
        last_coupling_info = None

        for _ in range(spe):
            idxs = _sample_shared_idxs(common_valid_starts, batch_size)
            goub_batch = goub_dataset.sample(batch_size, idxs=idxs)
            critic_batch = critic_dataset.sample(batch_size, idxs=idxs)

            if critic_name == 'deas':
                goub_agent, critic_batch, actor_batch, coupling_info = _build_joint_batches_deas(
                    goub_agent,
                    critic_agent,
                    goub_batch,
                    critic_batch,
                    actor_config,
                )
            else:
                goub_agent, critic_batch, actor_batch, coupling_info = _build_joint_batches_dqc(
                    goub_agent,
                    critic_agent,
                    goub_batch,
                    critic_batch,
                    actor_config,
                )
            goub_agent, goub_info = goub_agent.update(goub_batch)
            critic_agent, critic_info = critic_agent.update(critic_batch)
            if actor_agent is not None and actor_batch is not None:
                actor_batch_for_update = _rescore_actor_batch_for_update(actor_batch, critic_agent, actor_config)
                actor_agent, actor_info = actor_agent.update(actor_batch_for_update, critic_agent)
            else:
                actor_info = None

            last_goub_info = goub_info
            last_critic_info = critic_info
            last_actor_info = actor_info
            last_coupling_info = coupling_info

            goub_losses.append(float(goub_info['phase1/loss']))
            critic_losses.append(extract_critic_total_loss(critic_name, critic_info))
            value_losses.append(extract_value_loss(critic_name, critic_info))
            if actor_info is not None:
                actor_losses.append(extract_actor_loss(critic_name, actor_info))
            critic_scores.append(float(coupling_info['coupling/critic_score_mean']))

        gstep = epoch * spe
        if epoch % FLAGS.log_every_n_epochs == 0 and last_goub_info is not None:
            metrics = {}
            metrics.update({f'train/goub/{k}': float(v) for k, v in last_goub_info.items()})
            metrics.update({f'train/critic/{k}': float(v) for k, v in last_critic_info.items()})
            if last_actor_info is not None:
                metrics.update({f'train/actor/{k}': float(v) for k, v in last_actor_info.items()})
            metrics.update({f'train/{k}': float(v) for k, v in last_coupling_info.items()})
            metrics['train/goub/loss_epoch_mean'] = float(np.mean(goub_losses))
            metrics['train/critic/loss_epoch_mean'] = float(np.mean(critic_losses))
            metrics['train/value/loss_epoch_mean'] = float(np.mean(value_losses))
            metrics['train/actor/enabled'] = 1.0 if actor_enabled else 0.0
            if actor_losses:
                metrics['train/actor/loss_epoch_mean'] = float(np.mean(actor_losses))
            metrics['train/coupling/critic_score_epoch_mean'] = float(np.mean(critic_scores))
            metrics['train/critic/primary_score'] = extract_critic_primary_score(critic_name, last_critic_info)
            metrics['train/epoch'] = float(epoch)
            metrics['time/wall_sec'] = time.time() - last_log
            metrics['time/total_sec'] = time.time() - first_time
            last_log = time.time()
            if FLAGS.use_wandb:
                wandb.log(metrics, step=gstep)
            train_logger.log(metrics, step=gstep)
            actor_loss_str = (
                f"{metrics['train/actor/loss_epoch_mean']:.6f}" if 'train/actor/loss_epoch_mean' in metrics else 'disabled'
            )
            run_logger.info(
                'epoch=%d goub=%.6f critic=%.6f actor=%s coupling_score=%.6f',
                epoch,
                metrics['train/goub/loss_epoch_mean'],
                metrics['train/critic/loss_epoch_mean'],
                actor_loss_str,
                metrics['train/coupling/critic_score_epoch_mean'],
            )

        if epoch % FLAGS.save_every_n_epochs == 0:
            save_agent(goub_agent, goub_ckpt_dir, epoch)
            save_agent(critic_agent, critic_ckpt_dir, epoch)
            if actor_agent is not None:
                save_agent(actor_agent, actor_ckpt_dir, epoch)

    train_logger.close()
    run_logger.info('done run_dir=%s', run_dir)


if __name__ == '__main__':
    app.run(main)
