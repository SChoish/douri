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
_DEFAULT_JOINT_HORIZON = 25


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
flags.DEFINE_integer('batch_size', 256, 'Shared batch size for GOUB, critic, and actor.')
flags.DEFINE_integer(
    'joint_horizon', _DEFAULT_JOINT_HORIZON, 'Shared horizon for goub_N, subgoal_steps, and full_chunk_horizon.'
)
flags.DEFINE_integer('plan_candidates', 1, 'Number of GOUB candidate plans scored by the critic.')
flags.DEFINE_integer('proposal_topk', 1, 'How many critic-ranked GOUB proposals to pass to the actor.')
flags.DEFINE_float('plan_noise_scale', 1.0, 'Noise scale used for stochastic GOUB plan sampling.')
flags.DEFINE_boolean(
    'stochastic_plan_candidates',
    False,
    'Whether extra GOUB proposal candidates use stochastic sampling. When plan_candidates=1, mean ODE '
    '(sample_plan with noise_scale=0) is always used.',
)
flags.DEFINE_boolean('measure_timing', False, 'Whether to measure and log per-phase wall-clock timings.')
flags.DEFINE_integer('eval_freq', 0, 'Run validation loss and env evaluation every N epochs; <= 0 disables.')
flags.DEFINE_string('eval_task_ids', '1,2,3,4,5', 'Comma-separated OGBench task ids for env evaluation.')
flags.DEFINE_integer('eval_episodes_per_task', 5, 'Number of env evaluation episodes to run for each task id.')
flags.DEFINE_integer('eval_max_chunks', 50, 'Maximum action chunks to execute per evaluation episode.')
flags.DEFINE_float('eval_goal_tol', 0.5, 'Goal tolerance for marking env evaluation success.')
flags.DEFINE_string('eval_goal_dims', '0,1', 'Comma-separated observation dims used for env goal distance.')

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


def _accumulate_metric_sums(metric_sums: dict[str, float], info: dict | None) -> None:
    if info is None:
        return
    for key, value in info.items():
        metric_sums[key] = metric_sums.get(key, 0.0) + float(value)


def _emit_metric_means(metrics: dict[str, float], prefix: str, metric_sums: dict[str, float], count: int) -> None:
    if count < 1:
        return
    for key, total in metric_sums.items():
        metrics[f'{prefix}/{key}_epoch_mean'] = float(total / count)


def _accumulate_time_sums(time_sums: dict[str, float], values: dict[str, float] | None) -> None:
    if values is None:
        return
    for key, value in values.items():
        time_sums[key] = time_sums.get(key, 0.0) + float(value)


def _emit_time_sums(metrics: dict[str, float], prefix: str, time_sums: dict[str, float], count: int) -> None:
    if count < 1:
        return
    for key, total in time_sums.items():
        metrics[f'{prefix}/{key}_epoch_sec'] = float(total)
        metrics[f'{prefix}/{key}_step_sec'] = float(total / count)


def _format_epoch_log(metrics: dict[str, float]) -> str:
    parts = [
        f"goub={metrics['train/goub/loss_epoch_mean']:.6f}",
        f"critic={metrics['train/critic/loss_epoch_mean']:.6f}",
    ]
    if 'train/actor/loss_epoch_mean' in metrics:
        parts.append(f"actor={metrics['train/actor/loss_epoch_mean']:.6f}")
    else:
        parts.append('actor=disabled')
    parts.append(f"coupling={metrics['train/coupling/critic_score_epoch_mean']:.6f}")

    detail_keys = [
        ('goub_g', 'train/goub/phase1/loss_goub_epoch_mean'),
        ('goub_path', 'train/goub/phase1/loss_path_step_epoch_mean'),
        ('goub_roll', 'train/goub/phase1/loss_roll_epoch_mean'),
        ('goub_sub', 'train/goub/phase1/loss_subgoal_epoch_mean'),
        ('goub_idm', 'train/goub/phase1/loss_idm_epoch_mean'),
        ('critic_chunk', 'train/critic/chunk_critic/critic_loss_epoch_mean'),
        ('critic_distill', 'train/critic/action_critic/distill_loss_epoch_mean'),
        ('critic_value', 'train/critic/action_critic/value_loss_epoch_mean'),
        ('actor_q', 'train/actor/spi_actor/q_mean_epoch_mean'),
        ('actor_prox', 'train/actor/spi_actor/prox_mean_epoch_mean'),
        ('actor_entropy', 'train/actor/spi_actor/rho_entropy_epoch_mean'),
        ('t_data', 'time/data_epoch_sec'),
        ('t_build', 'time/build_batches_epoch_sec'),
        ('t_sg', 'time/build/predict_subgoal_epoch_sec'),
        ('t_mean', 'time/build/mean_ode_epoch_sec'),
        ('t_plan', 'time/build/plan_det_epoch_sec'),
        ('t_sample', 'time/build/sample_plan_epoch_sec'),
        ('t_idm', 'time/build/idm_epoch_sec'),
        ('t_score', 'time/build/score_epoch_sec'),
        ('t_goub', 'time/goub_update_epoch_sec'),
        ('t_critic', 'time/critic_update_epoch_sec'),
        ('t_actor_rescore', 'time/actor_rescore_epoch_sec'),
        ('t_actor', 'time/actor_update_epoch_sec'),
        ('t_epoch', 'time/epoch_compute_sec'),
    ]
    for label, key in detail_keys:
        if key in metrics:
            parts.append(f'{label}={metrics[key]:.6f}')
    return ' '.join(parts)


def _parse_int_list(text: str) -> tuple[int, ...]:
    items = [item.strip() for item in str(text).split(',') if item.strip()]
    return tuple(int(item) for item in items)


def _goal_distance(s: np.ndarray, g: np.ndarray, dims: tuple[int, ...] | None) -> float:
    if dims:
        idx = np.asarray(dims, dtype=np.int32)
        return float(np.linalg.norm(s[idx] - g[idx]))
    return float(np.linalg.norm(s - g))


def _goal_within_tol(s: np.ndarray, g: np.ndarray, dims: tuple[int, ...] | None, tol: float) -> bool:
    if tol <= 0.0:
        return False
    return _goal_distance(s, g, dims) <= float(tol)


def _apply_joint_horizon(goub_config: Any, critic_config: Any) -> tuple[Any, Any]:
    joint_horizon = int(FLAGS.joint_horizon)
    if joint_horizon < 1:
        raise ValueError(f'joint_horizon must be >= 1, got {joint_horizon}.')
    goub_config['goub_N'] = joint_horizon
    goub_config['subgoal_steps'] = joint_horizon
    critic_config['full_chunk_horizon'] = joint_horizon
    return goub_config, critic_config


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


def _eval_batch_size(common_valid_starts: np.ndarray, batch_size: int) -> int:
    return max(1, min(int(batch_size), int(len(common_valid_starts))))


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
) -> tuple[GOUBDynamicsAgent, dict, dict, dict[str, float]]:
    obs = np.asarray(goub_batch['observations'], dtype=np.float32)
    high_goals = np.asarray(goub_batch['high_actor_goals'], dtype=np.float32)
    measure_timing = bool(FLAGS.measure_timing)
    timing = {}

    if measure_timing:
        t0 = time.perf_counter()
        predicted_subgoals = np.asarray(goub_agent.predict_subgoal(obs, high_goals), dtype=np.float32)
        timing['predict_subgoal'] = time.perf_counter() - t0
    else:
        predicted_subgoals = np.asarray(goub_agent.predict_subgoal(obs, high_goals), dtype=np.float32)

    plan_candidates = max(1, int(FLAGS.plan_candidates))
    proposal_horizon = int(actor_config['actor_chunk_horizon'])
    plan_rng = goub_agent.rng
    if measure_timing:
        timing['mean_ode'] = 0.0
        timing['plan_det'] = 0.0
    sample_plan_time = 0.0

    if plan_candidates == 1:
        if measure_timing:
            t0 = time.perf_counter()
            sampled = goub_agent.sample_plan(
                obs,
                predicted_subgoals,
                plan_rng,
                noise_scale=0.0,
            )
            timing['mean_ode'] = time.perf_counter() - t0
        else:
            sampled = goub_agent.sample_plan(
                obs,
                predicted_subgoals,
                plan_rng,
                noise_scale=0.0,
            )
        plan_rng, _ = jax.random.split(plan_rng)
        trajectories = [np.asarray(sampled['trajectory'], dtype=np.float32)]
    else:
        if measure_timing:
            t0 = time.perf_counter()
            det_plan = np.asarray(goub_agent.plan(obs, predicted_subgoals)['trajectory'], dtype=np.float32)
            timing['plan_det'] = time.perf_counter() - t0
        else:
            det_plan = np.asarray(goub_agent.plan(obs, predicted_subgoals)['trajectory'], dtype=np.float32)
        trajectories = [det_plan]

        sample_noise_scale = float(FLAGS.plan_noise_scale) if bool(FLAGS.stochastic_plan_candidates) else 0.0
        for _ in range(plan_candidates - 1):
            plan_rng, sample_rng = jax.random.split(plan_rng)
            if measure_timing:
                t0 = time.perf_counter()
                sampled = goub_agent.sample_plan(
                    obs,
                    predicted_subgoals,
                    sample_rng,
                    noise_scale=sample_noise_scale,
                )
                sample_plan_time += time.perf_counter() - t0
            else:
                sampled = goub_agent.sample_plan(
                    obs,
                    predicted_subgoals,
                    sample_rng,
                    noise_scale=sample_noise_scale,
                )
            trajectories.append(np.asarray(sampled['trajectory'], dtype=np.float32))
    if measure_timing:
        timing['sample_plan'] = sample_plan_time
    goub_agent = goub_agent.replace(rng=plan_rng)

    candidate_trajectories = np.stack(trajectories, axis=1)  # [B, N, T, D]
    flat_trajectories = candidate_trajectories.reshape(-1, candidate_trajectories.shape[2], candidate_trajectories.shape[3])
    if measure_timing:
        t0 = time.perf_counter()
        candidate_actions = _idm_actions_from_trajectories(goub_agent, flat_trajectories, proposal_horizon)
        timing['idm'] = time.perf_counter() - t0
    else:
        candidate_actions = _idm_actions_from_trajectories(goub_agent, flat_trajectories, proposal_horizon)
    candidate_actions = candidate_actions.reshape(
        candidate_trajectories.shape[0],
        candidate_trajectories.shape[1],
        proposal_horizon,
        -1,
    )  # [B, N, ha, A]
    if candidate_actions.shape[1] == 1:
        candidate_scores = np.zeros((candidate_actions.shape[0], 1), dtype=np.float32)
        if measure_timing:
            timing['score'] = 0.0
    else:
        if measure_timing:
            t0 = time.perf_counter()
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
            timing['score'] = time.perf_counter() - t0
        else:
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
    return goub_agent, actor_batch, coupling_info, timing


def _rescore_actor_batch_for_update(actor_batch: dict, critic_agent: Any, actor_config: Any) -> dict:
    obs = np.asarray(actor_batch['observations'], dtype=np.float32)
    goals = np.asarray(actor_batch['spi_goals'], dtype=np.float32)
    candidates = np.asarray(actor_batch['candidate_partial_chunks'], dtype=np.float32)  # [B, N, ha, A]
    if candidates.shape[1] == 1:
        return {
            'observations': obs,
            'spi_goals': goals,
            'proposal_partial_chunks': candidates,
            'proposal_scores': np.zeros((obs.shape[0], 1), dtype=np.float32),
            'valids': np.asarray(actor_batch['valids'], dtype=np.float32),
        }
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
) -> tuple[GOUBDynamicsAgent, dict, dict | None, dict, dict[str, float]]:
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
        return goub_agent, critic_batch, None, coupling_info, {}
    goub_agent, actor_batch, coupling_info, build_timing = _build_actor_batch_from_goub(
        goub_agent, critic_agent, goub_batch, actor_config
    )
    return goub_agent, critic_batch, actor_batch, coupling_info, build_timing


def _build_joint_batches_dqc(
    goub_agent: GOUBDynamicsAgent,
    critic_agent: Any,
    goub_batch: dict,
    critic_batch: dict,
    actor_config: Any,
) -> tuple[GOUBDynamicsAgent, dict, dict | None, dict, dict[str, float]]:
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
        return goub_agent, critic_batch, None, coupling_info, {}
    goub_agent, actor_batch, coupling_info, build_timing = _build_actor_batch_from_goub(
        goub_agent, critic_agent, goub_batch, actor_config
    )
    return goub_agent, critic_batch, actor_batch, coupling_info, build_timing


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
    goub_config, critic_config = _apply_joint_horizon(goub_config, critic_config)
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
    shared_batch = int(FLAGS.batch_size)
    if shared_batch < 1:
        raise ValueError(f'batch_size must be >= 1, got {shared_batch}.')
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


def _evaluate_env_tasks(
    env,
    goub_agent: GOUBDynamicsAgent,
    actor_agent: Any,
    actor_config: Any,
    *,
    task_ids: tuple[int, ...],
    episodes_per_task: int,
    max_chunks: int,
    goal_tol: float,
    goal_dims: tuple[int, ...] | None,
) -> dict[str, float]:
    if actor_agent is None or not task_ids:
        return {}

    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    horizon = int(actor_config['actor_chunk_horizon'])
    task_successes = []
    metrics = {}

    for task_id in task_ids:
        episode_successes = []
        for _ in range(max(1, int(episodes_per_task))):
            ob, info = env.reset(options=dict(task_id=int(task_id), render_goal=False))
            if 'goal' not in info:
                raise RuntimeError(f'Env reset(task_id={task_id}) did not provide info["goal"].')
            obs = np.asarray(ob, dtype=np.float32).reshape(-1)
            goal = np.asarray(info['goal'], dtype=np.float32).reshape(-1)
            success = _goal_within_tol(obs, goal, goal_dims, goal_tol)

            for _ in range(max(1, int(max_chunks))):
                if success:
                    break
                predicted_subgoal = np.asarray(goub_agent.predict_subgoal(obs, goal), dtype=np.float32).reshape(-1)
                action_chunk = np.asarray(actor_agent.sample_actions(obs, predicted_subgoal), dtype=np.float32).reshape(horizon, -1)
                for action in action_chunk:
                    clipped = np.clip(action, low, high)
                    ob, reward, terminated, truncated, info = env.step(clipped)
                    obs = np.asarray(ob, dtype=np.float32).reshape(-1)
                    success_flag = bool(info.get('success', False)) if isinstance(info, dict) else False
                    success = success or success_flag or _goal_within_tol(obs, goal, goal_dims, goal_tol)
                    if success or terminated or truncated:
                        break
                if success or terminated or truncated:
                    break

            episode_successes.append(1.0 if success else 0.0)

        task_success_rate = float(np.mean(episode_successes))
        metrics[f'eval/task_{task_id}/success_rate'] = task_success_rate
        task_successes.append(task_success_rate)

    metrics['eval/success_rate_mean'] = float(np.mean(task_successes))
    metrics['eval/num_tasks'] = float(len(task_ids))
    metrics['eval/episodes_per_task'] = float(max(1, int(episodes_per_task)))
    return metrics


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

    train_logger = CsvLogger(os.path.join(run_dir, 'train.csv'), flush_every_n=1)
    first_time = time.time()
    last_log = time.time()
    measure_timing = bool(FLAGS.measure_timing)
    eval_freq = int(FLAGS.eval_freq)
    eval_task_ids = _parse_int_list(FLAGS.eval_task_ids)
    eval_episodes_per_task = max(1, int(FLAGS.eval_episodes_per_task))
    eval_goal_dims = _parse_int_list(FLAGS.eval_goal_dims)
    eval_goal_dims = eval_goal_dims if len(eval_goal_dims) > 0 else None
    eval_goal_tol = float(FLAGS.eval_goal_tol)
    eval_max_chunks = max(1, int(FLAGS.eval_max_chunks))

    epoch_iter = range(1, FLAGS.train_epochs + 1)
    if FLAGS.use_tqdm:
        epoch_iter = tqdm.tqdm(epoch_iter, smoothing=0.1, dynamic_ncols=True)

    for epoch in epoch_iter:
        if measure_timing:
            epoch_start = time.perf_counter()
        goub_losses = []
        critic_losses = []
        value_losses = []
        actor_losses = []
        actor_enabled = bool(actor_agent is not None)
        critic_scores = []
        data_time = 0.0
        build_time = 0.0
        build_detail_times = {}
        goub_time = 0.0
        critic_time = 0.0
        actor_rescore_time = 0.0
        actor_time = 0.0
        goub_metric_sums = {}
        critic_metric_sums = {}
        actor_metric_sums = {}
        coupling_metric_sums = {}
        last_goub_info = None
        last_critic_info = None
        last_actor_info = None
        last_coupling_info = None

        for _ in range(spe):
            if measure_timing:
                t0 = time.perf_counter()
            idxs = _sample_shared_idxs(common_valid_starts, batch_size)
            goub_batch = goub_dataset.sample(batch_size, idxs=idxs)
            critic_batch = critic_dataset.sample(batch_size, idxs=idxs)
            if measure_timing:
                data_time += time.perf_counter() - t0

            if measure_timing:
                t0 = time.perf_counter()
            if critic_name == 'deas':
                goub_agent, critic_batch, actor_batch, coupling_info, build_detail_info = _build_joint_batches_deas(
                    goub_agent,
                    critic_agent,
                    goub_batch,
                    critic_batch,
                    actor_config,
                )
            else:
                goub_agent, critic_batch, actor_batch, coupling_info, build_detail_info = _build_joint_batches_dqc(
                    goub_agent,
                    critic_agent,
                    goub_batch,
                    critic_batch,
                    actor_config,
                )
            if measure_timing:
                build_time += time.perf_counter() - t0
                _accumulate_time_sums(build_detail_times, build_detail_info)

            if measure_timing:
                t0 = time.perf_counter()
            goub_agent, goub_info = goub_agent.update(goub_batch)
            if measure_timing:
                goub_time += time.perf_counter() - t0

            if measure_timing:
                t0 = time.perf_counter()
            critic_agent, critic_info = critic_agent.update(critic_batch)
            if measure_timing:
                critic_time += time.perf_counter() - t0

            if actor_agent is not None and actor_batch is not None:
                if measure_timing:
                    t0 = time.perf_counter()
                actor_batch_for_update = _rescore_actor_batch_for_update(actor_batch, critic_agent, actor_config)
                if measure_timing:
                    actor_rescore_time += time.perf_counter() - t0

                if measure_timing:
                    t0 = time.perf_counter()
                actor_agent, actor_info = actor_agent.update(actor_batch_for_update, critic_agent)
                if measure_timing:
                    actor_time += time.perf_counter() - t0
            else:
                actor_info = None

            last_goub_info = goub_info
            last_critic_info = critic_info
            last_actor_info = actor_info
            last_coupling_info = coupling_info

            _accumulate_metric_sums(goub_metric_sums, goub_info)
            _accumulate_metric_sums(critic_metric_sums, critic_info)
            _accumulate_metric_sums(actor_metric_sums, actor_info)
            _accumulate_metric_sums(coupling_metric_sums, coupling_info)

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
            _emit_metric_means(metrics, 'train/goub', goub_metric_sums, len(goub_losses))
            _emit_metric_means(metrics, 'train/critic', critic_metric_sums, len(critic_losses))
            _emit_metric_means(metrics, 'train/actor', actor_metric_sums, len(actor_losses))
            _emit_metric_means(metrics, 'train/coupling', coupling_metric_sums, len(critic_scores))
            metrics['train/epoch'] = float(epoch)
            if eval_freq > 0 and epoch % eval_freq == 0:
                metrics.update(
                    _evaluate_env_tasks(
                        env,
                        goub_agent,
                        actor_agent,
                        actor_config,
                        task_ids=eval_task_ids,
                        episodes_per_task=eval_episodes_per_task,
                        max_chunks=eval_max_chunks,
                        goal_tol=eval_goal_tol,
                        goal_dims=eval_goal_dims,
                    )
                )
            if measure_timing:
                metrics['time/data_epoch_sec'] = data_time
                metrics['time/build_batches_epoch_sec'] = build_time
                _emit_time_sums(metrics, 'time/build', build_detail_times, spe)
                metrics['time/goub_update_epoch_sec'] = goub_time
                metrics['time/critic_update_epoch_sec'] = critic_time
                metrics['time/actor_rescore_epoch_sec'] = actor_rescore_time
                metrics['time/actor_update_epoch_sec'] = actor_time
                metrics['time/epoch_compute_sec'] = time.perf_counter() - epoch_start
                metrics['time/data_step_sec'] = data_time / spe
                metrics['time/build_batches_step_sec'] = build_time / spe
                metrics['time/goub_update_step_sec'] = goub_time / spe
                metrics['time/critic_update_step_sec'] = critic_time / spe
                metrics['time/actor_rescore_step_sec'] = actor_rescore_time / spe
                metrics['time/actor_update_step_sec'] = actor_time / spe if actor_enabled else 0.0
            metrics['time/wall_sec'] = time.time() - last_log
            metrics['time/total_sec'] = time.time() - first_time
            last_log = time.time()
            if FLAGS.use_wandb:
                wandb.log(metrics, step=gstep)
            train_logger.log(metrics, step=gstep)
            run_logger.info(
                'epoch=%d %s',
                epoch,
                _format_epoch_log(metrics),
            )
            if eval_freq > 0 and epoch % eval_freq == 0:
                run_logger.info('=== EVAL START epoch=%d ===', epoch)
                for task_id in eval_task_ids:
                    task_key = f'eval/task_{task_id}/success_rate'
                    if task_key in metrics:
                        run_logger.info('task_%d success_rate=%.6f', task_id, metrics[task_key])
                run_logger.info(
                    'eval epoch=%d success_rate_mean=%.6f num_tasks=%d episodes_per_task=%d',
                    epoch,
                    metrics.get('eval/success_rate_mean', float('nan')),
                    int(metrics.get('eval/num_tasks', 0.0)),
                    int(metrics.get('eval/episodes_per_task', 0.0)),
                )
                run_logger.info('=== EVAL END epoch=%d ===', epoch)

        if epoch % FLAGS.save_every_n_epochs == 0:
            save_agent(goub_agent, goub_ckpt_dir, epoch)
            save_agent(critic_agent, critic_ckpt_dir, epoch)
            if actor_agent is not None:
                save_agent(actor_agent, actor_ckpt_dir, epoch)

    train_logger.close()
    run_logger.info('done run_dir=%s', run_dir)


if __name__ == '__main__':
    app.run(main)
