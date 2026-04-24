"""Joint offline training for GOUB + DQC critic + SPI actor."""

from __future__ import annotations

from functools import partial
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
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
    DQCCriticAgent,
    extract_critic_primary_score,
    get_config as get_critic_config,
    validate_joint_mode,
)
from agents.actor import JointActorAgent, get_actor_config
from agents.goub_dynamics import GOUBDynamicsAgent, get_dynamics_config
from utils.datasets import Dataset, PathHGCDataset
from utils.dqc_sequence_dataset import DQCActionSeqDataset
from utils.env_utils import make_env_and_datasets
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb
from utils.run_io import goal_distance, goal_within_tol, parse_int_list

FLAGS = flags.FLAGS
_DEFAULT_JOINT_HORIZON = 25


def _impl_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _default_yaml_path():
    return os.path.join(_impl_dir(), 'config', 'antmaze_large_navigate.yaml')


def _sanitize_token(s: str) -> str:
    s = re.sub(r'[^\w.\-]+', '_', s)
    return s[:120] if len(s) > 120 else s


def _block_until_ready(tree: Any) -> Any:
    """Synchronize a JAX pytree so wall-clock timing reflects real compute."""

    def _ready(x):
        return x.block_until_ready() if hasattr(x, 'block_until_ready') else x

    return jax.tree_util.tree_map(_ready, tree)


flags.DEFINE_string('run_config', '', 'YAML config; empty uses config/antmaze_large_navigate.yaml.')
flags.DEFINE_string('runs_root', '', 'Run root; default <repo>/runs.')
flags.DEFINE_string('resume_run_dir', '', 'Existing run dir to resume in-place.')
flags.DEFINE_integer('resume_epoch', 0, 'Checkpoint epoch to resume from; 0 disables resume.')
flags.DEFINE_boolean(
    'resume_use_run_snapshot_config',
    True,
    'When resuming: if --run_config is not set on argv, load hyperparameters from '
    'resume_run_dir/flags.json (preferred; written as a temp YAML) or else config_used.yaml, '
    'so checkpoints and hparams match the original run.',
)
flags.DEFINE_string('run_group', 'Debug', 'W&B group.')
flags.DEFINE_integer('seed', 0, 'Seed.')
flags.DEFINE_string('env_name', 'antmaze-medium-navigate-v0', 'OGBench env / dataset name.')
flags.DEFINE_integer('train_epochs', 10, 'Training epochs.')
flags.DEFINE_integer('log_every_n_epochs', 1, 'Log interval (epochs).')
flags.DEFINE_integer('save_every_n_epochs', 10, 'Checkpoint interval.')
flags.DEFINE_boolean('use_wandb', False, 'W&B.')
flags.DEFINE_boolean('use_tqdm', False, 'tqdm over epochs.')
flags.DEFINE_integer('batch_size', 256, 'Shared batch size for GOUB, critic, and actor.')
flags.DEFINE_integer(
    'joint_horizon', _DEFAULT_JOINT_HORIZON, 'Shared horizon for goub_N, subgoal_steps, and full_chunk_horizon.'
)
flags.DEFINE_integer('plan_candidates', 1, 'Number of GOUB candidate plans scored by the critic.')
flags.DEFINE_float('plan_noise_scale', 1.0, 'Noise scale used for stochastic GOUB plan sampling.')
flags.DEFINE_boolean('measure_timing', False, 'Whether to measure and log per-phase wall-clock timings.')
flags.DEFINE_integer('eval_freq', 0, 'Run validation loss and env evaluation every N epochs; <= 0 disables.')
flags.DEFINE_string('eval_task_ids', '1,2,3,4,5', 'Comma-separated OGBench task ids for env evaluation.')
flags.DEFINE_integer('eval_episodes_per_task', 10, 'Number of env evaluation episodes to run for each task id.')
flags.DEFINE_integer('eval_max_chunks', 50, 'Maximum action chunks to execute per evaluation episode.')
flags.DEFINE_float('eval_goal_tol', 0.5, 'Goal tolerance for marking env evaluation success.')
flags.DEFINE_string('eval_goal_dims', '0,1', 'Comma-separated observation dims used for env goal distance.')

_SPI_ACTOR_KEYS = {
    'spi_tau',
    'spi_beta',
    'spi_actor_layer_norm',
    'spi_q_norm_eps',
    'spi_conditioned',
}

def _steps_per_epoch(dataset_size: int, batch_size: int) -> int:
    return max(1, math.ceil(dataset_size / batch_size))


def _load_yaml(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _resolve_resume_snapshot_config_path(run_dir: str) -> str | None:
    """Prefer flags.json (full merged hparams at run start); else config_used.yaml; else None."""
    fj = os.path.join(run_dir, 'flags.json')
    if not os.path.isfile(fj):
        used = os.path.join(run_dir, 'config_used.yaml')
        return used if os.path.isfile(used) else None
    with open(fj, encoding='utf-8') as fp:
        snap = json.load(fp)
    fg = dict(snap.get('flags') or {})
    skip = {
        'resume_run_dir',
        'resume_epoch',
        'run_config',
        'runs_root',
        'help',
        'helpshort',
        'helpfull',
        'helpxml',
        '?',
    }
    root: dict[str, Any] = {}
    for key, value in fg.items():
        if key in skip:
            continue
        if hasattr(FLAGS, key):
            root[key] = value
    if snap.get('goub') is not None:
        root['goub'] = snap['goub']
    if snap.get('critic_agent') is not None:
        root['critic_agent'] = snap['critic_agent']
    if snap.get('actor') is not None:
        root['actor'] = snap['actor']
    fd, tmp_path = tempfile.mkstemp(prefix='resume_flags_', suffix='.yaml', text=True)
    os.close(fd)
    with open(tmp_path, 'w', encoding='utf-8') as out:
        yaml.safe_dump(root, out, sort_keys=False, default_flow_style=False)
    return tmp_path


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


def _setup_file_logger(run_dir: str, *, resume_epoch: int = 0) -> tuple[logging.Logger, str]:
    if resume_epoch > 0:
        ts = time.strftime('%Y%m%d_%H%M%S')
        log_name = f'run_resume_from{int(resume_epoch)}_{ts}.log'
    else:
        log_name = 'run.log'
    log_path = os.path.join(run_dir, log_name)
    logger = logging.getLogger('joint')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)
    logger.propagate = False
    return logger, log_path


def _update_config(config: Any, updates: dict) -> Any:
    for key, value in updates.items():
        config[key] = value
    return config


def _accumulate_metric_sums(metric_sums: dict, info: dict | None) -> None:
    """Accumulate scalars *on device*; ``float()`` is deferred to log-emit time.

    Avoids one host sync per metric per step (~30 syncs/step otherwise), which
    serialised the GPU pipeline against the Python loop.
    """
    if info is None:
        return
    for key, value in info.items():
        prev = metric_sums.get(key)
        metric_sums[key] = value if prev is None else prev + value


def _emit_metric_means(metrics: dict[str, float], prefix: str, metric_sums: dict, count: int) -> None:
    if count < 1 or not metric_sums:
        return
    inv = 1.0 / float(count)
    # Single device→host transfer for the whole tree avoids one sync per key.
    host_sums = jax.device_get(metric_sums)
    for key, total in host_sums.items():
        metrics[f'{prefix}/{key}_epoch_mean'] = float(total) * inv


def _to_host_metrics(prefix: str, info: dict | None) -> dict[str, float]:
    if not info:
        return {}
    host = jax.device_get(info)
    return {f'{prefix}/{k}': float(v) for k, v in host.items()}


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
        ('t_prop', 'time/build/proposal_build_epoch_sec'),
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


def _make_critic_dataset(train_plain: dict, critic_config: Any):
    dataset = Dataset.create(**train_plain)
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


@partial(jax.jit, static_argnames=('horizon',))
def _idm_actions_from_trajectories_jit(network: Any, trajectories: jnp.ndarray, horizon: int) -> jnp.ndarray:
    prev_states = trajectories[:, :horizon, :]
    next_states = trajectories[:, 1 : horizon + 1, :]
    flat_prev = prev_states.reshape(-1, prev_states.shape[-1])
    flat_next = next_states.reshape(-1, next_states.shape[-1])
    pred = network.select('idm_net')(flat_prev, flat_next)
    return jnp.asarray(pred, dtype=jnp.float32).reshape(trajectories.shape[0], horizon, -1)


def _idm_actions_from_trajectories(goub_agent: GOUBDynamicsAgent, trajectories: np.ndarray, horizon: int) -> jnp.ndarray:
    if trajectories.shape[1] <= horizon:
        raise ValueError(
            f'GOUB trajectory length {trajectories.shape[1]} is too short for horizon={horizon}. '
            'Increase goub_N / subgoal_steps or reduce chunk horizons.'
        )
    trajectories = jnp.asarray(trajectories, dtype=jnp.float32)
    return _idm_actions_from_trajectories_jit(goub_agent.network, trajectories, horizon)


def _rank_candidate_actions(
    candidate_actions: jnp.ndarray,
    scores: jnp.ndarray,
    keep_topk: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    keep_topk = max(1, min(int(keep_topk), candidate_actions.shape[1]))
    order = jnp.argsort(-scores, axis=1)[:, :keep_topk]
    gather_idx = order[:, :, None, None]
    gathered = jnp.take_along_axis(candidate_actions, gather_idx, axis=1)
    gathered_scores = jnp.take_along_axis(scores, order, axis=1)
    return jnp.asarray(gathered, dtype=jnp.float32), jnp.asarray(gathered_scores, dtype=jnp.float32)


@partial(jax.jit, static_argnames=('keep_topk', 'use_partial_critic'))
def _score_and_rank_candidate_actions(
    critic_agent: Any,
    obs: jnp.ndarray,
    goals: jnp.ndarray,
    candidates: jnp.ndarray,
    network_params: Any,
    *,
    keep_topk: int,
    use_partial_critic: bool,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    rescored = jnp.asarray(
        critic_agent.score_action_chunks(
            obs,
            goals,
            candidates,
            network_params=network_params,
            use_partial_critic=use_partial_critic,
        ),
        dtype=jnp.float32,
    )
    return _rank_candidate_actions(candidates, rescored, keep_topk=keep_topk)


def _build_actor_batch_from_goub(
    goub_agent: GOUBDynamicsAgent,
    critic_agent: Any,
    goub_batch: dict,
    actor_config: Any,
) -> tuple[GOUBDynamicsAgent, dict, dict, dict[str, float]]:
    obs = jnp.asarray(goub_batch['observations'], dtype=jnp.float32)
    high_goals = jnp.asarray(goub_batch['high_actor_goals'], dtype=jnp.float32)
    measure_timing = bool(FLAGS.measure_timing)
    timing = {}

    plan_candidates = max(1, int(FLAGS.plan_candidates))
    proposal_horizon = int(actor_config['actor_chunk_horizon'])
    if measure_timing:
        t0 = time.perf_counter()
    sample_noise_scale = float(FLAGS.plan_noise_scale) if plan_candidates > 1 else 0.0
    predicted_subgoals, candidate_actions, plan_rng = goub_agent.build_actor_proposals(
        obs,
        high_goals,
        goub_agent.rng,
        proposal_horizon=proposal_horizon,
        plan_candidates=plan_candidates,
        sample_noise_scale=sample_noise_scale,
    )
    if measure_timing:
        _block_until_ready((predicted_subgoals, candidate_actions, plan_rng))
        timing['proposal_build'] = time.perf_counter() - t0
    else:
        timing = {}
    goub_agent = goub_agent.replace(rng=plan_rng)
    # ``spi_goals`` is the conditioning vector both π(s, g) and Q(s, g, a) see in
    # ``JointActorAgent.actor_loss``. ``actor_config.spi_conditioned`` selects which:
    #   'subgoal' (default): GOUB ``predict_subgoal(obs, high_goals)`` — local subgoal.
    #   'goal'             : the global ``high_goals`` itself — π/Q targeted at final goal.
    # Candidate proposal chunks are unchanged in either mode (still planned to the
    # subgoal); only the conditioning vector flips, keeping π/Q consistent.
    spi_cond = str(actor_config.get('spi_conditioned', 'subgoal')).lower()
    if spi_cond == 'goal':
        spi_goals = high_goals
    elif spi_cond == 'subgoal':
        spi_goals = predicted_subgoals
    else:
        raise ValueError(
            f"actor.spi_conditioned must be 'subgoal' or 'goal', got {spi_cond!r}."
        )
    actor_batch = {
        'observations': obs,
        'spi_goals': spi_goals,
        # Candidate action chunks generated from GOUB proposals; rescored after critic update.
        # Shape: [B, N, ha, A]
        'candidate_partial_chunks': candidate_actions,
        'valids': jnp.ones((obs.shape[0], proposal_horizon), dtype=jnp.float32),
    }

    nan = jnp.full((), jnp.nan, dtype=jnp.float32)
    coupling_info = {
        'coupling/predicted_subgoal_norm': jnp.linalg.norm(predicted_subgoals, axis=-1).mean(),
        'coupling/critic_score_mean': nan,
        'coupling/critic_score_max': nan,
        'coupling/critic_score_min': nan,
        'coupling/proposal_count': jnp.asarray(float(candidate_actions.shape[1]), dtype=jnp.float32),
    }
    return goub_agent, actor_batch, coupling_info, timing


def _rescore_actor_batch_for_update(actor_batch: dict, critic_agent: Any, actor_config: Any) -> tuple[dict, dict]:
    obs = jnp.asarray(actor_batch['observations'], dtype=jnp.float32)
    goals = jnp.asarray(actor_batch['spi_goals'], dtype=jnp.float32)
    candidates = jnp.asarray(actor_batch['candidate_partial_chunks'], dtype=jnp.float32)  # [B, N, ha, A]
    if candidates.shape[1] == 1:
        zero = jnp.zeros((), dtype=jnp.float32)
        return (
            {
                'observations': obs,
                'spi_goals': goals,
                'proposal_partial_chunks': candidates,
                'proposal_scores': jnp.zeros((obs.shape[0], 1), dtype=jnp.float32),
                'valids': jnp.asarray(actor_batch['valids'], dtype=jnp.float32),
            },
            {
                'coupling/critic_score_mean': zero,
                'coupling/critic_score_max': zero,
                'coupling/critic_score_min': zero,
            },
        )
    proposal_chunks, proposal_scores = _score_and_rank_candidate_actions(
        critic_agent,
        obs,
        goals,
        candidates,
        critic_agent.network.params,
        keep_topk=int(candidates.shape[1]),
        use_partial_critic=True,
    )
    return (
        {
            'observations': obs,
            'spi_goals': goals,
            # Shape: [B, K, ha, A]
            'proposal_partial_chunks': proposal_chunks,
            # Shape: [B, K]
            'proposal_scores': proposal_scores,
            'valids': jnp.asarray(actor_batch['valids'], dtype=jnp.float32),
        },
        {
            'coupling/critic_score_mean': proposal_scores.mean(),
            'coupling/critic_score_max': proposal_scores.max(),
            'coupling/critic_score_min': proposal_scores.min(),
        },
    )


def _build_joint_batches_dqc(
    goub_agent: GOUBDynamicsAgent,
    critic_agent: Any,
    goub_batch: dict,
    critic_batch: dict,
    actor_config: Any,
) -> tuple[GOUBDynamicsAgent, dict, dict, dict, dict[str, float]]:
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


def _prepare_joint_configs(goub_updates: dict, critic_updates: dict, actor_updates: dict):
    goub_config = _update_config(get_dynamics_config(), goub_updates)
    critic_config = _update_config(get_critic_config(), critic_updates)
    actor_config = _merge_actor_updates(get_actor_config(), actor_updates)
    goub_config, critic_config = _apply_joint_horizon(goub_config, critic_config)
    actor_config['actor_chunk_horizon'] = int(critic_config['action_chunk_horizon'])
    # Subgoal-value bonus net shares parameters with the critic value net, so its
    # architecture must mirror the critic; force-sync here so users only configure it once.
    goub_config['subgoal_value_hidden_dims'] = tuple(int(x) for x in critic_config['value_hidden_dims'])
    goub_config['subgoal_value_layer_norm'] = bool(critic_config['layer_norm'])
    validate_joint_mode(critic_config, actor_config)
    shared_batch = int(FLAGS.batch_size)
    if shared_batch < 1:
        raise ValueError(f'batch_size must be >= 1, got {shared_batch}.')
    goub_config['batch_size'] = shared_batch
    critic_config['batch_size'] = shared_batch
    actor_config['batch_size'] = shared_batch
    _require_matching_frame_stack(goub_config, critic_config)
    return goub_config, critic_config, actor_config


def _create_critic_agent(seed: int, ex: dict, critic_config):
    return DQCCriticAgent.create(
        seed,
        ex['observations'],
        ex['full_chunk_actions'],
        ex['action_chunk_actions'],
        critic_config,
        ex_goals=ex.get('value_goals'),
    )


def _create_actor_agent(seed: int, ex_goub: dict, actor_config):
    return JointActorAgent.create(
        seed,
        ex_goub['observations'],
        actor_config,
        ex_goals=ex_goub.get('high_actor_targets'),
    )


def _extract_critic_value_params(critic_agent: Any) -> Any | None:
    if critic_agent is None:
        return None
    return critic_agent.network.params.get('modules_value', None)


def _execute_action_chunk(
    env,
    obs: np.ndarray,
    goal: np.ndarray,
    action_chunk: np.ndarray,
    *,
    low: np.ndarray,
    high: np.ndarray,
    goal_dims: tuple[int, ...] | None,
    goal_tol: float,
) -> tuple[np.ndarray, bool, bool, bool]:
    success = goal_within_tol(obs, goal, goal_dims, goal_tol)
    terminated = False
    truncated = False
    for action in np.asarray(action_chunk, dtype=np.float32):
        clipped = np.clip(action, low, high)
        ob, _reward, terminated, truncated, info = env.step(clipped)
        obs = np.asarray(ob, dtype=np.float32).reshape(-1)
        success_flag = bool(info.get('success', False)) if isinstance(info, dict) else False
        success = success or success_flag or goal_within_tol(obs, goal, goal_dims, goal_tol)
        if success or terminated or truncated:
            break
    return obs, success, bool(terminated), bool(truncated)


def _idm_action_chunk(
    goub_agent: GOUBDynamicsAgent,
    obs: np.ndarray,
    predicted_subgoal: np.ndarray,
    horizon: int,
) -> np.ndarray:
    traj = np.asarray(goub_agent.plan(obs, predicted_subgoal)['trajectory'], dtype=np.float32)
    if traj.ndim != 2:
        raise RuntimeError(f'Expected single-trajectory plan with rank 2, got shape={traj.shape}.')
    action_chunk = np.asarray(_idm_actions_from_trajectories(goub_agent, traj[None, ...], horizon), dtype=np.float32)
    return action_chunk[0]


def _evaluate_env_tasks(
    env,
    goub_agent: GOUBDynamicsAgent,
    actor_agent: Any,
    actor_config: Any,
    critic_config: Any,
    critic_value_params: Any | None = None,
    *,
    task_ids: tuple[int, ...],
    episodes_per_task: int,
    max_chunks: int,
    goal_tol: float,
    goal_dims: tuple[int, ...] | None,
) -> dict[str, float]:
    if not task_ids:
        return {}

    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    actor_horizon = int(actor_config['actor_chunk_horizon'])
    idm_horizon = int(critic_config['action_chunk_horizon'])
    # Mirror training-time conditioning: 'subgoal' → π sees GOUB predict_subgoal(obs, g);
    # 'goal' → π sees the global goal directly. IDM eval below always uses predicted
    # subgoals because GOUB IDM is the chunk planner targeting subgoal endpoints.
    actor_spi_cond = str(actor_config.get('spi_conditioned', 'subgoal')).lower()
    actor_task_successes = []
    idm_task_successes = []
    metrics = {}

    for task_id in task_ids:
        actor_episode_successes = []
        idm_episode_successes = []
        for _ in range(max(1, int(episodes_per_task))):
            ob, info = env.reset(options=dict(task_id=int(task_id), render_goal=False))
            if 'goal' not in info:
                raise RuntimeError(f'Env reset(task_id={task_id}) did not provide info["goal"].')
            obs = np.asarray(ob, dtype=np.float32).reshape(-1)
            goal = np.asarray(info['goal'], dtype=np.float32).reshape(-1)
            success = goal_within_tol(obs, goal, goal_dims, goal_tol)
            terminated = False
            truncated = False

            for _ in range(max(1, int(max_chunks))):
                if success or terminated or truncated:
                    break
                predicted_subgoal = np.asarray(goub_agent.infer_subgoal(obs, goal), dtype=np.float32).reshape(-1)
                actor_cond = predicted_subgoal if actor_spi_cond == 'subgoal' else goal
                action_chunk = np.asarray(actor_agent.sample_actions(obs, actor_cond), dtype=np.float32).reshape(
                    actor_horizon, -1
                )
                obs, success, terminated, truncated = _execute_action_chunk(
                    env,
                    obs,
                    goal,
                    action_chunk,
                    low=low,
                    high=high,
                    goal_dims=goal_dims,
                    goal_tol=goal_tol,
                )
            actor_episode_successes.append(1.0 if success else 0.0)

            ob, info = env.reset(options=dict(task_id=int(task_id), render_goal=False))
            if 'goal' not in info:
                raise RuntimeError(f'Env reset(task_id={task_id}) did not provide info["goal"].')
            obs = np.asarray(ob, dtype=np.float32).reshape(-1)
            goal = np.asarray(info['goal'], dtype=np.float32).reshape(-1)
            success = goal_within_tol(obs, goal, goal_dims, goal_tol)
            terminated = False
            truncated = False

            for _ in range(max(1, int(max_chunks))):
                if success or terminated or truncated:
                    break
                predicted_subgoal = np.asarray(goub_agent.infer_subgoal(obs, goal), dtype=np.float32).reshape(-1)
                action_chunk = _idm_action_chunk(goub_agent, obs, predicted_subgoal, idm_horizon)
                obs, success, terminated, truncated = _execute_action_chunk(
                    env,
                    obs,
                    goal,
                    action_chunk,
                    low=low,
                    high=high,
                    goal_dims=goal_dims,
                    goal_tol=goal_tol,
                )
            idm_episode_successes.append(1.0 if success else 0.0)

        task_success_rate = float(np.mean(actor_episode_successes))
        metrics[f'eval/task_{task_id}/success_rate'] = task_success_rate
        actor_task_successes.append(task_success_rate)

        idm_task_success_rate = float(np.mean(idm_episode_successes))
        metrics[f'eval_idm/task_{task_id}/success_rate'] = idm_task_success_rate
        idm_task_successes.append(idm_task_success_rate)

    metrics['eval/success_rate_mean'] = float(np.mean(actor_task_successes))
    metrics['eval_idm/success_rate_mean'] = float(np.mean(idm_task_successes))
    metrics['eval/num_tasks'] = float(len(task_ids))
    metrics['eval/episodes_per_task'] = float(max(1, int(episodes_per_task)))
    return metrics


def main(_):
    impl = _impl_dir()
    resume_run_dir = FLAGS.resume_run_dir.strip()
    resume_epoch = int(FLAGS.resume_epoch)
    if bool(resume_run_dir) != bool(resume_epoch > 0):
        raise ValueError('resume_run_dir and resume_epoch must be provided together.')

    cfg_path = FLAGS.run_config.strip() or _default_yaml_path()
    resume_snapshot_path: str | None = None
    if (
        resume_run_dir
        and resume_epoch > 0
        and FLAGS.resume_use_run_snapshot_config
        and not _argv_sets_flag('run_config')
    ):
        resume_snapshot_path = _resolve_resume_snapshot_config_path(os.path.abspath(resume_run_dir))
        if resume_snapshot_path is not None:
            cfg_path = resume_snapshot_path
        else:
            print(
                f'[joint] WARN resume_use_run_snapshot_config but no flags.json or config_used.yaml '
                f'in {resume_run_dir!r}; using default run_config: {cfg_path}',
                file=sys.stderr,
            )

    goub_updates, critic_updates, actor_updates = {}, {}, {}
    if os.path.isfile(cfg_path):
        goub_updates, critic_updates, actor_updates = _apply_yaml_to_flags(_load_yaml(cfg_path))
    elif FLAGS.run_config.strip():
        raise FileNotFoundError(f'run_config YAML not found: {cfg_path}')
    else:
        raise FileNotFoundError(f'run_config YAML not found: {cfg_path}')

    goub_config, critic_config, actor_config = _prepare_joint_configs(
        goub_updates,
        critic_updates,
        actor_updates,
    )

    runs_root = FLAGS.runs_root.strip() or os.path.join(impl, 'runs')
    if resume_run_dir:
        run_dir = os.path.abspath(resume_run_dir)
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f'resume_run_dir not found: {run_dir}')
    else:
        ts = time.strftime('%Y%m%d_%H%M%S')
        env_tok = _sanitize_token(FLAGS.env_name)
        run_folder = f'{ts}_joint_dqc_seed{FLAGS.seed}_{env_tok}'
        run_dir = os.path.join(runs_root, run_folder)
    ckpt_root = os.path.join(run_dir, 'checkpoints')
    goub_ckpt_dir = os.path.join(ckpt_root, 'goub')
    critic_ckpt_dir = os.path.join(ckpt_root, 'critic')
    actor_ckpt_dir = os.path.join(ckpt_root, 'actor')
    os.makedirs(goub_ckpt_dir, exist_ok=True)
    os.makedirs(critic_ckpt_dir, exist_ok=True)
    os.makedirs(actor_ckpt_dir, exist_ok=True)
    if os.path.isfile(cfg_path) and not resume_run_dir:
        shutil.copy2(cfg_path, os.path.join(run_dir, 'config_used.yaml'))

    exp_name = get_exp_name(FLAGS.seed, env_name=FLAGS.env_name, agent_name='joint_dqc')
    if FLAGS.use_wandb:
        setup_wandb(project='OGBench-Joint', group=FLAGS.run_group, name=exp_name)

    run_logger, run_log_path = _setup_file_logger(run_dir, resume_epoch=resume_epoch if resume_run_dir else 0)
    run_logger.info('run_dir=%s critic=dqc', run_dir)
    run_logger.info('log_path=%s', run_log_path)
    if resume_snapshot_path is not None:
        run_logger.info('resume hyperparameters from snapshot file: %s', resume_snapshot_path)

    env, train_plain, _ = make_env_and_datasets(FLAGS.env_name, frame_stack=critic_config['frame_stack'])
    action_dim = int(np.asarray(env.action_space.shape).prod())
    critic_config['action_dim'] = action_dim
    actor_config['action_dim'] = action_dim

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

    goub_dataset = PathHGCDataset(Dataset.create(**train_plain), goub_config)
    critic_dataset = _make_critic_dataset(train_plain, critic_config)
    common_valid_starts = _intersect_valid_starts(goub_dataset, critic_dataset)
    if int(goub_config['goub_N']) < int(actor_config['actor_chunk_horizon']):
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
    critic_agent = _create_critic_agent(FLAGS.seed, ex_critic, critic_config)
    actor_agent = _create_actor_agent(FLAGS.seed, ex_goub, actor_config)
    if resume_run_dir:
        goub_agent = restore_agent(goub_agent, goub_ckpt_dir, resume_epoch)
        critic_agent = restore_agent(critic_agent, critic_ckpt_dir, resume_epoch)
        actor_agent = restore_agent(actor_agent, actor_ckpt_dir, resume_epoch)

    batch_size = int(goub_config['batch_size'])
    spe = _steps_per_epoch(len(common_valid_starts), batch_size)
    run_logger.info(
        'shared_valid_starts=%d batch_size=%d steps_per_epoch=%d goub_h=%d critic_h=%d actor_h=%d',
        len(common_valid_starts),
        batch_size,
        spe,
        int(goub_config['subgoal_steps']),
        int(critic_config.get('full_chunk_horizon', 0)),
        int(actor_config.get('actor_chunk_horizon', 0)),
    )

    train_logger = CsvLogger(os.path.join(run_dir, 'train.csv'), resume=bool(resume_run_dir), flush_every_n=1)
    first_time = time.time()
    last_log = time.time()
    measure_timing = bool(FLAGS.measure_timing)
    eval_freq = int(FLAGS.eval_freq)
    eval_task_ids = parse_int_list(FLAGS.eval_task_ids)
    eval_episodes_per_task = max(1, int(FLAGS.eval_episodes_per_task))
    eval_goal_dims = parse_int_list(FLAGS.eval_goal_dims)
    eval_goal_dims = eval_goal_dims if len(eval_goal_dims) > 0 else None
    eval_goal_tol = float(FLAGS.eval_goal_tol)
    eval_max_chunks = max(1, int(FLAGS.eval_max_chunks))

    start_epoch = resume_epoch + 1 if resume_run_dir else 1
    epoch_iter = range(start_epoch, FLAGS.train_epochs + 1)
    if FLAGS.use_tqdm:
        epoch_iter = tqdm.tqdm(epoch_iter, smoothing=0.1, dynamic_ncols=True)

    for epoch in epoch_iter:
        if measure_timing:
            epoch_start = time.perf_counter()
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
            goub_agent, critic_batch, actor_batch, coupling_info, build_detail_info = _build_joint_batches_dqc(
                goub_agent,
                critic_agent,
                goub_batch,
                critic_batch,
                actor_config,
            )
            if measure_timing:
                _block_until_ready((critic_batch, actor_batch))
                build_time += time.perf_counter() - t0
                _accumulate_time_sums(build_detail_times, build_detail_info)

            if measure_timing:
                t0 = time.perf_counter()
            goub_agent, goub_info = goub_agent.update(
                goub_batch,
                critic_value_params=_extract_critic_value_params(critic_agent),
            )
            if measure_timing:
                _block_until_ready(goub_info)
                goub_time += time.perf_counter() - t0

            if measure_timing:
                t0 = time.perf_counter()
            critic_agent, critic_info = critic_agent.update(critic_batch)
            if measure_timing:
                _block_until_ready(critic_info)
                critic_time += time.perf_counter() - t0

            if measure_timing:
                t0 = time.perf_counter()
            actor_batch_for_update, score_coupling_info = _rescore_actor_batch_for_update(
                actor_batch, critic_agent, actor_config
            )
            coupling_info = dict(coupling_info)
            coupling_info.update(score_coupling_info)
            if measure_timing:
                _block_until_ready(actor_batch_for_update)
                actor_rescore_time += time.perf_counter() - t0

            if measure_timing:
                t0 = time.perf_counter()
            actor_agent, actor_info = actor_agent.update(actor_batch_for_update, critic_agent)
            if measure_timing:
                _block_until_ready(actor_info)
                actor_time += time.perf_counter() - t0

            last_goub_info = goub_info
            last_critic_info = critic_info
            last_actor_info = actor_info
            last_coupling_info = coupling_info

            _accumulate_metric_sums(goub_metric_sums, goub_info)
            _accumulate_metric_sums(critic_metric_sums, critic_info)
            _accumulate_metric_sums(actor_metric_sums, actor_info)
            _accumulate_metric_sums(coupling_metric_sums, coupling_info)

        gstep = epoch * spe
        steps_done = spe
        if epoch % FLAGS.log_every_n_epochs == 0 and last_goub_info is not None:
            metrics = {}
            metrics.update(_to_host_metrics('train/goub', last_goub_info))
            metrics.update(_to_host_metrics('train/critic', last_critic_info))
            metrics.update(_to_host_metrics('train/actor', last_actor_info))
            metrics.update(_to_host_metrics('train', last_coupling_info))
            metrics['train/critic/primary_score'] = extract_critic_primary_score(last_critic_info)
            _emit_metric_means(metrics, 'train/goub', goub_metric_sums, steps_done)
            _emit_metric_means(metrics, 'train/critic', critic_metric_sums, steps_done)
            _emit_metric_means(metrics, 'train/actor', actor_metric_sums, steps_done)
            _emit_metric_means(metrics, 'train/coupling', coupling_metric_sums, steps_done)
            # Backward-compatible aliases for legacy log/dashboard consumers.
            _alias = {
                'train/goub/loss_epoch_mean': 'train/goub/phase1/loss_epoch_mean',
                'train/critic/loss_epoch_mean': 'train/critic/dqc_critic/total_loss_epoch_mean',
                'train/value/loss_epoch_mean': 'train/critic/action_critic/value_loss_epoch_mean',
                'train/actor/loss_epoch_mean': 'train/actor/spi_actor/actor_loss_epoch_mean',
                'train/coupling/critic_score_epoch_mean': 'train/coupling/coupling/critic_score_mean_epoch_mean',
            }
            for dst, src in _alias.items():
                if src in metrics:
                    metrics[dst] = metrics[src]
            metrics['train/epoch'] = float(epoch)
            if eval_freq > 0 and epoch % eval_freq == 0:
                metrics.update(
                    _evaluate_env_tasks(
                        env,
                        goub_agent,
                        actor_agent,
                        actor_config,
                        critic_config,
                        critic_value_params=_extract_critic_value_params(critic_agent),
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
                metrics['time/actor_update_step_sec'] = actor_time / spe
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
                num_tasks = int(metrics.get('eval/num_tasks', 0.0))
                episodes_per_task = int(metrics.get('eval/episodes_per_task', 0.0))
                run_logger.info(
                    '=== EVAL START epoch=%d num_tasks=%d episodes_per_task=%d ===',
                    epoch,
                    num_tasks,
                    episodes_per_task,
                )
                run_logger.info('[IDM POLICY]')
                run_logger.info(
                    'idm success_rate_mean=%.2f',
                    metrics.get('eval_idm/success_rate_mean', float('nan')),
                )
                for task_id in eval_task_ids:
                    task_key = f'eval_idm/task_{task_id}/success_rate'
                    if task_key in metrics:
                        run_logger.info('idm task_%d=%.2f', task_id, metrics[task_key])
                run_logger.info('[ACTOR POLICY]')
                run_logger.info(
                    'actor success_rate_mean=%.2f',
                    metrics.get('eval/success_rate_mean', float('nan')),
                )
                for task_id in eval_task_ids:
                    task_key = f'eval/task_{task_id}/success_rate'
                    if task_key in metrics:
                        run_logger.info('actor task_%d=%.2f', task_id, metrics[task_key])
                run_logger.info('=== EVAL END epoch=%d ===', epoch)

        if epoch % FLAGS.save_every_n_epochs == 0:
            save_agent(goub_agent, goub_ckpt_dir, epoch)
            save_agent(critic_agent, critic_ckpt_dir, epoch)
            save_agent(actor_agent, actor_ckpt_dir, epoch)

    train_logger.close()
    run_logger.info('done run_dir=%s', run_dir)


if __name__ == '__main__':
    app.run(main)
