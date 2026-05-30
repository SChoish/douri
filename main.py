"""Offline training for linear-SDE dynamics + critic + SPI actor."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
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
    CriticAgent,
    extract_critic_primary_score,
    get_config as get_critic_config,
    validate_config,
)
from agents.actor import ActorAgent, get_actor_config
from agents.dynamics import DynamicsAgent, get_dynamics_config
from utils.datasets import Dataset, PathHGCDataset
from utils.critic_sequence_dataset import CriticSequenceDataset
from utils.env_utils import make_env_and_datasets
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, get_wandb_video, setup_wandb
from utils.ogbench_eval_rollout import rollout_chunked_eval_episode
from utils.run_io import parse_int_list
from utils.goal_representation import infer_phi_goal_obs_indices, normalize_phi_goal_obs_indices

FLAGS = flags.FLAGS
_DEFAULT_HORIZON = 25


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


def _require_gpu_jax(logger: logging.Logger) -> None:
    """Fail fast if JAX did not pick the CUDA GPU backend (avoids silent CPU training)."""
    if bool(FLAGS.allow_cpu):
        logger.warning(
            'allow_cpu=True: skipping GPU-only check (jax default_backend=%s devices=%s)',
            jax.default_backend(),
            jax.devices(),
        )
        return
    backend = str(jax.default_backend()).lower()
    devs = jax.devices()
    dev_str = ', '.join(str(d) for d in devs)
    cuda_vis = os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')
    jax_plat = os.environ.get('JAX_PLATFORMS', '<unset>')
    if backend != 'gpu':
        msg = (
            f'GPU-only mode: JAX default_backend is {backend!r} (expected "gpu"). devices=[{dev_str}]. '
            f'CUDA_VISIBLE_DEVICES={cuda_vis!r} JAX_PLATFORMS={jax_plat!r}. '
            'Install a CUDA-enabled jaxlib matching your driver (e.g. cuSPARSE), confirm `nvidia-smi`, '
            'or pass --allow_cpu=True for intentional CPU runs.'
        )
        logger.error(msg)
        raise RuntimeError(msg)
    logger.info('GPU-only check passed: default_backend=%s device_count=%d', backend, len(devs))


flags.DEFINE_string('run_config', '', 'YAML config; empty uses config/antmaze_large_navigate.yaml.')
flags.DEFINE_string('runs_root', '', 'Run root; default <repo>/runs.')
flags.DEFINE_string(
    'dataset_dir',
    '',
    'Optional dataset directory override. If this points to a sharded OGBench directory, load and concatenate all '
    'train/val NPZ shards from there.',
)
flags.DEFINE_string('resume_run_dir', '', 'Existing run dir: resume from checkpoint or reuse path (see resume_epoch).')
flags.DEFINE_integer(
    'resume_epoch',
    0,
    'With resume_run_dir: load params_<epoch>.pkl and continue if >0; if 0, reuse that run_dir from epoch 1 '
    'without restoring (for empty/stale dirs). Must be 0 when resume_run_dir is empty.',
)
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
flags.DEFINE_integer('train_epochs', 1000, 'Training epochs.')
flags.DEFINE_integer('log_every_n_epochs', 10, 'Log interval (epochs).')
flags.DEFINE_integer('save_every_n_epochs', 100, 'Checkpoint interval.')
flags.DEFINE_boolean('use_wandb', False, 'W&B.')
flags.DEFINE_boolean('use_tqdm', False, 'tqdm over epochs.')
flags.DEFINE_integer('batch_size', 1024, 'Shared batch size for dynamics, critic, and actor.')
flags.DEFINE_integer(
    'horizon', _DEFAULT_HORIZON, 'Shared horizon for dynamics_N, subgoal_steps, and full_chunk_horizon.'
)
flags.DEFINE_integer('plan_candidates', 1, 'Number of dynamics candidate plans scored by the critic.')
flags.DEFINE_float('plan_noise_scale', 1.0, 'Noise scale used for stochastic dynamics plan sampling.')
flags.DEFINE_boolean('measure_timing', False, 'Whether to measure and log per-phase wall-clock timings.')
flags.DEFINE_boolean(
    'async_prefetch',
    True,
    'Overlap host-side batch sampling with GPU work via a single-worker prefetch thread.',
)
flags.DEFINE_integer(
    'eval_freq',
    100,
    'Run validation loss and env evaluation every N epochs; <= 0 disables.',
)
flags.DEFINE_string('eval_task_ids', '1,2,3,4,5', 'Comma-separated OGBench task ids for env evaluation.')
flags.DEFINE_integer('eval_episodes_per_task', 10, 'Number of env evaluation episodes to run for each task id.')
flags.DEFINE_integer(
    'final_eval_episodes_per_task',
    50,
    'If > 0, override eval_episodes_per_task for the final training epoch evaluation only.',
)
flags.DEFINE_integer('eval_max_chunks', 200, 'Maximum action chunks to execute per evaluation episode.')
flags.DEFINE_integer(
    'eval_video_episodes_per_task',
    0,
    'Extra env episodes per task recorded to W&B video only (not included in success stats); '
    'episode index >= eval_episodes_per_task uses render_goal=True.',
)
flags.DEFINE_integer(
    'eval_video_frame_skip',
    4,
    'Save env.render() every N env steps during video episodes (plus last frame when done).',
)
flags.DEFINE_integer('eval_video_fps', 15, 'FPS for W&B eval videos built from eval_video_episodes_per_task.')
flags.DEFINE_boolean(
    'subgoal_override_goal',
    False,
    'Inference/eval ablation: ignore predicted subgoals and condition IDM/actor directly on the final goal.',
)
flags.DEFINE_boolean(
    'allow_cpu',
    False,
    'If True, allow JAX to run on CPU when CUDA is unavailable. Default False: require GPU and exit '
    'with an error if jax.default_backend() is not gpu (no silent CPU fallback).',
)

_SPI_ACTOR_KEYS = {
    'spi_tau',
    'spi_beta',
    'spi_actor_layer_norm',
    'spi_q_norm_eps',
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
    if snap.get('dynamics') is not None:
        root['dynamics'] = snap['dynamics']
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
    dynamics_updates = data.pop('dynamics', None)
    critic_updates = data.pop('critic_agent', None)
    actor_updates = data.pop('actor', None)
    for name, updates in [('dynamics', dynamics_updates), ('critic_agent', critic_updates), ('actor', actor_updates)]:
        if updates is not None and not isinstance(updates, dict):
            raise ValueError(f'YAML key "{name}" must be a mapping.')

    # Allow forward-bridge planner knobs at the top level for ergonomic YAML
    # (route them into dynamics_updates so the dynamics-agent config picks them up).
    if 'planner_type' in data:
        dynamics_updates = dynamics_updates or {}
        dynamics_updates.setdefault('planner_type', data.pop('planner_type'))
    if 'forward_bridge' in data:
        dynamics_updates = dynamics_updates or {}
        dynamics_updates.setdefault('forward_bridge', data.pop('forward_bridge'))

    # Flatten ``dynamics.forward_bridge: { mode, noise_scale, ... }`` into the
    # individual ``forward_bridge_*`` keys recognised by the dynamics config.
    if isinstance(dynamics_updates, dict) and isinstance(dynamics_updates.get('forward_bridge'), dict):
        fb = dynamics_updates.pop('forward_bridge')
        for k, v in fb.items():
            dynamics_updates.setdefault(f'forward_bridge_{k}', v)

    for key, value in data.items():
        if not hasattr(FLAGS, key):
            raise ValueError(f'Unknown YAML top-level key: {key!r}')
        if _argv_sets_flag(key):
            continue
        setattr(FLAGS, key, value)
    return dynamics_updates or {}, critic_updates or {}, actor_updates or {}


def _setup_file_logger(run_dir: str, *, resume_epoch: int = 0) -> tuple[logging.Logger, str]:
    if resume_epoch > 0:
        ts = time.strftime('%Y%m%d_%H%M%S')
        log_name = f'run_resume_from{int(resume_epoch)}_{ts}.log'
    else:
        log_name = 'run.log'
    log_path = os.path.join(run_dir, log_name)
    logger = logging.getLogger('train')
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


def compute_state_normalization_stats(dataset: dict, eps: float = 1e-6) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Compute state-normalization stats from the full offline training dataset."""
    obs = np.asarray(dataset['observations'], dtype=np.float32)
    if obs.ndim != 2:
        raise ValueError(f'state_normalization expects 2D observations, got shape={obs.shape}.')
    mean = obs.mean(axis=0)
    std = np.maximum(obs.std(axis=0), float(eps))
    return tuple(float(x) for x in mean), tuple(float(x) for x in std)


def _attach_state_normalization_stats(dynamics_config: Any, train_plain: dict) -> None:
    """Populate dynamics state-normalization stats from the full offline train set."""
    if not bool(dynamics_config.get('state_normalization', False)):
        return
    obs = np.asarray(train_plain['observations'], dtype=np.float32)
    if obs.ndim != 2:
        raise ValueError(f'state_normalization expects 2D observations, got shape={obs.shape}.')
    eps = float(dynamics_config.get('state_normalization_eps', 1e-6))
    mean, std = compute_state_normalization_stats(train_plain, eps)
    dynamics_config['state_mean'] = mean
    dynamics_config['state_std'] = std


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
        f"dyn={metrics['train/dynamics/phase1/loss_epoch_mean']:.6f}",
        f"critic={metrics['train/critic/total_loss_epoch_mean']:.6f}",
    ]
    actor_key = 'train/actor/spi_actor/actor_loss_epoch_mean'
    if actor_key in metrics:
        parts.append(f"actor={metrics[actor_key]:.6f}")
    else:
        parts.append('actor=disabled')
    parts.append(f"coupling={metrics['train/coupling/critic_score_mean_epoch_mean']:.6f}")

    detail_keys = [
        ('dyn_g', 'train/dynamics/phase1/loss_dynamics_epoch_mean'),
        ('dyn_path', 'train/dynamics/phase1/loss_path_step_epoch_mean'),
        ('dyn_roll', 'train/dynamics/phase1/loss_roll_epoch_mean'),
        ('dyn_sub', 'train/dynamics/phase1/loss_subgoal_epoch_mean'),
        ('dyn_idm', 'train/dynamics/phase1/loss_idm_epoch_mean'),
        ('fb_path', 'train/dynamics/forward_bridge/loss_path_interior_epoch_mean'),
        ('fb_next', 'train/dynamics/forward_bridge/loss_path_next_epoch_mean'),
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
        ('t_dyn', 'time/dynamics_update_epoch_sec'),
        ('t_critic', 'time/critic_update_epoch_sec'),
        ('t_actor_rescore', 'time/actor_rescore_epoch_sec'),
        ('t_actor', 'time/actor_update_epoch_sec'),
        ('t_epoch', 'time/epoch_compute_sec'),
    ]
    for label, key in detail_keys:
        if key in metrics:
            parts.append(f'{label}={metrics[key]:.6f}')
    return ' '.join(parts)


def _apply_horizon(dynamics_config: Any, critic_config: Any) -> tuple[Any, Any]:
    horizon = int(FLAGS.horizon)
    if horizon < 1:
        raise ValueError(f'horizon must be >= 1, got {horizon}.')
    dynamics_config['dynamics_N'] = horizon
    dynamics_config['subgoal_steps'] = horizon
    critic_config['full_chunk_horizon'] = horizon
    return dynamics_config, critic_config


def _env_max_episode_steps(env: Any) -> int:
    """Return the environment episode cap advertised by Gym/Gymnasium wrappers."""
    spec = getattr(env, 'spec', None)
    max_steps = getattr(spec, 'max_episode_steps', None) if spec is not None else None
    if max_steps is None:
        max_steps = getattr(env, '_max_episode_steps', None)
    if max_steps is None:
        raise ValueError(
            'max_goal_steps="env" requested, but the environment does not expose max_episode_steps.'
        )
    max_steps = int(max_steps)
    if max_steps < 1:
        raise ValueError(f'env max_episode_steps must be >= 1, got {max_steps}.')
    return max_steps


def _resolve_max_goal_steps_from_env(config: Any, env: Any) -> bool:
    """Resolve ``max_goal_steps: env`` in-place. Return True when resolved."""
    if bool(config.get('max_goal_steps_from_env', False)):
        config['max_goal_steps'] = _env_max_episode_steps(env)
        return True
    value = config.get('max_goal_steps', None)
    if isinstance(value, str) and value.lower() in ('env', 'env_max_episode_steps', 'max_episode_steps'):
        with config.ignore_type():
            config['max_goal_steps'] = _env_max_episode_steps(env)
        return True
    return False


def _require_matching_frame_stack(dynamics_config: Any, critic_config: Any) -> None:
    frame_stacks = {
        'dynamics': dynamics_config.get('frame_stack', None),
        'critic': critic_config.get('frame_stack', None),
    }
    if len({str(v) for v in frame_stacks.values()}) != 1:
        raise ValueError(f'Training requires matching frame_stack across modules, got {frame_stacks}.')


def _make_critic_dataset(train_plain: dict, critic_config: Any):
    dataset = Dataset.create(**train_plain)
    return CriticSequenceDataset(dataset, critic_config)


def _intersect_valid_starts(dynamics_dataset: PathHGCDataset, critic_dataset: Any) -> np.ndarray:
    common = np.intersect1d(dynamics_dataset.path_valid_idxs, critic_dataset.valid_starts, assume_unique=False)
    common = np.asarray(common, dtype=np.int64)
    if len(common) == 0:
        raise ValueError('No shared valid starts across dynamics and critic datasets.')
    return common


def _sample_shared_idxs(common_valid_starts: np.ndarray, batch_size: int) -> np.ndarray:
    picked = np.random.randint(len(common_valid_starts), size=batch_size)
    return common_valid_starts[picked]


def _prepare_train_batch(
    common_valid_starts: np.ndarray,
    batch_size: int,
    dynamics_dataset,
    critic_dataset,
):
    """Sample one (dynamics, critic) batch from the host datasets.

    Pure CPU work; safe to run on a prefetch worker thread because the only
    shared state is NumPy's global RNG, and we serialize prefetch through a
    single-worker executor so randint() calls remain deterministic in order.
    """
    idxs = _sample_shared_idxs(common_valid_starts, batch_size)
    dynamics_batch = dynamics_dataset.sample(batch_size, idxs=idxs)
    critic_batch = critic_dataset.sample(batch_size, idxs=idxs)
    return dynamics_batch, critic_batch


def _eval_batch_size(common_valid_starts: np.ndarray, batch_size: int) -> int:
    return max(1, min(int(batch_size), int(len(common_valid_starts))))


@partial(jax.jit, static_argnames=('horizon',))
def _idm_actions_from_trajectories_jit(
    network: Any,
    trajectories: jnp.ndarray,
    horizon: int,
    *,
    state_mean: jnp.ndarray | None = None,
    state_std: jnp.ndarray | None = None,
) -> jnp.ndarray:
    prev_states = trajectories[:, :horizon, :]
    next_states = trajectories[:, 1 : horizon + 1, :]
    flat_prev = prev_states.reshape(-1, prev_states.shape[-1])
    flat_next = next_states.reshape(-1, next_states.shape[-1])
    if state_mean is not None and state_std is not None:
        flat_prev = (flat_prev - state_mean) / state_std
        flat_next = (flat_next - state_mean) / state_std
    pred = network.select('idm_net')(flat_prev, flat_next)
    return jnp.asarray(pred, dtype=jnp.float32).reshape(trajectories.shape[0], horizon, -1)


def _idm_actions_from_trajectories(dynamics_agent: DynamicsAgent, trajectories: np.ndarray, horizon: int) -> jnp.ndarray:
    if trajectories.shape[1] <= horizon:
        raise ValueError(
            f'Dynamics trajectory length {trajectories.shape[1]} is too short for horizon={horizon}. '
            'Increase dynamics_N / subgoal_steps or reduce chunk horizons.'
        )
    trajectories = jnp.asarray(trajectories, dtype=jnp.float32)
    return dynamics_agent._idm_actions_from_trajectories(trajectories, horizon)


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


@partial(jax.jit, static_argnames=('keep_topk', 'use_partial_critic'))
def _rescore_with_stats_jit(
    critic_agent: Any,
    obs: jnp.ndarray,
    spi_goals: jnp.ndarray,
    critic_goals: jnp.ndarray,
    candidates: jnp.ndarray,
    valids: jnp.ndarray,
    network_params: Any,
    *,
    keep_topk: int,
    use_partial_critic: bool,
) -> tuple[dict, dict]:
    """Single-graph rescore + ranking + score statistics.

    Replaces a chain of separate dispatches (score → rank → mean/max/min/gap)
    with one compiled function so that all stats live in the same XLA graph
    as the critic forward.
    """
    proposal_chunks, proposal_scores = _score_and_rank_candidate_actions(
        critic_agent,
        obs,
        critic_goals,
        candidates,
        network_params,
        keep_topk=keep_topk,
        use_partial_critic=use_partial_critic,
    )
    score_mean = proposal_scores.mean()
    score_max = proposal_scores.max()
    score_min = proposal_scores.min()
    if proposal_scores.shape[1] >= 2:
        gap = (proposal_scores[:, 0] - proposal_scores[:, 1]).mean()
    else:
        gap = jnp.zeros((), dtype=jnp.float32)
    out_batch = {
        'observations': obs,
        'spi_goals': spi_goals,
        'proposal_partial_chunks': proposal_chunks,
        'proposal_scores': proposal_scores,
        'valids': valids,
    }
    coupling_stats = {
        'critic_score_mean': score_mean,
        'critic_score_max': score_max,
        'critic_score_min': score_min,
        'critic_score_gap_top1_top2': gap,
    }
    return out_batch, coupling_stats


@partial(jax.jit, static_argnames=('use_partial_critic',))
def _rescore_top1_proposal_with_stats_jit(
    critic_agent: Any,
    obs: jnp.ndarray,
    spi_goals: jnp.ndarray,
    high_goals: jnp.ndarray,
    critic_goals: jnp.ndarray,
    candidates: jnp.ndarray,
    valids: jnp.ndarray,
    network_params: Any,
    *,
    use_partial_critic: bool,
) -> tuple[dict, dict]:
    """Score ``[B, K, ...]`` candidates and keep the global best proposal.

    Dynamics may generate ``K = U*N`` action proposals from ``U`` sampled
    subgoal endpoints and ``N`` bridge/action samples per endpoint.  The SPI
    actor should condition on the subgoal associated with the winning proposal,
    so this keeps one global best candidate and forwards its goal as
    ``spi_goals``.
    """
    q_scores = jnp.asarray(
        critic_agent.score_action_chunks(
            obs,
            critic_goals,
            candidates,
            network_params=network_params,
            use_partial_critic=use_partial_critic,
        ),
        dtype=jnp.float32,
    )
    if hasattr(critic_agent, 'score_transitive_subgoals'):
        v_scores = jnp.asarray(
            critic_agent.score_transitive_subgoals(
                obs,
                critic_goals,
                high_goals,
                network_params=network_params,
            ),
            dtype=jnp.float32,
        )
    else:
        v_scores = jnp.zeros_like(q_scores)
    mode = str(critic_agent.config.get('proposal_score_mode', 'q_only')).lower()
    if mode == 'q_only':
        scores = q_scores
    elif mode == 'v_only':
        scores = v_scores
    elif mode == 'q_plus_v':
        scores = (
            float(critic_agent.config.get('proposal_q_weight', 1.0)) * q_scores
            + float(critic_agent.config.get('proposal_v_weight', 1.0)) * v_scores
        )
    else:
        raise ValueError(
            "proposal_score_mode must be one of 'q_only', 'v_only', or 'q_plus_v', "
            f"got {mode!r}."
        )
    best_idx = jnp.argmax(scores, axis=1)
    best_chunks = jnp.take_along_axis(candidates, best_idx[:, None, None, None], axis=1)
    best_scores = jnp.take_along_axis(scores, best_idx[:, None], axis=1)
    if critic_goals is not None and critic_goals.ndim == 3:
        best_goals = jnp.take_along_axis(critic_goals, best_idx[:, None, None], axis=1)[:, 0, :]
    else:
        best_goals = spi_goals

    score_mean = best_scores.mean()
    score_max = best_scores.max()
    score_min = best_scores.min()
    if scores.shape[1] >= 2:
        sorted_scores = jnp.sort(scores, axis=1)[:, ::-1]
        gap = (sorted_scores[:, 0] - sorted_scores[:, 1]).mean()
    else:
        gap = jnp.zeros((), dtype=jnp.float32)

    out_batch = {
        'observations': obs,
        'spi_goals': jnp.asarray(best_goals, dtype=jnp.float32),
        'proposal_partial_chunks': jnp.asarray(best_chunks, dtype=jnp.float32),
        'proposal_scores': jnp.asarray(best_scores, dtype=jnp.float32),
        'valids': valids,
    }
    coupling_stats = {
        'critic_score_mean': score_mean,
        'critic_score_max': score_max,
        'critic_score_min': score_min,
        'critic_score_gap_top1_top2': gap,
        'critic_score_pre_best_mean': scores.mean(),
        'critic_score_pre_best_max': scores.max(),
        'coupling/proposal_q_score_mean': q_scores.mean(),
        'coupling/proposal_v_score_mean': v_scores.mean(),
        'coupling/proposal_combined_score_mean': scores.mean(),
    }
    return out_batch, coupling_stats


@jax.jit
def _proposal_goal_stats_jit(
    actor_goal_mean: jnp.ndarray,
    candidate_goals: jnp.ndarray,
) -> dict:
    """Compute proposal-goal coupling stats in a single fused dispatch."""
    return {
        'predicted_subgoal_norm': jnp.linalg.norm(actor_goal_mean, axis=-1).mean(),
        'proposal_goal_norm_mean': jnp.linalg.norm(candidate_goals, axis=-1).mean(),
        'proposal_goal_std_mean': candidate_goals.std(axis=1).mean(),
    }


def _build_actor_batch_from_dynamics(
    dynamics_agent: DynamicsAgent,
    critic_agent: Any,
    dynamics_batch: dict,
    actor_config: Any,
) -> tuple[DynamicsAgent, dict, dict, dict[str, float]]:
    obs = jnp.asarray(dynamics_batch['observations'], dtype=jnp.float32)
    high_goals = jnp.asarray(dynamics_batch['high_actor_goals'], dtype=jnp.float32)
    measure_timing = bool(FLAGS.measure_timing)
    timing = {}

    plan_candidates = max(1, int(FLAGS.plan_candidates))
    proposal_horizon = int(actor_config['actor_chunk_horizon'])
    if measure_timing:
        t0 = time.perf_counter()
    sample_noise_scale = float(FLAGS.plan_noise_scale) if plan_candidates > 1 else 0.0
    actor_goal_mean, candidate_actions, candidate_goals, plan_rng = dynamics_agent.build_actor_proposals(
        obs,
        high_goals,
        dynamics_agent.rng,
        proposal_horizon=proposal_horizon,
        plan_candidates=plan_candidates,
        sample_noise_scale=sample_noise_scale,
    )
    if measure_timing:
        _block_until_ready((actor_goal_mean, candidate_actions, candidate_goals, plan_rng))
        timing['proposal_build'] = time.perf_counter() - t0
    else:
        timing = {}
    dynamics_agent = dynamics_agent.replace(rng=plan_rng)
    # ``spi_goals`` is provisional here.  Rescoring replaces it with the
    # subgoal attached to the single best proposal before the actor update.
    use_mean_for_actor = bool(dynamics_agent.config.get('subgoal_use_mean_for_actor_goal', True))
    spi_goals = actor_goal_mean if use_mean_for_actor else high_goals
    actor_batch = {
        'observations': obs,
        'spi_goals': spi_goals,
        # Candidate action chunks generated from dynamics proposals; rescored after critic update.
        # Shape: [B, N, ha, A]
        'candidate_partial_chunks': candidate_actions,
        'valids': jnp.ones((obs.shape[0], proposal_horizon), dtype=jnp.float32),
        # Per-candidate sub-goal endpoints for critic rescoring (deterministic mode: mean broadcast).
        # Shape: [B, N, D]
        'candidate_goals': candidate_goals,
        'high_actor_goals': high_goals,
        # Each subgoal contributes ``plan_candidates`` bridge/action samples.
        # Rescoring keeps the global best proposal across the full candidate axis.
        'candidate_group_size': plan_candidates,
    }

    nan = jnp.full((), jnp.nan, dtype=jnp.float32)
    proposal_goal_stats = _proposal_goal_stats_jit(actor_goal_mean, candidate_goals)
    coupling_info = {
        **proposal_goal_stats,
        'critic_score_mean': nan,
        'critic_score_max': nan,
        'critic_score_min': nan,
        'critic_score_gap_top1_top2': nan,
        'proposal_count': jnp.asarray(float(candidate_actions.shape[1]), dtype=jnp.float32),
    }
    return dynamics_agent, actor_batch, coupling_info, timing


def _rescore_actor_batch_for_update(actor_batch: dict, critic_agent: Any, actor_config: Any) -> tuple[dict, dict]:
    obs = jnp.asarray(actor_batch['observations'], dtype=jnp.float32)
    goals = jnp.asarray(actor_batch['spi_goals'], dtype=jnp.float32)
    high_goals = jnp.asarray(actor_batch.get('high_actor_goals', goals), dtype=jnp.float32)
    candidates = jnp.asarray(actor_batch['candidate_partial_chunks'], dtype=jnp.float32)  # [B, N, ha, A]
    valids = jnp.asarray(actor_batch['valids'], dtype=jnp.float32)
    # Optional per-candidate sub-goal endpoints (distributional subgoal mode).
    cand_goals_in = actor_batch.get('candidate_goals', None)
    if cand_goals_in is not None:
        critic_goals = jnp.asarray(cand_goals_in, dtype=jnp.float32)  # [B, N, D]
    else:
        critic_goals = goals  # [B, D] - shared
    force_rescore_single = False
    if hasattr(critic_agent, '_is_direct_chunk_trl'):
        force_rescore_single = bool(critic_agent._is_direct_chunk_trl())
    if hasattr(critic_agent, '_is_state_transitive'):
        force_rescore_single = force_rescore_single or bool(critic_agent._is_state_transitive())
    force_rescore_single = force_rescore_single or bool(
        critic_agent.config.get('rescore_single_candidate', False)
    )
    # Fast path: single candidate -> skip critic call entirely (unless forced).
    if candidates.shape[1] == 1 and not force_rescore_single:
        zero = jnp.zeros((), dtype=jnp.float32)
        if cand_goals_in is not None:
            selected_goals = critic_goals[:, 0, :] if critic_goals.ndim == 3 else critic_goals
        else:
            selected_goals = goals
        return (
            {
                'observations': obs,
                'spi_goals': selected_goals,
                'proposal_partial_chunks': candidates,
                'proposal_scores': jnp.zeros((obs.shape[0], 1), dtype=jnp.float32),
                'valids': valids,
            },
            {
                'critic_score_mean': zero,
                'critic_score_max': zero,
                'critic_score_min': zero,
                'critic_score_gap_top1_top2': zero,
                'critic_score_pre_best_mean': zero,
                'critic_score_pre_best_max': zero,
                'proposal_best_of_n': jnp.asarray(1.0, dtype=jnp.float32),
                'proposal_pre_best_count': jnp.asarray(1.0, dtype=jnp.float32),
                'proposal_post_best_count': jnp.asarray(1.0, dtype=jnp.float32),
                'proposal_count': jnp.asarray(1.0, dtype=jnp.float32),
                'coupling/proposal_q_score_mean': zero,
                'coupling/proposal_v_score_mean': zero,
                'coupling/proposal_combined_score_mean': zero,
            },
        )
    # Multi-candidate path: score all U*N proposals and keep one global best.
    out_batch, stats = _rescore_top1_proposal_with_stats_jit(
        critic_agent,
        obs,
        goals,
        high_goals,
        critic_goals,
        candidates,
        valids,
        critic_agent.network.params,
        use_partial_critic=True,
    )
    stats = dict(stats)
    stats['proposal_best_of_n'] = jnp.asarray(float(candidates.shape[1]), dtype=jnp.float32)
    stats['proposal_pre_best_count'] = jnp.asarray(float(candidates.shape[1]), dtype=jnp.float32)
    stats['proposal_post_best_count'] = jnp.asarray(1.0, dtype=jnp.float32)
    stats['proposal_count'] = stats['proposal_post_best_count']
    return out_batch, stats


def _build_train_batches(
    dynamics_agent: DynamicsAgent,
    critic_agent: Any,
    dynamics_batch: dict,
    critic_batch: dict,
    actor_config: Any,
) -> tuple[DynamicsAgent, dict, dict, dict, dict[str, float]]:
    dynamics_agent, actor_batch, coupling_info, build_timing = _build_actor_batch_from_dynamics(
        dynamics_agent, critic_agent, dynamics_batch, actor_config
    )
    return dynamics_agent, critic_batch, actor_batch, coupling_info, build_timing


def _merge_actor_updates(actor_config: Any, actor_updates: dict) -> Any:
    ignored = sorted(k for k in actor_updates.keys() if k not in _SPI_ACTOR_KEYS)
    if ignored:
        logging.warning('Ignoring deprecated non-SPI actor keys: %s', ', '.join(ignored))
    for key in _SPI_ACTOR_KEYS:
        if key in actor_updates:
            actor_config[key] = actor_updates[key]
    return actor_config


def _prepare_configs(dynamics_updates: dict, critic_updates: dict, actor_updates: dict):
    dynamics_config = _update_config(get_dynamics_config(), dynamics_updates)
    critic_config = _update_config(get_critic_config(), critic_updates)
    actor_config = _merge_actor_updates(get_actor_config(), actor_updates)
    dynamics_config, critic_config = _apply_horizon(dynamics_config, critic_config)
    actor_config['actor_chunk_horizon'] = int(critic_config['action_chunk_horizon'])
    # Subgoal-value bonus net shares parameters with the critic value net, so its
    # architecture must mirror the critic; force-sync here so users only configure it once.
    dynamics_config['subgoal_value_hidden_dims'] = tuple(int(x) for x in critic_config['value_hidden_dims'])
    dynamics_config['subgoal_value_layer_norm'] = bool(critic_config['layer_norm'])
    dynamics_config['subgoal_value_goal_representation'] = str(
        critic_config.get('goal_representation', 'full'),
    )
    dynamics_config['critic_type'] = str(critic_config.get('critic_type', 'dqc'))
    dynamics_config['algorithm'] = str(critic_config.get('algorithm', 'dqc'))
    if str(critic_config.get('critic_type', 'dqc')).lower() in ('state_transitive', 'transitive_v_local_q') or str(
        critic_config.get('algorithm', 'dqc')
    ).lower() in ('state_transitive', 'transitive_v_local_q'):
        bonus_type = str(critic_config.get('subgoal_value_bonus_type', 'transitive_ratio')).lower()
        dynamics_config['subgoal_value_bonus_type'] = (
            'transitive_ratio' if bonus_type == 'single_value' else bonus_type
        )
        dynamics_config['subgoal_value_ratio_eps'] = float(critic_config.get('subgoal_value_ratio_eps', 1e-6))
    phi_idxs = normalize_phi_goal_obs_indices(critic_config.get('phi_goal_obs_indices', ()))
    critic_config['phi_goal_obs_indices'] = phi_idxs
    dynamics_config['phi_goal_obs_indices'] = phi_idxs
    # Propagate env_name so that 'phi' goal representation can dispatch to the
    # correct ManipSpace oracle layout (cube xyz vs. puzzle binary button state).
    env_name_for_phi = str(FLAGS.env_name)
    dynamics_config['env_name'] = env_name_for_phi
    critic_config['env_name'] = env_name_for_phi
    validate_config(critic_config, actor_config)
    shared_batch = int(FLAGS.batch_size)
    if shared_batch < 1:
        raise ValueError(f'batch_size must be >= 1, got {shared_batch}.')
    dynamics_config['batch_size'] = shared_batch
    critic_config['batch_size'] = shared_batch
    actor_config['batch_size'] = shared_batch
    _require_matching_frame_stack(dynamics_config, critic_config)
    return dynamics_config, critic_config, actor_config


def _create_critic_agent(seed: int, ex: dict, critic_config):
    critic_type = str(critic_config.get('critic_type', 'dqc')).lower()
    ex_full = ex['full_chunk_actions'] if critic_type == 'dqc' else None
    return CriticAgent.create(
        seed,
        ex['observations'],
        ex_full,
        ex['action_chunk_actions'],
        critic_config,
        ex_goals=ex.get('value_goals'),
    )


def _create_actor_agent(seed: int, ex_dynamics: dict, actor_config):
    return ActorAgent.create(
        seed,
        ex_dynamics['observations'],
        actor_config,
        ex_goals=ex_dynamics.get('high_actor_targets'),
    )


def _extract_critic_value_params(critic_agent: Any) -> Any | None:
    if critic_agent is None:
        return None
    if hasattr(critic_agent, '_is_state_transitive') and bool(critic_agent._is_state_transitive()):
        return critic_agent.network.params.get('modules_target_value', None)
    return critic_agent.network.params.get('modules_value', None)


def _idm_action_chunk(
    dynamics_agent: DynamicsAgent,
    obs: np.ndarray,
    predicted_subgoal: np.ndarray,
    horizon: int,
) -> np.ndarray:
    traj = np.asarray(dynamics_agent.plan(obs, predicted_subgoal)['trajectory'], dtype=np.float32)
    if traj.ndim != 2:
        raise RuntimeError(f'Expected single-trajectory plan with rank 2, got shape={traj.shape}.')
    action_chunk = np.asarray(_idm_actions_from_trajectories(dynamics_agent, traj[None, ...], horizon), dtype=np.float32)
    return action_chunk[0]


def _evaluate_env_tasks(
    env,
    dynamics_agent: DynamicsAgent,
    actor_agent: Any,
    actor_config: Any,
    critic_config: Any,
    *,
    task_ids: tuple[int, ...],
    episodes_per_task: int,
    max_chunks: int,
    video_episodes_per_task: int = 0,
    video_frame_skip: int = 4,
    video_fps: int = 15,
    wandb_enabled: bool = False,
    subgoal_override_goal: bool = False,
) -> dict[str, Any]:
    """OGBench-style eval: success is decided **only** by ``info['success']`` (any step). No tolerance diagnostic."""
    if not task_ids:
        return {}

    low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    actor_horizon = int(actor_config['actor_chunk_horizon'])
    idm_horizon = int(critic_config['action_chunk_horizon'])
    actor_task_successes: list[float] = []
    idm_task_successes: list[float] = []
    metrics: dict[str, Any] = {}
    wandb_media: dict[str, Any] = {}

    num_eval = max(0, int(episodes_per_task))
    num_video = max(0, int(video_episodes_per_task))
    total_eps = num_eval + num_video
    if num_eval < 1:
        raise ValueError('episodes_per_task (stat eval) must be >= 1')
    if total_eps <= 0:
        raise ValueError('eval needs num_eval_episodes + num_video_episodes > 0')

    actor_video_by_task: dict[int, np.ndarray] = {}
    idm_video_by_task: dict[int, np.ndarray] = {}

    def _actor_chunk(obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        if bool(subgoal_override_goal):
            pred = np.asarray(goal, dtype=np.float32).reshape(-1)
        else:
            pred = np.asarray(dynamics_agent.infer_subgoal(obs, goal), dtype=np.float32).reshape(-1)
        return np.asarray(actor_agent.sample_actions(obs, pred), dtype=np.float32).reshape(actor_horizon, -1)

    def _idm_chunk(obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        if bool(subgoal_override_goal):
            pred = np.asarray(goal, dtype=np.float32).reshape(-1)
        else:
            pred = np.asarray(dynamics_agent.infer_subgoal(obs, goal), dtype=np.float32).reshape(-1)
        return _idm_action_chunk(dynamics_agent, obs, pred, idm_horizon)

    for task_id in task_ids:
        actor_episode_successes: list[float] = []
        idm_episode_successes: list[float] = []

        for ep_ix in range(total_eps):
            should_render = ep_ix >= num_eval
            count_stats = ep_ix < num_eval
            render_goal = bool(should_render)
            ob, info = env.reset(options=dict(task_id=int(task_id), render_goal=render_goal))
            if 'goal' not in info:
                raise RuntimeError(f'Env reset(task_id={task_id}) did not provide info["goal"].')
            obs = np.asarray(ob, dtype=np.float32).reshape(-1)
            goal = np.asarray(info['goal'], dtype=np.float32).reshape(-1)
            goal_frame = info.get('goal_rendered')
            if goal_frame is not None:
                goal_frame = np.asarray(goal_frame, dtype=np.uint8)

            record_wb = bool(wandb_enabled and should_render and num_video > 0 and ep_ix == num_eval)
            actor_buf: list[np.ndarray] = [] if record_wb else []

            ok_env = rollout_chunked_eval_episode(
                env,
                obs,
                goal,
                low,
                high,
                max_chunks,
                sample_action_chunk=_actor_chunk,
                render_buf=actor_buf if record_wb else None,
                goal_frame=goal_frame,
                should_render=bool(record_wb),
                video_frame_skip=video_frame_skip,
            )
            if count_stats:
                actor_episode_successes.append(1.0 if ok_env else 0.0)
            if record_wb and actor_buf:
                actor_video_by_task[int(task_id)] = np.stack(actor_buf, axis=0)

            ob, info = env.reset(options=dict(task_id=int(task_id), render_goal=render_goal))
            if 'goal' not in info:
                raise RuntimeError(f'Env reset(task_id={task_id}) did not provide info["goal"].')
            obs = np.asarray(ob, dtype=np.float32).reshape(-1)
            goal = np.asarray(info['goal'], dtype=np.float32).reshape(-1)
            goal_frame = info.get('goal_rendered')
            if goal_frame is not None:
                goal_frame = np.asarray(goal_frame, dtype=np.uint8)

            idm_buf: list[np.ndarray] = [] if record_wb else []
            ok_env_i = rollout_chunked_eval_episode(
                env,
                obs,
                goal,
                low,
                high,
                max_chunks,
                sample_action_chunk=_idm_chunk,
                render_buf=idm_buf if record_wb else None,
                goal_frame=goal_frame,
                should_render=bool(record_wb),
                video_frame_skip=video_frame_skip,
            )
            if count_stats:
                idm_episode_successes.append(1.0 if ok_env_i else 0.0)
            if record_wb and idm_buf:
                idm_video_by_task[int(task_id)] = np.stack(idm_buf, axis=0)

        task_success_rate = float(np.mean(actor_episode_successes))
        metrics[f'eval/task_{task_id}/success_rate'] = task_success_rate
        metrics[f'evaluation/task_{task_id}_success'] = task_success_rate
        actor_task_successes.append(task_success_rate)

        idm_task_success_rate = float(np.mean(idm_episode_successes))
        metrics[f'eval_idm/task_{task_id}/success_rate'] = idm_task_success_rate
        metrics[f'evaluation/idm_task_{task_id}_success'] = idm_task_success_rate
        idm_task_successes.append(idm_task_success_rate)

    metrics['eval/success_rate_mean'] = float(np.mean(actor_task_successes))
    metrics['eval_idm/success_rate_mean'] = float(np.mean(idm_task_successes))
    metrics['evaluation/overall_success'] = metrics['eval/success_rate_mean']
    metrics['evaluation/overall_idm_success'] = metrics['eval_idm/success_rate_mean']
    metrics['eval/num_tasks'] = float(len(task_ids))
    metrics['eval/episodes_per_task'] = float(num_eval)
    metrics['eval/video_episodes_per_task'] = float(num_video)
    metrics['eval/total_episodes_per_task'] = float(total_eps)

    if wandb_enabled and num_video > 0 and len(actor_video_by_task) == len(task_ids) == len(idm_video_by_task):
        n_cols = max(1, len(task_ids))
        actor_stack = [actor_video_by_task[int(t)] for t in task_ids]
        idm_stack = [idm_video_by_task[int(t)] for t in task_ids]
        wandb_media['eval/wandb_video_actor'] = get_wandb_video(actor_stack, n_cols=n_cols, fps=int(video_fps))
        wandb_media['eval/wandb_video_idm'] = get_wandb_video(idm_stack, n_cols=n_cols, fps=int(video_fps))

    metrics.update(wandb_media)
    return metrics


def main(_):
    impl = _impl_dir()
    resume_run_dir = FLAGS.resume_run_dir.strip()
    resume_epoch = int(FLAGS.resume_epoch)
    if resume_epoch < 0:
        raise ValueError('resume_epoch must be >= 0.')
    if resume_epoch > 0 and not resume_run_dir:
        raise ValueError('resume_epoch > 0 requires resume_run_dir.')
    restoring_ckpt = bool(resume_run_dir and resume_epoch > 0)

    cfg_path = FLAGS.run_config.strip() or _default_yaml_path()
    resume_snapshot_path: str | None = None
    if (
        resume_run_dir
        and FLAGS.resume_use_run_snapshot_config
        and not _argv_sets_flag('run_config')
    ):
        resume_snapshot_path = _resolve_resume_snapshot_config_path(os.path.abspath(resume_run_dir))
        if resume_snapshot_path is not None:
            cfg_path = resume_snapshot_path
        else:
            print(
                f'[train] WARN resume_use_run_snapshot_config but no flags.json or config_used.yaml '
                f'in {resume_run_dir!r}; using default run_config: {cfg_path}',
                file=sys.stderr,
            )

    dynamics_updates, critic_updates, actor_updates = {}, {}, {}
    if os.path.isfile(cfg_path):
        dynamics_updates, critic_updates, actor_updates = _apply_yaml_to_flags(_load_yaml(cfg_path))
    elif FLAGS.run_config.strip():
        raise FileNotFoundError(f'run_config YAML not found: {cfg_path}')
    else:
        raise FileNotFoundError(f'run_config YAML not found: {cfg_path}')

    dynamics_config, critic_config, actor_config = _prepare_configs(
        dynamics_updates,
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
        run_folder = f'{ts}_seed{FLAGS.seed}_{env_tok}'
        run_dir = os.path.join(runs_root, run_folder)
    ckpt_root = os.path.join(run_dir, 'checkpoints')
    dynamics_ckpt_dir = os.path.join(ckpt_root, 'dynamics')
    critic_ckpt_dir = os.path.join(ckpt_root, 'critic')
    actor_ckpt_dir = os.path.join(ckpt_root, 'actor')
    os.makedirs(dynamics_ckpt_dir, exist_ok=True)
    os.makedirs(critic_ckpt_dir, exist_ok=True)
    os.makedirs(actor_ckpt_dir, exist_ok=True)
    if os.path.isfile(cfg_path) and not resume_run_dir:
        shutil.copy2(cfg_path, os.path.join(run_dir, 'config_used.yaml'))

    exp_name = get_exp_name(FLAGS.seed, env_name=FLAGS.env_name, agent_name='train')
    if FLAGS.use_wandb:
        setup_wandb(project='OGBench', group=FLAGS.run_group, name=exp_name)

    run_logger, run_log_path = _setup_file_logger(run_dir, resume_epoch=resume_epoch if restoring_ckpt else 0)
    run_logger.info('run_dir=%s', run_dir)
    run_logger.info('log_path=%s', run_log_path)
    if resume_snapshot_path is not None:
        run_logger.info('resume hyperparameters from snapshot file: %s', resume_snapshot_path)
    # Force PJRT/CUDA initialisation before MuJoCo/EGL environment creation.
    # In some shells, delaying the first JAX device touch until after env setup
    # can make cuInit fail and silently fall back to CPU.
    jax_devices = jax.devices()
    run_logger.info('jax_backend=%s jax_devices=%s', jax.default_backend(), jax_devices)
    _require_gpu_jax(run_logger)

    env, train_plain, _ = make_env_and_datasets(
        FLAGS.env_name,
        frame_stack=critic_config['frame_stack'],
        dataset_dir=FLAGS.dataset_dir,
        render_mode='rgb_array',
    )
    _attach_state_normalization_stats(dynamics_config, train_plain)
    obs_dim_env = int(np.prod(env.observation_space.shape))
    phi_idxs = normalize_phi_goal_obs_indices(critic_config.get('phi_goal_obs_indices', ()))
    if not phi_idxs:
        phi_idxs = infer_phi_goal_obs_indices(str(FLAGS.env_name), obs_dim_env)
        critic_config['phi_goal_obs_indices'] = phi_idxs
        dynamics_config['phi_goal_obs_indices'] = phi_idxs
    resolved_dyn_goal_cap = _resolve_max_goal_steps_from_env(dynamics_config, env)
    resolved_critic_goal_cap = _resolve_max_goal_steps_from_env(critic_config, env)
    action_dim = int(np.asarray(env.action_space.shape).prod())
    critic_config['action_dim'] = action_dim
    actor_config['action_dim'] = action_dim
    if resolved_dyn_goal_cap or resolved_critic_goal_cap:
        run_logger.info(
            'resolved max_goal_steps from env max_episode_steps=%d (dynamics=%s critic=%s)',
            _env_max_episode_steps(env),
            dynamics_config.get('max_goal_steps', None),
            critic_config.get('max_goal_steps', None),
        )

    with open(os.path.join(run_dir, 'flags.json'), 'w', encoding='utf-8') as f:
        json.dump(
            dict(
                flags=get_flag_dict(),
                dynamics=dynamics_config.to_dict(),
                critic_agent=critic_config.to_dict(),
                actor=actor_config.to_dict(),
            ),
            f,
            indent=2,
        )

    dynamics_dataset = PathHGCDataset(Dataset.create(**train_plain), dynamics_config)
    critic_dataset = _make_critic_dataset(train_plain, critic_config)
    common_valid_starts = _intersect_valid_starts(dynamics_dataset, critic_dataset)
    if int(dynamics_config['dynamics_N']) < int(actor_config['actor_chunk_horizon']):
        raise ValueError(
            f'dynamics_N={int(dynamics_config["dynamics_N"])} must be >= actor_chunk_horizon={int(actor_config["actor_chunk_horizon"])} '
            'for critic-ranked dynamics proposals.'
        )

    np.random.seed(FLAGS.seed)
    ex_idxs = _sample_shared_idxs(common_valid_starts, int(dynamics_config['batch_size']))
    ex_dynamics = dynamics_dataset.sample(len(ex_idxs), idxs=ex_idxs)
    ex_critic = critic_dataset.sample(len(ex_idxs), idxs=ex_idxs)

    dynamics_agent = DynamicsAgent.create(
        FLAGS.seed,
        ex_dynamics['observations'],
        dynamics_config,
        ex_actions=ex_dynamics['actions'],
    )
    critic_agent = _create_critic_agent(FLAGS.seed, ex_critic, critic_config)
    actor_agent = _create_actor_agent(FLAGS.seed, ex_dynamics, actor_config)
    if restoring_ckpt:
        dynamics_agent = restore_agent(dynamics_agent, dynamics_ckpt_dir, resume_epoch)
        critic_agent = restore_agent(critic_agent, critic_ckpt_dir, resume_epoch)
        actor_agent = restore_agent(actor_agent, actor_ckpt_dir, resume_epoch)

    batch_size = int(dynamics_config['batch_size'])
    spe = _steps_per_epoch(len(common_valid_starts), batch_size)
    measure_timing = bool(FLAGS.measure_timing)
    eval_freq = int(FLAGS.eval_freq)
    eval_task_ids = parse_int_list(FLAGS.eval_task_ids)
    eval_episodes_per_task = max(1, int(FLAGS.eval_episodes_per_task))
    final_eval_episodes_per_task = max(0, int(FLAGS.final_eval_episodes_per_task))
    eval_max_chunks = max(1, int(FLAGS.eval_max_chunks))
    run_logger.info(
        'shared_valid_starts=%d batch_size=%d steps_per_epoch=%d dyn_h=%d critic_h=%d actor_h=%d',
        len(common_valid_starts),
        batch_size,
        spe,
        int(dynamics_config['subgoal_steps']),
        int(critic_config.get('full_chunk_horizon', 0)),
        int(actor_config.get('actor_chunk_horizon', 0)),
    )
    run_logger.info(
        'run_setup env=%s seed=%d train_epochs=%d start_epoch=%d save_every=%d async_prefetch=%s action_dim=%d',
        FLAGS.env_name,
        int(FLAGS.seed),
        int(FLAGS.train_epochs),
        int(resume_epoch + 1 if restoring_ckpt else 1),
        int(FLAGS.save_every_n_epochs),
        bool(FLAGS.async_prefetch),
        action_dim,
    )
    run_logger.info(
        'dynamics planner=%s model=%s theta_schedule=%s theta_total=%.4g progress_alpha=%.4g bridge_gamma_inv=%.4g lambda=%.4g beta_min=%.4g beta_max=%.4g',
        str(dynamics_config.get('planner_type', '')),
        str(dynamics_config.get('dynamics_model_type', '')),
        str(dynamics_config.get('theta_schedule', '')),
        float(dynamics_config.get('theta_total', 0.0)),
        float(dynamics_config.get('progress_alpha', 0.0)),
        float(dynamics_config.get('bridge_gamma_inv', 0.0)),
        float(dynamics_config.get('dynamics_lambda', 0.0)),
        float(dynamics_config.get('dynamics_beta_min', 0.0)),
        float(dynamics_config.get('dynamics_beta_max', 0.0)),
    )
    run_logger.info(
        'subgoal mode=%s stochastic_loss=%s target_mode=%s steps=%d samples_U=%d plan_candidates_N=%d total_proposals=%d temperature=%.4g value_alpha=%.4g value_style=%s value_expectile=%.4g value_gap_scale=%.4g value_weight_max=%.4g use_mean_for_actor_goal=%s',
        str(dynamics_config.get('subgoal_distribution', '')),
        str(dynamics_config.get('subgoal_stochastic_loss', 'mse')),
        str(dynamics_config.get('subgoal_target_mode', 'absolute')),
        int(dynamics_config.get('subgoal_steps', 0)),
        int(dynamics_config.get('subgoal_num_samples', 1)),
        int(FLAGS.plan_candidates),
        int(dynamics_config.get('subgoal_num_samples', 1)) * int(FLAGS.plan_candidates),
        float(dynamics_config.get('subgoal_temperature', 0.0)),
        float(dynamics_config.get('subgoal_value_alpha', 0.0)),
        str(dynamics_config.get('subgoal_value_style', 'exponential')),
        float(dynamics_config.get('subgoal_value_expectile', 0.7)),
        float(dynamics_config.get('subgoal_value_gap_scale', 1.0)),
        float(dynamics_config.get('subgoal_value_weight_max', 0.0)),
        bool(dynamics_config.get('subgoal_use_mean_for_actor_goal', True)),
    )
    run_logger.info(
        'planner_sampling plan_noise_scale=%.4g forward_bridge_mode=%s forward_bridge_use_path_loss=%s path_loss_weight=%.4g rollout_horizon=%d rollout_loss_weight=%.4g',
        float(FLAGS.plan_noise_scale),
        str(dynamics_config.get('forward_bridge_mode', '')),
        bool(dynamics_config.get('forward_bridge_use_path_loss', True)),
        float(dynamics_config.get('path_loss_weight', 0.0)),
        int(dynamics_config.get('rollout_horizon', 0)),
        float(dynamics_config.get('rollout_loss_weight', 0.0)),
    )
    run_logger.info(
        'critic_actor type=%s use_chunk_critic=%s critic_chunk_h=%d action_chunk_h=%d spi_tau=%.4g spi_beta=%.4g '
        'q_agg=%s distill=%s kappa_d=%.4g implicit_backup=%s kappa_b=%.4g discount=%.4g',
        str(critic_config.get('critic_type', 'dqc')),
        str(bool(critic_config.get('use_chunk_critic', False))),
        int(critic_config.get('full_chunk_horizon', 0)),
        int(critic_config.get('action_chunk_horizon', 0)),
        float(actor_config.get('spi_tau', 0.0)),
        float(actor_config.get('spi_beta', 0.0)),
        str(critic_config.get('q_agg', '')),
        str(critic_config.get('distill_method', '')),
        float(critic_config.get('kappa_d', 0.0)),
        str(critic_config.get('implicit_backup_type', '')),
        float(critic_config.get('kappa_b', 0.0)),
        float(critic_config.get('discount', 0.0)),
    )
    run_logger.info(
        'eval eval_freq=%d eval_tasks=%s eval_episodes=%d final_eval_episodes=%d eval_max_chunks=%d '
        'video_episodes_per_task=%d video_frame_skip=%d primary_success=any_step_info_success',
        eval_freq,
        ','.join(str(x) for x in eval_task_ids),
        eval_episodes_per_task,
        final_eval_episodes_per_task,
        eval_max_chunks,
        int(FLAGS.eval_video_episodes_per_task),
        int(FLAGS.eval_video_frame_skip),
    )

    train_logger = CsvLogger(os.path.join(run_dir, 'train.csv'), resume=restoring_ckpt, flush_every_n=1)
    first_time = time.time()
    last_log = time.time()

    start_epoch = resume_epoch + 1 if restoring_ckpt else 1
    epoch_iter = range(start_epoch, FLAGS.train_epochs + 1)
    if FLAGS.use_tqdm:
        epoch_iter = tqdm.tqdm(epoch_iter, smoothing=0.1, dynamic_ncols=True)

    # Async batch prefetch: a single worker thread overlaps host-side numpy
    # slicing for batch N+1 with GPU work for batch N. Single-worker
    # ThreadPoolExecutor preserves the order of np.random calls so the
    # sampling sequence stays deterministic given the seed.
    use_async_prefetch = bool(FLAGS.async_prefetch)
    prefetch_pool = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix='train-prefetch')
        if use_async_prefetch
        else None
    )

    def _submit_prefetch() -> Future:
        return prefetch_pool.submit(
            _prepare_train_batch,
            common_valid_starts,
            batch_size,
            dynamics_dataset,
            critic_dataset,
        )

    next_batch_future: Future | None = _submit_prefetch() if prefetch_pool is not None else None

    for epoch in epoch_iter:
        if measure_timing:
            epoch_start = time.perf_counter()
        data_time = 0.0
        build_time = 0.0
        build_detail_times = {}
        dynamics_time = 0.0
        critic_time = 0.0
        actor_rescore_time = 0.0
        actor_time = 0.0
        dynamics_metric_sums = {}
        critic_metric_sums = {}
        actor_metric_sums = {}
        coupling_metric_sums = {}
        last_dynamics_info = None
        last_critic_info = None
        last_actor_info = None
        last_coupling_info = None

        for _ in range(spe):
            if measure_timing:
                t0 = time.perf_counter()
            if next_batch_future is not None:
                dynamics_batch, critic_batch = next_batch_future.result()
                next_batch_future = _submit_prefetch()
            else:
                idxs = _sample_shared_idxs(common_valid_starts, batch_size)
                dynamics_batch = dynamics_dataset.sample(batch_size, idxs=idxs)
                critic_batch = critic_dataset.sample(batch_size, idxs=idxs)
            if measure_timing:
                data_time += time.perf_counter() - t0

            if measure_timing:
                t0 = time.perf_counter()
            dynamics_agent, critic_batch, actor_batch, coupling_info, build_detail_info = _build_train_batches(
                dynamics_agent,
                critic_agent,
                dynamics_batch,
                critic_batch,
                actor_config,
            )
            if measure_timing:
                _block_until_ready((critic_batch, actor_batch))
                build_time += time.perf_counter() - t0
                _accumulate_time_sums(build_detail_times, build_detail_info)

            if measure_timing:
                t0 = time.perf_counter()
            dynamics_agent, dynamics_info = dynamics_agent.update(
                dynamics_batch,
                critic_value_params=_extract_critic_value_params(critic_agent),
            )
            if measure_timing:
                _block_until_ready(dynamics_info)
                dynamics_time += time.perf_counter() - t0

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

            last_dynamics_info = dynamics_info
            last_critic_info = critic_info
            last_actor_info = actor_info
            last_coupling_info = coupling_info

            _accumulate_metric_sums(dynamics_metric_sums, dynamics_info)
            _accumulate_metric_sums(critic_metric_sums, critic_info)
            _accumulate_metric_sums(actor_metric_sums, actor_info)
            _accumulate_metric_sums(coupling_metric_sums, coupling_info)

        gstep = epoch * spe
        steps_done = spe
        if epoch % FLAGS.log_every_n_epochs == 0 and last_dynamics_info is not None:
            metrics = {}
            metrics.update(_to_host_metrics('train/dynamics', last_dynamics_info))
            metrics.update(_to_host_metrics('train/critic', last_critic_info))
            metrics.update(_to_host_metrics('train/actor', last_actor_info))
            metrics.update(_to_host_metrics('train/coupling', last_coupling_info))
            metrics['train/critic/primary_score'] = extract_critic_primary_score(last_critic_info)
            _emit_metric_means(metrics, 'train/dynamics', dynamics_metric_sums, steps_done)
            _emit_metric_means(metrics, 'train/critic', critic_metric_sums, steps_done)
            _emit_metric_means(metrics, 'train/actor', actor_metric_sums, steps_done)
            _emit_metric_means(metrics, 'train/coupling', coupling_metric_sums, steps_done)
            metrics['train/epoch'] = float(epoch)
            if eval_freq > 0 and epoch % eval_freq == 0:
                eval_episode_count = (
                    final_eval_episodes_per_task
                    if final_eval_episodes_per_task > 0 and epoch == int(FLAGS.train_epochs)
                    else eval_episodes_per_task
                )
                metrics.update(
                    _evaluate_env_tasks(
                        env,
                        dynamics_agent,
                        actor_agent,
                        actor_config,
                        critic_config,
                        task_ids=eval_task_ids,
                        episodes_per_task=eval_episode_count,
                        max_chunks=eval_max_chunks,
                        video_episodes_per_task=int(FLAGS.eval_video_episodes_per_task),
                        video_frame_skip=int(FLAGS.eval_video_frame_skip),
                        video_fps=int(FLAGS.eval_video_fps),
                        wandb_enabled=bool(FLAGS.use_wandb),
                        subgoal_override_goal=bool(FLAGS.subgoal_override_goal),
                    )
                )
            if measure_timing:
                metrics['time/data_epoch_sec'] = data_time
                metrics['time/build_batches_epoch_sec'] = build_time
                _emit_time_sums(metrics, 'time/build', build_detail_times, spe)
                metrics['time/dynamics_update_epoch_sec'] = dynamics_time
                metrics['time/critic_update_epoch_sec'] = critic_time
                metrics['time/actor_rescore_epoch_sec'] = actor_rescore_time
                metrics['time/actor_update_epoch_sec'] = actor_time
                metrics['time/epoch_compute_sec'] = time.perf_counter() - epoch_start
                metrics['time/data_step_sec'] = data_time / spe
                metrics['time/build_batches_step_sec'] = build_time / spe
                metrics['time/dynamics_update_step_sec'] = dynamics_time / spe
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
                video_eps = int(metrics.get('eval/video_episodes_per_task', 0.0))
                run_logger.info(
                    '=== EVAL START epoch=%d num_tasks=%d stat_episodes_per_task=%d video_episodes_per_task=%d ===',
                    epoch,
                    num_tasks,
                    episodes_per_task,
                    video_eps,
                )
                run_logger.info('[IDM POLICY] primary_success=any_step_info_success')
                run_logger.info(
                    'idm env_success_rate_mean=%.2f',
                    metrics.get('eval_idm/success_rate_mean', float('nan')),
                )
                for task_id in eval_task_ids:
                    task_key = f'eval_idm/task_{task_id}/success_rate'
                    if task_key in metrics:
                        run_logger.info('idm task_%d env=%.2f', task_id, metrics[task_key])
                run_logger.info('[ACTOR POLICY] (same success definition)')
                run_logger.info(
                    'actor env_success_rate_mean=%.2f',
                    metrics.get('eval/success_rate_mean', float('nan')),
                )
                for task_id in eval_task_ids:
                    task_key = f'eval/task_{task_id}/success_rate'
                    if task_key in metrics:
                        run_logger.info('actor task_%d env=%.2f', task_id, metrics[task_key])
                run_logger.info('=== EVAL END epoch=%d ===', epoch)

        if epoch % FLAGS.save_every_n_epochs == 0:
            save_agent(dynamics_agent, dynamics_ckpt_dir, epoch)
            save_agent(critic_agent, critic_ckpt_dir, epoch)
            save_agent(actor_agent, actor_ckpt_dir, epoch)

    if prefetch_pool is not None:
        # Cancel any outstanding prefetch and tear down the worker thread.
        if next_batch_future is not None and not next_batch_future.done():
            next_batch_future.cancel()
        prefetch_pool.shutdown(wait=False, cancel_futures=True)

    train_logger.close()
    run_logger.info('done run_dir=%s', run_dir)


if __name__ == '__main__':
    app.run(main)
