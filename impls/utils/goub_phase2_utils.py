from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict

from agents.goub_phase1 import GOUBPhase1Agent, get_config as get_phase1_config
from agents.goub_phase1_path import GOUBPhase1PathAgent, get_config as get_phase1_path_config
from utils.flax_utils import merge_checkpoint_state_dict
from utils.inverse_dynamics_train import InverseDynamicsMLP


@dataclass(frozen=True)
class FrozenGOUBBundle:
    phase1_agent: Any
    idm_model: InverseDynamicsMLP
    idm_params: Any
    env_name: str
    phase1_agent_name: str
    checkpoint_epoch: int
    checkpoint_path: Path


def list_checkpoint_suffixes(checkpoints_dir: Path) -> list[int]:
    out = []
    for p in checkpoints_dir.glob('params_*.pkl'):
        m = re.search(r'params_(\d+)\.pkl$', p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def resolve_checkpoint_epoch(checkpoints_dir: Path, requested_epoch: int) -> int:
    suffixes = list_checkpoint_suffixes(checkpoints_dir)
    if not suffixes:
        raise FileNotFoundError(f'No params_*.pkl in {checkpoints_dir}')
    if int(requested_epoch) < 0:
        return suffixes[-1]
    if int(requested_epoch) in suffixes:
        return int(requested_epoch)
    nearest = min(suffixes, key=lambda x: abs(x - int(requested_epoch)))
    return int(nearest)


def load_phase1_run_flags(run_dir: str | Path) -> tuple[ConfigDict, str, str]:
    run_dir = Path(run_dir).resolve()
    flags_path = run_dir / 'flags.json'
    if not flags_path.is_file():
        raise FileNotFoundError(f'Missing flags.json under {run_dir}')
    with open(flags_path, 'r', encoding='utf-8') as f:
        flags = json.load(f)

    env_name = str(flags.get('env_name') or '').strip()
    if not env_name:
        raise KeyError('phase1 flags.json must contain env_name')

    agent_flags = flags.get('agent') or {}
    phase1_agent_name = str(agent_flags.get('agent_name') or 'goub_phase1').strip()
    if phase1_agent_name == 'goub_phase1_path':
        cfg = get_phase1_path_config()
    elif phase1_agent_name == 'goub_phase1':
        cfg = get_phase1_config()
    else:
        raise ValueError(
            f'Unsupported frozen phase1 agent_name={phase1_agent_name!r}. '
            "Expected 'goub_phase1' or 'goub_phase1_path'."
        )

    for k, v in agent_flags.items():
        cfg[k] = v
    return cfg, env_name, phase1_agent_name


def load_checkpoint_into_agent(agent, checkpoint_path: str | Path):
    checkpoint_path = Path(checkpoint_path).resolve()
    with open(checkpoint_path, 'rb') as f:
        load_dict = pickle.load(f)
    template = flax.serialization.to_state_dict(agent)
    merged = merge_checkpoint_state_dict(template, load_dict['agent'])
    return flax.serialization.from_state_dict(agent, merged)


def _extract_embedded_idm(phase1_agent, obs_dim: int):
    ptree = phase1_agent.network.params
    if 'idm_net' in ptree:
        idm_params = ptree['idm_net']
    elif 'modules_idm_net' in ptree:
        idm_params = ptree['modules_idm_net']
    else:
        return None, None

    idm_model = InverseDynamicsMLP(
        obs_dim=int(obs_dim),
        action_dim=int(phase1_agent.config['idm_action_dim']),
        hidden_dims=tuple(int(x) for x in phase1_agent.config['idm_hidden_dims']),
    )
    return idm_model, idm_params


def _load_standalone_idm(idm_checkpoint: str | Path, obs_dim: int):
    idm_checkpoint = Path(idm_checkpoint).resolve()
    if not idm_checkpoint.is_file():
        raise FileNotFoundError(f'Standalone IDM checkpoint not found: {idm_checkpoint}')
    with open(idm_checkpoint, 'rb') as f:
        idm_ckpt = pickle.load(f)

    ckpt_obs_dim = int(idm_ckpt['obs_dim'])
    if ckpt_obs_dim != int(obs_dim):
        raise ValueError(f'IDM obs_dim mismatch: checkpoint={ckpt_obs_dim}, expected={int(obs_dim)}')

    idm_model = InverseDynamicsMLP(
        obs_dim=ckpt_obs_dim,
        action_dim=int(idm_ckpt['action_dim']),
        hidden_dims=tuple(int(x) for x in idm_ckpt['hidden_dims']),
    )
    return idm_model, idm_ckpt['params']


def load_frozen_goub_bundle(
    *,
    phase1_run_dir: str | Path,
    phase1_checkpoint_epoch: int,
    example_observations: np.ndarray,
    example_actions: np.ndarray,
    seed: int = 0,
    idm_checkpoint: str = '',
) -> FrozenGOUBBundle:
    run_dir = Path(phase1_run_dir).resolve()
    checkpoints_dir = run_dir / 'checkpoints'
    if not checkpoints_dir.is_dir():
        raise FileNotFoundError(f'No checkpoints/ under {run_dir}')

    phase1_cfg, env_name, phase1_agent_name = load_phase1_run_flags(run_dir)
    checkpoint_epoch = resolve_checkpoint_epoch(checkpoints_dir, int(phase1_checkpoint_epoch))
    checkpoint_path = checkpoints_dir / f'params_{checkpoint_epoch}.pkl'

    ex_obs = jnp.asarray(example_observations, dtype=jnp.float32)
    ex_act = jnp.asarray(example_actions, dtype=jnp.float32)

    if phase1_agent_name == 'goub_phase1_path':
        phase1_agent = GOUBPhase1PathAgent.create(seed, ex_obs, phase1_cfg, ex_actions=ex_act)
    else:
        phase1_agent = GOUBPhase1Agent.create(seed, ex_obs, phase1_cfg, ex_actions=ex_act)
    phase1_agent = load_checkpoint_into_agent(phase1_agent, checkpoint_path)

    idm_ckpt_str = str(idm_checkpoint).strip()
    if idm_ckpt_str:
        idm_model, idm_params = _load_standalone_idm(idm_ckpt_str, ex_obs.shape[-1])
    else:
        idm_model, idm_params = _extract_embedded_idm(phase1_agent, ex_obs.shape[-1])
        if idm_model is None:
            raise FileNotFoundError(
                'Frozen phase1 checkpoint does not contain embedded idm_net. '
                'Pass top-level idm_checkpoint to use a standalone IDM pickle.'
            )

    return FrozenGOUBBundle(
        phase1_agent=phase1_agent,
        idm_model=idm_model,
        idm_params=idm_params,
        env_name=env_name,
        phase1_agent_name=phase1_agent_name,
        checkpoint_epoch=checkpoint_epoch,
        checkpoint_path=checkpoint_path,
    )


class FrozenGOUBProposalGenerator:
    """Generate action proposals from frozen GOUB phase1 + frozen IDM."""

    def __init__(
        self,
        frozen_bundle: FrozenGOUBBundle,
        *,
        action_low: np.ndarray,
        action_high: np.ndarray,
        num_action_samples: int,
        action_noise_std: float,
        include_mean_action: bool,
        include_dataset_action: bool,
        planner_noise_scale: float,
        num_planner_samples: int,
    ):
        self.bundle = frozen_bundle
        self.action_low = jnp.asarray(action_low, dtype=jnp.float32).reshape(-1)
        self.action_high = jnp.asarray(action_high, dtype=jnp.float32).reshape(-1)
        self.num_action_samples = int(num_action_samples)
        self.action_noise_std = float(action_noise_std)
        self.include_mean_action = bool(include_mean_action)
        self.include_dataset_action = bool(include_dataset_action)
        self.planner_noise_scale = float(planner_noise_scale)
        self.num_planner_samples = int(num_planner_samples)
        self._idm_apply = jax.jit(lambda p, o, on: self.bundle.idm_model.apply({'params': p}, o, on))

    def build(self, batch: dict[str, np.ndarray], seed: int) -> dict[str, jnp.ndarray]:
        observations = jnp.asarray(batch['observations'], dtype=jnp.float32)
        goals = jnp.asarray(batch['value_goals'], dtype=jnp.float32)
        dataset_actions = jnp.asarray(batch['actions'], dtype=jnp.float32)

        plan_rng = jax.random.PRNGKey(int(seed))
        subgoals = self.bundle.phase1_agent.predict_subgoal(observations, goals)

        # Deterministic bridge step (anchor for IDM mean, logging, action-noise center).
        plan_det = self.bundle.phase1_agent.plan(observations, subgoals)
        planned_next_det = jnp.asarray(plan_det['next_step'], dtype=jnp.float32)
        a_mean = self._idm_apply(self.bundle.idm_params, observations, planned_next_det)
        a_mean = jnp.clip(a_mean, self.action_low, self.action_high)
        planned_next_obs = planned_next_det

        candidates = []
        if self.planner_noise_scale > 0.0:
            if self.num_planner_samples < 1:
                raise ValueError(
                    'num_planner_samples must be >= 1 when planner_noise_scale > 0 (stochastic planner sampling).'
                )
            if self.include_mean_action:
                candidates.append(a_mean[:, None, :])
            stoch_actions = []
            for i in range(self.num_planner_samples):
                rng_i = jax.random.fold_in(plan_rng, i)
                plan_stoch = self.bundle.phase1_agent.sample_plan(
                    observations,
                    subgoals,
                    rng_i,
                    noise_scale=self.planner_noise_scale,
                )
                next_i = jnp.asarray(plan_stoch['next_step'], dtype=jnp.float32)
                a_i = self._idm_apply(self.bundle.idm_params, observations, next_i)
                a_i = jnp.clip(a_i, self.action_low, self.action_high)
                stoch_actions.append(a_i)
            candidates.append(jnp.stack(stoch_actions, axis=1))
        elif self.include_mean_action:
            candidates.append(a_mean[:, None, :])

        if self.num_action_samples > 0:
            noise_rng = jax.random.fold_in(plan_rng, 10_000)
            noise = jax.random.normal(
                noise_rng,
                (observations.shape[0], self.num_action_samples, a_mean.shape[-1]),
                dtype=jnp.float32,
            )
            noisy_actions = a_mean[:, None, :] + self.action_noise_std * noise
            noisy_actions = jnp.clip(noisy_actions, self.action_low, self.action_high)
            candidates.append(noisy_actions)

        if self.include_dataset_action:
            candidates.append(jnp.clip(dataset_actions, self.action_low, self.action_high)[:, None, :])

        if not candidates:
            raise ValueError(
                'Proposal generator has zero candidates. Enable include_mean_action, include_dataset_action, '
                'or set num_action_samples > 0.'
            )

        candidate_actions = jnp.concatenate(candidates, axis=1)
        return {
            'candidate_actions': candidate_actions,
            'a_mean': a_mean,
            'planned_next_obs': planned_next_obs,
            'subgoals': jnp.asarray(subgoals, dtype=jnp.float32),
        }
