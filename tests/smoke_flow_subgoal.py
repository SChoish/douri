"""Standalone smoke checks for DynamicsAgent subgoal distribution modes.

This complements the pytest coverage in ``tests/test_distributional_subgoal.py``
and avoids any OGBench dependency:

    PYTHONPATH=. python tests/smoke_flow_subgoal.py
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp

from agents.dynamics import DynamicsAgent, get_dynamics_config


def _config(mode: str):
    cfg = get_dynamics_config()
    cfg.batch_size = 4
    cfg.dynamics_N = 3
    cfg.subgoal_steps = 3
    cfg.subgoal_hidden_dims = (16, 16)
    cfg.path_residual_hidden_dims = (16, 16)
    cfg.residual_model_hidden_dims = (16, 16)
    cfg.idm_hidden_dims = (16, 16)
    cfg.subgoal_value_hidden_dims = (16, 16)
    cfg.layer_norm = False
    cfg.subgoal_value_layer_norm = False
    cfg.subgoal_goal_representation = 'full'
    cfg.subgoal_value_goal_representation = 'full'
    cfg.subgoal_distribution = mode
    cfg.subgoal_num_samples = 3
    cfg.subgoal_flow_steps = 2
    cfg.subgoal_flow_use_value_bonus = False
    cfg.subgoal_flow_noise_scale = 1.0
    cfg.subgoal_value_alpha = 0.0
    cfg.subgoal_target_mode = 'displacement'
    cfg.residual_target_mode = 'displacement'
    return cfg


def _batch(batch_size: int, state_dim: int, action_dim: int, horizon: int):
    observations = jnp.linspace(-1.0, 1.0, batch_size * state_dim, dtype=jnp.float32).reshape(batch_size, state_dim)
    next_observations = observations + 0.05
    high_actor_goals = observations + 0.5
    high_actor_targets = observations + 0.25
    actions = jnp.ones((batch_size, action_dim), dtype=jnp.float32) * 0.1
    steps = jnp.linspace(0.0, 1.0, horizon + 1, dtype=jnp.float32)
    trajectory_segment = observations[:, None, :] + steps[None, :, None] * 0.25
    return {
        'observations': observations,
        'next_observations': next_observations,
        'actions': actions,
        'high_actor_goals': high_actor_goals,
        'high_actor_targets': high_actor_targets,
        'trajectory_segment': trajectory_segment,
    }


def _create(mode: str):
    batch_size = 4
    state_dim = 5
    action_dim = 2
    ex_observations = jnp.zeros((batch_size, state_dim), dtype=jnp.float32)
    ex_actions = jnp.zeros((batch_size, action_dim), dtype=jnp.float32)
    return DynamicsAgent.create(0, ex_observations, _config(mode), ex_actions=ex_actions)


def main() -> None:
    for mode in ('deterministic', 'diag_gaussian', 'flow'):
        _create(mode)
        print(f'created {mode}')

    agent = _create('flow')
    batch = _batch(batch_size=4, state_dim=5, action_dim=2, horizon=3)
    pred = agent.predict_subgoal(batch['observations'], batch['high_actor_goals'])
    assert pred.shape == (4, 5), pred.shape
    candidates, mu = agent.sample_subgoal_candidates(
        batch['observations'],
        batch['high_actor_goals'],
        jax.random.PRNGKey(123),
        num_candidates=3,
        include_mean=True,
    )
    assert candidates.shape == (4, 3, 5), candidates.shape
    assert mu.shape == (4, 5), mu.shape
    agent, info = agent.update(batch)
    assert bool(jnp.isfinite(info['phase1/loss'])), info['phase1/loss']
    assert bool(jnp.isfinite(info['phase1/subgoal_flow_loss'])), info['phase1/subgoal_flow_loss']
    print('flow smoke passed')


if __name__ == '__main__':
    main()
