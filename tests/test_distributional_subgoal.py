"""Unit tests for the distributional-subgoal + linear dynamics refactor.

These tests are intentionally lightweight (no offline dataset, no actor / critic
training) so they can be run standalone with::

    PYTHONPATH=. python -m pytest tests/test_distributional_subgoal.py
    PYTHONPATH=. python tests/test_distributional_subgoal.py    # also works

They cover the contract changes only:
1. backward compatibility of the deterministic subgoal mode
2. linear dynamics schedule exposes gamma_inv and bridge arrays
3. distributional-subgoal sampling shape correctness
4. critic ``score_action_chunks`` accepts both ``[B, D]`` and ``[B, N, D]`` goals
5. ``plan_candidates=1`` and ``plan_candidates>1`` both succeed
6. distributional subgoal loss path is finite (no NaNs / Infs)
7. dynamics-config defaults remain usable
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import jax
import jax.numpy as jnp

from utils.goub import make_goub_schedule, bridge_sample, model_mean
from agents.goub_dynamics import (
    GOUBDynamicsAgent,
    get_dynamics_config,
)
from agents.critic import DQCCriticAgent, get_config as get_critic_config


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

STATE_DIM = 4
ACTION_DIM = 2
BATCH = 8


def _make_dynamics_agent(subgoal_distribution: str, bridge_gamma_inv: float = 0.0):
    cfg = get_dynamics_config()
    cfg.goub_N = 4
    cfg.subgoal_steps = 4
    cfg.rollout_horizon = 2
    cfg.subgoal_distribution = subgoal_distribution
    cfg.bridge_gamma_inv = bridge_gamma_inv
    cfg.eps_hidden_dims = (32, 32)
    cfg.subgoal_hidden_dims = (32, 32)
    cfg.subgoal_value_hidden_dims = (32, 32)
    cfg.idm_hidden_dims = (32, 32)
    ex_obs = np.zeros((BATCH, STATE_DIM), dtype=np.float32)
    ex_act = np.zeros((BATCH, ACTION_DIM), dtype=np.float32)
    return GOUBDynamicsAgent.create(seed=0, ex_observations=ex_obs, ex_actions=ex_act, config=cfg)


def _make_critic_agent():
    cfg = get_critic_config()
    cfg.action_chunk_horizon = 2
    cfg.full_chunk_horizon = 4
    cfg.value_hidden_dims = (32, 32)
    cfg.action_dim = ACTION_DIM
    ex_obs = np.zeros((BATCH, STATE_DIM), dtype=np.float32)
    ex_full = np.zeros((BATCH, cfg.full_chunk_horizon * ACTION_DIM), dtype=np.float32)
    ex_part = np.zeros((BATCH, cfg.action_chunk_horizon * ACTION_DIM), dtype=np.float32)
    return DQCCriticAgent.create(
        seed=0,
        ex_observations=ex_obs,
        ex_full_chunk_actions=ex_full,
        ex_action_chunk_actions=ex_part,
        config=cfg,
        ex_goals=ex_obs,
    )


# ---------------------------------------------------------------------------
# 1. backward-compatible deterministic mode
# ---------------------------------------------------------------------------

def test_deterministic_subgoal_backward_compat():
    agent = _make_dynamics_agent('deterministic')
    obs = jnp.zeros((BATCH, STATE_DIM), dtype=jnp.float32)
    g = jnp.zeros((BATCH, STATE_DIM), dtype=jnp.float32)
    sg = agent.predict_subgoal(obs, g)
    assert sg.shape == (BATCH, STATE_DIM), sg.shape
    # infer_subgoal must remain a backward-compatible alias.
    sg2 = agent.infer_subgoal(obs, g)
    assert jnp.allclose(sg, sg2)
    mu, log_std = agent.infer_subgoal_distribution(obs, g)
    # In deterministic mode the distribution helper must still return a mean;
    # log_std is filled with the configured floor.
    assert mu.shape == (BATCH, STATE_DIM)
    assert log_std.shape == (BATCH, STATE_DIM)
    assert jnp.allclose(mu, sg)


# ---------------------------------------------------------------------------
# 2. linear dynamics schedule
# ---------------------------------------------------------------------------

def test_linear_dynamics_schedule_and_model_mean_are_finite():
    schedule = make_goub_schedule(N=8, beta_min=0.1, beta_max=20.0, lambda_=1.0, bridge_gamma_inv=0.0)
    assert schedule['bridge_w'].shape == (9,)
    assert schedule['bridge_var'].shape == (9,)
    assert 'dynamics_phi_iK' in schedule
    assert 'dynamics_omega_iK' in schedule
    assert float(schedule['gamma_inv']) == 0.0

    rng = jax.random.PRNGKey(0)
    x0 = jax.random.normal(rng, (BATCH, STATE_DIM))
    xT = jax.random.normal(jax.random.fold_in(rng, 1), (BATCH, STATE_DIM))
    n = jnp.full((BATCH,), 3, dtype=jnp.int32)
    x_n = bridge_sample(x0, xT, n, schedule, rng)
    eps = jax.random.normal(jax.random.fold_in(rng, 2), (BATCH, STATE_DIM))
    mu = model_mean(x_n, x0, xT, eps, n, schedule)
    assert np.all(np.isfinite(np.asarray(x_n)))
    assert np.all(np.isfinite(np.asarray(mu)))


def test_linear_dynamics_resolves_gamma_inv_correctly():
    # The schedule must thread bridge_gamma_inv into a `gamma_inv` entry so
    # downstream agents can query the exact configured denominator offset.
    s_soft = make_goub_schedule(N=8, bridge_gamma_inv=0.5)
    assert abs(float(s_soft['gamma_inv']) - 0.5) < 1e-6
    assert 'dynamics_bridge_w' in s_soft
    assert 'dynamics_bridge_var' in s_soft

    # Negative inverse gamma must raise.
    raised = False
    try:
        make_goub_schedule(N=4, bridge_gamma_inv=-1.0)
    except ValueError:
        raised = True
    assert raised


# ---------------------------------------------------------------------------
# 3. distributional subgoal sampling shape correctness
# ---------------------------------------------------------------------------

def test_distributional_subgoal_sampling_shapes():
    agent = _make_dynamics_agent('diag_gaussian')
    obs = jnp.zeros((BATCH, STATE_DIM), dtype=jnp.float32)
    g = jnp.zeros((BATCH, STATE_DIM), dtype=jnp.float32)
    mu, log_std = agent.infer_subgoal_distribution(obs, g)
    assert mu.shape == (BATCH, STATE_DIM)
    assert log_std.shape == (BATCH, STATE_DIM)
    # sample_subgoal_candidates returns ([B, N, D], [B, D])
    cand, mu2 = agent.sample_subgoal_candidates(
        obs, g, jax.random.PRNGKey(0), num_candidates=5, include_mean=True,
    )
    assert cand.shape == (BATCH, 5, STATE_DIM)
    assert mu2.shape == (BATCH, STATE_DIM)
    # In diag_gaussian mode candidate 0 is pinned to the mean when include_mean=True.
    np.testing.assert_allclose(np.asarray(cand[:, 0, :]), np.asarray(mu2), rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# 4. critic accepts per-candidate goals [B, N, D]
# ---------------------------------------------------------------------------

def test_critic_score_action_chunks_with_per_candidate_goals():
    critic = _make_critic_agent()
    n_cand = 4
    ha = int(critic.config['action_chunk_horizon'])
    obs = jnp.zeros((BATCH, STATE_DIM), dtype=jnp.float32)
    # Shape required by score_action_chunks for candidates: [B, N, ha, A].
    chunks = jnp.zeros((BATCH, n_cand, ha, ACTION_DIM), dtype=jnp.float32)

    shared_goals = jnp.zeros((BATCH, STATE_DIM), dtype=jnp.float32)
    per_cand_goals = jnp.zeros((BATCH, n_cand, STATE_DIM), dtype=jnp.float32)

    s_shared = critic.score_action_chunks(obs, shared_goals, chunks, use_partial_critic=True)
    s_per = critic.score_action_chunks(obs, per_cand_goals, chunks, use_partial_critic=True)
    assert s_shared.shape == (BATCH, n_cand)
    assert s_per.shape == (BATCH, n_cand)
    # With identical goals they should match numerically.
    np.testing.assert_allclose(np.asarray(s_shared), np.asarray(s_per), rtol=1e-5, atol=1e-6)

    # Mismatched candidate count must raise.
    bad_goals = jnp.zeros((BATCH, n_cand + 1, STATE_DIM), dtype=jnp.float32)
    raised = False
    try:
        critic.score_action_chunks(obs, bad_goals, chunks, use_partial_critic=True)
    except Exception:
        raised = True
    assert raised, 'expected per-candidate goal/chunk mismatch to raise'


# ---------------------------------------------------------------------------
# 5. plan_candidates=1 and >1 both succeed (deterministic + diag_gaussian)
# ---------------------------------------------------------------------------

def _check_build_actor_proposals(agent, plan_candidates: int):
    obs = jnp.zeros((BATCH, STATE_DIM), dtype=jnp.float32)
    g = jnp.zeros((BATCH, STATE_DIM), dtype=jnp.float32)
    mu, cand_actions, cand_goals, _ = agent.build_actor_proposals(
        obs, g, jax.random.PRNGKey(0),
        proposal_horizon=2, plan_candidates=plan_candidates, sample_noise_scale=0.0,
    )
    assert mu.shape == (BATCH, STATE_DIM)
    assert cand_actions.shape == (BATCH, plan_candidates, 2, ACTION_DIM)
    assert cand_goals.shape == (BATCH, plan_candidates, STATE_DIM)


def test_build_actor_proposals_deterministic_n1_n_gt_1():
    agent = _make_dynamics_agent('deterministic')
    _check_build_actor_proposals(agent, plan_candidates=1)
    _check_build_actor_proposals(agent, plan_candidates=4)


def test_build_actor_proposals_diag_gaussian_n1_n_gt_1():
    agent = _make_dynamics_agent('diag_gaussian')
    _check_build_actor_proposals(agent, plan_candidates=1)
    _check_build_actor_proposals(agent, plan_candidates=4)


# ---------------------------------------------------------------------------
# 6. distributional subgoal loss has no NaN / Inf
# ---------------------------------------------------------------------------

def _make_phase1_batch():
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32)
    target = rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32)
    # 'trajectory_segment' must have N+1 = 5 states.
    seg = rng.standard_normal((BATCH, 5, STATE_DIM)).astype(np.float32)
    actions = rng.standard_normal((BATCH, ACTION_DIM)).astype(np.float32)
    next_obs = rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32)
    return {
        'observations': jnp.asarray(obs),
        'next_observations': jnp.asarray(next_obs),
        'high_actor_goals': jnp.asarray(obs),
        'high_actor_targets': jnp.asarray(target),
        'trajectory_segment': jnp.asarray(seg),
        'actions': jnp.asarray(actions),
    }


def test_distributional_subgoal_loss_is_finite():
    agent = _make_dynamics_agent('diag_gaussian')
    batch = _make_phase1_batch()
    new_agent, info = agent.update(batch, critic_value_params=None)
    for k, v in info.items():
        v = np.asarray(v)
        assert np.all(np.isfinite(v)), f'non-finite log value at {k}: {v}'
    # Required new keys are present.
    for required in (
        'phase1/subgoal_nll',
        'phase1/subgoal_std_mean',
        'phase1/subgoal_std_max',
        'phase1/subgoal_fr_spi',
        'phase1/subgoal_mode',
        'dynamics/bridge_gamma_inv',
        'dynamics/gamma_inv',
    ):
        assert required in info, f'missing log key {required}'
    # subgoal_mode == 1.0 in diag_gaussian mode.
    assert float(info['phase1/subgoal_mode']) == 1.0


def test_deterministic_subgoal_loss_is_finite_and_logs_match_legacy():
    agent = _make_dynamics_agent('deterministic')
    batch = _make_phase1_batch()
    _, info = agent.update(batch, critic_value_params=None)
    for k, v in info.items():
        assert np.all(np.isfinite(np.asarray(v))), f'non-finite log value at {k}: {v}'
    # In deterministic mode the new metrics should be zero placeholders.
    assert float(info['phase1/subgoal_mode']) == 0.0
    assert float(info['phase1/subgoal_nll']) == 0.0


# ---------------------------------------------------------------------------
# 7. dynamics-config defaults remain backward-compatible
# ---------------------------------------------------------------------------

def test_dynamics_config_defaults_are_usable():
    cfg = get_dynamics_config()
    assert str(cfg.subgoal_distribution) == 'deterministic'
    assert bool(cfg.subgoal_use_mean_for_actor_goal) is True
    assert float(cfg.subgoal_fr_spi_weight) == 0.0


if __name__ == '__main__':
    failures = []
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f'  PASS  {name}')
            except Exception as e:  # pragma: no cover
                failures.append((name, e))
                print(f'  FAIL  {name}: {type(e).__name__}: {e}')
    if failures:
        sys.exit(1)
    print('\nAll tests passed.')
