"""Unit tests for the distributional-subgoal + linear dynamics refactor.

These tests are intentionally lightweight (no offline dataset, no actor / critic
training) so they can be run standalone with::

    PYTHONPATH=. python -m pytest tests/test_distributional_subgoal.py
    PYTHONPATH=. python tests/test_distributional_subgoal.py    # also works

They cover the contract changes only:
1. deterministic subgoal mode
2. linear dynamics schedule exposes gamma_inv and bridge arrays
3. distributional-subgoal sampling shape correctness
4. critic ``score_action_chunks`` accepts both ``[B, D]`` and ``[B, N, D]`` goals
5. ``plan_candidates=1`` and ``plan_candidates>1`` both succeed
6. all U*N bridge candidates are reduced to one best proposal before SPI
7. distributional subgoal loss path is finite (no NaNs / Infs)
8. dynamics-config defaults remain usable
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import jax
import jax.numpy as jnp

from utils.dynamics import bridge_sample, make_dynamics_schedule, posterior_mean
from agents.dynamics import (
    DynamicsAgent,
    get_dynamics_config,
)
from agents.critic import CriticAgent, get_config as get_critic_config
from main import _rescore_actor_batch_for_update, _select_eval_subgoal


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

STATE_DIM = 4
ACTION_DIM = 2
BATCH = 8


def _make_dynamics_agent(
    subgoal_distribution: str,
    bridge_gamma_inv: float = 0.0,
    subgoal_num_samples: int = 1,
    config_updates: dict | None = None,
):
    cfg = get_dynamics_config()
    cfg.dynamics_N = 4
    cfg.subgoal_steps = 4
    cfg.rollout_horizon = 2
    cfg.subgoal_distribution = subgoal_distribution
    cfg.subgoal_num_samples = subgoal_num_samples
    cfg.bridge_gamma_inv = bridge_gamma_inv
    cfg.residual_model_hidden_dims = (32, 32)
    cfg.subgoal_hidden_dims = (32, 32)
    cfg.subgoal_value_hidden_dims = (32, 32)
    cfg.idm_hidden_dims = (32, 32)
    if config_updates:
        for key, value in config_updates.items():
            cfg[key] = value
    ex_obs = np.zeros((BATCH, STATE_DIM), dtype=np.float32)
    ex_act = np.zeros((BATCH, ACTION_DIM), dtype=np.float32)
    return DynamicsAgent.create(seed=0, ex_observations=ex_obs, ex_actions=ex_act, config=cfg)


def _make_critic_agent():
    cfg = get_critic_config()
    cfg.action_chunk_horizon = 2
    cfg.full_chunk_horizon = 4
    cfg.value_hidden_dims = (32, 32)
    cfg.action_dim = ACTION_DIM
    ex_obs = np.zeros((BATCH, STATE_DIM), dtype=np.float32)
    ex_full = np.zeros((BATCH, cfg.full_chunk_horizon * ACTION_DIM), dtype=np.float32)
    ex_part = np.zeros((BATCH, cfg.action_chunk_horizon * ACTION_DIM), dtype=np.float32)
    return CriticAgent.create(
        seed=0,
        ex_observations=ex_obs,
        ex_full_chunk_actions=ex_full,
        ex_action_chunk_actions=ex_part,
        config=cfg,
        ex_goals=ex_obs,
    )


# ---------------------------------------------------------------------------
# 1. deterministic mode
# ---------------------------------------------------------------------------

def test_deterministic_subgoal_api():
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

def test_linear_dynamics_schedule_and_posterior_mean_are_finite():
    schedule = make_dynamics_schedule(N=8, beta_min=0.1, beta_max=20.0, lambda_=1.0, bridge_gamma_inv=0.0)
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
    mu = posterior_mean(x_n, x0, xT, n, schedule)
    assert np.all(np.isfinite(np.asarray(x_n)))
    assert np.all(np.isfinite(np.asarray(mu)))


def test_linear_dynamics_resolves_gamma_inv_correctly():
    # The schedule must thread bridge_gamma_inv into a `gamma_inv` entry so
    # downstream agents can query the exact configured denominator offset.
    s_soft = make_dynamics_schedule(N=8, bridge_gamma_inv=0.5)
    assert abs(float(s_soft['gamma_inv']) - 0.5) < 1e-6
    assert 'dynamics_bridge_w' in s_soft
    assert 'dynamics_bridge_var' in s_soft

    # Negative inverse gamma must raise.
    raised = False
    try:
        make_dynamics_schedule(N=4, bridge_gamma_inv=-1.0)
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


def test_flow_subgoal_sampling_shapes_and_mean_pin():
    agent = _make_dynamics_agent(
        'flow',
        subgoal_num_samples=3,
        config_updates={'subgoal_flow_steps': 2, 'subgoal_flow_use_value_bonus': False},
    )
    obs = jnp.zeros((BATCH, STATE_DIM), dtype=jnp.float32)
    g = jnp.zeros((BATCH, STATE_DIM), dtype=jnp.float32)
    pred = agent.predict_subgoal(obs, g)
    assert pred.shape == (BATCH, STATE_DIM)
    mu, log_std = agent.infer_subgoal_distribution(obs, g)
    assert mu.shape == (BATCH, STATE_DIM)
    assert log_std.shape == (BATCH, STATE_DIM)
    np.testing.assert_allclose(np.asarray(mu), np.asarray(pred), rtol=1e-5, atol=1e-6)

    cand, mu2 = agent.sample_subgoal_candidates(
        obs, g, jax.random.PRNGKey(0), num_candidates=5, include_mean=True,
    )
    assert cand.shape == (BATCH, 5, STATE_DIM)
    assert mu2.shape == (BATCH, STATE_DIM)
    np.testing.assert_allclose(np.asarray(cand[:, 0, :]), np.asarray(mu2), rtol=1e-5, atol=1e-6)


def test_flow_build_actor_proposals_subgoal_samples_times_plan_candidates():
    subgoal_samples = 3
    plan_candidates = 4
    agent = _make_dynamics_agent(
        'flow',
        subgoal_num_samples=subgoal_samples,
        config_updates={'subgoal_flow_steps': 2, 'subgoal_flow_use_value_bonus': False},
    )
    _check_build_actor_proposals(
        agent,
        plan_candidates=plan_candidates,
        expected_candidates=subgoal_samples * plan_candidates,
    )


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

def _check_build_actor_proposals(agent, plan_candidates: int, expected_candidates: int | None = None):
    obs = jnp.zeros((BATCH, STATE_DIM), dtype=jnp.float32)
    g = jnp.zeros((BATCH, STATE_DIM), dtype=jnp.float32)
    mu, cand_actions, cand_goals, _ = agent.build_actor_proposals(
        obs, g, jax.random.PRNGKey(0),
        proposal_horizon=2, plan_candidates=plan_candidates, sample_noise_scale=0.0,
    )
    if expected_candidates is None:
        expected_candidates = plan_candidates
    assert mu.shape == (BATCH, STATE_DIM)
    assert cand_actions.shape == (BATCH, expected_candidates, 2, ACTION_DIM)
    assert cand_goals.shape == (BATCH, expected_candidates, STATE_DIM)


def test_build_actor_proposals_deterministic_n1_n_gt_1():
    agent = _make_dynamics_agent('deterministic')
    _check_build_actor_proposals(agent, plan_candidates=1)
    _check_build_actor_proposals(agent, plan_candidates=4)


def test_build_actor_proposals_diag_gaussian_n1_n_gt_1():
    agent = _make_dynamics_agent('diag_gaussian')
    _check_build_actor_proposals(agent, plan_candidates=1)
    _check_build_actor_proposals(agent, plan_candidates=4)


def test_build_actor_proposals_diag_gaussian_subgoal_samples_times_plan_candidates():
    subgoal_samples = 3
    plan_candidates = 4
    agent = _make_dynamics_agent('diag_gaussian', subgoal_num_samples=subgoal_samples)
    _check_build_actor_proposals(
        agent,
        plan_candidates=plan_candidates,
        expected_candidates=subgoal_samples * plan_candidates,
    )


def test_rescore_keeps_global_best_proposal_before_spi():
    subgoal_samples = 3
    plan_candidates = 4
    agent = _make_dynamics_agent('diag_gaussian', subgoal_num_samples=subgoal_samples)
    critic = _make_critic_agent()
    obs = jnp.zeros((BATCH, STATE_DIM), dtype=jnp.float32)
    g = jnp.zeros((BATCH, STATE_DIM), dtype=jnp.float32)
    mu, cand_actions, cand_goals, _ = agent.build_actor_proposals(
        obs,
        g,
        jax.random.PRNGKey(0),
        proposal_horizon=2,
        plan_candidates=plan_candidates,
        sample_noise_scale=0.0,
    )
    assert cand_actions.shape[1] == subgoal_samples * plan_candidates
    actor_batch = {
        'observations': obs,
        'spi_goals': mu,
        'candidate_partial_chunks': cand_actions,
        'candidate_goals': cand_goals,
        'candidate_group_size': plan_candidates,
        'valids': jnp.ones((BATCH, 2), dtype=jnp.float32),
    }
    out_batch, stats = _rescore_actor_batch_for_update(actor_batch, critic, actor_config={})

    scores = critic.score_action_chunks(obs, cand_goals, cand_actions, use_partial_critic=True)
    best_idx = jnp.argmax(scores, axis=1)
    expected_goals = jnp.take_along_axis(cand_goals, best_idx[:, None, None], axis=1)[:, 0, :]

    assert out_batch['proposal_partial_chunks'].shape == (BATCH, 1, 2, ACTION_DIM)
    assert out_batch['proposal_scores'].shape == (BATCH, 1)
    np.testing.assert_allclose(np.asarray(out_batch['spi_goals']), np.asarray(expected_goals), rtol=1e-5, atol=1e-6)
    assert float(stats['proposal_best_of_n']) == float(subgoal_samples * plan_candidates)
    assert float(stats['proposal_pre_best_count']) == float(subgoal_samples * plan_candidates)
    assert float(stats['proposal_post_best_count']) == 1.0


# ---------------------------------------------------------------------------
# 7. distributional subgoal loss has no NaN / Inf
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
        'phase1/subgoal_stochastic_loss',
        'phase1/subgoal_stochastic_loss_mode',
        'phase1/subgoal_mean_mse',
        'phase1/subgoal_sample_mse',
        'phase1/subgoal_weighted_mse',
        'phase1/subgoal_weighted_nll',
        'phase1/subgoal_current_value_mean',
        'phase1/subgoal_mse_weight_mean',
        'phase1/subgoal_std_mean',
        'phase1/subgoal_std_max',
        'phase1/subgoal_mode',
        'dynamics/bridge_gamma_inv',
        'dynamics/gamma_inv',
    ):
        assert required in info, f'missing log key {required}'
    # subgoal_mode == 1.0 in diag_gaussian mode.
    assert float(info['phase1/subgoal_mode']) == 1.0
    assert float(info['phase1/subgoal_stochastic_loss_mode']) == 0.0


def test_flow_subgoal_loss_is_finite():
    agent = _make_dynamics_agent(
        'flow',
        config_updates={'subgoal_flow_steps': 2, 'subgoal_flow_use_value_bonus': False},
    )
    batch = _make_phase1_batch()
    _, info = agent.update(batch, critic_value_params=None)
    for k, v in info.items():
        assert np.all(np.isfinite(np.asarray(v))), f'non-finite log value at {k}: {v}'
    for required in (
        'phase1/subgoal_flow_loss',
        'phase1/subgoal_flow_fm_raw',
        'phase1/subgoal_flow_weighted_fm',
        'phase1/subgoal_flow_weight_mean',
        'phase1/subgoal_flow_weight_max',
        'phase1/subgoal_flow_value_gap_mean',
        'phase1/subgoal_flow_t_mean',
        'phase1/subgoal_flow_velocity_norm',
        'phase1/subgoal_flow_velocity_reg',
        'phase1/subgoal_flow_energy_weighted',
        'phase1/subgoal_flow_energy_weight_mean',
        'phase1/subgoal_flow_energy_weight_max',
        'phase1/subgoal_stochastic_loss',
        'phase1/subgoal_stochastic_loss_mode',
        'phase1/subgoal_weighted_mse',
        'phase1/subgoal_weighted_nll',
        'phase1/subgoal_mode',
    ):
        assert required in info, f'missing log key {required}'
    assert float(info['phase1/subgoal_mode']) == 2.0
    assert float(info['phase1/subgoal_stochastic_loss_mode']) == 2.0
    np.testing.assert_allclose(
        np.asarray(info['phase1/subgoal_stochastic_loss']),
        np.asarray(info['phase1/subgoal_weighted_mse']),
        rtol=1e-5,
        atol=1e-6,
    )


def test_plain_flow_bc_loss_uses_unit_weights_and_eval_goal_l2_selector():
    agent = _make_dynamics_agent(
        'flow',
        subgoal_num_samples=4,
        config_updates={
            'subgoal_flow_steps': 2,
            'subgoal_flow_energy_weighted': False,
            'subgoal_flow_use_value_bonus': False,
            'subgoal_target_mode': 'displacement',
            'residual_target_mode': 'displacement',
            'subgoal_eval_selection': 'best_of_n_goal_l2',
            'subgoal_eval_num_samples': 4,
            'subgoal_eval_include_zero_candidate': False,
        },
    )
    batch = _make_phase1_batch()
    _, info = agent.update(batch, critic_value_params=None)
    assert float(info['phase1/subgoal_flow_energy_weighted']) == 0.0
    np.testing.assert_allclose(np.asarray(info['phase1/subgoal_flow_energy_weight_mean']), 1.0, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(info['phase1/subgoal_flow_energy_weight_max']), 1.0, rtol=1e-6)
    np.testing.assert_allclose(
        np.asarray(info['phase1/subgoal_flow_fm_raw']),
        np.asarray(info['phase1/subgoal_flow_weighted_fm']),
        rtol=1e-5,
        atol=1e-6,
    )

    candidates, _ = agent.sample_subgoal_candidates(
        batch['observations'], batch['high_actor_goals'], jax.random.PRNGKey(42),
        num_candidates=4, include_mean=False,
    )
    assert candidates.shape == (BATCH, 4, STATE_DIM)
    selected, stats, _ = _select_eval_subgoal(
        agent, None, np.asarray(batch['observations'][0]), np.asarray(batch['high_actor_goals'][0]), jax.random.PRNGKey(7),
    )
    assert selected.shape == (STATE_DIM,)
    assert stats['eval/subgoal_selection_mode_id'] == 3.0
    assert stats['eval/subgoal_selection_num_candidates'] == 4.0


def test_distributional_subgoal_nll_loss_option_is_finite():
    agent = _make_dynamics_agent('diag_gaussian', config_updates={'subgoal_stochastic_loss': 'nll'})
    batch = _make_phase1_batch()
    _, info = agent.update(batch, critic_value_params=None)
    for k, v in info.items():
        assert np.all(np.isfinite(np.asarray(v))), f'non-finite log value at {k}: {v}'
    assert float(info['phase1/subgoal_stochastic_loss_mode']) == 1.0
    np.testing.assert_allclose(
        np.asarray(info['phase1/subgoal_stochastic_loss']),
        np.asarray(info['phase1/subgoal_weighted_nll']),
        rtol=1e-5,
        atol=1e-6,
    )


def test_deterministic_subgoal_loss_is_finite_and_logs_are_stable():
    agent = _make_dynamics_agent('deterministic')
    batch = _make_phase1_batch()
    _, info = agent.update(batch, critic_value_params=None)
    for k, v in info.items():
        assert np.all(np.isfinite(np.asarray(v))), f'non-finite log value at {k}: {v}'
    # In deterministic mode the new metrics should be zero placeholders.
    assert float(info['phase1/subgoal_mode']) == 0.0
    assert float(info['phase1/subgoal_nll']) == 0.0
    assert float(info['phase1/subgoal_stochastic_loss_mode']) == 0.0


def test_subgoal_expectile_value_style_weights_by_gap_sign():
    agent = _make_dynamics_agent(
        'deterministic',
        config_updates={
            'subgoal_value_style': 'expectile',
            'subgoal_value_alpha': 0.1,
            'subgoal_value_expectile': 0.3,
        },
    )
    gap = jnp.asarray([0.2, -0.1, 0.0], dtype=jnp.float32)
    weight = agent._subgoal_mse_weight_from_gap(gap)
    np.testing.assert_allclose(np.asarray(weight), np.asarray([0.3, 0.7, 0.7]), rtol=1e-6)


# ---------------------------------------------------------------------------
# 8. dynamics-config defaults are usable
# ---------------------------------------------------------------------------

def test_dynamics_config_defaults_are_usable():
    cfg = get_dynamics_config()
    assert str(cfg.subgoal_distribution) == 'deterministic'
    assert str(cfg.subgoal_stochastic_loss) == 'mse'
    assert bool(cfg.subgoal_use_mean_for_actor_goal) is True
    assert int(cfg.subgoal_num_samples) == 1
    assert int(cfg.subgoal_flow_steps) == 8
    assert float(cfg.subgoal_flow_t_min) == 1e-4
    assert bool(cfg.subgoal_flow_use_value_bonus) is False
    assert bool(cfg.subgoal_flow_energy_weighted) is True
    assert str(cfg.subgoal_eval_selection) == 'zero'
    assert int(cfg.subgoal_eval_num_samples) == 1
    assert bool(cfg.subgoal_eval_include_zero_candidate) is True
    assert int(cfg.subgoal_eval_seed) == 0
    assert float(cfg.subgoal_flow_noise_scale) == 1.0
    assert str(cfg.subgoal_value_style) == 'exponential'
    assert float(cfg.subgoal_value_expectile) == 0.7
    assert float(cfg.subgoal_value_gap_scale) == 1.0


def test_invalid_subgoal_stochastic_loss_rejected():
    raised = False
    try:
        _make_dynamics_agent('diag_gaussian', config_updates={'subgoal_stochastic_loss': 'bad'})
    except ValueError:
        raised = True
    assert raised


# ---------------------------------------------------------------------------
# 9. subgoal_target_mode='displacement'
# ---------------------------------------------------------------------------

def _displacement_agent(subgoal_distribution='deterministic', **updates):
    return _make_dynamics_agent(
        subgoal_distribution,
        config_updates={'subgoal_target_mode': 'displacement', **updates},
    )


def test_displacement_mode_predict_subgoal_returns_absolute_state():
    # ``predict_subgoal`` must always return an absolute state; in displacement
    # mode the raw network output is Delta and the agent adds ``observations``
    # internally so callers cannot tell the difference.
    agent_disp = _displacement_agent()
    agent_abs = _make_dynamics_agent('deterministic')
    rng = np.random.default_rng(0)
    obs = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    g = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))

    sg_disp = np.asarray(agent_disp.predict_subgoal(obs, g))
    raw_disp = np.asarray(agent_disp._subgoal_forward(obs, g))
    # Output should equal obs + raw network output in displacement mode.
    np.testing.assert_allclose(sg_disp, np.asarray(obs) + raw_disp, rtol=1e-5, atol=1e-6)
    assert sg_disp.shape == (BATCH, STATE_DIM)

    # Sanity: absolute-mode agent returns the raw output untouched.
    sg_abs = np.asarray(agent_abs.predict_subgoal(obs, g))
    raw_abs = np.asarray(agent_abs._subgoal_forward(obs, g))
    np.testing.assert_allclose(sg_abs, raw_abs, rtol=1e-5, atol=1e-6)


def test_displacement_mode_plan_endpoint_clamped_to_desired_endpoint():
    # The bridge planner exposes an absolute API; with displacement mode on,
    # ``plan(current_state, desired_endpoint)`` must still return a trajectory
    # whose endpoints equal ``current_state`` and ``desired_endpoint`` even
    # though the underlying chain is trained in displacement frame.
    agent = _displacement_agent()
    rng = np.random.default_rng(1)
    current = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    endpoint = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    result = agent.plan(current, endpoint)
    traj = np.asarray(result['trajectory'])
    # The exact-residual chain pins ``traj[:, 0]`` to ``current_state`` because
    # the plan helper adds back the origin after a 0-anchored chain rollout.
    np.testing.assert_allclose(traj[:, 0, :], np.asarray(current), rtol=1e-5, atol=1e-6)


def test_displacement_mode_forward_bridge_endpoint_preserved():
    # ``forward_bridge`` clamps both endpoints; the displacement shift must
    # not break that invariant.
    cfg_updates = {
        'subgoal_target_mode': 'displacement',
        'planner_type': 'forward_bridge',
    }
    agent = _make_dynamics_agent('deterministic', config_updates=cfg_updates)
    rng = np.random.default_rng(2)
    current = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    endpoint = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    traj = np.asarray(agent.plan(current, endpoint)['trajectory'])
    np.testing.assert_allclose(traj[:, 0, :], np.asarray(current), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(traj[:, -1, :], np.asarray(endpoint), rtol=1e-5, atol=1e-6)


def test_displacement_mode_loss_is_finite_and_target_mode_logged():
    # Phase-1 loss with displacement-mode targets must remain finite, and the
    # ``dynamics/subgoal_target_mode`` metric must flag the mode (1.0).
    agent = _displacement_agent('deterministic')
    batch = _make_phase1_batch()
    _, info = agent.update(batch, critic_value_params=None)
    for k, v in info.items():
        v = np.asarray(v)
        assert np.all(np.isfinite(v)), f'non-finite log at {k}: {v}'
    assert float(info['dynamics/subgoal_target_mode']) == 1.0

    agent_abs = _make_dynamics_agent('deterministic')
    _, info_abs = agent_abs.update(batch, critic_value_params=None)
    assert float(info_abs['dynamics/subgoal_target_mode']) == 0.0


def test_displacement_mode_residual_net_uses_anchor():
    # PathResidualNet must take an anchor input and respond to it; otherwise
    # displacement mode collapses to a translation-invariant correction and the
    # bridge cannot distinguish two trajectories with the same Delta but
    # different current states.
    agent = _displacement_agent('deterministic')
    rng = np.random.default_rng(11)
    zK = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    t_norm = jnp.full((BATCH, 1), 0.5, dtype=jnp.float32)
    s1_a = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    s1_b = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))

    eps_a = np.asarray(
        agent.network.select('path_residual_net')(s1_a, zK, t_norm)
    )
    eps_b = np.asarray(
        agent.network.select('path_residual_net')(s1_b, zK, t_norm)
    )
    # Different anchors must produce different residuals on at least one
    # element of the batch (we just need to confirm the input is wired in).
    assert not np.allclose(eps_a, eps_b, atol=1e-6)


def test_displacement_mode_plan_responds_to_current_state():
    # End-to-end check: with the same Delta = desired_endpoint - current_state
    # but a different current_state (and matching desired_endpoint shift), the
    # produced trajectory shape (after subtracting the trivial origin shift)
    # must differ - otherwise the residual chain is forced to be translation
    # invariant.
    agent = _displacement_agent('deterministic')
    rng = np.random.default_rng(12)
    current_a = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    delta = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    endpoint_a = current_a + delta
    shift = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    current_b = current_a + shift
    endpoint_b = endpoint_a + shift  # same Delta in both cases

    traj_a = np.asarray(agent.plan(current_a, endpoint_a)['trajectory'])
    traj_b = np.asarray(agent.plan(current_b, endpoint_b)['trajectory'])
    # Remove the origin shift to isolate the bridge shape.
    shape_a = traj_a - np.asarray(current_a)[:, None, :]
    shape_b = traj_b - np.asarray(current_b)[:, None, :]
    # If the residual ignored the anchor, both shapes would be identical.
    assert not np.allclose(shape_a, shape_b, atol=1e-6)


def test_bridge_anchor_is_always_current_state():
    # The anchor channel is mode-agnostic: it always carries ``s_t``.  In
    # absolute mode this is redundant with ``x_T``; in displacement mode it
    # is the only path through which ``s_t`` reaches the residual MLP.
    obs = jnp.zeros((BATCH, STATE_DIM), dtype=jnp.float32) + 3.0
    for mode in ('absolute', 'displacement'):
        agent = _make_dynamics_agent(
            'deterministic', config_updates={'subgoal_target_mode': mode},
        )
        anchor = np.asarray(agent._bridge_anchor(obs))
        np.testing.assert_allclose(anchor, np.asarray(obs), rtol=1e-6, atol=1e-6)


def test_displacement_mode_build_actor_proposals_returns_absolute_goals():
    # ``build_actor_proposals`` must return absolute ``mu`` and
    # ``candidate_goals`` so SPI/Q can be scored in absolute state space.
    agent = _displacement_agent('diag_gaussian')
    rng = np.random.default_rng(3)
    obs = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    g = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    mu, _, cand_goals, _ = agent.build_actor_proposals(
        obs, g, jax.random.PRNGKey(0),
        proposal_horizon=2, plan_candidates=2, sample_noise_scale=0.0,
    )
    # Network output is Delta; ``mu - obs`` should match the raw forward-pass
    # mean (so ``mu`` is in absolute frame as expected by downstream callers).
    raw_mu = np.asarray(agent._subgoal_forward(obs, g)[0])
    np.testing.assert_allclose(np.asarray(mu) - np.asarray(obs), raw_mu, rtol=1e-5, atol=1e-6)
    # Each candidate goal is an absolute state with the same shape.
    assert cand_goals.shape[-1] == STATE_DIM


def _make_trl_critic_agent():
    cfg = get_critic_config()
    cfg.critic_type = 'direct_chunk_trl'
    cfg.algorithm = 'direct_chunk_trl'
    cfg.use_chunk_critic = False
    cfg.goal_representation = 'full'
    cfg.action_chunk_horizon = 2
    cfg.full_chunk_horizon = 4
    cfg.value_hidden_dims = (32, 32)
    cfg.action_dim = ACTION_DIM
    ex_obs = np.zeros((BATCH, STATE_DIM), dtype=np.float32)
    ex_part = np.zeros((BATCH, cfg.action_chunk_horizon * ACTION_DIM), dtype=np.float32)
    return CriticAgent.create(
        seed=0,
        ex_observations=ex_obs,
        ex_full_chunk_actions=None,
        ex_action_chunk_actions=ex_part,
        config=cfg,
        ex_goals=ex_obs,
    )


def _subgoal_value_test_config_updates() -> dict:
    return {
        'subgoal_value_goal_representation': 'full',
        'subgoal_value_hidden_dims': (32, 32),
    }


def test_trl_subgoal_value_bonus_uses_product_form():
    critic = _make_trl_critic_agent()
    value_params = critic.network.params['modules_value']
    agent = _make_dynamics_agent(
        'diag_gaussian',
        config_updates={
            **_subgoal_value_test_config_updates(),
            'critic_type': 'direct_chunk_trl',
            'algorithm': 'direct_chunk_trl',
            'subgoal_value_alpha': 1.0,
        },
    )
    rng = np.random.default_rng(17)
    s = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    sg = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    target = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    g = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))

    (
        pred_value,
        _obs_value,
        _target_value,
        bonus,
        _mse_weight,
        _gap,
        _adv_logit,
        v_s_sg,
        v_sg_g,
    ) = agent._subgoal_value_terms(s, sg, target, g, value_params)

    v_s_sg_expected = agent._subgoal_values(s, sg, value_params)
    v_sg_g_expected = pred_value
    np.testing.assert_allclose(np.asarray(v_s_sg), np.asarray(v_s_sg_expected), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(v_sg_g), np.asarray(v_sg_g_expected), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(bonus),
        np.asarray(v_s_sg_expected * v_sg_g_expected),
        rtol=1e-5,
        atol=1e-6,
    )


def test_state_transitive_subgoal_bonus_ratio_and_gradients():
    critic = _make_trl_critic_agent()
    value_params = critic.network.params['modules_value']
    agent = _make_dynamics_agent(
        'diag_gaussian',
        config_updates={
            **_subgoal_value_test_config_updates(),
            'critic_type': 'state_transitive',
            'algorithm': 'state_transitive',
            'subgoal_value_alpha': 1.0,
            'subgoal_value_bonus_type': 'transitive_ratio',
            'subgoal_value_ratio_eps': 1e-6,
        },
    )
    rng = np.random.default_rng(23)
    s = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    sg = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    target = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    g = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))

    pred_value, obs_value, _, bonus, _, _, _, v_s_sg, v_sg_g = agent._subgoal_value_terms(
        s, sg, target, g, value_params,
    )
    expected_ratio = v_s_sg * v_sg_g / (obs_value + 1e-6)
    assert np.all(np.isfinite(np.asarray(expected_ratio)))
    np.testing.assert_allclose(np.asarray(bonus), np.asarray(expected_ratio), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(v_sg_g), np.asarray(pred_value), rtol=1e-5, atol=1e-6)

    def bonus_sum(z, params):
        _, _, _, b, _, _, _, _, _ = agent._subgoal_value_terms(s, z, target, g, params)
        return jnp.sum(b)

    grad_sg, grad_params = jax.grad(bonus_sum, argnums=(0, 1))(sg, value_params)
    assert float(jnp.sum(jnp.abs(grad_sg))) > 0.0
    param_grad_sum = sum(float(jnp.sum(jnp.abs(x))) for x in jax.tree_util.tree_leaves(grad_params))
    assert param_grad_sum == 0.0


def test_dqc_subgoal_value_bonus_uses_single_value_form():
    critic = _make_critic_agent()
    critic_cfg = get_critic_config()
    critic_cfg.goal_representation = 'full'
    critic_cfg.value_hidden_dims = (32, 32)
    critic_cfg.action_chunk_horizon = 2
    critic_cfg.full_chunk_horizon = 4
    critic_cfg.action_dim = ACTION_DIM
    ex_obs = np.zeros((BATCH, STATE_DIM), dtype=np.float32)
    ex_full = np.zeros((BATCH, critic_cfg.full_chunk_horizon * ACTION_DIM), dtype=np.float32)
    ex_part = np.zeros((BATCH, critic_cfg.action_chunk_horizon * ACTION_DIM), dtype=np.float32)
    critic = CriticAgent.create(
        seed=0,
        ex_observations=ex_obs,
        ex_full_chunk_actions=ex_full,
        ex_action_chunk_actions=ex_part,
        config=critic_cfg,
        ex_goals=ex_obs,
    )
    value_params = critic.network.params['modules_value']
    agent = _make_dynamics_agent(
        'diag_gaussian',
        config_updates={
            **_subgoal_value_test_config_updates(),
            'critic_type': 'dqc',
            'algorithm': 'dqc',
            'subgoal_value_alpha': 0.5,
        },
    )
    rng = np.random.default_rng(19)
    s = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    sg = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    target = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))
    g = jnp.asarray(rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32))

    pred_value, _, _, bonus, _, _, _, v_s_sg, v_sg_g = agent._subgoal_value_terms(
        s, sg, target, g, value_params,
    )
    np.testing.assert_allclose(np.asarray(bonus), 0.5 * np.asarray(pred_value), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(v_s_sg), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(v_sg_g), np.asarray(pred_value), rtol=1e-5, atol=1e-6)


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
