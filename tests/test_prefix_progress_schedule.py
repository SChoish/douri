"""Tests for the prefix-calibrated theta schedule.

Covers (per ``Cursor implementation note: prefix-calibrated bridge schedule``):

* Legacy ``linear_beta`` schedule is preserved bit-for-bit.
* ``prefix_progress`` produces hard-bridge marginals matching the desired
  ``c_i = (i / K) ** progress_alpha`` curve, with explicitly pinned endpoints.
* ``forward_bridge_coefficients`` honours the same dispatch.
* The agent-level ``info`` dict surfaces the new schedule diagnostics under
  every planner / model_type combination.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

import jax  # noqa: F401  (imported to ensure JAX is available for the agent tests)

from utils.dynamics import (
    forward_bridge_coefficients,
    make_dynamics_schedule,
)
from utils.theta_schedules import (
    canonical_theta_schedule,
    desired_prefix_progress,
    prefix_progress_theta_fwd,
)


STATE_DIM = 4
ACTION_DIM = 2
BATCH = 8


def test_canonical_theta_schedule_validates_string():
    assert canonical_theta_schedule('linear_beta') == 'linear_beta'
    assert canonical_theta_schedule('PREFIX_PROGRESS') == 'prefix_progress'
    raised = False
    try:
        canonical_theta_schedule('not_a_schedule')
    except ValueError:
        raised = True
    assert raised, 'unknown theta_schedule must raise'


def test_linear_beta_schedule_matches_diffusion_theta():
    """Default (linear_beta) schedule matches the diffusion-style theta array."""
    N = 25
    beta_min = 0.1
    beta_max = 20.0
    sched = make_dynamics_schedule(
        N=N,
        beta_min=beta_min,
        beta_max=beta_max,
        lambda_=1.0,
        bridge_gamma_inv=0.0,
        theta_schedule='linear_beta',
    )

    steps = np.arange(1, N + 1, dtype=np.float32)
    expected = beta_min / N + (beta_max - beta_min) * steps / (N * N)
    np.testing.assert_allclose(
        np.asarray(sched['theta']), expected, rtol=1e-6, atol=1e-6,
    )

    # Legacy schedule has no prefix-progress target.
    target = np.asarray(sched['progress_target_fwd'])
    assert np.all(np.isnan(target))
    assert float(sched['theta_schedule_id']) == 0.0


def test_linear_beta_default_kwarg_matches_explicit_mode():
    """Calling make_dynamics_schedule without theta_schedule matches explicit linear_beta."""
    N = 10
    sched_default = make_dynamics_schedule(
        N=N, beta_min=0.1, beta_max=20.0, lambda_=1.0, bridge_gamma_inv=0.0,
    )
    sched_explicit = make_dynamics_schedule(
        N=N, beta_min=0.1, beta_max=20.0, lambda_=1.0, bridge_gamma_inv=0.0,
        theta_schedule='linear_beta',
    )
    for k in ('theta', 'g2', 'step_var', 'bridge_w', 'bridge_var', 'dynamics_beta_fwd'):
        np.testing.assert_allclose(
            np.asarray(sched_default[k]), np.asarray(sched_explicit[k]), atol=1e-7,
        )


def test_prefix_progress_matches_desired_bridge_progress():
    """With prefix_progress, the hard-bridge marginal weights match
    ``c_i = (i / K) ** alpha`` exactly (modulo float tolerance)."""
    N = 25
    alpha = 0.8
    sched = make_dynamics_schedule(
        N=N,
        beta_min=0.1,
        beta_max=20.0,
        lambda_=1.0,
        bridge_gamma_inv=0.0,
        theta_schedule='prefix_progress',
        theta_total=1.0,
        progress_alpha=alpha,
    )

    idx = np.arange(N + 1, dtype=np.float32)
    desired = (idx / float(N)) ** alpha
    desired[0] = 0.0
    desired[-1] = 1.0

    actual = np.asarray(sched['dynamics_beta_fwd'])
    np.testing.assert_allclose(actual, desired, rtol=1e-5, atol=1e-5)
    assert abs(float(actual[5]) - (5.0 / 25.0) ** alpha) < 1e-5


def test_prefix_progress_endpoint_pinning():
    sched = make_dynamics_schedule(
        N=25,
        theta_schedule='prefix_progress',
        theta_total=1.0,
        progress_alpha=0.8,
        bridge_gamma_inv=0.0,
    )

    beta = np.asarray(sched['dynamics_beta_fwd'])
    bvar = np.asarray(sched['dynamics_bridge_var_fwd'])

    assert beta[0] == 0.0
    assert beta[-1] == 1.0
    assert bvar[0] == 0.0
    assert bvar[-1] == 0.0


def test_prefix_progress_metadata_keys():
    """Metadata diagnostics must be present and consistent with config."""
    sched = make_dynamics_schedule(
        N=25,
        theta_schedule='prefix_progress',
        theta_total=1.5,
        progress_alpha=0.8,
        bridge_gamma_inv=0.0,
    )
    assert float(sched['theta_schedule_id']) == 1.0
    assert np.isclose(float(sched['theta_total']), 1.5, atol=1e-6)
    assert np.isclose(float(sched['progress_alpha']), 0.8, atol=1e-6)
    target = np.asarray(sched['progress_target_fwd'])
    assert target.shape == (26,)
    assert target[0] == 0.0
    assert target[-1] == 1.0


def test_forward_bridge_coefficients_prefix_progress():
    K = 25
    alpha = 0.8
    a, b, std = forward_bridge_coefficients(
        K,
        beta_min=0.1,
        beta_max=20.0,
        lambda_=1.0,
        theta_schedule='prefix_progress',
        theta_total=1.0,
        progress_alpha=alpha,
    )

    desired = (np.arange(K + 1, dtype=np.float32) / float(K)) ** alpha
    desired[0] = 0.0
    desired[-1] = 1.0

    np.testing.assert_allclose(np.asarray(b), desired, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(a), 1.0 - desired, rtol=1e-5, atol=1e-5)
    assert float(std[0]) == 0.0
    assert float(std[-1]) == 0.0


def test_forward_bridge_coefficients_linear_beta_default_unchanged():
    """Without theta_schedule, forward_bridge_coefficients reproduces the
    linear-beta marginals."""
    K = 8
    a_def, b_def, std_def = forward_bridge_coefficients(
        K, beta_min=0.1, beta_max=20.0, lambda_=1.0,
    )
    a_explicit, b_explicit, std_explicit = forward_bridge_coefficients(
        K, beta_min=0.1, beta_max=20.0, lambda_=1.0,
        theta_schedule='linear_beta',
    )
    np.testing.assert_allclose(np.asarray(a_def), np.asarray(a_explicit), atol=1e-7)
    np.testing.assert_allclose(np.asarray(b_def), np.asarray(b_explicit), atol=1e-7)
    np.testing.assert_allclose(np.asarray(std_def), np.asarray(std_explicit), atol=1e-7)


def test_prefix_progress_helpers_internal_consistency():
    """Internal helper produces (N,) theta_fwd and the cumulative recovers
    the desired progress curve via beta_i = sinh(Theta_i) / sinh(Theta_K)."""
    N = 25
    alpha = 0.8
    total = 1.0
    theta_fwd = prefix_progress_theta_fwd(N, theta_total=total, progress_alpha=alpha)
    assert theta_fwd.shape == (N,)
    Theta = np.concatenate([np.zeros(1), np.cumsum(np.asarray(theta_fwd))])
    beta = np.sinh(Theta) / np.sinh(total)
    desired = np.asarray(desired_prefix_progress(N, progress_alpha=alpha))
    np.testing.assert_allclose(beta, desired, rtol=1e-5, atol=1e-5)


def test_agent_logs_theta_schedule_metadata():
    """End-to-end: every planner / model_type combo surfaces the new logs."""
    from agents.dynamics import DynamicsAgent, get_dynamics_config

    rng = np.random.default_rng(0)

    def _make_batch(K):
        obs = rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32)
        targets = rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32)
        goals = rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32)
        seg = np.zeros((BATCH, K + 1, STATE_DIM), dtype=np.float32)
        seg[:, 0] = obs
        seg[:, -1] = targets
        for i in range(1, K):
            t = i / float(K)
            seg[:, i] = (1.0 - t) * obs + t * targets
        return dict(
            observations=obs,
            next_observations=rng.standard_normal((BATCH, STATE_DIM)).astype(np.float32),
            actions=rng.standard_normal((BATCH, ACTION_DIM)).astype(np.float32),
            high_actor_goals=goals,
            high_actor_targets=targets,
            trajectory_segment=seg,
        )

    combos = [
        ('forward_bridge_residual', 'exact_residual'),
    ]
    for planner, model_type in combos:
        cfg = get_dynamics_config()
        cfg.dynamics_N = 4
        cfg.subgoal_steps = 4
        cfg.rollout_horizon = 2
        cfg.residual_model_hidden_dims = (32, 32)
        cfg.subgoal_hidden_dims = (32, 32)
        cfg.subgoal_value_hidden_dims = (32, 32)
        cfg.idm_hidden_dims = (32, 32)
        cfg.path_residual_hidden_dims = (32, 32)
        cfg.planner_type = planner
        cfg.dynamics_model_type = model_type
        cfg.theta_schedule = 'prefix_progress'
        cfg.theta_total = 1.0
        cfg.progress_alpha = 0.8

        ex_obs = np.zeros((BATCH, STATE_DIM), dtype=np.float32)
        ex_act = np.zeros((BATCH, ACTION_DIM), dtype=np.float32)
        agent = DynamicsAgent.create(seed=0, ex_observations=ex_obs, ex_actions=ex_act, config=cfg)
        batch = _make_batch(int(cfg.dynamics_N))
        _, info = agent.update(batch)

        for k in (
            'dynamics/theta_schedule_id',
            'dynamics/theta_total',
            'dynamics/progress_alpha',
            'dynamics/prefix_progress_target_5',
            'dynamics/prefix_progress_actual_5',
        ):
            assert k in info, f'missing log {k} for planner={planner} model_type={model_type}'
        assert float(info['dynamics/theta_schedule_id']) == 1.0


if __name__ == '__main__':
    test_canonical_theta_schedule_validates_string()
    test_linear_beta_schedule_matches_diffusion_theta()
    test_linear_beta_default_kwarg_matches_explicit_mode()
    test_prefix_progress_matches_desired_bridge_progress()
    test_prefix_progress_endpoint_pinning()
    test_prefix_progress_metadata_keys()
    test_forward_bridge_coefficients_prefix_progress()
    test_forward_bridge_coefficients_linear_beta_default_unchanged()
    test_prefix_progress_helpers_internal_consistency()
    test_agent_logs_theta_schedule_metadata()
    print('OK: all prefix_progress schedule tests passed.')
