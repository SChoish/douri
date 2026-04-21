"""Smoke test for GOUB dynamics implementation.

Run from the repository root (directory containing ``main_*.py`` and ``agents/``):
    python smoke_test_goub.py

Checklist
---------
1. Schedule boundary conditions and variance identity.
2. Bridge sampling shapes and finiteness.
3. Posterior mean shapes, finiteness, and n=1 collapse to x_0.
4. model_mean shapes and finiteness for n in {1, ..., N-1}.
5. Agent creation.
6. Training loss shape, finiteness, positivity.
7. Loss decreases on a tiny synthetic dataset (200 steps).
8. plan() shapes and finiteness for single and batched inputs.
9. **next_step changes when network parameters change** (the critical test).
10. All outputs finite.
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

from agents.goub_dynamics import GOUBDynamicsAgent, get_dynamics_config
from utils.goub import bridge_sample, make_goub_schedule, model_mean, posterior_mean

PASS = 0
FAIL = 0


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'  [PASS] {name}')
    else:
        FAIL += 1
        print(f'  [FAIL] {name}  {detail}')


def _make_test_config(N: int = 10, *, batch_size: int = 64, lr: float = 3e-4):
    config = get_dynamics_config()
    config['goub_N'] = N
    config['subgoal_steps'] = N
    config['batch_size'] = batch_size
    config['lr'] = lr
    config['idm_loss_weight'] = 0.0
    return config


def _make_dynamics_batch(B: int, D: int, N: int):
    obs = jnp.linspace(0.0, 1.0, B * D, dtype=jnp.float32).reshape(B, D)
    target = jnp.linspace(1.0, 2.0, B * D, dtype=jnp.float32).reshape(B, D)
    high_goal = 0.5 * (obs + target)
    alphas = jnp.linspace(0.0, 1.0, N + 1, dtype=jnp.float32).reshape(1, N + 1, 1)
    segment = obs[:, None, :] * (1.0 - alphas) + target[:, None, :] * alphas
    next_obs = segment[:, 1, :]
    actions = jnp.zeros((B, 2), dtype=jnp.float32)
    return {
        'observations': obs,
        'next_observations': next_obs,
        'high_actor_goals': high_goal,
        'high_actor_targets': target,
        'trajectory_segment': segment,
        'actions': actions,
    }


# ------------------------------------------------------------------
# 1. Schedule
# ------------------------------------------------------------------

def test_schedule():
    print('\n--- Schedule ---')
    N = 50
    sched = make_goub_schedule(N, beta_min=0.1, beta_max=20.0, lambda_=1.0)

    for k, v in sched.items():
        if hasattr(v, 'shape'):
            print(f'  {k}: shape={v.shape}  min={float(v.min()):.6f}  max={float(v.max()):.6f}')
        else:
            print(f'  {k}: {float(v):.6f}')

    check('bridge_var[0] == 0', jnp.isclose(sched['bridge_var'][0], 0.0, atol=1e-10))
    check('bridge_var[N] ~ 0', jnp.isclose(sched['bridge_var'][N], 0.0, atol=1e-6),
          f'got {float(sched["bridge_var"][N]):.2e}')
    check('bridge_w[0] == 1', jnp.isclose(sched['bridge_w'][0], 1.0, atol=1e-6))
    check('bridge_w[N] ~ 0', jnp.isclose(sched['bridge_w'][N], 0.0, atol=1e-6),
          f'got {float(sched["bridge_w"][N]):.2e}')

    check('bar_theta monotone', bool(jnp.all(jnp.diff(sched['bar_theta']) > 0)))
    check('bar_theta_nN monotone decreasing', bool(jnp.all(jnp.diff(sched['bar_theta_nN']) <= 0)))

    total_id = sched['bar_sigma2'][N] - (
        jnp.exp(-2 * sched['bar_theta_nN']) * sched['bar_sigma2'] + sched['bar_sigma2_nN']
    )
    check('total variance identity', bool(jnp.allclose(total_id, 0.0, atol=1e-5)),
          f'max residual {float(jnp.abs(total_id).max()):.2e}')


# ------------------------------------------------------------------
# 2. Bridge sampling
# ------------------------------------------------------------------

def test_bridge_sampling():
    print('\n--- Bridge sampling ---')
    N = 50
    sched = make_goub_schedule(N)
    rng = jax.random.PRNGKey(0)

    B, D = 256, 10
    x_0 = jnp.zeros((B, D))
    x_T = jnp.ones((B, D))

    for step in [1, N // 2, N - 1]:
        n = jnp.full((B,), step, dtype=jnp.int32)
        x_n = bridge_sample(x_0, x_T, n, sched, rng)
        check(f'n={step} shape', x_n.shape == (B, D))
        check(f'n={step} finite', bool(jnp.all(jnp.isfinite(x_n))))

    n_lo = jnp.full((B,), 1, dtype=jnp.int32)
    n_hi = jnp.full((B,), N - 1, dtype=jnp.int32)
    x_lo = bridge_sample(x_0, x_T, n_lo, sched, rng)
    x_hi = bridge_sample(x_0, x_T, n_hi, sched, rng)
    check('n=1 closer to x_0', float(jnp.abs(x_lo - x_0).mean()) < float(jnp.abs(x_hi - x_0).mean()))


# ------------------------------------------------------------------
# 3. Posterior mean
# ------------------------------------------------------------------

def test_posterior_mean():
    print('\n--- Posterior mean ---')
    N = 50
    sched = make_goub_schedule(N)
    rng = jax.random.PRNGKey(42)

    B, D = 256, 10
    x_0 = jnp.zeros((B, D))
    x_T = jnp.ones((B, D))

    for step in [1, 2, N // 2, N - 1, N]:
        n = jnp.full((B,), step, dtype=jnp.int32)
        x_n = x_T if step == N else bridge_sample(x_0, x_T, n, sched, rng)
        mu = posterior_mean(x_n, x_0, x_T, n, sched)
        check(f'n={step} shape', mu.shape == (B, D))
        check(f'n={step} finite', bool(jnp.all(jnp.isfinite(mu))))

    n1 = jnp.full((B,), 1, dtype=jnp.int32)
    x_1 = bridge_sample(x_0, x_T, n1, sched, rng)
    mu_0 = posterior_mean(x_1, x_0, x_T, n1, sched)
    err = float(jnp.abs(mu_0 - x_0).max())
    check(f'n=1 posterior ~ x_0 (err={err:.2e})', err < 1e-4)


# ------------------------------------------------------------------
# 4. model_mean
# ------------------------------------------------------------------

def test_model_mean():
    print('\n--- Model mean ---')
    N = 50
    sched = make_goub_schedule(N)
    rng = jax.random.PRNGKey(1)

    B, D = 64, 10
    x_0 = jnp.zeros((B, D))
    x_T = jnp.ones((B, D))

    for step in [1, N // 2, N - 1]:
        n = jnp.full((B,), step, dtype=jnp.int32)
        x_n = bridge_sample(x_0, x_T, n, sched, rng)
        eps_pred = jnp.zeros((B, D))
        mu = model_mean(x_n, x_T, eps_pred, n, sched)
        check(f'n={step} shape', mu.shape == (B, D))
        check(f'n={step} finite', bool(jnp.all(jnp.isfinite(mu))))


# ------------------------------------------------------------------
# 5. Agent creation
# ------------------------------------------------------------------

def test_agent_creation():
    print('\n--- Agent creation ---')
    config = _make_test_config(N=10)

    B, D = 4, 10
    ex_obs = jnp.zeros((B, D))
    ex_act = jnp.zeros((1, 2), dtype=jnp.float32)

    agent = GOUBDynamicsAgent.create(0, ex_obs, config, ex_actions=ex_act)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(agent.network.params))
    print(f'  Total parameters: {n_params:,}')
    check('agent created', agent is not None)


# ------------------------------------------------------------------
# 6. Agent loss
# ------------------------------------------------------------------

def test_agent_loss():
    print('\n--- Agent loss ---')
    config = _make_test_config(N=10)

    B, D = 64, 10
    agent = GOUBDynamicsAgent.create(
        0, jnp.zeros((1, D)), config, ex_actions=jnp.zeros((1, 2), dtype=jnp.float32)
    )
    batch = _make_dynamics_batch(B, D, int(config['goub_N']))

    rng = jax.random.PRNGKey(0)
    loss, info = agent.total_loss(batch, agent.network.params, rng=rng)

    print(f'  loss = {float(loss):.6f}')
    for k, v in info.items():
        print(f'  {k} = {float(v):.6f}')
    check('loss finite', bool(jnp.isfinite(loss)))
    check('loss > 0', float(loss) > 0)
    check('all info finite', all(bool(jnp.isfinite(v)) for v in info.values()))


# ------------------------------------------------------------------
# 7. Loss decreases
# ------------------------------------------------------------------

def test_loss_decreases():
    print('\n--- Loss decrease (200 steps) ---')
    config = _make_test_config(N=10, batch_size=64, lr=1e-3)

    D = 10
    N = int(config['goub_N'])
    agent = GOUBDynamicsAgent.create(
        0, jnp.zeros((1, D)), config, ex_actions=jnp.zeros((1, 2), dtype=jnp.float32)
    )

    np.random.seed(42)
    obs_data = np.random.randn(256, D).astype(np.float32)
    target_data = np.random.randn(256, D).astype(np.float32)

    losses = []
    for step in range(200):
        idx = np.random.choice(256, 64)
        obs = jnp.array(obs_data[idx])
        target = jnp.array(target_data[idx])
        high_goal = target * 0.1
        alphas = jnp.linspace(0.0, 1.0, N + 1, dtype=jnp.float32).reshape(1, N + 1, 1)
        segment = obs[:, None, :] * (1.0 - alphas) + target[:, None, :] * alphas
        batch = {
            'observations': obs,
            'next_observations': segment[:, 1, :],
            'high_actor_goals': high_goal,
            'high_actor_targets': target,
            'trajectory_segment': segment,
            'actions': jnp.zeros((64, 2), dtype=jnp.float32),
        }
        agent, info = agent.update(batch)
        losses.append(float(info['phase1/loss']))

    early = np.mean(losses[:20])
    late = np.mean(losses[-20:])
    print(f'  early loss (mean first 20): {early:.6f}')
    print(f'  late  loss (mean last  20): {late:.6f}')
    check(f'loss decreased ({early:.4f} -> {late:.4f})', late < early)


# ------------------------------------------------------------------
# 8. plan() shapes and finiteness
# ------------------------------------------------------------------

def test_inference():
    print('\n--- Inference ---')
    config = _make_test_config(N=10)
    N = config['goub_N']

    D = 10
    agent = GOUBDynamicsAgent.create(
        0, jnp.zeros((1, D)), config, ex_actions=jnp.zeros((1, 2), dtype=jnp.float32)
    )

    # Single input
    current = jnp.ones(D)
    endpoint = jnp.zeros(D)
    result = agent.plan(current, endpoint)
    check('next_step shape (single)', result['next_step'].shape == (D,))
    check('trajectory shape (single)', result['trajectory'].shape == (N + 1, D))
    check('next_step finite', bool(jnp.all(jnp.isfinite(result['next_step']))))
    check('trajectory finite', bool(jnp.all(jnp.isfinite(result['trajectory']))))
    check('traj[0] == current', bool(jnp.allclose(result['trajectory'][0], current, atol=1e-5)))

    # Batched input
    B = 8
    current_b = jnp.ones((B, D))
    endpoint_b = jnp.zeros((B, D))
    result_b = agent.plan(current_b, endpoint_b)
    check('next_step shape (batch)', result_b['next_step'].shape == (B, D))
    check('trajectory shape (batch)', result_b['trajectory'].shape == (B, N + 1, D))

    print(f'  traj[0, :3]  = {result["trajectory"][0, :3]}  (should be ~1)')
    print(f'  traj[-1, :3] = {result["trajectory"][-1, :3]}')


# ------------------------------------------------------------------
# 9. next_step is learned (CRITICAL)
# ------------------------------------------------------------------

def test_next_step_is_learned():
    print('\n--- next_step is learned ---')
    config = _make_test_config(N=10, lr=1e-3)

    D = 10
    N = int(config['goub_N'])
    agent = GOUBDynamicsAgent.create(
        0, jnp.zeros((1, D)), config, ex_actions=jnp.zeros((1, 2), dtype=jnp.float32)
    )

    current = jnp.ones(D)
    endpoint = jnp.zeros(D)

    ns_before = agent.plan(current, endpoint)['next_step']

    # Train for a few steps to change parameters
    for _ in range(20):
        obs = jax.random.normal(jax.random.PRNGKey(0), (64, D))
        targets = jax.random.normal(jax.random.PRNGKey(1), (64, D))
        alphas = jnp.linspace(0.0, 1.0, N + 1, dtype=jnp.float32).reshape(1, N + 1, 1)
        segment = obs[:, None, :] * (1.0 - alphas) + targets[:, None, :] * alphas
        batch = {
            'observations': obs,
            'next_observations': segment[:, 1, :],
            'high_actor_goals': targets * 0.1,
            'high_actor_targets': targets,
            'trajectory_segment': segment,
            'actions': jnp.zeros((64, 2), dtype=jnp.float32),
        }
        agent, _ = agent.update(batch)

    ns_after = agent.plan(current, endpoint)['next_step']

    diff = float(jnp.abs(ns_before - ns_after).max())
    print(f'  max |next_step_before - next_step_after| = {diff:.6f}')
    check(f'next_step changed after training (diff={diff:.6f})', diff > 1e-4)

    # Verify next_step differs for different endpoints (depends on x_0)
    ep_a = jnp.zeros(D)
    ep_b = jnp.ones(D) * 2.0
    ns_a = agent.plan(current, ep_a)['next_step']
    ns_b = agent.plan(current, ep_b)['next_step']
    ep_diff = float(jnp.abs(ns_a - ns_b).max())
    print(f'  max |ns(ep_a) - ns(ep_b)| = {ep_diff:.6f}')
    check(f'next_step depends on endpoint (diff={ep_diff:.6f})', ep_diff > 1e-4)


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

if __name__ == '__main__':
    print('=' * 60)
    print('GOUB dynamics — Smoke Test')
    print('=' * 60)

    test_schedule()
    test_bridge_sampling()
    test_posterior_mean()
    test_model_mean()
    test_agent_creation()
    test_agent_loss()
    test_loss_decreases()
    test_inference()
    test_next_step_is_learned()

    print('\n' + '=' * 60)
    print(f'Results:  {PASS} passed,  {FAIL} failed')
    print('=' * 60)
    if FAIL > 0:
        sys.exit(1)
