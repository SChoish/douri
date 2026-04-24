"""GOUB-inspired bridge diffusion schedule and math helpers.

This module implements bridge diffusion helpers inspired by the Generalized
Ornstein-Uhlenbeck Bridge (GOUB) framework.  It is **not** a paper-exact
reproduction of any specific GOUB publication.  Differences:

* **Schedule** -- standard linear-beta diffusion schedule.
* **g_n^2** -- uses the continuous-time approximation ``2 lambda^2 theta_n``
  (stored as ``g2``) rather than the exact discrete variance
  ``lambda^2 (1 - exp(-2 theta_n))`` (stored as ``step_var``).
  ``step_var`` is used only inside the analytic posterior for accuracy.
* **model_mean** -- follows the GOUB epsilon-to-mean conversion and is
  numerically valid for n in {1, ..., N-1}.  At the boundary n = N the
  formula is singular (bridge_var[N] = 0); the agent handles that step
  with a learned residual parameterisation instead.

Indexing conventions
--------------------
* Diffusion steps n = 0 (clean endpoint x_0) to N (noisy start x_T).
* Per-step arrays ``theta``, ``g2``, ``step_var``:
  shape (N,), index k corresponds to step n = k + 1.
* Cumulative arrays ``bar_theta``, ``bar_sigma2``, ``bar_theta_nN``,
  ``bar_sigma2_nN``, ``bridge_var``, ``bridge_w``:
  shape (N+1,), index n corresponds to step n.

UniDB-GOU soft bridge
---------------------
Vanilla GOUB corresponds to the Doob h-transform limit ``gamma -> infinity``
of the UniDB family.  Setting ``bridge_type='unidb_gou'`` with a finite
``bridge_gamma`` switches to the soft-bridge variant by replacing every
``bar_sigma_{n:N}^2`` denominator with ``gamma^{-1} + bar_sigma_{n:N}^2``
(and similarly for ``bar_sigma_N^2`` in the bridge interpolation weight).
Per the design note, only the *means* (``bridge_w`` and the drift coefficient
in ``model_mean``) absorb the soft-bridge denominator; the conditional
``bridge_var`` keeps the gamma=infinity expression for sampling stability.
With ``bridge_type='goub'`` (default) the schedule reproduces the original
behaviour bit-for-bit.
"""

import jax
import jax.numpy as jnp


_VALID_BRIDGE_TYPES = ('goub', 'unidb_gou')


def _resolve_gamma_inv(bridge_type: str, bridge_gamma: float) -> float:
    """Map (bridge_type, bridge_gamma) to the soft-bridge denominator offset."""
    if bridge_type not in _VALID_BRIDGE_TYPES:
        raise ValueError(
            f"bridge_type must be one of {_VALID_BRIDGE_TYPES}, got {bridge_type!r}."
        )
    if bridge_type == 'goub':
        return 0.0
    gamma = float(bridge_gamma)
    if not (gamma > 0.0):
        raise ValueError(
            f'bridge_gamma must be > 0 for unidb_gou, got {bridge_gamma!r}.'
        )
    return 1.0 / gamma


def make_goub_schedule(
    N: int,
    beta_min: float = 0.1,
    beta_max: float = 20.0,
    lambda_: float = 1.0,
    bridge_type: str = 'goub',
    bridge_gamma: float = 1.0e7,
):
    """Precompute all GOUB-inspired schedule quantities.

    Args:
        N: number of diffusion steps.
        beta_min, beta_max, lambda_: linear-beta OU schedule parameters.
        bridge_type: ``'goub'`` (Doob h-transform, ``gamma -> infinity``) or
            ``'unidb_gou'`` (UniDB-GOU soft bridge with finite ``gamma``).
        bridge_gamma: finite gamma for ``'unidb_gou'``; ignored when
            ``bridge_type='goub'``.
    """
    gamma_inv = _resolve_gamma_inv(bridge_type, bridge_gamma)

    steps = jnp.arange(1, N + 1, dtype=jnp.float32)

    # Per-step OU rate (linear schedule)
    theta = beta_min / N + (beta_max - beta_min) * steps / (N * N)  # (N,)

    # Approximate per-step diffusion variance (model mean & loss weight)
    g2 = 2.0 * lambda_ ** 2 * theta  # (N,)

    # Exact per-step transition variance (analytic posterior only)
    step_var = lambda_ ** 2 * (1.0 - jnp.exp(-2.0 * theta))  # (N,)

    # Cumulative OU rate  bar_theta[n] = sum_{i=1}^{n} theta_i
    bar_theta = jnp.concatenate([jnp.zeros(1), jnp.cumsum(theta)])  # (N+1,)

    # Marginal variance  bar_sigma^2[n] = lambda^2 (1 - exp(-2 bar_theta_n))
    bar_sigma2 = lambda_ ** 2 * (1.0 - jnp.exp(-2.0 * bar_theta))  # (N+1,)

    # Remaining quantities  bar_theta_{n:N} = bar_theta_N - bar_theta_n
    bar_theta_nN = bar_theta[-1] - bar_theta  # (N+1,)
    bar_sigma2_nN = lambda_ ** 2 * (1.0 - jnp.exp(-2.0 * bar_theta_nN))  # (N+1,)
    bar_sigma2_N = bar_sigma2[-1]  # scalar

    # Soft-bridge denominators: gamma_inv == 0 reproduces vanilla GOUB.
    gamma_inv_arr = jnp.asarray(gamma_inv, dtype=jnp.float32)
    bar_sigma2_nN_gamma = gamma_inv_arr + bar_sigma2_nN  # (N+1,)
    bar_sigma2_N_gamma = gamma_inv_arr + bar_sigma2_N    # scalar

    # Bridge variance kept at the gamma=infinity form for sampling stability.
    bridge_var = bar_sigma2 * bar_sigma2_nN / jnp.maximum(bar_sigma2_N, 1e-12)  # (N+1,)
    # Bridge interpolation weight uses the soft-bridge denominator.
    bridge_w = bar_sigma2_nN_gamma * jnp.exp(-bar_theta) / jnp.maximum(bar_sigma2_N_gamma, 1e-12)  # (N+1,)

    return dict(
        theta=theta, g2=g2, step_var=step_var,
        bar_theta=bar_theta, bar_sigma2=bar_sigma2,
        bar_theta_nN=bar_theta_nN, bar_sigma2_nN=bar_sigma2_nN,
        bar_sigma2_nN_gamma=bar_sigma2_nN_gamma,
        bar_sigma2_N=bar_sigma2_N,
        bar_sigma2_N_gamma=bar_sigma2_N_gamma,
        bridge_var=bridge_var, bridge_w=bridge_w,
        gamma_inv=gamma_inv_arr,
    )


def bridge_sample(x_0, x_T, n, schedule, rng):
    """Sample x_n from the forward bridge q(x_n | x_0, x_T).

    Args:
        x_0: Target endpoint, shape (B, D).
        x_T: Current state, shape (B, D).
        n: Step indices, shape (B,), values in {1, ..., N-1}.
        schedule: Output of ``make_goub_schedule``.
        rng: JAX PRNG key.

    Returns:
        x_n of shape (B, D).
    """
    w = schedule['bridge_w'][n][..., None]
    var = schedule['bridge_var'][n][..., None]
    mean = w * x_0 + (1.0 - w) * x_T
    return mean + jnp.sqrt(jnp.maximum(var, 1e-12)) * jax.random.normal(rng, x_0.shape)


def posterior_mean(x_n, x_0, x_T, n, schedule):
    """Analytic posterior mean  E[x_{n-1} | x_n, x_0, x_T].

    Combines the one-step reverse ``N(mu_back, sigma_back^2)`` with the bridge
    marginal at step n-1 ``N(bridge_mean_{n-1}, bridge_var_{n-1})`` via Bayes.

    Valid for n in {1, ..., N}.  At n = 1 the bridge variance at step 0 is
    zero, so the posterior collapses to x_0 as expected.

    Args:
        x_n: Current bridge state, shape (B, D).
        x_0: Target endpoint, shape (B, D).
        x_T: Current observation, shape (B, D).
        n: Step indices, shape (B,), values in {1, ..., N}.
        schedule: Output of ``make_goub_schedule``.

    Returns:
        mu_{n-1} of shape (B, D).
    """
    k = n - 1  # 0-based index into (N,) arrays
    theta_n = schedule['theta'][k][..., None]
    svar = schedule['step_var'][k][..., None]

    exp_t = jnp.exp(theta_n)
    mu_back = exp_t * x_n - (exp_t - 1.0) * x_T
    sig_back2 = svar * exp_t ** 2

    nm1 = n - 1  # step n-1 for (N+1,) arrays
    bvar = schedule['bridge_var'][nm1][..., None]
    bw = schedule['bridge_w'][nm1][..., None]
    bmean = bw * x_0 + (1.0 - bw) * x_T

    return (bvar * mu_back + sig_back2 * bmean) / (sig_back2 + bvar + 1e-12)


def model_mean(x_n, x_T, eps_pred, n, schedule):
    """Parameterised posterior mean  mu_theta_{n-1}  (GOUB / UniDB-GOU).

    ::

        mu_theta = x_n
                 - (theta_n + g_n^2 exp(-2 bar_theta_{n:N})
                              / (gamma^{-1} + bar_sigma_{n:N}^2)) (x_T - x_n)
                 - (g_n^2 / sqrt(bridge_var_n)) eps_theta

    With ``gamma^{-1} = 0`` (i.e. ``bridge_type='goub'``) this collapses to the
    vanilla GOUB drift; with finite ``gamma`` it implements the UniDB-GOU
    soft-bridge drift.  ``bridge_var`` itself is kept at the gamma=infinity
    expression for sampling stability.

    **Valid for n in {1, ..., N-1} only.**  At n = N the bridge variance is
    zero, making the eps coefficient singular.  The agent handles that
    boundary with a separate residual parameterisation.

    Args:
        x_n: Current state, shape (B, D).
        x_T: Bridge endpoint, shape (B, D).
        eps_pred: Network output, shape (B, D).
        n: Step indices, shape (B,), values in {1, ..., N-1}.
        schedule: Output of ``make_goub_schedule``.

    Returns:
        mu_theta_{n-1} of shape (B, D).
    """
    k = n - 1
    theta_n = schedule['theta'][k][..., None]
    g2_n = schedule['g2'][k][..., None]

    bt_nN = schedule['bar_theta_nN'][n][..., None]
    bs2_nN_gamma = schedule['bar_sigma2_nN_gamma'][n][..., None]
    bsp2 = schedule['bridge_var'][n][..., None]
    bsp = jnp.sqrt(jnp.maximum(bsp2, 1e-12))

    drift = theta_n + g2_n * jnp.exp(-2.0 * bt_nN) / jnp.maximum(bs2_nN_gamma, 1e-12)
    return x_n - drift * (x_T - x_n) - (g2_n / bsp) * eps_pred


def reverse_std(n, schedule):
    """Return sqrt(g_n^2) for reverse sampling.

    Args:
        n: Step indices, shape (B,), values in {1, ..., N}.
        schedule: Output of ``make_goub_schedule``.

    Returns:
        std of shape (B, 1).
    """
    k = n - 1
    g2_n = schedule['g2'][k][..., None]
    return jnp.sqrt(jnp.maximum(g2_n, 1e-12))


def sample_from_reverse_mean(mu, n, schedule, rng, noise_scale=1.0):
    """Sample x_{n-1} ~ N(mu, noise_scale^2 * g_n^2 I).

    Args:
        mu: Reverse-step mean, shape (B, D).
        n: Step indices, shape (B,), values in {1, ..., N}.
        schedule: Output of ``make_goub_schedule``.
        rng: JAX PRNG key.
        noise_scale: Optional temperature multiplier (scalar or 0-dim array).

    Returns:
        Sampled x_{n-1}, shape (B, D).
    """
    std = reverse_std(n, schedule)
    noise = jax.random.normal(rng, mu.shape)
    ns = jnp.asarray(noise_scale, dtype=jnp.float32)
    return mu + ns * std * noise
