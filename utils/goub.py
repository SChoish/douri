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


_VALID_BRIDGE_TYPES = ('goub', 'unidb_gou', 'theta_linear')


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
            f'bridge_gamma must be > 0 for {bridge_type}, got {bridge_gamma!r}.'
        )
    return 1.0 / gamma



def _theta_linear_arrays(theta_fwd, g2_fwd, step_var_fwd, gamma_inv=0.0):
    """Exact linear-SDE bridge arrays in forward state time.

    This uses the exact per-step discretization

        r_{i+1} = exp(theta_i) r_i + eta_i,
        eta_i ~ N(0, exp(2 theta_i) step_var_i I).

    The returned arrays are converted to the legacy diffusion index ``n`` via
    ``i = N - n`` so existing calls ``bridge_w[n]`` / ``bridge_var[n]`` keep the
    same syntax as the GOUB helpers.
    """
    theta_fwd = jnp.asarray(theta_fwd, dtype=jnp.float32)
    g2_fwd = jnp.asarray(g2_fwd, dtype=jnp.float32)
    step_var_fwd = jnp.asarray(step_var_fwd, dtype=jnp.float32)
    N = int(theta_fwd.shape[0])
    A = jnp.exp(theta_fwd)
    q2 = step_var_fwd * A ** 2

    # P_i = Var[r_i | r_0 = 0].
    Ps = [jnp.asarray(0.0, dtype=jnp.float32)]
    for i in range(N):
        Ps.append(A[i] ** 2 * Ps[-1] + q2[i])
    P = jnp.stack(Ps)  # (N+1,)

    # Phi_{i:K} = prod_{l=i}^{K-1} A_l.
    Phis = [None] * (N + 1)
    Phis[N] = jnp.asarray(1.0, dtype=jnp.float32)
    for i in range(N - 1, -1, -1):
        Phis[i] = A[i] * Phis[i + 1]
    Phi = jnp.stack(Phis)

    # Omega_{i:K} = Var[r_K | r_i] with r_i fixed.
    Oms = [None] * (N + 1)
    Oms[N] = jnp.asarray(0.0, dtype=jnp.float32)
    for i in range(N - 1, -1, -1):
        Oms[i] = q2[i] * Phi[i + 1] ** 2 + Oms[i + 1]
    Omega = jnp.stack(Oms)

    gamma_inv_arr = jnp.asarray(gamma_inv, dtype=jnp.float32)
    denom = jnp.maximum(P[-1] + gamma_inv_arr, 1e-12)

    beta = P * Phi / denom
    bridge_var = P * (Omega + gamma_inv_arr) / denom
    bridge_var = jnp.maximum(bridge_var, 0.0)

    # Hard endpoint bridge should pin endpoints exactly.
    if float(gamma_inv) == 0.0:
        beta = beta.at[0].set(0.0).at[-1].set(1.0)
        bridge_var = bridge_var.at[0].set(0.0).at[-1].set(0.0)

    # Legacy index n corresponds to forward index i = N - n.
    return dict(
        theta_fwd=theta_fwd,
        g2_fwd=g2_fwd,
        step_var_fwd=step_var_fwd,
        theta_legacy=theta_fwd[::-1],
        g2_legacy=g2_fwd[::-1],
        step_var_legacy=step_var_fwd[::-1],
        theta_linear_A_fwd=A,
        theta_linear_A=A[::-1],
        theta_linear_q2_fwd=q2,
        theta_linear_q2=q2[::-1],
        theta_linear_P_fwd=P,
        theta_linear_P=P[::-1],
        theta_linear_phi_iK_fwd=Phi,
        theta_linear_phi_iK=Phi[::-1],
        theta_linear_omega_iK_fwd=Omega,
        theta_linear_omega_iK=Omega[::-1],
        theta_linear_beta_fwd=beta,
        theta_linear_bridge_w=beta[::-1],
        theta_linear_bridge_var_fwd=bridge_var,
        theta_linear_bridge_var=bridge_var[::-1],
    )

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

    theta_linear = {}
    if bridge_type == 'theta_linear':
        theta_linear = _theta_linear_arrays(theta[::-1], g2[::-1], step_var[::-1], gamma_inv=gamma_inv)
        # In theta-linear mode, arrays indexed by legacy n correspond to
        # forward state-time step i = N - n.  Pass reversed forward arrays so
        # schedule['theta'][n - 1] remains the original legacy theta_n.
        theta = theta_linear['theta_legacy']
        g2 = theta_linear['g2_legacy']
        step_var = theta_linear['step_var_legacy']
        bridge_w = theta_linear['theta_linear_bridge_w']
        bridge_var = theta_linear['theta_linear_bridge_var']

    out = dict(
        theta=theta, g2=g2, step_var=step_var,
        bar_theta=bar_theta, bar_sigma2=bar_sigma2,
        bar_theta_nN=bar_theta_nN, bar_sigma2_nN=bar_sigma2_nN,
        bar_sigma2_nN_gamma=bar_sigma2_nN_gamma,
        bar_sigma2_N=bar_sigma2_N,
        bar_sigma2_N_gamma=bar_sigma2_N_gamma,
        bridge_var=bridge_var, bridge_w=bridge_w,
        gamma_inv=gamma_inv_arr,
    )
    out.update(theta_linear)
    return out


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



def theta_linear_posterior_mean(x_n, x_0, x_T, n, schedule):
    """Exact posterior teacher for the exp-discretized linear-SDE bridge."""
    return posterior_mean(x_n, x_0, x_T, n, schedule)


def theta_linear_model_mean(x_n, x_0, x_T, eps_pred, n, schedule):
    """SDE/Euler model mean for theta-linear mode with explicit h-control."""
    k = n - 1
    theta_i = schedule['theta'][k][..., None]
    g2_i = schedule['g2'][k][..., None]
    phi_iK = schedule['theta_linear_phi_iK'][n][..., None]
    omega_iK = schedule['theta_linear_omega_iK'][n][..., None]
    gamma_inv = schedule.get('gamma_inv', jnp.asarray(0.0, dtype=jnp.float32))
    gamma_inv = jnp.asarray(gamma_inv, dtype=jnp.float32)
    bvar_i = schedule['bridge_var'][n][..., None]

    r_i = x_n - x_T
    delta = x_0 - x_T
    h_drift = g2_i * phi_iK / jnp.maximum(omega_iK + gamma_inv, 1e-12) * (delta - phi_iK * r_i)
    scale = g2_i / jnp.sqrt(jnp.maximum(bvar_i, 1e-12))
    return x_n + theta_i * r_i + h_drift - scale * eps_pred

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



def theta_linear_forward_bridge_coefficients(
    K: int,
    *,
    beta_min: float,
    beta_max: float,
    lambda_: float,
    eps: float = 1.0e-6,
):
    """Closed-form forward bridge marginals for theta-linear mode."""
    if K < 1:
        raise ValueError(f'K must be >= 1, got {K}.')
    K_int = int(K)
    steps = jnp.arange(1, K_int + 1, dtype=jnp.float32)
    theta = beta_min / float(K_int) + (beta_max - beta_min) * steps / float(K_int * K_int)
    g2 = 2.0 * float(lambda_) ** 2 * theta
    step_var = float(lambda_) ** 2 * (1.0 - jnp.exp(-2.0 * theta))
    arr = _theta_linear_arrays(theta, g2, step_var, gamma_inv=0.0)
    b = arr['theta_linear_beta_fwd']
    std = jnp.sqrt(jnp.maximum(arr['theta_linear_bridge_var_fwd'], 0.0))
    a = 1.0 - b
    a = a.at[0].set(1.0).at[-1].set(0.0)
    b = b.at[0].set(0.0).at[-1].set(1.0)
    std = std.at[0].set(0.0).at[-1].set(0.0)
    return a, b, std

def forward_bridge_coefficients(
    K: int,
    *,
    beta_min: float,
    beta_max: float,
    lambda_: float,
    eps: float = 1.0e-6,
    bridge_type: str = 'goub',
):
    """Closed-form GOUB *forward* bridge coefficients in DOURI ``z_0 -> z_K`` form.

    Returns three arrays ``a, b, std`` of shape ``(K + 1,)`` such that

        z_i | z_0, z_K ~ N( a_i * z_0 + b_i * z_K , std_i^2 * I )

    using the GOUB Proposition 3.1 formulae with the same per-step OU rate
    schedule as :func:`make_goub_schedule` (linear-beta, ``delta_l = 1``):

        theta_l         = beta_min / N + (beta_max - beta_min) * l / N^2  (l = 1..N)
        bar_theta_{0:i} = sum_{l=1}^{i} theta_l
        bar_sigma2_{a:b}^2 = lambda^2 * (1 - exp(-2 * bar_theta_{a:b}))

        a_i   = exp(-bar_theta_{0:i}) * bar_sigma2_{i:K}^2 / bar_sigma2_{0:K}^2
        b_i   = (1 - exp(-bar_theta_{0:i})) * bar_sigma2_{i:K}^2 / bar_sigma2_{0:K}^2
              + exp(-2 * bar_theta_{i:K}) * bar_sigma2_{0:i}^2 / bar_sigma2_{0:K}^2
        std_i^2 = bar_sigma2_{0:i}^2 * bar_sigma2_{i:K}^2 / bar_sigma2_{0:K}^2

    The endpoints are *clamped* exactly after the numerical computation so that

        a[0]  = 1, b[0]  = 0, std[0]  = 0
        a[-1] = 0, b[-1] = 1, std[-1] = 0

    are guaranteed regardless of ``eps`` / floating-point error.

    The convention here is the DOURI state-space *forward* time direction:
    index 0 = current state ``z_0``, index K = subgoal ``z_K``.  The
    ``make_goub_schedule`` arrays use the GOUB diffusion-time index
    ``n = 0`` = clean endpoint, ``n = N`` = noisy endpoint, so we re-derive
    ``theta`` / ``bar_theta`` here for clarity rather than reusing the
    pre-computed schedule directly (this also lets callers re-coefficient at
    a smaller ``K`` than ``goub_N`` without re-running the schedule).
    """
    if K < 1:
        raise ValueError(f'K must be >= 1, got {K}.')
    if str(bridge_type).lower() == 'theta_linear':
        return theta_linear_forward_bridge_coefficients(
            K,
            beta_min=beta_min,
            beta_max=beta_max,
            lambda_=lambda_,
            eps=eps,
        )

    K_int = int(K)
    steps = jnp.arange(1, K_int + 1, dtype=jnp.float32)
    theta = beta_min / float(K_int) + (beta_max - beta_min) * steps / float(K_int * K_int)

    bar_theta = jnp.concatenate([jnp.zeros(1, dtype=jnp.float32), jnp.cumsum(theta)])  # (K+1,)
    bar_total = bar_theta[-1]
    bar_theta_iK = bar_total - bar_theta  # (K+1,)

    lam2 = float(lambda_) ** 2
    bar_sigma2_0i = lam2 * (1.0 - jnp.exp(-2.0 * bar_theta))
    bar_sigma2_iK = lam2 * (1.0 - jnp.exp(-2.0 * bar_theta_iK))
    bar_sigma2_0K = bar_sigma2_0i[-1]
    denom = jnp.maximum(bar_sigma2_0K, jnp.asarray(eps, dtype=jnp.float32))

    exp_neg_0i = jnp.exp(-bar_theta)
    exp_neg2_iK = jnp.exp(-2.0 * bar_theta_iK)

    a = exp_neg_0i * bar_sigma2_iK / denom
    b = (1.0 - exp_neg_0i) * bar_sigma2_iK / denom + exp_neg2_iK * bar_sigma2_0i / denom
    std2 = bar_sigma2_0i * bar_sigma2_iK / denom
    std = jnp.sqrt(jnp.maximum(std2, 0.0))

    # Exact endpoint clamp - both for numerical safety and to match the
    # mathematical bridge definition independent of ``eps``.
    a = a.at[0].set(1.0).at[-1].set(0.0)
    b = b.at[0].set(0.0).at[-1].set(1.0)
    std = std.at[0].set(0.0).at[-1].set(0.0)

    return a, b, std
