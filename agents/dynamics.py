"""Linear-SDE dynamics agent and shared planner components.

Single source of truth for the dynamics model used by training and
inference. Training uses the path-supervised dynamics objective on top of
the endpoint-conditioned bridge planner.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from agents.critic import ScalarValueNet
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.dynamics import (
    bridge_sample,
    exact_residual_model_mean,
    forward_bridge_coefficients,
    make_dynamics_schedule,
    posterior_mean,
    sample_from_reverse_mean,
)
from utils.goal_representation import assert_phi_goal_obs_indices, goal_representation, normalize_phi_goal_obs_indices


_VALID_PLANNER_TYPES = ('forward_bridge_residual',)
_VALID_FORWARD_BRIDGE_MODES = ('mean', 'sample')
_VALID_DYNAMICS_MODEL_TYPES = ('exact_residual',)


def _planner_type(config) -> str:
    """Return the canonical planner_type string from the agent config."""
    pt = str(config.get('planner_type', 'forward_bridge_residual')).lower()
    if pt not in _VALID_PLANNER_TYPES:
        raise ValueError(
            f'planner_type must be one of {_VALID_PLANNER_TYPES}, got {pt!r}.'
        )
    return pt


def _forward_bridge_mode(config) -> str:
    mode = str(config.get('forward_bridge_mode', 'mean')).lower()
    if mode not in _VALID_FORWARD_BRIDGE_MODES:
        raise ValueError(
            f'forward_bridge_mode must be one of {_VALID_FORWARD_BRIDGE_MODES}, got {mode!r}.'
        )
    return mode


def _dynamics_model_type(config) -> str:
    """Return the canonical dynamics_model_type string from the agent config.

    ``exact_residual`` uses the exact bridge posterior mean as the base transition
      with a learned data residual; trained via path/rollout consistency.
    """
    mode = str(config.get('dynamics_model_type', 'exact_residual')).lower()
    if mode not in _VALID_DYNAMICS_MODEL_TYPES:
        raise ValueError(
            f'dynamics_model_type must be one of {_VALID_DYNAMICS_MODEL_TYPES}, got {mode!r}.'
        )
    return mode


def _dynamics_model_type_metric(config) -> float:
    _dynamics_model_type(config)
    return 1.0


from utils.inverse_dynamics import InverseDynamicsMLP, parse_hidden_dims
from utils.networks import MLP


class SinusoidalEmbedding(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, t):
        half = self.dim // 2
        freq = jnp.exp(-jnp.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / half)
        args = t[..., None].astype(jnp.float32) * freq
        return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


class ResidualNet(nn.Module):
    """Reverse-chain residual network.

    State-time notation:
        ``r_0 = 0``, ``r_K = Delta = s_{t+K} - s_t``.
    Reverse-chain implementation notation:
        ``x_T = r_0`` and ``x_0 = r_K``.

    Public training/planning paths pass the absolute current state ``s_t``
    through the dedicated ``anchor`` channel.  In ``absolute`` mode this is
    redundant with ``x_T`` (both equal ``s_t``); in ``displacement`` mode it is
    the only path through which ``s_t`` reaches the network, matching the PDF's
    ``M_r(r_i, s_t, Delta)`` interface and preventing the learned correction
    from being forced to be translation-invariant.
    """

    hidden_dims: Sequence[int]
    state_dim: int
    time_embed_dim: int = 64
    layer_norm: bool = True

    @nn.compact
    def __call__(self, x_n, x_T, x_0, anchor, n):
        t_emb = SinusoidalEmbedding(self.time_embed_dim)(n)
        inp = jnp.concatenate([x_n, x_T, x_0, anchor, t_emb], axis=-1)
        return MLP(
            hidden_dims=(*self.hidden_dims, self.state_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )(inp)


class SubgoalEstimatorNet(nn.Module):
    """Deterministic point subgoal estimator."""

    hidden_dims: Sequence[int]
    state_dim: int
    layer_norm: bool = True
    goal_representation: str = 'full'
    phi_goal_obs_indices: tuple[int, ...] = ()
    env_name: str = ''

    @nn.compact
    def __call__(self, observations, high_actor_goals):
        goal_inp = goal_representation(
            high_actor_goals,
            self.goal_representation,
            self.phi_goal_obs_indices,
            env_name=self.env_name,
        )
        inp = jnp.concatenate([observations, goal_inp], axis=-1)
        return MLP(
            hidden_dims=(*self.hidden_dims, self.state_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )(inp)


class DistributionalSubgoalEstimatorNet(nn.Module):
    """Diagonal-Gaussian subgoal estimator (``subgoal_distribution='diag_gaussian'``).

    Returns ``(mu, log_std)`` for the raw subgoal frame (absolute
    ``s_{t+K}`` in legacy mode, displacement ``Delta`` in displacement mode).
    The phase-1 implementation intentionally supports two stochastic losses:
    the PDF-style reparameterized sample MSE and an NLL option used by some
    configs; both can include the optional ``- alpha * V(sample_abs, g)``
    bonus after mapping a displacement sample back to absolute state space.
    Actor proposals sample endpoint candidates from this distribution in
    :meth:`_DynamicsAgentCore.sample_subgoal_candidates`.
    """

    hidden_dims: Sequence[int]
    state_dim: int
    layer_norm: bool = True
    log_std_min: float = -5.0
    log_std_max: float = 1.0
    goal_representation: str = 'full'
    phi_goal_obs_indices: tuple[int, ...] = ()
    env_name: str = ''

    @nn.compact
    def __call__(self, observations, high_actor_goals):
        goal_inp = goal_representation(
            high_actor_goals,
            self.goal_representation,
            self.phi_goal_obs_indices,
            env_name=self.env_name,
        )
        inp = jnp.concatenate([observations, goal_inp], axis=-1)
        trunk = MLP(
            hidden_dims=tuple(self.hidden_dims),
            activate_final=True,
            layer_norm=self.layer_norm,
        )(inp)
        mu = nn.Dense(self.state_dim, name='mu_head')(trunk)
        log_std_raw = nn.Dense(self.state_dim, name='log_std_head')(trunk)
        log_std = jnp.clip(log_std_raw, self.log_std_min, self.log_std_max)
        return mu, log_std


def _subgoal_mode(config) -> str:
    return str(config.get('subgoal_distribution', 'deterministic')).lower()


def _subgoal_stochastic_loss(config) -> str:
    return str(config.get('subgoal_stochastic_loss', 'mse')).lower()


def _subgoal_target_mode(config) -> str:
    """Target representation for the subgoal estimator and the bridge.

    - ``'absolute'`` (default, legacy): subgoal_net predicts the absolute
      next-K state ``s_{t+K}``; the bridge interpolates between ``s_t`` and
      ``s_{t+K}`` in absolute state space.
    - ``'displacement'``: PDF-aligned translated chart.  The subgoal_net
      predicts ``Delta = s_{t+K} - s_t`` and the bridge/residual model is
      trained in the local frame ``r_i = s_{t+i} - s_t``.  External APIs still
      hand off absolute states / trajectories; the displacement frame only
      exists inside :class:`DynamicsAgent`.
    """
    mode = str(config.get('subgoal_target_mode', 'absolute')).lower()
    if mode not in ('absolute', 'displacement'):
        raise ValueError(
            "subgoal_target_mode must be 'absolute' or 'displacement', "
            f'got {mode!r}.'
        )
    return mode


def _is_displacement_mode(config) -> bool:
    return _subgoal_target_mode(config) == 'displacement'

def _residual_target_mode(config) -> str:
    """Target representation for residual/bridge inputs (independent of subgoal mode)."""
    mode = str(config.get('residual_target_mode', 'absolute')).lower()
    if mode not in ('absolute', 'displacement'):
        raise ValueError(
            "residual_target_mode must be 'absolute' or 'displacement', "
            f'got {mode!r}.'
        )
    return mode


def _subgoal_target_mode_id(config) -> float:
    """Scalar id for CSV/W&B logging. 0 = absolute, 1 = displacement."""
    return 1.0 if _is_displacement_mode(config) else 0.0


class _DynamicsAgentCore(flax.struct.PyTreeNode):
    """Shared linear-SDE dynamics planner / inference core."""

    rng: Any
    network: Any
    schedule: Any
    config: Any = nonpytree_field()

    # ------------------------------------------------------------------
    # Displacement-frame helpers (``subgoal_target_mode='displacement'``)
    # ------------------------------------------------------------------
    #
    # In displacement mode the bridge / residual net are trained and queried in
    # the local frame ``s' = s - s_t`` so that ``s_T' = 0`` and ``s_0' = Delta``.
    # External APIs (``plan``, ``predict_subgoal``, etc.) still expose absolute
    # states; the helpers below centralise the affine shift.

    def _is_displacement_mode(self) -> bool:
        return _is_displacement_mode(self.config)

    def _is_residual_displacement_mode(self) -> bool:
        return _residual_target_mode(self.config) == 'displacement'

    def _displacement_origin(self, current_state: jnp.ndarray) -> jnp.ndarray:
        """Origin to subtract for the displacement frame.

        Returns ``current_state`` when ``residual_target_mode='displacement'``
        and a zero tensor otherwise (so callers can unconditionally subtract).
        """
        if self._is_residual_displacement_mode():
            return jnp.asarray(current_state, dtype=jnp.float32)
        return jnp.zeros_like(jnp.asarray(current_state, dtype=jnp.float32))

    def _bridge_anchor(self, current_state_abs: jnp.ndarray) -> jnp.ndarray:
        """Anchor input passed to the residual nets - always the absolute ``s_t``.

        Displacement mode shifts the bridge inputs to ``x_T = 0`` and
        ``x_0 = Delta``, which would otherwise hide ``s_t`` from the residual
        net.  Feeding ``s_t`` as a dedicated ``anchor`` channel restores that
        information.  In absolute mode the anchor is redundant with ``x_T``
        (both equal ``s_t``) but keeping the input always defined keeps the
        network shape stable across modes and removes a mode-specific code
        path.
        """
        return jnp.asarray(current_state_abs, dtype=jnp.float32)

    def _subgoal_abs_from_raw(self, observations: jnp.ndarray, raw: jnp.ndarray) -> jnp.ndarray:
        """Map raw ``subgoal_net`` output to an absolute next-K state.

        ``raw`` is interpreted as ``Delta`` in displacement mode and as
        ``s_{t+K}`` in absolute mode.
        """
        if self._is_displacement_mode():
            return jnp.asarray(observations, dtype=raw.dtype) + raw
        return raw

    def _subgoal_target_for_loss(
        self, observations: jnp.ndarray, target_abs: jnp.ndarray,
    ) -> jnp.ndarray:
        """Convert the absolute ``high_actor_targets`` into the subgoal-net's loss frame."""
        if self._is_displacement_mode():
            return jnp.asarray(target_abs, dtype=jnp.float32) - jnp.asarray(observations, dtype=jnp.float32)
        return jnp.asarray(target_abs, dtype=jnp.float32)

    def _learned_reverse_mean(
        self, x_n, x_T, x_0, n, schedule, goal=None, params=None, anchor=None,
    ):
        if anchor is None:
            # Backward-compatible low-level fallback for legacy private calls.
            # Public planners and training paths pass an explicit anchor; in
            # displacement mode callers should use ``plan`` / ``sample_plan`` so
            # the unshifted ``s_t`` is threaded through this channel.
            anchor = jnp.zeros_like(x_T)
        eps = self.network.select('residual_net')(
            x_n, x_T, x_0, anchor, n.astype(jnp.float32), params=params,
        )
        # Exact bridge posterior mean as base transition + learned residual.
        # Valid for all n in {1, ..., N}; no boundary override needed because
        # posterior_moments(...) already handles the terminal step.
        mu, _, _ = exact_residual_model_mean(
            x_n,
            x_0,
            x_T,
            eps,
            n,
            schedule,
            residual_scale=float(self.config.get('exact_residual_scale', 1.0)),
        )
        return mu, eps

    def _reverse_step(
        self, x_n, x_T, x_0, n, rng, stochastic, noise_scale, params=None, goal=None, anchor=None,
    ):
        mu, eps = self._learned_reverse_mean(
            x_n, x_T, x_0, n, self.schedule, goal=goal, params=params, anchor=anchor,
        )
        ns = jnp.asarray(noise_scale, dtype=jnp.float32)

        def take_sample(_):
            return sample_from_reverse_mean(mu, n, self.schedule, rng, noise_scale=ns)

        def take_mean(_):
            return mu

        x_new = jax.lax.cond(jnp.asarray(stochastic, dtype=jnp.bool_), take_sample, take_mean, operand=None)
        return x_new, eps

    def _idm_loss_term(self, batch, grad_params):
        idm_w = float(self.config.get('idm_loss_weight', 1.0))
        if idm_w <= 0.0:
            return jnp.array(0.0), jnp.array(0.0)
        o = jnp.asarray(batch['observations'], dtype=jnp.float32)
        on = jnp.asarray(batch['next_observations'], dtype=jnp.float32)
        a = jnp.asarray(batch['actions'], dtype=jnp.float32)
        pred = self.network.select('idm_net')(o, on, params=grad_params)
        mse = jnp.mean((pred - a) ** 2)
        return idm_w * mse, mse

    @jax.jit
    def update(self, batch, critic_value_params=None):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng, critic_value_params=critic_value_params)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    # ------------------------------------------------------------------
    # Forward-bridge planner (closed-form, endpoint-conditioned)
    # ------------------------------------------------------------------
    #
    # For ``planner_type='forward_bridge_residual'`` (default) ``plan()`` /
    # ``sample_plan()`` take the learned residual reverse chain.  For
    # ``planner_type='forward_bridge'`` and ``planner_type='forward_bridge_residual'``
    # they instead branch into the closed-form Gaussian bridge below.  The
    # convention follows DOURI's state-space *forward* time direction:
    #
    #     z_0 = current state
    #     z_K = subgoal endpoint
    #     trajectory[:, 0] = z_0
    #     trajectory[:, K] = z_K
    #
    # which already matches the existing reverse-chain output convention
    # (``traj[:, 0] = current_state`` and ``traj[:, -1]`` ~ predicted subgoal),
    # so downstream consumers (IDM action decoding, rollout/subgoal.py, etc.)
    # can be re-used without changes.

    def forward_bridge_coefficients(self, K, *, bridge_gamma_inv: float | None = None):
        """Return ``(a, b, std)`` of shape ``(K + 1,)`` for the linear-SDE forward bridge.

        Coefficients satisfy ``z_i | z_0, z_K ~ N(a_i z_0 + b_i z_K, std_i^2 I)``
        with the exact endpoint constraints
        ``a[0]=1, b[0]=0, std[0]=0`` and ``a[K]=0, b[K]=1, std[K]=0``.

        ``K`` is treated as a *static* Python int (it determines the size of
        the returned arrays). ``bridge_gamma_inv`` defaults to the agent config,
        so forward bridge planners use the same finite-gamma setting as the
        reverse-score bridge math.
        """
        gamma_inv = (
            float(self.config.get('bridge_gamma_inv', 0.0))
            if bridge_gamma_inv is None else float(bridge_gamma_inv)
        )
        return forward_bridge_coefficients(
            int(K),
            beta_min=float(self.config['dynamics_beta_min']),
            beta_max=float(self.config['dynamics_beta_max']),
            lambda_=float(self.config['dynamics_lambda']),
            bridge_gamma_inv=gamma_inv,
            theta_schedule=str(self.config.get('theta_schedule', 'linear_beta')),
            theta_total=float(self.config.get('theta_total', 1.0)),
            progress_alpha=float(self.config.get('progress_alpha', 0.8)),
        )

    def forward_bridge_plan(
        self,
        z0: jnp.ndarray,
        zK: jnp.ndarray,
        *,
        sample: bool = False,
        noise_scale: float = 0.0,
        num_steps: int | None = None,
        rng=None,
    ) -> jnp.ndarray:
        """Closed-form forward-bridge path ``z_0 -> z_K``.

        Args:
            z0: ``(B, state_dim)`` current state.
            zK: ``(B, state_dim)`` subgoal endpoint.
            sample: if ``True`` add per-step Gaussian noise scaled by ``std_i``.
                **Note**: this samples each marginal independently and is *not*
                an exact correlated bridge sample - it is intended only as an
                ablation / regularisation lever.  Endpoints are clamped after
                noise injection so ``path[:, 0] = z0`` and ``path[:, -1] = zK``
                always hold.
            noise_scale: scalar noise multiplier (``0`` falls back to mean).
            num_steps: planning horizon ``K``; defaults to ``self.config['dynamics_N']``.
            rng: required when ``sample=True`` and ``noise_scale > 0``.

        Returns:
            ``path`` of shape ``(B, K + 1, state_dim)``.
        """
        n_total = int(self.config['dynamics_N'])
        K = n_total if num_steps is None else int(num_steps)
        if K < 1 or K > n_total:
            raise ValueError(f'num_steps must be in [1, {n_total}], got {K}.')

        a, b, std = self.forward_bridge_coefficients(K)
        mu = a[None, :, None] * z0[:, None, :] + b[None, :, None] * zK[:, None, :]

        if sample and float(noise_scale) > 0.0:
            if rng is None:
                raise ValueError('forward_bridge_plan(sample=True, noise_scale>0) requires an rng.')
            noise = jax.random.normal(rng, mu.shape)
            path = mu + jnp.asarray(noise_scale, dtype=jnp.float32) * std[None, :, None] * noise
            # Endpoint clamp - keeps ``path[:, 0] = z0`` and ``path[:, -1] = zK``
            # exact even after the independent-marginal noise injection.
            path = path.at[:, 0, :].set(z0).at[:, -1, :].set(zK)
        else:
            path = mu

        return path

    def forward_bridge_residual_plan(
        self,
        z0: jnp.ndarray,
        zK: jnp.ndarray,
        *,
        sample: bool = False,
        noise_scale: float = 0.0,
        num_steps: int | None = None,
        rng=None,
        params=None,
        anchor=None,
    ) -> jnp.ndarray:
        """Forward bridge mean + endpoint-preserving learned residual.

        ``z_hat_i = mu_i + w_i * r_theta(z_0, z_K, anchor, i)`` where
        ``mu_i = a_i z_0 + b_i z_K`` is the closed-form bridge mean,
        ``r_theta`` is :class:`PathResidualNet`, and the quadratic schedule
        ``w_i = i*(K - i)/K^2`` zeros out at the endpoints so ``z_hat_0 = z_0``
        and ``z_hat_K = z_K`` are preserved exactly (modulo the explicit
        endpoint clamp at the end).  ``anchor`` is the absolute current state
        used to condition the path-residual MLP; in displacement mode it equals
        ``s_t`` so that translation invariance is *not* enforced.
        """
        n_total = int(self.config['dynamics_N'])
        K = n_total if num_steps is None else int(num_steps)
        if K < 1 or K > n_total:
            raise ValueError(f'num_steps must be in [1, {n_total}], got {K}.')

        a, b, std = self.forward_bridge_coefficients(K)
        mu = a[None, :, None] * z0[:, None, :] + b[None, :, None] * zK[:, None, :]

        idx = jnp.arange(K + 1, dtype=jnp.float32)
        w = idx * (float(K) - idx) / float(K * K)
        t_norm = jnp.broadcast_to(idx[None, :] / float(K), (z0.shape[0], K + 1))

        if anchor is None:
            if self._is_residual_displacement_mode():
                raise ValueError(
                    'forward_bridge_residual_plan in displacement mode requires '
                    'anchor=s_t (the absolute current state).  Use plan() / '
                    'sample_plan() unless this is an intentional low-level call.'
                )
            # Absolute mode: ``z0`` already equals the current state ``s_t``,
            # so it is the natural anchor fallback.
            anchor = z0
        # Reuse the shared residual_net with inputs R(s_1, delta, i/K):
        # x_n := forward-bridge mean state at step i, x_T := s_1(anchor),
        # x_0 := delta(zK), n := i.
        n_idx = jnp.broadcast_to(jnp.arange(K + 1, dtype=jnp.int32)[None, :], t_norm.shape)
        residual = jax.vmap(
            lambda mu_b, anchor_b, zK_b, n_b: self.network.select('residual_net')(
                mu_b, anchor_b, zK_b, anchor_b, n_b, params=params,
            ),
            in_axes=(0, 0, 0, 0),
        )(mu, anchor, zK, n_idx)
        path = mu + w[None, :, None] * residual

        if sample and float(noise_scale) > 0.0:
            if rng is None:
                raise ValueError('forward_bridge_residual_plan(sample=True, noise_scale>0) requires an rng.')
            noise = jax.random.normal(rng, path.shape)
            path = path + jnp.asarray(noise_scale, dtype=jnp.float32) * std[None, :, None] * noise

        # Endpoint clamp - safety net; ``w_0 = w_K = 0`` already preserves
        # endpoints in the deterministic case, but the explicit ``set`` keeps
        # things exact under sampled noise and floating-point round-off.
        path = path.at[:, 0, :].set(z0).at[:, -1, :].set(zK)
        return path

    def _forward_bridge_path_at_indices(
        self,
        z0: jnp.ndarray,
        zK: jnp.ndarray,
        indices: jnp.ndarray,
        *,
        planner: str,
        params=None,
        anchor=None,
    ) -> jnp.ndarray:
        """Evaluate a forward-bridge path only at selected full-horizon indices."""
        N = int(self.config['dynamics_N'])
        a, b, _ = self.forward_bridge_coefficients(N)
        idx = jnp.asarray(indices, dtype=jnp.int32)
        idx_f = idx.astype(jnp.float32)
        mu = a[idx][None, :, None] * z0[:, None, :] + b[idx][None, :, None] * zK[:, None, :]
        if anchor is None:
            if self._is_residual_displacement_mode():
                raise ValueError(
                    '_forward_bridge_path_at_indices in displacement mode requires anchor=s_t.'
                )
            anchor = z0
        n_idx = jnp.broadcast_to(idx[None, :], (z0.shape[0], idx.shape[0]))
        residual = jax.vmap(
            lambda mu_b, anchor_b, zK_b, n_b: self.network.select('residual_net')(
                mu_b, anchor_b, zK_b, anchor_b, n_b, params=params,
            ),
            in_axes=(0, 0, 0, 0),
        )(mu, anchor, zK, n_idx)
        w = idx_f * (float(N) - idx_f) / float(N * N)
        path = mu + w[None, :, None] * residual
        path = jnp.where((idx == 0)[None, :, None], z0[:, None, :], path)
        path = jnp.where((idx == N)[None, :, None], zK[:, None, :], path)
        return path

    def _exact_residual_chain_plan(
        self, current_state, desired_endpoint, num_steps: int | None = None, *, goal=None, anchor=None,
    ):
        x_T = current_state
        x_0_goal = desired_endpoint
        n_total = int(self.config['dynamics_N'])
        steps_to_roll = n_total if num_steps is None else int(num_steps)
        if steps_to_roll < 1 or steps_to_roll > n_total:
            raise ValueError(f'num_steps must be in [1, {n_total}], got {steps_to_roll}.')
        batch_size = x_T.shape[0]
        if anchor is None:
            anchor = jnp.zeros_like(x_T)

        def scan_body(x, step_n):
            n = jnp.full((batch_size,), step_n, dtype=jnp.int32)
            x_new, _ = self._learned_reverse_mean(
                x, x_T, x_0_goal, n, self.schedule, goal=goal, anchor=anchor,
            )
            return x_new, x_new

        steps = jnp.arange(n_total, n_total - steps_to_roll, -1)
        _, traj_body = jax.lax.scan(scan_body, x_T, steps)
        traj = jnp.concatenate([x_T[None], traj_body], axis=0)
        return jnp.swapaxes(traj, 0, 1)

    def _exact_residual_chain_sample_plan(
        self, current_state, desired_endpoint, rng, noise_scale: float, num_steps: int | None = None,
        *, goal=None, anchor=None,
    ):
        x_T = current_state
        x_0_goal = desired_endpoint
        n_total = int(self.config['dynamics_N'])
        steps_to_roll = n_total if num_steps is None else int(num_steps)
        if steps_to_roll < 1 or steps_to_roll > n_total:
            raise ValueError(f'num_steps must be in [1, {n_total}], got {steps_to_roll}.')
        batch_size = x_T.shape[0]
        step_rngs = jax.random.split(rng, steps_to_roll)
        if anchor is None:
            anchor = jnp.zeros_like(x_T)

        def scan_body(x, inputs):
            step_n, step_rng = inputs
            n = jnp.full((batch_size,), step_n, dtype=jnp.int32)
            x_new, _ = self._reverse_step(
                x, x_T, x_0_goal, n, step_rng, True, noise_scale,
                params=None, goal=goal, anchor=anchor,
            )
            return x_new, x_new

        steps = jnp.arange(n_total, n_total - steps_to_roll, -1)
        _, traj_body = jax.lax.scan(scan_body, x_T, (steps, step_rngs))
        traj = jnp.concatenate([x_T[None], traj_body], axis=0)
        return jnp.swapaxes(traj, 0, 1)

    def _shift_to_displacement_frame(
        self, current_state: jnp.ndarray, desired_endpoint: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return ``(origin, z0, zK, anchor)`` for the underlying planner.

        In displacement mode the bridge / residual chain is trained with
        ``x_T = 0`` and ``x_0 = Delta``; we shift the planner inputs accordingly
        and recover absolute states later by adding ``origin``.  ``anchor`` is
        the absolute current state that we feed to the residual nets so they
        can break translation invariance in displacement mode.
        """
        origin = self._displacement_origin(current_state)
        z0 = current_state - origin
        zK = desired_endpoint - origin
        anchor = self._bridge_anchor(current_state)
        return origin, z0, zK, anchor

    @partial(jax.jit, static_argnames=('num_steps',))
    def plan(self, current_state, desired_endpoint, *, num_steps: int | None = None, goal=None):
        squeeze = current_state.ndim == 1
        if squeeze:
            current_state = current_state[None]
            desired_endpoint = desired_endpoint[None]
            if goal is not None:
                goal = goal[None]

        origin, z0, zK, anchor = self._shift_to_displacement_frame(current_state, desired_endpoint)

        traj_local = self.forward_bridge_residual_plan(
            z0, zK, sample=False, noise_scale=0.0, num_steps=num_steps, anchor=anchor,
        )

        traj = traj_local + origin[:, None, :]
        result = {'next_step': traj[:, 1, :], 'trajectory': traj}
        if squeeze:
            result = jax.tree_util.tree_map(lambda x: x[0], result)
        return result

    @partial(jax.jit, static_argnames=('noise_scale', 'num_steps'))
    def sample_plan(self, current_state, desired_endpoint, rng, noise_scale: float = 1.0, num_steps: int | None = None, *, goal=None):
        squeeze = current_state.ndim == 1
        if squeeze:
            current_state = current_state[None]
            desired_endpoint = desired_endpoint[None]
            if goal is not None:
                goal = goal[None]

        origin, z0, zK, anchor = self._shift_to_displacement_frame(current_state, desired_endpoint)

        sample_flag = _forward_bridge_mode(self.config) == 'sample'
        traj_local = self.forward_bridge_residual_plan(
            z0, zK, sample=sample_flag, noise_scale=noise_scale,
            num_steps=num_steps, rng=rng, anchor=anchor,
        )

        traj = traj_local + origin[:, None, :]
        result = {'next_step': traj[:, 1, :], 'trajectory': traj}
        if squeeze:
            result = jax.tree_util.tree_map(lambda x: x[0], result)
        return result

    def _sample_plan_trajectory(
        self, current_state, desired_endpoint, rng, noise_scale: float, num_steps: int | None = None, *, goal=None,
    ):
        origin, z0, zK, anchor = self._shift_to_displacement_frame(current_state, desired_endpoint)
        sample_flag = _forward_bridge_mode(self.config) == 'sample'
        traj_local = self.forward_bridge_residual_plan(
            z0, zK, sample=sample_flag, noise_scale=noise_scale,
            num_steps=num_steps, rng=rng, anchor=anchor,
        )
        return traj_local + origin[:, None, :]

    @partial(jax.jit, static_argnames=('num_candidates', 'include_mean', 'noise_scale', 'num_steps'))
    def sample_plan_candidates(
        self,
        current_state,
        desired_endpoint,
        rng,
        *,
        num_candidates: int,
        noise_scale: float = 1.0,
        include_mean: bool = True,
        num_steps: int | None = None,
        goal=None,
    ):
        squeeze = current_state.ndim == 1
        if squeeze:
            current_state = current_state[None]
            desired_endpoint = desired_endpoint[None]
            if goal is not None:
                goal = goal[None]

        if include_mean:
            det = self.plan(current_state, desired_endpoint, num_steps=num_steps, goal=goal)['trajectory'][:, None, ...]
            if num_candidates == 1:
                out = det
            else:
                sample_rngs = jax.random.split(rng, num_candidates - 1)
                sampled = jax.vmap(
                    lambda r: self._sample_plan_trajectory(
                        current_state, desired_endpoint, r, noise_scale, num_steps=num_steps, goal=goal,
                    ),
                    in_axes=0,
                )(sample_rngs)
                sampled = jnp.swapaxes(sampled, 0, 1)
                out = jnp.concatenate([det, sampled], axis=1)
        else:
            sample_rngs = jax.random.split(rng, num_candidates)
            sampled = jax.vmap(
                lambda r: self._sample_plan_trajectory(
                    current_state, desired_endpoint, r, noise_scale, num_steps=num_steps, goal=goal,
                ),
                in_axes=0,
            )(sample_rngs)
            out = jnp.swapaxes(sampled, 0, 1)

        if squeeze:
            out = out[0]
        return out

    def _subgoal_forward(self, observations, high_actor_goals, params=None):
        """Raw subgoal-net forward.

        Returns either a single tensor (deterministic mode) or a ``(mu, log_std)``
        tuple (diag_gaussian mode).  Used internally; external callers should
        prefer ``infer_subgoal`` / ``infer_subgoal_distribution``.
        """
        return self.network.select('subgoal_net')(observations, high_actor_goals, params=params)

    @jax.jit
    def predict_subgoal(self, observations, high_actor_goals):
        """Backward-compatible: returns the deterministic / mean subgoal point.

        In ``subgoal_target_mode='displacement'`` the raw network output is the
        predicted displacement ``Delta``; this helper adds ``observations`` so
        callers always see an *absolute* next-K state, regardless of mode.
        """
        squeeze = observations.ndim == 1
        if squeeze:
            observations = observations[None]
            high_actor_goals = high_actor_goals[None]
        out = self._subgoal_forward(observations, high_actor_goals)
        if isinstance(out, tuple):
            out = out[0]
        out = self._subgoal_abs_from_raw(observations, out)
        if squeeze:
            out = out[0]
        return out

    @jax.jit
    def infer_subgoal(self, observations, high_actor_goals):
        """Backward-compatible alias for :meth:`predict_subgoal`."""
        return self.predict_subgoal(observations, high_actor_goals)

    @jax.jit
    def infer_subgoal_mean(self, observations, high_actor_goals):
        """Distributional API: returns mu (== deterministic point in deterministic mode)."""
        return self.predict_subgoal(observations, high_actor_goals)

    @jax.jit
    def infer_subgoal_distribution(self, observations, high_actor_goals):
        """Distributional API: returns ``(mu, log_std)``.

        ``mu`` is always an *absolute* next-K state (so downstream callers do
        not need to know about ``subgoal_target_mode``).  ``log_std`` is the
        same in either frame because the displacement shift ``s_t`` is
        deterministic.  In deterministic mode ``log_std`` is filled with
        ``log_std_min`` so the distribution degenerates to a point mass under
        any sampler.
        """
        squeeze = observations.ndim == 1
        if squeeze:
            observations = observations[None]
            high_actor_goals = high_actor_goals[None]
        out = self._subgoal_forward(observations, high_actor_goals)
        if isinstance(out, tuple):
            mu, log_std = out
        else:
            mu = out
            log_std_min = float(self.config.get('subgoal_log_std_min', -5.0))
            log_std = jnp.full_like(mu, log_std_min)
        mu = self._subgoal_abs_from_raw(observations, mu)
        if squeeze:
            mu = mu[0]
            log_std = log_std[0]
        return mu, log_std

    @partial(jax.jit, static_argnames=('num_candidates', 'include_mean'))
    def sample_subgoal_candidates(
        self,
        observations,
        high_actor_goals,
        rng,
        *,
        num_candidates: int,
        include_mean: bool = True,
    ):
        """Sample ``num_candidates`` subgoal endpoints.

        - In ``diag_gaussian`` mode this draws from ``q(g_sub | s, g_high)``
          with optional ``subgoal_temperature`` scaling and optionally pins
          the first sample to ``mu``.
        - In ``deterministic`` mode every candidate equals the predicted
          point (subgoal-distribution sampling is a no-op there; trajectory
          stochasticity still flows through ``plan_noise_scale``).

        Returns ``(candidate_goals [B, N, D], mu [B, D])``.
        """
        squeeze = observations.ndim == 1
        if squeeze:
            observations = observations[None]
            high_actor_goals = high_actor_goals[None]
        obs = jnp.asarray(observations, dtype=jnp.float32)
        goals = jnp.asarray(high_actor_goals, dtype=jnp.float32)
        if num_candidates < 1:
            raise ValueError(f'num_candidates must be >= 1, got {num_candidates}.')

        out = self._subgoal_forward(obs, goals)
        if isinstance(out, tuple):
            mu_raw, log_std = out
            std = jnp.exp(log_std) * float(self.config.get('subgoal_temperature', 1.0))
            n_sample = num_candidates - 1 if include_mean else num_candidates
            if n_sample > 0:
                eps = jax.random.normal(rng, (n_sample, mu_raw.shape[0], mu_raw.shape[-1]))
                sampled = mu_raw[None, :, :] + eps * std[None, :, :]
                sampled = jnp.swapaxes(sampled, 0, 1)  # [B, n_sample, D]
            else:
                sampled = jnp.zeros((mu_raw.shape[0], 0, mu_raw.shape[-1]), dtype=mu_raw.dtype)
            if include_mean:
                candidates_raw = jnp.concatenate([mu_raw[:, None, :], sampled], axis=1)
            else:
                candidates_raw = sampled
        else:
            mu_raw = out
            candidates_raw = jnp.broadcast_to(
                mu_raw[:, None, :], (mu_raw.shape[0], num_candidates, mu_raw.shape[-1])
            )

        # In displacement mode the raw output is ``Delta``; convert candidates
        # back to absolute states so downstream callers (``sample_plan*`` /
        # rescoring) always see absolute endpoints.
        if self._is_displacement_mode():
            mu = obs + mu_raw
            candidates = obs[:, None, :] + candidates_raw
        else:
            mu = mu_raw
            candidates = candidates_raw

        if squeeze:
            mu = mu[0]
            candidates = candidates[0]
        return candidates, mu

    @jax.jit
    def plan_from_high_goal(self, current_state, high_actor_goals):
        endpoint = self.predict_subgoal(current_state, high_actor_goals)
        return self.plan(current_state, endpoint, goal=high_actor_goals)

    def _idm_actions_from_trajectories(self, trajectories: jnp.ndarray, horizon: int) -> jnp.ndarray:
        prev_states = trajectories[:, :horizon, :]
        next_states = trajectories[:, 1 : horizon + 1, :]
        flat_prev = prev_states.reshape(-1, prev_states.shape[-1])
        flat_next = next_states.reshape(-1, next_states.shape[-1])
        pred = self.network.select('idm_net')(flat_prev, flat_next)
        return jnp.asarray(pred, dtype=jnp.float32).reshape(trajectories.shape[0], horizon, -1)

    @partial(jax.jit, static_argnames=('proposal_horizon', 'plan_candidates', 'sample_noise_scale'))
    def build_actor_proposals(
        self,
        observations,
        high_actor_goals,
        rng,
        *,
        proposal_horizon: int,
        plan_candidates: int,
        sample_noise_scale: float = 0.0,
    ):
        """Build candidate action chunks for the SPI rescoring path.

        Returns
        -------
        actor_goal_mean : ``[B, D]``
            Mean subgoal point used as ``spi_goals`` for the actor when
            ``subgoal_use_mean_for_actor_goal=True`` (default).
        candidate_actions : ``[B, N, ha, A]``
            Decoded action chunks for the proposals.  In deterministic mode
            ``N = plan_candidates``.  In ``diag_gaussian`` mode the candidate
            axis is ``U * N`` where ``U = subgoal_num_samples`` sampled
            subgoal endpoints and ``N = plan_candidates`` bridge/action
            samples per endpoint.
        candidate_goals : ``[B, N, D]`` or ``[B, U*N, D]``
            Per-candidate subgoal endpoints used both to drive bridge
            sampling and to rescore each candidate with its own goal.  In
            deterministic mode this is just the mean broadcast across ``N``.
        new_rng : updated PRNG key.
        """
        obs = jnp.asarray(observations, dtype=jnp.float32)
        goals = jnp.asarray(high_actor_goals, dtype=jnp.float32)
        sub_mode = _subgoal_mode(self.config)
        planner = _planner_type(self.config)
        use_full_bridge_prefix = (
            planner in ('forward_bridge', 'forward_bridge_residual')
            and int(proposal_horizon) < int(self.config['dynamics_N'])
        )

        if sub_mode == 'diag_gaussian':
            sub_rng, plan_rng, new_rng = jax.random.split(rng, 3)
            subgoal_samples = max(1, int(self.config.get('subgoal_num_samples', 1)))
            candidate_goals, mu = self.sample_subgoal_candidates(
                obs,
                goals,
                sub_rng,
                num_candidates=subgoal_samples,
                include_mean=True,
            )
            per_subgoal_rngs = jax.random.split(plan_rng, subgoal_samples)
            cand_endpoints = jnp.swapaxes(candidate_goals, 0, 1)  # [U, B, D]
            traj_noise = float(sample_noise_scale)

            def _per_subgoal(rng_u, endpoint_u):
                trajs = self.sample_plan_candidates(
                    obs,
                    endpoint_u,
                    rng_u,
                    num_candidates=plan_candidates,
                    noise_scale=traj_noise,
                    include_mean=True,
                    num_steps=None if use_full_bridge_prefix else proposal_horizon,
                    goal=goals,
                )
                if use_full_bridge_prefix:
                    trajs = trajs[:, :, : proposal_horizon + 1, :]
                return trajs

            candidate_trajectories = jax.vmap(_per_subgoal, in_axes=(0, 0))(per_subgoal_rngs, cand_endpoints)
            candidate_trajectories = jnp.swapaxes(candidate_trajectories, 0, 1)  # [B, U, N, K+1, D]
            B, U, Np = candidate_trajectories.shape[:3]
            candidate_trajectories = candidate_trajectories.reshape(
                B, U * Np, candidate_trajectories.shape[-2], candidate_trajectories.shape[-1]
            )
            candidate_goals = jnp.repeat(candidate_goals[:, :, None, :], plan_candidates, axis=2)
            candidate_goals = candidate_goals.reshape(obs.shape[0], subgoal_samples * plan_candidates, -1)
        else:
            mu = self._subgoal_forward(obs, goals)
            if isinstance(mu, tuple):  # safety net (shouldn't happen in deterministic mode)
                mu = mu[0]
            # Map raw subgoal-net output to absolute frame.  In absolute mode
            # this is a no-op; in displacement mode this adds the current state
            # so the bridge planner downstream still receives absolute endpoints.
            mu = self._subgoal_abs_from_raw(obs, mu)
            if plan_candidates == 1:
                if use_full_bridge_prefix:
                    origin, z0, zK, anchor = self._shift_to_displacement_frame(obs, mu)
                    indices = jnp.arange(0, proposal_horizon + 1, dtype=jnp.int32)
                    traj_local = self._forward_bridge_path_at_indices(
                        z0, zK, indices, planner=planner, anchor=anchor,
                    )
                    sampled = {'trajectory': traj_local + origin[:, None, :]}
                else:
                    sampled = self.sample_plan(
                        obs,
                        mu,
                        rng,
                        noise_scale=0.0,
                        num_steps=proposal_horizon,
                        goal=goals,
                    )
                new_rng, _ = jax.random.split(rng)
                candidate_trajectories = sampled['trajectory'][:, None, ...]
            else:
                new_rng, sample_rng = jax.random.split(rng)
                candidate_trajectories = self.sample_plan_candidates(
                    obs,
                    mu,
                    sample_rng,
                    num_candidates=plan_candidates,
                    noise_scale=sample_noise_scale,
                    include_mean=True,
                    num_steps=None if use_full_bridge_prefix else proposal_horizon,
                    goal=goals,
                )
                if use_full_bridge_prefix:
                    candidate_trajectories = candidate_trajectories[:, :, : proposal_horizon + 1, :]
            candidate_goals = jnp.broadcast_to(mu[:, None, :], (mu.shape[0], plan_candidates, mu.shape[-1]))

        flat_trajectories = candidate_trajectories.reshape(
            -1, candidate_trajectories.shape[2], candidate_trajectories.shape[3]
        )
        candidate_actions = self._idm_actions_from_trajectories(flat_trajectories, proposal_horizon)
        candidate_actions = candidate_actions.reshape(
            candidate_trajectories.shape[0], candidate_trajectories.shape[1], proposal_horizon, -1
        )
        return mu, candidate_actions, candidate_goals, new_rng

    @classmethod
    def create(cls, seed, ex_observations, config, ex_actions=None):
        assert config['dynamics_N'] >= 2, 'Dynamics requires N >= 2 diffusion steps.'
        if ex_actions is None:
            raise ValueError(
                '_DynamicsAgentCore.create requires ex_actions shaped (B, A) to build the inverse-dynamics head.'
            )

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        state_dim = ex_observations.shape[-1]
        action_dim = int(ex_actions.shape[-1])
        phi_idxs = normalize_phi_goal_obs_indices(config.get('phi_goal_obs_indices', ()))
        env_name_for_phi = str(config.get('env_name', ''))
        sg_rep = str(config.get('subgoal_goal_representation', config.get('goal_representation', 'full'))).lower()
        assert_phi_goal_obs_indices(
            int(state_dim),
            sg_rep,
            phi_idxs,
            where='DynamicsAgent.create (subgoal_goal_representation)',
            env_name=env_name_for_phi,
        )
        val_rep = str(config.get('subgoal_value_goal_representation', 'full')).lower()
        assert_phi_goal_obs_indices(
            int(state_dim),
            val_rep,
            phi_idxs,
            where='DynamicsAgent.create (subgoal_value_goal_representation)',
            env_name=env_name_for_phi,
        )
        idm_hidden = config.get('idm_hidden_dims', (512, 512, 512))
        if isinstance(idm_hidden, str):
            idm_hidden = parse_hidden_dims(idm_hidden)
        else:
            idm_hidden = tuple(int(x) for x in idm_hidden)

        schedule = make_dynamics_schedule(
            N=config['dynamics_N'],
            beta_min=config['dynamics_beta_min'],
            beta_max=config['dynamics_beta_max'],
            lambda_=config['dynamics_lambda'],
            bridge_gamma_inv=float(config.get('bridge_gamma_inv', 0.0)),
            theta_schedule=str(config.get('theta_schedule', 'linear_beta')),
            theta_total=float(config.get('theta_total', 1.0)),
            progress_alpha=float(config.get('progress_alpha', 0.8)),
        )

        residual_net_def = ResidualNet(
            hidden_dims=tuple(config['residual_model_hidden_dims']),
            state_dim=state_dim,
            time_embed_dim=config['time_embed_dim'],
            layer_norm=config['layer_norm'],
        )
        sub_mode = _subgoal_mode(config)
        stochastic_loss = _subgoal_stochastic_loss(config)
        if stochastic_loss not in ('mse', 'nll'):
            raise ValueError(
                f"subgoal_stochastic_loss must be 'mse' or 'nll', got {stochastic_loss!r}."
            )
        if sub_mode == 'deterministic':
            subgoal_def = SubgoalEstimatorNet(
                hidden_dims=tuple(config['subgoal_hidden_dims']),
                state_dim=state_dim,
                layer_norm=config['layer_norm'],
                goal_representation=str(
                    config.get('subgoal_goal_representation', config.get('goal_representation', 'full')),
                ),
                phi_goal_obs_indices=phi_idxs,
                env_name=env_name_for_phi,
            )
        elif sub_mode == 'diag_gaussian':
            subgoal_def = DistributionalSubgoalEstimatorNet(
                hidden_dims=tuple(config['subgoal_hidden_dims']),
                state_dim=state_dim,
                layer_norm=config['layer_norm'],
                log_std_min=float(config.get('subgoal_log_std_min', -5.0)),
                log_std_max=float(config.get('subgoal_log_std_max', 1.0)),
                goal_representation=str(
                    config.get('subgoal_goal_representation', config.get('goal_representation', 'full')),
                ),
                phi_goal_obs_indices=phi_idxs,
                env_name=env_name_for_phi,
            )
        else:
            raise ValueError(
                f"subgoal_distribution must be 'deterministic' or 'diag_gaussian', got {sub_mode!r}."
            )
        idm_def = InverseDynamicsMLP(
            obs_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=idm_hidden,
        )

        batch_size = ex_observations.shape[0]
        dummy_x = ex_observations
        dummy_g = ex_observations
        dummy_n = jnp.ones((batch_size,), dtype=jnp.float32)
        dummy_next = jnp.zeros_like(ex_observations)

        # ResidualNet signature takes an explicit ``anchor`` input.  Public
        # training/planning paths pass the absolute current state ``s_t``
        # (redundant with ``x_T`` in absolute mode, essential context in
        # displacement mode); the initializer must therefore see the matching
        # shape.
        network_info = dict(
            residual_net=(residual_net_def, (dummy_x, dummy_x, dummy_x, dummy_x, dummy_n)),
            subgoal_net=(subgoal_def, (dummy_x, dummy_g)),
            idm_net=(idm_def, (dummy_x, dummy_next)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        cfg_out = {
            **dict(config),
            'idm_action_dim': action_dim,
            'idm_hidden_dims': idm_hidden,
            'phi_goal_obs_indices': phi_idxs,
        }
        return cls(
            rng=rng,
            network=network,
            schedule=schedule,
            config=flax.core.FrozenDict(**cfg_out),
        )


class DynamicsAgent(_DynamicsAgentCore):
    """Linear-SDE dynamics agent with path supervision and rollout consistency."""

    @classmethod
    def create(cls, seed, ex_observations, config, ex_actions=None):
        if bool(config.get('require_matching_horizon', True)):
            gn = int(config['dynamics_N'])
            sk = int(config['subgoal_steps'])
            if gn != sk:
                raise ValueError(
                    f'Dynamics: require_matching_horizon expects dynamics_N ({gn}) == '
                    f'subgoal_steps ({sk}). Disable with require_matching_horizon: false '
                    'only if you accept misaligned indices.'
                )
        return super().create(seed, ex_observations, config, ex_actions=ex_actions)

    def _subgoal_mse_weight_from_gap(self, gap: jnp.ndarray) -> jnp.ndarray:
        value_style = str(self.config.get('subgoal_value_style', 'exponential')).lower()
        if value_style == 'exponential':
            gap_scale = jnp.asarray(
                float(self.config.get('subgoal_value_gap_scale', 1.0)), dtype=jnp.float32,
            )
            weight = jnp.exp(gap_scale * gap)
            weight_max = float(self.config.get('subgoal_value_weight_max', 0.0))
            if weight_max > 0.0:
                weight = jnp.minimum(weight, jnp.asarray(weight_max, dtype=jnp.float32))
            return weight

        if value_style == 'expectile':
            expectile = float(self.config.get('subgoal_value_expectile', 0.7))
            if not 0.0 <= expectile <= 1.0:
                raise ValueError(
                    'subgoal_value_expectile must be in [0, 1] when subgoal_value_style="expectile".'
                )
            expectile_arr = jnp.asarray(expectile, dtype=jnp.float32)
            return jnp.where(gap > 0.0, expectile_arr, 1.0 - expectile_arr)

        raise ValueError(
            f"Unknown subgoal_value_style={value_style!r}; expected 'exponential' or 'expectile'."
        )

    def _subgoal_adv_logit_from_gap(self, gap: jnp.ndarray) -> jnp.ndarray:
        """Return the scalar inside the exponential subgoal weight."""
        value_style = str(self.config.get('subgoal_value_style', 'exponential')).lower()
        if value_style == 'exponential':
            gap_scale = jnp.asarray(
                float(self.config.get('subgoal_value_gap_scale', 1.0)), dtype=jnp.float32,
            )
            return gap_scale * gap
        return jnp.zeros_like(gap)

    def _subgoal_values(
        self,
        states: jnp.ndarray,
        high_actor_goals: jnp.ndarray,
        critic_value_params: Any | None,
    ) -> jnp.ndarray:
        zeros = jnp.zeros((states.shape[0],), dtype=jnp.float32)
        if critic_value_params is None:
            return zeros

        value_def = ScalarValueNet(
            tuple(int(x) for x in self.config.get('subgoal_value_hidden_dims', (512, 512, 512))),
            layer_norm=bool(self.config.get('subgoal_value_layer_norm', True)),
            goal_representation=str(self.config.get('subgoal_value_goal_representation', 'full')),
            phi_goal_obs_indices=normalize_phi_goal_obs_indices(self.config.get('phi_goal_obs_indices', ())),
            env_name=str(self.config.get('env_name', '')),
        )
        value_logits = value_def.apply({'params': critic_value_params}, states, high_actor_goals)
        return jax.nn.sigmoid(jnp.asarray(value_logits, dtype=jnp.float32))

    def _subgoal_value_terms(
        self,
        observations: jnp.ndarray,
        pred_subgoals: jnp.ndarray,
        target_subgoals: jnp.ndarray,
        high_actor_goals: jnp.ndarray,
        critic_value_params: Any | None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        if critic_value_params is None:
            pred_value = jnp.zeros((observations.shape[0],), dtype=jnp.float32)
            obs_value = jnp.zeros_like(pred_value)
            target_value = jnp.zeros_like(pred_value)
        else:
            states = jnp.concatenate([pred_subgoals, observations, target_subgoals], axis=0)
            goals = jnp.concatenate([high_actor_goals, high_actor_goals, high_actor_goals], axis=0)
            values = self._subgoal_values(states, goals, critic_value_params)
            pred_value, obs_value, target_value = jnp.split(values, 3, axis=0)

        gap = target_value - obs_value
        adv_logit = self._subgoal_adv_logit_from_gap(gap)
        mse_weight = self._subgoal_mse_weight_from_gap(gap)
        alpha = jnp.asarray(float(self.config.get('subgoal_value_alpha', 0.0)), dtype=jnp.float32)
        subgoal_value_bonus = jnp.where(alpha > 0.0, alpha * pred_value, jnp.zeros_like(pred_value))
        return pred_value, obs_value, target_value, subgoal_value_bonus, mse_weight, gap, adv_logit

    def _compute_subgoal_loss(self, batch, grad_params, rng_fr, critic_value_params):
        """Compute the subgoal-net training loss + companion logging tensors.

        Shared by ``exact_residual_chain`` and the ``forward_bridge`` /
        ``forward_bridge_residual`` paths so the subgoal estimator is trained
        identically across planners.

        Note on the PDF: deterministic subgoals match the value-guided
        regression form directly.  Stochastic subgoals intentionally add an
        implementation choice between the PDF-style reparameterized sample-MSE
        objective and a weighted Gaussian NLL objective via
        ``subgoal_stochastic_loss``.  In displacement mode, value terms are
        always evaluated after reconstructing the absolute sample
        ``observations + Delta``.

        In ``subgoal_target_mode='displacement'`` the raw network output is the
        predicted displacement ``Delta`` and the supervised target becomes
        ``high_actor_targets - observations``.  The companion value terms still
        operate on *absolute* states (``observations``, ``observations +
        Delta``, ``high_actor_targets``) so the critic value head sees the
        same inputs in either mode.
        """
        s = batch['observations']
        g_high = batch['high_actor_goals']
        target_abs = batch['high_actor_targets']
        target = self._subgoal_target_for_loss(s, target_abs)
        sub_mode = _subgoal_mode(self.config)

        if sub_mode == 'diag_gaussian':
            stochastic_loss = _subgoal_stochastic_loss(self.config)
            pred_mu, pred_log_std = self.network.select('subgoal_net')(s, g_high, params=grad_params)
            pred_std = jnp.exp(pred_log_std)
            inv_var = jnp.exp(-2.0 * pred_log_std)
            mean_diff = target - pred_mu
            nll_per_sample = 0.5 * jnp.sum(
                mean_diff ** 2 * inv_var + 2.0 * pred_log_std + jnp.log(2.0 * jnp.pi), axis=-1
            )
            eps = jax.random.normal(rng_fr, pred_mu.shape, dtype=pred_mu.dtype)
            pred_sample = pred_mu + eps * pred_std
            sample_diff = target - pred_sample
            subgoal_mse = jnp.mean(sample_diff ** 2, axis=-1)
            mean_mse = jnp.mean(mean_diff ** 2, axis=-1)
            # Map the predicted sample back to absolute state for the value
            # head (which expects absolute observations).  In absolute mode
            # ``_subgoal_abs_from_raw`` is the identity.
            pred_sample_abs = self._subgoal_abs_from_raw(s, pred_sample)
            (
                subgoal_value,
                current_value,
                target_value,
                subgoal_value_bonus,
                subgoal_mse_weight,
                subgoal_value_gap,
                subgoal_adv_logit,
            ) = self._subgoal_value_terms(
                s, pred_sample_abs, target_abs, g_high, critic_value_params
            )
            # Weight better dataset targets more strongly, without adding a gradient path through exp(V gap).
            subgoal_weight = jax.lax.stop_gradient(subgoal_mse_weight)
            weighted_subgoal_mse = subgoal_weight * subgoal_mse
            weighted_subgoal_nll = subgoal_weight * nll_per_sample
            if stochastic_loss == 'mse':
                stochastic_loss_term = jnp.mean(weighted_subgoal_mse)
                stochastic_loss_mode = jnp.asarray(0.0, dtype=jnp.float32)
            elif stochastic_loss == 'nll':
                stochastic_loss_term = jnp.mean(weighted_subgoal_nll)
                stochastic_loss_mode = jnp.asarray(1.0, dtype=jnp.float32)
            else:
                raise ValueError(
                    f"subgoal_stochastic_loss must be 'mse' or 'nll', got {stochastic_loss!r}."
                )
            loss_sub = stochastic_loss_term - jnp.mean(subgoal_value_bonus)
            pred_sg_out = pred_sample
            subgoal_extra_info = {
                'phase1/subgoal_nll': jnp.mean(nll_per_sample),
                'phase1/subgoal_stochastic_loss': stochastic_loss_term,
                'phase1/subgoal_stochastic_loss_mode': stochastic_loss_mode,
                'phase1/subgoal_mean_mse': jnp.mean(mean_mse),
                'phase1/subgoal_sample_mse': jnp.mean(subgoal_mse),
                'phase1/subgoal_weighted_mse': jnp.mean(weighted_subgoal_mse),
                'phase1/subgoal_weighted_nll': jnp.mean(weighted_subgoal_nll),
                'phase1/subgoal_std_mean': jnp.mean(pred_std),
                'phase1/subgoal_std_max': jnp.max(pred_std),
                'phase1/subgoal_mode': jnp.asarray(1.0, dtype=jnp.float32),
                'phase1/subgoal_current_value_mean': jnp.mean(current_value),
                'phase1/subgoal_target_value_mean': jnp.mean(target_value),
                'phase1/subgoal_mse_weight_mean': jnp.mean(subgoal_mse_weight),
                'phase1/subgoal_value_gap_mean': jnp.mean(subgoal_value_gap),
                'phase1/subgoal_value_gap_min': jnp.min(subgoal_value_gap),
                'phase1/subgoal_value_gap_max': jnp.max(subgoal_value_gap),
                'phase1/subgoal_adv_logit_mean': jnp.mean(subgoal_adv_logit),
                'phase1/subgoal_adv_logit_min': jnp.min(subgoal_adv_logit),
                'phase1/subgoal_adv_logit_max': jnp.max(subgoal_adv_logit),
            }
        else:
            pred_sg = self.network.select('subgoal_net')(s, g_high, params=grad_params)
            subgoal_mse = jnp.mean((pred_sg - target) ** 2, axis=-1)
            pred_sg_abs = self._subgoal_abs_from_raw(s, pred_sg)
            (
                subgoal_value,
                current_value,
                target_value,
                subgoal_value_bonus,
                subgoal_mse_weight,
                subgoal_value_gap,
                subgoal_adv_logit,
            ) = self._subgoal_value_terms(
                s, pred_sg_abs, target_abs, g_high, critic_value_params
            )
            weighted_subgoal_mse = jax.lax.stop_gradient(subgoal_mse_weight) * subgoal_mse
            loss_sub = jnp.mean(weighted_subgoal_mse) - jnp.mean(subgoal_value_bonus)
            pred_sg_out = pred_sg
            zero = jnp.asarray(0.0, dtype=jnp.float32)
            subgoal_extra_info = {
                'phase1/subgoal_nll': zero,
                'phase1/subgoal_stochastic_loss': jnp.mean(weighted_subgoal_mse),
                'phase1/subgoal_stochastic_loss_mode': zero,
                'phase1/subgoal_std_mean': zero,
                'phase1/subgoal_std_max': zero,
                'phase1/subgoal_mode': zero,
                'phase1/subgoal_weighted_mse': jnp.mean(weighted_subgoal_mse),
                'phase1/subgoal_weighted_nll': zero,
                'phase1/subgoal_current_value_mean': jnp.mean(current_value),
                'phase1/subgoal_target_value_mean': jnp.mean(target_value),
                'phase1/subgoal_mse_weight_mean': jnp.mean(subgoal_mse_weight),
                'phase1/subgoal_value_gap_mean': jnp.mean(subgoal_value_gap),
                'phase1/subgoal_value_gap_min': jnp.min(subgoal_value_gap),
                'phase1/subgoal_value_gap_max': jnp.max(subgoal_value_gap),
                'phase1/subgoal_adv_logit_mean': jnp.mean(subgoal_adv_logit),
                'phase1/subgoal_adv_logit_min': jnp.min(subgoal_adv_logit),
                'phase1/subgoal_adv_logit_max': jnp.max(subgoal_adv_logit),
            }
        return loss_sub, subgoal_mse, subgoal_value, subgoal_value_bonus, pred_sg_out, subgoal_extra_info

    def _path_eval_slice(self) -> tuple[int, ...]:
        pev = self.config.get('path_eval_slice')
        if pev is None:
            return (0, 1)
        return tuple(int(x) for x in pev)

    def _check_segment_horizon(self, K: int, N: int, where: str) -> None:
        """Validate that ``trajectory_segment`` length matches ``dynamics_N``.

        The exact-residual-chain and forward-bridge paths both index the
        segment as ``segment[:, K - n]`` / ``segment[:, h]`` and assume
        ``K == dynamics_N``.  We check this at trace time so misconfigured
        ``PathHGCDataset`` horizons fail loudly rather than producing silent
        out-of-bounds reads.
        """
        if not bool(self.config.get('require_matching_horizon', True)):
            return
        if int(K) != int(N):
            raise ValueError(
                f'{where}: trajectory_segment horizon K={int(K)} must match '
                f'dynamics_N={int(N)}. Check PathHGCDataset, subgoal_steps, '
                'and dynamics_N.'
            )

    def _exact_residual_reverse_loss_block(self, batch, grad_params, rng1, rng2):
        """Reverse-chain dynamics + path + rollout losses (shared exact-residual math).

        Used only by ``_total_loss_exact_residual_chain``.  Forward-bridge
        residual training uses ``PathResidualNet`` instead of this reverse-chain
        ``ResidualNet`` block.
        """
        x_T_abs = batch['observations']
        x_0_abs = batch['high_actor_targets']
        segment_abs = jnp.asarray(batch['trajectory_segment'], dtype=jnp.float32)
        disp_origin = self._displacement_origin(x_T_abs)
        anchor = self._bridge_anchor(x_T_abs)
        x_T = x_T_abs - disp_origin
        x_0 = x_0_abs - disp_origin
        segment = segment_abs - disp_origin[:, None, :]
        B = x_T.shape[0]
        N = int(self.config['dynamics_N'])
        K = int(segment.shape[1]) - 1
        self._check_segment_horizon(K, N, '_exact_residual_reverse_loss_block')

        n = jax.random.randint(rng1, (B,), 1, N + 1)
        is_boundary = n == N
        n_safe = jnp.minimum(n, N - 1)

        x_n_bridge = bridge_sample(x_0, x_T, n_safe, self.schedule, rng2)
        x_n = jnp.where(is_boundary[..., None], x_T, x_n_bridge)
        mu_true = posterior_mean(x_n, x_0, x_T, n, self.schedule)
        mu_pred, eps_pred = self._learned_reverse_mean(
            x_n, x_T, x_0, n, self.schedule, params=grad_params, anchor=anchor,
        )
        g2_n = self.schedule['g2'][n - 1]
        weight = 1.0 / (2.0 * jnp.maximum(g2_n, 1e-12))
        loss_bridge_mean_match = (weight * jnp.abs(mu_true - mu_pred).sum(axis=-1)).mean()

        loss_residual_reg = jnp.mean(jnp.sum(eps_pred ** 2, axis=-1))
        loss_dynamics = (
            float(self.config.get('exact_residual_bridge_match_weight', 0.0)) * loss_bridge_mean_match
            + float(self.config.get('exact_residual_reg_weight', 1.0e-4)) * loss_residual_reg
        )

        row = jnp.arange(B, dtype=jnp.int32)
        x_n_real = segment[row, K - n, :]
        x_prev_real = segment[row, K - n + 1, :]
        mu_pred_path, _ = self._learned_reverse_mean(
            x_n_real, x_T, x_0, n, self.schedule, params=grad_params, anchor=anchor,
        )
        diff_p = mu_pred_path - x_prev_real
        loss_path = jnp.abs(diff_p).sum(axis=-1).mean()

        H_cfg = int(self.config.get('rollout_horizon', 5))
        H_eff = max(1, min(H_cfg, N))
        hs = jnp.arange(1, H_eff + 1, dtype=jnp.int32)

        def roll_body(x, h):
            step_n = N - h + 1
            n_b = jnp.full((B,), step_n, dtype=jnp.int32)
            mu_r, _ = self._learned_reverse_mean(
                x, x_T, x_0, n_b, self.schedule, params=grad_params, anchor=anchor,
            )
            tgt = segment[row, h, :]
            err = jnp.abs(mu_r - tgt).sum(axis=-1)
            return mu_r, err

        _, errs = jax.lax.scan(roll_body, segment[:, 0, :], hs)
        loss_roll = jnp.mean(errs)

        n_N = jnp.full((B,), N, dtype=jnp.int32)
        xNm1, _ = self._learned_reverse_mean(
            x_T, x_T, x_0, n_N, self.schedule, params=grad_params, anchor=anchor,
        )
        xNm1_norm = jnp.linalg.norm(xNm1, axis=-1).mean()
        s1 = segment[:, 1, :]
        first_step_l1 = jnp.abs(xNm1 - s1).sum(axis=-1).mean()
        idx_xy = jnp.asarray(self._path_eval_slice(), dtype=jnp.int32)
        d_xy = xNm1[:, idx_xy] - s1[:, idx_xy]
        first_step_xy_l2 = jnp.sqrt(jnp.mean(d_xy**2))

        return {
            'segment': segment,
            'anchor': anchor,
            'x_T': x_T,
            'x_0': x_0,
            'B': B,
            'N': N,
            'K': K,
            'loss_dynamics': loss_dynamics,
            'loss_path': loss_path,
            'loss_roll': loss_roll,
            'loss_bridge_mean_match': loss_bridge_mean_match,
            'loss_residual_reg': loss_residual_reg,
            'eps_pred': eps_pred,
            'mu_true': mu_true,
            'mu_pred': mu_pred,
            'n': n,
            'xNm1': xNm1,
            'xNm1_norm': xNm1_norm,
            'first_step_l1': first_step_l1,
            'first_step_xy_l2': first_step_xy_l2,
        }

    def _total_loss_exact_residual_chain(self, batch, grad_params, rng, critic_value_params):
        """Mean-matching + path-aligned reverse + short rollout + subgoal MSE."""
        sg_w = float(self.config.get('subgoal_loss_weight', 1.0))
        w_g = float(self.config.get('dynamics_loss_weight', 1.0))
        w_p = float(self.config.get('path_loss_weight', 1.0))
        w_r = float(self.config.get('rollout_loss_weight', 1.0))

        rng1, rng2, rng_fr = jax.random.split(rng, 3)
        rev = self._exact_residual_reverse_loss_block(batch, grad_params, rng1, rng2)
        loss_dynamics = rev['loss_dynamics']
        loss_path = rev['loss_path']
        loss_roll = rev['loss_roll']
        loss_bridge_mean_match = rev['loss_bridge_mean_match']
        loss_residual_reg = rev['loss_residual_reg']
        eps_pred = rev['eps_pred']
        mu_true = rev['mu_true']
        mu_pred = rev['mu_pred']
        n = rev['n']
        xNm1 = rev['xNm1']
        xNm1_norm = rev['xNm1_norm']
        first_step_l1 = rev['first_step_l1']
        first_step_xy_l2 = rev['first_step_xy_l2']
        B = rev['B']

        (loss_sub, subgoal_mse, subgoal_value, subgoal_value_bonus,
         pred_sg_out, subgoal_extra_info) = self._compute_subgoal_loss(
            batch, grad_params, rng_fr, critic_value_params,
        )

        loss = w_g * loss_dynamics + w_p * loss_path + w_r * loss_roll + sg_w * loss_sub

        idm_term, loss_idm_unw = self._idm_loss_term(batch, grad_params)
        loss = loss + idm_term

        info = {
            'phase1/loss': loss,
            'phase1/loss_dynamics': loss_dynamics,
            'phase1/loss_bridge_mean_match': loss_bridge_mean_match,
            'phase1/loss_residual_reg': loss_residual_reg,
            'phase1/loss_path_step': loss_path,
            'phase1/loss_roll': loss_roll,
            'phase1/loss_subgoal': loss_sub,
            'phase1/loss_subgoal_mse': subgoal_mse.mean(),
            'phase1/subgoal_value_mean': subgoal_value.mean(),
            'phase1/subgoal_value_bonus_mean': subgoal_value_bonus.mean(),
            'phase1/loss_idm': loss_idm_unw,
            'phase1/first_step_l1': first_step_l1,
            'phase1/first_step_xy_l2': first_step_xy_l2,
            'phase1/eps_norm': jnp.linalg.norm(eps_pred, axis=-1).mean(),
            'phase1/mu_true_norm': jnp.linalg.norm(mu_true, axis=-1).mean(),
            'phase1/mu_pred_norm': jnp.linalg.norm(mu_pred, axis=-1).mean(),
            'phase1/xN_minus_1_norm': xNm1_norm,
            'phase1/bridge_step_mean': n.astype(jnp.float32).mean(),
            'phase1/planner_type': jnp.asarray(0.0, dtype=jnp.float32),
            'dynamics/model_type': jnp.asarray(
                _dynamics_model_type_metric(self.config), dtype=jnp.float32,
            ),
        }
        # ``pred_sg_out`` lives in the subgoal-net output frame (Delta in
        # displacement mode, absolute state in absolute mode); ``raw`` keys
        # track that frame, ``abs`` keys track the absolute state space so the
        # two are comparable on a per-mode basis.
        target_raw = self._subgoal_target_for_loss(
            batch['observations'], batch['high_actor_targets'],
        )
        pred_abs = self._subgoal_abs_from_raw(batch['observations'], pred_sg_out)
        info['phase1/subgoal_pred_raw_norm'] = jnp.linalg.norm(pred_sg_out, axis=-1).mean()
        info['phase1/subgoal_target_raw_norm'] = jnp.linalg.norm(target_raw, axis=-1).mean()
        info['phase1/subgoal_pred_abs_norm'] = jnp.linalg.norm(pred_abs, axis=-1).mean()
        info['phase1/subgoal_target_abs_norm'] = jnp.linalg.norm(
            batch['high_actor_targets'], axis=-1
        ).mean()
        # Backward-compatible aliases - both raw-frame so old dashboards
        # stay coherent across modes (in displacement mode the raw frame is
        # the displacement frame; legacy plots that paired pred vs target
        # would otherwise mix coordinate systems).
        info['phase1/subgoal_pred_norm'] = info['phase1/subgoal_pred_raw_norm']
        info['phase1/subgoal_target_norm'] = info['phase1/subgoal_target_raw_norm']
        info.update(subgoal_extra_info)
        info['dynamics/bridge_gamma_inv'] = jnp.asarray(
            float(self.config.get('bridge_gamma_inv', 0.0)), dtype=jnp.float32
        )
        info['dynamics/gamma_inv'] = self.schedule['gamma_inv']
        info['dynamics/subgoal_target_mode'] = jnp.asarray(
            _subgoal_target_mode_id(self.config), dtype=jnp.float32
        )
        info.update(self._theta_schedule_info())
        return loss, info

    def _total_loss_forward_bridge(self, batch, grad_params, rng, critic_value_params, planner: str):
        """Forward-bridge path-supervised loss (planner_type in {forward_bridge,
        forward_bridge_residual}).

        - ``forward_bridge``: closed-form bridge mean only; reverse-chain terms
          are not trained (logged as zero).
        - ``forward_bridge_residual``: trains only the endpoint-preserving
          ``PathResidualNet`` on top of the closed-form forward bridge.  It does
          not train the reverse-chain ``ResidualNet`` because that network is not
          used by the forward-bridge-residual planner at inference time.
        - Subgoal + IDM match the exact-residual path (single ``_compute_subgoal_loss``
          / ``_idm_loss_term`` per step).
        """
        x_T_abs = batch['observations']
        segment_abs = jnp.asarray(batch['trajectory_segment'], dtype=jnp.float32)
        disp_origin = self._displacement_origin(x_T_abs)
        anchor = self._bridge_anchor(x_T_abs)
        segment = segment_abs - disp_origin[:, None, :]
        N = int(self.config['dynamics_N'])
        K = int(segment.shape[1]) - 1
        self._check_segment_horizon(K, N, '_total_loss_forward_bridge')
        sg_w = float(self.config.get('subgoal_loss_weight', 1.0))
        w_p = float(self.config.get('path_loss_weight', 1.0))
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        Bsz = segment.shape[0]
        Ddim = segment.shape[-1]

        _, rng_fr = jax.random.split(rng, 2)
        loss_bridge_mean_match = loss_residual_reg = zero
        eps_pred = jnp.zeros((Bsz, Ddim), dtype=jnp.float32)
        mu_true = jnp.zeros((Bsz, Ddim), dtype=jnp.float32)
        mu_pred = jnp.zeros((Bsz, Ddim), dtype=jnp.float32)
        n = jnp.ones((Bsz,), dtype=jnp.int32)

        z0 = segment[:, 0, :]
        zK = segment[:, -1, :]

        path_loss_horizon = int(self.config.get('forward_bridge_path_loss_horizon', 0) or 0)
        if 0 < path_loss_horizon < N:
            H = max(1, min(path_loss_horizon, N - 1))
            prefix_idx = jnp.arange(0, H + 1, dtype=jnp.int32)
            indices = jnp.concatenate([prefix_idx, jnp.asarray([N], dtype=jnp.int32)], axis=0)
            path_pred = self._forward_bridge_path_at_indices(
                z0, zK, indices, planner=planner, params=grad_params, anchor=anchor,
            )
            segment_path = segment[:, indices, :]
        else:
            if planner == 'forward_bridge_residual':
                path_pred = self.forward_bridge_residual_plan(
                    z0, zK, sample=False, noise_scale=0.0, num_steps=N,
                    params=grad_params, anchor=anchor,
                )
            else:
                path_pred = self.forward_bridge_plan(
                    z0, zK, sample=False, noise_scale=0.0, num_steps=N,
                )
            segment_path = segment

        path_steps = int(path_pred.shape[1]) - 1
        has_endpoint = path_loss_horizon <= 0 or path_loss_horizon >= N
        interior_stop = -1 if has_endpoint else path_pred.shape[1]
        if path_pred.shape[1] > 2:
            diff_interior = path_pred[:, 1:interior_stop, :] - segment_path[:, 1:interior_stop, :]
            loss_fb_interior = jnp.mean(jnp.abs(diff_interior).sum(axis=-1))
        else:
            loss_fb_interior = jnp.zeros((), dtype=jnp.float32)

        diff_next = path_pred[:, 1, :] - segment_path[:, 1, :]
        loss_fb_next = jnp.mean(jnp.abs(diff_next).sum(axis=-1))

        use_path = bool(self.config.get('forward_bridge_use_path_loss', True))
        path_fb_term = (loss_fb_interior + loss_fb_next) if use_path else zero

        (loss_sub, subgoal_mse, subgoal_value, subgoal_value_bonus,
         pred_sg_out, subgoal_extra_info) = self._compute_subgoal_loss(
            batch, grad_params, rng_fr, critic_value_params,
        )

        idm_term, loss_idm_unw = self._idm_loss_term(batch, grad_params)
        loss = w_p * path_fb_term + sg_w * loss_sub + idm_term

        if path_pred.shape[1] > 2:
            bridge_path_mse = jnp.mean((path_pred[:, 1:interior_stop, :] - segment_path[:, 1:interior_stop, :]) ** 2)
        else:
            bridge_path_mse = zero
        bridge_next_mse = jnp.mean((path_pred[:, 1, :] - segment_path[:, 1, :]) ** 2)
        bridge_final_mse = jnp.mean((path_pred[:, -1, :] - segment_path[:, -1, :]) ** 2)
        bridge_endpoint_start_mse = jnp.mean((path_pred[:, 0, :] - segment_path[:, 0, :]) ** 2)
        bridge_endpoint_end_mse = bridge_final_mse

        dist_per_step = jnp.linalg.norm(path_pred - zK[:, None, :], axis=-1).mean(axis=0)
        first_idx = 1 if path_steps >= 1 else 0
        mid_idx = max(1, path_steps // 2)
        last_idx = path_steps

        s1 = segment_path[:, 1, :]
        first_step_l1_fb = jnp.mean(jnp.abs(path_pred[:, 1, :] - s1).sum(axis=-1))
        idx_xy = jnp.asarray(self._path_eval_slice(), dtype=jnp.int32)
        d_xy_fb = path_pred[:, 1, :][:, idx_xy] - s1[:, idx_xy]
        first_step_xy_l2_fb = jnp.sqrt(jnp.mean(d_xy_fb ** 2))

        planner_id = 1.0 if planner == 'forward_bridge' else 2.0
        fb_diag = {
            'forward_bridge/loss_path_interior': loss_fb_interior,
            'forward_bridge/loss_path_next': loss_fb_next,
            'forward_bridge/first_step_l1_fb_path': first_step_l1_fb,
            'forward_bridge/first_step_xy_l2_fb_path': first_step_xy_l2_fb,
            'forward_bridge/path_mse': bridge_path_mse,
            'forward_bridge/next_mse': bridge_next_mse,
            'forward_bridge/final_mse': bridge_final_mse,
            'forward_bridge/endpoint_start_mse': bridge_endpoint_start_mse,
            'forward_bridge/endpoint_end_mse': bridge_endpoint_end_mse,
            'forward_bridge/dist_to_subgoal_step_1': dist_per_step[first_idx],
            'forward_bridge/dist_to_subgoal_step_mid': dist_per_step[mid_idx],
            'forward_bridge/dist_to_subgoal_step_last': dist_per_step[last_idx],
        }
        info = {
            'phase1/loss': loss,
            'phase1/loss_dynamics': zero,
            'phase1/loss_bridge_mean_match': loss_bridge_mean_match,
            'phase1/loss_residual_reg': loss_residual_reg,
            'phase1/loss_path_step': loss_fb_next,
            'phase1/loss_roll': zero,
            'phase1/loss_subgoal': loss_sub,
            'phase1/loss_subgoal_mse': subgoal_mse.mean(),
            'phase1/subgoal_value_mean': subgoal_value.mean(),
            'phase1/subgoal_value_bonus_mean': subgoal_value_bonus.mean(),
            'phase1/loss_idm': loss_idm_unw,
            'phase1/first_step_l1': first_step_l1_fb,
            'phase1/first_step_xy_l2': first_step_xy_l2_fb,
            'phase1/eps_norm': jnp.linalg.norm(eps_pred, axis=-1).mean(),
            'phase1/mu_true_norm': jnp.linalg.norm(mu_true, axis=-1).mean(),
            'phase1/mu_pred_norm': jnp.linalg.norm(mu_pred, axis=-1).mean(),
            'phase1/xN_minus_1_norm': jnp.linalg.norm(path_pred[:, 1, :], axis=-1).mean(),
            'phase1/bridge_step_mean': n.astype(jnp.float32).mean(),
            'phase1/planner_type': jnp.asarray(planner_id, dtype=jnp.float32),
            'dynamics/model_type': jnp.asarray(
                _dynamics_model_type_metric(self.config), dtype=jnp.float32,
            ),
            **fb_diag,
        }
        # Split prediction / target norms by frame (see exact_residual_chain
        # branch for the rationale).
        target_raw = self._subgoal_target_for_loss(
            batch['observations'], batch['high_actor_targets'],
        )
        pred_abs = self._subgoal_abs_from_raw(batch['observations'], pred_sg_out)
        info['phase1/subgoal_pred_raw_norm'] = jnp.linalg.norm(pred_sg_out, axis=-1).mean()
        info['phase1/subgoal_target_raw_norm'] = jnp.linalg.norm(target_raw, axis=-1).mean()
        info['phase1/subgoal_pred_abs_norm'] = jnp.linalg.norm(pred_abs, axis=-1).mean()
        info['phase1/subgoal_target_abs_norm'] = jnp.linalg.norm(
            batch['high_actor_targets'], axis=-1
        ).mean()
        # Backward-compatible aliases - both raw-frame (see exact-residual
        # branch for the rationale).
        info['phase1/subgoal_pred_norm'] = info['phase1/subgoal_pred_raw_norm']
        info['phase1/subgoal_target_norm'] = info['phase1/subgoal_target_raw_norm']
        info.update(subgoal_extra_info)
        info['dynamics/bridge_gamma_inv'] = jnp.asarray(
            float(self.config.get('bridge_gamma_inv', 0.0)), dtype=jnp.float32
        )
        info['dynamics/gamma_inv'] = self.schedule['gamma_inv']
        info['dynamics/subgoal_target_mode'] = jnp.asarray(
            _subgoal_target_mode_id(self.config), dtype=jnp.float32
        )
        info.update(self._theta_schedule_info())
        return loss, info

    def _theta_schedule_info(self) -> dict:
        """Common theta-schedule diagnostics shared by every loss path.

        Always logs the schedule id, ``theta_total``, ``progress_alpha`` and
        the *actual* hard-bridge marginal weight at step
        ``min(5, dynamics_N)``. The prefix-progress *target* curve is only
        defined for the ``prefix_progress`` schedule; logging it under
        ``linear_beta`` would emit ``NaN``, so we omit that key outside
        prefix-progress mode.
        """
        sched = self.schedule
        prefix_idx = min(5, int(self.config['dynamics_N']))
        info = {
            'dynamics/theta_schedule_id': sched['theta_schedule_id'],
            'dynamics/theta_total': sched['theta_total'],
            'dynamics/progress_alpha': sched['progress_alpha'],
            'dynamics/prefix_progress_actual_5': sched['dynamics_beta_fwd'][prefix_idx],
        }
        # Schedule-id 1.0 == prefix_progress; use a static config check (not the
        # jax array) so the dict layout is fixed across jit traces.
        if str(self.config.get('theta_schedule', 'linear_beta')).lower() == 'prefix_progress':
            info['dynamics/prefix_progress_target_5'] = sched['progress_target_fwd'][prefix_idx]
        return info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None, critic_value_params=None):
        """Path-supervised Phase1 loss; dispatches on ``planner_type``."""
        return self._total_loss_forward_bridge(
            batch, grad_params, rng, critic_value_params, 'forward_bridge_residual',
        )


def _get_common_config():
    """Common defaults for linear dynamics training and rollout."""
    return ml_collections.ConfigDict(
        dict(
            lr=3e-4,
            batch_size=1024,
            dynamics_N=25,
            dynamics_beta_min=0.1,
            dynamics_beta_max=20.0,
            dynamics_lambda=1.0,
            # Linear-SDE bridge denominator offset. 0.0 is the hard endpoint bridge.
            bridge_gamma_inv=0.0,
            # Theta schedule selector. ``linear_beta`` keeps the diffusion-style
            # schedule. ``prefix_progress`` calibrates
            # the hard-bridge marginal interpolation so that the actor-visible
            # prefix already reaches a meaningful fraction of the subgoal
            # displacement (``c_i = (i / K) ** progress_alpha``); ``theta_total``
            # controls the cumulative rate ``Theta_K``.
            theta_schedule='prefix_progress',
            theta_total=1.0,
            progress_alpha=0.8,
            residual_model_hidden_dims=(512, 512, 512),
            time_embed_dim=64,
            layer_norm=True,
            subgoal_loss_weight=1.0,
            subgoal_value_alpha=0.3,
            subgoal_value_style='exponential',
            subgoal_value_expectile=0.7,
            subgoal_value_gap_scale=1.0,
            # Optional high clip for exp(subgoal_value_gap_scale * gap). 0 disables
            # clipping and preserves historical behavior.
            subgoal_value_weight_max=0.0,
            # Goal input to the subgoal estimator. 'full' preserves historical
            # behavior; 'phi' uses task goal-representation channels
            # (ManipSpace cube positions, else maze xy).
            subgoal_goal_representation='phi',
            # Subgoal target representation:
            #   - 'absolute'    : subgoal_net predicts s_{t+K}, bridge is trained
            #                     in absolute state space (legacy behavior).
            #   - 'displacement': subgoal_net predicts Delta = s_{t+K} - s_t and
            #                     the bridge / residual chain is trained in the
            #                     local frame with z0 = 0, zK = Delta.  Aligns
            #                     with the prior that endpoints are small offsets
            #                     from the current state.  External APIs still
            #                     expose absolute states.
            subgoal_target_mode='absolute',
            # Residual/bridge conditioning frame, separate from subgoal_target_mode:
            #   - absolute: residual net uses (s_1, s_K, i/K)
            #   - displacement: residual net uses (s_1, delta, i/K), delta=s_K-s_1
            residual_target_mode='absolute',
            # NOTE: in the standard `main.py` training path, these two are
            # overwritten in `_prepare_configs` to mirror the critic's
            # `value_hidden_dims` / `layer_norm` so that the borrowed critic
            # head loads cleanly. They are honored only when DynamicsAgent is
            # constructed outside of `main.py` (e.g. unit tests).
            subgoal_value_hidden_dims=(512, 512, 512),
            subgoal_value_layer_norm=True,
            subgoal_value_goal_representation='full',
            subgoal_hidden_dims=(512, 512, 512),
            # Distributional subgoal controls (default: deterministic point).
            subgoal_distribution='diag_gaussian',
            # Stochastic subgoals intentionally support either the PDF-style
            # reparameterized sample-MSE objective or a weighted Gaussian NLL
            # objective used by selected experiment configs.
            subgoal_stochastic_loss='nll',
            subgoal_num_samples=4,
            subgoal_log_std_min=-5.0,
            subgoal_log_std_max=1.0,
            subgoal_temperature=1.0,
            subgoal_use_mean_for_actor_goal=True,
            discount=0.99,
            subgoal_steps=25,
            # When False: PathHGCDataset overrides high_actor_targets with the
            # K-step horizon endpoint s_{t+K}, so the dynamics bridge / subgoal_net teacher is
            # always K steps ahead even if the episode goal s_{t_g} is closer than K.
            # Default True: clip per-row to s_{min(t+K, t_g)} for both the bridge endpoint
            # (high_actor_targets) and the subgoal_net teacher, and pad trajectory_segment
            # tail with s_{t_g} for steps beyond t_g. This trains the bridge to "arrive
            # and stay" at close goals so subgoal predictions near the goal stay
            # in-distribution and reduces hovering near the goal.
            clip_path_to_goal=True,
            # Path-supervised planner switch.  Default 'exact_residual_chain'
            # uses the learned residual reverse chain.  Alternatives:
            #   'forward_bridge'          : closed-form forward
            #                               bridge mean (no learned path params).
            #   'forward_bridge_residual' : bridge mean + endpoint-preserving
            #                               learned residual (PathResidualNet).
            planner_type='forward_bridge_residual',
            forward_bridge_mode='mean',
            forward_bridge_use_path_loss=True,
            # If >0, train forward-bridge path supervision only on the prefix
            # steps 1..H (plus the exact endpoint for diagnostics).  This keeps
            # PathResidualNet aligned with the actor/IDM chunk horizon without
            # evaluating all K bridge states on every update.
            forward_bridge_path_loss_horizon=0,
            idm_loss_weight=1.0,
            idm_hidden_dims=(512, 512, 512),
            value_p_curgoal=0.2,
            value_p_trajgoal=0.5,
            value_p_randomgoal=0.3,
            value_geom_sample=False,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            # Optional cap for same-trajectory sampled goals. None/<=0 keeps
            # the historical behavior of sampling up to the episode terminal.
            max_goal_steps=None,
            max_goal_steps_from_env=True,
            gc_negative=True,
            p_aug=0.0,
            frame_stack=ml_collections.config_dict.placeholder(int),
        )
    )


def get_dynamics_config():
    """Defaults for linear-SDE dynamics training."""
    c = _get_common_config()
    c.require_matching_horizon = True
    c.dynamics_loss_weight = 1.0
    c.path_loss_weight = 1.0
    c.rollout_loss_weight = 1.0
    c.rollout_horizon = 5
    c.path_eval_slice = [0, 1]
    c.idm_loss_weight = 1.0
    c.idm_hidden_dims = (512, 512, 512)
    # Dynamics model parameterization: exact bridge posterior mean plus a
    # variance-scaled learned residual.
    c.dynamics_model_type = 'exact_residual'
    c.exact_residual_scale = 1.0
    c.exact_residual_reg_weight = 1.0e-4
    c.exact_residual_bridge_match_weight = 0.0
    return c
