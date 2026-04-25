"""GOUB dynamics agent and shared planner components.

This module is the single source of truth for GOUB training/inference.
Training uses the path-supervised dynamics objective on top of the
endpoint-conditioned bridge planner.
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
from utils.goub import (
    bridge_sample,
    forward_bridge_coefficients,
    make_goub_schedule,
    model_mean,
    posterior_mean,
    sample_from_reverse_mean,
    theta_linear_model_mean,
    theta_linear_posterior_mean,
)


_VALID_PLANNER_TYPES = ('reverse_score', 'forward_bridge', 'forward_bridge_residual')
_VALID_FORWARD_BRIDGE_MODES = ('mean', 'sample')


def _planner_type(config) -> str:
    """Return the canonical planner_type string from the agent config."""
    pt = str(config.get('planner_type', 'reverse_score')).lower()
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


def _bridge_type(config) -> str:
    return str(config.get('bridge_type', 'goub')).lower()


def _is_theta_linear_bridge(config) -> bool:
    return _bridge_type(config) == 'theta_linear'


def _bridge_type_metric(config) -> float:
    bt = _bridge_type(config)
    if bt == 'unidb_gou':
        return 1.0
    if bt == 'theta_linear':
        return 2.0
    return 0.0

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


class GOUBEpsilonNet(nn.Module):
    hidden_dims: Sequence[int]
    state_dim: int
    time_embed_dim: int = 64
    layer_norm: bool = True

    @nn.compact
    def __call__(self, x_n, x_T, x_0, n):
        t_emb = SinusoidalEmbedding(self.time_embed_dim)(n)
        inp = jnp.concatenate([x_n, x_T, x_0, t_emb], axis=-1)
        return MLP(
            hidden_dims=(*self.hidden_dims, self.state_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )(inp)


class SubgoalEstimatorNet(nn.Module):
    """Deterministic point subgoal estimator (legacy, ``subgoal_distribution='deterministic'``)."""

    hidden_dims: Sequence[int]
    state_dim: int
    layer_norm: bool = True

    @nn.compact
    def __call__(self, observations, high_actor_goals):
        inp = jnp.concatenate([observations, high_actor_goals], axis=-1)
        return MLP(
            hidden_dims=(*self.hidden_dims, self.state_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )(inp)


class DistributionalSubgoalEstimatorNet(nn.Module):
    """Diagonal-Gaussian subgoal estimator (``subgoal_distribution='diag_gaussian'``).

    Returns ``(mu, log_std)`` so the GOUB phase1 loss can mix NLL, mean MSE,
    a small log-std regulariser, and the existing critic-V subgoal bonus.
    Stochasticity for actor proposals is sampled from this distribution
    in :meth:`_GOUBAgentCore.sample_subgoal_candidates`.
    """

    hidden_dims: Sequence[int]
    state_dim: int
    layer_norm: bool = True
    log_std_min: float = -5.0
    log_std_max: float = 1.0

    @nn.compact
    def __call__(self, observations, high_actor_goals):
        inp = jnp.concatenate([observations, high_actor_goals], axis=-1)
        trunk = MLP(
            hidden_dims=tuple(self.hidden_dims),
            activate_final=True,
            layer_norm=self.layer_norm,
        )(inp)
        mu = nn.Dense(self.state_dim, name='mu_head')(trunk)
        log_std_raw = nn.Dense(self.state_dim, name='log_std_head')(trunk)
        log_std = jnp.clip(log_std_raw, self.log_std_min, self.log_std_max)
        return mu, log_std


class PathResidualNet(nn.Module):
    """Per-step path residual network for ``forward_bridge_residual`` planner.

    Takes the bridge endpoints ``(z_0, z_K)`` and a normalised time index
    ``t_norm = i / K`` and outputs a per-step residual ``r_theta(z_0, z_K, i)``
    of shape ``(B, T, state_dim)``.  The agent multiplies this residual by a
    quadratic schedule ``w_i = i*(K-i)/K^2`` so that ``w_0 = w_K = 0`` and the
    bridge endpoints are always preserved exactly.
    """

    hidden_dims: Sequence[int]
    state_dim: int
    time_embed_dim: int = 64
    layer_norm: bool = True

    @nn.compact
    def __call__(self, z0, zK, t_norm):
        # z0, zK: (B, D);  t_norm: (B, T) with values in [0, 1].
        t_emb = SinusoidalEmbedding(self.time_embed_dim)(t_norm)  # (B, T, time_embed_dim)
        z0_b = jnp.broadcast_to(z0[:, None, :], (z0.shape[0], t_norm.shape[1], z0.shape[1]))
        zK_b = jnp.broadcast_to(zK[:, None, :], (zK.shape[0], t_norm.shape[1], zK.shape[1]))
        inp = jnp.concatenate([z0_b, zK_b, t_emb], axis=-1)
        return MLP(
            hidden_dims=(*self.hidden_dims, self.state_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )(inp)


def _subgoal_mode(config) -> str:
    return str(config.get('subgoal_distribution', 'deterministic')).lower()


class _GOUBAgentCore(flax.struct.PyTreeNode):
    """Shared GOUB planner / inference core."""

    rng: Any
    network: Any
    schedule: Any
    config: Any = nonpytree_field()

    def _learned_reverse_mean(self, x_n, x_T, x_0, n, schedule, params=None):
        n_total = self.config['goub_N']
        n_safe = jnp.minimum(n, n_total - 1)
        is_boundary = n == n_total

        eps = self.network.select('eps_net')(x_n, x_T, x_0, n.astype(jnp.float32), params=params)
        if _is_theta_linear_bridge(self.config):
            mu_inner = theta_linear_model_mean(x_n, x_0, x_T, eps, n_safe, schedule)
        else:
            mu_inner = model_mean(x_n, x_T, eps, n_safe, schedule)
        mu_boundary = x_T + eps
        mu = jnp.where(is_boundary[..., None], mu_boundary, mu_inner)
        return mu, eps

    def _reverse_step(self, x_n, x_T, x_0, n, rng, stochastic, noise_scale, params=None):
        mu, eps = self._learned_reverse_mean(x_n, x_T, x_0, n, self.schedule, params=params)
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
    def total_loss(self, batch, grad_params, rng=None, critic_value_params=None):
        x_T = batch['observations']
        x_0 = batch['high_actor_targets']
        batch_size = x_T.shape[0]
        n_total = self.config['goub_N']
        sg_w = float(self.config.get('subgoal_loss_weight', 1.0))

        rng1, rng2 = jax.random.split(rng)
        n = jax.random.randint(rng1, (batch_size,), 1, n_total + 1)
        is_boundary = n == n_total
        n_safe = jnp.minimum(n, n_total - 1)

        x_n_bridge = bridge_sample(x_0, x_T, n_safe, self.schedule, rng2)
        x_n = jnp.where(is_boundary[..., None], x_T, x_n_bridge)
        if _is_theta_linear_bridge(self.config):
            mu_true = theta_linear_posterior_mean(x_n, x_0, x_T, n, self.schedule)
        else:
            mu_true = posterior_mean(x_n, x_0, x_T, n, self.schedule)
        mu_pred, eps_pred = self._learned_reverse_mean(x_n, x_T, x_0, n, self.schedule, params=grad_params)

        g2_n = self.schedule['g2'][n - 1]
        weight = 1.0 / (2.0 * jnp.maximum(g2_n, 1e-12))
        loss_goub = (weight * jnp.abs(mu_true - mu_pred).sum(axis=-1)).mean()

        sg_out = self.network.select('subgoal_net')(
            batch['observations'],
            batch['high_actor_goals'],
            params=grad_params,
        )
        pred_sg = sg_out[0] if isinstance(sg_out, tuple) else sg_out
        loss_sub = jnp.mean((pred_sg - batch['high_actor_targets']) ** 2)
        loss = loss_goub + sg_w * loss_sub

        idm_term, loss_idm_unw = self._idm_loss_term(batch, grad_params)
        loss = loss + idm_term

        n_N = jnp.full((batch_size,), n_total, dtype=jnp.int32)
        xNm1, _ = self._learned_reverse_mean(x_T, x_T, x_0, n_N, self.schedule, params=grad_params)
        xNm1_norm = jnp.linalg.norm(xNm1, axis=-1).mean()

        info = {
            'phase1/loss': loss,
            'phase1/loss_goub': loss_goub,
            'phase1/loss_subgoal': loss_sub,
            'phase1/loss_idm': loss_idm_unw,
            'phase1/eps_norm': jnp.linalg.norm(eps_pred, axis=-1).mean(),
            'phase1/mu_true_norm': jnp.linalg.norm(mu_true, axis=-1).mean(),
            'phase1/mu_pred_norm': jnp.linalg.norm(mu_pred, axis=-1).mean(),
            'phase1/xN_minus_1_norm': xNm1_norm,
            'phase1/bridge_step_mean': n.astype(jnp.float32).mean(),
        }
        info['phase1/subgoal_pred_norm'] = jnp.linalg.norm(pred_sg, axis=-1).mean()
        info['phase1/subgoal_target_norm'] = jnp.linalg.norm(batch['high_actor_targets'], axis=-1).mean()
        info['bridge/bridge_type'] = jnp.asarray(_bridge_type_metric(self.config), dtype=jnp.float32)
        info['bridge/bridge_gamma'] = jnp.asarray(
            float(self.config.get('bridge_gamma', 1.0e7)), dtype=jnp.float32
        )
        return loss, info

    @jax.jit
    def update(self, batch, critic_value_params=None):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng, critic_value_params=critic_value_params)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    # ------------------------------------------------------------------
    # GOUB forward-bridge planner (closed-form, endpoint-conditioned)
    # ------------------------------------------------------------------
    #
    # For ``planner_type='reverse_score'`` (default) ``plan()`` / ``sample_plan()``
    # take the existing learned-eps reverse chain.  For
    # ``planner_type='forward_bridge'`` and ``planner_type='forward_bridge_residual'``
    # they instead branch into the closed-form Gaussian bridge below.  The
    # convention follows DOURI's state-space *forward* time direction:
    #
    #     z_0 = current state          (= x_T in legacy GOUB diffusion-time)
    #     z_K = subgoal endpoint       (= x_0 in legacy GOUB diffusion-time)
    #     trajectory[:, 0] = z_0
    #     trajectory[:, K] = z_K
    #
    # which already matches the existing reverse-chain output convention
    # (``traj[:, 0] = current_state`` and ``traj[:, -1]`` ~ predicted subgoal),
    # so downstream consumers (IDM action decoding, rollout/subgoal.py, etc.)
    # can be re-used without changes.

    def forward_bridge_coefficients(self, K, *, eps: float | None = None):
        """Return ``(a, b, std)`` of shape ``(K + 1,)`` for the GOUB forward bridge.

        Coefficients satisfy ``z_i | z_0, z_K ~ N(a_i z_0 + b_i z_K, std_i^2 I)``
        with the exact endpoint constraints
        ``a[0]=1, b[0]=0, std[0]=0`` and ``a[K]=0, b[K]=1, std[K]=0``.

        ``K`` is treated as a *static* Python int (it determines the size of
        the returned arrays); ``eps`` defaults to ``self.config['forward_bridge_eps']``
        when ``None``.
        """
        eps_val = float(self.config.get('forward_bridge_eps', 1.0e-6)) if eps is None else float(eps)
        return forward_bridge_coefficients(
            int(K),
            beta_min=float(self.config['goub_beta_min']),
            beta_max=float(self.config['goub_beta_max']),
            lambda_=float(self.config['goub_lambda']),
            eps=eps_val,
            bridge_type=str(self.config.get('bridge_type', 'goub')),
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
            num_steps: planning horizon ``K``; defaults to ``self.config['goub_N']``.
            rng: required when ``sample=True`` and ``noise_scale > 0``.

        Returns:
            ``path`` of shape ``(B, K + 1, state_dim)``.
        """
        n_total = int(self.config['goub_N'])
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
    ) -> jnp.ndarray:
        """Forward bridge mean + endpoint-preserving learned residual.

        ``z_hat_i = mu_i + w_i * r_theta(z_0, z_K, i)`` where
        ``mu_i = a_i z_0 + b_i z_K`` is the closed-form bridge mean,
        ``r_theta`` is :class:`PathResidualNet`, and the quadratic schedule
        ``w_i = i*(K - i)/K^2`` zeros out at the endpoints so ``z_hat_0 = z_0``
        and ``z_hat_K = z_K`` are preserved exactly (modulo the explicit
        endpoint clamp at the end).
        """
        n_total = int(self.config['goub_N'])
        K = n_total if num_steps is None else int(num_steps)
        if K < 1 or K > n_total:
            raise ValueError(f'num_steps must be in [1, {n_total}], got {K}.')

        a, b, std = self.forward_bridge_coefficients(K)
        mu = a[None, :, None] * z0[:, None, :] + b[None, :, None] * zK[:, None, :]

        idx = jnp.arange(K + 1, dtype=jnp.float32)
        w = idx * (float(K) - idx) / float(K * K)
        t_norm = jnp.broadcast_to(idx[None, :] / float(K), (z0.shape[0], K + 1))

        residual = self.network.select('path_residual_net')(z0, zK, t_norm, params=params)
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

    def _reverse_score_plan(self, current_state, desired_endpoint, num_steps: int | None = None):
        x_T = current_state
        x_0_goal = desired_endpoint
        n_total = int(self.config['goub_N'])
        steps_to_roll = n_total if num_steps is None else int(num_steps)
        if steps_to_roll < 1 or steps_to_roll > n_total:
            raise ValueError(f'num_steps must be in [1, {n_total}], got {steps_to_roll}.')
        batch_size = x_T.shape[0]

        def scan_body(x, step_n):
            n = jnp.full((batch_size,), step_n, dtype=jnp.int32)
            x_new, _ = self._learned_reverse_mean(x, x_T, x_0_goal, n, self.schedule)
            return x_new, x_new

        steps = jnp.arange(n_total, n_total - steps_to_roll, -1)
        _, traj_body = jax.lax.scan(scan_body, x_T, steps)
        traj = jnp.concatenate([x_T[None], traj_body], axis=0)
        return jnp.swapaxes(traj, 0, 1)

    def _reverse_score_sample_plan(
        self, current_state, desired_endpoint, rng, noise_scale: float, num_steps: int | None = None
    ):
        x_T = current_state
        x_0_goal = desired_endpoint
        n_total = int(self.config['goub_N'])
        steps_to_roll = n_total if num_steps is None else int(num_steps)
        if steps_to_roll < 1 or steps_to_roll > n_total:
            raise ValueError(f'num_steps must be in [1, {n_total}], got {steps_to_roll}.')
        batch_size = x_T.shape[0]
        step_rngs = jax.random.split(rng, steps_to_roll)

        def scan_body(x, inputs):
            step_n, step_rng = inputs
            n = jnp.full((batch_size,), step_n, dtype=jnp.int32)
            x_new, _ = self._reverse_step(x, x_T, x_0_goal, n, step_rng, True, noise_scale, params=None)
            return x_new, x_new

        steps = jnp.arange(n_total, n_total - steps_to_roll, -1)
        _, traj_body = jax.lax.scan(scan_body, x_T, (steps, step_rngs))
        traj = jnp.concatenate([x_T[None], traj_body], axis=0)
        return jnp.swapaxes(traj, 0, 1)

    @partial(jax.jit, static_argnames=('num_steps',))
    def plan(self, current_state, desired_endpoint, *, num_steps: int | None = None):
        squeeze = current_state.ndim == 1
        if squeeze:
            current_state = current_state[None]
            desired_endpoint = desired_endpoint[None]

        planner = _planner_type(self.config)
        if planner == 'forward_bridge':
            traj = self.forward_bridge_plan(
                current_state, desired_endpoint,
                sample=False, noise_scale=0.0, num_steps=num_steps,
            )
        elif planner == 'forward_bridge_residual':
            traj = self.forward_bridge_residual_plan(
                current_state, desired_endpoint,
                sample=False, noise_scale=0.0, num_steps=num_steps,
            )
        else:
            traj = self._reverse_score_plan(current_state, desired_endpoint, num_steps=num_steps)

        result = {'next_step': traj[:, 1, :], 'trajectory': traj}
        if squeeze:
            result = jax.tree_util.tree_map(lambda x: x[0], result)
        return result

    @partial(jax.jit, static_argnames=('noise_scale', 'num_steps'))
    def sample_plan(self, current_state, desired_endpoint, rng, noise_scale: float = 1.0, num_steps: int | None = None):
        squeeze = current_state.ndim == 1
        if squeeze:
            current_state = current_state[None]
            desired_endpoint = desired_endpoint[None]

        planner = _planner_type(self.config)
        if planner == 'forward_bridge':
            sample_flag = _forward_bridge_mode(self.config) == 'sample'
            traj = self.forward_bridge_plan(
                current_state, desired_endpoint,
                sample=sample_flag, noise_scale=noise_scale,
                num_steps=num_steps, rng=rng,
            )
        elif planner == 'forward_bridge_residual':
            sample_flag = _forward_bridge_mode(self.config) == 'sample'
            traj = self.forward_bridge_residual_plan(
                current_state, desired_endpoint,
                sample=sample_flag, noise_scale=noise_scale,
                num_steps=num_steps, rng=rng,
            )
        else:
            traj = self._reverse_score_sample_plan(
                current_state, desired_endpoint, rng, noise_scale, num_steps=num_steps,
            )

        result = {'next_step': traj[:, 1, :], 'trajectory': traj}
        if squeeze:
            result = jax.tree_util.tree_map(lambda x: x[0], result)
        return result

    def _sample_plan_trajectory(
        self, current_state, desired_endpoint, rng, noise_scale: float, num_steps: int | None = None
    ):
        planner = _planner_type(self.config)
        if planner == 'forward_bridge':
            sample_flag = _forward_bridge_mode(self.config) == 'sample'
            return self.forward_bridge_plan(
                current_state, desired_endpoint,
                sample=sample_flag, noise_scale=noise_scale,
                num_steps=num_steps, rng=rng,
            )
        if planner == 'forward_bridge_residual':
            sample_flag = _forward_bridge_mode(self.config) == 'sample'
            return self.forward_bridge_residual_plan(
                current_state, desired_endpoint,
                sample=sample_flag, noise_scale=noise_scale,
                num_steps=num_steps, rng=rng,
            )
        return self._reverse_score_sample_plan(
            current_state, desired_endpoint, rng, noise_scale, num_steps=num_steps,
        )

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
    ):
        squeeze = current_state.ndim == 1
        if squeeze:
            current_state = current_state[None]
            desired_endpoint = desired_endpoint[None]

        if include_mean:
            det = self.plan(current_state, desired_endpoint, num_steps=num_steps)['trajectory'][:, None, ...]
            if num_candidates == 1:
                out = det
            else:
                sample_rngs = jax.random.split(rng, num_candidates - 1)
                sampled = jax.vmap(
                    lambda r: self._sample_plan_trajectory(
                        current_state, desired_endpoint, r, noise_scale, num_steps=num_steps
                    ),
                    in_axes=0,
                )(sample_rngs)
                sampled = jnp.swapaxes(sampled, 0, 1)
                out = jnp.concatenate([det, sampled], axis=1)
        else:
            sample_rngs = jax.random.split(rng, num_candidates)
            sampled = jax.vmap(
                lambda r: self._sample_plan_trajectory(
                    current_state, desired_endpoint, r, noise_scale, num_steps=num_steps
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
        """Backward-compatible: returns the deterministic / mean subgoal point."""
        squeeze = observations.ndim == 1
        if squeeze:
            observations = observations[None]
            high_actor_goals = high_actor_goals[None]
        out = self._subgoal_forward(observations, high_actor_goals)
        if isinstance(out, tuple):
            out = out[0]
        if squeeze:
            out = out[0]
        return out

    @jax.jit
    def infer_subgoal(self, observations, high_actor_goals):
        """Backward-compatible alias for :meth:`predict_subgoal`."""
        return self.predict_subgoal(observations, high_actor_goals)

    @jax.jit
    def infer_subgoal_mean(self, observations, high_actor_goals):
        """Distributional API: returns mu (== deterministic point in legacy mode)."""
        return self.predict_subgoal(observations, high_actor_goals)

    @jax.jit
    def infer_subgoal_distribution(self, observations, high_actor_goals):
        """Distributional API: returns ``(mu, log_std)``.

        In deterministic mode ``log_std`` is filled with ``log_std_min`` so the
        distribution degenerates to a point mass under any sampler.
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
            mu, log_std = out
            std = jnp.exp(log_std) * float(self.config.get('subgoal_temperature', 1.0))
            n_sample = num_candidates - 1 if include_mean else num_candidates
            if n_sample > 0:
                eps = jax.random.normal(rng, (n_sample, mu.shape[0], mu.shape[-1]))
                sampled = mu[None, :, :] + eps * std[None, :, :]
                sampled = jnp.swapaxes(sampled, 0, 1)  # [B, n_sample, D]
            else:
                sampled = jnp.zeros((mu.shape[0], 0, mu.shape[-1]), dtype=mu.dtype)
            if include_mean:
                candidates = jnp.concatenate([mu[:, None, :], sampled], axis=1)
            else:
                candidates = sampled
        else:
            mu = out
            candidates = jnp.broadcast_to(mu[:, None, :], (mu.shape[0], num_candidates, mu.shape[-1]))

        if squeeze:
            mu = mu[0]
            candidates = candidates[0]
        return candidates, mu

    @jax.jit
    def plan_from_high_goal(self, current_state, high_actor_goals):
        endpoint = self.predict_subgoal(current_state, high_actor_goals)
        return self.plan(current_state, endpoint)

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
            Decoded action chunks for the ``N = plan_candidates`` proposals.
        candidate_goals : ``[B, N, D]``
            Per-candidate subgoal endpoints used both to drive bridge
            sampling and to rescore each candidate with its own goal.  In
            deterministic mode this is just the mean broadcast across ``N``.
        new_rng : updated PRNG key.
        """
        obs = jnp.asarray(observations, dtype=jnp.float32)
        goals = jnp.asarray(high_actor_goals, dtype=jnp.float32)
        sub_mode = _subgoal_mode(self.config)

        if sub_mode == 'diag_gaussian':
            sub_rng, plan_rng, new_rng = jax.random.split(rng, 3)
            candidate_goals, mu = self.sample_subgoal_candidates(
                obs,
                goals,
                sub_rng,
                num_candidates=plan_candidates,
                include_mean=True,
            )
            per_rngs = jax.random.split(plan_rng, plan_candidates)
            cand_endpoints = jnp.swapaxes(candidate_goals, 0, 1)  # [N, B, D]
            traj_noise = float(sample_noise_scale)

            def _per_candidate(rng_k, endpoint_k):
                return self._sample_plan_trajectory(
                    obs, endpoint_k, rng_k, traj_noise, num_steps=proposal_horizon
                )

            candidate_trajectories = jax.vmap(_per_candidate, in_axes=(0, 0))(per_rngs, cand_endpoints)
            candidate_trajectories = jnp.swapaxes(candidate_trajectories, 0, 1)  # [B, N, K+1, D]
        else:
            mu = self._subgoal_forward(obs, goals)
            if isinstance(mu, tuple):  # safety net (shouldn't happen in deterministic mode)
                mu = mu[0]
            if plan_candidates == 1:
                sampled = self.sample_plan(
                    obs,
                    mu,
                    rng,
                    noise_scale=0.0,
                    num_steps=proposal_horizon,
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
                    num_steps=proposal_horizon,
                )
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
        assert config['goub_N'] >= 2, 'GOUB requires N >= 2 diffusion steps.'
        if ex_actions is None:
            raise ValueError(
                '_GOUBAgentCore.create requires ex_actions shaped (B, A) to build the inverse-dynamics head.'
            )

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        state_dim = ex_observations.shape[-1]
        action_dim = int(ex_actions.shape[-1])
        idm_hidden = config.get('idm_hidden_dims', (512, 512, 512))
        if isinstance(idm_hidden, str):
            idm_hidden = parse_hidden_dims(idm_hidden)
        else:
            idm_hidden = tuple(int(x) for x in idm_hidden)

        schedule = make_goub_schedule(
            N=config['goub_N'],
            beta_min=config['goub_beta_min'],
            beta_max=config['goub_beta_max'],
            lambda_=config['goub_lambda'],
            bridge_type=str(config.get('bridge_type', 'goub')),
            bridge_gamma=float(config.get('bridge_gamma', 1.0e7)),
        )

        eps_net_def = GOUBEpsilonNet(
            hidden_dims=tuple(config['eps_hidden_dims']),
            state_dim=state_dim,
            time_embed_dim=config['time_embed_dim'],
            layer_norm=config['layer_norm'],
        )
        sub_mode = _subgoal_mode(config)
        if sub_mode == 'deterministic':
            subgoal_def = SubgoalEstimatorNet(
                hidden_dims=tuple(config['subgoal_hidden_dims']),
                state_dim=state_dim,
                layer_norm=config['layer_norm'],
            )
        elif sub_mode == 'diag_gaussian':
            subgoal_def = DistributionalSubgoalEstimatorNet(
                hidden_dims=tuple(config['subgoal_hidden_dims']),
                state_dim=state_dim,
                layer_norm=config['layer_norm'],
                log_std_min=float(config.get('subgoal_log_std_min', -5.0)),
                log_std_max=float(config.get('subgoal_log_std_max', 1.0)),
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

        network_info = dict(
            eps_net=(eps_net_def, (dummy_x, dummy_x, dummy_x, dummy_n)),
            subgoal_net=(subgoal_def, (dummy_x, dummy_g)),
            idm_net=(idm_def, (dummy_x, dummy_next)),
        )
        # Optionally register the forward-bridge residual MLP.  Only created when
        # ``planner_type == 'forward_bridge_residual'`` so existing reverse-score
        # checkpoints stay loadable bit-for-bit (no extra params introduced).
        planner_type = str(config.get('planner_type', 'reverse_score')).lower()
        if planner_type == 'forward_bridge_residual':
            residual_hidden = config.get('residual_hidden_dims', config.get('eps_hidden_dims', (512, 512, 512)))
            if isinstance(residual_hidden, str):
                residual_hidden = parse_hidden_dims(residual_hidden)
            else:
                residual_hidden = tuple(int(x) for x in residual_hidden)
            residual_def = PathResidualNet(
                hidden_dims=residual_hidden,
                state_dim=state_dim,
                time_embed_dim=int(config['time_embed_dim']),
                layer_norm=bool(config['layer_norm']),
            )
            horizon = int(config['goub_N']) + 1
            dummy_t = jnp.zeros((batch_size, horizon), dtype=jnp.float32)
            network_info['path_residual_net'] = (residual_def, (dummy_x, dummy_x, dummy_t))
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        cfg_out = {**dict(config), 'idm_action_dim': action_dim, 'idm_hidden_dims': idm_hidden}
        return cls(
            rng=rng,
            network=network,
            schedule=schedule,
            config=flax.core.FrozenDict(**cfg_out),
        )


class GOUBDynamicsAgent(_GOUBAgentCore):
    """GOUB dynamics agent with path supervision and rollout consistency."""

    @classmethod
    def create(cls, seed, ex_observations, config, ex_actions=None):
        if bool(config.get('require_matching_horizon', True)):
            gn = int(config['goub_N'])
            sk = int(config['subgoal_steps'])
            if gn != sk:
                raise ValueError(
                    f'GOUBDynamics: require_matching_horizon expects goub_N ({gn}) == '
                    f'subgoal_steps ({sk}). Disable with require_matching_horizon: false '
                    'only if you accept misaligned indices.'
                )
        return super().create(seed, ex_observations, config, ex_actions=ex_actions)

    def _subgoal_value_bonus(
        self,
        pred_subgoals: jnp.ndarray,
        high_actor_goals: jnp.ndarray,
        critic_value_params: Any | None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        alpha = float(self.config.get('subgoal_value_alpha', 0.0))
        zeros = jnp.zeros((pred_subgoals.shape[0],), dtype=jnp.float32)
        if alpha <= 0.0 or critic_value_params is None:
            return zeros, zeros

        value_def = ScalarValueNet(
            tuple(int(x) for x in self.config.get('subgoal_value_hidden_dims', (512, 512, 512))),
            layer_norm=bool(self.config.get('subgoal_value_layer_norm', True)),
        )
        value_logits = value_def.apply({'params': critic_value_params}, pred_subgoals, high_actor_goals)
        value = jax.nn.sigmoid(jnp.asarray(value_logits, dtype=jnp.float32))
        return value, jnp.asarray(alpha, dtype=jnp.float32) * value

    def _compute_subgoal_loss(self, batch, grad_params, rng_fr, critic_value_params):
        """Compute the subgoal-net training loss + companion logging tensors.

        Shared by both the legacy reverse-score path and the
        ``forward_bridge`` / ``forward_bridge_residual`` path so the subgoal
        estimator is trained identically across planners.
        """
        s = batch['observations']
        g_high = batch['high_actor_goals']
        target = batch['high_actor_targets']
        sub_mode = _subgoal_mode(self.config)

        if sub_mode == 'diag_gaussian':
            pred_mu, pred_log_std = self.network.select('subgoal_net')(s, g_high, params=grad_params)
            pred_std = jnp.exp(pred_log_std)
            inv_var = jnp.exp(-2.0 * pred_log_std)
            diff_sg = target - pred_mu
            nll_per_sample = 0.5 * jnp.sum(
                diff_sg ** 2 * inv_var + 2.0 * pred_log_std + jnp.log(2.0 * jnp.pi), axis=-1
            )
            subgoal_mse = jnp.mean(diff_sg ** 2, axis=-1)
            var_reg_per_sample = jnp.mean(pred_log_std ** 2, axis=-1)

            w_fr = float(self.config.get('subgoal_fr_spi_weight', 0.0))
            if w_fr > 0.0 and critic_value_params is not None:
                k_fr = 4
                fr_eps = jax.random.normal(rng_fr, (s.shape[0], k_fr, target.shape[-1]))
                cand = jax.lax.stop_gradient(
                    pred_mu[:, None, :] + pred_std[:, None, :] * fr_eps
                )
                value_def = ScalarValueNet(
                    tuple(int(x) for x in self.config.get('subgoal_value_hidden_dims', (512, 512, 512))),
                    layer_norm=bool(self.config.get('subgoal_value_layer_norm', True)),
                )
                cand_flat = cand.reshape(-1, cand.shape[-1])
                g_high_flat = jnp.repeat(g_high[:, None, :], k_fr, axis=1).reshape(-1, g_high.shape[-1])
                v_logits = value_def.apply({'params': critic_value_params}, cand_flat, g_high_flat)
                v_cand = jax.nn.sigmoid(jnp.asarray(v_logits, dtype=jnp.float32)).reshape(s.shape[0], k_fr)
                tau_fr = float(self.config.get('subgoal_fr_spi_tau', 1.0))
                rho_fr = jax.lax.stop_gradient(jax.nn.softmax(v_cand / jnp.maximum(tau_fr, 1e-6), axis=1))
                log_q = -0.5 * jnp.sum(
                    ((cand - pred_mu[:, None, :]) ** 2) * inv_var[:, None, :]
                    + 2.0 * pred_log_std[:, None, :]
                    + jnp.log(2.0 * jnp.pi),
                    axis=-1,
                )
                fr_term = -jnp.mean(jnp.sum(rho_fr * log_q, axis=1))
            else:
                fr_term = jnp.asarray(0.0, dtype=jnp.float32)

            subgoal_value, subgoal_value_bonus = self._subgoal_value_bonus(
                pred_mu, g_high, critic_value_params
            )

            w_nll = float(self.config.get('subgoal_nll_weight', 1.0))
            w_mse = float(self.config.get('subgoal_mse_weight', 0.25))
            w_var = float(self.config.get('subgoal_var_reg_weight', 1.0e-4))
            loss_sub = (
                w_nll * jnp.mean(nll_per_sample)
                + w_mse * jnp.mean(subgoal_mse)
                + w_var * jnp.mean(var_reg_per_sample)
                - jnp.mean(subgoal_value_bonus)
                + w_fr * fr_term
            )
            pred_sg_out = pred_mu
            subgoal_extra_info = {
                'phase1/subgoal_nll': jnp.mean(nll_per_sample),
                'phase1/subgoal_std_mean': jnp.mean(pred_std),
                'phase1/subgoal_std_max': jnp.max(pred_std),
                'phase1/subgoal_fr_spi': fr_term,
                'phase1/subgoal_mode': jnp.asarray(1.0, dtype=jnp.float32),
            }
        else:
            pred_sg = self.network.select('subgoal_net')(s, g_high, params=grad_params)
            subgoal_mse = jnp.mean((pred_sg - target) ** 2, axis=-1)
            subgoal_value, subgoal_value_bonus = self._subgoal_value_bonus(
                pred_sg, g_high, critic_value_params
            )
            loss_sub = jnp.mean(subgoal_mse) - jnp.mean(subgoal_value_bonus)
            pred_sg_out = pred_sg
            zero = jnp.asarray(0.0, dtype=jnp.float32)
            subgoal_extra_info = {
                'phase1/subgoal_nll': zero,
                'phase1/subgoal_std_mean': zero,
                'phase1/subgoal_std_max': zero,
                'phase1/subgoal_fr_spi': zero,
                'phase1/subgoal_mode': zero,
            }
        return loss_sub, subgoal_mse, subgoal_value, subgoal_value_bonus, pred_sg_out, subgoal_extra_info

    def _path_eval_slice(self) -> tuple[int, ...]:
        pev = self.config.get('path_eval_slice')
        if pev is None:
            return (0, 1)
        return tuple(int(x) for x in pev)

    def _total_loss_reverse_score(self, batch, grad_params, rng, critic_value_params):
        """Legacy GOUB mean-matching + path-aligned reverse + short rollout + subgoal MSE."""
        x_T = batch['observations']
        x_0 = batch['high_actor_targets']
        segment = jnp.asarray(batch['trajectory_segment'], dtype=jnp.float32)
        B = x_T.shape[0]
        N = int(self.config['goub_N'])
        K = int(segment.shape[1]) - 1
        sg_w = float(self.config.get('subgoal_loss_weight', 1.0))
        w_g = float(self.config.get('goub_loss_weight', 1.0))
        w_p = float(self.config.get('path_loss_weight', 1.0))
        w_r = float(self.config.get('rollout_loss_weight', 0.25))

        rng1, rng2, rng_fr = jax.random.split(rng, 3)
        n = jax.random.randint(rng1, (B,), 1, N + 1)

        is_boundary = n == N
        n_safe = jnp.minimum(n, N - 1)

        # --- L_goub ---
        x_n_bridge = bridge_sample(x_0, x_T, n_safe, self.schedule, rng2)
        x_n = jnp.where(is_boundary[..., None], x_T, x_n_bridge)
        if _is_theta_linear_bridge(self.config):
            mu_true = theta_linear_posterior_mean(x_n, x_0, x_T, n, self.schedule)
        else:
            mu_true = posterior_mean(x_n, x_0, x_T, n, self.schedule)
        mu_pred, eps_pred = self._learned_reverse_mean(
            x_n, x_T, x_0, n, self.schedule, params=grad_params,
        )
        g2_n = self.schedule['g2'][n - 1]
        weight = 1.0 / (2.0 * jnp.maximum(g2_n, 1e-12))
        loss_goub = (weight * jnp.abs(mu_true - mu_pred).sum(axis=-1)).mean()

        # --- L_path: real x_n along segment, same n ---
        row = jnp.arange(B, dtype=jnp.int32)
        x_n_real = segment[row, K - n, :]
        x_prev_real = segment[row, K - n + 1, :]
        mu_pred_path, _ = self._learned_reverse_mean(
            x_n_real, x_T, x_0, n, self.schedule, params=grad_params,
        )
        diff_p = mu_pred_path - x_prev_real
        loss_path = jnp.abs(diff_p).sum(axis=-1).mean()

        # --- L_roll: recursive deterministic reverse vs segment prefix ---
        H_cfg = int(self.config.get('rollout_horizon', 5))
        H_eff = max(1, min(H_cfg, N))
        hs = jnp.arange(1, H_eff + 1, dtype=jnp.int32)

        def roll_body(x, h):
            step_n = N - h + 1
            n_b = jnp.full((B,), step_n, dtype=jnp.int32)
            mu_r, _ = self._learned_reverse_mean(
                x, x_T, x_0, n_b, self.schedule, params=grad_params,
            )
            tgt = segment[row, h, :]
            err = jnp.abs(mu_r - tgt).sum(axis=-1)
            return mu_r, err

        _, errs = jax.lax.scan(roll_body, segment[:, 0, :], hs)
        loss_roll = jnp.mean(errs)

        # --- L_subgoal ---
        (loss_sub, subgoal_mse, subgoal_value, subgoal_value_bonus,
         pred_sg_out, subgoal_extra_info) = self._compute_subgoal_loss(
            batch, grad_params, rng_fr, critic_value_params,
        )

        loss = w_g * loss_goub + w_p * loss_path + w_r * loss_roll + sg_w * loss_sub

        idm_term, loss_idm_unw = self._idm_loss_term(batch, grad_params)
        loss = loss + idm_term

        n_N = jnp.full((B,), N, dtype=jnp.int32)
        xNm1, _ = self._learned_reverse_mean(
            x_T, x_T, x_0, n_N, self.schedule, params=grad_params,
        )
        xNm1_norm = jnp.linalg.norm(xNm1, axis=-1).mean()

        s1 = segment[:, 1, :]
        first_step_l1 = jnp.abs(xNm1 - s1).sum(axis=-1).mean()
        idx_xy = jnp.asarray(self._path_eval_slice(), dtype=jnp.int32)
        d_xy = xNm1[:, idx_xy] - s1[:, idx_xy]
        first_step_xy_l2 = jnp.sqrt(jnp.mean(d_xy**2))

        info = {
            'phase1/loss': loss,
            'phase1/loss_goub': loss_goub,
            'phase1/loss_path_step': loss_path,
            'phase1/loss_roll': loss_roll,
            'phase1/loss_subgoal': loss_sub,
            'phase1/loss_subgoal_mse': subgoal_mse.mean(),
            'phase1/subgoal_value_mean': subgoal_value.mean(),
            'phase1/subgoal_value_bonus_mean': subgoal_value_bonus.mean(),
            'phase1/loss_idm': loss_idm_unw,
            'phase1/first_step_l1': first_step_l1,
            'phase1/first_step_xy_l2': first_step_xy_l2,
            'phase1/roll_h_l1': loss_roll,
            'phase1/eps_norm': jnp.linalg.norm(eps_pred, axis=-1).mean(),
            'phase1/mu_true_norm': jnp.linalg.norm(mu_true, axis=-1).mean(),
            'phase1/mu_pred_norm': jnp.linalg.norm(mu_pred, axis=-1).mean(),
            'phase1/xN_minus_1_norm': xNm1_norm,
            'phase1/bridge_step_mean': n.astype(jnp.float32).mean(),
            'phase1/planner_type': jnp.asarray(0.0, dtype=jnp.float32),
        }
        info['phase1/subgoal_pred_norm'] = jnp.linalg.norm(pred_sg_out, axis=-1).mean()
        info['phase1/subgoal_target_norm'] = jnp.linalg.norm(batch['high_actor_targets'], axis=-1).mean()
        info.update(subgoal_extra_info)
        info['bridge/bridge_type'] = jnp.asarray(_bridge_type_metric(self.config), dtype=jnp.float32)
        info['bridge/bridge_gamma'] = jnp.asarray(
            float(self.config.get('bridge_gamma', 1.0e7)), dtype=jnp.float32
        )
        return loss, info

    def _total_loss_forward_bridge(self, batch, grad_params, rng, critic_value_params, planner: str):
        """Forward-bridge path-supervised loss (planner_type in {forward_bridge,
        forward_bridge_residual}).

        - Bridge mean (and optional learned residual) is teacher-forced with the
          *true* segment endpoints ``z_0 = segment[:, 0]`` and
          ``z_K = segment[:, -1]``, so ``loss_path`` directly measures interior
          MSE against the on-policy segment.
        - Reverse-score / boundary / rollout losses are skipped (logged as 0).
        - The subgoal estimator (``subgoal_net``) is trained exactly as in
          reverse-score mode via ``_compute_subgoal_loss``.
        - IDM is always trained (separate network).
        """
        x_T = batch['observations']
        x_0 = batch['high_actor_targets']
        segment = jnp.asarray(batch['trajectory_segment'], dtype=jnp.float32)
        N = int(self.config['goub_N'])
        K = int(segment.shape[1]) - 1
        sg_w = float(self.config.get('subgoal_loss_weight', 1.0))
        w_p = float(self.config.get('path_loss_weight', 1.0))

        # NOTE: forward-bridge planner expects ``goub_N == subgoal_steps == K``.
        # ``GOUBDynamicsAgent.create`` already validates ``goub_N == subgoal_steps``;
        # asserting against ``segment.shape[1]`` here would be redundant, and a
        # hard JIT assert is expensive, so rely on PathHGCDataset/horizon plumbing.
        _, _, rng_fr = jax.random.split(rng, 3)

        z0 = segment[:, 0, :]
        zK = segment[:, -1, :]

        if planner == 'forward_bridge_residual':
            path_pred = self.forward_bridge_residual_plan(
                z0, zK, sample=False, noise_scale=0.0, num_steps=N, params=grad_params,
            )
        else:
            path_pred = self.forward_bridge_plan(
                z0, zK, sample=False, noise_scale=0.0, num_steps=N,
            )

        # Teacher-forced path loss (interior steps + first step).  ``loss_next``
        # is reported separately so it is comparable to reverse-score's
        # ``phase1/first_step_l1`` and ``phase1/loss_path_step``.
        diff_next = path_pred[:, 1, :] - segment[:, 1, :]
        loss_next = jnp.mean(jnp.abs(diff_next).sum(axis=-1))
        if K >= 2:
            diff_interior = path_pred[:, 1:-1, :] - segment[:, 1:-1, :]
            loss_path = jnp.mean(jnp.abs(diff_interior).sum(axis=-1))
        else:
            loss_path = jnp.zeros((), dtype=jnp.float32)

        zero = jnp.asarray(0.0, dtype=jnp.float32)
        use_path = bool(self.config.get('forward_bridge_use_path_loss', True))
        path_term = (loss_path + loss_next) if use_path else zero

        (loss_sub, subgoal_mse, subgoal_value, subgoal_value_bonus,
         pred_sg_out, subgoal_extra_info) = self._compute_subgoal_loss(
            batch, grad_params, rng_fr, critic_value_params,
        )

        loss = w_p * path_term + sg_w * loss_sub

        idm_term, loss_idm_unw = self._idm_loss_term(batch, grad_params)
        loss = loss + idm_term

        # Forward-bridge specific diagnostics.
        bridge_path_mse = jnp.mean((path_pred[:, 1:-1, :] - segment[:, 1:-1, :]) ** 2) if K >= 2 else zero
        bridge_next_mse = jnp.mean((path_pred[:, 1, :] - segment[:, 1, :]) ** 2)
        bridge_final_mse = jnp.mean((path_pred[:, -1, :] - segment[:, -1, :]) ** 2)
        bridge_endpoint_start_mse = jnp.mean((path_pred[:, 0, :] - segment[:, 0, :]) ** 2)
        bridge_endpoint_end_mse = bridge_final_mse  # alias kept for compatibility with spec

        # Per-step distance to subgoal (averaged over batch).  We report scalar
        # samples for csv/wandb compatibility (no vector logging required).
        dist_per_step = jnp.linalg.norm(path_pred - zK[:, None, :], axis=-1).mean(axis=0)
        first_idx = 1 if K >= 1 else 0
        mid_idx = max(1, K // 2)
        last_idx = K

        s1 = segment[:, 1, :]
        first_step_l1 = jnp.mean(jnp.abs(path_pred[:, 1, :] - s1).sum(axis=-1))
        idx_xy = jnp.asarray(self._path_eval_slice(), dtype=jnp.int32)
        d_xy = path_pred[:, 1, :][:, idx_xy] - s1[:, idx_xy]
        first_step_xy_l2 = jnp.sqrt(jnp.mean(d_xy ** 2))

        planner_id = 1.0 if planner == 'forward_bridge' else 2.0
        info = {
            'phase1/loss': loss,
            # Reverse-score legacy keys logged as 0 to keep the metrics dict
            # shape-stable for existing csv/wandb consumers.
            'phase1/loss_goub': zero,
            'phase1/loss_roll': zero,
            'phase1/loss_path_step': loss_next,
            'phase1/loss_subgoal': loss_sub,
            'phase1/loss_subgoal_mse': subgoal_mse.mean(),
            'phase1/subgoal_value_mean': subgoal_value.mean(),
            'phase1/subgoal_value_bonus_mean': subgoal_value_bonus.mean(),
            'phase1/loss_idm': loss_idm_unw,
            'phase1/first_step_l1': first_step_l1,
            'phase1/first_step_xy_l2': first_step_xy_l2,
            'phase1/roll_h_l1': zero,
            'phase1/eps_norm': zero,
            'phase1/mu_true_norm': jnp.linalg.norm(segment[:, 1, :], axis=-1).mean(),
            'phase1/mu_pred_norm': jnp.linalg.norm(path_pred[:, 1, :], axis=-1).mean(),
            'phase1/xN_minus_1_norm': jnp.linalg.norm(path_pred[:, 1, :], axis=-1).mean(),
            'phase1/bridge_step_mean': zero,
            'phase1/planner_type': jnp.asarray(planner_id, dtype=jnp.float32),
            # Forward-bridge specific.
            'forward_bridge/loss_path_interior': loss_path,
            'forward_bridge/loss_path_next': loss_next,
            'forward_bridge/path_mse': bridge_path_mse,
            'forward_bridge/next_mse': bridge_next_mse,
            'forward_bridge/final_mse': bridge_final_mse,
            'forward_bridge/endpoint_start_mse': bridge_endpoint_start_mse,
            'forward_bridge/endpoint_end_mse': bridge_endpoint_end_mse,
            'forward_bridge/dist_to_subgoal_step_1': dist_per_step[first_idx],
            'forward_bridge/dist_to_subgoal_step_mid': dist_per_step[mid_idx],
            'forward_bridge/dist_to_subgoal_step_last': dist_per_step[last_idx],
        }
        info['phase1/subgoal_pred_norm'] = jnp.linalg.norm(pred_sg_out, axis=-1).mean()
        info['phase1/subgoal_target_norm'] = jnp.linalg.norm(batch['high_actor_targets'], axis=-1).mean()
        info.update(subgoal_extra_info)
        info['bridge/bridge_type'] = jnp.asarray(_bridge_type_metric(self.config), dtype=jnp.float32)
        info['bridge/bridge_gamma'] = jnp.asarray(
            float(self.config.get('bridge_gamma', 1.0e7)), dtype=jnp.float32
        )
        return loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None, critic_value_params=None):
        """Path-supervised Phase1 loss; dispatches on ``planner_type``."""
        planner = _planner_type(self.config)
        if planner in ('forward_bridge', 'forward_bridge_residual'):
            return self._total_loss_forward_bridge(
                batch, grad_params, rng, critic_value_params, planner,
            )
        return self._total_loss_reverse_score(
            batch, grad_params, rng, critic_value_params,
        )


def _get_common_config():
    """Common defaults for GOUB dynamics training and rollout."""
    return ml_collections.ConfigDict(
        dict(
            agent_name='goub_dynamics',
            lr=3e-4,
            batch_size=1024,
            goub_N=25,
            goub_beta_min=0.1,
            goub_beta_max=20.0,
            goub_lambda=1.0,
            # UniDB-GOU soft-bridge controls.
            # bridge_type='goub' reproduces vanilla GOUB exactly.
            # bridge_type='unidb_gou' activates the finite-gamma soft bridge.
            # bridge_type='theta_linear' uses the self-consistent theta-linear
            # forward bridge with the same linear-beta theta scheduler.
            bridge_type='goub',
            bridge_gamma=1.0e7,
            eps_hidden_dims=(512, 512, 512),
            time_embed_dim=64,
            layer_norm=True,
            subgoal_loss_weight=1.0,
            subgoal_value_alpha=0.1,
            subgoal_value_hidden_dims=(512, 512, 512),
            subgoal_value_layer_norm=True,
            subgoal_hidden_dims=(512, 512, 512),
            # Distributional subgoal controls (default: legacy deterministic point).
            subgoal_distribution='deterministic',
            subgoal_log_std_min=-5.0,
            subgoal_log_std_max=1.0,
            subgoal_temperature=1.0,
            subgoal_use_mean_for_actor_goal=True,
            subgoal_nll_weight=1.0,
            subgoal_mse_weight=0.25,
            subgoal_var_reg_weight=1.0e-4,
            subgoal_fr_spi_weight=0.0,
            subgoal_fr_spi_tau=1.0,
            discount=0.99,
            subgoal_steps=25,
            # When False (default): PathHGCDataset overrides high_actor_targets with the
            # K-step horizon endpoint s_{t+K}, so the GOUB bridge / subgoal_net teacher is
            # always K steps ahead even if the episode goal s_{t_g} is closer than K.
            # When True: clip per-row to s_{min(t+K, t_g)} for both the bridge endpoint
            # (high_actor_targets) and the subgoal_net teacher, and pad trajectory_segment
            # tail with s_{t_g} for steps beyond t_g. This trains the bridge to "arrive
            # and stay" at close goals so subgoal predictions near the goal stay
            # in-distribution and reduces hovering near the goal.
            clip_path_to_goal=True,
            # Path-supervised planner switch.  Default 'reverse_score' keeps
            # the legacy learned-eps reverse chain bit-for-bit.  Alternatives:
            #   'forward_bridge'          : closed-form GOUB Prop. 3.1 forward
            #                               bridge mean (no learned path params).
            #   'forward_bridge_residual' : bridge mean + endpoint-preserving
            #                               learned residual (PathResidualNet).
            planner_type='reverse_score',
            forward_bridge_mode='mean',
            forward_bridge_noise_scale=0.0,
            forward_bridge_eps=1.0e-6,
            forward_bridge_train_epsilon=False,
            forward_bridge_use_path_loss=True,
            forward_bridge_use_rollout_loss=False,
            # Hidden dims for the optional PathResidualNet (forward_bridge_residual).
            residual_hidden_dims=(512, 512, 512),
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
            gc_negative=True,
            p_aug=0.0,
            frame_stack=ml_collections.config_dict.placeholder(int),
        )
    )


def get_dynamics_config():
    """Defaults for GOUB dynamics training."""
    c = _get_common_config()
    c.require_matching_horizon = True
    c.goub_loss_weight = 1.0
    c.path_loss_weight = 1.0
    c.rollout_loss_weight = 1.0
    c.rollout_horizon = 5
    c.path_eval_slice = [0, 1]
    c.idm_loss_weight = 1.0
    c.idm_hidden_dims = (512, 512, 512)
    return c
