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
    make_goub_schedule,
    model_mean,
    posterior_mean,
    sample_from_reverse_mean,
)
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
        mu_true = posterior_mean(x_n, x_0, x_T, n, self.schedule)
        mu_pred, eps_pred = self._learned_reverse_mean(x_n, x_T, x_0, n, self.schedule, params=grad_params)

        g2_n = self.schedule['g2'][n - 1]
        weight = 1.0 / (2.0 * jnp.maximum(g2_n, 1e-12))
        loss_goub = (weight * jnp.abs(mu_true - mu_pred).sum(axis=-1)).mean()

        pred_sg = self.network.select('subgoal_net')(
            batch['observations'],
            batch['high_actor_goals'],
            params=grad_params,
        )
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
        return loss, info

    @jax.jit
    def update(self, batch, critic_value_params=None):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng, critic_value_params=critic_value_params)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=('num_steps',))
    def plan(self, current_state, desired_endpoint, *, num_steps: int | None = None):
        squeeze = current_state.ndim == 1
        if squeeze:
            current_state = current_state[None]
            desired_endpoint = desired_endpoint[None]

        x_T = current_state
        x_0_goal = desired_endpoint
        n_total = self.config['goub_N']
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
        traj = jnp.swapaxes(traj, 0, 1)
        next_step = traj_body[0]

        result = {'next_step': next_step, 'trajectory': traj}
        if squeeze:
            result = jax.tree_util.tree_map(lambda x: x[0], result)
        return result

    @partial(jax.jit, static_argnames=('noise_scale', 'num_steps'))
    def sample_plan(self, current_state, desired_endpoint, rng, noise_scale: float = 1.0, num_steps: int | None = None):
        squeeze = current_state.ndim == 1
        if squeeze:
            current_state = current_state[None]
            desired_endpoint = desired_endpoint[None]

        x_T = current_state
        x_0_goal = desired_endpoint
        n_total = self.config['goub_N']
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
        traj = jnp.swapaxes(traj, 0, 1)
        next_step = traj_body[0]

        result = {'next_step': next_step, 'trajectory': traj}
        if squeeze:
            result = jax.tree_util.tree_map(lambda x: x[0], result)
        return result

    def _sample_plan_trajectory(
        self, current_state, desired_endpoint, rng, noise_scale: float, num_steps: int | None = None
    ):
        x_T = current_state
        x_0_goal = desired_endpoint
        n_total = self.config['goub_N']
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

    @jax.jit
    def predict_subgoal(self, observations, high_actor_goals):
        squeeze = observations.ndim == 1
        if squeeze:
            observations = observations[None]
            high_actor_goals = high_actor_goals[None]
        out = self.network.select('subgoal_net')(observations, high_actor_goals)
        if squeeze:
            out = out[0]
        return out

    @jax.jit
    def infer_subgoal(self, observations, high_actor_goals):
        squeeze = observations.ndim == 1
        if squeeze:
            observations = observations[None]
            high_actor_goals = high_actor_goals[None]
        pred = self.network.select('subgoal_net')(observations, high_actor_goals)
        if squeeze:
            pred = pred[0]
        return pred

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
        obs = jnp.asarray(observations, dtype=jnp.float32)
        goals = jnp.asarray(high_actor_goals, dtype=jnp.float32)
        predicted_subgoals = self.infer_subgoal(obs, goals)

        if plan_candidates == 1:
            sampled = self.sample_plan(
                obs,
                predicted_subgoals,
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
                predicted_subgoals,
                sample_rng,
                num_candidates=plan_candidates,
                noise_scale=sample_noise_scale,
                include_mean=True,
                num_steps=proposal_horizon,
            )

        flat_trajectories = candidate_trajectories.reshape(
            -1, candidate_trajectories.shape[2], candidate_trajectories.shape[3]
        )
        candidate_actions = self._idm_actions_from_trajectories(flat_trajectories, proposal_horizon)
        candidate_actions = candidate_actions.reshape(
            candidate_trajectories.shape[0], candidate_trajectories.shape[1], proposal_horizon, -1
        )
        return predicted_subgoals, candidate_actions, new_rng

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
        )

        eps_net_def = GOUBEpsilonNet(
            hidden_dims=tuple(config['eps_hidden_dims']),
            state_dim=state_dim,
            time_embed_dim=config['time_embed_dim'],
            layer_norm=config['layer_norm'],
        )
        subgoal_def = SubgoalEstimatorNet(
            hidden_dims=tuple(config['subgoal_hidden_dims']),
            state_dim=state_dim,
            layer_norm=config['layer_norm'],
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

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None, critic_value_params=None):
        """GOUB mean-matching + path-aligned reverse + short rollout + subgoal MSE."""
        x_T = batch['observations']
        x_0 = batch['high_actor_targets']
        segment = jnp.asarray(batch['trajectory_segment'], dtype=jnp.float32)
        B = x_T.shape[0]
        N = int(self.config['goub_N'])
        K = int(segment.shape[1]) - 1
        # With PathHGCDataset, K should match N; avoid hard assert inside jit for speed.
        sg_w = float(self.config.get('subgoal_loss_weight', 1.0))
        w_g = float(self.config.get('goub_loss_weight', 1.0))
        w_p = float(self.config.get('path_loss_weight', 1.0))
        w_r = float(self.config.get('rollout_loss_weight', 0.25))

        rng1, rng2 = jax.random.split(rng)
        n = jax.random.randint(rng1, (B,), 1, N + 1)

        is_boundary = n == N
        n_safe = jnp.minimum(n, N - 1)

        # --- L_goub ---
        x_n_bridge = bridge_sample(x_0, x_T, n_safe, self.schedule, rng2)
        x_n = jnp.where(is_boundary[..., None], x_T, x_n_bridge)
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
        s = batch['observations']
        g_high = batch['high_actor_goals']
        target = batch['high_actor_targets']
        pred_sg = self.network.select('subgoal_net')(s, g_high, params=grad_params)
        subgoal_mse = jnp.mean((pred_sg - target) ** 2, axis=-1)
        subgoal_value, subgoal_value_bonus = self._subgoal_value_bonus(pred_sg, g_high, critic_value_params)
        loss_sub = jnp.mean(subgoal_mse) - jnp.mean(subgoal_value_bonus)
        pred_sg_out = pred_sg

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
        pev = self.config.get('path_eval_slice')
        if pev is None:
            pev_t = (0, 1)
        else:
            pev_t = tuple(int(x) for x in pev)
        idx_xy = jnp.asarray(pev_t, dtype=jnp.int32)
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
        }
        info['phase1/subgoal_pred_norm'] = jnp.linalg.norm(pred_sg_out, axis=-1).mean()
        info['phase1/subgoal_target_norm'] = jnp.linalg.norm(batch['high_actor_targets'], axis=-1).mean()

        return loss, info


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
            eps_hidden_dims=(512, 512, 512),
            time_embed_dim=64,
            layer_norm=True,
            subgoal_loss_weight=1.0,
            subgoal_value_alpha=0.1,
            subgoal_value_hidden_dims=(512, 512, 512),
            subgoal_value_layer_norm=True,
            subgoal_hidden_dims=(512, 512, 512),
            discount=0.99,
            subgoal_steps=25,
            idm_loss_weight=1.0,
            idm_hidden_dims=(512, 512, 512),
            value_p_curgoal=0.2,
            value_p_trajgoal=0.5,
            value_p_randomgoal=0.3,
            value_geom_sample=True,
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
    c.rollout_loss_weight = 0.25
    c.rollout_horizon = 5
    c.path_eval_slice = [0, 1]
    c.idm_loss_weight = 1.0
    c.idm_hidden_dims = (512, 512, 512)
    return c
