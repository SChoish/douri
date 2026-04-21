"""GOUB-inspired Phase-1 with **offline trajectory path supervision**.

This is **not** paper-exact GOUB. It keeps the existing GOUB-inspired bridge mean-matching
objective and adds:

* **Step-aligned path loss** — match learned reverse means to real ``s_{t+1..K}`` along the
  same sampled segment when using the same diffusion index ``n``.
* **Short rollout consistency** — deterministic recursive reverse steps vs real prefixes.

See :mod:`agents.goub_phase1` for the endpoint-only baseline. Training entrypoint:
``main_goub_phase1_path.py`` with :class:`utils.datasets.PathHGCDataset`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from agents.goub_phase1 import GOUBPhase1Agent, get_config as goub_phase1_base_config
from utils.goub import bridge_sample, posterior_mean


class GOUBPhase1PathAgent(GOUBPhase1Agent):
    """Same inference API as :class:`GOUBPhase1Agent`; training loss adds path + rollout terms."""

    def _select_loss_slice(self, values, targets, cfg_key: str):
        """Apply an optional static config slice to ``values`` and ``targets``."""
        sel = self.config.get(cfg_key)
        if sel is None or (isinstance(sel, (list, tuple)) and len(sel) == 0):
            return values, targets
        idx = jnp.asarray(tuple(int(x) for x in sel), dtype=jnp.int32)
        return values[:, idx], targets[:, idx]

    @classmethod
    def create(cls, seed, ex_observations, config, ex_actions=None):
        if bool(config.get('require_matching_horizon', True)):
            gn = int(config['goub_N'])
            sk = int(config['subgoal_steps'])
            if gn != sk:
                raise ValueError(
                    f'GOUBPhase1Path: require_matching_horizon expects goub_N ({gn}) == '
                    f'subgoal_steps ({sk}). Disable with require_matching_horizon: false '
                    'only if you accept misaligned indices.'
                )
        return super().create(seed, ex_observations, config, ex_actions=ex_actions)

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """GOUB mean-matching + path-aligned reverse + short rollout + subgoal MSE."""
        x_T = batch['observations']
        x_0 = batch['high_actor_targets']
        segment = jnp.asarray(batch['trajectory_segment'], dtype=jnp.float32)
        B = x_T.shape[0]
        N = int(self.config['goub_N'])
        K = int(segment.shape[1]) - 1
        # With PathHGCDataset, K should match N; avoid hard assert inside jit for speed.
        train_sg = bool(self.config.get('train_subgoal_estimator', True))
        sg_w = float(self.config.get('subgoal_loss_weight', 1.0))
        w_g = float(self.config.get('goub_loss_weight', 1.0))
        w_p = float(self.config.get('path_loss_weight', 1.0))
        w_r = float(self.config.get('rollout_loss_weight', 0.25))

        rng1, rng2 = jax.random.split(rng)
        n = jax.random.randint(rng1, (B,), 1, N + 1)

        is_boundary = n == N
        n_safe = jnp.minimum(n, N - 1)

        # --- L_goub (same as baseline) ---
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
        mu_pred_path, x_prev_real = self._select_loss_slice(mu_pred_path, x_prev_real, 'path_loss_slice')
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
            mu_r_eval, tgt_eval = self._select_loss_slice(mu_r, tgt, 'rollout_loss_slice')
            err = jnp.abs(mu_r_eval - tgt_eval).sum(axis=-1)
            return mu_r, err

        _, errs = jax.lax.scan(roll_body, segment[:, 0, :], hs)
        loss_roll = jnp.mean(errs)

        # --- L_subgoal ---
        if train_sg and sg_w > 0.0:
            s = batch['observations']
            g_high = batch['high_actor_goals']
            target = batch['high_actor_targets']
            pred_sg = self.network.select('subgoal_net')(s, g_high, params=grad_params)
            loss_sub = jnp.mean((pred_sg - target) ** 2)
            pred_sg_out = pred_sg
        else:
            loss_sub = jnp.array(0.0)
            pred_sg_out = jnp.zeros_like(x_0)

        loss = w_g * loss_goub + w_p * loss_path + w_r * loss_roll
        if train_sg and sg_w > 0.0:
            loss = loss + sg_w * loss_sub

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
        if train_sg and sg_w > 0.0:
            info['phase1/subgoal_pred_norm'] = jnp.linalg.norm(pred_sg_out, axis=-1).mean()
        else:
            info['phase1/subgoal_pred_norm'] = jnp.array(0.0)
        info['phase1/subgoal_target_norm'] = jnp.linalg.norm(batch['high_actor_targets'], axis=-1).mean()

        return loss, info


def get_config():
    """Defaults for path-supervised Phase-1 (extends baseline GOUB config)."""
    c = goub_phase1_base_config()
    c.agent_name = 'goub_phase1_path'
    c.dataset_class = 'PathHGCDataset'
    c.path_dynamics = True
    c.require_matching_horizon = True
    c.goub_loss_weight = 1.0
    c.path_loss_weight = 1.0
    c.rollout_loss_weight = 0.25
    c.rollout_horizon = 5
    c.path_loss_slice = None  # optional list of observation indices; None = full state
    c.rollout_loss_slice = None  # optional list of observation indices; None = full state
    c.path_eval_slice = [0, 1]
    c.idm_loss_weight = 1.0
    c.idm_hidden_dims = (512, 512, 512)
    return c
