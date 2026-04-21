"""GOUB-inspired Phase-1 agent: endpoint-conditioned next-step planner.

Design notes
------------
This is a GOUB-*inspired* bridge diffusion planner, not a paper-exact GOUB
reproduction.  See ``utils/goub.py`` for the specific approximations used.

**Boundary handling (n = N).**  The standard model_mean formula is singular
at n = N because bridge_var[N] = 0.  We handle this with a *learned residual
parameterisation*:

* For n in {1, ..., N-1}: ``mu_pred = model_mean(x_n, x_T, eps, n)``
  (standard GOUB-inspired epsilon-to-mean conversion).
* For n = N (boundary): ``mu_pred = x_T + eps``  where eps is the raw
  network output.  At this step x_n = x_T always, so the network learns
  the delta from x_T to the analytic posterior target.

Both cases are trained jointly with the same L1 mean-matching loss weighted
by ``1 / (2 g_n^2)``.

**What ``next_step`` means.**  ``plan()`` runs the full deterministic reverse
chain from x_N = x_T down to x_0.  ``sample_plan()`` runs the same chain but
adds Gaussian noise at each step (N(mu, g_n^2 I) with optional temperature).
Every step—including the first one at n = N—uses the learned epsilon network.
``next_step`` is x_{N-1}, the first reverse output, and serves as the next-step
planner target for downstream RL / inverse dynamics.

**Subgoal estimator.**  A separate MLP ``subgoal_net`` maps
``(observations, high_actor_goals)`` to a predicted state vector supervised
with MSE against ``high_actor_targets`` (same teacher used as GOUB
``x_0``).  At deployment, ``predict_subgoal(s, g)`` can supply the bridge
endpoint for ``plan(s, predict_subgoal(s, g))`` when the dataset target is
unavailable.
"""

from functools import partial
from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.goub import (
    bridge_sample,
    make_goub_schedule,
    model_mean,
    posterior_mean,
    sample_from_reverse_mean,
)
from utils.networks import MLP


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for the diffusion timestep."""
    dim: int

    @nn.compact
    def __call__(self, t):
        half = self.dim // 2
        freq = jnp.exp(-jnp.log(10000.0) * jnp.arange(half, dtype=jnp.float32) / half)
        args = t[..., None].astype(jnp.float32) * freq
        return jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)


class GOUBEpsilonNet(nn.Module):
    """Epsilon prediction network for the GOUB-inspired bridge.

    Concatenates ``[x_n, x_T, x_0, emb(n)]`` and maps through an MLP to
    produce an output of dimension ``state_dim``.

    At n < N the output is interpreted as a noise prediction that is converted
    to a mean via ``model_mean``.  At n = N the output is interpreted as a
    residual from x_T (see agent docstring).
    """
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
    """Predicts ``high_actor_targets`` from ``(s, high_actor_goals)`` in state space."""

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


class GOUBPhase1Agent(flax.struct.PyTreeNode):
    """GOUB-inspired Phase-1 agent (flax PyTreeNode)."""

    rng: Any
    network: Any
    schedule: Any
    config: Any = nonpytree_field()

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _learned_reverse_mean(self, x_n, x_T, x_0, n, schedule, params=None):
        """One reverse-step mean that is safe for all n in {1, ..., N}.

        * n < N  →  model_mean (GOUB-inspired eps-to-mean)
        * n = N  →  x_T + eps  (learned boundary residual)

        Returns:
            mu_theta_{n-1}, eps
        """
        N = self.config['goub_N']
        n_safe = jnp.minimum(n, N - 1)
        is_boundary = (n == N)

        eps = self.network.select('eps_net')(
            x_n, x_T, x_0, n.astype(jnp.float32), params=params,
        )

        mu_inner = model_mean(x_n, x_T, eps, n_safe, schedule)
        mu_boundary = x_T + eps
        mu = jnp.where(is_boundary[..., None], mu_boundary, mu_inner)
        return mu, eps

    def _reverse_step(
        self,
        x_n,
        x_T,
        x_0,
        n,
        rng,
        stochastic,
        noise_scale,
        params=None,
    ):
        """Apply one reverse step: mean only, or mean + g_n-scaled Gaussian noise."""
        mu, eps = self._learned_reverse_mean(x_n, x_T, x_0, n, self.schedule, params=params)
        ns = jnp.asarray(noise_scale, dtype=jnp.float32)

        def take_sample(_):
            return sample_from_reverse_mean(mu, n, self.schedule, rng, noise_scale=ns)

        def take_mean(_):
            return mu

        x_new = jax.lax.cond(jnp.asarray(stochastic, dtype=jnp.bool_), take_sample, take_mean, operand=None)
        return x_new, eps

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Phase-1 loss: GOUB L1 mean-matching + optional subgoal MSE.

        GOUB: epsilon network over n in {1, ..., N}.  Subgoal estimator:
        MSE between ``subgoal_net(s, high_actor_goals)`` and
        ``high_actor_targets`` (weighted by ``subgoal_loss_weight``).
        """
        x_T = batch['observations']
        x_0 = batch['high_actor_targets']
        B = x_T.shape[0]
        N = self.config['goub_N']
        train_sg = bool(self.config.get('train_subgoal_estimator', True))
        sg_w = float(self.config.get('subgoal_loss_weight', 1.0))

        rng1, rng2 = jax.random.split(rng)

        # Sample n uniformly from {1, ..., N}  (boundary included)
        n = jax.random.randint(rng1, (B,), 1, N + 1)

        is_boundary = (n == N)
        n_safe = jnp.minimum(n, N - 1)

        # x_n: bridge sample for n < N, pinned x_T for n = N
        x_n_bridge = bridge_sample(x_0, x_T, n_safe, self.schedule, rng2)
        x_n = jnp.where(is_boundary[..., None], x_T, x_n_bridge)

        # Analytic target (well-defined for all n in {1, ..., N})
        mu_true = posterior_mean(x_n, x_0, x_T, n, self.schedule)

        # Learned prediction
        mu_pred, eps_pred = self._learned_reverse_mean(
            x_n, x_T, x_0, n, self.schedule, params=grad_params,
        )

        # L1 mean-matching loss weighted by 1 / (2 g_n^2)
        g2_n = self.schedule['g2'][n - 1]  # (B,)
        weight = 1.0 / (2.0 * jnp.maximum(g2_n, 1e-12))
        loss_goub = (weight * jnp.abs(mu_true - mu_pred).sum(axis=-1)).mean()

        if train_sg and sg_w > 0.0:
            s = batch['observations']
            g_high = batch['high_actor_goals']
            target = batch['high_actor_targets']
            pred_sg = self.network.select('subgoal_net')(s, g_high, params=grad_params)
            loss_sub = jnp.mean((pred_sg - target) ** 2)
            loss = loss_goub + sg_w * loss_sub
        else:
            loss_sub = jnp.array(0.0)
            loss = loss_goub
            pred_sg = jnp.zeros_like(x_0)

        # x_{N-1} norm (learned boundary step; same bridge endpoint x_0 as training batch)
        n_N = jnp.full((B,), N, dtype=jnp.int32)
        xNm1, _ = self._learned_reverse_mean(
            x_T, x_T, x_0, n_N, self.schedule, params=grad_params,
        )
        xNm1_norm = jnp.linalg.norm(xNm1, axis=-1).mean()

        # -- Logging (fixed keys for CSV / W&B) --
        # phase1/loss = total objective; phase1/loss_goub / phase1/loss_subgoal are decomposed.
        info = {
            'phase1/loss': loss,
            'phase1/loss_goub': loss_goub,
            'phase1/loss_subgoal': loss_sub,
            'phase1/eps_norm': jnp.linalg.norm(eps_pred, axis=-1).mean(),
            'phase1/mu_true_norm': jnp.linalg.norm(mu_true, axis=-1).mean(),
            'phase1/mu_pred_norm': jnp.linalg.norm(mu_pred, axis=-1).mean(),
            'phase1/xN_minus_1_norm': xNm1_norm,
            'phase1/bridge_step_mean': n.astype(jnp.float32).mean(),
        }
        if train_sg and sg_w > 0.0:
            info['phase1/subgoal_pred_norm'] = jnp.linalg.norm(pred_sg, axis=-1).mean()
        else:
            info['phase1/subgoal_pred_norm'] = jnp.array(0.0)
        info['phase1/subgoal_target_norm'] = jnp.linalg.norm(batch['high_actor_targets'], axis=-1).mean()

        return loss, info

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    @jax.jit
    def update(self, batch):
        """Single gradient step; returns ``(new_agent, info_dict)``."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        return self.replace(network=new_network, rng=new_rng), info

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @jax.jit
    def plan(self, current_state, desired_endpoint):
        """Deterministic learned reverse pass.

        **Every step—including the first at n = N—uses the learned network.**
        ``next_step`` is the first learned reverse output x_{N-1}.

        Args:
            current_state: s_k, shape ``(B, D)`` or ``(D,)``.
            desired_endpoint: tilde_s_k, shape ``(B, D)`` or ``(D,)``.

        Returns:
            dict with ``next_step`` (x_{N-1}) and ``trajectory`` (x_N … x_0).
        """
        squeeze = current_state.ndim == 1
        if squeeze:
            current_state = current_state[None]
            desired_endpoint = desired_endpoint[None]

        x_T = current_state
        x_0_goal = desired_endpoint
        N = self.config['goub_N']
        B = x_T.shape[0]

        def scan_body(x, step_n):
            n = jnp.full((B,), step_n, dtype=jnp.int32)
            x_new, _ = self._learned_reverse_mean(x, x_T, x_0_goal, n, self.schedule)
            return x_new, x_new

        steps = jnp.arange(N, 0, -1)  # N, N-1, …, 1
        _, traj_body = jax.lax.scan(scan_body, x_T, steps)
        # traj_body: (N, B, D) → x_{N-1}, x_{N-2}, …, x_0

        # Assemble trajectory: [x_N, x_{N-1}, …, x_0]
        traj = jnp.concatenate([x_T[None], traj_body], axis=0)  # (N+1, B, D)
        traj = jnp.swapaxes(traj, 0, 1)  # (B, N+1, D)

        next_step = traj_body[0]  # x_{N-1}, first *learned* reverse output

        result = {'next_step': next_step, 'trajectory': traj}
        if squeeze:
            result = jax.tree_util.tree_map(lambda x: x[0], result)
        return result

    @jax.jit
    def sample_plan(self, current_state, desired_endpoint, rng, noise_scale: float = 1.0):
        """Stochastic learned reverse pass.

        Samples each reverse step from N(mu_theta_{n-1}, noise_scale^2 * g_n^2 I).

        Args:
            current_state: s_k, shape (B, D) or (D,).
            desired_endpoint: tilde_s_k, shape (B, D) or (D,).
            rng: JAX PRNG key.
            noise_scale: Temperature on the per-step Gaussian (scalar).

        Returns:
            dict with ``next_step`` (sampled x_{N-1}) and ``trajectory`` (sampled [x_N, …, x_0]).
        """
        squeeze = current_state.ndim == 1
        if squeeze:
            current_state = current_state[None]
            desired_endpoint = desired_endpoint[None]

        x_T = current_state
        x_0_goal = desired_endpoint
        N = self.config['goub_N']
        B = x_T.shape[0]

        step_rngs = jax.random.split(rng, N)
        noise_scale = jnp.asarray(noise_scale, dtype=jnp.float32)

        def scan_body(x, inputs):
            step_n, step_rng = inputs
            n = jnp.full((B,), step_n, dtype=jnp.int32)
            x_new, _ = self._reverse_step(
                x,
                x_T,
                x_0_goal,
                n,
                step_rng,
                True,
                noise_scale,
                params=None,
            )
            return x_new, x_new

        steps = jnp.arange(N, 0, -1)  # N, N-1, …, 1
        _, traj_body = jax.lax.scan(scan_body, x_T, (steps, step_rngs))

        traj = jnp.concatenate([x_T[None], traj_body], axis=0)
        traj = jnp.swapaxes(traj, 0, 1)

        next_step = traj_body[0]

        result = {'next_step': next_step, 'trajectory': traj}
        if squeeze:
            result = jax.tree_util.tree_map(lambda x: x[0], result)
        return result

    @jax.jit
    def predict_subgoal(self, observations, high_actor_goals):
        """Predict subgoal state ``tilde{s}`` from current state and high-level goal.

        Shapes: ``observations`` and ``high_actor_goals`` are ``(B, D)`` or ``(D,)``
        (batched vs single handled like ``plan``).
        """
        squeeze = observations.ndim == 1
        if squeeze:
            observations = observations[None]
            high_actor_goals = high_actor_goals[None]
        out = self.network.select('subgoal_net')(observations, high_actor_goals)
        if squeeze:
            out = out[0]
        return out

    @jax.jit
    def plan_from_high_goal(self, current_state, high_actor_goals):
        """Plan bridge using the *estimated* subgoal endpoint.

        ``desired_endpoint = predict_subgoal(s, g)``, then ``plan(s, endpoint)``.
        """
        endpoint = self.predict_subgoal(current_state, high_actor_goals)
        return self.plan(current_state, endpoint)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, seed, ex_observations, config):
        """Create a new GOUB-inspired Phase-1 agent.

        Args:
            seed: Random seed.
            ex_observations: Example observation batch, shape ``(B, D)``.
            config: ``ml_collections.ConfigDict``.
        """
        assert config['goub_N'] >= 2, 'GOUB requires N >= 2 diffusion steps.'

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        state_dim = ex_observations.shape[-1]

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

        B = ex_observations.shape[0]
        dummy_x = ex_observations
        dummy_g = ex_observations
        dummy_n = jnp.ones((B,), dtype=jnp.float32)

        network_info = dict(
            eps_net=(eps_net_def, (dummy_x, dummy_x, dummy_x, dummy_n)),
            subgoal_net=(subgoal_def, (dummy_x, dummy_g)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        return cls(
            rng=rng,
            network=network,
            schedule=schedule,
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent identity.
            agent_name='goub_phase1',
            lr=3e-4,
            batch_size=1024,
            # GOUB-inspired schedule (match subgoal horizon by default).
            goub_N=25,
            goub_beta_min=0.1,
            goub_beta_max=20.0,
            goub_lambda=1.0,
            # Epsilon network.
            eps_hidden_dims=(512, 512, 512),
            time_embed_dim=64,
            layer_norm=True,
            # Subgoal estimator: (s, high_actor_goals) -> high_actor_targets (MSE).
            train_subgoal_estimator=True,
            subgoal_loss_weight=1.0,
            subgoal_hidden_dims=(512, 512, 512),
            # Dataset (reuse HGCDataset).
            dataset_class='HGCDataset',
            discount=0.99,
            subgoal_steps=25,
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
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
    return config
