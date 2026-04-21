"""Inverse dynamics MLP training (no absl flags — safe to import from any entrypoint)."""

from __future__ import annotations

import json
import logging
import math
import os
import pickle
import time
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
from flax import linen as nn
from flax.training import train_state

from utils.datasets import Dataset
from utils.env_utils import make_env_and_datasets
from utils.networks import MLP


class InverseDynamicsMLP(nn.Module):
    """Predict a_t from concatenated (s_t, s_{t+1})."""

    obs_dim: int
    action_dim: int
    hidden_dims: tuple[int, ...]

    @nn.compact
    def __call__(self, obs: jnp.ndarray, next_obs: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([obs, next_obs], axis=-1)
        return MLP(
            hidden_dims=(*self.hidden_dims, self.action_dim),
            activate_final=False,
            layer_norm=True,
        )(x)


def _steps_per_epoch(dataset_size: int, batch_size: int) -> int:
    return max(1, math.ceil(dataset_size / batch_size))


def _batch_to_jax(batch: dict) -> dict:
    return {
        'observations': jnp.asarray(batch['observations'], dtype=jnp.float32),
        'next_observations': jnp.asarray(batch['next_observations'], dtype=jnp.float32),
        'actions': jnp.asarray(batch['actions'], dtype=jnp.float32),
    }


def parse_hidden_dims(hidden_dims: str | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(hidden_dims, str):
        return tuple(int(x.strip()) for x in hidden_dims.split(',') if x.strip())
    return hidden_dims


@dataclass
class JointIdmBundle:
    """Mutable container for GOUB+IDM joint training (one IDM epoch = ``steps_per_epoch`` Adam steps)."""

    model: InverseDynamicsMLP
    state: train_state.TrainState
    train_ds: Dataset
    val_ds: Dataset | None
    steps_per_epoch: int
    obs_dim: int
    action_dim: int
    hidden_dims: tuple[int, ...]
    env_name: str
    train_step: Any
    eval_mse: Any


def init_joint_idm(
    *,
    train_ds: Dataset,
    val_ds: Dataset | None,
    env_name: str,
    seed: int,
    batch_size: int,
    lr: float,
    hidden_dims: str | tuple[int, ...],
) -> JointIdmBundle:
    """Build IDM + optimizer; ``steps_per_epoch`` = full pass over transitions at ``batch_size``."""
    hidden_dims_t = parse_hidden_dims(hidden_dims)
    ex = train_ds.sample(1)
    obs_dim = int(ex['observations'].shape[-1])
    action_dim = int(ex['actions'].shape[-1])

    model = InverseDynamicsMLP(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims_t,
    )
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    dummy_o = jnp.zeros((1, obs_dim), jnp.float32)
    dummy_n = jnp.zeros((1, obs_dim), jnp.float32)
    params = model.init(init_rng, dummy_o, dummy_n)['params']
    tx = optax.adam(lr)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    steps_pe = _steps_per_epoch(int(train_ds.size), batch_size)

    @jax.jit
    def train_step(st: train_state.TrainState, batch: dict):
        def loss_fn(p):
            pred = model.apply({'params': p}, batch['observations'], batch['next_observations'])
            return jnp.mean((pred - batch['actions']) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(st.params)
        return st.apply_gradients(grads=grads), loss

    @jax.jit
    def eval_mse(st: train_state.TrainState, batch: dict):
        pred = model.apply({'params': st.params}, batch['observations'], batch['next_observations'])
        return jnp.mean((pred - batch['actions']) ** 2)

    return JointIdmBundle(
        model=model,
        state=state,
        train_ds=train_ds,
        val_ds=val_ds,
        steps_per_epoch=steps_pe,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims_t,
        env_name=env_name,
        train_step=train_step,
        eval_mse=eval_mse,
    )


def joint_idm_train_one_epoch(bundle: JointIdmBundle, batch_size: int) -> float:
    """One IDM epoch (dataset-sized pass); updates ``bundle.state`` in place."""
    losses: list[float] = []
    state = bundle.state
    for _ in range(bundle.steps_per_epoch):
        batch = _batch_to_jax(bundle.train_ds.sample(batch_size))
        state, loss = bundle.train_step(state, batch)
        losses.append(float(loss))
    bundle.state = state
    return float(np.mean(losses))


def joint_idm_eval_val(bundle: JointIdmBundle, batch_size: int) -> float:
    if bundle.val_ds is None:
        return float('nan')
    vb = _batch_to_jax(bundle.val_ds.sample(min(batch_size * 4, bundle.val_ds.size)))
    return float(bundle.eval_mse(bundle.state, vb))


def save_joint_idm_checkpoint(bundle: JointIdmBundle, ckpt_path: str, epoch: int) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(ckpt_path)), exist_ok=True)
    with open(ckpt_path, 'wb') as f:
        pickle.dump(
            {
                'params': jax.device_get(bundle.state.params),
                'obs_dim': bundle.obs_dim,
                'action_dim': bundle.action_dim,
                'hidden_dims': bundle.hidden_dims,
                'env_name': bundle.env_name,
                'epoch': epoch,
            },
            f,
        )


def train_inverse_dynamics_run(
    *,
    env_name: str,
    seed: int,
    run_dir: str,
    train_epochs: int = 1000,
    batch_size: int = 512,
    lr: float = 3e-4,
    hidden_dims: str | tuple[int, ...] = (512, 512, 512),
    log_every_n_epochs: int = 10,
    save_every_n_epochs: int = 100,
    use_tqdm: bool = True,
    parent_run_dir: str | None = None,
    log: logging.Logger | None = None,
) -> str:
    """Train IDM; write ``run_dir/{flags.json,train.csv,checkpoints/}``.

    ``parent_run_dir``: if set, recorded in ``flags.json`` (e.g. GOUB phase-1 ``run_dir``).

    Returns:
        ``run_dir`` (absolute path).
    """
    run_dir = os.path.abspath(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    def _emit(msg: str) -> None:
        if log is not None:
            log.info('%s', msg)
        else:
            print(msg)

    hidden_dims_t = parse_hidden_dims(hidden_dims)

    flag_dict = {
        'env_name': env_name,
        'seed': seed,
        'train_epochs': train_epochs,
        'batch_size': batch_size,
        'lr': lr,
        'hidden_dims': ','.join(str(h) for h in hidden_dims_t),
        'log_every_n_epochs': log_every_n_epochs,
        'save_every_n_epochs': save_every_n_epochs,
        'use_tqdm': use_tqdm,
        'parent_run_dir': parent_run_dir,
    }
    with open(os.path.join(run_dir, 'flags.json'), 'w', encoding='utf-8') as f:
        json.dump(flag_dict, f, default=str, indent=2)

    rng = jax.random.PRNGKey(seed)
    np.random.seed(seed)

    _env, train_raw, val_raw = make_env_and_datasets(env_name, frame_stack=None)
    train_ds = train_raw if isinstance(train_raw, Dataset) else Dataset.create(**train_raw)
    val_ds = None if val_raw is None else (val_raw if isinstance(val_raw, Dataset) else Dataset.create(**val_raw))

    ex = train_ds.sample(1)
    obs_dim = int(ex['observations'].shape[-1])
    action_dim = int(ex['actions'].shape[-1])
    ds_size = int(train_ds.size)

    model = InverseDynamicsMLP(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims_t,
    )
    rng, init_rng = jax.random.split(rng)
    dummy_o = jnp.zeros((1, obs_dim), jnp.float32)
    dummy_n = jnp.zeros((1, obs_dim), jnp.float32)
    params = model.init(init_rng, dummy_o, dummy_n)['params']
    tx = optax.adam(lr)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    steps_pe = _steps_per_epoch(ds_size, batch_size)
    total_steps = train_epochs * steps_pe

    @jax.jit
    def train_step(state: train_state.TrainState, batch: dict):
        def loss_fn(p):
            pred = model.apply({'params': p}, batch['observations'], batch['next_observations'])
            return jnp.mean((pred - batch['actions']) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    @jax.jit
    def eval_mse(state: train_state.TrainState, batch: dict):
        pred = model.apply({'params': state.params}, batch['observations'], batch['next_observations'])
        return jnp.mean((pred - batch['actions']) ** 2)

    train_csv = os.path.join(run_dir, 'train.csv')
    with open(train_csv, 'w', encoding='utf-8') as f:
        f.write('epoch,train_mse,val_mse,elapsed_sec\n')

    t0 = time.time()
    epoch_iter = range(1, train_epochs + 1)
    if use_tqdm:
        epoch_iter = tqdm.tqdm(epoch_iter, desc='idm_epochs')

    for epoch in epoch_iter:
        losses = []
        for _ in range(steps_pe):
            batch = train_ds.sample(batch_size)
            batch = _batch_to_jax(batch)
            state, loss = train_step(state, batch)
            losses.append(float(loss))

        train_mse = float(np.mean(losses))

        val_mse = float('nan')
        do_log = log_every_n_epochs > 0 and (
            epoch % log_every_n_epochs == 0 or epoch == 1 or epoch == train_epochs
        )
        if val_ds is not None and do_log:
            vb = val_ds.sample(min(batch_size * 4, val_ds.size))
            vb = _batch_to_jax(vb)
            val_mse = float(eval_mse(state, vb))

        elapsed = time.time() - t0
        if do_log:
            with open(train_csv, 'a', encoding='utf-8') as f:
                f.write(f'{epoch},{train_mse:.8f},{val_mse:.8f},{elapsed:.2f}\n')
            _emit(
                f'[idm] epoch {epoch}/{train_epochs}  train_mse={train_mse:.6f}  val_mse={val_mse}  '
                f'steps/epoch={steps_pe}  elapsed={elapsed:.1f}s'
            )

        if save_every_n_epochs > 0 and (epoch % save_every_n_epochs == 0 or epoch == train_epochs):
            path = os.path.join(ckpt_dir, f'params_{epoch}.pkl')
            with open(path, 'wb') as f:
                pickle.dump(
                    {
                        'params': jax.device_get(state.params),
                        'obs_dim': obs_dim,
                        'action_dim': action_dim,
                        'hidden_dims': hidden_dims_t,
                        'env_name': env_name,
                        'epoch': epoch,
                    },
                    f,
                )
            _emit(f'[idm] Saved {path}')

    _emit(f'[idm] Done. run_dir={run_dir}  total_gradient_steps={total_steps}')
    return run_dir
