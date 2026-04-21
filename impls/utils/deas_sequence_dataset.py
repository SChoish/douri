"""Offline dataset sampling for DEAS-style action-sequence critics.

Batches expose:
  * ``observations`` ``[B, D]`` — ``s_t``
  * ``actions`` ``[B, L, A]`` — ``a_{t : t+L-1}`` (inclusive of ``L`` steps; ``L = critic_action_sequence``)
  * ``next_observations`` ``[B, D]`` — ``s_{t + nstep_options * L}``
  * ``chunk_return`` ``[B]`` — two-discount n-step chunk return (see agent docstring)
  * ``bootstrap_discount`` ``[B]`` — ``gamma2 ** (nstep_options * L)`` (broadcast scalar, stored per batch row)
  * ``masks`` ``[B]`` — continuation mask for the Bellman bootstrap (0 if terminal inside the forward horizon)
  * ``step_rewards`` ``[B, nstep_options, L]`` — raw rewards for logging ``batch_rewards_mean``

Episode boundaries: start indices ``t`` are drawn only from positions with
``t + nstep_options * L <= terminal_index`` within the same trajectory.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from utils.datasets import Dataset, batched_random_crop


@dataclasses.dataclass
class DEASActionSeqDataset:
    """Action-sequence windows for detached distributional critic / value training."""

    dataset: Dataset
    config: Any
    preprocess_frame_stack: bool = True

    def __post_init__(self):
        self.size = self.dataset.size
        (self.terminal_locs,) = np.nonzero(self.dataset['terminals'] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == self.size - 1

        self.L = int(self.config['critic_action_sequence'])
        self.nopt = int(self.config.get('nstep_options', 1))
        if self.L < 1:
            raise ValueError('critic_action_sequence (L) must be >= 1.')
        if self.nopt < 1:
            raise ValueError('nstep_options must be >= 1.')

        self.gamma1 = float(self.config['gamma1'])
        self.gamma2 = float(self.config['gamma2'])

        horizon = self.nopt * self.L
        valids = np.zeros(self.size, dtype=np.float32)
        for start, end in zip(self.initial_locs, self.terminal_locs):
            last_start = int(end) - horizon
            if last_start >= int(start):
                valids[int(start) : last_start + 1] = 1.0
        (self.valid_starts,) = np.nonzero(valids > 0)
        if len(self.valid_starts) == 0:
            raise ValueError(
                'DEASActionSeqDataset: no valid starts for '
                f'nstep_options={self.nopt}, L={self.L}. Episodes may be too short.'
            )

        if self.config.get('frame_stack') is not None:
            assert 'next_observations' not in self.dataset
            if self.preprocess_frame_stack:
                stacked = self._stack_all_observations(np.arange(self.size))
                self.dataset = Dataset(self.dataset.copy(dict(observations=stacked)))

    def _stack_all_observations(self, idxs: np.ndarray) -> np.ndarray:
        fs = int(self.config['frame_stack'])
        initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
        rets = []
        for i in reversed(range(fs)):
            cur = np.maximum(idxs - i, initial_state_idxs)
            rets.append(jax.tree_util.tree_map(lambda arr: np.asarray(arr[cur]), self.dataset['observations']))
        return jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *rets)

    def get_observations(self, idxs: np.ndarray):
        if self.config.get('frame_stack') is None or self.preprocess_frame_stack:
            return jax.tree_util.tree_map(lambda arr: np.asarray(arr[idxs]), self.dataset['observations'])
        return self._stack_observations_runtime(idxs)

    def _stack_observations_runtime(self, idxs: np.ndarray) -> np.ndarray:
        fs = int(self.config['frame_stack'])
        initial_state_idxs = self.initial_locs[np.searchsorted(self.initial_locs, idxs, side='right') - 1]
        rets = []
        for i in reversed(range(fs)):
            cur = np.maximum(idxs - i, initial_state_idxs)
            rets.append(jax.tree_util.tree_map(lambda arr: np.asarray(arr[cur]), self.dataset['observations']))
        return jax.tree_util.tree_map(lambda *args: np.concatenate(args, axis=-1), *rets)

    def augment(self, batch: dict, keys: list[str]) -> None:
        p_aug = self.config.get('p_aug')
        if not p_aug or p_aug <= 0:
            return
        padding = 3
        batch_size = len(batch[keys[0]])
        crop_froms = np.random.randint(0, 2 * padding + 1, (batch_size, 2))
        crop_froms = np.concatenate([crop_froms, np.zeros((batch_size, 1), dtype=np.int64)], axis=1)
        for key in keys:
            batch[key] = jax.tree_util.tree_map(
                lambda arr: np.array(batched_random_crop(arr, crop_froms, padding))
                if len(arr.shape) == 4
                else arr,
                batch[key],
            )

    def sample(self, batch_size: int, idxs: np.ndarray | None = None, evaluation: bool = False) -> dict:
        if idxs is None:
            idxs = self.valid_starts[np.random.randint(len(self.valid_starts), size=batch_size)]
        idxs = np.asarray(idxs, dtype=np.int64)
        self._validate_starts(idxs)

        B = len(idxs)
        L, J = self.L, self.nopt
        horizon = J * L

        obs = np.asarray(self.get_observations(idxs), dtype=np.float32)
        next_idx = idxs + horizon
        next_obs = np.asarray(self.get_observations(next_idx), dtype=np.float32)

        # actions[b, j, k] = action at time idxs[b] + j*L + k
        rel = (np.arange(J) * L)[:, None] + np.arange(L)[None, :]  # [J, L]
        act_idx = idxs[:, None, None] + rel[None, :, :]  # [B, J, L]
        actions_jl = np.asarray(self.dataset['actions'][act_idx], dtype=np.float32)
        # Critic uses the first chunk only for Q(s, a_{t:t+L-1}); dataset actions for all j enter chunk_return.
        actions = actions_jl[:, 0, :, :]  # [B, L, A]

        rew_idx = act_idx  # rewards aligned with transitions at same time index as actions in compact datasets
        step_rewards = np.asarray(self.dataset['rewards'][rew_idx], dtype=np.float32)  # [B, J, L]

        gam1 = self.gamma1
        gam2 = self.gamma2
        k_pow = gam1 ** np.arange(L, dtype=np.float32)[None, None, :]  # [1,1,L]
        inner = np.sum(k_pow * step_rewards, axis=-1)  # [B, J]
        j_fac = gam2 ** (np.arange(J, dtype=np.float32) * L)[None, :]  # [1,J]
        chunk_return = np.sum(j_fac * inner, axis=-1)  # [B]

        bootstrap_discount = np.full(B, gam2 ** (J * L), dtype=np.float32)

        # Continuation: no terminal on transitions used for the forward horizon (exclusive of final next state).
        terms = np.asarray(self.dataset['terminals'], dtype=np.float32)
        term_windows = np.stack([terms[idxs + t] for t in range(horizon)], axis=-1)  # [B, horizon]
        masks = (1.0 - np.max(term_windows, axis=-1)).astype(np.float32)

        batch = {
            'observations': obs,
            'actions': actions,
            'next_observations': next_obs,
            'chunk_return': chunk_return.astype(np.float32),
            'bootstrap_discount': bootstrap_discount,
            'masks': masks,
            'step_rewards': step_rewards,
        }

        p_aug = self.config.get('p_aug')
        if p_aug is not None and float(p_aug) > 0 and not evaluation:
            if np.random.rand() < float(p_aug):
                self.augment(batch, ['observations', 'next_observations'])

        return batch

    def _validate_starts(self, idxs: np.ndarray) -> None:
        horizon = self.nopt * self.L
        finals = self.terminal_locs[np.searchsorted(self.terminal_locs, idxs)]
        bad = idxs + horizon > finals
        if np.any(bad):
            raise ValueError('DEASActionSeqDataset: sampled starts cross episode boundaries.')
