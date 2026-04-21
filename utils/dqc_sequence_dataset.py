"""Dataset sampler for DQC full/action-chunk training."""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import numpy as np

from utils.datasets import (
    Dataset,
    augment_batch_images,
    gather_stacked_observations,
)


@dataclasses.dataclass
class DQCActionSeqDataset:
    """Samples full/action chunks without crossing episode boundaries."""

    dataset: Dataset
    config: Any
    preprocess_frame_stack: bool = True

    def __post_init__(self):
        self.size = self.dataset.size
        (self.terminal_locs,) = np.nonzero(self.dataset['terminals'] > 0)
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])
        assert self.terminal_locs[-1] == self.size - 1

        self.full_chunk_horizon = int(self.config['full_chunk_horizon'])
        self.action_chunk_horizon = int(self.config['action_chunk_horizon'])
        if self.full_chunk_horizon < 1:
            raise ValueError('full_chunk_horizon must be >= 1.')
        if self.action_chunk_horizon < 1:
            raise ValueError('action_chunk_horizon must be >= 1.')
        if self.action_chunk_horizon > self.full_chunk_horizon:
            raise ValueError('action_chunk_horizon must be <= full_chunk_horizon.')

        valids = np.zeros(self.size, dtype=np.float32)
        for start, end in zip(self.initial_locs, self.terminal_locs):
            last_start = int(end) - self.full_chunk_horizon
            if last_start >= int(start):
                valids[int(start) : last_start + 1] = 1.0
        (self.valid_starts,) = np.nonzero(valids > 0)
        if len(self.valid_starts) == 0:
            raise ValueError(
                f'No valid starts for full_chunk_horizon={self.full_chunk_horizon} in dataset size={self.size}.'
            )

        # Cached metadata for hot-path sampling.
        self.final_state_for_idx = np.empty(self.size, dtype=np.int64)
        for start, end in zip(self.initial_locs, self.terminal_locs):
            self.final_state_for_idx[int(start) : int(end) + 1] = int(end)
        self.full_offsets = np.arange(self.full_chunk_horizon, dtype=np.int64)
        self.action_offsets = np.arange(self.action_chunk_horizon, dtype=np.int64)
        self.discount_pows = np.power(float(self.config['discount']), self.full_offsets.astype(np.float32))
        self.valids_template = np.ones((self.action_chunk_horizon,), dtype=np.float32)
        self.full_chunk_horizon_template = np.full((1,), self.full_chunk_horizon, dtype=np.float32)
        # valid_starts guarantee no terminal inside full chunk; masks are always ones.
        self.full_chunk_masks_template = np.ones((1,), dtype=np.float32)

        if self.config.get('frame_stack') is not None:
            assert 'next_observations' not in self.dataset
            if self.preprocess_frame_stack:
                stacked_observations = self.get_stacked_observations(np.arange(self.size))
                self.dataset = Dataset(self.dataset.copy(dict(observations=stacked_observations)))

    def get_observations(self, idxs):
        if self.config.get('frame_stack') is None or self.preprocess_frame_stack:
            return jax.tree_util.tree_map(lambda arr: np.asarray(arr[idxs]), self.dataset['observations'])
        return self.get_stacked_observations(idxs)

    def get_stacked_observations(self, idxs):
        return gather_stacked_observations(
            self.dataset['observations'],
            idxs,
            self.initial_locs,
            int(self.config['frame_stack']),
        )

    def _lookup_finals(self, idxs: np.ndarray) -> np.ndarray:
        return self.final_state_for_idx[idxs]

    def _build_full_chunk_indices(self, idxs: np.ndarray) -> np.ndarray:
        return idxs[:, None] + self.full_offsets[None, :]

    def sample_goals(self, idxs):
        batch_size = len(idxs)
        random_goal_idxs = self.dataset.get_random_idxs(batch_size)
        final_state_idxs = self._lookup_finals(idxs)
        if bool(self.config['value_geom_sample']):
            offsets = np.random.geometric(p=1 - float(self.config['discount']), size=batch_size)
            traj_goal_idxs = np.minimum(idxs + offsets, final_state_idxs)
        else:
            distances = np.random.rand(batch_size)
            traj_goal_idxs = np.round(
                (np.minimum(idxs + 1, final_state_idxs) * distances + final_state_idxs * (1 - distances))
            ).astype(int)
        p_cur = float(self.config['value_p_curgoal'])
        p_traj = float(self.config['value_p_trajgoal'])
        if p_cur == 1.0:
            return idxs
        goal_idxs = np.where(np.random.rand(batch_size) < p_traj / (1.0 - p_cur), traj_goal_idxs, random_goal_idxs)
        goal_idxs = np.where(np.random.rand(batch_size) < p_cur, idxs, goal_idxs)
        return goal_idxs

    def augment(self, batch: dict, keys: list[str]) -> None:
        p_aug = self.config.get('p_aug')
        if not p_aug or float(p_aug) <= 0:
            return
        augment_batch_images(batch, keys, padding=3)

    def _validate_starts(self, idxs: np.ndarray) -> None:
        finals = self._lookup_finals(idxs)
        bad = idxs + self.full_chunk_horizon > finals
        if np.any(bad):
            raise ValueError('DQCActionSeqDataset sampled starts crossing episode boundaries.')

    def sample(self, batch_size: int, idxs: np.ndarray | None = None, evaluation: bool = False) -> dict:
        if idxs is None:
            idxs = self.valid_starts[np.random.randint(len(self.valid_starts), size=batch_size)]
        idxs = np.asarray(idxs, dtype=np.int64)
        self._validate_starts(idxs)

        obs = np.asarray(self.get_observations(idxs), dtype=np.float32)
        actions_step = np.asarray(self.dataset['actions'][idxs], dtype=np.float32)
        next_obs = np.asarray(self.get_observations(idxs + 1), dtype=np.float32)

        full_idx = self._build_full_chunk_indices(idxs)
        full_chunk_actions_3d = np.asarray(self.dataset['actions'][full_idx], dtype=np.float32)  # [B, H_full, A]
        action_chunk_actions_3d = full_chunk_actions_3d[:, : self.action_chunk_horizon, :]
        full_chunk_actions = full_chunk_actions_3d.reshape(full_chunk_actions_3d.shape[0], -1)
        action_chunk_actions = action_chunk_actions_3d.reshape(action_chunk_actions_3d.shape[0], -1)

        value_goal_idxs = self.sample_goals(idxs)
        value_goals = np.asarray(self.get_observations(value_goal_idxs), dtype=np.float32)
        success_steps = (full_idx == value_goal_idxs[:, None]).astype(np.float32)
        reward_offset = 1.0 if bool(self.config.get('gc_negative', True)) else 0.0
        step_rewards = success_steps - reward_offset
        full_chunk_rewards = np.sum(step_rewards * self.discount_pows[None, :], axis=1).astype(np.float32)

        full_chunk_masks = np.repeat(self.full_chunk_masks_template, idxs.shape[0]).astype(np.float32)
        full_chunk_next_observations = np.asarray(
            self.get_observations(idxs + self.full_chunk_horizon), dtype=np.float32
        )
        full_chunk_horizon = np.repeat(self.full_chunk_horizon_template, idxs.shape[0]).astype(np.float32)
        valids = np.repeat(self.valids_template[None, :], idxs.shape[0], axis=0).astype(np.float32)

        batch = {
            'observations': obs,
            'actions': actions_step,
            'next_observations': next_obs,
            'value_goals': value_goals,
            'full_chunk_actions': full_chunk_actions,
            'action_chunk_actions': action_chunk_actions,
            'full_chunk_next_observations': full_chunk_next_observations,
            'full_chunk_rewards': full_chunk_rewards,
            'full_chunk_masks': full_chunk_masks,
            'full_chunk_horizon': full_chunk_horizon,
            'valids': valids,
        }

        if not evaluation:
            self.augment(batch, ['observations', 'next_observations', 'value_goals', 'full_chunk_next_observations'])
        return batch
