"""Smoke tests for the ``critic_type`` modes (DQC, chunk-IQL, and TRL).

Run::

    PYTHONPATH=. python -m pytest tests/test_critic_modes.py -v
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import jax
import jax.numpy as jnp

from agents.critic import CriticAgent, get_config as get_critic_config, validate_config
from utils.critic_sequence_dataset import CriticSequenceDataset
from utils.datasets import Dataset


STATE_DIM = 6
ACTION_DIM = 3
BATCH = 12


def _make_dummy_dataset(num_episodes: int = 4, ep_len: int = 50):
    n = num_episodes * ep_len
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((n, STATE_DIM)).astype(np.float32)
    actions = rng.uniform(-1.0, 1.0, size=(n, ACTION_DIM)).astype(np.float32)
    terminals = np.zeros((n,), dtype=np.float32)
    for k in range(num_episodes):
        terminals[(k + 1) * ep_len - 1] = 1.0
    return Dataset.create(observations=obs, actions=actions, terminals=terminals)


def _make_critic_config(critic_type: str):
    cfg = get_critic_config()
    cfg.action_chunk_horizon = 4
    cfg.full_chunk_horizon = 8
    cfg.value_hidden_dims = (32, 32)
    cfg.action_dim = ACTION_DIM
    cfg.batch_size = BATCH
    cfg.frame_stack = None
    cfg.critic_type = critic_type
    if critic_type in ('iql', 'trl', 'state_transitive', 'direct_chunk_trl'):
        cfg.use_chunk_critic = False
    if critic_type in ('trl', 'state_transitive', 'direct_chunk_trl'):
        cfg.algorithm = critic_type if critic_type != 'direct_chunk_trl' else 'trl'
        cfg.goal_representation = 'full'
        cfg.value_base_horizon = cfg.action_chunk_horizon
    validate_config(cfg)
    return cfg


def _critic_dataset_for(cfg):
    base = _make_dummy_dataset()
    return CriticSequenceDataset(base, cfg)


def _build_critic(cfg, ex_batch):
    ex_full = ex_batch['full_chunk_actions'] if cfg.critic_type == 'dqc' else None
    return CriticAgent.create(
        seed=0,
        ex_observations=ex_batch['observations'],
        ex_full_chunk_actions=ex_full,
        ex_action_chunk_actions=ex_batch['action_chunk_actions'],
        config=dict(cfg),
        ex_goals=ex_batch['value_goals'],
    )


def _assert_finite_info(info, keys, prefix=''):
    for k in keys:
        v = float(info[k])
        assert np.isfinite(v), f'{prefix}{k!r} not finite: {v}'


def test_dataset_emits_action_chunk_fields_in_all_modes():
    for critic_type in ('dqc', 'iql', 'trl', 'state_transitive'):
        cfg = _make_critic_config(critic_type)
        ds = _critic_dataset_for(cfg)
        batch = ds.sample(BATCH)
        for key in (
            'observations',
            'full_chunk_actions',
            'action_chunk_actions',
            'full_chunk_next_observations',
            'full_chunk_rewards',
            'full_chunk_masks',
            'full_chunk_horizon',
            'action_chunk_next_observations',
            'action_chunk_rewards',
            'action_chunk_masks',
            'action_chunk_horizon_per_sample',
        ):
            assert key in batch, f'[{critic_type}] missing batch key {key!r}'
        assert batch['action_chunk_next_observations'].shape == (BATCH, STATE_DIM)
        assert batch['action_chunk_rewards'].shape == (BATCH,)
        assert batch['action_chunk_masks'].shape == (BATCH,)
        assert batch['action_chunk_horizon_per_sample'].shape == (BATCH,)
        assert np.all(batch['action_chunk_horizon_per_sample'] >= 0.0)
        assert np.all(batch['action_chunk_horizon_per_sample'] <= float(cfg.action_chunk_horizon))


def test_trl_dataset_emits_state_transitive_fields():
    cfg = _make_critic_config('trl')
    ds = _critic_dataset_for(cfg)
    batch = ds.sample(BATCH)
    for key in (
        'value_goals',
        'trans_v_split_observations',
        'trans_v_left_goals',
        'trans_v_right_observations',
        'trans_v_right_goals',
        'trans_v_valid_mask',
        'value_offsets',
        'trans_v_split_offsets',
        'value_base_goals',
        'value_base_offsets',
        'q_goals',
        'q_goal_offsets',
    ):
        assert key in batch, f'trl missing batch key {key!r}'
    assert batch['value_base_goals'].shape == (BATCH, STATE_DIM)
    assert np.all(batch['value_base_offsets'] >= 1.0)
    assert np.all(batch['value_base_offsets'] <= float(cfg.value_base_horizon))
    valid = batch['trans_v_valid_mask'] > 0.0
    if np.any(valid):
        assert np.all(batch['trans_v_split_offsets'][valid] > 0.0)
        assert np.all(batch['trans_v_split_offsets'][valid] < batch['value_offsets'][valid])


def test_trl_aliases_canonicalize_to_trl():
    cfg = get_critic_config()
    cfg.critic_type = 'direct_chunk_trl'
    cfg.algorithm = 'direct_chunk_trl'
    cfg.use_chunk_critic = True
    cfg.action_chunk_horizon = 4
    cfg.full_chunk_horizon = 8
    cfg.action_dim = ACTION_DIM
    validate_config(cfg)
    assert cfg['critic_type'] == 'trl'
    assert cfg['algorithm'] == 'trl'
    assert bool(cfg['use_chunk_critic']) is False


def test_trl_update_has_finite_losses_and_no_target_gradients():
    cfg = _make_critic_config('trl')
    ds = _critic_dataset_for(cfg)
    ex_batch = ds.sample(BATCH)
    critic = _build_critic(cfg, ex_batch)
    assert critic._is_trl()
    params = critic.network.params
    for k in ('modules_action_critic', 'modules_target_action_critic', 'modules_value', 'modules_target_value'):
        assert k in params, f'trl missing {k!r}'
    _, info = critic.update(ex_batch)
    _assert_finite_info(
        info,
        (
            'value/loss',
            'value/base_loss',
            'value/tri_loss',
            'local_q/loss',
            'local_q/target_v_mean',
            'action_critic/trl_loss',
        ),
        prefix='trl ',
    )

    def loss_fn(params):
        loss, _ = critic.total_loss(ex_batch, params)
        return loss

    grads = jax.grad(loss_fn)(critic.network.params)

    def _tree_abs_sum(tree):
        leaves = jax.tree_util.tree_leaves(tree)
        return float(sum(np.asarray(jnp.sum(jnp.abs(x))) for x in leaves))

    assert _tree_abs_sum(grads['modules_value']) > 0.0
    assert _tree_abs_sum(grads['modules_action_critic']) > 0.0
    assert _tree_abs_sum(grads['modules_target_value']) == 0.0
    assert _tree_abs_sum(grads['modules_target_action_critic']) == 0.0


def test_trl_zero_valid_mask_has_zero_tri_loss():
    cfg = _make_critic_config('trl')
    ds = _critic_dataset_for(cfg)
    ex_batch = ds.sample(BATCH)
    ex_batch = dict(ex_batch)
    ex_batch['trans_v_valid_mask'] = np.zeros_like(ex_batch['trans_v_valid_mask'])
    critic = _build_critic(cfg, ex_batch)
    _, info = critic.update(ex_batch)
    assert float(info['value/tri_loss']) == 0.0
    assert float(info['value/trans_valid_fraction']) == 0.0
    _assert_finite_info(
        info,
        (
            'loss/total',
            'value/self_loss',
            'value/base_loss',
            'local_q/loss',
            'value/tri_distance_weight_mean',
        ),
        prefix='trl zero-mask ',
    )


def test_trl_distance_reweight_logs_finite():
    cfg = _make_critic_config('trl')
    ds = _critic_dataset_for(cfg)
    ex_batch = ds.sample(BATCH)
    critic = _build_critic(cfg, ex_batch)
    _, info = critic.update(ex_batch)
    _assert_finite_info(
        info,
        (
            'value/tri_loss',
            'value/tri_distance_weight_mean',
            'value/value_offset_mean',
            'value/split_offset_mean',
        ),
        prefix='trl reweight ',
    )


def test_trl_rescores_single_candidate():
    from main import _rescore_actor_batch_for_update

    cfg = _make_critic_config('trl')
    ds = _critic_dataset_for(cfg)
    ex_batch = ds.sample(BATCH)
    critic = _build_critic(cfg, ex_batch)
    ha = int(cfg.action_chunk_horizon)
    actor_batch = {
        'observations': ex_batch['observations'],
        'spi_goals': ex_batch['value_goals'],
        'candidate_partial_chunks': ex_batch['action_chunk_actions'].reshape(BATCH, 1, ha, ACTION_DIM),
        'valids': np.ones((BATCH, ha), dtype=np.float32),
    }
    out_batch, stats = _rescore_actor_batch_for_update(actor_batch, critic, actor_config={})
    assert out_batch['proposal_scores'].shape == (BATCH, 1)
    assert np.all(np.isfinite(out_batch['proposal_scores']))
    assert np.isfinite(float(stats['critic_score_mean']))


def test_critic_dataset_clips_backup_to_in_window_goal_by_default():
    cfg = _make_critic_config('iql')
    cfg.discount = 0.9
    ds = _critic_dataset_for(cfg)
    idxs = np.arange(BATCH, dtype=np.int64)
    ds.sample_goals = lambda sampled_idxs: np.asarray(sampled_idxs, dtype=np.int64) + 2

    batch = ds.sample(BATCH, idxs=idxs)

    expected_next = np.asarray(ds.get_observations(idxs + 2), dtype=np.float32)
    np.testing.assert_allclose(batch['action_chunk_next_observations'], expected_next)
    np.testing.assert_allclose(batch['full_chunk_next_observations'], expected_next)
    np.testing.assert_array_equal(batch['action_chunk_horizon_per_sample'], np.full((BATCH,), 2.0, dtype=np.float32))
    np.testing.assert_array_equal(batch['full_chunk_horizon'], np.full((BATCH,), 2.0, dtype=np.float32))
    np.testing.assert_array_equal(batch['action_chunk_masks'], np.zeros((BATCH,), dtype=np.float32))
    np.testing.assert_array_equal(batch['full_chunk_masks'], np.zeros((BATCH,), dtype=np.float32))
    np.testing.assert_allclose(batch['action_chunk_rewards'], np.full((BATCH,), 0.9**2, dtype=np.float32))
    np.testing.assert_allclose(batch['full_chunk_rewards'], np.full((BATCH,), 0.9**2, dtype=np.float32))


def test_critic_dataset_can_disable_goal_clipping():
    cfg = _make_critic_config('iql')
    cfg.discount = 0.9
    cfg.clip_chunk_to_goal = False
    ds = _critic_dataset_for(cfg)
    idxs = np.arange(BATCH, dtype=np.int64)
    ds.sample_goals = lambda sampled_idxs: np.asarray(sampled_idxs, dtype=np.int64) + 2

    batch = ds.sample(BATCH, idxs=idxs)

    np.testing.assert_allclose(batch['action_chunk_next_observations'], ds.get_observations(idxs + cfg.action_chunk_horizon))
    np.testing.assert_allclose(batch['full_chunk_next_observations'], ds.get_observations(idxs + cfg.full_chunk_horizon))
    np.testing.assert_array_equal(
        batch['action_chunk_horizon_per_sample'],
        np.full((BATCH,), cfg.action_chunk_horizon, dtype=np.float32),
    )
    np.testing.assert_array_equal(batch['full_chunk_horizon'], np.full((BATCH,), cfg.full_chunk_horizon, dtype=np.float32))
    np.testing.assert_array_equal(batch['action_chunk_masks'], np.ones((BATCH,), dtype=np.float32))
    np.testing.assert_array_equal(batch['full_chunk_masks'], np.ones((BATCH,), dtype=np.float32))
    np.testing.assert_allclose(batch['action_chunk_rewards'], np.full((BATCH,), 0.9**2, dtype=np.float32))
    np.testing.assert_allclose(batch['full_chunk_rewards'], np.full((BATCH,), 0.9**2, dtype=np.float32))


def test_dqc_init_and_one_update_step():
    cfg = _make_critic_config('dqc')
    ds = _critic_dataset_for(cfg)
    ex_batch = ds.sample(BATCH)
    critic = _build_critic(cfg, ex_batch)
    params = critic.network.params
    for k in ('modules_action_critic', 'modules_target_action_critic', 'modules_value',
              'modules_chunk_critic', 'modules_target_chunk_critic'):
        assert k in params, f'DQC missing {k!r}'
    _, info = critic.update(ex_batch)
    assert 'chunk_critic/critic_loss' in info
    assert 'action_critic/distill_loss' in info
    assert 'action_critic/value_loss' in info
    for k in ('chunk_critic/critic_loss', 'action_critic/distill_loss', 'action_critic/value_loss'):
        v = float(info[k])
        assert np.isfinite(v), f'DQC {k!r} not finite: {v}'


def test_iql_skips_chunk_critic_and_runs_one_update_step():
    cfg = _make_critic_config('iql')
    ds = _critic_dataset_for(cfg)
    ex_batch = ds.sample(BATCH)
    critic = _build_critic(cfg, ex_batch)
    params = critic.network.params
    for k in ('modules_chunk_critic', 'modules_target_chunk_critic'):
        assert k not in params, f'IQL should not init {k!r}'
    for k in ('modules_action_critic', 'modules_target_action_critic', 'modules_value'):
        assert k in params, f'IQL missing {k!r}'
    _, info = critic.update(ex_batch)
    assert float(info['chunk_critic/critic_loss']) == 0.0
    for k in ('action_critic/distill_loss', 'action_critic/value_loss'):
        v = float(info[k])
        assert np.isfinite(v), f'IQL {k!r} not finite: {v}'


def test_trl_skips_dqc_chunk_critics_and_runs_trl_update_step():
    cfg = _make_critic_config('trl')
    ds = _critic_dataset_for(cfg)
    ex_batch = ds.sample(BATCH)
    critic = _build_critic(cfg, ex_batch)
    params = critic.network.params
    for k in ('modules_chunk_critic', 'modules_target_chunk_critic'):
        assert k not in params, f'TRL should not init {k!r}'
    for k in ('modules_action_critic', 'modules_target_action_critic', 'modules_value', 'modules_target_value'):
        assert k in params, f'TRL missing {k!r}'
    _, info = critic.update(ex_batch)
    assert float(info['chunk_critic/critic_loss']) == 0.0
    _assert_finite_info(info, (
        'loss/total',
        'value/self_loss',
        'value/base_loss',
        'value/tri_loss',
        'local_q/loss',
        'action_critic/trl_loss',
        'action_critic/q_part_mean',
    ), prefix='TRL ')


def test_trl_head_shapes_and_score_action_chunks():
    cfg = _make_critic_config('trl')
    ds = _critic_dataset_for(cfg)
    ex_batch = ds.sample(BATCH)
    critic = _build_critic(cfg, ex_batch)
    q_logits = critic.network.select('action_critic')(
        jnp.asarray(ex_batch['observations']),
        jnp.asarray(ex_batch['value_goals']),
        jnp.asarray(ex_batch['action_chunk_actions']),
    )
    v_logits = critic.network.select('value')(
        jnp.asarray(ex_batch['observations']),
        jnp.asarray(ex_batch['value_goals']),
    )
    assert q_logits.shape == (int(cfg.num_qs), BATCH)
    assert v_logits.shape == (BATCH,)
    scores_flat = critic.score_action_chunks(
        jnp.asarray(ex_batch['observations']),
        jnp.asarray(ex_batch['value_goals']),
        jnp.asarray(ex_batch['action_chunk_actions']),
        network_params=critic.network.params,
    )
    scores_4d = critic.score_action_chunks(
        jnp.asarray(ex_batch['observations']),
        jnp.asarray(ex_batch['value_goals']),
        jnp.asarray(ex_batch['action_chunk_actions']).reshape(
            BATCH, 1, cfg.action_chunk_horizon, ACTION_DIM
        ),
        network_params=critic.network.params,
    )
    np.testing.assert_allclose(np.asarray(scores_flat), np.asarray(scores_4d))
    _, info = critic.update(ex_batch)
    assert float(info['action_critic/trl_loss']) == float(info['loss/total'])


def test_score_action_chunks_works_in_all_modes():
    for critic_type in ('dqc', 'iql', 'trl', 'state_transitive'):
        cfg = _make_critic_config(critic_type)
        ds = _critic_dataset_for(cfg)
        ex_batch = ds.sample(BATCH)
        critic = _build_critic(cfg, ex_batch)
        obs = jnp.asarray(ex_batch['observations'], dtype=jnp.float32)
        goals = jnp.asarray(ex_batch['value_goals'], dtype=jnp.float32)
        n_cand = 5
        cand = np.random.uniform(
            -1.0, 1.0,
            size=(BATCH, n_cand, cfg.action_chunk_horizon, ACTION_DIM),
        ).astype(np.float32)
        scores = critic.score_action_chunks(
            obs, goals, jnp.asarray(cand),
            network_params=critic.network.params,
        )
        assert scores.shape == (BATCH, n_cand), f'[{critic_type}] {scores.shape}'
        assert np.all(np.isfinite(np.asarray(scores))), f'[{critic_type}] non-finite scores'


def test_validate_config_iql_and_trl_force_use_chunk_critic_off():
    for critic_type in ('iql', 'trl', 'state_transitive'):
        cfg = _make_critic_config(critic_type)
        cfg.use_chunk_critic = True
        validate_config(cfg)
        assert bool(cfg.use_chunk_critic) is False


def test_validate_config_rejects_unknown_critic_type():
    cfg = _make_critic_config('dqc')
    cfg.critic_type = 'foo'
    try:
        validate_config(cfg)
    except ValueError as e:
        assert 'critic_type' in str(e)
        return
    raise AssertionError('validate_config must reject unknown critic_type')
