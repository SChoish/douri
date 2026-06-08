"""Shared constants/builders for gap / weight_max / gamma tune sweeps (Set A + Set B)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from yaml_run_config import (
    LONG_HORIZON_VALUE_GOAL_SAMPLING,
    MAZE_ACTOR_GOAL_SAMPLING,
    MAZE_VALUE_GOAL_SAMPLING,
    PUZZLE_ACTOR_GOAL_SAMPLING,
    PUZZLE_VALUE_GOAL_SAMPLING,
    build_trl_run_config,
)

ENV_SPECS: dict[str, dict[str, Any]] = {
    'p3': {
        'env_name': 'puzzle-3x3-play-v0',
        'regime': 'standard',
        'policy_family': 'puzzle',
        'value_distance_weight_power': 0.5,
        'default_gamma': 0.995,
        'gamma_set_a_alt': 0.999,
        'gamma_set_b_g2': 0.999,
        'batch_size': 1024,
    },
    'p4': {
        'env_name': 'puzzle-4x4-play-v0',
        'regime': 'standard',
        'policy_family': 'puzzle',
        'value_distance_weight_power': 2.0,
        'default_gamma': 0.995,
        'gamma_set_a_alt': 0.999,
        'gamma_set_b_g2': 0.999,
        'batch_size': 1024,
    },
    'ct': {
        'env_name': 'cube-triple-play-v0',
        'regime': 'standard',
        'policy_family': 'maze',
        'value_distance_weight_power': 1.0,
        'default_gamma': 0.995,
        'gamma_set_a_alt': 0.999,
        'gamma_set_b_g2': 0.999,
        'batch_size': 4096,
    },
    'ag': {
        'env_name': 'antmaze-giant-navigate-v0',
        'regime': 'standard',
        'policy_family': 'maze',
        'value_distance_weight_power': 0.0,
        'default_gamma': 0.995,
        'gamma_set_a_alt': 0.999,
        'gamma_set_b_g2': 0.999,
        'batch_size': 1024,
    },
    'hg': {
        'env_name': 'humanoidmaze-giant-navigate-v0',
        'regime': 'long_horizon',
        'policy_family': 'maze',
        'value_distance_weight_power': 0.0,
        'default_gamma': 0.999,
        'gamma_set_a_alt': 0.995,
        'gamma_set_b_g2': 0.995,
        'batch_size': 8192,
    },
}

SET_A_VARIANTS: dict[str, dict[str, Any]] = {
    'gap3': {'gap': 3.0, 'wmax': 5.0, 'gamma': 'default'},
    'gap5': {'gap': 5.0, 'wmax': 5.0, 'gamma': 'default'},
    'wmax3': {'gap': 10.0, 'wmax': 3.0, 'gamma': 'default'},
    'altg': {'gap': 10.0, 'wmax': 5.0, 'gamma': 'set_a_alt'},
}

SET_B_VARIANTS: dict[str, dict[str, Any]] = {
    'gap7': {'gap': 7.0, 'wmax': 5.0, 'gamma': 'default'},
    'gap15': {'gap': 15.0, 'wmax': 5.0, 'gamma': 'default'},
    'wmax10': {'gap': 10.0, 'wmax': 10.0, 'gamma': 'default'},
    'g2': {'gap': 10.0, 'wmax': 5.0, 'gamma': 'set_b_g2'},
}

GAP1_VARIANTS: dict[str, dict[str, Any]] = {
    'w5_g999': {'gap': 1.0, 'wmax': 5.0, 'discount': 0.999},
    'w5_g995': {'gap': 1.0, 'wmax': 5.0, 'discount': 0.995},
    'w10_g999': {'gap': 1.0, 'wmax': 10.0, 'discount': 0.999},
    'w10_g995': {'gap': 1.0, 'wmax': 10.0, 'discount': 0.995},
}


_TRL_CRITIC_ALIASES = ('trl', 'chunk_trl', 'direct_chunk_trl', 'state_transitive', 'transitive_v_local_q')


def apply_legacy_critic_type(cri: dict[str, Any], critic_type: str) -> str:
    """Normalize legacy critic_type aliases for grid YAML writers."""
    ct = str(critic_type).lower()
    if ct in _TRL_CRITIC_ALIASES:
        cri['critic_type'] = 'trl'
        cri['algorithm'] = 'trl'
        cri['use_chunk_critic'] = False
        return 'trl'
    cri['critic_type'] = ct
    if ct == 'iql':
        cri['use_chunk_critic'] = False
    return ct


def resolve_gamma(env_spec: dict[str, Any], gamma_key: str) -> float:
    if gamma_key == 'default':
        return float(env_spec['default_gamma'])
    if gamma_key == 'set_a_alt':
        return float(env_spec['gamma_set_a_alt'])
    if gamma_key == 'set_b_g2':
        return float(env_spec['gamma_set_b_g2'])
    raise ValueError(f'unknown gamma key: {gamma_key!r}')


def _goal_sampling_for_env(env_spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if env_spec['policy_family'] == 'puzzle':
        return deepcopy(PUZZLE_VALUE_GOAL_SAMPLING), deepcopy(PUZZLE_ACTOR_GOAL_SAMPLING)
    if env_spec['regime'] == 'long_horizon':
        return deepcopy(LONG_HORIZON_VALUE_GOAL_SAMPLING), deepcopy(MAZE_ACTOR_GOAL_SAMPLING)
    return deepcopy(MAZE_VALUE_GOAL_SAMPLING), deepcopy(MAZE_ACTOR_GOAL_SAMPLING)


def build_tune_config(
    *,
    env_prefix: str,
    variant_suffix: str,
    variant: dict[str, Any],
    run_group_prefix: str,
    set_label: str,
) -> dict[str, Any]:
    del set_label  # reserved for manifest labels
    env_spec = ENV_SPECS[env_prefix]
    value_sampling, actor_sampling = _goal_sampling_for_env(env_spec)
    if 'discount' in variant:
        gamma = float(variant['discount'])
    else:
        gamma = resolve_gamma(env_spec, str(variant['gamma']))

    return build_trl_run_config(
        env_name=str(env_spec['env_name']),
        run_group=f'{run_group_prefix}{env_prefix}_{variant_suffix}',
        gap_scale=float(variant['gap']),
        weight_max=float(variant['wmax']),
        discount=gamma,
        value_distance_weight_power=float(env_spec['value_distance_weight_power']),
        batch_size=int(env_spec['batch_size']),
        value_goal_sampling=value_sampling,
        actor_goal_sampling=actor_sampling,
    )


def config_sort_key(path_name: str) -> tuple[str, str]:
    stem = path_name.removesuffix('.yaml')
    if '_' not in stem:
        return stem, ''
    prefix, suffix = stem.split('_', 1)
    suffix_order = {
        'gap3': 0,
        'gap5': 1,
        'gap7': 2,
        'gap15': 3,
        'wmax3': 4,
        'wmax10': 5,
        'altg': 6,
        'g2': 7,
        'w5_g999': 0,
        'w5_g995': 1,
        'w10_g999': 2,
        'w10_g995': 3,
    }
    env_order = ['p3', 'p4', 'ct', 'ag', 'hg']
    return (f'{env_order.index(prefix):02d}_{prefix}', f'{suffix_order.get(suffix, 99):02d}_{suffix}')
