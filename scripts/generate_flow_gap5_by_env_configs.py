#!/usr/bin/env python3
"""Emit Flow + TRL YAMLs per env (plain Flow-BC, baseline-aligned critic/training)."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from yaml_run_config import (
    MAZE_ACTOR_GOAL_SAMPLING,
    MAZE_VALUE_GOAL_SAMPLING,
    PUZZLE_ACTOR_GOAL_SAMPLING,
    PUZZLE_VALUE_GOAL_SAMPLING,
    build_trl_run_config,
    dump_run_config_yaml,
)

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / 'config' / 'flow_gap5_by_env'

FLOW_DYNAMICS_PATCH: dict = {
    'subgoal_distribution': 'flow',
    'subgoal_flow_energy_weighted': False,
    'subgoal_flow_use_value_bonus': False,
    'subgoal_value_alpha': 0.0,
    'subgoal_value_gap_scale': 5.0,
    'subgoal_value_weight_max': 5.0,
    'subgoal_flow_steps': 8,
    'subgoal_flow_t_min': 1.0e-4,
    'subgoal_flow_noise_scale': 1.0,
    'subgoal_temperature': 1.0,
    'subgoal_num_samples': 4,
    'subgoal_eval_selection': 'best_of_n_value',
    'subgoal_eval_num_samples': 4,
    'subgoal_eval_include_zero_candidate': False,
    'subgoal_eval_seed': 0,
    'subgoal_stochastic_loss': 'mse',
}

ENV_SPECS: list[dict] = [
    {
        'stem': 'puzzle_3x3',
        'env_name': 'puzzle-3x3-play-v0',
        'batch_size': 1024,
        'discount': 0.99,
        'value_distance_weight_power': 0.5,
        'value_goal_sampling': PUZZLE_VALUE_GOAL_SAMPLING,
        'actor_goal_sampling': PUZZLE_ACTOR_GOAL_SAMPLING,
        'kappa_b': 0.6,
        'kappa_d': 0.6,
    },
    {
        'stem': 'antmaze_medium',
        'env_name': 'antmaze-medium-navigate-v0',
        'batch_size': 1024,
        'discount': 0.99,
        'value_distance_weight_power': 0.0,
        'kappa_b': 0.7,
        'kappa_d': 0.7,
    },
    {
        'stem': 'antmaze_large',
        'env_name': 'antmaze-large-navigate-v0',
        'batch_size': 1024,
        'discount': 0.995,
        'value_distance_weight_power': 0.0,
        'kappa_b': 0.9,
        'kappa_d': 0.9,
    },
    {
        'stem': 'antmaze_giant',
        'env_name': 'antmaze-giant-navigate-v0',
        'batch_size': 1024,
        'discount': 0.99,
        'value_distance_weight_power': 0.0,
        'kappa_b': 0.8,
        'kappa_d': 0.8,
    },
    {
        'stem': 'cube_single',
        'env_name': 'cube-single-play-v0',
        'batch_size': 1024,
        'discount': 0.99,
        'value_distance_weight_power': 1.0,
        'kappa_b': 0.9,
        'kappa_d': 0.9,
    },
    {
        'stem': 'cube_double',
        'env_name': 'cube-double-play-v0',
        'batch_size': 1024,
        'discount': 0.99,
        'value_distance_weight_power': 1.0,
        'kappa_b': 0.6,
        'kappa_d': 0.6,
    },
    {
        'stem': 'cube_triple',
        'env_name': 'cube-triple-play-v0',
        'batch_size': 4096,
        'discount': 0.995,
        'value_distance_weight_power': 1.0,
        'kappa_b': 0.8,
        'kappa_d': 0.8,
    },
]


def _build_config(spec: dict) -> dict:
    dynamics_overrides = deepcopy(FLOW_DYNAMICS_PATCH)
    value_sampling = spec.get('value_goal_sampling', MAZE_VALUE_GOAL_SAMPLING)
    actor_sampling = spec.get('actor_goal_sampling', MAZE_ACTOR_GOAL_SAMPLING)
    return build_trl_run_config(
        env_name=str(spec['env_name']),
        run_group=f'FlowGap5TRL_rd_sd_{spec["stem"]}',
        gap_scale=5.0,
        weight_max=5.0,
        discount=float(spec['discount']),
        value_distance_weight_power=float(spec['value_distance_weight_power']),
        batch_size=int(spec['batch_size']),
        dynamics_overrides=dynamics_overrides,
        critic_overrides={
            'kappa_b': float(spec['kappa_b']),
            'kappa_d': float(spec['kappa_d']),
        },
        value_goal_sampling=deepcopy(value_sampling),
        actor_goal_sampling=deepcopy(actor_sampling),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for spec in ENV_SPECS:
        cfg = _build_config(spec)
        out_trl = OUT / f'flow_gap5_trl_{spec["stem"]}_rd_sd.yaml'
        header = (
            f'# Plain Flow-BC + TRL baseline, gap=5.0, wmax=5.0, rd_sd\n'
            f'# env={cfg["env_name"]}\n'
        )
        dump_run_config_yaml(out_trl, cfg, header=header)
        print(out_trl)

        plain = deepcopy(cfg)
        plain['run_group'] = f'FlowGap5_rd_sd_{spec["stem"]}'
        out_plain = OUT / f'flow_gap5_{spec["stem"]}_rd_sd.yaml'
        header_plain = (
            f'# Plain Flow-BC + TRL baseline (alias), gap=5.0, wmax=5.0, rd_sd\n'
            f'# env={plain["env_name"]}\n'
        )
        dump_run_config_yaml(out_plain, plain, header=header_plain)
        print(out_plain)


if __name__ == '__main__':
    main()
