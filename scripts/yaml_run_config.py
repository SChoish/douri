"""Ordered TRL run-config builders matching hand-written experiment YAMLs."""

from __future__ import annotations

from typing import Any

import yaml

# Goal sampling presets (critic_agent section).
MAZE_VALUE_GOAL_SAMPLING: dict[str, Any] = {
    'value_p_curgoal': 0.0,
    'value_p_trajgoal': 1.0,
    'value_p_randomgoal': 0.0,
    'value_geom_sample': True,
}

PUZZLE_VALUE_GOAL_SAMPLING: dict[str, Any] = {
    'value_p_curgoal': 0.0,
    'value_p_trajgoal': 1.0,
    'value_p_randomgoal': 0.0,
    'value_geom_sample': True,
}

LONG_HORIZON_VALUE_GOAL_SAMPLING: dict[str, Any] = {
    'value_p_curgoal': 0.0,
    'value_p_trajgoal': 1.0,
    'value_p_randomgoal': 0.0,
    'value_geom_sample': False,
}

MAZE_ACTOR_GOAL_SAMPLING: dict[str, Any] = {
    'actor_p_curgoal': 0.0,
    'actor_p_trajgoal': 1.0,
    'actor_p_randomgoal': 0.0,
    'actor_geom_sample': False,
}

PUZZLE_ACTOR_GOAL_SAMPLING: dict[str, Any] = {
    'actor_p_curgoal': 0.0,
    'actor_p_trajgoal': 0.5,
    'actor_p_randomgoal': 0.5,
    'actor_geom_sample': True,
}


def _dynamics_block(
    *,
    gap_scale: float,
    weight_max: float,
    dynamics_overrides: dict[str, Any] | None = None,
    actor_goal_sampling: dict[str, Any] | None = None,
) -> dict[str, Any]:
    block: dict[str, Any] = {
        'max_goal_steps_from_env': False,
        'clip_path_to_goal': True,
        'subgoal_distribution': 'diag_gaussian',
        'subgoal_stochastic_loss': 'nll',
        'subgoal_num_samples': 1,
        'subgoal_value_alpha': 0.0,
        'subgoal_value_gap_scale': float(gap_scale),
        'subgoal_value_weight_max': float(weight_max),
        'subgoal_goal_representation': 'phi',
        'subgoal_target_mode': 'displacement',
        'planner_type': 'forward_bridge_residual',
        'forward_bridge_path_loss_horizon': 5,
        'theta_schedule': 'prefix_progress',
        'theta_total': 1.0,
        'progress_alpha': 0.8,
        'residual_target_mode': 'displacement',
    }
    if actor_goal_sampling:
        block.update(actor_goal_sampling)
    if dynamics_overrides:
        block.update(dynamics_overrides)
    return block


def _critic_block(
    *,
    discount: float,
    value_distance_weight_power: float,
    value_goal_sampling: dict[str, Any],
    critic_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    block: dict[str, Any] = {
        'algorithm': 'trl',
        'critic_type': 'trl',
        'use_chunk_critic': False,
        'action_chunk_horizon': 5,
        'full_chunk_horizon': 25,
        'discount': float(discount),
        'tau_v': 0.7,
        'lambda_v_self': 1.0,
        'lambda_v_base': 1.0,
        'lambda_v_tri': 1.0,
        'value_base_horizon': 5,
        'value_transitive_reweight': True,
        'value_distance_weight_power': float(value_distance_weight_power),
        'lambda_q_local': 1.0,
        'q_target_from_value': True,
        'goal_representation': 'full',
        'kappa_b': 0.9,
        'kappa_d': 0.9,
        'max_goal_steps_from_env': False,
        'clip_chunk_to_goal': True,
        'subgoal_value_bonus_type': 'transitive_product',
        'subgoal_value_log_eps': 1.0e-6,
        'subgoal_value_ratio_eps': 1.0e-3,
        'subgoal_value_ratio_clip': 5.0,
    }
    block.update(value_goal_sampling)
    if critic_overrides:
        block.update(critic_overrides)
    return block


def build_trl_run_config(
    *,
    env_name: str,
    run_group: str,
    gap_scale: float,
    weight_max: float,
    discount: float,
    value_distance_weight_power: float,
    batch_size: int = 1024,
    train_epochs: int = 600,
    log_every_n_epochs: int = 10,
    save_every_n_epochs: int = 100,
    horizon: int = 25,
    plan_candidates: int = 1,
    plan_noise_scale: float = 1.0,
    eval_freq: int = 100,
    eval_task_ids: str = '1,2,3,4,5',
    eval_episodes_per_task: int = 10,
    final_eval_episodes_per_task: int = 25,
    eval_max_chunks: int = 200,
    eval_video_episodes_per_task: int = 0,
    dynamics_overrides: dict[str, Any] | None = None,
    critic_overrides: dict[str, Any] | None = None,
    actor_overrides: dict[str, Any] | None = None,
    value_goal_sampling: dict[str, Any] | None = None,
    actor_goal_sampling: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal TRL + diag_gaussian run config in canonical field order."""
    if value_goal_sampling is None:
        value_goal_sampling = MAZE_VALUE_GOAL_SAMPLING
    actor_block = {'spi_beta': 1.0, 'spi_tau': 5.0}
    if actor_overrides:
        actor_block.update(actor_overrides)

    return {
        'env_name': str(env_name),
        'run_group': str(run_group),
        'train_epochs': int(train_epochs),
        'log_every_n_epochs': int(log_every_n_epochs),
        'save_every_n_epochs': int(save_every_n_epochs),
        'horizon': int(horizon),
        'plan_candidates': int(plan_candidates),
        'plan_noise_scale': float(plan_noise_scale),
        'eval_freq': int(eval_freq),
        'eval_task_ids': str(eval_task_ids),
        'eval_episodes_per_task': int(eval_episodes_per_task),
        'final_eval_episodes_per_task': int(final_eval_episodes_per_task),
        'eval_max_chunks': int(eval_max_chunks),
        'eval_video_episodes_per_task': int(eval_video_episodes_per_task),
        'dynamics': _dynamics_block(
            gap_scale=gap_scale,
            weight_max=weight_max,
            dynamics_overrides=dynamics_overrides,
            actor_goal_sampling=actor_goal_sampling,
        ),
        'critic_agent': _critic_block(
            discount=discount,
            value_distance_weight_power=value_distance_weight_power,
            value_goal_sampling=value_goal_sampling,
            critic_overrides=critic_overrides,
        ),
        'actor': actor_block,
        'batch_size': int(batch_size),
    }


def dump_run_config_yaml(path, cfg: dict[str, Any], header: str = '') -> None:
    with open(path, 'w', encoding='utf-8') as f:
        if header:
            f.write(header if header.endswith('\n') else header + '\n')
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
