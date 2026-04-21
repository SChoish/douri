"""Unified critic package: DEAS and DQC critic-only stacks."""

from __future__ import annotations

import math
import ml_collections

from .common import BinaryChunkCritic, ScalarValueNet
from .deas import DEASSeqCriticAgent
from .dqc import DQCCriticAgent

CRITIC_REGISTRY = {
    'deas': DEASSeqCriticAgent,
    'dqc': DQCCriticAgent,
}


def normalize_critic_name(critic_name: str) -> str:
    key = str(critic_name).lower().strip()
    if key not in CRITIC_REGISTRY:
        raise ValueError(f"Unknown critic={critic_name!r}. Available: {sorted(CRITIC_REGISTRY.keys())}")
    return key


def get_critic_class(critic_name: str):
    return CRITIC_REGISTRY[normalize_critic_name(critic_name)]


def validate_joint_mode(
    critic_name: str,
    critic_config,
    actor_config=None,
    *,
    plan_candidates: int | None = None,
    proposal_topk: int | None = None,
    deas_spi_requested: bool = False,
) -> None:
    key = normalize_critic_name(critic_name)
    if key == 'dqc':
        action_chunk_horizon = int(critic_config.get('action_chunk_horizon', 0))
        full_chunk_horizon = int(critic_config.get('full_chunk_horizon', 0))
        if action_chunk_horizon < 1:
            raise ValueError('DQC joint mode requires action_chunk_horizon >= 1.')
        if full_chunk_horizon < action_chunk_horizon:
            raise ValueError(
                f'DQC joint mode requires full_chunk_horizon >= action_chunk_horizon, '
                f'got full_chunk_horizon={full_chunk_horizon}, action_chunk_horizon={action_chunk_horizon}.'
            )
    if plan_candidates is not None and proposal_topk is not None and int(proposal_topk) > int(plan_candidates):
        raise ValueError(
            f'proposal_topk must be <= plan_candidates, got proposal_topk={proposal_topk}, '
            f'plan_candidates={plan_candidates}.'
        )
    if actor_config is None:
        return
    if key == 'deas' and (deas_spi_requested or bool(actor_config.get('use_spi_actor', False))):
        raise ValueError(
            'DEAS joint mode is critic-only by default in this codebase; SPI actor is supported only for DQC.'
        )
    if bool(actor_config.get('use_spi_actor', False)) and int(actor_config.get('actor_chunk_horizon', 0)) < 1:
        raise ValueError('Actor-enabled joint training requires actor_chunk_horizon >= 1.')


def extract_value_loss(critic_name: str, info: dict) -> float:
    key = normalize_critic_name(critic_name)
    if key == 'deas':
        return float(info['value/value_loss'])
    return float(info['action_critic/value_loss'])


def extract_primary_critic_loss(critic_name: str, info: dict) -> float:
    key = normalize_critic_name(critic_name)
    if key == 'deas':
        return float(info['critic/critic_loss'])
    if 'chunk_critic/critic_loss' in info:
        return float(info['chunk_critic/critic_loss'])
    return float(info['action_critic/distill_loss'])


def extract_critic_total_loss(critic_name: str, info: dict) -> float:
    key = normalize_critic_name(critic_name)
    if key == 'deas':
        return extract_value_loss(key, info) + extract_primary_critic_loss(key, info)
    return float(info['dqc_critic/total_loss'])


def extract_critic_primary_score(critic_name: str, info: dict) -> float:
    key = normalize_critic_name(critic_name)
    if key == 'deas':
        return float(info['critic/q_mean'])
    if 'chunk_critic/q_mean' in info:
        return float(info['chunk_critic/q_mean'])
    return float(info['action_critic/q_part_mean'])


def extract_actor_loss(critic_name: str, info: dict | None) -> float:
    normalize_critic_name(critic_name)
    if info is None or 'spi_actor/actor_loss' not in info:
        return math.nan
    return float(info['spi_actor/actor_loss'])


def get_config():
    return ml_collections.ConfigDict(
        dict(
            critic='deas',
            agent_name='critic',
            lr=3e-4,
            batch_size=256,
            tau=0.005,
            layer_norm=True,
            frame_stack=None,
            p_aug=0.0,
            q_agg='min',
            # DEAS defaults
            expectile=0.7,
            num_atoms=51,
            v_min=-200.0,
            v_max=0.0,
            sigma=1.0,
            num_critic_ensembles=2,
            gamma1=0.99,
            gamma2=0.99,
            full_chunk_horizon=25,
            action_chunk_horizon=10,
            nstep_options=1,
            dataset_class='DEASActionSeqDataset',
            critic_hidden_dims=(256, 256, 256),
            value_hidden_dims=(256, 256, 256),
            # DQC defaults
            discount=0.99,
            num_qs=2,
            use_chunk_critic=True,
            distill_method='expectile',
            kappa_d=0.7,
            implicit_backup_type='expectile',
            kappa_b=0.7,
            use_spi_actor=False,
            spi_tau=0.5,
            spi_beta=10.0,
            spi_num_samples=32,
            spi_candidate_source='external',
            spi_use_partial_critic=True,
            spi_actor_hidden_dims=(256, 256, 256),
            spi_actor_layer_norm=True,
            spi_eval_use_actor=False,
            spi_dist_normalize_by_dim=True,
            spi_warmstart_steps=0,
            action_dim=2,
            value_p_curgoal=0.2,
            value_p_trajgoal=0.5,
            value_p_randomgoal=0.3,
            value_geom_sample=True,
            gc_negative=True,
        )
    )


__all__ = [
    'BinaryChunkCritic',
    'ScalarValueNet',
    'DEASSeqCriticAgent',
    'DQCCriticAgent',
    'normalize_critic_name',
    'validate_joint_mode',
    'extract_value_loss',
    'extract_primary_critic_loss',
    'extract_critic_total_loss',
    'extract_critic_primary_score',
    'extract_actor_loss',
    'get_critic_class',
    'get_config',
]
