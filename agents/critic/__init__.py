"""Unified critic package: DEAS and DQC critic-only stacks."""

from __future__ import annotations

import ml_collections

from .common import BinaryChunkCritic, ScalarValueNet
from .deas import DEASSeqCriticAgent
from .dqc import DQCCriticAgent

CRITIC_REGISTRY = {
    'deas': DEASSeqCriticAgent,
    'dqc': DQCCriticAgent,
}


def get_critic_class(critic_name: str):
    key = str(critic_name).lower().strip()
    if key not in CRITIC_REGISTRY:
        raise ValueError(f"Unknown critic={critic_name!r}. Available: {sorted(CRITIC_REGISTRY.keys())}")
    return CRITIC_REGISTRY[key]


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
            full_chunk_horizon=4,
            action_chunk_horizon=4,
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
    'get_critic_class',
    'get_config',
]
