"""One-step compile/smoke test for DEAS seq critic (no OGBench download)."""

import numpy as np

from agents.deas_seq_critic import DEASSeqCriticAgent, get_config
from utils.datasets import Dataset
from utils.deas_sequence_dataset import DEASActionSeqDataset


def main():
    T = 64
    obs_dim, act_dim = 5, 2
    obs = np.random.randn(T, obs_dim).astype(np.float32)
    act = np.random.randn(T, act_dim).astype(np.float32)
    rew = (np.random.randn(T) * 0.01).astype(np.float32)
    term = np.zeros(T, dtype=np.float32)
    term[-1] = 1.0
    val = np.ones(T, dtype=np.float32)
    ds = Dataset.create(observations=obs, actions=act, rewards=rew, terminals=term, valids=val)

    cfg = get_config()
    cfg['critic_action_sequence'] = 3
    cfg['nstep_options'] = 1
    cfg['batch_size'] = 8
    cfg['num_atoms'] = 11
    cfg['v_min'] = -10.0
    cfg['v_max'] = 10.0

    dset = DEASActionSeqDataset(ds, cfg)
    b = dset.sample(8)
    assert b['actions'].shape == (8, 3, act_dim), b['actions'].shape
    assert b['step_rewards'].shape == (8, 1, 3), b['step_rewards'].shape

    agent = DEASSeqCriticAgent.create(0, b['observations'][:1], b['actions'][:1], cfg)
    agent, info = agent.update(b)
    assert 'critic/critic_loss' in info and 'value/value_loss' in info
    import jax
    import jax.numpy as jnp

    p = jax.nn.softmax(jnp.zeros((3, int(cfg['num_atoms']))), axis=-1)
    assert np.allclose(np.asarray(p.sum(axis=-1)), 1.0, atol=1e-5)
    print('smoke_deas_compile_ok', float(info['critic/critic_loss']), float(info['value/value_loss']))


if __name__ == '__main__':
    main()
