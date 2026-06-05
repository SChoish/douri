#!/usr/bin/env python3
"""Generate YAML configs for puzzle forward-bridge-residual + displacement grid sweep."""

from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / 'config' / 'grid_fbr_displacement_puzzle'
BASELINE_REL = Path('config') / 'puzzle_3x3_play_table_phi_disp_diag_gaussian.yaml'

ENVS = (
    ('puzzle-3x3-play-v0', 'puzzle_3x3', None),  # (env_name, file_prefix, discount override None=baseline)
    ('puzzle-4x4-play-v0', 'puzzle_4x4', None),
    ('puzzle-4x6-play-v0', 'puzzle_4x6', 0.995),
)

ALPHAS = (0.0, 0.1, 0.3, 0.5)
GAPS = (0.0, 1.0, 5.0, 10.0, 20.0)
KAPPAS = (0.6, 0.7, 0.8, 0.9)

EXPECTED_PHI = {
    'puzzle-3x3-play-v0': (55, 9),
    'puzzle-4x4-play-v0': (83, 16),
    'puzzle-4x6-play-v0': (115, 24),
}


def _num_token(x: float) -> str:
    return str(x).replace('.', 'p').replace('-', 'm')


def _assert_dynamics_key():
    from agents.dynamics import get_dynamics_config

    cfg = get_dynamics_config()
    if 'forward_bridge_path_loss_horizon' not in cfg:
        raise RuntimeError(
            'agents.dynamics.get_dynamics_config() has no key forward_bridge_path_loss_horizon; '
            'refusing to generate configs silently.'
        )


def _load_baseline() -> dict:
    path = REPO_ROOT / BASELINE_REL
    if not path.is_file():
        raise FileNotFoundError(f'Baseline YAML not found: {path}')
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _deep_merge_baseline(base: dict) -> dict:
    """Shallow copy top-level + nested dicts we override."""
    out = dict(base)
    dyn = dict(base.get('dynamics') or {})
    crit = dict(base.get('critic_agent') or {})
    act = dict(base.get('actor') or {})
    out['dynamics'] = dyn
    out['critic_agent'] = crit
    out['actor'] = act
    return out


def _phi_sanity(env_name: str) -> tuple[bool, str]:
    try:
        os.environ.setdefault('MUJOCO_GL', 'egl')
        from utils.goal_representation import manip_button_state_indices

        import ogbench

        out = ogbench.make_env_and_datasets(str(env_name), compact_dataset=True, env_only=True)
        env = out[0] if isinstance(out, tuple) else out
        while hasattr(env, 'env'):
            env = env.env
        obs_dim = int(env.observation_space.shape[0])
        exp_obs, exp_phi = EXPECTED_PHI[str(env_name)]
        if obs_dim != exp_obs:
            return False, f'{env_name}: obs_dim={obs_dim} expected {exp_obs}'
        idx = manip_button_state_indices(obs_dim)
        if len(idx) != exp_phi:
            return False, f'{env_name}: phi len={len(idx)} expected {exp_phi}'
        return True, f'{env_name}: obs_dim={obs_dim} phi_dim={len(idx)} ok'
    except Exception as e:
        return False, f'{env_name}: phi sanity skipped/failed: {e!r}'


def _validate_discount(path: Path, env_name: str, forced_4995: float | None) -> tuple[bool, str]:
    with open(path, encoding='utf-8') as f:
        root = yaml.safe_load(f) or {}
    disc = (root.get('critic_agent') or {}).get('discount', None)
    if disc is None:
        return False, f'{path.name}: critic_agent.discount missing'
    d = float(disc)
    if forced_4995 is not None:
        ok = abs(d - float(forced_4995)) < 1e-9
        return ok, f'{path.name}: discount={d} (required {forced_4995})'
    # Baseline puzzle table uses 0.99; do not silently force 0.995.
    if abs(d - 0.995) < 1e-9 and '4x6' not in env_name:
        return (
            False,
            f'{path.name}: discount is 0.995 but env {env_name} must keep baseline (not 0.995).',
        )
    return True, f'{path.name}: discount={d} (baseline path)'


def main() -> int:
    _assert_dynamics_key()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = _load_baseline()

    written = 0
    discount_reports: list[tuple[str, str]] = []
    phi_reports: list[tuple[str, str]] = []

    for env_name, fprefix, disc_override in ENVS:
        ok_p, msg_p = _phi_sanity(env_name)
        phi_reports.append(('OK' if ok_p else 'WARN', msg_p))

        for alpha, gap, kappa in itertools.product(ALPHAS, GAPS, KAPPAS):
            stem = f'{fprefix}_a{_num_token(alpha)}_gap{_num_token(gap)}_k{_num_token(kappa)}.yaml'
            path = OUT_DIR / stem

            cfg = _deep_merge_baseline(base)
            cfg['env_name'] = env_name
            cfg['run_group'] = f'grid_fbr_disp_{fprefix}_a{_num_token(alpha)}_gap{_num_token(gap)}_k{_num_token(kappa)}'
            cfg['train_epochs'] = 400
            cfg['batch_size'] = 1024
            cfg['horizon'] = 25
            cfg['plan_candidates'] = 1
            cfg['eval_freq'] = 100
            cfg['save_every_n_epochs'] = 100
            cfg['log_every_n_epochs'] = 10

            d = cfg['dynamics']
            d['subgoal_distribution'] = 'diag_gaussian'
            d['subgoal_stochastic_loss'] = 'nll'
            d['subgoal_goal_representation'] = 'phi'
            d['subgoal_target_mode'] = 'displacement'
            d['planner_type'] = 'forward_bridge_residual'
            d['forward_bridge_path_loss_horizon'] = 5
            d['subgoal_value_alpha'] = float(alpha)
            d['subgoal_value_gap_scale'] = float(gap)

            c = cfg['critic_agent']
            c['goal_representation'] = 'phi'
            c['kappa_b'] = float(kappa)
            c['kappa_d'] = float(kappa)
            if disc_override is not None:
                c['discount'] = float(disc_override)

            with open(path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
            written += 1

            ok_d, msg_d = _validate_discount(path, env_name, disc_override)
            discount_reports.append(('OK' if ok_d else 'FAIL', msg_d))
            if not ok_d:
                print(f'ERROR: {msg_d}', file=sys.stderr)

    # Post-validate all 4x6 discounts
    for p in sorted(OUT_DIR.glob('puzzle_4x6_*.yaml')):
        with open(p, encoding='utf-8') as f:
            root = yaml.safe_load(f) or {}
        d = float((root.get('critic_agent') or {})['discount'])
        if abs(d - 0.995) > 1e-9:
            print(f'ERROR: {p.name} must have critic_agent.discount==0.995, got {d}', file=sys.stderr)
            return 2

    print(f'Wrote {written} configs under {OUT_DIR.relative_to(REPO_ROOT)}')
    print('--- discount validation ---')
    for st, m in discount_reports:
        print(f'  [{st}] {m}')
    print('--- puzzle phi sanity (env_name-based extraction; manip_button_state_indices) ---')
    for st, m in phi_reports:
        print(f'  [{st}] {m}')
    ex = sorted(OUT_DIR.glob('puzzle_*.yaml'))[:5]
    print('--- example paths ---')
    for e in ex:
        print(' ', e.relative_to(REPO_ROOT))
    if any(s != 'OK' for s, _ in discount_reports):
        return 2
    if any(s != 'OK' for s, _ in phi_reports):
        print('WARNING: phi sanity had issues; configs still written (check env/MuJoCo).', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
