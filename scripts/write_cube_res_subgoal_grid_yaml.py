#!/usr/bin/env python3
"""Emit one cube-play YAML for residual × subgoal target-mode grid sweep.

Baseline hparams (gap, κ) follow the manip leaderboard table. The template merges:

  - ``../douri/config/cube_double_play_baseline.yaml`` (manip: U=4, α=0.3, phi, …)
  - ``../douri/config/grid_fbr_displacement_antmaze/antmaze_medium_a0p0_gap5p0_k0p7.yaml``
    (Pathbridger FBR + prefix_progress fields; ``subgoal_num_samples`` stays 4)

Only ``residual_target_mode``, ``subgoal_target_mode``, gap, κ, env, and run metadata change.

Usage:
  python scripts/write_cube_res_subgoal_grid_yaml.py \\
    --scale triple --baseline t1 --residual displacement --subgoal absolute \\
    --time-embedding on --state-normalization off \\
    --out scripts/sweep_generated/cube_res_subgoal_grid_600ep/cube_triple_t1_rd_sa.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[1]
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from sweep_res_subgoal_grid_lib import (  # noqa: E402
    RES_SHORT,
    SG_SHORT,
    apply_grid_modes,
    deep_merge,
    load_yaml,
    sanitize_dynamics,
)

_DEFAULT_DOURI = _REPO.parent / 'douri'

# Top-2 gap·κ per env (leaderboard); α=0.3 from cube_double_play_baseline.
BASELINES: dict[str, dict] = {
    't1': {'scale': 'triple', 'gap': 5.0, 'kappa': 0.8},
    't2': {'scale': 'triple', 'gap': 5.0, 'kappa': 0.6},
    'd1': {'scale': 'double', 'gap': 5.0, 'kappa': 0.6},
    'd2': {'scale': 'double', 'gap': 5.0, 'kappa': 0.7},
    's1': {'scale': 'single', 'gap': 1.0, 'kappa': 0.9},
    's2': {'scale': 'single', 'gap': 5.0, 'kappa': 0.9},
}

SCALES: dict[str, dict] = {
    'single': {
        'env_name': 'cube-single-play-v0',
        'baselines': ('s1', 's2'),
        'batch_size': 1024,
    },
    'double': {
        'env_name': 'cube-double-play-v0',
        'baselines': ('d1', 'd2'),
        'batch_size': 1024,
    },
    'triple': {
        'env_name': 'cube-triple-play-v0',
        'baselines': ('t1', 't2'),
        'batch_size': 4096,
    },
}

_CUBE_TEMPLATE_REL = Path('config/cube_double_play_baseline.yaml')
_FBR_TEMPLATE_REL = Path(
    'config/grid_fbr_displacement_antmaze/antmaze_medium_a0p0_gap5p0_k0p7.yaml'
)


def _load_cube_template(douri_root: Path) -> dict:
    cube_path = douri_root / _CUBE_TEMPLATE_REL
    fbr_path = douri_root / _FBR_TEMPLATE_REL
    if not cube_path.is_file():
        raise FileNotFoundError(f'douri cube template not found: {cube_path}')
    if not fbr_path.is_file():
        raise FileNotFoundError(f'douri FBR grid template not found: {fbr_path}')

    cube = load_yaml(cube_path)
    fbr = load_yaml(fbr_path)

    root: dict = {
        k: v
        for k, v in cube.items()
        if k not in ('dynamics', 'critic_agent', 'actor', 'env_name', 'run_group', 'train_epochs')
    }
    for k in ('log_every_n_epochs', 'save_every_n_epochs', 'horizon', 'plan_candidates', 'eval_freq'):
        if k in fbr:
            root[k] = fbr[k]

    cube_dyn = dict(cube.get('dynamics') or {})
    fbr_dyn = sanitize_dynamics(dict(fbr.get('dynamics') or {}))
    dyn = deep_merge(cube_dyn, fbr_dyn)
    # Keep manip cohort scalars from cube baseline (FBR antmaze grid uses α=0, U=1).
    dyn['subgoal_value_alpha'] = float(cube_dyn.get('subgoal_value_alpha', 0.3))
    dyn['subgoal_num_samples'] = int(cube_dyn.get('subgoal_num_samples', 4))
    root['dynamics'] = dyn

    root['critic_agent'] = dict(cube.get('critic_agent') or {})
    root['actor'] = dict(cube.get('actor') or fbr.get('actor') or {})
    return root


def build_config(
    *,
    douri_root: Path,
    scale: str,
    baseline: str,
    residual_mode: str,
    subgoal_mode: str,
    use_time_embedding: bool,
    state_normalization: bool,
    critic_type: str,
    train_epochs: int,
) -> tuple[dict, dict]:
    if scale not in SCALES:
        raise ValueError(f'unknown scale {scale!r}, expected one of {sorted(SCALES)}')
    meta = SCALES[scale]
    if baseline not in meta['baselines']:
        raise ValueError(
            f'baseline {baseline!r} invalid for {scale!r}; expected one of {meta["baselines"]}'
        )
    hp = BASELINES[baseline]
    if hp['scale'] != scale:
        raise ValueError(f'baseline {baseline!r} belongs to scale {hp["scale"]!r}, not {scale!r}')

    env_name = str(meta['env_name'])
    root = _load_cube_template(douri_root)
    root['env_name'] = env_name

    dyn = dict(root['dynamics'])
    dyn['subgoal_value_gap_scale'] = float(hp['gap'])
    root['dynamics'] = dyn

    cri = dict(root['critic_agent'])
    cri['kappa_b'] = float(hp['kappa'])
    cri['kappa_d'] = float(hp['kappa'])
    cri['discount'] = 0.99
    ct_short = str(critic_type).lower()
    cri['critic_type'] = ct_short
    if ct_short in ('trl', 'chunk_trl', 'direct_chunk_trl'):
        cri['algorithm'] = 'direct_chunk_trl'
    if ct_short in ('iql', 'trl', 'chunk_trl', 'direct_chunk_trl'):
        cri['use_chunk_critic'] = False
    root['critic_agent'] = cri

    res_short = RES_SHORT[residual_mode]
    sg_short = SG_SHORT[subgoal_mode]
    te_short = 'te1' if use_time_embedding else 'te0'
    sn_short = 'sn1' if state_normalization else 'sn0'
    tag = f'{scale}_{baseline}_{res_short}_{sg_short}_{te_short}_{sn_short}'
    if ct_short != 'dqc':
        tag = f'{tag}_{ct_short}'
    root = apply_grid_modes(
        root,
        residual_mode=residual_mode,
        subgoal_mode=subgoal_mode,
        run_group_prefix='sweep600_cube_res_sg',
        tag=tag,
        train_epochs=train_epochs,
        batch_size=int(meta['batch_size']),
    )
    dyn = root.setdefault('dynamics', {})
    dyn['use_time_embedding'] = bool(use_time_embedding)
    dyn['state_normalization'] = bool(state_normalization)

    sweep_meta = {
        'tag': tag,
        'scale': scale,
        'baseline': baseline,
        'gap': hp['gap'],
        'kappa': hp['kappa'],
        'douri_cube_template': str(douri_root / _CUBE_TEMPLATE_REL),
        'douri_fbr_template': str(douri_root / _FBR_TEMPLATE_REL),
        'residual_target_mode': residual_mode,
        'subgoal_target_mode': subgoal_mode,
        'use_time_embedding': use_time_embedding,
        'state_normalization': state_normalization,
        'critic_type': ct_short,
    }
    return root, sweep_meta


def _on_off(value: str) -> bool:
    value = str(value).strip().lower()
    if value in ('on', 'true', '1', 'yes'):
        return True
    if value in ('off', 'false', '0', 'no'):
        return False
    raise argparse.ArgumentTypeError("expected one of: on/off, true/false, 1/0, yes/no")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--douri-root', type=Path, default=_DEFAULT_DOURI)
    p.add_argument('--scale', choices=tuple(SCALES), required=True)
    p.add_argument('--baseline', required=True, help='t1/t2, d1/d2, or s1/s2')
    p.add_argument('--residual', choices=('absolute', 'displacement'), required=True)
    p.add_argument('--subgoal', choices=('absolute', 'displacement'), required=True)
    p.add_argument('--time-embedding', type=_on_off, default=True)
    p.add_argument('--state-normalization', type=_on_off, default=False)
    p.add_argument('--critic-type', choices=('dqc', 'iql', 'trl', 'chunk_trl', 'direct_chunk_trl'), default='dqc')
    p.add_argument('--train_epochs', type=int, default=600)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()

    cfg, meta = build_config(
        douri_root=args.douri_root.resolve(),
        scale=args.scale,
        baseline=args.baseline,
        residual_mode=args.residual,
        subgoal_mode=args.subgoal,
        use_time_embedding=bool(args.time_embedding),
        state_normalization=bool(args.state_normalization),
        critic_type=str(args.critic_type),
        train_epochs=args.train_epochs,
    )

    dyn = cfg.get('dynamics') or {}
    cri = cfg.get('critic_agent') or {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(
            '# Auto-generated by scripts/write_cube_res_subgoal_grid_yaml.py\n'
            f'# cube template: {meta["douri_cube_template"]}\n'
            f'# FBR template: {meta["douri_fbr_template"]}\n'
            f'# baseline={args.baseline} gap={meta["gap"]} kappa={meta["kappa"]} '
            f'residual={args.residual} subgoal={args.subgoal}\n'
            f'# alpha={dyn.get("subgoal_value_alpha")} subgoal_dist={dyn.get("subgoal_distribution")} '
            f'U={dyn.get("subgoal_num_samples")} batch={cfg.get("batch_size")} '
            f'time_embedding={bool(args.time_embedding)} state_normalization={bool(args.state_normalization)} '
            f'critic_type={args.critic_type}\n'
        )
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    print(f'wrote {args.out} tag={meta["tag"]}')


if __name__ == '__main__':
    main()
