#!/usr/bin/env python3
"""Emit one antmaze YAML for the residual × subgoal target-mode grid sweep.

Baseline hparams are taken from ``../douri/runs/<ref_run>_seed0_<env>/config_used.yaml``
(top-2 @ epoch 600 leaderboard runs). Only the grid axes and run metadata change:

  - ``residual_target_mode``: absolute | displacement
  - ``subgoal_target_mode``: absolute | displacement

Keys present in douri but not in Pathbridger (e.g. ``residual_envelope``) are dropped.

Usage:
  python scripts/write_antmaze_res_subgoal_grid_yaml.py \\
    --scale medium --baseline m1 --residual displacement --subgoal absolute \\
    --time-embedding on --state-normalization off \\
    --out scripts/sweep_generated/antmaze_res_subgoal_grid_600ep/antmaze_medium_m1_rd_sa.yaml
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
    douri_run_config_path,
    load_yaml,
    sanitize_dynamics,
)

_DEFAULT_DOURI = _REPO.parent / 'douri'

# Top-2 @600 (IDM/Actor leaderboard); config from douri run ``config_used.yaml``.
BASELINES: dict[str, dict[str, str]] = {
    'g1': {'ref_run': '20260518_155648', 'scale': 'giant'},
    'g2': {'ref_run': '20260517_234529', 'scale': 'giant'},
    'l1': {'ref_run': '20260520_073104', 'scale': 'large'},
    'l2': {'ref_run': '20260520_221741', 'scale': 'large'},
    'm1': {'ref_run': '20260521_144335', 'scale': 'medium'},
    'm2': {'ref_run': '20260521_133620', 'scale': 'medium'},
}

SCALES: dict[str, dict] = {
    'giant': {'env_name': 'antmaze-giant-navigate-v0', 'baselines': ('g1', 'g2')},
    'large': {'env_name': 'antmaze-large-navigate-v0', 'baselines': ('l1', 'l2')},
    'medium': {'env_name': 'antmaze-medium-navigate-v0', 'baselines': ('m1', 'm2')},
}

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
    bl = BASELINES[baseline]
    if bl['scale'] != scale:
        raise ValueError(f'baseline {baseline!r} belongs to scale {bl["scale"]!r}, not {scale!r}')

    env_name = str(meta['env_name'])
    ref_run = str(bl['ref_run'])
    cfg_path = douri_run_config_path(douri_root, ref_run, env_name)
    if not cfg_path.is_file():
        raise FileNotFoundError(
            f'douri baseline config not found: {cfg_path}\n'
            f'Set --douri-root or clone runs next to Pathbridger.'
        )

    root = load_yaml(cfg_path)
    root['env_name'] = env_name
    res_short = RES_SHORT[residual_mode]
    sg_short = SG_SHORT[subgoal_mode]
    te_short = 'te1' if use_time_embedding else 'te0'
    sn_short = 'sn1' if state_normalization else 'sn0'
    ct_short = str(critic_type).lower()
    tag = f'{scale}_{baseline}_{res_short}_{sg_short}_{te_short}_{sn_short}'
    if ct_short != 'dqc':
        tag = f'{tag}_{ct_short}'
    root = apply_grid_modes(
        root,
        residual_mode=residual_mode,
        subgoal_mode=subgoal_mode,
        run_group_prefix='sweep600_res_sg',
        tag=tag,
        train_epochs=train_epochs,
        batch_size=1024,
    )
    dyn = root.setdefault('dynamics', {})
    dyn['use_time_embedding'] = bool(use_time_embedding)
    dyn['state_normalization'] = bool(state_normalization)
    cri = root.setdefault('critic_agent', {})
    cri['critic_type'] = ct_short
    if ct_short in ('trl', 'chunk_trl', 'direct_chunk_trl'):
        cri['algorithm'] = 'direct_chunk_trl'
    if ct_short in ('iql', 'trl', 'chunk_trl', 'direct_chunk_trl'):
        cri['use_chunk_critic'] = False

    sweep_meta = {
        'tag': tag,
        'scale': scale,
        'baseline': baseline,
        'ref_run': ref_run,
        'douri_config': str(cfg_path),
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
    p.add_argument('--baseline', required=True, help='g1/g2, l1/l2, or m1/m2')
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
            '# Auto-generated by scripts/write_antmaze_res_subgoal_grid_yaml.py\n'
            f'# douri baseline: {meta["douri_config"]}\n'
            f'# ref_run={meta["ref_run"]} residual={args.residual} subgoal={args.subgoal}\n'
            f'# (from douri) alpha={dyn.get("subgoal_value_alpha")} gap={dyn.get("subgoal_value_gap_scale")} '
            f'kappa_b={cri.get("kappa_b")} kappa_d={cri.get("kappa_d")} '
            f'subgoal_dist={dyn.get("subgoal_distribution")} '
            f'time_embedding={bool(args.time_embedding)} state_normalization={bool(args.state_normalization)} '
            f'critic_type={args.critic_type}\n'
        )
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
    print(f'wrote {args.out} tag={meta["tag"]}')


if __name__ == '__main__':
    main()
