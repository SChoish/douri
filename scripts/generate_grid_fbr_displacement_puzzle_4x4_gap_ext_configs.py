#!/usr/bin/env python3
"""Generate puzzle 4x4 alpha=0 configs for extended gap scales (50, 100).

Gap 0 configs already exist from the main grid generator.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / 'config' / 'grid_fbr_displacement_puzzle'
TEMPLATE = OUT_DIR / 'puzzle_4x4_a0p0_gap20p0_k0p6.yaml'

ENVS = ('puzzle-4x4-play-v0',)
ALPHA = 0.0
GAPS = (50.0, 100.0)
KAPPAS = (0.6, 0.7, 0.8, 0.9)


def _num_token(x: float) -> str:
    return str(x).replace('.', 'p').replace('-', 'm')


def main() -> int:
    if not TEMPLATE.is_file():
        print(f'Template missing: {TEMPLATE}', file=sys.stderr)
        return 2
    with open(TEMPLATE, encoding='utf-8') as f:
        base = yaml.safe_load(f) or {}

    written = 0
    for gap in GAPS:
        for kappa in KAPPAS:
            stem = (
                f'puzzle_4x4_a{_num_token(ALPHA)}_gap{_num_token(gap)}_k{_num_token(kappa)}.yaml'
            )
            path = OUT_DIR / stem
            cfg = yaml.safe_load(yaml.safe_dump(base)) or {}
            cfg['env_name'] = ENVS[0]
            cfg['run_group'] = (
                f'grid_fbr_disp_puzzle_4x4_a{_num_token(ALPHA)}_gap{_num_token(gap)}_k{_num_token(kappa)}'
            )
            cfg['dynamics']['subgoal_value_alpha'] = float(ALPHA)
            cfg['dynamics']['subgoal_value_gap_scale'] = float(gap)
            cfg['critic_agent']['kappa_b'] = float(kappa)
            cfg['critic_agent']['kappa_d'] = float(kappa)
            with open(path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
            written += 1
            print(path.relative_to(REPO_ROOT))
    print(f'Wrote {written} configs under {OUT_DIR.relative_to(REPO_ROOT)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
