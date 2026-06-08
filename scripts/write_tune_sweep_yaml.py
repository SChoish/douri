#!/usr/bin/env python3
"""Emit TRL tune-sweep YAMLs for gap1 / Set A (v2) / Set B (gw_b).

Usage:
  python scripts/write_tune_sweep_yaml.py --set gap1
  python scripts/write_tune_sweep_yaml.py --set v2 --probe
  python scripts/write_tune_sweep_yaml.py --set gw_b
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from tune_sweep_common import (
    ENV_SPECS,
    GAP1_VARIANTS,
    SET_A_VARIANTS,
    SET_B_VARIANTS,
    build_tune_config,
)
from yaml_run_config import dump_run_config_yaml

REPO = Path(__file__).resolve().parent.parent

SWEEP_SETS: dict[str, dict[str, Any]] = {
    'gap1': {
        'variants': GAP1_VARIANTS,
        'outdir': REPO / 'config' / 'sweep_tune_gap1',
        'run_group_prefix': 'tune_g1_',
        'set_label': 'gap1',
        'header_title': 'gap1 sweep: TRL + diag_gaussian',
        'manifest_note': '5 envs × 4 gap1 variants',
    },
    'v2': {
        'variants': SET_A_VARIANTS,
        'outdir': REPO / 'config' / 'sweep_tune_v2',
        'run_group_prefix': 'tune_gw_',
        'set_label': 'set_a',
        'header_title': 'Set A tune_gw: TRL + diag_gaussian',
        'manifest_note': '5 envs × 4 Set-A variants',
    },
    'gw_b': {
        'variants': SET_B_VARIANTS,
        'outdir': REPO / 'config' / 'sweep_tune_gw_b',
        'run_group_prefix': 'tune_gwb_',
        'set_label': 'set_b',
        'header_title': 'Set B tune_gwb: TRL + diag_gaussian',
        'manifest_note': '5 envs × 4 Set-B variants',
    },
}


def _probe_env(env_name: str) -> tuple[bool, str]:
    try:
        os.environ.setdefault('MUJOCO_GL', 'egl')
        import ogbench

        out = ogbench.make_env_and_datasets(str(env_name), compact_dataset=True, env_only=True)
        env = out[0] if isinstance(out, tuple) else out
        while hasattr(env, 'env'):
            env = env.env
        obs_dim = int(env.observation_space.shape[0])
        return True, f'{env_name}: obs_dim={obs_dim} ok'
    except Exception as e:
        return False, f'{env_name}: probe failed: {e!r}'


def _probe_all_envs() -> None:
    seen: set[str] = set()
    for spec in ENV_SPECS.values():
        name = str(spec['env_name'])
        if name in seen:
            continue
        seen.add(name)
        ok, msg = _probe_env(name)
        print(('OK' if ok else 'WARN'), msg)


def write_sweep_set(set_name: str, outdir: Path | None = None) -> list[str]:
    spec = SWEEP_SETS[set_name]
    out = outdir or Path(spec['outdir'])
    out.mkdir(parents=True, exist_ok=True)
    manifest: list[str] = []
    for env_prefix in ENV_SPECS:
        for variant_suffix, variant in spec['variants'].items():
            cfg = build_tune_config(
                env_prefix=env_prefix,
                variant_suffix=variant_suffix,
                variant=variant,
                run_group_prefix=str(spec['run_group_prefix']),
                set_label=str(spec['set_label']),
            )
            fname = f'{env_prefix}_{variant_suffix}.yaml'
            out_path = out / fname
            header = (
                f'# {spec["header_title"]}\n'
                f'# env={cfg["env_name"]} gap={variant["gap"]} wmax={variant["wmax"]} '
                f'gamma={cfg["critic_agent"]["discount"]}\n'
            )
            dump_run_config_yaml(out_path, cfg, header=header)
            manifest.append(str(out_path.relative_to(REPO)))
            print(out_path)

    manifest_path = out / '_manifest.txt'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write(f'# {len(manifest)} configs ({spec["manifest_note"]})\n')
        for line in manifest:
            f.write(line + '\n')
    print(f'Manifest: {manifest_path}')
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description='Generate TRL tune-sweep YAML configs.')
    p.add_argument('--set', choices=tuple(SWEEP_SETS), required=True, help='gap1, v2 (Set A), or gw_b (Set B)')
    p.add_argument('--probe', action='store_true', help='Probe env obs dims only (no YAML write).')
    p.add_argument('--outdir', type=Path, default=None, help='Override output directory.')
    args = p.parse_args()

    if args.probe:
        _probe_all_envs()
        return
    write_sweep_set(args.set, outdir=args.outdir)


if __name__ == '__main__':
    main()
