"""Shared helpers for residual × subgoal target-mode grid sweeps."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

RES_SHORT = {'absolute': 'ra', 'displacement': 'rd'}
SG_SHORT = {'absolute': 'sa', 'displacement': 'sd'}

# douri-only dynamics keys (not in Pathbridger ``get_dynamics_config``).
DOURI_DROP_DYNAMICS = frozenset({'residual_envelope'})


def load_yaml(path: Path) -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)  # type: ignore[arg-type]
        else:
            out[key] = value
    return out


def sanitize_dynamics(dyn: dict) -> dict:
    return {k: v for k, v in dyn.items() if k not in DOURI_DROP_DYNAMICS}


def apply_grid_modes(
    root: dict,
    *,
    residual_mode: str,
    subgoal_mode: str,
    run_group_prefix: str,
    tag: str,
    train_epochs: int,
    batch_size: int | None = None,
) -> dict:
    root = copy.deepcopy(root)
    root['train_epochs'] = int(train_epochs)
    root.setdefault('eval_freq', 100)
    root.setdefault('log_every_n_epochs', 10)
    root.setdefault('save_every_n_epochs', 100)
    root.setdefault('horizon', 25)
    root.setdefault('plan_candidates', 1)
    if batch_size is not None:
        root['batch_size'] = int(batch_size)
    root.setdefault('batch_size', 1024)

    dyn = sanitize_dynamics(dict(root.get('dynamics') or {}))
    dyn['residual_target_mode'] = residual_mode
    dyn['subgoal_target_mode'] = subgoal_mode
    root['dynamics'] = dyn
    root['run_group'] = f'{run_group_prefix}_{tag}'
    return root


def douri_run_config_path(douri_root: Path, ref_run: str, env_name: str) -> Path:
    return douri_root / 'runs' / f'{ref_run}_seed0_{env_name}' / 'config_used.yaml'
