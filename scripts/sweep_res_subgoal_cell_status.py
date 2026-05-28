#!/usr/bin/env python3
"""Resolve skip / resume / fresh-run for one residual×subgoal sweep cell.

Prints key=value lines for bash (see sweep_antmaze_res_subgoal_grid_600ep.sh).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

EPOCH_TRAIN = re.compile(r"\| INFO \| epoch=(\d+) ")
EPOCH_EVAL_END = re.compile(r"=== EVAL END epoch=(\d+) ===")
DONE_LINE = re.compile(r"done run_dir=")
CKPT_EPOCH = re.compile(r"params_(\d+)\.pkl$")


def _read_run_group(flags_path: Path) -> str | None:
    try:
        data = json.loads(flags_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return (data.get("flags") or {}).get("run_group")


def _find_run_dirs(runs_root: Path, run_group: str) -> list[Path]:
    if not runs_root.is_dir():
        return []
    out: list[Path] = []
    for child in runs_root.iterdir():
        if not child.is_dir():
            continue
        flags = child / "flags.json"
        if not flags.is_file():
            continue
        if _read_run_group(flags) == run_group:
            out.append(child)
    return out


def _max_ckpt_epoch(run_dir: Path) -> int:
    best = 0
    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.is_dir():
        return 0
    for agent_dir in ckpt_root.iterdir():
        if not agent_dir.is_dir():
            continue
        for p in agent_dir.glob("params_*.pkl"):
            m = CKPT_EPOCH.search(p.name)
            if m:
                best = max(best, int(m.group(1)))
    return best


def _max_logged_epoch(run_dir: Path) -> int:
    best = 0
    for log_name in ("run.log",):
        log_path = run_dir / log_name
        if not log_path.is_file():
            continue
        text = log_path.read_text(encoding="utf-8", errors="replace")
        for pat in (EPOCH_TRAIN, EPOCH_EVAL_END):
            for m in pat.finditer(text):
                best = max(best, int(m.group(1)))
        for extra in sorted(run_dir.glob("run_resume_from*.log")):
            t = extra.read_text(encoding="utf-8", errors="replace")
            for pat in (EPOCH_TRAIN, EPOCH_EVAL_END):
                for m in pat.finditer(t):
                    best = max(best, int(m.group(1)))
    return best


def _is_done(run_dir: Path, train_epochs: int) -> bool:
    ckpt = run_dir / "checkpoints" / "dynamics" / f"params_{train_epochs}.pkl"
    if not ckpt.is_file():
        return False
    for log_name in ("run.log",):
        lp = run_dir / log_name
        if lp.is_file() and DONE_LINE.search(lp.read_text(encoding="utf-8", errors="replace")):
            return True
    for extra in sorted(run_dir.glob("run_resume_from*.log")):
        if DONE_LINE.search(extra.read_text(encoding="utf-8", errors="replace")):
            return True
    return False


def resolve_cell(
    *,
    runs_root: Path,
    run_group: str,
    train_epochs: int,
) -> dict[str, str | int]:
    cands = _find_run_dirs(runs_root, run_group)
    if not cands:
        return {
            "action": "run",
            "run_dir": "",
            "resume_epoch": 0,
            "last_epoch": 0,
            "reason": "no_existing_run",
        }

    # Prefer run with highest checkpoint, then newest mtime.
    def sort_key(p: Path) -> tuple[int, float]:
        return (_max_ckpt_epoch(p), p.stat().st_mtime)

    run_dir = max(cands, key=sort_key)
    if _is_done(run_dir, train_epochs):
        return {
            "action": "skip",
            "run_dir": str(run_dir.resolve()),
            "resume_epoch": train_epochs,
            "last_epoch": train_epochs,
            "reason": "completed",
        }

    ckpt_ep = _max_ckpt_epoch(run_dir)
    log_ep = _max_logged_epoch(run_dir)
    last_ep = max(ckpt_ep, log_ep)

    if ckpt_ep > 0 and ckpt_ep < train_epochs:
        return {
            "action": "resume",
            "run_dir": str(run_dir.resolve()),
            "resume_epoch": ckpt_ep,
            "last_epoch": last_ep,
            "reason": f"checkpoint@{ckpt_ep}",
        }

    if last_ep > 0 and last_ep < train_epochs and ckpt_ep == 0:
        return {
            "action": "resume",
            "run_dir": str(run_dir.resolve()),
            "resume_epoch": 0,
            "last_epoch": last_ep,
            "reason": "partial_no_ckpt_reuse_dir",
        }

    if last_ep >= train_epochs and not _is_done(run_dir, train_epochs):
        return {
            "action": "resume",
            "run_dir": str(run_dir.resolve()),
            "resume_epoch": ckpt_ep if ckpt_ep > 0 else 0,
            "last_epoch": last_ep,
            "reason": "past_target_epoch_rerun_finalize",
        }

    return {
        "action": "run",
        "run_dir": str(run_dir.resolve()),
        "resume_epoch": 0,
        "last_epoch": last_ep,
        "reason": "stale_dir_start_fresh" if cands else "no_existing_run",
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--run-group", required=True)
    p.add_argument("--train-epochs", type=int, default=600)
    args = p.parse_args()

    info = resolve_cell(
        runs_root=args.runs_root.resolve(),
        run_group=args.run_group,
        train_epochs=int(args.train_epochs),
    )
    for k, v in info.items():
        print(f"{k}={v}")


if __name__ == "__main__":
    main()
