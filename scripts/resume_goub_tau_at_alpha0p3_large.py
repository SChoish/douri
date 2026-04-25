#!/usr/bin/env python3
"""Resume alpha=0.3 / tau in {5,10,20} large-navigate sweep: checkpoint resume + fresh tau if missing.

Uses ``main.py`` flags ``--resume_run_dir`` and ``--resume_epoch`` (omit ``--run_config`` so
``flags.json`` snapshot is used). For a tau with no matching run, starts a new run from the
sweep YAML.

See also: ``scripts/launch_goub_tau_at_alpha0p3_large.sh`` (cold start). Heatmaps: same
``scripts/plot_goub_tau_alpha_sweep_heatmaps.py`` with ``--taus 5,10 --alphas 0.3`` and
``--run-group antmaze_navigate_goub_tau_at_alpha0p3_sweep``.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RUN_GROUP = "antmaze_navigate_goub_tau_at_alpha0p3_sweep"
ENV_GLOB = "*_joint_dqc_seed0_antmaze-large-navigate-v0"
TRAIN_LOG_EPOCH = re.compile(r"\|\s*INFO\s*\|\s*epoch=(\d+)\s+goub=")


def _load_cfg(run_dir: Path) -> dict | None:
    p = run_dir / "config_used.yaml"
    if not p.is_file():
        return None
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _latest_ckpt_epoch(ckpt_goub: Path) -> int:
    if not ckpt_goub.is_dir():
        return 0
    best = 0
    for child in ckpt_goub.glob("params_*.pkl"):
        stem = child.stem  # params_300
        if not stem.startswith("params_"):
            continue
        try:
            ep = int(stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        best = max(best, ep)
    return best


def _max_training_epoch_from_logs(run_dir: Path) -> int:
    best = 0
    for logf in sorted(run_dir.glob("run*.log")):
        try:
            text = logf.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for m in TRAIN_LOG_EPOCH.finditer(text):
            best = max(best, int(m.group(1)))
    return best


def _min_ckpt_epoch_across_agents(run_dir: Path) -> int:
    """Use the minimum latest epoch so goub/critic/actor all have params_<ep>.pkl."""
    subs = ("goub", "critic", "actor")
    epochs = []
    for sub in subs:
        d = run_dir / "checkpoints" / sub
        ep = _latest_ckpt_epoch(d)
        if ep == 0:
            return 0
        epochs.append(ep)
    return min(epochs)


def _collect_runs_for_tau(runs_root: Path, tau: float) -> list[Path]:
    out: list[Path] = []
    for run_dir in sorted(runs_root.glob(ENV_GLOB)):
        if not run_dir.is_dir():
            continue
        cfg = _load_cfg(run_dir)
        if not cfg or cfg.get("run_group") != RUN_GROUP:
            continue
        actor = cfg.get("actor") or {}
        try:
            t = float(actor.get("spi_tau", -1.0))
        except (TypeError, ValueError):
            continue
        if abs(t - tau) > 1e-6:
            continue
        out.append(run_dir)
    return out


def _pick_run_to_resume(candidates: list[Path]) -> Path | None:
    if not candidates:
        return None
    with_ckpt = [r for r in candidates if _min_ckpt_epoch_across_agents(r) > 0]
    pool = with_ckpt if with_ckpt else candidates
    return max(
        pool,
        key=lambda rd: (
            _min_ckpt_epoch_across_agents(rd),
            _max_training_epoch_from_logs(rd),
            rd.name,
        ),
    )


def _plan_tau(
    runs_root: Path,
    tau: float,
    cfg_dir: Path,
    train_epochs_default: int = 400,
) -> tuple[str, Path | None, int, Path | None]:
    """Returns (action, run_dir_or_none, resume_epoch, yaml_for_fresh)."""
    tau_tag = str(tau).replace(".", "p", 1)
    yaml_path = cfg_dir / f"antmaze_large_navigate_goub_alpha0p3_tau{tau_tag}.yaml"
    cands = _collect_runs_for_tau(runs_root, tau)
    chosen = _pick_run_to_resume(cands)
    if chosen is None:
        return ("fresh", None, 0, yaml_path)

    cfg = _load_cfg(chosen) or {}
    train_epochs = int(cfg.get("train_epochs", train_epochs_default))
    ck = _min_ckpt_epoch_across_agents(chosen)
    log_max = _max_training_epoch_from_logs(chosen)

    if log_max >= train_epochs:
        return ("skip", chosen, ck, yaml_path)

    if ck <= 0:
        return ("fresh", None, 0, yaml_path)

    return ("resume", chosen, ck, yaml_path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--python",
        default="",
        help="Python binary (default: $PYTHON, else miniconda offrl, else sys.executable)",
    )
    p.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    p.add_argument("--config-dir", type=Path, default=ROOT / "config" / "sweep_goub_tau_at_alpha0p3")
    p.add_argument(
        "--taus",
        default="5.0,10.0,20.0",
        help="Comma-separated spi_tau values (default: 5.0,10.0,20.0)",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    py = (args.python or "").strip() or os.environ.get("PYTHON", "").strip()
    if not py:
        cand = Path("/home/choi/miniconda3/envs/offrl/bin/python")
        py = str(cand) if cand.is_file() else sys.executable

    runs_root: Path = args.runs_root
    cfg_dir: Path = args.config_dir
    taus = [float(x.strip()) for x in str(args.taus).split(",") if x.strip()]

    main_py = ROOT / "main.py"
    if not main_py.is_file():
        raise SystemExit(f"missing {main_py}")

    for tau in taus:
        action, run_dir, resume_ep, yaml_path = _plan_tau(runs_root, tau, cfg_dir)
        print(f"[tau={tau}] action={action} run_dir={run_dir} resume_epoch={resume_ep}")

        if action == "skip":
            print(f"  -> skip (already reached train_epochs or no work)")
            continue

        if action == "fresh":
            if not yaml_path.is_file():
                raise SystemExit(f"missing config {yaml_path}")
            cmd = [py, str(main_py), f"--run_config={yaml_path}"]
            print(f"  -> {' '.join(cmd)}")
            if not args.dry_run:
                subprocess.run(cmd, cwd=str(ROOT), check=True)
            continue

        if action == "resume":
            assert run_dir is not None and resume_ep > 0
            cmd = [
                py,
                str(main_py),
                f"--resume_run_dir={run_dir.resolve()}",
                f"--resume_epoch={resume_ep}",
            ]
            print(f"  -> {' '.join(cmd)}")
            if not args.dry_run:
                subprocess.run(cmd, cwd=str(ROOT), check=True)
            continue

        raise RuntimeError(f"unknown action {action}")


if __name__ == "__main__":
    main()
