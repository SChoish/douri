#!/usr/bin/env python3
"""AntMaze-large vanilla GOUB τ × α 통합 히트맵.

Composite: **3 rows × 4 cols**
  row 0: IDM policy task-mean success (절대값, viridis 0~1)
  row 1: Actor policy task-mean success (절대값, viridis 0~1)
  row 2: Actor uplift over IDM = (Actor − IDM) / IDM (RdBu_r 중심 0)
  cols : eval epoch ∈ {100, 200, 300, 400}

기본 그리드는 두 sweep run_group을 합쳐 τ∈{0.5,1,5,10,20} × α∈{0,0.1,0.3,0.5,1} (5×5) 한 장.
각 run 디렉터리의 ``run*.log`` (재개 ``run_resume_from*.log`` 포함)를 모두 읽어
``=== EVAL START epoch=… ===`` 블록의 ``idm/actor success_rate_mean``을 채웁니다.

Usage:
  python scripts/plot_goub_tau_alpha_sweep_heatmaps.py
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib import patheffects as pe

EPOCHS = (100, 200, 300, 400)
RUN_GLOB_DEFAULT = "*_joint_dqc_seed0_antmaze-large-navigate-v0"
RUN_GROUPS_DEFAULT = (
    "antmaze_navigate_goub_tau_alpha_sweep",
    "antmaze_navigate_goub_tau_at_alpha0p3_sweep",
)
DEFAULT_TAUS = (0.5, 1.0, 5.0, 10.0, 20.0)
DEFAULT_ALPHAS = (0.0, 0.1, 0.3, 0.5, 1.0)


def parse_eval_means(log_text: str) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    lines = log_text.splitlines()
    i = 0
    while i < len(lines):
        m = re.search(r"=== EVAL START epoch=(\d+)", lines[i])
        if not m:
            i += 1
            continue
        ep = int(m.group(1))
        idm_mean = actor_mean = None
        i += 1
        while i < len(lines):
            if "=== EVAL END" in lines[i]:
                break
            lm = re.search(r"idm success_rate_mean=([\d.]+)", lines[i])
            if lm:
                idm_mean = float(lm.group(1))
            lm = re.search(r"actor success_rate_mean=([\d.]+)", lines[i])
            if lm:
                actor_mean = float(lm.group(1))
            i += 1
        if idm_mean is not None and actor_mean is not None:
            out[ep] = (idm_mean, actor_mean)
        i += 1
    return out


def read_joint_run_logs(run_dir: Path) -> str:
    parts: list[str] = []
    for logf in sorted(run_dir.glob("run*.log")):
        try:
            parts.append(logf.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return "\n".join(parts)


def collect_matrices(
    runs_root: Path,
    run_glob: str,
    run_groups: tuple[str, ...],
    taus: tuple[float, ...],
    alphas: tuple[float, ...],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], list[str]]:
    """두 sweep run_group을 한 (alpha × tau) 그리드에 합쳐 IDM/Actor 절대값 매트릭스를 채움."""
    idm_mats = {ep: np.full((len(alphas), len(taus)), np.nan) for ep in EPOCHS}
    actor_mats = {ep: np.full((len(alphas), len(taus)), np.nan) for ep in EPOCHS}
    notes: list[str] = []
    group_set = set(run_groups)
    tau_set = {float(t) for t in taus}
    alpha_set = {float(a) for a in alphas}

    for d in sorted(runs_root.glob(run_glob)):
        cfg_path = d / "config_used.yaml"
        if not cfg_path.is_file():
            continue
        if not any(d.glob("run*.log")):
            continue
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if cfg.get("run_group") not in group_set:
            continue
        g = cfg.get("goub") or {}
        a = cfg.get("actor") or {}
        try:
            tau = float(a.get("spi_tau"))
            alpha = float(g.get("subgoal_value_alpha"))
        except (TypeError, ValueError):
            notes.append(f"skip {d.name}: missing tau/alpha")
            continue
        if tau not in tau_set or alpha not in alpha_set:
            notes.append(f"skip {d.name}: tau={tau} alpha={alpha}")
            continue
        ti = taus.index(tau)
        ai = alphas.index(alpha)
        means = parse_eval_means(read_joint_run_logs(d))
        for ep in EPOCHS:
            pair = means.get(ep)
            if pair is None:
                continue
            idm_mats[ep][ai, ti] = pair[0]
            actor_mats[ep][ai, ti] = pair[1]

    return idm_mats, actor_mats, notes


def actor_uplift(idm: np.ndarray, actor: np.ndarray) -> np.ndarray:
    """Relative uplift (actor − idm) / idm. idm==0 또는 NaN → NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (actor - idm) / idm
    out = np.where(np.isfinite(out), out, np.nan)
    return out


def plot_composite(
    idm_mats: dict[int, np.ndarray],
    actor_mats: dict[int, np.ndarray],
    taus: tuple[float, ...],
    alphas: tuple[float, ...],
    out_path: Path,
    suptitle_extra: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 4, figsize=(21, 12.0), constrained_layout=True)
    pe_white = [pe.withStroke(linewidth=2, foreground="white")]

    abs_kwargs = {"cmap": "viridis", "vmin": 0.0, "vmax": 1.0}
    abs_fmt = lambda v: "—" if not np.isfinite(v) else f"{v:.2f}"
    pct_fmt = lambda v: "—" if not np.isfinite(v) else f"{v * 100:+.0f}%"

    for col, ep in enumerate(EPOCHS):
        idm = idm_mats[ep]
        actor = actor_mats[ep]
        uplift = actor_uplift(idm, actor)

        rows = (
            ("IDM success", idm, abs_kwargs, abs_fmt),
            ("Actor success", actor, abs_kwargs, abs_fmt),
            (
                "Actor uplift vs IDM  (Actor − IDM)/IDM",
                uplift,
                {"cmap": "RdBu_r", "vmin": -1.0, "vmax": 1.0},
                pct_fmt,
            ),
        )

        last_row = len(rows) - 1
        for row, (name, mat, im_kwargs, fmt) in enumerate(rows):
            ax = axes[row, col]
            Z = np.ma.masked_invalid(mat)
            im = ax.imshow(
                Z,
                origin="lower",
                aspect="auto",
                extent=[-0.5, len(taus) - 0.5, -0.5, len(alphas) - 0.5],
                **im_kwargs,
            )
            ax.set_xticks(range(len(taus)))
            ax.set_xticklabels([str(t) for t in taus])
            ax.set_yticks(range(len(alphas)))
            ax.set_yticklabels([str(a) for a in alphas])
            if row == last_row:
                ax.set_xlabel(r"actor.spi_tau ($\tau$)")
            if col == 0:
                ax.set_ylabel("goub.subgoal_value_alpha (α)")
            ax.set_title(f"{name}  |  epoch {ep}")

            for i in range(len(alphas)):
                for j in range(len(taus)):
                    ax.text(
                        j,
                        i,
                        fmt(mat[i, j]),
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=9,
                        path_effects=pe_white,
                    )
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    fig.suptitle(
        "AntMaze large — GOUB τ × α  (rows: IDM success, Actor success, Actor relative uplift over IDM)\n"
        + suptitle_extra,
        fontsize=11,
    )
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", type=Path, default=Path(__file__).resolve().parent.parent / "runs")
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "docs" / "figures" / "goub_tau_alpha_sweep_antmaze_large_heatmaps.png",
    )
    p.add_argument("--run-glob", type=str, default=RUN_GLOB_DEFAULT)
    p.add_argument(
        "--run-groups",
        type=str,
        default=",".join(RUN_GROUPS_DEFAULT),
        help="콤마 구분 run_group 목록 — 모두 합쳐 한 그리드에 배치.",
    )
    p.add_argument("--taus", type=str, default=",".join(map(str, DEFAULT_TAUS)))
    p.add_argument("--alphas", type=str, default=",".join(map(str, DEFAULT_ALPHAS)))
    args = p.parse_args()

    taus = tuple(float(x.strip()) for x in args.taus.split(",") if x.strip())
    alphas = tuple(float(x.strip()) for x in args.alphas.split(",") if x.strip())
    run_groups = tuple(s.strip() for s in args.run_groups.split(",") if s.strip())

    idm_mats, actor_mats, notes = collect_matrices(
        args.runs_root, args.run_glob, run_groups, taus, alphas
    )
    for n in notes:
        print(n)
    plot_composite(
        idm_mats,
        actor_mats,
        taus,
        alphas,
        args.out,
        suptitle_extra=f"run_groups={'+'.join(run_groups)}",
    )
    print("wrote", args.out)


if __name__ == "__main__":
    main()
