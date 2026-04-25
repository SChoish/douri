#!/usr/bin/env python3
"""Regenerate ``docs/eval_task_scores_ep100_ep200_20260424_joint_antmaze_large.md`` and the
``runs/…`` copy from all ``runs/*_joint_dqc_seed0_antmaze-large-navigate-v0`` runs.

Preserves the appendix + notes suffix (from ``### 표 해석 메모`` onward) in the existing docs file.
YAML에 없는 ``goub`` / ``actor`` / top-level 키는 ``get_dynamics_config`` / ``get_actor_config`` / ``main`` 기본값으로 채움.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
RUN_GLOB = "*_joint_dqc_seed0_antmaze-large-navigate-v0"
EPOCHS = (100, 200, 300, 400, 500, 600)
SUFFIX_ANCHOR = "### 표 해석 메모"

_HEATMAP_BLURB = (
    "**sweep 시각화 (단일 통합 그림, τ×α = 4×4):** "
    "`docs/figures/goub_tau_alpha_sweep_antmaze_large_heatmaps.png` — "
    "**3행×4열**: (1) IDM 태스크 평균 success, (2) Actor 태스크 평균 success, "
    "(3) Actor 상대 증가율 `(Actor−IDM)/IDM` (RdBu_r, 0 중심). "
    "두 sweep(`antmaze_navigate_goub_tau_alpha_sweep` + "
    "`antmaze_navigate_goub_tau_at_alpha0p3_sweep`)을 한 그리드에 합칩니다 "
    "(τ∈{0.5,1,5,10,20}, α∈{0,0.1,0.3,0.5,1}). "
    "생성: `python scripts/plot_goub_tau_alpha_sweep_heatmaps.py`\n\n"
    "**α=0.3 고정, τ∈{5,10,20} (large) 운영:** 설정 `config/sweep_goub_tau_at_alpha0p3/`, "
    "실행 `scripts/launch_goub_tau_at_alpha0p3_large.sh` / "
    "재개 `scripts/resume_goub_tau_at_alpha0p3_large.sh` — 그림은 위 통합 PNG에 포함됩니다.\n\n"
)

_MAIN_TOP_DEFAULTS: dict[str, Any] = {
    "train_epochs": 10,
    "log_every_n_epochs": 1,
    "save_every_n_epochs": 100,
    "batch_size": 256,
    "plan_candidates": 1,
    "plan_noise_scale": 1.0,
    "joint_horizon": 25,
    "eval_freq": 100,
    "eval_episodes_per_task": 10,
    "eval_max_chunks": 200,
}


def _ensure_repo_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


def merge_goub(cfg: dict) -> dict[str, Any]:
    _ensure_repo_path()
    from agents.goub_dynamics import get_dynamics_config

    base: dict[str, Any] = dict(get_dynamics_config().to_dict())
    base.update(cfg.get("goub") or {})
    return base


def merge_actor(cfg: dict) -> dict[str, Any]:
    _ensure_repo_path()
    from agents.actor import get_actor_config

    base: dict[str, Any] = dict(get_actor_config().to_dict())
    base.update(cfg.get("actor") or {})
    return base


def merge_top_level(cfg: dict) -> dict[str, Any]:
    out = dict(_MAIN_TOP_DEFAULTS)
    for k, v in cfg.items():
        if k in ("goub", "critic_agent", "actor") or v is None:
            continue
        out[k] = v
    return out


def format_joint_run_diff_markdown(cfg: dict) -> str:
    g = merge_goub(cfg)
    a = merge_actor(cfg)
    top = merge_top_level(cfg)
    tau = a.get("spi_tau", "—")
    alpha = g.get("subgoal_value_alpha", "—")
    bt = g.get("bridge_type", "goub")
    clip = g.get("clip_path_to_goal", True)
    clip_s = "true" if clip is True else ("false" if clip is False else str(clip))
    te = top.get("train_epochs", "—")
    pc = top.get("plan_candidates", 1)
    rg = top.get("run_group", "")
    parts = [
        f"`spi_tau={tau}`",
        f"`subgoal_α={alpha}`",
        f"`bridge_type={bt}`",
        f"`clip_path_to_goal={clip_s}`",
        f"`train_epochs={te}`",
    ]
    if int(pc) != 1:
        parts.append(f"`plan_candidates={pc}`")
    parts.append(f"`run_group={rg}`")
    return " ".join(parts)


def read_joint_run_logs(run_dir: Path) -> str:
    parts: list[str] = []
    for logf in sorted(run_dir.glob("run*.log")):
        try:
            parts.append(logf.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
    return "\n".join(parts)


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


def fmt_pair(pair: tuple[float, float] | None) -> str:
    if pair is None:
        return "—"
    a, b = pair
    return f"{a:.2f} / {b:.2f}"


def discover_rows(runs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for d in sorted(runs_root.glob(RUN_GLOB)):
        if not d.is_dir():
            continue
        cfgp = d / "config_used.yaml"
        if not cfgp.is_file():
            continue
        cfg = yaml.safe_load(cfgp.read_text(encoding="utf-8")) or {}
        if not any(d.glob("run*.log")):
            continue
        short = d.name.split("_", 2)[1]
        means = parse_eval_means(read_joint_run_logs(d))
        rows.append(
            {
                "dir": d.name,
                "short": short,
                "cfg": cfg,
                "means": means,
            }
        )
    return rows


def build_tables_markdown(rows: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("## 런 디렉터리 (짧은 ID)\n\n")
    lines.append("| run 디렉터리 | 짧은 ID |\n|--------------|---------|\n")
    for r in rows:
        lines.append(f"| `{r['dir']}` | **{r['short']}** |\n")

    lines.append("\n## 런별 차이점 (`config_used.yaml` + 코드 기본값 병합)\n\n")
    lines.append("| run | 차이점 |\n|-----|--------|\n")
    for r in rows:
        lines.append(f"| **{r['short']}** | {format_joint_run_diff_markdown(r['cfg'])} |\n")

    lines.append("\n## EVAL mean: IDM / Actor (태스크 평균)\n\n")
    lines.append(
        "형식: `IDM mean` / `Actor mean`. 해당 epoch에 `EVAL` 블록이 없으면 `—`. "
        "로그는 각 런의 ``run*.log``(``run.log`` + ``run_resume_from*.log`` 등)를 합쳐 파싱합니다.\n\n"
    )
    lines.append("| run | ep100 (IDM / Actor) | ep200 (IDM / Actor) | ep300 (IDM / Actor) | ep400 (IDM / Actor) | ep500 (IDM / Actor) | ep600 (IDM / Actor) |\n")
    lines.append("|-----|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|\n")
    for r in rows:
        means: dict[int, tuple[float, float]] = r["means"]
        cells = [f"**{r['short']}**"] + [fmt_pair(means.get(ep)) for ep in EPOCHS]
        lines.append("| " + " | ".join(cells) + " |\n")

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["run_id"] + [f"ep{ep}_idm_mean" for ep in EPOCHS] + [f"ep{ep}_actor_mean" for ep in EPOCHS])
    for r in rows:
        means = r["means"]
        row = [r["short"]]
        for ep in EPOCHS:
            p = means.get(ep)
            row.append("" if p is None else f"{p[0]:.4f}")
        for ep in EPOCHS:
            p = means.get(ep)
            row.append("" if p is None else f"{p[1]:.4f}")
        w.writerow(row)
    csv_body = buf.getvalue().strip()

    lines.append("\n### 동일 표 (CSV)\n\n```csv\n")
    lines.append(csv_body)
    lines.append("\n```\n\n")
    return "".join(lines)


def build_log_path_list(rows: list[dict[str, Any]]) -> str:
    lines = ["## 원본 로그 경로\n\n", "로컬 `runs/` 아래 (저장소에서는 `.gitignore`):\n\n"]
    for r in rows:
        lines.append(f"- `runs/{r['dir']}/run.log`\n")
    return "".join(lines)


def _suffix_from_existing_docs(docs_path: Path) -> str:
    text = docs_path.read_text(encoding="utf-8")
    if SUFFIX_ANCHOR not in text:
        raise SystemExit(f"missing anchor {SUFFIX_ANCHOR!r} in {docs_path}")
    return text[text.index(SUFFIX_ANCHOR) :]


def _docs_header(*, updated_line: str) -> str:
    return (
        "# AntMaze large joint — 런별 설정 요약 및 EVAL mean (ep100–600)\n\n"
        "각 셀의 성공률은 각 런의 ``run*.log`` 안의 `=== EVAL START epoch=… ===` 블록에서 읽은 "
        "**태스크 평균** (`idm` / `actor`의 `success_rate_mean`)입니다. "
        "`RE-EVAL` 블록은 아래 표에 넣지 않았습니다(094849 부록 참고).\n\n"
        "공통: `num_tasks=5`, `episodes_per_task=10`.\n\n"
        "로컬 메모 사본: `runs/eval_task_scores_ep100_ep200_20260424_joint_antmaze_large.md`.\n\n"
        f"{updated_line}\n\n"
        + _HEATMAP_BLURB
    )


def _runs_header(*, updated_line: str) -> str:
    return (
        "# AntMaze large joint — 런별 설정 요약 및 EVAL mean (ep100–600)\n\n"
        "각 셀의 성공률은 각 런의 ``run*.log`` 안의 `=== EVAL START epoch=… ===` 블록에서 읽은 "
        "**태스크 평균** (`idm` / `actor`의 `success_rate_mean`)입니다. "
        "`RE-EVAL` 블록은 아래 표에 넣지 않았습니다(094849 부록 참고).\n\n"
        "공통: `num_tasks=5`, `episodes_per_task=10`.\n\n"
        "동일 내용 Git 추적본: `docs/eval_task_scores_ep100_ep200_20260424_joint_antmaze_large.md`.\n\n"
        f"{updated_line}\n\n"
        + _HEATMAP_BLURB
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    args = p.parse_args()

    docs_path = ROOT / "docs" / "eval_task_scores_ep100_ep200_20260424_joint_antmaze_large.md"
    runs_path = ROOT / "runs" / "eval_task_scores_ep100_ep200_20260424_joint_antmaze_large.md"
    if not docs_path.is_file():
        raise SystemExit(f"missing {docs_path}")

    rows = discover_rows(args.runs_root)
    ts = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    updated = (
        f"**갱신:** {ts} — 아래 표·CSV 블록·원본 로그 목록은 "
        "`python scripts/update_eval_task_scores_joint_antmaze_large.py` 로 재생성."
    )

    suffix = _suffix_from_existing_docs(docs_path)
    # Strip trailing "## 원본 로그 경로" section from suffix if present (regenerated below).
    if "## 원본 로그 경로" in suffix:
        suffix = suffix[: suffix.index("## 원본 로그 경로")].rstrip() + "\n\n"

    body = build_tables_markdown(rows)
    log_list = build_log_path_list(rows)
    full_docs = _docs_header(updated_line=updated) + body + suffix + log_list
    full_runs = _runs_header(updated_line=updated) + body + suffix + log_list

    docs_path.write_text(full_docs, encoding="utf-8")
    runs_path.write_text(full_runs, encoding="utf-8")
    print("wrote", docs_path)
    print("wrote", runs_path)


if __name__ == "__main__":
    main()
