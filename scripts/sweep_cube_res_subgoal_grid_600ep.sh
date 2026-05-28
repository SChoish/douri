#!/usr/bin/env bash
# 6 cube-play baseline (gap·κ) × 4 (residual × subgoal) = 24 runs @ 600 epochs.
#
# Baselines (leaderboard gap·κ):
#   triple t1: gap=5 κ=0.8   t2: gap=5 κ=0.6
#   double d1: gap=5 κ=0.6   d2: gap=5 κ=0.7
#   single s1: gap=1 κ=0.9   s2: gap=5 κ=0.9
#
# Template: douri cube_double_play_baseline + antmaze FBR grid yaml (Pathbridger fields).
#
# Usage:
#   cd /path/to/Pathbridger
#   export PYTHONPATH=.
#   export MUJOCO_GL=egl
#   bash scripts/sweep_cube_res_subgoal_grid_600ep.sh
#
# Optional: DOURI_ROOT, PYTHON_BIN, RUNS_ROOT, TRAIN_EPOCHS,
#   SCALES="triple double single"  BASELINE_FILTER="t1"

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DOURI_ROOT="${DOURI_ROOT:-$REPO_ROOT/../douri}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GEN="$REPO_ROOT/scripts/write_cube_res_subgoal_grid_yaml.py"
OUTDIR="${OUTDIR:-$REPO_ROOT/scripts/sweep_generated/cube_res_subgoal_grid_600ep}"
LOGDIR="${LOGDIR:-$REPO_ROOT/scripts/sweep_logs/cube_res_subgoal_grid_600ep}"
RUNS_ROOT="${RUNS_ROOT:-$REPO_ROOT/runs}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-600}"
SCALES="${SCALES:-triple double single}"

export PYTHONPATH=.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
unset JAX_PLATFORM_NAME JAX_PLATFORMS 2>/dev/null || true

mkdir -p "$OUTDIR" "$LOGDIR"

TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$LOGDIR/sweep_master_${TS}.log"
echo "[$(date -Is)] cube res×subgoal sweep start epochs=$TRAIN_EPOCHS scales=$SCALES" | tee "$MASTER_LOG"

run_cell () {
  local scale="$1" baseline="$2" residual="$3" subgoal="$4"
  local res_short sg_short
  case "$residual" in
    displacement) res_short=rd ;;
    absolute) res_short=ra ;;
    *) echo "bad residual: $residual" >&2; exit 1 ;;
  esac
  case "$subgoal" in
    displacement) sg_short=sd ;;
    absolute) sg_short=sa ;;
    *) echo "bad subgoal: $subgoal" >&2; exit 1 ;;
  esac

  if [[ -n "${BASELINE_FILTER:-}" && "$baseline" != "$BASELINE_FILTER" ]]; then
    return 0
  fi

  local yname="cube_${scale}_${baseline}_${res_short}_${sg_short}.yaml"
  local ypath="$OUTDIR/$yname"
  local rlog="$LOGDIR/${scale}_${baseline}_${res_short}_${sg_short}_${TS}.log"

  echo "=== $(date -Is) GEN $scale $baseline residual=$residual subgoal=$subgoal ===" | tee -a "$MASTER_LOG"
  "$PYTHON_BIN" "$GEN" \
    --douri-root "$DOURI_ROOT" \
    --scale "$scale" \
    --baseline "$baseline" \
    --residual "$residual" \
    --subgoal "$subgoal" \
    --train_epochs "$TRAIN_EPOCHS" \
    --out "$ypath"

  echo "=== $(date -Is) TRAIN $ypath ===" | tee -a "$MASTER_LOG"
  "$PYTHON_BIN" main.py \
    --run_config="$ypath" \
    --runs_root="$RUNS_ROOT" \
    2>&1 | tee -a "$rlog" | tee -a "$MASTER_LOG"
  echo "=== $(date -Is) DONE $ypath ===" | tee -a "$MASTER_LOG"
}

grid_for_scale () {
  local scale="$1"
  local -a baselines=()
  case "$scale" in
    triple) baselines=(t1 t2) ;;
    double) baselines=(d1 d2) ;;
    single) baselines=(s1 s2) ;;
    *)
      echo "unknown scale: $scale" >&2
      exit 1
      ;;
  esac
  local b residual subgoal
  for b in "${baselines[@]}"; do
    for residual in displacement absolute; do
      for subgoal in displacement absolute; do
        run_cell "$scale" "$b" "$residual" "$subgoal"
      done
    done
  done
}

for scale in $SCALES; do
  grid_for_scale "$scale"
done

echo "[$(date -Is)] cube sweep finished. Master: $MASTER_LOG"
