#!/usr/bin/env bash
# AntMaze residual × subgoal grid with te0/sn1 (time embedding off, state norm on).
# Same 24-cell layout as sweep_antmaze_res_subgoal_grid_600ep.sh @ 600 epochs.
#
# Usage:
#   cd /path/to/Pathbridger
#   export PYTHONPATH=. MUJOCO_GL=egl
#   bash scripts/sweep_antmaze_res_subgoal_grid_te0_sn1_600ep.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DOURI_ROOT="${DOURI_ROOT:-$REPO_ROOT/../douri}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/offrl/bin/python}"
GEN="$REPO_ROOT/scripts/write_antmaze_res_subgoal_grid_yaml.py"
STATUS="$REPO_ROOT/scripts/sweep_res_subgoal_cell_status.py"
OUTDIR="${OUTDIR:-$REPO_ROOT/scripts/sweep_generated/antmaze_res_subgoal_grid_te0_sn1_600ep}"
LOGDIR="${LOGDIR:-$REPO_ROOT/scripts/sweep_logs/antmaze_res_subgoal_grid_te0_sn1_600ep}"
RUNS_ROOT="${RUNS_ROOT:-$REPO_ROOT/runs}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-600}"
SWEEP_QUEUE="${SWEEP_QUEUE:-top1_then_top2}"
SKIP_CELLS="${SKIP_CELLS:-large_l1_rd_sa}"

export PYTHONPATH=.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
unset JAX_PLATFORM_NAME JAX_PLATFORMS 2>/dev/null || true

mkdir -p "$OUTDIR" "$LOGDIR"

TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$LOGDIR/sweep_master_${TS}.log"
echo "[$(date -Is)] te0_sn1 sweep start repo=$REPO_ROOT epochs=$TRAIN_EPOCHS queue=$SWEEP_QUEUE skip=$SKIP_CELLS" | tee "$MASTER_LOG"

_run_group_for () {
  local scale="$1" baseline="$2" res_short="$3" sg_short="$4"
  echo "sweep600_res_sg_${scale}_${baseline}_${res_short}_${sg_short}_te0_sn1"
}

_run_cell () {
  local scale="$1" baseline="$2" residual="$3" subgoal="$4"
  local res_short sg_short cell_key
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
  cell_key="${scale}_${baseline}_${res_short}_${sg_short}"

  if [[ -n "${BASELINE_FILTER:-}" && "$baseline" != "$BASELINE_FILTER" ]]; then
    return 0
  fi
  if [[ -n "${CELL_FILTER:-}" && "${res_short}_${sg_short}" != "$CELL_FILTER" ]]; then
    return 0
  fi
  if [[ ",${SKIP_CELLS}," == *",${cell_key},"* ]]; then
    echo "=== $(date -Is) SKIP $cell_key (in SKIP_CELLS) ===" | tee -a "$MASTER_LOG"
    return 0
  fi

  local yname="antmaze_${scale}_${baseline}_${res_short}_${sg_short}_te0_sn1.yaml"
  local ypath="$OUTDIR/$yname"
  local rlog="$LOGDIR/${cell_key}_${TS}.log"
  local run_group
  run_group="$(_run_group_for "$scale" "$baseline" "$res_short" "$sg_short")"

  echo "=== $(date -Is) CELL $cell_key (group=$run_group) ===" | tee -a "$MASTER_LOG"

  local status_line action="" run_dir="" resume_epoch="0" last_epoch="0" reason=""
  while IFS= read -r status_line; do
    case "$status_line" in
      action=*) action="${status_line#action=}" ;;
      run_dir=*) run_dir="${status_line#run_dir=}" ;;
      resume_epoch=*) resume_epoch="${status_line#resume_epoch=}" ;;
      last_epoch=*) last_epoch="${status_line#last_epoch=}" ;;
      reason=*) reason="${status_line#reason=}" ;;
    esac
  done < <("$PYTHON_BIN" "$STATUS" \
    --runs-root "$RUNS_ROOT" \
    --run-group "$run_group" \
    --train-epochs "$TRAIN_EPOCHS")
  echo "  status: action=$action run_dir=$run_dir resume_epoch=$resume_epoch last_epoch=$last_epoch reason=$reason" | tee -a "$MASTER_LOG"

  if [[ "$action" == "skip" ]]; then
    echo "  SKIP (already finished): $run_dir" | tee -a "$MASTER_LOG"
    return 0
  fi

  echo "=== $(date -Is) GEN $cell_key ===" | tee -a "$MASTER_LOG"
  "$PYTHON_BIN" "$GEN" \
    --douri-root "$DOURI_ROOT" \
    --scale "$scale" \
    --baseline "$baseline" \
    --residual "$residual" \
    --subgoal "$subgoal" \
    --time-embedding off \
    --state-normalization on \
    --train_epochs "$TRAIN_EPOCHS" \
    --out "$ypath"

  local -a train_args=(main.py --runs_root="$RUNS_ROOT")
  if [[ "$action" == "resume" || ( "$action" == "run" && -n "$run_dir" ) ]]; then
    train_args+=(--resume_run_dir="$run_dir" --resume_epoch="$resume_epoch")
    if [[ "$resume_epoch" -gt 0 ]]; then
      train_args+=(--resume_use_run_snapshot_config=true)
    else
      train_args+=(--run_config="$ypath")
    fi
    echo "=== $(date -Is) RESUME $run_dir from epoch $resume_epoch ===" | tee -a "$MASTER_LOG"
  else
    train_args+=(--run_config="$ypath")
    echo "=== $(date -Is) TRAIN $ypath ===" | tee -a "$MASTER_LOG"
  fi

  "$PYTHON_BIN" "${train_args[@]}" 2>&1 | tee -a "$rlog" | tee -a "$MASTER_LOG"
  echo "=== $(date -Is) DONE $cell_key ===" | tee -a "$MASTER_LOG"
}

_grid_for_baseline () {
  local scale="$1" baseline="$2"
  local residual subgoal
  for residual in displacement absolute; do
    for subgoal in displacement absolute; do
      _run_cell "$scale" "$baseline" "$residual" "$subgoal"
    done
  done
}

_run_queue_top1 () {
  _grid_for_baseline large l1
  _grid_for_baseline medium m1
  _grid_for_baseline giant g1
}

_run_queue_top2 () {
  _grid_for_baseline large l2
  _grid_for_baseline medium m2
  _grid_for_baseline giant g2
}

case "$SWEEP_QUEUE" in
  top1_only) _run_queue_top1 ;;
  top2_only) _run_queue_top2 ;;
  top1_then_top2 | *) _run_queue_top1; _run_queue_top2 ;;
esac

echo "[$(date -Is)] te0_sn1 sweep queue finished. Master: $MASTER_LOG"
