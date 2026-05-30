#!/usr/bin/env bash
# Large-only te0/sn1 residual × subgoal sweep @ 600 epochs.
# Configs: scripts/sweep_generated/antmaze_res_subgoal_grid_te0_sn1_600ep/large_only/
#
# Usage:
#   cd /path/to/Pathbridger
#   export PYTHONPATH=. MUJOCO_GL=egl
#   nohup bash scripts/sweep_antmaze_large_te0_sn1_600ep.sh \
#     > scripts/sweep_logs/antmaze_res_subgoal_grid_te0_sn1_600ep/nohup.out 2>&1 &
#   echo $! > scripts/sweep_logs/antmaze_res_subgoal_grid_te0_sn1_600ep/nohup.pid

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$HOME/miniconda3/envs/offrl/bin/python}"
STATUS="$REPO_ROOT/scripts/sweep_res_subgoal_cell_status.py"
OUTDIR="$REPO_ROOT/scripts/sweep_generated/antmaze_res_subgoal_grid_te0_sn1_600ep/large_only"
LOGDIR="${LOGDIR:-$REPO_ROOT/scripts/sweep_logs/antmaze_res_subgoal_grid_te0_sn1_600ep}"
RUNS_ROOT="${RUNS_ROOT:-$REPO_ROOT/runs}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-600}"
SKIP_CELLS="${SKIP_CELLS:-large_l1_rd_sa}"

export PYTHONPATH=.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
unset JAX_PLATFORM_NAME JAX_PLATFORMS 2>/dev/null || true

mkdir -p "$LOGDIR"

TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$LOGDIR/sweep_large_master_${TS}.log"
echo "[$(date -Is)] large te0_sn1 sweep start epochs=$TRAIN_EPOCHS skip=$SKIP_CELLS" | tee "$MASTER_LOG"

_run_group_from_yaml () {
  "$PYTHON_BIN" - <<'PY' "$1"
import sys, yaml
from pathlib import Path
data = yaml.safe_load(Path(sys.argv[1]).read_text(encoding='utf-8')) or {}
print(data.get('run_group', ''))
PY
}

_run_cell_yaml () {
  local ypath="$1"
  local yname cell_key run_group action="" run_dir="" resume_epoch="0" last_epoch="0" reason=""

  yname="$(basename "$ypath")"
  cell_key="${yname#antmaze_}"
  cell_key="${cell_key/_te0_sn1.yaml/}"

  if [[ ",${SKIP_CELLS}," == *",${cell_key},"* ]]; then
    echo "=== $(date -Is) SKIP $cell_key (in SKIP_CELLS) ===" | tee -a "$MASTER_LOG"
    return 0
  fi

  run_group="$(_run_group_from_yaml "$ypath")"
  if [[ -z "$run_group" ]]; then
    echo "ERROR: missing run_group in $ypath" | tee -a "$MASTER_LOG"
    exit 1
  fi

  echo "=== $(date -Is) CELL $cell_key (group=$run_group) ===" | tee -a "$MASTER_LOG"

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

  local rlog="$LOGDIR/${cell_key}_${TS}.log"
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

mapfile -t YAMLS < <(ls -1 "$OUTDIR"/antmaze_large_*.yaml | sort)
for ypath in "${YAMLS[@]}"; do
  _run_cell_yaml "$ypath"
done

echo "[$(date -Is)] large te0_sn1 sweep finished. Master: $MASTER_LOG"
