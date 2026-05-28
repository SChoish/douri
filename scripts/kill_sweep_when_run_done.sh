#!/usr/bin/env bash
# Wait until RUN_DIR finishes TRAIN_EPOCHS, then kill sweep process tree.
set -euo pipefail

RUN_DIR="${1:?run_dir required}"
TRAIN_EPOCHS="${2:-600}"
# PIDs to kill after done (sweep bash / nohup master); training python exits on its own.
shift 2 || true
KILL_PIDS=("$@")

CKPT="${RUN_DIR}/checkpoints/dynamics/params_${TRAIN_EPOCHS}.pkl"
LOG="${RUN_DIR}/run.log"

echo "[$(date -Is)] watch RUN_DIR=$RUN_DIR target_epoch=$TRAIN_EPOCHS kill_pids=${KILL_PIDS[*]:-<none>}"

while true; do
  if [[ -f "$CKPT" ]] && [[ -f "$LOG" ]] && grep -q 'done run_dir=' "$LOG" 2>/dev/null; then
    echo "[$(date -Is)] done detected (ckpt + log). killing sweep..."
    for pid in "${KILL_PIDS[@]}"; do
      if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        echo "  kill $pid"
        kill -TERM "$pid" 2>/dev/null || true
      fi
    done
    sleep 2
    for pid in "${KILL_PIDS[@]}"; do
      if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        kill -KILL "$pid" 2>/dev/null || true
      fi
    done
    echo "[$(date -Is)] sweep stopped."
    exit 0
  fi
  if [[ -f "$LOG" ]] && grep -q "=== EVAL END epoch=${TRAIN_EPOCHS} ===" "$LOG" 2>/dev/null; then
    # Final eval done; wait briefly for done line + ckpt save
    sleep 15
    continue
  fi
  sleep 20
done
