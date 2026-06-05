#!/usr/bin/env bash
# puzzle-3x3-play-v0 표 실험 8개 config를 GPU 한 장에서 순차 학습.
# 사용: CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_puzzle_3x3_table_nohup.sh &
#
# 첫 config만 따로 끝낸 뒤 나머지부터 이어가기:
#   PUZZLE_3X3_TABLE_START_AT=2 CUDA_VISIBLE_DEVICES=0 nohup bash scripts/run_puzzle_3x3_table_nohup.sh &
# (1-based 인덱스; 기본 1 = 전부)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# libcusparse etc. for jax_cuda12 + default JAX_PLATFORMS=cuda (override JAX_PLATFORMS if needed).
# shellcheck disable=SC1091
source "$ROOT/scripts/jax_cuda_env.sh"

PY="${PYTHON:-python3}"
if ! command -v "$PY" &>/dev/null; then
  PY="python"
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
BATCH_LOGDIR="${RUNS_BATCH_LOGDIR:-$ROOT/runs/puzzle_3x3_table_batch_${STAMP}}"
mkdir -p "$BATCH_LOGDIR"

echo "[puzzle batch] repo=$ROOT gpu=CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" | tee "$BATCH_LOGDIR/00_batch_meta.log"
echo "[puzzle batch] python=$PY" | tee -a "$BATCH_LOGDIR/00_batch_meta.log"
echo "[puzzle batch] logdir=$BATCH_LOGDIR" | tee -a "$BATCH_LOGDIR/00_batch_meta.log"

START_AT="${PUZZLE_3X3_TABLE_START_AT:-1}"
if [[ "$START_AT" -lt 1 || "$START_AT" -gt 8 ]]; then
  echo "[puzzle batch] ERROR: PUZZLE_3X3_TABLE_START_AT must be 1..8, got $START_AT" >&2
  exit 1
fi
echo "[puzzle batch] PUZZLE_3X3_TABLE_START_AT=$START_AT (configs before this index are skipped)" | tee -a "$BATCH_LOGDIR/00_batch_meta.log"

CONFIGS=(
  config/puzzle_3x3_play_table_phi_disp.yaml
  config/puzzle_3x3_play_table_phi_abs.yaml
  config/puzzle_3x3_play_table_full_disp.yaml
  config/puzzle_3x3_play_table_full_abs.yaml
  config/puzzle_3x3_play_table_phi_disp_diag_gaussian.yaml
  config/puzzle_3x3_play_table_phi_abs_diag_gaussian.yaml
  config/puzzle_3x3_play_table_full_disp_diag_gaussian.yaml
  config/puzzle_3x3_play_table_full_abs_diag_gaussian.yaml
)

i=0
for cfg in "${CONFIGS[@]}"; do
  i=$((i + 1))
  base="$(basename "$cfg" .yaml)"
  echo "" | tee -a "$BATCH_LOGDIR/00_batch_meta.log"
  if [[ "$i" -lt "$START_AT" ]]; then
    echo "========== SKIP [$i/8] $(date -Is) $cfg (START_AT=$START_AT) ==========" | tee -a "$BATCH_LOGDIR/00_batch_meta.log"
    continue
  fi
  echo "========== [$i/8] $(date -Is) $cfg ==========" | tee -a "$BATCH_LOGDIR/00_batch_meta.log"
  set +e
  "$PY" main.py --run_config="$cfg" >>"$BATCH_LOGDIR/${base}.log" 2>&1
  ec=$?
  set -e
  echo "========== exit_code=$ec $(date -Is) $cfg ==========" | tee -a "$BATCH_LOGDIR/00_batch_meta.log"
  if [[ "$ec" -ne 0 ]]; then
    echo "[puzzle batch] FAILED on $cfg (exit $ec); stopping batch." | tee -a "$BATCH_LOGDIR/00_batch_meta.log"
    exit "$ec"
  fi
done

echo "[puzzle batch] finished from index $START_AT through 8 $(date -Is)" | tee -a "$BATCH_LOGDIR/00_batch_meta.log"
