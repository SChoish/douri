#!/usr/bin/env bash
# Sequential sweep: phi/full x abs/displacement, planner forward_bridge_residual.
# Usage: ./run_antmaze_fbres_table_sweep_nohup.sh [medium|large]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
MAZE="${1:-medium}"
CONDA_ENV="${CONDA_ENV:-offrl}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${ROOT}/nohup_antmaze_${MAZE}_fbres_table_sweep_${TS}.log"

if [[ "$MAZE" != medium && "$MAZE" != large ]]; then
  echo "usage: $0 [medium|large]" >&2
  exit 1
fi

if [[ "$MAZE" == large ]]; then
  configs=(
    config/antmaze_large_navigate_table_phi_abs.yaml
    config/antmaze_large_navigate_table_phi_disp.yaml
    config/antmaze_large_navigate_table_full_abs.yaml
    config/antmaze_large_navigate_table_full_disp.yaml
  )
else
  configs=(
    config/antmaze_medium_navigate_table_phi_abs.yaml
    config/antmaze_medium_navigate_table_phi_disp.yaml
    config/antmaze_medium_navigate_table_full_abs.yaml
    config/antmaze_medium_navigate_table_full_disp.yaml
  )
fi

{
  echo "=== sweep start $(date -Is) maze=${MAZE} conda=${CONDA_ENV} ==="
  for c in "${configs[@]}"; do
    echo "=== $(date -Is) ${c} ==="
    conda run -n "${CONDA_ENV}" python main.py --run_config="${c}"
  done
  echo "=== sweep done $(date -Is) ==="
} >>"${LOG}" 2>&1
