#!/usr/bin/env bash
set -euo pipefail

# AntMaze large linear dynamics sweep.
# Runs alpha in {0.1, 0.3, 0.5} and spi_tau in {5, 10}, skipping
# alpha=0.3 / tau=10 because runs/20260425_224314 already covers it.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/choi/miniconda3/envs/offrl/bin/python}"
LOG_DIR="${ROOT_DIR}/nohup_logs"
mkdir -p "${LOG_DIR}"

CONFIGS=(
  "config/sweep_dynamics_tau_alpha/antmaze_large_navigate_dynamics_tau5p0_alpha0p1.yaml"
  "config/sweep_dynamics_tau_alpha/antmaze_large_navigate_dynamics_tau10p0_alpha0p1.yaml"
  "config/sweep_dynamics_tau_alpha/antmaze_large_navigate_dynamics_tau5p0_alpha0p3.yaml"
  "config/sweep_dynamics_tau_alpha/antmaze_large_navigate_dynamics_tau5p0_alpha0p5.yaml"
  "config/sweep_dynamics_tau_alpha/antmaze_large_navigate_dynamics_tau10p0_alpha0p5.yaml"
)

cd "${ROOT_DIR}"
for cfg in "${CONFIGS[@]}"; do
  stem="$(basename "${cfg}" .yaml)"
  ts="$(date +%Y%m%d_%H%M%S)"
  log="${LOG_DIR}/${stem}_${ts}.log"
  echo "=== ${cfg} ==="
  echo "log=${log}"
  "${PYTHON_BIN}" main.py --run_config="${cfg}" 2>&1 | tee "${log}"
done
