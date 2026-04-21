#!/usr/bin/env bash
# GOUB dynamics: OGBench antmaze navigate + stitch, 1000 epochs each env.
# Checkpoints: runs/<ts>_goub_dynamics_seed0_<env>/checkpoints/params_<epoch>.pkl
#
# Usage:
#   cd <douri repo root> && nohup bash scripts/run_goub_dynamics_antmaze_1000.sh &
# Log file is written under runs/nohup_goub_dynamics_antmaze_<timestamp>.log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMPL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${IMPL_DIR}"

RUNS_DIR="${IMPL_DIR}/runs"
mkdir -p "${RUNS_DIR}"
LOG_FILE="${RUNS_DIR}/nohup_goub_dynamics_antmaze_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "nohup batch log: ${LOG_FILE}"
echo "Checkpoints per env under: ${RUNS_DIR}/<timestamp>_goub_dynamics_seed0_<env_name>/"

CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
else
  eval "$(conda shell.bash hook 2>/dev/null)" || true
fi

conda activate offrl

ENVS=(
  antmaze-medium-navigate-v0
  antmaze-large-navigate-v0
  antmaze-giant-navigate-v0
  antmaze-teleport-navigate-v0
  antmaze-medium-stitch-v0
  antmaze-large-stitch-v0
  antmaze-giant-stitch-v0
  antmaze-teleport-stitch-v0
)

for env in "${ENVS[@]}"; do
  echo "========== START ${env} $(date -Is) =========="
  python main_goub_dynamics.py \
    --env_name="${env}" \
    --train_epochs=1000 \
    --save_every_n_epochs=100 \
    --log_every_n_epochs=10 \
    --use_tqdm=false
  echo "========== DONE ${env} $(date -Is) =========="
done

echo "ALL DONE $(date -Is)"
