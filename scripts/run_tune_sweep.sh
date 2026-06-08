#!/usr/bin/env bash
# Shared runner for gap / weight_max / gamma tune sweeps.
#
# Usage:
#   bash scripts/run_tune_sweep.sh v2
#   bash scripts/run_tune_sweep.sh gw_b
#   bash scripts/run_tune_sweep.sh gap1
#
# Env:
#   GPU_ID, PYTHON_BIN, SEED (default 0)

set -euo pipefail

SET="${1:-}"
if [[ -z "${SET}" ]]; then
  echo "usage: $0 <v2|gw_b|gap1>" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/scripts:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
PYTHON_BIN="${PYTHON_BIN:-/home/choi/miniconda3/envs/offrl/bin/python}"
WITH_CUDA="${ROOT}/scripts/with_jax_cuda.sh"
LOG_DIR="${ROOT}/nohup_logs"
mkdir -p "$LOG_DIR"
WRITER="${ROOT}/scripts/write_tune_sweep_yaml.py"

case "${SET}" in
  v2)
    CONFIG_DIR="${ROOT}/config/sweep_tune_v2"
    LOG_TAG="tune_gw"
    ;;
  gw_b)
    CONFIG_DIR="${ROOT}/config/sweep_tune_gw_b"
    LOG_TAG="tune_gwb"
    ;;
  gap1)
    CONFIG_DIR="${ROOT}/config/sweep_tune_gap1"
    LOG_TAG="tune_g1"
    ;;
  *)
    echo "unknown set: ${SET} (expected v2, gw_b, or gap1)" >&2
    exit 1
    ;;
esac

MASTER_LOG="${LOG_DIR}/${LOG_TAG}_master.log"
"${PYTHON_BIN}" "${WRITER}" --set "${SET}"
"${PYTHON_BIN}" "${WRITER}" --set "${SET}" --probe

echo "[$(date -Is)] ${LOG_TAG} sweep start GPU=${GPU_ID} seed=${SEED}" | tee -a "$MASTER_LOG"

mapfile -t configs < <(
  CONFIG_DIR="${CONFIG_DIR}" "${PYTHON_BIN}" - <<'PY'
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), 'scripts'))
from tune_sweep_common import config_sort_key

config_dir = os.environ['CONFIG_DIR']
paths = [
    p for p in glob.glob(os.path.join(config_dir, '*.yaml'))
    if not os.path.basename(p).startswith('_')
]
for p in sorted(paths, key=lambda x: config_sort_key(os.path.basename(x))):
    print(p)
PY
)

if ((${#configs[@]} == 0)); then
  echo "No configs in ${CONFIG_DIR}" | tee -a "$MASTER_LOG"
  exit 1
fi

for cfg in "${configs[@]}"; do
  base="$(basename "$cfg" .yaml)"
  env_name="$("${PYTHON_BIN}" -c "import yaml; print(yaml.safe_load(open('${cfg}'))['env_name'])")"
  log="${LOG_DIR}/${LOG_TAG}_${base}.log"
  echo "[$(date -Is)] START ${base} env=${env_name}" | tee -a "$MASTER_LOG"
  CUDA_VISIBLE_DEVICES="${GPU_ID}" bash "${WITH_CUDA}" "${PYTHON_BIN}" -u main.py \
    --run_config "${cfg}" \
    --env_name "${env_name}" \
    --seed "${SEED}" \
    --async_prefetch \
    --nouse_wandb \
    --nouse_tqdm \
    2>&1 | tee "${log}"
  echo "[$(date -Is)] DONE ${base}" | tee -a "$MASTER_LOG"
done

echo "[$(date -Is)] ${LOG_TAG} sweep complete" | tee -a "$MASTER_LOG"
