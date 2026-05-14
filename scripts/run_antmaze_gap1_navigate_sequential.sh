#!/usr/bin/env bash
# antmaze medium / large / giant navigate with subgoal_value_gap_scale=1.0 (phi yaml).
# Sequential GPU runs (nohup + wait per stage).
#
# Usage:
#   cd /home/choi/douri
#   export CUDA_VISIBLE_DEVICES=0
#   nohup bash scripts/run_antmaze_gap1_navigate_sequential.sh \
#     > nohup_logs/seq_antmaze_gap1_only_$(date +%Y%m%d_%H%M%S).log 2>&1 &

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

LOG_DIR="${LOG_DIR:-${ROOT_DIR}/nohup_logs}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT_DIR}/runs}"
PYTHON_BIN="${PYTHON_BIN:-/home/choi/miniconda3/envs/offrl/bin/python}"
mkdir -p "${LOG_DIR}"

CONFIGS=(
  "config/antmaze_medium_navigate_phi_gap1_ha5_kb93_kd80_tau5.yaml"
  "config/antmaze_large_navigate_phi_gap1_ha5_kb93_kd80_tau5.yaml"
  "config/antmaze_giant_navigate_phi_gap1_ha5_kb93_kd80_tau5.yaml"
)

export PYTHONPATH=.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
unset JAX_PLATFORM_NAME JAX_PLATFORMS 2>/dev/null || true

orch_ts="$(date +%Y%m%d_%H%M%S)"
echo "[${orch_ts}] sequential trainer root=${ROOT_DIR}"
echo "[${orch_ts}] python=${PYTHON_BIN} runs_root=${RUNS_ROOT} mujoco_gl=${MUJOCO_GL} cuda_visible_devices=${CUDA_VISIBLE_DEVICES-<unset>}"
echo "[${orch_ts}] stages=${#CONFIGS[@]}"

stage=0
for CONFIG_PATH in "${CONFIGS[@]}"; do
  stage=$((stage + 1))
  if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[stage ${stage}] missing file: ${CONFIG_PATH}" >&2
    exit 1
  fi

  stem="$(basename "${CONFIG_PATH}" .yaml)"
  ts="$(date +%Y%m%d_%H%M%S)"
  step_log="${LOG_DIR}/seq${stage}_${stem}_${ts}.log"

  echo "[stage ${stage}/${#CONFIGS[@]}] start config=${CONFIG_PATH}"
  echo "[stage ${stage}] log=${step_log}"

  cmd=(
    "${PYTHON_BIN}" "${ROOT_DIR}/main.py"
    "--run_config=${CONFIG_PATH}"
    "--runs_root=${RUNS_ROOT}"
  )

  nohup "${cmd[@]}" >>"${step_log}" 2>&1 &
  pid=$!
  echo "[stage ${stage}] pid=${pid}"
  if ! wait "${pid}"; then
    echo "[stage ${stage}] FAILED exit!=0 pid=${pid} log=${step_log}" >&2
    exit 1
  fi
  echo "[stage ${stage}] done"
done

echo "[$(date +%Y%m%d_%H%M%S)] all stages completed ok (${#CONFIGS[@]} runs)"
