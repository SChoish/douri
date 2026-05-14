#!/usr/bin/env bash
# 4 환경(cube_double, puzzle_3x3, antmaze_giant_navigate, antmaze_teleport_navigate)을
# train_epochs=1000, horizon=40 으로 nohup + wait 으로 한 번에 하나씩 학습한다.
#
# 사용 예:
#   cd /home/choi/douri
#   nohup bash scripts/run_4env_h40_e1000_sequential.sh \
#     > nohup_logs/seq_4env_h40_e1000_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# 환경 변수 override (선택):
#   PYTHON_BIN  학습용 python (default: /home/choi/miniconda3/envs/offrl/bin/python)
#   RUNS_ROOT   runs 디렉터리 root (default: <repo>/runs)
#   LOG_DIR     단계별 로그 위치 (default: <repo>/nohup_logs)
#   MUJOCO_GL   default egl

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

LOG_DIR="${LOG_DIR:-${ROOT_DIR}/nohup_logs}"
RUNS_ROOT="${RUNS_ROOT:-${ROOT_DIR}/runs}"
PYTHON_BIN="${PYTHON_BIN:-/home/choi/miniconda3/envs/offrl/bin/python}"
mkdir -p "${LOG_DIR}"

CONFIGS=(
  "config/cube_double.yaml"
  "config/puzzle_3x3.yaml"
  "config/antmaze_giant_navigate.yaml"
  "config/antmaze_teleport_navigate.yaml"
)

EXTRA_ARGS=(
  "--train_epochs=1000"
  "--horizon=40"
)

export PYTHONPATH=.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
unset JAX_PLATFORM_NAME JAX_PLATFORMS CUDA_VISIBLE_DEVICES 2>/dev/null || true

orch_ts="$(date +%Y%m%d_%H%M%S)"
echo "[${orch_ts}] sequential trainer root=${ROOT_DIR}"
echo "[${orch_ts}] python=${PYTHON_BIN} runs_root=${RUNS_ROOT} mujoco_gl=${MUJOCO_GL}"
echo "[${orch_ts}] stages=${#CONFIGS[@]} extra_args=${EXTRA_ARGS[*]}"

stage=0
for CONFIG_PATH in "${CONFIGS[@]}"; do
  stage=$((stage + 1))
  if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[stage ${stage}] SKIP missing file: ${CONFIG_PATH}" >&2
    exit 1
  fi

  stem="$(basename "${CONFIG_PATH}" .yaml)"
  ts="$(date +%Y%m%d_%H%M%S)"
  step_log="${LOG_DIR}/seq${stage}_${stem}_h40_e1000_${ts}.log"

  echo "[stage ${stage}/${#CONFIGS[@]}] start config=${CONFIG_PATH}"
  echo "[stage ${stage}] log=${step_log}"

  cmd=(
    "${PYTHON_BIN}" "${ROOT_DIR}/main.py"
    "--run_config=${CONFIG_PATH}"
    "--runs_root=${RUNS_ROOT}"
    "${EXTRA_ARGS[@]}"
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
