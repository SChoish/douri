#!/usr/bin/env bash
# Resume top-N configs from phase2 CSV by IDM success @ epoch 200.
#
#   CUDA_VISIBLE_DEVICES=0 bash scripts/with_jax_cuda.sh bash scripts/resume_grid_top3_idm.sh
#
# Env:
#   CSV_PATH   (default: sweep_results/puzzle_fbr_displacement_grid_phase2.csv)
#   TOP_N      (default: 3)
#   RANK_EPOCH (default: 200)
#   DRY_RUN=1  print selection only

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=".:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUNS_ROOT="${RUNS_ROOT:-runs}"
SEED="${SEED:-0}"
CSV_PATH="${CSV_PATH:-${ROOT}/sweep_results/puzzle_fbr_displacement_grid_phase2.csv}"
TOP_N="${TOP_N:-3}"
RANK_EPOCH="${RANK_EPOCH:-200}"
DRY_RUN="${DRY_RUN:-0}"
MET_PY="${ROOT}/scripts/puzzle_fbr_displacement_grid_metrics.py"
TS="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG_ROOT="${ROOT}/launch_logs/grid_fbr_displacement_puzzle_phase2_resume/${TS}"
mkdir -p "${LAUNCH_LOG_ROOT}"

if [[ ! -f "${CSV_PATH}" ]]; then
  echo "[top3] ERROR: CSV not found: ${CSV_PATH}" >&2
  exit 2
fi

rank_json="${LAUNCH_LOG_ROOT}/rank_top${TOP_N}.json"
"${PYTHON_BIN}" "${MET_PY}" rank-top-idm --csv "${CSV_PATH}" --epoch "${RANK_EPOCH}" --top "${TOP_N}" | tee "${rank_json}"

n_picked="$("${PYTHON_BIN}" -c "import json,sys; print(len(json.load(sys.stdin)['top']))" < "${rank_json}")"
if [[ "${n_picked}" -eq 0 ]]; then
  echo "[top3] ERROR: no completed runs to rank in ${CSV_PATH}" >&2
  exit 3
fi

echo "[top3] resuming ${n_picked} run(s) to epoch 400 (IDM @${RANK_EPOCH})"

idx=0
while IFS= read -r item; do
  idx=$((idx + 1))
  run_dir="$("${PYTHON_BIN}" -c "import json,sys; print(json.loads(sys.argv[1])['run_dir'])" "${item}")"
  cfg="$("${PYTHON_BIN}" -c "import json,sys; print(json.loads(sys.argv[1])['config_path'])" "${item}")"
  idm="$("${PYTHON_BIN}" -c "import json,sys; print(json.loads(sys.argv[1])['idm_at_epoch'])" "${item}")"
  base="$(basename "${cfg}" .yaml)"
  step_log="${LAUNCH_LOG_ROOT}/${base}_resume400.log"

  echo "================================================================================"
  echo "[top3] #${idx}/${n_picked} idm@${RANK_EPOCH}=${idm} run_dir=${run_dir}"
  echo "[top3] config=${cfg}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[top3] DRY_RUN=1: skip train"
    continue
  fi

  if grep -qE 'epoch=400 ' "${run_dir}/run.log" 2>/dev/null; then
    echo "[top3] SKIP (already reached epoch 400): ${run_dir}"
    continue
  fi

  set +e
  "${PYTHON_BIN}" main.py \
    --run_config="${cfg}" \
    --runs_root="${RUNS_ROOT}" \
    --seed="${SEED}" \
    --train_epochs=400 \
    --resume_run_dir="${run_dir}" \
    --resume_epoch=200 \
    >>"${step_log}" 2>&1
  ec=$?
  set -e
  if [[ "${ec}" -ne 0 ]]; then
    echo "[top3] FAILED exit=${ec}; log=${step_log}" >&2
    exit "${ec}"
  fi
  echo "[top3] OK resume -> 400; log=${step_log}"
done < <("${PYTHON_BIN}" -c "
import json, sys
for row in json.load(open(sys.argv[1]))['top']:
    print(json.dumps(row))
" "${rank_json}")

echo "================================================================================"
echo "[top3] done. rank file: ${rank_json}"
