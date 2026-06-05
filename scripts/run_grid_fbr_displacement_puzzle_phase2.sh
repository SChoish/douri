#!/usr/bin/env bash
# Phase-2 puzzle grid: alpha in {0.3,0.5}, gap in {5,10,20}, stage1 only (200 epochs).
# After all complete: bash scripts/resume_grid_top3_idm.sh
#
#   SKIP_GENERATE=1 CUDA_VISIBLE_DEVICES=0 nohup bash scripts/with_jax_cuda.sh \
#     bash scripts/run_grid_fbr_displacement_puzzle_phase2.sh >> puzzle_grid_phase2.master.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=".:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUNS_ROOT="${RUNS_ROOT:-runs}"
SEED="${SEED:-0}"
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-0}"
SKIP_GENERATE="${SKIP_GENERATE:-1}"
PUZZLE_TAG="${PUZZLE_TAG:-3x3}"

TS="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG_ROOT="${ROOT}/launch_logs/grid_fbr_displacement_puzzle_phase2/${TS}"
CSV_OUT="${ROOT}/sweep_results/puzzle_fbr_displacement_grid_phase2.csv"
MET_PY="${ROOT}/scripts/puzzle_fbr_displacement_grid_metrics.py"
GEN_PY="${ROOT}/scripts/generate_grid_fbr_displacement_puzzle_configs.py"

mkdir -p "${LAUNCH_LOG_ROOT}" "${ROOT}/sweep_results"

if [[ "${SKIP_GENERATE}" != "1" ]]; then
  echo "[phase2] generating configs..."
  "${PYTHON_BIN}" "${GEN_PY}"
fi

mapfile -t CONFIGS < <("${PYTHON_BIN}" - "${ROOT}" "${PUZZLE_TAG}" <<'PY'
import re, sys
from pathlib import Path

root = Path(sys.argv[1])
tag = sys.argv[2]
allowed_alpha = {0.3, 0.5}
allowed_gap = {5.0, 10.0, 20.0}
prefix = f"puzzle_{tag}_"

def dec(s):
    return float(s.replace("p", ".").replace("m", "-"))

pat = re.compile(
    r"^puzzle_(3x3|4x4|4x6)_a([^_]+)_gap([^_]+)_k([^\.]+)\.yaml$"
)
paths = []
for p in sorted((root / "config/grid_fbr_displacement_puzzle").glob(f"{prefix}*.yaml")):
    m = pat.match(p.name)
    if not m or m.group(1) != tag:
        continue
    a, g = dec(m.group(2)), dec(m.group(3))
    if a in allowed_alpha and g in allowed_gap:
        paths.append(str(p))
for line in paths:
    print(line)
PY
)
NCFG="${#CONFIGS[@]}"
echo "[phase2] puzzle_${PUZZLE_TAG}: ${NCFG} configs (alpha in {0.3,0.5}, gap in {5,10,20}, 200ep only)"

if [[ "${NCFG}" -eq 0 ]]; then
  echo "[phase2] ERROR: no configs matched filter" >&2
  exit 2
fi

read_yaml_top() {
  local file="$1" key="$2"
  "${PYTHON_BIN}" -c "import yaml,sys; d=yaml.safe_load(open(sys.argv[1])); print(d.get(sys.argv[2],''))" "$file" "$key"
}

read_yaml_critic() {
  local file="$1" key="$2"
  "${PYTHON_BIN}" -c "import yaml,sys; d=yaml.safe_load(open(sys.argv[1])); c=d.get('critic_agent')or{}; print(c.get(sys.argv[2],''))" "$file" "$key"
}

find_run_dir_fallback() {
  local env_name="$1" seed="$2"
  find "${RUNS_ROOT}" -maxdepth 1 -type d -name "*_seed${seed}_${env_name}" -printf '%T@\t%p\n' 2>/dev/null | sort -nr | head -1 | cut -f2- || true
}

stage1_run_dir_for_config() {
  local cfg="$1" env_name="$2"
  local cfg_base
  cfg_base="$(basename "$cfg")"
  local d rc
  while IFS= read -r d; do
    [[ -n "${d}" && -f "${d}/run.log" ]] || continue
    grep -qE 'done run_dir=' "${d}/run.log" || continue
    grep -qE 'epoch=200 ' "${d}/run.log" || continue
    [[ -f "${d}/flags.json" ]] || continue
    rc="$("${PYTHON_BIN}" -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('flags',{}).get('run_config',''))" "${d}/flags.json")"
    if [[ "${rc}" == *"${cfg_base}"* ]]; then
      echo "${d}"
      return 0
    fi
  done < <(find "${RUNS_ROOT}" -maxdepth 1 -type d -name "*_seed${SEED}_${env_name}" -printf '%T@\t%p\n' 2>/dev/null | sort -nr | cut -f2-)
  return 1
}

resolve_run_dir() {
  local log_path="$1" env_name="$2" seed="$3"
  local from_log
  from_log="$("${PYTHON_BIN}" "${MET_PY}" parse-run-dir --log "${log_path}" 2>/dev/null | "${PYTHON_BIN}" -c "import json,sys; d=json.load(sys.stdin); print(d.get('run_dir') or '')")"
  if [[ -n "${from_log}" ]]; then
    echo "${from_log}"
    return 0
  fi
  find_run_dir_fallback "${env_name}" "${seed}"
}

append_csv_row() {
  "${PYTHON_BIN}" "${MET_PY}" append-row --csv "${CSV_OUT}" --json "$1"
}

parse_cfg_nums() {
  "${PYTHON_BIN}" -c "
import re, json, sys
from pathlib import Path
p = Path(sys.argv[1])
m = re.match(r'^puzzle_(3x3|4x4|4x6)_a(?P<a>[^_]+)_gap(?P<g>[^_]+)_k(?P<k>[^.]+)\\.yaml\$', p.name)
dec = lambda s: float(s.replace('p','.').replace('m','-'))
print(json.dumps({'alpha': dec(m.group('a')), 'gap': dec(m.group('g')), 'kappa': dec(m.group('k'))}))
" "$1"
}

write_result_row() {
  export GRID_ROW_ENV_NAME="$1"
  export GRID_ROW_CONFIG_PATH="$2"
  export GRID_ROW_ALPHA="$3"
  export GRID_ROW_GAP="$4"
  export GRID_ROW_KAPPA="$5"
  export GRID_ROW_DISCOUNT="$6"
  export GRID_ROW_BATCH="$7"
  export GRID_ROW_SEED="$8"
  export GRID_ROW_RUN_DIR="$9"
  export GRID_ROW_STAGE1="${10}"
  export GRID_ROW_METRIC_NAME="${11}"
  export GRID_ROW_B200="${12}"
  export GRID_ROW_M200="${13}"
  export GRID_ROW_CONT400="${14}"
  export GRID_ROW_B400="${15}"
  export GRID_ROW_F400="${16}"
  export GRID_ROW_IDM200="${17}"
  export GRID_ROW_ACT200="${18}"
  export GRID_ROW_IDM400="${19}"
  export GRID_ROW_ACT400="${20}"
  export GRID_ROW_STATUS="${21}"
  export GRID_ROW_NOTES="${22}"
  export GRID_ROW_JSON_OUT="${23}"
  "${PYTHON_BIN}" - <<'PY'
import json, os
row = {
  'env_name': os.environ['GRID_ROW_ENV_NAME'],
  'config_path': os.environ['GRID_ROW_CONFIG_PATH'],
  'alpha': os.environ['GRID_ROW_ALPHA'],
  'gap': os.environ['GRID_ROW_GAP'],
  'kappa': os.environ['GRID_ROW_KAPPA'],
  'discount': os.environ['GRID_ROW_DISCOUNT'],
  'batch_size': os.environ['GRID_ROW_BATCH'],
  'seed': os.environ['GRID_ROW_SEED'],
  'run_dir': os.environ['GRID_ROW_RUN_DIR'],
  'stage1_completed': os.environ['GRID_ROW_STAGE1'],
  'metric_name': os.environ['GRID_ROW_METRIC_NAME'],
  'best_metric_upto_200': os.environ['GRID_ROW_B200'],
  'metric_at_200': os.environ['GRID_ROW_M200'],
  'continued_to_400': os.environ['GRID_ROW_CONT400'],
  'best_metric_upto_400': os.environ['GRID_ROW_B400'],
  'final_metric_400': os.environ['GRID_ROW_F400'],
  'idm_best_upto_200': os.environ['GRID_ROW_IDM200'],
  'actor_best_upto_200': os.environ['GRID_ROW_ACT200'],
  'idm_final_400': os.environ['GRID_ROW_IDM400'],
  'actor_final_400': os.environ['GRID_ROW_ACT400'],
  'status': os.environ['GRID_ROW_STATUS'],
  'notes': os.environ['GRID_ROW_NOTES'],
}
with open(os.environ['GRID_ROW_JSON_OUT'], 'w', encoding='utf-8') as f:
    json.dump(row, f)
PY
}

cfg_idx=0
for cfg in "${CONFIGS[@]}"; do
  cfg_idx=$((cfg_idx + 1))
  base="$(basename "$cfg" .yaml)"
  step_log="${LAUNCH_LOG_ROOT}/${base}.log"
  echo "================================================================================"
  echo "[phase2] config=${cfg} (${cfg_idx}/${NCFG})"

  env_name="$(read_yaml_top "$cfg" env_name)"
  discount="$(read_yaml_critic "$cfg" discount)"
  batch_size="$(read_yaml_top "$cfg" batch_size)"
  nums="$(parse_cfg_nums "$cfg")"
  alpha="$(echo "$nums" | "${PYTHON_BIN}" -c "import json,sys; print(json.load(sys.stdin)['alpha'])")"
  gap="$(echo "$nums" | "${PYTHON_BIN}" -c "import json,sys; print(json.load(sys.stdin)['gap'])")"
  kappa="$(echo "$nums" | "${PYTHON_BIN}" -c "import json,sys; print(json.load(sys.stdin)['kappa'])")"

  tmp200="${LAUNCH_LOG_ROOT}/${base}_m200.json"
  row_json="${LAUNCH_LOG_ROOT}/${base}_row.json"

  status="ok"
  notes="phase2_200ep_only"
  run_dir=""
  stage1_ok="false"
  idm_at_200=""
  act_at_200=""
  best200=""
  m200=""

  ec1=0
  if existing_run="$(stage1_run_dir_for_config "$cfg" "$env_name")"; then
    echo "[phase2] SKIP train (already 200ep complete): ${existing_run}" | tee -a "${step_log}"
    run_dir="${existing_run}"
  else
    set +e
    "${PYTHON_BIN}" main.py \
      --run_config="${cfg}" \
      --runs_root="${RUNS_ROOT}" \
      --seed="${SEED}" \
      --train_epochs=200 \
      >>"${step_log}" 2>&1
    ec1=$?
    set -e
    run_dir=""
  fi

  if [[ "${ec1}" -ne 0 ]]; then
    status="failed"
    notes="main.py exit=${ec1}; log=${step_log}"
    run_dir="$(resolve_run_dir "${step_log}" "${env_name}" "${SEED}" || true)"
    write_result_row "${env_name}" "${cfg}" "${alpha}" "${gap}" "${kappa}" "${discount}" "${batch_size}" "${SEED}" \
      "${run_dir}" "false" "" "" "" "false" "" "" "" "" "" "" "${status}" "${notes}" "${row_json}"
    append_csv_row "${row_json}"
    [[ "${CONTINUE_ON_FAIL}" == "1" ]] || exit "${ec1}"
    continue
  fi

  if [[ -z "${run_dir:-}" ]]; then
    run_dir="$(resolve_run_dir "${step_log}" "${env_name}" "${SEED}")"
  fi
  if [[ -z "${run_dir}" ]]; then
    status="failed"
    notes="could not resolve run_dir"
    write_result_row "${env_name}" "${cfg}" "${alpha}" "${gap}" "${kappa}" "${discount}" "${batch_size}" "${SEED}" \
      "" "false" "" "" "" "false" "" "" "" "" "" "" "${status}" "${notes}" "${row_json}"
    append_csv_row "${row_json}"
    [[ "${CONTINUE_ON_FAIL}" == "1" ]] || exit 3
    continue
  fi

  "${PYTHON_BIN}" "${MET_PY}" parse-metrics --run-dir "${run_dir}" --upto-epoch 200 >"${tmp200}"
  ok200="$("${PYTHON_BIN}" -c "import json,sys; print('true' if json.load(open(sys.argv[1]))['ok'] else 'false')" "${tmp200}")"
  metric_name="$("${PYTHON_BIN}" -c "import json,sys; v=json.load(open(sys.argv[1])).get('metric_name'); print('' if v is None else v)" "${tmp200}")"
  best200="$("${PYTHON_BIN}" -c "import json,sys; v=json.load(open(sys.argv[1])).get('best'); print('' if v is None else v)" "${tmp200}")"
  m200="$("${PYTHON_BIN}" -c "import json,sys; v=json.load(open(sys.argv[1])).get('at_cutoff'); print('' if v is None else v)" "${tmp200}")"
  idm_at_200="$("${PYTHON_BIN}" -c "import json,sys; v=json.load(open(sys.argv[1])).get('idm_at_cutoff'); print('' if v is None else v)" "${tmp200}")"
  act_at_200="$("${PYTHON_BIN}" -c "import json,sys; v=json.load(open(sys.argv[1])).get('actor_at_cutoff'); print('' if v is None else v)" "${tmp200}")"

  if [[ "${ok200}" == "true" ]]; then
    stage1_ok="true"
  else
    status="failed"
    notes="metrics parse failed: $( "${PYTHON_BIN}" -c "import json,sys; print(json.load(open(sys.argv[1])).get('reason',''))" "${tmp200}" )"
  fi

  write_result_row "${env_name}" "${cfg}" "${alpha}" "${gap}" "${kappa}" "${discount}" "${batch_size}" "${SEED}" \
    "${run_dir}" "${stage1_ok}" "${metric_name}" "${best200}" "${m200}" "false" "" "" \
    "${idm_at_200}" "${act_at_200}" "" "" "${status}" "${notes}" "${row_json}"
  append_csv_row "${row_json}"

  if [[ "${status}" == "failed" ]] && [[ "${CONTINUE_ON_FAIL}" != "1" ]]; then
    exit 4
  fi
done

echo "================================================================================"
echo "[phase2] finished. CSV: ${CSV_OUT}"
echo "[phase2] Next: bash scripts/resume_grid_top3_idm.sh"
