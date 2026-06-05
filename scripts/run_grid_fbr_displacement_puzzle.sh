#!/usr/bin/env bash
# Sequential puzzle grid: train 200 epochs, then optionally resume to 400 by eval metric rule.
# From repo root: bash scripts/run_grid_fbr_displacement_puzzle.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=".:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUNS_ROOT="${RUNS_ROOT:-runs}"
SEED="${SEED:-0}"
GRID_START_INDEX="${GRID_START_INDEX:-1}"
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-0}"
DRY_RUN="${DRY_RUN:-0}"
if [[ "${DRY_RUN}" == "1" ]]; then
  SKIP_GENERATE="${SKIP_GENERATE:-1}"
else
  SKIP_GENERATE="${SKIP_GENERATE:-0}"
fi

CONFIG_GLOB_ORDER=(
  "${ROOT}/config/grid_fbr_displacement_puzzle/puzzle_3x3_"*.yaml
  "${ROOT}/config/grid_fbr_displacement_puzzle/puzzle_4x4_"*.yaml
  "${ROOT}/config/grid_fbr_displacement_puzzle/puzzle_4x6_"*.yaml
)

TS="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG_ROOT="${ROOT}/launch_logs/grid_fbr_displacement_puzzle/${TS}"
CSV_OUT="${ROOT}/sweep_results/puzzle_fbr_displacement_grid.csv"
MET_PY="${ROOT}/scripts/puzzle_fbr_displacement_grid_metrics.py"
GEN_PY="${ROOT}/scripts/generate_grid_fbr_displacement_puzzle_configs.py"

mkdir -p "${LAUNCH_LOG_ROOT}" "${ROOT}/sweep_results"

if [[ "${SKIP_GENERATE}" != "1" ]]; then
  echo "[grid] generating configs..."
  "${PYTHON_BIN}" "${GEN_PY}"
fi

mapfile -t CONFIGS < <(printf '%s\n' "${CONFIG_GLOB_ORDER[@]}" | sort)
NCFG="${#CONFIGS[@]}"
echo "[grid] found ${NCFG} config files"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[grid] DRY_RUN=1: execution order (first 12):"
  printf '  %s\n' "${CONFIGS[@]:0:12}"
  echo "  ... total ${NCFG}"
  exit 0
fi

read_yaml_top() {
  local file="$1" key="$2"
  "${PYTHON_BIN}" -c "import yaml,sys; d=yaml.safe_load(open(sys.argv[1])); print(d.get(sys.argv[2],''))" "$file" "$key"
}

read_yaml_critic() {
  local file="$1" key="$2"
  "${PYTHON_BIN}" -c "import yaml,sys; d=yaml.safe_load(open(sys.argv[1])); c=d.get('critic_agent')or{}; print(c.get(sys.argv[2],''))" "$file" "$key"
}

validate_4x6_discount() {
  local cfg="$1"
  local base
  base="$(basename "$cfg")"
  [[ "${base}" == puzzle_4x6_* ]] || return 0
  local disc
  disc="$(read_yaml_critic "$cfg" discount)"
  local ok
  ok="$("${PYTHON_BIN}" -c "import sys; print('1' if abs(float(sys.argv[1])-0.995)<1e-9 else '0')" "${disc}")"
  if [[ "${ok}" != "1" ]]; then
    echo "[grid] ERROR: ${base} must have critic_agent.discount==0.995 (got ${disc})" >&2
    return 1
  fi
  echo "[grid] OK discount=0.995 for ${base}"
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
  # prints JSON one line: alpha,gap,kappa
  "${PYTHON_BIN}" -c "
import re, json, sys
from pathlib import Path
p = Path(sys.argv[1])
m = re.match(r'^puzzle_(3x3|4x4|4x6)_a(?P<a>[^_]+)_gap(?P<g>[^_]+)_k(?P<k>[^.]+)\\.yaml\$', p.name)
if not m:
    raise SystemExit('bad config name: '+p.name)
dec = lambda s: float(s.replace('p','.').replace('m','-'))
print(json.dumps({'alpha': dec(m.group('a')), 'gap': dec(m.group('g')), 'kappa': dec(m.group('k'))}))
" "$1"
}

resume_stage2_wanted() {
  "${PYTHON_BIN}" -c "import json,sys,math; d=json.load(open(sys.argv[1]))
if not d.get('ok'):
    raise SystemExit(1)
a=d.get('at_cutoff'); b=d.get('best')
if a is None or b is None:
    raise SystemExit(1)
a=float(a); b=float(b)
if math.isnan(a) or math.isnan(b):
    raise SystemExit(1)
raise SystemExit(0 if a+1e-8>=b else 1)" "$1"
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
  [[ -e "$cfg" ]] || continue
  cfg_idx=$((cfg_idx + 1))
  if [[ "${cfg_idx}" -lt "${GRID_START_INDEX}" ]]; then
    echo "[grid] SKIP index ${cfg_idx}/${NCFG} (GRID_START_INDEX=${GRID_START_INDEX}): ${cfg}"
    continue
  fi
  base="$(basename "$cfg" .yaml)"
  step_log="${LAUNCH_LOG_ROOT}/${base}.log"
  echo "================================================================================"
  echo "[grid] config=${cfg} (${cfg_idx}/${NCFG})"
  validate_4x6_discount "$cfg" || exit 2

  env_name="$(read_yaml_top "$cfg" env_name)"
  discount="$(read_yaml_critic "$cfg" discount)"
  batch_size="$(read_yaml_top "$cfg" batch_size)"
  nums="$(parse_cfg_nums "$cfg")"
  alpha="$(echo "$nums" | "${PYTHON_BIN}" -c "import json,sys; print(json.load(sys.stdin)['alpha'])")"
  gap="$(echo "$nums" | "${PYTHON_BIN}" -c "import json,sys; print(json.load(sys.stdin)['gap'])")"
  kappa="$(echo "$nums" | "${PYTHON_BIN}" -c "import json,sys; print(json.load(sys.stdin)['kappa'])")"

  tmp200="${LAUNCH_LOG_ROOT}/${base}_m200.json"
  tmp400="${LAUNCH_LOG_ROOT}/${base}_m400.json"
  row_json="${LAUNCH_LOG_ROOT}/${base}_row.json"

  status="ok"
  notes=""
  run_dir=""
  stage1_ok="false"
  metric_name=""
  best200=""
  m200=""
  cont400="false"
  best400=""
  fin400=""
  idm200b=""
  act200b=""
  idm400f=""
  act400f=""

  ec1=0
  if existing_run="$(stage1_run_dir_for_config "$cfg" "$env_name")"; then
    echo "[grid] SKIP stage1 train (already complete): ${existing_run}" | tee -a "${step_log}"
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
    notes="stage1 main.py exit=${ec1}; log=${step_log}"
    run_dir="$(resolve_run_dir "${step_log}" "${env_name}" "${SEED}" || true)"
    write_result_row "${env_name}" "${cfg}" "${alpha}" "${gap}" "${kappa}" "${discount}" "${batch_size}" "${SEED}" \
      "${run_dir}" "false" "" "" "" "false" "" "" "" "" "" "" "${status}" "${notes}" "${row_json}"
    append_csv_row "${row_json}"
    if [[ "${CONTINUE_ON_FAIL}" != "1" ]]; then
      exit "${ec1}"
    fi
    continue
  fi

  if [[ -z "${run_dir:-}" ]]; then
    run_dir="$(resolve_run_dir "${step_log}" "${env_name}" "${SEED}")"
  fi
  if [[ -z "${run_dir}" ]]; then
    status="failed"
    notes="could not resolve run_dir; log=${step_log}"
    write_result_row "${env_name}" "${cfg}" "${alpha}" "${gap}" "${kappa}" "${discount}" "${batch_size}" "${SEED}" \
      "" "false" "" "" "" "false" "" "" "" "" "" "" "${status}" "${notes}" "${row_json}"
    append_csv_row "${row_json}"
    [[ "${CONTINUE_ON_FAIL}" == "1" ]] || exit 3
    continue
  fi

  train_csv="${run_dir}/train.csv"
  "${PYTHON_BIN}" "${MET_PY}" parse-metrics --run-dir "${run_dir}" --upto-epoch 200 >"${tmp200}"

  metric_src="$("${PYTHON_BIN}" -c "import json,sys; print(json.load(open(sys.argv[1])).get('source',''))" "${tmp200}")"

  metric_name="$("${PYTHON_BIN}" -c "import json,sys; v=json.load(open(sys.argv[1]))['metric_name']; print('' if v is None else v)" "${tmp200}")"
  best200="$("${PYTHON_BIN}" -c "import json,sys; v=json.load(open(sys.argv[1]))['best']; print('' if v is None else v)" "${tmp200}")"
  m200="$("${PYTHON_BIN}" -c "import json,sys; v=json.load(open(sys.argv[1]))['at_cutoff']; print('' if v is None else v)" "${tmp200}")"
  idm200b="$("${PYTHON_BIN}" -c "import json,sys; v=json.load(open(sys.argv[1]))['idm_best']; print('' if v is None else v)" "${tmp200}")"
  act200b="$("${PYTHON_BIN}" -c "import json,sys; v=json.load(open(sys.argv[1]))['actor_best']; print('' if v is None else v)" "${tmp200}")"
  ok200="$("${PYTHON_BIN}" -c "import json,sys; print('true' if json.load(open(sys.argv[1]))['ok'] else 'false')" "${tmp200}")"

  if [[ "${ok200}" == "true" ]]; then
    stage1_ok="true"
  else
    notes="stage1 parse: $( "${PYTHON_BIN}" -c "import json,sys; print(json.load(open(sys.argv[1])).get('reason',''))" "${tmp200}" ) source=${metric_src}"
  fi

  if [[ "${stage1_ok}" == "true" ]] && resume_stage2_wanted "${tmp200}"; then
    set +e
    "${PYTHON_BIN}" main.py \
      --run_config="${cfg}" \
      --runs_root="${RUNS_ROOT}" \
      --seed="${SEED}" \
      --train_epochs=400 \
      --resume_run_dir="${run_dir}" \
      --resume_epoch=200 \
      >>"${step_log}" 2>&1
    ec2=$?
    set -e
    if [[ "${ec2}" -ne 0 ]]; then
      status="failed"
      notes="stage2 resume exit=${ec2}; log=${step_log}"
    else
      cont400="true"
      "${PYTHON_BIN}" "${MET_PY}" parse-metrics --run-dir "${run_dir}" --upto-epoch 400 >"${tmp400}"
      best400="$("${PYTHON_BIN}" -c "import json,sys; print(json.load(open(sys.argv[1]))['best'])" "${tmp400}")"
      fin400="$("${PYTHON_BIN}" -c "import json,sys; print(json.load(open(sys.argv[1]))['at_cutoff'])" "${tmp400}")"
      idm400f="$("${PYTHON_BIN}" -c "import json,sys; print(json.load(open(sys.argv[1]))['idm_at_cutoff'])" "${tmp400}")"
      act400f="$("${PYTHON_BIN}" -c "import json,sys; print(json.load(open(sys.argv[1]))['actor_at_cutoff'])" "${tmp400}")"
    fi
  fi

  write_result_row "${env_name}" "${cfg}" "${alpha}" "${gap}" "${kappa}" "${discount}" "${batch_size}" "${SEED}" \
    "${run_dir}" "${stage1_ok}" "${metric_name}" "${best200}" "${m200}" "${cont400}" "${best400}" "${fin400}" \
    "${idm200b}" "${act200b}" "${idm400f}" "${act400f}" "${status}" "${notes}" "${row_json}"
  append_csv_row "${row_json}"

  if [[ "${status}" == "failed" ]] && [[ "${CONTINUE_ON_FAIL}" != "1" ]]; then
    exit 4
  fi

done

echo "================================================================================"
echo "[grid] sweep finished. CSV: ${CSV_OUT}"
echo "[grid] per-config logs: ${LAUNCH_LOG_ROOT}"
