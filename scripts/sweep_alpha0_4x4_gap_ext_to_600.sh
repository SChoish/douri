#!/usr/bin/env bash
# Puzzle 4x4 alpha=0: gap in {0,50,100}, kappa descending 0.9->0.6.
# Stage1 200ep, resume to 600 if metric at 200 >= best upto 200.
#
#   CUDA_VISIBLE_DEVICES=0 nohup bash scripts/with_jax_cuda.sh \
#     bash scripts/sweep_alpha0_4x4_gap_ext_to_600.sh >> alpha0_4x4_gap_ext_sweep600.master.log 2>&1 &

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=".:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUNS_ROOT="${RUNS_ROOT:-runs}"
SEED="${SEED:-0}"
TARGET_EPOCHS="${TARGET_EPOCHS:-600}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-200}"
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-1}"
MET_PY="${ROOT}/scripts/puzzle_fbr_displacement_grid_metrics.py"
GEN_PY="${ROOT}/scripts/generate_grid_fbr_displacement_puzzle_4x4_gap_ext_configs.py"
CSV_OUT="${ROOT}/sweep_results/puzzle_fbr_displacement_grid_alpha0_4x4_gap_ext_600.csv"

TS="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG_ROOT="${ROOT}/launch_logs/alpha0_4x4_gap_ext_sweep600/${TS}"
mkdir -p "${LAUNCH_LOG_ROOT}" "${ROOT}/sweep_results"

"${PYTHON_BIN}" "${GEN_PY}"

# gap order: 0, 50, 100 · kappa order: 0.9 -> 0.6
GAPS=(0p0 50p0 100p0)
KAPPAS=(0p9 0p8 0p7 0p6)
CONFIGS=()
for g in "${GAPS[@]}"; do
  for k in "${KAPPAS[@]}"; do
    cfg="${ROOT}/config/grid_fbr_displacement_puzzle/puzzle_4x4_a0p0_gap${g}_k${k}.yaml"
    if [[ ! -f "${cfg}" ]]; then
      echo "[4x4-gap-ext] missing config: ${cfg}" >&2
      exit 1
    fi
    CONFIGS+=("${cfg}")
  done
done
NCFG="${#CONFIGS[@]}"
echo "[4x4-gap-ext] ${NCFG} configs (alpha=0, gap in {0,50,100}, kappa 0.9->0.6)"

resume_stage2_wanted() {
  "${PYTHON_BIN}" -c "import json,sys,math; d=json.load(open(sys.argv[1]))
if not d.get('ok'): raise SystemExit(1)
a,b=d.get('at_cutoff'),d.get('best')
if a is None or b is None: raise SystemExit(1)
a,b=float(a),float(b)
if math.isnan(a) or math.isnan(b): raise SystemExit(1)
raise SystemExit(0 if a+1e-8>=b else 1)" "$1"
}

append_csv_row() {
  "${PYTHON_BIN}" "${MET_PY}" append-row --csv "${CSV_OUT}" --json "$1"
}

write_result_row() {
  export GRID_ROW_ENV_NAME="$1" GRID_ROW_CONFIG_PATH="$2" GRID_ROW_ALPHA="$3"
  export GRID_ROW_GAP="$4" GRID_ROW_KAPPA="$5" GRID_ROW_DISCOUNT="$6"
  export GRID_ROW_BATCH="$7" GRID_ROW_SEED="$8" GRID_ROW_RUN_DIR="$9"
  export GRID_ROW_STAGE1="${10}" GRID_ROW_METRIC_NAME="${11}" GRID_ROW_B200="${12}"
  export GRID_ROW_M200="${13}" GRID_ROW_CONT400="${14}" GRID_ROW_B400="${15}"
  export GRID_ROW_F400="${16}" GRID_ROW_IDM200="${17}" GRID_ROW_ACT200="${18}"
  export GRID_ROW_IDM400="${19}" GRID_ROW_ACT400="${20}" GRID_ROW_STATUS="${21}"
  export GRID_ROW_NOTES="${22}" GRID_ROW_JSON_OUT="${23}"
  "${PYTHON_BIN}" - <<'PY'
import json, os
row = {k.replace('GRID_ROW_','').lower(): os.environ[k] for k in os.environ if k.startswith('GRID_ROW_') and k != 'GRID_ROW_JSON_OUT'}
out = {
  'env_name': row.get('env_name',''),
  'config_path': row.get('config_path',''),
  'alpha': row.get('alpha',''),
  'gap': row.get('gap',''),
  'kappa': row.get('kappa',''),
  'discount': row.get('discount',''),
  'batch_size': row.get('batch',''),
  'seed': row.get('seed',''),
  'run_dir': row.get('run_dir',''),
  'stage1_completed': row.get('stage1',''),
  'metric_name': row.get('metric_name',''),
  'best_metric_upto_200': row.get('b200',''),
  'metric_at_200': row.get('m200',''),
  'continued_to_400': row.get('cont400',''),
  'best_metric_upto_400': row.get('b400',''),
  'final_metric_400': row.get('f400',''),
  'idm_best_upto_200': row.get('idm200',''),
  'actor_best_upto_200': row.get('act200',''),
  'idm_final_400': row.get('idm400',''),
  'actor_final_400': row.get('act400',''),
  'status': row.get('status',''),
  'notes': row.get('notes',''),
}
with open(os.environ['GRID_ROW_JSON_OUT'], 'w', encoding='utf-8') as f:
    json.dump(out, f)
PY
}

read_yaml_top() {
  "${PYTHON_BIN}" -c "import yaml,sys; d=yaml.safe_load(open(sys.argv[1])); print(d.get(sys.argv[2],''))" "$1" "$2"
}

read_yaml_critic() {
  "${PYTHON_BIN}" -c "import yaml,sys; d=yaml.safe_load(open(sys.argv[1])); c=d.get('critic_agent')or{}; print(c.get(sys.argv[2],''))" "$1" "$2"
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

cfg_idx=0
for cfg in "${CONFIGS[@]}"; do
  cfg_idx=$((cfg_idx + 1))
  base="$(basename "${cfg}" .yaml)"
  step_log="${LAUNCH_LOG_ROOT}/${base}.log"
  echo "================================================================================"
  echo "[4x4-gap-ext] (${cfg_idx}/${NCFG}) ${cfg}"

  env_name="$(read_yaml_top "${cfg}" env_name)"
  discount="$(read_yaml_critic "${cfg}" discount)"
  batch_size="$(read_yaml_top "${cfg}" batch_size)"
  nums="$(parse_cfg_nums "${cfg}")"
  alpha="$(echo "${nums}" | "${PYTHON_BIN}" -c "import json,sys; print(json.load(sys.stdin)['alpha'])")"
  gap="$(echo "${nums}" | "${PYTHON_BIN}" -c "import json,sys; print(json.load(sys.stdin)['gap'])")"
  kappa="$(echo "${nums}" | "${PYTHON_BIN}" -c "import json,sys; print(json.load(sys.stdin)['kappa'])")"

  tmp200="${LAUNCH_LOG_ROOT}/${base}_m200.json"
  tmp600="${LAUNCH_LOG_ROOT}/${base}_m600.json"
  row_json="${LAUNCH_LOG_ROOT}/${base}_row.json"

  set +e
  "${PYTHON_BIN}" main.py \
    --run_config="${cfg}" \
    --runs_root="${RUNS_ROOT}" \
    --seed="${SEED}" \
    --train_epochs="${STAGE1_EPOCHS}" \
    >>"${step_log}" 2>&1
  ec1=$?
  set -e
  if [[ "${ec1}" -ne 0 ]]; then
    echo "[4x4-gap-ext] stage1 FAILED exit=${ec1}"
    [[ "${CONTINUE_ON_FAIL}" == "1" ]] || exit "${ec1}"
    continue
  fi

  run_dir="$(find "${RUNS_ROOT}" -maxdepth 1 -type d -name "*_seed${SEED}_${env_name}" -printf '%T@\t%p\n' 2>/dev/null | sort -nr | head -1 | cut -f2-)"
  "${PYTHON_BIN}" "${MET_PY}" parse-metrics --run-dir "${run_dir}" --upto-epoch "${STAGE1_EPOCHS}" >"${tmp200}"

  cont600="false"
  idm600="" act600="" b600="" f600=""
  if resume_stage2_wanted "${tmp200}"; then
    set +e
    "${PYTHON_BIN}" main.py \
      --run_config="${cfg}" \
      --runs_root="${RUNS_ROOT}" \
      --seed="${SEED}" \
      --train_epochs="${TARGET_EPOCHS}" \
      --resume_run_dir="${run_dir}" \
      --resume_epoch="${STAGE1_EPOCHS}" \
      >>"${step_log}" 2>&1
    ec2=$?
    set -e
    if [[ "${ec2}" -eq 0 ]]; then
      cont600="true"
      "${PYTHON_BIN}" "${MET_PY}" parse-metrics --run-dir "${run_dir}" --upto-epoch "${TARGET_EPOCHS}" >"${tmp600}"
      b600="$("${PYTHON_BIN}" -c "import json; print(json.load(open('${tmp600}'))['best'])")"
      f600="$("${PYTHON_BIN}" -c "import json; print(json.load(open('${tmp600}'))['at_cutoff'])")"
      idm600="$("${PYTHON_BIN}" -c "import json; print(json.load(open('${tmp600}'))['idm_at_cutoff'])")"
      act600="$("${PYTHON_BIN}" -c "import json; print(json.load(open('${tmp600}'))['actor_at_cutoff'])")"
    fi
  fi

  metric_name="$("${PYTHON_BIN}" -c "import json; print(json.load(open('${tmp200}'))['metric_name'])")"
  b200="$("${PYTHON_BIN}" -c "import json; print(json.load(open('${tmp200}'))['best'])")"
  m200="$("${PYTHON_BIN}" -c "import json; print(json.load(open('${tmp200}'))['at_cutoff'])")"
  idm200="$("${PYTHON_BIN}" -c "import json; print(json.load(open('${tmp200}'))['idm_best'])")"
  act200="$("${PYTHON_BIN}" -c "import json; print(json.load(open('${tmp200}'))['actor_best'])")"

  write_result_row "${env_name}" "${cfg}" "${alpha}" "${gap}" "${kappa}" "${discount}" "${batch_size}" "${SEED}" \
    "${run_dir}" "true" "${metric_name}" "${b200}" "${m200}" "${cont600}" "${b600}" "${f600}" \
    "${idm200}" "${act200}" "${idm600}" "${act600}" "ok" "" "${row_json}"
  append_csv_row "${row_json}"
done

echo "[4x4-gap-ext] done. CSV=${CSV_OUT}"
