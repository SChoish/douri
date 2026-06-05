#!/usr/bin/env bash
# Resume alpha=0 puzzle grid runs to 600 epochs.
# Skips runs whose last-eval IDM and Actor means are both 0%.
#
#   CUDA_VISIBLE_DEVICES=0 nohup bash scripts/with_jax_cuda.sh \
#     bash scripts/resume_alpha0_to_600.sh >> alpha0_resume600.master.log 2>&1 &

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
CSV_PATH="${CSV_PATH:-${ROOT}/sweep_results/puzzle_fbr_displacement_grid.csv}"
DRY_RUN="${DRY_RUN:-0}"
CONTINUE_ON_FAIL="${CONTINUE_ON_FAIL:-1}"

TS="$(date +%Y%m%d_%H%M%S)"
LAUNCH_LOG_ROOT="${ROOT}/launch_logs/alpha0_resume600/${TS}"
mkdir -p "${LAUNCH_LOG_ROOT}"

plan_json="${LAUNCH_LOG_ROOT}/plan.json"
"${PYTHON_BIN}" - "${CSV_PATH}" "${plan_json}" <<'PY'
import csv, json, math, re, sys
from pathlib import Path

csv_path, out_path = Path(sys.argv[1]), Path(sys.argv[2])
rows = [
    r for r in csv.DictReader(open(csv_path))
    if abs(float(r['alpha']) - 0.0) < 1e-9 and abs(float(r['gap']) - 0.0) > 1e-9
]

def max_trained_epoch(run_dir: Path) -> int:
    ep = 0
    for logf in [run_dir / 'run.log'] + sorted(run_dir.glob('run_resume_from*.log')):
        if logf.exists():
            for m in re.finditer(r'\|\s*INFO\s*\|\s*epoch=(\d+)\s', logf.read_text(errors='replace')):
                ep = max(ep, int(m.group(1)))
    ck_root = run_dir / 'checkpoints'
    if ck_root.exists():
        for ck in ck_root.rglob('params_*.pkl'):
            m = re.search(r'params_(\d+)\.pkl', ck.name)
            if m:
                ep = max(ep, int(m.group(1)))
    return ep

def last_eval_means(run_dir: Path) -> tuple[int, float, float]:
    pat_start = re.compile(r'===\s*EVAL\s+START\s+epoch=(\d+)')
    pat_idm = re.compile(r'idm\s+env_success_rate_mean=([\d.]+)')
    pat_actor = re.compile(r'actor\s+env_success_rate_mean=([\d.]+)')
    last_ep = -1
    idm = actor = float('nan')
    for logf in sorted([run_dir / 'run.log'] + list(run_dir.glob('run_resume_from*.log'))):
        if not logf.exists():
            continue
        cur = None
        for line in logf.read_text(errors='replace').splitlines():
            m = pat_start.search(line)
            if m:
                cur = int(m.group(1))
            if cur is None or cur < 0:
                continue
            m2, m3 = pat_idm.search(line), pat_actor.search(line)
            if m2 or m3:
                if cur > last_ep:
                    last_ep, idm, actor = cur, float('nan'), float('nan')
                if cur == last_ep:
                    if m2:
                        idm = float(m2.group(1))
                    if m3:
                        actor = float(m3.group(1))
    return last_ep, idm, actor

def has_ckpt(run_dir: Path, epoch: int) -> bool:
    ck = run_dir / 'checkpoints'
    if not ck.exists():
        return False
    for sub in ('dynamics', 'critic', 'actor'):
        if (ck / sub / f'params_{epoch}.pkl').is_file():
            return True
    return any(ck.rglob(f'params_{epoch}.pkl'))

resume, skip_zero, skip_gap0, skip_other = [], [], [], []
for r in rows:
    rd = Path(r['run_dir'])
    if not rd.is_absolute():
        rd = Path('.') / rd
    name = Path(r['config_path']).name
    if not rd.is_dir():
        skip_other.append({'name': name, 'reason': 'no run_dir'})
        continue
    max_ep = max_trained_epoch(rd)
    if max_ep < 200:
        skip_other.append({'name': name, 'reason': f'incomplete max_ep={max_ep}'})
        continue
    last_ep, idm, act = last_eval_means(rd)
    if (math.isnan(idm) or idm <= 1e-9) and (math.isnan(act) or act <= 1e-9):
        skip_zero.append({'name': name, 'last_eval_ep': last_ep})
        continue
    resume_from = 400 if max_ep >= 400 else 200
    if not has_ckpt(rd, resume_from):
        skip_other.append({'name': name, 'reason': f'no ckpt @ {resume_from}'})
        continue
    resume.append({
        'config_path': str(Path(r['config_path']).resolve()),
        'run_dir': str(rd.resolve()),
        'resume_epoch': resume_from,
        'max_ep': max_ep,
        'last_eval_ep': last_ep,
        'idm_last': idm,
        'actor_last': act,
        'name': name,
    })

out = {'resume': resume, 'skip_zero': skip_zero, 'skip_gap0': skip_gap0, 'skip_other': skip_other}
out_path.write_text(json.dumps(out, indent=2))
print(json.dumps({k: len(v) for k, v in out.items()}))
PY

echo "[alpha0-600] plan: ${plan_json}"
echo "[alpha0-600] target_epochs=${TARGET_EPOCHS}"

n_resume="$("${PYTHON_BIN}" -c "import json; print(len(json.load(open('${plan_json}'))['resume']))")"
if [[ "${n_resume}" -eq 0 ]]; then
  echo "[alpha0-600] ERROR: nothing to resume" >&2
  exit 2
fi

idx=0
while IFS= read -r item; do
  idx=$((idx + 1))
  cfg="$("${PYTHON_BIN}" -c "import json,sys; print(json.loads(sys.argv[1])['config_path'])" "${item}")"
  run_dir="$("${PYTHON_BIN}" -c "import json,sys; print(json.loads(sys.argv[1])['run_dir'])" "${item}")"
  resume_ep="$("${PYTHON_BIN}" -c "import json,sys; print(json.loads(sys.argv[1])['resume_epoch'])" "${item}")"
  name="$("${PYTHON_BIN}" -c "import json,sys; print(json.loads(sys.argv[1])['name'])" "${item}")"
  base="$(basename "${cfg}" .yaml)"
  step_log="${LAUNCH_LOG_ROOT}/${base}_resume${TARGET_EPOCHS}.log"

  echo "================================================================================"
  echo "[alpha0-600] (${idx}/${n_resume}) ${name} resume@${resume_ep} -> ${TARGET_EPOCHS}"
  echo "[alpha0-600] run_dir=${run_dir}"

  already="$("${PYTHON_BIN}" -c "
import re, sys
from pathlib import Path
rd = Path(sys.argv[1])
target = int(sys.argv[2])
ep = 0
for logf in [rd/'run.log'] + sorted(rd.glob('run_resume_from*.log')):
    if logf.is_file():
        for m in re.finditer(r'\|\s*INFO\s*\|\s*epoch=(\d+)\s', logf.read_text(errors='replace')):
            ep = max(ep, int(m.group(1)))
print('yes' if ep >= target else 'no')
" "${run_dir}" "${TARGET_EPOCHS}")"
  if [[ "${already}" == "yes" ]]; then
    echo "[alpha0-600] SKIP already at epoch >= ${TARGET_EPOCHS}: ${run_dir}"
    continue
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[alpha0-600] DRY_RUN=1: skip train"
    continue
  fi

  set +e
  "${PYTHON_BIN}" main.py \
    --run_config="${cfg}" \
    --runs_root="${RUNS_ROOT}" \
    --seed="${SEED}" \
    --train_epochs="${TARGET_EPOCHS}" \
    --resume_run_dir="${run_dir}" \
    --resume_epoch="${resume_ep}" \
    >>"${step_log}" 2>&1
  ec=$?
  set -e
  if [[ "${ec}" -ne 0 ]]; then
    echo "[alpha0-600] FAILED exit=${ec}; log=${step_log}" >&2
    if [[ "${CONTINUE_ON_FAIL}" != "1" ]]; then
      exit "${ec}"
    fi
    continue
  fi
  echo "[alpha0-600] OK -> ${TARGET_EPOCHS}; log=${step_log}"
done < <("${PYTHON_BIN}" -c "
import json, sys
for row in json.load(open(sys.argv[1]))['resume']:
    print(json.dumps(row))
" "${plan_json}")

echo "================================================================================"
echo "[alpha0-600] 3x3 done. logs: ${LAUNCH_LOG_ROOT}"
echo "[alpha0-600] chaining -> sweep_alpha0_4x4_to_600.sh (alpha=0, gap>0)"
exec bash "${ROOT}/scripts/sweep_alpha0_4x4_to_600.sh"
