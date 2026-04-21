#!/usr/bin/env bash
# GOUB dynamics (1000 ep, embedded IDM in main loop) for antmaze envs that
# do not yet have a matching run folder under runs/.
#
# After each env finishes training: deterministic state rollout + inv-dyn rollout
# (planner_noise_scale=0, action_chunk_horizon=10) into the same run_dir.
#
# Usage (from douri repo root):
#   nohup bash scripts/run_goub_dynamics_antmaze_missing_1000_rollouts.sh &
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMPL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${IMPL_DIR}"

RUNS_DIR="${IMPL_DIR}/runs"
mkdir -p "${RUNS_DIR}"
LOG_FILE="${RUNS_DIR}/nohup_goub_dynamics_missing_rollouts_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "log: ${LOG_FILE}"

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

ALL_ENVS=(
  antmaze-medium-navigate-v0
  antmaze-large-navigate-v0
  antmaze-giant-navigate-v0
  antmaze-teleport-navigate-v0
  antmaze-medium-stitch-v0
  antmaze-large-stitch-v0
  antmaze-giant-stitch-v0
  antmaze-teleport-stitch-v0
)

has_dynamics_run() {
  local env="$1"
  compgen -G "${RUNS_DIR}"/*_goub_dynamics_seed0_"${env}" >/dev/null
}

latest_dynamics_run_dir() {
  local env="$1"
  ls -dt "${RUNS_DIR}"/*_goub_dynamics_seed0_"${env}" 2>/dev/null | head -1
}

rollout_artifacts() {
  local R="$1"
  echo "--- rollouts for ${R} ---"
  python rollout_subgoal_goub.py \
    --run_dir="${R}" \
    --checkpoint_epoch=1000 \
    --traj_idx=0 \
    --max_steps=1000 \
    --out_path="${R}/deterministic_state_rollout.png" \
    --fps=60
  python rollout_idm_goub.py \
    --run_dir="${R}" \
    --checkpoint_epoch=1000 \
    --traj_idx=0 \
    --max_steps=1000 \
    --action_chunk_horizon=10 \
    --planner_noise_scale=0 \
    --out_path="${R}/deterministic_inv_dyn_rollout_goub1000_freq10.png" \
    --fps=60
}

for env in "${ALL_ENVS[@]}"; do
  if has_dynamics_run "${env}"; then
    echo "========== SKIP (dynamics run exists): ${env} =========="
    continue
  fi
  echo "========== TRAIN ${env} $(date -Is) =========="
  python main_goub_dynamics.py \
    --env_name="${env}" \
    --train_epochs=1000 \
    --save_every_n_epochs=100 \
    --log_every_n_epochs=10 \
    --use_tqdm=false

  R="$(latest_dynamics_run_dir "${env}")"
  if [[ -z "${R}" || ! -d "${R}" ]]; then
    echo "ERROR: could not resolve run_dir for ${env}"
    exit 1
  fi
  if [[ ! -f "${R}/checkpoints/params_1000.pkl" ]]; then
    echo "ERROR: missing ${R}/checkpoints/params_1000.pkl"
    exit 1
  fi
  rollout_artifacts "${R}"
  echo "========== DONE ${env} $(date -Is) =========="
done

echo "ALL DONE $(date -Is)"
