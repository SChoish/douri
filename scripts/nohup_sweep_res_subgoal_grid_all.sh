#!/usr/bin/env bash
# Antmaze (24) then cube (24) residual×subgoal grids — for nohup background launch.
#
#   cd /home/offrl/Pathbridger
#   export PYTHONPATH=.
#   nohup bash scripts/nohup_sweep_res_subgoal_grid_all.sh \
#     > nohup_logs/sweep_res_subgoal_all_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#   echo $!

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH=.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHON_BIN="${PYTHON_BIN:-/home/offrl/miniconda3/envs/offrl/bin/python}"
export DOURI_ROOT="${DOURI_ROOT:-$REPO_ROOT/../douri}"
export RUNS_ROOT="${RUNS_ROOT:-$REPO_ROOT/runs}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-600}"

echo "[$(date -Is)] nohup master start repo=$REPO_ROOT python=$PYTHON_BIN"
"$PYTHON_BIN" -c "import jax; print('jax', jax.default_backend(), jax.devices())"

echo "[$(date -Is)] === antmaze sweep (24 runs) ==="
bash "$REPO_ROOT/scripts/sweep_antmaze_res_subgoal_grid_600ep.sh"

echo "[$(date -Is)] === cube sweep (24 runs) ==="
bash "$REPO_ROOT/scripts/sweep_cube_res_subgoal_grid_600ep.sh"

echo "[$(date -Is)] nohup master done"
