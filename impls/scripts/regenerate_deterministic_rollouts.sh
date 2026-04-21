#!/usr/bin/env bash
# Reproduce the 2026-04-17 antmaze Phase-1 rollout artifacts:
#   - deterministic_state_rollout.{png,mp4}  → GOUB only (iterative), checkpoints/params_<EPOCH>.pkl
#   - deterministic_inv_dyn_rollout_freq<F>.{png,mp4} → chunked_idm (idm_net inside GOUB ckpt, or legacy idm/checkpoints)
#
# Usage:
#   cd impls && bash scripts/regenerate_deterministic_rollouts.sh
#   EPOCH=500 bash scripts/regenerate_deterministic_rollouts.sh
#   SKIP_IDM=1 bash scripts/regenerate_deterministic_rollouts.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EPOCH="${EPOCH:-1000}"
IDM_FREQ="${IDM_FREQ:-1}"
SKIP_IDM="${SKIP_IDM:-0}"
TRAJ_IDX="${TRAJ_IDX:-0}"
MAX_STEPS="${MAX_STEPS:-1000}"
FPS="${FPS:-60}"

if [[ -x "${CONDA_PREFIX:-}/bin/python" ]] && "${CONDA_PREFIX}/bin/python" -c "import flax" 2>/dev/null; then
  PY="${CONDA_PREFIX}/bin/python"
elif [[ -x /home/offrl/miniconda3/envs/offrl/bin/python ]]; then
  PY=/home/offrl/miniconda3/envs/offrl/bin/python
else
  PY=python3
fi

RUNS=(
  runs/20260417_034835_seed0_antmaze-medium-navigate-v0
  runs/20260417_042615_seed0_antmaze-large-navigate-v0
  runs/20260417_050230_seed0_antmaze-giant-navigate-v0
  runs/20260417_054040_seed0_antmaze-teleport-navigate-v0
  runs/20260417_061834_seed0_antmaze-medium-stitch-v0
  runs/20260417_065748_seed0_antmaze-large-stitch-v0
  runs/20260417_073737_seed0_antmaze-giant-stitch-v0
  runs/20260417_081610_seed0_antmaze-teleport-stitch-v0
)

for rel in "${RUNS[@]}"; do
  R="$ROOT/$rel"
  echo "=== state rollout: $R (GOUB checkpoints/params_${EPOCH}.pkl only) ==="
  "$PY" rollout_subgoal_goub.py \
    --run_dir="$R" \
    --checkpoint_epoch="$EPOCH" \
    --traj_idx="$TRAJ_IDX" \
    --max_steps="$MAX_STEPS" \
    --out_path="$R/deterministic_state_rollout.png" \
    --fps="$FPS"

  if [[ "$SKIP_IDM" != "1" ]]; then
    IDM_ARGS=()
    IDM_PKL="$R/idm/checkpoints/params_${EPOCH}.pkl"
    if [[ -f "$IDM_PKL" ]]; then
      IDM_ARGS=(--idm_checkpoint="$IDM_PKL")
      echo "=== IDM env rollout: $R (standalone $IDM_PKL, inv_dyn_planner_freq=$IDM_FREQ) ==="
    else
      echo "=== IDM env rollout: $R (embedded idm_net in GOUB ckpt, inv_dyn_planner_freq=$IDM_FREQ) ==="
    fi
    "$PY" rollout_idm_goub.py \
      --run_dir="$R" \
      --checkpoint_epoch="$EPOCH" \
      --traj_idx="$TRAJ_IDX" \
      --max_steps="$MAX_STEPS" \
      "${IDM_ARGS[@]}" \
      --inv_dyn_planner_freq="$IDM_FREQ" \
      --out_path="$R/deterministic_inv_dyn_rollout_freq${IDM_FREQ}.png" \
      --fps="$FPS"
  fi
done

echo "Done."
