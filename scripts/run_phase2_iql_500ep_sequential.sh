#!/usr/bin/env bash
# Sequential GOUB phase2 jobs with IQL fine-tuning (one GPU job at a time).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PHASE2_PYTHON:-/home/offrl/miniconda3/envs/offrl/bin/python}"

run_one() {
  local tag="$1" env="$2" p1="$3" idm="$4" total="$5" dist="$6" fine="$7"
  echo "==== BEGIN ${tag} $(date -Is) ===="
  if [[ -n "${idm}" ]]; then
    "${PY}" main_goub_phase2_policy.py \
      --run_config=config/goub_phase2_policy_antmaze.yaml \
      --run_group=phase2_iql_500ep \
      --seed=0 \
      --env_name="${env}" \
      --phase1_run_dir="${p1}" \
      --phase1_checkpoint_epoch=1000 \
      --idm_checkpoint="${idm}" \
      --train_steps="${total}" \
      --distill_steps="${dist}" \
      --finetune_steps="${fine}" \
      --log_interval=5000 \
      --eval_interval=100000 \
      --save_interval=100000 \
      --use_wandb=false \
      --use_tqdm=false \
      --agent.rl_algo=iql
  else
    "${PY}" main_goub_phase2_policy.py \
      --run_config=config/goub_phase2_policy_antmaze.yaml \
      --run_group=phase2_iql_500ep \
      --seed=0 \
      --env_name="${env}" \
      --phase1_run_dir="${p1}" \
      --phase1_checkpoint_epoch=1000 \
      --train_steps="${total}" \
      --distill_steps="${dist}" \
      --finetune_steps="${fine}" \
      --log_interval=5000 \
      --eval_interval=100000 \
      --save_interval=100000 \
      --use_wandb=false \
      --use_tqdm=false \
      --agent.rl_algo=iql
  fi
  echo "==== END ${tag} $(date -Is) ===="
}

# 500 epochs == 500 * ceil(dataset_size / batch_size); batch_size=1024 from agent config.
# Navigate variants: spe=978 -> total=489000 (30% distill).
run_one navigate_medium antmaze-medium-navigate-v0 \
  runs/20260418_124123_goub_phase1_path_seed0_antmaze-medium-navigate-v0 \
  runs/20260418_124123_goub_phase1_path_seed0_antmaze-medium-navigate-v0/standalone_idm_1000ep/checkpoints/params_1000.pkl \
  489000 146700 342300

run_one navigate_large antmaze-large-navigate-v0 \
  runs/20260418_140216_goub_phase1_path_seed0_antmaze-large-navigate-v0 \
  runs/20260418_140216_goub_phase1_path_seed0_antmaze-large-navigate-v0/standalone_idm_1000ep/checkpoints/params_1000.pkl \
  489000 146700 342300

run_one navigate_giant antmaze-giant-navigate-v0 \
  runs/20260418_163208_goub_phase1_path_seed0_antmaze-giant-navigate-v0 \
  runs/20260418_163208_goub_phase1_path_seed0_antmaze-giant-navigate-v0/standalone_idm_1000ep/checkpoints/params_1000.pkl \
  489000 146700 342300

# Embedded IDM in phase1 checkpoint (omit --idm_checkpoint).
run_one teleport_navigate antmaze-teleport-navigate-v0 \
  runs/20260418_215812_goub_phase1_path_seed0_antmaze-teleport-navigate-v0 \
  "" \
  489000 146700 342300

# Stitch variants: spe=982 -> total=491000.
run_one stitch_medium antmaze-medium-stitch-v0 \
  runs/20260418_234103_goub_phase1_path_seed0_antmaze-medium-stitch-v0 \
  "" \
  491000 147300 343700

run_one stitch_large antmaze-large-stitch-v0 \
  runs/20260419_011017_goub_phase1_path_seed0_antmaze-large-stitch-v0 \
  "" \
  491000 147300 343700

run_one stitch_giant antmaze-giant-stitch-v0 \
  runs/20260419_023804_goub_phase1_path_seed0_antmaze-giant-stitch-v0 \
  "" \
  491000 147300 343700

run_one teleport_stitch antmaze-teleport-stitch-v0 \
  runs/20260419_040613_goub_phase1_path_seed0_antmaze-teleport-stitch-v0 \
  "" \
  491000 147300 343700

echo "All jobs finished $(date -Is)"
