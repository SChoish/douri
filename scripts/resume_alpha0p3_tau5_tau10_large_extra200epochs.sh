#!/usr/bin/env bash
# 093720 (τ=5), 123020 (τ=10) — alpha=0.3 large joint 런을 epoch-400 체크포인트에서 이어서
# **200 epoch만** 추가 학습합니다 (총 train_epochs=600까지, ep401–600).
#
# 다른 학습이 끝난 뒤 실행하세요. 두 런은 **순차**로 돌며, 각각 nohup + wait 로 GPU 한 대만 사용합니다.
#
#   cd /home/choi/douri
#   chmod +x scripts/resume_alpha0p3_tau5_tau10_large_extra200epochs.sh
#   nohup ./scripts/resume_alpha0p3_tau5_tau10_large_extra200epochs.sh \
#     > nohup_logs/resume_alpha0p3_extra200_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PYTHON="${PYTHON:-/home/choi/miniconda3/envs/offrl/bin/python}"
LOG_DIR="${LOG_DIR:-$REPO/nohup_logs}"
mkdir -p "$LOG_DIR"

# 마지막으로 저장된 공통 체크포인트 epoch (goub/critic/actor 모두 params_<ep>.pkl 있어야 함).
RESUME_EPOCH="${RESUME_EPOCH:-400}"
# 그 이후 몇 epoch 더 돌릴지 (총 목표 = RESUME_EPOCH + EXTRA_EPOCHS).
EXTRA_EPOCHS="${EXTRA_EPOCHS:-200}"
TARGET_EPOCH=$((RESUME_EPOCH + EXTRA_EPOCHS))

# 사용자 요청 순서: 123020 먼저, 그다음 093720
RUNS=(
  "$REPO/runs/20260425_123020_joint_dqc_seed0_antmaze-large-navigate-v0"
  "$REPO/runs/20260425_093720_joint_dqc_seed0_antmaze-large-navigate-v0"
)

echo "[resume-extra200] repo=$REPO resume_epoch=$RESUME_EPOCH train_epochs=$TARGET_EPOCH (+$EXTRA_EPOCHS) start $(date -Iseconds)"

for run_dir in "${RUNS[@]}"; do
  if [[ ! -d "$run_dir" ]]; then
    echo "[resume-extra200] ERROR missing $run_dir" >&2
    exit 1
  fi
  for agent in goub critic actor; do
    ckpt="$run_dir/checkpoints/$agent/params_${RESUME_EPOCH}.pkl"
    if [[ ! -f "$ckpt" ]]; then
      echo "[resume-extra200] ERROR missing $ckpt" >&2
      exit 1
    fi
  done
done

for run_dir in "${RUNS[@]}"; do
  base="$(basename "$run_dir")"
  ts="$(date +%Y%m%d_%H%M%S)"
  log="$LOG_DIR/resume_extra200_${base}_${ts}.log"
  echo "[resume-extra200] === $base === $(date -Iseconds)"
  echo "[resume-extra200] log=$log"
  # --run_config 없음 → flags.json 스냅샷으로 하이퍼 유지; --train_epochs 만 CLI로 덮어써 총 600까지.
  nohup "$PYTHON" main.py \
    "--resume_run_dir=$run_dir" \
    "--resume_epoch=$RESUME_EPOCH" \
    "--train_epochs=$TARGET_EPOCH" \
    >"$log" 2>&1 </dev/null &
  wait $!
  echo "[resume-extra200] done $base $(date -Iseconds)"
done

echo "[resume-extra200] all done $(date -Iseconds)"
