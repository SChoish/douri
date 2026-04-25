#!/usr/bin/env bash
# Sequentially sweep antmaze-{large,giant}-navigate vanilla GOUB over
# (actor.spi_tau, goub.subgoal_value_alpha) = {0.5, 1.0, 5.0} x {0.0, 0.5, 1.0}
# Trains for 400 epochs each (eval at 100/200/300/400, ckpt at 100/200/300/400).
#
# Each invocation creates its own runs/<timestamp>_..._<env_name> directory with
# run.log, train.csv, and checkpoints. Each job runs under nohup (detach +
# SIGHUP-safe) but we wait between jobs so only one training uses the GPU at a
# time. Stdout/stderr: nohup_logs/<maze>_goub_tau*_alpha*_<timestamp>.log
#
# Full sweep in background from your shell:
#   nohup ./scripts/launch_goub_tau_alpha_sweep.sh >nohup_logs/launcher_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -euo pipefail

cd /home/choi/douri

PYTHON=/home/choi/miniconda3/envs/offrl/bin/python
CONFIG_DIR=config/sweep_goub_tau_alpha
LOG_DIR=nohup_logs
mkdir -p "$LOG_DIR"

TAUS=("0.5" "1.0" "5.0")
ALPHAS=("0.0" "0.5" "1.0")
# maze key must match config basename: antmaze_<key>_navigate_goub_*.yaml
MAZE_KEYS=("large" "giant")

echo "[launcher] start $(date -Iseconds)"
echo "[launcher] mazes=${MAZE_KEYS[*]} configs in $CONFIG_DIR"

for maze_key in "${MAZE_KEYS[@]}"; do
  cfg_prefix="antmaze_${maze_key}_navigate_goub"
  for tau in "${TAUS[@]}"; do
    for alpha in "${ALPHAS[@]}"; do
      tau_tag=${tau//./p}
      alpha_tag=${alpha//./p}
      cfg="$CONFIG_DIR/${cfg_prefix}_tau${tau_tag}_alpha${alpha_tag}.yaml"
      if [[ ! -f "$cfg" ]]; then
        echo "[launcher] MISSING $cfg" >&2
        exit 1
      fi
      ts=$(date +%Y%m%d_%H%M%S)
      log="$LOG_DIR/train_${maze_key}_goub_tau${tau_tag}_alpha${alpha_tag}_${ts}.log"
      echo "[launcher] === maze=${maze_key} tau=${tau} alpha=${alpha} === $(date -Iseconds)"
      echo "[launcher] cfg=$cfg log=$log"
      nohup "$PYTHON" main.py --run_config="$cfg" >"$log" 2>&1 </dev/null &
      wait $!
      echo "[launcher] done maze=${maze_key} tau=${tau} alpha=${alpha} $(date -Iseconds)"
    done
  done
done

echo "[launcher] done $(date -Iseconds)"
