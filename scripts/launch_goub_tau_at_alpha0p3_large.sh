#!/usr/bin/env bash
# Large navigate only: vanilla GOUB with goub.subgoal_value_alpha=0.3 fixed,
# sweep actor.spi_tau in {5.0, 10.0, 20.0}. 400 epochs, eval every 100.
#
# Configs: config/sweep_goub_tau_at_alpha0p3/antmaze_large_navigate_goub_alpha0p3_tau*.yaml
#
#   nohup ./scripts/launch_goub_tau_at_alpha0p3_large.sh >nohup_logs/launcher_tau_alpha0p3_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -euo pipefail

cd /home/choi/douri

PYTHON=/home/choi/miniconda3/envs/offrl/bin/python
CONFIG_DIR=config/sweep_goub_tau_at_alpha0p3
LOG_DIR=nohup_logs
mkdir -p "$LOG_DIR"

TAUS=("5.0" "10.0" "20.0")

echo "[launcher] start $(date -Iseconds) large navigate alpha=0.3 tau sweep"

for tau in "${TAUS[@]}"; do
  tau_tag=${tau//./p}
  cfg="$CONFIG_DIR/antmaze_large_navigate_goub_alpha0p3_tau${tau_tag}.yaml"
  if [[ ! -f "$cfg" ]]; then
    echo "[launcher] MISSING $cfg" >&2
    exit 1
  fi
  ts=$(date +%Y%m%d_%H%M%S)
  log="$LOG_DIR/train_large_goub_alpha0p3_tau${tau_tag}_${ts}.log"
  echo "[launcher] === tau=${tau} alpha=0.3 (fixed) === $(date -Iseconds)"
  echo "[launcher] cfg=$cfg log=$log"
  nohup "$PYTHON" main.py --run_config="$cfg" >"$log" 2>&1 </dev/null &
  wait $!
  echo "[launcher] done tau=${tau} $(date -Iseconds)"
done

echo "[launcher] done $(date -Iseconds)"
