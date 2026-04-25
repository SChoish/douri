#!/usr/bin/env bash
# Resume alpha=0.3, tau in {5,10} large sweep: loads latest common checkpoint per tau
# (see scripts/resume_goub_tau_at_alpha0p3_large.py). Missing tau starts from YAML.
#
#   nohup ./scripts/resume_goub_tau_at_alpha0p3_large.sh >nohup_logs/resume_tau_alpha0p3_$(date +%Y%m%d_%H%M%S).log 2>&1 &
set -euo pipefail

cd /home/choi/douri

PYTHON="${PYTHON:-/home/choi/miniconda3/envs/offrl/bin/python}"
export PYTHON

echo "[resume-launcher] start $(date -Iseconds)"
"$PYTHON" scripts/resume_goub_tau_at_alpha0p3_large.py "$@"
echo "[resume-launcher] done $(date -Iseconds)"
