#!/usr/bin/env bash
# Stitch stages removed; this entrypoint now runs the same 4 navigate-only stages.
# See scripts/run_antmaze4_navigate_a03_pc1_sequential.sh for the active CONFIG list.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${ROOT_DIR}/scripts/run_antmaze4_navigate_a03_pc1_sequential.sh"
