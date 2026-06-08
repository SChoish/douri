#!/usr/bin/env bash
# gap=1, wmax {5,10}, gamma {0.999,0.995} sweep (TRL + diag_gaussian).
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_tune_sweep.sh" gap1 "$@"
