#!/usr/bin/env bash
# Set A gap / weight_max / gamma sweep (TRL + diag_gaussian).
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_tune_sweep.sh" v2 "$@"
