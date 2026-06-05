#!/usr/bin/env bash
# Run any command with Douri JAX+CUDA library path and default JAX_PLATFORMS=cuda.
# Example: bash scripts/with_jax_cuda.sh python3 main.py --run_config=config/foo.yaml
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/jax_cuda_env.sh"
exec "$@"
