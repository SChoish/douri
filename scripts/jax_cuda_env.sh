#!/usr/bin/env bash
# This file is intended to be sourced (see scripts/with_jax_cuda.sh for a wrapper).
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "source scripts/jax_cuda_env.sh  또는  bash scripts/with_jax_cuda.sh <command>" >&2
  exit 1
fi

# Prepend pip-bundled NVIDIA shared libs so jax[cuda] / jax_plugins.xla_cuda12 finds
# libcusparse and friends (otherwise JAX may fall back to CPU with no obvious error).
#
# Usage (from repo root or any cwd):
#   source "$(git rev-parse --show-toplevel 2>/dev/null || dirname "$0")/scripts/jax_cuda_env.sh"
# Or: bash scripts/with_jax_cuda.sh python3 main.py ...
#
# Override: JAX_PLATFORMS=cpu (still prepends LD paths; unset JAX_PLATFORMS before source to skip cuda default).

[[ -n "${DOURI_JAX_CUDA_ENV_EXPORTED:-}" ]] && return 0

_jax_cuda_py="${PYTHON:-python3}"
if ! command -v "$_jax_cuda_py" &>/dev/null; then
  _jax_cuda_py="python3"
fi

_extra_ld="$("$_jax_cuda_py" -c '
import sys
from pathlib import Path
for d in sys.path:
    base = Path(d) / "nvidia"
    if (base / "cusparse" / "lib").is_dir():
        names = (
            "cusparse", "cublas", "cuda_runtime", "cudnn", "cufft",
            "cusolver", "curand", "nvjitlink",
        )
        parts = []
        for n in names:
            p = base / n / "lib"
            if p.is_dir():
                parts.append(str(p))
        print(":".join(parts))
        break
' 2>/dev/null || true)"

if [[ -n "${_extra_ld}" ]]; then
  export LD_LIBRARY_PATH="${_extra_ld}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export DOURI_JAX_CUDA_ENV_EXPORTED=1
