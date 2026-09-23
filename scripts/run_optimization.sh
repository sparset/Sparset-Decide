#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
runtime="$HOME/.cache/sparset-decide/profiling-20260918"
export CC="$runtime/tools/zig-cc"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
exec "$runtime/venv/bin/python" -u "$root/scripts/benchmark_optimizations.py" "$@"
