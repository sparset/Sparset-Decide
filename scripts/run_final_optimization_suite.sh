#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
runtime="$HOME/.cache/sparset-decide/profiling-20260918"
"$runtime/venv/bin/python" -u "$root/scripts/run_optimization_tests.py"
for stage in fused combined; do
    bash "$root/scripts/run_optimization.sh" --stage "$stage"
done
