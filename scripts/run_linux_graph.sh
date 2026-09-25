#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
PROFILE_ENV="$HOME/.cache/subset/profiling-20260918"
"$PROFILE_ENV/venv/bin/python" - "$root" "$PROFILE_ENV" <<'PY'
from pathlib import Path
import shutil, sys
root=Path(sys.argv[1])
runtime=Path(sys.argv[2])
source=root/".cache/decision-engine/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
target=runtime/"model"
if not target.exists():
    print("Copying pinned model snapshot to Linux-local storage for the graph test",flush=True)
    shutil.copytree(source,target)
for file in source.iterdir():
    if file.is_file() and file.stat().st_size!=(target/file.name).stat().st_size:
        raise RuntimeError("Copied model file size mismatch: "+file.name)
print("Local snapshot ready",flush=True)
PY
exec "$PROFILE_ENV/venv/bin/python" -u "$root/scripts/profile_cuda_graph.py" --model "$PROFILE_ENV/model" --output "$root/outputs/decision-engine/profiling-20260918/cuda-graph"
