#!/usr/bin/env bash
set -euo pipefail
PROFILE_ENV="$HOME/.cache/sparset-decide/profiling-20260918"
mkdir -p "$PROFILE_ENV/tools"
if [ ! -x "$PROFILE_ENV/tools/uv" ]; then
  python3 - "$PROFILE_ENV" <<'PY'
import sys, urllib.request, tarfile
from pathlib import Path
root=Path(sys.argv[1])
archive=root/"uv.tar.gz"
urllib.request.urlretrieve("https://github.com/astral-sh/uv/releases/download/0.12.15/uv-x86_64-unknown-linux-gnu.tar.gz", archive)
with tarfile.open(archive) as tar:
    # Extract only the tool executable to an explicitly chosen file.
    member=next(m for m in tar.getmembers() if m.name.endswith("/uv") and m.isfile())
    (root/"tools/uv").write_bytes(tar.extractfile(member).read())
(root/"tools/uv").chmod(0o755)
PY
fi
export UV_CACHE_DIR="$PROFILE_ENV/uv-cache"
"$PROFILE_ENV/tools/uv" --version
if [ ! -x "$PROFILE_ENV/venv/bin/python" ]; then
  "$PROFILE_ENV/tools/uv" venv --python /usr/bin/python3 "$PROFILE_ENV/venv"
fi
"$PROFILE_ENV/tools/uv" pip install --python "$PROFILE_ENV/venv/bin/python" "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
"$PROFILE_ENV/tools/uv" pip install --python "$PROFILE_ENV/venv/bin/python" "transformers==5.5.4"
"$PROFILE_ENV/venv/bin/python" - <<'PY'
import torch,transformers
print("torch",torch.__version__,"transformers",transformers.__version__)
print("gpu",torch.cuda.get_device_name())
print("activities",torch.profiler.supported_activities())
print("flash_available",torch.backends.cuda.is_flash_attention_available())
PY


if [ "$#" -gt 0 ]; then exec "$PROFILE_ENV/venv/bin/python" -u "$@"; fi
