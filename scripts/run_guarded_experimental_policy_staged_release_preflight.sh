#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
PYTHON_BIN=${PYTHON:-/home/kai/anaconda3/envs/lunar-explorer/bin/python}

cd "$REPO_ROOT"
exec "$PYTHON_BIN" "$SCRIPT_DIR/run_guarded_experimental_policy_staged_release_preflight.py" "$@"
