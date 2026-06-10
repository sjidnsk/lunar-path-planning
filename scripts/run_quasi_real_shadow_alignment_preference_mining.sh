#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-/home/kai/anaconda3/envs/lunar-explorer/bin/python}"

exec "$PYTHON_BIN" "$SCRIPT_DIR/run_quasi_real_shadow_alignment_preference_mining.py" "$@"
