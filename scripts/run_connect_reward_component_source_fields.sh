#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec "$PYTHON_BIN" "$SCRIPT_DIR/run_connect_reward_component_source_fields.py" "$@"
