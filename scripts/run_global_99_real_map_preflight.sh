#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python3}"
CONFIG_PATH="${1:-$ROOT_DIR/configs/global_99_real_map_preflight_v1.json}"
OUTPUT_ROOT="${2:-$ROOT_DIR/outputs/path_feedback_batch_global_99_real_map_preflight_v1}"

"$PYTHON_BIN" "$ROOT_DIR/scripts/run_global_99_real_map_preflight.py" \
  --config "$CONFIG_PATH" \
  --output-root "$OUTPUT_ROOT" \
  --repo-root "$ROOT_DIR"
