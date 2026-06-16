#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-${PY:-python3}}"
CONFIG="${1:-$ROOT_DIR/configs/xunce_design_freeze_v1.json}"
OUTPUT_ROOT="${2:-$ROOT_DIR/outputs/path_feedback_batch_xunce_design_freeze_v1}"

"$PYTHON_BIN" "$ROOT_DIR/scripts/run_xunce_design_freeze_audit.py" \
  --config "$CONFIG" \
  --output-root "$OUTPUT_ROOT" \
  --repo-root "$ROOT_DIR"
