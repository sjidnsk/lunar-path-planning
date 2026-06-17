#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_PYTHON="/home/kai/anaconda3/envs/lunar-explorer/bin/python"
PYTHON_BIN="${PYTHON:-$DEFAULT_PYTHON}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

CONFIG="${CONFIG:-$REPO_ROOT/configs/xunce_high_fidelity_real_map_comparison_v1.json}"
OUT="${OUT:-$REPO_ROOT/outputs/path_feedback_batch_xunce_high_fidelity_real_map_comparison_v1}"

"$PYTHON_BIN" "$SCRIPT_DIR/run_xunce_high_fidelity_real_map_comparison.py" \
  --config "$CONFIG" \
  --output-root "$OUT" \
  --repo-root "$REPO_ROOT"
