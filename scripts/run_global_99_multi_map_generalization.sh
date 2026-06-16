#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$REPO_ROOT/configs/global_99_multi_map_generalization_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/path_feedback_batch_global_99_multi_map_generalization_v1}"

"$PYTHON_BIN" "$REPO_ROOT/scripts/run_global_99_multi_map_generalization.py" \
  --config "$CONFIG" \
  --output-root "$OUTPUT_ROOT" \
  --repo-root "$REPO_ROOT"
