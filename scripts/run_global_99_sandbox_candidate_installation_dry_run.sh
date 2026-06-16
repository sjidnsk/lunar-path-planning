#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-/home/kai/anaconda3/envs/lunar-explorer/bin/python}"
CONFIG="${1:-configs/global_99_sandbox_candidate_installation_dry_run_v1.json}"
OUTPUT_ROOT="${2:-outputs/path_feedback_batch_global_99_sandbox_candidate_installation_dry_run_v1}"

"$PYTHON_BIN" scripts/run_global_99_sandbox_candidate_installation_dry_run.py \
  --config "$CONFIG" \
  --output-root "$OUTPUT_ROOT"
