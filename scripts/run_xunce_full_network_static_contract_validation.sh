#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-/home/kai/anaconda3/envs/lunar-explorer/bin/python}"
CONFIG="${1:-configs/xunce_full_network_static_contract_validation_v1.json}"
OUTPUT_ROOT="${2:-outputs/path_feedback_batch_xunce_full_network_static_contract_validation_v1}"

"$PYTHON_BIN" scripts/run_xunce_full_network_static_contract_validation.py \
  --config "$CONFIG" \
  --output-root "$OUTPUT_ROOT"
