#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-/home/kai/anaconda3/envs/lunar-explorer/bin/python}"
CONFIG="${1:-configs/xunce_full_network_ablation_experiments_v1.json}"
OUTPUT_ROOT="${2:-outputs/path_feedback_batch_xunce_full_network_ablation_experiments_v1}"

"$PYTHON_BIN" scripts/run_xunce_full_network_ablation_experiments.py \
  --config "$CONFIG" \
  --output-root "$OUTPUT_ROOT"
