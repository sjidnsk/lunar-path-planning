#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
CONFIG="${1:-configs/xunce_sandbox_candidate_preflight_v1.json}"
OUTPUT_ROOT="${2:-outputs/path_feedback_batch_xunce_sandbox_candidate_preflight_v1}"

"${PYTHON_BIN}" scripts/run_xunce_sandbox_candidate_preflight.py \
  --config "${CONFIG}" \
  --output-root "${OUTPUT_ROOT}"
