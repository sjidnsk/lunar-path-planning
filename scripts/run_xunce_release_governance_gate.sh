#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
CONFIG="${1:-configs/xunce_release_governance_gate_v1.json}"
OUTPUT_ROOT="${2:-outputs/path_feedback_batch_xunce_release_governance_gate_v1}"

"${PYTHON_BIN}" scripts/run_xunce_release_governance_gate.py \
  --config "${CONFIG}" \
  --output-root "${OUTPUT_ROOT}"
