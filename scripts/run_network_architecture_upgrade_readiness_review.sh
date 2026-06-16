#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$REPO_ROOT/configs/network_architecture_upgrade_readiness_review_v1.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/path_feedback_batch_network_architecture_upgrade_readiness_review_v1}"

"$PYTHON_BIN" "$REPO_ROOT/scripts/run_network_architecture_upgrade_readiness_review.py" \
  --config "$CONFIG" \
  --output-root "$OUTPUT_ROOT" \
  --repo-root "$REPO_ROOT"
