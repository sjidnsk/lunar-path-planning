#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_PYTHON="/home/kai/anaconda3/envs/lunar-explorer/bin/python"
PYTHON_BIN="${PYTHON:-$DEFAULT_PYTHON}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

CANARY_ROOT="${CANARY_ROOT:-outputs/path_feedback_batch_quasi_real_guarded_formal_ppo_rollout_canary_v1}"
BATCH_ROOT="${BATCH_ROOT:-outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1}"
OUTPUT="${OUTPUT:-outputs/path_feedback_batch_quasi_real_guarded_formal_ppo_stability_holdout_validation_v1}"
CONFIG="${CONFIG:-configs/quasi_real_guarded_formal_ppo_stability_holdout_validation_v1.json}"

rm -rf "$REPO_ROOT/$OUTPUT"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_quasi_real_guarded_formal_ppo_stability_holdout_validation.sh" \
  --canary-root "$CANARY_ROOT" \
  --batch-root "$BATCH_ROOT" \
  --output-root "$OUTPUT" \
  --config "$CONFIG"
