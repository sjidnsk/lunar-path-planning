#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-/home/kai/anaconda3/envs/lunar-explorer/bin/python}"

AUTHORIZATION_ROOT="${AUTHORIZATION_ROOT:-outputs/path_feedback_batch_guarded_formal_ppo_training_authorization_v1}"
BATCH_ROOT="${BATCH_ROOT:-outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1}"
OUTPUT="${OUTPUT:-outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1}"
CONFIG="${CONFIG:-configs/guarded_formal_ppo_training_run_v1.json}"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_guarded_formal_ppo_training_run.sh" \
  --authorization-root "$AUTHORIZATION_ROOT" \
  --batch-root "$BATCH_ROOT" \
  --config "$CONFIG" \
  --output-root "$OUTPUT"
