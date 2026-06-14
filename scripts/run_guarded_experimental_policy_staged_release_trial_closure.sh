#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-/home/kai/anaconda3/envs/lunar-explorer/bin/python}"

PREFLIGHT_ROOT="${PREFLIGHT_ROOT:-outputs/path_feedback_batch_guarded_experimental_policy_staged_release_preflight_v1}"
SHADOW_ROOT="${SHADOW_ROOT:-outputs/path_feedback_batch_guarded_experimental_policy_shadow_release_trial_v1}"
BATCH_ROOT="${BATCH_ROOT:-outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1}"
OUTPUT="${OUTPUT:-outputs/path_feedback_batch_guarded_experimental_policy_staged_release_trial_v1}"
CONFIG="${CONFIG:-configs/guarded_experimental_policy_staged_release_trial_v1.json}"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_guarded_experimental_policy_staged_release_trial.sh" \
  --preflight-root "$PREFLIGHT_ROOT" \
  --shadow-release-trial-root "$SHADOW_ROOT" \
  --batch-root "$BATCH_ROOT" \
  --config "$CONFIG" \
  --output-root "$OUTPUT"
