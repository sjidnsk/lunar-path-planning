#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_PYTHON="/home/kai/anaconda3/envs/lunar-explorer/bin/python"
PYTHON_BIN="${PYTHON:-$DEFAULT_PYTHON}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

INSTALL_CANARY_ROOT="${INSTALL_CANARY_ROOT:-outputs/path_feedback_batch_guarded_experimental_policy_install_canary_dry_run_v1}"
MULTIHORIZON_SHADOW_ROOT="${MULTIHORIZON_SHADOW_ROOT:-outputs/path_feedback_batch_selected_formal_ppo_candidate_multihorizon_shadow_rollout_v1}"
BATCH_ROOT="${BATCH_ROOT:-outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1}"
OUTPUT="${OUTPUT:-outputs/path_feedback_batch_guarded_experimental_policy_shadow_release_trial_v1}"
CONFIG="${CONFIG:-configs/guarded_experimental_policy_shadow_release_trial_v1.json}"

rm -rf "$REPO_ROOT/$OUTPUT"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_guarded_experimental_policy_shadow_release_trial.sh" \
  --install-canary-root "$INSTALL_CANARY_ROOT" \
  --multihorizon-shadow-root "$MULTIHORIZON_SHADOW_ROOT" \
  --batch-root "$BATCH_ROOT" \
  --output-root "$OUTPUT" \
  --config "$CONFIG"
