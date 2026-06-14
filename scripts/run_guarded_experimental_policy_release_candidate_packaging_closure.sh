#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON:-${PYTHON_BIN:-python3}}"
DECISION_ROOT="${DECISION_ROOT:-outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_decision_review_v1}"
BATCH_ROOT="${BATCH_ROOT:-outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1}"
OUTPUT="${OUTPUT:-outputs/path_feedback_batch_guarded_experimental_policy_release_candidate_packaging_v1}"
CONFIG="${CONFIG:-configs/guarded_experimental_policy_release_candidate_packaging_v1.json}"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_guarded_experimental_policy_release_candidate_packaging.sh" \
  --decision-root "$DECISION_ROOT" \
  --batch-root "$BATCH_ROOT" \
  --output-root "$OUTPUT" \
  --config "$CONFIG"
