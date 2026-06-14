#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-/home/kai/anaconda3/envs/lunar-explorer/bin/python}"

FORMAL_PREFLIGHT_ROOT="${FORMAL_PREFLIGHT_ROOT:-outputs/path_feedback_batch_quasi_real_guarded_formal_ppo_preflight_v1}"
FORMAL_ROLLOUT_CANARY_ROOT="${FORMAL_ROLLOUT_CANARY_ROOT:-outputs/path_feedback_batch_quasi_real_guarded_formal_ppo_rollout_canary_v1}"
FORMAL_STABILITY_HOLDOUT_ROOT="${FORMAL_STABILITY_HOLDOUT_ROOT:-outputs/path_feedback_batch_quasi_real_guarded_formal_ppo_stability_holdout_validation_v1}"
CANDIDATE_SELECTION_ROOT="${CANDIDATE_SELECTION_ROOT:-outputs/path_feedback_batch_quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout_v1}"
PROMOTION_DECISION_REVIEW_ROOT="${PROMOTION_DECISION_REVIEW_ROOT:-outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_decision_review_v1}"
CANARY_PREFLIGHT_ROOT="${CANARY_PREFLIGHT_ROOT:-outputs/path_feedback_batch_guarded_experimental_policy_staged_release_canary_preflight_v1}"
BATCH_ROOT="${BATCH_ROOT:-outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1}"
OUTPUT="${OUTPUT:-outputs/path_feedback_batch_guarded_formal_ppo_training_authorization_v1}"
CONFIG="${CONFIG:-configs/guarded_formal_ppo_training_authorization_v1.json}"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_guarded_formal_ppo_training_authorization.sh" \
  --formal-preflight-root "$FORMAL_PREFLIGHT_ROOT" \
  --formal-rollout-canary-root "$FORMAL_ROLLOUT_CANARY_ROOT" \
  --formal-stability-holdout-root "$FORMAL_STABILITY_HOLDOUT_ROOT" \
  --candidate-selection-root "$CANDIDATE_SELECTION_ROOT" \
  --promotion-decision-review-root "$PROMOTION_DECISION_REVIEW_ROOT" \
  --canary-preflight-root "$CANARY_PREFLIGHT_ROOT" \
  --batch-root "$BATCH_ROOT" \
  --config "$CONFIG" \
  --output-root "$OUTPUT"
