#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_PYTHON="/home/kai/anaconda3/envs/lunar-explorer/bin/python"
PYTHON_BIN="${PYTHON:-$DEFAULT_PYTHON}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

SRC="${SRC:-outputs/path_feedback_batch_ppo_collector_clean_src_v1}"
BASE="${BASE:-outputs/path_feedback_batch_ppo_collector_candidate_v1}"
RAW_BASE="${RAW_BASE:-outputs/path_feedback_batch_sequential_multi_step_opportunity_baseline_candidate_v1}"
COLLECTOR="${COLLECTOR:-outputs/path_feedback_batch_ppo_rollout_collector_dry_run_v1}"
UPDATE="${UPDATE:-outputs/path_feedback_batch_limited_ppo_update_smoke_v1}"
SEQ="${SEQ:-outputs/path_feedback_batch_limited_ppo_update_sequential_v1}"
UPDATED_COLLECTOR="${UPDATED_COLLECTOR:-outputs/path_feedback_batch_limited_ppo_update_collector_v1}"

DEV="${DEV:-outputs/path_feedback_batch_sequential_multi_step_opportunity_dev_v1}"
VAL="${VAL:-outputs/path_feedback_batch_sequential_multi_step_opportunity_val_v1}"
TEST="${TEST:-outputs/path_feedback_batch_sequential_multi_step_opportunity_test_v1}"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_sequential_multi_step_opportunity_closure.sh"
PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_ppo_rollout_collector_closure.sh"

rm -rf "$UPDATE" "$SEQ" "$UPDATED_COLLECTOR"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_limited_ppo_update_smoke.sh" \
  --source-root "$SRC" \
  --base-candidate-root "$BASE" \
  --collector-root "$COLLECTOR" \
  --output-root "$UPDATE" \
  --config "$REPO_ROOT/configs/limited_ppo_update_smoke_v1.json"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_raw_policy_generalization_evaluation.sh" \
  --source-root "$SRC" \
  --dev-root "$DEV" \
  --val-root "$VAL" \
  --test-root "$TEST" \
  --baseline-candidate-root "$RAW_BASE" \
  --candidate-root "$UPDATE" \
  --config "$REPO_ROOT/configs/raw_policy_generalization_evaluation_v1.json"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_policy_gated_sequential_canary_rollout.sh" \
  --source-root "$SRC" \
  --candidate-root "$UPDATE" \
  --batch-root "$SEQ" \
  --config "$REPO_ROOT/configs/policy_gated_sequential_multi_step_opportunity_rollout_v1.json"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_ppo_rollout_collector_dry_run.sh" \
  --sequential-root "$SEQ" \
  --candidate-root "$UPDATE" \
  --output-root "$UPDATED_COLLECTOR" \
  --config "$REPO_ROOT/configs/ppo_rollout_collector_dry_run_v1.json"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_policy_training_readiness_review.sh" \
  --batch-root "$SRC" \
  --config "$REPO_ROOT/configs/policy_training_readiness_review_v1.json" \
  --raw-policy-generalization-evaluation-summary "$UPDATE/raw-policy-generalization-evaluation-summary.json" \
  --policy-gated-sequential-canary-rollout-summary "$SEQ/policy-gated-sequential-canary-rollout-summary.json" \
  --ppo-rollout-collector-summary "$UPDATED_COLLECTOR/ppo-rollout-collector-summary.json" \
  --limited-ppo-update-smoke-summary "$UPDATE/limited-ppo-update-smoke-summary.json"
