#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_PYTHON="/home/kai/anaconda3/envs/lunar-explorer/bin/python"
PYTHON_BIN="${PYTHON:-$DEFAULT_PYTHON}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

SOURCE_SRC="${SOURCE_SRC:-outputs/path_feedback_batch_ppo_collector_clean_src_v1}"
SRC="${SRC:-outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1}"
ITERATIVE="${ITERATIVE:-outputs/path_feedback_batch_iterative_ppo_mini_loop_stability_v1}"
BASE="${BASE:-$ITERATIVE/round-02/update}"
RAW_BASE="${RAW_BASE:-outputs/path_feedback_batch_sequential_multi_step_opportunity_baseline_candidate_v1}"
DEV="${DEV:-outputs/path_feedback_batch_sequential_multi_step_opportunity_dev_v1}"
VAL="${VAL:-outputs/path_feedback_batch_sequential_multi_step_opportunity_val_v1}"
TEST="${TEST:-outputs/path_feedback_batch_sequential_multi_step_opportunity_test_v1}"
OUT="${OUT:-outputs/path_feedback_batch_guarded_ppo_rollout_pilot_v1}"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_iterative_ppo_mini_loop_stability_closure.sh"

rm -rf "$SRC" "$OUT"
cp -a "$SOURCE_SRC" "$SRC"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_guarded_ppo_rollout_pilot.sh" \
  --source-root "$SRC" \
  --base-candidate-root "$BASE" \
  --raw-baseline-candidate-root "$RAW_BASE" \
  --dev-root "$DEV" \
  --val-root "$VAL" \
  --test-root "$TEST" \
  --output-root "$OUT" \
  --config "$REPO_ROOT/configs/guarded_ppo_rollout_pilot_v1.json"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_policy_training_readiness_review.sh" \
  --batch-root "$SRC" \
  --config "$REPO_ROOT/configs/policy_training_readiness_review_v1.json" \
  --raw-policy-generalization-evaluation-summary "$OUT/final/raw-policy-generalization-evaluation-summary.json" \
  --policy-gated-sequential-canary-rollout-summary "$OUT/final/sequential/policy-gated-sequential-canary-rollout-summary.json" \
  --ppo-rollout-collector-summary "$OUT/final/collector/ppo-rollout-collector-summary.json" \
  --limited-ppo-update-smoke-summary "$OUT/update/limited-ppo-update-smoke-summary.json" \
  --iterative-ppo-mini-loop-stability-summary "$ITERATIVE/iterative-ppo-mini-loop-stability-summary.json" \
  --guarded-ppo-rollout-pilot-summary "$OUT/guarded-ppo-rollout-pilot-summary.json"
