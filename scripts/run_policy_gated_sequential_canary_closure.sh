#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_PYTHON="/home/kai/anaconda3/envs/lunar-explorer/bin/python"
PYTHON_BIN="${PYTHON:-$DEFAULT_PYTHON}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

SRC="${SRC:-outputs/path_feedback_batch_sequential_canary_clean_src_v1}"
BASE="${BASE:-outputs/path_feedback_batch_sequential_canary_baseline_candidate_v1}"
DEV="${DEV:-outputs/path_feedback_batch_sequential_canary_dev_v1}"
TRAIN="${TRAIN:-outputs/path_feedback_batch_sequential_canary_train_v1}"
VAL="${VAL:-outputs/path_feedback_batch_sequential_canary_val_v1}"
TEST="${TEST:-outputs/path_feedback_batch_sequential_canary_test_v1}"
CAND="${CAND:-outputs/path_feedback_batch_sequential_canary_candidate_v1}"
SEQUENTIAL="${SEQUENTIAL:-outputs/path_feedback_batch_policy_gated_sequential_canary_rollout_v1}"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_batch_path_feedback_validation.sh" \
  --matrix "$REPO_ROOT/configs/path_feedback_batch_planner_validated_trainable_target_mining_v1.json" \
  --output-root "$SRC"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_anchor_projection_candidate_generation.sh" \
  --batch-root "$SRC" \
  --config "$REPO_ROOT/configs/anchor_projection_candidate_generation_v1.json"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_anchor_projection_evidence_contract.sh" \
  --batch-root "$SRC" \
  --config "$REPO_ROOT/configs/anchor_projection_evidence_contract_v1.json"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_planner_validated_trainable_target_mining.sh" \
  --batch-root "$SRC" \
  --config "$REPO_ROOT/configs/planner_validated_trainable_target_mining_v1.json"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_planner_validated_training_input_materialization.sh" \
  --batch-root "$SRC" \
  --config "$REPO_ROOT/configs/planner_validated_training_input_materialization_v1.json"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_limited_policy_training_dry_run.sh" \
  --batch-root "$SRC" \
  --config "$REPO_ROOT/configs/limited_policy_training_dry_run_v1.json"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_counterfactual_preference_training_samples.sh" \
  --batch-root "$SRC" \
  --config "$REPO_ROOT/configs/counterfactual_preference_training_samples_v1.json"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_counterfactual_preference_training_dry_run.sh" \
  --batch-root "$SRC" \
  --config "$REPO_ROOT/configs/counterfactual_preference_training_dry_run_v1.json"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_unified_policy_sample_registry.sh" \
  --batch-root "$SRC" \
  --config "$REPO_ROOT/configs/unified_policy_sample_registry_v1.json"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_residual_boundary_preference_training_dry_run.sh" \
  --batch-root "$SRC" \
  --config "$REPO_ROOT/configs/residual_boundary_preference_training_dry_run_v1.json"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_hybrid_policy_training_dry_run.sh" \
  --batch-root "$SRC" \
  --config "$REPO_ROOT/configs/hybrid_policy_training_dry_run_v1.json"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_controlled_hybrid_policy_training_candidate.sh" \
  --source-root "$SRC" \
  --output-root "$BASE" \
  --config "$REPO_ROOT/configs/controlled_hybrid_policy_training_candidate_v1.json"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_batch_path_feedback_validation.sh" \
  --matrix "$REPO_ROOT/configs/path_feedback_batch_scenario_disjoint_policy_candidate_evaluation_v1.json" \
  --output-root "$DEV"
PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_batch_path_feedback_validation.sh" \
  --matrix "$REPO_ROOT/configs/path_feedback_batch_raw_policy_generalization_train_v1.json" \
  --output-root "$TRAIN"
PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_batch_path_feedback_validation.sh" \
  --matrix "$REPO_ROOT/configs/path_feedback_batch_raw_policy_generalization_val_v1.json" \
  --output-root "$VAL"
PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_batch_path_feedback_validation.sh" \
  --matrix "$REPO_ROOT/configs/path_feedback_batch_raw_policy_generalization_test_v1.json" \
  --output-root "$TEST"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_raw_policy_generalization_closure.sh" \
  --source-root "$SRC" \
  --dev-root "$DEV" \
  --train-root "$TRAIN" \
  --val-root "$VAL" \
  --test-root "$TEST" \
  --baseline-candidate-root "$BASE" \
  --candidate-root "$CAND"

set +e
PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_policy_gated_sequential_canary_rollout.sh" \
  --source-root "$SRC" \
  --candidate-root "$CAND" \
  --batch-root "$SEQUENTIAL" \
  --config "$REPO_ROOT/configs/policy_gated_sequential_canary_rollout_v1.json"
SEQUENTIAL_STATUS=$?
set -e

set +e
PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_policy_training_readiness_review.sh" \
  --batch-root "$SRC" \
  --config "$REPO_ROOT/configs/policy_training_readiness_review_v1.json" \
  --raw-policy-generalization-evaluation-summary "$CAND/raw-policy-generalization-evaluation-summary.json" \
  --policy-gated-sequential-canary-rollout-summary "$SEQUENTIAL/policy-gated-sequential-canary-rollout-summary.json"
READINESS_STATUS=$?
set -e

if [[ "$SEQUENTIAL_STATUS" -ne 0 ]]; then
  exit "$SEQUENTIAL_STATUS"
fi
exit "$READINESS_STATUS"
