#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_PYTHON="/home/kai/anaconda3/envs/lunar-explorer/bin/python"
PYTHON_BIN="${PYTHON:-$DEFAULT_PYTHON}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

SOURCE_ROOT=""
DEV_ROOT=""
TRAIN_ROOT=""
VAL_ROOT=""
TEST_ROOT=""
BASELINE_CANDIDATE_ROOT=""
CANDIDATE_ROOT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-root)
      SOURCE_ROOT="$2"
      shift 2
      ;;
    --dev-root)
      DEV_ROOT="$2"
      shift 2
      ;;
    --train-root)
      TRAIN_ROOT="$2"
      shift 2
      ;;
    --val-root)
      VAL_ROOT="$2"
      shift 2
      ;;
    --test-root)
      TEST_ROOT="$2"
      shift 2
      ;;
    --baseline-candidate-root)
      BASELINE_CANDIDATE_ROOT="$2"
      shift 2
      ;;
    --candidate-root)
      CANDIDATE_ROOT="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$SOURCE_ROOT" || -z "$DEV_ROOT" || -z "$TRAIN_ROOT" || -z "$VAL_ROOT" || -z "$TEST_ROOT" || -z "$BASELINE_CANDIDATE_ROOT" || -z "$CANDIDATE_ROOT" ]]; then
  echo "usage: $0 --source-root ROOT --dev-root ROOT --train-root ROOT --val-root ROOT --test-root ROOT --baseline-candidate-root ROOT --candidate-root ROOT" >&2
  exit 2
fi

for split_root in "$DEV_ROOT" "$TRAIN_ROOT" "$VAL_ROOT" "$TEST_ROOT"; do
  PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_fresh_holdout_policy_candidate_evaluation.sh" \
    --source-root "$SOURCE_ROOT" \
    --candidate-root "$BASELINE_CANDIDATE_ROOT" \
    --batch-root "$split_root" \
    --config "$REPO_ROOT/configs/scenario_disjoint_policy_candidate_evaluation_v1.json"

  PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_scenario_disjoint_policy_rollout_evaluation.sh" \
    --source-root "$SOURCE_ROOT" \
    --candidate-root "$BASELINE_CANDIDATE_ROOT" \
    --batch-root "$split_root" \
    --config "$REPO_ROOT/configs/scenario_disjoint_policy_rollout_evaluation_v1.json"
done

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_raw_policy_regression_mining.sh" \
  --source-root "$SOURCE_ROOT" \
  --holdout-root "$DEV_ROOT" \
  --config "$REPO_ROOT/configs/raw_policy_regression_mining_v1.json"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_raw_policy_regression_mining.sh" \
  --source-root "$SOURCE_ROOT" \
  --holdout-root "$TRAIN_ROOT" \
  --config "$REPO_ROOT/configs/raw_policy_regression_mining_v1.json"

for eval_root in "$VAL_ROOT" "$TEST_ROOT"; do
  PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_raw_policy_regression_mining.sh" \
    --source-root "$SOURCE_ROOT" \
    --holdout-root "$eval_root" \
    --config "$REPO_ROOT/configs/raw_policy_regression_mining_diagnostic_v1.json"
done

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_raw_policy_generalization_candidate.sh" \
  --source-root "$SOURCE_ROOT" \
  --train-mining-root "$TRAIN_ROOT" \
  --dev-mining-root "$DEV_ROOT" \
  --val-diagnostic-root "$VAL_ROOT" \
  --test-diagnostic-root "$TEST_ROOT" \
  --output-root "$CANDIDATE_ROOT" \
  --config "$REPO_ROOT/configs/raw_policy_generalization_candidate_v1.json"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_raw_policy_generalization_evaluation.sh" \
  --source-root "$SOURCE_ROOT" \
  --dev-root "$DEV_ROOT" \
  --val-root "$VAL_ROOT" \
  --test-root "$TEST_ROOT" \
  --baseline-candidate-root "$BASELINE_CANDIDATE_ROOT" \
  --candidate-root "$CANDIDATE_ROOT" \
  --config "$REPO_ROOT/configs/raw_policy_generalization_evaluation_v1.json"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_policy_training_readiness_review.sh" \
  --batch-root "$SOURCE_ROOT" \
  --config "$REPO_ROOT/configs/policy_training_readiness_review_v1.json" \
  --raw-policy-generalization-evaluation-summary "$CANDIDATE_ROOT/raw-policy-generalization-evaluation-summary.json"
