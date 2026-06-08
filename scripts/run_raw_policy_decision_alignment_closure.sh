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
BASELINE_CANDIDATE_ROOT=""
CANDIDATE_ROOT=""
HOLDOUT_ROOT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-root)
      SOURCE_ROOT="$2"
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
    --holdout-root)
      HOLDOUT_ROOT="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$SOURCE_ROOT" || -z "$BASELINE_CANDIDATE_ROOT" || -z "$CANDIDATE_ROOT" || -z "$HOLDOUT_ROOT" ]]; then
  echo "usage: $0 --source-root ROOT --baseline-candidate-root ROOT --candidate-root ROOT --holdout-root ROOT" >&2
  exit 2
fi

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_scenario_disjoint_policy_rollout_evaluation.sh" \
  --source-root "$SOURCE_ROOT" \
  --candidate-root "$BASELINE_CANDIDATE_ROOT" \
  --batch-root "$HOLDOUT_ROOT" \
  --config "$REPO_ROOT/configs/scenario_disjoint_policy_rollout_evaluation_v1.json"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_raw_policy_regression_mining.sh" \
  --source-root "$SOURCE_ROOT" \
  --holdout-root "$HOLDOUT_ROOT" \
  --config "$REPO_ROOT/configs/raw_policy_regression_mining_v1.json"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_raw_policy_decision_alignment_candidate.sh" \
  --source-root "$SOURCE_ROOT" \
  --raw-mining-root "$HOLDOUT_ROOT" \
  --output-root "$CANDIDATE_ROOT" \
  --config "$REPO_ROOT/configs/raw_policy_decision_alignment_candidate_v1.json"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_raw_policy_strict_rollout_evaluation.sh" \
  --source-root "$SOURCE_ROOT" \
  --candidate-root "$CANDIDATE_ROOT" \
  --batch-root "$HOLDOUT_ROOT" \
  --config "$REPO_ROOT/configs/raw_policy_strict_rollout_evaluation_v1.json"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_policy_training_readiness_review.sh" \
  --batch-root "$SOURCE_ROOT" \
  --config "$REPO_ROOT/configs/policy_training_readiness_review_v1.json" \
  --raw-policy-strict-rollout-evaluation-summary "$HOLDOUT_ROOT/raw-policy-strict-rollout-evaluation-summary.json"
