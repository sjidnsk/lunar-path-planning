#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_PYTHON="/home/kai/anaconda3/envs/lunar-explorer/bin/python"
PYTHON_BIN="${PYTHON:-$DEFAULT_PYTHON}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

SRC="${SRC:-outputs/path_feedback_batch_sequential_multi_step_opportunity_clean_src_v1}"
BASE="${BASE:-outputs/path_feedback_batch_sequential_multi_step_opportunity_baseline_candidate_v1}"
DEV="${DEV:-outputs/path_feedback_batch_sequential_multi_step_opportunity_dev_v1}"
TRAIN="${TRAIN:-outputs/path_feedback_batch_sequential_multi_step_opportunity_train_v1}"
VAL="${VAL:-outputs/path_feedback_batch_sequential_multi_step_opportunity_val_v1}"
TEST="${TEST:-outputs/path_feedback_batch_sequential_multi_step_opportunity_test_v1}"
CAND="${CAND:-outputs/path_feedback_batch_sequential_multi_step_opportunity_candidate_v1}"
WARMUP_SEQ="${WARMUP_SEQ:-outputs/path_feedback_batch_policy_gated_sequential_multi_step_opportunity_warmup_v1}"
PREFLIGHT_SEQ="${PREFLIGHT_SEQ:-outputs/path_feedback_batch_policy_gated_sequential_multi_step_opportunity_preflight_v1}"
SEQ="${SEQ:-outputs/path_feedback_batch_policy_gated_sequential_multi_step_opportunity_rollout_v1}"
STATIC="${STATIC:-outputs/path_feedback_batch_sequential_multi_step_opportunity_v1}"
FAILED_SEQUENTIAL="${FAILED_SEQUENTIAL:-outputs/path_feedback_batch_policy_gated_sequential_canary_rollout_v1}"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_batch_path_feedback_validation.sh" \
  --matrix "$REPO_ROOT/configs/path_feedback_batch_sequential_multi_step_opportunity_v1.json" \
  --output-root "$STATIC"

set +e
PYTHON="$PYTHON_BIN" \
SRC="$SRC" \
BASE="$BASE" \
DEV="$DEV" \
TRAIN="$TRAIN" \
VAL="$VAL" \
TEST="$TEST" \
CAND="$CAND" \
SEQUENTIAL="$WARMUP_SEQ" \
FAILED_SEQUENTIAL="$FAILED_SEQUENTIAL" \
bash "$SCRIPT_DIR/run_sequential_safe_choice_calibration_closure.sh"
SAFE_CHOICE_STATUS=$?
set -e

if [[ ! -f "$CAND/experimental-hybrid-policy-candidate.pt" ]]; then
  echo "sequential safe-choice candidate was not produced; upstream status=$SAFE_CHOICE_STATUS" >&2
  exit "${SAFE_CHOICE_STATUS:-1}"
fi

run_multi_step_rollout() {
  local batch_root="$1"
  PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_policy_gated_sequential_canary_rollout.sh" \
    --source-root "$SRC" \
    --candidate-root "$CAND" \
    --batch-root "$batch_root" \
    --config "$REPO_ROOT/configs/policy_gated_sequential_multi_step_opportunity_rollout_v1.json"
}

run_diagnosis() {
  local batch_root="$1"
  PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_sequential_multi_step_opportunity_diagnosis.sh" \
    --batch-root "$batch_root" \
    --config "$REPO_ROOT/configs/sequential_multi_step_opportunity_diagnosis_v1.json"
}

set +e
run_multi_step_rollout "$PREFLIGHT_SEQ"
PREFLIGHT_SEQUENTIAL_STATUS=$?
run_diagnosis "$PREFLIGHT_SEQ"
DIAGNOSIS_STATUS=$?
set -e

MISSED_COUNT="$("$PYTHON_BIN" - "$PREFLIGHT_SEQ/sequential-multi-step-opportunity-diagnosis-summary.json" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    print(0)
else:
    print(int(json.loads(path.read_text()).get("policy_missed_existing_opportunity_count", 0)))
PY
)"
DIAGNOSIS_PASSED="$("$PYTHON_BIN" - "$PREFLIGHT_SEQ/sequential-multi-step-opportunity-diagnosis-summary.json" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    print("false")
else:
    print("true" if json.loads(path.read_text()).get("status") == "passed" else "false")
PY
)"

if [[ "$PREFLIGHT_SEQUENTIAL_STATUS" -ne 0 && "$DIAGNOSIS_PASSED" == "true" && "$MISSED_COUNT" -gt 0 ]]; then
  PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_sequential_canary_failure_mining.sh" \
    --batch-root "$PREFLIGHT_SEQ" \
    --config "$REPO_ROOT/configs/sequential_canary_failure_mining_v1.json"

  PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_sequential_safe_choice_calibration_candidate.sh" \
    --source-root "$SRC" \
    --train-mining-root "$TRAIN" \
    --dev-mining-root "$DEV" \
    --sequential-mining-root "$PREFLIGHT_SEQ" \
    --val-diagnostic-root "$VAL" \
    --test-diagnostic-root "$TEST" \
    --output-root "$CAND" \
    --config "$REPO_ROOT/configs/sequential_safe_choice_calibration_candidate_v1.json"

  PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_raw_policy_generalization_evaluation.sh" \
    --source-root "$SRC" \
    --dev-root "$DEV" \
    --val-root "$VAL" \
    --test-root "$TEST" \
    --baseline-candidate-root "$BASE" \
    --candidate-root "$CAND" \
    --config "$REPO_ROOT/configs/raw_policy_generalization_evaluation_v1.json"

  set +e
  run_multi_step_rollout "$SEQ"
  SEQUENTIAL_STATUS=$?
  set -e
else
  rm -rf "$SEQ"
  cp -a "$PREFLIGHT_SEQ" "$SEQ"
  SEQUENTIAL_STATUS=$PREFLIGHT_SEQUENTIAL_STATUS
fi

set +e
PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_policy_training_readiness_review.sh" \
  --batch-root "$SRC" \
  --config "$REPO_ROOT/configs/policy_training_readiness_review_v1.json" \
  --raw-policy-generalization-evaluation-summary "$CAND/raw-policy-generalization-evaluation-summary.json" \
  --policy-gated-sequential-canary-rollout-summary "$SEQ/policy-gated-sequential-canary-rollout-summary.json"
READINESS_STATUS=$?
set -e

if [[ "$SEQUENTIAL_STATUS" -ne 0 ]]; then
  exit "$SEQUENTIAL_STATUS"
fi
if [[ "$DIAGNOSIS_STATUS" -ne 0 ]]; then
  exit "$DIAGNOSIS_STATUS"
fi
exit "$READINESS_STATUS"
