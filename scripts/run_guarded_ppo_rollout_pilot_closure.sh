#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_PYTHON="/home/kai/anaconda3/envs/lunar-explorer/bin/python"
PYTHON_BIN="${PYTHON:-$DEFAULT_PYTHON}"
PROGRESS_MODE="auto"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --progress)
      PROGRESS_MODE="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

SOURCE_SRC="${SOURCE_SRC:-outputs/path_feedback_batch_ppo_collector_clean_src_v1}"
SRC="${SRC:-outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1}"
ITERATIVE="${ITERATIVE:-outputs/path_feedback_batch_quasi_real_iterative_ppo_mini_loop_stability_v1}"
BASE="${BASE:-$ITERATIVE/round-02/update}"
RAW_BASE="${RAW_BASE:-outputs/path_feedback_batch_sequential_multi_step_opportunity_baseline_candidate_v1}"
DEV="${DEV:-outputs/path_feedback_batch_sequential_multi_step_opportunity_dev_v1}"
VAL="${VAL:-outputs/path_feedback_batch_sequential_multi_step_opportunity_val_v1}"
TEST="${TEST:-outputs/path_feedback_batch_sequential_multi_step_opportunity_test_v1}"
QUASI_REAL="${QUASI_REAL:-outputs/path_feedback_batch_quasi_real_safe_better_opportunity_expansion_v1}"
OUT="${OUT:-outputs/path_feedback_batch_guarded_ppo_rollout_pilot_v1}"
PROGRESS_RUN_ID="${PROGRESS_RUN_ID:-guarded-ppo-rollout-pilot-$(date -u +%Y%m%dT%H%M%SZ)}"

emit_progress() {
  "$PYTHON_BIN" "$REPO_ROOT/scripts/training_progress.py" emit \
    --output-root "$OUT" \
    --mode "$PROGRESS_MODE" \
    --run-id "$PROGRESS_RUN_ID" \
    "$@"
}

finalize_progress() {
  "$PYTHON_BIN" "$REPO_ROOT/scripts/training_progress.py" finalize \
    --output-root "$OUT" \
    --mode "$PROGRESS_MODE" \
    --run-id "$PROGRESS_RUN_ID" \
    "$@"
}

rm -rf "$SRC" "$OUT"
cp -a "$SOURCE_SRC" "$SRC"

export TRAINING_PROGRESS_ROOT="$OUT"
export TRAINING_PROGRESS_MODE="$PROGRESS_MODE"
export TRAINING_PROGRESS_RUN_ID="$PROGRESS_RUN_ID"

emit_progress \
  --stage guarded_ppo_rollout_pilot_closure \
  --status start \
  --current 0 \
  --total 3 \
  --message "guarded PPO rollout pilot closure"

emit_progress \
  --stage iterative_precondition \
  --status start \
  --current 1 \
  --total 3 \
  --message "refresh iterative PPO mini-loop stability"

if PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_quasi_real_iterative_ppo_mini_loop_stability_closure.sh"; then
  emit_progress \
    --stage iterative_precondition \
    --status passed \
    --current 1 \
    --total 3 \
    --summary-path "$ITERATIVE/quasi-real-iterative-ppo-mini-loop-stability-summary.json" \
    --message "iterative PPO mini-loop stability refreshed"
else
  emit_progress \
    --stage iterative_precondition \
    --status failed \
    --current 1 \
    --total 3 \
    --summary-path "$ITERATIVE/quasi-real-iterative-ppo-mini-loop-stability-summary.json" \
    --message "iterative PPO mini-loop stability failed"
  finalize_progress --status failed --recommended-debug-artifact "$ITERATIVE/quasi-real-iterative-ppo-mini-loop-stability-summary.json"
  exit 1
fi

emit_progress \
  --stage guarded_pilot \
  --status start \
  --current 2 \
  --total 3 \
  --message "run guarded PPO rollout pilot"

if PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_guarded_ppo_rollout_pilot.sh" \
  --source-root "$SRC" \
  --base-candidate-root "$BASE" \
  --raw-baseline-candidate-root "$RAW_BASE" \
  --dev-root "$DEV" \
  --val-root "$VAL" \
  --test-root "$TEST" \
  --quasi-real-root "$QUASI_REAL" \
  --output-root "$OUT" \
  --config "$REPO_ROOT/configs/guarded_ppo_rollout_pilot_v1.json" \
  --progress "$PROGRESS_MODE"; then
  emit_progress \
    --stage guarded_pilot \
    --status passed \
    --current 2 \
    --total 3 \
    --summary-path "$OUT/guarded-ppo-rollout-pilot-summary.json" \
    --message "guarded PPO rollout pilot completed"
else
  emit_progress \
    --stage guarded_pilot \
    --status failed \
    --current 2 \
    --total 3 \
    --summary-path "$OUT/guarded-ppo-rollout-pilot-summary.json" \
    --message "guarded PPO rollout pilot failed"
  finalize_progress --status failed --recommended-debug-artifact "$OUT/guarded-ppo-rollout-pilot-summary.json"
  exit 1
fi

emit_progress \
  --stage readiness \
  --status start \
  --current 3 \
  --total 3 \
  --message "validate guarded PPO rollout pilot readiness"

READINESS_ERR="$(mktemp)"
if READINESS_OUTPUT="$(PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_policy_training_readiness_review.sh" \
  --batch-root "$SRC" \
  --config "$REPO_ROOT/configs/policy_training_readiness_review_v1.json" \
  --guarded-ppo-rollout-pilot-summary "$OUT/guarded-ppo-rollout-pilot-summary.json" \
  --validate-only 2>"$READINESS_ERR")"; then
  cat "$READINESS_ERR" >&2
  printf '%s\n' "$READINESS_OUTPUT"
  READINESS_STATUS="$(printf '%s\n' "$READINESS_OUTPUT" | "$PYTHON_BIN" -c 'import json,sys; print(json.load(sys.stdin).get("training_readiness_status") or "")')"
  emit_progress \
    --stage readiness \
    --status passed \
    --current 3 \
    --total 3 \
    --summary-path "$SRC/policy-training-readiness-review-summary.json" \
    --message "readiness $READINESS_STATUS" \
    --metric "readiness_status=$READINESS_STATUS"
  emit_progress \
    --stage guarded_ppo_rollout_pilot_closure \
    --status passed \
    --current 3 \
    --total 3 \
    --message "guarded PPO rollout pilot closure passed"
  finalize_progress --status passed --readiness-status "$READINESS_STATUS"
else
  READINESS_RC=$?
  cat "$READINESS_ERR" >&2
  emit_progress \
    --stage readiness \
    --status failed \
    --current 3 \
    --total 3 \
    --summary-path "$SRC/policy-training-readiness-review-summary.json" \
    --message "readiness validation failed"
  finalize_progress --status failed --recommended-debug-artifact "$SRC/policy-training-readiness-review-summary.json"
  rm -f "$READINESS_ERR"
  exit "$READINESS_RC"
fi
rm -f "$READINESS_ERR"
