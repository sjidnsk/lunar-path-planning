#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_PYTHON="/home/kai/anaconda3/envs/lunar-explorer/bin/python"
PYTHON_BIN="${PYTHON:-$DEFAULT_PYTHON}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

SRC="${SRC:-outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1}"
BASE="${BASE:-outputs/path_feedback_batch_quasi_real_teacher_distillation_candidate_v1}"
QUASI="${QUASI:-outputs/path_feedback_batch_quasi_real_safe_better_opportunity_expansion_v1}"
GUARDED_TEACHER="${GUARDED_TEACHER:-outputs/path_feedback_batch_quasi_real_guarded_teacher_following_pilot_v1}"
COLLECTOR="${COLLECTOR:-outputs/path_feedback_batch_quasi_real_ppo_collector_dry_run_v1}"
UPDATE="${UPDATE:-outputs/path_feedback_batch_limited_quasi_real_ppo_update_smoke_v1}"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_quasi_real_ppo_collector_closure.sh"

rm -rf "$REPO_ROOT/$UPDATE"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_limited_quasi_real_ppo_update_smoke.sh" \
  --source-root "$SRC" \
  --base-candidate-root "$BASE" \
  --collector-root "$COLLECTOR" \
  --output-root "$UPDATE" \
  --quasi-real-root "$QUASI" \
  --guarded-teacher-following-root "$GUARDED_TEACHER" \
  --config "$REPO_ROOT/configs/limited_quasi_real_ppo_update_smoke_v1.json"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_policy_training_readiness_review.sh" \
  --batch-root "$SRC" \
  --config "$REPO_ROOT/configs/policy_training_readiness_review_v1.json" \
  --quasi-real-guarded-teacher-following-pilot-summary \
    "$UPDATE/post_update_quasi_real_teacher_following/quasi-real-guarded-teacher-following-pilot-summary.json" \
  --ppo-rollout-collector-summary \
    "$UPDATE/post_update_quasi_real_collector/ppo-rollout-collector-summary.json" \
  --limited-quasi-real-ppo-update-smoke-summary \
    "$UPDATE/limited-quasi-real-ppo-update-smoke-summary.json" \
  --validate-only
