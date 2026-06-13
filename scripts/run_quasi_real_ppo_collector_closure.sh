#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_PYTHON="/home/kai/anaconda3/envs/lunar-explorer/bin/python"
PYTHON_BIN="${PYTHON:-$DEFAULT_PYTHON}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

GUARDED_TEACHER="${GUARDED_TEACHER:-outputs/path_feedback_batch_quasi_real_guarded_teacher_following_pilot_v1}"
COLLECTOR="${COLLECTOR:-outputs/path_feedback_batch_quasi_real_ppo_collector_dry_run_v1}"
SRC="${SRC:-outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1}"

rm -rf "$REPO_ROOT/$COLLECTOR"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_quasi_real_ppo_collector_dry_run.sh" \
  --guarded-teacher-following-root "$GUARDED_TEACHER" \
  --output-root "$COLLECTOR" \
  --config "$REPO_ROOT/configs/quasi_real_ppo_collector_dry_run_v1.json"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_policy_training_readiness_review.sh" \
  --batch-root "$SRC" \
  --config "$REPO_ROOT/configs/policy_training_readiness_review_v1.json" \
  --quasi-real-guarded-teacher-following-pilot-summary \
    "$GUARDED_TEACHER/quasi-real-guarded-teacher-following-pilot-summary.json" \
  --ppo-rollout-collector-summary \
    "$COLLECTOR/ppo-rollout-collector-summary.json" \
  --validate-only
