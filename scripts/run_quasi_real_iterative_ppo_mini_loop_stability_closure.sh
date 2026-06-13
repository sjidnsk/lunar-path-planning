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
INITIAL="${INITIAL:-outputs/path_feedback_batch_quasi_real_teacher_distillation_candidate_v1}"
QUASI="${QUASI:-outputs/path_feedback_batch_quasi_real_safe_better_opportunity_expansion_v1}"
OUT="${OUT:-outputs/path_feedback_batch_quasi_real_iterative_ppo_mini_loop_stability_v1}"
CONFIG="${CONFIG:-configs/quasi_real_iterative_ppo_mini_loop_stability_v1.json}"

rm -rf "$REPO_ROOT/$OUT"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_quasi_real_iterative_ppo_mini_loop_stability.sh" \
  --source-root "$SRC" \
  --initial-candidate-root "$INITIAL" \
  --quasi-real-root "$QUASI" \
  --output-root "$OUT" \
  --config "$REPO_ROOT/$CONFIG"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_policy_training_readiness_review.sh" \
  --batch-root "$SRC" \
  --config "$REPO_ROOT/configs/policy_training_readiness_review_v1.json" \
  --iterative-ppo-mini-loop-stability-summary \
    "$OUT/quasi-real-iterative-ppo-mini-loop-stability-summary.json" \
  --validate-only
