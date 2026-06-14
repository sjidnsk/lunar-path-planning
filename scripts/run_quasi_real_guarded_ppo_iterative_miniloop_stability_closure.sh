#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
OUTPUT_ROOT="outputs/path_feedback_batch_quasi_real_guarded_ppo_iterative_miniloop_stability_v1"

"${PYTHON_BIN}" scripts/run_quasi_real_guarded_ppo_iterative_miniloop_stability.py \
  --expansion-root outputs/path_feedback_batch_quasi_real_trainable_context_expansion_v1 \
  --scale512-root outputs/path_feedback_batch_quasi_real_trainable_context_expansion_v1/scale512_rerun \
  --batch-root outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1 \
  --output-root "${OUTPUT_ROOT}" \
  --config configs/quasi_real_guarded_ppo_iterative_miniloop_stability_v1.json

"${PYTHON_BIN}" scripts/run_policy_training_readiness_review.py \
  --batch-root outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1 \
  --config configs/policy_training_readiness_review_v1.json \
  --quasi-real-guarded-ppo-iterative-miniloop-stability-summary \
    "${OUTPUT_ROOT}/quasi-real-guarded-ppo-iterative-miniloop-stability-summary.json" \
  --validate-only
