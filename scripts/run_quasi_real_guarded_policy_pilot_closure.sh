#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON:-/home/kai/anaconda3/envs/lunar-explorer/bin/python}"

SOURCE_ROOT="${SOURCE_ROOT:-outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1}"
CANDIDATE_ROOT="${CANDIDATE_ROOT:-outputs/path_feedback_batch_quasi_real_shadow_alignment_candidate_v1}"
QUASI_REAL_ROOT="${QUASI_REAL_ROOT:-outputs/path_feedback_batch_quasi_real_map_domain_gap_v1}"
CUDA_ROOT="${CUDA_ROOT:-outputs/path_feedback_batch_policy_training_cuda_device_support_v1}"
SHADOW_ROOT="${SHADOW_ROOT:-outputs/path_feedback_batch_quasi_real_shadow_policy_behavior_v1}"
ALIGNMENT_ROOT="${ALIGNMENT_ROOT:-outputs/path_feedback_batch_quasi_real_shadow_alignment_candidate_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/path_feedback_batch_quasi_real_guarded_policy_pilot_v1}"
CONFIG="${CONFIG:-configs/quasi_real_guarded_policy_pilot_v1.json}"

rm -rf "$REPO_ROOT/$OUTPUT_ROOT"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_quasi_real_guarded_policy_pilot.sh" \
  --source-root "$SOURCE_ROOT" \
  --candidate-root "$CANDIDATE_ROOT" \
  --quasi-real-root "$QUASI_REAL_ROOT" \
  --alignment-summary "$ALIGNMENT_ROOT/quasi-real-shadow-alignment-summary.json" \
  --output-root "$OUTPUT_ROOT" \
  --config "$CONFIG"

PYTHON="$PYTHON_BIN" bash "$SCRIPT_DIR/run_policy_training_readiness_review.sh" \
  --batch-root "$SOURCE_ROOT" \
  --config configs/policy_training_readiness_review_v1.json \
  --policy-training-cuda-device-support-summary "$CUDA_ROOT/policy-training-cuda-device-support-summary.json" \
  --quasi-real-map-domain-gap-summary "$QUASI_REAL_ROOT/quasi-real-map-domain-gap-summary.json" \
  --quasi-real-shadow-policy-behavior-summary "$SHADOW_ROOT/quasi-real-shadow-policy-behavior-summary.json" \
  --quasi-real-shadow-alignment-summary "$ALIGNMENT_ROOT/quasi-real-shadow-alignment-summary.json" \
  --quasi-real-guarded-policy-pilot-summary "$OUTPUT_ROOT/quasi-real-guarded-policy-pilot-summary.json"
