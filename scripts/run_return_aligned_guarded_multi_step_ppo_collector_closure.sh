#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/path_feedback_batch_return_aligned_guarded_multi_step_ppo_collector_expansion_v1}"
GUARDED_ROOT="${GUARDED_ROOT:-outputs/path_feedback_batch_guarded_ppo_rollout_pilot_v1}"
EVIDENCE_FREEZE_SUMMARY="${EVIDENCE_FREEZE_SUMMARY:-outputs/path_feedback_batch_guarded_ppo_evidence_freeze_v1/guarded-ppo-evidence-freeze-summary.json}"
BATCH_ROOT="${BATCH_ROOT:-outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1}"
CONFIG="${CONFIG:-configs/return_aligned_guarded_multi_step_ppo_collector_expansion_v1.json}"
READINESS_CONFIG="${READINESS_CONFIG:-configs/policy_training_readiness_review_v1.json}"

"$PYTHON_BIN" scripts/run_return_aligned_guarded_multi_step_ppo_collector_expansion.py \
  --guarded-root "$GUARDED_ROOT" \
  --evidence-freeze-summary "$EVIDENCE_FREEZE_SUMMARY" \
  --output-root "$OUTPUT_ROOT" \
  --config "$CONFIG"

bash scripts/run_policy_training_readiness_review.sh \
  --batch-root "$BATCH_ROOT" \
  --config "$READINESS_CONFIG" \
  --return-aligned-guarded-multistep-collector-summary "$OUTPUT_ROOT/return-aligned-collector-summary.json" \
  --validate-only
