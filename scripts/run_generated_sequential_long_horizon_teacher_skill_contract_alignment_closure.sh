#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON:-/home/kai/anaconda3/envs/lunar-explorer/bin/python}"
DIAGNOSIS_ROOT="${DIAGNOSIS_ROOT:-outputs/path_feedback_batch_quasi_real_generated_sequential_contract_compatibility_diagnosis_v1}"
ACCOUNTING_ROOT="${ACCOUNTING_ROOT:-outputs/path_feedback_batch_generated_sequential_gate_metric_accounting_audit_v1}"
UPDATE_ROOT="${UPDATE_ROOT:-outputs/path_feedback_batch_limited_quasi_real_ppo_update_smoke_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/path_feedback_batch_generated_sequential_long_horizon_teacher_skill_contract_alignment_v1}"
CONFIG="${CONFIG:-configs/generated_sequential_long_horizon_teacher_skill_contract_alignment_v1.json}"
READINESS_BATCH_ROOT="${READINESS_BATCH_ROOT:-outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1}"
READINESS_CONFIG="${READINESS_CONFIG:-configs/policy_training_readiness_review_v1.json}"

"$PYTHON_BIN" scripts/run_generated_sequential_long_horizon_teacher_skill_contract_alignment.py \
  --diagnosis-root "$DIAGNOSIS_ROOT" \
  --accounting-audit-root "$ACCOUNTING_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --config "$CONFIG" \
  --quasi-real-teacher-following-summary \
    "$UPDATE_ROOT/post_update_quasi_real_teacher_following/quasi-real-guarded-teacher-following-pilot-summary.json" \
  --quasi-real-collector-summary \
    "$UPDATE_ROOT/post_update_quasi_real_collector/ppo-rollout-collector-summary.json"

"$PYTHON_BIN" scripts/run_policy_training_readiness_review.py \
  --batch-root "$READINESS_BATCH_ROOT" \
  --config "$READINESS_CONFIG" \
  --quasi-real-guarded-teacher-following-pilot-summary \
    "$UPDATE_ROOT/post_update_quasi_real_teacher_following/quasi-real-guarded-teacher-following-pilot-summary.json" \
  --ppo-rollout-collector-summary \
    "$UPDATE_ROOT/post_update_quasi_real_collector/ppo-rollout-collector-summary.json" \
  --limited-quasi-real-ppo-update-smoke-summary \
    "$UPDATE_ROOT/limited-quasi-real-ppo-update-smoke-summary.json" \
  --generated-sequential-gate-metric-accounting-audit-summary \
    "$ACCOUNTING_ROOT/generated-sequential-gate-metric-accounting-audit-summary.json" \
  --generated-sequential-long-horizon-teacher-skill-contract-summary \
    "$OUTPUT_ROOT/long-horizon-teacher-skill-contract-summary.json" \
  --validate-only
