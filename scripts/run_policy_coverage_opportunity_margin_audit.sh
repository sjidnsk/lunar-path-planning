#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/kai/anaconda3/envs/lunar-explorer/bin/python}"

"${PYTHON}" scripts/run_policy_coverage_opportunity_margin_audit.py \
  --coverage-driven-root outputs/path_feedback_batch_coverage_driven_ppo_improvement_run_v1 \
  --reward-refinement-root outputs/path_feedback_batch_coverage_aware_reward_refinement_v1 \
  --coverage-signal-root outputs/path_feedback_batch_exploration_coverage_signal_audit_v1 \
  --coverage-performance-root outputs/path_feedback_batch_exploration_coverage_performance_evaluation_v1 \
  --output-root outputs/path_feedback_batch_policy_coverage_opportunity_margin_audit_v1
