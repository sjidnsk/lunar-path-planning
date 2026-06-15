#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/kai/anaconda3/envs/lunar-explorer/bin/python}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${PYTHON}" "${REPO_ROOT}/scripts/run_refined_coverage_driven_ppo_improvement_run.py" \
  --stage5a2-root "${REPO_ROOT}/outputs/path_feedback_batch_policy_differentiating_counterfactual_coverage_rollouts_v1" \
  --coverage-driven-root "${REPO_ROOT}/outputs/path_feedback_batch_coverage_driven_ppo_improvement_run_v1" \
  --formal-training-root "${REPO_ROOT}/outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1" \
  --post-training-replay-root "${REPO_ROOT}/outputs/path_feedback_batch_guarded_formal_ppo_post_training_stability_replay_v1" \
  --selected-candidate-root "${REPO_ROOT}/outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_preflight_v1" \
  --coverage-signal-root "${REPO_ROOT}/outputs/path_feedback_batch_exploration_coverage_signal_audit_v1" \
  --coverage-performance-root "${REPO_ROOT}/outputs/path_feedback_batch_exploration_coverage_performance_evaluation_v1" \
  --reward-refinement-root "${REPO_ROOT}/outputs/path_feedback_batch_coverage_aware_reward_refinement_v1" \
  --output-root "${REPO_ROOT}/outputs/path_feedback_batch_refined_coverage_driven_ppo_improvement_run_v2" \
  --repo-root "${REPO_ROOT}"
