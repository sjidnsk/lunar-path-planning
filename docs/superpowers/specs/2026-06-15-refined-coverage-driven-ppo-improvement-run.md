# Refined Coverage-Driven PPO Improvement Run v2

## Purpose

This stage follows `Policy-Differentiating Counterfactual Coverage Rollouts v1`.
Stage 5A.2 produced 51 safe better-than-teacher non-teacher candidates with real
counterfactual exploration coverage evidence. This stage tests whether those
candidates can become reward, advantage, and margin training signal for a
guarded offline PPO update.

It may run a local experimental PPO update. It must not publish checkpoints,
replace the default policy, connect a real executor, relax guards, modify the
network/action space/default A*, or claim performance.

## Inputs

- `outputs/path_feedback_batch_policy_differentiating_counterfactual_coverage_rollouts_v1/`
- `outputs/path_feedback_batch_coverage_driven_ppo_improvement_run_v1/`
- `outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1/`
- `outputs/path_feedback_batch_guarded_formal_ppo_post_training_stability_replay_v1/`
- `outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_preflight_v1/`
- `outputs/path_feedback_batch_exploration_coverage_signal_audit_v1/`
- `outputs/path_feedback_batch_exploration_coverage_performance_evaluation_v1/`
- `outputs/path_feedback_batch_coverage_aware_reward_refinement_v1/`

## Implementation

Implemented entry points:

- `scripts/run_refined_coverage_driven_ppo_improvement_run.py`
- `scripts/run_refined_coverage_driven_ppo_improvement_run.sh`
- `tests/test_refined_coverage_driven_ppo_improvement_run.py`

Output root:

- `outputs/path_feedback_batch_refined_coverage_driven_ppo_improvement_run_v2/`

Produced artifacts:

- `refined-coverage-driven-ppo-improvement-run-summary.json`
- `coverage-driven-ppo-improvement-run-summary.json`
- `refined-coverage-ppo-batch/refined-trainable-transitions.jsonl`
- `coverage-aware-ppo-batch/ppo-rollout-episodes.jsonl`
- `coverage-aware-ppo-batch/ppo-rollout-transitions.jsonl`
- `coverage-aware-ppo-batch/ppo-rollout-collector-summary.json`
- `advantage-margin-audit.jsonl`
- `coverage-driven-ppo-update-summary.json`
- `coverage-driven-experimental-policy-candidate.pt`
- `coverage-driven-experimental-policy-candidate-metadata.json`
- `coverage-driven-ppo-replay-audit.json`
- `refined-coverage-driven-ppo-performance-metric-table.jsonl`
- `stage5a-rerun-summary.json`
- `refined-coverage-driven-ppo-improvement-run-report.md`

## Current Result

The current run fails the performance gate but passes the signal
materialization and PPO-consumption checks:

- `status=failed`
- `reason_codes=["post_update_policy_teacher_equivalent", "no_coverage_return_improvement", "no_cumulative_coverage_rate_delta_improvement", "fallback_dominates"]`
- `next_required_change=tune_refined_reward_margin_or_expand_safe_better_pairs`
- `safe_better_training_pair_count=51`
- `refined_trainable_transition_count=51`
- `counterfactual_advantage_nonzero_count=51`
- `optimizer_train_transition_count=51`
- `old_log_prob_max_abs_error=0.0`
- `old_value_max_abs_error=0.0`
- `parameter_l2_delta=0.0017301534299004352`
- `policy_argmax_changed_count=0`
- `coverage_return_improvement=-81.582958126732`
- `cumulative_coverage_rate_delta_improvement=-89.45694198338`
- `valuable_area_covered_improvement=0.0`
- `fallback_rate=1.0`
- `controlled_regression_count=0`
- `fallback_gain_contamination_count=0`
- `publishes_checkpoint=false`
- `replaces_default_policy=false`
- `performance_claimed=false`

## Interpretation

The refined batch now carries the missing decision-time training signal: safe
non-teacher actions have positive `counterfactual_coverage_advantage`,
`teacher_margin_target`, and `coverage_rank_margin`. The offline PPO update is
numerically valid and writes an experimental checkpoint.

The updated policy still does not change its effective argmax toward those
actions. Replay therefore cannot show coverage return improvement. This is a
reward/margin strength or sample-density failure, not a coverage telemetry
failure and not a release candidate.

## Acceptance

This stage can pass only when:

- `safe_better_training_pair_count>0`
- `counterfactual_advantage_nonzero_count>0`
- `policy_argmax_changed_count>0`
- `coverage_return_improvement>0`
- `cumulative_coverage_rate_delta_improvement>0`
- `valuable_area_covered_improvement>=0`
- `coverage_efficiency_regression=false`
- `controlled_regression_count=0`
- `fallback_gain_contamination_count=0`
- `publishes_checkpoint=false`
- `replaces_default_policy=false`
- `performance_claimed=false`

If the first two signal gates fail, route to
`fix_refined_advantage_materialization`. If signal exists but the policy remains
teacher-equivalent or coverage return does not improve, route to
`tune_refined_reward_margin_or_expand_safe_better_pairs`.

## Verification

Validated with:

```bash
/home/kai/anaconda3/envs/lunar-explorer/bin/python -m pytest tests/test_refined_coverage_driven_ppo_improvement_run.py tests/test_policy_differentiating_counterfactual_coverage_rollouts.py tests/test_policy_coverage_opportunity_margin_audit.py -q
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python bash scripts/run_refined_coverage_driven_ppo_improvement_run.sh
jq '{status,reason_codes,next_required_change,coverage_return_improvement,cumulative_coverage_rate_delta_improvement,policy_argmax_changed_count,performance_claimed}' outputs/path_feedback_batch_refined_coverage_driven_ppo_improvement_run_v2/refined-coverage-driven-ppo-improvement-run-summary.json
git diff --check
```
