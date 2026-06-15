# Refined Coverage Reward/Margin Tuning v1

## Purpose

This stage follows `Refined Coverage-Driven PPO Improvement Run v2`.
Refined v2 proved that 51 safe better-than-teacher non-teacher candidates can
be materialized as trainable counterfactual advantage/margin records, but the
post-update policy still did not improve exploration coverage. This stage tests
whether explicit PPO `transition_info` advantages and returns can make the
same safe-better signal strong enough to change policy action ranking.

It may run local experimental PPO updates. It must not publish checkpoints,
replace the default policy, connect a real executor, relax guards, modify the
network/action space/default A*, or claim performance.

## Inputs

- `outputs/path_feedback_batch_refined_coverage_driven_ppo_improvement_run_v2/`
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

- `scripts/run_refined_coverage_reward_margin_tuning.py`
- `scripts/run_refined_coverage_reward_margin_tuning.sh`
- `tests/test_refined_coverage_reward_margin_tuning.py`

Output root:

- `outputs/path_feedback_batch_refined_coverage_reward_margin_tuning_v1/`

Produced artifacts:

- `refined-coverage-reward-margin-tuning-summary.json`
- `tuning-config-audit.jsonl`
- `rejection-report.json`
- `refined-coverage-reward-margin-tuning-report.md`
- `tuning-configs/<config_id>/coverage-aware-ppo-batch/ppo-rollout-episodes.jsonl`
- `tuning-configs/<config_id>/coverage-driven-ppo-update-summary.json`
- `tuning-configs/<config_id>/coverage-driven-ppo-replay-audit.json`
- `tuning-configs/<config_id>/coverage-driven-ppo-performance-metric-table.jsonl`
- `tuning-configs/<config_id>/stage5a-rerun/`

## Current Result

The current run fails the performance gate while proving explicit PPO
advantage/return materialization:

- `status=failed`
- `reason_codes=["post_update_policy_teacher_equivalent", "no_coverage_return_improvement", "no_cumulative_coverage_rate_delta_improvement", "fallback_dominates"]`
- `next_required_change=expand_safe_better_pair_generation_across_families`
- `tuning_config_count=4`
- `executed_tuning_config_count=4`
- `best_config_id=advantage_x3`
- `safe_better_training_pair_count=51`
- `ppo_advantage_nonzero_count=204`
- `policy_argmax_changed_count=0`
- `coverage_return_improvement=-81.582958126732`
- `cumulative_coverage_rate_delta_improvement=-89.45694198338`
- `valuable_area_covered_improvement=0.0`
- `coverage_efficiency_regression=false`
- `accepted_policy_activation_rate=0.0`
- `fallback_rate=1.0`
- `controlled_regression_count=0`
- `fallback_gain_contamination_count=0`
- `runs_new_ppo_update=true`
- `publishes_checkpoint=false`
- `replaces_default_policy=false`
- `performance_claimed=false`

The evaluated configs were `advantage_x3`, `advantage_x5`, `advantage_x8`,
and `reward_margin_x5`. Each config wrote nonzero `ppo_advantage` and
`ppo_return` into transition `info` and used limited PPO
`training.return_source=transition_info`.

## Interpretation

The failure is no longer an advantage materialization problem. PPO can consume
the explicit tuned advantage fields, and all four updates are numerically
valid. The remaining blocker is that 51 safe-better pairs across only two
families are not enough to shift the policy argmax and produce accepted
coverage gain under replay.

The next stage should expand safe-better pair generation across more scenario
families and longer horizons, while keeping fallback/source coverage gain
separate from policy gain.

## Acceptance

This stage can pass only when:

- `ppo_advantage_nonzero_count>0`
- `policy_argmax_changed_count>0`
- `coverage_return_improvement>0`
- `cumulative_coverage_rate_delta_improvement>0`
- `valuable_area_covered_improvement>=0`
- `coverage_efficiency_regression=false`
- `fallback_rate<0.5`
- `controlled_regression_count=0`
- `fallback_gain_contamination_count=0`
- `publishes_checkpoint=false`
- `replaces_default_policy=false`
- `performance_claimed=false`

If explicit advantage fields cannot be materialized, route to
`fix_tuning_advantage_materialization`. If tuning materializes but the policy
remains teacher-equivalent or coverage return does not improve, route to
`expand_safe_better_pair_generation_across_families`.

## Verification

Validated with:

```bash
/home/kai/anaconda3/envs/lunar-explorer/bin/python -m pytest tests/test_refined_coverage_reward_margin_tuning.py tests/test_refined_coverage_driven_ppo_improvement_run.py tests/test_policy_coverage_opportunity_margin_audit.py -q
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python bash scripts/run_refined_coverage_reward_margin_tuning.sh
jq '{status,reason_codes,next_required_change,best_config_id,policy_argmax_changed_count,coverage_return_improvement,cumulative_coverage_rate_delta_improvement,performance_claimed}' outputs/path_feedback_batch_refined_coverage_reward_margin_tuning_v1/refined-coverage-reward-margin-tuning-summary.json
git diff --check
```
