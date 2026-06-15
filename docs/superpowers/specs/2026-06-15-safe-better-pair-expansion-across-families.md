# Safe-Better Pair Expansion Across Families v1

## Purpose

This stage follows `Refined Coverage Reward/Margin Tuning v1`. The tuning run
proved that explicit PPO `ppo_advantage` and `ppo_return` fields can be
materialized, but all four configs still left the post-update policy
teacher-equivalent with no coverage-return improvement. The remaining blocker
is safe-better sample density and family diversity, not another reward scale on
the same 51 pairs.

This stage expands and audits candidate-vs-teacher safe-better coverage pairs
across scenario families. It does not run PPO, publish checkpoints, replace the
default policy, connect a real executor, relax guards, modify the
network/action space/default A*, or claim performance.

## Inputs

- `outputs/path_feedback_batch_refined_coverage_reward_margin_tuning_v1/`
- `outputs/path_feedback_batch_policy_differentiating_counterfactual_coverage_rollouts_v1/`
- `outputs/path_feedback_batch_quasi_real_safe_better_opportunity_expansion_v1/`
- `outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1/`
- `outputs/path_feedback_batch_guarded_formal_ppo_post_training_stability_replay_v1/`
- `outputs/path_feedback_batch_exploration_coverage_signal_audit_v1/`
- `outputs/path_feedback_batch_coverage_aware_reward_refinement_v1/`

## Implementation

Implemented entry points:

- `scripts/run_safe_better_pair_expansion_across_families.py`
- `scripts/run_safe_better_pair_expansion_across_families.sh`
- `tests/test_safe_better_pair_expansion_across_families.py`

Output root:

- `outputs/path_feedback_batch_safe_better_pair_expansion_across_families_v1/`

Produced artifacts:

- `safe-better-pair-expansion-summary.json`
- `expanded-safe-better-pairs.jsonl`
- `expanded-counterfactual-coverage-rollouts.jsonl`
- `family-safe-better-gap-report.json`
- `safe-better-pair-expansion-report.md`

## Contract

The stage reads the Stage 5A.2 candidate overlay and counterfactual coverage
rows, groups candidates by `context_id + episode_id + step_index`, finds the
teacher action, and recomputes:

- `coverage_advantage = candidate.expected_coverage_rate_delta - teacher.expected_coverage_rate_delta`
- `new_area_advantage`
- `information_gain_advantage`
- `valuable_coverage_advantage`
- `path_cost_delta`
- `risk_delta`
- `energy_delta`

Only `train` split candidates that are guard-clean, fallback-free,
counterfactual-source-backed, and have positive coverage advantage are marked
`ppo_trainable=true`. `validation` and `test` positives remain diagnostic-only.

## Current Result

The current run is a controlled failure that expands the data but does not yet
pass the cross-family gate:

- `status=failed`
- `reason_codes=["safe_better_family_count_below_threshold", "family_safe_better_gap_low_observation_count"]`
- `next_required_change=expand_safe_better_pair_generation_across_families`
- `safe_better_than_teacher_candidate_count=615`
- `safe_better_than_teacher_family_count=3`
- `trainable_safe_better_pair_count=615`
- `diagnostic_safe_better_pair_count=0`
- `family_safe_better_counts={"mixed_risk":231,"rim_or_steep_slope":186,"smooth_high_confidence":198}`
- `low_observation_count=0` in the family gap report
- `missing_counterfactual_source_count=0`
- `fallback_gain_contamination_count=0`
- `controlled_regression_count=0`
- `runs_new_ppo_update=false`
- `publishes_checkpoint=false`
- `replaces_default_policy=false`
- `performance_claimed=false`

## Interpretation

The stage proves that the current Stage 5A.2 overlay contains much more
trainable safe-better signal than the 51 pairs used by Stage 5B.2, but that
signal still covers only three families. The missing bridge is targeted
`low_observation_count` counterfactual opportunity generation. Until at least 8
trainable low-observation safe-better pairs exist, the next refined PPO run
would still be trained on an imbalanced coverage-improvement distribution.

## Acceptance

This stage can pass only when:

- `safe_better_than_teacher_candidate_count>=128`
- `safe_better_than_teacher_family_count>=4`
- every target family has at least 8 trainable safe-better pairs
- `missing_counterfactual_source_count=0`
- `fallback_gain_contamination_count=0`
- `controlled_regression_count=0`
- `runs_new_ppo_update=false`
- `publishes_checkpoint=false`
- `replaces_default_policy=false`
- `performance_claimed=false`

If the pair count or family coverage is insufficient, the stage must fail with
explicit `reason_codes`; it must not use placeholder or default-zero coverage
gain to pass the gate.

## Verification

Validated with:

```bash
/home/kai/anaconda3/envs/lunar-explorer/bin/python -m pytest tests/test_safe_better_pair_expansion_across_families.py tests/test_policy_differentiating_counterfactual_coverage_rollouts.py tests/test_refined_coverage_reward_margin_tuning.py -q
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python bash scripts/run_safe_better_pair_expansion_across_families.sh
jq '{status,reason_codes,next_required_change,safe_better_than_teacher_candidate_count,safe_better_than_teacher_family_count,missing_counterfactual_source_count,performance_claimed}' outputs/path_feedback_batch_safe_better_pair_expansion_across_families_v1/safe-better-pair-expansion-summary.json
git diff --check
```
