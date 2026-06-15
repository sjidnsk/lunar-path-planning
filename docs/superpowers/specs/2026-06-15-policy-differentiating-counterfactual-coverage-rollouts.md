# Policy-Differentiating Counterfactual Coverage Rollouts v1

## Purpose

This stage follows `Candidate-Level Exploration Coverage Materialization v1`.
Stage 5A.1 proved that executed teacher/selected actions had real
path-feedback coverage, but 3,456 non-teacher candidates lacked counterfactual
coverage source fields. This stage fills that decision-time data gap from
existing offline artifacts only. It does not run PPO, publish checkpoints,
replace the default policy, connect a real executor, relax guards, or claim
performance.

## Inputs

- `outputs/path_feedback_batch_policy_coverage_opportunity_margin_audit_v1/`
- `outputs/path_feedback_batch_candidate_level_coverage_materialization_v1/`
- `outputs/path_feedback_batch_coverage_driven_ppo_improvement_run_v1/`
- `outputs/path_feedback_batch_coverage_aware_reward_refinement_v1/`
- `outputs/path_feedback_batch_exploration_coverage_signal_audit_v1/`
- `outputs/path_feedback_batch_exploration_coverage_performance_evaluation_v1/`
- `outputs/path_feedback_batch_quasi_real_safe_better_opportunity_expansion_v1/`

## Required Audit

For each Stage 5A.1 missing non-teacher candidate, the stage links the action
row back to quasi-real path-feedback candidates by context/scenario/action/cell
keys, then computes counterfactual fields from:

- candidate `diagnostics.expanded_cells`
- path-planner sidecar `passable_mask`
- sidecar terrain `confidence`
- prior executed coverage state within the same episode
- candidate `path_cost`, `risk`, `energy_cost`, and `utility`

Produced fields include `expected_coverage_rate_delta`,
`expected_new_coverage_area`, `information_gain`, `value`,
`valuable_coverage_proxy`, source paths, match method, source confidence,
missing reason codes, and fallback/guard provenance. Evidence gaps stay
missing; they are not converted to zero.

## Implementation

Implemented entry points:

- `scripts/run_policy_differentiating_counterfactual_coverage_rollouts.py`
- `scripts/run_policy_differentiating_counterfactual_coverage_rollouts.sh`
- `tests/test_policy_differentiating_counterfactual_coverage_rollouts.py`

Output root:

- `outputs/path_feedback_batch_policy_differentiating_counterfactual_coverage_rollouts_v1/`

Produced artifacts:

- `policy-differentiating-counterfactual-coverage-rollouts-summary.json`
- `counterfactual-coverage-rollouts.jsonl`
- `candidate-level-coverage-overlay.jsonl`
- `source-link-audit.json`
- `family-action-gap-report.json`
- `stage5a1-rerun-summary.json`
- `policy-differentiating-counterfactual-coverage-rollouts-report.md`
- `stage5a1-rerun/`
- `stage5a1-augmented-quasi-real-input/`

## Current Result

The current run passes as a data-routing stage:

- `status=passed`
- `reason_codes=[]`
- `next_required_change=rerun_coverage_driven_ppo_with_refined_reward_or_advantage`
- `input_missing_candidate_count=3456`
- `counterfactual_coverage_candidate_count=3456`
- `candidate_expected_coverage_nonzero_count=3690`
- `candidate_information_gain_nonzero_count=3690`
- `candidate_value_nonzero_count=1638`
- `safe_better_than_teacher_candidate_count=51`
- `safe_better_than_teacher_family_count=2`
- `missing_counterfactual_source_count=0`
- `fallback_gain_contamination_count=0`
- `controlled_regression_count=0`
- `stage5a1_rerun_status=passed`
- `stage5a_overlay_rerun_status=passed`
- `runs_new_ppo_update=false`
- `publishes_checkpoint=false`
- `replaces_default_policy=false`
- `performance_claimed=false`
- `formal_release_claimed=false`

All 3,456 previously missing non-teacher candidates are connected through
`counterfactual_candidate_expanded_cells_sidecar`. The generated
`candidate-level-coverage-overlay.jsonl` has 5,508 rows and can be consumed by
Stage 5A.

## Interpretation

This is not a model performance improvement. It proves that decision-time
counterfactual coverage evidence now exists and exposes safe better-than-teacher
candidate actions. The next stage should rerun coverage-driven PPO with refined
reward, advantage, or margin learning that can use these candidates.

## Acceptance

This stage passes when:

- `counterfactual_coverage_candidate_count>0`
- `missing_counterfactual_source_count=0`
- `safe_better_than_teacher_candidate_count>0`
- `fallback_gain_contamination_count=0`
- `controlled_regression_count=0`
- `stage5a1_rerun_status=passed`
- `stage5a_overlay_rerun_status=passed`
- all release/performance flags remain false

If evidence is still missing, it must fail with
`next_required_change=expand_counterfactual_coverage_source_generation` and list
missing family/action/source buckets.

## Verification

Validated with:

```bash
/home/kai/anaconda3/envs/lunar-explorer/bin/python -m pytest tests/test_policy_differentiating_counterfactual_coverage_rollouts.py tests/test_candidate_level_coverage_materialization.py tests/test_policy_coverage_opportunity_margin_audit.py -q
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python bash scripts/run_policy_differentiating_counterfactual_coverage_rollouts.sh
jq '{status,reason_codes,next_required_change,counterfactual_coverage_candidate_count,safe_better_than_teacher_candidate_count,performance_claimed,runs_new_ppo_update}' outputs/path_feedback_batch_policy_differentiating_counterfactual_coverage_rollouts_v1/policy-differentiating-counterfactual-coverage-rollouts-summary.json
git diff --check
```
