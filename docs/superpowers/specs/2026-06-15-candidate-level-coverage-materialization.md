# Candidate-Level Exploration Coverage Materialization v1

## Purpose

This stage follows `Policy Coverage Opportunity / Margin Audit v1`. Stage 5A
proved that the post-update policy stayed teacher-equivalent and that candidate
decision features had no coverage signal. This stage materializes a
candidate-level coverage overlay from existing offline artifacts only. It does
not run PPO, publish checkpoints, replace the default policy, connect a real
executor, relax guards, or claim performance.

## Inputs

- `outputs/path_feedback_batch_policy_coverage_opportunity_margin_audit_v1/`
- `outputs/path_feedback_batch_coverage_driven_ppo_improvement_run_v1/`
- `outputs/path_feedback_batch_coverage_aware_reward_refinement_v1/`
- `outputs/path_feedback_batch_exploration_coverage_signal_audit_v1/`
- `outputs/path_feedback_batch_exploration_coverage_performance_evaluation_v1/`
- `outputs/path_feedback_batch_quasi_real_safe_better_opportunity_expansion_v1/`

## Required Audit

For each Stage 5A action candidate, materialize:

- `expected_coverage_rate_delta`
- `expected_new_coverage_area`
- `information_gain`
- `value`
- `valuable_coverage_proxy`
- `path_cost`, `risk`, and `energy_cost`
- `source_path`, `match_method`, and `source_confidence`
- missing-source reason codes when coverage source is absent

Executed teacher/selected actions may use actual path-feedback coverage as
`executed_action_actual_path_feedback`. Non-teacher candidates require real
counterfactual candidate coverage; missing fields must remain missing, not
default to zero.

## Implementation

Implemented entry points:

- `scripts/run_candidate_level_coverage_materialization.py`
- `scripts/run_candidate_level_coverage_materialization.sh`
- `tests/test_candidate_level_coverage_materialization.py`

Stage 5A was extended with:

- `--candidate-coverage-overlay`
- `candidate_coverage_overlay_connected_count`
- action-row overlay source fields

Output root:

- `outputs/path_feedback_batch_candidate_level_coverage_materialization_v1/`

Produced artifacts:

- `candidate-level-coverage-materialization-summary.json`
- `candidate-level-coverage-overlay.jsonl`
- `counterfactual-opportunity-audit.jsonl`
- `source-link-audit.json`
- `candidate-level-coverage-rejection-report.json`
- `candidate-level-coverage-materialization-report.md`
- `overlay-fed-stage5a-rerun-summary.json`

## Current Result

The current materialization run fails as expected because counterfactual
candidate coverage is not present:

- `status=failed`
- `next_required_change=generate_policy_differentiating_coverage_rollouts`
- `reason_codes=["candidate_coverage_source_missing", "counterfactual_candidate_coverage_source_missing", "no_safe_better_than_teacher_candidate"]`
- `context_count=2052`
- `action_candidate_row_count=5508`
- `overlay_connected_candidate_count=2052`
- `missing_candidate_coverage_source_count=3456`
- `candidate_expected_coverage_nonzero_count=2052`
- `candidate_information_gain_nonzero_count=2052`
- `candidate_value_nonzero_count=0`
- `counterfactual_coverage_candidate_count=0`
- `safe_better_than_teacher_candidate_count=0`
- `fallback_gain_contamination_count=0`
- `controlled_regression_count=0`

The overlay-fed Stage 5A rerun passed as a diagnostic and consumed the overlay,
but still routes to `collect_more_policy_differentiating_coverage` because the
nonzero coverage belongs to executed teacher-equivalent actions, not safe
non-teacher alternatives.

Missing rollout requirements are grouped by family:

- `smooth_high_confidence`: 864 candidate rows
- `rim_or_steep_slope`: 873 candidate rows
- `low_observation_count`: 837 candidate rows
- `mixed_risk`: 882 candidate rows

## Follow-Up Stage

`Policy-Differentiating Counterfactual Coverage Rollouts v1` now consumes this
failure as its input. It reads the missing Stage 5A.1 overlay/source-link rows
and computes non-teacher counterfactual coverage from existing quasi-real
path-feedback candidate `diagnostics.expanded_cells` plus path-planner
sidecars.

The follow-up run is rooted at
`outputs/path_feedback_batch_policy_differentiating_counterfactual_coverage_rollouts_v1/`
and passes as a data-routing stage:

- `status=passed`
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

This follow-up still does not claim performance. It only proves that Stage 5A.1
can now be rerun with real non-teacher counterfactual coverage and can route the
pipeline back to refined coverage-driven PPO.

## Acceptance

If future artifacts provide real non-teacher counterfactual coverage and safe
better alternatives, this stage should output:

- `status=passed`
- `next_required_change=rerun_coverage_driven_ppo_with_refined_reward_or_advantage`
- `counterfactual_coverage_candidate_count>0`
- `safe_better_than_teacher_candidate_count>0`

If counterfactual source evidence is missing, this stage must fail with
`next_required_change=generate_policy_differentiating_coverage_rollouts`.

It must keep `runs_new_ppo_update=false`, `publishes_checkpoint=false`,
`replaces_default_policy=false`, `performance_claimed=false`, and
`formal_release_claimed=false`.

## Verification

Validated with:

```bash
/home/kai/anaconda3/envs/lunar-explorer/bin/python -m pytest tests/test_candidate_level_coverage_materialization.py tests/test_policy_coverage_opportunity_margin_audit.py -q
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python bash scripts/run_candidate_level_coverage_materialization.sh
jq '{status,reason_codes,next_required_change,candidate_expected_coverage_nonzero_count,safe_better_than_teacher_candidate_count,performance_claimed,runs_new_ppo_update}' outputs/path_feedback_batch_candidate_level_coverage_materialization_v1/candidate-level-coverage-materialization-summary.json
git diff --check
```
