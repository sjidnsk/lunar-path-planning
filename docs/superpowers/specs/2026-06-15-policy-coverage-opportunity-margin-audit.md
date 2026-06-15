# Policy Coverage Opportunity / Margin Audit v1

## Purpose

This is the Stage 5A diagnostic after `Coverage-Driven PPO Improvement Run v1`
failed with `no_coverage_return_improvement`.

The goal is to determine whether the failed PPO improvement run needs reward and
advantage refinement, or whether the current data lacks enough safe
policy-differentiating coverage opportunities. This stage does not run PPO,
publish checkpoints, replace the default policy, connect a real executor, relax
guards, or claim performance.

## Evidence Basis

Current Stage 5 evidence:

- `status=failed`
- `coverage_driven_ppo_improvement_status=failed`
- `reason_codes=["no_coverage_return_improvement"]`
- `coverage_return_improvement=0.0`
- `cumulative_coverage_rate_delta_improvement=0.0`
- `valuable_area_covered_improvement=45.824562342498`
- `accepted_policy_activation_rate=1.0`
- `fallback_rate=0.0`
- `teacher_agreement_rate=1.0`
- `controlled_regression_count=0`

Replay shows that post-update raw policy action, controlled action, and teacher
action are identical on all 2,052 audited rows. The problem is not guard
rejection; the update did not change the final action choice.

## Inputs

- `outputs/path_feedback_batch_coverage_driven_ppo_improvement_run_v1/`
- `outputs/path_feedback_batch_coverage_aware_reward_refinement_v1/`
- `outputs/path_feedback_batch_exploration_coverage_signal_audit_v1/`
- `outputs/path_feedback_batch_exploration_coverage_performance_evaluation_v1/`
- Frozen selected candidate and post-update experimental checkpoint metadata.

## Required Audit

The audit must expand every context into action-level candidate rows and compare:

- teacher action
- pre-improvement selected PPO action
- post-update raw policy action
- post-update controlled action
- best safe non-teacher alternative, if present
- coverage gain and cumulative coverage contribution
- valuable coverage
- path cost, risk, and energy
- action mask reachability
- guard rejection or controlled regression reasons
- policy logit/probability margin before and after update
- candidate-level coverage features visible before action selection

Required summary fields:

- `safe_better_than_teacher_count`
- `safe_better_than_teacher_family_count`
- `policy_argmax_changed_count`
- `post_update_teacher_equal_raw_count`
- `post_update_teacher_equal_controlled_count`
- `teacher_margin_sample_count`
- `missing_teacher_signal_count`
- `candidate_expected_coverage_nonzero_count`
- `candidate_information_gain_nonzero_count`
- `candidate_value_nonzero_count`
- `fallback_gain_contamination_count`
- `controlled_regression_count`

## Decision Rules

If enough safe better alternatives exist, the next stage is reward/objective
refinement. Candidate changes include:

- reduce or condition `teacher_skill_retention_bonus`
- strengthen coverage-return advantage
- add pairwise or margin loss for safe better alternatives
- normalize reward components so coverage gain can flip action ranking
- preserve hard guard and controlled regression penalties

If safe better alternatives are missing or too sparse, the next stage is data
collection/materialization. Candidate changes include:

- collect policy-differentiating coverage rollouts
- materialize counterfactual candidate coverage fields
- populate decision-time `expected_coverage_rate_delta`, `information_gain`,
  and `value`
- add explicit teacher-margin and safe-alternative labels

## Acceptance

Passing this diagnostic does not mean model performance improved. It passes when
it produces a clear route:

- `next_required_change=refine_coverage_reward_or_advantage` when safe better
  alternatives are sufficient.
- `next_required_change=collect_more_policy_differentiating_coverage` when
  opportunity evidence is insufficient.

It must keep `publishes_checkpoint=false`, `replaces_default_policy=false`,
`runs_new_ppo_update=false`, `performance_claimed=false`, and
`formal_release_claimed=false`.

## Implementation

Implemented entry points:

- `scripts/run_policy_coverage_opportunity_margin_audit.py`
- `scripts/run_policy_coverage_opportunity_margin_audit.sh`
- `tests/test_policy_coverage_opportunity_margin_audit.py`

The script also supports an optional candidate-level coverage overlay:

```bash
--candidate-coverage-overlay outputs/path_feedback_batch_candidate_level_coverage_materialization_v1/candidate-level-coverage-overlay.jsonl
```

When provided, Stage 5A indexes overlay rows by decision context
(`context_id`, `episode_id`, `step_index`, `action_index`) before broader
scenario/cell fallbacks, merges real candidate-level coverage fields into the
decision-time feature map, and reports
`candidate_coverage_overlay_connected_count`. Missing overlay rows or rows with
`coverage_source_available=false` are not treated as real zero-coverage
evidence.

`Policy-Differentiating Counterfactual Coverage Rollouts v1` now produces a
fully connected overlay at:

```bash
--candidate-coverage-overlay outputs/path_feedback_batch_policy_differentiating_counterfactual_coverage_rollouts_v1/candidate-level-coverage-overlay.jsonl
```

The overlay-fed rerun through Stage 5A.1 and Stage 5A reports
`safe_better_than_teacher_candidate_count=51`,
`safe_better_than_teacher_family_count=2`,
`fallback_gain_contamination_count=0`, and `controlled_regression_count=0`.
That is a route back to refined coverage-driven PPO, not a performance claim.

Output root:

- `outputs/path_feedback_batch_policy_coverage_opportunity_margin_audit_v1/`

Produced artifacts:

- `policy-coverage-opportunity-margin-audit-summary.json`
- `policy-coverage-action-level-audit.jsonl`
- `policy-margin-audit.json`
- `candidate-feature-audit.json`
- `policy-coverage-opportunity-rejection-report.json`
- `policy-coverage-opportunity-margin-audit-report.md`

## Current Result

The current run passes as a diagnostic and routes to data/materialization:

- `status=passed`
- `next_required_change=collect_more_policy_differentiating_coverage`
- `reason_codes=["candidate_coverage_features_missing", "no_safe_better_than_teacher_alternatives", "post_update_policy_teacher_equivalent"]`
- `context_count=2052`
- `action_candidate_row_count=5508`
- `safe_better_than_teacher_count=0`
- `safe_better_than_teacher_family_count=0`
- `policy_argmax_changed_count=0`
- `post_update_teacher_equal_raw_count=2052`
- `post_update_teacher_equal_controlled_count=2052`
- `teacher_margin_sample_count=2052`
- `missing_teacher_signal_count=0`
- `candidate_expected_coverage_nonzero_count=0`
- `candidate_information_gain_nonzero_count=0`
- `candidate_value_nonzero_count=0`

The latest overlay-fed rerun from Stage 5A.2 changes this route: candidate-level
counterfactual coverage is now available for the previously missing non-teacher
actions, so the current follow-up route is
`rerun_coverage_driven_ppo_with_refined_reward_or_advantage`.
- `fallback_gain_contamination_count=0`
- `controlled_regression_count=0`

Interpretation: the guard did not block a better action, and the PPO update did
not change policy argmax. The audited candidate observations still contain no
decision-time coverage direction, because `expected_coverage_rate_delta`,
`information_gain`, and `value` are zero for every candidate. The next stage
should collect or materialize policy-differentiating coverage evidence before
another PPO improvement run.

The follow-on materialization stage is documented in
`docs/superpowers/specs/2026-06-15-candidate-level-coverage-materialization.md`.

## Verification

Validated with:

```bash
/home/kai/anaconda3/envs/lunar-explorer/bin/python -m pytest tests/test_candidate_level_coverage_materialization.py tests/test_policy_coverage_opportunity_margin_audit.py -q
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python bash scripts/run_policy_coverage_opportunity_margin_audit.sh
jq '{status,reason_codes,next_required_change,performance_claimed,runs_new_ppo_update}' outputs/path_feedback_batch_policy_coverage_opportunity_margin_audit_v1/policy-coverage-opportunity-margin-audit-summary.json
git diff --check
```
