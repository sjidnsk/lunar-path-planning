# Quasi-Real Teacher-Equivalent Validation v1

## Background

Current quasi-real evidence shows the expanded LOLA root has 108 contexts across
4 ROI groups and 9 start cells. Bridge, domain-gap, context-id, action-mask,
fallback, safety, contract, path/risk, and source-selection gates are clean, but
strict top-k=3 opportunity diagnosis reports no safe alternative or safe-better
opportunity.

This means the quasi-real mainline should first validate teacher-equivalent
behavior. If the teacher/source-selected candidate is already the
non-regressive choice in the available top-k set, `source_aligned` is valid
policy behavior. Safe-better search remains useful, but it is a value branch,
not a prerequisite for learning the teacher.

## Scope

Add a shadow-only validation stage:

- `configs/quasi_real_teacher_equivalent_validation_v1.json`
- `scripts/run_quasi_real_teacher_equivalent_validation.py/.sh`
- `scripts/run_quasi_real_teacher_equivalent_validation_closure.sh`
- output root `outputs/path_feedback_batch_quasi_real_teacher_equivalent_validation_v1/`

The stage reuses quasi-real shadow policy scoring and gate logic. It does not
add an action space, does not execute policy takeover, and does not write PPO
transitions.

## Behavior Classes

- `source_aligned`: teacher-equivalent normal behavior.
- `policy_changed_gate_passed`: safe disagreement; accepted as diagnostic, not
  required for success.
- `policy_changed_gate_rejected`: unsafe disagreement; stage failure.
- `not_scored`: scoring or bridge failure; stage failure.

## Outputs

- `quasi-real-teacher-equivalent-decisions.jsonl`
- `quasi-real-teacher-equivalent-summary.json`
- `quasi-real-teacher-equivalent-disagreement-report.json`
- `quasi-real-teacher-equivalent-group-report.md`

The summary reports context count, policy decision count, teacher-aligned count,
teacher agreement rate, safe and unsafe disagreement counts, ROI-group teacher
agreement summary, all gate regression counters, non-goal flags, and git
provenance.

## Readiness

`scripts/run_policy_training_readiness_review.py` accepts
`--quasi-real-teacher-equivalent-validation-summary` and may advance to
`quasi_real_teacher_equivalent_validated`.

Acceptance gates:

- `status=passed`, `reason_codes=[]`
- `teacher_equivalent_context_count>=48`
- `policy_decision_count==teacher_equivalent_context_count`
- `roi_group_count>=4`
- `teacher_agreement_rate>=0.90`
- `context_id_missing_count=0`
- `unsafe_disagreement_count=0`
- `policy_changed_gate_rejected_count=0`
- invalid action mask, fallback/open-grid, safety, contract, path/risk, and
  source-selection regression counters all `0`
- `runs_ppo_update=false`
- `policy_takes_control=false`
- `publishes_checkpoint=false`
- `replaces_default_policy=false`
- `performance_claimed=false`
- current git provenance matches consumed evidence

If safe-better opportunity is still absent, that does not block this stage.
It only blocks the parallel value branch.

## Current Result

The validation tooling is implemented, but current evidence does not yet pass
the teacher-equivalent gate.

Default guarded candidate on the 108-context expanded LOLA root:

- `teacher_agreement_rate=0.8333`
- `teacher_aligned_count=90`
- `unsafe_disagreement_count=18`
- `path_cost_regression_count=18`
- `risk_regression_count=5`

Existing quasi-real shadow-alignment candidate probe:

- `teacher_agreement_rate=0.8796`
- `teacher_aligned_count=95`
- `unsafe_disagreement_count=13`
- `path_cost_regression_count=13`
- `risk_regression_count=4`

This confirms the next blocker is teacher distillation/alignment under
quasi-real path-cost and mixed-risk terrain, not safe-better opportunity
generation.

## Verification

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_quasi_real_teacher_equivalent_validation.py \
  tests/test_quasi_real_shadow_policy_behavior_audit.py \
  tests/test_quasi_real_safe_alternative_opportunity_diagnosis.py \
  tests/test_policy_training_readiness_review.py

PYTHON=$P bash scripts/run_quasi_real_teacher_equivalent_validation_closure.sh

PYTHON=$P bash scripts/run_policy_training_readiness_review.sh \
  --batch-root outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1 \
  --config configs/policy_training_readiness_review_v1.json \
  --quasi-real-safe-better-opportunity-expansion-summary \
    outputs/path_feedback_batch_quasi_real_safe_better_opportunity_expansion_v1/quasi-real-safe-better-opportunity-expansion-summary.json \
  --quasi-real-teacher-equivalent-validation-summary \
    outputs/path_feedback_batch_quasi_real_teacher_equivalent_validation_v1/quasi-real-teacher-equivalent-summary.json \
  --validate-only
```

## Non-Goals

This stage does not run PPO, write PPO transitions, execute policy takeover,
publish or replace a checkpoint, change the network/action space/default A*,
relax distance/path-risk/source-selection gates, claim Ackermann-feasible
trajectory, treat IRIS/GCS/path-planner diagnostics as training release
evidence, or claim policy performance improvement.
