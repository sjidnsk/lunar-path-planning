# Quasi-Real Guarded Teacher-Following Pilot v1

## Background

`Quasi-Real Teacher Distillation Robustness v1` closes the quasi-real
teacher-equivalent alignment gap:

- `teacher_equivalent_context_count=108`
- post-distillation `teacher_agreement_rate=1.0`
- `unsafe_disagreement_count=0`
- `policy_changed_gate_rejected_count=0`
- path/risk/source-selection regression counters are all `0`

The quasi-real top-k evidence still has no safe-better alternative. That blocks
the value-improvement branch, not the teacher-following mainline. This stage
therefore validates guarded teacher-following behavior instead of requiring a
policy-changed safe-better choice.

## Scope

Added artifacts:

- `configs/quasi_real_guarded_teacher_following_pilot_v1.json`
- `scripts/run_quasi_real_guarded_teacher_following_pilot.py/.sh`
- `scripts/run_quasi_real_guarded_teacher_following_closure.sh`

Output root:

- `outputs/path_feedback_batch_quasi_real_guarded_teacher_following_pilot_v1/`

Output files:

- `quasi-real-guarded-teacher-following-decisions.jsonl`
- `quasi-real-guarded-teacher-following-pilot-summary.json`
- `quasi-real-guarded-teacher-following-rejection-report.json`
- `quasi-real-guarded-teacher-following-group-report.md`

## Decision Semantics

The runner reuses quasi-real shadow scoring and gate logic, then maps decisions
into guarded teacher-following records:

- `source_aligned` -> `controlled_choice_source=policy_teacher_aligned`
- `policy_changed_gate_passed` -> `controlled_choice_source=policy_safe_disagreement`
- `policy_changed_gate_rejected` -> `controlled_choice_source=teacher_fallback`

`source_aligned` is a valid teacher-following step. `policy_changed_gate_passed`
is allowed as a safe disagreement. `policy_changed_gate_rejected` fails the
stage and must be reported with gate reason codes.

## Readiness

`scripts/run_policy_training_readiness_review.py` accepts
`--quasi-real-guarded-teacher-following-pilot-summary` and may advance to
`quasi_real_guarded_teacher_following_pilot_evaluated` when:

- summary `status=passed`
- `teacher_following_pilot_verdict=teacher_following_pilot_validated`
- `quasi_real_context_count>=108`
- `policy_decision_count==quasi_real_context_count`
- `roi_group_count>=4`
- `teacher_agreement_rate>=0.90`
- `teacher_following_step_count>=90`
- `unsafe_disagreement_count=0`
- invalid mask, fallback/open-grid, safety, contract, path/risk, and
  source-selection regression counters all `0`
- consumed evidence git provenance matches current HEAD

## Verification

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_quasi_real_guarded_teacher_following_pilot.py \
  tests/test_quasi_real_teacher_equivalent_validation.py \
  tests/test_quasi_real_teacher_distillation_robustness.py \
  tests/test_policy_training_readiness_review.py

PYTHON=$P bash scripts/run_quasi_real_guarded_teacher_following_closure.sh

PYTHON=$P bash scripts/run_policy_training_readiness_review.sh \
  --batch-root outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1 \
  --config configs/policy_training_readiness_review_v1.json \
  --quasi-real-safe-better-opportunity-expansion-summary \
    outputs/path_feedback_batch_quasi_real_safe_better_opportunity_expansion_v1/quasi-real-safe-better-opportunity-expansion-summary.json \
  --quasi-real-teacher-distillation-summary \
    outputs/path_feedback_batch_quasi_real_teacher_distillation_candidate_v1/quasi-real-teacher-distillation-summary.json \
  --quasi-real-guarded-teacher-following-pilot-summary \
    outputs/path_feedback_batch_quasi_real_guarded_teacher_following_pilot_v1/quasi-real-guarded-teacher-following-pilot-summary.json \
  --validate-only
```

## Non-Goals

No PPO optimizer update, no PPO transition materialization, no checkpoint
publication, no default policy replacement, no network/action space/default A*
changes, no distance/path-risk/source-selection gate relaxation, no
Ackermann-feasible trajectory claim, no IRIS/GCS/path-planner diagnostic release
claim, and no policy performance claim.
