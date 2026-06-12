# Quasi-Real Teacher Distillation Robustness v1

## Background

`Quasi-Real Teacher-Equivalent Validation v1` showed that safe-better
opportunity absence should not block the quasi-real mainline, but the default
guarded candidate still failed teacher-equivalent validation on the expanded
LOLA root:

- `teacher_equivalent_context_count=108`
- `teacher_agreement_rate=0.8333`
- `unsafe_disagreement_count=18`
- `path_cost_regression_count=18`
- `risk_regression_count=5`

The issue is not that quasi-real safe-better alternatives are missing. The issue
is that the policy can rank path-cost/risk-regressive alternatives above the
source-selected teacher. This stage calibrates teacher imitation without PPO,
policy takeover, checkpoint publication, or gate relaxation.

## Scope

Added artifacts:

- `configs/quasi_real_teacher_distillation_taxonomy_v1.json`
- `configs/quasi_real_teacher_distillation_dataset_v1.json`
- `configs/quasi_real_teacher_distillation_preference_v1.json`
- `configs/quasi_real_teacher_distillation_candidate_v1.json`
- `scripts/run_quasi_real_teacher_distillation_taxonomy.py/.sh`
- `scripts/run_quasi_real_teacher_distillation_dataset.py/.sh`
- `scripts/run_quasi_real_teacher_distillation_preference_mining.py/.sh`
- `scripts/run_quasi_real_teacher_distillation_candidate.py/.sh`
- `scripts/run_quasi_real_teacher_distillation_closure.sh`

Output roots:

- `outputs/path_feedback_batch_quasi_real_teacher_distillation_taxonomy_v1/`
- `outputs/path_feedback_batch_quasi_real_teacher_distillation_dataset_v1/`
- `outputs/path_feedback_batch_quasi_real_teacher_distillation_preference_v1/`
- `outputs/path_feedback_batch_quasi_real_teacher_distillation_candidate_v1/`
- `outputs/path_feedback_batch_quasi_real_teacher_distillation_validation_v1/`

## Method

The first implementation attempt trained only `teacher > current raw unsafe
choice`. That reduced some failures, but new regressive alternatives surfaced.
The final implementation expands the signal to `teacher > every
gate-regressive alternative` in the quasi-real top-k candidate set.

The taxonomy summary still reports the 18 raw unsafe disagreements uniquely, but
the training taxonomy JSONL contains all teacher-vs-regressive-alternative
pairs. Each pair is keyed by `taxonomy_record_id = scenario + source_context +
alternative_context`, so multiple alternatives in the same scenario cannot
overwrite each other.

Training samples remain pairwise preference only:

- preferred: source-selected teacher candidate
- alternative: gate-regressive candidate
- no hard positive
- no PPO transition
- no checkpoint publication
- no default policy replacement

## Current Evidence

The current closure passes:

- baseline unsafe disagreements classified: `18/18`
- `path_cost_only_regression_count=13`
- `path_risk_joint_regression_count=5`
- `bridge_or_feedback_gap_count=0`
- `action_mask_or_contract_gap_count=0`
- distillation candidate pairs: `216`
- train preference samples: `648`
- `hard_positive_added_count=0`
- `ppo_transition_added_count=0`
- `holdout_leakage_count=0`
- post-distillation `teacher_agreement_rate=1.0`
- post-distillation `unsafe_disagreement_count=0`
- invalid mask, fallback/open-grid, safety, contract, path/risk, and
  source-selection regression counters all `0`

## Readiness

`scripts/run_policy_training_readiness_review.py` accepts
`--quasi-real-teacher-distillation-summary` and may advance to
`quasi_real_teacher_distillation_robustness_evaluated` when:

- distillation summary `status=passed`
- `teacher_distillation_verdict=teacher_distillation_robustness_validated`
- taxonomy, preference, leakage, and post-distillation validation gates pass
- consumed evidence git provenance matches current HEAD
- non-goal flags remain false

This readiness status means quasi-real teacher-equivalent shadow calibration is
closed. It does not mean formal PPO rollout, quasi-real policy takeover,
checkpoint publication, default policy replacement, Ackermann-feasible
trajectory, or policy performance improvement.

## Verification

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_quasi_real_teacher_equivalent_validation.py \
  tests/test_quasi_real_teacher_distillation_robustness.py \
  tests/test_quasi_real_shadow_alignment.py \
  tests/test_policy_training_readiness_review.py

PYTHON=$P bash scripts/run_quasi_real_teacher_distillation_closure.sh

PYTHON=$P bash scripts/run_policy_training_readiness_review.sh \
  --batch-root outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1 \
  --config configs/policy_training_readiness_review_v1.json \
  --quasi-real-safe-better-opportunity-expansion-summary \
    outputs/path_feedback_batch_quasi_real_safe_better_opportunity_expansion_v1/quasi-real-safe-better-opportunity-expansion-summary.json \
  --quasi-real-teacher-equivalent-validation-summary \
    outputs/path_feedback_batch_quasi_real_teacher_distillation_validation_v1/quasi-real-teacher-equivalent-summary.json \
  --quasi-real-teacher-distillation-summary \
    outputs/path_feedback_batch_quasi_real_teacher_distillation_candidate_v1/quasi-real-teacher-distillation-summary.json \
  --validate-only
```

## Non-Goals

No PPO optimizer update, no PPO transition materialization, no quasi-real policy
takeover, no network/action space/default A* changes, no distance/path-risk or
source-selection gate relaxation, no checkpoint publication, no default policy
replacement, no performance claim, no Ackermann-feasible trajectory claim, and
no use of IRIS/GCS/path-planner diagnostics as training release evidence.
