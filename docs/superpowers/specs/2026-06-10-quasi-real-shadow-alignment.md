# Quasi-Real Shadow Failure Taxonomy and Anti-Overfit Alignment v1

## Summary

Baseline quasi-real shadow audit found one failed context:
`lola_qreal_mixed_risk_test_011`. Source action was `1`, raw policy selected
action `2`, and the raw alternative triggered both `path_cost_regression` and
`risk_regression`. This stage treats that context as a failure seed, not as a
single sample to memorize.

The goal is to classify the failure, derive disjoint train/val/holdout
quasi-real variants, mine rule-shaped hard-negative preferences, run a small
pairwise calibration from the guarded experimental checkpoint, and verify that
holdout plus the original ROI audit no longer regress.

## Artifacts

- `configs/quasi_real_shadow_failure_taxonomy_v1.json`
- `configs/quasi_real_shadow_alignment_splits_v1.json`
- `configs/quasi_real_shadow_alignment_preference_v1.json`
- `configs/quasi_real_shadow_alignment_candidate_v1.json`
- `scripts/run_quasi_real_shadow_failure_taxonomy.py/.sh`
- `scripts/run_quasi_real_shadow_alignment_dataset.py/.sh`
- `scripts/run_quasi_real_shadow_alignment_preference_mining.py/.sh`
- `scripts/run_quasi_real_shadow_alignment_candidate.py/.sh`
- `scripts/run_quasi_real_shadow_alignment_closure.sh`
- `outputs/path_feedback_batch_quasi_real_shadow_failure_taxonomy_v1/`
- `outputs/path_feedback_batch_quasi_real_shadow_alignment_dataset_v1/`
- `outputs/path_feedback_batch_quasi_real_shadow_alignment_preference_v1/`
- `outputs/path_feedback_batch_quasi_real_shadow_alignment_candidate_v1/`

## Evidence

Functional closure currently reports:

- taxonomy `status=passed`.
- `failure_count=1`.
- `path_risk_joint_regression_count=1`.
- bridge/action-mask/contract gap counts are `0`.
- train/val/holdout split counts are `3/3/3`.
- `context_id_overlap_count=0`.
- `scenario_id_overlap_count=0`.
- `slice_id_overlap_count=0`.
- `quasi_real_hard_negative_preference_count=3`.
- `hard_positive_added_count=0`.
- `ppo_transition_added_count=0`.
- post-calibration holdout path/risk/source-selection regression counts are `0`.
- original 12-ROI shadow regression count is `0`.

The calibration starts from the guarded experimental checkpoint and uses only
pairwise hard-negative preference. It does not create PPO rollout transitions
and does not add action-label positives.

## Readiness

Readiness accepts `--quasi-real-shadow-alignment-summary` and may advance only
to `quasi_real_shadow_alignment_evaluated` when:

- alignment summary `status=passed`;
- `alignment_verdict=acceptable_for_quasi_real_shadow_audit`;
- no split leakage;
- holdout gate-rejected/path/risk/source-selection regression counts are `0`;
- original ROI regression count is `0`;
- no over-conservative policy is reported;
- no checkpoint publication, default replacement, or performance claim exists;
- all consumed evidence provenance matches current HEAD.

Because tracked code/docs changed during implementation, a clean-HEAD refresh is
required before formal readiness can be accepted.

## Verification

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_quasi_real_shadow_policy_behavior_audit.py \
  tests/test_quasi_real_map_domain_gap_evaluation.py \
  tests/test_policy_training_readiness_review.py \
  tests/test_quasi_real_shadow_failure_taxonomy.py \
  tests/test_quasi_real_shadow_alignment.py

PYTHON=$P bash scripts/run_quasi_real_shadow_alignment_closure.sh

PYTHON=$P bash scripts/run_policy_training_readiness_review.sh \
  --batch-root outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1 \
  --config configs/policy_training_readiness_review_v1.json \
  --policy-training-cuda-device-support-summary outputs/path_feedback_batch_policy_training_cuda_device_support_v1/policy-training-cuda-device-support-summary.json \
  --quasi-real-map-domain-gap-summary outputs/path_feedback_batch_quasi_real_map_domain_gap_v1/quasi-real-map-domain-gap-summary.json \
  --quasi-real-shadow-policy-behavior-summary outputs/path_feedback_batch_quasi_real_shadow_policy_behavior_v1/quasi-real-shadow-policy-behavior-summary.json \
  --quasi-real-shadow-alignment-summary outputs/path_feedback_batch_quasi_real_shadow_alignment_candidate_v1/quasi-real-shadow-alignment-summary.json \
  --validate-only
```

## Non-Goals

- No quasi-real policy takeover.
- No formal PPO rollout.
- No checkpoint publication or default-policy replacement.
- No network/action-space/default-A* change.
- No distance/path-risk/source-selection contract relaxation.
- No Ackermann-feasible trajectory claim.
- No IRIS/GCS/path-planner diagnostic treated as training release evidence.
- No policy performance claim.
