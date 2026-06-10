# Current-HEAD Sequential/Guarded Evidence Re-closure v1

## Summary

Quasi-real shadow alignment has functionally passed, but formal readiness still
depends on current-HEAD upstream evidence. The blocker is not quasi-real
alignment. It is generated sequential/guarded evidence freshness and multi-step
coverage: stale roots can trigger `current_git_provenance_mismatch`, and older
sequential roots did not provide enough episodes with consecutive safe takeover.

This stage fixes the current-HEAD generated sequential boundary, refreshes
guarded/CUDA/quasi-real evidence, and only then allows readiness to advance to
`quasi_real_shadow_alignment_evaluated`.

## Changes

- Pin two additional sequential starts in
  `configs/policy_gated_sequential_multi_step_opportunity_rollout_v1.json`:
  `seq-mixed_stress_detour-b=[8,9]` and
  `seq-path_complexity_benefit-b=[8,9]`.
- Extend sequential opportunity diagnosis with `family_opportunity_summary`,
  including safe-better, policy-used, policy-missed, rejected, missing,
  accepted-takeover, multi-step accepted, and family `gap_reason` counts.
- Keep the existing quasi-real alignment logic unchanged. The baseline mixed-risk
  failure remains a failure seed and hard-negative preference source, not a hard
  positive or PPO transition.
- Refresh evidence from a clean HEAD: sequential multi-step closure, guarded
  pilot, CUDA smoke, quasi-real domain gap, shadow audit, alignment closure, and
  final readiness validate-only.

## Acceptance

- Sequential diagnosis: 36 episodes, 108 steps, safe-better step count at least
  24, multi-step opportunity episode count at least 12, six families covered,
  and zero opportunity exclusions.
- Sequential rollout: 36 completed episodes, at least 24 policy takeover steps,
  at least 24 accepted takeover steps, at least 12 accepted-better steps, six
  accepted takeover families, at least 12 multi-step accepted episodes, six
  families with multi-step accepted episodes, zero rejected policy choice, zero
  state continuity violation, zero episode fallback, and zero safety/contract/
  path/risk/source-selection regression.
- Guarded/CUDA/quasi-real: guarded pilot passes; CUDA smoke passes and checkpoint
  remains CPU-loadable; quasi-real domain-gap verdict remains
  `acceptable_for_next_pilot`; baseline shadow audit reproduces the mixed-risk
  failure; alignment closure passes without holdout or original ROI regression.
- Final readiness: no `current_git_provenance_mismatch`,
  `training_readiness_status=quasi_real_shadow_alignment_evaluated`, and
  `training_blockers=[]`.

## Verification

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_sequential_multi_step_opportunity_generation.py \
  tests/test_policy_gated_sequential_canary_rollout.py \
  tests/test_policy_training_readiness_review.py \
  tests/test_quasi_real_shadow_failure_taxonomy.py \
  tests/test_quasi_real_shadow_alignment.py

cd dev-platform-constraints && \
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q tests/test_npz_validation_maps.py
cd ..

PYTHON=$P bash scripts/run_sequential_multi_step_opportunity_closure.sh
PYTHON=$P bash scripts/run_guarded_ppo_rollout_pilot_closure.sh
PYTHON=$P bash scripts/run_policy_training_cuda_device_support_smoke.sh \
  --source-root outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1 \
  --base-candidate-root outputs/path_feedback_batch_guarded_ppo_rollout_pilot_v1/update \
  --collector-root outputs/path_feedback_batch_guarded_ppo_rollout_pilot_v1/final/collector \
  --output-root outputs/path_feedback_batch_policy_training_cuda_device_support_v1 \
  --config configs/policy_training_cuda_device_support_v1.json
PYTHON=$P bash scripts/run_quasi_real_map_domain_gap_closure.sh
PYTHON=$P bash scripts/run_quasi_real_shadow_policy_behavior_closure.sh || true
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

- No formal PPO rollout.
- No quasi-real policy takeover.
- No checkpoint publication or default policy replacement.
- No network/action-space/default-A* change.
- No distance/path-risk/source-selection contract relaxation.
- No Ackermann-feasible trajectory claim.
- No IRIS/GCS/path-planner diagnostic treated as release evidence.
- No policy performance claim.
