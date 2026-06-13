# Quasi-Real / Generated Sequential Contract Compatibility Diagnosis v1

## Summary

This stage diagnoses the blocker left by `Limited Quasi-Real PPO Update Smoke v1`.
The PPO update itself is numerically clean and quasi-real post-update gates pass,
but generated sequential post-update canary fails. The goal is not to run another
PPO update or relax a gate; it is to determine whether the failure is update
induced, pre-existing contract mismatch, stale/unreplayable base evidence, or a
gate accounting/metric mismatch.

## Artifacts

- `configs/quasi_real_generated_sequential_contract_compatibility_diagnosis_v1.json`
- `scripts/run_quasi_real_generated_sequential_contract_compatibility_diagnosis.py`
- `scripts/run_quasi_real_generated_sequential_contract_compatibility_diagnosis.sh`
- `scripts/run_quasi_real_generated_sequential_contract_compatibility_closure.sh`
- `outputs/path_feedback_batch_quasi_real_generated_sequential_contract_compatibility_diagnosis_v1/`

## Inputs

- Failed update smoke root:
  `outputs/path_feedback_batch_limited_quasi_real_ppo_update_smoke_v1/`
- Base quasi-real teacher-distillation candidate:
  `outputs/path_feedback_batch_quasi_real_teacher_distillation_candidate_v1/`
- Generated source root:
  `outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1/`

The diagnosis consumes the failed update wrapper summary, post-update generated
sequential summary/steps/rejection report, post-update quasi-real
teacher-following summary, and post-update quasi-real collector summary.

## Logic

1. Clone the base quasi-real candidate into `diagnostic-base-candidate/`.
2. Preserve checkpoint weights, refresh current git provenance, and mark the
   clone diagnostic, experimental, non-publishing, and non-default.
3. Run generated sequential canary on the diagnostic base candidate.
4. Reuse or replay the updated generated sequential evidence.
5. Compare failed episode/step/family records, rejection reasons, action indices,
   logits, target cells, and path/risk deltas.
6. Emit one verdict:
   `ppo_update_induced_generated_regression`,
   `pre_existing_generated_sequential_contract_mismatch`,
   `stale_or_unreplayable_base_candidate`,
   `gate_accounting_or_metric_mismatch`, or `diagnosis_inconclusive`.

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_quasi_real_generated_sequential_contract_compatibility_diagnosis.py \
  tests/test_limited_quasi_real_ppo_update_smoke.py \
  tests/test_limited_ppo_update_smoke.py \
  tests/test_quasi_real_ppo_collector_dry_run.py \
  tests/test_policy_training_readiness_review.py

PYTHON=$P bash scripts/run_quasi_real_generated_sequential_contract_compatibility_closure.sh

jq '{status, reason_codes, diagnosis_verdict, failed_step_count, base_generated_sequential_status, updated_generated_sequential_status, recommended_next_action}' \
  outputs/path_feedback_batch_quasi_real_generated_sequential_contract_compatibility_diagnosis_v1/quasi-real-generated-sequential-contract-compatibility-summary.json

git diff --check
```

## Non-Goals

- No new PPO update.
- No iterative PPO mini-loop.
- No checkpoint publication or default-policy replacement.
- No network/action-space/default A* change.
- No distance/path-risk/source-selection gate relaxation.
- No Ackermann-feasible trajectory claim.
- No IRIS/GCS/path-planner diagnostic promotion to training release evidence.
- No removal of generated sequential from the acceptance contract without a
  separate follow-up plan.
