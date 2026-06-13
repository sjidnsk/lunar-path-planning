# Generated Sequential Gate Metric / Accounting Audit v1

## Summary

This stage audits generated sequential accounting after compatibility diagnosis
reported `gate_accounting_or_metric_mismatch`. The goal is to separate raw policy
probe rejection from controlled rollout cumulative regression. It does not run
PPO, relax generated sequential acceptance gates, publish checkpoints, or advance
readiness.

## Key Changes

- Add `configs/generated_sequential_gate_metric_accounting_audit_v1.json`.
- Add `scripts/run_generated_sequential_gate_metric_accounting_audit.py/.sh`.
- Add `scripts/run_generated_sequential_gate_metric_accounting_closure.sh`.
- Write outputs under
  `outputs/path_feedback_batch_generated_sequential_gate_metric_accounting_audit_v1/`.
- Split generated sequential summary counters into raw policy probe counters and
  controlled rollout counters.
- Keep `canary_rejected_policy_choice_count` as a blocker; do not remove generated
  sequential from the acceptance contract.
- Extend readiness with
  `--generated-sequential-gate-metric-accounting-audit-summary`; a passed audit
  refines the blocker to `generated_sequential_contract_alignment_required` but
  does not advance readiness.

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_generated_sequential_gate_metric_accounting_audit.py \
  tests/test_policy_gated_sequential_canary_rollout.py \
  tests/test_quasi_real_generated_sequential_contract_compatibility_diagnosis.py \
  tests/test_limited_quasi_real_ppo_update_smoke.py \
  tests/test_policy_training_readiness_review.py

PYTHON=$P bash scripts/run_generated_sequential_gate_metric_accounting_closure.sh

jq '{status, reason_codes, legacy_mismatch_count, raw_policy_path_cost_regression_count, raw_policy_risk_regression_count, controlled_path_cost_regression_count, controlled_risk_regression_count, diagnosis_verdict_after_origin_split, recommended_next_action}' \
  outputs/path_feedback_batch_generated_sequential_gate_metric_accounting_audit_v1/generated-sequential-gate-metric-accounting-audit-summary.json

git diff --check
```

## Acceptance

- Audit summary passes with no reason codes.
- The six legacy mismatches are explained as raw policy probe regressions.
- Corrected controlled cumulative path/risk regression counts are zero.
- Generated sequential still remains failed due rejected raw choices and
  insufficient multi-step accepted coverage.
- Readiness remains `needs_training_contract_refinement` and recommends
  `generated_sequential_contract_alignment_required`.

## Non-Goals

- No PPO update or iterative PPO mini-loop.
- No checkpoint publication or default-policy replacement.
- No network/action-space/default A* change.
- No distance/path-risk/source-selection gate relaxation.
- No Ackermann-feasible trajectory claim.
- No IRIS/GCS/path-planner diagnostic promotion to training release evidence.
