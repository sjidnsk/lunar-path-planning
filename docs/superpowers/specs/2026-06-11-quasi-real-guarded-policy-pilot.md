# Quasi-Real Guarded Policy Pilot v1

## Summary

The project has reached `quasi_real_shadow_alignment_evaluated`: LOLA quasi-real ROI data is bridged into path-feedback, baseline mixed-risk shadow failure is classified, and anti-overfit alignment removes holdout/original ROI path-risk regressions without adding hard positives or PPO transitions.

This stage adds a guarded quasi-real pilot. The policy may propose a different action on each quasi-real context, but it is accepted only if the existing gate stack passes. Rejected changes fall back to source-selected and are reported. This is not a PPO collector, not a checkpoint release, and not a performance claim.

## Artifacts

- `configs/quasi_real_guarded_policy_pilot_v1.json`
- `scripts/run_quasi_real_guarded_policy_pilot.py`
- `scripts/run_quasi_real_guarded_policy_pilot.sh`
- `scripts/run_quasi_real_guarded_policy_pilot_closure.sh`
- `outputs/path_feedback_batch_quasi_real_guarded_policy_pilot_v1/`

Outputs:

- `quasi-real-guarded-policy-decisions.jsonl`
- `quasi-real-guarded-policy-pilot-summary.json`
- `quasi-real-guarded-policy-rejection-report.json`
- `quasi-real-guarded-policy-group-report.md`

## Contract

The pilot reuses quasi-real shadow policy scoring and gate logic. Each changed policy choice must satisfy:

- action mask valid
- candidate present
- reachable
- no replan
- no fallback/open-grid
- contract safe
- no path-cost regression
- no risk regression
- no source-selection regression

Decision records use `controlled_choice_source=policy|source|source_fallback`. Source fallback records are diagnostic only. No PPO transition is written.

## Readiness

`run_policy_training_readiness_review.py` accepts `--quasi-real-guarded-policy-pilot-summary` and may advance to `quasi_real_guarded_policy_pilot_evaluated`.

Pass conditions:

- `status=passed`
- `guarded_pilot_verdict=acceptable_for_quasi_real_collector_dry_run`
- `quasi_real_context_count>=12`
- `policy_decision_count==quasi_real_context_count`
- `roi_group_count>=4`
- `context_id_missing_count=0`
- `policy_changed_gate_passed_count>0`
- `policy_changed_gate_rejected_count=0`
- invalid action mask, fallback/open-grid, safety, contract, path/risk, and source-selection regression counters are all 0
- current git provenance matches consumed evidence

Failure routing:

- no accepted changed policy choice: `quasi_real_guarded_policy_over_conservative`
- rejected changed choice: `quasi_real_guarded_gate_regression`
- action mask or contract break: `quasi_real_guarded_action_mask_contract_gap`
- missing/scoring bridge issue: `quasi_real_guarded_policy_scoring_failed`

## Verification

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_quasi_real_guarded_policy_pilot.py \
  tests/test_quasi_real_shadow_policy_behavior_audit.py \
  tests/test_quasi_real_shadow_alignment.py \
  tests/test_policy_training_readiness_review.py

PYTHON=$P bash scripts/run_quasi_real_guarded_policy_pilot_closure.sh
```

## Non-Goals

- no PPO optimizer update
- no PPO transition materialization
- no checkpoint publication
- no default policy replacement
- no network/action-space/default-A* change
- no distance/path-risk/source-selection gate relaxation
- no Ackermann-feasible trajectory claim
- no IRIS/GCS/path-planner diagnostic promoted to training release evidence
- no policy performance claim
