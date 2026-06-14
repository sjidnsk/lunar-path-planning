# Guarded Experimental Policy Staged Release Trial v1

## Summary

This stage runs a local/offline guarded staged release trial after
`Guarded Experimental Policy Staged Release Preflight v1`. It allows a small
number of experimental policy control activations only when the source shadow
row is gate-clean. The default policy remains authoritative, and this stage does
not publish or replace any checkpoint.

## Inputs

- Preflight summary:
  `outputs/path_feedback_batch_guarded_experimental_policy_staged_release_preflight_v1/guarded-experimental-policy-staged-release-preflight-summary.json`
- Shadow step comparison:
  `outputs/path_feedback_batch_guarded_experimental_policy_shadow_release_trial_v1/shadow-release-step-comparison.jsonl`

## Outputs

- `outputs/path_feedback_batch_guarded_experimental_policy_staged_release_trial_v1/guarded-experimental-policy-staged-release-trial-summary.json`
- `staged-release-activation-ledger.jsonl`
- `staged-release-controlled-regression-audit.json`
- `staged-release-fallback-rejection-report.json`
- `staged-release-kill-switch-drill.json`
- `staged-release-rollback-drill.json`
- `staged-release-telemetry-drill.json`
- `staged-release-trial-readiness-validate-only.json`
- `staged-release-trial-report.md`

## Contract

Experimental control may activate only when:

- `controlled_choice_source="policy"`
- gate reason codes are empty
- controlled regression reason codes are empty
- observation/log_prob/value/reward are present and finite
- action mask is valid

Fallback, rejected, missing, invalid-mask, non-finite, or controlled-regression
rows remain diagnostic-only and cannot count as control activations.

## Acceptance

- summary `status=passed`, `reason_codes=[]`
- `staged_release_trial_verdict=eligible_for_guarded_staged_release_canary`
- `staged_release_enabled=true`, `default_policy_authoritative=true`
- `1 <= experimental_control_activation_count <= 64`
- diagnostic/fallback/rejected control activation count is 0
- controlled safety/contract/path-risk/source-selection regression counts are 0
- activation-level missing/non-finite/invalid-mask counts are 0
- kill-switch, rollback, and telemetry drills pass
- default policy unchanged
- no checkpoint publication, default replacement, performance claim, or formal-training-ready claim
- readiness accepts `--guarded-experimental-policy-staged-release-trial-summary`
  and returns `guarded_experimental_policy_staged_release_trial_evaluated`

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_guarded_experimental_policy_staged_release_trial.py \
  tests/test_guarded_experimental_policy_staged_release_preflight.py \
  tests/test_policy_training_readiness_review.py

PYTHON=$P bash scripts/run_guarded_experimental_policy_staged_release_trial_closure.sh

PYTHON=$P bash scripts/run_policy_training_readiness_review.sh \
  --batch-root outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1 \
  --config configs/policy_training_readiness_review_v1.json \
  --guarded-experimental-policy-staged-release-trial-summary \
    outputs/path_feedback_batch_guarded_experimental_policy_staged_release_trial_v1/guarded-experimental-policy-staged-release-trial-summary.json \
  --validate-only

git diff --check
```

## Non-Goals

No real executor connection, no online release, no new PPO update, no checkpoint
publication, no default policy replacement, no network/action-space/default-A*
change, no gate relaxation, no new raw data download, no Ackermann-feasible
trajectory claim, and no use of IRIS/GCS/path-planner diagnostics as training or
release approval.
