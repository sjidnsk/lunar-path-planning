# Guarded Experimental Policy Shadow Release Trial v1

## Summary

This stage follows `Guarded Experimental Policy Install Canary Dry-Run v1`.
It runs the packaged experimental policy as a shadow-only release trial: the
experimental policy can produce side-channel decisions and gate audit records,
but it never controls the authoritative output and never replaces the default
policy.

## Inputs

- Install canary summary:
  `outputs/path_feedback_batch_guarded_experimental_policy_install_canary_dry_run_v1/guarded-experimental-policy-install-canary-dry-run-summary.json`
- Multi-horizon shadow rollout summary and steps:
  `outputs/path_feedback_batch_selected_formal_ppo_candidate_multihorizon_shadow_rollout_v1/`

## Outputs

- `configs/guarded_experimental_policy_shadow_release_trial_v1.json`
- `scripts/run_guarded_experimental_policy_shadow_release_trial.py`
- `scripts/run_guarded_experimental_policy_shadow_release_trial.sh`
- `scripts/run_guarded_experimental_policy_shadow_release_trial_closure.sh`
- `tests/test_guarded_experimental_policy_shadow_release_trial.py`
- `outputs/path_feedback_batch_guarded_experimental_policy_shadow_release_trial_v1/guarded-experimental-policy-shadow-release-trial-summary.json`
- `outputs/path_feedback_batch_guarded_experimental_policy_shadow_release_trial_v1/shadow-release-runtime-manifest.json`
- `outputs/path_feedback_batch_guarded_experimental_policy_shadow_release_trial_v1/shadow-release-step-comparison.jsonl`
- `outputs/path_feedback_batch_guarded_experimental_policy_shadow_release_trial_v1/shadow-release-rejection-report.json`
- `outputs/path_feedback_batch_guarded_experimental_policy_shadow_release_trial_v1/shadow-release-risk-reward-audit.json`
- `outputs/path_feedback_batch_guarded_experimental_policy_shadow_release_trial_v1/shadow-release-readiness-validate-only.json`
- `outputs/path_feedback_batch_guarded_experimental_policy_shadow_release_trial_v1/shadow-release-report.md`

## Acceptance Gates

- Summary `status=passed`, `reason_codes=[]`.
- `shadow_release_trial_verdict=eligible_for_guarded_staged_release_preflight`.
- `shadow_step_count>=256`.
- `unique_shadow_context_count>=256`.
- Package and consumer checkpoint SHA-256/size match.
- Default policy remains unchanged and rollback audit passed.
- Experimental policy remains shadow-only: `shadow_policy_takes_control=false`.
- Missing observation/log_prob/value, invalid mask, and non-finite logits/log_prob/value/reward counts are all 0.
- Controlled safety/contract/path-risk/source-selection regression counts are all 0.
- Shadow rejection/fallback rows are diagnostic only and do not count as controlled regression.
- Readiness validate-only returns
  `guarded_experimental_policy_shadow_release_trial_evaluated` with no blockers
  or reason codes.

## Verification

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python
OUT=outputs/path_feedback_batch_guarded_experimental_policy_shadow_release_trial_v1
SUM=$OUT/guarded-experimental-policy-shadow-release-trial-summary.json

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_guarded_experimental_policy_shadow_release_trial.py \
  tests/test_guarded_experimental_policy_install_canary_dry_run.py \
  tests/test_policy_training_readiness_review.py

PYTHON=$P bash scripts/run_guarded_experimental_policy_shadow_release_trial_closure.sh

PYTHON=$P bash scripts/run_policy_training_readiness_review.sh \
  --batch-root outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1 \
  --config configs/policy_training_readiness_review_v1.json \
  --guarded-experimental-policy-shadow-release-trial-summary $SUM \
  --validate-only

jq '{status,reason_codes,shadow_release_trial_verdict,readiness_status,shadow_step_count,unique_shadow_context_count,controlled_regression_count}' $SUM
git diff --check
```

## Current Result

The current closure passes with `shadow_step_count=2052`,
`unique_shadow_context_count=684`, `controlled_regression_count=0`, no shadow
fallback/rejection diagnostics in the consumed multi-horizon batch, and
readiness status `guarded_experimental_policy_shadow_release_trial_evaluated`.

## Non-Goals

- No new PPO update.
- No checkpoint publication.
- No default policy replacement.
- No network, action-space, or default A* changes.
- No relaxation of distance/path-risk/source-selection gates.
- No new raw data download.
- No Ackermann-feasible trajectory claim.
- No use of IRIS/GCS/path-planner diagnostics as training release evidence.
- No policy performance or formal-training-ready claim.
