# Guarded Experimental Policy Staged Release Preflight v1

## Summary

This stage follows `Guarded Experimental Policy Shadow Release Trial v1`.
It audits the release controls needed before any guarded staged release. It
does not execute staged release and does not allow the experimental policy to
control the authoritative output.

## Input

- Shadow release trial summary:
  `outputs/path_feedback_batch_guarded_experimental_policy_shadow_release_trial_v1/guarded-experimental-policy-shadow-release-trial-summary.json`

## Outputs

- `configs/guarded_experimental_policy_staged_release_preflight_v1.json`
- `scripts/run_guarded_experimental_policy_staged_release_preflight.py`
- `scripts/run_guarded_experimental_policy_staged_release_preflight.sh`
- `scripts/run_guarded_experimental_policy_staged_release_preflight_closure.sh`
- `tests/test_guarded_experimental_policy_staged_release_preflight.py`
- `outputs/path_feedback_batch_guarded_experimental_policy_staged_release_preflight_v1/guarded-experimental-policy-staged-release-preflight-summary.json`
- `outputs/path_feedback_batch_guarded_experimental_policy_staged_release_preflight_v1/staged-release-preflight-manifest.json`
- `outputs/path_feedback_batch_guarded_experimental_policy_staged_release_preflight_v1/staged-release-gate-threshold-audit.json`
- `outputs/path_feedback_batch_guarded_experimental_policy_staged_release_preflight_v1/staged-release-kill-switch-audit.json`
- `outputs/path_feedback_batch_guarded_experimental_policy_staged_release_preflight_v1/staged-release-rollback-audit.json`
- `outputs/path_feedback_batch_guarded_experimental_policy_staged_release_preflight_v1/staged-release-telemetry-audit.json`
- `outputs/path_feedback_batch_guarded_experimental_policy_staged_release_preflight_v1/staged-release-preflight-readiness-validate-only.json`
- `outputs/path_feedback_batch_guarded_experimental_policy_staged_release_preflight_v1/staged-release-preflight-report.md`

## Acceptance Gates

- Summary `status=passed`, `reason_codes=[]`.
- `staged_release_preflight_verdict=eligible_for_guarded_staged_release_trial`.
- Source shadow release trial summary is passed and current.
- `staged_release_enabled=false`.
- `default_policy_authoritative=true`.
- `experimental_control_activation_count=0`.
- `shadow_step_count>=512`.
- `unique_shadow_context_count>=256`.
- Missing observation/log_prob/value, invalid mask, and non-finite logits/log_prob/value/reward counts are all 0.
- Controlled safety/contract/path-risk/source-selection regression counts are all 0.
- Gate-threshold, kill-switch, rollback, and telemetry audits pass.
- Readiness validate-only returns
  `guarded_experimental_policy_staged_release_preflight_evaluated` with no
  blockers or reason codes.

## Verification

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python
OUT=outputs/path_feedback_batch_guarded_experimental_policy_staged_release_preflight_v1
SUM=$OUT/guarded-experimental-policy-staged-release-preflight-summary.json

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_guarded_experimental_policy_staged_release_preflight.py \
  tests/test_guarded_experimental_policy_shadow_release_trial.py \
  tests/test_policy_training_readiness_review.py

PYTHON=$P bash scripts/run_guarded_experimental_policy_staged_release_preflight_closure.sh

PYTHON=$P bash scripts/run_policy_training_readiness_review.sh \
  --batch-root outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1 \
  --config configs/policy_training_readiness_review_v1.json \
  --guarded-experimental-policy-staged-release-preflight-summary $SUM \
  --validate-only

jq '{status,reason_codes,staged_release_preflight_verdict,readiness_status,staged_release_enabled,default_policy_authoritative,experimental_control_activation_count}' $SUM
git diff --check
```

## Non-Goals

- No actual staged release.
- No experimental policy control takeover.
- No new PPO update.
- No checkpoint publication.
- No default policy replacement.
- No network, action-space, or default A* changes.
- No relaxation of distance/path-risk/source-selection gates.
- No new raw data download.
- No Ackermann-feasible trajectory claim.
- No use of IRIS/GCS/path-planner diagnostics as training release evidence.
- No policy performance or formal-training-ready claim.
