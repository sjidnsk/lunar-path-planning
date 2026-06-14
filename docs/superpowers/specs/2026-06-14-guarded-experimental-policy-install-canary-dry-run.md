# Guarded Experimental Policy Install Canary Dry-Run v1

## Summary

This stage follows `Guarded Experimental Policy Release Candidate Packaging v1`.
It simulates installing the packaged experimental policy in an isolated sandbox,
loads the checkpoint through the sandbox consumer path, and runs a guarded
64-step canary dry-run. It is not a policy release.

## Inputs

- Packaging summary:
  `outputs/path_feedback_batch_guarded_experimental_policy_release_candidate_packaging_v1/guarded-experimental-policy-release-candidate-packaging-summary.json`
- Package manifest and checkpoint:
  `outputs/path_feedback_batch_guarded_experimental_policy_release_candidate_packaging_v1/release-candidate-package/`
- Multi-horizon holdout steps resolved from the packaging preflight lineage.

## Outputs

- `configs/guarded_experimental_policy_install_canary_dry_run_v1.json`
- `scripts/run_guarded_experimental_policy_install_canary_dry_run.py`
- `scripts/run_guarded_experimental_policy_install_canary_dry_run.sh`
- `scripts/run_guarded_experimental_policy_install_canary_dry_run_closure.sh`
- `tests/test_guarded_experimental_policy_install_canary_dry_run.py`
- `outputs/path_feedback_batch_guarded_experimental_policy_install_canary_dry_run_v1/guarded-experimental-policy-install-canary-dry-run-summary.json`
- `outputs/path_feedback_batch_guarded_experimental_policy_install_canary_dry_run_v1/install-canary-sandbox-manifest.json`
- `outputs/path_feedback_batch_guarded_experimental_policy_install_canary_dry_run_v1/install-canary-package-consumer-audit.json`
- `outputs/path_feedback_batch_guarded_experimental_policy_install_canary_dry_run_v1/install-canary-step-audit.jsonl`
- `outputs/path_feedback_batch_guarded_experimental_policy_install_canary_dry_run_v1/install-canary-rollback-audit.json`
- `outputs/path_feedback_batch_guarded_experimental_policy_install_canary_dry_run_v1/install-canary-readiness-validate-only.json`
- `outputs/path_feedback_batch_guarded_experimental_policy_install_canary_dry_run_v1/install-canary-report.md`

## Acceptance Gates

- Summary `status=passed`, `reason_codes=[]`.
- `install_canary_verdict=eligible_for_guarded_shadow_release_trial`.
- Package checkpoint hash/size matches the sandbox consumer audit.
- Sandbox manifest passed.
- Default policy snapshot is unchanged.
- `canary_step_count>=64`.
- Missing observation/log_prob/value, invalid mask, and non-finite logits/log_prob/value/reward counts are all 0.
- Controlled safety/contract/path-risk/source-selection regression counts are all 0.
- Rollback/default audit passed.
- Readiness validate-only returns
  `guarded_experimental_policy_install_canary_dry_run_evaluated` with no blockers
  or reason codes.

## Verification

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python
OUT=outputs/path_feedback_batch_guarded_experimental_policy_install_canary_dry_run_v1
SUM=$OUT/guarded-experimental-policy-install-canary-dry-run-summary.json
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_guarded_experimental_policy_install_canary_dry_run.py \
  tests/test_guarded_experimental_policy_release_candidate_packaging.py \
  tests/test_policy_training_readiness_review.py
PYTHON=$P bash scripts/run_guarded_experimental_policy_install_canary_dry_run_closure.sh
PYTHON=$P bash scripts/run_policy_training_readiness_review.sh \
  --batch-root outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1 \
  --config configs/policy_training_readiness_review_v1.json \
  --guarded-experimental-policy-install-canary-dry-run-summary $SUM \
  --validate-only
jq '{status,reason_codes,install_canary_verdict,readiness_status,canary_step_count,controlled_regression_count}' $SUM
git diff --check
```

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

## Current Result

The current closure passes with 64 canary steps, zero controlled regression,
matching package/consumer SHA-256
`9d9539c685ab965739c91958bf9cbfe90329c460b4bdbcc35881875aa62f0aa2`, and
readiness status `guarded_experimental_policy_install_canary_dry_run_evaluated`.
The next stage is a guarded shadow release trial, not a default-policy release.
