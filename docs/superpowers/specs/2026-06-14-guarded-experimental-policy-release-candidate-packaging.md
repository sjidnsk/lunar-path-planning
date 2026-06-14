# Guarded Experimental Policy Release Candidate Packaging v1

## Purpose

This stage packages the selected formal PPO experimental candidate after the
promotion decision review. It is a packaging and audit step only: a passed
summary means the candidate is eligible for a later guarded install/canary
dry-run, not that the checkpoint is published or installed as the default
policy.

## Inputs

- Decision review root:
  `outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_decision_review_v1/`
- Decision review summary:
  `selected-formal-ppo-candidate-promotion-decision-review-summary.json`
- Required verdict:
  `eligible_for_guarded_release_candidate_packaging`
- Expected checkpoint SHA-256:
  `9d9539c685ab965739c91958bf9cbfe90329c460b4bdbcc35881875aa62f0aa2`

The packager follows the decision review lineage and copies the experimental
checkpoint into an isolated package root without touching any default policy
path.

## Outputs

- Config:
  `configs/guarded_experimental_policy_release_candidate_packaging_v1.json`
- Runner:
  `scripts/run_guarded_experimental_policy_release_candidate_packaging.py/.sh`
- Closure:
  `scripts/run_guarded_experimental_policy_release_candidate_packaging_closure.sh`
- Tests:
  `tests/test_guarded_experimental_policy_release_candidate_packaging.py`
- Output root:
  `outputs/path_feedback_batch_guarded_experimental_policy_release_candidate_packaging_v1/`

Output files:

- `guarded-experimental-policy-release-candidate-packaging-summary.json`
- `release-candidate-package-manifest.json`
- `checkpoint-hash-audit.json`
- `checkpoint-load-audit.json`
- `rollback-audit.json`
- `packaging-readiness-validate-only.json`
- `release-candidate-packaging-report.md`

## Acceptance

- Summary `status=passed`, `reason_codes=[]`.
- `package_verdict=eligible_for_guarded_install_dry_run`.
- Original and packaged checkpoint SHA-256/size match the decision review.
- Load smoke passes on at least 64 multi-horizon observation samples with no
  invalid mask, missing observation, or non-finite logits/log-prob/value.
- Rollback audit passes: the package is isolated and deletable, and the original
  experimental candidate remains traceable.
- Readiness accepts
  `--guarded-experimental-policy-release-candidate-packaging-summary` and returns
  `guarded_experimental_policy_release_candidate_packaging_evaluated`.

## Non-Goals

- Do not run a new PPO update.
- Do not execute install/canary.
- Do not publish or replace any checkpoint/default policy.
- Do not modify network, action space, default A*, or distance/path-risk/source-selection gates.
- Do not download new raw data.
- Do not claim Ackermann-feasible trajectory output.
- Do not treat IRIS/GCS/path-planner diagnostics as training release evidence.
- Do not claim policy performance or formal training readiness.

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_guarded_experimental_policy_release_candidate_packaging.py \
  tests/test_selected_formal_ppo_candidate_promotion_decision_review.py \
  tests/test_policy_training_readiness_review.py

PYTHON=$P bash scripts/run_guarded_experimental_policy_release_candidate_packaging_closure.sh

jq '{status,reason_codes,package_verdict,readiness_status}' \
  outputs/path_feedback_batch_guarded_experimental_policy_release_candidate_packaging_v1/guarded-experimental-policy-release-candidate-packaging-summary.json
```
