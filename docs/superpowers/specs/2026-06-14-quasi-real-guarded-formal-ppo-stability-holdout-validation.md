# Quasi-Real Guarded Formal PPO Stability & Holdout Validation v1

## Summary

This stage follows the passed guarded formal PPO rollout canary. It turns the
single canary road test into a seed-and-budget stability matrix with
validation/test holdout diagnostics. It is not checkpoint publication, default
policy replacement, a policy-performance claim, or a formal-training-ready
claim.

## Inputs

- Canary root:
  `outputs/path_feedback_batch_quasi_real_guarded_formal_ppo_rollout_canary_v1/`
- Canary summary:
  `quasi-real-guarded-formal-ppo-rollout-canary-summary.json`
- Canary seed summaries, progress JSONL, rollback manifest, and the referenced
  684 train split, gate-clean quasi-real transition steps.

## Artifacts

- `configs/quasi_real_guarded_formal_ppo_stability_holdout_validation_v1.json`
- `scripts/run_quasi_real_guarded_formal_ppo_stability_holdout_validation.py/.sh`
- `scripts/run_quasi_real_guarded_formal_ppo_stability_holdout_validation_closure.sh`
- `tests/test_quasi_real_guarded_formal_ppo_stability_holdout_validation.py`
- `outputs/path_feedback_batch_quasi_real_guarded_formal_ppo_stability_holdout_validation_v1/`

The output root writes:

- `quasi-real-guarded-formal-ppo-stability-holdout-validation-summary.json`
- `formal-ppo-stability-baseline-manifest.json`
- `formal-ppo-stability-matrix.jsonl`
- `formal-ppo-stability-training-curves.json`
- `formal-ppo-stability-holdout-audit.json`
- `formal-ppo-stability-family-regression-report.json`
- `formal-ppo-stability-rollback-manifest.json`
- `formal-ppo-stability-readiness-validate-only.json`
- `formal-ppo-stability-report.md`

## Acceptance Gates

- input Canary summary is passed and traceable
- input trainable, optimizer, and unique context counts are all 684
- validation/test/fallback/teacher fallback/non-empty gate reason trainable
  counts are 0
- missing observation/log_prob/value counts are 0
- non-finite reward/return/advantage/loss/gradient counts are 0
- seed count is at least 5 and budget count is at least 2
- all seed-budget runs pass
- old log_prob/value max abs error `<=1e-4`
- `abs(approx_kl)<=0.25`
- `max_grad_norm_after_clip<=1.0`
- parameter delta is positive for every run
- teacher agreement `>=0.95`
- train/validation/test controlled regression counts are 0
- safety/contract/path-risk/source-selection regression counts are 0
- family regression count is 0
- baseline and rollback manifests are present
- no checkpoint publication, default-policy replacement, performance claim, or
  formal-training-ready claim
- readiness status is
  `quasi_real_guarded_formal_ppo_stability_holdout_validated`

## Current Closure

The current closure passes with 684 trainable transitions, 5 seeds, 6 budget
settings, 30/30 passed runs, `teacher_agreement_rate=1.0`, controlled
regression 0, validation/test controlled regression 0, family regression 0,
old `log_prob/value` error `0.0/0.0`, max `abs(approx_kl)` about `2.92e-5`,
and max clipped grad norm `1.0`.

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_quasi_real_guarded_formal_ppo_stability_holdout_validation.py \
  tests/test_quasi_real_guarded_formal_ppo_rollout_canary.py \
  tests/test_policy_training_readiness_review.py

PYTHON=$P bash scripts/run_quasi_real_guarded_formal_ppo_stability_holdout_validation_closure.sh

jq '{status, reason_codes, seed_count, budget_count, passed_run_count, readiness_status, controlled_regression_count}' \
  outputs/path_feedback_batch_quasi_real_guarded_formal_ppo_stability_holdout_validation_v1/quasi-real-guarded-formal-ppo-stability-holdout-validation-summary.json

git diff --check
```

## Non-Goals

No checkpoint publication, no default policy replacement, no network/action
space/default-A* change, no distance/path-risk/source-selection gate
relaxation, no new raw-data download, no Ackermann-feasible trajectory claim,
no IRIS/GCS or path-planner diagnostic promotion to training release evidence,
no policy-performance claim, and no formal-training-ready claim.
