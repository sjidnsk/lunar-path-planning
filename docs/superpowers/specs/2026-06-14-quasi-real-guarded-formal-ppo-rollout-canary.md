# Quasi-Real Guarded Formal PPO Rollout Canary v1

## Summary

This stage is the first guarded formal PPO rollout canary after the passed
formal preflight. It is a conservative, reversible training road test over the
684 train split, gate-clean quasi-real transitions. It does not publish a
checkpoint, replace the default policy, claim policy performance, or claim
formal training readiness.

## Artifacts

- `configs/quasi_real_guarded_formal_ppo_rollout_canary_v1.json`
- `scripts/run_quasi_real_guarded_formal_ppo_rollout_canary.py/.sh`
- `scripts/run_quasi_real_guarded_formal_ppo_rollout_canary_closure.sh`
- `tests/test_quasi_real_guarded_formal_ppo_rollout_canary.py`
- `outputs/path_feedback_batch_quasi_real_guarded_formal_ppo_rollout_canary_v1/`

The output root writes:

- `quasi-real-guarded-formal-ppo-rollout-canary-summary.json`
- `formal-rollout-canary-seed-summaries.jsonl`
- `formal-rollout-canary-training-curves.json`
- `formal-rollout-canary-progress.jsonl`
- `formal-rollout-canary-gate-audit.json`
- `formal-rollout-canary-rollback-manifest.json`
- `formal-rollout-canary-readiness-validate-only.json`
- `formal-rollout-canary-report.md`

## Acceptance Gates

- input formal preflight summary is passed and traceable
- freeze manifest and preflight rollback manifest are present
- summary `status=passed`, `reason_codes=[]`
- `input_trainable_transition_count=684`
- `optimizer_train_transition_count=684`
- `unique_trainable_context_count=684`
- validation/test/fallback/teacher fallback trainable counts are 0
- missing observation/log_prob/value counts are 0
- non-finite reward/return/advantage/loss/gradient counts are 0
- `seed_count=3`, `passed_seed_count=3`
- old log_prob/value max abs error `<=1e-4`
- `abs(approx_kl)<=0.25`
- `max_grad_norm_after_clip<=1.0`
- parameter delta is positive for every seed
- teacher agreement `>=0.95`
- controlled safety/contract/path-risk/source-selection regression counts are 0
- rollback manifest points back to the formal preflight baseline
- no checkpoint publication, no default-policy replacement, no performance
  claim, no formal training ready claim
- readiness status `quasi_real_guarded_formal_ppo_rollout_canary_evaluated`

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_quasi_real_guarded_formal_ppo_rollout_canary.py \
  tests/test_quasi_real_guarded_formal_ppo_preflight.py \
  tests/test_policy_training_readiness_review.py

PYTHON=$P bash scripts/run_quasi_real_guarded_formal_ppo_rollout_canary_closure.sh

jq '{status, reason_codes, seed_count, passed_seed_count, readiness_status, controlled_regression_count}' \
  outputs/path_feedback_batch_quasi_real_guarded_formal_ppo_rollout_canary_v1/quasi-real-guarded-formal-ppo-rollout-canary-summary.json

git diff --check
```

## Non-Goals

No checkpoint publication, no default policy replacement, no network/action
space/default-A* change, no distance/path-risk/source-selection gate relaxation,
no new raw-data download, no Ackermann-feasible trajectory claim, no IRIS/GCS or
path-planner diagnostic promotion to training release evidence, no policy
performance claim, and no formal training ready claim.
