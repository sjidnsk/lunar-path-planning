# Quasi-Real Guarded Formal PPO Preflight v1

## Summary

This stage adds the formal PPO preflight after the frozen quasi-real iterative
mini-loop baseline. It does not start formal PPO rollout. It validates whether
the frozen 684 train split, gate-clean quasi-real transitions can be consumed by
three seed-level tiny PPO smokes without teacher-skill or controlled-gate
regression.

## Artifacts

- `configs/quasi_real_guarded_formal_ppo_preflight_v1.json`
- `scripts/run_quasi_real_guarded_formal_ppo_preflight.py/.sh`
- `scripts/run_quasi_real_guarded_formal_ppo_preflight_closure.sh`
- `tests/test_quasi_real_guarded_formal_ppo_preflight.py`
- `outputs/path_feedback_batch_quasi_real_guarded_formal_ppo_preflight_v1/`

The output root writes:

- `quasi-real-guarded-formal-ppo-preflight-summary.json`
- `formal-preflight-seed-summaries.jsonl`
- `formal-preflight-training-curves.json`
- `formal-preflight-gate-audit.json`
- `formal-preflight-rollback-manifest.json`
- `formal-preflight-readiness-validate-only.json`
- `formal-preflight-report.md`

## Acceptance Gates

- summary `status=passed`, `reason_codes=[]`
- frozen mini-loop evidence and manifest are present and passed
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
- parameter delta is positive
- teacher agreement `>=0.95`
- controlled safety/contract/path-risk/source-selection regression counts are 0
- rollback manifest exists
- no checkpoint publication, no default-policy replacement, no performance
  claim, no formal training ready claim
- readiness status `quasi_real_guarded_formal_ppo_preflight_evaluated`

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_quasi_real_guarded_formal_ppo_preflight.py \
  tests/test_quasi_real_guarded_ppo_iterative_miniloop_evidence_freeze.py \
  tests/test_quasi_real_guarded_ppo_iterative_miniloop_stability.py \
  tests/test_policy_training_readiness_review.py

PYTHON=$P bash scripts/run_quasi_real_guarded_formal_ppo_preflight_closure.sh

jq '{status, reason_codes, input_trainable_transition_count, optimizer_train_transition_count, seed_count, passed_seed_count, readiness_status, controlled_regression_count}' \
  outputs/path_feedback_batch_quasi_real_guarded_formal_ppo_preflight_v1/quasi-real-guarded-formal-ppo-preflight-summary.json

git diff --check
```

## Current Result

The current closure passed with 684 input trainable transitions, 684 optimizer
transitions, three passed seeds, teacher agreement 1.0, old log_prob/value error
0.0/0.0, max abs KL about `1.98e-5`, clipped grad norm within 1.0, controlled
regression count 0, and readiness
`quasi_real_guarded_formal_ppo_preflight_evaluated`.

## Non-Goals

No formal PPO rollout, no new raw-data download, no checkpoint publication, no
default policy replacement, no network/action-space/default-A* change, no
distance/path-risk/source-selection gate relaxation, no Ackermann-feasible
trajectory claim, and no IRIS/GCS/path-planner diagnostic promotion to training
release evidence.
