# Formal PPO Candidate Selection & Long-Horizon Holdout v1

## Summary

This stage follows the passed quasi-real guarded formal PPO stability and
holdout validation. It selects one auditable experimental candidate from the
30-run seed/budget matrix, then evaluates a longer horizon-10 holdout without
running a new PPO update. It is not checkpoint publication, default policy
replacement, a policy-performance claim, or a formal-training-ready claim.

## Inputs

- Stability root:
  `outputs/path_feedback_batch_quasi_real_guarded_formal_ppo_stability_holdout_validation_v1/`
- Stability summary:
  `quasi-real-guarded-formal-ppo-stability-holdout-validation-summary.json`
- Stability matrix:
  `formal-ppo-stability-matrix.jsonl`
- Baseline manifest, rollback manifest, and the referenced 684 train split,
  gate-clean quasi-real transition steps.

## Artifacts

- `configs/quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout_v1.json`
- `scripts/run_quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout.py/.sh`
- `scripts/run_quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout_closure.sh`
- `tests/test_quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout.py`
- `outputs/path_feedback_batch_quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout_v1/`

The output root writes:

- `quasi-real-guarded-formal-ppo-candidate-selection-long-horizon-holdout-summary.json`
- `candidate-selection-audit.json`
- `long-horizon-holdout-episodes.jsonl`
- `long-horizon-holdout-steps.jsonl`
- `long-horizon-return-audit.json`
- `long-horizon-holdout-split-report.json`
- `long-horizon-family-report.json`
- `selected-candidate-manifest.json`
- `candidate-selection-rollback-manifest.json`
- `candidate-selection-readiness-validate-only.json`
- `candidate-selection-long-horizon-holdout-report.md`

## Acceptance Gates

- input stability/holdout summary is passed and traceable
- selected candidate comes from the 30-run stability matrix
- candidate selection is deterministic and not based on one loss/reward field
- eligible candidate count is positive
- horizon is at least 10
- validation/test/fallback/teacher fallback/non-empty gate reason trainable
  counts are 0
- missing observation/log_prob/value counts are 0
- non-finite reward/return/advantage/loss/gradient counts are 0
- long-horizon discounted return and advantage are finite
- old log_prob/value max abs error `<=1e-4`
- `abs(approx_kl)<=0.25`
- `max_grad_norm_after_clip<=1.0`
- selected candidate parameter delta is positive
- teacher agreement `>=0.95`
- controlled safety/contract/path-risk/source-selection regression counts are 0
- family regression count is 0
- no new PPO update is run
- no checkpoint publication, default-policy replacement, performance claim, or
  formal-training-ready claim
- readiness status is
  `quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout_evaluated`

## Current Closure

The current closure passes with selected seed `0`, selected budget
`epochs1_lr3e-6`, `eligible_candidate_count=30`, `horizon=10`, 684
long-horizon steps, 68 completed horizon-10 episodes, 4 tail steps,
`teacher_agreement_rate=1.0`, controlled regression 0, family regression 0,
and no publication or formal-ready claim.

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout.py \
  tests/test_quasi_real_guarded_formal_ppo_stability_holdout_validation.py \
  tests/test_policy_training_readiness_review.py

PYTHON=$P bash scripts/run_quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout_closure.sh

jq '{status,reason_codes,selected_seed,selected_budget,horizon,readiness_status}' \
  outputs/path_feedback_batch_quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout_v1/quasi-real-guarded-formal-ppo-candidate-selection-long-horizon-holdout-summary.json

git diff --check
```

## Non-Goals

No new PPO update, no checkpoint publication, no default policy replacement,
no network/action space/default-A* change, no distance/path-risk/source-selection
gate relaxation, no new raw-data download, no Ackermann-feasible trajectory
claim, no IRIS/GCS or path-planner diagnostic promotion to training release
evidence, no policy-performance claim, and no formal-training-ready claim.
