# Selected Formal PPO Candidate Multi-Horizon Shadow Rollout v1

## Summary

This stage follows the passed `Formal PPO Candidate Selection & Long-Horizon
Holdout v1`. It keeps the selected experimental candidate fixed and performs a
read-only multi-horizon shadow rollout over horizons 10, 20, and 30. It does not
run a new PPO update, publish a checkpoint, replace the default policy, claim
policy performance, or claim formal training readiness.

## Inputs

- Candidate-selection root:
  `outputs/path_feedback_batch_quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout_v1/`
- Candidate-selection summary:
  `quasi-real-guarded-formal-ppo-candidate-selection-long-horizon-holdout-summary.json`
- Selected-candidate manifest:
  `selected-candidate-manifest.json`
- Input steps:
  `long-horizon-holdout-steps.jsonl`

## Artifacts

- `configs/selected_formal_ppo_candidate_multihorizon_shadow_rollout_v1.json`
- `scripts/run_selected_formal_ppo_candidate_multihorizon_shadow_rollout.py/.sh`
- `scripts/run_selected_formal_ppo_candidate_multihorizon_shadow_rollout_closure.sh`
- `tests/test_selected_formal_ppo_candidate_multihorizon_shadow_rollout.py`
- `outputs/path_feedback_batch_selected_formal_ppo_candidate_multihorizon_shadow_rollout_v1/`

The output root writes:

- `multihorizon-shadow-rollout-summary.json`
- `multihorizon-shadow-rollout-episodes.jsonl`
- `multihorizon-shadow-rollout-steps.jsonl`
- `multihorizon-return-audit.json`
- `multihorizon-rejection-report.json`
- `multihorizon-family-report.json`
- `multihorizon-readiness-validate-only.json`
- `multihorizon-shadow-rollout-report.md`

## Acceptance Gates

- input candidate-selection summary is passed and traceable
- selected seed and budget match the selected-candidate manifest
- horizons are exactly `[10,20,30]`
- all horizons produce completed episodes
- trainable input remains 684 unique train contexts
- shadow trainable transition records equal 2052 across the three horizon views
- validation/test/fallback/teacher fallback/non-empty gate reason trainable counts are 0
- missing observation/log_prob/value counts are 0
- non-finite reward/return/advantage and shadow return/advantage counts are 0
- controlled safety/contract/path-risk/source-selection regression counts are 0
- family regression count is 0
- teacher agreement is at least 0.95
- multi-step discounted return is used
- no new PPO update is run
- no checkpoint publication, default-policy replacement, performance claim, or
  formal-training-ready claim
- readiness status is
  `selected_formal_ppo_candidate_multihorizon_shadow_rollout_evaluated`

## Current Closure

The current closure passes with selected seed `0`, selected budget
`epochs1_lr3e-6`, horizons `[10,20,30]`,
`input_trainable_transition_count=684`,
`shadow_trainable_transition_count=2052`,
`unique_trainable_context_count=684`, completed episode counts `68/34/22`,
`teacher_agreement_rate=1.0`, controlled regression 0, family regression 0, and
no publication or formal-ready claim.

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_selected_formal_ppo_candidate_multihorizon_shadow_rollout.py \
  tests/test_quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout.py \
  tests/test_policy_training_readiness_review.py

PYTHON=$P bash scripts/run_selected_formal_ppo_candidate_multihorizon_shadow_rollout_closure.sh

jq '{status, reason_codes, horizons, controlled_regression_count, family_regression_count, readiness_status}' \
  outputs/path_feedback_batch_selected_formal_ppo_candidate_multihorizon_shadow_rollout_v1/multihorizon-shadow-rollout-summary.json

git diff --check
```

## Non-Goals

No new PPO update, no checkpoint publication, no default policy replacement,
no network/action space/default-A* change, no distance/path-risk/source-selection
gate relaxation, no raw-data download, no Ackermann-feasible trajectory claim,
no IRIS/GCS or path-planner diagnostic promotion to training release evidence,
no policy-performance claim, and no formal-training-ready claim.
