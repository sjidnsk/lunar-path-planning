# Quasi-Real Guarded PPO Rollout Pilot v1

## Summary

This stage evaluates the return-aligned experimental policy in quasi-real
guarded multi-step rollout. It does not run a new PPO update, publish a
checkpoint, or replace the default policy. The goal is to prove that the
post-update policy remains teacher-skill aligned under horizon-3 guarded
execution, with gate/fallback protection and readiness-consumable evidence.

## Inputs

- Updated experimental candidate root:
  `outputs/path_feedback_batch_return_aligned_guarded_ppo_update_smoke_v1/`
- Quasi-real root:
  `outputs/path_feedback_batch_quasi_real_safe_better_opportunity_expansion_v1/`
- Required upstream summaries:
  `return-aligned-guarded-ppo-update-smoke-summary.json`,
  post-update quasi-real teacher-following summary and decisions,
  post-update long-horizon summary, and post-update return-aligned replay
  summary.

## Artifacts

- `configs/quasi_real_guarded_ppo_rollout_pilot_v1.json`
- `scripts/run_quasi_real_guarded_ppo_rollout_pilot.py/.sh`
- `scripts/run_quasi_real_guarded_ppo_rollout_pilot_closure.sh`
- `tests/test_quasi_real_guarded_ppo_rollout_pilot.py`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_rollout_pilot_v1/`
- `README.md`
- `docs/算法设计与系统架构报告.md`

Output files:

- `quasi-real-guarded-ppo-rollout-pilot-summary.json`
- `quasi-real-guarded-ppo-rollout-episodes.jsonl`
- `quasi-real-guarded-ppo-rollout-steps.jsonl`
- `quasi-real-guarded-ppo-rollout-rejection-report.json`
- `quasi-real-guarded-ppo-rollout-reward-audit.json`

## Contract

The pilot groups quasi-real decisions into horizon-3 single-path episodes. Each
step reconstructs `PolicyObservation`, records action/log-prob/value/reward,
and computes discounted return and advantage. Policy-aligned or safe-disagree
teacher-following decisions are normalized to `controlled_choice_source=policy`
when the policy actually controls the step. Fallback and rejected raw probes
stay diagnostic.

Trainable transitions must be:

- `split=train`
- `controlled_choice_source=policy`
- empty gate reason codes
- complete observation/action/log-prob/value
- finite reward, return, and advantage

Validation/test, source fallback, teacher fallback, unsafe decisions, and
non-empty gate reason rows are diagnostic-only.

## Readiness

Readiness accepts
`--quasi-real-guarded-ppo-rollout-pilot-summary` and may advance to
`quasi_real_guarded_ppo_rollout_pilot_evaluated` only when the pilot summary is
passed, reason codes are empty, provenance matches current HEAD, trainable
count is at least 24, collector replay passes, long-horizon verdict remains
`long_horizon_teacher_skill_contract_aligned`, and controlled regression counts
are all zero.

## Current Evidence

The current pilot summary reports:

- `status=passed`, `reason_codes=[]`
- `episode_count=36`, `step_count=108`
- `ppo_trainable_transition_count=36`
- `diagnostic_transition_count=72`
- `controlled_regression_count=0`
- controlled safety/contract/path-risk/source-selection regression: all `0`
- `teacher_agreement_rate=1.0`
- quasi-real collector replay `status=passed`, trainable count `36`
- post-pilot long-horizon verdict:
  `long_horizon_teacher_skill_contract_aligned`

Readiness validate-only returns
`training_readiness_status=quasi_real_guarded_ppo_rollout_pilot_evaluated` with
`training_blockers=[]`.

## Verification

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_quasi_real_guarded_ppo_rollout_pilot.py \
  tests/test_return_aligned_guarded_ppo_update_smoke.py \
  tests/test_policy_training_readiness_review.py

PYTHON=$P bash scripts/run_quasi_real_guarded_ppo_rollout_pilot_closure.sh

PYTHON=$P bash scripts/run_policy_training_readiness_review.sh \
  --batch-root outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1 \
  --config configs/policy_training_readiness_review_v1.json \
  --quasi-real-guarded-ppo-rollout-pilot-summary \
    outputs/path_feedback_batch_quasi_real_guarded_ppo_rollout_pilot_v1/quasi-real-guarded-ppo-rollout-pilot-summary.json \
  --validate-only

jq '{status, reason_codes, episode_count, step_count, controlled_regression_count, teacher_agreement_rate}' \
  outputs/path_feedback_batch_quasi_real_guarded_ppo_rollout_pilot_v1/quasi-real-guarded-ppo-rollout-pilot-summary.json

git diff --check
```

## Non-Goals

No formal PPO rollout, no new PPO parameter update, no checkpoint publication,
no default-policy replacement, no network/action-space/default-A* changes, no
gate relaxation, no Ackermann-feasible trajectory claim, no IRIS/GCS/path
planner diagnostic as training release evidence, no policy performance claim,
and no formal training-ready claim.
