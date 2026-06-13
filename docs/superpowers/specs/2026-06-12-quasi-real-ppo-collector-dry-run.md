# Quasi-Real PPO Collector Dry-Run v1

## Summary

`Quasi-Real Guarded Teacher-Following Pilot v1` passed on 108 quasi-real LOLA
contexts. `Quasi-Real PPO Collector Dry-Run v1` turns those guarded decisions
into PPO-consumable rollout artifacts without running PPO updates.

The default split contract is strict: only `train` split decisions with
`controlled_choice_source=policy_teacher_aligned` or
`controlled_choice_source=policy_safe_disagreement`, no gate reason codes, and
valid rollout fields are trainable. `validation` and `test` splits are
diagnostic-only.

## Scope

Added artifacts:

- `configs/quasi_real_ppo_collector_dry_run_v1.json`
- `scripts/run_quasi_real_ppo_collector_dry_run.py/.sh`
- `scripts/run_quasi_real_ppo_collector_closure.sh`
- `tests/test_quasi_real_ppo_collector_dry_run.py`

Output root:

- `outputs/path_feedback_batch_quasi_real_ppo_collector_dry_run_v1/`

Output files:

- `ppo-rollout-episodes.jsonl`
- `ppo-rollout-transitions.jsonl`
- `ppo-rollout-collector-summary.json`
- `ppo-rollout-rejection-report.json`
- `ppo-rollout-reward-audit.json`

## Behavior

The collector reads
`outputs/path_feedback_batch_quasi_real_guarded_teacher_following_pilot_v1/`,
loads the guarded teacher-following summary, derives `candidate_root` and
`quasi_real_root`, consumes
`quasi-real-guarded-teacher-following-decisions.jsonl`, and reconstructs
`PolicyObservation` from quasi-real path-feedback scenario candidates.

Each trainable context becomes a single-step episode with `done=true`.
`log_prob` and `value` are read from the decision record when present or
computed from the distillation candidate checkpoint. Missing observation,
missing `log_prob`, missing `value`, or non-finite reward fails the collector.

Reward is intentionally narrow:

- `teacher_following_bonus=1.0` for trainable teacher-aligned steps
- `safe_disagreement_bonus=1.0` for trainable safe disagreements
- `gate_regression_penalty` is audit-only for diagnostic/rejected records

## Current Evidence

`PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python bash
scripts/run_quasi_real_ppo_collector_closure.sh` passes and reports:

- `episode_count=108`
- `step_count=108`
- `materialized_episode_count=36`
- `ppo_trainable_transition_count=36`
- `diagnostic_transition_count=72`
- `source_fallback_trainable_count=0`
- invalid/empty mask, missing log-prob/value, non-finite reward, fallback,
  safety, contract, path/risk, and source-selection regression counters all `0`
- reward audit is finite and trainable rewards are not all zero
- readiness validate-only reaches
  `training_readiness_status=ppo_rollout_collector_dry_run_evaluated`

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_quasi_real_ppo_collector_dry_run.py \
  tests/test_ppo_rollout_collector_dry_run.py \
  tests/test_quasi_real_guarded_teacher_following_pilot.py \
  tests/test_policy_training_readiness_review.py

PYTHON=$P bash scripts/run_quasi_real_ppo_collector_closure.sh

PYTHON=$P bash scripts/run_policy_training_readiness_review.sh \
  --batch-root outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1 \
  --config configs/policy_training_readiness_review_v1.json \
  --quasi-real-guarded-teacher-following-pilot-summary \
    outputs/path_feedback_batch_quasi_real_guarded_teacher_following_pilot_v1/quasi-real-guarded-teacher-following-pilot-summary.json \
  --ppo-rollout-collector-summary \
    outputs/path_feedback_batch_quasi_real_ppo_collector_dry_run_v1/ppo-rollout-collector-summary.json \
  --validate-only
```

## Non-Goals

No PPO optimizer update, no checkpoint publication, no default policy
replacement, no network/action space/default A* change, no distance/path-risk or
source-selection gate relaxation, no Ackermann-feasible trajectory claim, no
IRIS/GCS diagnostic promoted to training release evidence, and no formal
training readiness or policy performance claim.

The next stage is `Limited Quasi-Real PPO Update Smoke`, not broad PPO training.
