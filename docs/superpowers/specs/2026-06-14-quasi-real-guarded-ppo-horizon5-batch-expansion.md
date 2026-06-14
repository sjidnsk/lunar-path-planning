# Quasi-Real Guarded PPO Horizon-5 Batch Expansion v1

## Summary

This stage starts from the passed `Quasi-Real Guarded PPO Stability Replay &
Acceptance Contract Refinement v1` evidence and performs the first quasi-real
guarded rollout expansion. It is not formal PPO and does not run a new optimizer
update. It expands the audited rollout ledger to `horizon=5`, at least 96
episodes, and at least 96 PPO-trainable transitions while preserving strict
split isolation and gate-clean trainability.

## Artifacts

- `configs/quasi_real_guarded_ppo_horizon5_batch_expansion_v1.json`
- `scripts/run_quasi_real_guarded_ppo_horizon5_batch_expansion.py`
- `scripts/run_quasi_real_guarded_ppo_horizon5_batch_expansion.sh`
- `scripts/run_quasi_real_guarded_ppo_horizon5_batch_expansion_closure.sh`
- `tests/test_quasi_real_guarded_ppo_horizon5_batch_expansion.py`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_horizon5_batch_expansion_v1/quasi-real-guarded-ppo-horizon5-batch-expansion-summary.json`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_horizon5_batch_expansion_v1/quasi-real-guarded-ppo-horizon5-batch-expansion-episodes.jsonl`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_horizon5_batch_expansion_v1/quasi-real-guarded-ppo-horizon5-batch-expansion-steps.jsonl`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_horizon5_batch_expansion_v1/quasi-real-guarded-ppo-horizon5-batch-expansion-reward-audit.json`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_horizon5_batch_expansion_v1/quasi-real-guarded-ppo-horizon5-batch-expansion-rejection-report.json`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_horizon5_batch_expansion_v1/horizon5-batch-expansion-comparison.jsonl`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_horizon5_batch_expansion_v1/horizon5-batch-expansion-readiness-validate-only.json`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_horizon5_batch_expansion_v1/horizon5-batch-expansion-report.md`

## Contract

The runner reads the stability replay summary, acceptance contract, freeze
summary, and freeze manifest. The inputs must be passed and provenance
traceable. The runner follows the freeze summary back to the same quasi-real
guarded pilot steps, deterministically rebuilds 96 five-step episodes, and
recalculates discounted return and advantage over the full episode.

Only train split, controlled policy, gate-clean, fully materialized, finite
steps can be PPO-trainable. Validation/test split, source fallback, teacher
fallback, raw probe rejection, non-empty gate reasons, and path-planner/IRIS/GCS
diagnostics remain diagnostic-only.

The stage performs three deterministic replays of the expanded ledger and
compares split counts, trainable masks, action indexes, gate reasons,
controlled regression reasons, reward, discounted return, and advantage.

## Current Evidence

The current closure writes a passed summary with:

- `status=passed`
- `reason_codes=[]`
- `horizon=5`
- `episode_count=96`
- `step_count=480`
- `ppo_trainable_transition_count=162`
- `diagnostic_transition_count=318`
- `replay_count=3`
- `passed_replay_count=3`
- `readiness_status=quasi_real_guarded_ppo_horizon5_batch_expansion_evaluated`
- `controlled_regression_count=0`
- `teacher_agreement_rate=1.0`
- `baseline_replay_behavior_drift_count=0`
- `quasi_real_collector_replay_status=passed`
- `quasi_real_collector_replay_trainable_transition_count=162`
- `long_horizon_verdict=long_horizon_teacher_skill_contract_aligned`
- `runs_ppo_update=false`
- `publishes_checkpoint=false`
- `replaces_default_policy=false`
- `performance_claimed=false`
- `formal_training_ready_claimed=false`

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_quasi_real_guarded_ppo_horizon5_batch_expansion.py \
  tests/test_quasi_real_guarded_ppo_stability_replay.py \
  tests/test_quasi_real_guarded_ppo_rollout_pilot.py \
  tests/test_policy_training_readiness_review.py

PYTHON=$P bash scripts/run_quasi_real_guarded_ppo_horizon5_batch_expansion_closure.sh

jq '{status, reason_codes, horizon, episode_count, step_count, ppo_trainable_transition_count, replay_count, passed_replay_count, readiness_status, controlled_regression_count}' \
  outputs/path_feedback_batch_quasi_real_guarded_ppo_horizon5_batch_expansion_v1/quasi-real-guarded-ppo-horizon5-batch-expansion-summary.json

git diff --check
```

## Non-Goals

No formal PPO rollout, new PPO update, checkpoint publication, default-policy
replacement, network/action-space/default-A* change, gate relaxation,
Ackermann-feasible trajectory claim, policy performance claim, formal
training-ready claim, or IRIS/GCS/path-planner diagnostic training release is
made by this stage. Formal PPO preflight still requires a later expansion to at
least 512 trainable transitions plus multi-seed stability evidence.
