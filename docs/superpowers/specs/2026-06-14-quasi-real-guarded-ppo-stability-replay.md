# Quasi-Real Guarded PPO Stability Replay & Acceptance Contract Refinement v1

## Summary

This stage starts from the passed `Quasi-Real Guarded PPO Evidence Freeze v1`
bundle and replays the quasi-real guarded PPO rollout pilot three times. It does
not run a new PPO update or change model parameters. Its purpose is to prove the
guarded rollout evidence is reproducible and to make the pre-formal-training
acceptance contract explicit.

## Artifacts

- `configs/quasi_real_guarded_ppo_stability_replay_v1.json`
- `scripts/run_quasi_real_guarded_ppo_stability_replay.py`
- `scripts/run_quasi_real_guarded_ppo_stability_replay.sh`
- `scripts/run_quasi_real_guarded_ppo_stability_replay_closure.sh`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_stability_replay_v1/quasi-real-guarded-ppo-stability-replay-summary.json`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_stability_replay_v1/stability-replay-comparison.jsonl`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_stability_replay_v1/acceptance-contract-refinement.json`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_stability_replay_v1/stability-replay-progress-events.jsonl`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_stability_replay_v1/quasi-real-guarded-ppo-stability-readiness-validate-only.json`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_stability_replay_v1/stability-replay-report.md`

## Contract

The runner reads the freeze summary and manifest from
`outputs/path_feedback_batch_quasi_real_guarded_ppo_evidence_freeze_v1/`,
verifies that the frozen evidence is passed, then replays the guarded PPO pilot
to `replay-00/`, `replay-01/`, and `replay-02/`. Each replay must pass with 36
episodes, 108 steps, 36 PPO-trainable transitions, 72 diagnostic transitions,
zero validation/test/source-fallback trainable leakage, zero missing
observation/log-prob/value, finite reward/return/advantage, zero controlled
safety/contract/path-risk/source-selection regression, and teacher agreement at
least 0.9.

The comparison file records whether each replay matches the frozen baseline.
Step signatures compare episode id, step index, context/scenario id, split,
controlled source, trainable flag, action indexes, gate reason codes, controlled
regression reasons, reward, discounted return, and advantage. Allowed
differences are limited to paths, generated timestamps, commands, returncode
metadata, and git provenance fingerprints.

The acceptance contract separates hard gates from diagnostic-only evidence.
Validation/test, source fallback, teacher fallback, raw policy probe rejection,
non-empty gate reasons, and IRIS/GCS/path-planner diagnostics remain
diagnostic-only and cannot silently become trainable release evidence.

## Current Evidence

The current closure writes a passed summary with:

- `status=passed`
- `reason_codes=[]`
- `replay_count=3`
- `passed_replay_count=3`
- `readiness_status=quasi_real_guarded_ppo_stability_replay_evaluated`
- `episode_count=36`
- `step_count=108`
- `ppo_trainable_transition_count=36`
- `diagnostic_transition_count=72`
- `controlled_regression_count=0`
- `teacher_agreement_rate=1.0`
- `baseline_replay_behavior_drift_count=0`
- `quasi_real_collector_replay_status=passed`
- `long_horizon_verdict=long_horizon_teacher_skill_contract_aligned`
- `acceptance_contract_refined=true`
- `runs_ppo_update=false`
- `publishes_checkpoint=false`
- `replaces_default_policy=false`
- `performance_claimed=false`
- `formal_training_ready_claimed=false`

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_quasi_real_guarded_ppo_stability_replay.py \
  tests/test_quasi_real_guarded_ppo_evidence_freeze.py \
  tests/test_quasi_real_guarded_ppo_rollout_pilot.py \
  tests/test_policy_training_readiness_review.py

PYTHON=$P bash scripts/run_quasi_real_guarded_ppo_stability_replay_closure.sh

jq '{status, reason_codes, replay_count, passed_replay_count, readiness_status, controlled_regression_count}' \
  outputs/path_feedback_batch_quasi_real_guarded_ppo_stability_replay_v1/quasi-real-guarded-ppo-stability-replay-summary.json

git diff --check
```

## Non-Goals

No new PPO update, batch expansion, formal rollout, checkpoint publication,
default-policy replacement, network/action-space/default-A* change, gate
relaxation, Ackermann-feasible trajectory claim, policy performance claim,
formal training-ready claim, or IRIS/GCS/path-planner diagnostic training
release is made by this stage.
