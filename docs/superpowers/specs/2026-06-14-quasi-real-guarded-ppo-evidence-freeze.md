# Quasi-Real Guarded PPO Evidence Freeze v1

## Summary

This stage freezes the passed `Quasi-Real Guarded PPO Rollout Pilot v1`
evidence into a reproducible audit package. It is a closure and provenance
stage only: it may refresh the guarded pilot to current provenance before
freezing, but it does not run a new PPO update, publish a checkpoint, replace
the default policy, relax gates, or claim policy performance.

The freeze exists because the authoritative readiness result must come from an
explicit validate-only run against the quasi-real pilot summary. A previously
written `policy-training-readiness-review-summary.json` under the batch root can
be stale after the worktree changes, so the freeze records stale written
readiness as diagnostic evidence instead of letting it override the explicit
validate-only result.

## Artifacts

- `configs/quasi_real_guarded_ppo_evidence_freeze_v1.json`
- `scripts/run_quasi_real_guarded_ppo_evidence_freeze.py`
- `scripts/run_quasi_real_guarded_ppo_evidence_freeze.sh`
- `scripts/run_quasi_real_guarded_ppo_evidence_freeze_closure.sh`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_evidence_freeze_v1/quasi-real-guarded-ppo-evidence-freeze-summary.json`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_evidence_freeze_v1/quasi-real-guarded-ppo-evidence-manifest.json`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_evidence_freeze_v1/quasi-real-guarded-ppo-readiness-validate-only.json`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_evidence_freeze_v1/quasi-real-guarded-ppo-evidence-freeze-report.md`

## Inputs

The freeze reads:

- `outputs/path_feedback_batch_quasi_real_guarded_ppo_rollout_pilot_v1/quasi-real-guarded-ppo-rollout-pilot-summary.json`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_rollout_pilot_v1/quasi-real-guarded-ppo-rollout-episodes.jsonl`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_rollout_pilot_v1/quasi-real-guarded-ppo-rollout-steps.jsonl`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_rollout_pilot_v1/quasi-real-guarded-ppo-rollout-rejection-report.json`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_rollout_pilot_v1/quasi-real-guarded-ppo-rollout-reward-audit.json`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_rollout_pilot_v1/quasi_real_collector_replay/ppo-rollout-collector-summary.json`
- `outputs/path_feedback_batch_return_aligned_guarded_ppo_update_smoke_v1/post_update_long_horizon/long-horizon-teacher-skill-contract-summary.json`
- `outputs/path_feedback_batch_return_aligned_guarded_ppo_update_smoke_v1/return-aligned-guarded-ppo-update-smoke-summary.json`

## Contract

The freeze verifies that the quasi-real pilot is still passed, has 36 episodes,
108 steps, 36 PPO-trainable train transitions, 72 diagnostic validation/test
transitions, zero controlled regressions, and teacher agreement at least 0.9.
It checks that the collector replay is passed, the long-horizon verdict is
`long_horizon_teacher_skill_contract_aligned`, and the return-aligned update
smoke summary is passed while still marking all publication/performance/formal
training flags false.

It then runs readiness validate-only with:

```bash
scripts/run_policy_training_readiness_review.sh \
  --batch-root outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1 \
  --config configs/policy_training_readiness_review_v1.json \
  --quasi-real-guarded-ppo-rollout-pilot-summary \
    outputs/path_feedback_batch_quasi_real_guarded_ppo_rollout_pilot_v1/quasi-real-guarded-ppo-rollout-pilot-summary.json \
  --validate-only
```

The expected readiness status is
`quasi_real_guarded_ppo_rollout_pilot_evaluated` with empty blockers and reason
codes.

## Current Evidence

The current closure writes a passed freeze summary with:

- `status=passed`
- `reason_codes=[]`
- `pilot_status=passed`
- `pilot_episode_count=36`
- `pilot_step_count=108`
- `pilot_ppo_trainable_transition_count=36`
- `pilot_diagnostic_transition_count=72`
- `pilot_controlled_regression_count=0`
- `pilot_teacher_agreement_rate=1.0`
- `collector_replay_status=passed`
- `collector_replay_trainable_transition_count=36`
- `long_horizon_verdict=long_horizon_teacher_skill_contract_aligned`
- `readiness_status=quasi_real_guarded_ppo_rollout_pilot_evaluated`
- `stale_written_readiness_summary_detected=true`
- `runs_ppo_update=false`
- `publishes_checkpoint=false`
- `replaces_default_policy=false`
- `performance_claimed=false`
- `formal_training_ready_claimed=false`

The manifest covers nine required artifacts with sha256 digests and records the
optional stale written readiness summary separately.

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_quasi_real_guarded_ppo_evidence_freeze.py \
  tests/test_quasi_real_guarded_ppo_rollout_pilot.py \
  tests/test_policy_training_readiness_review.py

PYTHON=$P bash scripts/run_quasi_real_guarded_ppo_evidence_freeze_closure.sh

jq '{status, reason_codes, pilot_status, readiness_status, stale_written_readiness_summary_detected}' \
  outputs/path_feedback_batch_quasi_real_guarded_ppo_evidence_freeze_v1/quasi-real-guarded-ppo-evidence-freeze-summary.json

git diff --check
```

## Non-Goals

No new PPO update, formal rollout, checkpoint publication, default-policy
replacement, network/action-space/default-A* change, gate relaxation,
Ackermann-feasible trajectory claim, policy performance claim, formal training
ready claim, or IRIS/GCS/path-planner diagnostic training release is made by
this stage.
