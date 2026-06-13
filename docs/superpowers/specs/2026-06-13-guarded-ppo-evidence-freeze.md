# Guarded PPO Evidence Freeze & Reproducible Closure v1

## Summary

This stage freezes the already-passed guarded PPO rollout pilot evidence into a
single reproducible package. It does not run a new PPO update, change gates, or
advance formal training claims. It records which summaries were trusted, which
artifacts existed, their hashes, and why the final readiness status is
`guarded_ppo_rollout_pilot_evaluated`.

## Artifacts

- `configs/guarded_ppo_evidence_freeze_v1.json`
- `scripts/run_guarded_ppo_evidence_freeze.py/.sh`
- `outputs/path_feedback_batch_guarded_ppo_evidence_freeze_v1/guarded-ppo-evidence-freeze-summary.json`
- `outputs/path_feedback_batch_guarded_ppo_evidence_freeze_v1/evidence-manifest.json`
- `outputs/path_feedback_batch_guarded_ppo_evidence_freeze_v1/readiness-final.json`
- `outputs/path_feedback_batch_guarded_ppo_evidence_freeze_v1/progress-consistency-report.json`
- `outputs/path_feedback_batch_guarded_ppo_evidence_freeze_v1/reproducibility-report.md`

## Contract

The freeze reads:

- `outputs/path_feedback_batch_guarded_ppo_rollout_pilot_v1/guarded-ppo-rollout-pilot-summary.json`
- `outputs/path_feedback_batch_guarded_ppo_rollout_pilot_v1/training-progress-summary.json`
- `outputs/path_feedback_batch_guarded_ppo_rollout_pilot_v1/training-progress-events.jsonl`

It then performs an explicit readiness validate-only run with
`--guarded-ppo-rollout-pilot-summary` and writes that result to
`readiness-final.json`. Any stale readiness file under the guarded clean source
root is compared against the explicit result and reported as diagnostic drift.

## Passing Criteria

- guarded pilot summary status is `passed` and reason codes are empty
- progress summary status is `passed`, failed stage count is 0, and progress
  events are present
- final readiness is `guarded_ppo_rollout_pilot_evaluated`
- final readiness blockers and reason codes are empty
- every required manifest artifact exists and has a sha256 digest
- stale readiness drift, if present, is reported but does not override the
  explicit guarded-summary readiness result

## Non-Goals

No PPO update, batch expansion, formal training, checkpoint publication, default
policy replacement, network/action-space/default-A* change, gate relaxation,
Ackermann-feasible trajectory claim, performance claim, or IRIS/GCS diagnostic
training release is made by this stage.
