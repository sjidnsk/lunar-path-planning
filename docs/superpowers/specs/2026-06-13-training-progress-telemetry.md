# Training Progress Telemetry & Progress Bar v1

## Summary

`Training Progress Telemetry & Progress Bar v1` adds observability to long
training and closure runs after `Quasi-Real Guarded PPO Rollout Pilot v1`.
It does not change PPO math, rewards, gates, checkpoint behavior, or readiness
semantics. It only records where the run is, what stage is active, and which
artifact should be inspected if the run fails.

## Scope

New artifacts:

- `scripts/training_progress.py`
- `configs/training_progress_telemetry_v1.json`
- `training-progress-events.jsonl`
- `training-progress-summary.json`

Updated entry points:

- `scripts/run_guarded_ppo_rollout_pilot_closure.sh`
- `scripts/run_guarded_ppo_rollout_pilot.py`
- `scripts/run_quasi_real_iterative_ppo_mini_loop_stability.py`
- `scripts/run_limited_ppo_update_smoke.py`
- `scripts/run_policy_gated_sequential_canary_rollout.py`

## Contract

Each supported entry point accepts `--progress auto|plain|jsonl|off`.
Plain progress renders to stderr. Existing JSON stdout remains machine-readable
and compatible with `jq` and closure automation. Structured events are written
as JSONL under the active output root.

Event records use `training-progress-event/v1` and include:

- `run_id`
- `stage`
- `status`
- `current` / `total`
- `round_index`
- `step_index`
- `message`
- `elapsed_seconds`
- `output_root`
- `summary_path`
- `reason_codes`
- `metrics`

The summary uses `training-progress-summary/v1` and reports the final run
status, event count, stage count, failed stage count, last stage/status,
last reason codes, readiness status, and recommended debug artifact.

## Stage Coverage

The progress stream covers:

- guarded closure start/finalize
- iterative precondition
- iterative round start/pass/fail
- quasi-real teacher-following
- quasi-real collector
- PPO update
- generated sequential compatibility/accounting/long-horizon checks
- guarded pilot stages
- generated sequential step progress, such as `sequential step 2/3`
- readiness validation

PPO events record optimizer transition count, epoch, loss, approx KL, grad norm,
parameter L2 delta, and finite/non-finite counters when available.

## Acceptance

The guarded pilot closure must generate:

- `outputs/path_feedback_batch_guarded_ppo_rollout_pilot_v1/training-progress-events.jsonl`
- `outputs/path_feedback_batch_guarded_ppo_rollout_pilot_v1/training-progress-summary.json`

The final progress summary should report `status=passed`,
`failed_stage_count=0`, and
`readiness_status=guarded_ppo_rollout_pilot_evaluated` when the underlying
guarded pilot closure passes.

`--progress off` must preserve old behavior by writing no progress artifacts.

## Non-Goals

This stage does not expand the PPO batch, start formal training, publish a
checkpoint, replace the default policy, change network/action space/default A*,
relax safety/contract/path-risk/source-selection gates, claim
Ackermann-feasible trajectory output, treat IRIS/GCS/path-planner diagnostics as
training release evidence, or change readiness semantics.
