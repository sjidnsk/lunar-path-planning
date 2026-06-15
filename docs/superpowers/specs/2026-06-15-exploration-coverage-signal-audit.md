# Exploration Coverage Signal Audit v1

## Purpose

`Exploration Coverage Signal Audit v1` is the signal gate after guarded formal
PPO training and selected-candidate stabilization. Its job is to verify that
exploration coverage telemetry is real, non-default, attributable, and usable
before any coverage-performance claim or coverage-aware PPO reward training.

This stage does not run PPO, publish checkpoints, replace the default policy, or
claim policy performance. It answers one question: is the coverage instrument
connected well enough to score exploration?

## Current Evidence Boundary

Current formal PPO evidence proves training stability:

- `outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1/formal-ppo-training-run-summary.json`
  has `status=passed`, 5 passed seeds, 684 optimizer train transitions,
  `teacher_agreement_rate=1.0`, and zero controlled regression.
- Existing selected-candidate promotion evidence selects seed `0` from
  `epochs1_lr3e-6`, checkpoint SHA-256
  `9d9539c685ab965739c91958bf9cbfe90329c460b4bdbcc35881875aa62f0aa2`.

That evidence does not prove coverage performance. Before
`Connect Real Exploration Coverage Signal v1`, sampled rollout episodes still
showed `coverage_rate_delta=0.0` and `cumulative_coverage_rate_delta=0.0`.
The current audit now proves the actual coverage signal is connected well enough
to score exploration, but it still does not prove that the PPO candidate improves
coverage versus teacher or source/default baselines.

## Inputs

- Formal training run summary:
  `outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1/formal-ppo-training-run-summary.json`
- Formal training seed summaries:
  `outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1/formal-ppo-training-run-seed-summaries.jsonl`
- Candidate selection or promotion manifest when present:
  `outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_preflight_v1/promotion-candidate-manifest.json`
- Candidate shadow steps when present:
  `outputs/path_feedback_batch_selected_formal_ppo_candidate_multihorizon_shadow_rollout_v1/multihorizon-shadow-rollout-steps.jsonl`
- Upstream batch, path-feedback, sidecar, and rollout artifacts referenced by
  those summaries.

## Required Checks

- `initial_coverage_rate` exists where the rollout claims a coverage transition.
- `final_coverage_rate` exists where the rollout claims a coverage transition.
- `coverage_rate_delta` is present and finite.
- At least one audited policy/source/teacher execution path has non-zero actual
  coverage gain, unless the stage fails explicitly.
- `cumulative_coverage_rate_delta` equals the audited step-level deltas.
- `expected_coverage_rate_delta` is separated from actual post-execution
  coverage gain.
- Multi-step episodes update exploration state rather than replaying static
  observations.
- Coverage gain is attributable by actor: selected PPO candidate, teacher,
  source/default policy, and fallback.
- Coverage gain is sourced from real map, sidecar, or path-feedback artifacts,
  not missing/default zero fields.
- Fallback coverage gain is reported separately and cannot be counted as PPO
  policy improvement.

## Outputs

Implemented runner:

- `scripts/run_exploration_coverage_signal_audit.py`
- `scripts/run_exploration_coverage_signal_audit.sh`

Output root:

`outputs/path_feedback_batch_exploration_coverage_signal_audit_v1/`

Required files:

- `exploration-coverage-signal-audit-summary.json`
- `coverage-field-presence-audit.json`
- `coverage-delta-audit.jsonl`
- `coverage-attribution-audit.json`
- `coverage-state-transition-audit.json`
- `coverage-signal-rejection-report.json`
- `exploration-coverage-signal-audit-report.md`

Documentation updates are required in:

- `README.md`
- `docs/算法设计与系统架构报告.md`
- `docs/superpowers/specs/2026-06-15-exploration-coverage-signal-audit.md`

## Acceptance

- Summary `status=passed`, `reason_codes=[]`.
- `coverage_signal_status=passed`.
- `coverage_field_presence_status=passed`.
- `actual_coverage_gain_source` is one of `map`, `sidecar`, or
  `path_feedback`, never `default_zero`.
- `nonzero_actual_coverage_delta_count>0`.
- `coverage_delta_default_zero_count=0` for trainable policy evaluation rows.
- `expected_actual_coverage_confusion_count=0`.
- `multi_step_state_update_verified=true`.
- `policy_source_fallback_coverage_attribution_status=passed`.
- `fallback_coverage_gain_claimed_as_policy_gain_count=0`.
- `controlled_regression_count=0`.
- `publishes_checkpoint=false`, `replaces_default_policy=false`.
- `runs_new_ppo_update=false`.
- `performance_claimed=false`, `formal_training_ready_claimed=false`.

If the audit cannot prove real coverage signal, it must fail with:

`insufficient_exploration_coverage_signal`

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

$P -m pytest tests/test_exploration_coverage_signal_audit.py

PYTHON=$P bash scripts/run_exploration_coverage_signal_audit.sh \
  --formal-training-root outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1 \
  --selected-candidate-root outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_preflight_v1 \
  --shadow-root outputs/path_feedback_batch_selected_formal_ppo_candidate_multihorizon_shadow_rollout_v1 \
  --output-root outputs/path_feedback_batch_exploration_coverage_signal_audit_v1

jq '{status,reason_codes,coverage_signal_status,next_required_change}' \
  outputs/path_feedback_batch_exploration_coverage_signal_audit_v1/exploration-coverage-signal-audit-summary.json

git diff --check
```

The current audited repository state now passes this signal gate after
`Connect Real Exploration Coverage Signal v1`:

- `status=passed`
- `coverage_signal_status=passed`
- `coverage_field_presence_status=passed`
- `actual_coverage_gain_source=path_feedback`
- `nonzero_actual_coverage_delta_count=2052`
- `coverage_delta_default_zero_count=0`
- `expected_actual_coverage_confusion_count=0`
- `multi_step_state_update_verified=true`
- `fallback_coverage_gain_claimed_as_policy_gain_count=0`
- `controlled_regression_count=0`
- `next_required_change=run_exploration_coverage_performance_evaluation`

Implementation notes:

- `scripts/run_quasi_real_trainable_context_expansion.py` carries real
  path-feedback coverage deltas into materialized trainable rows and rolls them
  into step-level initial/final/cumulative coverage state.
- `scripts/run_quasi_real_guarded_ppo_scale512_multiseed_preflight.py`
  preserves those fields into PPO collector rollout episodes and episode
  metrics.
- `scripts/run_exploration_coverage_signal_audit.py` can read legacy shadow
  artifacts by backfilling missing coverage fields from existing path-feedback
  summaries, then using `shadow_episode_id` / `shadow_step_index` for continuous
  multi-step state validation.

This is not a PPO performance result. It only proves that Stage 2 has a usable
actual coverage signal. Stage 3 must still run a coverage-performance evaluation
before any PPO performance improvement, coverage-aware reward refinement, or
release claim.

## Non-Goals

- Do not run a new PPO update.
- Do not publish or replace any checkpoint/default policy.
- Do not modify network, action space, or default A*.
- Do not relax distance/path-risk/source-selection gates.
- Do not treat teacher agreement as coverage performance.
- Do not treat expected coverage as actual executed coverage.
- Do not count fallback/source coverage as PPO policy improvement.
- Do not claim Ackermann-feasible trajectory output.
- Do not treat IRIS/GCS/path-planner diagnostics as training release evidence.
- Do not claim policy performance, formal training readiness, or deployability.
