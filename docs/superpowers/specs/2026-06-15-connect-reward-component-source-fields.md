# Connect Reward Component Source Fields v1

## Background

Stage 2 proved the actual exploration coverage signal is connected:
`outputs/path_feedback_batch_exploration_coverage_signal_audit_v1/` has
2,052 nonzero `coverage_rate_delta` rows from `path_feedback`, with no
expected/actual confusion, no fallback policy-gain contamination, and no
controlled regression.

Stage 3 remained a strict performance failure: the selected PPO candidate is
teacher-equivalent and has not improved coverage return or valuable coverage.
Stage 4 originally failed because `valuable_area_bonus`,
`information_gain_bonus`, and `risk_penalty` had no trustworthy source fields.

## Goal

Connect provenance-backed source fields for the three missing coverage-aware
reward components without running PPO, publishing checkpoints, replacing the
default policy, or claiming performance.

## Implementation

The connector is implemented by:

- `scripts/run_connect_reward_component_source_fields.py`
- `scripts/run_connect_reward_component_source_fields.sh`
- `tests/test_connect_reward_component_source_fields.py`

It reads:

- Stage 2 coverage signal output
- Stage 3 coverage performance output
- Stage 4 reward refinement output
- path-feedback summaries from the teacher-distillation and safe-better
  expansion batches

It writes:

- `outputs/path_feedback_batch_connect_reward_component_source_fields_v1/connect-reward-component-source-fields-summary.json`
- `outputs/path_feedback_batch_connect_reward_component_source_fields_v1/source-field-wiring-audit.json`
- `outputs/path_feedback_batch_connect_reward_component_source_fields_v1/component-provenance.jsonl`
- `outputs/path_feedback_batch_connect_reward_component_source_fields_v1/replay-validation.json`
- `outputs/path_feedback_batch_connect_reward_component_source_fields_v1/connect-reward-component-source-fields-rejection-report.json`
- `outputs/path_feedback_batch_connect_reward_component_source_fields_v1/connect-reward-component-source-fields-report.md`

The Stage 4 reward refinement script now consumes a matching connector overlay
when the connector summary roots match the current `coverage_signal_root` and
`coverage_performance_root`.

## Source Contract

- `valuable_area_bonus` uses
  `path_feedback.coverage_rate_delta*path_feedback.candidates.utility`.
- `information_gain_bonus` uses `path_feedback.coverage_rate_delta`.
- `risk_penalty` uses `path_feedback.candidates.risk`.

The connector rejects expected-only values, missing candidate provenance,
fallback policy-gain contamination, and controlled regression with explicit
reason codes.

## Current Evidence

Connector output:

- `status=passed`
- `reward_component_source_field_status=passed`
- `reason_codes=[]`
- `audited_row_count=2052`
- `connected_row_count=2052`
- `path_feedback_match_method_counts={"context":108,"scenario_action":1944}`
- `expected_actual_coverage_confusion_count=0`
- `fallback_coverage_gain_claimed_as_policy_gain_count=0`
- `controlled_regression_count=0`

Stage 4 rerun:

- `status=passed`
- `reward_refinement_status=passed`
- `reason_codes=[]`
- `source_field_missing_component_count=0`
- `component_source_overlay_row_count=2052`
- `next_required_change=coverage_driven_ppo_improvement_run`

The selected candidate remains teacher-equivalent:
`selected_candidate_performance_improved=false` and
`coverage_aware_reward_improvement=0.0`.

## Verification

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python
F=outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1
C=outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_preflight_v1
S=outputs/path_feedback_batch_exploration_coverage_signal_audit_v1
E=outputs/path_feedback_batch_exploration_coverage_performance_evaluation_v1
R=outputs/path_feedback_batch_coverage_aware_reward_refinement_v1
O=outputs/path_feedback_batch_connect_reward_component_source_fields_v1
$P -m pytest tests/test_connect_reward_component_source_fields.py tests/test_coverage_aware_reward_refinement.py
PYTHON=$P bash scripts/run_connect_reward_component_source_fields.sh --coverage-signal-root $S --coverage-performance-root $E --reward-refinement-root $R --output-root $O
PYTHON=$P bash scripts/run_coverage_aware_reward_refinement.sh --formal-training-root $F --selected-candidate-root $C --coverage-signal-root $S --coverage-performance-root $E --output-root $R
jq '{status,reason_codes,next_required_change,runs_new_ppo_update}' $O/connect-reward-component-source-fields-summary.json
git diff --check
```

## Non-Goals

This stage does not run PPO updates, expand training, publish or replace any
checkpoint/default policy, change the network/action space/default A*, loosen
guards, connect a real executor, or make a formal performance/release claim.
