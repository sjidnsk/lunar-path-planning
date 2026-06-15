# Exploration Coverage Performance Evaluation v1

## Purpose

`Exploration Coverage Performance Evaluation v1` is Stage 3 after the real
coverage signal audit. It answers whether the selected guarded formal PPO
candidate improves multi-step exploration coverage versus teacher or
source/default baselines without making path cost, risk, fallback, or controlled
regression worse.

This stage is read-only. It does not run PPO, publish checkpoints, replace the
default policy, connect a real executor, or make a formal release claim.

## Inputs

- Formal training:
  `outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1/`
- Post-training stability replay:
  `outputs/path_feedback_batch_guarded_formal_ppo_post_training_stability_replay_v1/`
- Selected candidate promotion preflight:
  `outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_preflight_v1/`
- Selected candidate multihorizon shadow rollout:
  `outputs/path_feedback_batch_selected_formal_ppo_candidate_multihorizon_shadow_rollout_v1/`
- Exploration coverage signal audit:
  `outputs/path_feedback_batch_exploration_coverage_signal_audit_v1/`

The evaluator uses the Stage 2 `coverage-delta-audit.jsonl` as the authoritative
actual coverage source and enriches rows from shadow steps for path cost, risk,
energy cost, teacher action, policy activation, and fallback attribution.

## Required Checks

- Selected PPO candidate rows are present.
- At least one comparator is present or safely derived. A teacher comparator may
  only be derived when shadow rows are explicitly `policy_teacher_aligned` and
  selected and teacher actions are identical.
- Actual coverage is not replaced by expected coverage.
- Fallback/source gain is not counted as PPO policy gain.
- `controlled_regression_count=0`.
- Selected PPO improves `coverage_return` and
  `cumulative_coverage_rate_delta` over the best baseline.
- Selected PPO improves valuable coverage when valuable coverage evidence is
  available.
- Coverage gain per path cost, risk, and energy does not regress beyond the
  allowed tolerance.
- Fallback does not dominate total gain and selected fallback rate does not
  regress.

## Outputs

Implemented runner:

- `scripts/run_exploration_coverage_performance_evaluation.py`
- `scripts/run_exploration_coverage_performance_evaluation.sh`

Output root:

`outputs/path_feedback_batch_exploration_coverage_performance_evaluation_v1/`

Required files:

- `exploration-coverage-performance-evaluation-summary.json`
- `coverage-performance-metric-table.jsonl`
- `coverage-performance-comparison-audit.json`
- `coverage-performance-rejection-report.json`
- `exploration-coverage-performance-evaluation-report.md`

Documentation updates are required in:

- `README.md`
- `docs/算法设计与系统架构报告.md`
- `docs/superpowers/specs/2026-06-15-exploration-coverage-performance-evaluation.md`

## Acceptance

Passing requires:

- Summary `status=passed` and `coverage_performance_status=passed`.
- `reason_codes=[]`.
- Selected PPO has higher `coverage_return` and
  `cumulative_coverage_rate_delta` than the best teacher or source/default
  baseline.
- Valuable coverage and coverage efficiency do not regress.
- Path cost, risk, energy cost, fallback rate, and controlled regression remain
  within gates.
- `runs_new_ppo_update=false`, `publishes_checkpoint=false`,
  `replaces_default_policy=false`, and `formal_release_claimed=false`.

Failing must be explicit, with reason codes such as:

- `coverage_performance_not_improved`
- `valuable_coverage_not_improved`
- `coverage_efficiency_regression`
- `path_cost_regression`
- `risk_regression`
- `fallback_dominates_gain`
- `insufficient_comparator_evidence`
- `expected_actual_coverage_confusion`
- `fallback_coverage_gain_claimed_as_policy_gain`
- `controlled_regression_present`

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

$P -m pytest tests/test_exploration_coverage_performance_evaluation.py

PYTHON=$P bash scripts/run_exploration_coverage_performance_evaluation.sh \
  --formal-training-root outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1 \
  --post-training-replay-root outputs/path_feedback_batch_guarded_formal_ppo_post_training_stability_replay_v1 \
  --selected-candidate-root outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_preflight_v1 \
  --shadow-root outputs/path_feedback_batch_selected_formal_ppo_candidate_multihorizon_shadow_rollout_v1 \
  --coverage-signal-root outputs/path_feedback_batch_exploration_coverage_signal_audit_v1 \
  --output-root outputs/path_feedback_batch_exploration_coverage_performance_evaluation_v1

jq '{status,reason_codes,next_required_change,coverage_performance_status,controlled_regression_count}' \
  outputs/path_feedback_batch_exploration_coverage_performance_evaluation_v1/exploration-coverage-performance-evaluation-summary.json

git diff --check
```

## Current Result

The current repository state runs successfully but the performance evaluation
fails, as it should:

- `status=failed`
- `coverage_performance_status=failed`
- `reason_codes=["coverage_performance_not_improved",
  "valuable_coverage_not_improved"]`
- `selected_seed=0`
- `selected_budget=epochs1_lr3e-6`
- `best_baseline_actor=teacher`
- `coverage_return_improvement=0.0`
- `cumulative_coverage_rate_delta_improvement=0.0`
- `valuable_area_covered_improvement=0.0`
- `controlled_regression_count=0`
- `next_required_change=coverage_aware_reward_refinement`

The teacher baseline is not an independent rollout here. It is a conservative
action-equivalent comparator derived from `policy_teacher_aligned` shadow rows.
Selected PPO and teacher have the same coverage return and cumulative coverage
delta, so this stage cannot claim PPO performance improvement.

## Non-Goals

- Do not run a new PPO update.
- Do not expand training.
- Do not refine reward inside this evaluator.
- Do not publish or replace checkpoints/default policy.
- Do not modify network, action space, or default A*.
- Do not relax gates.
- Do not treat teacher agreement as exploration performance.
- Do not treat expected coverage as actual executed coverage.
- Do not count fallback/source gain as PPO policy improvement.
- Do not connect a real executor.
- Do not make a formal performance claim or release decision.
