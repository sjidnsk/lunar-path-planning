# Coverage-Aware Reward Refinement v1

## Purpose

`Coverage-Aware Reward Refinement v1` is Stage 4 after exploration coverage
performance evaluation. It audits and re-scores the current evidence with a
coverage-aware reward contract while staying read-only.

This stage does not run PPO, publish checkpoints, replace the default policy,
connect a real executor, or claim performance. It answers whether the project
has trustworthy fields for a reward that keeps teacher skill while rewarding
real multi-step exploration coverage gain.

## Inputs

- Formal PPO training:
  `outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1/`
- Post-training stability replay:
  `outputs/path_feedback_batch_guarded_formal_ppo_post_training_stability_replay_v1/`
- Selected candidate promotion preflight:
  `outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_preflight_v1/`
- Exploration coverage signal audit:
  `outputs/path_feedback_batch_exploration_coverage_signal_audit_v1/`
- Exploration coverage performance evaluation:
  `outputs/path_feedback_batch_exploration_coverage_performance_evaluation_v1/`

The runner joins Stage 2 `coverage-delta-audit.jsonl` with selected-candidate
shadow steps and Stage 3 comparison output.

## Reward Contract

```text
reward =
  coverage_gain_bonus
+ valuable_area_bonus
+ information_gain_bonus
+ teacher_skill_retention_bonus
- path_cost_penalty
- risk_penalty
- energy_penalty
- fallback_penalty
- controlled_regression_penalty
```

Required source rules:

- `coverage_gain_bonus` must use actual executed coverage delta from `map`,
  `sidecar`, or `path_feedback`, never expected coverage.
- `valuable_area_bonus` and `information_gain_bonus` must have real value or
  information fields; missing indicators and default zero placeholders cannot
  create positive reward.
- `path_cost_penalty`, `risk_penalty`, and `energy_penalty` must use real
  fields or non-missing candidate features.
- `fallback_penalty` must keep fallback/source gain separate from PPO policy
  gain.
- `controlled_regression_penalty` must block any controlled regression.
- `teacher_skill_retention_bonus` may reward teacher-aligned or safe
  disagreement evidence, but it is not a performance claim.

## Outputs

Implemented runner:

- `scripts/run_coverage_aware_reward_refinement.py`
- `scripts/run_coverage_aware_reward_refinement.sh`

Output root:

`outputs/path_feedback_batch_coverage_aware_reward_refinement_v1/`

Required files:

- `coverage-aware-reward-refinement-summary.json`
- `reward-component-audit.jsonl`
- `source-field-audit.json`
- `reward-rescore-comparison.json`
- `reward-refinement-rejection-report.json`
- `coverage-aware-reward-refinement-report.md`

Documentation updates are required in:

- `README.md`
- `docs/算法设计与系统架构报告.md`
- `docs/superpowers/specs/2026-06-15-coverage-aware-reward-refinement.md`

## Acceptance

Passing requires:

- Summary `status=passed` and `reward_refinement_status=passed`.
- `reason_codes=[]`.
- Every reward component has a trustworthy source field.
- `expected_actual_coverage_confusion_count=0`.
- `fallback_coverage_gain_claimed_as_policy_gain_count=0`.
- `controlled_regression_count=0`.
- `next_required_change=coverage_driven_ppo_improvement_run`.
- `runs_new_ppo_update=false`, `publishes_checkpoint=false`,
  `replaces_default_policy=false`, `performance_claimed=false`, and
  `formal_release_claimed=false`.

Failing must be explicit, with reason codes such as:

- `reward_component_source_missing`
- `valuable_coverage_signal_missing`
- `information_gain_signal_missing`
- `risk_signal_missing`
- `expected_actual_coverage_confusion`
- `fallback_policy_gain_contamination`
- `controlled_regression_present`

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

$P -m pytest tests/test_coverage_aware_reward_refinement.py

PYTHON=$P bash scripts/run_coverage_aware_reward_refinement.sh \
  --formal-training-root outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1 \
  --selected-candidate-root outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_preflight_v1 \
  --coverage-signal-root outputs/path_feedback_batch_exploration_coverage_signal_audit_v1 \
  --coverage-performance-root outputs/path_feedback_batch_exploration_coverage_performance_evaluation_v1 \
  --output-root outputs/path_feedback_batch_coverage_aware_reward_refinement_v1

jq '{status,reason_codes,next_required_change,reward_refinement_status,runs_new_ppo_update}' \
  outputs/path_feedback_batch_coverage_aware_reward_refinement_v1/coverage-aware-reward-refinement-summary.json

git diff --check
```

## Current Result

The current repository state has passed the reward refinement gate after
`Connect Reward Component Source Fields v1` supplied the provenance overlay:

- `status=passed`
- `reward_refinement_status=passed`
- `reason_codes=[]`
- `source_field_missing_component_count=0`
- `component_source_overlay_row_count=2052`
- `next_required_change=coverage_driven_ppo_improvement_run`
- `selected_teacher_equivalent=true`
- `selected_candidate_performance_improved=false`
- `coverage_aware_reward_improvement=0.0`
- `expected_actual_coverage_confusion_count=0`
- `fallback_coverage_gain_claimed_as_policy_gain_count=0`
- `controlled_regression_count=0`

This means the reward contract can now trust the connected source fields, but
the current selected PPO remains teacher-equivalent and cannot be claimed as a
performance improvement. The next proof must come from a coverage-driven PPO
improvement run, not from this read-only reward-contract pass.

## Non-Goals

- Do not run a new PPO update.
- Do not expand training.
- Do not publish or replace checkpoints/default policy.
- Do not modify network, action space, or default A*.
- Do not relax guards.
- Do not treat teacher agreement as exploration performance.
- Do not treat expected coverage as actual executed coverage.
- Do not count fallback/source gain as PPO policy improvement.
- Do not connect a real executor.
- Do not make a formal performance claim or release decision.
