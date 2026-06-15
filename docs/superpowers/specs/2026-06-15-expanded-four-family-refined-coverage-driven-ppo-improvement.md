# Expanded Four-Family Refined Coverage-Driven PPO Improvement v1

## Goal

Use the Stage 5B.3 four-family expanded safe-better set to rerun refined
coverage-driven PPO, and verify whether the policy can change action ranking
and improve multi-step exploration coverage under the existing guard.

## Inputs

- Safe-better root:
  `outputs/path_feedback_batch_safe_better_pair_expansion_across_families_v1/`
- Required source files:
  - `expanded-safe-better-pairs.jsonl`
  - `expanded-counterfactual-coverage-rollouts.jsonl`
  - `safe-better-pair-expansion-summary.json`
- The runner must not default back to the old Stage 5A.2 51-pair source.
- Trainable rows are restricted to `split=train`, `ppo_trainable=true`,
  source-backed, fallback-free, guard-clean, and coverage-advantage-positive
  candidates.

## Implementation

Implemented by:

- `scripts/run_expanded_four_family_refined_coverage_driven_ppo_improvement.py`
- `scripts/run_expanded_four_family_refined_coverage_driven_ppo_improvement.sh`

Output root:

- `outputs/path_feedback_batch_expanded_four_family_refined_coverage_driven_ppo_improvement_v1/`

The runner adapts the 5B.3 candidate-only pair rows into a Stage 5A.2-compatible
candidate coverage overlay and counterfactual rollout set. For low-observation
train decisions that do not exist in the old coverage-aware PPO batch, it
synthesizes compatible teacher old-policy transitions and recomputes old
`log_prob` / `value` from the selected base checkpoint. The PPO update is still
guarded and offline; it uses transition-info `ppo_advantage` / `ppo_return` and
writes only an experimental checkpoint for replay.

## Current Result

Current summary:

`outputs/path_feedback_batch_expanded_four_family_refined_coverage_driven_ppo_improvement_v1/expanded-four-family-refined-coverage-driven-ppo-improvement-summary.json`

Key fields:

```json
{
  "status": "failed",
  "reason_codes": ["coverage_efficiency_regression"],
  "next_required_change": "tune_cost_efficiency_aware_reward_or_candidate_filter",
  "safe_better_training_pair_count": 723,
  "safe_better_training_family_count": 4,
  "counterfactual_advantage_nonzero_count": 723,
  "policy_argmax_changed_count": 379,
  "coverage_return_improvement": 4.178973625356,
  "cumulative_coverage_rate_delta_improvement": 0.936713498108,
  "valuable_area_covered_improvement": 49.499451984438,
  "fallback_rate": 0.283540802213,
  "controlled_regression_count": 0,
  "fallback_gain_contamination_count": 0,
  "performance_claimed": false
}
```

This is meaningful progress, but not a performance pass. The expanded four
family signal changes policy ranking and improves coverage return, but it does
so with worse path-cost efficiency than the baseline. The correct next stage is
cost/efficiency-aware reward or candidate filtering, not shadow/canary.

## Validation

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
$PY -m pytest tests/test_expanded_four_family_refined_coverage_driven_ppo_improvement.py tests/test_refined_coverage_driven_ppo_improvement_run.py tests/test_safe_better_pair_expansion_across_families.py -q
PYTHON=$PY bash scripts/run_expanded_four_family_refined_coverage_driven_ppo_improvement.sh
jq '{status,reason_codes,safe_better_training_pair_count,safe_better_training_family_count,policy_argmax_changed_count,coverage_return_improvement,cumulative_coverage_rate_delta_improvement,fallback_rate,performance_claimed}' outputs/path_feedback_batch_expanded_four_family_refined_coverage_driven_ppo_improvement_v1/expanded-four-family-refined-coverage-driven-ppo-improvement-summary.json
git diff --check
```

## Non-Goals

- Do not publish a checkpoint.
- Do not replace the default policy.
- Do not connect a real executor.
- Do not modify network/action space/default A*.
- Do not relax the guard.
- Do not claim Ackermann-feasible trajectory.
- Do not treat IRIS/GCS/path-planner diagnostics as training release evidence.
- Do not claim model performance improvement while `status=failed`.
