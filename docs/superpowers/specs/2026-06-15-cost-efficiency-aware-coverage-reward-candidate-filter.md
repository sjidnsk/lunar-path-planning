# Cost-Efficiency-Aware Coverage Reward / Candidate Filter v1

## Background

Stage 5B.6 proved that expanded four-family refined PPO can change policy
argmax and improve coverage return, but it failed
`coverage_efficiency_regression`. The next bridge was therefore to keep the
real Stage 5B.3 four-family source-backed pairs while adding path/risk/energy
efficiency into candidate filtering and PPO advantage.

## Implementation

- Entry points:
  - `scripts/run_cost_efficiency_aware_coverage_reward_candidate_filter.py`
  - `scripts/run_cost_efficiency_aware_coverage_reward_candidate_filter.sh`
- Output root:
  - `outputs/path_feedback_batch_cost_efficiency_aware_coverage_reward_candidate_filter_v1/`
- Inputs:
  - Stage 5B.3 `expanded-safe-better-pairs.jsonl`
  - Stage 5B.3 `expanded-counterfactual-coverage-rollouts.jsonl`
  - Stage 5B.6 summary, replay audit, metric table, and refined transitions

The runner writes an efficiency audit, a filtered trainable batch, a
cost-efficiency advantage audit, compatible Stage 5A.2 inputs, and a comparable
performance input table. The comparable table evaluates teacher/pre-improvement
behavior on the same filtered decision set as the post-update replay.

## Reward And Filter Contract

Trainable rows must be train split, source-backed, fallback-free, guard-clean,
coverage-advantage-positive, and cost-efficiency-advantage-positive. High
coverage with high cost is downweighted through `cost_efficiency_weight` and
`cost_efficiency_ppo_advantage`; low-gain high-cost rows become
diagnostic-only. The PPO runner consumes `cost_efficiency_ppo_advantage` through
the existing transition-info advantage path.

## Current Result

Summary:
`outputs/path_feedback_batch_cost_efficiency_aware_coverage_reward_candidate_filter_v1/cost-efficiency-aware-coverage-reward-candidate-filter-summary.json`

Result:

- `status=passed`
- `reason_codes=[]`
- `next_required_change=shadow_canary_release_performance_validation_preflight`
- `trainable_pair_count=621`
- `safe_better_training_pair_count=621`
- `safe_better_training_family_count=4`
- family counts: `low_observation_count=9`, `mixed_risk=231`, `rim_or_steep_slope=186`, `smooth_high_confidence=195`
- `counterfactual_advantage_nonzero_count=621`
- `policy_argmax_changed_count=263`
- `coverage_return_improvement=54.96083408637`
- `cumulative_coverage_rate_delta_improvement=56.92136202257`
- `valuable_area_covered_improvement=25.987354935654`
- `coverage_efficiency_regression=false`
- `fallback_rate=0.220611916264`
- `controlled_regression_count=0`
- `fallback_gain_contamination_count=0`

## Scope Guards

- `publishes_checkpoint=false`
- `replaces_default_policy=false`
- `performance_claimed=false`
- Does not connect a real executor.
- Does not modify network/action space/default A*.
- Does not relax guard.
- Does not claim Ackermann-feasible trajectory.
- Does not treat IRIS/GCS/path-planner diagnostics as training release proof.

## Verification

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
$PY -m pytest tests/test_cost_efficiency_aware_coverage_reward_candidate_filter.py \
  tests/test_expanded_four_family_refined_coverage_driven_ppo_improvement.py \
  tests/test_safe_better_pair_expansion_across_families.py -q
PYTHON=$PY bash scripts/run_cost_efficiency_aware_coverage_reward_candidate_filter.sh
jq '{status,reason_codes,next_required_change,coverage_return_improvement,cumulative_coverage_rate_delta_improvement,valuable_area_covered_improvement,coverage_efficiency_regression,fallback_rate,performance_claimed}' \
  outputs/path_feedback_batch_cost_efficiency_aware_coverage_reward_candidate_filter_v1/cost-efficiency-aware-coverage-reward-candidate-filter-summary.json
git diff --check
```
