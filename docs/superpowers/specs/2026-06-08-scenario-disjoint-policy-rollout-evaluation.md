# Controlled Scenario-Disjoint Policy Rollout Evaluation v1

## Summary

当前严格泛化门禁已到 `scenario_disjoint_policy_candidate_evaluated`。本阶段新增受控 shadow rollout evaluation，
验证 local experimental checkpoint 若进入候选选择闭环，是否仍满足 action mask、contract、安全、fallback、
path/risk/source-selection 全部无退化。

## Interfaces

- Config: `configs/scenario_disjoint_policy_rollout_evaluation_v1.json`
- CLI: `scripts/run_scenario_disjoint_policy_rollout_evaluation.py`
- Wrapper: `scripts/run_scenario_disjoint_policy_rollout_evaluation.sh`
- HOLD outputs:
  - `scenario-disjoint-policy-rollout-decisions.jsonl`
  - `scenario-disjoint-policy-rollout-regression-report.json`
  - `scenario-disjoint-policy-rollout-evaluation-summary.json`

The evaluator consumes SRC batch/readiness evidence, CAND checkpoint plus metadata, and HOLD batch/fresh/path-feedback artifacts.

## Behavior

Default mode is `shadow_mode=true` and `controlled_selection_mode=false`.

For each HOLD scenario, the evaluator scores path-feedback candidates with the experimental policy network. It records the raw
policy top candidate, then applies controlled gates before recording the shadow decision. A different candidate is acceptable only
when it is in the action mask, contract-safe, has no fallback/open-grid or safety regression, and does not regress path cost, risk,
or source-selection quality. Otherwise the controlled shadow decision stays with the source-selected candidate while raw regression
is reported diagnostically.

## Readiness Gate

`scripts/run_policy_training_readiness_review.py` can consume the rollout summary. Passing evidence advances readiness to
`scenario_disjoint_policy_rollout_evaluated` only when rollout summary is passed, reason codes are empty, context and decision counts
are positive, invalid action mask and regression counts are 0, and fallback/open-grid, safety, contract, path/risk, and
source-selection regression counts are all 0.

## Non-Goals

This stage does not start formal PPO rollout, publish a checkpoint, replace the default policy, modify network/action space/default A*,
relax the default distance contract, claim Ackermann-feasible trajectory, treat IRIS/GCS/path-planner diagnostics as training release
evidence, or claim policy performance improvement.
