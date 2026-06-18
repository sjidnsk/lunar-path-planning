# Xunce Stage 18F.1 Cost-Efficient Coverage Opportunity Refinement

## Summary

Stage 18E materialized candidate-level coverage opportunities and Stage 18D
confirmed that the enriched root is discriminative enough for dynamic rollout.
Stage 18F still blocked oracle separability because the cost-aware oracle gained
coverage with efficiency regressions. Stage 18F.1 refines the Stage 18E root so
candidate opportunities are not only higher coverage, but also non-regressive on
path cost, risk, and cost-adjusted coverage efficiency.

This is an offline counterfactual refinement stage. It does not train Xunce, run
PPO, publish checkpoints, replace the default policy, connect an executor, start
online canary, or download new maps.

## Runner

- Stage id: `xunce-cost-efficient-coverage-opportunity-refinement`
- Runner: `scripts/run_xunce_cost_efficient_coverage_opportunity_refinement.py`
- Config: `configs/xunce_cost_efficient_coverage_opportunity_refinement_v1.json`
- Output root:
  `outputs/path_feedback_batch_xunce_cost_efficient_coverage_opportunity_refinement_v1/`

## Candidate Fields

Each candidate in the refined root receives:

- `cost_efficiency_adjusted_coverage_delta`
- `risk_adjusted_coverage_delta`
- `budget_adjusted_coverage_delta`
- `coverage_cost_pareto_rank`
- `dominance_status`
- `safe_efficient_opportunity`
- `cost_efficiency_guard_passed`
- `risk_efficiency_guard_passed`
- `efficiency_refinement_source`

`safe_efficient_opportunity=true` requires reachable candidate, no open-grid
fallback, ROI-weighted coverage above incumbent, path cost no higher than
incumbent, risk no higher than incumbent, and cost-adjusted efficiency above
incumbent.

## Artifacts

- `xunce-cost-efficient-coverage-opportunity-summary.json`
- `xunce-cost-efficient-candidate-overlay.jsonl`
- `xunce-cost-efficient-spread-by-scenario.jsonl`
- `xunce-cost-efficient-roi-summary.json`
- `xunce-cost-efficient-decision-audit.json`
- `xunce-cost-efficient-coverage-opportunity-manifest.json`
- `xunce-cost-efficient-coverage-opportunity-report.md`
- `xunce-high-fidelity-real-map-roi-expansion-summary.json`
- `xunce-high-fidelity-real-map-slices.jsonl`
- `xunce-high-fidelity-path-feedback-audit.json`

## Stage 18F Compatibility

When Stage 18F detects `safe_efficient_opportunity`, the cost-aware oracle only
selects from safe-efficient candidates and prefers
`cost_efficiency_adjusted_coverage_delta` as its score. Greedy oracle still
selects by coverage for diagnostic contrast. If no safe-efficient candidates are
available, Stage 18F routes to `refine_cost_efficient_coverage_opportunity`.

## Decision Routes

- No Stage 18E/refined root: `run_candidate_level_coverage_opportunity_materialization`
- Missing efficiency inputs: `repair_candidate_materialization_inputs`
- No safe-efficient opportunity: `expand_roi_or_map_complexity`
- Safe-efficient opportunity exists but insufficient spread:
  `refine_cost_efficient_coverage_opportunity`
- Refined Stage 18F oracle separable:
  `run_stage18c_v2_with_refined_cost_efficient_root`
- Oracle separable but Xunce not better:
  `xunce_adapter_or_reward_iteration_required`

## Validation

```powershell
D:\conda_envs\lunar-explorer\python.exe -m pytest `
  tests\test_xunce_cost_efficient_coverage_opportunity_refinement.py `
  tests\test_xunce_oracle_separability_benchmark.py `
  tests\test_xunce_high_fidelity_exploration_coverage_comparison.py `
  tests\test_platform_stage_runner.py `
  -q

D:\conda_envs\lunar-explorer\python.exe scripts\run_stage.py `
  --stage xunce-cost-efficient-coverage-opportunity-refinement `
  --dry-run

D:\conda_envs\lunar-explorer\python.exe -m py_compile `
  scripts\run_xunce_cost_efficient_coverage_opportunity_refinement.py `
  scripts\run_xunce_oracle_separability_benchmark.py `
  scripts\run_stage.py
```

