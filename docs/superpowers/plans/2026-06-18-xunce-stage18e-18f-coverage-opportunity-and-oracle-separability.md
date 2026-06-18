# Xunce Stage 18E/18F Coverage Opportunity and Oracle Separability

## Goal

Stage 18D showed that candidate coverage spread was insufficient, so Stage 18C
could not distinguish better exploration from different routing. Stage 18E
materializes candidate-level coverage opportunities. Stage 18F then checks
whether oracle baselines can exploit those opportunities before further Xunce
model iteration.

## Stage 18E

Runner:
`scripts/run_xunce_candidate_level_coverage_opportunity_materialization.py`.

Config:
`configs/xunce_candidate_level_coverage_opportunity_materialization_v1.json`.

Output root:
`outputs/path_feedback_batch_xunce_candidate_level_coverage_opportunity_materialization_v1/`.

The stage reads Stage 18A ROI expansion evidence and Stage 18C coverage
comparison evidence. It writes an enriched Stage 18A-compatible root containing
`xunce-high-fidelity-real-map-roi-expansion-summary.json`,
`xunce-high-fidelity-real-map-slices.jsonl`, and
`xunce-high-fidelity-path-feedback-audit.json`.

Each candidate receives endpoint coverage, path-line coverage, expected coverage
rate, expected new covered cells, ROI-weighted coverage, revisit penalty,
coverage gain per path cost, coverage gain per risk, and opportunity rank. The
source is labeled `geometric_counterfactual_from_stage18a_candidate/v1`.

Passing routes to `run_oracle_separability_benchmark`. Low spread routes to
`repair_candidate_level_coverage_materialization`. Missing geometry routes to
`repair_candidate_materialization_inputs`. ROI spread gaps route to
`expand_roi_or_refine_roi_weighting`.

## Stage 18F

Runner:
`scripts/run_xunce_oracle_separability_benchmark.py`.

Config:
`configs/xunce_oracle_separability_benchmark_v1.json`.

Output root:
`outputs/path_feedback_batch_xunce_oracle_separability_benchmark_v1/`.

The stage consumes the Stage 18E materialized root and compares incumbent,
Xunce, greedy coverage oracle, and cost-aware coverage oracle on the same
candidate set. Oracles can only choose reachable, unmasked, no-open-grid
candidates.

Oracle separability requires positive greedy and cost-aware deltas against
incumbent, at least 60% oracle-better scenarios, at least three oracle-better
ROI groups, zero cost-aware efficiency regressions, zero oracle mask violations,
and zero open-grid fallback.

Routes:

- `expand_roi_or_map_complexity` when oracle cannot separate.
- `xunce_training_or_adapter_iteration_required` when oracle separates but
  Xunce does not.
- `refine_coverage_reward_and_cost_guard` when Xunce coverage improves with
  efficiency regression.
- `xunce_default_policy_candidate_authorization_preflight` only when Xunce
  coverage advantage is clean.

## Verification

```powershell
D:\conda_envs\lunar-explorer\python.exe -m pytest `
  tests\test_xunce_candidate_level_coverage_opportunity_materialization.py `
  tests\test_xunce_oracle_separability_benchmark.py `
  tests\test_xunce_coverage_discriminability_audit.py `
  tests\test_xunce_high_fidelity_exploration_coverage_comparison.py `
  tests\test_platform_stage_runner.py `
  -q

D:\conda_envs\lunar-explorer\python.exe scripts\run_stage.py `
  --stage xunce-candidate-level-coverage-opportunity-materialization `
  --dry-run

D:\conda_envs\lunar-explorer\python.exe scripts\run_stage.py `
  --stage xunce-oracle-separability-benchmark `
  --dry-run
```

## Non-Goals

This work does not download new maps, run PPO, publish checkpoints, replace the
default policy, connect an executor, start online canary traffic, change action
space, change default A*, or treat oracle as a deployable policy.
