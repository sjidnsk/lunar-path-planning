# Xunce Stage 18D / 18C-v2 Coverage Discriminability Repair

## Goal

Stage 18C-v1 showed many action disagreements between Xunce and the incumbent,
but no coverage return, coverage AUC, or new-cell separation. This is not enough
to conclude that the Xunce architecture is ineffective. The next step is to
test whether the evaluation task can distinguish better exploration from merely
different routing.

## Implemented Scope

- Add Stage 18D coverage discriminability audit:
  `scripts/run_xunce_coverage_discriminability_audit.py`.
- Add config:
  `configs/xunce_coverage_discriminability_audit_v1.json`.
- Register stage:
  `xunce-coverage-discriminability-audit`.
- Extend Stage 18C with v2 dynamic rollout options:
  `candidate_refresh_mode=dynamic_from_coverage_memory`,
  `coverage_metric_mode=path_line_plus_endpoint`,
  `include_oracle_baselines=true`, and
  `include_roi_weighted_coverage=true`.
- Add v2 artifacts:
  `xunce-exploration-coverage-comparison-v2-summary.json`,
  `xunce-exploration-coverage-v2-steps.jsonl`, and
  `xunce-exploration-coverage-v2-episodes.jsonl`.

## Stage 18D Audit Logic

Stage 18D reads Stage 18A ROI expansion evidence and Stage 18C coverage
comparison evidence. It reports candidate coverage spread, spread standard
deviation, Gini, Pareto frontier count, reachable candidate count, unique
candidate-cell ratio, static candidate reuse rate, nonzero coverage candidate
count, oracle coverage return, oracle-vs-model deltas, oracle regret, useful
disagreement opportunities, path/risk tradeoffs, and ROI groups with nonzero
spread.

Root-cause routes:

- `coverage_task_not_discriminative_static_candidate_reuse`
- `candidate_coverage_spread_insufficient`
- `coverage_metric_too_coarse`
- `map_or_roi_complexity_insufficient`
- `candidate_materialization_insufficient`
- `models_genuinely_no_coverage_advantage`
- `ready_for_stage18c_v2_dynamic_rollout`

## Stage 18C-v2 Logic

Stage 18C-v2 refreshes candidates from the current coverage memory, evaluates
path-line plus endpoint coverage, and compares Xunce, incumbent, greedy
coverage oracle, and cost-aware coverage oracle on the same scenario and step
state. Oracle policies cannot select masked or unreachable candidates.

Xunce can only route to authorization preflight when coverage return and
coverage AUC improve, Xunce oracle regret is lower than incumbent oracle regret,
efficiency and safety do not regress, mask violations are zero, and open-grid
fallback is zero.

## Non-Goals

This work does not publish checkpoints, replace the default policy, connect a
real executor, start online canary traffic, run PPO, change the action space,
change default A*, or download new map products by default.

## Verification

```powershell
D:\conda_envs\lunar-explorer\python.exe -m pytest `
  tests\test_xunce_coverage_discriminability_audit.py `
  tests\test_xunce_high_fidelity_exploration_coverage_comparison.py `
  tests\test_platform_stage_runner.py `
  -q

D:\conda_envs\lunar-explorer\python.exe scripts\run_stage.py `
  --stage xunce-coverage-discriminability-audit `
  --dry-run

D:\conda_envs\lunar-explorer\python.exe scripts\run_stage.py `
  --stage xunce-high-fidelity-exploration-coverage-comparison `
  --extra-arg --candidate-refresh-mode `
  --extra-arg dynamic_from_coverage_memory `
  --extra-arg --coverage-metric-mode `
  --extra-arg path_line_plus_endpoint `
  --extra-arg --include-oracle-baselines `
  --dry-run
```
