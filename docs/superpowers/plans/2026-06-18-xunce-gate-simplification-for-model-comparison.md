# Xunce Gate Simplification For Model Comparison

## Objective

Simplify the Stage 18I/18H/18F/18C evidence chain so model comparison is blocked only by evidence authenticity failures or candidate validity failures.

## Implemented Gate Semantics

Hard gate family 1: evidence authenticity.

- Blocks fake or missing true checkpoint inference.
- Blocks proxy selection.
- Blocks missing or unsupported checkpoints.
- Blocks `fallback_action_index_0`.
- Blocks candidate-cell mismatch during true incumbent binding.

Hard gate family 2: candidate validity.

- Blocks unvalidated proposals.
- Blocks unreachable candidates.
- Blocks open-grid fallback candidates.
- Blocks missing required cost, risk, or coverage provenance.
- Blocks zero valid candidates.

The following are diagnostics only and do not block Stage 18B / Stage 18C-v2 model comparison:

- `safe_efficient_candidate_count`
- `frontier_boundary_candidate_count`
- `roi_group_with_safe_efficient_candidate_count`
- `oracle_separable`
- path cost or coverage efficiency relative to incumbent
- whether Xunce has already shown a coverage advantage

## Output Contract

Related stages now expose consistent fields where applicable:

- `evidence_authenticity_gate_passed`
- `candidate_validity_gate_passed`
- `comparison_allowed`
- `blocking_reason_codes`
- `diagnostic_reason_codes`
- `diagnostic_recommended_change`
- `valid_candidate_count`
- `invalid_candidate_count`
- `valid_scenario_count`
- `invalid_scenario_count`
- `invalid_candidate_reason_counts`

`reason_codes` remains a backward-compatible alias for hard blocking reasons.

## Stage Changes

Stage 18I.2 passes when true incumbent binding and candidate validity are sound, even if frontier or safe-efficient counts are zero. It routes to `rerun_true_model_inference_and_binding` and records sampling issues as diagnostics.

Stage 18H.0 quantizes coverage, risk, and cost. Risk or cost regressions are reported but no longer fail the stage.

Stage 18F cost-aware oracle selects from valid candidates only. `safe_efficient_candidate` and `safe_efficient_opportunity` are reported, not hard pool restrictions. `oracle_separable=false` routes to Stage 18C-v2 comparison with diagnostics.

Stage 18C-v2 completes the Xunce-vs-incumbent rollout when true inference and candidate validity pass. It always routes successful comparisons to `review_xunce_incumbent_comparison_metrics`.

Stage 18I.1 closure checks only artifact completeness, true inference, true incumbent binding, and candidate validity. It does not fail on safe-efficient count, oracle separability, or Xunce advantage.

## Non-Goals

- No PPO.
- No actor/critic training.
- No checkpoint publication.
- No default policy replacement.
- No executor connection.
- No online canary.
- No new map download.
- No action-space, default A*, or model contract change.

## Verification

```powershell
D:\conda_envs\lunar-explorer\python.exe -m pytest `
  tests\test_xunce_gate_simplification.py `
  tests\test_xunce_risk_aware_frontier_nbv_candidate_repair.py `
  tests\test_xunce_risk_coverage_cost_quantization_audit.py `
  tests\test_xunce_oracle_separability_benchmark.py `
  tests\test_xunce_high_fidelity_exploration_coverage_comparison.py `
  tests\test_xunce_stage18i_evidence_closure_audit.py `
  -q
```

```powershell
D:\conda_envs\lunar-explorer\python.exe scripts\run_stage.py `
  --stage xunce-risk-aware-frontier-nbv-candidate-repair `
  --dry-run

D:\conda_envs\lunar-explorer\python.exe scripts\run_stage.py `
  --stage xunce-risk-coverage-cost-quantization-audit `
  --dry-run

D:\conda_envs\lunar-explorer\python.exe scripts\run_stage.py `
  --stage xunce-oracle-separability-benchmark `
  --dry-run

D:\conda_envs\lunar-explorer\python.exe scripts\run_stage.py `
  --stage xunce-stage18i-evidence-closure-audit `
  --dry-run
```
