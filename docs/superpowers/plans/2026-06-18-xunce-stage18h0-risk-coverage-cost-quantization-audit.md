# Xunce Stage 18H.0 Risk-Coverage-Cost Quantization Audit Plan

## Summary

Stage 18H.0 adds a decoupled candidate metric audit for the Xunce high-fidelity evidence chain. It consumes the true-incumbent-bound Stage 18G.0 root and writes coverage, risk, and cost atomic vectors for every candidate before any guard, Pareto, reward, or critic target is considered.

This stage is diagnostic only. It does not train actor/critic models, modify network/action space/default A*, publish checkpoints, replace default policy, connect executors, start online canary, or download new maps.

## Implementation

- Add `scripts/run_xunce_risk_coverage_cost_quantization_audit.py` with CLI support for `--config`, `--output-root`, `--repo-root`, and `--source-bound-coverage-root`.
- Add `configs/xunce_risk_coverage_cost_quantization_audit_v1.json` with Stage 18G.0 as the default source root and lightweight cost defaults.
- Register `xunce-risk-coverage-cost-quantization-audit` in `configs/stage_registry.json`.
- Write compatible artifacts so the output root can be passed to Stage 18F when safe-efficient candidates exist.

## Metrics

- Coverage vector: endpoint new cells, path-line new cells, ROI-weighted coverage, expected new cells, total new cells, revisit count, overlap count, overlap ratio, and provenance fields.
- Risk vector: candidate point risk, path risk mean/peak/p95, path risk exposure length, source, and primary risk attribution axis. V1 uses the existing candidate `risk` scalar as a path-risk proxy and records that source explicitly.
- Cost vector: path cost, budget used ratio, planning latency, and source. V1 does not include Ackermann, curvature, turning-rate, or executor feasibility costs.
- Relative audit: coverage/risk/cost/budget deltas against true incumbent, guard booleans, hard validity, safe-efficient candidate, Pareto status, failure axis, and reason codes.

## Validation

```powershell
D:\conda_envs\lunar-explorer\python.exe -m pytest `
  tests\test_xunce_risk_coverage_cost_quantization_audit.py `
  tests\test_xunce_safe_efficient_opportunity_root_cause_audit.py `
  tests\test_xunce_cost_efficient_coverage_opportunity_refinement.py `
  tests\test_platform_stage_runner.py `
  -q

D:\conda_envs\lunar-explorer\python.exe scripts\run_stage.py `
  --stage xunce-risk-coverage-cost-quantization-audit `
  --dry-run

D:\conda_envs\lunar-explorer\python.exe -m py_compile `
  scripts\run_xunce_risk_coverage_cost_quantization_audit.py `
  scripts\run_stage.py

git diff --check
```

## Decision Routes

- `run_true_incumbent_selection_binding` when the true incumbent binding root is missing or falls back to action index 0.
- `repair_path_feedback_candidate_validation` when positive proposals are not path-feedback validated.
- `repair_metric_family_separation` when metric coupling is detected.
- `repair_metric_normalization` when raw and normalized Pareto disagree.
- `calibrate_risk_margin_or_risk_attribution` when all coverage-positive candidates fail risk guard.
- `repair_risk_aware_candidate_generation` when only part of the coverage-positive set is risk-regressive.
- `expand_roi_or_map_complexity` when coverage-positive candidates are absent.
- `rerun_oracle_separability_with_quantized_root` when validated safe-efficient candidates exist across enough ROI groups.
