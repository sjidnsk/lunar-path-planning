# Xunce Stage 18I.1 Evidence Closure and Separability Validation Plan

## Summary

目标是把 Stage 18I 从“候选生成代码已实现”推进到“after-Stage18I 真实 evidence 闭环结论”。本阶段先修复 safe-efficient 字段兼容，然后运行 Stage 18I、真实模型推理、true incumbent binding、Stage 18H.0 量化、Stage 18F oracle separability、Stage 18C-v2 覆盖率对比，最后用只读 closure audit 汇总。

本阶段仍不训练 PPO、不发布 checkpoint、不替换默认策略、不连接 executor、不启动 online canary、不下载新地图。

## Scope

1. 字段兼容：
   - Stage 18I formal candidate 同时写 `safe_efficient_candidate` 和 `safe_efficient_opportunity`。
   - Stage 18H.0 quantized root 同时写两个字段，且值一致。
   - Stage 18F cost-aware oracle 同时接受两个字段；任一字段存在时，只能从 validated、reachable、non-open-grid、non-proposal 的 safe-efficient candidates 中选择。

2. 新增只读 closure audit：
   - Runner: `scripts/run_xunce_stage18i_evidence_closure_audit.py`
   - Config: `configs/xunce_stage18i_evidence_closure_audit_v1.json`
   - Stage: `xunce-stage18i-evidence-closure-audit`
   - Output root: `outputs/path_feedback_batch_xunce_stage18i_evidence_closure_v1/`

3. 真实 evidence 执行链：
   - Stage 18I candidate generation
   - Stage 18B after-Stage18I true checkpoint inference
   - Stage 18G.0 after-Stage18I true incumbent binding
   - Stage 18H.0 after-Stage18I quantization
   - Stage 18F after-Stage18I oracle separability using the Stage 18H.0 true-bound/quantized root
   - Stage 18C-v2 after-Stage18I coverage rollout
   - Stage 18I.1 closure audit

## Acceptance Criteria

- `xunce-stage18i-evidence-closure-summary.json` exists.
- Closure summary includes:
  - `stage18i_candidate_generation_passed`
  - `true_model_inference_executed`
  - `true_incumbent_selection_bound`
  - `safe_efficient_candidate_count`
  - `roi_group_with_safe_efficient_candidate_count`
  - `oracle_separable`
  - `cost_aware_oracle_efficiency_regression_count`
  - `xunce_coverage_advantage_established`
  - `next_required_change`
- Missing Stage 18I / Stage 18B / Stage 18G.0 / Stage 18H.0 / Stage 18F / Stage 18C-v2 artifacts produce explicit routes.
- If oracle separates but Xunce does not, route to `stage18j_coverage_cost_evaluator_critic_preflight`.
- Boundary fields remain false and `canary_traffic_fraction=0.0`.

## Verification

```powershell
D:\conda_envs\lunar-explorer\python.exe -m pytest `
  tests\test_xunce_risk_constrained_frontier_nbv_candidate_generation.py `
  tests\test_xunce_risk_coverage_cost_quantization_audit.py `
  tests\test_xunce_oracle_separability_benchmark.py `
  tests\test_xunce_stage18i_evidence_closure_audit.py `
  tests\test_platform_stage_runner.py `
  -q

D:\conda_envs\lunar-explorer\python.exe scripts\run_stage.py `
  --stage xunce-stage18i-evidence-closure-audit `
  --dry-run

D:\conda_envs\lunar-explorer\python.exe -m py_compile `
  scripts\run_xunce_stage18i_evidence_closure_audit.py `
  scripts\run_xunce_risk_constrained_frontier_nbv_candidate_generation.py `
  scripts\run_xunce_risk_coverage_cost_quantization_audit.py `
  scripts\run_xunce_oracle_separability_benchmark.py `
  scripts\run_stage.py

git diff --check
```

## Decision Rules

- Stage 18I fails: repair frontier-NBV sampling, path-feedback validation, or ROI/map complexity.
- Stage 18B fails: repair checkpoint availability or Stage 18B adapter.
- Stage 18G.0 fails: repair candidate cell alignment or true incumbent binding.
- Stage 18H.0 has no safe-efficient candidates: repair candidate generation or ROI/map complexity.
- Stage 18F oracle not separable: refine candidate generation / ROI complexity.
- Oracle separable but Xunce not better: enter Stage 18J evaluator / critic preflight.
- Xunce clean advantage: only proceed to authorization preflight, not default replacement.

## Non-Goals

- No PPO update.
- No checkpoint publication.
- No default policy replacement.
- No executor connection.
- No online canary.
- No new map download.
- No action space, default A*, executor, or checkpoint release flow change.
