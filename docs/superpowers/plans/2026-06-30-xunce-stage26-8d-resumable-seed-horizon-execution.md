# Stage26.8D Resumable Seed/Horizon Execution Pipeline

## 目标

Stage26.8D 将 Stage26.8C 的 H16/H20 多 seed 长命令，拆成可恢复的 `(horizon, seed, phase)` 实验流水线。每次默认只推进一个 phase，外部命令被中断后，可以从最近一个完整 summary 继续。

## 核心合同

- stage id: `xunce-stage26-8d-resumable-seed-horizon-execution`
- 固定 horizons: `16`、`20`
- 固定 seeds: `260801`、`260802`、`260803`
- phase 顺序: `stage26_1 -> stage26_2 -> stage26_3`
- 默认 `run_mode=run_next`
- 默认 `max_jobs_per_invocation=1`
- 支持 `aggregate_only` 只汇总现有 artifacts。
- 支持指定 horizon/seed/phase 做单 job 修复。
- Stage26.8C partial 输出只作为只读 carryover source，不复制、不覆盖、不改写。

## 关键判据

- 主指标仍为 `main_coverage_per_100m_delta`。
- AUC、总路程、Hybrid A* path cost 只作诊断。
- 仍保留：
  - `coverage_denominator_source=main_coverable_cells/v1`
  - `coverage_source=endpoint_theta_slope_obstacle_los/v1`
  - `path_cost_source=hybrid_astar_pose_path/v1`
  - `synthetic_source_kind=synthetic_terrain_obstacle_proxy/v1`
  - `action_space_type=hybrid_discrete_xy_continuous_theta/v1`
  - `hybrid_astar_candidate_eval_workers=4`
  - `max_traversable_slope_deg=30.0`

## 输出

- `xunce-stage26-8d-summary.json`
- `xunce-stage26-8d-job-plan.json`
- `xunce-stage26-8d-job-state.jsonl`
- `xunce-stage26-8d-carryover-audit.json`
- `xunce-stage26-8d-horizon-efficiency-aggregate.json`
- `xunce-stage26-8d-next-stage-routing.json`
- `xunce-stage26-8d-report.md`
- `xunce-stage26-8d-manifest.json`

## 路由

- 输入不可信：`rerun_stage26_8d_required_inputs`
- resume state/carryover 异常：`repair_stage26_8d_resume_state_contract`
- binding/safety/unreachable/fallback 失败：`repair_stage26_8d_job_binding_or_safety`
- update/KL/checkpoint 失败：`repair_stage26_8d_job_update_stability`
- 仍有未完成 job：`continue_stage26_8d_seed_horizon_jobs`
- H16/H20 多数 seed 单位路程主覆盖率提升：`run_stage26_9_synthetic_terrain_long_horizon_efficiency_pilot`
- 单 seed 正向：`expand_stage26_8d_seed_budget_at_best_horizon`
- 全部 clean 且全 0：`repair_stage26_synthetic_policy_update_signal_strength`
- 多数 seed 负向：`repair_stage26_synthetic_credit_assignment`

## 审查门

1. Pipeline contract review：确认只新增 orchestration，不改 reward、network、Hybrid A*、synthetic terrain、candidate generation。
2. Carryover/resume review：确认 Stage26.8C partial artifacts 只读引用，failed/incomplete root 不算成功。
3. Result/routing review：确认 aggregate 按 `main_coverage_per_100m_delta` 判定，未完成 job route 到 continue。

## 验证

```powershell
python -m pytest tests\test_xunce_stage26_8d_resumable_seed_horizon_execution.py tests\test_xunce_stage26_8c_resume_h16_h20_horizon_efficiency.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage26-8d
python -m py_compile scripts\run_xunce_stage26_8d_resumable_seed_horizon_execution.py scripts\run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot.py
python scripts\run_stage.py --stage xunce-stage26-8d-resumable-seed-horizon-execution --dry-run
```
