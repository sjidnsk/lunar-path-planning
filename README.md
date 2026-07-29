# lunar-path-planning

## 中文说明

月球无人平台自主探索与路径规划系统级仓库。

本仓库协调四个子项目：

- `path-planner`：网格 A* 与可选 Scout Mini Hybrid A* 位姿规划。
- `model-explorer`：探索决策编排与 path-feedback 实验。
- `dev-platform-constraints`：平台、地形、地图和约束合同。
- `visual-workbench`：以 artifact 为中心的证据浏览和验证工作台。

旧的 `a_gcs_ws-2.0.1` 执行层参考工程不再作为父仓库运行依赖。

`path-planner/cpp` 另提供 opt-in 的 C++20 多平台规划 v3：轮式、足式机体
参考和纯弹道飞跃式下一着陆参考共用强类型接口。该实现不替换默认 A*、不连接
executor；1 秒仅是 Release benchmark 的 P95 实验指标。

### 当前路线

当前 Xunce 主线处在 Stage26 synthetic terrain policy-signal 诊断阶段：

```text
Stage25.0
  连续 theta 动作空间基础

Stage26.0 -> Stage26.3
  synthetic rock/pit terrain 合同、collector、PPO update、post-update eval

Stage26.4
  worker=4 policy update signal-strength repair

Stage26.5
  synthetic discrete margin crossing calibration

Stage26.6 -> Stage26.7H
  synthetic credit direct-PPO-credit repair, main-coverable efficiency rerun, reachable theta repair, behavior-policy KL baseline repair, and eval binding repair

当前下一跳
  xunce-stage26-io1-artifact-path-contract-and-long-path-resilience
```

当前结论是：Stage26.8F 已证明 `main_coverage_per_100m_delta=0` 的首要原因不是继续盲跑 H20，而是已完成 eval 的多个 scenario 只有不同 `scenario_id`，实际 first cell、candidate 序列、covered 序列、selected action 序列和 episode metrics 高度重复。Stage26.8G 修复 scenario fixture 多样性：同一 synthetic terrain 下，每个 scenario 必须从真实不同的安全起点/seed/ROI 上下文出发；早期 AUC、总路程和 Hybrid A* path cost 仍只作诊断。

### 当前合同

```text
coverage_source = endpoint_theta_slope_obstacle_los/v1
path_cost_source = hybrid_astar_pose_path/v1
synthetic_source_kind = synthetic_terrain_obstacle_proxy/v1
action_space_type = hybrid_discrete_xy_continuous_theta/v1
max_traversable_slope_deg = 30.0
hybrid_astar_candidate_eval_workers = 4
```

重要边界：

- synthetic rock/pit 只能作为障碍代理，不是真实物理障碍真值。
- default grid A* 不被替换。
- Hybrid A* 是 opt-in 后端，不宣称 Ackermann feasible。
- 未经后续 release-governance 阶段授权，不发布 checkpoint、不替换 default policy、不连接真实 executor、不启动 canary。

## 文档地图

权威文档边界见 `docs/xunce-stage-documentation-index.md`。

| 文件或目录 | 职责 |
|---|---|
| `AGENTS.md` | agent 操作规则、当前硬边界、当前主线合同、最近关键阶段摘要。 |
| `README.md` | 面向人的项目入口、当前路线和文档入口。 |
| `docs/superpowers/plans/` | 每个阶段的完整执行计划。 |
| `outputs/.../report.md` | 每个阶段的真实执行结果。 |
| `docs/算法设计与系统架构报告.md` | 架构设计、系统演进和长期设计结论。 |
| `docs/superpowers/specs/` | 长期技术规格和接口合同。 |
| `configs/stage_registry.json` | 机器可执行 stage 注册。 |

## English

System-level repository for lunar rover autonomous exploration and path planning.

This repository coordinates four subprojects:

- `path-planner`: grid A* plus opt-in Scout Mini Hybrid A* pose planning.
- `model-explorer`: exploration decision orchestration and path-feedback experiments.
- `dev-platform-constraints`: platform, terrain, map, and constraint contracts.
- `visual-workbench`: artifact-first evidence browser and validation workbench.

The old `a_gcs_ws-2.0.1` execution-layer reference is intentionally excluded from the parent repository and is not a runtime dependency.

### Current Route

The active Xunce line is in Stage26 synthetic terrain policy-signal diagnostics:

```text
Stage25.0
  continuous theta action-space foundation

Stage26.0 -> Stage26.3
  synthetic rock/pit terrain contract, collector, PPO update, post-update eval

Stage26.4
  worker=4 policy update signal-strength repair

Stage26.5
  synthetic discrete margin crossing calibration

Stage26.6 -> Stage26.7H
  synthetic exploration credit assignment, main-coverable efficiency rerun, reachable-theta repair, behavior-policy KL baseline repair, and eval binding repair

Current next step
  xunce-stage26-io2-remaining-runner-long-path-migration
```

Stage26.8F showed that repeated zero `main_coverage_per_100m_delta` is currently dominated by scenario content duplication: different `scenario_id` values were sharing the same first cell, candidate sequence, covered-cell sequence, selected-action sequence, and episode metrics. Stage26.8G repairs the scenario fixture contract so each synthetic scenario uses a real distinct safe start/seed/ROI context before H16/H20 efficiency work resumes. Early AUC, total path length, and Hybrid A* path cost remain diagnostics only.

Stage26.8M generalizes the specialized Stage26.8D/8H/8I resumable runners into one configurable training pipeline. It manages `collector -> update -> eval_pre -> eval_post -> aggregate` jobs over horizon, seed, scenario-count, rollout-step, and update-combo dimensions. All experiment state lives under the D-drive output root; the repository only stores source, configs, tests, and docs.

Stage26.IO1 introduced long-path aware artifact IO, canonical short artifact aliases with legacy fallback, and short default output roots such as `D:/xunce/out/s26_8n`. Stage26.IO2 is the current path-governance step: it migrates the remaining active Stage21.4/21.5, Stage26.2/26.3, and Stage26.8 resume/update/eval runners so later Stage26.8N/8Q/8M work does not misreport missing artifacts on Windows long paths. Neither IO step changes PPO, reward, Hybrid A*, candidate generation, or synthetic terrain.

Historical route anchor: Stage26.5 diagnosed missing direct synthetic exploration credit and routed to `repair_stage26_synthetic_exploration_credit_assignment`.

### Current Contracts

```text
coverage_source = endpoint_theta_slope_obstacle_los/v1
path_cost_source = hybrid_astar_pose_path/v1
synthetic_source_kind = synthetic_terrain_obstacle_proxy/v1
action_space_type = hybrid_discrete_xy_continuous_theta/v1
max_traversable_slope_deg = 30.0
hybrid_astar_candidate_eval_workers = 4
```

Important boundaries:

- Synthetic rocks and pits are obstacle proxies, not physical obstacle truth.
- Default grid A* is not replaced.
- Hybrid A* is an opt-in backend and must not claim Ackermann feasibility.
- Unless a later release-governance stage explicitly authorizes it, no checkpoint may be published, no default policy may be replaced, no real executor may be connected, and no canary may be started.

## Documentation Map

Use `docs/xunce-stage-documentation-index.md` as the authoritative map for documentation responsibilities.

| File or directory | Responsibility |
|---|---|
| `AGENTS.md` | Agent operating rules, current hard boundaries, current mainline contracts, and recent key stage summaries. |
| `README.md` | Human-facing project overview, current route, and documentation entry points. |
| `docs/superpowers/plans/` | Full implementation plans for individual stages. |
| `outputs/.../report.md` | Real execution results for individual stage runs. |
| `docs/算法设计与系统架构报告.md` | Architecture design, system evolution, and durable design decisions. |
| `docs/superpowers/specs/` | Long-lived technical specifications and interface contracts. |
| `configs/stage_registry.json` | Machine-readable stage registration only. |

## Stage Registry

Use the Python stage runner for registered stages:

```powershell
python scripts\run_stage.py --list
python scripts\run_stage.py --stage xunce-stage26-5-synthetic-discrete-margin-crossing-calibration --dry-run
```

Large datasets and generated artifacts should stay outside Git, preferably under:

```text
D:\CodexDownloads\lunar-path-planning
```

## Platform Notes

The platform contract is centralized in:

```text
configs/platforms/agilex_scout_mini_piper_v1.json
```

The Scout Mini hard slope gate is aligned to:

```text
max_traversable_slope_deg = 30.0
```

## Development Notes

Use Windows-safe Python entry points first:

```powershell
python scripts\run_stage.py ...
python -m pytest ...
python -m py_compile ...
```

Do not commit raw datasets, large generated output roots, checkpoints, or training artifacts unless a later governance plan explicitly says so.
