# lunar-path-planning

## 中文说明

月球无人平台自主探索与路径规划系统级仓库。

本仓库协调四个子项目：

- `path-planner`：保留的子模块；提供默认 Python grid A*，以及可选 Scout Mini Hybrid A* 位姿规划。
- `dev-platform-constraints`：保留的子模块；提供平台、地形、地图和约束合同。
- `model-explorer`、`visual-workbench`：已退役的历史可复现与人工清理候选；物理移除仍须额外人工确认，详见 [路线退役与清理边界](docs/route-retirement-manifest.md)。

旧的 `a_gcs_ws-2.0.1` 执行层参考工程不再作为父仓库运行依赖。

`path-planner/cpp` 另提供 opt-in 的 C++20 多平台规划 v3：轮式、足式机体
参考和纯弹道飞跃式下一着陆参考共用强类型接口。该实现不替换默认 A*、不连接
executor；1 秒仅是 Release benchmark 的 P95 实验指标。

### 当前路线

- 高分辨率前沿 PPO / Stage6 是当前探索主线，权威入口为 `docs/ppo-highres-frontier-stage6.md`。
- 中期缩减规模双门槛 G1/G2/G3 是当前交付链，运行与验收入口为 `docs/xunce-midterm-dual-gate-runbook.md`。
- `path-planner/cpp` 的多平台路径规划 v3 是保留的 opt-in 能力；它不替换默认运行时，也不连接 executor。
- `dev-platform-constraints` 保留平台约束合同；平台对齐硬坡度阈值为 `max_traversable_slope_deg=30.0`。
- 默认路径规划算法是 `path-planner` 中的 Python grid A*：PPO adapter 使用 `path_planner.search.AStarPlanner`。Hybrid A* 和 v3 均为 opt-in 能力。

Xunce Stage18--26、path-feedback 与早期策略实验均已退役，仅保留为历史可复现和人工清理候选；本次路线退役不表示相关代码已删除。退役范围和人工清理边界的唯一详细入口是 [路线退役与清理边界](docs/route-retirement-manifest.md)。

### 长期边界

重要边界：

- synthetic rock/pit 只能作为障碍代理，不是真实物理障碍真值。
- `path-planner` 中的 Python grid A*（`AStarPlanner`）是默认路径规划算法，不被替换。
- Hybrid A* 与 v3 均为 opt-in 能力；Hybrid A* 不宣称 Ackermann feasible。
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
| `docs/route-retirement-manifest.md` | 退役范围和人工清理边界的唯一详细入口。 |

## English

System-level repository for lunar rover autonomous exploration and path planning.

This repository coordinates four subprojects:

- `path-planner`: a retained submodule that provides the default Python grid A* and opt-in Scout Mini Hybrid A* pose planning.
- `dev-platform-constraints`: a retained submodule for platform, terrain, map, and constraint contracts.
- `model-explorer` and `visual-workbench`: retired historical-reproducibility and manual-cleanup candidates. Physical removal still requires separate human confirmation; see the [route retirement and cleanup boundary](docs/route-retirement-manifest.md).

The old `a_gcs_ws-2.0.1` execution-layer reference is intentionally excluded from the parent repository and is not a runtime dependency.

### Current Route

- High-resolution frontier PPO / Stage6 is the current exploration route; its authoritative entry is `docs/ppo-highres-frontier-stage6.md`.
- The midterm reduced-scale dual-gate G1/G2/G3 chain is the current delivery route; its run and acceptance entry is `docs/xunce-midterm-dual-gate-runbook.md`.
- Multi-platform path planning v3 in `path-planner/cpp` is a retained opt-in capability; it does not replace the default runtime and does not connect an executor.
- `dev-platform-constraints` retains the platform-constraint contract, including the aligned hard slope threshold `max_traversable_slope_deg=30.0`.
- The default path-planning algorithm is Python grid A* in `path-planner`: the PPO adapter uses `path_planner.search.AStarPlanner`. Hybrid A* and v3 are opt-in capabilities.

Xunce Stage18--26, path-feedback, and early policy experiments are retired and remain only for historical reproducibility and as manual-cleanup candidates; retirement does not mean the code was deleted. The sole detailed entry for the retirement scope and manual-cleanup boundary is the [route retirement and cleanup boundary](docs/route-retirement-manifest.md).

### Long-Term Boundaries

Important boundaries:

- Synthetic rocks and pits are obstacle proxies, not physical obstacle truth.
- Python grid A* (`AStarPlanner`) in `path-planner` is the default path-planning algorithm and is not replaced.
- Hybrid A* and v3 are opt-in capabilities; Hybrid A* must not claim Ackermann feasibility.
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
| `docs/route-retirement-manifest.md` | The sole detailed entry for retirement scope and manual-cleanup boundaries. |

## Stage Registry

Use the Python stage runner for registered stages:

```powershell
python scripts\run_stage.py --list
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
