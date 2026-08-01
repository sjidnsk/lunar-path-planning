# 项目路线图

机器可读路线表为 `configs/mainline_routes_v1.json`；本文件说明默认与 opt-in 边界。

| 能力 | 状态 | 入口 | 边界 |
|---|---|---|---|
| Stage6 高分辨率前沿 PPO | 当前主线 | `scripts/run_ppo_stage6_standard.py` | 不训练、不发布 checkpoint。 |
| G1/G2/G3 双门槛 | 当前主线 | `scripts/run_xunce_mid_dual_g1_coverage.py`、`run_xunce_mid_dual_g2_planning_time.py`、`run_xunce_mid_dual_g3_closed_loop.py` | 仅合同和评估入口；不启动 canary。 |
| Python grid A* | 默认 | `src/lunar_exploration_ppo/integrations/path_planner_adapter.py` | `path_planner.search.AStarPlanner` 不被替换。 |
| Hybrid A* | opt-in | `path-planner` | 不宣称 Ackermann feasible。 |
| 多平台 path-planner v3 | opt-in | `path-planner/cpp` | 不替换默认 A*，不连接 executor。 |
| 平台约束 | 保留支撑 | `dev-platform-constraints` | `max_traversable_slope_deg=30.0`。 |

依赖关系：Stage6 和 G1/G2/G3 使用父仓库的共享 artifact helper；默认 A* 来自 `path-planner`；平台验证同时覆盖 `path-planner` 与 `dev-platform-constraints`。保留子模块仅为这两个。

退役路径已从父仓库索引中移除；恢复历史只能使用备份和外部 artifact 索引，不重新暴露历史 runner 或 stage registry。
