# lunar-path-planning

月球无人平台自主探索与路径规划的主线仓库。当前仅保留以下能力及其真实依赖：

- 高分辨率前沿 PPO / Stage6；入口：[Stage6](docs/ppo-highres-frontier-stage6.md)。
- 中期缩减规模双门槛 G1/G2/G3；入口：[运行手册](docs/xunce-midterm-dual-gate-runbook.md)。
- `path-planner`：默认 Python grid `AStarPlanner`，以及 opt-in 多平台规划 v3。
- `dev-platform-constraints`：平台、地形与地图约束合同，硬坡度阈值为 `max_traversable_slope_deg=30.0`。
- bootstrap 与 Windows/Ubuntu 平台验证；入口：[平台文档](docs/platform/windows-ubuntu-setup.md)。

默认规划算法不变：PPO adapter 使用 `path_planner.search.AStarPlanner`。Hybrid A* 和 v3 都是 opt-in；Hybrid A* 不宣称 Ackermann feasible。v3 不替换默认运行时、不连接 executor。

## 快速入口

```powershell
$env:PYTHONPATH = "$PWD\path-planner\src;$PWD\src"
D:\conda_envs\lunar-explorer\python.exe -m pytest tests/ppo_highres_frontier tests/test_xunce_mid_dual_*.py -q
D:\conda_envs\lunar-explorer\python.exe scripts\run_platform_validation_matrix.py --profile windows-non-drake --dry-run
```

详细导航见 [文档索引](docs/documentation-index.md)，机器可读入口见 `configs/mainline_routes_v1.json`。运行时产物、训练输出与 checkpoint 不进入 Git，应写入 D 盘输出根。

## 安全边界

- synthetic terrain 仅为 proxy，不得称为 `physical_obstacle_cells`。
- 不发布 checkpoint、不替换 default policy、不连接 executor、不启动 canary。
- 已退役路线的父仓库文件与 gitlink 已移除；本地目录、外部 artifacts 与备份不会被本仓库清理动作删除。
