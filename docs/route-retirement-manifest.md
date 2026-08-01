# 已完成路线退役清单

本次父仓库收敛已完成。冻结 fixture `tests/fixtures/route_retirement_candidates_v1.json` 保留以下已删除集合的可核验记录：

- path-feedback：29 项；
- Stage26：148 项；
- Stage18--25：250 项；
- 合计：427 项 fixture 候选；此外删除不属于当前 Stage6 基础链的早期策略实验、Xunce Stage0--17、path-v2、release/canary、旧文档和对应测试等依赖闭包。Stage6 仍依赖的 PPO Stage1--5 基础 workflow 保留。

`model-explorer` 与 `visual-workbench` 已从父仓库索引和 `.gitmodules` 移除；`path-planner`、`dev-platform-constraints` 保留。此操作不递归删除本地子模块目录，也不删除 D 盘外部 artifacts、备份、训练输出或 checkpoint。

恢复历史时应使用已完成备份和外部 artifact 索引，而非恢复父仓库的旧入口。当前父仓库只有 Stage6、G1/G2/G3、默认 grid A*、opt-in v3、平台约束与支持它们的 helper。
