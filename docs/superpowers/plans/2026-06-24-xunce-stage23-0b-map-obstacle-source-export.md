# Stage23.0B Map Obstacle Source Export

目标是修复 Stage23.0A 发现的源头问题：bounded high-fidelity 输入没有可复现的 LOS 障碍源，导致 candidate rows 全量 `obstacle_source_missing=true`。

本阶段做三件事：

1. `run_quasi_real_map_path_feedback_bridge.py` 从 `passable_mask == false` 物化 `blocked_cells`，并写 `blocked_source_kind=passable_mask_false`。
2. high-fidelity obstacle source selection 优先读取 sidecar 的显式 `obstacle_cells` / `blocked_cells`，再 fallback 到 `passable_mask == false`。
3. `run_xunce_stage23_0b_map_obstacle_source_export.py` 重跑 Stage23.0A/23.0，并把物理 obstacle、blocked proxy、仍缺 source 三种情况分开路由。

语义边界：

- `physical_obstacle_cells` 只来自显式物理障碍字段。
- `blocked_cells` 是不可通行代理，写成 `blocked_as_obstacle_proxy`，不能直接当物理遮挡。
- `no_go_cells` 默认不挡 LOS，只有 `no_go_blocks_los=true` 时纳入。
- `blocked_count`、`passable_ratio`、risk、slope 不能被提升为真实障碍物。

验收：

- sidecar 可导出 `blocked_cells`，hash/linkage 可复现。
- Stage23.0A rerun 不再因为可用 blocked source 未物化而全量 missing。
- Stage23.0 能生成 LOS audit rows；如果只有 proxy，则 route 到 `review_blocked_as_obstacle_proxy_semantics`。
- 不训练、不发布、不替换 default policy、不连接 executor、不启动 canary、不修改 default A*。
