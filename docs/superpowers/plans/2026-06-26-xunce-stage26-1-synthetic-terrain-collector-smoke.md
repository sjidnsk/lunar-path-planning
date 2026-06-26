# Stage26.1 Synthetic Terrain Collector Smoke

目标：把 Stage26.0 生成的 synthetic rock/pit augmented sidecar 接入真实
`Stage21.1 -> Stage21.2 -> Stage21.3` PPO 数据链路，证明 collector、reward、
batch 都能消费合成岩石/坑洞地形，而不是只停留在离线地图审计。

输入：

- Stage26.0 root:
  `D:\CodexDownloads\lunar-path-planning\stage26_synthetic_terrain_augmentation\outputs\path_feedback_batch_xunce_stage26_0_synthetic_rock_pit_terrain_augmentation_contract_v1`
- synthetic source kind: `synthetic_terrain_obstacle_proxy/v1`
- coverage source: `endpoint_theta_slope_obstacle_los/v1`
- path cost source: `hybrid_astar_pose_path/v1`
- platform-aligned max slope: `30.0`

实现范围：

- 新增 Stage26.1 wrapper/config/test/stage registry。
- Stage21.1 读取 Stage26.0 augmented sidecars，并在 transition `info` 中保留
  synthetic terrain model/hash/source kind、synthetic hard obstacle、LOS blocker、
  high-risk provenance、effective obstacle sources 和 `physical_obstacle_cells_written=false`。
- Stage21.2 reward 保留 synthetic provenance，并继续使用 slope-obstacle theta
  coverage 与 Hybrid A* pose path cost。
- Stage21.3 新增 synthetic terrain gate，拒绝缺 synthetic provenance、synthetic
  写成 physical obstacle、point-only/unobstructed theta/grid path fallback、默认 A*
  替换或 Ackermann feasibility 声明。

关键修复：

- Stage26.1 从 Stage26.0 map augmentation audit 派生 `platform_contract_hash`，
  写入短路径 sidecar、ROI slice row 和 Stage21.1 config。
- Stage21.1 的 Hybrid A* sampling mask 要求候选同时具备 reachable、正的
  `hybrid_astar_path_cost`、pose path hash、trajectory kind 和 path cost source，
  避免 zero-cost/no-op 候选进入 reward 合同。

真实 smoke 结果：

- Stage26.1 status: `passed`
- next route: `run_stage26_2_synthetic_terrain_ppo_update_smoke`
- transition/reward/batch row count: `8/8/8`
- synthetic transition/reward/batch contract missing count: `0`
- point-only reward fallback: `0`
- unobstructed theta fallback: `0`
- grid path fallback: `0`
- hard risk、mask violation、path planning failure、open-grid fallback: `0`

边界：

- 本阶段不运行 Stage21.4 PPO update。
- 不发布 checkpoint、不替换 default policy、不连接 executor、不启动 canary。
- synthetic terrain 仍是 proxy，不得宣称为真实 physical obstacle。
- 不修改 reward 目标、network、default A*、candidate generation 主算法。
