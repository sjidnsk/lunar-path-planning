# Stage23.0A Materialize Obstacle Sources For Endpoint Theta LOS

## Goal

补齐 Stage23.0 缺失的真实 LOS 障碍源 lineage。high-fidelity run 需要输出稳定的
`xunce-exploration-coverage-obstacle-sources.json`，candidate metric audit 只引用
`obstacle_source_id`、`obstacle_source_hash`、`obstacle_source_kind`，然后重跑 Stage23.0
验证 obstacle-aware endpoint theta coverage 是否 material。

## Source Semantics

- `obstacle_cells` / `obstacle_rectangles` 是优先级最高的物理遮挡源。
- `blocked_cells` / `blocked_rectangles` 可作为 `blocked_as_obstacle_proxy`。
- sidecar `passable_mask == false` 可物化为 `blocked_as_obstacle_proxy`。
- `blocked_count`、`passable_ratio` 等摘要字段不能当成 LOS 障碍地图。
- `no_go_cells` 默认不遮挡视线；只有 `no_go_blocks_los=true` 时才纳入。

## Artifacts

- `xunce-stage23-0a-summary.json`
- `xunce-stage23-0a-obstacle-source-audit.json`
- `xunce-stage23-0a-candidate-audit-source-linkage.json`
- `xunce-stage23-0a-rerun-stage23-0-summary.json`
- `xunce-stage23-0a-next-stage-routing.json`
- `xunce-stage23-0a-report.md`
- `xunce-stage23-0a-manifest.json`

## Routing

- No usable source: `repair_stage23_map_obstacle_source_export`
- Only blocked proxy source: `review_blocked_as_obstacle_proxy_semantics`
- Candidate source id/hash mismatch: `repair_stage23_0a_candidate_obstacle_source_linkage`
- Physical obstacle materiality after Stage23.0 rerun: `implement_stage23_1_obstacle_aware_theta_reward_contract`
- No material occlusion: `document_obstacle_occlusion_audit_only`

## Non-Goals

Stage23.0A 不启动 PPO、不发布 checkpoint、不替换 default policy、不连接 executor、不启动
canary、不做连续 theta、不做沿路径持续观测、不做坡度/高度/3D 遮挡，也不修改 default
A*、network 或 candidate generation 主算法。
