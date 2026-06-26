# Stage24.2 Hybrid A* Reward Path-Cost Contract

目标：把 Stage24.1 产出的 `hybrid_astar_pose_path/v1` 候选位姿路径成本接入 Stage21.2 reward 与 Stage21.3 batch gate，让后续 PPO 能使用到目标观测位姿 `(x,y,theta)` 的 Hybrid A* 成本。

## 输入

- Stage24.1 root: `D:\CodexDownloads\lunar-path-planning\stage24_hybrid_astar_pose_planner\outputs\path_feedback_batch_xunce_stage24_1_hybrid_astar_candidate_path_cost_integration_v1`
- Stage24.1 candidate Hybrid A* audit
- Stage21 coverage-constrained reward profile v2
- Stage23 平台对齐 slope-obstacle theta coverage lineage

## 实施范围

- 新增 runner: `scripts/run_xunce_stage24_2_hybrid_astar_reward_path_cost_contract.py`
- 新增 config: `configs/xunce_stage24_2_hybrid_astar_reward_path_cost_contract_v1.json`
- 扩展 Stage21.2：`require_hybrid_astar_path_cost_contract=true` 时使用 `hybrid_astar_path_cost` 作为 `metrics.path_cost_m`
- 扩展 Stage21.3：拒绝 grid-only path cost、缺 pose path hash、默认 A* 替换声明、Ackermann feasible 声明
- 注册 stage id: `xunce-stage24-2-hybrid-astar-reward-path-cost-contract`

## 验收

- Stage24.2 summary `status=passed`
- route 到 `run_stage24_3_hybrid_astar_path_cost_ppo_collector_smoke`
- reward replay 能区分 Hybrid A* pose path cost 与旧 grid A* cost
- Stage21.3 能接受 Hybrid batch，并拒绝 grid-only / Ackermann claim batch
- 默认 A* 未替换，Ackermann feasible 未宣称
- 所有 release/default-policy/executor/canary 字段为 false/0

## 非目标

不启动 PPO，不发布 checkpoint，不替换 default policy，不连接 executor，不启动 canary，不做 continuous theta，不做沿路径持续观测或 3D DEM LOS，不宣称性能提升。
