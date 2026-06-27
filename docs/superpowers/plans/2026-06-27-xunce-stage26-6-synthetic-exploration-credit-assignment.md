# Stage26.6 Synthetic Exploration Credit Assignment Plan

## 目标

完成 `Stage26.6：synthetic_exploration_credit_assignment`，修复 Stage26.5 暴露的两个根因：

1. 网络候选特征看不到 synthetic LOS、synthetic hard obstacle、Hybrid A* path cost 和 obstacle-aware coverage 信号。
2. best synthetic candidate 没有被采样成真实 trainable selected action，因此 PPO 没有给它直接 credit。

## 当前主线合同

```text
coverage_source=endpoint_theta_slope_obstacle_los/v1
path_cost_source=hybrid_astar_pose_path/v1
synthetic_source_kind=synthetic_terrain_obstacle_proxy/v1
action_space_type=hybrid_discrete_xy_continuous_theta/v1
max_traversable_slope_deg=30.0
hybrid_astar_candidate_eval_workers=4
```

## 实现范围

- 新增 `scripts/xunce_synthetic_exploration_credit.py`。
- 新增 `scripts/run_xunce_stage26_6_synthetic_exploration_credit_assignment.py`。
- 新增 `configs/xunce_stage26_6_synthetic_exploration_credit_assignment_v1.json`。
- 新增 `tests/test_xunce_stage26_6_synthetic_exploration_credit_assignment.py`。
- 扩展 Stage21.1 collector：候选特征覆盖、synthetic credit target、mixture behavior old logprob。
- 扩展 Stage21.3 batch gate：重算 synthetic credit behavior logprob。
- 扩展 Stage21.4 update audit：确认 ratio 分母使用 behavior old logprob。

## 关键设计

Stage26.6 不改网络结构，只复用 8 维 `candidate_features` 槽位，并写入 `xunce_batch_feature_semantic_map`：

```text
relative_distance_norm
hybrid_reachable
obstacle_aware_new_visible_norm
obstacle_aware_gain_per_hybrid_cost_norm
hybrid_astar_path_cost_norm
synthetic_los_blocker_pressure_norm
synthetic_hard_obstacle_pressure_norm
risk_or_clearance_proxy_norm
```

synthetic credit target 只能从 `action_mask`、`sampling_mask`、`hard_risk_clean_mask`、Hybrid A* reachable 全部允许的候选中选出。target 被选中时必须是真实 selected action，不允许只写 counterfactual row。

行为策略：

```text
behavior_policy_id=synthetic_credit_mixture_policy/v1
behavior_prob(a)=(1-p)*policy_prob(a)+p*I[a==target]
old_log_prob=old_behavior_point_log_prob+old_theta_log_prob
```

同时保留 `old_policy_*` 字段用于审计。

## 输出

默认输出根：

```text
D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/outputs/path_feedback_batch_xunce_stage26_6_synthetic_exploration_credit_assignment_v1
```

主要产物：

- `xunce-stage26-6-summary.json`
- `xunce-stage26-6-synthetic-credit-target-audit.jsonl`
- `xunce-stage26-6-behavior-logprob-audit.json`
- `xunce-stage26-6-candidate-feature-exposure-audit.json`
- `xunce-stage26-6-stage26-1-summary.json`
- `xunce-stage26-6-stage26-2-summary.json`
- `xunce-stage26-6-stage26-3-summary.json`
- `xunce-stage26-6-next-stage-routing.json`
- `xunce-stage26-6-report.md`
- `xunce-stage26-6-manifest.json`

## 边界

不发布 checkpoint，不替换 default policy，不连接 executor，不启动 canary。不改 reward 目标、不改网络结构、不重新生成 synthetic terrain、不做 continuous x/y、3D LOS 或沿路径持续观测。
