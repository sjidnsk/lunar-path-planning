# Xunce Stage 18.7 Candidate Count Scaling Audit

## 目标

验证一个窄假设：在不改变候选点生成机制、不新增候选类型、不修改 reward、network、action space 或 default A* 的前提下，只增加每步候选点数量，是否能缓解 Xunce 当前的候选数量瓶颈。

## Sweep 档位

| candidate_count | proposal_pool_limit |
|---:|---:|
| 6 | 48 |
| 12 | 96 |
| 24 | 192 |
| 36 | 288 |

每个 count 必须先跑 Stage 18.4E，且打开 `--emit-candidate-metric-audit`。随后跑对应 Stage 18.5 和 Stage 18.6。Stage 18.7 runner 只读消费这些 root，并输出跨 count 汇总。

## 固定项

- `candidate_refresh_mode=dynamic_frontier_nbv_in_process`
- `dynamic_candidate_generation_mode=map_aware_coverage_frontier_nbv`
- `dynamic_candidate_selection_mode=validated_pareto_diverse`
- `dynamic_candidate_validation_mode=in_process_path_planner_astar_batch`
- `coverage_metric_mode=path_line_plus_endpoint`
- `include_oracle_baselines=true`
- `include_roi_weighted_coverage=true`
- `rollout_steps=40`
- `required_scenario_count=24`
- canonical reward/guard profile v2 不变

除 `dynamic_max_candidates_per_step` 与 `dynamic_proposal_pool_limit_per_step` 外，normalized config 不得漂移。

## Runner 与输出

- runner: `scripts/run_xunce_stage18_7_candidate_count_scaling_audit.py`
- config: `configs/xunce_stage18_7_candidate_count_scaling_audit_v1.json`
- stage id: `xunce-stage18-7-candidate-count-scaling-audit`
- 默认大输出根: `D:\CodexDownloads\lunar-path-planning\stage18_7_candidate_count_scaling_audit\`

产物：

- `xunce-stage18-7-candidate-count-scaling-summary.json`
- `xunce-stage18-7-candidate-count-scaling-results.jsonl`
- `xunce-stage18-7-candidate-count-scaling-command-plan.json`
- `xunce-stage18-7-candidate-count-scaling-report.md`
- `xunce-stage18-7-manifest.json`

## 路由规则

- 缺 sweep root、candidate metric audit 或 Stage 18.6 root：`run_missing_candidate_count_sweeps_with_metric_audit`
- profile hash mismatch、stale root 或非数量字段漂移：`repair_stage18_7_lineage_or_config_drift`
- 36 档成本过高但低档可用：`continue_candidate_count_scaling_with_bounded_budget`
- 高 count 后仍缺 guard-clean candidate：`expand_candidate_generation_roi_complexity`
- guard-clean candidate 出现但 Xunce 不选：`refine_coverage_reward_and_cost_guard`
- Stage 18.6 clean 且 same-candidate-set guard-clean advantage 成立：`prepare_stage19_evaluator_critic_preflight`

无论 route 如何，`stage19_authorized=false`。

## 非目标

不启动 PPO，不发布 checkpoint，不替换 default policy，不连接真实 executor，不启动 canary，不修改 action space/default A*，不把 IRIS/GCS/path-planner 诊断当训练或发布放行证据。
