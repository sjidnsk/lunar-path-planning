# Stage 21.12 Coverage-Constrained Reward Profile Repair

## 目标

Stage21.12 修复 Stage21.11 暴露的问题：原 Stage21 coverage-first reward 与 `coverage-per-cost` 排序不够一致，导致 PPO 的训练信号没有稳定推动“覆盖进展相近时选择更低路径成本”的动作。

本阶段只做 reward profile 修复、离线 reward replay、advantage replay 和推荐配置生成。不启动正式 PPO，不发布 checkpoint，不替换 default policy，不连接 executor，不启动 canary，也不修改 network、action space 或 default A*。

## 实现

- 新增 `configs/xunce_stage21_coverage_constrained_ppo_reward_profile_v2.json`。
- 扩展 `model-explorer/src/model_explorer/policy/coverage_first_reward.py`，支持 `xunce-stage21-coverage-constrained-ppo-reward-profile/v2`。
- 新增 `scripts/run_xunce_stage21_12_coverage_constrained_reward_profile_repair.py`。
- 新增 `configs/xunce_stage21_12_coverage_constrained_reward_profile_repair_v1.json`。
- 新增测试 `tests/test_xunce_stage21_12_coverage_constrained_reward_profile_repair.py`。

v2 reward 保留 99% final coverage 为第一目标，同时新增 `coverage_per_cost_component`。该组件只在 hard-risk clean 且 coverage gain 大于 0 时生效，并使用 `coverage_gain / max(path_cost, floor)` 的 log-bounded 信号，避免极端低成本无限放大，也避免原先按 coverage gain 逐候选压顶导致高 coverage-per-cost 候选被压没。

## 当前证据

真实 Stage21.12 输出根：

`D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\outputs\path_feedback_batch_xunce_stage21_12_coverage_constrained_reward_profile_repair_v1`

结果：

- `status=passed`
- `next_required_change=run_stage21_13_coverage_constrained_multi_seed_ppo_smoke`
- `v1_reward_best_matches_coverage_per_cost_rate=0.20833333333333334`
- `v2_reward_best_matches_coverage_per_cost_rate=0.29583333333333334`
- `v2_alignment_improvement=0.0875`
- `v2_coverage_priority_violation_rate=0.020833333333333332`
- `v2_advantage_coverage_per_cost_correlation=0.318183174323286`
- `v2_hard_risk_trainable_count=0`
- `runs_new_ppo_update=false`

## 下一跳

Stage21.12 通过只说明新的 reward/advantage replay 信号可用于下一轮 pilot。下一阶段应执行 Stage21.13：使用推荐 Stage21.2/Stage21.6 config 运行 coverage-constrained multi-seed PPO smoke，验证稳定 reward 信号是否能转化为覆盖率或 coverage AUC 的实际提升。

即使 Stage21.13 有提升，也仍不能发布 checkpoint、替换 default policy、连接 executor 或启动 canary。
