# Stage21.10 Stage21.9 Repaired Multi-Seed PPO Pilot

## 目标

Stage21.10 用 Stage21.9 修复后的 PPO loss/advantage scale 配置重跑一次 Stage21.6 multi-seed PPO pilot，判断梯度稳定后 final coverage 和 coverage AUC 是否开始提升。

## 输入

- Stage21.9 root: `D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\outputs\path_feedback_batch_xunce_stage21_9_gradient_normalization_loss_scaling_repair_v1`
- Stage21.9 repaired Stage21.4 config: `xunce-stage21-9-repaired-stage21-4-config.json`
- Stage21.7 repaired Stage21.6 base config: `D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\outputs\path_feedback_batch_xunce_stage21_7_reward_collector_advantage_horizon_repair_v1\xunce-stage21-7-repaired-stage21-6-config.json`

## 输出

默认输出根：

```text
D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\outputs\path_feedback_batch_xunce_stage21_10_stage21_9_repaired_multi_seed_ppo_pilot_v1
```

主要产物：

- `xunce-stage21-10-summary.json`
- `xunce-stage21-10-repaired-stage21-4-config.json`
- `xunce-stage21-10-repaired-stage21-6-config.json`
- `xunce-stage21-10-stage21-6-result-summary.json`
- `xunce-stage21-10-next-stage-routing.json`
- `xunce-stage21-10-report.md`
- `xunce-stage21-10-manifest.json`
- nested Stage21.6 root: configured by `stage21_6_output_subdir`; the default Windows-safe config uses `s6/` to avoid 260-character path limits.

## 固定配置

Stage21.10 会生成专用 Stage21.4 config：

```text
learning_rate=2e-6
epochs=1
clip_ratio=0.2
max_grad_norm=1.0
advantage_clip_abs=5.0
normalize_minibatch_advantages=true
loss_scale=0.25
value_loss_coefficient=0.1
```

Stage21.6 repaired pilot 固定：

```text
seed_list=[2101,2102,2103]
required_scenario_count=8
rollout_steps=10
dynamic_max_candidates_per_step=36
dynamic_proposal_pool_limit_per_step=288
stage21_6_pre_clip_grad_norm_gate=25.0
```

## 判定规则

- Stage21.9 不是 `passed` 或 route 不对：`rerun_stage21_10_required_inputs`
- 边界字段打开：`resolve_stage21_10_boundary_rejections`
- Stage21.6 lineage 不通过：`rerun_stage21_10_required_inputs`
- pre-clip grad 超过 25、KL/entropy/post-clip 异常：`continue_stage21_9_gradient_normalization_loss_scaling_repair`
- 梯度稳定但 raw 或 capped coverage/AUC 没有同时提升：`repair_stage21_reward_signal_or_advantage_separation`
- raw 与 capped coverage/AUC 都提升，worst seed 不回退，边界为 0：`scale_stage21_ppo_pilot_scenarios_and_horizon`

## 非目标

Stage21.10 不是正式 PPO 训练，不发布 checkpoint，不替换 default policy，不连接真实 executor，不启动 canary，不修改 reward 目标、collector、network、action space 或 default A*。

## 执行结果

Stage21.10 已在 Windows-safe nested root `s6/` 完成 repaired Stage21.6 pilot。结果为 `status=failed`，`next_required_change=repair_stage21_reward_signal_or_advantage_separation`。三 seed 与 240 条 transition 均完成，lineage 通过，hard risk 与执行边界为 0，`pre_clip_grad_norm_max=2.334965467453003`，但 raw/capped final coverage 与 coverage AUC delta 均为 0.0。
