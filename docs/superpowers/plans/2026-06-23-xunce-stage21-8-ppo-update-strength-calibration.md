# Stage 21.8 PPO Update Strength Calibration

## Summary

Stage 21.8 用于校准 Stage21.7 repaired pilot 暴露出的 PPO 更新强度问题。Stage21.7 已把样本量扩大到 240 transitions，但 repaired pilot 三个 seed 全部 `grad_unstable`，`pre_clip_grad_norm_mean` 约为 41.98，`pre_clip_grad_norm_max` 约为 46.73，高于 Stage21.6 当前 pre-clip 稳定性门槛 25.0。

本阶段不是正式 PPO 训练，也不发布 checkpoint。它只做 bounded calibration sweep，寻找可再次尝试 Stage21.6 multi-seed pilot 的稳定 PPO update 配置。

## Inputs And Outputs

输入：

```text
D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\outputs\path_feedback_batch_xunce_stage21_7_reward_collector_advantage_horizon_repair_v1
D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\outputs\path_feedback_batch_xunce_stage21_7_repair_attempt_v1
```

输出：

```text
D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\outputs\path_feedback_batch_xunce_stage21_8_ppo_update_strength_calibration_v1
```

主要 artifacts：

```text
xunce-stage21-8-calibration-summary.json
xunce-stage21-8-sweep-results.jsonl
xunce-stage21-8-gradient-stability-audit.json
xunce-stage21-8-policy-shift-audit.json
xunce-stage21-8-recommended-stage21-6-config.json
xunce-stage21-8-next-stage-routing.json
xunce-stage21-8-report.md
xunce-stage21-8-manifest.json
```

## Calibration Rules

- 固定 repaired pilot scope：3 seeds、8 scenarios、10 rollout steps、36 candidates、288 proposal pool。
- Sweep 只改变 PPO update strength：learning rate、epochs、clip ratio、Stage21.4 train-time clip norm、Stage21.6 pre-clip grad gate。
- Stage21.4 的 `max_grad_norm` 是训练时 `clip_grad_norm_` 的裁剪阈值。
- Stage21.6/21.8 的 pre-clip grad gate 是审计门槛，用来判断原始梯度是否过大。
- 两个阈值必须分开写入和审计，不能混用。
- 缺少 seed 阶段状态、batch fingerprint、transition fingerprint、post-clip grad norm 时不能推荐配置。
- 任一 hard risk、mask violation、path-planning failure、open-grid fallback、checkpoint/release/canary 违规必须 hard fail。
- 推荐组合必须来自 Stage21.6 `status=passed` 的 sweep，且满足梯度稳定、KL/entropy 正常、policy shift 可观测、coverage/AUC 不退化。
- Runtime-blocked combo 默认不会自动重试；如需重试，必须显式设置 `retry_runtime_blocked_combinations=true` 或换新的 `combo_id` / `sweep_work_root`。

## Verification

```powershell
python -m pytest tests\test_xunce_stage21_8_ppo_update_strength_calibration.py tests\test_xunce_stage21_7_reward_collector_advantage_horizon_repair.py tests\test_xunce_stage21_6_multi_seed_ppo_pilot.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage21-8
python -m py_compile scripts\run_xunce_stage21_8_ppo_update_strength_calibration.py scripts\run_xunce_stage21_6_multi_seed_ppo_pilot.py scripts\run_xunce_stage21_4_tiny_ppo_update_smoke.py
python scripts\run_stage.py --stage xunce-stage21-8-ppo-update-strength-calibration --dry-run
```

## Boundaries

Stage21.8 不启动正式 PPO 训练，不发布 checkpoint，不替换 default policy，不连接真实 executor，不启动 canary，不修改 network/action space/default A*，不把 calibration smoke 宣称为真实性能提升。
