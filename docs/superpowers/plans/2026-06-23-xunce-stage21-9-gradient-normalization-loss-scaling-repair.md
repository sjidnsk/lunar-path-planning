# Stage 21.9 Gradient Normalization / Loss Scaling Repair

## Summary

Stage 21.9 修复 Stage21.8 暴露的 PPO raw gradient 过大问题。Stage21.8 已证明：降低 learning rate 会减小参数步长，但不会明显降低 `pre_clip_grad_norm`。因此本阶段不继续盲目调 learning rate，而是审计并修复 Stage21.4 的 loss/advantage scale。

## Inputs

```text
D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\outputs\path_feedback_batch_xunce_stage21_8_ppo_update_strength_calibration_v1
D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\stage21_8_sweep_runs_v1\lr2e-6_e1_c0p2_clip1_g25\stage21_6
```

## Outputs

```text
D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\outputs\path_feedback_batch_xunce_stage21_9_gradient_normalization_loss_scaling_repair_v1
```

Artifacts:

```text
xunce-stage21-9-diagnostic-summary.json
xunce-stage21-9-seed-loss-scale-diagnostics.jsonl
xunce-stage21-9-advantage-scale-audit.json
xunce-stage21-9-loss-component-gradient-audit.json
xunce-stage21-9-repaired-stage21-4-config.json
xunce-stage21-9-repaired-stage21-8-config.json
xunce-stage21-9-repaired-smoke-summary.json
xunce-stage21-9-next-stage-routing.json
xunce-stage21-9-report.md
xunce-stage21-9-manifest.json
```

## Repair Contract

- Confirm Stage21.3 train split advantage normalization is applied.
- Record total/policy/value/entropy loss gradient norms in Stage21.4.
- Generate a repaired Stage21.4 config with:
  - `advantage_clip_abs=5.0`
  - `normalize_minibatch_advantages=true`
  - `loss_scale=0.25`
  - `value_loss_coefficient=0.1`
- Execute one bounded Stage21.8 single-combo smoke with 7200 seconds per combo.
- Judge repair from the Stage21.8 sweep result row, not Stage21.4 `status=passed`.

## Routes

```text
rerun_stage21_9_required_inputs
repair_stage21_3_advantage_normalization_contract
repair_stage21_4_loss_component_scaling
repair_stage21_4_gradient_clipping_application
repair_stage21_4_ppo_update_numerics
calibrate_stage21_4_loss_scale_or_learning_rate_for_policy_shift
continue_stage21_9_gradient_normalization_loss_scaling_repair
rerun_stage21_6_multi_seed_pilot_with_stage21_9_repaired_config
```

## Verification

```powershell
python -m pytest tests\test_xunce_stage21_9_gradient_normalization_loss_scaling_repair.py tests\test_xunce_stage21_8_ppo_update_strength_calibration.py tests\test_xunce_stage21_4_tiny_ppo_update_smoke.py tests\test_xunce_stage21_6_multi_seed_ppo_pilot.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage21-9
python -m py_compile scripts\run_xunce_stage21_9_gradient_normalization_loss_scaling_repair.py scripts\run_xunce_stage21_4_tiny_ppo_update_smoke.py scripts\run_xunce_stage21_6_multi_seed_ppo_pilot.py
python scripts\run_stage.py --stage xunce-stage21-9-gradient-normalization-loss-scaling-repair --dry-run
```

## Boundaries

Stage21.9 不启动正式 PPO 训练，不发布 checkpoint，不替换 default policy，不连接真实 executor，不启动 canary，不修改 reward 目标、collector、network、action space 或 default A*，也不把 repaired smoke 宣称为真实性能提升。
