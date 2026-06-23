# Stage 21.7 Reward / Collector / Advantage / Horizon Repair

## Summary

Stage 21.7 修复 Stage 21.6 的当前 blocker：3 个 seed 都完成离线 PPO update，但 `mean_final_coverage_delta=0.0` 且 `mean_coverage_auc_delta=0.0`。本阶段不是正式训练或发布，而是诊断 reward、collector、advantage、PPO update 强度和 holdout horizon，生成一个可重跑 Stage21.6 的 repaired smoke 配置，并执行一次 repaired pilot smoke 或记录 runtime blocker。

输入根：

```text
D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\outputs\path_feedback_batch_xunce_stage21_6_multi_seed_ppo_pilot_v1
```

输出根：

```text
D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\outputs\path_feedback_batch_xunce_stage21_7_reward_collector_advantage_horizon_repair_v1
```

## Diagnostics

Stage21.7 必须输出五类审计：

- reward signal：reward 标准差、coverage reward 非零率、reward 与 coverage gain 的相关性。
- advantage signal：advantage 标准差、正负比例、advantage 与 coverage gain 的相关性。
- sample count：总 trainable transition 是否低于 200。
- PPO update strength：KL、parameter delta、pre/post action/probability/logit shift。
- holdout horizon：当前 4-step holdout 是否过短，不能说明 40-step/99% 覆盖能力。

任何 hard risk、mask violation、path planning failure、open-grid fallback、checkpoint release、default policy、executor 或 canary 边界违规都必须 hard fail。

## Repair

默认 repaired Stage21.6 config：

```text
seed_list=[2101,2102,2103]
required_scenario_count=8
rollout_steps=10
epochs=2
learning_rate=2e-5
dynamic_max_candidates_per_step=36
dynamic_proposal_pool_limit_per_step=288
```

该配置目标是把样本量提高到约 `3*8*10=240`，并让 holdout 更容易暴露 coverage/AUC 变化。输出必须写入 D 盘的新 root，不覆盖原 Stage21.6。

## Runner

```text
scripts/run_xunce_stage21_7_reward_collector_advantage_horizon_repair.py
configs/xunce_stage21_7_reward_collector_advantage_horizon_repair_v1.json
```

主要产物：

```text
xunce-stage21-7-diagnostic-summary.json
xunce-stage21-7-seed-diagnostics.jsonl
xunce-stage21-7-reward-signal-audit.json
xunce-stage21-7-advantage-signal-audit.json
xunce-stage21-7-policy-shift-audit.json
xunce-stage21-7-repaired-stage21-6-config.json
xunce-stage21-7-repaired-stage21-6-result-summary.json
xunce-stage21-7-next-stage-routing.json
xunce-stage21-7-report.md
xunce-stage21-7-manifest.json
```

## Verification

```powershell
python -m pytest tests\test_xunce_stage21_7_reward_collector_advantage_horizon_repair.py tests\test_xunce_stage21_6_multi_seed_ppo_pilot.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage21-7
python -m py_compile scripts\run_xunce_stage21_7_reward_collector_advantage_horizon_repair.py scripts\run_xunce_stage21_6_multi_seed_ppo_pilot.py
python scripts\run_stage.py --stage xunce-stage21-7-reward-collector-advantage-horizon-repair --dry-run
```

## Boundaries

Stage21.7 不启动正式 PPO 训练，不发布 checkpoint，不替换 default policy，不连接真实 executor，不启动 canary，不修改 network/action space/default A*，不把 smoke 结果宣称为真实性能提升。
