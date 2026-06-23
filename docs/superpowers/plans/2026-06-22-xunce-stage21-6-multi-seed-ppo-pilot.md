# Stage 21.6 Multi-Seed PPO Pilot Plan

## Summary

Stage 21.6 的目标是验证 Stage 21 纯 PPO 链路在多个 seed 下是否稳定，而不是发布模型。默认采用保守 smoke 预算：3 个 seed，每个 seed 重新执行 Stage21.1 -> Stage21.2 -> Stage21.3 -> Stage21.4 -> Stage21.5，holdout 仍为 2 scenarios x 4 steps。

Stage21.6 不复用同一个 Stage21.1 batch 假装多 seed。每个 seed 必须设置独立 `sampling_seed`，重新采集 Xunce on-policy stochastic transitions，然后通过同一个 coverage-first reward contract、batch validation、tiny PPO update 和 post-update offline trajectory evaluation 合同。

## Inputs

- Stage21.1 base config: `configs/xunce_stage21_1_on_policy_ppo_rollout_collector_v1.json`
- Stage21.2 base config: `configs/xunce_stage21_2_coverage_first_ppo_reward_contract_v1.json`
- Stage21.3 base config: `configs/xunce_stage21_3_ppo_batch_validation_v1.json`
- Stage21.4 base config: `configs/xunce_stage21_4_tiny_ppo_update_smoke_v1.json`
- Stage21.5 base config: `configs/xunce_stage21_5_post_update_offline_trajectory_evaluation_v1.json`
- Stage21.5 prerequisite root:
  `D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\outputs\path_feedback_batch_xunce_stage21_5_post_update_offline_trajectory_evaluation_v1`

## Default Conservative Pilot

- `seed_list=[2101, 2102, 2103]`
- `required_scenario_count=2`
- `rollout_steps=4`
- `dynamic_max_candidates_per_step=36`
- `dynamic_proposal_pool_limit_per_step=288`
- Stage21.4: `epochs=1`
- Stage21.4: `learning_rate=1e-5`
- `min_transition_count_for_performance_claim=200`

大于此范围的 seed 数、scenario 数、step 数、epoch 数或正式 24 x 40 holdout 需要用户确认。

## Runner

新增 runner:

`scripts/run_xunce_stage21_6_multi_seed_ppo_pilot.py`

新增配置:

`configs/xunce_stage21_6_multi_seed_ppo_pilot_v1.json`

默认输出根:

`D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\outputs\path_feedback_batch_xunce_stage21_6_multi_seed_ppo_pilot_v1`

每个 seed 的输出目录:

```text
seed_2101/stage21_1
seed_2101/stage21_2
seed_2101/stage21_3
seed_2101/stage21_4
seed_2101/stage21_5
...
```

## Runner Behavior

1. 校验 Stage21.5 prerequisite 已通过，下一跳为 `implement_stage21_6_multi_seed_ppo_pilot`，并且 release/executor/canary 字段均为 false。
2. 对每个 seed 生成临时 per-seed config：
   - Stage21.1: 覆盖 `sampling_seed`、`dynamic_validation_work_root` 和 smoke scope。
   - Stage21.2: 指向该 seed 的 Stage21.1 root。
   - Stage21.3: 指向该 seed 的 Stage21.1/21.2 roots。
   - Stage21.4: 指向该 seed 的 Stage21.3 root。
   - Stage21.5: 指向该 seed 的 Stage21.4 root，并使用同一 holdout smoke scope。
3. 顺序执行每个 seed，不并行，避免 A*/JSONL IO 干扰。
4. 读取每个 seed 的 Stage21.1/21.3/21.4/21.5 summary、checkpoint audit、gradient audit、loss audit 和 post-evaluation summary。
5. 汇总：
   - seed count
   - per-seed sampling seed、stage root 和 trainable batch fingerprint
   - trainable transition count mean/min/max/total
   - final coverage delta mean/std/variance/min/max
   - coverage AUC delta mean/std/variance/min/max
   - worst seed id
   - KL/entropy/grad norm mean/max
   - parameter delta mean/max
   - hard risk violation total
   - model inference failure、mask violation、unreachable selected、path-planning failure、open-grid fallback totals
   - path cost delta mean/max
   - soft risk exposure delta mean/max
   - checkpoint reload / experimental-only / release boundary totals
   - `sample_count_too_low_for_performance_claim`
6. 每个 seed 必须写入并校验 `stage21_3_batch_fingerprint` 与 `transition_id_fingerprint`。所有 seed 的 `sampling_seed`、Stage21.1 root、batch fingerprint 和 transition id fingerprint 必须唯一；重复视为复用同一个 batch，hard fail。
7. 为测试可控，runner 支持 `execute_seed_pipeline=false`，消费已经存在的 fake per-seed roots。

## Hard Gates

- config 中任何发布、替换 default policy、连接 executor、canary 或授权字段为 true，route 到 `resolve_stage21_6_multi_seed_boundary_rejections`。
- 任一 seed 的 checkpoint reload 失败、checkpoint 不是 experimental-only、或 seed artifact 声称 publish/replace/executor/canary/training authorization，route 到 `resolve_stage21_6_multi_seed_boundary_rejections`。
- 任一 seed 的 model inference failure、mask violation、unreachable selected、path-planning failure、open-grid fallback、hard risk violation 或 safety boundary violation 非 0，route 到 `repair_stage21_6_hard_risk_or_execution_boundary_regression`。
- KL 过大、entropy collapse、grad norm 非有限或超过预算，route 到 `repair_stage21_6_ppo_numerical_stability`。
- seed lineage 或 batch fingerprint 重复，route 到 `repair_stage21_6_seed_lineage_or_batch_reuse`。
- 平均 coverage delta / AUC delta 未正向提升，worst seed 回退，scenario regression 非 0，或 path cost / soft risk 超预算，route 到 `repair_stage21_6_reward_collector_advantage_or_horizon`。
- 只有所有 hard gates 通过，且平均覆盖、AUC、worst seed、成本和风险门禁均通过，才允许 route 到 `prepare_stage22_formal_pure_ppo_training_run`。

## Outputs

- `xunce-stage21-6-multi-seed-ppo-pilot-summary.json`
- `xunce-stage21-6-seed-results.jsonl`
- `xunce-stage21-6-aggregate-metrics.json`
- `xunce-stage21-6-lineage-audit.json`
- `xunce-stage21-6-next-stage-routing.json`
- `xunce-stage21-6-report.md`
- `xunce-stage21-6-manifest.json`

## Boundary

Stage21.6 只运行离线 PPO pilot。即使每个 seed 都生成 experimental checkpoint，也必须保持：

```text
stage21_6_authorized=false
training_or_release_authorized=false
publishes_checkpoint=false
replaces_default_policy=false
connects_real_executor=false
starts_online_canary=false
canary_traffic_fraction=0.0
```

`runs_new_ppo_update=true` 表示所有 seed 的离线 Stage21.4 PPO smoke update 均通过；不代表发布、替换默认策略或真实训练授权。

## Verification

```powershell
python -m pytest tests\test_xunce_stage21_6_multi_seed_ppo_pilot.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage21-6
python -m py_compile scripts\run_xunce_stage21_6_multi_seed_ppo_pilot.py
python scripts\run_stage.py --stage xunce-stage21-6-multi-seed-ppo-pilot --dry-run
```

完整 Stage21 回归：

```powershell
python -m pytest tests\test_xunce_stage21_0_pure_ppo_readiness_audit.py tests\test_xunce_stage21_1_on_policy_ppo_rollout_collector.py tests\test_xunce_stage21_2_coverage_first_ppo_reward_contract.py tests\test_stage21_coverage_first_reward.py tests\test_xunce_stage21_3_ppo_batch_validation.py tests\test_xunce_stage21_4_tiny_ppo_update_smoke.py tests\test_xunce_stage21_5_post_update_offline_trajectory_evaluation.py tests\test_xunce_stage21_6_multi_seed_ppo_pilot.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage21-0-6-final
```

## Current Expected Outcome

在当前保守 smoke 预算下，Stage21.6 可能完成真实 3-seed 离线 PPO pilot，但仍因平均覆盖 / AUC 无提升而失败。若发生该结果，正确下一阶段不是 Stage22，而是：

`repair_stage21_6_reward_collector_advantage_or_horizon`
