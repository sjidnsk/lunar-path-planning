# Stage26.8O Repair Aggressive Collector Trainable Sample Budget

目标：修复 Stage26.8N 暴露的 aggressive collector 有效训练样本严重不足问题。当前主因是连续 theta 下 selected `(x,y,theta)` 经常 Hybrid A* 不可达，导致 episode 在 step0 或早期终止，最终只得到 3 条 trainable transition。

## 范围

- 只修采样端有效样本生成：Stage21.1 collector、Stage21.3 batch gate、Stage26.8O runner/config/test。
- Stage26.8O 只跑 `Stage26.1 -> Stage21.1/21.2/21.3`，不跑 PPO update，不跑 post-update eval。
- 不改 reward、network、Hybrid A* 搜索语义、candidate generation、synthetic terrain。
- 不发布 checkpoint、不替换 default policy、不连接 executor、不启动 canary。

## 核心合同

- 新增 opt-in：
  - `selected_continuous_theta_reachability_guard_enabled=true`
  - `selected_continuous_theta_unreachable_resample_policy=reachable_theta_proposal/v1`
- 对 policy-selected 和 synthetic-credit-selected action 都适用。
- 若 selected theta 不可达，先在同一个 selected `(x,y)` 的 reachable theta proposals 中重采样。
- 若该 `(x,y)` 全部 theta 不可达，再从 sampling mask 中选择下一个 Hybrid A* reachable candidate。
- 若全候选不可达，才 terminal，并写 `no_selected_reachable_pose_candidate_terminal`。
- resampled theta/action 是真实 selected action，不是 counterfactual；必须写 behavior point/theta/total logprob，并由 Stage21.3 重算。

## Stage26.8O Runner

输入：

- Stage26.8N root 必须存在，且 route 为 `expand_stage26_8n_aggressive_sample_budget`。
- 复现 Stage26.8N aggressive sample：H16、seed=260801、scenario_count=6、rollout_steps=20、min_trainable_transition_count=100、worker=4。

输出 root：

```text
D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/outputs/path_feedback_batch_xunce_stage26_8o_repair_aggressive_collector_trainable_sample_budget_v1
```

主要 artifacts：

- `xunce-stage26-8o-summary.json`
- `xunce-stage26-8o-stage26-1-config.json`
- `xunce-stage26-8o-stage26-1-summary.json`
- `xunce-stage26-8o-reachability-guard-audit.json`
- `xunce-stage26-8o-recommended-stage26-8n-config.json`
- `xunce-stage26-8o-next-stage-routing.json`
- `xunce-stage26-8o-report.md`
- `xunce-stage26-8o-manifest.json`

## 路由

- 输入缺失或 Stage26.8N route 不匹配：`rerun_stage26_8o_required_inputs`
- guard 未生效：`repair_stage26_8o_selected_theta_reachability_guard`
- behavior logprob 不可重算：`repair_stage26_8o_behavior_logprob_contract`
- repaired collector 仍少于 100 rows：`expand_stage26_8o_sample_budget_or_start_pool`
- binding/safety/fallback 失败：`repair_stage26_8o_collector_binding_or_safety`
- collector clean 且 rows>=100：`rerun_stage26_8n_aggressive_update_sweep_with_repaired_collector`
- boundary 打开：`resolve_stage26_8o_boundary_rejections`

## 验证

```powershell
python -m pytest tests\test_xunce_stage26_8o_repair_aggressive_collector_trainable_sample_budget.py tests\test_xunce_stage21_1_on_policy_ppo_rollout_collector.py tests\test_xunce_stage21_3_ppo_batch_validation.py tests\test_xunce_stage26_8n_aggressive_sample_update_sweep.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage26-8o
python -m py_compile scripts\run_xunce_stage26_8o_repair_aggressive_collector_trainable_sample_budget.py scripts\run_xunce_stage21_1_on_policy_ppo_rollout_collector.py scripts\run_xunce_stage21_3_ppo_batch_validation.py
python scripts\run_stage.py --stage xunce-stage26-8o-repair-aggressive-collector-trainable-sample-budget --dry-run
python scripts\run_stage.py --stage xunce-stage26-8o-repair-aggressive-collector-trainable-sample-budget
```
