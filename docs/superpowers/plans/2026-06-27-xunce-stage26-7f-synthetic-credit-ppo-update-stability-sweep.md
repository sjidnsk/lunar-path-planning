# Stage26.7F Synthetic Credit PPO Update Stability Sweep

## 目标

Stage26.7D 已经修复 synthetic credit sampler 的连续 theta 可达性，但 Stage21.4 tiny PPO update 的 `post_update_approx_kl=2.698` 超过 `max_abs_approx_kl=1.5`。Stage26.7F 的目标是只调离线 PPO update 强度，不改 reward、network、Hybrid A*、candidate generation 或 synthetic terrain，找到 KL 稳定且 checkpoint 可 reload 的 experimental-only combo。

## 输入

- Stage26.7D root:
  `D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/outputs/path_feedback_batch_xunce_stage26_7d_repair_synthetic_credit_sampler_continuous_theta_reachability_v1`
- 必须满足：
  - `next_required_change=repair_stage26_7_credit_ppo_update_stability`
  - `trainable_transition_count=36`
  - `synthetic_credit_target_selected_count=36`
  - `selected_continuous_theta_hybrid_astar_unreachable_count=0`
  - `behavior_theta_logprob_recomputable=true`

## 执行边界

- 不重跑 collector，默认复用 Stage26.7D 短链路 `r/c1/s26_1`。
- 每个 combo 只运行 Stage26.2 update。
- 只有 Stage26.2 `passed`、checkpoint reload passed、`experimental_only=true` 且 `final_post_update_approx_kl<=1.5` 的 combo 才运行 Stage26.3 eval。
- `max_abs_approx_kl` 固定为 `1.5`，不得通过放宽阈值伪造稳定。
- `hybrid_astar_path_cost_delta` 只作为诊断，不作为硬失败；探索效率按 `main_coverage_per_100m_delta` 判断。

## 默认 Sweep

| combo | epochs | learning_rate | policy | value | loss_scale |
|---|---:|---:|---:|---:|---:|
| current_repro | 4 | 1e-5 | 1.0 | 0.02 | 0.25 |
| lr_half | 4 | 5e-6 | 1.0 | 0.02 | 0.25 |
| shallow_lr_half | 2 | 5e-6 | 1.0 | 0.02 | 0.25 |
| conservative | 1 | 5e-6 | 1.0 | 0.02 | 0.25 |
| ultra_conservative | 1 | 2.5e-6 | 1.0 | 0.02 | 0.25 |
| policy_soft | 2 | 5e-6 | 0.5 | 0.02 | 0.25 |

## 审查门

1. Config/sweep contract review: 只调 Stage21.4 update 强度，不改 reward/network/Hybrid A*/synthetic terrain，不放宽 KL。
2. Update result review: 稳定 combo 判定必须基于 KL、finite loss/grad、checkpoint reload 和 boundary。
3. Post-update eval review: 结论按 `main_coverage_per_100m_delta`，不把总路程增加当硬失败。

## 输出

Root:
`D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/outputs/path_feedback_batch_xunce_stage26_7f_synthetic_credit_ppo_update_stability_sweep_v1`

Artifacts:

- `xunce-stage26-7f-summary.json`
- `xunce-stage26-7f-update-sweep-results.jsonl`
- `xunce-stage26-7f-kl-stability-audit.json`
- `xunce-stage26-7f-checkpoint-audit.json`
- `xunce-stage26-7f-post-update-eval-audit.json`
- `xunce-stage26-7f-recommended-stage26-2-config.json`
- `xunce-stage26-7f-recommended-stage26-3-config.json`
- `xunce-stage26-7f-next-stage-routing.json`
- `xunce-stage26-7f-report.md`
- `xunce-stage26-7f-manifest.json`

## 路由

- 输入不可信：`rerun_stage26_7f_required_inputs`
- 更新前 KL baseline 已超阈：`repair_stage26_7_behavior_policy_kl_baseline`
- 所有 combo KL 仍超阈：`reduce_stage26_7_credit_update_strength`
- loss/gradient 非有限：`repair_stage26_7_credit_ppo_update_numerics`
- checkpoint reload/boundary 失败：`repair_stage26_7_credit_checkpoint_boundary`
- eval binding 失败：`repair_stage26_7_credit_post_update_eval_binding`
- 稳定 checkpoint 下 action 不变：`calibrate_stage26_synthetic_discrete_margin_crossing_after_credit`
- action 变了但 `main_coverage_per_100m_delta<0`：`repair_stage26_synthetic_credit_assignment`
- 主覆盖/AUC/单位路程覆盖改善且无 regression：`run_stage26_8_synthetic_terrain_multi_seed_ppo_pilot`
