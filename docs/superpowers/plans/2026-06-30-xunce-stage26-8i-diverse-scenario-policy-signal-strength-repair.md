# Stage26.8I Diverse Scenario Policy Signal Strength Repair

目标：在 Stage26.8H 已证明 diverse scenario eval binding 干净的基础上，复用 Stage26.8G `h16_seed260801/s26_1` batch 做受控 PPO update-strength sweep，判断 policy delta 是否足以跨过 argmax margin。

## 合同

- 不重跑 Stage26.1 collector，不重新生成 synthetic terrain。
- 只调 Stage26.2 / Stage21.4 update 强度；不改 reward、network、Hybrid A*、candidate generation。
- 只有 policy KL、loss/gradient、checkpoint boundary 稳定的 combo 才进入 Stage26.8H 可恢复 eval。
- 主指标仍为 `main_coverage_per_100m_delta`；AUC、总路程、Hybrid A* path cost 只作诊断。
- release/default-policy/executor/canary 全部保持 false/0。

## 默认 sweep

- `current_repro`: epochs=4, lr=1e-5, policy=1.0, value=0.02, entropy=0.01, loss_scale=0.25
- `lr_x2`: epochs=4, lr=2e-5, policy=1.0, value=0.02, entropy=0.01, loss_scale=0.25
- `depth_x2`: epochs=8, lr=1e-5, policy=1.0, value=0.02, entropy=0.01, loss_scale=0.25
- `policy_amp`: epochs=8, lr=2e-5, policy=2.0, value=0.01, entropy=0.005, loss_scale=0.5
- `value_off_probe`: epochs=4, lr=1e-5, policy=1.0, value=0.0, entropy=0.01, loss_scale=0.25

## 审计

- update sweep：KL、loss/gradient、checkpoint、boundary、parameter delta。
- margin audit：top1/top2 prob/logit margin、margin closure、estimated updates to cross margin。
- stable eval audit：selected action/viewpoint/theta change、main coverage per 100m delta、safety/fallback。

## 路由

- 输入不可信：`rerun_stage26_8i_required_inputs`
- Stage26.8H binding/safety 不干净：`repair_stage26_8h_eval_binding_or_safety`
- 所有 combo 不稳定：`repair_stage26_8i_update_stability`
- margin 字段缺失：`repair_stage26_8i_policy_margin_audit_binding`
- 概率仍极小且 margin 很大：`increase_stage26_synthetic_update_strength_or_sample_count`
- 概率变大但 action 不变：`calibrate_stage26_synthetic_discrete_margin_crossing_after_credit`
- action 变了但 per100m 负向：`repair_stage26_synthetic_credit_assignment`
- action 变了且 per100m 非负：`resume_stage26_8d_seed_horizon_jobs_with_diverse_scenarios`

## 验证

```powershell
python -m pytest tests\test_xunce_stage26_8i_diverse_scenario_policy_signal_strength_repair.py tests\test_xunce_stage26_8h_resumable_diverse_scenario_post_update_eval.py tests\test_xunce_stage26_7g_repair_behavior_policy_kl_baseline.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage26-8i
python -m py_compile scripts\run_xunce_stage26_8i_diverse_scenario_policy_signal_strength_repair.py scripts\run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke.py scripts\run_xunce_stage26_8h_resumable_diverse_scenario_post_update_eval.py
python scripts\run_stage.py --stage xunce-stage26-8i-diverse-scenario-policy-signal-strength-repair --dry-run
```
