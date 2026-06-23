# Stage 21.2 Coverage-First PPO Reward Contract Plan

## Summary

Stage 21.2 的目标是为纯 PPO 主线新增 coverage-first reward contract。它不覆盖 canonical v3 历史 profile，也不训练 PPO。Stage 21.2 读取 Stage 21.1 on-policy trainable transitions，用新的 coverage-first PPO reward profile 重新计算 reward components，验证 profile hash、hard-risk 失败语义、99% 覆盖目标路由和边界字段。

## Design

新增独立 helper：

- `model-explorer/src/model_explorer/policy/coverage_first_reward.py`

新增 profile：

- `configs/xunce_stage21_coverage_first_ppo_reward_profile_v1.json`

新增 runner：

- `scripts/run_xunce_stage21_2_coverage_first_ppo_reward_contract.py`

Profile 不复用 canonical v3 schema，避免影响 Stage 18-20 的历史证据。coverage-first profile 包含：

- `target_final_coverage_rate=0.99`
- `weights.step_coverage_gain`
- `weights.coverage_progress`
- `weights.final_coverage`
- `weights.success_99pct`
- `weights.path_cost`
- `weights.soft_risk`
- `weights.failure`
- `weights.hard_risk_failure`
- normalizers
- `risk_policy.path_cost_includes_risk_proxy=true`

Reward components 固定为：

- `step_coverage_gain_component`
- `coverage_progress_component`
- `final_coverage_bonus_component`
- `success_99pct_bonus_component`
- `path_cost_component`
- `soft_risk_component`
- `failure_component`
- `hard_risk_component`

## Rules

- 覆盖率是第一目标；99% final coverage 未达到时不能因为路径成本下降宣称成功。
- hard risk 不进入可抵消的普通 reward；如果 hard risk 触发且 reward 为正，`hard_risk_component` 必须把总 reward 压到 `<=0`。
- 如果 `path_cost_includes_risk_proxy=true`，`soft_risk_component` 必须保持 tiny/audit-weighted，不能重复大额扣分。
- Stage 21.2 不改 Stage 21.1 原始 batch，只输出 reward contract evaluation artifacts。

## Outputs

- `xunce-stage21-2-coverage-first-reward-summary.json`
- `xunce-stage21-2-reward-contract-evaluation.jsonl`
- `xunce-stage21-2-profile-audit.json`
- `xunce-stage21-2-next-stage-routing.json`
- `xunce-stage21-2-report.md`
- `xunce-stage21-2-manifest.json`

## Routes

- 输入缺失、profile mismatch、Stage21.1 未通过：`rerun_stage21_2_required_inputs`
- component set 或 hash 不稳定：`repair_stage21_2_coverage_first_reward_contract`
- component set、profile hash、hard risk 或 reward finite 检查失败：`repair_stage21_2_coverage_first_reward_contract`
- 40 步证据无法接近 99%：`stage21_12_rollout_horizon_or_mission_budget_scaling_for_99pct_coverage`
- contract 通过：`implement_stage21_3_ppo_batch_validation`

## Acceptance Criteria

- profile hash 稳定，不包含路径、时间、输出目录、git 状态。
- component set 固定且 signed。
- hard risk 不能被 coverage reward 抵消。
- path cost 与 soft risk 不重复大额扣分。
- Stage21.1 batch 可用新 profile 重算 reward components。
- 所有 training/release/executor/canary 字段保持 false。
- registry dry-run 支持 Stage21.2。
