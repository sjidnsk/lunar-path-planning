# Stage 21.1 Xunce On-Policy PPO Rollout Collector Plan

## Summary

Stage 21.1 的目标是补齐纯 PPO 主线的第一段真实数据合同：让 Xunce 在高保真动态候选集上按随机策略采样动作，并输出可供后续 PPO batch validation 使用的 transition。此阶段只采集数据，不训练、不发布、不替换 default policy、不连接 executor、不启动 canary。

## Scope

- 新增只读 collector runner：`scripts/run_xunce_stage21_1_on_policy_ppo_rollout_collector.py`。
- 新增配置：`configs/xunce_stage21_1_on_policy_ppo_rollout_collector_v1.json`。
- 复用 high-fidelity 动态候选生成、coverage helper、Xunce batch adapter 和 Xunce checkpoint loader。
- 不复用现有 evaluator 的 `_score_xunce_model()` 选择逻辑，因为它是 argmax。
- 采样使用 `torch.distributions.Categorical(logits=masked_logits)`，并叠加 `hard_risk_clean_mask`。
- 输出 transitions、episodes、trainable batch、sampling audit、reward audit、rejection report、summary/routing/report/manifest。

## Contract

每条 trainable transition 必须包含：

- `observation`
- `xunce_batch`
- `action_index`
- `old_log_prob`
- `old_value`
- `reward`
- `next_observation`
- `next_xunce_batch`
- `done`
- `info`

训练样本只允许：

- `action_mask[action_index] == true`
- `sampling_mask[action_index] == true`
- selected candidate 无 hard risk violation
- `old_log_prob`、`old_value`、`reward` 有限

## Acceptance Criteria

- Stage21.1 summary status passed。
- `trainable_transition_count >= min_trainable_transition_count`。
- `mask_violation_count == 0`。
- `hard_risk_violation_count == 0`。
- `non_finite_old_log_prob_count == 0`。
- `non_finite_old_value_count == 0`。
- `non_finite_reward_count == 0`。
- `old_log_prob_recompute_max_abs_error <= max_log_prob_recompute_abs_error`。
- 所有 training/release/executor/canary 字段保持 false。
- Stage registry dry-run 支持 `xunce-stage21-1-on-policy-ppo-rollout-collector`。

## Review Notes

子智能体只读审核指出两个 Critical 风险，计划已纳入：

- 不能直接用 `_scenario_to_model_inputs()` 的 `action_mask` 采样，必须叠加 hard-risk-clean mask。
- 不能复用 `_score_xunce_model()`，因为它内部使用 argmax，会破坏 stochastic on-policy 语义。

## Next Route

通过后进入：

`implement_stage21_2_coverage_first_ppo_reward_contract`
