# Stage 21.3 PPO Batch Validation Plan

## Summary

Stage 21.3 的目标是把 Stage 21.1 的 Xunce on-policy transitions 与 Stage 21.2 的 coverage-first reward evaluation 合并成训练前可验证的 PPO batch。该阶段只验证 batch，不训练、不保存可发布 checkpoint、不替换 default policy、不连接 executor、不启动 canary。

## Inputs

- Stage 21.1 collector root
- Stage 21.2 coverage-first reward contract root
- coverage-first PPO reward profile lineage

## Outputs

- `xunce-stage21-3-ppo-batch-validation-summary.json`
- `xunce-stage21-3-ppo-trainable-batch.jsonl`
- `xunce-stage21-3-ppo-batch-splits.json`
- `xunce-stage21-3-return-advantage-audit.jsonl`
- `xunce-stage21-3-lineage-audit.json`
- `xunce-stage21-3-next-stage-routing.json`
- `xunce-stage21-3-report.md`
- `xunce-stage21-3-manifest.json`

## Validation Rules

- Stage21.1 与 Stage21.2 必须都是 `passed`，且 route 分别指向下一阶段。
- Stage21.1 transition 与 Stage21.2 reward row 必须按 `transition_id` 一对一匹配。
- `transition_id` 不允许为空，不允许重复；reward row 也不允许重复。
- `scenario_id`、`step_index`、`done` 必须在 transition 与 reward row 之间一致。
- 每个 scenario 内 `step_index` 必须唯一、单调、连续。
- 每个 scenario 必须恰好一个 terminal transition，且 terminal 必须是最后一条。
- `trainable=true` 必须在 transition 和 reward row 两侧同时成立。
- reward row 不允许包含 `hard_risk_rejected`。
- `action_index` 必须是非负整数，并且 `action_mask[action_index]`、`sampling_mask[action_index]`、`hard_risk_clean_mask[action_index]` 都必须为 true。
- `old_log_prob`、`old_value`、reward、return、advantage 必须 finite。
- `old_log_prob_recompute_abs_error` 必须不超过配置阈值。
- 非 terminal transition 必须有 `next_observation`。
- return/advantage 按 scenario 计算 discounted return，遇到 done 重置。
- train/validation split 按 scenario id 稳定划分；每行 batch JSONL 必须写入 `stage21_3_split`。
- advantage normalization 只能用 train split 的统计量；validation 行不能参与统计量计算。

## Routes

- Boundary flags: `resolve_stage21_3_ppo_batch_validation_boundary_rejections`
- Missing inputs: `rerun_stage21_3_required_inputs`
- Duplicate IDs: `repair_stage21_3_duplicate_lineage_ids`
- Lineage mismatch: `repair_stage21_3_batch_lineage`
- Episode boundary failure: `repair_stage21_3_episode_boundary_contract`
- Non-trainable reward/transition: `repair_stage21_3_reward_trainability_contract`
- Required split missing: `expand_stage21_3_batch_for_train_validation_split`
- Invalid PPO contract: `repair_stage21_3_ppo_batch_contract`
- Passed batch validation: `implement_stage21_4_tiny_ppo_update_smoke`

## Current Smoke Result

当前 Stage21.3 smoke artifact 已通过：

- `trainable_transition_count=8`
- `duplicate_transition_id_count=0`
- `duplicate_reward_transition_id_count=0`
- `reward_trainable_false_count=0`
- `hard_risk_reward_rejection_count=0`
- `invalid_action_count=0`
- `hard_risk_violation_count=0`
- `missing_next_observation_count=0`
- `old_log_prob_recompute_max_abs_error=0.0`
- route: `implement_stage21_4_tiny_ppo_update_smoke`

输出根：

`D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\outputs\path_feedback_batch_xunce_stage21_3_ppo_batch_validation_v1`

## Acceptance Criteria

- Batch validation summary status passed.
- Batch row count equals transition count and reward row count.
- Duplicate transition/reward id count is 0.
- Episode terminal count mismatch is 0.
- Terminal-after-transition count is 0.
- Reward nontrainable count is 0.
- Hard-risk reward rejection count is 0.
- Profile hash mismatch count is 0.
- `return_finite_count == trainable_transition_count`.
- `advantage_finite_count == trainable_transition_count`.
- `invalid_action_count == 0`.
- `hard_risk_violation_count == 0`.
- `old_log_prob_recompute_max_abs_error` remains within threshold.
- Splits are deterministic and non-overlapping.
- Every output batch row includes `stage21_3_split`.
- All training/release/executor/canary fields remain false.
