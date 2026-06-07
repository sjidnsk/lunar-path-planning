# Anchor-Projection Evidence Contract v1

## 背景

`Platform-Aware Target/Anchor Contract v1` 已经能在 `platform_goal_feasibility/v1`
中区分 `policy_target_cell`、`execution_goal_cell` 和 `anchor_projection`。最新
current-HEAD evidence 根目录为
`outputs/path_feedback_batch_anchor_projection_contract_v1/`，旧根目录
`outputs/path_feedback_batch_platform_goal_alignment_v1/` 在提交后会触发
`current_git_provenance_mismatch`，不再作为训练输入。

当前训练阻塞不是 batch 跑不通：8 个 path-feedback run 全部 passed，
`open_grid_fallback_used_count=0`。阻塞点是 39 个
`platform_inflated_goal_blocked` 目标都只有 audit projection anchor，不能直接升级为
positive training evidence 或 negative evidence。

## 本阶段实现

本阶段新增 audit-only summary：

- config：`configs/anchor_projection_evidence_contract_v1.json`
- runner：`scripts/run_anchor_projection_evidence_contract.py`
- wrapper：`scripts/run_anchor_projection_evidence_contract.sh`
- test：`tests/test_anchor_projection_evidence_contract.py`
- summary schema：`anchor-projection-evidence-contract-summary/v1`

summary 消费 `goal-blocked-evidence-regeneration-summary/v1` 与
`policy-training-readiness-review-summary/v1`，不修改
`path-planner-route/v1`、`model-explorer-contract/v1`、`path-feedback-summary/v1` 的稳定字段语义。

## 契约规则

一个 `platform_inflated_goal_blocked` record 只有同时满足下列条件，才可判定为
`trainable_anchor_projection_contrast`：

- 有 `nearest_inflated_passable_anchor`。
- projection distance 不超过 config 中的米制和 cell 制阈值。
- `anchor_reachable=true`。
- `anchor_projection.training_use` 属于允许训练值。
- comparison scope 属于 projected-target training scope。
- comparison scope 不属于 `audit_proxy_anchor_not_same_cell`。

否则 record 保持 `nontrainable_blocked_target`，`sample_weight=0.0`，
`training_use=not_positive_evidence`。`unknown_contract_mismatch` 会进入 `unresolved`；
本阶段验收要求 `platform_goal_unresolved_count=0`。

## Current-HEAD Evidence

新 evidence root：
`outputs/path_feedback_batch_anchor_projection_contract_v1/`

关键计数：

- `current_git_provenance_mismatch_count=0`
- `git_provenance_mismatch_count=0`
- `platform_goal_contract_mismatch_count=39`
- `trainable_anchor_projection_count=0`
- `nontrainable_blocked_target_count=39`
- `platform_goal_anchor_available_count=39`
- `platform_goal_unresolved_count=0`
- `positive_training_evidence_contains_audit_proxy_anchor_count=0`
- `eligible_negative_evidence_candidate_count=0`
- `recommended_next_action=keep_platform_blocked_targets_out_of_training`

结论：当前 39 条 anchor projection 仍全部是 audit-only blocked target evidence。
这不支持 `source policy target selection` 已改善，也不支持启动 PPO。

## 非目标

- 不启动 PPO。
- 不修改 network architecture 或 action space。
- 不修改 default A* 语义。
- 不把 channel-aware opt-in audit 说成默认 route replacement。
- 不实现完整 GCS graph search。
- 不使用 C-space IRIS、IrisNp、IrisNp2 或 IrisZo。
- 不宣称 Ackermann-feasible trajectory。
