# Anchor Projection Contract-Aware Trainable Target v1

## 背景

最新证据根 `outputs/path_feedback_batch_anchor_projection_nontrainable_context_reduction_v1/`
显示 batch 8/8 passed、open-grid fallback=0、provenance mismatch=0，但 readiness 仍为
`needs_training_contract_refinement`，阻塞为
`anchor_projection_nontrainable_contexts_remain`。剩余 blocker 的本质不是审计解释不足，而是
候选生成/选择阶段没有优先产生“默认距离契约内、source-selected、PPO action mask 可消费”的训练目标。

## 目标

新增 opt-in 的 Contract-Aware Trainable Target Generation v1：

- 在 `AnchorProjectionCandidateConfig` 中加入 contract-aware same-action target 开关。
- 对 platform-footprint blocked 的原始 policy target 生成 same-action execution substitute。
- 保持 policy action index 为原 `top_goals` index，`execution_goal_cell` 指向可达 anchor。
- 只有 source-selected、contract-safe、无质量回退且 `ppo_consumable_action=true` 的候选可标记为
  `training_use=trainable_anchor_projection_contrast`。
- 新增 summary gate 报告 PPO 可消费训练目标计数、相对 60/36/48 基线的 delta 与下一步硬阻塞。

## 产物

- `configs/path_feedback_batch_anchor_projection_contract_aware_trainable_target_v1.json`
- `configs/anchor_projection_contract_aware_trainable_target_v1.json`
- `scripts/run_anchor_projection_contract_aware_trainable_target.py`
- `scripts/run_anchor_projection_contract_aware_trainable_target.sh`
- `tests/test_anchor_projection_contract_aware_trainable_target.py`
- 证据根：`outputs/path_feedback_batch_anchor_projection_contract_aware_trainable_target_v1/`

## 验收

- batch summary 必须满足 `failed_count=0`、`open_grid_fallback_used_count=0`。
- candidate/evidence/readiness/contract-aware summary 必须 `status=passed`、`reason_codes=[]`。
- 主成功条件：`ppo_consumable_trainable_target_count > 0`，且
  `nontrainable_blocked_target_count_delta < 0`、
  `candidate_contract_alignment_gap_count=0`。
- 若主成功失败，summary 必须设置
  `next_required_change=action_or_target_contract_change_required`，readiness 继续保留
  `anchor_projection_nontrainable_contexts_remain`。

当前验证预期/结果口径：

- `ppo_consumable_trainable_target_count=18`，证明 same-action substitute 可被 PPO action
  mask 消费。
- `candidate_contract_alignment_gap_count=0`。
- `nontrainable_blocked_target_count=60`，相对 60 基线未下降。
- 因此主成功门槛不成立；summary 必须记录
  `main_success_gate_failures=["nontrainable_blocked_target_count_not_reduced"]`，并设置
  `next_required_change=action_or_target_contract_change_required`。
- readiness review 不得宣称 training ready，必须保留
  `anchor_projection_nontrainable_contexts_remain`。

## 验证命令

```bash
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_batch_path_feedback_validation.sh \
  --matrix configs/path_feedback_batch_anchor_projection_contract_aware_trainable_target_v1.json

PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_anchor_projection_candidate_generation.sh \
  --batch-root outputs/path_feedback_batch_anchor_projection_contract_aware_trainable_target_v1 \
  --config configs/anchor_projection_candidate_generation_v1.json

PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_anchor_projection_evidence_contract.sh \
  --batch-root outputs/path_feedback_batch_anchor_projection_contract_aware_trainable_target_v1 \
  --config configs/anchor_projection_evidence_contract_v1.json

PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_anchor_projection_contract_aware_trainable_target.sh \
  --batch-root outputs/path_feedback_batch_anchor_projection_contract_aware_trainable_target_v1 \
  --config configs/anchor_projection_contract_aware_trainable_target_v1.json

PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_policy_training_readiness_review.sh \
  --batch-root outputs/path_feedback_batch_anchor_projection_contract_aware_trainable_target_v1 \
  --config configs/policy_training_readiness_review_v1.json
```

## 非目标

不启动 PPO；不修改 `training.py` 主训练逻辑；不改变 network/action space/default A*；不放宽默认
2 cells / 1.0 m distance contract；不宣称 Ackermann-feasible trajectory；不把 IRIS/GCS 诊断当训练放行。
