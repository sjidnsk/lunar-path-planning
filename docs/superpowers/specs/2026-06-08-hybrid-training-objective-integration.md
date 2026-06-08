# Hybrid Training Objective Integration v1

## Summary

证据根：
`outputs/path_feedback_batch_planner_validated_trainable_target_mining_v1/`。

本阶段新增 opt-in hybrid dry-run，将 24 个 `RolloutEpisode` action-label positive 与 54 个 pairwise
preference/negative 信号接入同一个本地训练 smoke。该阶段只验证联合 objective 可消费统一样本体系；
不启动正式 PPO，不发布 checkpoint，不宣称策略性能提升。

## Implementation

新增产物：

- `configs/hybrid_policy_training_dry_run_v1.json`
- `scripts/run_hybrid_policy_training_dry_run.py`
- `scripts/run_hybrid_policy_training_dry_run.sh`
- `tests/test_hybrid_policy_training_dry_run.py`

Readiness review 扩展：

- 新增 `--hybrid-policy-training-dry-run-summary`
- 支持读取 `hybrid-policy-training-dry-run-summary/v1`
- 通过时 `training_readiness_status=hybrid_training_dry_run_completed`
- 不输出正式 PPO ready 或 performance claim

Hybrid objective：

- Action-label loss：使用现有 `RolloutEpisode` / `PolicyObservation` / `action_index` 样本。
- Pairwise loss：使用 unified registry 中 24 个既有 preference pair 与 30 个 residual negative/boundary pair。
- Loss 权重由配置控制：`action_label_loss_weight`、`preference_loss_weight`、`residual_negative_loss_weight`。
- 默认 1 epoch、本地、固定 seed、`checkpoint_path=null`。

## Current Evidence

`hybrid-policy-training-dry-run-summary.json` 当前结果：

- `status=passed`
- `reason_codes=[]`
- `dry_run_status=passed`
- `action_label_positive_count=24`
- `existing_preference_pair_count=24`
- `residual_preference_pair_count=30`
- `pairwise_preference_signal_count=54`
- `hybrid_train_signal_count=78`
- `hard_positive_added_count=0`
- `invalid_action_mask_count=0`
- `empty_action_mask_count=0`
- `publishes_checkpoint=false`
- `performance_claimed=false`

## Validation

```bash
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_hybrid_policy_training_dry_run.sh \
  --batch-root outputs/path_feedback_batch_planner_validated_trainable_target_mining_v1 \
  --config configs/hybrid_policy_training_dry_run_v1.json

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/home/kai/anaconda3/envs/lunar-explorer/bin/python -m pytest -q \
  model-explorer/tests/test_model_explorer.py \
  tests/test_counterfactual_preference_training_samples.py \
  tests/test_limited_policy_training_dry_run_input_materialization.py \
  tests/test_unified_policy_sample_registry.py \
  tests/test_hybrid_policy_training_dry_run.py \
  tests/test_policy_training_readiness_review.py
```

## Boundaries

`hybrid_train_signal_count=78` 是联合 dry-run 训练信号数，不是 78 个 PPO hard positive。
`action_label_positive_count` 保持 24，`hard_positive_added_count=0`。

本阶段不改 `train_policy_on_episodes` 默认语义，不改 network/action space/default A*，不放宽默认 distance contract，
不发布 checkpoint，不宣称正式 PPO ready、策略性能提升或 Ackermann-feasible trajectory。
