# Unified Sample Taxonomy and Residual Boundary Preference v1

## Summary

证据根：
`outputs/path_feedback_batch_planner_validated_trainable_target_mining_v1/`。

本阶段把此前仍未可训练利用的 30 个 residual context 纳入统一样本体系，但不增加 PPO action hard positive。
最终样本体系覆盖 78 个 blocked contexts：24 个 action-label positive、24 个既有 counterfactual preference pair、
12 个 boundary-negative preference pair、18 个 blocked-target negative pair。

## Implementation

新增产物：

- `configs/unified_policy_sample_registry_v1.json`
- `configs/residual_boundary_preference_training_dry_run_v1.json`
- `scripts/run_unified_policy_sample_registry.py`
- `scripts/run_unified_policy_sample_registry.sh`
- `scripts/run_residual_boundary_preference_training_dry_run.py`
- `scripts/run_residual_boundary_preference_training_dry_run.sh`
- `tests/test_unified_policy_sample_registry.py`

`run_unified_policy_sample_registry.py` 读取 planner-validated mining、candidate generation、materialization、
counterfactual preference samples 与 exclusion report，输出：

- `unified-policy-sample-registry.jsonl`
- `unified-policy-sample-registry-summary.json`
- `unified-policy-sample-exclusion-report.json`

Residual 处理规则：

- 12 个 `npz_near_blocked_corridor` exclusion 生成为 `boundary_negative_preference_pair`，保留
  `binding_required=true`，不进入 `RolloutEpisode.action_index`。
- 18 个 `npz_dense_rock_choke` distance-blocked context 生成为 `blocked_target_negative_pair`，保留
  `hierarchical_subgoal_required=true`。
- 缺失 selected reference 或 candidate metric 时不合成样本，写入 exclusion report。

## Current Evidence

Registry summary:

- `status=passed`
- `reason_codes=[]`
- `action_label_positive_count=24`
- `existing_preference_pair_count=24`
- `boundary_negative_preference_pair_count=12`
- `blocked_target_negative_pair_count=18`
- `residual_trainable_signal_count=30`
- `pairwise_preference_signal_count=54`
- `unified_context_coverage_count=78`
- `hard_positive_added_count=0`
- `planner_validated_trainable_target_count=24`

Residual dry-run summary:

- `status=passed`
- `residual_preference_dry_run_status=passed`
- `residual_train_sample_count=30`
- `boundary_negative_preference_pair_count=12`
- `blocked_target_negative_pair_count=18`
- `publishes_checkpoint=false`
- `performance_claimed=false`

## Validation

```bash
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_unified_policy_sample_registry.sh \
  --batch-root outputs/path_feedback_batch_planner_validated_trainable_target_mining_v1 \
  --config configs/unified_policy_sample_registry_v1.json

PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_residual_boundary_preference_training_dry_run.sh \
  --batch-root outputs/path_feedback_batch_planner_validated_trainable_target_mining_v1 \
  --config configs/residual_boundary_preference_training_dry_run_v1.json

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
/home/kai/anaconda3/envs/lunar-explorer/bin/python -m pytest -q \
  model-explorer/tests/test_model_explorer.py \
  tests/test_counterfactual_preference_training_samples.py \
  tests/test_limited_policy_training_dry_run_input_materialization.py \
  tests/test_unified_policy_sample_registry.py \
  tests/test_policy_training_readiness_review.py
```

## Boundaries

`unified_context_coverage_count=78` 表示统一训练信号覆盖，不表示 78 个 PPO hard positive。
`action_label_positive_count` 保持 24，`hard_positive_added_count=0`。

本阶段不启动正式 PPO，不发布 checkpoint，不宣称策略性能提升，不改 `train_policy_on_episodes` 默认语义，
不改 network/action space/default A*，不放宽默认 distance contract，不宣称 Ackermann-feasible trajectory。
