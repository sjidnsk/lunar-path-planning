# Counterfactual Preference Training Samples v1

## 背景

Planner-Validated Trainable Target Mining v1 后，仍有
`source_selection_not_selected_count=36`。这些 context 不是普通训练正例：当前证据显示它们全部为
`target_binding_mode=synthetic_projection` 且 `ppo_consumable_action=false`，不能写入现有
`RolloutEpisode.action_index` hard label。它们的价值是记录 selected action 与未选中 alternative
之间的 path/risk/utility tradeoff。

## 目标

新增 opt-in counterfactual preference pipeline：

- 不增加 hard positive，`hard_positive_added_count=0`。
- 不调用默认 `train_policy_on_episodes` 处理这 36 条。
- 把 contract-safe not-selected context 转为 pairwise preference 样本。
- 非 contract-safe 或仍需 action binding / distance-contract 的样本进入 exclusion。
- preference dry-run 只验证 ranking loss 可消费，不发布 checkpoint，不宣称策略性能。

## 产物

- `configs/counterfactual_preference_training_samples_v1.json`
- `configs/counterfactual_preference_training_dry_run_v1.json`
- `scripts/run_counterfactual_preference_training_samples.py`
- `scripts/run_counterfactual_preference_training_samples.sh`
- `scripts/run_counterfactual_preference_training_dry_run.py`
- `scripts/run_counterfactual_preference_training_dry_run.sh`
- `tests/test_counterfactual_preference_training_samples.py`
- 输出：
  - `counterfactual-preference-training-samples.jsonl`
  - `counterfactual-preference-training-summary.json`
  - `counterfactual-preference-exclusion-report.json`
  - `counterfactual-preference-training-dry-run-summary.json`

## 当前验收结果

- `source_selection_not_selected_count=36`
- `preference_pair_count=24`
- `selected_over_alternative_negative_count=12`
- `tradeoff_preference_pair_count=12`
- `rejected_binding_or_distance_required_count=12`
- `rejected_quality_regression_count=0`
- `hard_positive_added_count=0`
- preference dry-run `status=passed`
- `preference_train_sample_count=24`
- `publishes_checkpoint=false`
- `performance_claimed=false`

## 验证命令

```bash
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_counterfactual_preference_training_samples.sh \
  --batch-root outputs/path_feedback_batch_planner_validated_trainable_target_mining_v1 \
  --config configs/counterfactual_preference_training_samples_v1.json

PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_counterfactual_preference_training_dry_run.sh \
  --batch-root outputs/path_feedback_batch_planner_validated_trainable_target_mining_v1 \
  --config configs/counterfactual_preference_training_dry_run_v1.json

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/kai/anaconda3/envs/lunar-explorer/bin/python -m pytest -q \
  model-explorer/tests/test_model_explorer.py \
  tests/test_counterfactual_preference_training_samples.py \
  tests/test_limited_policy_training_dry_run_input_materialization.py \
  tests/test_policy_training_readiness_review.py
```

## 非目标

不把 not-source-selected synthetic projection 当作 hard positive；不修改默认
`train_policy_on_episodes` 语义；不改 network/action space/default A*；不放宽默认 distance contract；
不启动正式 PPO；不发布 checkpoint；不宣称策略性能提升或 Ackermann-feasible trajectory。
