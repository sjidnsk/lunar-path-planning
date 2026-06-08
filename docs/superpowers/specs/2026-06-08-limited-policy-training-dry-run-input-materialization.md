# Limited Policy Training Dry-Run Input Materialization v1

## 背景

`outputs/path_feedback_batch_planner_validated_trainable_target_mining_v1/` 已通过
Planner-Validated Trainable Target Mining v1：`planner_validated_trainable_target_count=24`，
其中默认 contract 正例 18 个、planner-validated distance exception 正例 6 个，剩余
`excluded_nontrainable_count=54`。readiness 已进入
`ready_for_limited_policy_training_dry_run`，但这还不是训练链路可消费性证明。

## 目标

把 24 个 mining 正例物化为现有 `RolloutEpisode` / `RolloutTransition` JSONL，并执行一次
strict limited dry-run：

- 只接受 `selected_default_contract_trainable` 与
  `selected_planner_validated_distance_exception`。
- `transition.action_index == source_action_index`。
- `observation.action_mask[source_action_index] == true`。
- `execution_goal_cell`、`policy_target_cell`、`target_binding_mode`、
  `planner_validated_distance_exception` 写入 transition metadata。
- 剩余 54 个 blocker 只进入 exclusion report，不得转成训练正例。

## 产物

- `configs/planner_validated_training_input_materialization_v1.json`
- `configs/limited_policy_training_dry_run_v1.json`
- `scripts/run_planner_validated_training_input_materialization.py`
- `scripts/run_planner_validated_training_input_materialization.sh`
- `scripts/run_limited_policy_training_dry_run.py`
- `scripts/run_limited_policy_training_dry_run.sh`
- `tests/test_limited_policy_training_dry_run_input_materialization.py`
- 输出：
  - `planner-validated-rollout-episodes.jsonl`
  - `planner-validated-training-input-materialization-summary.json`
  - `planner-validated-training-exclusion-report.json`
  - `limited-policy-training-dry-run-summary.json`

## 当前验收结果

- materialization summary `status=passed`，`reason_codes=[]`。
- `input_positive_count=24`
- `default_contract_positive_count=18`
- `planner_validated_exception_positive_count=6`
- `excluded_nontrainable_count=54`
- `invalid_action_mask_count=0`
- `empty_action_mask_count=0`
- dry-run summary `dry_run_status=passed`
- `train_policy_sample_count=24`
- `publishes_checkpoint=false`
- `performance_claimed=false`

## 验证命令

```bash
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_planner_validated_training_input_materialization.sh \
  --batch-root outputs/path_feedback_batch_planner_validated_trainable_target_mining_v1 \
  --config configs/planner_validated_training_input_materialization_v1.json

PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_limited_policy_training_dry_run.sh \
  --batch-root outputs/path_feedback_batch_planner_validated_trainable_target_mining_v1 \
  --config configs/limited_policy_training_dry_run_v1.json

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/kai/anaconda3/envs/lunar-explorer/bin/python -m pytest -q \
  model-explorer/tests/test_model_explorer.py \
  tests/test_planner_validated_trainable_target_mining.py \
  tests/test_limited_policy_training_dry_run_input_materialization.py \
  tests/test_policy_training_readiness_review.py
```

## 非目标

不解决剩余 54 个 contract/source-selection blocker；不启动正式 PPO；不发布 checkpoint；
不宣称策略性能提升；不修改 `training.py` 主训练逻辑；不改 network/action space/default A*；
不放宽默认 distance contract；不宣称 Ackermann-feasible trajectory；不把 IRIS/GCS 诊断当训练放行。
