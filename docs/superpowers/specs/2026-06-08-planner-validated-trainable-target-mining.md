# Planner-Validated Trainable Target Mining v1

## 背景

Contract-Aware Trainable Target Generation v1 已证明当前 action mask 下可以生成
`18` 个 `same_action_execution_substitute`：
`ppo_consumable_action=true`、`contract_safe=true`、`source_selected`。但
`nontrainable_blocked_target_count` 仍为 `60`，readiness 继续保留
`anchor_projection_nontrainable_contexts_remain`。根因是“按可训练目标生成候选”还不等于
“planner 反馈后可作为训练正例”。

## 目标

新增 opt-in 的 post-planner mining/filter 层：

- 保留 contract-aware same-action candidate generation。
- 允许 3 cells / 1.5 m 作为 opt-in planner-validated distance exception。
- 默认 distance contract 仍为 2 cells / 1.0 m。
- 不绕过 source selection；未 source-selected 的 planner repair 只能进入 diagnostic。
- 每个 blocked context 只输出一个最终训练裁决。

最终裁决集合：

- `selected_default_contract_trainable`
- `selected_planner_validated_distance_exception`
- `rejected_distance_contract`
- `rejected_not_source_selected`
- `rejected_quality_regression`
- `rejected_not_ppo_consumable`

## 产物

- `configs/path_feedback_batch_planner_validated_trainable_target_mining_v1.json`
- `configs/planner_validated_trainable_target_mining_v1.json`
- `scripts/run_planner_validated_trainable_target_mining.py`
- `scripts/run_planner_validated_trainable_target_mining.sh`
- `tests/test_planner_validated_trainable_target_mining.py`
- 证据根：`outputs/path_feedback_batch_planner_validated_trainable_target_mining_v1/`

## 验收

必须满足：

- batch `failed_count=0`
- open-grid/fallback count 为 0
- safety regression count 为 0
- current git provenance mismatch 为 0
- summary `reason_codes=[]`
- `candidate_contract_alignment_gap_count=0`

主成功：

- `planner_validated_trainable_target_count > 18`
- `nontrainable_blocked_target_count < 60`
- readiness 不再因同一批 context 保留
  `anchor_projection_nontrainable_contexts_remain`

若主成功失败：

- 不宣称 training ready。
- readiness 继续 blocked。
- summary 设置
  `next_required_change=source_selection_or_target_contract_change_required`。

当前验收结果：

- batch 8/8 passed，`failed_count=0`，`open_grid_fallback_used_count=0`。
- candidate summary `status=passed`，`reason_codes=[]`，
  `current_git_provenance_mismatch_count=0`。
- candidate summary 仍为 `trainable_anchor_projection_count=18`、
  `nontrainable_blocked_target_count=60`。
- mining summary `planner_validated_trainable_target_count=24`、
  `default_contract_trainable_target_count=18`、
  `planner_validated_distance_exception_count=6`、
  `nontrainable_blocked_target_count=54`、
  `nontrainable_blocked_target_count_delta=-6`、
  `candidate_contract_alignment_gap_count=0`、
  `next_required_change=null`。
- readiness summary `training_readiness_status=ready_for_limited_policy_training_dry_run`，
  `training_blockers=[]`。

## 验证命令

```bash
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_batch_path_feedback_validation.sh \
  --matrix configs/path_feedback_batch_planner_validated_trainable_target_mining_v1.json

PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_anchor_projection_candidate_generation.sh \
  --batch-root outputs/path_feedback_batch_planner_validated_trainable_target_mining_v1 \
  --config configs/anchor_projection_candidate_generation_v1.json

PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_planner_validated_trainable_target_mining.sh \
  --batch-root outputs/path_feedback_batch_planner_validated_trainable_target_mining_v1 \
  --config configs/planner_validated_trainable_target_mining_v1.json

PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_policy_training_readiness_review.sh \
  --batch-root outputs/path_feedback_batch_planner_validated_trainable_target_mining_v1 \
  --config configs/policy_training_readiness_review_v1.json \
  --anchor-projection-candidate-generation-summary outputs/path_feedback_batch_planner_validated_trainable_target_mining_v1/anchor-projection-candidate-generation-summary.json \
  --planner-validated-trainable-target-mining-summary outputs/path_feedback_batch_planner_validated_trainable_target_mining_v1/planner-validated-trainable-target-mining-summary.json

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/kai/anaconda3/envs/lunar-explorer/bin/python -m pytest -q \
  model-explorer/tests/test_model_explorer.py \
  tests/test_anchor_projection_candidate_generation.py \
  tests/test_policy_training_readiness_review.py \
  tests/test_planner_validated_trainable_target_mining.py
```

## 非目标

不启动 PPO；不修改 `training.py` 主训练逻辑；不改 network/action space/default A*；
不放宽默认 distance contract；不宣称 Ackermann-feasible trajectory；不把 IRIS/GCS/path-planner
修复结果直接等价为训练放行。
