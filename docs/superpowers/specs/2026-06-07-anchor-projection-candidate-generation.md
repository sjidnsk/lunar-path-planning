# Anchor-Projection Candidate Generation v1

## 背景

`Anchor-Projection Evidence Contract v1` 证明了事后 audit proxy anchor 不能作为 positive
training evidence。旧 evidence root
`outputs/path_feedback_batch_anchor_projection_contract_v1/` 在父仓库提交
`d15c667 Add anchor projection evidence contract` 后已触发
`current_git_provenance_mismatch`，因此本阶段重新生成 current-HEAD evidence。

## 实现范围

本阶段把 anchor projection 前移到 `model-explorer` opt-in candidate generation 层：

- `AnchorProjectionCandidateConfig(enabled=True)` 开启 projected execution target candidate。
- 对 `original_passable=true`、`inflated_passable=false` 的 source policy target，基于
  `inflated_passable_mask` 找到 nearest inflated-passable anchor。
- 只有 anchor reachable 且满足阈值时，才对 projected execution target 重新调用 planner。
- projected candidate 保留 `policy_target_cell`、`execution_goal_cell`、
  `projected_anchor_cell`、projection distance、`anchor_reachable`、`comparison_scope`、
  `training_use`、`sample_weight`、`reject_reason` 和 `evidence_boundary`。
- 只有 source selection 实际选择 projected candidate 时，才标注
  `training_use=trainable_anchor_projection_contrast` 和 `sample_weight=1.0`。
- 未选中的 projected candidate 和所有 `audit_proxy_anchor_not_same_cell` 仍保持
  `training_use=not_positive_evidence`。

稳定 schema 只做 additive 扩展：`path-planner-route/v1`、
`model-explorer-contract/v1`、`path-feedback-summary/v1` 的既有字段语义不变。

## 新增产物

- `configs/path_feedback_batch_anchor_projection_candidate_generation_v1.json`
- `configs/anchor_projection_candidate_generation_v1.json`
- `scripts/run_anchor_projection_candidate_generation.py`
- `scripts/run_anchor_projection_candidate_generation.sh`
- `scripts/run_path_feedback_validation.sh --anchor-projection-candidate-generation`
- `tests/test_anchor_projection_candidate_generation.py`
- `model-explorer/tests/test_model_explorer.py` 中的 projected candidate 回归覆盖

## Current-HEAD Evidence

Evidence root：
`outputs/path_feedback_batch_anchor_projection_candidate_generation_v1/`

关键结果：

- batch：8 runs、8 passed、0 failed
- `current_git_provenance_mismatch_count=0`
- `git_provenance_mismatch_count=0`
- `open_grid_fallback_used_count=0`
- `fallback_or_open_grid_count=0`
- `safety_regression_count=0`
- `platform_goal_contract_mismatch_count=78`
- `trainable_anchor_projection_count=12`
- `nontrainable_blocked_target_count=66`
- `source_selected_candidate_changed_rate=0.15384615384615385`
- `positive_training_evidence_contains_audit_proxy_anchor_count=0`

旧 audit-only contract summary 仍保持：

- `trainable_anchor_projection_count=0`
- `nontrainable_blocked_target_count=42`
- `recommended_next_action=keep_platform_blocked_targets_out_of_training`

该结果是预期的：旧 summary 的责任是阻止 audit proxy 混入 positive evidence；新 summary 的
责任是统计 source-selected projected candidates。

## 验证命令

```bash
PYTHONPATH=model-explorer/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/kai/anaconda3/envs/lunar-explorer/bin/python -m pytest model-explorer/tests/test_model_explorer.py::PathPlanningAdapterTests tests/test_path_feedback_validation_script.py::PathFeedbackValidationScriptTests::test_anchor_projection_candidate_generation_is_explicit_opt_in tests/test_anchor_projection_candidate_generation.py tests/test_batch_path_feedback_validation.py -q

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/kai/anaconda3/envs/lunar-explorer/bin/python -m pytest tests/test_anchor_projection_evidence_contract.py tests/test_policy_training_readiness_review.py -q

bash scripts/run_batch_path_feedback_validation.sh --matrix configs/path_feedback_batch_anchor_projection_candidate_generation_v1.json --output-root outputs/path_feedback_batch_anchor_projection_candidate_generation_v1

bash scripts/run_anchor_projection_candidate_generation.sh --batch-root outputs/path_feedback_batch_anchor_projection_candidate_generation_v1 --config configs/anchor_projection_candidate_generation_v1.json
```

全链路 validate-only 已覆盖 batch、sample-quality、policy robustness、channel-aware、
readiness、goal-blocked、audit-only anchor contract 和新 candidate-generation summary。

## 当前结论

本阶段已经证明“测试案例不够复杂所以没效果”不是主要问题。真实问题是 anchor projection 之前
只存在于失败后的 audit proxy scope；前移到 candidate generation 后，source selection 已经
产生非零变化并得到 12 条 trainable projected target context。

但 PPO 仍不能启动，因为 66 个 blocked target context 仍不可训练，policy training readiness
仍是 `needs_training_contract_refinement`。下一步应做 `Anchor-Projection Coverage Expansion v1`：
分析 nontrainable context 的 reject reasons、anchor reachability、projection distance 和 ranking
margin，扩大 source-selected projected target 覆盖率。

## 非目标

- 不启动 PPO。
- 不修改 network architecture 或 action space。
- 不修改 default A* 语义。
- 不把 opt-in candidate generation 宣称为默认 route replacement。
- 不实现完整 GCS graph search。
- 不使用 C-space IRIS、IrisNp、IrisNp2 或 IrisZo。
- 不宣称 Ackermann-feasible trajectory。
