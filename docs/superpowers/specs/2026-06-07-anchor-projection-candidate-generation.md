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

### Candidate Generation v1 基线结果

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

### Coverage Diagnosis + Selection Improvement v1 结果

本轮新增 opt-in `--anchor-projection-selection-path-cost-bonus 6.0`，并在 Risk Closure v1
加入 source-selection quality gate：
`--anchor-projection-max-selection-path-cost-regression 6.0` 与
`--anchor-projection-max-selection-risk-regression 0.5`。这些参数只进入
`model-explorer` 的 path-feedback manifest，不转发给 `path-planner`，不修改 default A*，
不启动 PPO，不放宽 fallback/open-grid/safety gate。若 source-selected projected candidate
相对最佳可行替代项超过 path/risk regression 阈值，则保持
`training_use=not_positive_evidence`。

刷新后的结果：

- batch：8 runs、8 passed、0 failed
- `current_git_provenance_mismatch_count=0`
- `git_provenance_mismatch_count=0`
- `open_grid_fallback_used_count=0`
- `fallback_or_open_grid_count=0`
- `safety_regression_count=0`
- `platform_goal_contract_mismatch_count=78`
- `trainable_anchor_projection_count=18`
- `nontrainable_blocked_target_count=60`
- `source_selected_candidate_changed_rate=0.23076923076923078`
- `source_selection_quality_regression_count=0`（Risk Closure v1 隔离重跑口径）
- `positive_training_evidence_contains_audit_proxy_anchor_count=0`

新增 `anchor_projection_coverage_diagnosis/v1` 解释了剩余阻塞：

- `projected_candidate_generated_count=42`
- `projected_candidate_source_selected_count=18`
- `projected_candidate_not_source_selected_count=24`
- `anchor_unreachable_not_generated_count=36`
- `nontrainable_primary_reason_counts={"anchor_unreachable":36,"source_candidate_not_selected":24}`
- projection distance：72 个 1-cell，6 个 2-cell
- generated-but-not-selected path-cost margin：约 2.17 到 10.88

结论：机制已经从 12/78 基线推进到 18/78 的 source-selection bonus + quality-gate 口径，
但剩余瓶颈不再是“测试样本不够复杂”，而是
36 个 anchor 不可达和 24 个 generated candidate 未被 source selection 选中。继续单纯加大
bonus 会扩大 path-cost tradeoff，不应直接作为 PPO readiness 的替代条件。

## Risk Closure v1

本轮新增统一 git provenance helper。新 evidence source snapshot 会记录 parent 与三个子模块
的 `sha`、`branch`、`dirty`、`tracked_modified_count`、`untracked_count`、
`ignored_untracked_count`；当 source snapshot 声明 dirty 时，`require_current_git_match=true`
必须失败。旧格式只含 SHA 的 legacy summary 仍按 SHA 兼容。

隔离重跑 root：
`outputs/path_feedback_batch_anchor_projection_candidate_generation_v1_risk_closure_check/`

- batch：8 runs、8 passed、0 failed
- `batch-run-index.json.git.dirty=true`
- candidate-generation summary：`status=failed`
- `reason_codes=["current_git_provenance_mismatch","git_provenance_mismatch"]`
- `current_git_provenance_mismatch_count=1`
- `git_provenance_mismatch_count=1`
- 算法计数仍为 `trainable_anchor_projection_count=18`、
  `nontrainable_blocked_target_count=60`、`source_selection_quality_regression_count=0`

该失败是预期验收：当前工作区未提交，不能把隔离输出作为正式 current-HEAD evidence。提交后
必须在 clean worktree 下重跑 batch、candidate-generation、contract 与 readiness validate-only。

Risk Closure v1 同时修复：

- `AStarPlanner`、`ChannelAwareAStarPlanner`、`RegionGraphGuidedPlanner` 的严格
  `prevent_corner_cutting`：对角移动两侧 side cell 任一 blocked 即禁止。
- `channel_aware_astar` selection 组合 gate：不能只因 high-cost exposure 下降就接受
  channel/path cost 明显退化的 candidate，reason 为 `channel_candidate_quality_regression`。
- `policy-training-readiness-review` 可选消费
  `anchor-projection-candidate-generation-summary/v1` 与
  `anchor-projection-evidence-contract-summary/v1`，并输出
  `anchor_projection_readiness` 及
  `anchor_projection_contract_trainable_count_below_candidate_generation` blocker。

当前 anchor-projection contract summary 仍保持 audit proxy 不得混入 positive evidence：

- `trainable_anchor_projection_count=3`
- `nontrainable_blocked_target_count=39`
- `positive_training_evidence_contains_audit_proxy_anchor_count=0`
- `recommended_next_action=rerun_policy_training_readiness_review_with_anchor_projection_contract`

## Risk Closure v2

Risk Closure v2 修复剩余工程风险：

- provenance source inspection 统一到 `scripts/git_provenance.py`。缺少
  `git_provenance.current` 的 source summary 在 `require_current_git_match=true` 时必须失败，
  输出 `current_git_provenance_missing` 与 `<label>_current_git_provenance_missing`。
- readiness 的 path/risk regression blocker 只看 source-selected quality regression 或
  trainable anchor-projection context；未选中 projected candidate 的 margin 仅进入
  `diagnostic_max_source_selection_*_margin_vs_best_alternative`。
- best alternative scope 固定为
  `reachable_non_replan_candidates_including_policy_and_projected_targets`，并在
  candidate annotation 与 candidate-generation summary 中记录 alternative role。
- `path-planner` direction-cone CLI batch 将旧
  `sampled_trajectory_collision` case 重定标为 `direction_cone_obstacle_detour`，因为严格
  corner-cutting 后该 3x3 中心障碍场景不再产生 sampled collision，真实 blocker 是
  `direction_cone_constraint_violation`。

本阶段仍不启动 PPO，也不把 GCS candidate 或 projected anchor 当作默认执行轨迹。

验证状态：

- parent/model-explorer/dev-platform/path-planner 测试均已通过对应全量命令。
- anchor-projection batch dry-run 展开 8 个 run，正式 dirty-worktree 隔离 batch 8/8 passed。
- dirty-worktree candidate-generation validate-only 按预期被 provenance gate 阻断：
  `current_git_provenance_mismatch_count=1`、`git_provenance_mismatch_count=1`，
  同时 `trainable_anchor_projection_count=18`，说明失败来自 dirty provenance 而不是候选生成回退。
- clean-worktree formal batch 8/8 passed；candidate-generation validate-only 与 summary generation
  均通过，`current_git_provenance_mismatch_count=0`、`git_provenance_mismatch_count=0`、
  `source_selection_quality_regression_count=0`。

该 summary 的责任仍是训练合约边界审计；不能与 candidate-generation 的 18/78 计数混作同一
训练输入口径。

## 验证命令

```bash
PYTHONPATH=model-explorer/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/kai/anaconda3/envs/lunar-explorer/bin/python -m pytest model-explorer/tests/test_model_explorer.py::PathPlanningAdapterTests tests/test_path_feedback_validation_script.py::PathFeedbackValidationScriptTests::test_anchor_projection_candidate_generation_is_explicit_opt_in tests/test_anchor_projection_candidate_generation.py tests/test_batch_path_feedback_validation.py -q

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/kai/anaconda3/envs/lunar-explorer/bin/python -m pytest tests/test_anchor_projection_evidence_contract.py tests/test_policy_training_readiness_review.py -q

bash scripts/run_batch_path_feedback_validation.sh --matrix configs/path_feedback_batch_anchor_projection_candidate_generation_v1.json --output-root outputs/path_feedback_batch_anchor_projection_candidate_generation_v1

bash scripts/run_anchor_projection_candidate_generation.sh --batch-root outputs/path_feedback_batch_anchor_projection_candidate_generation_v1 --config configs/anchor_projection_candidate_generation_v1.json

bash scripts/run_anchor_projection_candidate_generation.sh --batch-root outputs/path_feedback_batch_anchor_projection_candidate_generation_v1 --config configs/anchor_projection_candidate_generation_v1.json --validate-only

bash scripts/run_policy_training_readiness_review.sh --batch-root outputs/path_feedback_batch_anchor_projection_candidate_generation_v1 --config configs/policy_training_readiness_review_v1.json --anchor-projection-candidate-generation-summary outputs/path_feedback_batch_anchor_projection_candidate_generation_v1/anchor-projection-candidate-generation-summary.json --anchor-projection-evidence-contract-summary outputs/path_feedback_batch_anchor_projection_candidate_generation_v1/anchor-projection-evidence-contract-summary.json --validate-only
```

全链路 validate-only 已覆盖 batch、sample-quality、policy robustness、channel-aware、
readiness、goal-blocked、audit-only anchor contract 和新 candidate-generation summary。

## 当前结论

本阶段已经证明“测试案例不够复杂所以没效果”不是主要问题。真实问题是 anchor projection 之前
只存在于失败后的 audit proxy scope；前移到 candidate generation 后，source selection 已经
产生非零变化并得到 18 条 trainable projected target context。

`Anchor-Projection Readiness Contract Integration v1` 已把 candidate-generation summary 与
anchor-projection contract/readiness 的训练口径对齐。新的
`outputs/path_feedback_batch_anchor_projection_contract_integration_v1/` root 中：

- batch：8/8 passed，`open_grid_fallback_used_count=0`。
- candidate-generation：`status=passed`，`reason_codes=[]`，
  `trainable_anchor_projection_count=18`，
  `nontrainable_blocked_target_count=60`，
  `nontrainable_anchor_unreachable_count=36`，
  `nontrainable_source_candidate_not_selected_count=24`，
  `source_selection_quality_regression_count=0`，
  `positive_training_evidence_contains_audit_proxy_anchor_count=0`。
- evidence-contract：`contract_source=anchor_projection_candidate_generation_summary`，
  `trainable_anchor_projection_count=18`，
  `nontrainable_blocked_target_count=60`，
  `candidate_contract_alignment_gap_count=0`，
  `contract_blockers=[]`。
- policy readiness：`application_scope=anchor_projection_readiness_contract_review_only`，
  `anchor_projection_readiness_trainable_count=18`，
  `anchor_projection_candidate_contract_alignment_gap_count=0`，
  但仍为 `training_readiness_status=needs_training_contract_refinement`，
  `training_blockers=["anchor_projection_nontrainable_contexts_remain"]`。

PPO 仍不能启动。当前剩余瓶颈不再是 18/78 与 3/42 的 contract 口径不一致，而是
60 个 nontrainable context：36 个 `anchor_unreachable` 与 24 个
`source_candidate_not_selected`。下一阶段应分别分析平台 footprint/地形可达性与
source-selection margin，而不是扩大样本或放宽正样本边界。

## 非目标

- 不启动 PPO。
- 不修改 network architecture 或 action space。
- 不修改 default A* 语义。
- 不把 opt-in candidate generation 宣称为默认 route replacement。
- 不实现完整 GCS graph search。
- 不使用 C-space IRIS、IrisNp、IrisNp2 或 IrisZo。
- 不宣称 Ackermann-feasible trajectory。
