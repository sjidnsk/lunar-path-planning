# Anchor Projection Nontrainable Context Reduction / Source-Selection Candidate Quality v1

## Scope

本阶段基于
`outputs/path_feedback_batch_anchor_projection_distance_contract_relaxation_safety_audit_v1/`
的保守结论继续推进：distance audit 已通过，但 recommendation 仍为
`keep_current_training_distance_contract`，policy readiness 仍为
`needs_training_contract_refinement`，blocker 为
`anchor_projection_nontrainable_contexts_remain`。

目标不是启动 PPO，也不是直接放宽默认 distance contract，而是把剩余 60 个
nontrainable context 做成可复验的 source-selection / distance-rejection 去向审计。

非目标保持严格：不启动 PPO，不修改 `training.py`、`ppo.py`、`architectures.py`、
network、action space 或 default A*，不宣称 Ackermann-feasible trajectory，不使用
IrisNp/IrisNp2/IrisZo/C-space IRIS，不把 IRIS/GCS/channel-aware 诊断当训练放行。

## Implementation

新增入口：

- `scripts/run_anchor_projection_nontrainable_context_reduction.py`
- `scripts/run_anchor_projection_nontrainable_context_reduction.sh`
- `configs/anchor_projection_nontrainable_context_reduction_v1.json`
- `configs/path_feedback_batch_anchor_projection_nontrainable_context_reduction_v1.json`
- `tests/test_anchor_projection_nontrainable_context_reduction.py`

新 summary schema 为
`anchor-projection-nontrainable-context-reduction-summary/v1`，默认输出到：

`outputs/path_feedback_batch_anchor_projection_nontrainable_context_reduction_v1/anchor-projection-nontrainable-context-reduction-summary.json`

它消费同一 root 下的：

- `anchor-projection-candidate-generation-summary/v1`
- `anchor-projection-evidence-contract-summary/v1`
- `policy-training-readiness-review-summary/v1`
- `anchor-projection-distance-contract-relaxation-safety-audit-summary/v1`

## Decision Rule

summary 必须显式保留以下口径：

- `safe_default_training_conversion_count=0`
- `must_remain_blocked_count=nontrainable_blocked_target_count`
- `blocker_retained=true` when any nontrainable context remains
- `source_selection_quality_regression_count=0`
- `positive_training_evidence_contains_audit_proxy_anchor_count=0`
- `does_not_relax_default_distance_contract=true`

60 个 nontrainable context 的预期去向：

- 6 个：`opt_in_relaxation_followup_candidate`，只允许作为后续显式 opt-in contract review 候选。
- 6 个：`blocked_source_selected_distance_too_far`，对应 11 cells / 5.5 m 的 source-selected distance-rejected context。
- 24 个：`blocked_not_source_selected_distance_rejected`。
- 24 个：`blocked_source_candidate_not_selected_quality`。

只要 `nontrainable_blocked_target_count>0`，readiness 仍必须保持
`needs_training_contract_refinement` 与
`anchor_projection_nontrainable_contexts_remain`。

## Verification

目标证据根：

`outputs/path_feedback_batch_anchor_projection_nontrainable_context_reduction_v1/`

验证命令：

```bash
R=outputs/path_feedback_batch_anchor_projection_nontrainable_context_reduction_v1
bash scripts/run_batch_path_feedback_validation.sh --matrix configs/path_feedback_batch_anchor_projection_nontrainable_context_reduction_v1.json --output-root $R
bash scripts/run_anchor_projection_candidate_generation.sh --batch-root $R --config configs/anchor_projection_candidate_generation_v1.json
bash scripts/run_anchor_projection_evidence_contract.sh --batch-root $R --config configs/anchor_projection_evidence_contract_v1.json
bash scripts/run_policy_training_readiness_review.sh --batch-root $R --config configs/policy_training_readiness_review_v1.json
bash scripts/run_anchor_projection_distance_contract_relaxation_safety_audit.sh --batch-root $R --config configs/anchor_projection_distance_contract_relaxation_safety_audit_v1.json
bash scripts/run_anchor_projection_nontrainable_context_reduction.sh --batch-root $R --config configs/anchor_projection_nontrainable_context_reduction_v1.json
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/kai/anaconda3/envs/lunar-explorer/bin/python -m pytest tests/test_anchor_projection_candidate_generation.py tests/test_anchor_projection_evidence_contract.py tests/test_policy_training_readiness_review.py tests/test_anchor_projection_distance_contract_relaxation_safety_audit.py tests/test_anchor_projection_nontrainable_context_reduction.py -q
git diff --check
```

## Expected Outcome

本阶段完成后仍不等同于 PPO readiness。clean evidence 已在临时 clean worktree 生成，
并同步到主工作区目标 root。结论为：

- batch：8/8 passed，open-grid fallback 为 0。
- candidate/evidence-contract/readiness/distance audit/nontrainable reduction 均为
  `status=passed`、`reason_codes=[]`。
- provenance mismatch 为 0。
- readiness blocker 仍为 `anchor_projection_nontrainable_contexts_remain`。
- nontrainable-reduction：
  `recommendation=keep_training_blocker_focus_source_selection_candidate_quality`，
  `generated_not_source_selected_count=48`，
  `distance_contract_rejected_count=36`，
  `source_selected_distance_rejected_count=12`，
  `not_source_selected_distance_rejected_count=24`。
- `nontrainable_resolution_accounting`：
  `safe_default_training_conversion_count=0`，
  `opt_in_relaxation_followup_candidate_count=6`，
  `must_remain_blocked_count=60`。
- 分类计数：
  `opt_in_relaxation_followup_candidate=6`，
  `blocked_source_selected_distance_too_far=6`，
  `blocked_not_source_selected_distance_rejected=24`，
  `blocked_source_candidate_not_selected_quality=24`。
- 下一步应继续减少 source-candidate-not-selected 与 not-source-selected distance-rejected context，
  而不是直接放宽默认 distance contract。
