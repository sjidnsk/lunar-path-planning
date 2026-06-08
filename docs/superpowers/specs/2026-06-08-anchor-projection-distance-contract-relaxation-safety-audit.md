# Anchor Projection Distance-Contract Relaxation Safety Audit v1

## Scope

本阶段基于
`outputs/path_feedback_batch_anchor_projection_source_selection_distance_contract_v1/`
后的 evidence，新增 audit-only summary，用于判断 anchor projection 训练距离 contract
是否可以显式、opt-in 地放宽。

非目标保持严格：不启动 PPO，不修改 `training.py`、`ppo.py`、`architectures.py`、
network、action space 或 default A*，不宣称 Ackermann-feasible trajectory，不把 IRIS/GCS
或 channel-aware opt-in 诊断当训练放行依据。

## Implementation

新增入口：

- `scripts/run_anchor_projection_distance_contract_relaxation_safety_audit.py`
- `scripts/run_anchor_projection_distance_contract_relaxation_safety_audit.sh`
- `configs/anchor_projection_distance_contract_relaxation_safety_audit_v1.json`
- `configs/path_feedback_batch_anchor_projection_distance_contract_relaxation_safety_audit_v1.json`
- `tests/test_anchor_projection_distance_contract_relaxation_safety_audit.py`

新 summary schema 为
`anchor-projection-distance-contract-relaxation-safety-audit-summary/v1`。它在目标验证链路中紧跟
candidate-generation 运行，因此必需消费同一 batch root 下的：

- `anchor-projection-candidate-generation-summary/v1`

若同一 root 下已经存在以下 summary，也会一并校验 schema/status/provenance；若它们尚未生成，
audit 会把它们记录为 optional input，不阻断后续 evidence-contract 与 readiness gate：

- `anchor-projection-evidence-contract-summary/v1`
- `policy-training-readiness-review-summary/v1`

审计只使用 candidate-generation summary 的 current/provenance、fallback/open-grid、safety、
source-selection quality regression、audit proxy positive 与 `context_records`，不重新解释
path-feedback；后续 evidence-contract 与 readiness 命令仍必须独立通过。

`run_policy_training_readiness_review.py` 同步支持默认 anchor-only 自动检测：当同一 root 下存在
candidate-generation 与 evidence-contract summary，且旧 channel-aware 四件套不存在时，不需要额外传
`--anchor-projection-*` 参数，也会按
`application_scope=anchor_projection_readiness_contract_review_only` 运行。

## Decision Rule

默认配置只允许审计一个保守 opt-in profile：

- `max_opt_in_projection_distance_cells=3`
- `max_opt_in_projection_distance_m=1.5`
- `max_path_cost_margin_vs_selected=0.0`
- `max_risk_margin_vs_selected=0.0`
- `max_source_selection_quality_regression_count=0`
- `max_audit_proxy_positive_count=0`

只有当全部 distance-rejected context 都是 source-selected，且全部满足上述 path/risk/safety
门槛时，summary 才输出
`recommendation=opt_in_distance_contract_relaxation_profile_ready`。即使该条件成立，输出也
只是 opt-in profile，`default_contract_unchanged=true`，readiness 仍保持 contract refinement
语义，不等同于 PPO readiness。

若存在 not-source-selected distance-rejected context、远距离 7/11 cell context、fallback/open-grid、
safety regression、source-selection regression、audit proxy positive、contract alignment gap 或
provenance mismatch，则输出 `keep_current_training_distance_contract` 或 validation failure。

## Clean Evidence

正式验证在不包含本地 `AGENTS.md` 修改的 clean worktree 中完成，证据根为：

`outputs/path_feedback_batch_anchor_projection_distance_contract_relaxation_safety_audit_v1/`

验证命令：

```bash
R=outputs/path_feedback_batch_anchor_projection_distance_contract_relaxation_safety_audit_v1
bash scripts/run_batch_path_feedback_validation.sh --matrix configs/path_feedback_batch_anchor_projection_distance_contract_relaxation_safety_audit_v1.json --output-root $R
bash scripts/run_anchor_projection_candidate_generation.sh --batch-root $R --config configs/anchor_projection_candidate_generation_v1.json
bash scripts/run_anchor_projection_distance_contract_relaxation_safety_audit.sh --batch-root $R --config configs/anchor_projection_distance_contract_relaxation_safety_audit_v1.json
bash scripts/run_anchor_projection_evidence_contract.sh --batch-root $R --config configs/anchor_projection_evidence_contract_v1.json
bash scripts/run_policy_training_readiness_review.sh --batch-root $R --config configs/policy_training_readiness_review_v1.json
```

验证结果：

- batch：`run_count=8`，`passed_count=8`，`failed_count=0`，
  `open_grid_fallback_used_count=0`。
- candidate-generation：`status=passed`，`reason_codes=[]`，
  `current_git_provenance_mismatch_count=0`，`git_provenance_mismatch_count=0`，
  `trainable_anchor_projection_count=18`，
  `nontrainable_blocked_target_count=60`，
  `source_selected_but_distance_rejected_count=12`，
  `distance_contract_rejected_source_selected_count=12`，
  `nontrainable_source_candidate_not_selected_count=48`，
  `source_selection_quality_regression_count=0`，
  `positive_training_evidence_contains_audit_proxy_anchor_count=0`。
- distance-contract audit：`status=passed`，`reason_codes=[]`，
  `recommendation=keep_current_training_distance_contract`，
  `distance_contract_rejected_count=36`，
  `source_selected_distance_rejected_count=12`，
  `not_source_selected_distance_rejected_count=24`，
  `eligible_source_selected_distance_rejected_count=6`，
  `ineligible_source_selected_distance_rejected_count=6`，
  `ready_for_opt_in_relaxation=false`。
- evidence-contract：`status=passed`，`reason_codes=[]`，
  `contract_blockers=[]`，`candidate_contract_alignment_gap_count=0`。
- readiness：`status=passed`，`reason_codes=[]`，
  `training_readiness_status=needs_training_contract_refinement`，
  `training_blockers=["anchor_projection_nontrainable_contexts_remain"]`。

## Expected Outcome From Prior Evidence

本阶段保持保守结论：

- `78 = 18 trainable + 60 nontrainable`
- distance-rejected 共 36，其中 12 个 source-selected、24 个 not-source-selected
- 3 cells / 1.5 m 有 18 个，7 cells / 3.5 m 有 12 个，11 cells / 5.5 m 有 6 个
- source-selected distance-rejected 的 12 个样本 path/risk margin 为 0，但 6 个 11-cell
  样本超过 opt-in audit 上限，且仍存在 24 个 not-source-selected distance-rejected context

因此推荐结论为 `keep_current_training_distance_contract`，PPO 仍不启动。
