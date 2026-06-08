# Anchor Projection Source-Selection & Distance-Contract Calibration v1

## Scope

本阶段基于
`outputs/path_feedback_batch_anchor_reachability_aware_candidate_generation_v1`
后的 current-HEAD evidence，把 anchor projection 训练输入校准从
`anchor_unreachable` 诊断推进到 source-selection 与训练距离 contract 诊断。

非目标保持严格：不启动 PPO，不修改 `training.py`、`ppo.py`、`architectures.py`、
network、action space 或 default A*，不宣称 Ackermann-feasible trajectory，不把 IRIS/GCS
诊断当训练放行依据，不删除压力样本降低失败数。

## Implementation

`anchor-projection-candidate-generation-summary/v1` 新增并在
`anchor_projection_coverage_diagnosis` 中同步保留以下字段：

- `source_selected_but_distance_rejected_count`
- `distance_contract_rejected_source_selected_count`
- `distance_contract_rejected_by_distance_bin`
- `source_candidate_not_selected_by_best_alternative_reason`
- `source_selection_quality_tradeoff_summary`

`anchor-projection-evidence-contract-summary/v1` 原样透传上述字段，并继续以 candidate
summary 为当前 contract source。`policy-training-readiness-review-summary/v1` 在
`anchor_projection_readiness` 和顶层 `anchor_projection_*` 字段中消费这些值，用于解释
剩余 blocker，不改变训练放行阈值。

## Evidence

证据根：
`outputs/path_feedback_batch_anchor_projection_source_selection_distance_contract_v1/`。

验证结果：

- batch：`passed_count=8`，`failed_count=0`，`open_grid_fallback_used_count=0`。
- candidate-generation：`status=passed`，`reason_codes=[]`，
  `current_git_provenance_mismatch_count=0`，`git_provenance_mismatch_count=0`，
  `fallback_or_open_grid_count=0`，`safety_regression_count=0`。
- evidence-contract：`status=passed`，`contract_blockers=[]`，
  `candidate_contract_alignment_gap_count=0`。
- readiness：`status=passed`，
  `training_readiness_status=needs_training_contract_refinement`，
  `training_blockers=["anchor_projection_nontrainable_contexts_remain"]`。

## Nontrainable Decomposition

总量仍为 `platform_goal_contract_mismatch_count=78`，
`trainable_anchor_projection_count=18`，
`nontrainable_blocked_target_count=60`。

60 个 nontrainable context 的完整分解：

- `source_candidate_not_selected=48`。
- `source_selected_but_distance_rejected_count=12`。
- `distance_contract_rejected_by_distance_bin.count=36`，
  `source_selected_count=12`，`not_source_selected_count=24`。
- 距离分箱：
  - 3 cells / 1.5 m：18 个，6 个 source-selected，12 个 not-selected，
    scenario 为 `npz_near_blocked_corridor`。
  - 7 cells / 3.5 m：12 个，全部 not-selected，
    scenario 为 `npz_dense_rock_choke`。
  - 11 cells / 5.5 m：6 个，全部 source-selected，
    scenario 为 `npz_dense_rock_choke`。
- 48 个 not-selected 的归因：
  `distance_contract_rejected=24`、`higher_path_cost_and_risk=12`、
  `higher_path_cost=12`、`higher_risk=0`、`lower_utility_or_coverage=0`、
  `ranking_weight_tradeoff_or_unobserved_utility=0`、
  `no_selected_candidate_comparison=0`。

## Decision

`source_selection_quality_regression_count=0`，且 source-selected distance-rejected 的
12 个样本相对 selected candidate 的 path/risk margin 均为 0；但这些样本仍超过当前
`max_trainable_projection_distance_cells=2` /
`max_trainable_projection_distance_m=1.0`。本阶段不放宽默认训练 contract，只记录：

`distance_contract_relaxation_recommendation=record_only_keep_current_training_distance_contract`

下一阶段应推进 Distance-Contract Relaxation Safety Audit：若要放宽距离阈值，必须证明远距离
reachable substitute 在 path/risk/safety、scenario/backend 分布和 readiness alignment 上无回退；
否则继续保持 audit-only，PPO 仍不启动。
