# Current-HEAD Hybrid Evidence Refresh and Readiness Closure v1

## Summary

证据根：
`outputs/path_feedback_batch_hybrid_current_head_readiness_closure_v1/`。

Hybrid Training Objective Integration v1 已经实现并推送，但旧 evidence root
`outputs/path_feedback_batch_planner_validated_trainable_target_mining_v1/`
中的 readiness review 因 `current_git_provenance_mismatch` 失败。本阶段只在当前
HEAD 下刷新证据、结论和文档，不新增样本类型，不启动正式训练。

## Scope

需要在新 evidence root 下重跑完整链路：

- batch path-feedback validation
- anchor-projection candidate generation
- anchor-projection evidence contract
- planner-validated trainable target mining
- training input materialization
- limited policy training dry-run
- counterfactual preference samples and dry-run
- unified policy sample registry
- residual boundary preference dry-run
- hybrid policy training dry-run
- policy training readiness review

## Expected Evidence

刷新后的 summary 必须保持既有训练信号口径：

- `action_label_positive_count=24`
- `pairwise_preference_signal_count=54`
- `hybrid_train_signal_count=78`
- `hard_positive_added_count=0`
- `invalid_action_mask_count=0`
- `empty_action_mask_count=0`
- `publishes_checkpoint=false`
- `performance_claimed=false`

Readiness review 只有在 provenance、fallback/open-grid、safety、contract 和
hybrid dry-run summary 全部通过时，才能记录
`training_readiness_status=hybrid_training_dry_run_completed`。

## Validation

```bash
ROOT=outputs/path_feedback_batch_hybrid_current_head_readiness_closure_v1
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTHON=$PY bash scripts/run_batch_path_feedback_validation.sh \
  --matrix configs/path_feedback_batch_planner_validated_trainable_target_mining_v1.json \
  --output-root $ROOT

PYTHON=$PY bash scripts/run_anchor_projection_candidate_generation.sh \
  --batch-root $ROOT \
  --config configs/anchor_projection_candidate_generation_v1.json

PYTHON=$PY bash scripts/run_anchor_projection_evidence_contract.sh \
  --batch-root $ROOT \
  --config configs/anchor_projection_evidence_contract_v1.json

PYTHON=$PY bash scripts/run_planner_validated_trainable_target_mining.sh \
  --batch-root $ROOT \
  --config configs/planner_validated_trainable_target_mining_v1.json

PYTHON=$PY bash scripts/run_planner_validated_training_input_materialization.sh \
  --batch-root $ROOT \
  --config configs/planner_validated_training_input_materialization_v1.json

PYTHON=$PY bash scripts/run_limited_policy_training_dry_run.sh \
  --batch-root $ROOT \
  --config configs/limited_policy_training_dry_run_v1.json

PYTHON=$PY bash scripts/run_counterfactual_preference_training_samples.sh \
  --batch-root $ROOT \
  --config configs/counterfactual_preference_training_samples_v1.json

PYTHON=$PY bash scripts/run_counterfactual_preference_training_dry_run.sh \
  --batch-root $ROOT \
  --config configs/counterfactual_preference_training_dry_run_v1.json

PYTHON=$PY bash scripts/run_unified_policy_sample_registry.sh \
  --batch-root $ROOT \
  --config configs/unified_policy_sample_registry_v1.json

PYTHON=$PY bash scripts/run_residual_boundary_preference_training_dry_run.sh \
  --batch-root $ROOT \
  --config configs/residual_boundary_preference_training_dry_run_v1.json

PYTHON=$PY bash scripts/run_hybrid_policy_training_dry_run.sh \
  --batch-root $ROOT \
  --config configs/hybrid_policy_training_dry_run_v1.json

PYTHON=$PY bash scripts/run_policy_training_readiness_review.sh \
  --batch-root $ROOT \
  --config configs/policy_training_readiness_review_v1.json \
  --anchor-projection-candidate-generation-summary \
    $ROOT/anchor-projection-candidate-generation-summary.json \
  --anchor-projection-evidence-contract-summary \
    $ROOT/anchor-projection-evidence-contract-summary.json \
  --planner-validated-trainable-target-mining-summary \
    $ROOT/planner-validated-trainable-target-mining-summary.json \
  --hybrid-policy-training-dry-run-summary \
    $ROOT/hybrid-policy-training-dry-run-summary.json
```

```bash
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

本阶段只关闭 current-HEAD evidence/readiness mismatch。它不启动正式 PPO，不发布 checkpoint，不改
network/action space/default A*，不放宽默认 distance contract，不宣称 Ackermann-feasible
trajectory，不把 IRIS/GCS/path-planner 诊断当训练放行，也不宣称策略性能提升。
