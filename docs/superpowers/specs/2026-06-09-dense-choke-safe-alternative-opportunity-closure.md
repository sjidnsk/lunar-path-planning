# Dense-Choke Safe Alternative Opportunity Closure v1

## Background

`policy_gated_canary_opportunity_quality_evaluated` 已证明 6 类 canary family 中有 5 类具备安全改选机会。唯一缺口是
`dense_choke_safe_bypass`：旧 summary 中 4 个 context、12 个 alternative、`acceptable_alternative_count=0`，
拒绝原因集中在 `risk_regression=12`、`path_cost_regression=8`、`invalid_action_mask=4`。

本阶段不让 policy 更激进，也不放宽 gate；目标是修复 dense-choke opportunity generation，让它产生真实合同内、
action-mask 可消费、path/risk 不回退的 safe bypass alternative。

## Scope

- 新增 `policy_canary_dense_choke_opportunity` scenario set。
- 新增 full-family canary matrix/config：
  - `configs/path_feedback_batch_policy_gated_canary_full_family_opportunity_v1.json`
  - `configs/policy_gated_canary_full_family_opportunity_v1.json`
- 新增 dense-choke diagnosis：
  - `scripts/run_dense_choke_safe_alternative_diagnosis.py`
  - `scripts/run_dense_choke_safe_alternative_diagnosis.sh`
  - `dense-choke-safe-alternative-diagnosis-summary.json`
  - `dense-choke-safe-alternative-diagnosis.md`
  - `outputs/dense_choke_safe_alternative_visual_diagnostics_v1/index.html`
- 新增 closure wrapper：
  - `scripts/run_dense_choke_safe_alternative_opportunity_closure.sh`
- 扩展 canary/readiness：
  - `dense_choke_acceptable_alternative_count`
  - `dense_choke_accepted_policy_choice_count`
  - `canary_full_family_opportunity_passed`
  - readiness status `policy_gated_canary_full_family_opportunity_evaluated`
  - dense-specific `next_required_change=dense_choke_opportunity_generation_gap`

## Evidence Roots

- SRC: `outputs/path_feedback_batch_dense_choke_opportunity_clean_src_v1/`
- CAND: `outputs/path_feedback_batch_dense_choke_opportunity_candidate_v1/`
- CANARY: `outputs/path_feedback_batch_policy_gated_canary_full_family_opportunity_v1/`

## Clean Result

Clean-HEAD closure passed:

- Batch `failed_count=0`, fallback/open-grid=0.
- Canary `status=passed`, `reason_codes=[]`.
- `policy_decision_count=32`.
- `canary_opportunity_context_count=32`.
- `policy_changed_decision_count=18`.
- `canary_accepted_policy_choice_count=18`.
- `canary_rejected_policy_choice_count=0`.
- `scenario_family_count=6`.
- `family_with_acceptable_alternative_count=6`.
- `accepted_scenario_family_count=6`.
- `dense_choke_acceptable_alternative_count=8`.
- `dense_choke_accepted_policy_choice_count=8`.
- `canary_full_family_opportunity_passed=true`.
- Candidate/checkpoint provenance matches current source.
- controlled/raw regression and all invalid-mask/fallback/safety/contract/path/risk/source-selection gates are 0.
- Readiness `training_readiness_status=policy_gated_canary_full_family_opportunity_evaluated`, `training_blockers=[]`.

## Acceptance

- Batch `failed_count=0`; fallback/open-grid=0; safety regression=0.
- Candidate/checkpoint provenance match current source.
- `scenario_family_count=6`.
- `family_with_acceptable_alternative_count=6`.
- `accepted_scenario_family_count=6`.
- `dense_choke_acceptable_alternative_count>0`.
- `dense_choke_accepted_policy_choice_count>0`.
- `canary_accepted_policy_choice_count>=12`.
- `canary_rejected_policy_choice_count=0`.
- controlled/raw regression=0.
- invalid action mask, fallback/open-grid, safety, contract, path/risk, source-selection regression all 0.
- readiness `status=passed`, `training_readiness_status=policy_gated_canary_full_family_opportunity_evaluated`, `training_blockers=[]`.

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTHON=$P bash scripts/run_dense_choke_safe_alternative_diagnosis.sh \
  --batch-root outputs/path_feedback_batch_policy_gated_canary_opportunity_quality_v1

PYTHON=$P bash scripts/run_batch_path_feedback_validation.sh \
  --matrix configs/path_feedback_batch_policy_gated_canary_full_family_opportunity_v1.json

PYTHON=$P bash scripts/run_dense_choke_safe_alternative_opportunity_closure.sh

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_dense_choke_safe_alternative_opportunity.py \
  tests/test_policy_gated_canary_rollout.py \
  tests/test_policy_training_readiness_review.py \
  tests/test_batch_path_feedback_validation.py
```

## Non-Goals

- No formal PPO rollout.
- No checkpoint publication or default policy replacement.
- No network/action-space/default-A* change.
- No distance-contract relaxation.
- No Ackermann-feasible trajectory claim.
- No use of IRIS/GCS/path-planner diagnostics as training release evidence.
- No policy performance claim.
