# Formal Performance Claim / Release Decision v1

## 背景

Stage 6 `Shadow / Canary Release Performance Validation Preflight v1` 已通过，下一跳为 `formal_performance_claim_release_decision`。本阶段不是训练，也不是发布，而是把当前离线 guarded shadow/canary 证据整理成可审计的性能声明与发布边界决策。

## 目标

新增 `scripts/run_formal_performance_claim_release_decision.py` 与 `scripts/run_formal_performance_claim_release_decision.sh`，输出到 `outputs/path_feedback_batch_formal_performance_claim_release_decision_v1/`。

核心 summary 为 `formal-performance-claim-release-decision-summary.json`，并同时写出 evidence ledger、metric consistency audit、claim scope audit、release boundary audit、provenance audit、decision matrix、scoped claim statement、rejection report 和 markdown report。

## 输入

- Stage 6 summary、long-horizon validation、coverage efficiency audit、guard/fallback audit、kill-switch audit、rollback audit、telemetry audit、eligibility ledger。
- Stage 5B.7 cost-efficiency summary、metric table、guard replay audit。
- Formal training summary、post-training replay summary、selected candidate summary。

## 验收

- `status=passed`、`reason_codes=[]`。
- `decision_verdict=approved_for_scoped_offline_performance_claim`。
- `scoped_offline_performance_claim_approved=true`。
- coverage return、cumulative coverage、valuable coverage improvement 全部大于 0。
- `coverage_efficiency_regression=false`、`controlled_regression_count=0`、`fallback_gain_contamination_count=0`、`fallback_rate<0.5`。
- Stage 6 long-horizon、kill-switch、rollback、telemetry 均 passed。
- `checkpoint_publication_approved=false`、`default_policy_replacement_approved=false`、`real_executor_connection_approved=false`。
- `publishes_checkpoint=false`、`replaces_default_policy=false`、`connects_real_executor=false`。
- 更新项目文档：`README.md` 与 `docs/算法设计与系统架构报告.md` 必须说明 Stage 7 决策范围、允许声明和剩余非目标。

## 当前允许声明

只允许声明：当前 5B.7 candidate 在当前离线 guarded shadow/canary 证据范围内，相比同集合 teacher/pre-improvement baseline 具备受限覆盖性能提升证据。

## 当前结果

Summary:
`outputs/path_feedback_batch_formal_performance_claim_release_decision_v1/formal-performance-claim-release-decision-summary.json`

- `status=passed`
- `reason_codes=[]`
- `decision_verdict=approved_for_scoped_offline_performance_claim`
- `scoped_offline_performance_claim_approved=true`
- `performance_claim_scope=scoped_offline_guarded_shadow_canary_only`
- `coverage_return_improvement=54.96083408637`
- `cumulative_coverage_rate_delta_improvement=56.92136202257`
- `valuable_area_covered_improvement=25.987354935654`
- `coverage_efficiency_regression=false`
- `fallback_rate=0.220611916264`
- `stage6_long_horizon_shadow_passed=true`
- `kill_switch_audit_passed=true`
- `rollback_audit_passed=true`
- `telemetry_audit_passed=true`
- `checkpoint_publication_approved=false`
- `default_policy_replacement_approved=false`
- `real_executor_connection_approved=false`
- `publishes_checkpoint=false`
- `replaces_default_policy=false`
- `connects_real_executor=false`

## 禁止声明

- 不发布 checkpoint。
- 不替换 default policy。
- 不连接真实执行器。
- 不放松 guard。
- 不修改 network/action space/default A*。
- 不声明 Ackermann-feasible trajectory。
- 不把 IRIS/GCS/path-planner 诊断当训练放行。
- 不声明真实世界或无限场景性能。

## 失败路由

Known reason codes include `stage6_not_passed`, `claim_scope_overbroad`, `coverage_metric_inconsistent`, `coverage_efficiency_regression`, `fallback_dominates`, `guard_or_rollback_not_passed`, `telemetry_missing_coverage_gain`, `release_boundary_violation`, and `provenance_or_docs_missing`.

## 验证

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
$PY -m pytest tests/test_formal_performance_claim_release_decision.py tests/test_shadow_canary_release_performance_validation_preflight.py -q
PYTHON=$PY bash scripts/run_formal_performance_claim_release_decision.sh
jq '{status,reason_codes,decision_verdict,scoped_offline_performance_claim_approved,checkpoint_publication_approved,default_policy_replacement_approved,real_executor_connection_approved}' \
  outputs/path_feedback_batch_formal_performance_claim_release_decision_v1/formal-performance-claim-release-decision-summary.json
git diff --check
```
