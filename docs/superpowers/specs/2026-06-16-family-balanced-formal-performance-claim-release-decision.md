# Family-Balanced Formal Performance Claim / Release Decision v1

## Summary

This stage reviews the passed `Family-Balanced Shadow/Canary Preflight v1` evidence and decides whether a scoped offline family-balanced guarded shadow/canary performance claim is allowed.

It is a decision/audit stage only. It does not train PPO, publish a checkpoint, replace the default policy, connect a real executor, or authorize Stage 17 default-policy installation.

## Inputs

- `outputs/path_feedback_batch_family_balanced_shadow_canary_preflight_v1/family-balanced-shadow-canary-preflight-summary.json`
- Family-balanced shadow/canary long-horizon, family-generalization, efficiency, guard/fallback, kill-switch, rollback, telemetry, lineage, and release-boundary audits.
- `outputs/path_feedback_batch_family_balanced_coverage_driven_ppo_rerun_v1/family-balanced-coverage-driven-ppo-rerun-summary.json`
- `outputs/path_feedback_batch_family_balanced_algorithm_gap_closure_v1/family-balanced-algorithm-gap-closure-summary.json`
- Formal training, post-training replay, and selected candidate summaries.
- `README.md` and `docs/算法设计与系统架构报告.md`.

## Outputs

- `family-balanced-formal-performance-claim-release-decision-summary.json`
- `family-balanced-performance-evidence-ledger.json`
- `family-balanced-metric-consistency-audit.json`
- `family-balanced-claim-scope-audit.json`
- `family-balanced-low-observation-limitation-audit.json`
- `family-balanced-release-boundary-audit.json`
- `family-balanced-provenance-audit.json`
- `family-balanced-decision-matrix.json`
- `family-balanced-scoped-claim-statement.md`
- `family-balanced-formal-performance-claim-release-decision-rejection-report.json`
- `family-balanced-formal-performance-claim-release-decision-report.md`

## Acceptance

The summary passes only when:

- `status=passed`
- `reason_codes=[]`
- `decision_verdict=approved_for_family_balanced_scoped_offline_performance_claim`
- `family_balanced_scoped_offline_performance_claim_approved=true`
- `allowed_claim_scope=scoped_offline_family_balanced_guarded_shadow_canary_only`
- aggregate coverage return, cumulative coverage rate delta, and valuable area covered improvements are positive
- `coverage_efficiency_regression=false`
- `fallback_rate<0.5`
- `fallback_gain_contamination_count=0`
- `controlled_regression_count=0`
- `family_generalization_audit_passed=true`
- `low_observation_shadow_passed=true`
- `low_observation_limitation_acknowledged=true`
- `low_observation_coverage_return_outperforms_teacher=false`
- `next_required_change=family_balanced_scoped_claim_publication_evidence_freeze`
- `checkpoint_publication_approved=false`
- `default_policy_replacement_approved=false`
- `real_executor_connection_approved=false`

## Low-Observation Boundary

The low-observation family is explicitly a scoped limitation. Current evidence may pass low-observation shadow checks while still showing negative low-observation coverage-return improvement relative to teacher. The claim text must say this limitation directly and must not claim low-observation coverage return outperforms teacher.

中文边界：低观测 family 是本阶段必须公开写入的限制项，不能把 aggregate family-balanced 性能声明扩大成“低观测 coverage return 已超过 teacher”。

## Failure Reasons

- `family_balanced_shadow_canary_not_passed`
- `long_horizon_shadow_not_passed`
- `coverage_metric_inconsistent`
- `claim_scope_overbroad`
- `low_observation_limitation_missing`
- `low_observation_overclaimed`
- `coverage_efficiency_regression`
- `fallback_dominates`
- `fallback_gain_contamination`
- `controlled_regression_detected`
- `guard_rollback_or_telemetry_not_passed`
- `lineage_incomplete`
- `release_boundary_violation`
- `provenance_or_docs_missing`

## Validation

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
$PY -m pytest tests/test_family_balanced_formal_performance_claim_release_decision.py \
  tests/test_family_balanced_shadow_canary_preflight.py -q

PYTHON=$PY bash scripts/run_family_balanced_formal_performance_claim_release_decision.sh

jq '{status,reason_codes,decision_verdict,family_balanced_scoped_offline_performance_claim_approved,low_observation_limitation_acknowledged,checkpoint_publication_approved,default_policy_replacement_approved,real_executor_connection_approved}' \
  outputs/path_feedback_batch_family_balanced_formal_performance_claim_release_decision_v1/family-balanced-formal-performance-claim-release-decision-summary.json

git diff --check
```

## Non-Goals

不发布 checkpoint，不替换 default policy，不连接真实执行器，不恢复 Stage 17 default-policy installation，不运行 PPO，不扩大 rollout，不修改 reward，不修改 network/action space/default A*，不放松 guard，不声明真实世界性能，不声明 Ackermann-feasible trajectory，不把 IRIS/GCS/path-planner 诊断当发布证明。
