# Shadow / Canary Release Performance Validation Preflight v1

## 背景

5B.7 `Cost-Efficiency-Aware Coverage Reward / Candidate Filter v1` 已通过，当前下一跳是 `shadow_canary_release_performance_validation_preflight`。本阶段只验证候选策略是否具备进入发布决策审查的离线证据，不发布 checkpoint，不替换 default policy，不连接真实执行器。

## 目标

新增 `scripts/run_shadow_canary_release_performance_validation_preflight.py` 与 `scripts/run_shadow_canary_release_performance_validation_preflight.sh`，输出到 `outputs/path_feedback_batch_shadow_canary_release_performance_validation_preflight_v1/`。

核心 summary 为 `shadow-canary-release-performance-validation-preflight-summary.json`，并同时写出 runtime manifest、shadow step comparison、long-horizon validation、coverage efficiency audit、guard/fallback audit、kill-switch audit、rollback audit、telemetry audit、eligibility ledger、rejection report 和 markdown report。

## 输入

- 5B.7 summary、filtered batch、refined transitions、guard replay audit、metric table、stage5a rerun、compatible performance table。
- formal training summary、post-training stability replay summary、selected candidate preflight summary。
- 不得把旧 5A.2 51 pair 或旧 selected-formal shadow 当当前模型证据。

## 验收

- `status=passed`、`reason_codes=[]`、`long_horizon_shadow_passed=true`。
- coverage return、cumulative coverage、valuable coverage improvement 全部大于 0。
- `coverage_efficiency_regression=false`、`controlled_regression_count=0`、`fallback_gain_contamination_count=0`、`fallback_rate<0.5`。
- `shadow_policy_takes_control=false`、`experimental_control_activation_count=0`、`kill_switch_audit_passed=true`、`rollback_audit_passed=true`、`telemetry_audit_passed=true`。
- `publishes_checkpoint=false`、`replaces_default_policy=false`、`connects_real_executor=false`、`performance_claimed=false`。
- 更新项目文档：`README.md` 与 `docs/算法设计与系统架构报告.md` 必须说明 Stage 6 产物、验证边界和剩余非目标。

## 当前结果

Summary:
`outputs/path_feedback_batch_shadow_canary_release_performance_validation_preflight_v1/shadow-canary-release-performance-validation-preflight-summary.json`

- `status=passed`
- `reason_codes=[]`
- `next_required_change=formal_performance_claim_release_decision`
- `long_horizon_shadow_passed=true`
- horizons: `[10,20,30]`
- `coverage_return_improvement=54.96083408637`
- `cumulative_coverage_rate_delta_improvement=56.92136202257`
- `valuable_area_covered_improvement=25.987354935654`
- `coverage_efficiency_regression=false`
- `fallback_rate=0.220611916264`
- `safe_better_training_family_count=4`
- `controlled_regression_count=0`
- `fallback_gain_contamination_count=0`
- `shadow_policy_takes_control=false`
- `experimental_control_activation_count=0`
- `kill_switch_audit_passed=true`
- `rollback_audit_passed=true`
- `telemetry_audit_passed=true`
- `publishes_checkpoint=false`
- `replaces_default_policy=false`
- `connects_real_executor=false`
- `performance_claimed=false`

The long-horizon validation writes all/family/scenario/context rollups. Local negative windows are retained as diagnostic risk evidence; the pass gate is aggregate all-horizon improvement under the offline guard boundary.

## 失败路由

Known reason codes include `input_cost_efficiency_stage_not_passed`, `long_horizon_coverage_regression`, `coverage_efficiency_regression`, `shadow_controlled_regression`, `fallback_dominates`, `shadow_policy_took_control`, `kill_switch_failed`, `rollback_failed`, `telemetry_missing_coverage_gain`, and `family_balance_failed`.

## 验证

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
$PY -m pytest tests/test_shadow_canary_release_performance_validation_preflight.py tests/test_cost_efficiency_aware_coverage_reward_candidate_filter.py -q
PYTHON=$PY bash scripts/run_shadow_canary_release_performance_validation_preflight.sh
jq '{status,reason_codes,next_required_change,long_horizon_shadow_passed,coverage_return_improvement,coverage_efficiency_regression,fallback_rate,kill_switch_audit_passed,rollback_audit_passed,telemetry_audit_passed,performance_claimed}' \
  outputs/path_feedback_batch_shadow_canary_release_performance_validation_preflight_v1/shadow-canary-release-performance-validation-preflight-summary.json
git diff --check
```

## 非目标

- 不训练 PPO。
- 不发布 checkpoint。
- 不替换 default policy。
- 不连接真实执行器。
- 不放松 guard。
- 不修改 network/action space/default A*。
- 不声明 Ackermann-feasible trajectory。
- 不把 IRIS/GCS/path-planner 诊断当训练放行。
- 不声明正式性能提升。
