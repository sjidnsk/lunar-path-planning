# Family-Balanced Shadow/Canary Preflight v1

## Background

`Family-Balanced Coverage-Driven PPO Rerun v1` passed and routed to
`family_balanced_shadow_canary_preflight`. This stage validates the rerun
checkpoint and evidence under an offline guarded shadow/canary preflight before
any formal scoped performance claim decision.

It is not a release stage and does not continue Stage 17 default-policy
installation.

## Implementation

- Runner: `scripts/run_family_balanced_shadow_canary_preflight.py`
- Shell entrypoint: `scripts/run_family_balanced_shadow_canary_preflight.sh`
- Output root:
  `outputs/path_feedback_batch_family_balanced_shadow_canary_preflight_v1/`

The runner reads the family-balanced rerun root, gap-closure summary, formal
training summary, post-training replay summary, and selected candidate summary.
It uses the rerun compatible performance metric table as the fair same-input
baseline and writes long-horizon, family generalization, coverage efficiency,
guard/fallback, kill-switch, rollback, telemetry, lineage, and release-boundary
audits.

## Output Contract

The stage writes:

- `family-balanced-shadow-canary-preflight-summary.json`
- `family-balanced-shadow-canary-runtime-manifest.json`
- `family-balanced-shadow-canary-step-comparison.jsonl`
- `family-balanced-shadow-canary-long-horizon-validation.json`
- `family-balanced-shadow-canary-family-generalization-audit.json`
- `family-balanced-shadow-canary-coverage-efficiency-audit.json`
- `family-balanced-shadow-canary-guard-fallback-audit.json`
- `family-balanced-shadow-canary-kill-switch-audit.json`
- `family-balanced-shadow-canary-rollback-audit.json`
- `family-balanced-shadow-canary-telemetry-audit.json`
- `family-balanced-shadow-canary-lineage-audit.json`
- `family-balanced-shadow-canary-release-boundary-audit.json`
- `family-balanced-shadow-canary-rejection-report.json`
- `family-balanced-shadow-canary-preflight-report.md`

Passing summary contract:

- `status=passed`
- `reason_codes=[]`
- `preflight_verdict=eligible_for_family_balanced_formal_performance_claim_release_decision`
- `family_balanced_shadow_canary_preflight_passed=true`
- `family_balanced_formal_performance_claim_release_decision_approved=true`
- `long_horizon_shadow_passed=true`
- `coverage_return_improvement>0`
- `cumulative_coverage_rate_delta_improvement>0`
- `valuable_area_covered_improvement>0`
- `coverage_efficiency_regression=false`
- `family_generalization_audit_passed=true`
- `low_observation_shadow_passed=true`
- `fallback_rate<0.5`
- `fallback_gain_contamination_count=0`
- `controlled_regression_count=0`
- `kill_switch_audit_passed=true`
- `rollback_audit_passed=true`
- `telemetry_audit_passed=true`
- `next_required_change=family_balanced_formal_performance_claim_release_decision`
- release boundaries remain false.

## Failure Reasons

- `family_balanced_rerun_not_passed`
- `family_balanced_rerun_not_routed_to_shadow_canary`
- `long_horizon_coverage_regression`
- `family_generalization_regression`
- `low_observation_shadow_regression`
- `coverage_efficiency_regression`
- `fallback_dominates`
- `fallback_gain_contamination`
- `controlled_regression_detected`
- `shadow_policy_took_control`
- `kill_switch_failed`
- `rollback_failed`
- `telemetry_missing_coverage_gain`
- `lineage_incomplete`
- `release_boundary_violation`
- `docs_not_updated`

## Verification

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
$PY -m pytest tests/test_family_balanced_shadow_canary_preflight.py \
  tests/test_family_balanced_coverage_driven_ppo_rerun.py \
  tests/test_shadow_canary_release_performance_validation_preflight.py -q

PYTHON=$PY bash scripts/run_family_balanced_shadow_canary_preflight.sh

jq '{status,reason_codes,preflight_verdict,long_horizon_shadow_passed,coverage_return_improvement,cumulative_coverage_rate_delta_improvement,valuable_area_covered_improvement,coverage_efficiency_regression,fallback_rate,next_required_change,checkpoint_publication_approved,default_policy_replacement_approved,real_executor_connection_approved}' \
  outputs/path_feedback_batch_family_balanced_shadow_canary_preflight_v1/family-balanced-shadow-canary-preflight-summary.json

git diff --check
```

## Documentation

Update `README.md` and `docs/算法设计与系统架构报告.md` with the stage status,
output root, next gate, and non-release boundary.

## Non-Goals

- 不训练 PPO。
- 不发布 checkpoint。
- 不替换 default policy。
- 不连接真实执行器。
- 不继续 Stage 17 default-policy installation。
- 不放松 guard。
- 不修改 network/action space/default A*。
- 不声明 Ackermann-feasible trajectory。
- 不声明真实世界性能。
- 不把 IRIS/GCS/path-planner 诊断当发布证明。
