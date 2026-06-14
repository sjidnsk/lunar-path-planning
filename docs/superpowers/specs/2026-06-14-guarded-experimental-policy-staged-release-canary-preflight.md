# Guarded Experimental Policy Staged Release Canary Preflight v1

## Summary

This stage adds an offline canary preflight after `Guarded Experimental Policy
Staged Release Trial v1`. It reads the frozen staged trial summary and activation
ledger, selects a capped set of canary-eligible offline dry-run rows, and audits
the operational guardrails needed before any future guarded canary dry run.

It does not start an online canary and does not connect a real executor.

## Inputs

- `outputs/path_feedback_batch_guarded_experimental_policy_staged_release_trial_v1/guarded-experimental-policy-staged-release-trial-summary.json`
- `outputs/path_feedback_batch_guarded_experimental_policy_staged_release_trial_v1/staged-release-activation-ledger.jsonl`

The staged trial input must be `status=passed`, have no `reason_codes`, report
`staged_release_trial_verdict=eligible_for_guarded_staged_release_canary`, keep
the default policy authoritative, and have no controlled regression.

## Outputs

- `configs/guarded_experimental_policy_staged_release_canary_preflight_v1.json`
- `scripts/run_guarded_experimental_policy_staged_release_canary_preflight.py/.sh`
- `scripts/run_guarded_experimental_policy_staged_release_canary_preflight_closure.sh`
- `tests/test_guarded_experimental_policy_staged_release_canary_preflight.py`
- `outputs/path_feedback_batch_guarded_experimental_policy_staged_release_canary_preflight_v1/`
- `guarded-experimental-policy-staged-release-canary-preflight-summary.json`
- `staged-canary-preflight-manifest.json`
- `staged-canary-eligibility-ledger.jsonl`
- `staged-canary-rejection-report.json`
- `staged-canary-kill-switch-audit.json`
- `staged-canary-rollback-audit.json`
- `staged-canary-telemetry-audit.json`
- `staged-canary-automatic-downgrade-audit.json`
- `staged-canary-budget-audit.json`
- `staged-canary-operator-approval-audit.json`
- `staged-canary-preflight-readiness-validate-only.json`
- `staged-canary-preflight-report.md`

## Eligibility

Only staged trial activation rows that remain gate-clean can become
canary-eligible. The row must use `control_mode=experimental`,
`controlled_choice_source=policy`, empty gate reason codes, empty controlled
regression reason codes, finite log_prob/value/reward/path/risk values, and no
positive controlled path/risk delta.

Fallback, rejected, diagnostic, missing, non-finite, or controlled-regression
rows remain diagnostic-only and cannot enter the eligibility ledger.

## Acceptance Gates

- summary `status=passed`, `reason_codes=[]`
- `staged_release_canary_preflight_verdict=eligible_for_guarded_staged_release_canary_dry_run`
- `staged_canary_enabled=false`
- `connects_real_executor=false`
- `default_policy_authoritative=true`
- `canary_traffic_fraction<=0.01`
- `1 <= canary_eligible_activation_count <= 16`
- diagnostic/fallback/rejected/missing/non-finite/control-regression eligible counts are all 0
- controlled safety/contract/path-risk/source-selection regression counts are all 0
- kill-switch, rollback, telemetry, automatic downgrade, budget, and operator approval audits all pass
- readiness advances to `guarded_experimental_policy_staged_release_canary_preflight_evaluated`

## Non-Goals

- Do not start an online canary.
- Do not connect a real executor or robot.
- Do not run new PPO.
- Do not publish or replace a checkpoint.
- Do not replace the default policy.
- Do not modify network, action space, or default A*.
- Do not relax distance, path-risk, or source-selection gates.
- Do not download new raw data.
- Do not claim Ackermann-feasible trajectory.
- Do not treat IRIS/GCS/path-planner diagnostics as training or release evidence.
- Do not claim policy performance improvement or formal training readiness.
