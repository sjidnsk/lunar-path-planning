# Stage 18.11 Path Cost Weight Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Calibrate Xunce path-cost reward weight under strict v3 evidence while preserving 99% final coverage as the primary mission objective.

**Architecture:** Stage 18.11 is read-only calibration. It consumes strict v3 Stage 18.4E candidate metric audits, performs one-step observed-candidate-set replay across stable path-cost profile variants, and optionally consumes diagnostic reward-rerank oracle rollouts. It does not train, publish, replace the default policy, connect an executor, or start canary traffic.

**Tech Stack:** Python stdlib JSON/JSONL, existing Stage runner, canonical v3 reward helper, pytest.

---

## Summary

- Add `scripts/run_xunce_stage18_11_path_cost_weight_calibration.py`.
- Add stable v3 profile variants that only change `soft_reward_components.path_cost_weight`.
- Add optional `canonical_reward_rerank_oracle` to the high-fidelity coverage runner as diagnostic-only.
- Register `xunce-stage18-11-path-cost-weight-calibration`.
- Route to `stage18_12_rollout_horizon_or_mission_budget_scaling_for_99pct_coverage` when observed and diagnostic final coverage remain below 99%.

## Key Outputs

```text
xunce-stage18-11-path-cost-weight-calibration-summary.json
xunce-stage18-11-weight-replay-results.jsonl
xunce-stage18-11-count-weight-matrix.json
xunce-stage18-11-reward-rerank-command-plan.json
xunce-stage18-11-next-stage-routing.json
xunce-stage18-11-report.md
xunce-stage18-11-manifest.json
```

## Validation

```powershell
python -m pytest `
  tests\test_xunce_stage18_11_path_cost_weight_calibration.py `
  tests\test_xunce_high_fidelity_exploration_coverage_comparison.py `
  tests\test_platform_stage_runner.py `
  tests\test_canonical_reward_profile_v3.py `
  tests\test_canonical_reward_components.py `
  -q --basetemp outputs\pytest-stage18-11-path-cost-weight

python -m py_compile `
  scripts\run_xunce_stage18_11_path_cost_weight_calibration.py `
  scripts\run_xunce_high_fidelity_exploration_coverage_comparison.py `
  model-explorer\src\model_explorer\policy\canonical_reward.py

python scripts\run_stage.py --stage xunce-stage18-11-path-cost-weight-calibration --dry-run
```
