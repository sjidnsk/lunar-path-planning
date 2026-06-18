# Xunce Exploration Coverage Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Stage 18C to compare Xunce and incumbent policies on multi-step offline exploration coverage.

**Architecture:** Reuse Stage 18B checkpoint loading and true inference scoring, then wrap it in a 24-scenario x 10-step offline shadow rollout that updates deterministic coverage memory. Path-feedback rows provide candidates, path/risk/cost, and coverage evidence, but model action selection must come from true checkpoint inference.

**Tech Stack:** Python 3.12, PyTorch, pytest, JSON/JSONL artifacts, existing Xunce Stage 18A/18B evidence roots.

---

## Tasks

- [x] Add `tests/test_xunce_high_fidelity_exploration_coverage_comparison.py` with true tiny Xunce/incumbent checkpoint fixtures.
- [x] Add `scripts/run_xunce_high_fidelity_exploration_coverage_comparison.py`.
- [x] Add `configs/xunce_high_fidelity_exploration_coverage_comparison_v1.json`.
- [x] Register `xunce-high-fidelity-exploration-coverage-comparison` in `configs/stage_registry.json`.
- [x] Extend platform stage runner tests so the new stage dry-runs through the Python runner.
- [x] Update `README.md`, `docs/算法设计与系统架构报告.md`, and `docs/superpowers/specs/2026-06-16-topology-aware-coverage-policy-network-design.md`.

## Outputs

Default output root:

```text
outputs/path_feedback_batch_xunce_high_fidelity_exploration_coverage_comparison_v1/
```

Artifacts:

```text
xunce-exploration-coverage-comparison-summary.json
xunce-exploration-coverage-episodes.jsonl
xunce-exploration-coverage-steps.jsonl
xunce-exploration-coverage-model-inference.jsonl
xunce-exploration-coverage-roi-breakdown.json
xunce-exploration-coverage-decision-audit.json
xunce-exploration-coverage-comparison-report.md
```

## Acceptance Criteria

- `true_model_inference_executed=true` and `proxy_selection_used=false`.
- Both checkpoints load read-only; missing or unsupported checkpoints fail explicitly.
- Xunce and incumbent run over the same scenario, step, candidate batch, and action mask.
- Masked or unreachable candidates are not selected.
- Coverage memory updates across 10 rollout steps; revisits do not count as new coverage.
- Summary includes coverage, efficiency, safety/feasibility, and model-behavior metrics.
- Coverage advantage requires positive coverage return and coverage AUC deltas with no mask, fallback, safety, path-cost, or risk-efficiency regression.
- The stage never approves default-policy replacement, checkpoint publication, PPO training, executor connection, online canary, or real-world performance claims.

## Verification

```powershell
D:\conda_envs\lunar-explorer\python.exe -m pytest `
  tests\test_xunce_high_fidelity_exploration_coverage_comparison.py `
  tests\test_xunce_high_fidelity_real_map_comparison.py `
  tests\test_xunce_high_fidelity_real_map_roi_expansion.py `
  tests\test_platform_stage_runner.py `
  -q

D:\conda_envs\lunar-explorer\python.exe scripts\run_stage.py `
  --stage xunce-high-fidelity-exploration-coverage-comparison `
  --dry-run
```

## Non-Goals

- Do not start PPO.
- Do not publish checkpoints.
- Do not replace default policy.
- Do not connect a real executor or online canary.
- Do not modify action space, default A*, or model-explorer contract.
- Do not download new LOLA or illumination/shadow products by default.
