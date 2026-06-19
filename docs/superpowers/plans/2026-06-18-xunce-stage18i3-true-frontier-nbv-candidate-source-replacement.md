# Xunce Stage 18I.3 True Frontier-NBV Candidate Source Replacement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Replace the relabel-only Stage 18I.2 candidate source with an offline frontier/NBV proposal source, validate proposals with path-feedback-backed evidence, fix Stage 18C-v2 oracle rollout semantics, and rerun Xunce vs incumbent comparison on the new root.

**Architecture:** Stage 18I.3 emits an unbound Stage 18A-compatible candidate root. New proposals are generated from ROI geometry, start cell, and true incumbent binding, but only exact path-feedback-backed validated rows enter the formal action set. Stage 18C-v2 separates checkpoint model inference from offline oracle rollout and no longer moves dynamic candidate cells without validated path evidence.

**Tech Stack:** Python 3.12, `pathlib`, JSON/JSONL artifacts, `model-explorer` path-feedback evidence, pytest, Windows non-Drake offline profile.

---

## Summary

Stage 18I.2 showed that the previous implementation still depended on old candidate rows. Stage 18I.3 adds a real proposal source and a fail-closed validation adapter:

- Generate `incumbent_neighborhood`, `frontier_boundary`, and `roi_undercovered_boundary` proposals from ROI/start/incumbent geometry.
- Keep proposals as `proposal_only=true` until validation attaches `reachable`, `path_cost`, `risk`, and `open_grid_fallback_used=false`.
- Write a Stage 18A-compatible root for Stage 18B/G/H/F/C consumption.
- Fix Stage 18C-v2 so oracle baselines execute offline rollout and dynamic candidates require validated path evidence.

## Key Changes

- Add `scripts/run_xunce_true_frontier_nbv_candidate_source_replacement.py`, `scripts/xunce_frontier_nbv_validation.py`, and `configs/xunce_true_frontier_nbv_candidate_source_replacement_v1.json`.
- Register `xunce-true-frontier-nbv-candidate-source-replacement` in `configs/stage_registry.json`.
- Update `scripts/run_xunce_high_fidelity_exploration_coverage_comparison.py` so oracle policies use `policy_inference_kind=oracle_offline_policy` and can execute without checkpoint inference, while Xunce/incumbent still require real checkpoint inference.
- Map legacy `dynamic_from_coverage_memory` to `dynamic_validated_only`; dynamic candidates must have validated path-feedback fields or remain static.
- Update Stage 18I closure audit to recognize `xunce-true-frontier-nbv-candidate-source-summary.json`.

## Test Plan

Run:

```powershell
D:\conda_envs\lunar-explorer\python.exe -m pytest `
  tests\test_xunce_frontier_nbv_validation.py `
  tests\test_xunce_true_frontier_nbv_candidate_source_replacement.py `
  tests\test_xunce_high_fidelity_exploration_coverage_comparison.py `
  tests\test_xunce_stage18i_evidence_closure_audit.py `
  tests\test_platform_stage_runner.py `
  -q

D:\conda_envs\lunar-explorer\python.exe scripts\run_stage.py `
  --stage xunce-true-frontier-nbv-candidate-source-replacement `
  --dry-run

D:\conda_envs\lunar-explorer\python.exe -m py_compile `
  scripts\run_xunce_true_frontier_nbv_candidate_source_replacement.py `
  scripts\xunce_frontier_nbv_validation.py `
  scripts\run_xunce_high_fidelity_exploration_coverage_comparison.py `
  scripts\run_xunce_stage18i_evidence_closure_audit.py `
  scripts\run_stage.py

git diff --check
```

## Assumptions

- The v1 validation adapter is deliberately conservative: it uses exact-cell matching against existing path-feedback-backed candidates and fails closed when no evidence exists.
- This stage does not train actor/critic, start PPO, publish checkpoints, replace the default policy, connect an executor, start online canary, or download new maps.
- A real after-Stage18I.3 comparison still requires rerunning Stage 18B, Stage 18G.0, Stage 18H.0, Stage 18F, and Stage 18C-v2 on the new unbound root.
