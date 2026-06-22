# Xunce Stage 18.9 Trajectory Risk Boundary Reward Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate hard risk filtering from reward ranking and make whole-trajectory audit the Stage 19 preflight authority.

**Architecture:** Add path-level risk semantics, canonical reward/guard profile v3, and Stage 18.9 trajectory audit. Stage 18.6/18.7 remain diagnostics and cannot independently authorize readiness.

**Tech Stack:** Python stdlib JSON/JSONL, existing Stage runner, `model_explorer.policy.canonical_reward`, pytest.

---

## Tasks

- [x] Add `scripts/xunce_path_risk_semantics.py` and write path risk boundary fields from A* validation.
- [x] Add `configs/xunce_canonical_reward_guard_profile_v3.json` and v3 reward component support.
- [x] Add `scripts/run_xunce_stage18_9_trajectory_risk_boundary_reward_audit.py` with summary, JSONL, routing, report, and manifest artifacts.
- [x] Register `xunce-stage18-9-trajectory-risk-boundary-reward-audit` in `configs/stage_registry.json`.
- [x] Extend Stage 18 pipeline with optional `stage18_9_trajectory_risk_reward_root`.
- [x] Extend Stage 18.4E artifacts with path-level hard/soft risk fields for candidate, episode, and pair outputs.
- [x] Gate v3 PPO consumers on Stage 18.9 readiness.
- [x] Update README, architecture report, and topology-aware policy spec.

## Verification

```powershell
python -m pytest tests\test_xunce_path_risk_semantics.py tests\test_canonical_reward_profile_v3.py tests\test_xunce_stage18_9_trajectory_risk_boundary_reward_audit.py -q --basetemp outputs\pytest-stage18-9-core
python -m pytest tests\test_xunce_stage18_research_evidence_pipeline.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage18-9-pipeline
python -m py_compile scripts\xunce_path_risk_semantics.py scripts\run_xunce_stage18_9_trajectory_risk_boundary_reward_audit.py scripts\xunce_stage18_pipeline.py
python scripts\run_stage.py --stage xunce-stage18-9-trajectory-risk-boundary-reward-audit --dry-run
```

## Non-Goals

- Do not start PPO.
- Do not publish checkpoints.
- Do not replace default policy.
- Do not connect a real executor.
- Do not start online canary.
- Do not claim physical risk or Ackermann feasibility.
