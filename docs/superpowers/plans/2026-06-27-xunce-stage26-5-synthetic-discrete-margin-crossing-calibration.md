# Stage26.5 Synthetic Discrete Margin Crossing Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose why Stage26 synthetic terrain PPO updates move probabilities but do not change the selected `(x,y,theta)` action.

**Architecture:** Stage26.5 is a read-only audit runner over Stage26.4 artifacts. It strong-joins each combo's pre/post inference rows, measures selected-vs-best candidate logit/probability/rank gap closure, checks whether the best counterfactual candidate ever received direct PPO credit, and audits whether candidate features expose synthetic LOS and Hybrid A* path-cost signals.

**Tech Stack:** Python runner, JSON/JSONL artifacts, pytest, existing Stage26 output conventions.

---

### Task 1: Stage26.5 Runner

**Files:**
- Create: `scripts/run_xunce_stage26_5_synthetic_discrete_margin_crossing_calibration.py`
- Create: `configs/xunce_stage26_5_synthetic_discrete_margin_crossing_calibration_v1.json`
- Modify: `configs/stage_registry.json`

- [ ] Write failing tests for Stage26.5 input routing, worker-count routing, margin trend calculations, credit binding, and feature exposure.
- [ ] Implement the read-only runner with `xunce-stage26-5-summary.json`, margin audit, logit/rank trend JSONL, credit audit, feature audit, recommended config, routing, report, and manifest.
- [ ] Register `xunce-stage26-5-synthetic-discrete-margin-crossing-calibration`.

### Task 2: Documentation

**Files:**
- Create: `docs/superpowers/plans/2026-06-27-xunce-stage26-5-synthetic-discrete-margin-crossing-calibration.md`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/superpowers/specs/2026-06-16-topology-aware-coverage-policy-network-design.md`
- Modify: `docs/算法设计与系统架构报告.md`

- [ ] Document that Stage26.5 skips Stage26.4A equivalence and uses Stage26.4 worker=4 as the default evidence source.
- [ ] Document that Stage26.5 is diagnostic-only and does not run PPO or publish checkpoints.

### Task 3: Verification

**Commands:**

```powershell
python -m pytest tests\test_xunce_stage26_5_synthetic_discrete_margin_crossing_calibration.py tests\test_xunce_stage26_4_synthetic_policy_update_signal_strength_repair.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage26-5
python -m py_compile scripts\run_xunce_stage26_5_synthetic_discrete_margin_crossing_calibration.py scripts\run_xunce_stage26_4_synthetic_policy_update_signal_strength_repair.py
python scripts\run_stage.py --stage xunce-stage26-5-synthetic-discrete-margin-crossing-calibration --dry-run
```

Expected: tests and compile pass; dry-run resolves the new stage.
