# Stage26.8C Resume H16/H20 Horizon Efficiency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resume the Stage26.8A long-horizon coverage-efficiency pilot from H16/H20 after Stage26.8B repaired terminal Hybrid A* reachability handling.

**Architecture:** Stage26.8C is a wrapper only. It validates Stage26.8B, reads H12 as a baseline audit, then runs Stage26.8 for H16 and H20 into a new output root.

**Tech Stack:** Python runner/config/tests, existing Stage26.8 multi-seed pilot, JSON/JSONL artifacts.

---

### Task 1: Stage26.8C Runner

**Files:**
- Create: `scripts/run_xunce_stage26_8c_resume_h16_h20_horizon_efficiency.py`

- [ ] Add a Stage26.8B input gate requiring `status=passed` and `next_required_change=resume_stage26_8a_from_h16_h20`.
- [ ] Read Stage26.8A H12 only as baseline; do not rerun or fail on its zero delta.
- [ ] Generate Stage26.8 configs for H16 and H20 with matching collector/eval rollout steps.
- [ ] Force `main_coverable_cells/v1`, `main_coverable_coverage_efficiency/v1`, worker=4, and `stage21_5_timeout_seconds=0.0`.
- [ ] Write summary, horizon results, aggregate, terminal carryover audit, routing, report, and manifest.

### Task 2: Config, Registry, and Tests

**Files:**
- Create: `configs/xunce_stage26_8c_resume_h16_h20_horizon_efficiency_v1.json`
- Modify: `configs/stage_registry.json`
- Create: `tests/test_xunce_stage26_8c_resume_h16_h20_horizon_efficiency.py`
- Modify: `tests/test_platform_stage_runner.py`

- [ ] Register `xunce-stage26-8c-resume-h16-h20-horizon-efficiency`.
- [ ] Test rejection of non-resume Stage26.8B roots.
- [ ] Test only H16/H20 are generated and H12 is not rerun.
- [ ] Test H16/H20 config propagation and no reuse of old H16 failed output.
- [ ] Test route decisions for majority positive, single positive, zero delta, update, binding/safety, credit, and boundary.

### Task 3: Verification

- [ ] Run targeted pytest for Stage26.8C, Stage26.8B, Stage26.8, and stage runner.
- [ ] Run `py_compile` on Stage26.8C and Stage26.8.
- [ ] Run stage dry-run.
- [ ] Run the real Stage26.8C stage and review H16/H20 results.

### Assumptions

- Stage26.8B repaired collector semantics are already present in the working tree.
- H12 remains a baseline audit and is not a success/failure gate.
- AUC, total path length, and Hybrid A* path-cost deltas are diagnostic only.
- No checkpoint is published, no default policy is replaced, no executor is connected, and no canary is started.
