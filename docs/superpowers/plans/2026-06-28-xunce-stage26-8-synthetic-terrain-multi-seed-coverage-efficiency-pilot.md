# Stage26.8 Synthetic Terrain Multi-Seed Coverage Efficiency Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a bounded multi-seed synthetic-terrain pilot whose primary success metric is main-coverable coverage per 100m.

**Architecture:** Stage26.8 wraps the existing Stage26.1 -> Stage26.2 -> Stage26.3 chain per seed. Stage21.5 and Stage26.3 gain an opt-in efficiency success metric while their legacy AUC/final-coverage behavior remains the default.

**Tech Stack:** Python stage runners, JSON configs, pytest, existing Stage26 synthetic terrain artifacts.

---

### Task 1: Efficiency Metric Contract

**Files:**
- Modify: `scripts/run_xunce_stage21_5_post_update_offline_trajectory_evaluation.py`
- Modify: `scripts/run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke.py`
- Test: `tests/test_xunce_stage21_5_post_update_offline_trajectory_evaluation.py`
- Test: `tests/test_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke.py`

- [ ] Add `post_update_success_metric=main_coverable_coverage_efficiency/v1`.
- [ ] In Stage21.5, keep legacy AUC/final coverage as default.
- [ ] In efficiency mode, gate on final coverage and coverage per 100m; keep AUC and path cost diagnostic.
- [ ] Preserve safety, unreachable, fallback, and binding hard failures.
- [ ] Run targeted pytest for Stage21.5 and Stage26.3 metric cases.

### Task 2: Stage26.8 Multi-Seed Runner

**Files:**
- Create: `scripts/run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot.py`
- Create: `configs/xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot_v1.json`
- Test: `tests/test_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot.py`

- [ ] Validate Stage26.7H clean lineage before running.
- [ ] For seeds `[260801, 260802, 260803]`, run Stage26.1, Stage26.2, and Stage26.3 independently.
- [ ] Force `coverage_denominator_source=main_coverable_cells/v1` and `post_update_success_metric=main_coverable_coverage_efficiency/v1`.
- [ ] Aggregate seed-level efficiency, binding, safety, and update status.
- [ ] Route to Stage26.9 only when the majority of clean seeds improve main coverage per 100m.

### Task 3: Registry, Docs, Verification

**Files:**
- Modify: `configs/stage_registry.json`
- Modify: `tests/test_platform_stage_runner.py`
- Modify: `README.md`, `AGENTS.md`, and long-form docs as short current-state references.

- [ ] Register `xunce-stage26-8-synthetic-terrain-multi-seed-coverage-efficiency-pilot`.
- [ ] Add dry-run support test coverage.
- [ ] Document that AUC is a diagnostic, not the Stage26.8 primary gate.
- [ ] Run full Stage26.8 verification commands.

### Review Gates

- [ ] Review 1: metric contract review.
- [ ] Review 2: multi-seed runner review.
- [ ] Review 3: full result review.

### Acceptance Criteria

- Stage26.8 summary is readable and reports seed-level efficiency.
- AUC and total path cost are diagnostic only.
- Safety, unreachable, binding, fallback, checkpoint, executor, and canary boundaries remain strict.
- No checkpoint is published and default policy is not replaced.
