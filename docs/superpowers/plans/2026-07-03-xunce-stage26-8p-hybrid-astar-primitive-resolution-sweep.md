# Stage26.8P Hybrid A* Primitive Resolution Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify whether Stage26.8O trainable sample shortage is caused by coarse Hybrid A* motion primitives.

**Architecture:** Add a Stage26.8P wrapper that repeatedly invokes Stage26.1 collector with identical synthetic terrain, diverse scenario fixture, candidate settings, and reachability guard, changing only `hybrid_astar_primitive_duration_s`. The wrapper is resumable: `run_next` advances one primitive combo, while `aggregate_only` only scans artifacts and rewrites audits.

**Tech Stack:** Python runner, JSON/JSONL artifacts, existing Stage26.1 collector/reward/batch chain, pytest.

---

### Task 1: Runner, Config, And Registry

**Files:**
- Create: `scripts/run_xunce_stage26_8p_hybrid_astar_primitive_resolution_sweep.py`
- Create: `configs/xunce_stage26_8p_hybrid_astar_primitive_resolution_sweep_v1.json`
- Modify: `configs/stage_registry.json`

- [ ] Add config validation for Stage26.8O input root, boundary fields, primitive combo list, and D-drive output root.
- [ ] Implement `run_next` and `aggregate_only`; `run_next` must run at most one primitive combo.
- [ ] Build Stage26.1 configs from the base Stage26.1 config and override only scenario/sample settings, reachability guard, and Hybrid A* primitive fields.
- [ ] Register `xunce-stage26-8p-hybrid-astar-primitive-resolution-sweep`.

### Task 2: Audits And Routing

**Files:**
- Modify: `scripts/run_xunce_stage26_8p_hybrid_astar_primitive_resolution_sweep.py`

- [ ] Read Stage26.1 summary, Stage21.1 trainable rows, and rejection rows for each combo.
- [ ] Write sweep rows with trainable count, terminal count, reachable-count stats, reward/batch row counts, lineage, and runtime.
- [ ] Route to Stage26.8N if a fine primitive reaches `min_trainable_transition_count`.
- [ ] Route to sample expansion with the best primitive if rows improve materially but remain below target.
- [ ] Route to start-pool/candidate reachability repair if all primitive settings remain weak.

### Task 3: Tests And Validation

**Files:**
- Create: `tests/test_xunce_stage26_8p_hybrid_astar_primitive_resolution_sweep.py`

- [ ] Test wrong Stage26.8O route rejection.
- [ ] Test `run_next` executes one combo and only changes primitive duration across combos.
- [ ] Test `aggregate_only` runs no Stage26.1 work.
- [ ] Test fine primitive success route and recommended config.
- [ ] Test weak/no-improvement route and material-short-improvement route.
- [ ] Validate with pytest, py_compile, and run_stage dry-run.

### Task 4: Review Gates

- [ ] Primitive contract review: only primitive resolution changes; reward/network/candidate generation/synthetic terrain remain unchanged.
- [ ] Collector/batch review: Stage26.1 config passes primitive fields and reachability guard; Stage21.3 batch compatibility is not weakened.
- [ ] Result/routing review: conclusions are based on reachable counts, terminal reasons, trainable rows, and runtime diagnostics.
