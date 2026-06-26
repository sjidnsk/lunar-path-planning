# Stage26.4A Parallelize Stage21.1 Hybrid A* Candidate Costs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Stage21.1 collector Hybrid A* candidate path-cost evaluation configurable and candidate-parallel, then verify worker=4 is equivalent to worker=1 on Stage26 synthetic terrain lineage.

**Architecture:** Stage21.1 keeps serial behavior by default. When `hybrid_astar_candidate_eval_workers > 1`, it uses a process pool per candidate batch and restores results by candidate index. Stage26.4A wraps Stage26.1 twice and compares serial/parallel outputs without running PPO.

**Tech Stack:** Python, pytest, ProcessPoolExecutor, existing Stage21/Stage26 runners.

---

### Task 1: Stage21.1 Collector Parallel Candidate Evaluation

**Files:**
- Modify: `scripts/run_xunce_stage21_1_on_policy_ppo_rollout_collector.py`
- Test: `tests/test_xunce_stage21_1_on_policy_ppo_rollout_collector.py`

- [ ] Add failing tests for worker count, candidate order restoration, and per-candidate failure isolation.
- [ ] Add `hybrid_astar_candidate_eval_workers`, default `1`, to Stage21.1 config normalization and high-fidelity overrides.
- [ ] Add a module-level worker function for Windows-safe process pool execution.
- [ ] In `_hybrid_astar_path_cost_metadata`, use `ProcessPoolExecutor` only when Hybrid A* is enabled, workers > 1, and more than one candidate exists.
- [ ] Restore all rows by `candidate_index` and write audit fields into metadata.
- [ ] Keep serial behavior unchanged for worker `1`.

### Task 2: Stage26 Wrapper Propagation

**Files:**
- Modify: `scripts/run_xunce_stage26_1_synthetic_terrain_collector_smoke.py`
- Modify: `scripts/run_xunce_stage26_4_synthetic_policy_update_signal_strength_repair.py`
- Modify: `configs/xunce_stage26_1_synthetic_terrain_collector_smoke_v1.json`
- Test: `tests/test_xunce_stage26_1_synthetic_terrain_collector_smoke.py`

- [ ] Add failing test proving Stage26.1 writes the worker count into generated Stage21.1 config.
- [ ] Add config normalization in Stage26.1.
- [ ] Pass the worker count through Stage26.1 to Stage21.1.
- [ ] Pass Stage26.4 worker count through its Stage26.1 rerun config.

### Task 3: Stage26.4A Serial/Parallel Equivalence Runner

**Files:**
- Create: `scripts/run_xunce_stage26_4a_parallelize_stage21_1_hybrid_astar_candidate_costs.py`
- Create: `configs/xunce_stage26_4a_parallelize_stage21_1_hybrid_astar_candidate_costs_v1.json`
- Test: `tests/test_xunce_stage26_4a_parallelize_stage21_1_hybrid_astar_candidate_costs.py`

- [ ] Add tests for serial/parallel equivalence pass and mismatch routing.
- [ ] Run Stage26.1 twice: `serial_workers_1/s26_1` and `parallel_workers_4/s26_1`.
- [ ] Compare selected action, candidate set hash, Hybrid A* costs, pose path hashes, reachability, synthetic lineage, and platform lineage.
- [ ] Write runtime, equivalence, worker, routing, report, summary, and manifest artifacts.
- [ ] Route clean equivalence to `rerun_stage26_4_synthetic_policy_update_signal_strength_with_parallel_collector`.

### Task 4: Registry, Docs, Verification

**Files:**
- Modify: `configs/stage_registry.json`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/算法设计与系统架构报告.md`
- Modify: `docs/superpowers/specs/2026-06-16-topology-aware-coverage-policy-network-design.md`

- [ ] Register the new stage id.
- [ ] Document that Stage26.4A parallelizes candidate evaluations only, not a single Hybrid A* search.
- [ ] Run targeted pytest, py_compile, and stage dry-run.
- [ ] Confirm no release/default-policy/executor/canary boundary fields are opened.
