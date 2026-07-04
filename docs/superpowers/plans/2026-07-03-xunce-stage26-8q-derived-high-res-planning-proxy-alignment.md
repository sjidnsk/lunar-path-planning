# Stage26.8Q Derived High-Res Planning Proxy Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a collector-only Stage26.8Q audit that tests whether a derived 1m Hybrid A* planning proxy fixes Stage26 synthetic terrain trainable sample scarcity.

**Architecture:** Keep the original Stage26 synthetic sidecar immutable and derive an opt-in planning-only `CostGrid` for Hybrid A* path-cost evaluation. The policy action space, candidate indices, masks, behavior logprob, reward, PPO and network remain unchanged.

**Tech Stack:** Python runners, JSON/JSONL artifacts, pytest, existing Stage26.1/Stage21.1/Hybrid A* modules.

---

### Task 1: Hybrid A* closed-key resolution opt-in

**Files:**
- Modify: `path-planner/src/path_planner/search/hybrid_astar.py`
- Test: `tests/test_xunce_stage26_8q_derived_high_res_planning_proxy_alignment.py`

- [ ] Add optional `PosePlanRequest.closed_key_xy_resolution_m`.
- [ ] Keep default `None` behavior using `(grid cell, theta_bin)`.
- [ ] When provided, key by metric `floor((x-origin)/resolution)` and write diagnostics `xy_resolution_m_theta_bin/v1`.

### Task 2: Derived planning proxy and world-goal binding

**Files:**
- Modify: `scripts/xunce_hybrid_astar_candidate_path_cost.py`
- Modify: `scripts/run_xunce_stage21_1_on_policy_ppo_rollout_collector.py`
- Test: `tests/test_xunce_stage26_8q_derived_high_res_planning_proxy_alignment.py`

- [ ] Add `build_derived_high_res_planning_proxy_grid()` with conservative constant inheritance.
- [ ] Keep `candidate_viewpoint` as coarse action cell.
- [ ] In proxy mode, add `candidate_goal_world_pose` from source-grid cell center.
- [ ] Pass `closed_key_xy_resolution_m` through serial and parallel Hybrid A* evaluation.
- [ ] Emit proxy provenance in transition info.

### Task 3: Stage26.8Q runner

**Files:**
- Create: `scripts/run_xunce_stage26_8q_derived_high_res_planning_proxy_alignment.py`
- Create: `configs/xunce_stage26_8q_derived_high_res_planning_proxy_alignment_v1.json`
- Modify: `configs/stage_registry.json`
- Test: `tests/test_xunce_stage26_8q_derived_high_res_planning_proxy_alignment.py`

- [ ] Implement `run_next` and `aggregate_only`.
- [ ] Compare `current_baseline` with `aligned_1m_proxy`.
- [ ] Audit trainable rows, reachable counts, terminal reasons, proxy hash and binding mismatches.
- [ ] Route to Stage26.8N if aligned proxy reaches 100 trainable rows; otherwise route to sample expansion or start/candidate reachability repair.

### Task 4: Validation

- [ ] Run:

```powershell
python -m pytest tests\test_xunce_stage26_8q_derived_high_res_planning_proxy_alignment.py tests\test_xunce_stage26_8p_hybrid_astar_primitive_resolution_sweep.py tests\test_xunce_stage26_1_synthetic_terrain_collector_smoke.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage26-8q
python -m py_compile scripts\run_xunce_stage26_8q_derived_high_res_planning_proxy_alignment.py scripts\xunce_hybrid_astar_candidate_path_cost.py scripts\run_xunce_stage21_1_on_policy_ppo_rollout_collector.py
python scripts\run_stage.py --stage xunce-stage26-8q-derived-high-res-planning-proxy-alignment --dry-run
```

- [ ] Use subagent read-only reviews for proxy contract, collector binding, and final result semantics before claiming completion.
