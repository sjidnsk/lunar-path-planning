# Stage24.0 Hybrid A* Pose Path Planner Full Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an opt-in Scout Mini differential/skid-steer Hybrid A* foundation that plans to `(x,y,theta)` pose goals.

**Architecture:** Add a separate `path_planner.search.hybrid_astar` backend without changing the default grid A*. Stage24.0 uses bounded offline smoke scenarios to prove pose-state planning, turn-in-place primitives, rectangular footprint collision, and 30 degree slope-obstacle hard gates.

**Tech Stack:** Python, numpy, pytest, existing `path-planner` core models, Xunce stage runner conventions.

---

### Task 1: Hybrid A* Planner

**Files:**
- Create: `path-planner/src/path_planner/search/hybrid_astar.py`
- Modify: `path-planner/src/path_planner/search/__init__.py`
- Test: `tests/test_xunce_stage24_0_hybrid_astar_pose_path_planner_full_foundation.py`

- [ ] Add `Pose2D`, `MotionPrimitive`, `PosePlanRequest`, `PosePlanResult`, and `HybridAStarPlanner`.
- [ ] Keep old `PlanRequest`, `PlanResult`, and `AStarPlanner` unchanged.
- [ ] Implement differential/skid-steer primitives including turn-in-place.
- [ ] Use rectangular footprint collision and hard obstacle passability.
- [ ] Emit `trajectory_kind=hybrid_astar_pose_path` and `ackermann_feasible_claimed=false`.

### Task 2: Stage24.0 Runner

**Files:**
- Create: `scripts/run_xunce_stage24_0_hybrid_astar_pose_path_planner_full_foundation.py`
- Create: `configs/xunce_stage24_0_hybrid_astar_pose_path_planner_full_foundation_v1.json`
- Modify: `configs/stage_registry.json`
- Test: `tests/test_platform_stage_runner.py`

- [ ] Load Scout Mini platform contract and verify `max_traversable_slope_deg=30.0`.
- [ ] Run smoke cases for turn-in-place, straight pose path, slope obstacle detour, and blocked footprint.
- [ ] Write summary, smoke JSONL, grid-vs-hybrid audit, footprint audit, routing, report, and manifest.
- [ ] Keep all release/default-policy/executor/canary fields false or zero.

### Task 3: Tests And Verification

**Files:**
- Test: `tests/test_xunce_stage24_0_hybrid_astar_pose_path_planner_full_foundation.py`
- Test: `path-planner/tests/test_hybrid_astar.py`

- [ ] Verify Hybrid A* changes theta while staying in the same cell.
- [ ] Verify blocked footprint returns `goal_blocked`.
- [ ] Verify slope-blocked proxy cells are treated as hard obstacles.
- [ ] Verify Stage24.0 routes to Stage24.1 when smoke passes.
- [ ] Verify `run_stage.py --dry-run` supports Stage24.0.

### Task 4: Documentation

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/算法设计与系统架构报告.md`
- Modify: `docs/superpowers/specs/2026-06-16-topology-aware-coverage-policy-network-design.md`

- [ ] Document Stage24.0 as opt-in Hybrid A* foundation.
- [ ] State it does not replace default grid A*, run PPO, publish checkpoints, connect executor, or claim performance.
- [ ] State the platform model is differential/skid-steer, not Ackermann.

### Verification Commands

```powershell
python -m pytest tests\test_xunce_stage24_0_hybrid_astar_pose_path_planner_full_foundation.py path-planner\tests\test_astar.py path-planner\tests\test_planning_grid.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage24-0
python -m py_compile path-planner\src\path_planner\search\hybrid_astar.py scripts\run_xunce_stage24_0_hybrid_astar_pose_path_planner_full_foundation.py
python scripts\run_stage.py --stage xunce-stage24-0-hybrid-astar-pose-path-planner-full-foundation --dry-run
```
