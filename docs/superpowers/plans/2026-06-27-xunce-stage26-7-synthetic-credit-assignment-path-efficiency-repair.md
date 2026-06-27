# Stage26.7 Synthetic Credit Assignment Path-Efficiency Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair Stage26.6 synthetic credit assignment so direct-credit target selection favors coverage-per-Hybrid-A* path-cost instead of local coverage alone.

**Architecture:** Keep Stage26.6 feature exposure, behavior logprob, continuous theta, synthetic proxy terrain, and PPO plumbing unchanged. Add a path-efficiency score version in the synthetic credit helper, propagate audit fields through Stage21.1, and run a Stage26.7 bounded v1/v2 target-score sweep through Stage26.1 -> Stage26.2 -> Stage26.3.

**Tech Stack:** Python, pytest, JSON/JSONL stage artifacts, existing Stage21/26 runners.

---

### Task 1: Path-Efficiency Target Contract

**Files:**
- Modify: `scripts/xunce_synthetic_exploration_credit.py`
- Modify: `scripts/run_xunce_stage21_1_on_policy_ppo_rollout_collector.py`
- Modify: `scripts/run_xunce_stage26_1_synthetic_terrain_collector_smoke.py`
- Test: `tests/test_xunce_stage26_7_synthetic_credit_assignment_path_efficiency_repair.py`

- [ ] Add `path_efficiency_v2` target scoring with explicit Hybrid A* path-cost and synthetic pressure penalties.
- [ ] Add `path_efficiency_max_cost_norm` filtering with `path_efficiency_filter_relaxed` audit when strict filtering leaves no valid candidate.
- [ ] Propagate `synthetic_credit_score_version`, `selected_target_hybrid_cost_norm`, and `selected_target_gain_per_cost_norm` into Stage21.1 transition/info rows.
- [ ] Keep synthetic terrain as `synthetic_terrain_obstacle_proxy/v1`; do not write physical obstacle payload.

### Task 2: Stage26.7 Runner

**Files:**
- Create: `scripts/run_xunce_stage26_7_synthetic_credit_assignment_path_efficiency_repair.py`
- Create: `configs/xunce_stage26_7_synthetic_credit_assignment_path_efficiency_repair_v1.json`
- Modify: `configs/stage_registry.json`
- Modify: `tests/test_platform_stage_runner.py`

- [ ] Validate Stage26.6 input: `status=failed`, `next_required_change=repair_stage26_synthetic_credit_assignment`, feature/logprob fixed, target rows selected.
- [ ] Run score sweep combos: `v1_current_baseline`, `path_efficiency_v2_default`, and `path_efficiency_v2_strict_cost`.
- [ ] For each combo, orchestrate Stage26.1 -> Stage26.2 -> Stage26.3 into isolated subdirectories.
- [ ] Emit summary, sweep JSONL, target audit, behavior audit, eval comparison, recommended configs, routing, report, and manifest.

### Task 3: Review And Verification

**Files:**
- Test: `tests/test_xunce_stage26_7_synthetic_credit_assignment_path_efficiency_repair.py`

- [ ] Review gate 1: score/filter contract, synthetic proxy semantics, no reward/network/Hybrid A* semantic changes.
- [ ] Review gate 2: collector propagation, Stage21.3 behavior logprob audit, v2 target selected rows.
- [ ] Review gate 3: full-chain result, path-cost delta, coverage-per-100m, safety/boundary.
- [ ] Run:

```powershell
python -m pytest tests\test_xunce_stage26_7_synthetic_credit_assignment_path_efficiency_repair.py tests\test_xunce_stage26_6_synthetic_exploration_credit_assignment.py tests\test_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage26-7
python -m py_compile scripts\run_xunce_stage26_7_synthetic_credit_assignment_path_efficiency_repair.py scripts\xunce_synthetic_exploration_credit.py scripts\run_xunce_stage21_1_on_policy_ppo_rollout_collector.py
python scripts\run_stage.py --stage xunce-stage26-7-synthetic-credit-assignment-path-efficiency-repair --dry-run
```

### Assumptions

- Stage26.7 inherits continuous theta from Stage25/26; it does not introduce a new action-space change.
- Stage26.7 does not publish checkpoints, replace the default policy, connect executor, or start canary.
- Stage26.7 does not change reward objectives, network structure, default A*, Hybrid A* search semantics, or synthetic terrain generation.
