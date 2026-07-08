# Frontier-First Candidate Prefilter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将动态候选生成改为 frontier-first、连通性前置、局部兜底、少量多样性补充，减少无效 endpoint 进入 A*/Hybrid A* 验证。

**Architecture:** 核心实现集中在 `scripts/xunce_dynamic_frontier_nbv.py`：先构造可信 valid/free cells，再生成 raw proposals、执行 cheap prefilter、按 validation budget 送路径验证。theta 与 obstacle-aware 零新增可见过滤在候选扩展/模型输入前完成，不改变 PPO、网络结构、动作空间或 Hybrid A* pose gate。

**Tech Stack:** Python, pytest, existing Xunce scripts/config JSON.

---

### Task 1: Valid Cells And Cheap Prefilter

**Files:**
- Modify: `scripts/xunce_dynamic_frontier_nbv.py`
- Test: `tests/test_xunce_dynamic_frontier_nbv.py`

- [ ] Add `ValidCellsContext` and remove all-bounds fallback.
- [ ] Use `sidecar.passable_mask ∩ roi_valid_cells` when both are present.
- [ ] Reject missing valid-cell source with `valid_cells_source_missing`.
- [ ] Add cheap 8-neighbor BFS distance with no corner cutting.
- [ ] Add audit fields for passable/free/component/distance/reject reason.

### Task 2: Frontier-First Proposal Budgeting

**Files:**
- Modify: `scripts/xunce_dynamic_frontier_nbv.py`
- Test: `tests/test_xunce_dynamic_frontier_nbv.py`

- [ ] Keep `coverage_frontier_boundary` as primary source.
- [ ] Use `undercovered_component_boundary` only as supplement.
- [ ] Disable centroid and low-cost bridge by default.
- [ ] Add `dynamic_path_validation_candidate_budget`; final 6 uses 12, final 36 uses 48.
- [ ] Send only `path_validation_attempted=true` rows into `validate_candidate_cells()`.

### Task 3: Theta And Obstacle-Aware Zero-Gain Filtering

**Files:**
- Modify: `scripts/xunce_theta_viewpoint_candidates.py`
- Modify: `scripts/run_xunce_high_fidelity_exploration_coverage_comparison.py`
- Modify: `scripts/run_xunce_stage21_1_on_policy_ppo_rollout_collector.py`
- Test: `tests/test_xunce_theta_viewpoint_candidates.py`
- Test: `tests/test_xunce_high_fidelity_exploration_coverage_comparison.py`

- [ ] Drop theta viewpoints with `theta_new_visible_cell_count <= 0`.
- [ ] Record raw/drop viewpoint counts.
- [ ] When obstacle-aware LOS is enabled, drop candidates with zero obstacle-aware new visible cells before model input.

### Task 4: Config And Audit

**Files:**
- Modify: `scripts/run_xunce_high_fidelity_exploration_coverage_comparison.py`
- Modify: Stage21-26 config JSON files with dynamic candidate budgets.

- [ ] Expose `path_validation_attempt_count`, `prefilter_reject_reason_counts`, and `valid_cells_source_counts`.
- [ ] Set final=6 configs to raw 24 / validation 12.
- [ ] Set final=36 configs to raw 96 / validation 48.
- [ ] Set final=10 configs to raw 32 / validation 16.

### Task 5: Verification

- [ ] Run `python -m pytest tests/test_xunce_dynamic_frontier_nbv.py -v`.
- [ ] Run `python -m pytest tests/test_xunce_frontier_nbv_validation.py -v`.
- [ ] Run `python -m pytest tests/test_xunce_theta_viewpoint_candidates.py tests/test_xunce_stage21_1_on_policy_ppo_rollout_collector.py -v`.
- [ ] Run `python -m pytest tests/test_xunce_high_fidelity_exploration_coverage_comparison.py -v`.
- [ ] Run Stage26 smoke regressions if time allows.
