# Stage26.8A Expand Seed Or Horizon Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a bounded horizon ladder over Stage26.8 with `rollout_steps=12/16/20` to determine whether synthetic credit coverage-efficiency gains need a longer trajectory horizon.

**Architecture:** Stage26.8A is a wrapper over the existing Stage26.8 runner. It generates one Stage26.8 config per horizon, writes each run to an isolated subdirectory, and aggregates the horizon-level `main_coverage_per_100m_delta` result. It does not change reward, network, Hybrid A* search semantics, synthetic terrain generation, or deployment boundaries.

**Tech Stack:** Python stage runner scripts, JSON configs/artifacts, pytest, existing Stage26.8 chain.

---

### Task 1: Horizon Ladder Wrapper

**Files:**
- Create: `scripts/run_xunce_stage26_8a_expand_seed_or_horizon_budget.py`
- Create: `configs/xunce_stage26_8a_expand_seed_or_horizon_budget_v1.json`
- Modify: `configs/stage_registry.json`

- [ ] Add a Stage26.8A runner that reads the clean Stage26.8 `expand_stage26_8_seed_or_horizon_budget` output.
- [ ] Generate Stage26.8 configs for horizons `12`, `16`, and `20`.
- [ ] Force generated Stage26.3 eval configs to use `stage21_5_timeout_seconds=0.0`; long-horizon eval must finish naturally instead of being killed by a 7200-second wrapper timeout.
- [ ] Run each horizon into `h12/s26_8`, `h16/s26_8`, and `h20/s26_8`.
- [ ] Aggregate horizon-level positive, zero, negative, binding/safety, and update counts.
- [ ] Route by Stage26.8 seed pass counts, where the primary efficiency signal is `main_coverage_per_100m_delta` and each seed must still have Stage26.3 pass, non-regressed final main coverage, and clean lineage/safety; keep AUC and path cost diagnostic only.

### Task 2: Tests

**Files:**
- Create: `tests/test_xunce_stage26_8a_expand_seed_or_horizon_budget.py`

- [ ] Test rejection of non-clean Stage26.8 input.
- [ ] Test that H12/H16/H20 configs set collector/eval rollout steps equal to the horizon.
- [ ] Test majority-positive route to Stage26.9.
- [ ] Test single-positive route to seed-budget expansion.
- [ ] Test all-zero route to policy-signal repair.
- [ ] Test boundary, binding/safety, and update-failure route priority.

### Task 3: Documentation And Verification

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/算法设计与系统架构报告.md`
- Modify: `docs/superpowers/specs/2026-06-16-topology-aware-coverage-policy-network-design.md`

- [ ] Record Stage26.8A as the current route.
- [ ] Preserve the contract that `main_coverage_per_100m_delta` is primary and AUC/path cost are diagnostic.
- [ ] Run:

```powershell
python -m pytest tests\test_xunce_stage26_8a_expand_seed_or_horizon_budget.py tests\test_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage26-8a
python -m py_compile scripts\run_xunce_stage26_8a_expand_seed_or_horizon_budget.py scripts\run_xunce_stage26_8_synthetic_terrain_multi_seed_coverage_efficiency_pilot.py
python scripts\run_stage.py --stage xunce-stage26-8a-expand-seed-or-horizon-budget --dry-run
```

### Acceptance Criteria

- Stage26.8A produces summary, horizon results, aggregate, recommended config, routing, report, and manifest.
- H12/H16/H20 use identical seeds and differ only by collector/eval horizon.
- No checkpoint is published, no default policy is replaced, no executor is connected, and no canary is started.
- Three read-only review gates report no Critical or Important findings.
