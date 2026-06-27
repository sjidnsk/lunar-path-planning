# Stage26.5B Documentation Boundary Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split Xunce stage documentation responsibilities so `AGENTS.md`, `README.md`, long-form docs, plans, reports, and stage registry no longer duplicate the same stage history.

**Architecture:** Keep `AGENTS.md` and `README.md` short, add a documentation index as the source of truth for file responsibilities, and add a read-only runner that audits the boundary. Do not change PPO, reward, Hybrid A*, candidate generation, checkpoints, policies, executors, or canary behavior.

**Tech Stack:** Markdown, JSON stage registry, Python read-only audit runner, pytest.

---

### Task 1: Add Boundary Regression Test

**Files:**
- Create: `tests/test_xunce_documentation_boundary_consolidation.py`

- [ ] Write tests that fail when `AGENTS.md` is longer than 180 lines, has more than 5 stage sections, misses current contracts, or omits the documentation index.
- [ ] Test that `README.md` has `Documentation Map` and the current Stage26.5 route.
- [ ] Test that `docs/xunce-stage-documentation-index.md` lists file responsibilities and current route.
- [ ] Test that `configs/stage_registry.json` keeps Stage26.5 and registers Stage26.5B.

### Task 2: Consolidate Human And Agent Docs

**Files:**
- Modify: `AGENTS.md`
- Modify: `README.md`
- Create: `docs/xunce-stage-documentation-index.md`

- [ ] Rewrite `AGENTS.md` as a short boundary document with language, file safety, D-drive output, git protection, current hard boundaries, current contracts, and only recent key stage summaries.
- [ ] Rewrite `README.md` as a project overview, current route summary, contract summary, and documentation map.
- [ ] Add the documentation index explaining where plans, results, architecture, specs, registry, README, and AGENTS content belong.

### Task 3: Mark Long-Form Docs As Architecture/Spec Sources

**Files:**
- Modify: `docs/算法设计与系统架构报告.md`
- Modify: `docs/superpowers/specs/2026-06-16-topology-aware-coverage-policy-network-design.md`

- [ ] Add current contract summary and documentation-boundary notes.
- [ ] Do not delete historical content in this stage; only clarify that future smoke execution logs belong in outputs and plans.

### Task 4: Add Read-Only Stage Audit

**Files:**
- Create: `scripts/run_xunce_stage26_5b_documentation_boundary_consolidation.py`
- Create: `configs/xunce_stage26_5b_documentation_boundary_consolidation_v1.json`
- Modify: `configs/stage_registry.json`
- Modify: `tests/test_platform_stage_runner.py`

- [ ] Implement a read-only runner that writes summary, audit, routing, report, and manifest artifacts.
- [ ] Route to `continue_stage26_6_synthetic_exploration_credit_assignment` only when docs and registry boundaries pass.
- [ ] Register `xunce-stage26-5b-documentation-boundary-consolidation`.

### Task 5: Verify

- [ ] Run `python -m pytest tests\test_xunce_documentation_boundary_consolidation.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage26-5b`.
- [ ] Run `python -m py_compile scripts\run_xunce_stage26_5b_documentation_boundary_consolidation.py`.
- [ ] Run `python scripts\run_stage.py --stage xunce-stage26-5b-documentation-boundary-consolidation --dry-run`.
- [ ] Run `python scripts\run_stage.py --stage xunce-stage26-5b-documentation-boundary-consolidation`.
