# Xunce Stage 18I Risk-Constrained Frontier-NBV Candidate Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an offline, path-feedback-validated candidate generation stage that creates risk-constrained frontier-guided NBV candidates for Xunce Stage 18 follow-up evidence.

**Architecture:** Stage 18I consumes the Stage 18A ROI expansion root and Stage 18H.0 quantization summary, generates frontier/NBV proposals, only promotes proposals with inherited path-feedback validation, and writes an unbound ROI/path-feedback root compatible with Stage 18B, Stage 18C-v2, Stage 18F, and later Stage 18G.0 binding. It does not train, publish, replace policies, connect an executor, or start canary traffic.

**Tech Stack:** Python 3.12, JSON/JSONL artifacts, existing `run_stage.py` registry, pytest.

---

### Task 1: Add Stage 18I Runner and Config

**Files:**
- Create: `scripts/run_xunce_risk_constrained_frontier_nbv_candidate_generation.py`
- Create: `configs/xunce_risk_constrained_frontier_nbv_candidate_generation_v1.json`
- Modify: `configs/stage_registry.json`

- [ ] Implement CLI support for `--config`, `--output-root`, `--repo-root`, `--source-roi-expansion-root`, and `--source-quantization-root`.
- [ ] Load Stage 18A-compatible `xunce-high-fidelity-real-map-roi-expansion-summary.json`, `xunce-high-fidelity-real-map-slices.jsonl`, and `xunce-high-fidelity-path-feedback-audit.json`.
- [ ] Generate frontier/NBV proposal rows from each scenario, preserving proposal provenance.
- [ ] Promote only proposals with path-feedback validation fields (`reachable`, `path_cost`, `risk`, no open-grid fallback) into formal candidates.
- [ ] Write compatible root files and Stage 18I-specific proposals, validation, rejection, ROI, Pareto, manifest, and report artifacts.
- [ ] Register `xunce-risk-constrained-frontier-nbv-candidate-generation` in `configs/stage_registry.json`.

### Task 2: Add Tests

**Files:**
- Create: `tests/test_xunce_risk_constrained_frontier_nbv_candidate_generation.py`
- Modify: `tests/test_platform_stage_runner.py`

- [ ] Add fixture coverage for 24 scenarios, 8 ROI groups, validated safe-efficient candidates, risky high-coverage candidates, unvalidated positive proposals, low-spread failure, and missing Stage 18A evidence.
- [ ] Assert action indices are stable and equal candidate array position.
- [ ] Assert Stage 18I does not write true incumbent binding fields.
- [ ] Assert platform registry dry-run uses current Python and no shell wrapper.

### Task 3: Update Documentation and Verify

**Files:**
- Modify: `README.md`
- Modify: `docs/算法设计与系统架构报告.md`
- Modify: `docs/superpowers/specs/2026-06-16-topology-aware-coverage-policy-network-design.md`

- [ ] Document that Stage 18I is candidate generation, not model training.
- [ ] Document that the Stage 18I root is unbound and must pass Stage 18B true inference and Stage 18G.0 true incumbent binding before Stage 18H.0.
- [ ] Run the Stage 18I test, registry dry-run, py_compile, and `git diff --check` verification commands.
