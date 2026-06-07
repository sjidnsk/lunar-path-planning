# Anchor Projection Candidate Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move platform blocked anchor projection from audit-only evidence into an opt-in source candidate generation path.

**Architecture:** Keep existing route and contract schemas stable, add additive metadata to `model-explorer` path candidate evaluations, and expose the behavior through an opt-in path feedback validation flag. A root summary script will verify trainable vs audit-only counts against current-HEAD provenance.

**Tech Stack:** Python dataclasses, pytest/unittest, Bash validation wrappers, JSON summaries.

---

### Task 1: Model Explorer Candidate Generation

**Files:**
- Modify: `model-explorer/src/model_explorer/policy/planning.py`
- Modify: `model-explorer/src/model_explorer/policy/feedback_selection.py`
- Test: `model-explorer/tests/test_model_explorer.py`

- [ ] **Step 1: Write failing tests**

Add tests proving opt-in projected candidates replan to the nearest inflated-passable anchor, carry `trainable_anchor_projection_contrast`, and stay non-trainable when the anchor is unreachable.

- [ ] **Step 2: Run tests to verify RED**

Run:
`PYTHONPATH=model-explorer/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/kai/anaconda3/envs/lunar-explorer/bin/python -m pytest model-explorer/tests/test_model_explorer.py::PathPlanningAdapterTests -q`

Expected: new anchor projection tests fail because no opt-in generation API exists.

- [ ] **Step 3: Implement minimal candidate generation**

Add optional evaluation config and projected execution target evaluation after an original `platform_inflated_goal_blocked` result has a reachable anchor within thresholds.

- [ ] **Step 4: Verify GREEN**

Run the same pytest target and confirm all `PathPlanningAdapterTests` pass.

### Task 2: Opt-In Validation Wiring

**Files:**
- Modify: `model-explorer/src/model_explorer/policy/path_feedback.py`
- Modify: `scripts/run_path_feedback_validation.sh`
- Create: `configs/path_feedback_batch_anchor_projection_candidate_generation_v1.json`
- Test: `tests/test_batch_path_feedback_validation.py`

- [ ] **Step 1: Write failing tests**

Add tests that a manifest planner setting enables anchor projection candidate generation without changing default validation behavior.

- [ ] **Step 2: Implement opt-in flag**

Add `--anchor-projection-candidate-generation` to `run_path_feedback_validation.sh`, store it in manifest planner config and acceptance metadata, and pass it to `evaluate_candidate_paths`.

- [ ] **Step 3: Verify focused tests**

Run:
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/kai/anaconda3/envs/lunar-explorer/bin/python -m pytest tests/test_batch_path_feedback_validation.py model-explorer/tests/test_model_explorer.py::PathPlanningAdapterTests -q`

### Task 3: Candidate Generation Summary

**Files:**
- Create: `configs/anchor_projection_candidate_generation_v1.json`
- Create: `scripts/run_anchor_projection_candidate_generation.py`
- Create: `scripts/run_anchor_projection_candidate_generation.sh`
- Test: `tests/test_anchor_projection_candidate_generation.py`

- [ ] **Step 1: Write failing summary tests**

Cover trainable count, nontrainable count, audit proxy exclusion, source selected candidate changed rate, provenance mismatch, fallback and safety guards.

- [ ] **Step 2: Implement summary script**

Read batch summaries and scenario path feedback candidates, classify only source-selected projected candidates as trainable, and fail validation when provenance or safety gates fail.

- [ ] **Step 3: Verify summary tests**

Run:
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/kai/anaconda3/envs/lunar-explorer/bin/python -m pytest tests/test_anchor_projection_candidate_generation.py tests/test_anchor_projection_evidence_contract.py -q`

### Task 4: Evidence Refresh and Documentation

**Files:**
- Modify: `docs/算法设计与系统架构报告.md`
- Create/Modify: `docs/superpowers/specs/2026-06-07-anchor-projection-candidate-generation.md`

- [ ] **Step 1: Run current-HEAD evidence chain**

Generate `outputs/path_feedback_batch_anchor_projection_candidate_generation_v1/` with the opt-in matrix, then run downstream validate-only stages and the new candidate generation summary.

- [ ] **Step 2: Update docs**

Record output root, key counts, commands, source/trainable vs audit-only boundary, and non-goals.

- [ ] **Step 3: Final verification**

Run focused pytest and validate-only commands fresh before claiming completion.
