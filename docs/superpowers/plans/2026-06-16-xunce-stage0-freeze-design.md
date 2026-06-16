# Xunce Stage 0 Freeze Design Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the current `巡策 / Topology-Aware Coverage Policy Network Design` as a clean, documented, auditable research baseline before any evidence refresh or network implementation.

**Architecture:** Stage 0 is a documentation and evidence-boundary audit, not a training or release stage. A small runner checks the required docs, the complete 0-17 stage chain, plan-first rule, non-goal boundaries, and key upstream Global 99 summaries, then writes frozen audit artifacts under an independent output root.

**Tech Stack:** Python 3 standard library, existing root `scripts/` audit style, pytest, markdown docs, JSON artifacts.

---

## Scope

This stage freezes the design baseline only. It does not add a policy architecture, train PPO, publish a checkpoint, modify action space/default A*, replace default policy, connect a real executor, or start online canary.

## Input Evidence

- `README.md`
- `docs/算法设计与系统架构报告.md`
- `docs/superpowers/specs/2026-06-16-global-99-exploration-coverage-line.md`
- `docs/superpowers/specs/2026-06-16-topology-aware-coverage-policy-network-design.md`
- `outputs/path_feedback_batch_global_99_controlled_default_policy_candidate_installation_preflight_v1/global-99-controlled-default-policy-candidate-installation-preflight-summary.json`
- `outputs/path_feedback_batch_network_architecture_upgrade_readiness_review_v1/network-architecture-upgrade-readiness-summary.json`
- `outputs/path_feedback_batch_global_99_real_map_multi_roi_generalization_v1/global-99-real-map-multi-roi-generalization-summary.json`

## Output Artifacts

Output root: `outputs/path_feedback_batch_xunce_design_freeze_v1/`

- `xunce-design-freeze-summary.json`
- `xunce-design-freeze-manifest.json`
- `xunce-design-document-audit.json`
- `xunce-design-evidence-audit.json`
- `xunce-design-boundary-audit.json`
- `xunce-design-freeze-rejection-report.json`
- `xunce-design-freeze-report.md`

## Interface

Config path: `configs/xunce_design_freeze_v1.json`

Required config fields:

- `schema_version="xunce-design-freeze-config/v1"`
- `documents`
- `source_controlled_installation_root`
- `source_network_readiness_root`
- `source_real_map_multi_roi_root`
- `require_complete_stage_chain=true`
- `require_plan_first_rule=true`
- `require_closed_boundaries=true`

Runner entrypoints:

- `scripts/run_xunce_design_freeze_audit.py`
- `scripts/run_xunce_design_freeze_audit.sh`

Pass `next_required_change`: `current_head_evidence_refresh`

Failure `next_required_change`:

- `fix_xunce_design_freeze` for missing/incomplete docs.
- `fix_global_99_evidence_chain` for missing/failed/conflicting source evidence.

## Tasks

### Task 1: Add Stage 0 Failing Tests

**Files:**
- Create: `tests/test_xunce_design_freeze_audit.py`

- [x] **Step 1: Write tests for pass, incomplete stage chain, and source boundary violation.**
- [x] **Step 2: Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/test_xunce_design_freeze_audit.py -q`.**
- [x] **Step 3: Confirm tests fail because `scripts.run_xunce_design_freeze_audit` is missing.**

### Task 2: Implement Runner, Config, and Shell Entrypoint

**Files:**
- Create: `scripts/run_xunce_design_freeze_audit.py`
- Create: `scripts/run_xunce_design_freeze_audit.sh`
- Create: `configs/xunce_design_freeze_v1.json`

- [ ] **Step 1: Implement config loading and path normalization.**
- [ ] **Step 2: Implement document audit for `巡策`, `Topology-Aware Coverage Policy Network Design`, all stages `0` through `17`, plan-first rule, and non-goal phrases.**
- [ ] **Step 3: Implement source evidence audit for controlled installation, network readiness, and real-map multi-ROI summaries.**
- [ ] **Step 4: Implement boundary audit for checkpoint/default-policy/executor/online-canary/PPO/network/action-space/default-A* fields.**
- [ ] **Step 5: Write summary, manifest, audit, rejection, and report artifacts.**
- [ ] **Step 6: Run the Stage 0 test and confirm it passes.**

### Task 3: Update Project Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/算法设计与系统架构报告.md`
- Modify: `docs/superpowers/specs/2026-06-16-global-99-exploration-coverage-line.md`
- Modify: `docs/superpowers/specs/2026-06-16-topology-aware-coverage-policy-network-design.md`

- [ ] **Step 1: Replace the short 5-step development order with the complete 0-17 stage chain in the 巡策 spec.**
- [ ] **Step 2: Add the Stage 0 runner/config/output root to README and architecture report.**
- [ ] **Step 3: Keep Global 99 spec clear that 巡策 is parallel research, not a release gate.**

### Task 4: Verify and Publish Stage 0

**Files:**
- Stage all Stage 0 files and docs only.

- [ ] **Step 1: Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/test_xunce_design_freeze_audit.py -q`.**
- [ ] **Step 2: Run `PYTHON=$PY bash scripts/run_xunce_design_freeze_audit.sh`.**
- [ ] **Step 3: Run `rg -n "巡策|Topology-Aware|xunce|current_head_evidence_refresh" README.md docs scripts tests configs`.**
- [ ] **Step 4: Run `git diff --check`.**
- [ ] **Step 5: Commit with message `Add xunce design freeze audit`.**
- [ ] **Step 6: Push current branch and verify remote HEAD matches local HEAD.**

## Acceptance Gate

- Stage 0 summary is `status=passed`.
- `next_required_change=current_head_evidence_refresh`.
- `complete_stage_chain_count=18`.
- All boundary fields remain false.
- Required docs contain the complete 0-17 chain and plan-first rule.
- Stage 0 tests pass.
- The commit is pushed and remote HEAD matches local HEAD.
