# Xunce Stage 1 Current-HEAD Evidence Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify that the just-frozen 巡策 design baseline is anchored to the current clean Git HEAD and that key Global 99 / real-map / controlled-installation evidence is still status-clean and boundary-clean before literature or network work starts.

**Architecture:** Stage 1 is an evidence-refresh audit. It consumes Stage 0 output plus selected Global 99 evidence summaries, checks current worktree cleanliness, validates source statuses and boundary fields, enforces provenance match for sources that declare `git_provenance.current`, records legacy missing-provenance counts without silently treating them as current, and emits a pass/fail next gate.

**Tech Stack:** Python 3 standard library, existing `git_provenance.py`, pytest, JSON audit artifacts, markdown report.

---

## Scope

This stage does not search literature, change observation schema, implement a network, train PPO, publish a checkpoint, replace default policy, connect executor, or start canary. Its only job is to decide whether current evidence is clean enough to enter `network_literature_bottleneck_review`.

## Input Evidence

- `outputs/path_feedback_batch_xunce_design_freeze_v1/xunce-design-freeze-summary.json`
- `outputs/path_feedback_batch_global_99_controlled_default_policy_candidate_installation_preflight_v1/global-99-controlled-default-policy-candidate-installation-preflight-summary.json`
- `outputs/path_feedback_batch_global_99_real_map_shadow_replay_v1/global-99-real-map-shadow-replay-summary.json`
- `outputs/path_feedback_batch_global_99_real_map_multi_roi_generalization_v1/global-99-real-map-multi-roi-generalization-summary.json`
- `outputs/path_feedback_batch_network_architecture_upgrade_readiness_review_v1/network-architecture-upgrade-readiness-summary.json`

## Output Artifacts

Output root: `outputs/path_feedback_batch_xunce_current_head_evidence_refresh_v1/`

- `xunce-current-head-evidence-refresh-summary.json`
- `xunce-current-head-evidence-refresh-manifest.json`
- `xunce-current-head-source-status-audit.json`
- `xunce-current-head-git-provenance-audit.json`
- `xunce-current-head-boundary-audit.json`
- `xunce-current-head-evidence-refresh-rejection-report.json`
- `xunce-current-head-evidence-refresh-report.md`

## Interface

Config path: `configs/xunce_current_head_evidence_refresh_v1.json`

Required fields:

- `schema_version="xunce-current-head-evidence-refresh-config/v1"`
- source roots for Stage 0, controlled installation, real-map shadow replay, real-map multi-ROI, and network readiness.
- `require_current_worktree_clean=true`
- `require_stage0_passed=true`
- `allow_legacy_missing_git_provenance=true`
- `require_closed_boundaries=true`

Passing `next_required_change`: `network_literature_bottleneck_review`

Failure `next_required_change`:

- `fix_xunce_design_freeze` when Stage 0 is missing, failed, or does not point to `current_head_evidence_refresh`.
- `fix_current_head_evidence_refresh` when current git is dirty, a source with provenance is dirty/mismatched, a source status fails, or source boundary fields are open.

## Tasks

### Task 1: Add Failing Tests

**Files:**
- Create: `tests/test_xunce_current_head_evidence_refresh.py`

- [ ] **Step 1:** Test current clean fixture passes with Stage 0 current git match and legacy missing provenance counted.
- [ ] **Step 2:** Test dirty/mismatched source provenance fails.
- [ ] **Step 3:** Test failed Stage 0 routes to `fix_xunce_design_freeze`.
- [ ] **Step 4:** Test source boundary violation fails.
- [ ] **Step 5:** Run the test and confirm it fails because the runner is missing.

### Task 2: Implement Runner and Entrypoints

**Files:**
- Create: `scripts/run_xunce_current_head_evidence_refresh.py`
- Create: `scripts/run_xunce_current_head_evidence_refresh.sh`
- Create: `configs/xunce_current_head_evidence_refresh_v1.json`

- [ ] **Step 1:** Implement config loading and source path resolution.
- [ ] **Step 2:** Load current `git_snapshot(repo_root)` and require clean parent/submodules.
- [ ] **Step 3:** Audit source statuses and expected `next_required_change`.
- [ ] **Step 4:** For sources with `git_provenance.current`, compare parent SHA and dirty state to current HEAD.
- [ ] **Step 5:** Count legacy missing git provenance separately when allowed.
- [ ] **Step 6:** Audit boundary fields.
- [ ] **Step 7:** Write summary, manifest, audits, rejection report, and markdown report.

### Task 3: Update Docs

**Files:**
- Modify: `README.md`
- Modify: `docs/算法设计与系统架构报告.md`
- Modify: `docs/superpowers/specs/2026-06-16-topology-aware-coverage-policy-network-design.md`
- Modify: `docs/superpowers/specs/2026-06-16-global-99-exploration-coverage-line.md`

- [ ] **Step 1:** Add Stage 1 runner/config/output root and pass next gate.
- [ ] **Step 2:** Explain legacy missing provenance is recorded, while dirty/mismatched provenance blocks progression.
- [ ] **Step 3:** Preserve non-goals and Global 99 separation.

### Task 4: Verify and Publish Stage 1

- [ ] **Step 1:** Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/test_xunce_current_head_evidence_refresh.py tests/test_xunce_design_freeze_audit.py -q`.
- [ ] **Step 2:** Run `PYTHON=$PY bash scripts/run_xunce_design_freeze_audit.sh`.
- [ ] **Step 3:** Run `PYTHON=$PY bash scripts/run_xunce_current_head_evidence_refresh.sh`.
- [ ] **Step 4:** Run the Goal regression command for `tests/test_global_99_*.py` and `tests/test_network_architecture_upgrade_readiness_review.py`.
- [ ] **Step 5:** Run `rg -n "巡策|xunce|current_head_evidence_refresh|network_literature_bottleneck_review" README.md docs scripts tests configs`.
- [ ] **Step 6:** Run `git diff --check`.
- [ ] **Step 7:** Commit `Add xunce current head evidence refresh` and push.

## Acceptance Gate

- Summary `status=passed`.
- `next_required_change=network_literature_bottleneck_review`.
- Current git is clean and matches the just-pushed Stage 0 commit.
- Stage 0 source is passed and points to `current_head_evidence_refresh`.
- Any source with provenance is clean and current; legacy missing provenance is counted explicitly.
- Boundary fields remain closed.
