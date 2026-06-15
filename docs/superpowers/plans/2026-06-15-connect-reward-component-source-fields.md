# Connect Reward Component Source Fields Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire real provenance-backed source fields for coverage-aware reward components that Stage 4 currently rejects.

**Architecture:** Add a read-only connector stage that joins Stage 2 coverage delta rows, selected-candidate shadow steps, and path-feedback candidate records. The connector writes a provenance overlay; Stage 4 consumes that overlay when its input roots match, without mutating Stage 2 artifacts.

**Tech Stack:** Python stdlib JSON/JSONL scripts, bash runner, unittest/pytest, existing `git_provenance` helper.

---

### Task 1: Red Tests For Source Wiring

**Files:**
- Create: `tests/test_connect_reward_component_source_fields.py`

- [ ] Write tests that build minimal Stage 2/3/4 fixture roots plus path-feedback summaries.
- [ ] Assert the connector writes summary, source-field wiring audit, component provenance JSONL, replay validation, rejection report, and markdown report.
- [ ] Assert `actual_valuable_area_covered`, `actual_information_gain`, and `risk` are positive and provenance-backed when path-feedback candidates match.
- [ ] Assert missing source, expected-field-only, fallback contamination, and controlled regression cases fail with explicit reason codes.
- [ ] Run the new test file and confirm it fails because the connector script does not exist yet.

### Task 2: Connector Stage

**Files:**
- Create: `scripts/run_connect_reward_component_source_fields.py`
- Create: `scripts/run_connect_reward_component_source_fields.sh`

- [ ] Implement JSON/JSONL readers, path resolution, and CLI args for coverage signal, coverage performance, reward refinement, optional path-feedback summaries, and output root.
- [ ] Build a shadow-step index using context and episode-step keys.
- [ ] Build a path-feedback candidate index by context and by `scenario_id + action_index`.
- [ ] For each delta row, resolve the controlled action from shadow, match the candidate, and write provenance fields.
- [ ] Derive `actual_valuable_area_covered = max(coverage_rate_delta, 0) * utility`.
- [ ] Derive `actual_information_gain = max(coverage_rate_delta, 0)` from the actual path-feedback coverage signal.
- [ ] Use path-feedback candidate `risk` as trusted risk source.
- [ ] Reject rows with expected-only values, fallback policy-gain contamination, controlled regression, or missing candidate provenance.
- [ ] Write summary, `source-field-wiring-audit.json`, `component-provenance.jsonl`, `replay-validation.json`, `connect-reward-component-source-fields-rejection-report.json`, and report markdown.

### Task 3: Stage 4 Overlay Consumption

**Files:**
- Modify: `scripts/run_coverage_aware_reward_refinement.py`
- Modify: `tests/test_coverage_aware_reward_refinement.py`

- [ ] Add default overlay discovery at `outputs/path_feedback_batch_connect_reward_component_source_fields_v1/`.
- [ ] Only consume overlay when the connector summary roots match the current Stage 4 input roots.
- [ ] Merge provenance fields by context and episode-step keys before computing reward components.
- [ ] Preserve all read-only boundary flags and existing rejection logic.
- [ ] Add a Stage 4 test proving missing direct fields pass after a matching connector overlay.

### Task 4: Real Run And Docs

**Files:**
- Modify: `README.md`
- Modify: `docs/算法设计与系统架构报告.md`
- Create: `docs/superpowers/specs/2026-06-15-connect-reward-component-source-fields.md`

- [ ] Run the new tests and existing reward refinement tests.
- [ ] Run connector on current artifacts into `outputs/path_feedback_batch_connect_reward_component_source_fields_v1/`.
- [ ] Rerun Stage 4 and verify the three missing source reason codes are cleared.
- [ ] Run `git diff --check`.
- [ ] Update docs with evidence, outputs, remaining non-goals, and the next gate.
