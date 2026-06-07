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

### Task 5: Coverage Diagnosis + Selection Improvement

**Files:**
- Modify: `model-explorer/src/model_explorer/policy/planning.py`
- Modify: `model-explorer/src/model_explorer/policy/path_feedback.py`
- Modify: `scripts/run_path_feedback_validation.sh`
- Modify: `configs/path_feedback_batch_anchor_projection_candidate_generation_v1.json`
- Modify: `scripts/run_anchor_projection_candidate_generation.py`
- Modify: `tests/test_anchor_projection_candidate_generation.py`
- Modify: `tests/test_path_feedback_validation_script.py`
- Modify: `model-explorer/tests/test_model_explorer.py`
- Modify: `docs/算法设计与系统架构报告.md`
- Modify: `docs/superpowers/specs/2026-06-07-anchor-projection-candidate-generation.md`

- [x] **Step 1: Write failing tests for diagnosis and opt-in selection bonus**

Added tests proving the summary exposes `anchor_projection_coverage_diagnosis/v1`, and that
`source_selection_path_cost_bonus` is manifest-only and opt-in.

- [x] **Step 2: Implement bounded source-selection bonus**

Added `AnchorProjectionCandidateConfig.source_selection_path_cost_bonus`. The bonus only affects
path-feedback source selection for feasible, non-replan `projected_execution_target` candidates.
It does not change default A*, PPO, network architecture, action space, or path-planner CLI args.

- [x] **Step 3: Implement coverage diagnosis summary**

Extended `anchor-projection-candidate-generation-summary/v1` with generated/source-selected counts,
nontrainable primary reasons, projection distance distribution, scenario/run breakdowns, and
path-cost/risk margin diagnostics.

- [x] **Step 4: Refresh evidence**

Regenerated `outputs/path_feedback_batch_anchor_projection_candidate_generation_v1/` with
`--anchor-projection-selection-path-cost-bonus 6.0`. Result:
`trainable_anchor_projection_count=18`, `nontrainable_blocked_target_count=60`,
`source_selected_candidate_changed_rate=0.23076923076923078`.

- [x] **Step 5: Validate readiness remains blocked**

Full downstream generation and validate-only chain passed. Policy training readiness remains
`needs_training_contract_refinement`; PPO is still not allowed.

- [x] **Step 6: Risk Closure v1**

Added unified dirty-aware git provenance snapshots for parent/submodule evidence gates, strict
corner-cutting semantics for A*/channel-aware/region-guided planners, source-selected
anchor-projection quality regression gates, channel-aware exposure-only regression rejection, and
optional anchor-projection candidate/contract inputs in policy readiness review.

Validation highlights:

- Existing related tests pass for provenance, readiness inputs, anchor-projection candidate
  generation, model-explorer path feedback, A*, channel-aware A*, and region-guided search.
- Isolated dirty-worktree batch
  `outputs/path_feedback_batch_anchor_projection_candidate_generation_v1_risk_closure_check/`
  completed 8/8 runs, then candidate-generation validation failed as expected with
  `current_git_provenance_mismatch_count=1` and `git_provenance_mismatch_count=1`.
- The same isolated summary preserved the algorithmic count
  `trainable_anchor_projection_count=18`, `nontrainable_blocked_target_count=60`, and
  `source_selection_quality_regression_count=0`; the failure is provenance, not candidate
  generation regression.

Next step: after committing these changes, rerun the same batch and downstream validate-only chain
from a clean worktree, then reconcile the 18/78 candidate-generation count with the 3/42
anchor-projection contract count before any training dry-run.

- [x] **Step 7: Risk Closure v2**

Closed the remaining risk items found in the follow-up scan:

- Shared provenance inspection now lives in `scripts/git_provenance.py`; downstream source
  summaries missing `git_provenance.current` fail with explicit missing-current reason codes.
- Legacy SHA-only snapshot compatibility remains only for present snapshots; a missing current
  snapshot is no longer treated as a weak match.
- Policy readiness now separates blocker margins from diagnostics: only source-selected quality
  regressions or trainable anchor-projection contexts can trip path/risk regression blockers.
- Anchor-projection best-alternative scope is explicit:
  `reachable_non_replan_candidates_including_policy_and_projected_targets`.
- The direction-cone CLI scenario batch was re-baselined from the stale
  `sampled_trajectory_collision` case to `direction_cone_obstacle_detour`; the strict
  corner-cutting path no longer collides with the center obstacle, and the real blocker is
  `direction_cone_constraint_violation`.

Validation completed:

- parent `tests`: `110 passed, 10 subtests passed`
- `model-explorer/tests`: `191 passed, 13 subtests passed`
- `dev-platform-constraints/tests`: `86 passed, 1 skipped, 36 subtests passed`
- `path-planner/tests` from submodule cwd with `PYTHONPATH=src`: pytest exited 0
- anchor-projection batch dry-run: 8 planned runs
- dirty-worktree formal batch:
  `outputs/path_feedback_batch_anchor_projection_candidate_generation_v1_risk_closure_v2_check/`
  completed 8/8 runs
- dirty-worktree candidate-generation validate-only failed as intended with
  `current_git_provenance_mismatch_count=1`, `git_provenance_mismatch_count=1`, and
  `trainable_anchor_projection_count=18`
- clean-worktree formal batch:
  `outputs/path_feedback_batch_anchor_projection_candidate_generation_v1_risk_closure_v2_final_clean_check/`
  completed 8/8 runs
- clean-worktree candidate-generation validate-only and summary generation passed with
  `current_git_provenance_mismatch_count=0`, `git_provenance_mismatch_count=0`,
  `trainable_anchor_projection_count=18`, and `source_selection_quality_regression_count=0`

The remaining blocker is no longer provenance; it is reconciling candidate-generation evidence with
the anchor-projection evidence contract before any training dry-run.

- [x] **Step 8: Anchor-Projection Readiness Contract Integration v1**

Aligned candidate-generation, evidence-contract, and policy-readiness counting around the same
projected execution target contract.

- Candidate-generation summaries now expose top-level nontrainable cause counts:
  `nontrainable_anchor_unreachable_count`,
  `nontrainable_source_candidate_not_selected_count`, and `audit_proxy_positive_count`.
- Evidence-contract can use `anchor-projection-candidate-generation-summary/v1` as the contract
  source when old goal-blocked/readiness audit summaries are absent, while preserving the old
  audit-only mode for legacy roots.
- Policy readiness now supports an anchor-only review mode and keeps the old channel-aware/audit
  counts separate from candidate-generation counts.

Validation:

- targeted anchor-projection/readiness/provenance tests:
  `27 passed, 5 subtests passed`
- parent `tests`: `113 passed, 10 subtests passed`
- `model-explorer/tests`: `191 passed, 13 subtests passed`
- integration root:
  `outputs/path_feedback_batch_anchor_projection_contract_integration_v1/`
  completed 8/8 batch runs; candidate-generation and evidence-contract both report
  `trainable_anchor_projection_count=18`, `nontrainable_blocked_target_count=60`,
  and `candidate_contract_alignment_gap_count=0`.

Readiness remains blocked by `anchor_projection_nontrainable_contexts_remain`; the next algorithm
work is to split the 36 `anchor_unreachable` contexts from the 24
`source_candidate_not_selected` contexts and address their separate causes.
