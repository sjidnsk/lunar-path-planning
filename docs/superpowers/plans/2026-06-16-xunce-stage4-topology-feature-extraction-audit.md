# Xunce Stage 4 Topology Feature Extraction Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the Stage 3 additive topology observation contract can be populated deterministically from synthetic Global 99 coverage state without training, publishing, or changing policy behavior.

**Architecture:** Stage 4 is an extraction audit gate. It consumes the Stage 3 contract, a deterministic synthetic scenario fixture, and existing coverage/frontier helper functions. The runner expands geometry, builds coverage memory, enumerates frontier candidates, extracts candidate topology fields, edge fields, and memory-token fields, then writes audits proving field completeness, missing-indicator behavior, deterministic ordering, and closed release/training/executor boundaries.

**Tech Stack:** Python 3 standard library, existing `global_99_coverage_contract.py`, existing `frontier_coverage_planner_common.py`, JSON/JSONL artifacts, pytest, shell runner, markdown docs.

---

## Scope

This stage audits feature generation only. It does not implement a new network, train PPO, publish checkpoints, replace default policy, connect executor, start online canary, modify action space/default A*, call path-planner, or claim performance.

## Input Evidence

- `outputs/path_feedback_batch_xunce_topology_observation_contract_v1/xunce-topology-observation-contract-summary.json`
- `outputs/path_feedback_batch_xunce_topology_observation_contract_v1/xunce-topology-observation-contract.json`
- `configs/xunce_topology_observation_contract_v1.json`
- Existing helper modules:
  - `scripts/global_99_coverage_contract.py`
  - `scripts/frontier_coverage_planner_common.py`

## Output Artifacts

Output root: `outputs/path_feedback_batch_xunce_topology_feature_extraction_audit_v1/`

- `xunce-topology-feature-extraction-audit-summary.json`
- `xunce-topology-feature-extraction-audit-manifest.json`
- `xunce-topology-feature-extraction-candidates.jsonl`
- `xunce-topology-feature-extraction-edges.jsonl`
- `xunce-topology-feature-extraction-memory.json`
- `xunce-topology-feature-extraction-field-audit.json`
- `xunce-topology-feature-extraction-determinism-audit.json`
- `xunce-topology-feature-extraction-boundary-audit.json`
- `xunce-topology-feature-extraction-rejection-report.json`
- `xunce-topology-feature-extraction-audit-report.md`

## Interface

Config path: `configs/xunce_topology_feature_extraction_audit_v1.json`

Required fields:

- `schema_version="xunce-topology-feature-extraction-audit-config/v1"`
- `source_topology_observation_contract_root`
- `target_coverage_rate`
- `path_budget_m`
- `coverage_radius_cells`
- `revisit_penalty_weight`
- `new_coverage_weight`
- deterministic `scenario` with grid, start, ROI, blocked/unsafe rectangles, current covered cells, previous path cells, and recent history.

Passing `next_required_change`: `topology_aware_coverage_graph_proto`

Failure `next_required_change`:

- `fix_xunce_topology_observation_contract` when Stage 3 evidence is missing, failed, or not pointing to this stage.
- `fix_xunce_topology_feature_extraction_audit` for extraction, determinism, missing-indicator, or boundary failures.

## Extraction Rules

- Use `scenario_geometry()` to compute ROI, blocked/unsafe cells, reachable cells, and reachable-safe target denominator.
- Build coverage memory from `current_covered_cells`, `previous_path_cells`, and start-cell coverage footprint.
- Use `frontier_cells()` and `enumerate_frontier_candidates()` for stable frontier candidate ordering.
- Candidate fields:
  - `frontier_cluster_id`: candidate `cluster_index`.
  - `roi_group_id`: deterministic ROI group derived from grid quadrant.
  - `new_coverage_cell_count`: candidate new reachable-safe cells.
  - `coverage_overlap_count`: candidate event cells already in coverage memory.
  - `bfs_distance_from_current`: path length from current cell.
  - `path_bottleneck_score`: inverse local passable-neighbor count along path, max-normalized.
  - `revisit_path_cell_count`: path cells already in coverage memory.
  - `budget_fraction_cost`: path cost divided by `path_budget_m`.
  - `fallback_risk`: `1.0` when no new coverage or over budget, else `0.0`.
- Edge fields:
  - `same_frontier_cluster`, `same_roi_group`, `bfs_distance_between_candidates`, `coverage_overlap_ratio`, `shared_bottleneck`, `mutual_redundancy_score`.
- Memory fields:
  - `coverage_rate`, `remaining_budget_fraction`, `recent_path_cost_trend`, `recent_new_coverage_trend`, `revisit_rate`, `fallback_rate`, `roi_group_completion_ratio`.
- Missing indicators must be present for every Stage 3 contract field. Default fixture should have no missing candidate or memory fields.

## Tasks

### Task 1: Add Failing Tests

**Files:**
- Create: `tests/test_xunce_topology_feature_extraction_audit.py`

- [ ] Test default fixture passes, writes all artifacts, and routes to `topology_aware_coverage_graph_proto`.
- [ ] Test extracted candidate rows contain all Stage 3 candidate fields and missing indicators.
- [ ] Test edge rows contain pairwise topology fields and stable candidate ordering.
- [ ] Test memory token fields are finite and coverage/budget/revisit values are within valid ranges.
- [ ] Test missing Stage 3 summary routes to `fix_xunce_topology_observation_contract`.
- [ ] Test no frontier / invalid denominator routes to `fix_xunce_topology_feature_extraction_audit`.
- [ ] Test boundary violation fails.

### Task 2: Implement Runner and Entrypoints

**Files:**
- Create: `scripts/run_xunce_topology_feature_extraction_audit.py`
- Create: `scripts/run_xunce_topology_feature_extraction_audit.sh`
- Create: `configs/xunce_topology_feature_extraction_audit_v1.json`

- [ ] Load config and Stage 3 summary/contract.
- [ ] Validate fixture schema and geometry.
- [ ] Extract candidate rows, edge rows, and memory token.
- [ ] Verify field coverage, missing indicators, finite numeric values, deterministic candidate ranks, and boundary closure.
- [ ] Write summary, manifest, JSONL artifacts, audits, rejection report, and markdown report.

### Task 3: Update Docs

**Files:**
- Modify: `README.md`
- Modify: `docs/算法设计与系统架构报告.md`
- Modify: `docs/superpowers/specs/2026-06-16-topology-aware-coverage-policy-network-design.md`
- Modify: `docs/superpowers/specs/2026-06-16-global-99-exploration-coverage-line.md`

- [ ] Add Stage 4 runner/config/output root.
- [ ] Explain Stage 4 audits deterministic extraction only, not network/prototype/training.
- [ ] State that passing Stage 4 authorizes the small prototype stage only.

### Task 4: Verify and Publish

- [ ] Run Stage 4 tests.
- [ ] Run Stage 0-4 runners.
- [ ] Run Goal regression command.
- [ ] Run docs `rg` and `git diff --check`.
- [ ] Commit `Add xunce topology feature extraction audit` and push.

## Acceptance Gate

- Summary `status=passed`.
- `next_required_change=topology_aware_coverage_graph_proto`.
- Candidate, edge, and memory fields match Stage 3 contract.
- Missing indicators are explicit.
- Determinism audit passes.
- Release/training/executor/default-policy/network/action-space/default-A* boundaries remain closed.
