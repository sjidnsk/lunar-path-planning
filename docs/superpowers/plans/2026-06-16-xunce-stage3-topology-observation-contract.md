# Xunce Stage 3 Topology Observation Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Define and audit an additive topology observation contract for 巡策 without changing the existing `policy-observation/v1.1` extractor, scorer, checkpoint loader, action mask, or default policy behavior.

**Architecture:** Stage 3 is a contract gate. It consumes Stage 2 evidence and a proposed JSON contract for candidate topology fields, candidate edge fields, and coverage memory fields. The runner verifies the contract is additive, optional, missing-indicator backed, and compatible with current `PolicyObservation`, `TorchPolicyScorer`, checkpoint metadata, candidate mask, and old architectures.

**Tech Stack:** Python 3 standard library, JSON contract/config, pytest, existing policy source inspection, markdown docs.

---

## Scope

This stage only designs and audits the contract. It does not implement feature extraction, add a network, train PPO, publish checkpoints, replace default policy, connect executor, start canary, modify action space/default A*, or claim performance.

## Input Evidence

- `outputs/path_feedback_batch_xunce_network_literature_bottleneck_review_v1/xunce-network-literature-bottleneck-review-summary.json`
- `model-explorer/src/model_explorer/policy/features.py`
- `model-explorer/src/model_explorer/policy/torch_policy.py`
- `model-explorer/src/model_explorer/policy/training.py`
- `model-explorer/src/model_explorer/policy/architectures.py`

## Output Artifacts

Output root: `outputs/path_feedback_batch_xunce_topology_observation_contract_v1/`

- `xunce-topology-observation-contract-summary.json`
- `xunce-topology-observation-contract-manifest.json`
- `xunce-topology-observation-contract.json`
- `xunce-topology-observation-contract-audit.json`
- `xunce-topology-observation-compatibility-audit.json`
- `xunce-topology-observation-boundary-audit.json`
- `xunce-topology-observation-contract-rejection-report.json`
- `xunce-topology-observation-contract-report.md`

## Interface

Config path: `configs/xunce_topology_observation_contract_v1.json`

Required fields:

- `schema_version="xunce-topology-observation-contract-config/v1"`
- source Stage 2 root.
- policy source paths.
- `base_observation_schema_version="policy-observation/v1.1"`
- `extension_schema_version="xunce-topology-observation-extension/v1"`
- `contract.additive_only=true`
- `contract.preserve_action_mask=true`
- `contract.preserve_candidate_order=true`
- `contract.optional_by_default=true`
- `contract.require_missing_indicators=true`
- `candidate_topology_fields`
- `candidate_edge_fields`
- `coverage_memory_fields`

Passing `next_required_change`: `topology_feature_extraction_audit`

Failure `next_required_change`:

- `fix_xunce_network_literature_bottleneck_review` when Stage 2 is missing, failed, or not pointing to this stage.
- `fix_xunce_topology_observation_contract` for invalid contract or compatibility violations.

## Contract Rules

- The base schema remains `policy-observation/v1.1`.
- New fields are optional by default and must have `missing_indicator`.
- No field can remove/rename `candidate_features`, `global_features`, `action_mask`, `candidate_cells`, `candidate_missing_indicator_names`, or `candidate_missing_indicators`.
- Candidate topology fields describe per-candidate values such as `frontier_cluster_id`, `roi_group_id`, `new_coverage_cell_count`, `coverage_overlap_count`, `bfs_distance_from_current`, `path_bottleneck_score`, `revisit_path_cell_count`, `budget_fraction_cost`, and `fallback_risk`.
- Edge fields describe candidate-candidate relations such as `same_frontier_cluster`, `same_roi_group`, `bfs_distance_between_candidates`, `coverage_overlap_ratio`, `shared_bottleneck`, and `mutual_redundancy_score`.
- Memory fields describe compact global state such as `coverage_rate`, `remaining_budget_fraction`, `recent_path_cost_trend`, `recent_new_coverage_trend`, `revisit_rate`, `fallback_rate`, and `roi_group_completion_ratio`.
- The contract can be consumed by future Stage 4 extraction audit but must not require current legacy scorers/checkpoints to consume the new fields.

## Tasks

### Task 1: Add Failing Tests

**Files:**
- Create: `tests/test_xunce_topology_observation_contract.py`

- [ ] Test the default contract passes and writes all artifacts.
- [ ] Test Stage 2 failure routes to `fix_xunce_network_literature_bottleneck_review`.
- [ ] Test a non-additive contract fails.
- [ ] Test required new fields without missing indicators fail.
- [ ] Test base schema mutation fails.
- [ ] Test boundary violation fails.

### Task 2: Implement Runner and Entrypoints

**Files:**
- Create: `scripts/run_xunce_topology_observation_contract.py`
- Create: `scripts/run_xunce_topology_observation_contract.sh`
- Create: `configs/xunce_topology_observation_contract_v1.json`

- [ ] Load config and source Stage 2 summary.
- [ ] Validate field definitions: unique names, numeric/id/bool type families, optional/default, missing indicators.
- [ ] Inspect policy source files for `policy-observation/v1.1`, `action_mask`, missing indicators, `TorchPolicyScorer`, and architecture inventory.
- [ ] Verify scope boundaries remain closed.
- [ ] Write contract, summary, manifest, audits, rejection report, and markdown report.

### Task 3: Update Docs

**Files:**
- Modify: `README.md`
- Modify: `docs/算法设计与系统架构报告.md`
- Modify: `docs/superpowers/specs/2026-06-16-topology-aware-coverage-policy-network-design.md`
- Modify: `docs/superpowers/specs/2026-06-16-global-99-exploration-coverage-line.md`

- [ ] Add Stage 3 runner/config/output root.
- [ ] Explain Stage 3 is additive contract design only, not extraction, model, or training.
- [ ] List the required topology/edge/memory field groups.

### Task 4: Verify and Publish

- [ ] Run Stage 3 tests.
- [ ] Run Stage 0, 1, 2, and 3 runners.
- [ ] Run Goal regression command.
- [ ] Run docs `rg` and `git diff --check`.
- [ ] Commit `Add xunce topology observation contract` and push.

## Acceptance Gate

- Summary `status=passed`.
- `next_required_change=topology_feature_extraction_audit`.
- Contract is additive, optional, and missing-indicator backed.
- Existing observation/scorer/checkpoint/action-mask boundaries are preserved.
- Release/training/executor/default-policy boundaries remain closed.
