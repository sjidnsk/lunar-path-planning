# Xunce Stage 6 Prototype Mechanism Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate that the Stage 5 `topology_aware_coverage_graph_proto_v1` prototype actually uses topology edges and memory-token signals to change candidate ranking while preserving masks, guarded fallback boundaries, and closed release/training scope.

**Architecture:** Stage 6 is a mechanism validation gate. It consumes the Stage 5 prototype summary and Stage 4 feature artifacts, rebuilds the same research-only prototype with a deterministic mechanism-validation seed, and compares full-signal logits/ranking against edge-ablated and memory-ablated forward passes. It writes audits proving signal deltas, rank changes, mask stability, fallback safety, and boundary closure.

**Tech Stack:** Python 3, PyTorch, Stage 4 feature artifacts, Stage 5 prototype common module, JSON/JSONL artifacts, pytest, shell runner, markdown docs.

---

## Scope

This stage only validates prototype mechanism behavior. It does not train PPO, update weights, publish or write checkpoints, replace default policy, connect executor, start online canary, modify action space/default A*, call path-planner, or claim performance.

## Input Evidence

- `outputs/path_feedback_batch_xunce_topology_graph_proto_v1/xunce-topology-graph-proto-summary.json`
- `outputs/path_feedback_batch_xunce_topology_feature_extraction_audit_v1/xunce-topology-feature-extraction-candidates.jsonl`
- `outputs/path_feedback_batch_xunce_topology_feature_extraction_audit_v1/xunce-topology-feature-extraction-edges.jsonl`
- `outputs/path_feedback_batch_xunce_topology_feature_extraction_audit_v1/xunce-topology-feature-extraction-memory.json`

## Output Artifacts

Output root: `outputs/path_feedback_batch_xunce_proto_mechanism_validation_v1/`

- `xunce-proto-mechanism-validation-summary.json`
- `xunce-proto-mechanism-validation-manifest.json`
- `xunce-proto-mechanism-validation-signal-audit.json`
- `xunce-proto-mechanism-validation-mask-audit.json`
- `xunce-proto-mechanism-validation-fallback-audit.json`
- `xunce-proto-mechanism-validation-logit-comparison.jsonl`
- `xunce-proto-mechanism-validation-boundary-audit.json`
- `xunce-proto-mechanism-validation-rejection-report.json`
- `xunce-proto-mechanism-validation-report.md`

## Interface

Config path: `configs/xunce_proto_mechanism_validation_v1.json`

Required fields:

- `schema_version="xunce-proto-mechanism-validation-config/v1"`
- `source_topology_graph_proto_root`
- `source_topology_feature_extraction_root`
- `architecture="topology_aware_coverage_graph_proto_v1"`
- `hidden_dim`
- `message_passing_layers`
- `dropout=0.0`
- `mechanism_seed`
- `min_logit_delta`
- `require_edge_rank_change=true`
- `require_memory_rank_change=true`
- `action_mask_override`

Passing `next_required_change`: `architecture_contrast_evaluation`

Failure `next_required_change`:

- `fix_xunce_topology_graph_proto` when Stage 5 evidence is missing, failed, or not pointing to this stage.
- `fix_xunce_proto_mechanism_validation` for signal, mask, fallback, finite-output, or boundary failures.

## Mechanism Rules

- Build three deterministic forward passes:
  - `full`: original candidates, edges, and memory token.
  - `edge_ablated`: same candidates and memory token, but edge feature tensor is zeroed.
  - `memory_ablated`: same candidates and edges, but memory feature tensor is zeroed.
- Signal audit must report:
  - max absolute edge logit delta.
  - max absolute memory logit delta.
  - full, edge-ablated, and memory-ablated ranking.
  - whether edge ablation changes rank.
  - whether memory ablation changes rank.
- Mask audit must prove invalid candidates stay masked at or below `-1e8` and valid action probabilities sum to 1.
- Fallback audit must prove no fallback candidate is forced and no non-finite logits/value appear.
- This is mechanism validation, not a claim that the prototype improves coverage.

## Tasks

### Task 1: Add Failing Tests

**Files:**
- Create: `tests/test_xunce_proto_mechanism_validation.py`

- [ ] Test default fixture passes and writes all artifacts.
- [ ] Test edge and memory ablations both exceed `min_logit_delta` and change ranking.
- [ ] Test invalid action mask override remains masked and probability mass stays on valid candidates.
- [ ] Test missing/failed Stage 5 routes to `fix_xunce_topology_graph_proto`.
- [ ] Test insufficient signal delta routes to `fix_xunce_proto_mechanism_validation`.
- [ ] Test boundary violation fails.
- [ ] Test checkpoint/default-policy/executor/PPO/release boundaries remain closed.

### Task 2: Implement Runner and Entrypoint

**Files:**
- Create: `scripts/run_xunce_proto_mechanism_validation.py`
- Create: `scripts/run_xunce_proto_mechanism_validation.sh`
- Create: `configs/xunce_proto_mechanism_validation_v1.json`

- [ ] Load Stage 5 summary and Stage 4 candidate/edge/memory artifacts.
- [ ] Rebuild `TopologyAwareCoverageGraphPrototype` with config seed.
- [ ] Run full, edge-ablated, and memory-ablated forward passes.
- [ ] Write signal, mask, fallback, boundary audits, logit comparison JSONL, rejection report, manifest, summary, and report.

### Task 3: Update Docs

**Files:**
- Modify: `README.md`
- Modify: `docs/算法设计与系统架构报告.md`
- Modify: `docs/superpowers/specs/2026-06-16-topology-aware-coverage-policy-network-design.md`
- Modify: `docs/superpowers/specs/2026-06-16-global-99-exploration-coverage-line.md`

- [ ] Add Stage 6 runner/config/output root.
- [ ] Explain Stage 6 validates mechanism only, not performance.
- [ ] State that passing Stage 6 leads to Stage 7 architecture contrast evaluation.

### Task 4: Verify and Publish

- [ ] Run Stage 6 tests and Stage 0-6 tests.
- [ ] Run Stage 0-6 runners.
- [ ] Run Goal regression command.
- [ ] Run docs `rg` and `git diff --check`.
- [ ] Commit `Add xunce proto mechanism validation` and push.

## Acceptance Gate

- Summary `status=passed`.
- `next_required_change=architecture_contrast_evaluation`.
- Edge and memory ablations both change logits and ranking.
- Mask and fallback audits pass.
- No checkpoint is written.
- Release/training/executor/default-policy/action-space/default-A* boundaries remain closed.
