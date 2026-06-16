# Xunce Stage 5 Topology-Aware Coverage Graph Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the research-only small 巡策 prototype `topology_aware_coverage_graph_proto_v1` and prove it can consume Stage 4 candidate graph plus memory-token features to produce masked logits and a value estimate.

**Architecture:** Stage 5 is a prototype-forward audit gate, not training or release. It consumes Stage 4 candidates/edges/memory artifacts, builds tensors, instantiates a small PyTorch candidate-graph network, runs deterministic forward passes, and writes audits for shape, mask, finite outputs, parameter count, latency, and closed release/training/executor boundaries. The prototype lives in root `scripts/` so it does not register as a production trainable/default policy architecture yet.

**Tech Stack:** Python 3, PyTorch, JSON/JSONL artifacts, existing Stage 4 outputs, pytest, shell runner, markdown docs.

---

## Scope

This stage implements only a small research prototype and forward-pass audit. It does not run PPO, train weights, publish checkpoint, replace default policy, connect executor, start online canary, modify action space/default A*, call path-planner, or claim performance. Passing Stage 5 authorizes Stage 6 mechanism validation only.

## Input Evidence

- `outputs/path_feedback_batch_xunce_topology_feature_extraction_audit_v1/xunce-topology-feature-extraction-audit-summary.json`
- `outputs/path_feedback_batch_xunce_topology_feature_extraction_audit_v1/xunce-topology-feature-extraction-candidates.jsonl`
- `outputs/path_feedback_batch_xunce_topology_feature_extraction_audit_v1/xunce-topology-feature-extraction-edges.jsonl`
- `outputs/path_feedback_batch_xunce_topology_feature_extraction_audit_v1/xunce-topology-feature-extraction-memory.json`

## Output Artifacts

Output root: `outputs/path_feedback_batch_xunce_topology_graph_proto_v1/`

- `xunce-topology-graph-proto-summary.json`
- `xunce-topology-graph-proto-manifest.json`
- `xunce-topology-graph-proto-forward-pass-audit.json`
- `xunce-topology-graph-proto-shape-audit.json`
- `xunce-topology-graph-proto-parameter-audit.json`
- `xunce-topology-graph-proto-logits.jsonl`
- `xunce-topology-graph-proto-boundary-audit.json`
- `xunce-topology-graph-proto-rejection-report.json`
- `xunce-topology-graph-proto-report.md`

## Interface

Config path: `configs/xunce_topology_graph_proto_v1.json`

Required fields:

- `schema_version="xunce-topology-graph-proto-config/v1"`
- `source_topology_feature_extraction_root`
- `architecture="topology_aware_coverage_graph_proto_v1"`
- `hidden_dim`
- `message_passing_layers`
- `dropout=0.0`
- `seed`
- `max_forward_passes`
- `max_parameter_count`
- `max_forward_latency_ms`

Passing `next_required_change`: `xunce_proto_mechanism_validation`

Failure `next_required_change`:

- `fix_xunce_topology_feature_extraction_audit` when Stage 4 evidence is missing, failed, or not pointing to this stage.
- `fix_xunce_topology_graph_proto` for shape/mask/finite/parameter/latency/boundary failures.

## Prototype Contract

- Candidate input: Stage 4 candidate feature vectors in stable candidate order.
- Edge input: Stage 4 pairwise edge feature vectors and candidate pair indices.
- Memory input: Stage 4 compact coverage-memory feature vector.
- Network:
  - candidate encoder: Linear -> GELU -> LayerNorm.
  - edge encoder: Linear -> GELU.
  - one or more lightweight message-passing layers using indexed edge messages.
  - memory encoder: Linear -> GELU.
  - policy head: candidate embedding + memory context -> scalar logit.
  - value head: masked candidate pool + memory context -> scalar value.
- Mask behavior:
  - valid candidates are all Stage 4 candidates unless config provides an override mask.
  - invalid candidates must receive masked logits at or below `-1e8`.
  - action probabilities over valid candidates must sum to 1.
- No checkpoint write is allowed.

## Tasks

### Task 1: Add Failing Tests

**Files:**
- Create: `tests/test_xunce_topology_graph_proto.py`

- [ ] Test default Stage 4 fixture produces passed summary and all artifacts.
- [ ] Test prototype output shapes: logits/masked logits/action probabilities `[1, candidate_count]`, value `[1]`.
- [ ] Test invalid action mask override masks logits and preserves probability sum over valid candidates.
- [ ] Test candidate graph and memory token are marked used and edge count is positive.
- [ ] Test missing/failed Stage 4 routes to `fix_xunce_topology_feature_extraction_audit`.
- [ ] Test boundary violation fails.
- [ ] Test no checkpoint/default-policy/executor/PPO/release boundary remains closed.

### Task 2: Implement Prototype Common Module and Runner

**Files:**
- Create: `scripts/xunce_topology_graph_proto_common.py`
- Create: `scripts/run_xunce_topology_graph_proto.py`
- Create: `scripts/run_xunce_topology_graph_proto.sh`
- Create: `configs/xunce_topology_graph_proto_v1.json`

- [ ] Load Stage 4 summary/candidates/edges/memory.
- [ ] Convert candidate, edge, memory features to tensors.
- [ ] Implement `TopologyAwareCoverageGraphPrototype`.
- [ ] Run deterministic forward passes under `torch.no_grad()`.
- [ ] Write summary, manifest, forward/shape/parameter/boundary audits, logits JSONL, rejection report, and markdown report.

### Task 3: Update Docs

**Files:**
- Modify: `README.md`
- Modify: `docs/算法设计与系统架构报告.md`
- Modify: `docs/superpowers/specs/2026-06-16-topology-aware-coverage-policy-network-design.md`
- Modify: `docs/superpowers/specs/2026-06-16-global-99-exploration-coverage-line.md`

- [ ] Add Stage 5 runner/config/output root.
- [ ] Explain Stage 5 is a prototype forward audit only.
- [ ] State that passing Stage 5 leads to Stage 6 mechanism validation, not training.

### Task 4: Verify and Publish

- [ ] Run Stage 5 tests and relevant Stage 0-5 tests.
- [ ] Run Stage 0-5 runners.
- [ ] Run Goal regression command.
- [ ] Run docs `rg` and `git diff --check`.
- [ ] Commit `Add xunce topology graph prototype` and push.

## Acceptance Gate

- Summary `status=passed`.
- `next_required_change=xunce_proto_mechanism_validation`.
- Candidate graph and memory token are actually consumed.
- Masked logits/value shapes are correct and finite.
- Parameter and latency audits are within configured thresholds.
- No checkpoint is written.
- Release/training/executor/default-policy/action-space/default-A* boundaries remain closed.
