# Xunce Stage 2 Network Literature Bottleneck Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an auditable literature and project-bottleneck review for 巡策 before any topology observation schema or network prototype is implemented.

**Architecture:** Stage 2 consumes Stage 1 Current-HEAD evidence, existing Global 99 / real-map / network-readiness summaries, and a curated primary-source literature map. It decides whether current evidence shows coverage, generalization, fallback, latency, parameter-count, or no immediate network bottleneck, then routes to the next stage without training or changing network code.

**Tech Stack:** Python 3 standard library, JSON configs/artifacts, pytest, existing repository evidence summaries, markdown documentation.

---

## Scope

This stage is an audit and recommendation gate. It does not add topology fields, implement a model, train PPO, publish checkpoints, replace default policy, connect executor, start canary, modify action space/default A*, or claim real-world performance.

## Input Evidence

- `outputs/path_feedback_batch_xunce_current_head_evidence_refresh_v1/xunce-current-head-evidence-refresh-summary.json`
- `outputs/path_feedback_batch_network_architecture_upgrade_readiness_review_v1/network-architecture-upgrade-readiness-summary.json`
- `outputs/path_feedback_batch_global_99_multi_map_generalization_v1/global-99-multi-map-generalization-summary.json`
- `outputs/path_feedback_batch_global_99_real_map_multi_roi_generalization_v1/global-99-real-map-multi-roi-generalization-summary.json`
- `model-explorer/src/model_explorer/policy/architectures.py`
- `model-explorer/src/model_explorer/policy/features.py`

## Literature Map

Primary sources in the default config:

- Deep Sets, permutation-invariant set functions: https://arxiv.org/abs/1703.06114
- Set Transformer, attention over sets: https://arxiv.org/abs/1810.00825
- Attention for routing heuristics: https://arxiv.org/abs/1803.08475
- Graph Attention Networks: https://arxiv.org/abs/1710.10903
- Decision Transformer, trajectory sequence conditioning: https://arxiv.org/abs/2106.01345
- Mamba, efficient long-context sequence modeling: https://arxiv.org/abs/2312.00752
- Active Neural SLAM, modular learned exploration around planning: https://arxiv.org/abs/2004.05155
- Neural A*, learned path-planning heuristics for diagnostic inspiration: https://arxiv.org/abs/2009.07476

## Output Artifacts

Output root: `outputs/path_feedback_batch_xunce_network_literature_bottleneck_review_v1/`

- `xunce-network-literature-bottleneck-review-summary.json`
- `xunce-network-literature-bottleneck-review-manifest.json`
- `xunce-network-literature-map-audit.json`
- `xunce-project-bottleneck-audit.json`
- `xunce-network-research-scope-audit.json`
- `xunce-network-literature-bottleneck-rejection-report.json`
- `xunce-network-literature-bottleneck-review-report.md`

## Interface

Config path: `configs/xunce_network_literature_bottleneck_review_v1.json`

Required fields:

- `schema_version="xunce-network-literature-bottleneck-review-config/v1"`
- source roots for Stage 1, network readiness, multi-map, real-map multi-ROI.
- `literature_references`: list with id, title, url, family, project_relevance.
- `min_literature_reference_count=6`
- `require_current_head_refresh_passed=true`
- `require_closed_boundaries=true`

Passing `next_required_change`: `topology_observation_contract`

Failure `next_required_change`:

- `fix_xunce_current_head_evidence_refresh` when Stage 1 is missing, failed, dirty, or not pointing here.
- `fix_xunce_network_literature_bottleneck_review` when literature map, source evidence, or scope/boundary audit is invalid.

## Bottleneck Rules

- If network readiness fails or Stage 1 fails, this stage fails.
- If multi-map or real-map required scenarios fail, record `coverage_or_generalization_bottleneck=true`.
- If `policy_worse_than_baseline_count > 0`, `controlled_regression_count > 0`, or fallback rate exceeds `0.05`, record `fallback_or_policy_regression_bottleneck=true`.
- If architecture inventory lacks topology/graph/memory support, record `topology_representation_gap=true`.
- If coverage and generalization pass, no regressions exist, and readiness says `network_upgrade_recommended=false`, record `no_current_release_blocking_network_bottleneck=true`.
- Passing still routes to `topology_observation_contract`, because Stage 3 is additive contract design for research contrast, not a claim that network upgrade is required.

## Tasks

### Task 1: Add Failing Tests

**Files:**
- Create: `tests/test_xunce_network_literature_bottleneck_review.py`

- [ ] Test current clean evidence passes, records no current release-blocking network bottleneck, and routes to `topology_observation_contract`.
- [ ] Test insufficient literature references fails.
- [ ] Test failed Stage 1 routes to `fix_xunce_current_head_evidence_refresh`.
- [ ] Test policy regression or high fallback is attributed as a bottleneck without opening release boundaries.
- [ ] Test boundary violation fails.

### Task 2: Implement Runner and Entrypoints

**Files:**
- Create: `scripts/run_xunce_network_literature_bottleneck_review.py`
- Create: `scripts/run_xunce_network_literature_bottleneck_review.sh`
- Create: `configs/xunce_network_literature_bottleneck_review_v1.json`

- [ ] Load config and validate literature map.
- [ ] Load source summaries and architecture source files.
- [ ] Audit Stage 1 status and `next_required_change=network_literature_bottleneck_review`.
- [ ] Audit project bottlenecks from coverage, generalization, fallback, policy regression, architecture inventory, latency/params availability.
- [ ] Audit non-goal boundaries.
- [ ] Write all artifacts.

### Task 3: Update Docs

**Files:**
- Modify: `README.md`
- Modify: `docs/算法设计与系统架构报告.md`
- Modify: `docs/superpowers/specs/2026-06-16-topology-aware-coverage-policy-network-design.md`
- Modify: `docs/superpowers/specs/2026-06-16-global-99-exploration-coverage-line.md`

- [ ] Add Stage 2 runner/config/output root.
- [ ] Explain current evidence likely means no release-blocking network bottleneck.
- [ ] Explain Stage 3 remains additive observation-contract research, not training.

### Task 4: Verify and Publish

- [ ] Run Stage 2 tests.
- [ ] Run Stage 0, Stage 1, and Stage 2 runners.
- [ ] Run the Goal regression command.
- [ ] Run docs `rg` and `git diff --check`.
- [ ] Commit `Add xunce network literature bottleneck review` and push.

## Acceptance Gate

- Summary `status=passed`.
- `next_required_change=topology_observation_contract`.
- Literature reference count meets threshold and sources are primary URLs.
- Bottleneck attribution is explicit.
- Release/training/executor/default-policy boundaries remain closed.
- Docs describe Stage 2 and the no-direct-training boundary.
