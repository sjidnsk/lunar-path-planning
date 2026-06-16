# Xunce Stage 13 Controlled Training Candidate v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first controlled research training candidate for `xunce_full_network_v1` after the Stage 12 guarded authorization preflight.

**Architecture:** This stage reads the Stage 12 preflight summary, runs a deterministic bounded supervised surrogate update against `XunceFullNetworkV1`, records loss/gradient/checkpoint metadata, and writes a research-only checkpoint artifact. It is not a PPO update and it does not publish or install a checkpoint.

**Tech Stack:** Python, PyTorch, JSON artifacts, existing `scripts/xunce_full_network_common.py`, unittest/pytest, shell runner.

---

## Files

- Create: `configs/xunce_controlled_training_candidate_v1.json`
- Create: `scripts/run_xunce_controlled_training_candidate.py`
- Create: `scripts/run_xunce_controlled_training_candidate.sh`
- Create: `tests/test_xunce_controlled_training_candidate.py`
- Modify: `README.md`
- Modify: `docs/算法设计与系统架构报告.md`
- Modify: `docs/superpowers/specs/2026-06-16-global-99-exploration-coverage-line.md`
- Modify: `docs/superpowers/specs/2026-06-16-topology-aware-coverage-policy-network-design.md`

Output root:

`outputs/path_feedback_batch_xunce_controlled_training_candidate_v1/`

Artifacts:

- `xunce-controlled-training-candidate-summary.json`
- `xunce-controlled-training-candidate-manifest.json`
- `xunce-controlled-training-loss-audit.json`
- `xunce-controlled-training-gradient-audit.json`
- `xunce-controlled-training-checkpoint-metadata.json`
- `xunce-controlled-training-boundary-audit.json`
- `xunce-controlled-training-rejection-report.json`
- `xunce-controlled-training-candidate-report.md`
- `xunce-controlled-training-candidate.pt`

## Decision Contract

The runner must require Stage 12:

- `status=passed`
- `next_required_change=controlled_training_candidate`
- `controlled_training_candidate_authorized=true`
- boundary fields closed.

Passing summary:

- `status=passed`
- `controlled_training_candidate_passed=true`
- `runs_controlled_training_update=true`
- `runs_new_ppo_update=false`
- `ppo_update_executed=false`
- `writes_research_checkpoint=true`
- `publishes_checkpoint=false`
- `next_required_change=post_training_offline_evaluation`

Failure routing:

- Stage 12 missing/failed/wrong next: `fix_guarded_training_candidate_preflight`
- training loss does not improve enough: `fix_controlled_training_candidate`
- non-finite loss/gradient: `fix_controlled_training_candidate`
- boundary violation: `resolve_xunce_controlled_training_boundary_rejections`

## Task 1: Write Failing Tests

- [ ] Add `tests/test_xunce_controlled_training_candidate.py`.
- [ ] Cover default pass, missing Stage 12, boundary violation, insufficient loss improvement, and artifact writes.
- [ ] Run:

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/test_xunce_controlled_training_candidate.py -q
```

Expected before implementation: import/module failure for `run_xunce_controlled_training_candidate`.

## Task 2: Implement Runner and Config

- [ ] Add default config with deterministic seed, tiny bounded training step count, learning rate, and loss improvement threshold.
- [ ] Implement config validation and Stage 12 source audit.
- [ ] Instantiate `XunceFullNetworkV1` with contract dimensions from config.
- [ ] Run deterministic supervised surrogate update:
  - fixed synthetic candidate/context/memory/edge tensors;
  - valid action mask;
  - cross-entropy target over masked logits;
  - value MSE term;
  - finite gradient and loss checks.
- [ ] Write research checkpoint and metadata only under the output root.
- [ ] Explicitly keep release/default-policy/executor/PPO boundary fields closed.

## Task 3: Implement Shell Entrypoint

- [ ] Add `scripts/run_xunce_controlled_training_candidate.sh`.
- [ ] Support:

```bash
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python bash scripts/run_xunce_controlled_training_candidate.sh
```

## Task 4: Update Documentation

- [ ] Update README with Stage 13 command, output root, and next gate.
- [ ] Update architecture report with research-only checkpoint boundary.
- [ ] Update Global 99 spec and 巡策 design spec with `post_training_offline_evaluation`.

## Task 5: Verification

Run:

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/test_xunce_controlled_training_candidate.py -q
PYTHON=$PY bash scripts/run_xunce_controlled_training_candidate.sh
jq '{status,reason_codes,controlled_training_candidate_passed,loss_decreased,writes_research_checkpoint,publishes_checkpoint,runs_new_ppo_update,ppo_update_executed,next_required_change}' \
  outputs/path_feedback_batch_xunce_controlled_training_candidate_v1/xunce-controlled-training-candidate-summary.json
rg -n "Xunce Controlled Training Candidate v1|run_xunce_controlled_training_candidate|post_training_offline_evaluation" \
  README.md docs/算法设计与系统架构报告.md docs/superpowers/specs
git diff --check
```

Full regression before commit:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest \
  tests/test_xunce_design_freeze_audit.py \
  tests/test_xunce_current_head_evidence_refresh.py \
  tests/test_xunce_network_literature_bottleneck_review.py \
  tests/test_xunce_topology_observation_contract.py \
  tests/test_xunce_topology_feature_extraction_audit.py \
  tests/test_xunce_topology_graph_proto.py \
  tests/test_xunce_proto_mechanism_validation.py \
  tests/test_xunce_architecture_contrast_evaluation.py \
  tests/test_xunce_full_network_v1.py \
  tests/test_xunce_full_network_static_contract_validation.py \
  tests/test_xunce_full_network_ablation_experiments.py \
  tests/test_xunce_full_network_stress_evaluation.py \
  tests/test_xunce_guarded_training_candidate_preflight.py \
  tests/test_xunce_controlled_training_candidate.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/test_global_99_*.py tests/test_network_architecture_upgrade_readiness_review.py -q
```

## Non-goals

- No PPO update.
- No checkpoint publication or installation.
- No default-policy replacement.
- No executor connection or online canary.
- No action-space/default-A* change.
- No real-world performance claim.
