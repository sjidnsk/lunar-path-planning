# Xunce Stage 14 Post-Training Offline Evaluation v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluate the Stage 13 research checkpoint offline before any shadow/replay or sandbox governance.

**Architecture:** This stage reads the controlled training candidate summary, metadata, and research-only checkpoint. It loads the checkpoint read-only into `XunceFullNetworkV1`, compares trained metrics against a deterministic fresh model on synthetic topology-contract cases, and writes an audit decision.

**Tech Stack:** Python, PyTorch, JSON/JSONL artifacts, existing `scripts/xunce_full_network_common.py`, unittest/pytest, shell runner.

---

## Files

- Create: `configs/xunce_post_training_offline_evaluation_v1.json`
- Create: `scripts/run_xunce_post_training_offline_evaluation.py`
- Create: `scripts/run_xunce_post_training_offline_evaluation.sh`
- Create: `tests/test_xunce_post_training_offline_evaluation.py`
- Modify: `README.md`
- Modify: `docs/算法设计与系统架构报告.md`
- Modify: `docs/superpowers/specs/2026-06-16-global-99-exploration-coverage-line.md`
- Modify: `docs/superpowers/specs/2026-06-16-topology-aware-coverage-policy-network-design.md`

Output root:

`outputs/path_feedback_batch_xunce_post_training_offline_evaluation_v1/`

Artifacts:

- `xunce-post-training-offline-evaluation-summary.json`
- `xunce-post-training-offline-evaluation-manifest.json`
- `xunce-post-training-offline-evaluation-results.jsonl`
- `xunce-post-training-checkpoint-load-audit.json`
- `xunce-post-training-policy-delta-audit.json`
- `xunce-post-training-boundary-audit.json`
- `xunce-post-training-rejection-report.json`
- `xunce-post-training-offline-evaluation-report.md`

## Decision Contract

Source requirements:

- Stage 13 `status=passed`
- `next_required_change=post_training_offline_evaluation`
- `writes_research_checkpoint=true`
- `publishes_checkpoint=false`
- checkpoint and metadata exist.

Passing summary:

- `status=passed`
- `checkpoint_loaded=true`
- `checkpoint_read_only=true`
- `post_training_offline_evaluation_passed=true`
- `trained_loss_lower_than_fresh=true`
- `target_probability_improved=true`
- `runs_new_training_update=false`
- `runs_new_ppo_update=false`
- `publishes_checkpoint=false`
- `next_required_change=shadow_replay_validation`

Failure routing:

- source missing/failed/wrong next: `fix_controlled_training_candidate`
- checkpoint missing/invalid: `fix_controlled_training_candidate_checkpoint`
- offline metric regression: `fix_post_training_offline_evaluation`
- release/executor/default-policy boundary violation: `resolve_xunce_post_training_offline_boundary_rejections`

## Task 1: Write Failing Tests

- [ ] Add `tests/test_xunce_post_training_offline_evaluation.py`.
- [ ] Cover default pass, missing source, missing checkpoint, regressed checkpoint, and boundary violation.
- [ ] Run:

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/test_xunce_post_training_offline_evaluation.py -q
```

Expected before implementation: import/module failure for `run_xunce_post_training_offline_evaluation`.

## Task 2: Implement Runner and Config

- [ ] Validate config and source artifact paths.
- [ ] Load checkpoint with `map_location="cpu"` and without writing checkpoint files.
- [ ] Recreate deterministic evaluation cases matching the Stage 13 synthetic contract.
- [ ] Compare fresh model vs trained checkpoint:
  - target cross-entropy loss;
  - target probability;
  - target logit;
  - finite logits/value outputs;
  - latency and parameter count.
- [ ] Write JSONL case results plus checkpoint-load and policy-delta audits.
- [ ] Keep all release/default-policy/executor/PPO boundaries closed.

## Task 3: Implement Shell Entrypoint

- [ ] Add `scripts/run_xunce_post_training_offline_evaluation.sh`.
- [ ] Support:

```bash
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python bash scripts/run_xunce_post_training_offline_evaluation.sh
```

## Task 4: Update Documentation

- [ ] Update README with Stage 14 command, output root, and next gate.
- [ ] Update architecture report with read-only checkpoint evaluation boundary.
- [ ] Update Global 99 spec and 巡策 design spec with `shadow_replay_validation`.

## Task 5: Verification

Run:

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/test_xunce_post_training_offline_evaluation.py -q
PYTHON=$PY bash scripts/run_xunce_post_training_offline_evaluation.sh
jq '{status,reason_codes,checkpoint_loaded,post_training_offline_evaluation_passed,trained_loss_lower_than_fresh,target_probability_improved,next_required_change,publishes_checkpoint,runs_new_training_update,runs_new_ppo_update}' \
  outputs/path_feedback_batch_xunce_post_training_offline_evaluation_v1/xunce-post-training-offline-evaluation-summary.json
rg -n "Xunce Post-Training Offline Evaluation v1|run_xunce_post_training_offline_evaluation|shadow_replay_validation" \
  README.md docs/算法设计与系统架构报告.md docs/superpowers/specs
git diff --check
```

Full regression before commit:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/test_xunce_*.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/test_global_99_*.py tests/test_network_architecture_upgrade_readiness_review.py -q
```

## Non-goals

- No training update.
- No PPO update.
- No checkpoint write/publication/installation.
- No default-policy replacement.
- No executor connection or online canary.
- No real-world performance claim.
