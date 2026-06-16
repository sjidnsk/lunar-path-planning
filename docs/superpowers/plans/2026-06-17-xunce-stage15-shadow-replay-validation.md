# Xunce Stage 15 Shadow Replay Validation v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the Stage 14 post-training offline evaluation can be replayed deterministically before any sandbox packaging.

**Architecture:** This stage reads the Stage 14 summary/results and the Stage 13 research checkpoint. It reloads the checkpoint read-only, replays the same deterministic topology-contract cases, compares replay rows against Stage 14 source rows, and emits a source-match/determinism audit.

**Tech Stack:** Python, PyTorch, JSON/JSONL artifacts, existing `scripts/xunce_full_network_common.py`, existing Stage 13 synthetic batch helper, unittest/pytest, shell runner.

---

## Files

- Create: `configs/xunce_shadow_replay_validation_v1.json`
- Create: `scripts/run_xunce_shadow_replay_validation.py`
- Create: `scripts/run_xunce_shadow_replay_validation.sh`
- Create: `tests/test_xunce_shadow_replay_validation.py`
- Modify: `README.md`
- Modify: `docs/算法设计与系统架构报告.md`
- Modify: `docs/superpowers/specs/2026-06-16-global-99-exploration-coverage-line.md`
- Modify: `docs/superpowers/specs/2026-06-16-topology-aware-coverage-policy-network-design.md`

Output root:

`outputs/path_feedback_batch_xunce_shadow_replay_validation_v1/`

Artifacts:

- `xunce-shadow-replay-validation-summary.json`
- `xunce-shadow-replay-validation-manifest.json`
- `xunce-shadow-replay-results.jsonl`
- `xunce-shadow-replay-source-match-audit.json`
- `xunce-shadow-replay-checkpoint-load-audit.json`
- `xunce-shadow-replay-boundary-audit.json`
- `xunce-shadow-replay-rejection-report.json`
- `xunce-shadow-replay-validation-report.md`

## Decision Contract

Source requirements:

- Stage 14 `status=passed`
- `next_required_change=shadow_replay_validation`
- `checkpoint_loaded=true`
- `post_training_offline_evaluation_passed=true`
- source results JSONL exists.

Passing summary:

- `status=passed`
- `shadow_replay_passed=true`
- `source_match_audit_passed=true`
- `replay_determinism_passed=true`
- `checkpoint_loaded=true`
- `checkpoint_read_only=true`
- `starts_online_canary=false`
- `connects_real_executor=false`
- `publishes_checkpoint=false`
- `next_required_change=sandbox_candidate_preflight`

Failure routing:

- Stage 14 missing/failed/wrong next: `fix_post_training_offline_evaluation`
- checkpoint missing/invalid: `fix_controlled_training_candidate_checkpoint`
- source-match mismatch: `fix_xunce_shadow_replay_determinism`
- boundary violation: `resolve_xunce_shadow_replay_boundary_rejections`

## Task 1: Write Failing Tests

- [ ] Add `tests/test_xunce_shadow_replay_validation.py`.
- [ ] Cover default pass, missing Stage 14 source, missing checkpoint, source-match mismatch, and boundary violation.
- [ ] Run:

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/test_xunce_shadow_replay_validation.py -q
```

Expected before implementation: import/module failure for `run_xunce_shadow_replay_validation`.

## Task 2: Implement Runner and Config

- [ ] Validate config and source roots.
- [ ] Read Stage 14 summary and source result rows.
- [ ] Load Stage 13 checkpoint read-only.
- [ ] Replay deterministic cases with the trained checkpoint.
- [ ] Compare replay/source by `case_id`, target probability, target logit, trained loss, and finite outputs.
- [ ] Write replay rows, source-match audit, boundary audit, manifest, summary, rejection report, and markdown report.

## Task 3: Implement Shell Entrypoint

- [ ] Add `scripts/run_xunce_shadow_replay_validation.sh`.
- [ ] Support:

```bash
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python bash scripts/run_xunce_shadow_replay_validation.sh
```

## Task 4: Update Documentation

- [ ] Update README with Stage 15 command, output root, and next gate.
- [ ] Update architecture report with offline-only shadow/replay boundary.
- [ ] Update Global 99 spec and 巡策 design spec with `sandbox_candidate_preflight`.

## Task 5: Verification

Run:

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/test_xunce_shadow_replay_validation.py -q
PYTHON=$PY bash scripts/run_xunce_shadow_replay_validation.sh
jq '{status,reason_codes,shadow_replay_passed,source_match_audit_passed,replay_determinism_passed,next_required_change,publishes_checkpoint,connects_real_executor,starts_online_canary}' \
  outputs/path_feedback_batch_xunce_shadow_replay_validation_v1/xunce-shadow-replay-validation-summary.json
rg -n "Xunce Shadow Replay Validation v1|run_xunce_shadow_replay_validation|sandbox_candidate_preflight" \
  README.md docs/算法设计与系统架构报告.md docs/superpowers/specs
git diff --check
```

Full regression before commit:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/test_xunce_*.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/test_global_99_*.py tests/test_network_architecture_upgrade_readiness_review.py -q
```

## Non-goals

- No online canary.
- No real executor connection.
- No checkpoint publication or installation.
- No default-policy replacement.
- No PPO/training update.
- No real-world performance claim.
