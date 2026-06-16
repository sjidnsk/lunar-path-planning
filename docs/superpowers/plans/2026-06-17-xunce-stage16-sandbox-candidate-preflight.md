# Xunce Stage 16 Sandbox Candidate Preflight v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package the trained `xunce_full_network_v1` research checkpoint as a sandbox-only candidate and verify load, hash, kill-switch, rollback, telemetry, and isolation boundaries before release governance.

**Architecture:** This stage consumes Stage 15 shadow/replay evidence and the Stage 13 research checkpoint. It copies the checkpoint into an output-root sandbox package, hashes source and package copies, loads the packaged checkpoint in a sandbox dry-run, and emits governance audits. It does not install or publish the checkpoint.

**Tech Stack:** Python, PyTorch, JSON artifacts, SHA-256 hashing, existing `scripts/xunce_full_network_common.py`, unittest/pytest, shell runner.

---

## Files

- Create: `configs/xunce_sandbox_candidate_preflight_v1.json`
- Create: `scripts/run_xunce_sandbox_candidate_preflight.py`
- Create: `scripts/run_xunce_sandbox_candidate_preflight.sh`
- Create: `tests/test_xunce_sandbox_candidate_preflight.py`
- Modify: `README.md`
- Modify: `docs/算法设计与系统架构报告.md`
- Modify: `docs/superpowers/specs/2026-06-16-global-99-exploration-coverage-line.md`
- Modify: `docs/superpowers/specs/2026-06-16-topology-aware-coverage-policy-network-design.md`

Output root:

`outputs/path_feedback_batch_xunce_sandbox_candidate_preflight_v1/`

Artifacts:

- `xunce-sandbox-candidate-preflight-summary.json`
- `xunce-sandbox-candidate-preflight-manifest.json`
- `xunce-sandbox-package-manifest.json`
- `xunce-sandbox-checkpoint-hash-audit.json`
- `xunce-sandbox-load-audit.json`
- `xunce-sandbox-kill-switch-audit.json`
- `xunce-sandbox-rollback-audit.json`
- `xunce-sandbox-telemetry-audit.json`
- `xunce-sandbox-boundary-audit.json`
- `xunce-sandbox-rejection-report.json`
- `xunce-sandbox-candidate-preflight-report.md`
- `sandbox_package/xunce-controlled-training-candidate.pt`

## Decision Contract

Source requirements:

- Stage 15 `status=passed`
- `next_required_change=sandbox_candidate_preflight`
- `shadow_replay_passed=true`
- `source_match_audit_passed=true`
- source release/executor/online-canary boundaries closed.

Passing summary:

- `status=passed`
- `sandbox_candidate_preflight_passed=true`
- `sandbox_package_created=true`
- `sandbox_checkpoint_hash_verified=true`
- `sandbox_load_verified=true`
- `kill_switch_audit_passed=true`
- `rollback_audit_passed=true`
- `telemetry_audit_passed=true`
- `default_policy_read_only=true`
- `executor_isolation_passed=true`
- `publishes_checkpoint=false`
- `replaces_default_policy=false`
- `connects_real_executor=false`
- `starts_online_canary=false`
- `next_required_change=xunce_release_governance_gate`

Failure routing:

- Stage 15 missing/failed/wrong next: `fix_xunce_shadow_replay_validation`
- checkpoint missing/invalid: `fix_controlled_training_candidate_checkpoint`
- package hash/load failure: `fix_xunce_sandbox_candidate_preflight`
- boundary violation: `resolve_xunce_sandbox_candidate_boundary_rejections`

## Task 1: Write Failing Tests

- [ ] Add `tests/test_xunce_sandbox_candidate_preflight.py`.
- [ ] Cover default pass, missing Stage 15 source, missing checkpoint, invalid checkpoint, and boundary violation.
- [ ] Run:

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/test_xunce_sandbox_candidate_preflight.py -q
```

Expected before implementation: import/module failure for `run_xunce_sandbox_candidate_preflight`.

## Task 2: Implement Runner and Config

- [ ] Validate config and source roots.
- [ ] Read Stage 15 summary and validate boundary fields.
- [ ] Copy Stage 13 checkpoint into `sandbox_package/`.
- [ ] Compute SHA-256 of source and sandbox checkpoint and compare.
- [ ] Load sandbox checkpoint into `XunceFullNetworkV1` and run one deterministic forward pass.
- [ ] Write package manifest plus hash/load/kill-switch/rollback/telemetry/boundary audits.

## Task 3: Implement Shell Entrypoint

- [ ] Add `scripts/run_xunce_sandbox_candidate_preflight.sh`.

## Task 4: Update Documentation

- [ ] Update README, architecture report, Global 99 spec, and 巡策 design spec with Stage 16 command and next gate.

## Task 5: Verification

Run:

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/test_xunce_sandbox_candidate_preflight.py -q
PYTHON=$PY bash scripts/run_xunce_sandbox_candidate_preflight.sh
jq '{status,reason_codes,sandbox_candidate_preflight_passed,sandbox_package_created,sandbox_load_verified,kill_switch_audit_passed,rollback_audit_passed,telemetry_audit_passed,next_required_change,publishes_checkpoint,replaces_default_policy,connects_real_executor}' \
  outputs/path_feedback_batch_xunce_sandbox_candidate_preflight_v1/xunce-sandbox-candidate-preflight-summary.json
rg -n "Xunce Sandbox Candidate Preflight v1|run_xunce_sandbox_candidate_preflight|xunce_release_governance_gate" \
  README.md docs/算法设计与系统架构报告.md docs/superpowers/specs
git diff --check
```

Full regression before commit:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/test_xunce_*.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/test_global_99_*.py tests/test_network_architecture_upgrade_readiness_review.py -q
```

## Non-goals

- No default-policy installation.
- No checkpoint publication.
- No executor connection.
- No online canary.
- No PPO/training update.
- No real-world performance claim.
