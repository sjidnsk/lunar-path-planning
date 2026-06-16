# Xunce Stage 17 Release Governance Gate v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the completed sandbox candidate evidence into a final research-track governance decision without approving default-policy replacement.

**Architecture:** This stage consumes Stage 16 sandbox candidate preflight evidence, audits lineage/scope/boundaries, and writes a final governance verdict. It can declare the 巡策 research chain complete, but it cannot publish or install the checkpoint.

**Tech Stack:** Python, JSON artifacts, unittest/pytest, shell runner.

---

## Files

- Create: `configs/xunce_release_governance_gate_v1.json`
- Create: `scripts/run_xunce_release_governance_gate.py`
- Create: `scripts/run_xunce_release_governance_gate.sh`
- Create: `tests/test_xunce_release_governance_gate.py`
- Modify: `README.md`
- Modify: `docs/算法设计与系统架构报告.md`
- Modify: `docs/superpowers/specs/2026-06-16-global-99-exploration-coverage-line.md`
- Modify: `docs/superpowers/specs/2026-06-16-topology-aware-coverage-policy-network-design.md`

Output root:

`outputs/path_feedback_batch_xunce_release_governance_gate_v1/`

Artifacts:

- `xunce-release-governance-gate-summary.json`
- `xunce-release-governance-gate-manifest.json`
- `xunce-release-evidence-lineage-audit.json`
- `xunce-release-scope-audit.json`
- `xunce-release-boundary-audit.json`
- `xunce-release-governance-decision-audit.json`
- `xunce-release-rejection-report.json`
- `xunce-release-governance-gate-report.md`

## Decision Contract

Source requirements:

- Stage 16 `status=passed`
- `next_required_change=xunce_release_governance_gate`
- `sandbox_candidate_preflight_passed=true`
- `sandbox_load_verified=true`
- kill-switch / rollback / telemetry audits passed
- release/default-policy/executor/online-canary boundaries closed.

Passing summary:

- `status=passed`
- `release_governance_gate_passed=true`
- `release_governance_verdict=research_candidate_ready_for_human_governance_review`
- `xunce_research_chain_complete=true`
- `default_policy_replacement_approved=false`
- `real_world_release_approved=false`
- `publishes_checkpoint=false`
- `connects_real_executor=false`
- `starts_online_canary=false`
- `next_required_change=xunce_research_track_complete`

Failure routing:

- Stage 16 missing/failed/wrong next: `fix_xunce_sandbox_candidate_preflight`
- release boundary violation: `resolve_xunce_release_governance_boundary_rejections`
- governance audit failure: `fix_xunce_release_governance_gate`

## Task 1: Write Failing Tests

- [ ] Add `tests/test_xunce_release_governance_gate.py`.
- [ ] Cover default pass, missing Stage 16 source, failed/wrong Stage 16 source, boundary violation, and missing kill-switch/rollback/telemetry evidence.
- [ ] Run:

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/test_xunce_release_governance_gate.py -q
```

Expected before implementation: import/module failure for `run_xunce_release_governance_gate`.

## Task 2: Implement Runner and Config

- [ ] Validate config and source root.
- [ ] Read Stage 16 summary.
- [ ] Audit evidence lineage, scope, boundary, and governance decision fields.
- [ ] Write all artifacts and closed boundary fields.

## Task 3: Implement Shell Entrypoint

- [ ] Add `scripts/run_xunce_release_governance_gate.sh`.

## Task 4: Update Documentation

- [ ] Update README, architecture report, Global 99 spec, and 巡策 design spec with final gate and completion wording.

## Task 5: Verification

Run:

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/test_xunce_release_governance_gate.py -q
PYTHON=$PY bash scripts/run_xunce_release_governance_gate.sh
jq '{status,reason_codes,release_governance_gate_passed,release_governance_verdict,xunce_research_chain_complete,next_required_change,publishes_checkpoint,replaces_default_policy,connects_real_executor,starts_online_canary}' \
  outputs/path_feedback_batch_xunce_release_governance_gate_v1/xunce-release-governance-gate-summary.json
rg -n "Xunce Release Governance Gate v1|run_xunce_release_governance_gate|xunce_research_track_complete" \
  README.md docs/算法设计与系统架构报告.md docs/superpowers/specs
git diff --check
```

Full regression before commit:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/test_xunce_*.py -q
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/test_global_99_*.py tests/test_network_architecture_upgrade_readiness_review.py -q
```

## Non-goals

- No default-policy replacement approval.
- No real-world release approval.
- No checkpoint publication or installation.
- No executor connection.
- No online canary.
- No PPO/training update.
