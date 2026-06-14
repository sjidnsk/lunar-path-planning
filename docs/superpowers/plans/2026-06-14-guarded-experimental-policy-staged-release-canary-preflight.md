# Guarded Experimental Policy Staged Release Canary Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an offline guarded staged release canary preflight stage that audits canary eligibility, limit controls, rollback, telemetry, downgrade, budget, and operator approval without connecting a real executor.

**Architecture:** Mirror the existing staged release trial runner and readiness stage-only pattern. The new runner reads the frozen staged trial summary plus activation ledger, filters only gate-clean trial activations into a capped canary-eligible ledger, writes audit artifacts, invokes readiness validate-only, and reports a new stage-specific readiness status.

**Tech Stack:** Python scripts, JSON/JSONL artifacts, unittest/pytest, shell closure scripts, existing `scripts/run_policy_training_readiness_review.py`.

---

### Task 1: Canary Preflight Runner Tests

**Files:**
- Create: `tests/test_guarded_experimental_policy_staged_release_canary_preflight.py`
- Read: `tests/test_guarded_experimental_policy_staged_release_trial.py`

- [ ] Write a failing test for a passed staged trial with mixed ledger rows. Assert only gate-clean rows become canary-eligible, capped by `max_canary_control_activation_count`, and all audit flags pass.
- [ ] Run the new test file and confirm it fails because `scripts.run_guarded_experimental_policy_staged_release_canary_preflight` does not exist.

### Task 2: Readiness Tests

**Files:**
- Modify: `tests/test_policy_training_readiness_review.py`
- Read: staged trial readiness tests in the same file.

- [ ] Add a passing canary preflight summary helper and a test that `--guarded-experimental-policy-staged-release-canary-preflight-summary` advances readiness to `guarded_experimental_policy_staged_release_canary_preflight_evaluated`.
- [ ] Add a boundary test for wrong verdict, canary enabled, real executor, activation count out of bounds, controlled regression, and audit failure.
- [ ] Run the two new readiness tests and confirm the CLI rejects the unknown argument or lacks the new status.

### Task 3: Runner Implementation

**Files:**
- Create: `scripts/run_guarded_experimental_policy_staged_release_canary_preflight.py`
- Create: `scripts/run_guarded_experimental_policy_staged_release_canary_preflight.sh`
- Create: `scripts/run_guarded_experimental_policy_staged_release_canary_preflight_closure.sh`
- Create: `configs/guarded_experimental_policy_staged_release_canary_preflight_v1.json`

- [ ] Implement config/schema constants, input/output file defaults, CLI arguments, JSON helpers, and git provenance.
- [ ] Validate staged trial input: passed, expected verdict, staged release enabled, no regressions, kill-switch/rollback/telemetry passed, no publication claims, no real executor, clean/current provenance.
- [ ] Build canary eligibility rows from `staged-release-activation-ledger.jsonl`. Eligible rows require experimental control, policy source, no gate reasons, no controlled regression reasons, finite log_prob/value/reward, no fallback/rejected/diagnostic/missing/non-finite markers, and no controlled regression deltas.
- [ ] Cap eligible rows at `max_canary_control_activation_count=16`, set `staged_canary_enabled=false`, `connects_real_executor=false`, `default_policy_authoritative=true`, and `canary_traffic_fraction<=0.01`.
- [ ] Write manifest, eligibility ledger, kill-switch, rollback, telemetry, automatic downgrade, budget, operator approval, rejection report, summary, readiness JSON, and report markdown.
- [ ] Invoke readiness validate-only with the new CLI flag when pre-readiness checks pass.

### Task 4: Readiness Implementation

**Files:**
- Modify: `scripts/run_policy_training_readiness_review.py`

- [ ] Add new schema/status constants and CLI argument.
- [ ] Resolve the explicit canary preflight summary path and insert stage-only handling before staged trial.
- [ ] Implement `_analyze_guarded_experimental_policy_staged_release_canary_preflight_stage_only` and `_guarded_experimental_policy_staged_release_canary_preflight_readiness`.
- [ ] Validate verdict, source trial summary, required artifacts, canary disabled, no real executor, traffic <= 0.01, eligible count within 1..max, no diagnostic/fallback/rejected/missing/non-finite/control-regression eligibility, audits passed, no publication/performance/formal-ready claims, and clean/current git provenance.

### Task 5: Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/算法设计与系统架构报告.md`
- Create: `docs/superpowers/specs/2026-06-14-guarded-experimental-policy-staged-release-canary-preflight.md`

- [ ] Document the new output root, artifacts, status, acceptance gates, and boundary: offline preflight only, not an online canary.
- [ ] Preserve non-goals: no real executor, no new PPO, no checkpoint/default-policy replacement, no network/action-space/default-A* change, no gate relaxation, no Ackermann claim, no IRIS/GCS/path-planner diagnostics as release evidence.

### Task 6: Verification

- [ ] Run:

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q tests/test_guarded_experimental_policy_staged_release_canary_preflight.py tests/test_guarded_experimental_policy_staged_release_trial.py tests/test_policy_training_readiness_review.py
PYTHON=$P bash scripts/run_guarded_experimental_policy_staged_release_canary_preflight_closure.sh
PYTHON=$P bash scripts/run_policy_training_readiness_review.sh --batch-root outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1 --config configs/policy_training_readiness_review_v1.json --guarded-experimental-policy-staged-release-canary-preflight-summary outputs/path_feedback_batch_guarded_experimental_policy_staged_release_canary_preflight_v1/guarded-experimental-policy-staged-release-canary-preflight-summary.json --validate-only
git diff --check
```

- [ ] Inspect the generated summary for `status=passed`, `reason_codes=[]`, `verdict=eligible_for_guarded_staged_release_canary_dry_run`, `canary_eligible_activation_count<=16`, all audit flags passed, `staged_canary_enabled=false`, and `connects_real_executor=false`.
