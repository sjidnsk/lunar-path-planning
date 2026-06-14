# Guarded Formal PPO Training Authorization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline authorization stage that decides whether the existing guarded formal PPO evidence is sufficient to start the next controlled PPO training run.

**Architecture:** Follow the existing stage pattern: a standalone runner reads source summaries, writes an authorization summary plus audit artifacts, then invokes readiness validate-only against the newly written summary. Readiness gains one new explicit summary CLI and stage-only helper.

**Tech Stack:** Python stdlib, JSON/JSONL artifacts, pytest/unittest, existing git provenance helper and readiness script.

---

### Task 1: Authorization Runner Tests

**Files:**
- Create: `tests/test_guarded_formal_ppo_training_authorization.py`

- [ ] Write tests for a passed authorization summary with 684 authorized trainable transitions, seed/budget/stop/rollback/post-training manifests, no publication flags, and readiness status `guarded_formal_ppo_training_authorized`.
- [ ] Write tests that block authorization when a required source summary is failed or has stale git provenance.
- [ ] Write tests that reject validation/test/fallback/diagnostic trainable leakage and non-finite/missing counters.
- [ ] Write tests that verify config output files, docs updates, and non-goals.
- [ ] Run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q tests/test_guarded_formal_ppo_training_authorization.py` and confirm import/config failures before implementation.

### Task 2: Readiness Tests

**Files:**
- Modify: `tests/test_policy_training_readiness_review.py`

- [ ] Add a passed `guarded-formal-ppo-training-authorization-summary/v1` fixture.
- [ ] Add a test that `--guarded-formal-ppo-training-authorization-summary` advances readiness to `guarded_formal_ppo_training_authorized`.
- [ ] Add a blocker test for bad verdict, unexpected PPO update, publication/replacement, missing plan audits, or provenance mismatch.
- [ ] Run the targeted readiness tests and confirm the CLI argument/status are missing before implementation.

### Task 3: Runner, Config, and Closure

**Files:**
- Create: `configs/guarded_formal_ppo_training_authorization_v1.json`
- Create: `scripts/run_guarded_formal_ppo_training_authorization.py`
- Create: `scripts/run_guarded_formal_ppo_training_authorization.sh`
- Create: `scripts/run_guarded_formal_ppo_training_authorization_closure.sh`

- [ ] Implement source summary loading and git/current provenance checks for formal preflight, rollout canary, stability/holdout, candidate selection, promotion review, and canary preflight.
- [ ] Derive the authorized training input from stability/holdout counts and enforce `authorized_trainable_transition_count >= 684`.
- [ ] Write training input audit, budget manifest, seed plan, stop condition manifest, rollback manifest, post-training gate plan, readiness output, summary, and report.
- [ ] Ensure `runs_new_ppo_update=false`, checkpoint remains experimental, and no publication/replacement/performance/formal-ready claim is made.

### Task 4: Readiness Implementation

**Files:**
- Modify: `scripts/run_policy_training_readiness_review.py`

- [ ] Add schema/status constants and CLI argument.
- [ ] Add stage-only analyzer and `_guarded_formal_ppo_training_authorization_readiness`.
- [ ] Check all authorization counters, manifests, non-goals, and git provenance.
- [ ] Return `guarded_formal_ppo_training_authorized` only when the wrapper summary is passed and complete.

### Task 5: Docs and Verification

**Files:**
- Create: `docs/superpowers/specs/2026-06-14-guarded-formal-ppo-training-authorization.md`
- Modify: `README.md`
- Modify: `docs/算法设计与系统架构报告.md`

- [ ] Document that this is a training authorization gate, not a PPO run.
- [ ] Run the requested pytest command, closure, readiness validate-only, and `git diff --check`.
- [ ] Inspect the final summary fields against the goal acceptance gates.
