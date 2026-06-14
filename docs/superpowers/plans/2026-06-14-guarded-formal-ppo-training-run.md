# Guarded Formal PPO Training Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the authorized guarded formal PPO training pass across seeds 0-4 and expose the result to readiness as `guarded_formal_ppo_training_run_evaluated`.

**Architecture:** Reuse the existing limited PPO update smoke implementation as the per-seed optimizer kernel. Add a guarded formal wrapper that validates the authorization summary, runs one experimental update per seed, audits post-training gates from the per-seed summaries, writes progress/report artifacts, and calls readiness validate-only against the wrapper summary.

**Tech Stack:** Python, pytest/unittest, bash closure scripts, JSON summaries, existing model-explorer PPO utilities.

---

### Task 1: Training Run Tests

**Files:**
- Create: `tests/test_guarded_formal_ppo_training_run.py`
- Modify: `tests/test_policy_training_readiness_review.py`

- [ ] Write tests for a passing 5-seed training run using injected per-seed and readiness runners.
- [ ] Write tests for authorization leakage/post-gate failures blocking the wrapper.
- [ ] Write readiness tests for `--guarded-formal-ppo-training-run-summary`.
- [ ] Run targeted tests and confirm they fail because the new runner/config/readiness hook does not exist yet.

### Task 2: Training Run Runner

**Files:**
- Create: `scripts/run_guarded_formal_ppo_training_run.py`
- Create: `scripts/run_guarded_formal_ppo_training_run.sh`
- Create: `scripts/run_guarded_formal_ppo_training_run_closure.sh`
- Create: `configs/guarded_formal_ppo_training_run_v1.json`

- [ ] Load and validate the authorization summary.
- [ ] For each seed, call the limited PPO update smoke kernel with seed-specific output root and config.
- [ ] Aggregate seed status, old-policy reconstruction, finite counters, KL, gradient, teacher agreement, controlled regression, holdout/canary gates, and checkpoint flags.
- [ ] Write `formal-ppo-training-run-summary.json`, `formal-ppo-training-run-seed-summaries.json`, `formal-ppo-training-run-progress.jsonl`, `formal-ppo-training-run-readiness-validate-only.json`, and report markdown.
- [ ] Keep all checkpoints experimental and never publish or replace default policy.

### Task 3: Readiness Extension

**Files:**
- Modify: `scripts/run_policy_training_readiness_review.py`
- Modify: `tests/test_policy_training_readiness_review.py`

- [ ] Add `--guarded-formal-ppo-training-run-summary`.
- [ ] Add schema/status constants and readiness helper.
- [ ] Insert the new status above `guarded_formal_ppo_training_authorized`.
- [ ] Block on failed seeds, non-finite metrics, KL/grad violations, post-training gate regressions, publication/default-policy claims, or stale git provenance.

### Task 4: Docs And Verification

**Files:**
- Create: `docs/superpowers/specs/2026-06-14-guarded-formal-ppo-training-run.md`
- Modify: `README.md`
- Modify: `docs/算法设计与系统架构报告.md`

- [ ] Document that this is a controlled formal PPO training run, not release.
- [ ] Run the requested pytest command.
- [ ] Run `scripts/run_guarded_formal_ppo_training_run_closure.sh`.
- [ ] Run readiness validate-only with the new summary.
- [ ] Run `git diff --check`.
