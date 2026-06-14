# Guarded Formal PPO Post-Training Stability Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a post-training replay stage that repeatedly audits the five experimental candidates produced by Guarded Formal PPO Training Run v1 without running another PPO update.

**Architecture:** Reuse the existing formal training run summary and per-seed artifacts as the replay source. Add a focused runner that validates input provenance, loads seed summaries, invokes a replay runner per seed/replay, compares replay metrics against the original seed summary, writes audit artifacts, and calls readiness validate-only through a new CLI summary flag.

**Tech Stack:** Python stdlib scripts, JSON/JSONL artifacts, existing `git_provenance`, pytest/unittest, shell closure wrappers.

---

### Task 1: TDD Coverage

**Files:**
- Create: `tests/test_guarded_formal_ppo_post_training_stability_replay.py`
- Modify: `tests/test_policy_training_readiness_review.py`

- [ ] Write tests for the replay runner passing with 5 seeds and 3 replays each.
- [ ] Write tests that fail on missing seed checkpoint and replay behavior drift.
- [ ] Write tests for config/documentation/non-goal declarations.
- [ ] Write readiness tests for `guarded_formal_ppo_post_training_stability_replay_evaluated` and blocker cases.
- [ ] Run the new tests and confirm they fail because the module/config/CLI do not exist yet.

### Task 2: Replay Runner

**Files:**
- Create: `scripts/run_guarded_formal_ppo_post_training_stability_replay.py`
- Create: `scripts/run_guarded_formal_ppo_post_training_stability_replay.sh`
- Create: `scripts/run_guarded_formal_ppo_post_training_stability_replay_closure.sh`
- Create: `configs/guarded_formal_ppo_post_training_stability_replay_v1.json`

- [ ] Implement config validation and defaults.
- [ ] Read `formal-ppo-training-run-summary.json` and seed summaries.
- [ ] Validate input status, clean/current provenance, five passed seeds, and no publish/default/performance/formal-ready claims.
- [ ] Locate each seed checkpoint under `seed-XX/limited_ppo_update_smoke/experimental-hybrid-policy-candidate.pt`.
- [ ] Run three replay audits per seed through an injectable replay runner.
- [ ] Aggregate replay counts, trainable counts, finite counters, teacher agreement, controlled regression, holdout/canary statuses, and drift counts.
- [ ] Write progress JSONL, seed replay summaries JSONL, drift report JSONL, gate audit JSON, rollback manifest JSON, summary JSON, readiness JSON, and report Markdown.

### Task 3: Readiness Integration

**Files:**
- Modify: `scripts/run_policy_training_readiness_review.py`
- Modify: `tests/test_policy_training_readiness_review.py`

- [ ] Add CLI flag `--guarded-formal-ppo-post-training-stability-replay-summary`.
- [ ] Add stage-only analyzer and readiness helper.
- [ ] Require passed summary, replay count >= 15, all replays passed, zero drift/regression/leakage/non-finite/missing counters, no publication claims, and clean current provenance.
- [ ] Return `guarded_formal_ppo_post_training_stability_replay_evaluated` when complete.

### Task 4: Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/算法设计与系统架构报告.md`
- Create: `docs/superpowers/specs/2026-06-14-guarded-formal-ppo-post-training-stability-replay.md`

- [ ] Document that this is replay validation, not a new PPO update or release.
- [ ] Record the expected output root and verification commands.
- [ ] Preserve scope guards around no online canary, no real executor, no checkpoint publication, no default replacement, no gate relaxation, and no Ackermann claim.

### Task 5: Verification

- [ ] Run the target pytest command from the goal.
- [ ] Run `scripts/run_guarded_formal_ppo_post_training_stability_replay_closure.sh`.
- [ ] Run readiness validate-only with the new summary flag.
- [ ] Run `git diff --check`.
- [ ] Inspect the final summary fields required by the goal.
