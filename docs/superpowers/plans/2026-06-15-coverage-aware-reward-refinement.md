# Coverage-Aware Reward Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only Coverage-Aware Reward Refinement v1 audit that rescales existing coverage evidence into a verifiable reward contract without running PPO or claiming performance.

**Architecture:** The new runner reads formal PPO, post-training replay, selected-candidate, coverage-signal, and coverage-performance artifacts. It joins Stage 2 delta rows with selected-candidate shadow steps, audits each reward component source, writes per-row reward components, aggregates actor re-score comparisons, and fails when any required component is missing, contaminated, or based on expected/fallback data.

**Tech Stack:** Python standard library, existing JSON/JSONL artifacts, `unittest`/`pytest`, existing `scripts/git_provenance.py`.

---

### Task 1: Regression Tests

**Files:**
- Create: `tests/test_coverage_aware_reward_refinement.py`

- [ ] **Step 1: Write failing tests**

Add tests for actual coverage bonus, expected coverage rejection, missing valuable/information no-fake-bonus behavior, path/risk/energy penalties, fallback contamination, controlled regression penalty, teacher retention, read-only flags, artifact creation, and teacher-equivalent no-performance-improvement reporting.

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
/home/kai/anaconda3/envs/lunar-explorer/bin/python -m pytest tests/test_coverage_aware_reward_refinement.py -q
```

Expected: FAIL because `scripts.run_coverage_aware_reward_refinement` is not implemented yet.

### Task 2: Reward Refinement Runner

**Files:**
- Create: `scripts/run_coverage_aware_reward_refinement.py`
- Create: `scripts/run_coverage_aware_reward_refinement.sh`

- [ ] **Step 1: Implement CLI and artifact loading**

Create a runner that accepts `--formal-training-root`, optional `--post-training-replay-root`, `--selected-candidate-root`, `--coverage-signal-root`, `--coverage-performance-root`, and `--output-root`.

- [ ] **Step 2: Implement reward component source audit**

Compute:

```text
reward =
  coverage_gain_bonus
+ valuable_area_bonus
+ information_gain_bonus
+ teacher_skill_retention_bonus
- path_cost_penalty
- risk_penalty
- energy_penalty
- fallback_penalty
- controlled_regression_penalty
```

Reject expected coverage as actual gain, fallback/source gain as policy gain, missing valuable/information sources, and controlled regression.

- [ ] **Step 3: Write required artifacts**

Write summary JSON, reward-component audit JSONL, source-field audit JSON, comparison/re-score JSON, rejection report JSON, and markdown report under `outputs/path_feedback_batch_coverage_aware_reward_refinement_v1/`.

### Task 3: Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/算法设计与系统架构报告.md`
- Create: `docs/superpowers/specs/2026-06-15-coverage-aware-reward-refinement.md`

- [ ] **Step 1: Update project docs**

Document that Stage 4 is implemented as a read-only audit/preflight. Record whether the current evidence passes or fails, the exact output root, and the next required change.

- [ ] **Step 2: Preserve boundaries**

State that the stage does not run a new PPO update, publish or replace checkpoints, relax guards, connect a real executor, or make a formal performance claim.

### Task 4: Verification

- [ ] **Step 1: Run focused tests**

```bash
/home/kai/anaconda3/envs/lunar-explorer/bin/python -m pytest tests/test_coverage_aware_reward_refinement.py -q
```

- [ ] **Step 2: Run real audit**

```bash
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python bash scripts/run_coverage_aware_reward_refinement.sh \
  --formal-training-root outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1 \
  --selected-candidate-root outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_preflight_v1 \
  --coverage-signal-root outputs/path_feedback_batch_exploration_coverage_signal_audit_v1 \
  --coverage-performance-root outputs/path_feedback_batch_exploration_coverage_performance_evaluation_v1 \
  --output-root outputs/path_feedback_batch_coverage_aware_reward_refinement_v1
```

- [ ] **Step 3: Inspect gate summary**

```bash
jq '{status,reason_codes,next_required_change,reward_refinement_status,runs_new_ppo_update}' outputs/path_feedback_batch_coverage_aware_reward_refinement_v1/coverage-aware-reward-refinement-summary.json
git diff --check
```
