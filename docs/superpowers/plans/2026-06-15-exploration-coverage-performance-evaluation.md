# Exploration Coverage Performance Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Stage 3 coverage-performance evaluation for the guarded formal PPO candidate without making a release claim.

**Architecture:** Add a read-only evaluator that consumes the already-audited real coverage delta rows, enriches them with shadow rollout cost/risk/action context, aggregates selected PPO and comparator metrics, and writes pass/fail artifacts. Passing requires actual coverage improvement plus no unacceptable path-cost, risk, fallback, or controlled-regression regression.

**Tech Stack:** Python standard library, JSON/JSONL artifacts, existing git provenance helper, pytest/unittest tests, shell runner.

---

### Task 1: Tests

**Files:**
- Create: `tests/test_exploration_coverage_performance_evaluation.py`

- [ ] **Step 1: Write failing tests**

Cover:
- selected PPO beats teacher/source/default and passes
- selected PPO does not improve coverage and fails
- selected PPO improves coverage but path cost/risk/fallback efficiency regresses and fails
- expected coverage is confused with actual coverage and fails
- fallback/source gain is claimed as policy gain and fails
- controlled regression is nonzero and fails
- no comparator evidence fails

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
/home/kai/anaconda3/envs/lunar-explorer/bin/python -m pytest tests/test_exploration_coverage_performance_evaluation.py -q
```

Expected: fail because `scripts.run_exploration_coverage_performance_evaluation` does not exist.

### Task 2: Evaluator

**Files:**
- Create: `scripts/run_exploration_coverage_performance_evaluation.py`
- Create: `scripts/run_exploration_coverage_performance_evaluation.sh`

- [ ] **Step 1: Implement the minimal runner**

Read:
- formal training summary
- post-training replay summary
- selected candidate promotion summary
- multihorizon shadow summary and steps
- coverage signal audit summary and `coverage-delta-audit.jsonl`

Write:
- `exploration-coverage-performance-evaluation-summary.json`
- `coverage-performance-metric-table.jsonl`
- `coverage-performance-comparison-audit.json`
- `coverage-performance-rejection-report.json`
- `exploration-coverage-performance-evaluation-report.md`

- [ ] **Step 2: Run tests to verify GREEN**

Run:

```bash
/home/kai/anaconda3/envs/lunar-explorer/bin/python -m pytest tests/test_exploration_coverage_performance_evaluation.py -q
```

Expected: all tests pass.

### Task 3: Docs And Current Artifact Run

**Files:**
- Modify: `README.md`
- Modify: `docs/算法设计与系统架构报告.md`
- Create: `docs/superpowers/specs/2026-06-15-exploration-coverage-performance-evaluation.md`

- [ ] **Step 1: Run current evaluation**

Run the shell entrypoint against current outputs and inspect the summary with `jq`.

- [ ] **Step 2: Update docs**

Document that Stage 3 is implemented and read-only. If the current artifact fails, state the concrete reason codes and the next required change without claiming PPO performance improvement.

- [ ] **Step 3: Final verification**

Run targeted tests, the evaluator, `jq`, `git diff --check`, and confirm `docs/月面巡视探索.md` remains untouched.
