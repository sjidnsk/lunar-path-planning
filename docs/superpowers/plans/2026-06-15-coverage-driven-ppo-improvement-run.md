# Coverage-Driven PPO Improvement Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `Coverage-Driven PPO Improvement Run v1`, a guarded offline PPO update that consumes audited coverage-aware rewards and evaluates whether exploration coverage improves.

**Architecture:** Add one stage runner that reads current upstream summaries, materializes a coverage-aware collector from Stage 4 reward rows plus selected shadow observations, calls the existing limited PPO update implementation, then writes replay/performance/rejection artifacts. Keep all release boundaries false.

**Tech Stack:** Python stdlib, existing `scripts/run_limited_ppo_update_smoke.py`, JSON/JSONL artifacts, unittest/pytest.

---

### Task 1: Test The Stage Contract

**Files:**
- Create: `tests/test_coverage_driven_ppo_improvement_run.py`

- [ ] **Step 1: Write failing pass-path test**

Create a synthetic one-action policy checkpoint, Stage 4 reward rows with positive selected coverage reward, a lower teacher baseline, and matching shadow steps with finite `log_prob` and `value`. Assert the stage writes a coverage-aware batch, runs a PPO update, writes an experimental checkpoint, compares against teacher/source/default/pre-improvement baselines, and keeps publish/replacement/performance claims false.

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
/home/kai/anaconda3/envs/lunar-explorer/bin/python -m pytest tests/test_coverage_driven_ppo_improvement_run.py -q
```

Expected: fails because `scripts/run_coverage_driven_ppo_improvement_run.py` does not exist.

- [ ] **Step 3: Write failing failure-path tests**

Add tests for teacher-equivalent/no-improvement rows, missing reward source gate, and release boundary invariants. Expected reason codes must include `no_coverage_return_improvement`, `valuable_coverage_not_improved`, or `coverage_reward_source_not_passed` as appropriate.

### Task 2: Implement The Runner

**Files:**
- Create: `scripts/run_coverage_driven_ppo_improvement_run.py`
- Create: `scripts/run_coverage_driven_ppo_improvement_run.sh`

- [ ] **Step 1: Add CLI and upstream loading**

Implement arguments from the goal prompt, default post-training replay root, JSON/JSONL readers, path resolution, and upstream status validation for formal training, replay, selected candidate, Stage 2, Stage 4, and connector.

- [ ] **Step 2: Materialize the coverage-aware PPO batch**

Join `reward-component-audit.jsonl` with selected shadow steps by context or episode/step. Emit `coverage-aware-ppo-batch/ppo-rollout-episodes.jsonl`, `ppo-rollout-transitions.jsonl`, and `ppo-rollout-collector-summary.json`. Trainable rows must be train split, selected PPO actor, non-fallback, finite reward/log_prob/value, valid action mask, and no controlled regression.

- [ ] **Step 3: Run the guarded PPO update**

Call `run_limited_ppo_update_smoke` with base candidate root from selected-candidate summary, reward-driven output file names, one epoch, small learning rate, old-policy reconstruction limits, KL/grad gates, and trainable filter `controlled_choice_sources=["policy"]`, `splits=["train"]`.

- [ ] **Step 4: Evaluate post-update coverage performance**

Load the updated checkpoint, infer raw actions on audited observations, accept only guard-clean actions with real audited coverage evidence, aggregate `post_improvement_ppo` metrics, compare against teacher/source-default/pre-improvement selected PPO baselines, and write metric table, replay audit, rejection report, summary, and report.

### Task 3: Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/算法设计与系统架构报告.md`

- [ ] **Step 1: Document current Stage 5 behavior**

Add the new script/output paths, explain that the stage runs a guarded offline PPO update using coverage-aware reward, and state that failure is valid when no coverage improvement is proven.

- [ ] **Step 2: Preserve boundaries**

Document that no checkpoint is published, no default policy is replaced, no performance claim is made, and `docs/月面巡视探索.md` remains untouched.

### Task 4: Verification

- [ ] **Step 1: Run focused tests**

```bash
/home/kai/anaconda3/envs/lunar-explorer/bin/python -m pytest tests/test_coverage_driven_ppo_improvement_run.py tests/test_coverage_aware_reward_refinement.py tests/test_exploration_coverage_performance_evaluation.py -q
```

- [ ] **Step 2: Run the real stage**

```bash
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python bash scripts/run_coverage_driven_ppo_improvement_run.sh --formal-training-root outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1 --selected-candidate-root outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_preflight_v1 --coverage-signal-root outputs/path_feedback_batch_exploration_coverage_signal_audit_v1 --coverage-performance-root outputs/path_feedback_batch_exploration_coverage_performance_evaluation_v1 --reward-refinement-root outputs/path_feedback_batch_coverage_aware_reward_refinement_v1 --reward-source-root outputs/path_feedback_batch_connect_reward_component_source_fields_v1 --output-root outputs/path_feedback_batch_coverage_driven_ppo_improvement_run_v1
```

- [ ] **Step 3: Check boundaries and diff hygiene**

```bash
jq '{status,reason_codes,next_required_change,publishes_checkpoint,replaces_default_policy,performance_claimed}' outputs/path_feedback_batch_coverage_driven_ppo_improvement_run_v1/coverage-driven-ppo-improvement-run-summary.json
git diff --check
git diff -- docs/月面巡视探索.md
```
