# Stage26.10B Completion-Capable Formal PPO Shadow Trial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run Stage26.10B to test terminal-aware reward under H1024+ completion-capable PPO shadow trials.

**Architecture:** Add a resumable Stage26.10B orchestrator that wraps existing Stage26.8M update/eval jobs and adds L0 completion reachability calibration, L1 paired shadow comparison, L2 required profile comparison, terminal-credit audits, and route-first stopping. The stage writes D-drive artifacts and never changes PPO loss, network, action space, Hybrid A* pose gate, or candidate generation.

**Tech Stack:** Python runner scripts, JSON config/registry, pytest, existing Stage26 artifact IO helpers, Stage26.8M orchestration, Stage21.3 return/advantage artifacts.

---

### Task 1: Runner Contract Tests

**Files:**
- Create: `tests/test_xunce_stage26_10b_completion_capable_shadow_trial.py`
- Modify: `tests/test_platform_stage_runner.py`

- [ ] Write failing tests for config/schema/registry/default root, boundary rejections, profile derivation, L0 routing, L1 paired configs, required L2, terminal-credit blocker, and per100m short-sighted route.
- [ ] Run `python -m pytest tests/test_xunce_stage26_10b_completion_capable_shadow_trial.py tests/test_platform_stage_runner.py -q --basetemp D:/xunce/pytest/s26_10b_red`; expected: fails because runner/config/registry do not exist.

### Task 2: Stage26.10B Runner And Config

**Files:**
- Create: `scripts/run_xunce_stage26_10b_completion_capable_shadow_trial.py`
- Create: `configs/xunce_stage26_10b_completion_capable_shadow_trial_v1.json`
- Modify: `configs/stage_registry.json`

- [ ] Implement `xunce-stage26-10b-completion-capable-shadow-trial` with default output root `D:/xunce/out/s26_10b`.
- [ ] Write canonical artifacts: `xunce-stage26-10b-summary.json`, `xunce-stage26-10b-routing.json`, `xunce-stage26-10b-manifest.json`, `xunce-stage26-10b-matrix.jsonl`, `xunce-stage26-10b-job-state.jsonl`, `xunce-stage26-10b-report.md`.
- [ ] Generate fixed profiles: `terminal_off_control`, `baseline_v3`, `balanced_high`, `terminal_heavy`, `dead_end_high`; do not overwrite existing Stage26.10A artifacts.
- [ ] Implement route logic for L0 completion reachability, L1 paired shadow, required L2 comparison, terminal-credit repair, short-sighted efficiency repair, eval/binding/safety repair, readiness, and continue.
- [ ] Preserve all boundary flags as false and write source checkpoint identity into summary and manifest.

### Task 3: Integration And Metric Tests

**Files:**
- Modify: `tests/test_xunce_stage26_8m_generalized_resumable_training_pipeline.py`
- Modify: `tests/test_xunce_stage21_3_ppo_batch_validation.py`

- [ ] Add assertions that Stage26.10B derived Stage26.8M configs keep `modifies_ppo_loss=false`, `modifies_network=false`, `modifies_action_space=false`, `modifies_hybrid_astar_pose_gate=false`, `modifies_candidate_generation=false`.
- [ ] Add assertions that Stage21.3 terminal-aware reason codes remain nonblocking and return/advantage audit rows can be summarized by Stage26.10B terminal-credit audit.

### Task 4: Verification And Goal Execution

**Files:**
- Runtime only under `D:/xunce/out/s26_10b`

- [ ] Run `python -m pytest tests/test_xunce_stage26_10b_completion_capable_shadow_trial.py tests/test_xunce_stage26_8m_generalized_resumable_training_pipeline.py tests/test_xunce_stage21_3_ppo_batch_validation.py tests/test_platform_stage_runner.py -q --basetemp D:/xunce/pytest/s26_10b`.
- [ ] Run `git diff --check`.
- [ ] Start one bounded Stage26.10B invocation only after tests pass:
  `python scripts/run_xunce_stage26_10b_completion_capable_shadow_trial.py --config configs/xunce_stage26_10b_completion_capable_shadow_trial_v1.json --output-root D:/xunce/out/s26_10b --repo-root C:/Users/77634/.codex/worktrees/ca49/lunar-path-planning`
- [ ] Stop on any decision route other than `continue_stage26_10b_completion_shadow_trial`; otherwise report current pending job state.

### Task 5: Subagent Review Gates

**Files:**
- No direct edits

- [ ] After implementation, dispatch spec reviewer to check the plan contract, hard boundaries, and route semantics.
- [ ] After spec approval, dispatch code-quality reviewer to check locality, artifact IO, resumability, and test coverage.
- [ ] Before final report, dispatch read-only experiment reviewer to check latest summary/matrix/job-state evidence.

### Acceptance Criteria

- Tests above pass with D-drive pytest basetemp.
- Stage registry includes `xunce-stage26-10b-completion-capable-shadow-trial`.
- Summary/manifest include source checkpoint path, sha256, trained-policy identity, boundary flags, profile ids, L0/L1/L2 status, and route.
- Stage26.10B never publishes a checkpoint, replaces default policy, connects executor, starts canary, or changes PPO/network/action/gate/candidate contracts.
