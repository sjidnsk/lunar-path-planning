# Stage26.4 Synthetic Policy Update Signal Strength Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose and repair the Stage26.3 finding that synthetic rock/pit terrain PPO updates barely move action probabilities and do not change selected `(x,y,theta)`.

**Architecture:** Stage26.4 is an orchestration wrapper over Stage26.1, Stage26.2, and Stage26.3. It expands the synthetic collector sample count, runs bounded PPO update/eval combos, and summarizes policy/value balance plus action-signal deltas without changing PPO math, reward targets, the network, Hybrid A*, default A*, or candidate generation.

**Tech Stack:** Python runner scripts, JSON configs/artifacts, pytest, existing Stage21/Stage26 wrappers.

---

### Task 1: Stage26.4 Runner

**Files:**
- Create: `scripts/run_xunce_stage26_4_synthetic_policy_update_signal_strength_repair.py`
- Create: `configs/xunce_stage26_4_synthetic_policy_update_signal_strength_repair_v1.json`
- Modify: `configs/stage_registry.json`

- [x] Read Stage26.3 summary and require route `repair_stage26_synthetic_policy_update_signal_strength`.
- [x] Rerun Stage26.1 with `rollout_steps=8` and `min_trainable_transition_count=16`.
- [x] Run independent Stage26.2 -> Stage26.3 combos under short combo roots.
- [x] Write sweep, gradient, action, credit, recommendation, routing, report, and manifest artifacts.
- [x] Keep all release/default-policy/executor/canary fields false/0.

### Task 2: Stage26.4 Tests

**Files:**
- Create: `tests/test_xunce_stage26_4_synthetic_policy_update_signal_strength_repair.py`

- [x] Test collector-short route.
- [x] Test value-loss dominance route.
- [x] Test tiny probability route.
- [x] Test probability-moved-but-action-unchanged route.
- [x] Test coverage/AUC/path-cost improved route.
- [x] Test invalid Stage26.3 route rejection.

### Task 3: Documentation

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/算法设计与系统架构报告.md`
- Modify: `docs/superpowers/specs/2026-06-16-topology-aware-coverage-policy-network-design.md`

- [x] Document Stage26.4 as bounded offline repair evidence.
- [x] State that synthetic terrain remains a proxy and must not be written as physical obstacle truth.
- [x] State that Stage26.4 does not publish checkpoints, replace default policy, connect executor, or start canary traffic.

### Verification

Run:

```powershell
python -m pytest tests\test_xunce_stage26_4_synthetic_policy_update_signal_strength_repair.py tests\test_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke.py tests\test_xunce_stage26_2_synthetic_terrain_ppo_update_smoke.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage26-4
python -m py_compile scripts\run_xunce_stage26_4_synthetic_policy_update_signal_strength_repair.py scripts\run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke.py scripts\run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke.py scripts\run_xunce_stage21_4_tiny_ppo_update_smoke.py
python scripts\run_stage.py --stage xunce-stage26-4-synthetic-policy-update-signal-strength-repair --dry-run
```
