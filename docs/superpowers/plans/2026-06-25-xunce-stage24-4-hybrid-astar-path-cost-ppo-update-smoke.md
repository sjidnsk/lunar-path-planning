# Stage24.4 Hybrid A* Path Cost PPO Update Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify that the real Stage24.3 PPO batch with `hybrid_astar_pose_path/v1` path cost can be consumed by Stage21.4 tiny PPO update.

**Architecture:** Stage24.4 is a thin wrapper around Stage21.4. It first gates Stage24.3 readiness and audits the selected-action Hybrid A* path-cost contract in `s21_3`, then runs one offline experimental-only Stage21.4 update and verifies loss, gradients, checkpoint reload, and boundary metadata.

**Tech Stack:** Python stage runners, pytest, Stage21.4 tiny PPO update, Stage21.3 PPO batch artifacts.

---

### Task 1: Stage24.4 Wrapper And Config

**Files:**
- Create: `scripts/run_xunce_stage24_4_hybrid_astar_path_cost_ppo_update_smoke.py`
- Create: `configs/xunce_stage24_4_hybrid_astar_path_cost_ppo_update_smoke_v1.json`
- Modify: `configs/stage_registry.json`

- [ ] Add config schema `xunce-stage24-4-hybrid-astar-path-cost-ppo-update-smoke-config/v1`.
- [ ] Require Stage24.3 summary to be passed with route `run_stage24_4_hybrid_astar_path_cost_ppo_update_smoke`.
- [ ] Audit Stage24.3 `s21_3/xunce-stage21-3-ppo-trainable-batch.jsonl` before Stage21.4 runs.
- [ ] Generate `xunce-stage24-4-stage21-4-config.json` and call Stage21.4 with fixed smoke parameters.
- [ ] Register `xunce-stage24-4-hybrid-astar-path-cost-ppo-update-smoke`.

### Task 2: Hybrid Path Batch Audit

**Files:**
- Test: `tests/test_xunce_stage24_4_hybrid_astar_path_cost_ppo_update_smoke.py`

- [ ] Verify selected `action_index` binds to `candidate_viewpoints[action_index]`.
- [ ] Verify `path_cost_sources[action_index] == hybrid_astar_pose_path/v1`.
- [ ] Verify selected Hybrid A* path cost, pose path hash, trajectory kind, legacy grid cost, and grid-vs-hybrid delta match the row-level reward/batch fields.
- [ ] Reject grid-only path cost fallback, default A* replacement, or Ackermann-feasible claims.
- [ ] Verify `xunce_batch` action dimension matches viewpoint count.

### Task 3: Update Smoke Audit

**Files:**
- Runner outputs under `D:\CodexDownloads\lunar-path-planning\stage24_hybrid_astar_pose_planner\outputs\path_feedback_batch_xunce_stage24_4_hybrid_astar_path_cost_ppo_update_smoke_v1`

- [ ] Verify Stage21.4 loss rows are finite.
- [ ] Verify gradient audit has finite pre/post clip norms and component grad norms.
- [ ] Verify experimental checkpoint exists, reloads, is marked experimental-only, and stays under Stage24.4 `s21_4`.
- [ ] Verify all release/default-policy/executor/canary fields remain false/0.
- [ ] Route to `run_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke` only if all audits pass.

### Task 4: Documentation And Verification

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/算法设计与系统架构报告.md`
- Modify: `docs/superpowers/specs/2026-06-16-topology-aware-coverage-policy-network-design.md`

- [ ] Document that Stage24.4 proves update/checkpoint chain readiness only.
- [ ] Run:

```powershell
python -m pytest tests\test_xunce_stage24_4_hybrid_astar_path_cost_ppo_update_smoke.py tests\test_xunce_stage24_3_hybrid_astar_path_cost_ppo_collector_smoke.py tests\test_xunce_stage21_4_tiny_ppo_update_smoke.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage24-4
python -m py_compile scripts\run_xunce_stage24_4_hybrid_astar_path_cost_ppo_update_smoke.py scripts\run_xunce_stage21_4_tiny_ppo_update_smoke.py
python scripts\run_stage.py --stage xunce-stage24-4-hybrid-astar-path-cost-ppo-update-smoke --dry-run
```

- [ ] Run the real Stage24.4 smoke with the default config and output root.

### Assumptions

- Stage24.3 remains the source of truth for collector/reward/batch readiness.
- Stage21.4 is unchanged and only consumes the validated PPO tensors.
- Stage24.4 does not run Stage21.5, publish checkpoints, replace default policy, connect executor, start canary, replace default A*, or claim performance.
