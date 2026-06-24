# Stage22.3 Theta-Aware PPO Collector Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate that the real Stage21.1 -> Stage21.2 -> Stage21.3 PPO data path can produce, reward, and validate `(x,y,theta)` viewpoint actions.

**Architecture:** Stage22.3 is a wrapper smoke runner around existing Stage21.1, Stage21.2, and Stage21.3. It writes internal roots `s21_1`, `s21_2`, and `s21_3`, then audits transition, reward, and batch contracts before routing to Stage22.4.

**Tech Stack:** Python stdlib JSON/JSONL, existing Stage21 runners, pytest, stage registry.

---

### Task 1: Stage22.3 Runner And Config

**Files:**
- Create: `scripts/run_xunce_stage22_3_theta_aware_ppo_collector_smoke.py`
- Create: `configs/xunce_stage22_3_theta_aware_ppo_collector_smoke_v1.json`
- Modify: `configs/stage_registry.json`

- [ ] Add a runner that loads Stage22.2 summary and requires `status=passed` plus route `run_stage22_3_theta_aware_ppo_collector_smoke`.
- [ ] Generate Stage21.1 config with `theta_aware_candidate_viewpoints_enabled=true`, 8 theta bins, 90 degree FOV, 2 scenarios, and 4 rollout steps.
- [ ] Generate Stage21.2 config with `require_theta_aware_reward_contract=true`.
- [ ] Generate Stage21.3 config with both theta viewpoint and theta reward gates enabled.
- [ ] Register `xunce-stage22-3-theta-aware-ppo-collector-smoke` in `stage_registry.json`.

### Task 2: Contract Audit

**Files:**
- Modify: `scripts/run_xunce_stage22_3_theta_aware_ppo_collector_smoke.py`
- Test: `tests/test_xunce_stage22_3_theta_aware_ppo_collector_smoke.py`

- [ ] Audit Stage21.1 trainable transitions for `candidate_viewpoints`, `candidate_theta_deg`, `theta_new_visible_cell_counts`, `theta_coverage_hashes`, `theta_coverage_gain_per_path_costs`, `selected_viewpoint`, and `selected_theta_deg`.
- [ ] Fail if `action_mask`, `sampling_mask`, or `hard_risk_clean_mask` length differs from viewpoint action count.
- [ ] Fail if `action_index` does not bind to `selected_viewpoint`, or if selected viewpoint `(x,y)` differs from selected path target.
- [ ] Fail if Stage21.2 rows lack theta reward provenance, use point-only fallback, have inconsistent coverage delta, or bind to a different selected viewpoint.
- [ ] Fail if Stage21.3 batch rows do not preserve theta viewpoint/reward contracts.

### Task 3: Tests And Docs

**Files:**
- Create: `tests/test_xunce_stage22_3_theta_aware_ppo_collector_smoke.py`
- Modify: `tests/test_platform_stage_runner.py`
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/算法设计与系统架构报告.md`
- Modify: `docs/superpowers/specs/2026-06-16-topology-aware-coverage-policy-network-design.md`

- [ ] Cover pass, Stage22.2 input failure, missing theta transition fields, action/viewpoint mismatch, reward fallback, reward delta mismatch, empty substage rows, and boundary rejection.
- [ ] Add Stage22.3 to platform stage coverage.
- [ ] Document that Stage22.3 does not run PPO update, publish checkpoint, replace default policy, connect executor, start canary, introduce continuous theta, modify network, or change default A*.

### Verification

- [ ] `python -m pytest tests\test_xunce_stage22_3_theta_aware_ppo_collector_smoke.py tests\test_xunce_stage22_2_theta_aware_coverage_reward_contract.py tests\test_xunce_stage21_1_on_policy_ppo_rollout_collector.py tests\test_xunce_stage21_2_coverage_first_ppo_reward_contract.py tests\test_xunce_stage21_3_ppo_batch_validation.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage22-3`
- [ ] `python -m py_compile scripts\run_xunce_stage22_3_theta_aware_ppo_collector_smoke.py scripts\run_xunce_stage21_1_on_policy_ppo_rollout_collector.py scripts\run_xunce_stage21_2_coverage_first_ppo_reward_contract.py scripts\run_xunce_stage21_3_ppo_batch_validation.py`
- [ ] `python scripts\run_stage.py --stage xunce-stage22-3-theta-aware-ppo-collector-smoke --dry-run`
- [ ] `python scripts\run_xunce_stage22_3_theta_aware_ppo_collector_smoke.py --config configs\xunce_stage22_3_theta_aware_ppo_collector_smoke_v1.json --output-root D:\CodexDownloads\lunar-path-planning\stage22_theta_aware_sensor_action_space\outputs\path_feedback_batch_xunce_stage22_3_theta_aware_ppo_collector_smoke_v1 --repo-root .`
