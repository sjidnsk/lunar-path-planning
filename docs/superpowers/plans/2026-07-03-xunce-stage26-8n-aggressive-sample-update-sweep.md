# Stage26.8N Aggressive Sample + Update Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use the Stage26.8M resumable pipeline to run a more aggressive sample-count and PPO update-strength sweep after Stage26.8I showed policy deltas were too small to cross the argmax boundary.

**Architecture:** Stage26.8N is an orchestration wrapper. It writes a Stage26.8M config with a shared collector, stronger update combos, and controlled eval selection; Stage26.8M continues to execute one recoverable phase per invocation.

**Tech Stack:** Python runners, JSON/JSONL artifacts, pytest, existing Stage26.1/26.2/26.3/26.8M stages.

---

## Task 1: Add Stage26.8M Shared Collector Reuse

**Files:**
- Modify: `scripts/run_xunce_stage26_8m_generalized_resumable_training_pipeline.py`
- Test: `tests/test_xunce_stage26_8m_generalized_resumable_training_pipeline.py`

- [ ] Add opt-in `collector_reuse_policy=by_horizon_seed_scenario_rollout/v1`.
- [ ] Keep default behavior as `none/v1`.
- [ ] Put reused collectors under a short physical path such as `<output_root>/m/c<hash>` while preserving the full logical `job_id` in JSON/JSONL state. This avoids Windows MAX_PATH failures for Stage26.1 `src/...` artifacts.
- [ ] Put per-job phase roots under a short physical path such as `<output_root>/m/j<hash>` and temp configs under `<output_root>/m/g/<hash>`, while preserving logical job identity in state and reports.
- [ ] Ensure update configs point `stage26_1_root` at the shared collector.
- [ ] Ensure `run_next` executes a shared collector only once even when multiple update combos exist.
- [ ] Test that two update combos share one collector artifact and that old default behavior is unchanged.

## Task 2: Add Stage26.8N Wrapper

**Files:**
- Create: `scripts/run_xunce_stage26_8n_aggressive_sample_update_sweep.py`
- Create: `configs/xunce_stage26_8n_aggressive_sample_update_sweep_v1.json`
- Test: `tests/test_xunce_stage26_8n_aggressive_sample_update_sweep.py`

- [ ] Reject Stage26.8I inputs unless `next_required_change=increase_stage26_synthetic_update_strength_or_sample_count`.
- [ ] Generate an aggressive Stage26.8M config with scenario count 6, collector/eval rollout 20, shared collector reuse, and five update combos.
- [ ] Run Stage26.8M one controlled phase per invocation.
- [ ] Stop before update if shared collector has fewer than 100 trainable transitions.
- [ ] Select at most two stable combos for eval, preferring high `parameter_delta_l2` while avoiding policy KL close to 1.5.
- [ ] Write summary, audits, routing, report, and manifest under the D-drive output root.

## Task 3: Registry, Docs, And Validation

**Files:**
- Modify: `configs/stage_registry.json`
- Create: `docs/superpowers/plans/2026-07-03-xunce-stage26-8n-aggressive-sample-update-sweep.md`

- [ ] Register `xunce-stage26-8n-aggressive-sample-update-sweep`.
- [ ] Keep all release/default-policy/executor/canary fields false/0.
- [ ] Run:

```powershell
python -m pytest tests\test_xunce_stage26_8n_aggressive_sample_update_sweep.py tests\test_xunce_stage26_8m_generalized_resumable_training_pipeline.py tests\test_xunce_stage26_8i_diverse_scenario_policy_signal_strength_repair.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage26-8n
python -m py_compile scripts\run_xunce_stage26_8n_aggressive_sample_update_sweep.py scripts\run_xunce_stage26_8m_generalized_resumable_training_pipeline.py
python scripts\run_stage.py --stage xunce-stage26-8n-aggressive-sample-update-sweep --dry-run
```

## Acceptance Criteria

- Stage26.8N uses Stage26.8M as the resumable execution substrate.
- Multiple update combos can share one expensive collector artifact.
- The aggressive sample gate requires at least 100 trainable transitions before update sweep continues.
- Stable update combos are judged by policy KL, finite loss/gradient, checkpoint reload, and boundary metadata.
- AUC, total route length, and Hybrid A* path cost remain diagnostic; `main_coverage_per_100m_delta` is the primary efficiency metric.
