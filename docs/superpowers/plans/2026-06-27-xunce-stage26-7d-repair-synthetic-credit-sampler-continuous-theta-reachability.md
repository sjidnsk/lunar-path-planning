# Stage26.7D Repair Synthetic Credit Sampler Continuous-Theta Reachability

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` task-by-task. Use review gates before moving between sampler, batch gate, and full-chain execution.

**Goal:** Make the synthetic credit sampler choose a Hybrid A* reachable continuous theta, with auditable behavior point/theta log probabilities.

**Architecture:** Keep reward, network, candidate generation, and Hybrid A* search semantics unchanged. Stage21.1 changes only behavior-action sampling/provenance; Stage21.3 validates the resulting behavior logprob contract; Stage26.7D wraps the repaired Stage26.7C-compatible chain.

**Tech Stack:** Python runner scripts, pytest, JSON/JSONL stage artifacts.

---

### Task 1: Reachability Theta Proposal Contract

**Files:**
- Modify: `scripts/xunce_synthetic_exploration_credit.py`
- Test: `tests/test_xunce_stage26_7d_repair_synthetic_credit_sampler_continuous_theta_reachability.py`

- [x] Add `reachability_theta_proposal_mixture/v1` helper.
- [x] Include reachability probe theta, adjacent probe theta values, and current theta in the proposal set.
- [x] Choose only Hybrid A* reachable proposals and write `old_behavior_theta_log_prob`.
- [x] Preserve `synthetic_terrain_obstacle_proxy/v1` semantics.

### Task 2: Collector And Batch Logprob Binding

**Files:**
- Modify: `scripts/run_xunce_stage21_1_on_policy_ppo_rollout_collector.py`
- Modify: `scripts/run_xunce_stage21_3_ppo_batch_validation.py`
- Test: `tests/test_xunce_stage26_7d_repair_synthetic_credit_sampler_continuous_theta_reachability.py`

- [x] Make synthetic credit behavior select both target index and reachable theta.
- [x] Write `old_policy_theta_log_prob`, `old_behavior_theta_log_prob`, proposal list, selected proposal index, and reachable proposal count.
- [x] Reject behavior theta logprob mismatches in Stage21.3.
- [x] Keep older synthetic credit behavior batches compatible when no reachability theta policy is present.

### Task 3: Stage26.7D Wrapper

**Files:**
- Create: `scripts/run_xunce_stage26_7d_repair_synthetic_credit_sampler_continuous_theta_reachability.py`
- Create: `configs/xunce_stage26_7d_repair_synthetic_credit_sampler_continuous_theta_reachability_v1.json`
- Modify: `configs/stage_registry.json`
- Test: `tests/test_platform_stage_runner.py`

- [x] Require Stage26.7C route `repair_stage26_7_path_efficiency_credit_sampler`.
- [x] Rerun a Stage26.7C-compatible chain with `main_coverable_cells/v1`.
- [x] Summarize trainable rows, target-selected rows, unreachable selected theta count, behavior logprob status, and coverage-per-100m result.
- [x] Treat Hybrid A* path-cost delta as diagnostic only.

### Review Gates

- [x] Review 1: sampler contract and logprob semantics.
- [x] Review 2: collector/Stage21.3 behavior logprob gate.
- [x] Review 3: final runner/tests/docs and full-chain routing.

### Verification

```powershell
python -m pytest tests\test_xunce_stage26_7d_repair_synthetic_credit_sampler_continuous_theta_reachability.py tests\test_xunce_stage26_7c_main_coverable_coverage_efficiency_rerun.py tests\test_xunce_stage26_1_synthetic_terrain_collector_smoke.py tests\test_xunce_stage21_3_ppo_batch_validation.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage26-7d
python -m py_compile scripts\run_xunce_stage26_7d_repair_synthetic_credit_sampler_continuous_theta_reachability.py scripts\run_xunce_stage21_1_on_policy_ppo_rollout_collector.py scripts\run_xunce_stage21_3_ppo_batch_validation.py scripts\xunce_synthetic_exploration_credit.py
python scripts\run_stage.py --stage xunce-stage26-7d-repair-synthetic-credit-sampler-continuous-theta-reachability --dry-run
```
