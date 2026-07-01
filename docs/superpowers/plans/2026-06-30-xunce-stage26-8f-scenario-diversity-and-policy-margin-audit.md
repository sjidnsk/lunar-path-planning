# Stage26.8F Scenario Diversity And Policy Margin Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 只读诊断 Stage26.8D/8E 中 `main_coverage_per_100m_delta` 长期为 0 的根因。

**Architecture:** Stage26.8F 不推进实验队列，不运行 collector、PPO update 或 trajectory eval。它只读取 Stage26.8D job-state 和已完成 Stage26.3 pre/post artifacts，输出 scenario 多样性、policy margin、candidate opportunity 和 coverage proxy sensitivity 四类审计。

**Tech Stack:** Python runner、JSON/JSONL artifacts、pytest、stage registry。

---

### Task 1: Runner And Audit Contracts

**Files:**
- Create: `scripts/run_xunce_stage26_8f_scenario_diversity_and_policy_margin_audit.py`
- Create: `configs/xunce_stage26_8f_scenario_diversity_and_policy_margin_audit_v1.json`

- [ ] 实现只读 runner，读取 Stage26.8D summary 与 `xunce-stage26-8d-job-state.jsonl`。
- [ ] 只收集 `status=complete` 且 `phase=complete` 的 Stage26.3 roots；pending jobs 写入 source index，不得执行。
- [ ] 读取 completed eval 的 Stage26.3 summary/action/trajectory audit、pre/post inference、episodes、steps。
- [ ] 输出四类审计：scenario diversity、policy margin crossing、candidate opportunity、coverage metric sensitivity。
- [ ] 所有 release/default-policy/executor/canary 字段固定 false/0。

### Task 2: Routing And Reports

**Files:**
- Modify: `configs/stage_registry.json`
- Create: `docs/superpowers/plans/2026-06-30-xunce-stage26-8f-scenario-diversity-and-policy-margin-audit.md`

- [ ] 注册 stage id：`xunce-stage26-8f-scenario-diversity-and-policy-margin-audit`。
- [ ] 若 completed eval 少于阈值，route 到 `continue_stage26_8d_seed_horizon_jobs`。
- [ ] 若 scenario signature 大量重复，route 到 `repair_stage26_synthetic_scenario_diversity`。
- [ ] 若 action 不变且 policy delta 远小于 margin，route 到 `repair_stage26_synthetic_policy_update_signal_strength`。
- [ ] 若 candidate proxy 无差异，route 到 `repair_stage26_synthetic_candidate_generation_or_metric_sensitivity`。

### Task 3: Tests

**Files:**
- Create: `tests/test_xunce_stage26_8f_scenario_diversity_and_policy_margin_audit.py`

- [ ] 构造 fake Stage26.8D root 和 minimal Stage26.3 artifacts。
- [ ] 覆盖 scenario 重复、强绑定缺字段、policy delta 太小、candidate 字段缺失、completed eval 不足、boundary 打开。
- [ ] 验证 stage registry dry-run 可解析。

### Task 4: Verification

- [ ] Run:
  `python -m pytest tests\test_xunce_stage26_8f_scenario_diversity_and_policy_margin_audit.py tests\test_xunce_stage26_8d_resumable_seed_horizon_execution.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage26-8f`
- [ ] Run:
  `python -m py_compile scripts\run_xunce_stage26_8f_scenario_diversity_and_policy_margin_audit.py scripts\run_xunce_stage26_8d_resumable_seed_horizon_execution.py`
- [ ] Run:
  `python scripts\run_stage.py --stage xunce-stage26-8f-scenario-diversity-and-policy-margin-audit --dry-run`

### Acceptance

- Stage26.8F summary 明确回答 0 delta 的主因。
- 8F 不运行 Stage26.1/26.2/26.3，不继续盲跑 H20 或更多 seed。
- 所有 outputs 保留 synthetic hash、platform hash、30 度 slope、coverage/path/action-space lineage。
- 三轮审查无 Critical/Important。
