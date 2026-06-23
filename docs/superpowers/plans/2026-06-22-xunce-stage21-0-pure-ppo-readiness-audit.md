# Xunce Stage 21.0 Pure PPO Readiness Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立 Stage 21.0 只读 readiness audit，明确当前仓库距离 Xunce 纯 PPO 主线还缺什么。

**Architecture:** 新增一个只读 runner，静态审查 Xunce full network、high-fidelity rollout、PPO loss/update、canonical reward v3 与 stage registry。Runner 不训练、不采样、不写 checkpoint，只输出 summary/routing/report/manifest，并把下一跳收敛到 Stage 21.1 on-policy PPO collector。

**Tech Stack:** Python stdlib JSON、现有 canonical reward profile loader、stage registry、pytest、py_compile。

---

### Task 1: Stage 21.0 Runner And Config

**Files:**
- Create: `configs/xunce_stage21_0_pure_ppo_readiness_audit_v1.json`
- Create: `scripts/run_xunce_stage21_0_pure_ppo_readiness_audit.py`

- [ ] 新增 config，固定 D 盘输出根、canonical v3 profile、Stage18-20 最近证据根和所有训练/发布/executor/canary boundary false。
- [ ] 新增 runner，读取 config 和 repo root，扫描关键文件中的 Xunce 网络、checkpoint loader、high-fidelity rollout、PPO loss、rollout schema、limited PPO smoke、canonical v3 profile。
- [ ] 输出 `xunce-stage21-0-pure-ppo-readiness-summary.json`、`xunce-stage21-0-capability-audit.json`、`xunce-stage21-0-next-stage-routing.json`、`xunce-stage21-0-report.md`、`xunce-stage21-0-manifest.json`。
- [ ] 若 boundary 字段为 true，hard fail 到 `resolve_stage21_0_pure_ppo_readiness_boundary_rejections`。
- [ ] 正常通过时 route 到 `implement_stage21_1_xunce_on_policy_ppo_rollout_collector`。

### Task 2: Tests And Registry

**Files:**
- Create: `tests/test_xunce_stage21_0_pure_ppo_readiness_audit.py`
- Modify: `configs/stage_registry.json`
- Modify: `tests/test_platform_stage_runner.py`

- [ ] 测试 runner 能识别已有能力、需改进能力、缺失能力。
- [ ] 测试 boundary true 会 hard fail。
- [ ] 测试非 v3 profile 会失败。
- [ ] 注册 `xunce-stage21-0-pure-ppo-readiness-audit`，默认输出到 D 盘 Stage21 根。
- [ ] 更新 stage runner supported set 并验证 dry-run。

### Task 3: Documentation

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/算法设计与系统架构报告.md`
- Modify: `docs/superpowers/specs/2026-06-16-topology-aware-coverage-policy-network-design.md`

- [ ] 写明 Stage 21 新主线：Xunce on-policy rollout -> PPO trainable batch -> coverage-first reward -> tiny PPO update -> post-update eval -> multi-seed pilot。
- [ ] 写明 Stage20/20.1 oracle imitation 降级为诊断，不作为纯 PPO 主线门槛。
- [ ] 写明 Stage 21.0 不训练、不发布、不替换 policy、不连接 executor、不启动 canary。

### Verification

- [ ] `python -m pytest tests\test_xunce_stage21_0_pure_ppo_readiness_audit.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage21-0-readiness`
- [ ] `python -m py_compile scripts\run_xunce_stage21_0_pure_ppo_readiness_audit.py`
- [ ] `python scripts\run_stage.py --stage xunce-stage21-0-pure-ppo-readiness-audit --dry-run`
- [ ] `python scripts\run_xunce_stage21_0_pure_ppo_readiness_audit.py --config configs\xunce_stage21_0_pure_ppo_readiness_audit_v1.json --output-root D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\outputs\path_feedback_batch_xunce_stage21_0_pure_ppo_readiness_audit_v1 --repo-root .`
