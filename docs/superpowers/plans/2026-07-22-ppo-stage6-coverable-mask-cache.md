# Stage 6 Exact Coverable Mask Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 预计算并只读加载逐位等价的 Standard exact coverable masks，消除每个新 episode 约 129.55s 的重复初始化，再从 Update 49 安全恢复正式训练。

**Architecture:** 新模块负责 canonical key、原子 NPZ codec 和只读 manifest；独立 workflow/script 用 spawn workers 预热全部 1064 场景。`StandardTrainingEnv` 在构建 scenario 后从已绑定 manifest 加载 masks，并通过 `LunarExplorationEnv` 的显式 precomputed 参数注入；Smoke 和其他调用保持现有 process-local compute 路径。

**Tech Stack:** Python 3.12、NumPy、multiprocessing spawn、现有 ArtifactStore/path-security、pytest；不新增依赖。

## Global Constraints

- 官方 mask 算法与三个数组必须逐位等于 `compute_coverage_masks()` 当前实现。
- 正式 runner 缺 cache 或 cache 漂移必须 fail closed，不得静默重算或写 cache。
- cache 不能进入 policy/frontier；不改变 PPO、reward、sensor、planner、candidate、RNG、worker ordering 或 checkpoint schema。
- 固定 formal run、seed、Update49 checkpoint 与 Update50 attempt2/segment7 恢复边界。
- 实现者不 stage、不 commit、不启动正式 runner；Stage 6 Gate 获批前保持一个未提交阶段。
- 所有大型 entry 写 D 盘，不写 Git。

---

### Task 1: Exact cache key、codec 与环境注入

**Files:**
- Create: `src/lunar_exploration_ppo/env/coverage_cache.py`
- Modify: `src/lunar_exploration_ppo/env/env.py`
- Modify: `src/lunar_exploration_ppo/env/standard_training.py`
- Create: `tests/ppo_highres_frontier/test_stage6_coverage_cache.py`

**Interfaces:**
- Produces: `CoverageCacheKey.build(...)`, `serialize_coverage_entry(...) -> bytes`, `load_coverage_entry(...) -> CoverageMasks`, `Stage6CoverageManifest.load(...)`, and optional `precomputed_coverage_masks` injection into `LunarExplorationEnv`.
- Consumes: existing `CoverageMasks`, `compute_coverage_masks`, `ScenarioBundle`, `ArtifactStore.canonical_json_bytes`, secure path reads.

- [ ] Write RED tests for stable key, exact round-trip, corruption/key/path failures, missing formal entry, and no-policy-leakage.
- [ ] Run `D:/conda_envs/lunar-explorer/python.exe -m pytest tests/ppo_highres_frontier/test_stage6_coverage_cache.py -q`; require failures caused by missing cache interfaces.
- [ ] Implement canonical key and self-validating uncompressed NPZ bytes with `allow_pickle=False`; publish via ArtifactStore exclusive atomic write.
- [ ] Add explicit precomputed-mask injection. Validate geometry, mask shapes/dtypes/count and key metadata before use; retain existing compute path when no loader is supplied.
- [ ] Wire `StandardTrainingEnv` to a read-only manifest binding while preserving its existing constructor behavior when no binding is supplied.
- [ ] Run the new suite plus exact Stage1 coverage/env and Stage6 Standard-env focused tests; retain RED/GREEN commands and output in the task report.
- [ ] Build a review package without staging or committing; fresh reviewer must return separate specification and quality verdicts.

### Task 2: 可恢复并行预热与 cache 性能门

**Files:**
- Create: `src/lunar_exploration_ppo/workflows/stage6_coverage_cache.py`
- Create: `scripts/prewarm_ppo_stage6_coverage_cache.py`
- Create: `tests/ppo_highres_frontier/test_stage6_coverage_cache_workflow.py`
- Modify: `src/lunar_exploration_ppo/workflows/__init__.py`

**Interfaces:**
- Produces: `prewarm_stage6_coverage_cache(...) -> CoverageCacheSummary`, canonical `coverage-cache-manifest.json`, resumable parent-owned `progress.jsonl`.
- Consumes: Task 1 codec/key, production catalog/factory, existing resource/path/artifact utilities.

- [ ] Write RED tests for split cardinality 700/150/150/64, spawn-safe worker payload, interrupted resume, duplicate result, corrupt existing entry, worker exception and manifest determinism.
- [ ] Run the focused workflow tests and capture expected RED failures.
- [ ] Implement parent-owned deterministic scheduling and progress writing; workers compute only existing exact masks and return bytes/summary.
- [ ] Enforce D/RSS gates, entry exclusive publish, stable scenario ordering, full manifest and aggregate hash.
- [ ] Add `--dry-run-scenarios`, `--workers`, fixed formal run/cache root and explicit output root; production defaults prewarm all 1064 scenarios.
- [ ] Run a 2-scenario RED/GREEN reproduction, interrupted resume, then a representative cache-hit benchmark with warmup and repeated samples. Hard gate: median <=5s and >=20× versus 129.548168600013s.
- [ ] Fresh task reviewer checks exactness, Windows spawn, path security, resume, manifest and performance evidence; fix/re-review Critical/Important.

### Task 3: Stage 6 production binding与 ordinal6 恢复桥

**Files:**
- Modify: `src/lunar_exploration_ppo/workflows/stage6.py`
- Modify: `src/lunar_exploration_ppo/workflows/stage6_source_repair.py`
- Modify: `src/lunar_exploration_ppo/ppo/standard_training.py`
- Modify: `scripts/create_ppo_stage6_source_repair_amendment.py`
- Modify: `scripts/run_ppo_stage6_standard.py`
- Modify: `tests/ppo_highres_frontier/test_stage6_machine_preflight.py`
- Modify: `tests/ppo_highres_frontier/test_stage6_source_repair.py`
- Modify: `tests/ppo_highres_frontier/test_stage6_workflow.py`
- Modify: `tests/ppo_highres_frontier/test_stage6_training.py`

**Interfaces:**
- Produces: ordinal6 `source-repair-sensor-acceleration.json` extended to bind both exact sensor acceleration and exact coverable cache manifest; summary schema `stage6_source_repair_summary/v6`.
- Consumes: Task 1/2 source hashes, full 1064-entry cache manifest SHA/size, equivalence/performance/review evidence, ordinal5 parent and Update49 cutover.

- [ ] Write RED fixtures for valid ordinal1-5 + Update49 + discarded Update50 pre + sensor/cache evidence.
- [ ] Add cache manifest/source/review bindings to production source set and every Stage6 workflow/training/manifest consumer.
- [ ] Preflight must report `persistent_exact_manifest_read_only/v1`, verify one cold process cache hit, and reject process-local fallback.
- [ ] Fail closed on missing/extra scenario, wrong split count, cache/source/review/parent drift, Update50 partial mismatch, old process alive, repeat publish and post-publish tamper.
- [ ] Run source-repair/workflow/training/preflight focused suites; perform ordinal6 dry-run only.
- [ ] Fresh integrated specification/quality reviewer inspects the full sensor+cache+recovery package; fix/re-review Critical/Important.

### Task 4: 正式预热、最终授权与恢复

**Files and artifacts:**
- Cache root: `D:/xunce/cache/ppo_frontier/s6-standard-single-r1-20260718T220434Z/coverage-v1/`
- Evidence root: `D:/xunce/review/s6-coverable-cache-r1-<timestamp>/`
- Formal run root remains unchanged.

- [ ] Reconfirm no old runner/worker, latest=49, exact checkpoint/manifest/complete/policy hashes, and Update50 has only attempt1 pre.
- [ ] Run all-1064 prewarm with at most 16 spawn workers; monitor D/RSS and resume the same prewarm run after interruption rather than duplicating it.
- [ ] Audit 1064 unique entries and exact split counts; verify entry-set aggregate and manifest hash.
- [ ] Run cold-load/hit benchmark, representative legacy-vs-cache env equivalence, leakage audit and focused regressions.
- [ ] Publish ordinal6 exactly once, create new launch authorization and verify execution identity/source set/cache manifest.
- [ ] Launch the same formal run from Update49; require next resource record segment7 and Update50 attempt2 pre.
- [ ] Reactivate the existing `ppo-stage-6` heartbeat at approximately 30 minutes; no in-turn sleep or duplicate runner.
- [ ] After accepted Update50, compare end-to-end proxy against U47-49 mean 105.05 and median 107.64 minutes without equating cache microbenchmark to PPO performance.

## Self-review result

- All design requirements map to Tasks 1-4.
- No algorithm approximation, async collector, config-hash change or checkpoint-schema migration is included.
- Production runtime is read-only; only the prewarm workflow can create entries.
- Formal cache manifest covers all 1064 Standard scenarios, including validation/test/unseen phases.
