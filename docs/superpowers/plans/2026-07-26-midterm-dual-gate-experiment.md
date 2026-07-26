# Midterm Reduced Dual-Gate Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 `midterm_reduced_w8x3_update80/v1` 缩减规模实验链，使固定 update 80 能在 24 场景/split 的 G1 覆盖率资格试验、三平台 645 次正式调用的 G2 规划计时资格试验、G3 闭环交叉核验和独立 aggregate 复算中形成可审计的中期双门槛证据。

**Architecture:** 在现有 Standard v1、Stage 6 update 80、Path Planner v2 和默认 Stage 1 A* 上增加只读评价适配层。G1 通过显式冻结的 24 场景 schedule 调用现有 8-worker 评价内核；G2 从独立输入 bundle 重建三平台请求并在 worker 内计时；G3 复用 G1 闭环语义并记录与 G2 同构的规划调用字段；所有 runner 只编排、校验和写 artifact，aggregate 只信任逐行结果与 manifest。正式输入、参数集或 lineage 不完整时 fail closed 为 `blocked`。

**Tech Stack:** Python 3.12.13、PyTorch/CUDA FP32、NumPy、现有 Standard v1 spawn evaluator、Path Planner v1/v2、`time.perf_counter_ns()`、JSON/JSONL/NPZ、SHA-256、pytest、PowerShell。

## Global Constraints

- 设计依据：
  `docs/superpowers/specs/2026-07-26-midterm-dual-gate-experiment-design.md`。
- 规模合同固定为 `midterm_reduced_w8x3_update80/v1`；G1 是
  `8 workers × 3 scenes = 24 scenes/split`，不得恢复为 21，也不得运行
  64 后截断为 24。
- G1 checkpoint 固定为：
  - `D:/xunce/out/ppo_frontier/s6-standard-single-r1-20260724T000124Z/s6/checkpoints/seed-20260716/update-00000080/checkpoint.pt`
  - checkpoint SHA-256：
    `35e04c86f9f973af028fb08f0175d42ab45378d09e1f2b96ee6aad6e4c12b5b5`
  - policy-state SHA-256：
    `3123e6adde99be41e3bd5cc2f3843068e5416892395926d759c2c6a243ffd381`
- G1 正式 denominator source 固定为
  `reachable_observable_free_highres_cells/v1`；只有逐单元掩膜与
  `exact_reachable_safe_pose_range_los/v1` 完全一致并且 SHA-256 相同时，
  才记录语义别名。
- G1 Test-Q24、Test-C24、Unseen-24、G3 5+5 清单必须在任何正式结果产生前
  冻结；选择算法不得读取 checkpoint、动作、覆盖率或规划成功率。
- G2 正式调用数固定为 `3 × 43 × 5 = 645`；worker-1 对照、每平台 10 次
  warm-up 和冷启动记录不进入这 645 个正式样本。
- G2 以独立输入为资格前置条件。默认 repo 配置保持 blocked：
  - 每平台独立 primitive labels 不少于 3334；
  - 每平台独立 small-map optimum 不少于 1；
  - 每平台 33 个 Standard、10 个 Kilometer 请求及独立可达性标签；
  - oracle 与 provider 的来源、实现 hash 和审批证据必须分离。
- 当前内置 Hopper Gate5B 参数集的 `evidence_class=test_fixture` 且
  `formal_evidence_eligible=False`。不得把它提升为正式 G2/G3 证据；
  正式运行必须绑定另行批准的 simulation-proxy 参数集。批准前只允许
  diagnostic replay，G2/G3 正式状态必须为 `blocked`。
- 足式、飞跃式始终标记 simulation proxy；synthetic terrain 仍为 proxy，
  不得写成 `physical_obstacle_cells` 或实物平台证据。
- 不发布 checkpoint、不替换 default policy、不连接 executor、不启动
  canary；默认 A* 不被替换，Path Planner v2 只在 G2 和显式接口 replay 中
  使用。
- G3 轮式闭环继续使用 G1 的
  `stage1_path_planner_adapter/v1`，仅添加计时诊断，不把它描述成 Path
  Planner v2 性能；足式和飞跃式接口 replay 才使用各自 v2 provider。
- 正式输出根固定为 `D:/xunce/out/mid_dual/{g1,g2,g3,aggregate}`；输入放
  `D:/xunce/inputs/mid_dual`；pytest basetemp 放
  `D:/xunce/tmp/pytest-mid-dual`。不得把大规模输出写到 C 盘。
- 主线 artifact 必须通过 `scripts/xunce_artifact_io.py` 和
  `scripts/xunce_artifact_paths.py`；不得在 runner 中直接使用
  `Path.read_text/write_text/exists/is_file/open/mkdir` 读写正式 artifact。
- 每个正式 attempt 使用新 run ID；既有 root 不覆盖、不移动、不删除。
  resume 只接受连续、完整且 hash 有效的 phase 前缀。
- 所有中文文件 UTF-8；完成修改后必须用
  `D:/conda_envs/lunar-explorer/python.exe` 显式 UTF-8 读取验证。
- 当前工作区已有大量用户改动，且 G1 依赖的 Stage 6 文件仍有未提交内容。
  实施时不得 reset、clean、checkout 或回退这些内容；每项任务先记录
  allowed-file baseline。除非用户明确批准整合这些既有改动，实施阶段不做
  Git commit，只输出精确 diff 和测试证据。
- “一天完成”仅指正式输入、runner、缓存、Hopper 正式代理参数和硬件均已
  ready 后的 14–18 小时正式运行窗口；本计划的软件实现和外部独立输入生产
  不属于该窗口。

## File Map

### Root repository — create

- `scripts/xunce_mid_dual_contracts.py`
  - 统一规模常量、row schema、统计、bootstrap、门槛与状态路由。
- `scripts/xunce_mid_dual_artifacts.py`
  - run claim、phase 恢复、一次写入 final artifact、manifest 生成与复核。
- `scripts/freeze_xunce_mid_dual_scenarios.py`
  - 只基于静态环境描述量生成 Test-Q24/Test-C24/Unseen-24/G3/replay 清单。
- `scripts/xunce_mid_dual_g2_inputs.py`
  - G2 JSONL/NPZ codec、独立来源审计、三平台栈构建和 readiness audit。
- `scripts/run_xunce_mid_dual_g1_coverage.py`
- `scripts/run_xunce_mid_dual_g2_planning_time.py`
- `scripts/run_xunce_mid_dual_g3_closed_loop.py`
- `scripts/run_xunce_mid_dual_aggregate.py`
- `configs/xunce_mid_dual_scenario_freeze_v1.json`
- `configs/xunce_mid_dual_g1_coverage_v1.json`
- `configs/xunce_mid_dual_g2_planning_time_v1.json`
- `configs/xunce_mid_dual_g3_closed_loop_v1.json`
- `configs/xunce_mid_dual_aggregate_v1.json`
- `src/lunar_exploration_ppo/eval/midterm_reduced.py`
- `tests/test_xunce_mid_dual_contracts.py`
- `tests/test_xunce_mid_dual_scenario_freeze.py`
- `tests/test_xunce_mid_dual_g1_coverage.py`
- `tests/test_xunce_mid_dual_g2_inputs.py`
- `tests/test_xunce_mid_dual_g2_planning_time.py`
- `tests/test_xunce_mid_dual_g3_closed_loop.py`
- `tests/test_xunce_mid_dual_aggregate.py`
- `docs/xunce-midterm-dual-gate-runbook.md`

### Root repository — modify

- `scripts/xunce_artifact_paths.py`
  - 增加 mid-dual canonical artifact names；无 legacy 长文件名。
- `src/lunar_exploration_ppo/eval/standard.py`
  - 增加显式 job schedule 的公共执行入口和结构化 trace 返回值；
    canonical 16/64 API 行为保持不变。
- `src/lunar_exploration_ppo/env/coverage_cache.py`
  - 增加按 scenario ID 读取公开 denominator audit 的只读 API。
- `src/lunar_exploration_ppo/integrations/path_planner_adapter.py`
  - 在不改路线语义的前提下增加 v1 A* 五阶段纳秒计时诊断。
- `tests/ppo_highres_frontier/test_stage6_standard_eval.py`
  - 锁定原 16/64 合同不回归，并验证 24 只能走显式 reduced API。
- `tests/ppo_highres_frontier/test_foundation.py`
  - 锁定 A* 计时诊断不改变既有路线、失败原因和安全结果。

### `path-planner` submodule — create/modify

- Create `path-planner/src/path_planner/v2/formal_request_codec.py`
  - 严格 JSON metadata + NPZ terrain 的 `PlanningRequestV2` codec。
- Create `path-planner/src/path_planner/v2/timing.py`
  - 五阶段 timing row、nearest-rank 所需稳定字段和结果语义 digest。
- Create `path-planner/tests/test_v2_formal_request_codec.py`
- Create `path-planner/tests/test_v2_midterm_timing.py`
- Modify `path-planner/src/path_planner/v2/__init__.py`
  - 只导出 codec/timing 的稳定公共入口。

### Narrow authorized Hopper authority addendum

用户于 2026-07-27 确认一次性授权后，Task 7 增加一个前置的、additive
Hopper implementation-authority 子任务。完整设计与实施写集见：

- `docs/superpowers/specs/2026-07-27-midterm-hopper-internal-simulation-proxy-authority-design-addendum.md`
- `docs/superpowers/plans/2026-07-27-midterm-hopper-internal-simulation-proxy-authority.md`

该补充只允许 path-planner 识别
`hopper_generic_internal_computational_simulation_proxy_midterm_g2g3/v1`
的冻结计算语义；不授予 formal evidence，不改变 Gate5B fixture、默认配置、
192-action 生成、搜索、terrain、L2 safety 或真实硬件边界。Task 7 仍必须用独立
candidate/approval/input/hash 链解析 `formal_evidence_eligible`。

不修改 `plan_v2()`、wheel/legged/hopper provider 搜索语义、L2 safety 语义或
profile 默认值。上述 narrow addendum 仅增加 Hopper authority record/evaluator
及其一致性 seal。G2 的五段计时由 root executor 对“重建请求 → 构建平台栈 →
`plan_v2` → 独立 L2 recheck → 结果序列化”五个互不重叠的墙钟区间测量。

### Narrow authorized G2 independent truth-producer addendum

用户确认的一次性授权同时允许生成项目内部技术独立输入。Task 7 前新增一个隔离的
truth-producer 子项目，完整设计和实施写集见：

- `docs/superpowers/specs/2026-07-27-midterm-g2-independent-truth-producer-design-addendum.md`
- `docs/superpowers/plans/2026-07-27-midterm-g2-independent-truth-producer.md`

Producer 必须在独立源码根/进程中运行，不得 import 或调用本仓库、Path Planner provider、
oracle、L2 或 Task 7。它先冻结 `10002 labels + 3 optima + 129 requests` 的 source
bundle；Task 7 只能在 freeze 后做单向 codec/crosswalk/readiness。独立 reviewer 和
artifact-bound approval 未完成前，candidate 始终 `formal_evidence_eligible=false`。

---

## Task 1: Freeze the Baseline and Add Common Contracts

**Files:**

- Create: `scripts/xunce_mid_dual_contracts.py`
- Create: `tests/test_xunce_mid_dual_contracts.py`

### Step 1.1: Capture the relevant dirty baseline

- [ ] 记录所有 allowed files 的 `git status --short`、root commit、submodule
  commit 和 Python 版本；不得触碰无关 dirty files。
- [ ] 读取并记录 checkpoint 文件存在性和两个 SHA-256，但不加载 CUDA 模型。
- [ ] 记录以下正式 readiness 事实：
  `D:/xunce/out/mid_dual` 当前是否存在、G2 input bundle 是否存在、Hopper
  正式参数集是否存在。缺失是预期 blocked，不是实施测试失败。

### Step 1.2: Write the common-contract RED tests

- [ ] 先写以下精确测试：

```python
def test_scale_profile_freezes_w8x3_update80_contract() -> None: ...
def test_g1_requires_exactly_24_unique_jobs_and_eight_lanes_of_three() -> None: ...
def test_g2_requires_exactly_645_formal_calls() -> None: ...
def test_threshold_boundaries_are_inclusive_at_080_099_1000_2000() -> None: ...
def test_nearest_rank_uses_ceil_qn_one_based_index() -> None: ...
def test_episode_bootstrap_is_fixed_seed_and_uses_2000_resamples() -> None: ...
def test_nonfinite_values_block_instead_of_being_filtered() -> None: ...
def test_reduced_pass_fields_have_no_unqualified_pass_alias() -> None: ...
```

- [ ] 运行 RED：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:/conda_envs/lunar-explorer/python.exe' -m pytest `
  tests/test_xunce_mid_dual_contracts.py `
  -vv -p no:cacheprovider `
  --basetemp 'D:/xunce/tmp/pytest-mid-dual/task1-red'
```

Expected: collection fails because `xunce_mid_dual_contracts` does not exist.

### Step 1.3: Implement exact constants and row contracts

- [ ] Implement frozen constants:

```python
SCALE_PROFILE = "midterm_reduced_w8x3_update80/v1"
G1_EPISODES_PER_FORMAL_SPLIT = 24
G1_LANE_SIZES = (3, 3, 3, 3, 3, 3, 3, 3)
G2_PLATFORMS = ("wheel", "legged", "hopper")
G2_STANDARD_REQUESTS = 33
G2_KILOMETER_REQUESTS = 10
G2_REPEATS = 5
G2_FORMAL_CALLS = 645
BOOTSTRAP_RESAMPLES = 2000
MID_COVERAGE_THRESHOLD = 0.80
FINAL_COVERAGE_THRESHOLD = 0.99
MID_TIME_MS = 2000.0
FINAL_TIME_MS = 1000.0
```

- [ ] Add frozen/slots row dataclasses:
  `CoverageEpisodeRow`, `PlanningCallRow`, `ClosedLoopStepRow`,
  `InterfaceReplayRow`。每个 row 必须包含 `schema_version`,
  `scale_profile`, stable IDs、source hashes 和 finite-number checks。
- [ ] Implement:
  - `nearest_rank(values, q)`;
  - fixed-seed episode bootstrap percentile CI;
  - G1 per-split statistics and 23/24 threshold count;
  - G2 per `(platform, scale, outcome_kind)` statistics;
  - unique-request semantic consensus across five repeats;
  - three-state `passed/failed/blocked` routing。
- [ ] G2 reachable correctness 按 unique request 判定：同一 request 的 5 次
  repeat 都成功且 semantic digest 一致才算 request success。每平台 38 个
  reachable unique requests，因此 `>=0.99` 在该规模下等价于 `38/38`。
- [ ] 运行 GREEN：

```powershell
& 'D:/conda_envs/lunar-explorer/python.exe' -m pytest `
  tests/test_xunce_mid_dual_contracts.py `
  -vv -p no:cacheprovider `
  --basetemp 'D:/xunce/tmp/pytest-mid-dual/task1-green'
```

Expected: all tests pass.

---

## Task 2: Implement Recoverable Canonical Artifacts

**Files:**

- Create: `scripts/xunce_mid_dual_artifacts.py`
- Modify: `scripts/xunce_artifact_paths.py`
- Add tests to: `tests/test_xunce_mid_dual_contracts.py`

### Step 2.1: Add artifact-store RED tests

- [ ] Add exact tests:

```python
def test_run_store_refuses_an_existing_run_root() -> None: ...
def test_resume_accepts_only_a_contiguous_hash_valid_phase_prefix() -> None: ...
def test_incomplete_phase_is_not_merged_into_final_results() -> None: ...
def test_finalize_writes_the_seven_required_canonical_artifacts_once() -> None: ...
def test_manifest_hashes_every_artifact_except_itself() -> None: ...
def test_blocked_preflight_is_terminal_evidence_but_not_pass_evidence() -> None: ...
def test_artifact_module_uses_xunce_io_and_paths_only() -> None: ...
def test_lineage_audit_snapshots_dirty_required_sources_by_sha256() -> None: ...
def test_environment_audit_binds_python_cpu_gpu_threads_and_power_mode() -> None: ...
```

- [ ] RED command:

```powershell
& 'D:/conda_envs/lunar-explorer/python.exe' -m pytest `
  tests/test_xunce_mid_dual_contracts.py -k 'run_store or resume or manifest or artifact' `
  -vv -p no:cacheprovider `
  --basetemp 'D:/xunce/tmp/pytest-mid-dual/task2-red'
```

### Step 2.2: Add canonical artifact names

- [ ] Add these `ArtifactName` values to `xunce_artifact_paths.py`, each with
  no legacy aliases:

```python
MID_DUAL_CONFIG = ArtifactName("mid_dual_config", "config.json")
MID_DUAL_RESULTS = ArtifactName("mid_dual_results", "results.jsonl")
MID_DUAL_SUMMARY = ArtifactName("mid_dual_summary", "summary.json")
MID_DUAL_ROUTING = ArtifactName("mid_dual_routing", "routing.json")
MID_DUAL_MANIFEST = ArtifactName("mid_dual_manifest", "manifest.json")
MID_DUAL_PHASE_STATE = ArtifactName("mid_dual_phase_state", "phase-state.jsonl")
MID_DUAL_REPORT = ArtifactName("mid_dual_report", "report.md")
```

### Step 2.3: Implement phase-safe writes

- [ ] Implement `MidDualRunStore` with:
  - `create_new(run_root, effective_config)`;
  - `load_for_resume(run_root, expected_config_sha256)`;
  - `write_phase_attempt(phase_id, rows, audit)`;
  - `accept_phase(phase_id, attempt_id, row_sha256)`;
  - `capture_lineage(required_source_paths, root_commit, submodule_commit)`;
  - `capture_environment(environment_probe)`;
  - `finalize(summary, routing, report, extra_audits)`;
  - `verify_manifest(run_root)`.
- [ ] Phase rows live under short paths
  `phases/pNN/aNN/results.jsonl`; an incomplete attempt is retained but never
  referenced by accepted `phase-state.jsonl`。
- [ ] `finalize()` concatenates only accepted phase rows in phase order and
  refuses missing/duplicate IDs, config drift, manifest drift or an existing
  final artifact。
- [ ] Blocked preflight writes empty `results.jsonl` and all seven minimum
  artifacts with `formal_evidence_eligible=false`。
- [ ] `capture_lineage()` records root/submodule commit、branch、dirty status、
  original relative path、size and SHA-256。Every required dirty/untracked source
  is copied byte-for-byte through `artifact_io.copy_file()` to a short
  `lineage/sNNNN.bin` name and indexed in `lineage_audit.json`；unrelated dirty
  files are recorded only in the status inventory and are not copied。
- [ ] `capture_environment()` records Windows/version、CPU model/logical count、
  memory、GPU/driver/CUDA、Python executable/version、frozen dependency list、
  Python hash seed、thread variables、worker start method and power mode。Probe
  failure blocks formal execution instead of filling guessed values。
- [ ] Run GREEN and static forbidden-I/O scan:

```powershell
& 'D:/conda_envs/lunar-explorer/python.exe' -m pytest `
  tests/test_xunce_mid_dual_contracts.py `
  -vv -p no:cacheprovider `
  --basetemp 'D:/xunce/tmp/pytest-mid-dual/task2-green'

rg -n 'Path\\.(read_text|write_text|exists|is_file|open|mkdir)\\(' `
  scripts/xunce_mid_dual_artifacts.py
```

Expected: pytest passes; `rg` returns no matches.

---

## Task 3: Freeze Scenario Cohorts and Denominator Evidence

**Execution-source prerequisite:** before a real freeze, complete
`docs/superpowers/plans/2026-07-27-midterm-g1-scenario-source-materialization.md`.
That additive preparer creates only the policy-blind Standard/static/reset and
Stage6 denominator evidence already required by Task 3; it does not change
selection, scale, checkpoint, environment, or evaluation semantics.

**Files:**

- Create: `configs/xunce_mid_dual_scenario_freeze_v1.json`
- Create: `scripts/freeze_xunce_mid_dual_scenarios.py`
- Create: `tests/test_xunce_mid_dual_scenario_freeze.py`
- Modify: `src/lunar_exploration_ppo/env/coverage_cache.py`

### Step 3.1: Write selection and denominator RED tests

- [ ] Add exact tests:

```python
def test_descriptor_extraction_contains_only_policy_independent_fields() -> None: ...
def test_quantile_bins_are_stable_under_input_order_changes() -> None: ...
def test_freeze_selects_disjoint_test_q24_and_test_c24() -> None: ...
def test_freeze_selects_exact_unseen24_and_g3_five_plus_five() -> None: ...
def test_freeze_rejects_coverage_or_planner_result_fields() -> None: ...
def test_same_catalog_and_config_produce_byte_identical_manifest() -> None: ...
def test_denominator_audit_proves_mask_identity_and_nonempty_count() -> None: ...
def test_denominator_hash_drift_blocks_the_whole_manifest() -> None: ...
```

- [ ] RED command:

```powershell
& 'D:/conda_envs/lunar-explorer/python.exe' -m pytest `
  tests/test_xunce_mid_dual_scenario_freeze.py `
  -vv -p no:cacheprovider `
  --basetemp 'D:/xunce/tmp/pytest-mid-dual/task3-red'
```

### Step 3.2: Implement the policy-blind descriptor

- [ ] For every candidate scenario, load only catalog, static truth/cache and
  reset-time environment state. Produce:

```text
slope_p90_deg
hard_obstacle_fraction
start_pose_bin = (x_half, y_half, heading_quadrant)
initial_observed_coverable_fraction
initial_valid_frontier_count
coverable_cell_count
parent_roi
density_profile
```

- [ ] Numeric fields use stable rank tertiles sorted by
  `(value, scenario_id)`。Selection is greedy and deterministic. For each
  candidate, minimize this lexicographic score after hypothetical addition:

```text
(
  maximum_single_factor_bin_count,
  sum_of_squared_single_factor_bin_counts,
  repeated_full_stratum_count,
  parent_roi_count,
  density_profile_count,
  sha256(f"{selection_seed}:{scenario_id}")
)
```

- [ ] Run independently for Test-Q24, remaining Test-C24 and Unseen-24。
  Select G3 5+5 from already selected Q/Unseen with the same score plus a
  frozen static start-to-farthest-candidate distance bin。Select 3 Validation
  dry-run scenes and 3 replay scenes by the same policy-blind method。
- [ ] Reject any descriptor key containing
  `coverage_result`, `final_coverage`, `policy`, `checkpoint`,
  `planner_success`, `runtime` or `reward`。

### Step 3.3: Expose denominator audits

- [ ] Add a read-only `Stage6CoverageManifest.scenario_audit(scenario_id)`
  method that securely loads the bound NPZ entry and returns:
  scenario ID/hash、entry/key/hash、mask SHA-256、cell count、geometry、
  algorithm ID and exact flag。
- [ ] In the freeze script, reconstruct and compare the exact `coverable_mask`
  bytes. Record:

```text
coverage_denominator_source =
  reachable_observable_free_highres_cells/v1
coverage_denominator_algorithm =
  exact_reachable_safe_pose_range_los/v1
semantic_alias_proven = true
```

  only when the loaded mask, count and SHA all match. Any empty mask or drift
  aborts the whole freeze。
- [ ] Write the frozen manifest to
  `D:/xunce/inputs/mid_dual/scenarios/<freeze-id>/manifest.json` and the
  descriptor rows to `descriptors.jsonl` via xunce artifact I/O。
- [ ] GREEN command:

```powershell
& 'D:/conda_envs/lunar-explorer/python.exe' -m pytest `
  tests/test_xunce_mid_dual_scenario_freeze.py `
  tests/ppo_highres_frontier/test_stage6_coverage_cache.py `
  -vv -p no:cacheprovider `
  --basetemp 'D:/xunce/tmp/pytest-mid-dual/task3-green'
```

Expected: all tests pass; no policy/checkpoint is loaded by freeze tests.

---

## Task 4: Add the Explicit 24-Job Standard Evaluation API

**Files:**

- Create: `src/lunar_exploration_ppo/eval/midterm_reduced.py`
- Modify: `src/lunar_exploration_ppo/eval/standard.py`
- Modify: `tests/ppo_highres_frontier/test_stage6_standard_eval.py`
- Create: `tests/test_xunce_mid_dual_g1_coverage.py`

### Step 4.1: Write Standard compatibility RED tests

- [ ] Add exact tests:

```python
def test_canonical_builder_still_rejects_24_episodes() -> None: ...
def test_reduced_builder_accepts_only_explicit_frozen_24_ids() -> None: ...
def test_reduced_partition_is_eight_lanes_of_three() -> None: ...
def test_reduced_builder_cannot_build_64_then_slice() -> None: ...
def test_explicit_job_runner_reuses_standard_env_action_and_safety_contracts() -> None: ...
def test_existing_validation16_test64_unseen64_contracts_are_unchanged() -> None: ...
def test_trace_bundle_contains_episode_decision_and_step_rows_without_writing() -> None: ...
```

- [ ] RED commands:

```powershell
& 'D:/conda_envs/lunar-explorer/python.exe' -m pytest `
  tests/test_xunce_mid_dual_g1_coverage.py `
  tests/ppo_highres_frontier/test_stage6_standard_eval.py -k '24 or canonical' `
  -vv -p no:cacheprovider `
  --basetemp 'D:/xunce/tmp/pytest-mid-dual/task4-red'
```

### Step 4.2: Add an additive public execution seam

- [ ] In `standard.py`, add:

```python
@dataclass(frozen=True, slots=True)
class StandardEvaluationExecution:
    summary: EvaluationSummary
    episode_rows: tuple[dict[str, object], ...]
    decision_rows: tuple[dict[str, object], ...]
    step_rows: tuple[dict[str, object], ...]

def run_standard_evaluation_jobs(
    *,
    catalog: StandardScenarioCatalog,
    jobs: Sequence[StandardEvaluationJob],
    method: str,
    policy: nn.Module | None,
    policy_device: str | torch.device,
    bootstrap_resamples: int,
    bootstrap_seed: int,
    safety_contract: SafetyContract,
    config_sha256: str,
    resource_guard: Callable[[str], None] | None = None,
) -> StandardEvaluationExecution:
    ...
```

- [ ] `run_standard_evaluation()` 继续只接受 canonical counts 并返回原
  `EvaluationSummary`；它内部构建 canonical schedule 后委托新 seam。
- [ ] 新 seam 接受长度 16/24/64，但 24 只允许 test/unseen、8 lanes × 3、
  episode indexes 0..23、显式唯一 scenario IDs。它不负责选择 IDs。
- [ ] `_run_parallel_episode_batch` 仍只执行现有环境、候选、策略动作和状态推进，
  但把原 trace row、decision record 和每步结果复制到内存 trace bundle。
  旧 API 仍按原 `trace_path` 方式写 trace，保证兼容。
- [ ] 每个 step row 增加稳定 join key、pre/post observation SHA、selected
  candidate、theta、planned path、cumulative path length、coverage gain、
  coverage rate、termination and planner diagnostics。不得加入隐藏 truth
  数组。

### Step 4.3: Implement the reduced adapter and update80 loader

- [ ] `midterm_reduced.py` 实现：
  - 严格读取 frozen scenario manifest；
  - 将 manifest rows 转成 `StandardEvaluationJob`；
  - 调用 `run_standard_evaluation_jobs()`；
  - 调用现有 `load_stage4_policy_for_standard()` 加载 update 80，并再次核对
    固定路径、checkpoint hash、policy-state hash；
  - CUDA FP32 only，无 CPU fallback；
  - 从 trace 计算 steps/path-to-80、steps/path-to-99、分母 audit 和 reduced
    split summary。
- [ ] Adapter 不 import PPO trainer/update，不写 checkpoint，也不接受任意
  checkpoint override。
- [ ] Run GREEN:

```powershell
& 'D:/conda_envs/lunar-explorer/python.exe' -m pytest `
  tests/test_xunce_mid_dual_g1_coverage.py `
  tests/ppo_highres_frontier/test_stage6_standard_eval.py `
  -vv -p no:cacheprovider `
  --basetemp 'D:/xunce/tmp/pytest-mid-dual/task4-green'
```

Expected: all existing Standard tests and new 24-job tests pass.

---

## Task 5: Implement the G1 Runner

**Files:**

- Create: `configs/xunce_mid_dual_g1_coverage_v1.json`
- Create: `scripts/run_xunce_mid_dual_g1_coverage.py`
- Modify: `tests/test_xunce_mid_dual_g1_coverage.py`

### Step 5.1: Write G1 runner RED tests

- [ ] Add exact tests:

```python
def test_g1_config_pins_update80_and_rejects_any_override() -> None: ...
def test_g1_dry_run_uses_three_validation_scenes_only() -> None: ...
def test_g1_q24_must_pass_before_unseen24_runs() -> None: ...
def test_g1_test_c24_requires_a_new_repair_lineage_and_explicit_mode() -> None: ...
def test_g1_requires_23_of_24_and_macro_mean_for_both_thresholds() -> None: ...
def test_g1_replay_requires_identical_actions_curve_and_termination() -> None: ...
def test_g1_results_preserve_all_low_coverage_and_failure_rows() -> None: ...
def test_g1_hidden_truth_never_enters_policy_or_candidate_input() -> None: ...
```

- [ ] RED command:

```powershell
& 'D:/conda_envs/lunar-explorer/python.exe' -m pytest `
  tests/test_xunce_mid_dual_g1_coverage.py -k 'g1_' `
  -vv -p no:cacheprovider `
  --basetemp 'D:/xunce/tmp/pytest-mid-dual/task5-red'
```

### Step 5.2: Implement config and phase flow

- [ ] Freeze config schema fields for checkpoint, scenario manifest/hash,
  denominator source, 8 workers, lane sizes, max 128 steps, 0.99 environment
  terminal threshold, bootstrap seed/resamples and output root。
- [ ] CLI:

```text
--config <repo config>
--scenario-manifest <absolute D path>
--run-id <new stable id>
--mode preflight|dry-run|formal|test-c-confirmation
--repair-lineage <absolute path; only test-c-confirmation>
```

- [ ] Formal phases:
  `preflight → validation_dry_run → test_q24 → unseen24 → replay3 →
  recompute → finalize`。
- [ ] If Test-Q24 fails, stop before Unseen-24 and route `failed`; do not
  automatically execute Test-C24。`test-c-confirmation` requires a new code/config
  lineage hash and rejects reuse of the failed run root。
- [ ] Write mixed `results.jsonl` with explicit `row_kind` values
  `coverage_episode`, `decision`, `planner_call`。G1 gate statistics only consume
  `coverage_episode` rows for Test-Q24 and Unseen-24。
- [ ] G1 pass fields are exactly:
  `g1_coverage_80_passed`, `g1_coverage_99_passed` plus status; no bare
  `coverage_passed`。

### Step 5.3: Test CLI and blocked paths

- [ ] Use fixture policy/env adapters; do not load the real CUDA checkpoint in unit
  tests。
- [ ] Run:

```powershell
& 'D:/conda_envs/lunar-explorer/python.exe' -m pytest `
  tests/test_xunce_mid_dual_g1_coverage.py `
  -vv -p no:cacheprovider `
  --basetemp 'D:/xunce/tmp/pytest-mid-dual/task5-green'

& 'D:/conda_envs/lunar-explorer/python.exe' `
  scripts/run_xunce_mid_dual_g1_coverage.py `
  --config configs/xunce_mid_dual_g1_coverage_v1.json `
  --scenario-manifest 'D:/xunce/inputs/mid_dual/missing/manifest.json' `
  --run-id 'implementation-preflight-g1' `
  --mode preflight
```

Expected: tests pass; CLI creates a new blocked preflight root with reason
`scenario_manifest_missing` and never loads CUDA。

---

## Task 6: Add the Path Planner v2 Request Codec and Timing Primitive

**Files in `path-planner` submodule:**

- Create: `src/path_planner/v2/formal_request_codec.py`
- Create: `src/path_planner/v2/timing.py`
- Create: `tests/test_v2_formal_request_codec.py`
- Create: `tests/test_v2_midterm_timing.py`
- Modify: `src/path_planner/v2/__init__.py`

### Step 6.1: Write codec/timing RED tests in the submodule

- [ ] Add exact tests:

```python
def test_request_codec_round_trips_all_three_platform_requests() -> None: ...
def test_request_codec_rejects_pickle_object_dtype_and_noncanonical_json() -> None: ...
def test_request_codec_binds_every_terrain_layer_hash_and_request_sha256() -> None: ...
def test_timed_executor_uses_five_nonoverlapping_perf_counter_ns_intervals() -> None: ...
def test_phase_sum_equals_total_ns_exactly() -> None: ...
def test_timeout_outcome_cannot_encode_a_success_route() -> None: ...
def test_timing_wrapper_does_not_change_plan_v2_semantic_digest() -> None: ...
```

- [ ] RED:

```powershell
Push-Location path-planner
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:/conda_envs/lunar-explorer/python.exe' -m pytest `
  tests/test_v2_formal_request_codec.py `
  tests/test_v2_midterm_timing.py `
  -vv -p no:cacheprovider `
  --basetemp 'D:/xunce/tmp/pytest-mid-dual/task6-red'
Pop-Location
```

### Step 6.2: Implement a data-only codec

- [ ] JSON metadata contains request/platform/pose/objective/budget/timeout/
  accelerator/seed、geometry、provenance and per-layer NPZ hashes。
- [ ] NPZ accepts only exact uncompressed arrays:
  `<f8 elevation/slope/confidence` and exact bool traversable/hard/observed。
  `allow_pickle=False` always。
- [ ] Canonical `request_sha256` hashes canonical JSON plus each canonical layer
  byte chunk in fixed order；reconstruction must produce exact
  `PlanningRequestV2` and exact `TerrainSnapshotV2`。
- [ ] Codec does not serialize provider objects, callables, routes, labels or
  arbitrary imports。

### Step 6.3: Implement the five-stage timing helper

- [ ] `execute_timed_request_v2()` accepts callbacks supplied by the root runner:

```python
decode_request
build_platform_stack
plan_request
revalidate_success_route
assemble_result
clock_ns = time.perf_counter_ns
```

- [ ] It measures these exact sequential intervals:
  `input_validation_ns`, `platform_instantiation_ns`, `search_ns`,
  `complete_route_validation_ns`, `result_assembly_ns`。`total_ns` equals their
  integer sum; queue waiting and input-file read occur before the worker timer。
- [ ] For a success, `revalidate_success_route` must return complete L2 evidence。
  For a structured failure, validation phase records a checked failure contract。
  Any exception becomes a structured internal failure row, not a dropped sample。
- [ ] Run GREEN plus existing v2 tests available in the submodule:

```powershell
Push-Location path-planner
& 'D:/conda_envs/lunar-explorer/python.exe' -m pytest `
  tests/test_v2_formal_request_codec.py `
  tests/test_v2_midterm_timing.py `
  -vv -p no:cacheprovider `
  --basetemp 'D:/xunce/tmp/pytest-mid-dual/task6-green'
Pop-Location
```

Expected: all tests pass; `plan_v2` source and providers are unchanged.

---

## Task 7: Implement G2 Input Readiness and Independent-Source Audit

**Authorized prerequisite:** first complete and verify
`docs/superpowers/plans/2026-07-27-midterm-hopper-internal-simulation-proxy-authority.md`.
The submodule record is implementation support only. Task 7 must still reject
the Gate5B fixture and the unapproved candidate, and may resolve formal
eligibility only from the separately hashed candidate + approval + independent
input chain.

Also complete and independently audit
`docs/superpowers/plans/2026-07-27-midterm-g2-independent-truth-producer.md`.
Task 7 must not generate truth or import the producer as a library; it consumes
only a completed immutable truth bundle and writes a separate one-way execution
bundle/crosswalk.

**Files:**

- Create: `scripts/xunce_mid_dual_g2_inputs.py`
- Create: `tests/test_xunce_mid_dual_g2_inputs.py`
- Create: `configs/xunce_mid_dual_g2_planning_time_v1.json`

### Step 7.1: Write readiness RED tests

- [ ] Add exact tests:

```python
def test_default_config_is_blocked_and_contains_no_formal_input_paths() -> None: ...
def test_input_bundle_requires_3334_primitive_labels_per_platform() -> None: ...
def test_input_bundle_requires_one_small_map_optimum_per_platform() -> None: ...
def test_request_matrix_is_33_standard_plus_10_kilometer_per_platform() -> None: ...
def test_request_matrix_has_exact_reachable_hard_unreachable_counts() -> None: ...
def test_oracle_and_provider_source_identity_must_be_independent() -> None: ...
def test_request_hash_terrain_hash_and_label_join_are_one_to_one() -> None: ...
def test_hopper_test_fixture_is_rejected_as_formal_evidence() -> None: ...
def test_ppo_target_is_optional_for_reduced_contract_but_required_for_gate6_mode() -> None: ...
```

- [ ] RED command:

```powershell
& 'D:/conda_envs/lunar-explorer/python.exe' -m pytest `
  tests/test_xunce_mid_dual_g2_inputs.py `
  -vv -p no:cacheprovider `
  --basetemp 'D:/xunce/tmp/pytest-mid-dual/task7-red'
```

### Step 7.2: Freeze the input bundle schema

- [ ] Bundle root:

```text
D:/xunce/inputs/mid_dual/g2/<input-set-id>/
  manifest.json
  primitive-labels.jsonl
  small-map-optima.jsonl
  requests.jsonl
  source-attestations.json
  terrain/<terrain-sha256>.npz
```

- [ ] `requests.jsonl` rows contain `platform_kind`, `scale`,
  `difficulty_class`, `oracle_reachable`, stable request ID/hash, terrain blob
  hash, provider profile ID, independent source IDs and the canonical request
  metadata required by the submodule codec。
- [ ] Exact unique request counts:
  - Standard: 23 reachable + 7 hard reachable + 3 unreachable；
  - Kilometer: 6 reachable + 2 hard reachable + 2 unreachable；
  - each platform: 43 unique requests。
- [ ] `source-attestations.json` must bind producer ID/revision, implementation
  hash, approval ID and artifact hash。Code checks exact separation and hashes；
  human/organizational independence remains an external prerequisite and is
  reported, not fabricated。
- [ ] Full Gate6 mode additionally requires PPO target rows；reduced G2 does not
  claim full Gate6 and therefore does not require them。

### Step 7.3: Build only approved platform stacks

- [ ] Implement a frozen mapping from `platform_kind` to existing v2 profile,
  provider and L2 recheck APIs。Never accept a Python import path from input data。
- [ ] Wheel and legged stack builders validate `max_traversable_slope_deg=30.0`
  and approved capability IDs。
- [ ] Hopper builder accepts only a parameter-set record whose evidence explicitly
  says `simulation_proxy=True` and `formal_evidence_eligible=True`。The current
  Gate5B test fixture therefore produces blocker
  `approve_midterm_hopper_simulation_proxy_parameter_set`。
- [ ] Run GREEN:

```powershell
& 'D:/conda_envs/lunar-explorer/python.exe' -m pytest `
  tests/test_xunce_mid_dual_g2_inputs.py `
  -vv -p no:cacheprovider `
  --basetemp 'D:/xunce/tmp/pytest-mid-dual/task7-green'
```

---

## Task 8: Implement the G2 Timed Executor and Runner

**Files:**

- Create: `scripts/run_xunce_mid_dual_g2_planning_time.py`
- Create: `tests/test_xunce_mid_dual_g2_planning_time.py`
- Modify: `scripts/xunce_mid_dual_g2_inputs.py`

### Step 8.1: Write execution and aggregation RED tests

- [ ] Add exact tests:

```python
def test_formal_schedule_contains_exactly_645_calls() -> None: ...
def test_each_platform_scale_group_has_165_or_50_samples() -> None: ...
def test_timer_starts_inside_worker_after_queue_wait() -> None: ...
def test_formal_worker_count_is_four_and_diagnostic_worker_count_is_one() -> None: ...
def test_warmup_and_cold_start_rows_never_enter_formal_distribution() -> None: ...
def test_phase_nan_negative_or_sum_drift_blocks_run() -> None: ...
def test_no_call_over_2000ms_and_p95_mean_thresholds_are_per_group() -> None: ...
def test_unreachable_success_and_non_l2_success_fail_correctness_gate() -> None: ...
def test_static_cache_is_allowed_but_route_answer_cache_is_rejected() -> None: ...
def test_worker_one_and_four_have_identical_semantic_digests() -> None: ...
```

- [ ] RED:

```powershell
& 'D:/conda_envs/lunar-explorer/python.exe' -m pytest `
  tests/test_xunce_mid_dual_g2_planning_time.py `
  -vv -p no:cacheprovider `
  --basetemp 'D:/xunce/tmp/pytest-mid-dual/task8-red'
```

### Step 8.2: Implement the frozen schedule

- [ ] For each platform:
  - execute 10 non-formal warm-up requests；
  - record one cold-start call separately；
  - shuffle the 43 request IDs for each repeat using a fixed seed derived from
    `(platform, repeat_index)`；
  - submit formal work to `ThreadPoolExecutor(max_workers=4)`；
  - start `perf_counter_ns` only inside the worker；
  - sort output by `(platform, scale, request_id, repeat_index)` before write。
- [ ] Execute worker-1 diagnostic once per unique request after formal timing or
  in a separately scheduled low-load phase。Its 129 rows have
  `formal_sample=False`。
- [ ] Formal cache may hold immutable terrain/static validation data and must
  record key/hash/hits/lookups。A cache entry containing route primitives,
  PlanningSuccess/Failure, goal-specific answer or request ID is rejected。

### Step 8.3: Implement correctness and timing gates

- [ ] For every `(platform, scale)` all-call distribution compute count、mean、
  median、nearest-rank P95/P99、sample std、min/max、bootstrap CI and threshold
  proportions。Success/failure distributions are reported separately and do not
  replace the all-call gate；each nonempty success or structured-failure subgroup
  must also satisfy the same time threshold，so a slow failure class cannot be
  hidden by the combined distribution。
- [ ] Midterm pass per group:
  `mean<=2000`, `p95<=2000`, every call `<=2000`。
- [ ] Final-threshold reduced pass per group:
  `mean<=1000`, `p95<=1000`, at least 95% `<=1000`, every call `<=2000`。
- [ ] Correctness requires:
  no unreachable success、38/38 reachable unique request successes per platform、
  every success complete L2、every timeout no route、five repeats semantic
  consensus。
- [ ] Retain engineering margin diagnostics:
  Standard P95 `<=250 ms`, Kilometer P95 `<=750 ms`；these are not formal gate
  replacements。

### Step 8.4: Implement CLI and blocked preflight

- [ ] CLI:

```text
--config <repo config>
--input-bundle <absolute D path>
--run-id <new id>
--mode preflight|diagnostic|formal
```

- [ ] G2 formal refuses concurrent heavy-process preflight failures, input drift,
  missing platform approval or existing run root。
- [ ] Run GREEN and a default blocked preflight:

```powershell
& 'D:/conda_envs/lunar-explorer/python.exe' -m pytest `
  tests/test_xunce_mid_dual_g2_inputs.py `
  tests/test_xunce_mid_dual_g2_planning_time.py `
  -vv -p no:cacheprovider `
  --basetemp 'D:/xunce/tmp/pytest-mid-dual/task8-green'

& 'D:/conda_envs/lunar-explorer/python.exe' `
  scripts/run_xunce_mid_dual_g2_planning_time.py `
  --config configs/xunce_mid_dual_g2_planning_time_v1.json `
  --input-bundle 'D:/xunce/inputs/mid_dual/g2/missing' `
  --run-id 'implementation-preflight-g2' `
  --mode preflight
```

Expected: tests pass; CLI records missing independent inputs and Hopper approval
as blockers, with zero formal rows.

---

## Task 9: Add G3 Closed-Loop Timing and Cross-Checks

**Files:**

- Modify: `src/lunar_exploration_ppo/integrations/path_planner_adapter.py`
- Modify: `tests/ppo_highres_frontier/test_foundation.py`
- Create: `configs/xunce_mid_dual_g3_closed_loop_v1.json`
- Create: `scripts/run_xunce_mid_dual_g3_closed_loop.py`
- Create: `tests/test_xunce_mid_dual_g3_closed_loop.py`

### Step 9.1: Write v1 A* timing RED tests

- [ ] Add exact tests:

```python
def test_timing_diagnostics_have_five_nonoverlapping_ns_fields() -> None: ...
def test_timing_diagnostics_sum_to_total_ns() -> None: ...
def test_timing_does_not_change_route_cells_length_theta_or_failure_reason() -> None: ...
def test_early_failure_records_zero_for_unentered_phases() -> None: ...
```

- [ ] Split existing `PathPlannerAdapter.validate()` into private helpers only
  as needed to place timers around:
  input validation、GridSpec/CostGrid instantiation、A* search、complete path
  safety validation、result assembly。Do not change the A* request, neighbor
  policy, corner-cutting, reachability or safety checks。
- [ ] Run:

```powershell
& 'D:/conda_envs/lunar-explorer/python.exe' -m pytest `
  tests/ppo_highres_frontier/test_foundation.py `
  -vv -p no:cacheprovider `
  --basetemp 'D:/xunce/tmp/pytest-mid-dual/task9-adapter'
```

### Step 9.2: Write G3 RED tests

- [ ] Add exact tests:

```python
def test_g3_manifest_is_exactly_five_q24_plus_five_unseen24() -> None: ...
def test_wheel_step_joins_candidate_request_route_feedback_one_to_one() -> None: ...
def test_wheel_gate_requires_ten_of_ten_threshold_successes() -> None: ...
def test_wheel_g1_paired_coverage_delta_mean_is_at_least_minus_001() -> None: ...
def test_every_planner_call_preserves_the_g2_timing_field_contract() -> None: ...
def test_legged_and_hopper_each_have_three_platform_correct_replays() -> None: ...
def test_no_candidate_unreachable_timeout_and_stale_snapshot_fail_closed() -> None: ...
def test_hopper_diagnostic_fixture_cannot_make_g3_formal_pass() -> None: ...
```

### Step 9.3: Implement the G3 runner

- [ ] Consume the frozen G3 scenario IDs, a completed G1 root and a G2 input
  bundle/hash。Do not select scenes from G1/G2 results。
- [ ] Use the update80 reduced adapter for 10 wheel episodes and retain every
  decision/step row。Generate deterministic request IDs from
  `(scenario_id, step_index, pre_snapshot_sha256)`。
- [ ] Verify selected candidate endpoint/theta matches planner route endpoint and
  next-state feedback。Any mismatch, stale snapshot or partial success is a
  structured failure。
- [ ] Wheel pass requires average coverage threshold、10/10 scenes at threshold、
  planner P95 threshold、no `>2s`、no mask/safety/mismatch error and paired G1
  mean delta `>=-0.01`。
- [ ] Run 3 legged + 3 Hopper short v2 replays from frozen G2 requests。They check
  platform/request/result/timing correspondence only and never enter wheel
  coverage statistics。Without formally eligible Hopper stack, G3 formal is
  blocked even if diagnostic replay succeeds。
- [ ] Fixed failure replays use explicit fixture IDs and injected clocks/stale
  versions；they prove fail-closed semantics but do not enter performance
  distributions。
- [ ] Run GREEN:

```powershell
& 'D:/conda_envs/lunar-explorer/python.exe' -m pytest `
  tests/test_xunce_mid_dual_g3_closed_loop.py `
  tests/ppo_highres_frontier/test_foundation.py `
  -vv -p no:cacheprovider `
  --basetemp 'D:/xunce/tmp/pytest-mid-dual/task9-green'
```

---

## Task 10: Implement Independent Aggregate Recalculation

**Files:**

- Create: `configs/xunce_mid_dual_aggregate_v1.json`
- Create: `scripts/run_xunce_mid_dual_aggregate.py`
- Create: `tests/test_xunce_mid_dual_aggregate.py`

### Step 10.1: Write aggregate RED tests

- [ ] Add exact tests:

```python
def test_aggregate_reads_results_and_manifest_not_gate_summary_values() -> None: ...
def test_aggregate_recomputes_g1_g2_and_g3_from_raw_rows() -> None: ...
def test_aggregate_rejects_scale_profile_input_and_code_hash_drift() -> None: ...
def test_aggregate_rejects_missing_duplicate_and_cross_platform_rows() -> None: ...
def test_aggregate_blocks_when_recomputed_and_stored_summaries_differ() -> None: ...
def test_midterm_reduced_truth_table_requires_all_three_gates() -> None: ...
def test_final_threshold_reduced_truth_table_requires_all_three_gates() -> None: ...
def test_blocked_is_not_rendered_as_failed_or_passed() -> None: ...
def test_report_contains_reduced_scale_qualifier_and_exact_sample_counts() -> None: ...
```

- [ ] RED:

```powershell
& 'D:/conda_envs/lunar-explorer/python.exe' -m pytest `
  tests/test_xunce_mid_dual_aggregate.py `
  -vv -p no:cacheprovider `
  --basetemp 'D:/xunce/tmp/pytest-mid-dual/task10-red'
```

### Step 10.2: Implement raw-row recomputation

- [ ] CLI accepts absolute completed G1/G2/G3 roots and a new aggregate run ID。
- [ ] Verify each manifest first, then exact scale profile、config/input/code
  hashes、row schema、unique IDs and formal counts。
- [ ] Recompute:

```text
midterm_reduced_gate_passed =
  g1_coverage_80_passed
  AND g2_all_platforms_2s_passed
  AND g3_midterm_crosscheck_passed

final_threshold_reduced_gate_passed =
  g1_coverage_99_passed
  AND g2_all_platforms_1s_passed
  AND g3_final_crosscheck_passed
```

- [ ] If any gate is blocked, aggregate is blocked and both booleans are false；
  report must say“证据未就绪”，不得改写成指标失败。
- [ ] Generate one-page table with G1 Test/Unseen、G2 six
  platform-scale groups、G3 wheel/interface checks、sample counts and blockers。
- [ ] Run GREEN:

```powershell
& 'D:/conda_envs/lunar-explorer/python.exe' -m pytest `
  tests/test_xunce_mid_dual_aggregate.py `
  tests/test_xunce_mid_dual_contracts.py `
  -vv -p no:cacheprovider `
  --basetemp 'D:/xunce/tmp/pytest-mid-dual/task10-green'
```

---

## Task 11: Add the Runbook and Perform Implementation Verification

**Files:**

- Create: `docs/xunce-midterm-dual-gate-runbook.md`
- Verify all files above

### Step 11.1: Write the operational runbook

- [ ] Document two separate clocks:
  - readiness/implementation clock：runner、input、Hopper approval、cache；
  - formal execution clock：ready 后 14–18h，排期上限 20h。
- [ ] Include exact command order:
  scenario freeze → G1 preflight/dry-run → G2 preflight/diagnostic → formal
  source freeze → G1 formal → G2 formal → G3 formal → aggregate。
- [ ] State that G2 formal timing runs alone with no training/G1/heavy workload。
- [ ] Include recovery rules、Test-C24 branch、blocked/failed distinction and
  prohibited claims。

### Step 11.2: Run focused and regression suites

- [ ] Root focused suite:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'D:/conda_envs/lunar-explorer/python.exe' -m pytest `
  tests/test_xunce_mid_dual_contracts.py `
  tests/test_xunce_mid_dual_scenario_freeze.py `
  tests/test_xunce_mid_dual_g1_coverage.py `
  tests/test_xunce_mid_dual_g2_inputs.py `
  tests/test_xunce_mid_dual_g2_planning_time.py `
  tests/test_xunce_mid_dual_g3_closed_loop.py `
  tests/test_xunce_mid_dual_aggregate.py `
  -vv -p no:cacheprovider `
  --basetemp 'D:/xunce/tmp/pytest-mid-dual/focused'
```

- [ ] Stage 6 compatibility:

```powershell
& 'D:/conda_envs/lunar-explorer/python.exe' -m pytest `
  tests/ppo_highres_frontier/test_stage6_standard_eval.py `
  tests/ppo_highres_frontier/test_stage6_coverage_cache.py `
  tests/ppo_highres_frontier/test_foundation.py `
  -vv -p no:cacheprovider `
  --basetemp 'D:/xunce/tmp/pytest-mid-dual/stage6-regression'
```

- [ ] Path Planner submodule:

```powershell
Push-Location path-planner
& 'D:/conda_envs/lunar-explorer/python.exe' -m pytest `
  tests/test_v2_formal_request_codec.py `
  tests/test_v2_midterm_timing.py `
  -vv -p no:cacheprovider `
  --basetemp 'D:/xunce/tmp/pytest-mid-dual/path-v2'
Pop-Location
```

### Step 11.3: Run non-formal readiness probes

- [ ] Run scenario-freeze dry run into a new D input root and verify byte-stable
  rerun using a second new root。
- [ ] Run G1 preflight and three-scene Validation dry-run only after checkpoint
  and scenario hashes pass。
- [ ] Run G2 blocked preflight with repo default config；expected blockers are
  exact and no formal timing call executes。
- [ ] If an approved G2 bundle later arrives, run diagnostic only，verify
  worker1/4 semantics and cache audit，then require a separate explicit formal
  start。
- [ ] Do not run Test-Q24、Unseen-24、645-call formal G2、G3 formal or aggregate
  during implementation verification。

### Step 11.4: UTF-8, static boundary and diff review

- [ ] Explicit UTF-8 read:

```powershell
& 'D:/conda_envs/lunar-explorer/python.exe' -c `
  "from pathlib import Path; files=[p for p in Path('.').rglob('*') if p.suffix in {'.py','.json','.md'} and ('mid_dual' in p.name or 'midterm-dual' in p.name)]; [p.read_text(encoding='utf-8') for p in files]; print(len(files))"
```

- [ ] Scan for forbidden claims and direct formal artifact I/O:

```powershell
rg -n 'physical_obstacle_cells|replaces_default_policy|publishes_checkpoint|starts_online_canary' `
  scripts/xunce_mid_dual_*.py scripts/run_xunce_mid_dual_*.py `
  configs/xunce_mid_dual_*.json docs/xunce-midterm-dual-gate-runbook.md

rg -n 'Path\\.(read_text|write_text|exists|is_file|open|mkdir)\\(' `
  scripts/xunce_mid_dual_*.py scripts/run_xunce_mid_dual_*.py
```

- [ ] Review the exact allowed-file diff；confirm no formal D output was
  overwritten and no unrelated dirty file was staged。

## Formal Execution Handoff Gate

正式一天窗口只能在以下全部为 true 后开始：

```text
implementation_tests_passed
AND scenario_manifest_frozen
AND update80_checkpoint_hash_verified
AND coverage_cache_hashes_verified
AND independent_primitive_labels_ready_3x3334
AND independent_small_map_optima_ready_3x1
AND formal_requests_ready_3x43
AND independent_source_attestations_approved
AND hopper_formal_simulation_proxy_parameter_set_approved
AND g1_validation_dry_run_passed
AND g2_diagnostic_semantics_and_cache_audit_passed
AND machine_preflight_passed
```

若任一项为 false，handoff 状态是 `blocked`；不得用 Validation 结果、Gate5B
test fixture、自标注数据或较小请求集替代。

## Self-Review Checklist

- [ ] G1 在每个正式 split 是 24，不是 21，也不是 64 后截断。
- [ ] G2 正式样本是 645，warm-up/cold/worker1 不混入。
- [ ] G3 是 10 wheel + 3 legged + 3 hopper，wheel coverage 与 proxy replay
  分开。
- [ ] 80%/99%、2s/1s 均使用 inclusive boundary。
- [ ] G1 Test 和 Unseen 分开判定；G2 平台和规模分开判定。
- [ ] 独立输入与 Hopper 正式参数缺失都产生 blocked，不产生伪 pass。
- [ ] aggregate 只从 raw rows + manifest 复算。
- [ ] 所有 pass 字段保留 `reduced` 限定。
- [ ] 无 checkpoint 发布、default policy 替换、executor 或 canary。
- [ ] 正式执行的一天边界没有包含实现、外部输入生产或失败修复。
