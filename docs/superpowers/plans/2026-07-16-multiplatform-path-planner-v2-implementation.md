# 多平台能力感知路径规划 v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变 v1 默认行为的前提下，以显式 opt-in 的 `path_planner.v2` 提供 wheel、legged 与 hopper 的能力感知完整路线规划、全路线 L2 安全验证、失败证据、观测投影和正式 benchmark。

**Architecture:** v2 完全隔离在 `path-planner/src/path_planner/v2/`，共享 versioned contract、cell-center fine safety anchor、确定性搜索/验证与 artifact 合同，平台 provider 只实现自身状态、原语、成本和 oracle。根仓库 runner 只负责 Gate 证据与 benchmark 编排；PPO 仅通过离线薄适配层传入已观测 terrain 和目标 `(x, y, theta)`，不改 env、policy、reward、trainer 或 executor。

**Tech Stack:** Python 3.12、dataclasses、Enum、NumPy、pytest、JUnit XML、项目既有 `xunce_artifact_io.py` / `xunce_artifact_paths.py`。

## Global Constraints

- 工作树固定为 `D:/codex/worktrees/multiplatform-path-planner-v2`，分支固定为 `codex/multiplatform-path-planner-v2`，基线固定为 `b635740ee021258ef31811ec87c60add839fc5f9`。
- 保护 `C:/Users/77634/.codex/worktrees/ca49/lunar-path-planning` Stage6 dirty 工作树：不读取其文件、不修改、不运行命令；只能在 Git 元数据中把它登记为受保护外部工作树，不能声称已验证其前后内容不变。
- v1 `AStarPlanner`、CLI、`PathPlannerAdapter` 和默认 policy 行为保持默认；v2 只能通过 `path_planner.v2` 或显式 v2 adapter 选择。
- 不发布 checkpoint、不替换 default policy、不连接 executor、不启动 canary；所有 Gate artifact 固定写 `publishes_checkpoint=false`、`replaces_default_policy=false`、`connects_real_executor=false`、`starts_online_canary=false`。
- 不修改 PPO 算法、网络、reward 或训练语义；Gate 6 只提供离线 target-to-request adapter 和 benchmark。
- synthetic terrain 只能标记 `synthetic_terrain_obstacle_proxy/v1`，不得写成或宣称 `physical_obstacle_cells`。
- wheel 保持 `max_traversable_slope_deg=30.0`；Hybrid A* 仍为 opt-in 且 `ackermann_feasible_claimed=false`。
- v2 坐标冻结为 cell-center：`origin + (index + 0.5) * resolution`；不得复用 v1 `GridSpec.cell_to_world()` 的格点坐标作为 v2 cell center。
- fine safety anchor 分辨率固定 `0.5m`；unknown 接触、落足、起跳、落区或弹道包络一律 fail closed。
- 成功结果必须是从请求起点到请求目标的完整 typed route，且整条路线 L2 复验通过；部分路径、L0/L1 候选或只验证终点不得成功。
- 资源默认权重 `energy=0.5`、`time=0.5`；探索候选资源上限固定为最小可行资源成本的 `1.20`。
- 同输入、版本、profile、seed 的结构化决策字段必须字节稳定；候选、worker、cache 遍历全部使用稳定 key。
- 单请求硬超时 `2s`；Standard p95 目标 `<=250ms`，Kilometer p95 目标 `<=750ms`。
- 正式安全审计每平台至少 `10,000` primitive；Standard 每平台至少 100 episodes，Kilometer 每平台至少 30 episodes。
- Gate artifact 根固定 `D:/xunce/out/path_v2/g0` 至 `g6`；canonical 文件至少为 `config.json`、`summary.json`、`routing.json`、`manifest.json`、`report.md`、`results.jsonl` 或 `metrics.jsonl`、`phase-state.jsonl`、`review.json`。
- runner 的 artifact 读写必须走 `scripts/xunce_artifact_io.py`；新路径超过 180 字符 warning，达到 240 字符 fail closed。
- PPO Stage1 外部基线只指 `tests/ppo_highres_frontier/test_stage1_smoke_env.py` 的精确 13 项 allowlist；后续 Gate 的失败集合不得新增 nodeid、error 或 skip。
- Gate 5 未冻结的 hopper 包络/起跳高度/净空/落地 footprint/停止/能量参数不得擅自补成实机能力；缺值时实现必须返回稳定 `unsupported_capability` 并在 `routing.json` 标 blocker。
- Gate 6 缺少独立 oracle labels、小图 optimum、正式 Standard/Kilometer schedule 或 PPO target fixture 时，runner 必须输出明确 blocker，不能用自标注 fixture 宣称正式通过。

---

## File Structure

### Standalone v2 package

- `path-planner/src/path_planner/v2/contracts.py`：请求、结果、失败分类、typed route、成本、telemetry 与 validation evidence。
- `path-planner/src/path_planner/v2/serialization.py`：finite-only canonical JSON 与 SHA-256 identity。
- `path-planner/src/path_planner/v2/terrain.py`：cell-center geometry、`TerrainSnapshotV2` 与共享 `FineSafetyAnchorV2`。
- `path-planner/src/path_planner/v2/profiles.py`：版本化 profile 与 registry。
- `path-planner/src/path_planner/v2/providers/base.py`：平台 provider Protocol。
- `path-planner/src/path_planner/v2/search.py`：fine-only 安全核心、稳定队列和 2s deadline。
- `path-planner/src/path_planner/v2/validation.py`：L0/L1/L2 与整条路线复验。
- `path-planner/src/path_planner/v2/cache.py`：完整 key 的 fail-closed hash cache。
- `path-planner/src/path_planner/v2/hierarchy.py`：`r/2r/4r` conservative hint，不拥有最终安全裁决权。
- `path-planner/src/path_planner/v2/api.py`：显式 `plan_v2()` 调度，不暴露为 v1 默认。
- `path-planner/src/path_planner/v2/providers/wheel.py`：现有 Hybrid A* 的 typed wheel adapter。
- `path-planner/src/path_planner/v2/geometry.py`：平台无关 supercover、旋转矩形、凸包与扫掠 helper。
- `path-planner/src/path_planner/v2/oracles/legged.py`、`providers/legged.py`：四足静态爬行 proxy。
- `path-planner/src/path_planner/v2/ballistics.py`、`oracles/hopper.py`、`providers/hopper.py`：月面弹道 proxy。
- `path-planner/src/path_planner/v2/observation.py`：observed-only FOV/LOS observation projection。
- `path-planner/src/path_planner/v2/adapters/ppo_target.py`：PPO target 到 v2 request 的离线薄适配。
- `path-planner/src/path_planner/v2/benchmark.py`、`benchmark_fixtures.py`：指标聚合、独立 label/fixture 输入合同。

### Gate orchestration

- `scripts/path_v2_gate_artifacts.py`：所有 Gate 的 canonical artifact、hash manifest、phase state 和边界字段。
- `scripts/run_xunce_path_v2_g0_baseline_and_isolation.py`：Gate 0 测试与精确失败集合审计。
- `scripts/run_xunce_path_v2_gate_benchmark.py`：Gate 1–6 的可恢复审计/benchmark 编排。
- `configs/xunce_path_v2_g0_baseline_and_isolation_v1.json` 与 `configs/xunce_path_v2_gate{1..6}_*.json`：机器配置。
- `tests/test_xunce_path_v2_g0_baseline_and_isolation.py` 与 `tests/test_xunce_path_v2_gate_benchmark.py`：runner 与 artifact 合同。
- `docs/xunce-stage-documentation-index.md`：path-v2 spec、plan 与 D 盘 evidence 路由。

---

### Task 0: Gate 0 — 基线、隔离与证据冻结

**Files:**
- Create: `scripts/path_v2_gate_artifacts.py`
- Create: `scripts/run_xunce_path_v2_g0_baseline_and_isolation.py`
- Create: `configs/xunce_path_v2_g0_baseline_and_isolation_v1.json`
- Create: `tests/test_xunce_path_v2_g0_baseline_and_isolation.py`
- Modify: `configs/stage_registry.json`
- Modify: `docs/xunce-stage-documentation-index.md`
- Track: `docs/superpowers/specs/2026-07-16-multiplatform-path-planner-v2-design.md`
- Track: `docs/superpowers/plans/2026-07-16-multiplatform-path-planner-v2-implementation.md`

**Interfaces:**
- Consumes: two isolated pytest commands and their JUnit XML.
- Produces: `run_gate0(config_path: Path, output_root: Path, repo_root: Path, execute_tests: bool = True) -> dict[str, Any]` and the complete Gate 0 artifact set.

- [ ] **Step 1: Write failing Gate 0 runner tests**

```python
def test_gate0_accepts_green_v1_and_exact_external_allowlist(tmp_path: Path) -> None:
    summary = runner.run_gate0(
        config_path=_config(tmp_path),
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
        execute_tests=False,
    )
    assert summary["status"] == "passed"
    assert summary["path_planner"] == {"passed": 156, "skipped": 17, "failed": 0, "errors": 0}
    assert summary["ppo_stage1"]["known_failure_count"] == 13
    assert summary["ppo_stage1"]["unexpected_failure_nodeids"] == []
    assert summary["replaces_default_policy"] is False

def test_gate0_rejects_one_extra_ppo_failure(tmp_path: Path) -> None:
    config_path = _config(tmp_path, extra_failure_nodeid="tests/example.py::test_extra")
    summary = runner.run_gate0(
        config_path=config_path,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
        execute_tests=False,
    )
    assert summary["status"] == "failed"
    assert summary["next_required_change"] == "restore_exact_ppo_stage1_external_baseline"
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `D:/conda_envs/lunar-explorer/python.exe -m pytest -q tests/test_xunce_path_v2_g0_baseline_and_isolation.py`

Expected: FAIL because the runner/config do not exist.

- [ ] **Step 3: Implement exact JUnit, Git, import-origin and boundary audits**

```python
@dataclass(frozen=True)
class JUnitSummary:
    tests: int
    passed: int
    failures: int
    errors: int
    skipped: int
    failed_nodeids: Sequence[str]

PUBLIC_FUNCTIONS = (
    "parse_junit(path: Path) -> JUnitSummary",
    "audit_import_origins(python: Path, repo_root: Path) -> dict[str, Any]",
    "audit_git_identity(repo_root: Path, expected_branch: str, expected_base_commit: str) -> dict[str, Any]",
    "run_gate0(config_path: Path, output_root: Path, repo_root: Path, execute_tests: bool = True) -> dict[str, Any]",
)
```

`execute_tests=True` 必须设置 `PYTHONNOUSERSITE=1`、`PYTHONDONTWRITEBYTECODE=1`、`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`、D 盘 `TEMP/TMP/MPLCONFIGDIR` 和显式 D 工作树 `PYTHONPATH`。PPO pytest 返回码 1 只有在 `errors=0`、`skipped=0` 且 nodeid 集合精确等于 allowlist 时才是通过的外部基线。

Git 身份检查必须验证当前分支、linked-worktree、`git merge-base <expected_base_commit> HEAD == <expected_base_commit>` 和运行时 clean tree；不得要求证据提交后的 `HEAD` 仍等于起始基线提交。

- [ ] **Step 4: Write all canonical artifacts with hashes**

`path_v2_gate_artifacts.py` 必须提供：

```python
BOUNDARY_FIELDS = {
    "publishes_checkpoint": False,
    "replaces_default_policy": False,
    "connects_real_executor": False,
    "starts_online_canary": False,
}

def write_gate_artifacts(
    *, output_root: Path, config: dict[str, Any], summary: dict[str, Any],
    routing: dict[str, Any], rows: list[dict[str, Any]],
    phases: list[dict[str, Any]], review: dict[str, Any], report: str,
) -> dict[str, Any]:
    artifact_io.make_dirs(output_root)
    artifact_io.write_json(output_root / "config.json", config)
    artifact_io.write_json(output_root / "summary.json", summary)
    artifact_io.write_json(output_root / "routing.json", routing)
    artifact_io.write_jsonl(output_root / "results.jsonl", rows)
    artifact_io.write_jsonl(output_root / "phase-state.jsonl", phases)
    artifact_io.write_json(output_root / "review.json", review)
    artifact_io.write_text(output_root / "report.md", report)
    manifest = build_manifest_without_self_hash(output_root)
    artifact_io.write_json(output_root / "manifest.json", manifest)
    return manifest
```

Manifest 对除自身外的每个 artifact 保存相对路径、SHA-256 和 size；不得产生自引用 hash。

- [ ] **Step 5: Register the stage and documentation route**

注册：

```json
"xunce-path-v2-g0-baseline-and-isolation": {
  "script": "scripts/run_xunce_path_v2_g0_baseline_and_isolation.py",
  "default_config": "configs/xunce_path_v2_g0_baseline_and_isolation_v1.json",
  "default_output_root": "D:/xunce/out/path_v2/g0",
  "args": ["--config", "{config}", "--output-root", "{output_root}", "--repo-root", "{repo_root}"]
}
```

- [ ] **Step 6: Run focused tests and registry dry-run**

Run:

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest -q tests/test_xunce_path_v2_g0_baseline_and_isolation.py
D:/conda_envs/lunar-explorer/python.exe scripts/run_stage.py --stage xunce-path-v2-g0-baseline-and-isolation --dry-run
```

Expected: focused tests PASS；dry-run 解析到 `D:/xunce/out/path_v2/g0`。

- [ ] **Step 7: Commit Gate 0 implementation, then generate clean-tree evidence**

```powershell
git add docs/superpowers/specs/2026-07-16-multiplatform-path-planner-v2-design.md docs/superpowers/plans/2026-07-16-multiplatform-path-planner-v2-implementation.md docs/xunce-stage-documentation-index.md scripts/path_v2_gate_artifacts.py scripts/run_xunce_path_v2_g0_baseline_and_isolation.py configs/xunce_path_v2_g0_baseline_and_isolation_v1.json configs/stage_registry.json tests/test_xunce_path_v2_g0_baseline_and_isolation.py
git commit -m "build: freeze path planner v2 gate0 evidence"
D:/conda_envs/lunar-explorer/python.exe scripts/run_stage.py --stage xunce-path-v2-g0-baseline-and-isolation
```

Expected: `summary.status=passed`，v1 为 156 passed/17 optional skipped，PPO Stage1 精确 13 known external failures，无新增失败。

---

### Task 1: Gate 1A — Versioned public contracts and deterministic serialization

**Files:**
- Create: `path-planner/src/path_planner/v2/__init__.py`
- Create: `path-planner/src/path_planner/v2/contracts.py`
- Create: `path-planner/src/path_planner/v2/serialization.py`
- Create: `path-planner/tests/test_v2_contracts.py`
- Create: `path-planner/tests/test_v2_serialization.py`

**Interfaces:**
- Produces: `PlanningRequestV2`, `PlanningSuccessV2`, `PlanningFailureV2`, `PlanningOutcomeV2`, `FailureCategoryV2`, `PoseStateV2`, `TypedRouteV2`, `ValidationEvidenceV2`, `canonical_json_bytes()`。

- [ ] **Step 1: Write RED tests for schema, finite values and stable bytes**

```python
assert FailureCategoryV2.values() == (
    "invalid_request", "unsupported_capability", "unsafe_start", "unsafe_goal",
    "goal_pose_unreachable", "no_complete_route", "validation_failed",
    "resource_limit", "timeout", "internal_error",
)
assert canonical_json_bytes(success) == canonical_json_bytes(success)
with pytest.raises(ValueError, match="finite"):
    PoseStateV2(float("nan"), 0.0, 0.0)
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `cd path-planner; D:/conda_envs/lunar-explorer/python.exe -m pytest -q tests/test_v2_contracts.py tests/test_v2_serialization.py`

- [ ] **Step 3: Implement frozen dataclasses and canonical serialization**

```python
class PlatformKindV2(str, Enum):
    WHEEL = "wheel"
    LEGGED = "legged"
    HOPPER = "hopper"

@dataclass(frozen=True, slots=True)
class PlanningRequestV2:
    request_id: str
    platform_profile_id: str
    start_state: PoseStateV2
    goal_state: PoseStateV2
    terrain_snapshot: TerrainSnapshotV2
    objective_profile: ObjectiveProfileV2
    resource_budget: ResourceBudgetV2
    timeout_s: float
    accelerator_policy: AcceleratorPolicyV2
    determinism_seed: int

PlanningOutcomeV2 = PlanningSuccessV2 | PlanningFailureV2
```

Schema version 固定 `path-planner-v2-planning/v1`；mapping key 排序，tuple 保序，拒绝 NaN/inf。

- [ ] **Step 4: Run focused tests and commit**

Expected: PASS。

Commit: `feat: add path planner v2 public contracts`

---

### Task 2: Gate 1B — Cell-center terrain snapshot and fine safety anchor

**Files:**
- Create: `path-planner/src/path_planner/v2/terrain.py`
- Create: `path-planner/tests/test_v2_terrain.py`
- Create: `path-planner/tests/test_v2_fine_safety_anchor.py`

**Interfaces:**
- Produces: `FineGridGeometryV2`, `TerrainSnapshotV2`, `FineSafetyAnchorV2`, `SafetyQueryV2`, `snapshot_hash`。

- [ ] **Step 1: Write RED boundary/property tests**

```python
assert geometry.cell_center(Cell(0, 0)) == WorldPoint(0.25, 0.25)
assert geometry.world_to_cell(WorldPoint(0.25, 0.25)) == Cell(0, 0)
assert anchor.query(Cell(1, 1)).reason_code == "terrain_unknown"
assert anchor.query(Cell(2, 2)).reason_code == "terrain_hard_obstacle"
assert anchor.query(Cell(3, 3), max_slope_deg=30.0).reason_code == "terrain_slope_exceeded"
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `cd path-planner; D:/conda_envs/lunar-explorer/python.exe -m pytest -q tests/test_v2_terrain.py tests/test_v2_fine_safety_anchor.py`

- [ ] **Step 3: Implement immutable 0.5m snapshot and fail-closed queries**

```python
@dataclass(frozen=True, slots=True)
class TerrainSnapshotV2:
    geometry: FineGridGeometryV2
    elevation_m: np.ndarray
    slope_deg: np.ndarray
    traversable_mask: np.ndarray
    hard_obstacle_mask: np.ndarray
    observed_mask: np.ndarray
    confidence: np.ndarray
    provenance: TerrainProvenanceV2

class FineSafetyAnchorV2:
    query: Callable[[Cell, float], SafetyQueryV2]
    validate_cells: Callable[[Iterable[Cell], float], ValidationEvidenceV2]
```

所有数组 shape 必须一致、只读 copy；synthetic provenance 必须保留 `physical_obstacle_cells_written=False`。

- [ ] **Step 4: Run focused tests and commit**

Commit: `feat: add path planner v2 fine safety anchor`

---

### Task 3: Gate 1C — Profiles, provider boundary, API opt-in and Gate 1 evidence

**Files:**
- Create: `path-planner/src/path_planner/v2/profiles.py`
- Create: `path-planner/src/path_planner/v2/providers/__init__.py`
- Create: `path-planner/src/path_planner/v2/providers/base.py`
- Create: `path-planner/src/path_planner/v2/api.py`
- Create: `path-planner/tests/test_v2_profiles.py`
- Create: `path-planner/tests/test_v2_api.py`
- Create: `configs/xunce_path_v2_gate1_contract_v1.json`
- Create/Modify: `scripts/run_xunce_path_v2_gate_benchmark.py`
- Create: `tests/test_xunce_path_v2_gate_benchmark.py`
- Modify: `configs/stage_registry.json`

**Interfaces:**
- Produces: `PlatformProfileRegistryV2`, `PrimitiveProviderV2`, `plan_v2(request, *, registry, providers) -> PlanningOutcomeV2`。

- [ ] **Step 1: Write RED tests for explicit profile, single platform and v1 isolation**

```python
assert plan_v2(request_with_unknown_profile).category is FailureCategoryV2.UNSUPPORTED_CAPABILITY
assert "plan_v2" not in path_planner.__dict__
assert AStarPlanner().plan(grid, request).to_route_dict(spec)["schema_version"] == "path-planner-route/v1"
```

- [ ] **Step 2: Implement registry and provider Protocol**

```python
class PrimitiveProviderV2(Protocol):
    profile: PlatformProfileV2
    plan: Callable[[PlanningRequestV2, FineSafetyAnchorV2], PlanningOutcomeV2]

PlanV2Fn = Callable[[PlanningRequestV2, PlatformProfileRegistryV2, Mapping[str, PrimitiveProviderV2]], PlanningOutcomeV2]
```

API 先对 request/profile/terrain/start/goal 做验证；provider 未注册返回稳定 `unsupported_capability`，不 fallback v1。

- [ ] **Step 3: Add Gate 1 runner config and registry entry**

Gate 1 runner 执行 contract/terrain/API tests、v1 regression 和字节稳定重复，输出 `D:/xunce/out/path_v2/g1` 八件 artifact。

- [ ] **Step 4: Run Gate 1 focused and v1 regression tests**

```powershell
cd path-planner
D:/conda_envs/lunar-explorer/python.exe -m pytest -q tests/test_v2_contracts.py tests/test_v2_serialization.py tests/test_v2_terrain.py tests/test_v2_fine_safety_anchor.py tests/test_v2_profiles.py tests/test_v2_api.py
D:/conda_envs/lunar-explorer/python.exe -m pytest -q tests
```

Expected: v2 focused PASS；v1 仍为 156 passed/17 optional skipped。

- [ ] **Step 5: Commit and run Gate 1 evidence**

Commit: `feat: establish path planner v2 safety contracts`

Expected route: `implement_path_v2_wheel_provider`。

---

### Task 4A: Gate 2A — Public Hybrid replay, rejection hook and runtime contracts

**Files:**
- Modify: `path-planner/src/path_planner/core/models.py`
- Modify: `path-planner/src/path_planner/search/hybrid_astar.py`
- Modify: `path-planner/src/path_planner/search/__init__.py`
- Modify: `path-planner/src/path_planner/v2/contracts.py`
- Modify: `path-planner/src/path_planner/v2/profiles.py`
- Create: `path-planner/src/path_planner/v2/runtime.py`
- Modify: `path-planner/src/path_planner/v2/providers/base.py`
- Modify: `path-planner/src/path_planner/v2/api.py`
- Modify: `path-planner/src/path_planner/v2/__init__.py`
- Modify: `path-planner/tests/test_hybrid_astar.py`
- Modify: `path-planner/tests/test_v2_contracts.py`
- Modify: `path-planner/tests/test_v2_profiles.py`
- Modify: `path-planner/tests/test_v2_api.py`

**Interfaces:**
- Produces: public `PoseTransition`, `PoseSearchAudit`, `replay_motion_primitive()` and opt-in pose/transition validators on `HybridAStarPlanner.plan()`.
- Produces: `PlanningDeadlineV2`; provider execution receives the same absolute monotonic deadline created by `plan_v2()`.

- [ ] **Step 1: Write RED compatibility and rejection-continuation tests**

Public replay must include exact start/end samples and match the existing integration math. A rejected transition must not enter dominance/open state, and another transition/path must remain searchable. With validators omitted, v1 route/control/serialization semantics remain unchanged. Deadline expiry during grid heuristic preprocessing or search returns `FailureReason.TIMEOUT` with an empty route and truthful audit.

- [ ] **Step 2: Freeze goal and objective contracts**

`PlatformProfileV2` adds explicit `goal_position_tolerance_m` and `goal_heading_tolerance_rad`, both defaulting to zero. `plan_v2()` independently checks Euclidean position and wrap-safe heading without snapping the provider endpoint. `ObjectiveProfileV2` defaults to `energy_weight=0.5` and `time_weight=0.5`, with distance/risk zero, and rejects the all-zero objective.

The effective deadline is `min(request.timeout_s, 2.0)` from API entry. Late success is replaced by typed timeout failure. Existing v1 entry points and `DEFAULT_PLATFORM_KEY="yutu2"` remain unchanged.

- [ ] **Step 3: Run focused contracts and v1 Hybrid regression**

Run: `cd path-planner; D:/conda_envs/lunar-explorer/python.exe -m pytest -q tests/test_hybrid_astar.py tests/test_v2_contracts.py tests/test_v2_profiles.py tests/test_v2_api.py tests/test_astar.py`

- [ ] **Step 4: Commit**

Commit: `feat: add safe Hybrid transition hooks for path planner v2`

---

### Task 4B: Gate 2A — Typed wheel route and full-route L2 validation

**Files:**
- Create: `path-planner/src/path_planner/v2/providers/wheel.py`
- Create: `path-planner/src/path_planner/v2/geometry.py`
- Create: `path-planner/src/path_planner/v2/validation.py`
- Create: `path-planner/tests/test_v2_geometry.py`
- Create: `path-planner/tests/test_v2_wheel_provider.py`
- Create: `path-planner/tests/test_v2_route_validation.py`
- Modify: `path-planner/src/path_planner/v2/profiles.py`
- Modify: `path-planner/src/path_planner/v2/providers/__init__.py`
- Modify: `path-planner/src/path_planner/v2/__init__.py`

**Interfaces:**
- Produces: `WheelProfileV2`, `WheelMotionPrimitiveV2`, `WheelPrimitiveProviderV2`, `validate_route_l2()`。

- [ ] **Step 1: Write RED tests for pose semantics and boundaries**

```python
assert result.route.primitives[0].kind is PrimitiveKindV2.WHEEL_MOTION
assert result.validation_evidence.level is ValidationLevelV2.L2
assert result.validation_evidence.passed is True
assert result.search_telemetry.ackermann_feasible_claimed is False
assert DEFAULT_PLATFORM_KEY == "yutu2"
```

增加端点安全但中间 footprint 碰撞、unknown sweep、目标 theta 不可达、倒车开关和 30deg/30deg+epsilon 坡度测试。

- [ ] **Step 2: Implement explicit wheel profile and replayed typed segments**

```python
@dataclass(frozen=True, slots=True)
class WheelMotionPrimitiveV2:
    control_name: str
    start_state: PoseStateV2
    end_state: PoseStateV2
    samples: tuple[PoseStateV2, ...]
    duration_s: float
    distance_m: float
    energy_cost: float

class WheelPrimitiveProviderV2:
    plan: Callable[[PlanningRequestV2, FineSafetyAnchorV2], PlanningOutcomeV2]
```

`WheelProfileV2` 显式冻结差速/滑移转向、`0.612m x 0.580m` 包络、margin、倒车/原地转向、速度、角速度、integration dt、目标容差、`30.0deg` 坡度硬边界和版本化相对能耗 proxy；`.profile` 仍暴露精确 `PlatformProfileV2`。

使用现有 `HybridAStarPlanner`、`default_scout_mini_primitives()` 和 Task 4A 的公共 replay/transition validator 生成 opt-in pose route。每个候选边在入队前按 v2 cell-center rotated-footprint sweep 执行 fine L2；被拒边继续搜索。成功候选再独立 replay 并执行整条 L2 validation。不得导入或调用私有 `_apply_primitive`/`_footprint_cells`，不得读取 unknown truth 生成搜索代价。

`WheelMotionPrimitiveV2` 继承 `RoutePrimitiveV2`，额外冻结 control、samples、速度、角速度、倒车和原地转向字段。`start==goal` 返回一个零时长、零距离、零能耗且 L2 通过的 `hold` primitive。primitive `energy_cost` 是未加权的相对能耗；`CostBreakdownV2` 保存 objective 加权后的 component。真实 FOV/LOS 延后实现时，observation projection 必须明确标记“收益未计算”并返回零，不能伪造覆盖收益。

- [ ] **Step 3: Run focused tests and v1 regression**

Run: `cd path-planner; D:/conda_envs/lunar-explorer/python.exe -m pytest -q tests/test_v2_geometry.py tests/test_v2_wheel_provider.py tests/test_v2_route_validation.py tests/test_hybrid_astar.py tests/test_astar.py`

- [ ] **Step 4: Commit**

Commit: `feat: add opt-in wheel provider for path planner v2`

---

### Task 5: Gate 2B — Wheel audit, exact-map quality and Standard performance evidence

**Files:**
- Create: `path-planner/src/path_planner/v2/benchmark.py`
- Create: `path-planner/tests/test_v2_benchmark.py`
- Create: `configs/xunce_path_v2_gate2_wheel_v1.json`
- Modify: `scripts/run_xunce_path_v2_gate_benchmark.py`
- Modify: `tests/test_xunce_path_v2_gate_benchmark.py`
- Modify: `configs/stage_registry.json`

- [ ] **Step 1: Write RED metric tests**

`PrimitiveAuditRowV2` 必须把独立 expected label 与 provider result 分开；自标注行拒绝进入正式安全分母。

- [ ] **Step 2: Implement wheel audit aggregation**

计算 false positive、recall、完整 L2 比例、exact cost ratio、p50/p95/p99、timeout 和 reason histogram；固定 seed 和稳定 row ordering。

- [ ] **Step 3: Run Gate 2 evidence**

Gate 2 正式 runner 至少审计 10,000 wheel primitives。若独立 label 或 Standard schedule 缺失，`routing.json` 写 `provide_independent_wheel_oracle_labels` 或 `provide_standard_wheel_schedule`，不得伪造 pass；代码与安全合同通过但正式门可为 blocker。

- [ ] **Step 4: Commit**

Commit: `test: add wheel provider safety and performance gate`

---

### Task 6: Gate 3A — L0/L1/L2 lazy validation and complete-key cache

**Files:**
- Create: `path-planner/src/path_planner/v2/cache.py`
- Modify: `path-planner/src/path_planner/v2/validation.py`
- Create: `path-planner/tests/test_v2_lazy_validation.py`
- Create: `path-planner/tests/test_v2_cache.py`

**Interfaces:**
- Produces: `ValidationLevelV2`, `ValidationCacheKeyV2`, `ValidationCacheV2`。

- [ ] **Step 1: Write RED tests for fail-closed keys and L2-only success**

```python
with pytest.raises(ValueError, match="complete cache key"):
    ValidationCacheKeyV2(
        schema_version="",
        platform_profile_hash="profile",
        terrain_snapshot_hash="terrain",
        primitive_hash="primitive",
        validation_level=ValidationLevelV2.L2,
        objective_profile_hash=None,
    )
assert validate_route(
    route=route,
    anchor=anchor,
    profile=profile,
    max_level=ValidationLevelV2.L1,
).success is False
assert cache_on.route == cache_off.route
```

- [ ] **Step 2: Implement validation pipeline and cache**

Cache key 精确绑定 `schema_version`、`platform_profile_hash`、`terrain_snapshot_hash`、`primitive_hash`、`validation_level`，成本缓存另加 `objective_profile_hash`。任何缺值/schema drift 直接 miss 或拒绝，不共享错误结果。

- [ ] **Step 3: Run focused tests and commit**

Commit: `feat: add lazy validation and complete-key cache`

---

### Task 7: Gate 3B — Multi-heuristic and r/2r/4r conservative hierarchy

**Files:**
- Create: `path-planner/src/path_planner/v2/search.py`
- Create: `path-planner/src/path_planner/v2/hierarchy.py`
- Create: `path-planner/tests/test_v2_search.py`
- Create: `path-planner/tests/test_v2_hierarchy.py`
- Create: `configs/xunce_path_v2_gate3_accelerators_v1.json`
- Modify: `scripts/run_xunce_path_v2_gate_benchmark.py`

**Interfaces:**
- Produces: `StableSearchQueueV2`, `ConservativeHierarchyV2`, accelerator ablation results。

- [ ] **Step 1: Write RED equivalence and determinism tests**

对 fine-only、每个单 accelerator、full v2，断言 route safety/category/ordering 与 safe core 等价；单 worker/4 worker 和重复 hash seed 决策字段字节相同。

- [ ] **Step 2: Implement stable queues and conservative aggregation**

Anchor heuristic 保持 admissible；辅助队列只能建议 expansion。2r/4r cell 只有在所含 fine cells 全部已观测安全时才可标 safe；否则 unknown/blocked。任何 accelerator 异常关闭该开关并回到 fine-only。

- [ ] **Step 3: Run ablation Gate**

失败 accelerator 写 `disabled_accelerators` 和 reason，不阻塞 Gate 4；不得降低安全门。

- [ ] **Step 4: Commit**

Commit: `feat: add opt-in path planner v2 accelerators`

---

### Task 8: Gate 4A — Shared geometry and legged static-stability oracle

**Files:**
- Create: `path-planner/src/path_planner/v2/geometry.py`
- Create: `path-planner/src/path_planner/v2/oracles/__init__.py`
- Create: `path-planner/src/path_planner/v2/oracles/legged.py`
- Create: `path-planner/tests/test_v2_geometry.py`
- Create: `path-planner/tests/test_v2_legged_oracle.py`
- Modify: `path-planner/src/path_planner/v2/profiles.py`

- [ ] **Step 1: Write RED profile and geometry tests**

冻结：`simulation_proxy_static_crawl/v1`、body `0.60x0.40m`、foot rectangle `0.70x0.50m`、step length `0.50m`、step height `0.25m`、foothold slope `25deg`、support margin `0.05m`、local foothold grid `r/2`。

- [ ] **Step 2: Implement geometry helpers**

公开签名固定为 `convex_hull_xy(points) -> Sequence[WorldPoint]`、`point_margin_to_convex_polygon(point, polygon) -> float`、`oriented_rectangle_cells(center, theta_rad, length_m, width_m, geometry) -> Sequence[Cell]`、`sample_pose_sweep(start, end, step_m) -> Sequence[PoseStateV2]`。

- [ ] **Step 3: Implement stable legged reason taxonomy**

Oracle 逐项检查 `legged_foothold_unknown`、`legged_foothold_hard_obstacle`、`legged_foothold_slope_exceeded`、`legged_step_length_exceeded`、`legged_step_height_exceeded`、`legged_support_margin_insufficient`、`legged_body_sweep_unknown`、`legged_body_sweep_collision`、`legged_foot_sequence_invalid`。

- [ ] **Step 4: Run tests and commit**

Commit: `feat: add legged static stability oracle`

---

### Task 9: Gate 4B — Legged provider, complete route L2 and evidence

**Files:**
- Create: `path-planner/src/path_planner/v2/providers/legged.py`
- Create: `path-planner/tests/test_v2_legged_provider.py`
- Create: `configs/xunce_path_v2_gate4_legged_v1.json`
- Modify: `path-planner/src/path_planner/v2/providers/__init__.py`
- Modify: `path-planner/src/path_planner/v2/api.py`
- Modify: `path-planner/src/path_planner/v2/validation.py`
- Modify: `scripts/run_xunce_path_v2_gate_benchmark.py`

- [ ] **Step 1: Write RED typed-state/provider tests**

```python
class LegIdV2(str, Enum):
    FRONT_LEFT = "front_left"
    FRONT_RIGHT = "front_right"
    REAR_LEFT = "rear_left"
    REAR_RIGHT = "rear_right"
```

测试完整 foot-contact state key、稳定 foot sequence、严格 goal theta、端点安全但 body sweep 失败时整路否决。

- [ ] **Step 2: Implement deterministic provider and L2 route validation**

Provider 只生成满足 profile 几何界的候选；oracle 拥有安全裁决权。成功结果 capability 固定 `simulation_proxy`，不得出现 dynamic gait/real-robot claim。

- [ ] **Step 3: Run 10,000 primitive Gate or emit independent-label blocker**

正式 label 缺失时稳定 route 为 `provide_independent_legged_oracle_labels`；其余合同测试仍必须全绿。

- [ ] **Step 4: Commit**

Commit: `feat: add legged simulation proxy provider`

---

### Task 10: Gate 5A — Pure lunar ballistics and explicit proxy profile gaps

**Files:**
- Create: `path-planner/src/path_planner/v2/ballistics.py`
- Create: `path-planner/tests/test_v2_ballistics.py`
- Modify: `path-planner/src/path_planner/v2/profiles.py`

- [ ] **Step 1: Write RED ballistics tests**

冻结 `g=1.62`、速度 `{1.5,2.0,2.5,3.0}`、仰角 `{30,45,60}deg`、16 方位、`sigma=0.05*range+0.05m`、landing slope `15deg`、probability `0.99`，并验证 3.0m/s、45deg 最大水平距离约 5.56m。

- [ ] **Step 2: Implement pure deterministic functions**

公开签名固定为 `sample_ballistic_arc(start, speed_mps, elevation_rad, azimuth_rad, g_mps2, dt_s) -> Sequence[BallisticSampleV2]`、`normal_interval_mass(lo, hi, mean, sigma) -> float`、`landing_zone_cells(mean_xy, sigma_m, probability_threshold, geometry) -> Sequence[LandingCellMassV2]`。

- [ ] **Step 3: Encode missing envelope parameters as unsupported capability**

`HopperProfileV2` 的 `body_envelope_radius_m`、`launch_reference_height_m`、`arc_clearance_margin_m`、`landing_footprint_radius_m`、`stop_condition`、`energy_model` 不提供默认实机值。任一缺失时 provider construction 返回 `hopper_proxy_profile_incomplete`，而不是零值放行。

- [ ] **Step 4: Run tests and commit**

Commit: `feat: add lunar ballistic proxy primitives`

---

### Task 11: Gate 5B — Hopper envelope/landing oracle, provider and evidence

**Files:**
- Create: `path-planner/src/path_planner/v2/oracles/hopper.py`
- Create: `path-planner/src/path_planner/v2/providers/hopper.py`
- Create: `path-planner/tests/test_v2_hopper_oracle.py`
- Create: `path-planner/tests/test_v2_hopper_provider.py`
- Create: `configs/xunce_path_v2_gate5_hopper_v1.json`
- Modify: `path-planner/src/path_planner/v2/providers/__init__.py`
- Modify: `path-planner/src/path_planner/v2/api.py`
- Modify: `path-planner/src/path_planner/v2/validation.py`
- Modify: `scripts/run_xunce_path_v2_gate_benchmark.py`

- [ ] **Step 1: Write RED full-envelope and probability tests**

覆盖中心线安全但 envelope 边缘 unknown/碰撞、arc 越界、99% 落区含 unknown、坡度 `15+epsilon`、probability `0.99-epsilon`、目标姿态和停止失败。

- [ ] **Step 2: Implement stable hopper reasons**

固定 `hopper_launch_unknown`、`hopper_launch_unsafe`、`hopper_arc_unknown`、`hopper_arc_boundary_violation`、`hopper_arc_clearance_violation`、`hopper_landing_zone_unknown`、`hopper_landing_slope_exceeded`、`hopper_landing_probability_below_threshold`、`hopper_landing_theta_unreachable`、`hopper_stop_condition_failed`、`hopper_midcourse_correction_unsupported`。

- [ ] **Step 3: Implement provider only for complete explicit simulation profiles**

默认 repo config 保持 profile incomplete，并使 Gate 5 `routing.json` 明确 `freeze_hopper_simulation_proxy_profile_parameters`。测试 fixture 可显式传 proxy-only 参数验证算法，但报告不得把 fixture 值提升为正式/实机能力。

- [ ] **Step 4: Run Gate and commit**

Commit: `feat: add fail-closed hopper simulation proxy provider`

---

### Task 12: Gate 6A — Observed-only observation projection and offline PPO adapter

**Files:**
- Create: `path-planner/src/path_planner/v2/observation.py`
- Create: `path-planner/src/path_planner/v2/adapters/__init__.py`
- Create: `path-planner/src/path_planner/v2/adapters/ppo_target.py`
- Create: `path-planner/tests/test_v2_observation_projection.py`
- Create: `path-planner/tests/test_v2_ppo_target_adapter.py`
- Optional Modify: `src/lunar_exploration_ppo/integrations/path_planner_adapter.py` only if a PPO-package shim is required by tests.
- Create: `tests/ppo_highres_frontier/test_path_planner_v2_adapter.py`

- [ ] **Step 1: Write RED truth-leakage and target-preservation tests**

同一 observed snapshot 下改变 hidden truth，projection 与 request bytes 必须不变；PPO target `(x,y,theta)` 必须逐值保留；失败不可重选候选。

- [ ] **Step 2: Implement route projection**

wheel/legged 按固定弧长 route samples + endpoint theta；hopper 默认仅 launch/landing。LOS 遇第一个 unknown 即停止，不调用 `SensorUpdater.reveal()`。

- [ ] **Step 3: Implement offline adapter**

```python
@dataclass(frozen=True, slots=True)
class PpoTargetV2:
    x_m: float
    y_m: float
    theta_rad: float

BuildPpoRequestFn = Callable[
    [PpoTargetV2, ObservedTerrainInputV2, str, ObjectiveProfileV2, ResourceBudgetV2, float, AcceleratorPolicyV2, int],
    PlanningRequestV2,
]
PlanPpoTargetFn = Callable[[PlanningRequestV2], PlanningOutcomeV2]
```

不得修改 `env.py`、action execution、policy、trainer、reward。若添加 shim，只能在既有 `path_planner_adapter.py` 内新增显式类，旧 `PathPlannerAdapter` 的字节/行为测试保持不变。

- [ ] **Step 4: Run focused and foundation tests, then commit**

Commit: `feat: add offline ppo target adapter for path planner v2`

---

### Task 13: Gate 6B — Formal benchmark, ablation and release decision

**Files:**
- Create: `path-planner/src/path_planner/v2/benchmark_fixtures.py`
- Create: `path-planner/tests/test_v2_benchmark_metrics.py`
- Create: `configs/xunce_path_v2_gate6_formal_benchmark_v1.json`
- Modify: `scripts/run_xunce_path_v2_gate_benchmark.py`
- Modify: `tests/test_xunce_path_v2_gate_benchmark.py`
- Modify: `configs/stage_registry.json`

- [ ] **Step 1: Write RED benchmark schedule and blocker tests**

正式输入缺任一项时，断言 status 为 `blocked` 且 reason 精确列出：独立 10k labels、小图 optimum、Standard 100/平台、Kilometer 30/平台、PPO targets。不得自动生成并把同一 oracle 的输出当 label。

- [ ] **Step 2: Implement resumable phases**

固定 phases：`primitive_audit`、`exact_quality`、`standard`、`kilometer`、`ablation`、`determinism_worker_cache`、`aggregate`。每 phase 追加 `phase-state.jsonl`，完成后原子写 summary。

- [ ] **Step 3: Implement complete ablation and statistics**

矩阵固定：v1 A*、wheel Hybrid A* opt-in、v2 fine-only、`+multi-heuristic`、`+hierarchy`、`+lazy validation`、`+cache`、full v2。输出安全、reachable success、recall、resource cost、coverage efficiency、expanded、L0/L1/L2 rejection、cache hit、p50/p95/p99、timeout；coverage 用固定 seed paired episode bootstrap 95% CI。

- [ ] **Step 4: Enforce release boundaries**

即使全部内部 Gate 通过，`routing.json` 也必须保持 checkpoint/default/executor/canary 均未授权；primary route 只能是 `path_v2_internal_validation_passed_no_release_authority` 或具体 blocker。

- [ ] **Step 5: Run available Gate 6 evidence and commit**

若正式 fixture 尚缺，期望 runner 成功生成完整 artifact 且 status=`blocked`，不是 process error。

Commit: `test: add path planner v2 formal benchmark gate`

---

### Task 14: Whole-branch regression, final review and Goal closure

**Files:**
- Modify only files required by reviewer Critical/Important findings.
- Write runtime reports only under `D:/xunce/out/path_v2/g0..g6`.

- [ ] **Step 1: Re-run path-planner full regression**

Run with isolated Python 3.12 and D worktree `PYTHONPATH`:

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest -q -p no:cacheprovider path-planner/tests
```

Expected: all v2 tests pass；v1 original set remains 156 passed/17 optional skipped。

- [ ] **Step 2: Re-run PPO external baseline guard**

Run only `tests/ppo_highres_frontier/test_stage1_smoke_env.py` with isolated import origin.

Expected: exactly 13 allowlisted failures、43 pass、0 errors、0 skip；任何新增 nodeid 阻止完成。

- [ ] **Step 3: Validate artifact sets and UTF-8**

对 g0–g6 检查 canonical 文件、hash、path length、JSON finite、UTF-8 report、proxy/claim boundary。未到达的正式门必须有 blocker，不得写 pass。

- [ ] **Step 4: Request whole-branch code review**

使用 merge-base `b635740ee021258ef31811ec87c60add839fc5f9` 生成完整 review package；修复并重审全部 Critical/Important，Minor 进入 ledger。

- [ ] **Step 5: Finish without publishing or integration side effects**

只报告分支、commits、tests、Gate 状态和 D 盘 evidence；不 push、不建 PR、不发布 checkpoint、不安装 default policy、不连接 executor、不启动 canary，除非用户另行明确授权。

---

## Plan Self-Review

- Spec coverage：Gate 0–6、三平台、fine anchor、完整 L2、资源/探索排序、确定性、超时、cache、ablation、PPO 边界和 artifact 均有对应 task。
- Placeholder scan：计划没有用待定实现替代行为；设计未冻结的 hopper/formal fixture 输入被明确实现为稳定 blocker，而不是留空或猜值。
- Type consistency：`PlanningRequestV2 -> FineSafetyAnchorV2 -> PrimitiveProviderV2 -> PlanningOutcomeV2` 为唯一主链；platform typed primitive 统一装入 `TypedRouteV2`，Gate 6 adapter 仍返回同一 outcome。
- Isolation：所有生产 v2 代码位于 standalone package；根 runner 不反向污染 v1 public API；PPO shim 如有需要也只能扩展既有唯一 integration 文件。
