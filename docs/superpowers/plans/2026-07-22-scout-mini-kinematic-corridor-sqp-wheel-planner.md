# Scout Mini Kinematic Corridor SQP Wheel Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为当前 Scout Mini 差速/滑移转向平台新增显式 opt-in 的 `wheel_kinematic_corridor_sqp/v1`：在单一 2 秒硬期限内生成最多三条拓扑不同的二维走廊，以确定性运动学直接多重射击 SQP 求解分段常值 `(v, omega, duration)` 轨迹，并且只返回通过独立完整连续 L2 的第一条路线。

**Architecture:** 新能力与现有 `providers/wheel.py` Hybrid A* 完全并列。`wheel_corridors.py` 只生成二维引导，`wheel_sqp_initialization.py` 生成固定模式初值，`wheel_sqp_solver.py` 通过私有 SciPy SLSQP 适配层执行直接多重射击，`wheel_sqp_serialization.py` 负责严格 codec/规范化 identity，`wheel_sqp_validation.py` 从规范化序列化 segments 独立重建解析差速轨迹并用区间证明检查完整旋转矩形扫掠，`providers/wheel_sqp.py` 只负责编排，`wheel_sqp_api.py` 负责 capability seal 与轻量 codec/reseal postcondition，不重复昂贵扫掠。公共 v2 success 包装新的 `WheelKinematicRouteV2(TypedRouteV2)`，其 primitives 为 `WheelKinematicSegmentV2`；既有 v1、v2 Hybrid A* 和 Gate 0–6 证据语义不变。

**Tech Stack:** Python 3.12、frozen/slotted dataclasses、NumPy 2.x、SciPy 1.18.0 SLSQP、pytest、现有 canonical JSON/SHA-256、现有 `PlanningDeadlineV2` / `FineSafetyAnchorV2`、根仓库 artifact IO 与 D 盘 benchmark 输出。

## Global Constraints

- 规范设计固定为 `docs/superpowers/specs/2026-07-22-scout-mini-kinematic-corridor-sqp-wheel-planner-design.md`；实现不得改写已批准算法选择。
- 工作树固定为 `D:/codex/worktrees/multiplatform-path-planner-v2`，分支固定为 `codex/multiplatform-path-planner-v2`，原始基线固定为 `b635740ee021258ef31811ec87c60add839fc5f9`。
- 本工作是与既有 Gate 0–6 收口相互独立的新请求。不得改变 `D:/xunce/out/path_v2/g0` 至 `g6` 的既有结果、门定义、正式输入或状态。
- 保护 `C:/Users/77634/.codex/worktrees/ca49/lunar-path-planning` 的 Stage6 dirty 工作树；不在该路径读取、写入、测试、暂存、提交或清理。
- v1 A*、CLI、默认 policy 和根包默认导出保持不变；新能力只能通过 `path_planner.v2.plan_v2()` 加显式新 profile ID 选择。
- 不发布 checkpoint、不替换 default policy、不连接 executor、不启动 canary、不 push、不 release。
- 现有 `path-planner/src/path_planner/v2/providers/wheel.py`、`path-planner/src/path_planner/search/hybrid_astar.py` 和 `WheelMotionPrimitiveV2` 保持语义兼容；SQP 失败后严禁隐式调用 Hybrid A*。
- 单次请求只读取传入的 immutable `TerrainSnapshotV2`。不跨请求复用 open/closed、走廊、SQP warm start、L2 反例或路线；`determinism_seed` 参与 identity，但本能力不做随机 multi-start。
- 平台固定为当前 Scout Mini：差速/滑移转向、车体 `0.612m x 0.580m`、`max_traversable_slope_deg=30.0`、倒车可用、原地旋转可用；不声称 Ackermann、动力学、摩擦、侧滑或实机认证可行。
- 目标闭区间固定为位置 `<=0.25m`、wrap-safe 朝向 `<=0.08726646259971647rad`；输出保留真实解析 replay 终点，禁止 endpoint snapping。
- 共享 deadline 只能由 `plan_v2()` 创建，effective timeout 保持 `min(request.timeout_s, 2.0)`；走廊、初值、SQP、规范化、唯一完整连续 L2、修复、编码与 API 轻量 postcondition 使用同一个绝对 deadline。
- 只有 `wheel_sqp_validation.py` 的完整连续 L2 可以授予 `ValidationLevelV2.L2`。shooting nodes、optimizer clearance、观测 samples、二维走廊或 SciPy success flag 均不是安全权威。
- 每条走廊最多一次由稳定首反例驱动的 repair；numeric、identity、kinematic、resource、deadline 或 reseal 错误不得 repair。
- 第一条通过完整 L2 的路线立即返回；不得为了质量改善继续处理后续走廊，运行时不宣称最优。
- SciPy 只出现在私有 `wheel_sqp_solver.py` 懒加载适配层；公共 route、hash、failure 和 capability 语义不得包含 SciPy 私有状态或后端名称。`wheel-sqp` optional extra 精确锁定 `scipy==1.18.0`，未安装时只有显式新 capability 返回 typed unsupported，v1 与旧 v2 wheel 仍可运行；后端升级必须先升级 solver contract 或 capability revision。
- 所有中文 `.md`、`.py` 注释与测试字符串按 UTF-8 写入；写后用 Python `read_text(encoding="utf-8")` 验证，无 BOM、U+FFFD 或乱码。
- 新 benchmark 与临时大文件写入 `D:/xunce/out/path_v2/wsqp` 和 `D:/CodexDownloads/pip-cache`；不得把下载缓存、正式 corpus 或 benchmark artifact 写到 C 盘或提交到 Git。
- 正式 `oracle_reachable`、unsafe label 与 small-map optimum 必须来自独立、高预算、同能力边界且 `oracle_source_id != provider_source_id` 的输入。缺输入时 runner 必须返回稳定 blocked，不得自标注为通过。
- 正式门固定为：unsafe false-positive `0`、返回路线完整 L2 `100%`、oracle-reachable success `>=99%`、Standard p95 `<=250ms`、Kilometer p95 `<=750ms`、端到端 `<=2s`、small exact 首条路线 resource cost `<=1.10 * optimum`、重复运行决策 hash 一致。
- 每个任务严格 RED → GREEN → focused regression → review。子 agent 不暂存、不提交；主 agent 只在精确 diff、测试和复核通过后提交当前任务文件。

---

## File Structure

### Nested `path-planner` implementation

- Modify `path-planner/pyproject.toml`：只在 `wheel-sqp` optional extra 精确声明 `scipy==1.18.0`，不改变基础依赖。
- Modify `path-planner/src/path_planner/v2/profiles.py`：新增 capability 常量和 `WheelKinematicSQPProfileV2`，不改变 `WheelProfileV2`。
- Create `path-planner/src/path_planner/v2/wheel_sqp_contracts.py`：typed segment/route、solver/L2 result、counterexample、reserve、telemetry/evidence 和稳定 reason 合同。
- Create `path-planner/src/path_planner/v2/wheel_kinematics.py`：唯一解析差速积分、解析导数与派生 observation samples。
- Create `path-planner/src/path_planner/v2/wheel_sqp_serialization.py`：private candidate/public L2 route 两层严格 codec、规范化、hash 和 byte-stable round trip。
- Create `path-planner/src/path_planner/v2/wheel_corridors.py`：稳定二维 A*、Yen 有界候选、阻塞分量、离散拓扑签名与最多三走廊排序。
- Create `path-planner/src/path_planner/v2/wheel_sqp_initialization.py`：走廊简化、forward/reverse/turn 模式动态规划和直接多重射击初值。
- Create `path-planner/src/path_planner/v2/wheel_sqp_solver.py`：固定变量布局、目标/约束/Jacobian、私有 SLSQP 适配、deadline/resource taxonomy 和 repair constraint。
- Create `path-planner/src/path_planner/v2/wheel_sqp_validation.py`：规范化后解析 replay、identity/reseal、连续旋转矩形扫掠区间证明和首反例。
- Create `path-planner/src/path_planner/v2/providers/wheel_sqp.py`：最多三走廊、每走廊一次 repair、第一条完整 L2 成功立即返回。
- Create `path-planner/src/path_planner/v2/wheel_sqp_api.py`：新 capability 的 trusted dispatch、provider seal、独立 postcondition L2/reseal 和 timeout precedence。
- Modify `path-planner/src/path_planner/v2/api.py`：只增加新 capability 的显式分支。
- Modify `path-planner/src/path_planner/v2/providers/__init__.py` 与 `path-planner/src/path_planner/v2/__init__.py`：只通过 opt-in v2 导出新 surface。
- Modify `path-planner/src/path_planner/v2/observation.py`：仅补新 segment 的派生 sample 兼容性测试所需薄逻辑；安全语义不依赖这里。
- Create `path-planner/src/path_planner/v2/wheel_sqp_benchmark.py`：独立正式行合同、聚合和 threshold audit。

### Nested tests

- Create `path-planner/tests/test_v2_wheel_sqp_contracts.py`。
- Create `path-planner/tests/test_v2_wheel_kinematics.py`。
- Create `path-planner/tests/test_v2_wheel_sqp_serialization.py`。
- Create `path-planner/tests/test_v2_wheel_corridors.py`。
- Create `path-planner/tests/test_v2_wheel_sqp_initialization.py`。
- Create `path-planner/tests/test_v2_wheel_sqp_solver.py`。
- Create `path-planner/tests/test_v2_wheel_sqp_validation.py`。
- Create `path-planner/tests/test_v2_wheel_sqp_provider.py`。
- Create `path-planner/tests/test_v2_wheel_sqp_api.py`。
- Create `path-planner/tests/test_v2_wheel_sqp_conformance.py`。
- Create `path-planner/tests/test_v2_wheel_sqp_benchmark.py`。
- Modify `path-planner/tests/test_package_imports.py`、`test_v2_profiles.py`、`test_v2_api.py`、`test_v2_observation_projection.py`：依赖、导出、API 和观测兼容回归。

### Root benchmark orchestration

- Create `scripts/run_xunce_path_v2_wheel_sqp_formal.py`：只编排本能力的 local smoke、正式输入审计、实际 provider 执行、metrics、routing、manifest 和 report。
- Create `configs/xunce_path_v2_wheel_sqp_formal_v1.json`：输入路径、profile、门值、阶段与四项发布边界。
- Create `tests/test_xunce_path_v2_wheel_sqp_formal.py`：runner、输入 lineage、blocked/pass、artifact 和 threshold 测试。
- Modify `configs/stage_registry.json`：注册独立 `xunce-path-v2-wheel-sqp-formal`，默认 root `D:/xunce/out/path_v2/wsqp`。
- Modify `docs/xunce-stage-documentation-index.md`：登记 spec、plan 与 D 盘结果职责。

---

## Frozen Public and Internal Contracts

实现时先冻结以下 ID；字符串变化等同于合同变化：

```python
WHEEL_KINEMATIC_CORRIDOR_SQP_CAPABILITY_V2 = "wheel_kinematic_corridor_sqp/v1"
WHEEL_KINEMATIC_SEGMENT_SCHEMA_V2 = "wheel_kinematic_segment/v1"
WHEEL_KINEMATIC_SOLVER_CONTRACT_V2 = "wheel_kinematic_direct_multiple_shooting_sqp/v1"
WHEEL_KINEMATIC_CANONICALIZATION_V2 = "wheel_kinematic_decimal12_half_even/v1"
WHEEL_KINEMATIC_L2_VALIDATOR_V2 = "wheel_kinematic_continuous_rectangle_sweep_l2/v1"
WHEEL_KINEMATIC_CORRIDOR_SOURCE_V2 = "wheel_deterministic_yen_h_signature_corridors/v1"
WHEEL_KINEMATIC_INITIALIZER_V2 = "wheel_forward_reverse_turn_dp_initializer/v1"
WHEEL_KINEMATIC_L2_RESERVE_MODEL_V2 = "wheel_l2_reserve_model/v1"
WHEEL_KINEMATIC_CONTROL_SLEW_V2 = "wheel_segment_center_control_slew/v1"
WHEEL_KINEMATIC_OBSERVATION_SOURCE_V2 = "wheel_kinematic_derived_samples/v1"
```

新平台 profile 使用精确公共 `PlatformProfileV2`：

```python
PlatformProfileV2(
    profile_id="scout-mini-wheel-kinematic-sqp/v1",
    platform_kind=PlatformKindV2.WHEEL,
    capability_revision=WHEEL_KINEMATIC_CORRIDOR_SQP_CAPABILITY_V2,
    simulation_proxy=False,
    max_traversable_slope_deg=30.0,
    goal_position_tolerance_m=0.25,
    goal_heading_tolerance_rad=0.08726646259971647,
)
```

`WheelKinematicSQPProfileV2` 冻结这些 v1 数值；不允许从本次候选动态估计：

```python
body_length_m = 0.612
body_width_m = 0.580
footprint_safety_margin_m = 0.0
max_speed_mps = 1.0
max_angular_speed_radps = 0.7853981633974483
max_linear_accel_mps2 = 0.5
max_linear_decel_mps2 = 0.75
max_angular_accel_radps2 = 1.5707963267948966
min_segment_duration_s = 0.05
max_segment_duration_s = 120.0
max_segment_heading_change_rad = 1.5707963267948966
max_corridors = 3
max_corridor_candidates = 24
max_segments = 48
max_sqp_iterations = 40
max_sqp_function_evaluations = 4096
sqp_ftol = 1.0e-10
hard_constraint_tolerance = 1.0e-9
near_zero_omega_radps = 1.0e-8
min_nonzero_control = 1.0e-4
canonical_decimal_places = 12
max_l2_interval_records = 262_144
max_l2_candidate_cells = 1_000_000
max_l2_subdivision_depth = 24
continuous_separation_epsilon_m = 1.0e-9
repair_clearance_m = 1.0e-4
observation_sample_translation_m = 0.25
observation_sample_heading_rad = 0.08726646259971647
solver_memory_reservation_bytes = 16_777_216
relative_energy_proxy_id = "wheel_relative_motion_energy/v1"
translation_energy_per_m = 1.0
rotation_energy_per_rad = 0.2
idle_energy_per_s = 0.05
reverse_energy_multiplier = 1.25
energy_normalization = 1.0
time_normalization_s = 1.0
control_slew_regularization_weight = 1.0e-4
corridor_deviation_regularization_weight = 1.0e-3
clearance_soft_weight = 0.0
```

`WheelSQPModeV2` 精确包含 `FORWARD`、`REVERSE`、`TURN_LEFT`、`TURN_RIGHT`、`STOP`。方向切换必须插入正持续时间且 `v=omega=0` 的 `STOP` segment；initializer、SQP hard constraints 与 L2 都拒绝任意相邻 `v[k] * v[k + 1] < 0.0`。

分段间的“加减速”明确为 `wheel_segment_center_control_slew/v1` 离散命令变化率，不宣称连续轮地动力学。令 `tau_k=(dt[k]+dt[k+1])/2`、`s_k=abs(v[k])`：

```python
speed_up_k = max(0.0, s[k + 1] - s[k]) / tau_k
slow_down_k = max(0.0, s[k] - s[k + 1]) / tau_k
angular_slew_k = abs(omega[k + 1] - omega[k]) / tau_k
```

硬边界分别为 `speed_up<=0.5`、`slow_down<=0.75`、`angular_slew<=1.5707963267948966`。输入没有起始/终止速度，因此 v1 不对请求边界外的速度作推断，只检查相邻 segment。

`reverse` 必须等于 `v_mps < 0.0`；`turn_in_place` 必须等于 `v_mps == 0.0 and omega_radps != 0.0`。`STOP` 的两标志均为 false，正持续时间、零距离但保留 idle relative energy。profile 固定倒车和原地旋转均可用。

失败映射固定为：

| reason | `FailureCategoryV2` |
|---|---|
| `wheel_sqp_profile_unsupported`、`wheel_sqp_backend_unavailable`、`wheel_sqp_objective_unsupported`、`wheel_sqp_accelerator_required_unsupported` | `UNSUPPORTED_CAPABILITY` |
| `wheel_sqp_start_invalid` | `UNSAFE_START` |
| `wheel_sqp_goal_invalid` | `UNSAFE_GOAL` |
| `wheel_sqp_no_2d_corridor` | `GOAL_POSE_UNREACHABLE` |
| `wheel_sqp_initialization_failed`、`wheel_sqp_infeasible` | `NO_COMPLETE_ROUTE` |
| `wheel_sqp_candidate_l2_rejected`、`wheel_sqp_repair_l2_rejected` | `VALIDATION_FAILED` |
| `wheel_sqp_goal_tolerance_exceeded` | `GOAL_POSE_UNREACHABLE` |
| `wheel_sqp_corridor_budget_exceeded`、`wheel_sqp_resource_budget_exceeded` | `RESOURCE_LIMIT` |
| `planning_deadline_expired` | `TIMEOUT` |
| `wheel_sqp_numeric_contract_failed`、`wheel_sqp_identity_mismatch`、`wheel_sqp_internal_error` | `INTERNAL_ERROR` |

所有 failure 都返回 `route=None` 的 `PlanningFailureV2`；任何 candidate、repair 中间量或部分 route 都不得泄露为可执行输出。

The existing generic `plan_v2()` center-cell preflight remains before capability dispatch. Therefore an unsafe center cell keeps the existing exact reason `terrain_out_of_bounds`、`terrain_unknown`、`terrain_hard_obstacle`、`terrain_not_traversable` or `terrain_slope_exceeded` with `UNSAFE_START/UNSAFE_GOAL`. `wheel_sqp_start_invalid` and `wheel_sqp_goal_invalid` are reserved for the additional full-rectangle stationary footprint check after the generic center cell has passed.

---

## Implementation Tasks

### Task 1: Freeze Dependency, Profile, Typed Segment, Evidence, and Resource Contracts

**Files:**
- Modify: `path-planner/pyproject.toml`
- Modify: `path-planner/src/path_planner/v2/profiles.py`
- Create: `path-planner/src/path_planner/v2/wheel_sqp_contracts.py`
- Modify: `path-planner/src/path_planner/v2/__init__.py`
- Modify: `path-planner/tests/test_package_imports.py`
- Modify: `path-planner/tests/test_v2_profiles.py`
- Create: `path-planner/tests/test_v2_wheel_sqp_contracts.py`

**Interfaces:**
- Produces: `WheelKinematicSQPProfileV2`, `audit_wheel_kinematic_sqp_profile_v2()`, `WheelKinematicSegmentV2`, `WheelKinematicRouteV2`, `WheelSQPModeV2`, `WheelSQPStatusV2`, `WheelCorridorV2`, `WheelSQPInitialGuessV2`, `WheelSQPCandidateV2`, `WheelSQPOptimizationResultV2`, `WheelL2CounterexampleV2`, `WheelTrajectoryL2ResultV2`, `WheelSQPValidationEvidenceV2`, `WheelSQPSearchTelemetryV2`, `WheelSQPResourceLedgerV1`, `L2ReserveModelV1`。
- Consumes: existing exact `PlatformProfileV2`, `RoutePrimitiveV2`, `PoseStateV2`, `ValidationEvidenceV2`, `SearchTelemetryV2`, `Cell`, `PlanningDeadlineV2`。

- [ ] **Step 1: Write RED tests for dependency metadata, exact profile identity, typed segment, and immutable evidence**

```python
def test_path_planner_pins_the_solver_optional_extra() -> None:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'wheel-sqp = [' in text
    assert '"scipy==1.18.0"' in text


def test_wheel_sqp_profile_freezes_current_scout_mini_contract() -> None:
    profile = make_wheel_sqp_profile()
    assert profile.profile.capability_revision == "wheel_kinematic_corridor_sqp/v1"
    assert profile.profile.goal_position_tolerance_m == 0.25
    assert profile.profile.goal_heading_tolerance_rad == 0.08726646259971647
    assert profile.body_length_m == 0.612
    assert profile.body_width_m == 0.580
    assert profile.max_corridors == 3
    assert profile.max_segments == 48
    assert profile.relative_energy_proxy_id == "wheel_relative_motion_energy/v1"
    assert not hasattr(profile, "__dict__")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_traversable_slope_deg", 30.000000000000004),
        ("goal_position_tolerance_m", 0.25000000000000006),
        ("goal_heading_tolerance_rad", nextafter(0.08726646259971647, inf)),
        ("capability_revision", "wheel_kinematic_corridor_sqp/v2"),
    ],
)
def test_wheel_sqp_profile_rejects_boundary_or_identity_drift(field, value) -> None:
    with pytest.raises((TypeError, ValueError)):
        make_wheel_sqp_profile(platform_overrides={field: value})


def test_wheel_kinematic_segment_and_route_require_exact_hashes_and_l2() -> None:
    segment = make_segment()
    route = make_route(segment)
    assert type(segment) is WheelKinematicSegmentV2
    assert type(route) is WheelKinematicRouteV2
    assert isinstance(route, TypedRouteV2)
    assert segment.kind is PrimitiveKindV2.WHEEL_MOTION
    assert segment.energy_cost == segment.relative_energy
    assert segment.validation_level is ValidationLevelV2.L2
    assert len(segment.segment_hash) == 64
    assert len(route.route_hash) == 64
    assert len(route.source_candidate_hash) == 64
    assert route.request_hash == REQUEST_HASH
    assert route.profile_hash == PROFILE_HASH
    assert route.terrain_snapshot_hash == SNAPSHOT_HASH
    with pytest.raises(ValueError, match="reverse"):
        replace(segment, reverse=not segment.reverse)
```

Test exact subclass compatibility without changing old base serialization:

```python
def test_wheel_sqp_evidence_and_telemetry_are_base_compatible_but_exactly_typed() -> None:
    evidence = make_l2_evidence()
    telemetry = make_sqp_telemetry()
    assert isinstance(evidence, ValidationEvidenceV2)
    assert isinstance(telemetry, SearchTelemetryV2)
    assert type(evidence) is WheelSQPValidationEvidenceV2
    assert type(telemetry) is WheelSQPSearchTelemetryV2
    assert evidence.route_hash == "a" * 64
    assert evidence.candidate_hash == "b" * 64
    assert telemetry.corridor_count == 3
    assert canonical_json_bytes(ValidationEvidenceV2("base/v1", ValidationLevelV2.L2, True, ("ok",))) == (
        b'{"checks":["ok"],"level":"L2","passed":true,"validator_id":"base/v1"}'
    )
```

- [ ] **Step 2: Run RED tests**

Run from `path-planner/`:

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_package_imports.py tests/test_v2_profiles.py tests/test_v2_wheel_sqp_contracts.py -q -x
```

Expected: new tests fail only because SciPy metadata and the new wheel SQP surface do not exist; all pre-existing selected nodeids remain GREEN.

- [ ] **Step 3: Pin SciPy and implement the exact profile audit**

Add an optional extra to `path-planner/pyproject.toml` without changing the base dependency list:

```toml
[project.optional-dependencies]
dev = [
  "pytest>=8",
]
wheel-sqp = [
  "scipy==1.18.0",
]
```

Install into the D-drive environment without a C-drive download cache:

```powershell
New-Item -ItemType Directory -Force D:/CodexDownloads/pip-cache | Out-Null
$env:PIP_CACHE_DIR='D:/CodexDownloads/pip-cache'
D:/conda_envs/lunar-explorer/python.exe -m pip install -e ".[dev,wheel-sqp]"
```

Implement the profile as an exact capability wrapper, not as a mutation of `WheelProfileV2`:

```python
@dataclass(frozen=True, slots=True)
class WheelKinematicSQPProfileV2:
    profile: PlatformProfileV2
    steering_model: str = "differential_skid_steer"
    body_length_m: float = 0.612
    body_width_m: float = 0.580
    footprint_safety_margin_m: float = 0.0
    reverse_enabled: bool = True
    turn_in_place_enabled: bool = True
    max_speed_mps: float = 1.0
    max_angular_speed_radps: float = 0.7853981633974483
    max_linear_accel_mps2: float = 0.5
    max_linear_decel_mps2: float = 0.75
    max_angular_accel_radps2: float = 1.5707963267948966
    min_segment_duration_s: float = 0.05
    max_segment_duration_s: float = 120.0
    max_segment_heading_change_rad: float = 1.5707963267948966
    max_corridors: int = 3
    max_corridor_candidates: int = 24
    max_segments: int = 48
    max_sqp_iterations: int = 40
    max_sqp_function_evaluations: int = 4096
    sqp_ftol: float = 1.0e-10
    hard_constraint_tolerance: float = 1.0e-9
    near_zero_omega_radps: float = 1.0e-8
    min_nonzero_control: float = 1.0e-4
    canonical_decimal_places: int = 12
    max_l2_interval_records: int = 262_144
    max_l2_candidate_cells: int = 1_000_000
    max_l2_subdivision_depth: int = 24
    continuous_separation_epsilon_m: float = 1.0e-9
    repair_clearance_m: float = 1.0e-4
    observation_sample_translation_m: float = 0.25
    observation_sample_heading_rad: float = 0.08726646259971647
    solver_memory_reservation_bytes: int = 16_777_216
    relative_energy_proxy_id: str = WHEEL_RELATIVE_ENERGY_PROXY_ID_V2
    translation_energy_per_m: float = 1.0
    rotation_energy_per_rad: float = 0.2
    idle_energy_per_s: float = 0.05
    reverse_energy_multiplier: float = 1.25
    energy_normalization: float = 1.0
    time_normalization_s: float = 1.0
    control_slew_regularization_weight: float = 1.0e-4
    corridor_deviation_regularization_weight: float = 1.0e-3
    clearance_soft_weight: float = 0.0

    def __post_init__(self) -> None:
        audited = _reaudit_wheel_sqp_platform(self.profile)
        object.__setattr__(self, "profile", audited)
        _require_frozen_wheel_sqp_values(self)
```

`_reaudit_wheel_sqp_platform()` reconstructs a new exact `PlatformProfileV2` from exact built-in fields, enforces `PlatformKindV2.WHEEL`、`simulation_proxy is False`、固定 capability/tolerances/slope，再返回 audited copy。`audit_wheel_kinematic_sqp_profile_v2()` catches only contract exceptions and returns an immutable audit result; `KeyboardInterrupt`、`SystemExit`、`MemoryError` propagate。

- [ ] **Step 4: Implement frozen typed contracts and strict invariants**

The segment remains a `RoutePrimitiveV2`; the public route is one narrow `TypedRouteV2` subclass. Base serialization remains unchanged, and Task 9 adds an explicit exact-type observation branch for the derived `samples` carrier:

```python
@dataclass(frozen=True, slots=True)
class WheelKinematicSegmentV2(RoutePrimitiveV2):
    v_mps: float
    omega_radps: float
    reverse: bool
    turn_in_place: bool
    relative_energy: float
    samples: tuple[PoseStateV2, ...]
    segment_hash: str
    segment_schema_id: str = WHEEL_KINEMATIC_SEGMENT_SCHEMA_V2

    def __post_init__(self) -> None:
        RoutePrimitiveV2.__post_init__(self)
        _validate_exact_segment_fields(self)
        if self.energy_cost != self.relative_energy:
            raise ValueError("energy_cost must equal relative_energy")
        if self.reverse is not (self.v_mps < 0.0):
            raise ValueError("reverse flag must match v_mps")
        expected_turn = self.v_mps == 0.0 and self.omega_radps != 0.0
        if self.turn_in_place is not expected_turn:
            raise ValueError("turn_in_place flag must match controls")
        if self.validation_level is not ValidationLevelV2.L2:
            raise ValueError("wheel SQP public segment must be L2")


@dataclass(frozen=True, slots=True, kw_only=True)
class WheelKinematicRouteV2(TypedRouteV2):
    route_hash: str
    source_candidate_hash: str
    request_hash: str
    profile_hash: str
    terrain_snapshot_hash: str
    capability_revision: str
    solver_contract_id: str
    canonicalization_id: str
    validator_contract_id: str

    def __post_init__(self) -> None:
        TypedRouteV2.__post_init__(self)
        _validate_exact_route_identity(self)
```

Use keyword-only dataclass subclasses so existing base contracts and hashes do not gain empty extension fields:

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class WheelSQPValidationEvidenceV2(ValidationEvidenceV2):
    route_hash: str
    candidate_hash: str
    request_hash: str
    profile_hash: str
    terrain_snapshot_hash: str
    solver_contract_id: str
    checked_interval_count: int
    checked_cell_count: int
    repair_applied: bool


@dataclass(frozen=True, slots=True, kw_only=True)
class WheelSQPSearchTelemetryV2(SearchTelemetryV2):
    corridor_count: int
    selected_corridor_index: int | None
    corridor_expansions: int
    sqp_iterations: int
    sqp_function_evaluations: int
    l2_interval_count: int
    l2_cell_count: int
    repair_attempted: bool
    decision_hash: str | None
```

`L2ReserveModelV1.reserve_s()` uses exact finite arithmetic and the frozen formula below；它为唯一完整连续 L2、严格编码以及 API 的轻量 decode/reseal 预留时间，不重复计算完整扫掠：

```python
reserve = (
    0.015
    + 0.00025 * segment_count
    + 0.000002 * broadphase_cell_bound
    + 0.000001 * interval_record_bound
    + 0.000002 * encoded_state_bound
    + 0.000001 * encoded_scalar_bound
)
```

All counts are clamped to the profile/request caps before multiplication; overflow、non-finite、negative or reserve `>= deadline.remaining_s` returns `wheel_sqp_resource_budget_exceeded` or `planning_deadline_expired`, never a candidate.

- [ ] **Step 5: Export only through opt-in v2 and run GREEN tests**

Run:

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_package_imports.py tests/test_v2_profiles.py tests/test_v2_wheel_sqp_contracts.py tests/test_v2_wheel_contracts.py -q
```

Expected: all selected tests pass; existing `WheelProfileV2` and `WheelMotionPrimitiveV2` canonical bytes remain unchanged; `path_planner.__dict__` still has no new v2 symbols.

- [ ] **Step 6: Review and commit Task 1**

Run `git diff --check` in nested and parent repositories, verify the nested diff contains only Task 1 files, then have the main agent commit the nested change:

```powershell
git add -- pyproject.toml src/path_planner/v2/profiles.py src/path_planner/v2/wheel_sqp_contracts.py src/path_planner/v2/__init__.py tests/test_package_imports.py tests/test_v2_profiles.py tests/test_v2_wheel_sqp_contracts.py
git commit -m "feat: freeze Scout Mini wheel SQP contracts"
```

Integrate only the reviewed nested gitlink in the parent with commit message `build: integrate wheel SQP contracts`.

---

### Task 2: Implement Analytic Differential-Drive Kinematics, Canonicalization, and Strict Codec

**Files:**
- Create: `path-planner/src/path_planner/v2/wheel_kinematics.py`
- Create: `path-planner/src/path_planner/v2/wheel_sqp_serialization.py`
- Create: `path-planner/tests/test_v2_wheel_kinematics.py`
- Create: `path-planner/tests/test_v2_wheel_sqp_serialization.py`
- Modify: `path-planner/tests/test_v2_wheel_sqp_contracts.py`

**Interfaces:**
- Produces: `integrate_wheel_segment_v2()`, `wheel_segment_jacobian_v2()`, `sample_wheel_segment_v2()`, `wheel_relative_energy_v1()`, `canonicalize_wheel_scalar_v2()`, `canonicalize_wheel_heading_v2()`, `materialize_canonical_wheel_candidate_v2()`, `encode_wheel_candidate_v2()`, `decode_wheel_candidate_v2()`, `encode_wheel_route_v2()`, `decode_wheel_route_v2()`, `project_wheel_route_to_candidate_v1()`, `wheel_candidate_hash_v2()`, `wheel_segment_hash_v2()`, `wheel_route_hash_v2()`。
- Consumes: solver candidate values as untrusted finite scalars plus exact request/profile/snapshot/capability identity values; it does not inspect terrain cells or import SciPy、provider or old Hybrid replay。

- [ ] **Step 1: Write RED mathematical tests for straight, arc, reverse, turn, stop, and near-zero continuity**

```python
@pytest.mark.parametrize(
    ("v", "omega", "dt", "expected"),
    [
        (1.0, 0.0, 2.0, PoseStateV2(2.0, 0.0, 0.0)),
        (-1.0, 0.0, 2.0, PoseStateV2(-2.0, 0.0, 0.0)),
        (0.0, pi / 2.0, 1.0, PoseStateV2(0.0, 0.0, pi / 2.0)),
        (0.0, 0.0, 0.05, PoseStateV2(0.0, 0.0, 0.0)),
    ],
)
def test_analytic_wheel_integration_closed_forms(v, omega, dt, expected) -> None:
    actual = integrate_wheel_segment_v2(PoseStateV2(0.0, 0.0, 0.0), v, omega, dt)
    assert_pose_close(actual, expected, abs_tol=1.0e-14)


def test_arc_uses_the_exact_sinc_form() -> None:
    end = integrate_wheel_segment_v2(PoseStateV2(1.0, 2.0, 0.3), 0.7, 0.2, 1.5)
    radius = 0.7 / 0.2
    assert end.x_m == pytest.approx(1.0 + radius * (sin(0.6) - sin(0.3)), abs=1e-14)
    assert end.y_m == pytest.approx(2.0 - radius * (cos(0.6) - cos(0.3)), abs=1e-14)
    assert end.heading_rad == pytest.approx(0.6, abs=1e-14)


def test_sinc_branch_is_continuous_at_exact_half_angle_threshold() -> None:
    threshold_omega = 2.0e-4
    below = integrate_wheel_segment_v2(START, 0.8, nextafter(threshold_omega, 0.0), 1.0)
    at = integrate_wheel_segment_v2(START, 0.8, threshold_omega, 1.0)
    above = integrate_wheel_segment_v2(START, 0.8, nextafter(threshold_omega, inf), 1.0)
    assert_pose_close(below, at, abs_tol=2.0e-15)
    assert_pose_close(at, above, abs_tol=2.0e-15)


def test_relative_energy_has_one_shared_analytic_authority() -> None:
    assert wheel_relative_energy_v1(-0.5, 0.2, 2.0, PROFILE) == pytest.approx(1.43, abs=1e-15)
```

Compare the analytic 6-column Jacobian `(start_x,start_y,start_theta,v,omega,dt)` against a central finite-difference oracle on straight、near-zero、arc、reverse and turn cases with `rtol=2e-6, atol=2e-8`.

- [ ] **Step 2: Write RED canonicalization and strict codec tests**

```python
def test_canonicalization_is_half_even_decimal12_wrap_safe_and_zero_normalized() -> None:
    assert canonicalize_wheel_scalar_v2(-0.0) == 0.0
    assert copysign(1.0, canonicalize_wheel_scalar_v2(-0.0)) == 1.0
    assert canonicalize_wheel_scalar_v2(1.2345678901235) == 1.234567890124
    assert canonicalize_wheel_heading_v2(2.0 * pi) == 0.0
    assert canonicalize_wheel_heading_v2(-pi) == -pi


def test_materialization_preserves_the_exact_unwrapped_request_start() -> None:
    request = make_request(start=PoseStateV2(0.12345678901234567, 0.75, 2.0 * pi))
    candidate = materialize_candidate_for(request)
    assert candidate.start_state == request.start_state
    assert candidate.start_state.heading_rad == 2.0 * pi


def test_candidate_codec_round_trip_is_byte_stable_and_rejects_untrusted_variants() -> None:
    candidate = make_canonical_candidate()
    encoded = encode_wheel_candidate_v2(candidate)
    decoded = decode_wheel_candidate_v2(encoded)
    assert type(decoded) is CanonicalWheelCandidateV1
    assert encode_wheel_candidate_v2(decoded) == encoded
    assert decoded.candidate_hash == wheel_candidate_hash_v2(decoded)
    assert not hasattr(decoded, "validation_level")

    for mutated in (
        add_unknown_key(encoded),
        add_duplicate_key(encoded),
        replace_number_with_bool(encoded),
        replace_number_with_nan(encoded),
        drift_solver_contract(encoded),
        drift_segment_hash(encoded),
    ):
        with pytest.raises(WheelSQPCodecError):
            decode_wheel_candidate_v2(mutated)


def test_public_route_codec_is_byte_stable_but_accepts_only_exact_l2_route() -> None:
    route = make_l2_public_route_fixture()
    encoded = encode_wheel_route_v2(route)
    decoded = decode_wheel_route_v2(encoded)
    assert type(decoded) is WheelKinematicRouteV2
    assert encode_wheel_route_v2(decoded) == encoded
    assert decoded.route_hash == wheel_route_hash_v2(decoded)
    projected = project_wheel_route_to_candidate_v1(decoded, REQUEST, PROFILE, SNAPSHOT_HASH)
    assert wheel_candidate_hash_v2(projected) == decoded.source_candidate_hash
    assert all(p.segment_hash == wheel_segment_hash_v2(p) for p in decoded.primitives)
    mutated = replace_first_segment_validation_level(encoded, "L1")
    with pytest.raises(WheelSQPCodecError):
        decode_wheel_route_v2(mutated)
```

Also assert that segment hash excludes only its own `segment_hash`, route hash excludes only its own `route_hash`, and changing `samples`、cost、solver ID、source candidate hash、request hash、profile hash、terrain hash or actual endpoint changes the correct digest. `project_wheel_route_to_candidate_v1()` strips only L2/public-only fields, recomputes analytic endpoints、relative costs and derived samples, and returns the exact private semantic projection whose hash must equal `source_candidate_hash`.

- [ ] **Step 3: Run RED tests**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_wheel_kinematics.py tests/test_v2_wheel_sqp_serialization.py tests/test_v2_wheel_sqp_contracts.py -q -x
```

Expected: import/collection failures for the two new modules; all Task 1 contract tests remain GREEN.

- [ ] **Step 4: Implement the single analytic integration authority and Jacobian**

Use half-angle integration so the straight limit is continuous. The branch is chosen only by `abs(h) <= 1.0e-4`, where `h=omega*dt/2`:

```python
def _sinc_v1(h: float) -> float:
    if abs(h) <= 1.0e-4:
        h2 = h * h
        return 1.0 - h2 / 6.0 + h2 * h2 / 120.0 - h2 * h2 * h2 / 5040.0
    return sin(h) / h


def integrate_wheel_segment_v2(
    start: PoseStateV2,
    v_mps: float,
    omega_radps: float,
    duration_s: float,
) -> PoseStateV2:
    v = _finite(v_mps, "v_mps")
    omega = _finite(omega_radps, "omega_radps")
    dt = _positive(duration_s, "duration_s")
    h = 0.5 * omega * dt
    travel = v * dt * _sinc_v1(h)
    midpoint_heading = start.heading_rad + h
    return PoseStateV2(
        start.x_m + travel * cos(midpoint_heading),
        start.y_m + travel * sin(midpoint_heading),
        start.heading_rad + 2.0 * h,
    )
```

Implement `_sinc_prime_v1()` with the matching sixth-order branch and derive the analytic Jacobian from the same operations. Do not maintain a second `omega==0` formula. Optimizer, private candidate and public route states all keep one continuous unwrapped theta sequence. `canonicalize_wheel_heading_v2()` is used only for wrap-safe angular differences; it is never applied to stored request/route states.

`sample_wheel_segment_v2()` chooses the exact count:

```python
count = max(
    1,
    ceil(abs(v_mps) * duration_s / profile.observation_sample_translation_m),
    ceil(abs(omega_radps) * duration_s / profile.observation_sample_heading_rad),
)
times = tuple(duration_s * index / count for index in range(count + 1))
```

Samples are derived observation carriers and are included in segment hash, but never consumed by continuous L2.

- [ ] **Step 5: Implement private candidate materialization and the two strict codec layers**

Use `Decimal(format(value, ".17g")).quantize(Decimal("1e-12"), rounding=ROUND_HALF_EVEN)` for finite scalars and normalize signed zero to `0.0`. Wrap only angular differences to `[-pi, pi)` for comparisons; stored state headings remain unwrapped. Materialization order is fixed:

```python
for raw in candidate.segments:
    v = canonicalize_wheel_scalar_v2(raw.v_mps)
    omega = canonicalize_wheel_scalar_v2(raw.omega_radps)
    dt = canonicalize_positive_duration_v2(raw.duration_s)
    declared_end = canonicalize_unwrapped_pose_v2(raw.end_state)
    replay_end = canonicalize_unwrapped_pose_v2(integrate_wheel_segment_v2(current, v, omega, dt))
    require_pose_residual_within(declared_end, replay_end, profile.hard_constraint_tolerance)
    segment = CanonicalWheelSegmentV1(
        start_state=current,
        end_state=replay_end,
        v_mps=v,
        omega_radps=omega,
        duration_s=dt,
        mode=canonical_mode_for(v, omega),
    )
    segments.append(segment)
    current = replay_end

canonical_candidate = materialize_canonical_wheel_candidate_v2(
    candidate,
    segments=tuple(segments),
    actual_endpoint=current,
)
candidate_bytes = encode_wheel_candidate_v2(canonical_candidate)
```

Set `current = request.start_state` without quantizing any coordinate or heading; exact request-start equality is the one state-identity exception. This order prevents independently rounded seams. Before replacing a declared endpoint by replay, the raw solver residual must already pass; canonicalization is not a repair. Every later private/public state keeps the same decimal-12 unwrapped theta sequence. `CanonicalWheelCandidateV1` and `CanonicalWheelSegmentV1` have no validation level and cannot be used as public route primitives.

Both codecs use `json.loads(..., object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant)` and exact key sets at every level. They decode to fresh exact dataclasses, recompute hashes, check `encode(decode(bytes)) == bytes`, and reject non-UTF-8、BOM、extra keys、wrong bool/int distinctions、non-finite values or version drift. Complete L2 receives only freshly decoded private candidate bytes; the public route codec accepts only an exact already-promoted L2 route and never grants L2 itself.

- [ ] **Step 6: Run GREEN, compatibility, and commit**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_wheel_kinematics.py tests/test_v2_wheel_sqp_serialization.py tests/test_v2_wheel_sqp_contracts.py tests/test_v2_serialization.py tests/test_v2_wheel_contracts.py -q
```

Expected: all pass, including old canonical serialization bytes. After focused review, the main agent commits nested files as `feat: add canonical wheel SQP trajectory codec`, then integrates the reviewed gitlink only as `build: integrate wheel SQP trajectory codec`.

---

### Task 3: Generate at Most Three Deterministic Topologically Distinct 2D Corridors

**Files:**
- Create: `path-planner/src/path_planner/v2/wheel_corridors.py`
- Create: `path-planner/tests/test_v2_wheel_corridors.py`
- Modify: `path-planner/src/path_planner/v2/wheel_sqp_contracts.py`

**Interfaces:**
- Produces: `WheelCorridorGraphV1`, `WheelBlockedComponentV1`, `WheelTopologySignatureV1`, `WheelCorridorGenerationResultV2`, `generate_wheel_corridors_v2()`。
- Consumes: current immutable `TerrainSnapshotV2`, `FineSafetyAnchorV2`, request start/goal, resource ledger and shared deadline。

- [ ] **Step 1: Write RED tests for graph safety, stable A*, topology signatures, de-duplication, and limits**

```python
def test_corridor_graph_uses_only_current_observed_center_cell_safety() -> None:
    snapshot = make_snapshot(
        unknown={Cell(2, 1)},
        hard={Cell(2, 2)},
        not_traversable={Cell(2, 3)},
        slope={Cell(2, 4): nextafter(30.0, inf)},
    )
    graph = WheelCorridorGraphV1.from_snapshot(snapshot, max_slope_deg=30.0)
    assert all(not graph.passable(cell) for cell in (Cell(2, 1), Cell(2, 2), Cell(2, 3), Cell(2, 4)))
    assert graph.passable(Cell(1, 1))


def test_diagonal_corner_cut_is_forbidden() -> None:
    graph = graph_with_blocked({Cell(1, 0), Cell(0, 1)})
    assert Cell(1, 1) not in graph.neighbors(Cell(0, 0))


def test_three_homotopy_classes_are_stably_sorted_and_near_duplicates_do_not_fill_quota() -> None:
    result = generate_wheel_corridors_v2(three_route_island_snapshot(), START, GOAL, LEDGER, DEADLINE)
    assert result.reason_code == "wheel_sqp_corridors_ready"
    assert len(result.corridors) == 3
    assert len({c.topology_signature for c in result.corridors}) == 3
    assert result.corridors == tuple(sorted(result.corridors, key=corridor_order_key_v2))
    assert [c.path_hash for c in result.corridors] == EXPECTED_PATH_HASHES


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        ("disconnected", "wheel_sqp_no_2d_corridor"),
        ("expansion_cap", "wheel_sqp_corridor_budget_exceeded"),
        ("memory_cap", "wheel_sqp_resource_budget_exceeded"),
        ("deadline", "planning_deadline_expired"),
    ],
)
def test_corridor_generation_has_stable_failure_taxonomy(failure, reason) -> None:
    assert run_corridor_failure_case(failure).reason_code == reason
```

Add metamorphic tests: repeated calls、different `PYTHONHASHSEED` subprocesses and a copied equivalent snapshot produce identical ordered `(signature,path_hash)`; changing an unrelated confidence value does not change corridors; changing observed/hard/traversable/slope safety can change them and must change snapshot identity.

- [ ] **Step 2: Run RED tests**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_wheel_corridors.py -q -x
```

Expected: collection fails because `wheel_corridors.py` does not exist.

- [ ] **Step 3: Build the exact deterministic graph and blocked components**

The passable mask is exact and intentionally does not claim footprint safety:

```python
passable = (
    snapshot.observed_mask
    & ~snapshot.hard_obstacle_mask
    & snapshot.traversable_mask
    & (snapshot.slope_deg <= 30.0)
)
```

Use eight neighbors in the frozen order `N,E,S,W,NE,SE,SW,NW`; reject a diagonal if either adjacent cardinal cell is not passable. Edge length is `0.5` or `0.5*sqrt(2)`. Freeze guide cost:

```python
edge_cost = edge_length_m * (
    1.0
    + 0.05 * (destination_slope_deg / 30.0) ** 2
    + 0.10 / (1.0 + clearance_cells)
)
```

`clearance_cells` comes from a deterministic multi-source 8-neighbor distance pass over the same invalid mask. Stable A* uses Euclidean distance as an admissible lower bound and queue key:

```python
(f_cost, g_cost, cell.y, cell.x, predecessor_rank, insertion_id)
```

All `g` updates use exact binary64 values from the fixed formula and compare with no epsilon; exact ties use the lexicographic predecessor key. Count every popped non-stale record against the shared expansion ledger and check deadline before each pop and neighbor batch.

Build invalid components with 4-connectivity, ordered by the component's sorted `(y,x)` cell tuple and SHA-256. For each component choose the boundary-facing member minimizing `(distance_to_boundary, boundary_rank, y, x)`, with boundary rank `left,top,right,bottom`, then cast the reference cut from that member's cell interior to the selected map boundary at a quarter-cell offset. This avoids path edges lying collinearly on the cut.

- [ ] **Step 4: Implement the versioned discrete topology signature and bounded Yen enumeration**

Convert the cell path to center-point segments. For every ordered component cut, count signed crossings using a half-open intersection rule: include the lower endpoint, exclude the upper endpoint; positive sign is left-to-right relative to the oriented cut. The signature is the ordered tuple:

```python
tuple((component.component_hash, signed_crossing_count) for component in components)
```

The signature code rejects tangency/vertex ambiguity by applying the frozen quarter-cell cut and exact orientation predicates; it never uses a floating epsilon to choose a class.

Enumerate at most 24 loopless guide paths with deterministic Yen spur exclusions:

```python
raw_paths = yen_k_shortest_v1(
    graph,
    start_cell,
    goal_cell,
    candidate_cap=profile.max_corridor_candidates,
    deadline=deadline,
    ledger=ledger,
)
accepted = first_path_per_signature(raw_paths)
corridors = tuple(sorted(accepted, key=corridor_order_key_v2)[:3])
```

`corridor_order_key_v2()` is exactly `(guide_cost, path_length_m, topology_signature, path_hash)`. `path_hash` is SHA-256 over source ID、snapshot hash and ordered integer cells. If a first path exists but fewer than three distinct signatures exist, return the available set; only zero paths maps to `wheel_sqp_no_2d_corridor`.

- [ ] **Step 5: Run GREEN, old A* regression, and commit**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_wheel_corridors.py tests/test_v2_terrain.py tests/test_astar.py tests/test_channel_aware_astar.py -q
```

Expected: all pass; no edits under `path_planner/search/` or old wheel provider. Main agent commits nested files as `feat: add deterministic wheel SQP corridors`, then integrates the reviewed gitlink only as `build: integrate wheel SQP corridors`.

---

### Task 4: Build One Deterministic Forward/Reverse/Turn/Stop Initial Guess per Corridor

**Files:**
- Create: `path-planner/src/path_planner/v2/wheel_sqp_initialization.py`
- Create: `path-planner/tests/test_v2_wheel_sqp_initialization.py`
- Modify: `path-planner/src/path_planner/v2/wheel_sqp_contracts.py`
- Modify: `path-planner/src/path_planner/v2/wheel_kinematics.py`

**Interfaces:**
- Produces: `simplify_wheel_corridor_v2()`, `select_wheel_modes_v2()`, `initialize_wheel_trajectory_v2()` and exact `WheelSQPInitialGuessV2`。
- Consumes: one `WheelCorridorV2`, real request start/goal, request objective, exact wheel SQP profile, resource ledger and shared deadline。

- [ ] **Step 1: Write RED tests for real endpoints, mode DP, explicit stop, turn splitting, and segment cap**

```python
def test_initializer_uses_real_endpoints_and_never_snaps_to_cell_centers() -> None:
    request = make_request(
        start=PoseStateV2(0.61, 0.74, 0.2),
        goal=PoseStateV2(5.38, 3.11, -0.4),
    )
    points = simplify_wheel_corridor_v2(CORRIDOR, request, PROFILE, LEDGER, DEADLINE)
    assert points[0] == (request.start_state.x_m, request.start_state.y_m)
    assert points[-1] == (request.goal_state.x_m, request.goal_state.y_m)
    assert points[0] != ANCHOR.cell_center(CORRIDOR.cells[0])
    assert points[-1] != ANCHOR.cell_center(CORRIDOR.cells[-1])

    guess = initialize_wheel_trajectory_v2(CORRIDOR, request, PROFILE, LEDGER, DEADLINE)
    assert guess.start_state == request.start_state
    assert guess.requested_goal == request.goal_state
    assert guess.segments[0].start_state == request.start_state


def test_reverse_is_selected_for_a_rear_goal_and_sign_change_has_positive_stop() -> None:
    guess = initialize_wheel_trajectory_v2(REAR_SWITCH_CORRIDOR, REAR_REQUEST, PROFILE, LEDGER, DEADLINE)
    modes = tuple(segment.mode for segment in guess.segments)
    assert WheelSQPModeV2.REVERSE in modes
    for left, right in pairwise(guess.segments):
        if left.v_mps * right.v_mps < 0.0:
            pytest.fail("direct sign change escaped initializer")
    sign_switches = find_direction_switches(guess.segments)
    assert sign_switches
    assert all(guess.segments[index].mode is WheelSQPModeV2.STOP for index in sign_switches)
    assert all(guess.segments[index].duration_s >= 0.05 for index in sign_switches)


def test_large_rotation_is_split_into_at_most_quarter_turn_segments() -> None:
    guess = initialize_wheel_trajectory_v2(SAME_POSITION, HEADING_PI_REQUEST, PROFILE, LEDGER, DEADLINE)
    assert all(abs(s.omega_radps) * s.duration_s <= pi / 2 for s in guess.segments)
    assert all(s.mode in {WheelSQPModeV2.TURN_LEFT, WheelSQPModeV2.TURN_RIGHT} for s in guess.segments)


def test_exact_start_goal_produces_one_positive_stationary_segment() -> None:
    guess = initialize_wheel_trajectory_v2(SINGLE_CELL, EXACT_HOLD_REQUEST, PROFILE, LEDGER, DEADLINE)
    assert len(guess.segments) == 1
    stop = guess.segments[0]
    assert stop.mode is WheelSQPModeV2.STOP
    assert (stop.v_mps, stop.omega_radps, stop.duration_s) == (0.0, 0.0, 0.05)
```

Also test: reverse/turn capability flags are sealed true; line-of-sight simplification never crosses an invalid center cell or changes the topology signature; more than 48 necessary segments returns `wheel_sqp_initialization_failed`; deadline/resource failure takes precedence; no initial segment is L0/L1/L2 or a public `RoutePrimitiveV2`.

- [ ] **Step 2: Run RED tests**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_wheel_sqp_initialization.py -q -x
```

Expected: collection fails because the initializer module does not exist.

- [ ] **Step 3: Implement topology-preserving deterministic corridor simplification**

Build world points as `[real_start] + internal_cell_centers + [real_goal]`. Starting at each kept point, choose the farthest later point whose supercover contains only graph-passable cells and whose replacement does not change the corridor's frozen topology signature; ties choose the larger path index. Preserve every forced point adjacent to a reference-cut crossing. Split any straight leg longer than `max_speed_mps * max_segment_duration_s` into equal-distance pieces, with piece count determined by exact `ceil`.

If the resulting translation、turn and required stop count exceeds 48, return `wheel_sqp_initialization_failed`; do not truncate or merge across a blocked cut.

- [ ] **Step 4: Implement the exact two-mode dynamic program and positive-duration control schedule**

For each translation leg, define two states:

```text
FORWARD: body_heading = atan2(dy, dx), v = +0.5
REVERSE: body_heading = wrap(atan2(dy, dx) + pi), v = -0.5
```

Short legs reduce `abs(v)` so `dt>=0.05`; long legs split so `dt<=120.0`. DP cost uses only the request's supported distance/time/relative-energy weights and these exact energy equations:

```python
translation_energy = abs(v) * dt * profile.translation_energy_per_m
if v < 0.0:
    translation_energy *= profile.reverse_energy_multiplier
rotation_energy = abs(omega) * dt * profile.rotation_energy_per_rad
idle_energy = dt * profile.idle_energy_per_s
segment_relative_energy = translation_energy + rotation_energy + idle_energy
```

The initializer and optimizer both call the Task 2 `wheel_relative_energy_v1()` authority; the expanded formula above is its frozen semantic definition, not a second implementation.

Transition key is `(weighted_cost, reverse_segment_count, mode_switch_count, mode_rank_sequence)` with `FORWARD` rank before `REVERSE`. Insert pure turn segments at `abs(omega)*dt<=pi/2`; choose turn sign by the wrap-safe shortest angle, with exact `+pi` tied to left. Insert a `STOP(dt=0.05)` between opposite translation signs even if the heading is unchanged.

After constructing the schedule, increase adjacent segment durations deterministically, never beyond 120 seconds, until the frozen slew inequalities pass. Whenever a duration changes, recompute the translation speed as `signed_leg_length / dt` or the pure-turn rate as `signed_turn_angle / dt`, so the geometric displacement/heading of that initializer leg is unchanged. Re-audit mode lower bounds after each adjustment; if no bounded duration/control assignment exists, fail initialization. Starting at the request's exact pose, replay every initial segment through `integrate_wheel_segment_v2()` so shooting nodes are internally connected. The request goal remains a constraint target, not an overwritten endpoint.

- [ ] **Step 5: Run GREEN and deterministic regression**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_wheel_sqp_initialization.py tests/test_v2_wheel_corridors.py tests/test_v2_wheel_kinematics.py -q
```

Run the initializer corpus twice in fresh subprocesses with `PYTHONHASHSEED=0` and `1`; compare mode sequence and initial-guess digest byte-for-byte. Expected: all pass.

- [ ] **Step 6: Review and commit**

Review that no random call、wall-clock value、old Hybrid primitive or cross-request cache enters the initializer. Main agent commits nested files as `feat: add deterministic wheel SQP initializer`, then integrates the reviewed gitlink only as `build: integrate wheel SQP initializer`.

---

### Task 5: Implement Fixed-Layout Direct Multiple-Shooting SQP and Independent Candidate Audit

**Files:**
- Create: `path-planner/src/path_planner/v2/wheel_sqp_solver.py`
- Create: `path-planner/tests/test_v2_wheel_sqp_solver.py`
- Modify: `path-planner/src/path_planner/v2/wheel_sqp_contracts.py`
- Modify: `path-planner/src/path_planner/v2/wheel_kinematics.py`

**Interfaces:**
- Produces: `WheelSQPLayoutV1`, `WheelSQPProblemV1`, `WheelSQPConstraintAuditV1`, `audit_wheel_sqp_candidate_v2()`, `solve_wheel_sqp_v2()`。
- Consumes: one exact initial guess, request objective, exact profile, optional deterministic terrain/repair constraints, shared deadline, resource ledger and precomputed L2/encoding reserve。

- [ ] **Step 1: Write RED tests for exact layout, scaling, dynamics, terminal constraints, modes, and slew**

```python
def test_layout_has_six_values_per_segment_and_excludes_fixed_start() -> None:
    layout = WheelSQPLayoutV1(segment_count=2, start_state=START)
    z = layout.pack(
        states=(Q1, Q2),
        controls=((0.5, 0.0, 1.0), (0.0, pi / 4.0, 1.0)),
    )
    assert z.tolist() == [
        Q1.x_m, Q1.y_m, Q1.heading_rad, 0.5, 0.0, 1.0,
        Q2.x_m, Q2.y_m, Q2.heading_rad, 0.0, pi / 4.0, 1.0,
    ]
    unpacked = layout.unpack(z)
    assert layout.variable_count == 12
    assert layout.start_state is START
    assert unpacked.states == (Q1, Q2)
    assert layout.pack(states=unpacked.states, controls=unpacked.controls) == pytest.approx(z, abs=0.0)


def test_dynamics_and_terminal_constraints_use_unwrapped_theta_and_closed_bounds() -> None:
    problem = make_open_problem(goal_position_tolerance_m=0.25, goal_heading_tolerance_rad=pi / 36)
    at_boundary = candidate_with_goal_errors(position=0.25, heading=pi / 36)
    audit = audit_wheel_sqp_candidate_v2(problem, at_boundary)
    assert audit.passed
    assert terminal_inequalities(problem, at_boundary)[-2:] == pytest.approx((0.0, 0.0), abs=1e-15)
    assert not audit_wheel_sqp_candidate_v2(
        problem,
        candidate_with_goal_errors(position=nextafter(0.25, inf), heading=0.0),
    ).passed


def test_mode_and_slew_contract_rejects_direct_sign_flip_and_asymmetric_rate_overflow() -> None:
    assert audit_controls(((0.5, 0.0, 1.0), (-0.5, 0.0, 1.0))).reason_code == "wheel_sqp_mode_contract_failed"
    assert audit_controls(((0.0, 0.0, 0.05), (0.5, 0.0, 0.05))).reason_code == "wheel_sqp_linear_accel_exceeded"
    assert audit_controls(((0.5, 0.0, 0.05), (0.0, 0.0, 0.05))).reason_code == "wheel_sqp_linear_decel_exceeded"
    assert audit_controls(((0.0, 0.0, 1.0), (0.0, pi, 1.0))).reason_code == "wheel_sqp_angular_slew_exceeded"


def test_each_segment_heading_change_is_at_most_pi_over_two() -> None:
    assert audit_controls(((0.0, pi / 4, 2.0),)).passed
    assert audit_controls(((0.0, nextafter(pi / 4, inf), 2.0),)).reason_code == "wheel_sqp_segment_heading_change_exceeded"
```

Test normalized scales exactly: relative `(x,y)` scale `1m`, theta scale `pi`, `v` scale `1m/s`, omega scale `pi/4rad/s`, duration scale `1s`. No scale may depend on path length、map size、candidate set or corridor.

- [ ] **Step 2: Write RED backend and post-solve audit tests**

```python
def test_missing_or_wrong_scipy_is_typed_backend_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(solver, "_load_scipy_optimize_v1", lambda: None)
    result = solve_wheel_sqp_v2(PROBLEM, DEADLINE, LEDGER)
    assert result.reason_code == "wheel_sqp_backend_unavailable"
    assert result.candidate is None


def test_slsqp_success_flag_cannot_bypass_independent_residual_audit(monkeypatch) -> None:
    monkeypatch.setattr(solver, "_run_slsqp_v1", lambda *args, **kwargs: fake_success_with_bad_dynamics())
    result = solve_wheel_sqp_v2(PROBLEM, DEADLINE, LEDGER)
    assert result.reason_code == "wheel_sqp_numeric_contract_failed"
    assert result.candidate is None


@pytest.mark.parametrize(
    ("abort", "reason"),
    [
        ("deadline", "planning_deadline_expired"),
        ("evaluation", "wheel_sqp_resource_budget_exceeded"),
        ("memory", "wheel_sqp_resource_budget_exceeded"),
        ("nonfinite", "wheel_sqp_numeric_contract_failed"),
        ("infeasible", "wheel_sqp_infeasible"),
    ],
)
def test_solver_abort_taxonomy_never_returns_partial_candidate(abort, reason) -> None:
    result = run_solver_abort_case(abort)
    assert result.reason_code == reason
    assert result.candidate is None
```

- [ ] **Step 3: Run RED tests**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_wheel_sqp_solver.py -q -x
```

Expected: collection fails because the solver module does not exist.

- [ ] **Step 4: Implement fixed layout, normalized residuals, objective, and analytic Jacobians**

For `N` segments, freeze this order and never expose a backend-dependent permutation:

```text
z[6*k:6*k+6] = [x[k+1], y[k+1], theta_unwrapped[k+1], v[k], omega[k], dt[k]]
```

`q0` is the exact request start parameter. Dynamics equalities are the scaled difference between the declared next state and `integrate_wheel_segment_v2(qk,v,omega,dt)`. Terminal inequalities are:

```python
position_margin = 0.25**2 - (xN - x_goal)**2 - (yN - y_goal)**2
heading_margin = cos(thetaN - theta_goal) - cos(0.08726646259971647)
```

Mode bounds are fixed from the initializer sequence:

```text
FORWARD:   1.0e-4 <= v <= 1.0; omega is free within global bounds
REVERSE:  -1.0 <= v <= -1.0e-4; omega is free within global bounds
TURN_LEFT: v == 0.0; 1.0e-4 <= omega <= pi/4
TURN_RIGHT:v == 0.0; -pi/4 <= omega <= -1.0e-4
STOP:      v == 0.0; omega == 0.0
```

All modes enforce `0.05<=dt<=120.0` and `abs(omega)*dt<=pi/2`. Slew inequalities use the frozen speed-magnitude contract. Implement exact analytic Jacobians from `wheel_segment_jacobian_v2()`; tests compare every row against finite differences.

Reject `request.objective_profile.risk_weight != 0.0` before loading SciPy. The request objective is:

```python
distance = sum(abs(v) * dt for v, _, dt in controls)
relative_energy = sum(wheel_relative_energy_v1(v, omega, dt, profile) for v, omega, dt in controls)
duration = sum(dt for _, _, dt in controls)
request_cost = (
    objective.distance_weight * distance
    + objective.energy_weight * relative_energy / profile.energy_normalization
    + objective.time_weight * duration / profile.time_normalization_s
)
regularization = (
    1.0e-4 * squared_control_slew(controls)
    + 1.0e-3 * squared_corridor_node_deviation(states, corridor)
)
```

`clearance_soft_weight` is exactly zero；地图/坡度/净空只能作为 hard constraints，不能用代价换取违反。

- [ ] **Step 5: Implement the private lazy SLSQP adapter and deadline/resource abort**

`_load_scipy_optimize_v1()` imports only inside the explicit new provider path, verifies `scipy.__version__ == "1.18.0"`, and returns `scipy.optimize.minimize`、`Bounds`、`NonlinearConstraint`. Missing/wrong version maps to `wheel_sqp_backend_unavailable` without affecting old wheel imports.

Call the backend exactly once per attempt:

```python
raw = minimize(
    objective_fn,
    z0,
    method="SLSQP",
    jac=objective_jacobian,
    bounds=bounds,
    constraints=(equality_constraint, inequality_constraint),
    callback=iteration_callback,
    options={"ftol": 1.0e-10, "maxiter": 40, "disp": False},
)
```

Every objective、constraint、Jacobian and callback entry first checks: finite input, evaluation cap, memory ledger, deadline, and `deadline.remaining_s > reserve_s`. Abort with a private `_WheelSQPAbort(reason_code)` caught separately from unexpected exceptions. Do not set workers or start threads.

Record explicitly in code comments and tests: the compiled inner QP call is not Python-preemptible. The fixed 48-segment/40-iteration/4096-evaluation bounds and formal runtime gate are therefore promotion requirements；a late return after deadline always becomes timeout even if SciPy reports success.

- [ ] **Step 6: Independently audit raw output before creating an internal candidate**

Ignore `raw.success` as authority. Recompute all constraints and costs from a copied finite vector in fixed order. Candidate creation requires:

```python
max_abs_scaled_dynamics_residual <= 1.0e-9
min_inequality_margin >= -1.0e-9
all_mode_and_slew_checks_pass is True
deadline.expired is False
evaluations <= 4096
ledger.within_limits is True
```

The returned object is internal `WheelSQPCandidateV2`, not `RoutePrimitiveV2` and not L0/L1/L2. Preserve raw solver status only in non-semantic diagnostic telemetry; public decision hash uses audited normalized fields and reason only.

- [ ] **Step 7: Run GREEN and commit**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_wheel_sqp_solver.py tests/test_v2_wheel_sqp_initialization.py tests/test_v2_wheel_kinematics.py -q
```

Expected: all pass with no new skips. Main agent commits nested files as `feat: add deterministic wheel multiple shooting SQP`, then integrates the reviewed gitlink only as `build: integrate wheel SQP solver`.

---

### Task 6: Add Deterministic Terrain Hard Constraints and Reserve/Memory Admission

**Files:**
- Modify: `path-planner/src/path_planner/v2/wheel_sqp_solver.py`
- Modify: `path-planner/src/path_planner/v2/wheel_corridors.py`
- Modify: `path-planner/src/path_planner/v2/wheel_sqp_contracts.py`
- Modify: `path-planner/tests/test_v2_wheel_sqp_solver.py`
- Create: `path-planner/tests/test_v2_wheel_sqp_resources.py`

**Interfaces:**
- Produces: `WheelSQPTerrainGuideV1`, `WheelSQPResourceEstimateV1`, `build_wheel_sqp_terrain_constraints_v2()`, `estimate_wheel_sqp_attempt_resources_v2()`。
- Consumes: current snapshot only, corridor, initial mode schedule, request resource budget, exact profile and shared deadline。

- [ ] **Step 1: Write RED tests for the optimizer's hard terrain approximation and admission boundaries**

```python
def test_optimizer_terrain_set_matches_fine_safety_anchor_reason_set() -> None:
    guide = WheelSQPTerrainGuideV1.from_snapshot(
        make_snapshot(
            unknown={Cell(1, 1)},
            hard={Cell(2, 1)},
            not_traversable={Cell(3, 1)},
            slope={Cell(4, 1): nextafter(30.0, inf)},
        ),
        max_slope_deg=30.0,
    )
    assert guide.invalid_cells == (Cell(1, 1), Cell(2, 1), Cell(3, 1), Cell(4, 1))
    assert Cell(5, 1) not in guide.invalid_cells


def test_fixed_fraction_clearance_detects_mid_segment_obstacle_when_nodes_are_clear() -> None:
    problem = problem_with_obstacle_between_nodes()
    margins = problem.terrain_inequalities(problem.initial_vector)
    assert min(margins) < 0.0


def test_optimizer_clearance_uses_circumradius_as_a_sufficient_not_authoritative_bound() -> None:
    guide = make_guide()
    radius = hypot(0.612, 0.580) / 2.0
    assert guide.center_clearance_margin(point_at_distance(radius), HEADING) == pytest.approx(0.0)
    assert guide.center_clearance_margin(point_at_distance(nextafter(radius, 0.0)), HEADING) < 0.0


def test_attempt_does_not_start_when_l2_encoding_reserve_or_memory_is_insufficient(monkeypatch) -> None:
    monkeypatch.setattr(solver, "_run_slsqp_v1", forbidden_call)
    result = solve_wheel_sqp_v2(
        make_problem(),
        deadline_with_remaining(RESERVE_S),
        ledger_with_memory(PROFILE.solver_memory_reservation_bytes - 1),
    )
    assert result.reason_code in {"planning_deadline_expired", "wheel_sqp_resource_budget_exceeded"}
    assert result.candidate is None
```

Add exact `nextafter` tests for reserve、memory、candidate-cell、interval-record and route-state caps. Zero `max_memory_bytes` retains the existing v2 meaning “no caller memory limit” but still obeys fixed internal caps.

- [ ] **Step 2: Run RED tests**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_wheel_sqp_resources.py tests/test_v2_wheel_sqp_solver.py -q -x
```

Expected: new resource tests fail because terrain guide/admission interfaces do not exist; prior solver tests remain GREEN.

- [ ] **Step 3: Implement the deterministic invalid-cell index and conservative optimizer clearance**

The optimizer invalid set is exactly:

```python
invalid = (
    ~snapshot.observed_mask
    | snapshot.hard_obstacle_mask
    | ~snapshot.traversable_mask
    | (snapshot.slope_deg > 30.0)
)
```

Index invalid cell axis-aligned rectangles in stable `(row,column)` order. Freeze `terrain_reason_rank` as `out_of_bounds=0, unknown=1, hard_obstacle=2, not_traversable=3, slope_exceeded=4`; a cell matching multiple masks uses the first applicable rank. At each optimizer state and fixed analytic fractions `rho=(0.0,0.25,0.5,0.75,1.0)`, compute the exact Euclidean distance from vehicle center to the nearest invalid cell rectangle and to the map exterior. Tie nearest cells by `(terrain_reason_rank,row,column)`. Use the sufficient hard inequality:

```python
center_clearance_m - hypot(
    profile.body_length_m / 2.0 + profile.footprint_safety_margin_m,
    profile.body_width_m / 2.0 + profile.footprint_safety_margin_m,
) >= 0.0
```

Use the active nearest rectangle's analytic gradient; at an exact tie use the frozen first rectangle and record the tie count in non-semantic telemetry. This constraint may conservatively reject a feasible oriented rectangle but can never authorize L2. Corridor deviation remains a regularizer and cannot replace these inequalities.

- [ ] **Step 4: Implement exact work/memory estimates and two-stage reserve checks**

Before each original or repair SQP attempt, compute:

```python
decision_bytes = 8 * (6 * segment_count)
jacobian_bytes = 8 * constraint_count * (6 * segment_count)
solver_bytes = profile.solver_memory_reservation_bytes
l2_queue_bytes = 96 * interval_record_bound
codec_bytes = 64 * encoded_state_bound + encoded_scalar_bound * 16
required_bytes = decision_bytes + jacobian_bytes + solver_bytes + l2_queue_bytes + codec_bytes
```

Use checked integer arithmetic and reject overflow before allocation. `broadphase_cell_bound` is the stable sum of corridor leg AABB cells expanded by the footprint circumradius and clamped to `1_000_000`; `interval_record_bound` is `min(262_144, broadphase_cell_bound * ((1 << 25) - 1))` computed with saturation before multiplication. `encoded_state_bound` is derived from observation sampling and must be `<=request.resource_budget.max_route_states`.

Admission sequence is fixed:

1. before SLSQP: require memory/work caps and full `L2 + encode + API reseal` time reserve;
2. after candidate canonicalization, before L2: recompute actual broad-phase/state bounds and require `L2 + encode + API reseal` reserve;
3. during L2: continue checking deadline and keep the exact codec/reseal tail reserve;
4. after L2: any expired deadline replaces success with typed timeout.

`remaining_s <= reserve_s` is insufficient and fails before starting the next phase. Reserve arithmetic never grants success; it only prevents starting work that cannot leave a validation/encoding tail.

- [ ] **Step 5: Run GREEN and commit**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_wheel_sqp_resources.py tests/test_v2_wheel_sqp_solver.py tests/test_v2_wheel_corridors.py tests/test_v2_runtime.py -q
```

Expected: all pass. Review allocation-before-cap ordering and verify no unbounded Python list/array is created first. Main agent commits nested files as `feat: bind wheel SQP terrain and resource limits`, then integrates the reviewed gitlink only as `build: integrate wheel SQP resource limits`.

---

### Task 7: Implement Strict Serialized-Candidate Replay and Complete Continuous Rectangle-Sweep L2

**Files:**
- Create: `path-planner/src/path_planner/v2/wheel_sqp_validation.py`
- Create: `path-planner/tests/test_v2_wheel_sqp_validation.py`
- Modify: `path-planner/src/path_planner/v2/wheel_sqp_serialization.py`
- Modify: `path-planner/src/path_planner/v2/wheel_sqp_contracts.py`
- Modify: `path-planner/src/path_planner/v2/wheel_kinematics.py`

**Interfaces:**
- Produces: `validate_wheel_sqp_candidate_l2()`, `oriented_rectangle_cell_separation_v2()`, `prove_wheel_segment_sweep_v2()`, `promote_wheel_candidate_to_route_v2()`。
- Consumes: immutable canonical candidate bytes, exact request/anchor/profile, shared deadline, resource ledger and codec/reseal tail reserve；it does not import SciPy、initializer、corridor generator or provider。

- [ ] **Step 1: Write RED tests for strict decode, analytic replay, controls, cost, and endpoint contracts**

```python
def test_l2_decodes_fresh_candidate_and_replays_from_exact_request_start() -> None:
    candidate_bytes = encode_wheel_candidate_v2(make_candidate())
    result = validate_wheel_sqp_candidate_l2(candidate_bytes, REQUEST, ANCHOR, PROFILE, DEADLINE, LEDGER)
    assert result.passed
    assert result.actual_start == REQUEST.start_state
    assert result.actual_goal != REQUEST.goal_state  # no snap
    assert result.goal_position_error_m <= 0.25
    assert result.goal_heading_error_rad <= pi / 36


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ("node_disconnect", "wheel_sqp_candidate_l2_rejected"),
        ("wrong_replay_end", "wheel_sqp_candidate_l2_rejected"),
        ("nonpositive_dt", "wheel_sqp_numeric_contract_failed"),
        ("direct_sign_flip", "wheel_sqp_candidate_l2_rejected"),
        ("accel", "wheel_sqp_candidate_l2_rejected"),
        ("decel", "wheel_sqp_candidate_l2_rejected"),
        ("angular_slew", "wheel_sqp_candidate_l2_rejected"),
        ("heading_change", "wheel_sqp_candidate_l2_rejected"),
        ("cost", "wheel_sqp_identity_mismatch"),
        ("solver_id", "wheel_sqp_identity_mismatch"),
        ("snapshot_hash", "wheel_sqp_identity_mismatch"),
    ],
)
def test_l2_rejects_contract_drift_without_route_or_repair(mutation, reason) -> None:
    result = validate_mutated_candidate(mutation)
    assert result.reason_code == reason
    assert result.route is None
    assert result.counterexample is None
```

Candidate serialization must use private `CanonicalWheelCandidateV1` / `CanonicalWheelSegmentV1` types without `validation_level`; public `WheelKinematicSegmentV2` is created only after L2 passes.

- [ ] **Step 2: Write RED adversarial continuous-sweep tests**

```python
def test_nodes_and_observation_samples_safe_but_mid_arc_collision_is_rejected() -> None:
    candidate = arc_whose_declared_nodes_and_samples_miss(Cell(4, 3))
    result = validate_candidate(candidate, hard={Cell(4, 3)})
    assert not result.passed
    assert result.counterexample is not None
    assert result.counterexample.cell == Cell(4, 3)
    assert result.counterexample.repairable is True


def test_rotation_corner_sweep_and_single_point_tangency_are_unsafe() -> None:
    for candidate in (quarter_turn_corner_touch(), exact_cell_tangent()):
        result = validate_candidate(candidate)
        assert not result.passed
        assert result.reason_code == "wheel_sqp_candidate_l2_rejected"


@pytest.mark.parametrize(
    ("terrain_case", "terrain_reason"),
    [
        ("out_of_bounds", "terrain_out_of_bounds"),
        ("unknown", "terrain_unknown"),
        ("hard", "terrain_hard_obstacle"),
        ("not_traversable", "terrain_not_traversable"),
        ("slope", "terrain_slope_exceeded"),
    ],
)
def test_complete_footprint_uses_fine_safety_reason_set(terrain_case, terrain_reason) -> None:
    result = validate_terrain_case(terrain_case)
    assert result.counterexample.terrain_reason == terrain_reason


def test_slope_boundary_is_closed() -> None:
    assert validate_slope(30.0).passed
    assert not validate_slope(nextafter(30.0, inf)).passed
```

Add cases where no sampled pose collides but the interval does; where an arc extrema broad phase includes a cell omitted by endpoint AABB; where an unproven interval hits depth/record/cell/memory/deadline cap and fails closed with `counterexample.repairable is False`; and where repeated validation yields the exact same first counterexample.

- [ ] **Step 3: Run RED tests**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_wheel_sqp_validation.py -q -x
```

Expected: collection fails because the independent validator does not exist.

- [ ] **Step 4: Complete the pre-L2 candidate codec and post-L2 public route promotion boundary**

Use the two explicit codec layers frozen in Task 2 and add the only authorized promotion operation:

```python
canonical_candidate = materialize_canonical_wheel_candidate_v2(
    solved_candidate,
    request=request,
    profile=profile,
    terrain_snapshot_hash=snapshot_hash(request.terrain_snapshot),
)
candidate_bytes = encode_wheel_candidate_v2(canonical_candidate)
decoded_candidate = decode_wheel_candidate_v2(candidate_bytes)
l2_result = validate_wheel_sqp_candidate_l2(
    candidate_bytes,
    request,
    anchor,
    profile,
    deadline,
    ledger,
)
public_route = promote_wheel_candidate_to_route_v2(decoded_candidate, l2_result)
public_bytes = encode_wheel_route_v2(public_route)
```

Candidate fields include request/profile/snapshot/capability/solver/canonicalization/control-slew IDs, exact start、requested goal、canonical controls/nodes、relative costs and candidate hashes. They contain no L0/L1/L2 field and cannot satisfy `PlanningSuccessV2`.

Promotion is a pure operation that requires an exact passed `WheelTrajectoryL2ResultV2` bound to the decoded candidate hash. It derives public samples, constructs every `WheelKinematicSegmentV2(validation_level=L2)`, computes segment hashes, then constructs `WheelKinematicRouteV2` with `source_candidate_hash` plus exact request/profile/snapshot/capability/solver/canonicalization/validator identities and its non-self-referential route hash. `WheelSQPValidationEvidenceV2.candidate_hash` and `.route_hash` bind both sides of the operation. Immediately project the new public route back through `project_wheel_route_to_candidate_v1()` and require its recomputed hash to equal the decoded candidate、L2 receipt、route and evidence candidate hashes before returning it. Any deadline expiry before public encoding returns timeout, not a late route.

- [ ] **Step 5: Implement structural, identity, kinematic, resource, cost, and terminal L2**

Decode a fresh object and rederive request、profile and snapshot hashes independently. Recompute every analytic endpoint from the previous decoded endpoint; check exact start, connectivity, positive monotonic time, finite values, mode/STOP/sign rules, speed/omega/duration/90° bounds, discrete slew, route-state budget, relative energy, distance `abs(v)*dt`, weighted cost and real terminal errors. Use closed `<=` comparisons at the two goal tolerances.

Hash the input bytes before and after validation and verify all identity fields again before returning. Only expected contract errors become stable validation results; `KeyboardInterrupt`、`SystemExit`、`MemoryError` propagate.

- [ ] **Step 6: Implement exact-pose SAT separation and interval proof**

Let half dimensions including margin be `A=0.306+margin`、`B=0.290+margin`, and cell half-width `r=0.25`. At an exact pose, compute separation over world and body axes:

```python
axes = (
    (1.0, 0.0),
    (0.0, 1.0),
    (cos(theta), sin(theta)),
    (-sin(theta), cos(theta)),
)
gap = max(
    abs(dot(center - cell_center, axis))
    - (
        A * abs(dot(body_forward, axis))
        + B * abs(dot(body_lateral, axis))
        + r * (abs(axis[0]) + abs(axis[1]))
    )
    for axis in axes
)
```

`gap<=0.0` means overlap or contact and is unsafe. Build an analytic center-path AABB including circular-arc quadrant extrema, expand by `hypot(A,B)`, and enumerate candidate invalid cells before interval records.

For interval `[ta,tb]`, midpoint `tm` and half-width `H`, use:

```python
D = hypot(center_tm.x - cell_center.x, center_tm.y - cell_center.y) + abs(v) * H
L = abs(v) + abs(omega) * (D + A + B + 2.0 * r)
proved_separate = gap_at_midpoint > L * H + 1.0e-9
```

For each segment/candidate-cell pair, evaluate exact poses at `t=0` and `t=dt` first; any endpoint `gap<=0` is a known collision/contact counterexample. Then process the open interior interval. If proved, accept that cell/interval. If an exact midpoint has `gap<=0`, emit a repairable collision/contact counterexample. Otherwise subdivide left then right. Use one priority queue ordered by `(segment_index,time_lo,terrain_reason_rank,row,column,time_hi)` so the first counterexample is independent of set/dict iteration. OOB uses four versioned boundary pseudo-cells and the same proof pattern.

When depth 24、262,144 interval records、1,000,000 candidate cells、memory、numeric or deadline cap prevents proof, fail closed as unproven/resource/timeout and set `repairable=False`; do not reinterpret uncertainty as a known collision.

- [ ] **Step 7: Build the stable counterexample and passed receipt**

Known collision/contact counterexamples contain exactly:

```text
segment_index, candidate_segment_hash,
time_fraction_lo, time_fraction_mid, time_fraction_hi,
cell, terrain_reason, snapshot_hash, candidate_hash, repairable
```

Passed results carry actual terminal pose/errors, checked cells/intervals, candidate hash and all identity hashes. They do not yet carry a public route hash; promotion computes that hash and builds `WheelSQPValidationEvidenceV2` with the same receipt plus both candidate/public route hashes.

- [ ] **Step 8: Run GREEN, adversarial regression, and commit**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_wheel_sqp_validation.py tests/test_v2_wheel_sqp_serialization.py tests/test_v2_wheel_kinematics.py tests/test_v2_terrain.py tests/test_v2_geometry.py -q
```

Expected: all pass, no safety test skipped. Main agent obtains a fresh adversarial review, commits nested files as `feat: add continuous wheel rectangle L2`, then integrates the reviewed gitlink only as `build: integrate continuous wheel L2`.

---

### Task 8: Add One Deterministic Counterexample Repair and the First-Passing-Route Provider

**Files:**
- Create: `path-planner/src/path_planner/v2/providers/wheel_sqp.py`
- Create: `path-planner/tests/test_v2_wheel_sqp_provider.py`
- Modify: `path-planner/src/path_planner/v2/wheel_sqp_solver.py`
- Modify: `path-planner/src/path_planner/v2/wheel_sqp_validation.py`
- Modify: `path-planner/src/path_planner/v2/wheel_sqp_contracts.py`

**Interfaces:**
- Produces: `WheelKinematicSQPProviderV2`, `WheelSQPRepairConstraintV1`, `repair_constraint_from_counterexample_v2(counterexample: WheelL2CounterexampleV2, canonical_candidate_bytes: bytes, profile: WheelKinematicSQPProfileV2) -> WheelSQPRepairConstraintV1`。
- Consumes: exact profile and the already implemented corridor→initializer→solver→codec→L2 pipeline。

- [ ] **Step 1: Write RED tests for repair eligibility, separating-face choice, and three repair samples**

```python
def test_known_collision_selects_least_penetrating_face_with_frozen_tie_break() -> None:
    counterexample = known_collision_at_cell_center()
    repair = repair_constraint_from_counterexample_v2(counterexample, CANDIDATE_BYTES, PROFILE)
    assert repair.face in {"left", "right", "bottom", "top"}
    assert repair.face == "left"  # exact symmetric tie rank
    assert repair.time_fractions == (
        counterexample.time_fraction_lo,
        counterexample.time_fraction_mid,
        counterexample.time_fraction_hi,
    )
    assert repair.clearance_m == 1.0e-4


@pytest.mark.parametrize(
    "reason",
    ["unproven", "numeric", "identity", "kinematic", "resource", "deadline", "reseal"],
)
def test_non_geometric_or_unproven_failure_cannot_create_repair(reason) -> None:
    with pytest.raises(WheelSQPRepairNotAllowed):
        repair_constraint_from_counterexample_v2(nonrepairable_counterexample(reason), CANDIDATE_BYTES, PROFILE)
```

The face margins are exact oriented-rectangle support inequalities. For example, left of a cell:

```python
cell_left_x - center_x - (
    A * abs(cos(theta)) + B * abs(sin(theta))
) - 1.0e-4 >= 0.0
```

The helper strictly decodes `canonical_candidate_bytes`, verifies its hash equals the counterexample candidate hash, and never accepts a mutable solver object. Right、bottom、top use the corresponding signed axis and support. Choose the current candidate face with the largest margin (least penetration), ties `left,right,bottom,top`. Add that same face at counterexample `rho_lo,rho_mid,rho_hi`; do not choose a new face during SQP callbacks.

- [ ] **Step 2: Write RED provider orchestration tests**

```python
def test_first_full_l2_route_returns_immediately_without_later_corridors(monkeypatch) -> None:
    calls = install_three_corridor_fakes(monkeypatch, outcomes=("l2_pass", forbidden_call, forbidden_call))
    outcome = PROVIDER.plan(REQUEST, ANCHOR, DEADLINE)
    assert type(outcome) is PlanningSuccessV2
    assert type(outcome.route) is WheelKinematicRouteV2
    assert calls.optimized_corridors == [0]
    assert outcome.search_telemetry.selected_corridor_index == 0


def test_one_known_l2_counterexample_gets_exactly_one_repair(monkeypatch) -> None:
    calls = install_repair_fakes(monkeypatch, first="known_collision", repair="l2_pass")
    outcome = PROVIDER.plan(REQUEST, ANCHOR, DEADLINE)
    assert type(outcome) is PlanningSuccessV2
    assert calls.original_attempt_count == 1
    assert calls.repair_attempt_count == 1
    assert outcome.validation_evidence.repair_applied is True


def test_failed_repair_moves_to_next_corridor_and_never_repairs_twice(monkeypatch) -> None:
    calls = install_three_corridor_fakes(
        monkeypatch,
        outcomes=("known_collision_then_known_collision", "l2_pass", forbidden_call),
    )
    outcome = PROVIDER.plan(REQUEST, ANCHOR, DEADLINE)
    assert type(outcome) is PlanningSuccessV2
    assert calls.repair_counts == {0: 1, 1: 0}


def test_provider_source_contains_no_hybrid_fallback() -> None:
    source = inspect.getsource(wheel_sqp_provider_module)
    assert "HybridAStarPlanner" not in source
    assert "providers.wheel" not in source
```

Add provider tests for exact STOP success、real endpoint within but not equal to target、full-footprint unsafe start/goal、unsupported risk as `wheel_sqp_objective_unsupported`、required accelerator as `wheel_sqp_accelerator_required_unsupported`、missing SciPy as `wheel_sqp_backend_unavailable`、corridor/init/SQP/L2/resource/deadline mappings、all failures without route、unexpected exception type only、no state retained between calls and same decision hash across repeat calls. API integration tests separately prove the five existing generic `terrain_*` endpoint reasons remain unchanged when the center cell itself fails.

- [ ] **Step 3: Run RED tests**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_wheel_sqp_provider.py tests/test_v2_wheel_sqp_solver.py -q -x
```

Expected: provider import fails and repair tests fail; all previous solver tests remain GREEN.

- [ ] **Step 4: Implement the immutable provider and preflight**

Provider stores only the exact profile:

```python
@dataclass(frozen=True, slots=True)
class WheelKinematicSQPProviderV2:
    wheel_sqp_profile: WheelKinematicSQPProfileV2

    @property
    def profile(self) -> PlatformProfileV2:
        return self.wheel_sqp_profile.profile

    def plan(self, request, anchor, deadline) -> PlanningOutcomeV2:
        return _plan_wheel_sqp_v2(self, request, anchor, deadline)
```

Preflight order is exact: request/anchor/deadline types → profile deep reaudit → request profile/snapshot identity → optional SciPy version → supported objective (`risk_weight==0`) → accelerator policy (`REQUIRED` unsupported) → resource ledger → full rectangle stationary start → full rectangle stationary goal → exact start/goal special case → corridor generation. Deadline has precedence after every bounded step.

The start/goal footprint check calls a focused validator helper that uses the same rectangle/cell SAT and FineSafetyAnchor reason priority as route L2. API's existing center-cell check remains unchanged; this provider check can only reject additional unsafe footprint cases.

- [ ] **Step 5: Implement the exact corridor loop and one-repair state machine**

```python
for corridor_index, corridor in enumerate(corridors):
    require_attempt_reserve(corridor, request, profile, deadline, ledger)
    initial = initialize_wheel_trajectory_v2(corridor, request, profile, ledger, deadline)
    solved = solve_wheel_sqp_v2(problem(initial), deadline, ledger)
    if solved.candidate is None:
        record_failure(solved)
        continue

    candidate_bytes = canonicalize_and_encode_candidate(solved.candidate, request, profile)
    l2 = validate_wheel_sqp_candidate_l2(candidate_bytes, request, anchor, profile, deadline, ledger)
    if l2.passed:
        return build_success(candidate_bytes, l2, corridor_index, repair_applied=False)

    if l2.counterexample is not None and l2.counterexample.repairable:
        require_attempt_reserve(corridor, request, profile, deadline, ledger)
        repair = repair_constraint_from_counterexample_v2(l2.counterexample, candidate_bytes, profile)
        repaired = solve_wheel_sqp_v2(problem(initial, repair=repair), deadline, ledger)
        if repaired.candidate is None:
            record_repair_failure(repaired)
            continue
        repaired_bytes = canonicalize_and_encode_candidate(repaired.candidate, request, profile)
        repaired_l2 = validate_wheel_sqp_candidate_l2(
            repaired_bytes, request, anchor, profile, deadline, ledger
        )
        if repaired_l2.passed:
            return build_success(repaired_bytes, repaired_l2, corridor_index, repair_applied=True)
        record_repair_failure(repaired_l2)
```

No loop may execute repair twice. An unproven/cap/numeric/identity/deadline result skips repair. If any deadline check expires, return timeout immediately rather than trying the next corridor.

If all corridors fail, choose final reason by stable precedence: deadline → identity/numeric/internal → resource → repair L2 rejected → candidate L2 rejected → infeasible → initialization failed. Never return the “last exception wins” implicitly.

- [ ] **Step 6: Construct exact public success, cost, observation carrier, and telemetry**

Promotion creates the route once. Compute `CostBreakdownV2` from analytic arc distance、relative energy and duration; risk is zero. `ObservationProjectionV2` uses the route's derived samples and source `wheel_kinematic_derived_samples/v1`; it does not query unknown truth or grant safety. Cache evidence is fixed disabled with a new versioned namespace and `hit=False` because cross-request reuse is out of scope.

Populate `WheelSQPSearchTelemetryV2` with stable counts but exclude elapsed time from `decision_hash`. Always set `accelerator_used=False` and `ackermann_feasible_claimed=False`. The decision hash includes ordered corridor signatures/hashes、mode sequence、canonical candidate bytes hash、L2 result/counterexample/repair choice、final reason and route hash.

- [ ] **Step 7: Run GREEN, old wheel compatibility, and commit**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_wheel_sqp_provider.py tests/test_v2_wheel_sqp_validation.py tests/test_v2_wheel_provider.py tests/test_v2_wheel_contracts.py -q
```

Expected: all pass. Obtain fresh spec/quality/adversarial review; main agent commits nested files as `feat: add Scout Mini corridor SQP provider`, then integrates the reviewed gitlink only as `build: integrate Scout Mini SQP provider`.

---

### Task 9: Add Capability-Specific API Seal, Lightweight Reseal, Observation Compatibility, and Opt-In Exports

**Files:**
- Create: `path-planner/src/path_planner/v2/wheel_sqp_api.py`
- Create: `path-planner/tests/test_v2_wheel_sqp_api.py`
- Modify: `path-planner/src/path_planner/v2/api.py`
- Modify: `path-planner/src/path_planner/v2/observation.py`
- Modify: `path-planner/src/path_planner/v2/providers/__init__.py`
- Modify: `path-planner/src/path_planner/v2/__init__.py`
- Modify: `path-planner/tests/test_v2_api.py`
- Modify: `path-planner/tests/test_v2_observation_projection.py`

**Interfaces:**
- Produces: `dispatch_wheel_sqp_provider_v2()` and explicit v2 exports for the new profile/provider/route/segment IDs and types。
- Consumes: exact new provider, exact base profile, request/anchor/shared deadline; it never invokes the old wheel provider or a second complete continuous L2。

- [ ] **Step 1: Write RED tests for explicit dispatch and unchanged old paths**

```python
def test_new_capability_requires_explicit_profile_and_provider_registration() -> None:
    request, registry, provider = make_wheel_sqp_api_fixture()
    missing = plan_v2(request, registry=registry, providers={})
    assert missing.reason_code == "primitive_provider_unregistered"

    success = plan_v2(
        request,
        registry=registry,
        providers={provider.profile.profile_id: provider},
    )
    assert type(success) is PlanningSuccessV2
    assert type(success.route) is WheelKinematicRouteV2
    assert all(type(p) is WheelKinematicSegmentV2 for p in success.route.primitives)


def test_old_wheel_capability_still_uses_old_provider(monkeypatch) -> None:
    monkeypatch.setattr(wheel_sqp_api, "dispatch_wheel_sqp_provider_v2", forbidden_call)
    outcome = plan_v2(OLD_WHEEL_REQUEST, registry=OLD_REGISTRY, providers=OLD_PROVIDERS)
    assert type(outcome) is PlanningSuccessV2
    assert type(outcome.route.primitives[0]) is WheelMotionPrimitiveV2


def test_sqp_failure_never_calls_hybrid_provider(monkeypatch) -> None:
    monkeypatch.setattr(wheel_module.WheelPrimitiveProviderV2, "plan", forbidden_call)
    outcome = plan_v2(SQP_UNREACHABLE_REQUEST, registry=SQP_REGISTRY, providers=SQP_PROVIDERS)
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code.startswith("wheel_sqp_")
```

- [ ] **Step 2: Write RED forgery, drift, codec, and deadline postcondition tests**

```python
@pytest.mark.parametrize(
    "forgery",
    [
        "wrong_provider_type",
        "rebound_plan",
        "l2_callable_drift",
        "profile_drift",
        "snapshot_drift",
        "wrong_route_type",
        "wrong_segment_type",
        "segment_hash",
        "route_hash",
        "source_candidate_hash",
        "evidence_candidate_hash",
        "route_candidate_projection",
        "codec_round_trip",
        "cost",
        "evidence",
        "telemetry",
    ],
)
def test_api_rejects_forged_wheel_sqp_success(forgery) -> None:
    outcome = run_forged_api_case(forgery)
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "wheel_sqp_identity_mismatch"


def test_deadline_after_provider_or_reseal_replaces_success_with_timeout() -> None:
    for phase in ("provider_completion", "codec_reseal"):
        outcome = run_deadline_expiry_case(phase)
        assert outcome.category is FailureCategoryV2.TIMEOUT
        assert outcome.reason_code == "planning_deadline_expired"
```

Also assert the API does not run continuous sweep twice by replacing the trusted L2 callable with a counting wrapper before dispatcher capture and requiring exactly one call from provider execution.

- [ ] **Step 3: Run RED tests**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_wheel_sqp_api.py tests/test_v2_api.py -q -x
```

Expected: new import/dispatch tests fail; all old API nodeids remain GREEN.

- [ ] **Step 4: Implement the narrow trusted dispatcher and identity seal**

At module load, capture exact trusted objects:

```python
_TRUSTED_PROVIDER_TYPE = WheelKinematicSQPProviderV2
_TRUSTED_PROVIDER_PLAN = WheelKinematicSQPProviderV2.plan
_TRUSTED_L2_VALIDATOR = provider_module.validate_wheel_sqp_candidate_l2
_TRUSTED_ROUTE_ENCODER = encode_wheel_route_v2
_TRUSTED_ROUTE_DECODER = decode_wheel_route_v2
_TRUSTED_ROUTE_HASH = wheel_route_hash_v2
_TRUSTED_ROUTE_TO_CANDIDATE = project_wheel_route_to_candidate_v1
_TRUSTED_CANDIDATE_HASH = wheel_candidate_hash_v2
```

Before provider call require exact provider type、exact bound plan function/self、exact profile object/equality、trusted callable identities and matching request/profile/snapshot hashes. Capture a seal over provider identity、profile bytes、request bytes、snapshot hash、anchor identity and deadline object identity. Recompute it at provider completion and after postcondition; drift maps to `wheel_sqp_identity_mismatch` unless deadline has expired, in which case timeout wins.

Add only this branch in `plan_v2()` after generic provider resolution/profile match and before the generic provider call:

```python
if (
    profile.platform_kind is PlatformKindV2.WHEEL
    and profile.capability_revision == WHEEL_KINEMATIC_CORRIDOR_SQP_CAPABILITY_V2
):
    from path_planner.v2.wheel_sqp_api import dispatch_wheel_sqp_provider_v2
    return dispatch_wheel_sqp_provider_v2(request, profile, provider, anchor, deadline)
```

Other wheel profiles continue through the unchanged generic path.

- [ ] **Step 5: Implement lightweight success postcondition without a second sweep**

Require exact `PlanningSuccessV2`、`WheelKinematicRouteV2`、`WheelKinematicSegmentV2`、`WheelSQPValidationEvidenceV2`、`WheelSQPSearchTelemetryV2`、base cost/observation/cache types. Recompute public route bytes and hashes:

```python
encoded = _TRUSTED_ROUTE_ENCODER(success.route)
decoded = _TRUSTED_ROUTE_DECODER(encoded)
if _TRUSTED_ROUTE_ENCODER(decoded) != encoded:
    return identity_failure(request, profile, detail="route_codec_round_trip")
if decoded.route_hash != _TRUSTED_ROUTE_HASH(decoded):
    return identity_failure(request, profile, detail="route_hash_mismatch")

projected = _TRUSTED_ROUTE_TO_CANDIDATE(
    decoded,
    request,
    provider.wheel_sqp_profile,
    snapshot_hash(request.terrain_snapshot),
)
projected_hash = _TRUSTED_CANDIDATE_HASH(projected)
if projected_hash != decoded.source_candidate_hash:
    return identity_failure(request, profile, detail="candidate_projection_hash_mismatch")
```

Then require `projected_hash == evidence.candidate_hash` and recompute analytic segment endpoints、mode flags、derived samples、segment hashes、route cost、start/goal closed tolerances、request/profile/snapshot/capability/validator/solver IDs、`validation_level=L2`、`ackermann_feasible_claimed=False` and decision hash binding. The trusted provider type/function and trusted L2 callable identity prove which validator path produced the receipt; API does not repeat expensive cell/interval work.

Failures must be exact `PlanningFailureV2` bound to request/platform, carry no route, use accepted stable reason/category, and pass deadline/identity seal checks.

- [ ] **Step 6: Make observation projection accept the exact additive route type**

Keep route subclasses narrow:

```python
if type(route) not in (TypedRouteV2, WheelKinematicRouteV2):
    raise TypeError("route must be an exact supported TypedRouteV2")
```

For the new wheel route, consume each segment's exact derived `samples` tuple as before. Add tests proving the projection is identical for equal sample carriers and that changing samples changes segment/route hashes; do not call observation projection from L2.

- [ ] **Step 7: Export through `path_planner.v2` only and run GREEN**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_wheel_sqp_api.py tests/test_v2_api.py tests/test_v2_observation_projection.py tests/test_v2_wheel_provider.py tests/test_v2_hopper_api.py -q
```

Also run:

```powershell
D:/conda_envs/lunar-explorer/python.exe -c "import path_planner; assert 'WheelKinematicSQPProviderV2' not in path_planner.__dict__; import path_planner.v2 as v2; assert v2.WheelKinematicSQPProviderV2"
```

Expected: all pass; v1/default imports unchanged.

- [ ] **Step 8: Review and commit**

Obtain fresh API/identity review. Main agent commits nested files as `feat: expose opt-in wheel corridor SQP`, then integrates the reviewed gitlink only as `build: integrate opt-in wheel SQP API`.

---

### Task 10: Freeze End-to-End Conformance, Adversarial Scenarios, Determinism, and Compatibility

**Files:**
- Create: `path-planner/tests/test_v2_wheel_sqp_conformance.py`
- Create: `path-planner/tests/fixtures/wheel_sqp_conformance_v1.json`
- Modify: `path-planner/tests/test_package_imports.py`
- Modify: `path-planner/tests/test_v2_api.py`
- Modify: `path-planner/tests/test_v2_wheel_sqp_provider.py`

**Interfaces:**
- Produces: a frozen input/expected-semantics corpus and byte-stable decision/route hashes for the exact solver contract and approved environment。
- Consumes: only public `path_planner.v2` construction/dispatch for end-to-end cases; direct internal calls remain confined to focused module tests。

- [ ] **Step 1: Write the fixed conformance corpus and RED end-to-end expectations**

The UTF-8 JSON corpus contains exact finite terrain layers、request/profile fields and expected semantic result for:

```text
open_straight
open_goal_tolerance_boundary
same_pose_stop
heading_only_turn
rear_goal_reverse
forward_to_reverse_with_stop
s_obstacle
narrow_two_cell_corridor
dead_end_unreachable
three_topology_routes
unknown_boundary_reject
not_traversable_reject
slope_30_pass
slope_nextafter_30_reject
rotating_corner_collision
single_tangent_reject
repair_once_success
repair_once_then_next_corridor
kilometer_open_stress
```

Use this complete row shape; the other rows change only explicit values in the same exact-key schema:

```json
{
  "schema_version": "wheel_sqp_conformance_case/v1",
  "case_id": "same_pose_stop",
  "terrain": {
    "geometry": {
      "width": 3,
      "height": 3,
      "resolution_m": 0.5,
      "origin": [0.0, 0.0],
      "frame_id": "moon"
    },
    "elevation_m": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    "slope_deg": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    "traversable_mask": [[true, true, true], [true, true, true], [true, true, true]],
    "hard_obstacle_mask": [[false, false, false], [false, false, false], [false, false, false]],
    "observed_mask": [[true, true, true], [true, true, true], [true, true, true]],
    "confidence": [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
    "provenance": {
      "source_kind": "synthetic_terrain_obstacle_proxy/v1",
      "source_id": "wheel-sqp-conformance-same-pose-stop",
      "source_hash": "same-pose-stop-v1",
      "physical_obstacle_cells_written": false,
      "details": []
    }
  },
  "request": {
    "request_id": "wheel-sqp-conformance-same-pose-stop",
    "platform_profile_id": "scout-mini-wheel-kinematic-sqp/v1",
    "start_state": {"x_m": 0.75, "y_m": 0.75, "heading_rad": 0.0},
    "goal_state": {"x_m": 0.75, "y_m": 0.75, "heading_rad": 0.0},
    "objective_profile": {
      "distance_weight": 0.0,
      "risk_weight": 0.0,
      "energy_weight": 0.5,
      "time_weight": 0.5
    },
    "resource_budget": {
      "max_expanded_states": 100000,
      "max_route_states": 10000,
      "max_memory_bytes": 0
    },
    "timeout_s": 2.0,
    "accelerator_policy": "disabled",
    "determinism_seed": 20260722
  },
  "expected_semantics": {
    "outcome_type": "success",
    "failure_reason": null,
    "actual_endpoint": {"x_m": 0.75, "y_m": 0.75, "heading_rad": 0.0},
    "goal_position_relation": "equal",
    "goal_heading_relation": "equal",
    "corridor_signature_sequence": [],
    "mode_sequence": ["STOP"],
    "repair_applied": false
  }
}
```

The fixture file has an exact top-level `{schema_version, cases, case_payload_sha256}` object. `case_payload_sha256[case_id]` is computed over that case's canonical JSON bytes excluding no field and is verified before construction; it is an adjacent index, not a self-referential field inside the case. Unknown/missing keys, duplicate IDs or digest drift fail collection.

Each row initially freezes input bytes hash、expected success/failure reason、actual endpoint tolerance relation、corridor signature sequence、mode sequence and repair flag. Decision/route hash keys are intentionally absent—not empty sentinels—until Step 5 records independently reviewed exact values. The corpus never stores an “expected safe” label produced by the provider; safety expectations are separately constructed geometric adversarial cases.

```python
@pytest.mark.parametrize("case", load_conformance_cases())
def test_public_api_matches_wheel_sqp_conformance(case) -> None:
    outcome = execute_public_case(case)
    assert semantic_projection(outcome) == case.expected_semantics
    if case.expected_success:
        assert outcome.validation_evidence.validator_id == WHEEL_KINEMATIC_L2_VALIDATOR_V2
```

- [ ] **Step 2: Add cross-process determinism and static isolation RED tests**

```python
@pytest.mark.parametrize("hash_seed", ("0", "1"))
def test_conformance_semantics_are_process_stable(hash_seed, tmp_path) -> None:
    payload = run_conformance_subprocess(hash_seed, output_path=tmp_path / f"{hash_seed}.json")
    assert payload == EXPECTED_SEMANTIC_PAYLOAD


def test_new_capability_has_no_forbidden_runtime_edges() -> None:
    sources = read_new_wheel_sqp_sources()
    assert "HybridAStarPlanner" not in sources
    assert "default_scout_mini_primitives" not in sources
    assert "random." not in sources
    assert "threading" not in sources
    assert "multiprocessing" not in sources
    assert "pydrake" not in sources
    assert scipy_import_paths(sources) == ("path_planner/v2/wheel_sqp_solver.py",)
```

Assert there are no edits or imports that make v2 default: `path_planner/__init__.py`、CLI、v1 adapter、PPO env/policy/reward/trainer remain outside the nested diff.

- [ ] **Step 3: Run RED then implement only missing integration behavior**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_wheel_sqp_conformance.py tests/test_package_imports.py -q -x
```

Expected: semantic/hash expectations expose any incomplete integration. Fix only defects against the already frozen profile/solver/L2 contracts; do not silently change weights、caps、tie-breaks or IDs to fit golden outputs. If an approved constant must change, stop and revise spec/plan/capability revision first.

- [ ] **Step 4: Run full focused and nested regression**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest `
  tests/test_v2_wheel_sqp_contracts.py `
  tests/test_v2_wheel_kinematics.py `
  tests/test_v2_wheel_sqp_serialization.py `
  tests/test_v2_wheel_corridors.py `
  tests/test_v2_wheel_sqp_initialization.py `
  tests/test_v2_wheel_sqp_solver.py `
  tests/test_v2_wheel_sqp_resources.py `
  tests/test_v2_wheel_sqp_validation.py `
  tests/test_v2_wheel_sqp_provider.py `
  tests/test_v2_wheel_sqp_api.py `
  tests/test_v2_wheel_sqp_conformance.py -q
```

Then run in a fresh process:

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests -q
```

Expected: all non-optional tests pass; existing pydrake skips remain the only optional skips; no old wheel/API/serialization failure is added.

- [ ] **Step 5: Freeze reviewed golden hashes and commit**

Generate candidate semantic/hash payloads from two fresh identical runs, require byte equality, then have an independent reviewer recompute the route/segment hashes from canonical bytes and inspect every geometric adversarial expected result. Only after review, add the exact 64-hex decision/route hash keys and their assertions to the fixture/tests, reject missing or empty hash keys, and rerun Step 4.

Main agent commits nested files as `test: freeze wheel SQP conformance corpus`, then integrates the reviewed gitlink only as `build: integrate wheel SQP conformance corpus`.

---

### Task 11: Build an Independent Wheel-SQP Formal Benchmark Gate Without Changing Gate 2 or Gate 6

**Files:**
- Create: `path-planner/src/path_planner/v2/wheel_sqp_benchmark.py`
- Create: `path-planner/tests/test_v2_wheel_sqp_benchmark.py`
- Modify: `path-planner/src/path_planner/v2/__init__.py`
- Create: `scripts/run_xunce_path_v2_wheel_sqp_formal.py`
- Create: `configs/xunce_path_v2_wheel_sqp_formal_v1.json`
- Create: `tests/test_xunce_path_v2_wheel_sqp_formal.py`
- Modify: `scripts/xunce_artifact_paths.py`
- Modify: `configs/stage_registry.json`
- Modify: `docs/xunce-stage-documentation-index.md`

**Interfaces:**
- Produces: `run_wheel_sqp_benchmark_variant_v1()` plus independent input audits, actual execution rows, per-case/scale aggregates, determinism audit, threshold audit and the canonical eight-artifact stage under `D:/xunce/out/path_v2/wsqp`。
- Reuses: `PrimitiveAuditRowV2`、`ExactMapQualityRowV2`、`StandardEpisodeRowV2` aggregate semantics and `path_v2_gate_artifacts.py` atomic artifact writer。
- Does not modify: `benchmark_fixtures.GATE6_*` constants、Gate 2/6 configs、old output roots or PPO target contracts。

- [ ] **Step 1: Write RED typed-row and threshold aggregation tests**

```python
def test_final_case_thresholds_are_closed_and_nonempty() -> None:
    summary = aggregate_wheel_sqp_formal_v1(make_boundary_rows())
    assert summary.false_positive_count == 0
    assert summary.complete_l2_ratio == 1.0
    assert summary.reachable_success_ratio == 0.99
    assert summary.standard_runtime_p95_ms == 250.0
    assert summary.kilometer_runtime_p95_ms == 750.0
    assert summary.max_exact_cost_ratio == 1.10
    assert summary.hard_timeout_violation_count == 0
    assert summary.deterministic is True
    assert summary.passed is True


@pytest.mark.parametrize(
    ("mutation", "blocker"),
    [
        ("one_false_positive", "wheel_sqp_false_positive_gate_failed"),
        ("one_success_without_l2", "wheel_sqp_complete_l2_gate_failed"),
        ("reachable_0_989", "wheel_sqp_reachable_success_gate_failed"),
        ("standard_250_epsilon", "wheel_sqp_standard_runtime_gate_failed"),
        ("kilometer_750_epsilon", "wheel_sqp_kilometer_runtime_gate_failed"),
        ("runtime_2000_epsilon", "wheel_sqp_hard_timeout_gate_failed"),
        ("exact_1_10_epsilon", "wheel_sqp_exact_quality_gate_failed"),
        ("semantic_hash_drift", "wheel_sqp_determinism_gate_failed"),
        ("empty_reachable_denominator", "wheel_sqp_formal_denominator_empty"),
        ("empty_exact_success_denominator", "wheel_sqp_exact_quality_denominator_empty"),
    ],
)
def test_threshold_audit_fails_each_boundary_mutation(mutation, blocker) -> None:
    assert blocker in aggregate_wheel_sqp_formal_v1(make_mutated_rows(mutation)).blockers
```

- [ ] **Step 2: Write RED external-input lineage and actual-execution runner tests**

The runner accepts exactly four absolute D-drive formal inputs:

```text
independent_l2_labels       >= 10,000 rows
independent_small_map_optima >= 1 certified nonzero-optimum row
standard_schedules          >= 100 rows
kilometer_schedules         >= 30 rows
```

```python
def test_missing_formal_inputs_is_blocked_not_passed(tmp_path) -> None:
    summary = runner.run_wheel_sqp_formal(CONFIG, tmp_path / "out", REPO_ROOT, execute=False)
    assert summary["status"] == "blocked"
    assert summary["formal_metrics_status"] == "not_evaluated"
    assert summary["blockers"][0] == "wheel_sqp_independent_formal_inputs_missing"


def test_runner_rejects_subject_as_oracle_before_provider_execution(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "_execute_provider_case", forbidden_call)
    inputs = write_formal_inputs(tmp_path, oracle_source_id="wheel-sqp-subject/v1", provider_source_id="wheel-sqp-subject/v1")
    summary = runner.run_wheel_sqp_formal(config_for(inputs), tmp_path / "out", REPO_ROOT, execute=True)
    assert summary["status"] == "blocked"
    assert "wheel_sqp_oracle_provider_identity_not_independent" in summary["blockers"]


def test_schedule_files_supply_requests_and_oracle_labels_but_not_provider_results(tmp_path) -> None:
    row = valid_standard_row()
    assert "request" in row and "oracle_reachable" in row
    for forbidden in ("provider_success", "runtime_ms", "route_hash", "provider_complete_l2"):
        assert forbidden not in row
```

Also reject: relative/C-drive paths、duplicate JSON keys、non-UTF-8、hash mismatch、wrong capability/profile/solver/validator/cost IDs、wrong parent/nested input commit、`source_independent!=true`、oracle declarations that use subject corridors/SQP/L2、row count deficits、duplicate request IDs、mixed source identities or files read differently from their single bytes snapshot.

- [ ] **Step 3: Run RED tests**

From parent root:

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest path-planner/tests/test_v2_wheel_sqp_benchmark.py tests/test_xunce_path_v2_wheel_sqp_formal.py -q -x
```

Expected: import/runner/config failures because the new formal surface does not exist; old Gate runner tests remain untouched.

- [ ] **Step 4: Implement exact formal schemas, cases, phases, and actual execution**

Every input header binds:

```text
schema_version, input_kind, scenario_schema_id,
subject_capability_id, subject_profile_id, subject_profile_hash,
solver_contract_id, validator_contract_id,
cost_definition_id, gate_input_parent_commit, gate_input_nested_commit,
oracle_source_id, oracle_source_hash, provider_source_id, provider_source_hash,
source_independent=true,
oracle_uses_subject_corridors=false,
oracle_uses_subject_sqp=false,
oracle_uses_subject_l2_validator=false,
rows_sha256
```

The runner parses each file from one immutable bytes snapshot, verifies header/rows hash and exact keys, then executes repository code itself. Schedule rows use platform-neutral `wheel_sqp_formal_scenario/v1` terrain/start/goal/objective/resource fields; they do not contain a runtime `platform_profile_id`. The runner derives the exact case request/profile from the frozen table below. External files never provide trusted provider result fields. Its CLI exposes `--no-execute`, which maps exactly to `execute=False` for input/audit blocker evidence and never runs a provider.

Freeze cases:

```python
WHEEL_SQP_FORMAL_CASES_V1 = (
    "v1_astar",
    "wheel_hybrid_astar_opt_in",
    "wheel_sqp_single_corridor",
    "wheel_sqp_three_corridor",
    "wheel_sqp_three_corridor_one_l2_repair",
)
WHEEL_SQP_FINAL_CASE_V1 = "wheel_sqp_three_corridor_one_l2_repair"
```

Freeze dispatch separately from the external scenario:

| case | entry point | exact profile | attempt contract | role |
|---|---|---|---|---|
| `v1_astar` | `run_v1_reference_case_v1()` | current unchanged v1 default | n/a | reference only |
| `wheel_hybrid_astar_opt_in` | public `plan_v2()` | `scout-mini-wheel-hybrid-reference/v1` wrapping the unchanged `WheelProfileV2` / `wheel-provider-capability/v1` | existing Hybrid provider | reference only |
| `wheel_sqp_single_corridor` | private `run_wheel_sqp_benchmark_variant_v1()` | exact production `scout-mini-wheel-kinematic-sqp/v1` | `(max_corridors=1, repairs=0)` | ablation |
| `wheel_sqp_three_corridor` | private `run_wheel_sqp_benchmark_variant_v1()` | exact production `scout-mini-wheel-kinematic-sqp/v1` | `(max_corridors=3, repairs=0)` | ablation |
| `wheel_sqp_three_corridor_one_l2_repair` | public `plan_v2()` | exact production `scout-mini-wheel-kinematic-sqp/v1` | production `(3,1)` | final subject |

The benchmark-only variant receives an already audited production provider/profile and changes only the loop attempt limits; it is not exported from `path_planner.v2`, registered as a public profile or reachable from production dispatch. Each result row records the derived case profile/capability hash. Safety/L2 gates apply to every SQP success; formal success/runtime/quality thresholds apply to the final case only.

Freeze resumable phases:

```text
preflight
primitive_l2_audit
exact_quality
standard
kilometer
comparison_matrix
determinism
aggregate
```

Primitive audit compares the subject L2 result on externally supplied canonical candidates to independent labels. Standard/Kilometer dispatches each case through the frozen table and records its first returned result only; the final case must go through public `plan_v2()`. Exact quality excludes exact start==goal rows with zero optimum and requires certified positive optimum/lower-bound lineage.

- [ ] **Step 5: Implement deterministic matrix and exact thresholds**

For each schedule row run semantic checks under:

```text
PYTHONHASHSEED = 0, 1
outer_workers = 1, 4
```

The solver itself stays single-threaded. Outer workers write isolated row results, reassembled by `(scale,case_id,request_id,seed,worker)`; capability cache evidence remains fixed disabled for v1, so this plan does not add a cache matrix or cache implementation.

Compare normalized segment bytes、route hash、ordered corridor hashes/signatures、mode sequence、L2/counterexample/repair choice and success/failure reason. Exclude runtime and environment observation telemetry from semantic digest.

Threshold audit is exact:

```python
false_positive_count == 0
unknown_accept_count == 0
complete_l2_ratio == 1.0
reachable_success_ratio >= 0.99
standard_nearest_rank_p95_ms <= 250.0
kilometer_nearest_rank_p95_ms <= 750.0
hard_timeout_violation_count == 0
exact_final_case_success_count >= 1
max_exact_first_route_cost_ratio <= 1.10
determinism_mismatch_count == 0
```

- [ ] **Step 6: Write canonical artifacts and register the independent stage**

Reuse artifact IO/path helpers and write exactly:

```text
config.json
summary.json
routing.json
results.jsonl
phase-state.jsonl
review.json
report.md
manifest.json
```

Every summary/routing/report carries:

```json
{
  "publishes_checkpoint": false,
  "replaces_default_policy": false,
  "connects_real_executor": false,
  "starts_online_canary": false,
  "changes_v1_default": false,
  "uses_hybrid_runtime_fallback": false,
  "modifies_gate2_or_gate6": false
}
```

Register `xunce-path-v2-wheel-sqp-formal` with default output `D:/xunce/out/path_v2/wsqp`. Documentation index links to the approved spec、this plan and output `report.md`; it does not duplicate experimental conclusions.

- [ ] **Step 7: Run GREEN and dry-run artifact checks**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest path-planner/tests/test_v2_wheel_sqp_benchmark.py tests/test_xunce_path_v2_wheel_sqp_formal.py -q
D:/conda_envs/lunar-explorer/python.exe scripts/run_stage.py --stage xunce-path-v2-wheel-sqp-formal --dry-run
```

Run a fake-input execution only under a fresh D-drive temp root, then verify eight files、manifest hashes、no old g0–g6 writes and blocked status for non-independent labels. Do not write the canonical formal root in this task.

- [ ] **Step 8: Review and commit**

Obtain a fresh benchmark/lineage/artifact review. Commit nested benchmark contracts as `feat: add wheel SQP benchmark contracts`; integrate that reviewed gitlink, then commit root runner/config/registry/docs as `feat: add independent wheel SQP formal gate`. Do not modify existing Gate 2/6 case constants or configs.

---

### Task 12: Whole-Branch Regression, Formal Evidence, Independent Review, and Non-Publishing Handoff

**Files:**
- Modify only files required by reproducible RED tests for reviewer Critical/Important findings.
- Write runtime artifacts only under a fresh D-drive root below `D:/xunce/out/path_v2/wsqp`.
- Do not modify historical `D:/xunce/out/path_v2/g0..g6` artifacts.

- [ ] **Step 1: Run the exact repository and dependency preflight**

Verify:

```powershell
git branch --show-current
git merge-base --is-ancestor b635740ee021258ef31811ec87c60add839fc5f9 HEAD
git status --short
git -C path-planner status --short
git ls-files -s path-planner
git -C path-planner rev-parse HEAD
D:/conda_envs/lunar-explorer/python.exe -c "import scipy; assert scipy.__version__ == '1.18.0'"
```

Expected: branch is `codex/multiplatform-path-planner-v2`; baseline is ancestor; parent/nested tracked trees are clean; gitlink equals nested HEAD; exact optional backend is installed. Do not inspect the protected C-drive worktree.

- [ ] **Step 2: Run complete nested tests in a fresh process**

From parent root:

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest -q -p no:cacheprovider path-planner/tests --basetemp D:/xunce/tmp/wsqp_nested_full
```

Expected: all required tests pass; only the pre-existing optional pydrake tests skip. Record exact counts in the final report; do not copy an older count into new evidence.

- [ ] **Step 3: Run root runner tests and the inherited PPO baseline guard**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest -q -p no:cacheprovider tests/test_xunce_path_v2_wheel_sqp_formal.py tests/test_xunce_path_v2_gate_benchmark.py tests/test_xunce_path_v2_g0_baseline_and_isolation.py --basetemp D:/xunce/tmp/wsqp_root
```

Run the checked-in Gate 0 auditor under a fresh D-drive root; it fixes import origin to this D worktree, runs only the frozen path-planner baseline plus `tests/ppo_highres_frontier/test_stage1_smoke_env.py`, parses JUnit and compares exact nodeids with the existing allowlist:

```powershell
$gate0Root = "D:/xunce/tmp/wsqp_gate0/$([DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ', [Globalization.CultureInfo]::InvariantCulture))"
$repoRoot = (Resolve-Path .).Path
if (Test-Path -LiteralPath $gate0Root) { throw "Gate 0 audit root must be fresh" }
D:/conda_envs/lunar-explorer/python.exe scripts/run_xunce_path_v2_g0_baseline_and_isolation.py `
  --config configs/xunce_path_v2_g0_baseline_and_isolation_v1.json `
  --output-root $gate0Root `
  --repo-root $repoRoot
```

Expected: root runner tests pass; PPO has exactly the inherited 13 allowlisted failures, 43 passes, zero errors/skips, and no new failed nodeid. The inherited failures are external baseline, not a wheel-SQP regression.

- [ ] **Step 4: Audit formal inputs or emit the exact blocker**

If any of the four independent inputs is absent, run the formal runner once under a fresh D root with `execute=false` and require:

```powershell
$blockedRoot = "D:/xunce/out/path_v2/wsqp/blocked_$([DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ', [Globalization.CultureInfo]::InvariantCulture))"
if (Test-Path -LiteralPath $blockedRoot) { throw "blocked evidence root must be fresh" }
D:/conda_envs/lunar-explorer/python.exe scripts/run_xunce_path_v2_wheel_sqp_formal.py `
  --config configs/xunce_path_v2_wheel_sqp_formal_v1.json `
  --output-root $blockedRoot `
  --no-execute
$blockedExit = $LASTEXITCODE
if ($blockedExit -ne 1) { throw "blocked audit must exit 1" }
```

```text
status=blocked
formal_metrics_status=not_evaluated
first blocker=wheel_sqp_independent_formal_inputs_missing
no pass claim
```

That completes implementation readiness but not formal promotion. Do not generate replacement labels from the subject planner.

If all four files exist but any lineage, identity, encoding, hash or count audit fails, require a stable input-audit blocker and `formal_metrics_status=not_evaluated`; it must stop before provider execution and must not be rewritten as the missing-input blocker.

If all four independently reviewed inputs are present, snapshot their bytes once, record hashes and execute the exact eight phases under a fresh root:

```powershell
$parentCommit = (git rev-parse --short=8 HEAD).Trim()
$nestedCommit = (git -C path-planner rev-parse --short=8 HEAD).Trim()
$utcStamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssfffZ", [Globalization.CultureInfo]::InvariantCulture)
$runId = "{0}_{1}_{2}" -f $utcStamp, $parentCommit, $nestedCommit
$outputRoot = "D:/xunce/out/path_v2/wsqp/$runId"
if (Test-Path -LiteralPath $outputRoot) { throw "refusing to overwrite $outputRoot" }
D:/conda_envs/lunar-explorer/python.exe scripts/run_xunce_path_v2_wheel_sqp_formal.py `
  --config configs/xunce_path_v2_wheel_sqp_formal_v1.json `
  --output-root $outputRoot
```

No existing run directory may be overwritten or deleted. A failed/blocked run remains evidence and cannot be renamed to passed.

- [ ] **Step 5: Verify the formal gates and artifacts**

For a pass claim require all exact thresholds, nonempty formal denominators, all eight canonical files, finite JSON, strict UTF-8 report, manifest hashes, input hashes/commits and four publishing flags false. Recompute aggregate metrics from `results.jsonl` in an independent process and compare with `summary.json`.

If any threshold fails, record the exact blocker and keep the capability opt-in/unpromoted. Do not add request-internal post-success improvement、Hybrid fallback、larger timeout、looser L2 or hidden tolerance to make the report pass.

- [ ] **Step 6: Request fresh independent reviews**

Review package includes the approved spec, this plan, merge-base diff, nested gitlink history, exact test logs, conformance corpus, formal input headers/hashes, `results.jsonl`, summary/routing/manifest/report and current branch status.

Require four fresh reviews:

1. spec/contract coverage;
2. code quality and module boundaries;
3. adversarial continuous L2/repair/identity correctness;
4. benchmark independence、metrics、artifact and threshold correctness.

Fix every Critical/Important through a new failing test, rerun the affected focused suite and the complete Steps 2–5, then obtain a fresh re-review. Minor findings go into the report ledger.

- [ ] **Step 7: Finish the branch without publishing side effects**

Run `git diff --check`, strict UTF-8 scan and exact parent/nested status. The final handoff reports commits、tests、formal status/blockers and D-drive artifact path. It does not push、open a PR、publish a checkpoint、change default policy、connect executor、start canary or make the new profile default.

---

## Plan Self-Review

- Spec coverage: every approved section maps to a task—input/profile/output contracts (Task 1–2), three topological corridors (Task 3), deterministic initial modes (Task 4), direct multiple-shooting SQP (Task 5–6), normalized serialized complete continuous L2 (Task 2/7), one counterexample repair and first-pass return (Task 8), opt-in integration/no fallback (Task 9), conformance (Task 10), formal thresholds and independent labels (Task 11–12).
- Type consistency: unvalidated solver output stays `WheelSQPCandidateV2`/private canonical bytes; only passed L2 promotes exact `WheelKinematicSegmentV2` and `WheelKinematicRouteV2`; outer outcome remains exact `PlanningSuccessV2`/`PlanningFailureV2`.
- Promotion binding: route、evidence and L2 receipt carry the same candidate hash; promotion and API independently project public segments back to private candidate semantics and recheck analytic endpoints/samples without repeating the terrain sweep.
- State identity: the request start is retained byte-semantically as the exact first state; all later headings remain one deterministic unwrapped sequence, while only angular differences are wrapped for comparisons.
- Safety authority: optimizer center-clearance is only a sufficient hard approximation; public success requires decoded analytic replay and complete rotating-rectangle interval proof. Unknown、hard、not-traversable、OOB and `slope>30` all fail closed.
- Deadline/resources: one API-created deadline flows through every module; reserve is checked before SQP/repair and before L2; late success is impossible; compiled SLSQP non-preemptibility remains a formal runtime gate risk rather than a hidden exception.
- Determinism: variable order、scales、sinc branch、rounding、neighbor/component/path order、mode ties、SLSQP options、counterexample order、repair face、semantic digest and subprocess matrix are all explicit.
- Compatibility/isolation: v1/default、old Hybrid A* wheel、Gate 2/6、PPO and historical artifacts are outside the new runtime path; Hybrid appears only as benchmark reference.
- Scope control: v1 cache evidence stays disabled; this plan does not add a cache implementation, executor path, fallback or request-internal post-success improvement loop.
- Completeness scan: implementation behavior, numeric constants, IDs, commands, expected RED/GREEN outcomes and the missing-formal-input blocker are explicit; no unresolved token or value slot remains.
- External dependency: SciPy is a lazy optional extra pinned to 1.18.0; its absence affects only the explicit new capability and cannot break v1 import/use.
- Formal evidence: subject code never supplies its own oracle labels or optima. Missing independent data yields blocked, not a fabricated pass.

---

## Execution Handoff

After this plan is approved, choose one execution mode:

1. **Subagent-Driven (recommended):** execute Tasks 1–12 in this task, one bounded implementation worker plus fresh spec/quality review per task; shared files and nested gitlink integrations remain serial under the main agent.
2. **Inline Execution:** execute the same tasks serially in the main task using `superpowers:executing-plans`, with identical RED/GREEN/review/commit gates.

The plan approval authorizes implementation only after the user selects a mode; it never authorizes publishing, default replacement, executor, canary, push or release.
