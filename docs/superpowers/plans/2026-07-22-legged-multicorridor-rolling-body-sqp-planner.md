# Legged Multicorridor Rolling Body SQP Planner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增显式 opt-in 的 `legged_multicorridor_rolling_body_sqp/v1`：在同一不可变 `TerrainSnapshotV2` 和 `200–500ms` 期限内，生成最多三条拓扑不同的全局意图走廊，沿风险优先顺序求解时间戳机体 SQP，并且只返回通过独立完整连续机体包络 L2 的第一条滚动局部轨迹。

**Architecture:** 新能力不复用旧 `simulation_proxy_static_crawl/v1` 的足端搜索，也不伪装成完整到达任务终点的 `PlanningSuccessV2`。它通过独立的 `plan_legged_rolling_v1()` API 复用公共 `TerrainSnapshotV2`、`PlatformProfileV2`、`ObjectiveProfileV2`、`ResourceBudgetV2`、`PlanningDeadlineV2`、canonical JSON 与失败类别；足式专用 safety view 加入 provenance、控制器身份和最低置信度硬门，多尺度走廊只提供不可执行意图，直接多重射击 SQP 产生局部机体候选，风险模块只排序，独立 L2 才能把候选提升为 `TimedBodyTrajectoryV1`。

**Tech Stack:** Python 3.12、frozen/slotted dataclasses、NumPy 2.x、SciPy 1.18.0 SLSQP（懒加载 optional extra）、pytest、现有 SHA-256/canonical JSON、现有 `TerrainSnapshotV2` / `PlanningDeadlineV2`、根仓库 artifact IO 与 D 盘正式输出。

## Global Constraints

- 规范设计固定为 `docs/superpowers/specs/2026-07-22-legged-multicorridor-rolling-body-sqp-planner-design.md`；本计划不得改写已批准的职责边界、算法选择、失败语义或正式门。
- 计划开始编写时的稳定基线为父仓库 `b6e25e8`、嵌套 `path-planner@37fc145`；编写期间并发 Wheel-SQP 任务继续推进。执行前必须重新记录并审计实际父/嵌套起始提交；任何并发 dirty、untracked、nested commit 或待集成 gitlink 都不属于本计划，绝不暂存、覆盖、回退或删除。
- 在上述起始基线上，Scout Mini SQP 只有 profile/contracts、运动学与 canonical codec。即使后续 Wheel 任务新增 corridor/solver/L2/provider，本计划也不把这些 wheel-specific 模块作为前置依赖；足式代码使用独立模块，复用公共设计模式和已审计基础类型，不复用滑移转向语义。
- 本规格虽然覆盖安全地图、全局走廊、局部 SQP、风险、L2、API 与 benchmark，但这些部分共享同一 request/profile/controller/snapshot/candidate identity 和绝对 deadline；任何中间子系统都不能独立发布，因此保留一个计划、按 13 个可独立审查任务串行交付。
- 现有 v1 默认 A*、v2 wheel Hybrid A*、Scout Mini SQP、`simulation_proxy_static_crawl/v1`、hopper、PPO 默认路径和 Gate 0–6 保持行为兼容。
- 新能力只通过 `plan_legged_rolling_v1()` 与新的 profile ID 显式调用；不得修改 `plan_v2()` 的既有返回语义，不得把滚动局部成功包装为 `PlanningSuccessV2`。
- 四足稳定控制、步态、落足、支撑、接触力、MPC/WBC 与关节命令始终在外部单位；本能力只输出 `(x,y,yaw,v,t)` 机体轨迹和 `(a,omega,duration)` 分段控制。
- 请求只读取一个 `resolution_m=0.5` 的不可变 `TerrainSnapshotV2`。`1.0m/2.0m` 只能由该快照 fail-closed 聚合；新快照必须形成新请求，禁止跨快照复用 open/closed、走廊、SQP warm start、反例或 L2 receipt。
- 地图硬门固定为 `observed && !hard_obstacle && traversable && slope_deg<=30.0 && confidence>=profile.min_publish_confidence && in_bounds`。`nextafter(30.0,+inf)` 必须拒绝；学习风险不能改变任何硬门。
- `replan_period_s` 必须是精确有限内建 `float` 且位于 `[0.2,0.5]`；唯一 deadline 为 `started + min(request.timeout_s, request.replan_period_s)`。deadline 后不得返回 success。
- 平台 profile 的 `max_forward_speed_mps` 必须 `<=0.8`；`0.3m/s` 只是常用巡航参考，不是硬最小速度，停车合法。
- 每个快照最多三条不同 topology signature；按版本化风险引导键排序，第一条完整 L2 通过后立即返回，`selection_policy="first_l2_in_order/v1"` 且 `global_optimality_claimed=false`。
- SQP 状态固定为 `(x,y,yaw,v)`，控制固定为 `(a,omega)`，每段 `duration_s>0`；禁止随机 multi-start。方向切换必须经过显式 `v=0`，是否允许倒退由 profile 密封。
- 学习模型只接收只读、候选派生特征。缺失、非有限、schema 不兼容、非确定性或 OOD 必须使用非零确定性 fallback；模型不得改变 hard view/阈值，且对同一 `risk_neutral_geometry_digest` 的 hard/L2 verdict 及其失败 reason 必须不变。风险合法地改变走廊顺序，因此在 first-L2/deadline 策略下不声称最终 provider outcome/failure category 跨模式不变。
- 候选先 canonicalize/encode/decode 为新对象，再执行唯一完整连续 L2。每条走廊最多一次由稳定首反例驱动的 repair；identity、numeric、kinematic、resource 或 deadline 失败不可 repair。
- 不发布 checkpoint、不替换 default policy、不连接 executor、不启动 canary、不 push、不 release；不声明 Ackermann、足式动力学稳定、落足可行、接触力可行、抗滑或实机认证。
- 新正式输出使用短根目录 `D:/xunce/out/path_v2/lbsqp`，临时测试根使用 `D:/xunce/tmp/lbsqp_*`，下载/缓存使用 `D:/CodexDownloads`；不得把 corpus、benchmark artifact 或构建缓存写到 C 盘或提交到 Git。
- 所有中文 `.md`、`.py` 注释和测试文本必须 UTF-8；写后用 Python `read_text(encoding="utf-8")` 检查，无 BOM、U+FFFD 或 mojibake。
- 每个任务严格 RED → GREEN → focused regression → review → commit。实现 worker 只修改任务列出的文件；主 agent 审核后只暂存精确路径，并分别提交嵌套实现与父仓库 gitlink。

---

## File Structure

### Nested `path-planner` implementation

- Modify `path-planner/pyproject.toml`：增加独立 `legged-sqp` optional extra，精确锁定 `scipy==1.18.0`，不改变基础依赖或 `wheel-sqp`。
- Create `path-planner/src/path_planner/v2/legged_rolling_profiles.py`：足式滚动机体 profile、profile hash、严格审计与专用 registry。
- Create `path-planner/src/path_planner/v2/legged_rolling_contracts.py`：请求、机体状态、不可执行走廊、风险证据、L2 轨迹、复合成功和 telemetry 公共合同；失败继续复用共享 `PlanningFailureV2`。
- Create `path-planner/src/path_planner/v2/legged_body_sqp_contracts.py`：初值、私有候选、求解结果、资源 ledger、L2 counterexample/result 等内部合同。
- Create `path-planner/src/path_planner/v2/legged_rolling_safety.py`：provenance/controller/profile 绑定、最低置信度硬门、保守多尺度层和只读派生缓存。
- Modify `path-planner/src/path_planner/v2/terrain.py`：在不改变旧 digest 字节流的前提下增加按行分块的 snapshot-hash checkpoint 能力。
- Modify `path-planner/src/path_planner/v2/geometry.py`：抽出平台中性的 conservative rectangle pose-cell helper；既有 wheel helper 委托它并保持 byte-for-byte 结果。
- Create `path-planner/src/path_planner/v2/legged_rolling_risk.py`：风险特征、可选模型 Protocol、输出审计、确定性 fallback、union proxy 与字典序键。
- Create `path-planner/src/path_planner/v2/legged_rolling_corridors.py`：多尺度稳定 A*、阻塞分量、参考射线 topology signature、排除惩罚、细化和最多三走廊。
- Create `path-planner/src/path_planner/v2/legged_body_kinematics.py`：常值 `(a,omega)` 的机体传播、Jacobian、采样和时间/相对能耗/平滑代理。
- Create `path-planner/src/path_planner/v2/legged_rolling_subgoals.py`：`5–20m` 前视滚动子目标与最终任务目标保持逻辑。
- Create `path-planner/src/path_planner/v2/legged_body_sqp_initialization.py`：单一确定性 corridor-tangent 初值。
- Create `path-planner/src/path_planner/v2/legged_body_sqp_serialization.py`：候选/公开轨迹严格 codec、12 位 half-even canonicalization 与 identity hash。
- Create `path-planner/src/path_planner/v2/legged_body_sqp_solver.py`：固定变量布局、直接多重射击约束、私有 SciPy SLSQP 适配、deadline/resource taxonomy 与 repair constraint。
- Create `path-planner/src/path_planner/v2/legged_body_sqp_validation.py`：decode 后独立传播、连续矩形包络区间证明、首反例与 L2 receipt。
- Create `path-planner/src/path_planner/v2/providers/legged_rolling.py`：走廊顺序、SQP、风险、L2、一次 repair、首条通过返回与稳定失败编排。
- Create `path-planner/src/path_planner/v2/legged_rolling_api.py`：专用 trusted dispatch、唯一 deadline、profile/provider seal 与轻量 codec/L2 receipt postcondition。
- Create `path-planner/src/path_planner/v2/adapters/legged_rolling_target.py`：把既有 `PpoTargetV2` 仅映射为任务位置/朝向，同时由非 PPO 调用方显式提供权威 `TerrainSnapshotV2`、当前机体位姿/速度和控制器身份；不得复用会重写 provenance 的 `ObservedTerrainInputV2`。
- Modify `path-planner/src/path_planner/v2/adapters/__init__.py`、`providers/__init__.py` 与 `v2/__init__.py`：只增加显式新 surface，不改变旧默认导出含义。
- Create `path-planner/src/path_planner/v2/legged_rolling_benchmark.py`：正式 row、summary、聚合、门审计与独立输入身份合同。

### Nested tests

- Create `path-planner/tests/test_v2_legged_rolling_contracts.py`。
- Create `path-planner/tests/test_v2_legged_rolling_safety.py`。
- Create `path-planner/tests/test_v2_legged_rolling_risk.py`。
- Create `path-planner/tests/test_v2_legged_rolling_corridors.py`。
- Create `path-planner/tests/test_v2_legged_body_kinematics.py`。
- Create `path-planner/tests/test_v2_legged_rolling_subgoals.py`。
- Create `path-planner/tests/test_v2_legged_body_sqp_initialization.py`。
- Create `path-planner/tests/test_v2_legged_body_sqp_serialization.py`。
- Create `path-planner/tests/test_v2_legged_body_sqp_solver.py`。
- Create `path-planner/tests/test_v2_legged_body_sqp_validation.py`。
- Create `path-planner/tests/test_v2_legged_rolling_provider.py`。
- Create `path-planner/tests/test_v2_legged_rolling_api.py`。
- Create `path-planner/tests/test_v2_legged_rolling_conformance.py`。
- Create `path-planner/tests/test_v2_legged_rolling_benchmark.py`。
- Create `path-planner/tests/fixtures/legged_rolling_conformance_v1.json`。
- Modify only for explicit export/regression assertions: `path-planner/tests/test_package_imports.py`、`test_v2_geometry.py`、`test_v2_terrain.py`、`test_v2_profiles.py`、`test_v2_serialization.py`、`test_v2_route_validation.py`、`test_v2_legged_oracle.py`、`test_v2_api.py`、`test_v2_ppo_target_adapter.py`、`test_v2_wheel_sqp_serialization.py`。

### Root formal orchestration

- Create `scripts/run_xunce_path_v2_legged_body_sqp_formal.py`。
- Create `scripts/verify_xunce_path_v2_legged_body_sqp_formal.py`：stdlib-only second implementation of joins/denominators/gates；禁止 import subject benchmark/runner aggregator。
- Create `configs/xunce_path_v2_legged_body_sqp_formal_v1.json`。
- Create `tests/test_xunce_path_v2_legged_body_sqp_formal.py`。
- Modify `configs/stage_registry.json`：注册独立 `xunce-path-v2-legged-body-sqp-formal`，默认根为 `D:/xunce/out/path_v2/lbsqp`。
- Modify `docs/xunce-stage-documentation-index.md`：只登记 spec、plan 与 D 盘 report 职责，不复制实验结论。

---

## Frozen Public and Internal Contracts

以下 ID 在 Task 1 先冻结；字符串变化必须升级 schema/capability revision：

```python
LEGGED_ROLLING_CAPABILITY_V1 = "legged_multicorridor_rolling_body_sqp/v1"
LEGGED_ROLLING_REQUEST_SCHEMA_V1 = "legged_rolling_request/v1"
LEGGED_ROLLING_PLAN_SCHEMA_V1 = "legged_rolling_plan/v1"
LEGGED_ROLLING_PROFILE_SCHEMA_V1 = "legged_rolling_body_sqp_profile/v1"
LEGGED_HARD_SAFETY_VIEW_V1 = "legged_hard_safety_view/v1"
LEGGED_DERIVED_HIERARCHY_V1 = "legged_conservative_0p5_1p0_2p0_hierarchy/v1"
LEGGED_CORRIDOR_SOURCE_V1 = "legged_deterministic_multiscale_h_signature_corridors/v1"
LEGGED_SELECTION_POLICY_V1 = "first_l2_in_order/v1"
LEGGED_ROLLING_SUBGOAL_SCHEMA_V1 = "legged_rolling_subgoal/v1"
LEGGED_BODY_SEGMENT_SCHEMA_V1 = "legged_timed_body_segment/v1"
LEGGED_BODY_SOLVER_CONTRACT_V1 = "legged_body_direct_multiple_shooting_sqp/v1"
LEGGED_BODY_SQP_GEOMETRY_V1 = "legged_body_support_lattice_corridor_geometry/v1"
LEGGED_SLSQP_BACKEND_V1 = "scipy_slsqp_1p18_preloaded/v1"
LEGGED_BODY_CANONICALIZATION_V1 = "legged_body_decimal12_half_even/v1"
LEGGED_BODY_L2_VALIDATOR_V1 = "legged_body_continuous_rectangle_sweep_l2/v1"
LEGGED_BODY_L2_RESERVE_MODEL_V1 = "legged_body_l2_reserve_model/v1"
LEGGED_RISK_FEATURE_SCHEMA_V1 = "legged_body_route_risk_features/v1"
LEGGED_RISK_INFERENCE_RUNNER_V1 = "legged_isolated_bounded_risk_inference/v1"
LEGGED_RISK_INFERENCE_LEDGER_V1 = "legged_request_risk_inference_ledger/v1"
LEGGED_DETERMINISTIC_RISK_SOURCE_V1 = "deterministic_fallback"
LEGGED_DETERMINISTIC_RISK_MODEL_V1 = "legged_deterministic_risk_fallback/v1"
LEGGED_RISK_AGGREGATION_V1 = "independent_segment_union_proxy/v1"
RISK_DISTANCE_CLIP_M_V1 = 2.0
RISK_FALLBACK_AGGREGATION_BASE_S_V1 = 0.0001
RISK_FALLBACK_AGGREGATION_PER_SEGMENT_S_V1 = 0.000002
RISK_FALLBACK_AGGREGATION_BASE_BYTES_V1 = 16_384
RISK_FALLBACK_AGGREGATION_PER_SEGMENT_BYTES_V1 = 128
```

公共调用面固定为：

```text
plan_legged_rolling_v1(
    request: LeggedRollingRequestV1,
    *,
    registry: LeggedRollingProfileRegistryV1,
    providers: Mapping[str, LeggedRollingBodySQPProviderV1],
    monotonic_clock: MonotonicClockV2 = monotonic,
) -> LeggedRollingOutcomeV1
```

这是冻结的调用合同；Task 10 给出完整控制流。公共类型关系固定为：

```text
LeggedRollingOutcomeV1 = LeggedRollingPlanV1 | PlanningFailureV2
LeggedRollingPlanV1 is not PlanningSuccessV2
GlobalCorridorIntentV1.executable is always false
TimedBodyTrajectoryV1.validation_level is always L2
```

`LeggedRollingBodySQPProfileV1` 必须包含并 hash 以下字段：

```python
profile: PlatformProfileV2
controller_capability_id: str
controller_capability_hash: str
traversability_derivation_id: str
terrain_feature_source_id: str
terrain_feature_source_hash: str
body_length_m: float
body_width_m: float
safety_margin_m: float
min_publish_confidence: float
reverse_enabled: bool
max_forward_speed_mps: float
max_reverse_speed_mps: float
max_linear_accel_mps2: float
max_linear_decel_mps2: float
max_yaw_rate_radps: float
max_yaw_accel_radps2: float
min_segment_duration_s: float
max_segment_duration_s: float
lookahead_min_m: float
lookahead_nominal_m: float
lookahead_max_m: float
max_corridors: int = 3
max_corridor_candidates: int = 24
max_topology_crossing_tests: int = 16_777_216
max_segments: int = 48
max_sqp_iterations: int = 40
max_sqp_function_evaluations: int = 4096
sqp_ftol: float = 1.0e-10
hard_constraint_tolerance: float = 1.0e-9
max_risk_batch_inference_s: float = 0.004
max_total_risk_inference_s: float = 0.016
max_risk_feature_pose_samples: int = 8192
max_risk_feature_cells: int = 262_144
max_corridor_risk_segments: int = 2048
canonical_decimal_places: int = 12
max_l2_interval_records: int = 262_144
max_l2_candidate_cells: int = 1_000_000
max_l2_subdivision_depth: int = 24
continuous_separation_epsilon_m: float = 1.0e-9
repair_clearance_m: float = 1.0e-4
max_body_pose_cells: int = 4096
max_body_support_points: int = 1024
max_sqp_support_constraints: int = 131_072
max_sqp_broadphase_cells: int = 262_144
max_sqp_unsafe_cell_records: int = 65_536
max_sqp_distance_evaluations_per_callback: int = 16_777_216
max_snapshot_cells: int = 4_000_000
max_hierarchy_cells: int = 5_250_000
solver_memory_reservation_bytes: int = 16_777_216
```

Profile 审计必须要求：`PlatformKindV2.LEGGED`、新 capability、`simulation_proxy=False`、`max_traversable_slope_deg==30.0`、所有字段为精确内建类型且有限；三个 terrain-derivation/source ID 为非空精确字符串且 source hash 为 SHA-256；`0<max_forward_speed_mps<=0.8` 且 `0<=max_reverse_speed_mps<=0.8`；未启用倒退时 `max_reverse_speed_mps==0.0`，启用时必须严格大于零；加速、减速、最大横摆率和最大横摆加速度均严格大于零；`0<=min_publish_confidence<=1`；`5<=lookahead_min<=lookahead_nominal<=lookahead_max<=20`；嵌套 `PlatformProfileV2` 的 `goal_position_tolerance_m==0.25` 和 `goal_heading_tolerance_rad==0.08726646259971647` 与 Scout Mini SQP 合同对齐；滚动子目标位置使用 Euclidean norm，heading 使用绕回 `[-pi,pi)` 后的绝对误差；所有专用资源上限不得由请求动态放宽。具体机体尺寸、控制上限和上游 traversability/terrain-feature 身份由外部控制器 profile 提供并通过 profile hash 密封，不从地图或候选估计。

其余数值域同样冻结：`body_length_m/body_width_m>0`、`safety_margin_m>=0`、`0<min_segment_duration_s<=max_segment_duration_s`；所有 count/cap 字段必须是非 bool 的精确正 `int`，并满足 `1<=max_corridors<=3`、`max_corridors<=max_corridor_candidates<=24`、`max_topology_crossing_tests==16777216`、`1<=max_segments<=48`、`1<=max_sqp_iterations<=40`、`1<=max_sqp_function_evaluations<=4096`、L2 interval/cell/depth caps 不超过上列 v1 maxima、`max_body_pose_cells==4096`、`max_body_support_points==1024`、`max_sqp_support_constraints==131072`、`max_sqp_broadphase_cells==262144`、`max_sqp_unsafe_cell_records==65536`、`max_sqp_distance_evaluations_per_callback==16777216`、`max_risk_feature_pose_samples==8192`、`max_risk_feature_cells==262144`、`max_corridor_risk_segments==2048`、`max_snapshot_cells==4000000`、`max_hierarchy_cells==5250000`。Profile audit 令 `L=body_length+2*margin`、`W=body_width+2*margin`、`r=nextafter(hypot(L/2,W/2),+inf)`，以 `static_span=2*ceil(r/0.5)+3` 的平方作为任意 yaw 静态 rectangle-cell 上界；以 `(max(2,ceil(L/0.25)+1)*max(2,ceil(W/0.25)+1))` 作为固定 tensor support-lattice 点数。任一超 cap 则 profile unsupported；L2 动态 expansion 仍在请求内受同一 body-cell cap 约束并映射 typed resource failure，不允许尺寸/margin 演变成 internal error。Risk worker budgets require `0<max_risk_batch_inference_s<=max_total_risk_inference_s<=0.02`。`canonical_decimal_places` 必须恰为 `12`，`sqp_ftol==1.0e-10`、`hard_constraint_tolerance==1.0e-9`、`continuous_separation_epsilon_m==1.0e-9`、`repair_clearance_m==1.0e-4`，且 memory reservation 为正。滚动子目标容差是决策合同而非数值残差容差，SQP、L2、failure taxonomy 和 formal oracle 都显式使用嵌套且 profile-hashed 的两个 `PlatformProfileV2` 字段，不在专用 profile 复制第二份容差。测试可构造向下收紧的资源 cap，但不能放宽安全容差或改变 decimal12 语义。

资源 cap 规范性消歧：上段列出的 `==` 数值表示 dataclass 默认值与 v1 绝对上限，不表示唯一可接受值；实际 audit 规则是 `1<=field<=对应 v1 上限`（包括 topology、body-pose/support、SQP broadphase/unsafe/distance-evaluation、risk、snapshot/hierarchy 各 cap）。生产 profile 与测试 fixture 均可向下收紧，但请求不能动态放宽；不存在隐藏 test override。Profile audit 仅在尺寸派生数量超过 v1 绝对上限时判定 unsupported；若仅超过合法收紧后的 cap，必须在首个相关运行时预检中稳定返回 `legged_resource_budget_exceeded`，不落入 internal error。安全/数值容差与 decimal12 仍保持上段的 exact 合同。

公共成功链的字段固定为：

```python
@dataclass(frozen=True, slots=True)
class LeggedCorridorRiskGuideV1:
    segment_risk_upper_bounds: tuple[float, ...]
    max_upper_bound: float
    cumulative: float
    estimated_time_s: float
    estimated_energy: float
    guide_length_m: float
    risk_source: str
    model_id: str
    feature_schema_id: str
    calibration_id: str
    risk_aggregation_id: str = LEGGED_RISK_AGGREGATION_V1


@dataclass(frozen=True, slots=True)
class GlobalCorridorIntentV1:
    corridor_index: int
    cells_2m: tuple[Cell, ...]
    cells_1m: tuple[Cell, ...]
    cells_0p5m: tuple[Cell, ...]
    topology_signature: str
    risk_guide: LeggedCorridorRiskGuideV1
    terrain_snapshot_hash: str
    platform_profile_hash: str
    controller_capability_hash: str
    corridor_geometry_hash: str
    corridor_hash: str
    source_id: str = LEGGED_CORRIDOR_SOURCE_V1
    validation_level: ValidationLevelV2 = ValidationLevelV2.L1
    executable: bool = False


@dataclass(frozen=True, slots=True)
class TimedBodySegmentV1:
    start_state: LeggedBodyStateV1
    end_state: LeggedBodyStateV1
    start_time_s: float
    end_time_s: float
    a_mps2: float
    omega_radps: float
    duration_s: float
    distance_m: float
    relative_energy: float
    segment_hash: str
    segment_schema_id: str = LEGGED_BODY_SEGMENT_SCHEMA_V1
    validation_level: ValidationLevelV2 = ValidationLevelV2.L2


@dataclass(frozen=True, slots=True)
class LeggedBodyL2ReceiptV1:
    validator_id: str
    passed: bool
    validation_input_hash: str
    candidate_hash: str
    request_hash: str
    profile_hash: str
    controller_capability_hash: str
    terrain_snapshot_hash: str
    rolling_goal_hash: str
    risk_evidence_hash: str
    checked_interval_count: int
    checked_cell_count: int
    validated_distance_m: float
    validated_relative_energy: float
    validated_duration_s: float
    validated_control_slew: float
    cost_breakdown_hash: str
    repair_applied: bool


@dataclass(frozen=True, slots=True)
class TimedBodyTrajectoryV1:
    segments: tuple[TimedBodySegmentV1, ...]
    rolling_goal_state: PoseStateV2
    rolling_goal_hash: str
    source_candidate_hash: str
    request_hash: str
    profile_hash: str
    controller_capability_hash: str
    terrain_snapshot_hash: str
    risk_evidence_hash: str
    l2_receipt: LeggedBodyL2ReceiptV1
    trajectory_hash: str
    validation_level: ValidationLevelV2 = ValidationLevelV2.L2


@dataclass(frozen=True, slots=True)
class LeggedRollingPlanV1:
    request_id: str
    mission_goal_state: PoseStateV2
    rolling_goal_state: PoseStateV2
    global_corridor_intent: GlobalCorridorIntentV1
    local_body_trajectory: TimedBodyTrajectoryV1
    mission_complete: bool
    terrain_snapshot_hash: str
    platform_profile_hash: str
    controller_capability_hash: str
    risk_evidence: LeggedRiskEvidenceV1
    cost_breakdown: CostBreakdownV2
    telemetry: LeggedRollingTelemetryV1
    decision_hash: str
    selection_policy: str = LEGGED_SELECTION_POLICY_V1
    global_optimality_claimed: bool = False
    schema_version: str = LEGGED_ROLLING_PLAN_SCHEMA_V1
```

`LeggedRiskEvidenceV1` 字段固定为：

```python
segment_risk_upper_bounds: tuple[float, ...]
max_segment_risk_upper_bound: float
cumulative_route_risk: float
terrain_risk: float
clearance_risk: float
confidence_risk: float
kinematic_aggressiveness_risk: float
model_uncertainty: float
ood_detected: bool
risk_source: str
model_id: str
feature_schema_id: str
calibration_id: str
risk_aggregation_id: str
snapshot_hash: str
candidate_hash: str
fallback_reason: str | None
```

前 16 项精确对应规格第 8.2 节，`fallback_reason` 记录模型拒绝原因；`candidate_hash`、`snapshot_hash` 和派生 `risk_evidence_hash` 相互可复核。`LeggedRollingTelemetryV1` 继承 `SearchTelemetryV2`，增加：

```python
corridor_count: int
corridor_attempt_count: int
selected_corridor_index: int | None
ordered_corridor_hashes: tuple[str, ...]
sqp_iteration_count: int
sqp_function_evaluation_count: int
l2_attempt_count: int
l2_interval_count: int
l2_cell_count: int
repair_attempt_count: int
risk_fallback_count: int
cache_hit: bool
selection_key: tuple[float, float, float, float, float, str] | None
decision_hash: str | None
```

滚动子目标不是裸 `PoseStateV2`；其内部身份合同固定为：

```python
@dataclass(frozen=True, slots=True)
class LeggedRollingSubgoalV1:
    pose: PoseStateV2
    is_mission_goal: bool
    distance_from_start_m: float
    terminal_speed_reference_mps: float
    terminal_speed_is_hard: bool
    start_state_hash: str
    mission_goal_hash: str
    corridor_hash: str
    terrain_snapshot_hash: str
    profile_hash: str
    controller_capability_hash: str
    rolling_goal_hash: str
    schema_id: str = LEGGED_ROLLING_SUBGOAL_SCHEMA_V1
    memory_token: LeggedPlanningMemoryTokenV1 = field(repr=False, compare=False)
```

`rolling_goal_hash` covers every semantic field plus schema ID and explicitly excludes the nonserializable scoped memory token, whose request/ledger/owner/byte identity is separately checked. `is_mission_goal=True` additionally requires `pose` to equal the exact requested mission pose、`terminal_speed_reference_mps==0.0` and `terminal_speed_is_hard=True`; false requires the corridor-derived pose and distance to remain within the approved lookahead contract、`terminal_speed_reference_mps==min(0.3, profile.max_forward_speed_mps)` and `terminal_speed_is_hard=False`. The mission terminal speed is an exact hard equality independently rechecked by L2; the intermediate value is only a hashed initializer reference and never enters merit or becomes a minimum-speed safety gate.

内部 SQP 字段固定为：

```python
LeggedBodySQPInitialGuessV1(
    corridor_hash: str,
    rolling_goal_hash: str,
    states: tuple[LeggedBodyStateV1, ...],
    a_mps2: tuple[float, ...],
    omega_radps: tuple[float, ...],
    duration_s: tuple[float, ...],
    initializer_id: str,
    memory_token: LeggedPlanningMemoryTokenV1,
)

CanonicalLeggedBodyCandidateV1(
    candidate_hash: str,
    request_hash: str,
    profile_hash: str,
    controller_capability_hash: str,
    terrain_snapshot_hash: str,
    corridor_hash: str,
    corridor_geometry_hash: str,
    rolling_goal_hash: str,
    initial_guess_hash: str,
    solver_contract_id: str,
    sqp_geometry_id: str,
    canonicalization_id: str,
    backend_id: str,
    backend_version_hash: str,
    repair_applied: bool,
    repair_constraint_hash: str | None,
    states: tuple[LeggedBodyStateV1, ...],
    a_mps2: tuple[float, ...],
    omega_radps: tuple[float, ...],
    duration_s: tuple[float, ...],
    objective_value: float,
    status: LeggedBodySQPStatusV1,
)

LeggedBodyL2CounterexampleV1(
    request_hash: str,
    profile_hash: str,
    controller_capability_hash: str,
    terrain_snapshot_hash: str,
    corridor_hash: str,
    rolling_goal_hash: str,
    risk_evidence_hash: str,
    validation_input_hash: str,
    candidate_hash: str,
    segment_index: int,
    interval_start_s: float,
    interval_end_s: float,
    segment_phase_start: float,
    segment_phase_end: float,
    reason_code: str,
    cell: Cell | None,
    repairable: bool,
    counterexample_hash: str,
)
```

`validation_input_hash` covers the exact request/profile/controller/snapshot/corridor/rolling-goal/risk-evidence/candidate identities and validator ID before either verdict exists. `LeggedBodyRepairConstraintV1` carries that same complete lineage、`source_validation_input_hash`、`source_counterexample_hash`、`source_validation_record_hash`、segment index、normalized phase interval、cell、clearance and its own hash. `LeggedBodySQPOptimizationResultV1` carries status、an optional exact `CanonicalLeggedBodyCandidateV1`、iteration/function-evaluation/solve-attempt counts、optional reason、`reserve_audit: LeggedBodyL2TailReserveV1 | None` and two internal optional handles `solver_candidate_token/post_solver_tail_token`；only a feasible candidate requires the audit and both live handles. Their ledger/request identities must match, `solver_candidate_token.bytes==reserve_audit.solver_candidate_bytes`、`post_solver_tail_token.bytes==reserve_audit.tail_bytes-reserve_audit.solver_candidate_bytes`, and their sum equals the complete sealed tail. A result stopped after reserve assessment may retain an immutable audit but no handle；backend-unavailable/pre-assessment results have neither. Tokens are excluded from candidate/public hashes and cannot be serialized. Every non-feasible result has both handles `None` after releasing provisional reservations. `LeggedRiskEvaluationV1` is an internal pair of exact `evidence` and its newly transferred live `memory_token`.

`LeggedBodyTrajectoryL2ResultV1` carries the exact `validation_input_hash`、`passed`、optional public trajectory、optional receipt、optional counterexample、its own `validation_record_hash` and two internal optional handles `promotion_token/repair_context_token`. Exactly one ownership branch is valid: pass has only `promotion_token`; a repairable localized failure has only `repair_context_token`, which retains prior candidate bytes、risk evidence and failed-record bytes; a nonrepairable failure has neither after cleanup. All handles are nonserializable/excluded from semantic hashes. The record hash covers validation input、verdict and receipt/counterexample branch, so a repair can be authorized only by the exact preceding failed validation object. `CanonicalLeggedBodyCandidateV1` 在 Task 1 定义数据合同，Task 6 实现 materializer/codec，Task 7 才允许由可行 solver result 构造它；不存在第二个同义 candidate 类型。Initial candidates require `repair_applied=False/repair_constraint_hash=None`; repaired candidates require `True` and the exact accepted constraint hash。

跨 provider/API 的内部所有权类型固定为 `LeggedRollingPlanBuildV1(plan,canonical_candidate_bytes,memory_token)` 与 `LeggedProviderResultV1(outcome,canonical_candidate_bytes,output_token)`。Plan-build token 合并 L2 promotion、被选 corridor retained token、公开 plan 与仅供 API 后置复核的 canonical candidate bytes capacity；success wrapper 的 bytes 必须与 receipt candidate hash 一致且由同一 output token 覆盖，provider failure wrapper 的 bytes/token 必须均为 `None`。两类 wrapper 均不可复制/序列化且不是公共返回联合；Task 10 完成独立 encode→decode→encode 和全部后置校验后丢弃内部 bytes，才可将 plan publicize。

`LeggedBodySQPStatusV1` 枚举值固定为 `FEASIBLE`、`INFEASIBLE`、`BACKEND_UNAVAILABLE`、`NUMERIC_FAILURE`、`RESOURCE_EXHAUSTED`、`DEADLINE_EXPIRED`。Task 1 同时定义只在新能力内部使用的 `LeggedMapProfileMismatchV1`、`LeggedCorridorBudgetExceededV1`、`LeggedPlanningDeadlineExpiredV1`、`LeggedLocalSubgoalUnavailableV1`、`LeggedSQPInitializationFailedV1`、`LeggedSQPDeadlineExpiredV1`、`LeggedSQPResourceCutoffV1`、`LeggedPlanningMemoryIdentityErrorV1`、`LeggedBodyCodecErrorV1`，API 边界将它们映射为上表稳定 reason/category，不让私有异常类型逃逸。`LeggedCorridorBudgetExceededV1` 只映射 `legged_corridor_budget_exceeded`；两个 SQP stop 异常分别只映射 `planning_deadline_expired` 和 `legged_resource_budget_exceeded`；stale/wrong-ledger/wrong-owner/wrong-byte token 统一抛 memory identity error 并映射 `legged_identity_mismatch`，所有映射均保持 deadline precedence。

同一内部异常合同还定义 `LeggedProviderSealErrorV1`，仅表示 provider/trusted-ops 静态封印在请求派发前失配，并稳定映射 `legged_body_profile_unsupported`；它不与运行中 identity drift 混用。

失败映射固定为：

| reason | `FailureCategoryV2` |
|---|---|
| `legged_body_profile_unsupported` | `UNSUPPORTED_CAPABILITY` |
| `legged_map_profile_mismatch` | `INVALID_REQUEST` |
| `legged_start_invalid` | `UNSAFE_START` |
| `legged_goal_invalid` | `UNSAFE_GOAL` |
| `legged_no_global_corridor`、`legged_local_subgoal_unavailable`、`legged_goal_tolerance_exceeded` | `GOAL_POSE_UNREACHABLE` |
| `legged_sqp_initialization_failed`、`legged_sqp_infeasible` | `NO_COMPLETE_ROUTE` |
| `legged_candidate_l2_rejected`、`legged_repair_l2_rejected` | `VALIDATION_FAILED` |
| `legged_corridor_budget_exceeded`、`legged_resource_budget_exceeded` | `RESOURCE_LIMIT` |
| `planning_deadline_expired` | `TIMEOUT` |
| `legged_sqp_numeric_contract_failed`、`legged_identity_mismatch`、`legged_internal_error` | `INTERNAL_ERROR` |

任何 failure 均不包含 `TimedBodyTrajectoryV1`、candidate segments、SQP iterate 或可执行局部前缀。由于这是显式足式 API，即使 profile/provider 尚未解析，复用的 `PlanningFailureV2.platform_kind` 也固定为 `PlatformKindV2.LEGGED`；不得使用只允许通用 `platform_profile_unresolved` 的 `platform_kind=None` 分支。

---

## Implementation Tasks

### Task 1: Freeze the Opt-In Profile, Request, Result, Evidence, and Failure Contracts

**Files:**
- Modify: `path-planner/pyproject.toml`
- Create: `path-planner/src/path_planner/v2/legged_rolling_profiles.py`
- Create: `path-planner/src/path_planner/v2/legged_rolling_contracts.py`
- Create: `path-planner/src/path_planner/v2/legged_body_sqp_contracts.py`
- Modify: `path-planner/src/path_planner/v2/__init__.py`
- Modify: `path-planner/tests/test_package_imports.py`
- Modify: `path-planner/tests/test_v2_profiles.py`
- Create: `path-planner/tests/test_v2_legged_rolling_contracts.py`

**Interfaces:**
- Consumes: exact existing `PlatformProfileV2`、`PlatformKindV2`、`PoseStateV2`、`ObjectiveProfileV2`、`ResourceBudgetV2`、`FailureCategoryV2`、`FailureEvidenceV2`、`CostBreakdownV2`、`ValidationLevelV2`、`TerrainSnapshotV2`。
- Produces: all frozen constants above; the single decimal12 scalar primitive `canonicalize_legged_body_scalar_v1()`; `LeggedBodyStateV1`、`LeggedRollingRequestV1`、`LeggedRollingBodySQPProfileV1`、`LeggedRollingProfileRegistryV1`、`GlobalCorridorIntentV1`、`LeggedCorridorRiskGuideV1`、`LeggedRiskEvidenceV1`、`LeggedRollingSubgoalV1`、`TimedBodySegmentV1`、`TimedBodyTrajectoryV1`、`LeggedBodyL2ReceiptV1`、`LeggedRollingTelemetryV1`、`LeggedRollingPlanV1`、`LeggedRollingOutcomeV1`、request-scoped `LeggedPlanningMemoryLedgerV1` and internal SQP result/resource/counterexample/repair types。`LeggedRollingOutcomeV1` 的失败分支是共享 `PlanningFailureV2`。

Internal-only ownership results also include `LeggedOutputEscrowV1`、`LeggedPostSolverPhaseTokensV1`、`LeggedCandidateRepresentationTokensV1`、`LeggedCorridorAttemptResultV1`、`LeggedRollingPlanBuildV1`、`LeggedProviderResultV1` and `LeggedAPIPostconditionResultV1`；all are exact-type、noncopyable/nonserializable and excluded from public unions/hashes.

- [ ] **Step 1: Write RED tests for dependency isolation and exact profile identity**

```python
def test_legged_sqp_is_an_optional_exact_dependency() -> None:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'legged-sqp = [' in text
    assert text.count('"scipy==1.18.0"') == 2  # wheel-sqp and legged-sqp


def test_legged_rolling_profile_seals_controller_and_body_contract() -> None:
    profile = make_legged_rolling_profile()
    assert profile.profile.platform_kind is PlatformKindV2.LEGGED
    assert profile.profile.capability_revision == LEGGED_ROLLING_CAPABILITY_V1
    assert profile.profile.simulation_proxy is False
    assert profile.profile.max_traversable_slope_deg == 30.0
    assert profile.max_forward_speed_mps == 0.8
    assert profile.lookahead_min_m == 5.0
    assert profile.lookahead_nominal_m == 10.0
    assert profile.lookahead_max_m == 20.0
    assert len(profile.controller_capability_hash) == 64
    assert len(legged_rolling_profile_hash_v1(profile)) == 64
    assert not hasattr(profile, "__dict__")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_forward_speed_mps", nextafter(0.8, inf)),
        ("min_publish_confidence", nextafter(1.0, inf)),
        ("lookahead_min_m", nextafter(5.0, -inf)),
        ("lookahead_max_m", nextafter(20.0, inf)),
        ("max_corridors", 4),
    ],
)
def test_legged_rolling_profile_rejects_contract_drift(field, value) -> None:
    with pytest.raises((TypeError, ValueError)):
        make_legged_rolling_profile(**{field: value})
```

- [ ] **Step 2: Write RED tests for request bounds and non-masquerading output**

```python
@pytest.mark.parametrize("period", [0.2, 0.5])
def test_legged_request_accepts_exact_replan_boundaries(period: float) -> None:
    request = make_request(replan_period_s=period)
    assert type(request.replan_period_s) is float
    assert request.replan_period_s == period


@pytest.mark.parametrize("period", [nextafter(0.2, -inf), nextafter(0.5, inf), True])
def test_legged_request_rejects_invalid_replan_period(period: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        make_request(replan_period_s=period)


def test_rolling_success_is_not_generic_complete_goal_success() -> None:
    plan = make_plan(mission_complete=False)
    assert type(plan) is LeggedRollingPlanV1
    assert not isinstance(plan, PlanningSuccessV2)
    assert plan.global_corridor_intent.executable is False
    assert plan.local_body_trajectory.validation_level is ValidationLevelV2.L2


def test_failure_has_no_partial_executable_payload() -> None:
    failure = make_failure("legged_sqp_infeasible")
    assert type(failure) is PlanningFailureV2
    assert not hasattr(failure, "route")
    assert not hasattr(failure, "local_body_trajectory")
```

- [ ] **Step 3: Run RED tests**

Run from `path-planner/`:

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_package_imports.py tests/test_v2_profiles.py tests/test_v2_legged_rolling_contracts.py -q -x
```

Expected: failures are limited to the absent optional extra and new modules/types; all selected pre-existing tests remain green.

- [ ] **Step 4: Add the optional extra and implement exact public dataclasses**

Add without touching the base dependency list:

```toml
legged-sqp = [
  "scipy==1.18.0",
]
```

Implement the core request/state relation exactly:

```python
@dataclass(frozen=True, slots=True)
class LeggedBodyStateV1:
    x_m: float
    y_m: float
    yaw_rad: float
    v_mps: float


@dataclass(frozen=True, slots=True)
class LeggedRollingRequestV1:
    request_id: str
    platform_profile_id: str
    controller_capability_id: str
    current_body_state: LeggedBodyStateV1
    mission_goal_state: PoseStateV2
    terrain_snapshot: TerrainSnapshotV2
    objective_profile: ObjectiveProfileV2
    resource_budget: ResourceBudgetV2
    timeout_s: float
    replan_period_s: float
    determinism_seed: int
    schema_version: str = LEGGED_ROLLING_REQUEST_SCHEMA_V1

    def __post_init__(self) -> None:
        _require_exact_nonempty_string(self.request_id, "request_id")
        _require_exact_nonempty_string(self.platform_profile_id, "platform_profile_id")
        _require_exact_nonempty_string(self.controller_capability_id, "controller_capability_id")
        _require_exact_type(self.current_body_state, LeggedBodyStateV1, "current_body_state")
        _require_exact_type(self.mission_goal_state, PoseStateV2, "mission_goal_state")
        _require_exact_type(self.terrain_snapshot, TerrainSnapshotV2, "terrain_snapshot")
        _require_exact_type(self.objective_profile, ObjectiveProfileV2, "objective_profile")
        _require_exact_type(self.resource_budget, ResourceBudgetV2, "resource_budget")
        _require_exact_nonnegative_float(self.timeout_s, "timeout_s")
        period = _require_exact_finite_float(self.replan_period_s, "replan_period_s")
        if not 0.2 <= period <= 0.5:
            raise ValueError("replan_period_s must be in [0.2, 0.5]")
        _require_exact_int_not_bool(self.determinism_seed, "determinism_seed")
        if self.schema_version != LEGGED_ROLLING_REQUEST_SCHEMA_V1:
            raise ValueError("request schema mismatch")
```

Implement all listed public/internal frozen dataclasses with exact built-in type checks, finite scalar checks, immutable tuples, sorted unique keys, SHA-256 validation, and cross-field invariants. Define the decimal12 primitive once in `legged_rolling_contracts.py` before the public segment types; internal contracts and serialization import it rather than reimplementing it. Freeze `legged_rolling_request_hash_v1(request, *, terrain_snapshot_hash)` to hash every request scalar/state/objective/resource/schema field while replacing the large snapshot arrays with the exact already-audited SHA-256 argument; it rejects a malformed hash and is the sole request-hash helper used by solver/provider/formal joins. `TimedBodySegmentV1` requires positive duration、nonnegative analytic `distance_m`、exact `end_time_s == canonicalize_legged_body_scalar_v1(start_time_s + duration_s)`, L2, start/end velocity consistency and a valid segment hash; using raw binary-float equality here would reject valid decimal12 sums such as `0.1 + 0.2`. `LeggedRollingPlanV1` requires matching request/profile/controller/snapshot/candidate/trajectory/rolling-goal hashes, requires `legged_cost_breakdown_hash_v1(cost_breakdown)==local_body_trajectory.l2_receipt.cost_breakdown_hash`, and enforces the fixed selection/global-optimality flags.

`LeggedPlanningMemoryLedgerV1.create(request_hash, profile_hash, resource_budget)` is the only constructor for one planner-thread-confined mutable memory ledger per request. It stores the exact nonzero ceiling (or unlimited sentinel for `max_memory_bytes==0`), overflow-safe `live_bytes/peak_bytes`, monotonically increasing token IDs and an owner-thread ID. `reserve(owner, exact_bytes)` checks identity and ceiling before any allocation, then returns a single-use scoped token. `split(parent, ((owner_a,bytes_a),...))` requires child-byte sum to equal the parent exactly, invalidates the parent and creates ordered child tokens without changing live/peak bytes; `merge(children,new_owner)` requires same ledger/request, sorted unique live handles and overflow-safe summed bytes, invalidates every child and creates one handle without changing live/peak; `transfer(token,new_owner)` does the corresponding one-to-one operation. `release(token)` requires exact owner/token/byte agreement and cannot underflow. The sole additional terminal operation `publicize(output_token)` is callable only by the Task 10 root API scope after all postconditions and the final deadline check; it requires owner `api_validated_public_output`, invalidates the handle, subtracts its exact bytes and returns no token. It does not authorize another allocation or hide a peak—it marks the already-accounted immutable object as ownership transferred to the caller. Copy/deepcopy/pickle、stale token、double release and cross-thread use are rejected. API postcondition、hierarchy、corridor search/retained intents、risk features/fallback、SQP problem buffers、codec and L2 all reserve or consume declared tokens through this same object; exactly one ledger is created by Task 10 and injected downward, and no provider/module may create a competing request-memory total. Cached input arrays and the caller-owned immutable snapshot are not re-counted, but every new copy/materialization is. `max_memory_bytes==0` disables only the ceiling, never overflow checks or telemetry.

`LeggedOutputEscrowV1` is a noncopyable root-API-scope object created and registered before provider dispatch. `source_scope.handoff_to(destination_scope, token)` and `escrow.deposit_from(source_scope, token)` execute one ledger-locked transaction: validate source ownership/destination identity, register the live handle at the destination, then remove it from the source, with rollback before releasing the lock on any exception. There is never a detached/unowned state. Escrow holds at most one output handle and its root scope cleanup releases any deposited-but-unclaimed handle, including when `KeyboardInterrupt`、`SystemExit` or `MemoryError` interrupts provider return. `api_scope.claim_from_escrow(escrow, token)` is the inverse atomic transaction. Deposit/claim do not change live/peak bytes or token ID；wrong/duplicate/empty operations raise `LeggedPlanningMemoryIdentityErrorV1`.

- [ ] **Step 5: Implement strict profile audit and sorted registry**

```python
@dataclass(frozen=True, slots=True)
class LeggedRollingProfileRegistryV1:
    profiles: tuple[LeggedRollingBodySQPProfileV1, ...]

    def __post_init__(self) -> None:
        if type(self.profiles) is not tuple:
            raise TypeError("profiles must be exact tuple")
        audited = tuple(audit_legged_rolling_profile_v1(value) for value in self.profiles)
        ids = tuple(value.profile.profile_id for value in audited)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise ValueError("profiles must have unique sorted profile ids")
        object.__setattr__(self, "profiles", audited)

    def resolve(self, profile_id: str) -> LeggedRollingBodySQPProfileV1 | None:
        _require_exact_nonempty_string(profile_id, "profile_id")
        return next((value for value in self.profiles if value.profile.profile_id == profile_id), None)
```

`legged_rolling_profile_hash_v1()` must hash canonical JSON of every base/profile/controller/resource field; object address、wall time and registry order never enter the hash.

- [ ] **Step 6: Run GREEN and old-contract regressions**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_package_imports.py tests/test_v2_profiles.py tests/test_v2_legged_rolling_contracts.py tests/test_v2_contracts.py tests/test_v2_legged_provider.py -q
```

Expected: all pass; old `LeggedProfileV2` remains exactly `simulation_proxy_static_crawl/v1` and no old test imports the new capability implicitly.

- [ ] **Step 7: Review and commit only Task 1 files**

```powershell
git -C path-planner diff --check
git -C path-planner add pyproject.toml src/path_planner/v2/legged_rolling_profiles.py src/path_planner/v2/legged_rolling_contracts.py src/path_planner/v2/legged_body_sqp_contracts.py src/path_planner/v2/__init__.py tests/test_package_imports.py tests/test_v2_profiles.py tests/test_v2_legged_rolling_contracts.py
git -C path-planner commit -m "feat: freeze legged rolling planner contracts"
git add path-planner
git commit -m "build: integrate legged rolling planner contracts"
```

Before each `git add`, verify `git -C path-planner status --short` and leave every Wheel-SQP or unrelated path unstaged.

---

### Task 2: Build the Profile-Bound Hard Safety View and Conservative 0.5/1.0/2.0 Hierarchy

**Files:**
- Create: `path-planner/src/path_planner/v2/legged_rolling_safety.py`
- Modify: `path-planner/src/path_planner/v2/terrain.py`
- Modify: `path-planner/src/path_planner/v2/geometry.py`
- Create: `path-planner/tests/test_v2_legged_rolling_safety.py`
- Modify: `path-planner/tests/test_v2_geometry.py`
- Modify: `path-planner/tests/test_v2_terrain.py`

**Interfaces:**
- Consumes: `TerrainSnapshotV2`、`TerrainProvenanceV2.details`、`snapshot_hash()`、`FineGridGeometryV2`、`PlanningDeadlineV2`、`LeggedRollingBodySQPProfileV1`、request-scoped `LeggedPlanningMemoryLedgerV1`。
- Produces: scratch-token-capable `snapshot_hash_audit_with_checkpoints_v2()`/`snapshot_hash_with_checkpoints_v2()`、`LeggedSafetyQueryV1`、`LeggedHardSafetyViewV1.build(snapshot, profile, deadline, *, memory_ledger=None, hash_scratch_token=None)`、`LeggedHierarchyCellV1`、internal nonserializable `LeggedConservativeHierarchyV1(...,memory_token)` from `build(view, deadline, *, memory_ledger, cache=None)`、`LeggedDerivedMapCacheKeyV1`、`LeggedDerivedMapCacheV1`、`conservative_body_pose_cells()`。

- [ ] **Step 1: Write RED tests for every hard boundary and provenance binding**

```python
def test_slope_and_confidence_boundaries_are_closed() -> None:
    profile = make_profile(min_publish_confidence=0.8)
    assert make_view(slope=30.0, confidence=0.8, profile=profile).query(CELL).passed
    assert make_view(slope=nextafter(30.0, inf), confidence=0.8, profile=profile).query(CELL).reason_code == "terrain_slope_exceeded"
    assert make_view(slope=30.0, confidence=nextafter(0.8, -inf), profile=profile).query(CELL).reason_code == "terrain_confidence_below_profile"


def test_only_exact_half_meter_authority_snapshot_is_accepted() -> None:
    with pytest.raises(LeggedMapProfileMismatchV1):
        LeggedHardSafetyViewV1.build(make_snapshot(resolution_m=1.0), PROFILE, OPEN_DEADLINE)
    assert LeggedHardSafetyViewV1.build(make_snapshot(resolution_m=0.5), PROFILE, OPEN_DEADLINE).snapshot.geometry.resolution_m == 0.5


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_traversability_derivation_id",
        "wrong_controller_capability_id",
        "wrong_controller_capability_hash",
        "missing_terrain_feature_source_id",
        "wrong_terrain_feature_source_hash",
    ],
)
def test_map_profile_binding_drift_is_rejected(mutation: str) -> None:
    with pytest.raises(LeggedMapProfileMismatchV1):
        LeggedHardSafetyViewV1.build(snapshot_with_provenance_mutation(mutation), PROFILE, OPEN_DEADLINE)


def test_unknown_obstacle_not_traversable_and_oob_fail_closed() -> None:
    view = make_mixed_view()
    assert [view.query(cell).reason_code for cell in CASE_CELLS] == [
        "terrain_unknown",
        "terrain_hard_obstacle",
        "terrain_not_traversable",
        "terrain_out_of_bounds",
    ]


def test_synthetic_proxy_is_never_upgraded_to_physical_obstacle_truth() -> None:
    snapshot = make_snapshot(
        source_kind="synthetic_terrain_obstacle_proxy/v1",
        physical_obstacle_cells_written=False,
    )
    view = LeggedHardSafetyViewV1.build(snapshot, PROFILE, OPEN_DEADLINE)
    assert view.snapshot.provenance.physical_obstacle_cells_written is False
    assert view.snapshot.provenance.source_kind == "synthetic_terrain_obstacle_proxy/v1"
```

- [ ] **Step 2: Write RED tests for conservative aggregation, body cells, and cache equivalence**

```python
def test_one_unsafe_fine_child_blocks_1m_and_2m_parent() -> None:
    view = make_view_with_one_low_confidence_child()
    hierarchy = LeggedConservativeHierarchyV1.build(view, OPEN_DEADLINE, memory_ledger=MEMORY_LEDGER)
    assert hierarchy.query(scale=1, cell=Cell(0, 0)).passed is False
    assert hierarchy.query(scale=2, cell=Cell(0, 0)).passed is False
    assert hierarchy.query(scale=4, cell=Cell(0, 0)).passed is False


def test_incomplete_boundary_parent_is_out_of_bounds_not_partially_safe() -> None:
    hierarchy = LeggedConservativeHierarchyV1.build(make_view(width=5, height=5), OPEN_DEADLINE, memory_ledger=MEMORY_LEDGER)
    edge = hierarchy.query(scale=4, cell=Cell(1, 1))
    assert edge.passed is False
    assert "terrain_out_of_bounds" in edge.reason_codes


def test_platform_neutral_body_helper_preserves_existing_wheel_cells() -> None:
    generic = conservative_body_pose_cells(POSE, GEOMETRY, length_m=0.612, width_m=0.580, safety_margin_m=0.0)
    wheel = conservative_wheel_pose_cells(POSE, GEOMETRY, body_length_m=0.612, body_width_m=0.580, safety_margin_m=0.0)
    assert generic == wheel


def test_cache_on_off_changes_only_cache_telemetry() -> None:
    cold = build_safety_bundle(cache=None)
    cache = LeggedDerivedMapCacheV1()
    first = build_safety_bundle(cache=cache)
    second = build_safety_bundle(cache=cache)
    assert cold.semantic_digest == first.semantic_digest == second.semantic_digest
    assert (cold.cache_hit, first.cache_hit, second.cache_hit) == (False, False, True)


def test_derived_map_cache_is_single_entry_replacement_not_cross_request_growth() -> None:
    cache = LeggedDerivedMapCacheV1()
    for snapshot in THREE_DISTINCT_SNAPSHOTS:
        build_safety_bundle(snapshot=snapshot, cache=cache)
        assert cache.entry_count <= 1
        assert cache.stored_bytes <= LEGGED_DERIVED_CACHE_MAX_BYTES_V1
    assert cache.key.snapshot_hash == snapshot_hash(THREE_DISTINCT_SNAPSHOTS[-1])


def test_streamed_snapshot_hash_is_byte_identical_to_existing_hash() -> None:
    snapshot = make_large_snapshot()
    assert snapshot_hash_with_checkpoints_v2(snapshot, checkpoint=noop_checkpoint) == snapshot_hash(snapshot)


def test_one_row_ultrawide_snapshot_checks_deadline_inside_the_row() -> None:
    with pytest.raises(LeggedPlanningDeadlineExpiredV1):
        build_view_with_scripted_clock(make_snapshot(width=PROFILE.max_snapshot_cells, height=1))


def test_huge_provenance_is_rejected_or_stops_at_bounded_byte_checkpoint() -> None:
    with pytest.raises((LeggedMapProfileMismatchV1, LeggedPlanningDeadlineExpiredV1)):
        build_view_with_huge_details_and_scripted_clock()


def test_legacy_snapshot_hash_still_accepts_metadata_larger_than_legged_caps() -> None:
    snapshot = make_legacy_snapshot_with_257_details_and_70k_value()
    assert snapshot_hash(snapshot) == LEGACY_OVERSIZED_METADATA_GOLDEN_HASH
    with pytest.raises(LeggedMapProfileMismatchV1):
        LeggedHardSafetyViewV1.build(snapshot, PROFILE, OPEN_DEADLINE)


def test_hierarchy_preflights_cell_and_memory_bounds_before_allocation() -> None:
    ledger = make_memory_ledger(max_memory_bytes=HIERARCHY_BYTES)
    with pytest.raises(LeggedSQPResourceCutoffV1):
        LeggedConservativeHierarchyV1.build(VIEW, OPEN_DEADLINE, memory_ledger=ledger)
    assert HIERARCHY_ALLOCATION_COUNTER == 0


def test_one_row_ultrawide_hierarchy_checks_every_fixed_cell_batch() -> None:
    with pytest.raises(LeggedPlanningDeadlineExpiredV1):
        build_hierarchy_with_clock_crossing_mid_wide_row()


@pytest.mark.parametrize("phase", ["snapshot_hash", "start_body_cells", "goal_body_cells"])
def test_hard_safety_preprocessing_stops_at_the_shared_deadline(phase: str) -> None:
    with pytest.raises(LeggedPlanningDeadlineExpiredV1):
        build_or_validate_with_clock_expiring_mid_phase(phase)


def test_profile_whose_body_lattice_exceeds_v1_cap_is_rejected_before_request() -> None:
    with pytest.raises(ValueError, match="body.*cap"):
        audit_legged_rolling_profile_v1(make_oversized_body_profile())
```

- [ ] **Step 3: Run RED tests**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_legged_rolling_safety.py tests/test_v2_geometry.py tests/test_v2_terrain.py -q -x
```

Expected: failures name the absent safety module/helper; existing terrain and wheel geometry assertions stay green.

- [ ] **Step 4: Extract the generic rectangle helper without semantic drift**

Move the exact current body of `conservative_wheel_pose_cells()` into:

```python
def conservative_body_pose_cells(
    pose: Pose2D,
    geometry: FineGridGeometryV2,
    *,
    length_m: float,
    width_m: float,
    safety_margin_m: float,
    expansion_m: float = 0.0,
    checkpoint: Callable[[], None] | None = None,
    max_candidate_cells: int | None = None,
) -> tuple[Cell, ...]:
    return _conservative_oriented_rectangle_cells_v2(
        pose,
        geometry,
        length_m=length_m,
        width_m=width_m,
        safety_margin_m=safety_margin_m,
        expansion_m=expansion_m,
        checkpoint=checkpoint,
        max_candidate_cells=max_candidate_cells,
    )
```

Keep the old public helper as an exact argument-renaming delegate with both new options `None`. The private implementation preserves current rounding、`nextafter`、cell-square expansion and enumeration order. When provided, it computes the candidate bounding-box count before allocation and raises the caller's typed resource error if it exceeds `max_candidate_cells`; it invokes `checkpoint` before work, every `BODY_CELL_CHECKPOINT_BATCH_V1=256` candidate cells, after enumeration and before return. The `None` path remains byte-for-byte identical for existing wheel callers.

Refactor the current terrain digest into `snapshot_hash_with_checkpoints_v2(snapshot, *, checkpoint=None)`. It writes the exact existing domain/metadata/name/dtype/shape/total-byte-length framing, then feeds canonical C-order layer bytes in chunks of at most `SNAPSHOT_HASH_CELL_BATCH_V2=4096` elements and `SNAPSHOT_HASH_BYTE_BATCH_V2=1_048_576` bytes, whichever binds first; a single ultra-wide row is therefore split, and negative-zero normalization is per chunk. Metadata/provenance uses the identical existing canonical byte grammar but emits it incrementally with a checkpoint every `SNAPSHOT_METADATA_BYTE_BATCH_V2=4096` bytes; no single value may exceed `MAX_SNAPSHOT_METADATA_VALUE_BYTES_V2=65_536`, provenance detail count may not exceed `MAX_TERRAIN_PROVENANCE_DETAIL_COUNT_V1=256`, and total canonical metadata/provenance bytes may not exceed `MAX_SNAPSHOT_METADATA_BYTES_V2=1_048_576`. It calls the checkpoint before exact-shape/type audit、before and after provenance audit、before/after every bounded byte/cell chunk and before return. `snapshot_hash(snapshot)` delegates with `checkpoint=None`, and golden/equivalence tests prove the digest is unchanged; no full-layer/full-row copy、unbounded metadata serialization or one-shot `tobytes()` remains on the deadline-aware path.

Compatibility correction for that refactor is normative: implement `snapshot_hash_audit_with_checkpoints_v2(snapshot, *, checkpoint=None, limits=None) -> SnapshotHashAuditV2(snapshot_hash, canonical_byte_count)` and make `snapshot_hash_with_checkpoints_v2()` a digest-only delegate. The three metadata/provenance caps above live in `LEGGED_SNAPSHOT_METADATA_LIMITS_V1` and are enforced only when the specialized legged preflight passes `limits=LEGGED_SNAPSHOT_METADATA_LIMITS_V1`; `snapshot_hash()` and every old wheel/generic caller pass `limits=None`, retain the complete legacy acceptance domain, and still stream without an unbounded copy. The audit result obtains `canonical_byte_count` from the same bounded framing pass; callers must not rescan metadata or layer bytes merely to compute reserve input. A frozen oversized-legacy golden proves both unchanged digest and unchanged acceptance, while the same snapshot fails only at the opt-in legged boundary.

Planning-time scratch accounting is also exact: Task 2 owns the shared constant `SNAPSHOT_HASH_SCRATCH_BYTES_V2=1_052_672`, and the hash audit accepts optional paired `memory_ledger/scratch_memory_token`. Both `None` is the legacy/API-prehash mode and uses only one fixed local buffer of that size after the API's preflight ceiling check；both non-`None` is mandatory inside provider/L2 and borrows an exact live same-ledger token of that size, with no hidden byte buffer or metadata copy. The callee never releases the borrowed token；the owning scope releases it after the final hash checkpoint or during exception cleanup. Supplying only one、wrong bytes/owner or a second scratch allocation raises the typed memory identity/resource failure.

- [ ] **Step 5: Implement profile-bound queries and provenance audit**

```python
@dataclass(frozen=True, slots=True)
class LeggedHardSafetyViewV1:
    snapshot: TerrainSnapshotV2
    profile: LeggedRollingBodySQPProfileV1
    snapshot_hash: str
    snapshot_cell_count: int
    snapshot_canonical_byte_count: int
    profile_hash: str
    controller_capability_hash: str

    @classmethod
    def build(cls, snapshot: TerrainSnapshotV2, profile: LeggedRollingBodySQPProfileV1, deadline: PlanningDeadlineV2, *, memory_ledger=None, hash_scratch_token=None) -> "LeggedHardSafetyViewV1":
        _require_open_legged_deadline_v1(deadline)
        audited_profile = audit_legged_rolling_profile_v1(profile)
        audited_snapshot = _require_exact_snapshot(snapshot, checkpoint=lambda: _require_open_legged_deadline_v1(deadline))
        if audited_snapshot.geometry.resolution_m != 0.5:
            raise LeggedMapProfileMismatchV1("terrain resolution must be exactly 0.5m")
        cell_count = checked_mul(audited_snapshot.geometry.width, audited_snapshot.geometry.height)
        if cell_count > audited_profile.max_snapshot_cells:
            raise LeggedMapProfileMismatchV1("terrain snapshot exceeds profile cell cap")
        _audit_legged_provenance_binding(audited_snapshot.provenance, audited_profile, checkpoint=lambda: _require_open_legged_deadline_v1(deadline))
        snapshot_audit = snapshot_hash_audit_with_checkpoints_v2(
            audited_snapshot,
            checkpoint=lambda: _require_open_legged_deadline_v1(deadline),
            limits=LEGGED_SNAPSHOT_METADATA_LIMITS_V1,
            memory_ledger=memory_ledger,
            scratch_memory_token=hash_scratch_token,
        )
        _require_open_legged_deadline_v1(deadline)
        return cls(
            snapshot=audited_snapshot,
            profile=audited_profile,
            snapshot_hash=snapshot_audit.snapshot_hash,
            snapshot_cell_count=cell_count,
            snapshot_canonical_byte_count=snapshot_audit.canonical_byte_count,
            profile_hash=legged_rolling_profile_hash_v1(audited_profile),
            controller_capability_hash=audited_profile.controller_capability_hash,
        )

    def query(self, cell: Cell) -> LeggedSafetyQueryV1:
        if not self.snapshot.geometry.in_bounds(cell):
            return self._failed(cell, "terrain_out_of_bounds")
        row, column = cell.y, cell.x
        if not bool(self.snapshot.observed_mask[row, column]):
            return self._failed(cell, "terrain_unknown")
        if bool(self.snapshot.hard_obstacle_mask[row, column]):
            return self._failed(cell, "terrain_hard_obstacle")
        if not bool(self.snapshot.traversable_mask[row, column]):
            return self._failed(cell, "terrain_not_traversable")
        if float(self.snapshot.slope_deg[row, column]) > 30.0:
            return self._failed(cell, "terrain_slope_exceeded")
        if float(self.snapshot.confidence[row, column]) < self.profile.min_publish_confidence:
            return self._failed(cell, "terrain_confidence_below_profile")
        return self._passed(cell)
```

`validate_body_pose(pose, deadline, *, memory_ledger)` precomputes and reserves the bounded rectangle tuple/scratch bytes through the exact request ledger, passes the same deadline checkpoint and `profile.max_body_pose_cells` into `conservative_body_pose_cells()`, checks before/after every returned helper call and fails on the first `(y,x)` sorted rejected cell. It releases scratch before returning the immutable scalar query result. Runtime cap/memory exhaustion maps to `legged_resource_budget_exceeded`; expiry maps to `planning_deadline_expired`; neither becomes `legged_internal_error`. It never writes into snapshot arrays.

`_audit_legged_provenance_binding()` checks before work and every 64 details/4096 encoded bytes while reading the bounded unique canonical `details` tuple, and requires exact equality for all five approved bindings: `traversability_derivation_id`、`controller_capability_id`、`controller_capability_hash`、`terrain_feature_source_id` and `terrain_feature_source_hash`. The first、third and fifth values must also pass their exact string/SHA-256 contracts, and the expected derivation/source values come from the audited profile fields frozen in Task 1. Missing、duplicate、wrong-typed、over-cap or mismatched required bindings raise `LeggedMapProfileMismatchV1`; unrelated bounded extra provenance details may remain because the complete provenance is already covered by `snapshot_hash`.

- [ ] **Step 6: Implement conservative hierarchy and exact cache key**

Build scales `(1,2,4)` in stable `scale,y,x` order. A coarse cell is passable iff its complete `scale × scale` fine footprint lies in bounds and every child passes the hard view; a partial right/bottom footprint is an explicit `terrain_out_of_bounds` failure, never a smaller safe parent. Its reason tuple is the sorted unique child failures. Cache key is exactly:

```python
LeggedDerivedMapCacheKeyV1(
    schema_id=LEGGED_DERIVED_HIERARCHY_V1,
    snapshot_hash=view.snapshot_hash,
    profile_hash=view.profile_hash,
    controller_capability_hash=view.controller_capability_hash,
)
```

Before cache lookup or allocation, compute with overflow-safe integers `hierarchy_cell_count=sum(ceil(width/scale)*ceil(height/scale) for scale in (1,2,4))` and `hierarchy_bytes=HIERARCHY_BASE_BYTES_V1 + hierarchy_cell_count*HIERARCHY_CELL_RECORD_BYTES_V1`, where `HIERARCHY_BASE_BYTES_V1=4096` and `HIERARCHY_CELL_RECORD_BYTES_V1=32`. The count must not exceed `profile.max_hierarchy_cells`. Reserve the exact bytes from the request memory ledger before constructing a miss or copying a hit；equality with the remaining nonzero ceiling is allowed, one byte over fails before allocation as the typed resource cutoff. The deadline and exact memory-ledger identity are mandatory. Deadline checks occur before/after lookup、before/after reservation and every `HIERARCHY_CELL_CHECKPOINT_BATCH_V1=256` emitted/inspected cells even within one row；a hit after deadline is still timeout. The returned hierarchy stores that exact non-comparable/non-hashable `memory_token` field until provider-scope cleanup；semantic equality/hash/cache encoding exclude it, and every exception releases any provisional token without publishing a partial hierarchy.

`LeggedDerivedMapCacheV1` has the fixed persistent policy `single_exact_key_replacement/v1`: `LEGGED_DERIVED_CACHE_MAX_ENTRIES_V1=1` and `LEGGED_DERIVED_CACHE_MAX_BYTES_V1=168_004_096` (the exact v1 hierarchy cap bytes). It stores only one immutable canonical hierarchy object/byte payload. A matching exact key returns a defensively copied request-accounted value; a different key is built completely under the request ledger first, then an atomic deadline-checked replacement swaps the sole entry. A failed/expired build leaves the old entry intact. There is no accumulating LRU、second entry or request-budget exemption for the defensive copy, and cache contents/addresses never enter semantic hashes.

- [ ] **Step 7: Run GREEN and regression suites**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_legged_rolling_safety.py tests/test_v2_geometry.py tests/test_v2_terrain.py tests/test_v2_hierarchy.py tests/test_v2_wheel_provider.py tests/test_v2_wheel_sqp_contracts.py -q
```

Expected: all pass; existing wheel contacted-cell tuples remain unchanged.

- [ ] **Step 8: Review and commit Task 2**

```powershell
git -C path-planner diff --check
git -C path-planner add src/path_planner/v2/legged_rolling_safety.py src/path_planner/v2/terrain.py src/path_planner/v2/geometry.py tests/test_v2_legged_rolling_safety.py tests/test_v2_geometry.py tests/test_v2_terrain.py
git -C path-planner commit -m "feat: add legged profile-bound terrain safety"
git add path-planner
git commit -m "build: integrate legged terrain safety"
```

---

### Task 3: Implement Isolated Risk Features, Deterministic Fallback, and Lexicographic Keys

**Files:**
- Create: `path-planner/src/path_planner/v2/legged_rolling_risk.py`
- Create: `path-planner/tests/test_v2_legged_rolling_risk.py`

**Interfaces:**
- Consumes: immutable safety view、corridor cells or canonical candidate、profile、deadline、the exact request-scoped `LeggedRiskInferenceLedgerV1` and `LeggedPlanningMemoryLedgerV1` objects、an optional correctly owned pre-reserved candidate-phase memory token、caller-computed mandatory feature/downstream reserves and optional prestarted `LeggedRiskInferenceRunnerV1`; raw `LeggedRiskModelV1` is worker-only。
- Produces: `LeggedRiskFeaturesV1`、`LeggedRiskModelOutputV1`、internal `LeggedRiskEvaluationV1(evidence,memory_token)` for planner candidate/repair calls、request-scoped `LeggedRiskInferenceLedgerV1`、worker-only `LeggedRiskModelV1` Protocol、bounded `LeggedRiskInferenceRunnerV1`、`risk_fallback_aggregation_reserve_v1()`、`evaluate_legged_corridor_risk_v1()`、`evaluate_legged_candidate_risk_v1()`、`aggregate_segment_risk_v1()`、`legged_selection_key_v1()`。

- [ ] **Step 1: Write RED tests for union aggregation and fallback behavior**

```python
def test_union_proxy_is_exactly_the_approved_formula() -> None:
    outputs = (
        make_segment_risk(total=0.1, terrain=0.05, clearance=0.08),
        make_segment_risk(total=0.2, terrain=0.15, clearance=0.12),
    )
    evidence = aggregate_segment_risk_v1(outputs, candidate_hash="a" * 64, context=CONTEXT)
    assert evidence.max_segment_risk_upper_bound == 0.2
    assert evidence.cumulative_route_risk == pytest.approx(1.0 - (0.9 * 0.8))
    assert evidence.terrain_risk == 0.15
    assert evidence.clearance_risk == 0.12
    assert evidence.risk_aggregation_id == "independent_segment_union_proxy/v1"


@pytest.mark.parametrize("output", [None, float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_or_missing_model_uses_nonzero_deterministic_fallback(output: object) -> None:
    runner = prestarted_isolated_runner(StubRiskModel(output)) if output is not None else None
    evaluation = evaluate_legged_candidate_risk_v1(CANDIDATE, VIEW, PROFILE, runner=runner, ledger=LEDGER, memory_ledger=MEMORY_LEDGER, memory_token=RISK_PHASE_TOKEN, feature_reserve_s=FEATURE_RESERVE_S, downstream_reserve_s=RESERVE_S, deadline=DEADLINE)
    evidence = evaluation.evidence
    assert evidence.risk_source == "deterministic_fallback"
    assert evidence.cumulative_route_risk > 0.0
    assert evidence.fallback_reason is not None
```

- [ ] **Step 2: Write RED isolation and ordering tests**

```python
def test_model_never_receives_mutable_map_or_hard_view() -> None:
    transport = CapturingCanonicalFeatureTransport(valid_output())
    runner = make_exact_runner(transport=transport)
    evaluate_legged_candidate_risk_v1(CANDIDATE, VIEW, PROFILE, runner=runner, ledger=LEDGER, memory_ledger=MEMORY_LEDGER, memory_token=RISK_PHASE_TOKEN, feature_reserve_s=FEATURE_RESERVE_S, downstream_reserve_s=RESERVE_S, deadline=DEADLINE)
    assert type(transport.sent_batch) is tuple
    assert all(type(value) is LeggedRiskFeaturesV1 for value in transport.sent_batch)
    assert all(not hasattr(value, "terrain_snapshot") for value in transport.sent_batch)
    assert all(not hasattr(value, "hard_obstacle_mask") for value in transport.sent_batch)
    assert all(not hasattr(value, "traversable_mask") for value in transport.sent_batch)


def test_lexicographic_risk_dominates_time_energy_and_smoothness() -> None:
    safer_slower = key(max_risk=0.1, cumulative=0.2, time=10.0, energy=10.0, smoothness=10.0, tie="b")
    riskier_faster = key(max_risk=0.2, cumulative=0.0, time=0.0, energy=0.0, smoothness=0.0, tie="a")
    assert safer_slower < riskier_faster


def test_blocking_model_is_killed_and_falls_back_within_sealed_budget() -> None:
    runner = prestarted_isolated_runner(BlockingRiskModel())
    evidence, audit = evaluate_with_runner(runner, PROFILE)
    assert evidence.risk_source == "deterministic_fallback"
    assert evidence.fallback_reason == "risk_inference_timeout"
    assert audit.charged_inference_s == PROFILE.max_risk_batch_inference_s
    assert audit.worker_terminated is True


def test_feature_extraction_matches_checked_in_corridor_and_candidate_goldens() -> None:
    assert encode_features(extract_corridor_features(CORRIDOR_FIXTURE)) == CORRIDOR_FEATURE_GOLDEN
    assert encode_features(extract_candidate_features(CANDIDATE_FIXTURE)) == CANDIDATE_FEATURE_GOLDEN


def test_risk_distance_window_finds_edge_beyond_eight_neighborhood_and_saturates_when_empty() -> None:
    near = extract_candidate_features(EDGE_1P5M_AWAY)[0]
    open_area = extract_candidate_features(NO_EDGE_WITHIN_2M)[0]
    assert near.terrain_edge_distance_m == 1.5
    assert open_area.clearance_min_m == RISK_DISTANCE_CLIP_M_V1
    assert open_area.terrain_edge_distance_m == RISK_DISTANCE_CLIP_M_V1


def test_zero_speed_curvature_and_first_control_variation_are_frozen() -> None:
    features = extract_candidate_features(ZERO_SPEED_ROTATION)
    assert features[0].max_curvature_proxy == abs(OMEGA) / 1.0e-6
    assert features[0].control_variation == 0.0


def test_one_request_ledger_accumulates_and_next_request_starts_fresh() -> None:
    first = LeggedRiskInferenceLedgerV1.create(REQUEST_HASH, PROFILE)
    consume_two_batches(first)
    assert third_batch_uses_fallback_without_worker(first)
    second = LeggedRiskInferenceLedgerV1.create(OTHER_REQUEST_HASH, PROFILE)
    assert second.charged_inference_s == 0.0


def test_downstream_reserve_equality_skips_model_without_charging() -> None:
    evidence = evaluate_with_remaining_time(RESERVE_S + BATCH_SLOT_S)
    assert evidence.fallback_reason == "risk_inference_reserve_insufficient"
    assert LEDGER.charged_inference_s == 0.0


def test_feature_plus_downstream_reserve_equality_stops_before_extraction() -> None:
    with pytest.raises(LeggedSQPResourceCutoffV1):
        evaluate_with_remaining_time(FEATURE_RESERVE_S + RESERVE_S)
    assert FEATURE_COUNTER.started is False


def test_fast_and_slow_equal_outputs_charge_the_same_slots_and_allow_the_same_calls() -> None:
    fast = run_request_risk_batches(model_latency_s=0.0, initial_remaining_s=NEXT_SLOT_BOUNDARY)
    slow = run_request_risk_batches(model_latency_s=nextafter(BATCH_SLOT_S, -inf), initial_remaining_s=NEXT_SLOT_BOUNDARY)
    assert (fast.charged_inference_s, fast.sent_batch_count) == (slow.charged_inference_s, slow.sent_batch_count)
    assert fast.slot_end_times == slow.slot_end_times


def test_full_worker_slot_still_leaves_complete_fallback_and_l2_reserve() -> None:
    evidence, audit = evaluate_with_blocking_worker_consuming_full_slot()
    assert evidence.fallback_reason == "risk_inference_timeout"
    assert audit.fallback_completed is True
    assert audit.remaining_s > audit.downstream_reserve_s


def test_feature_work_cap_uses_complete_max_risk_fallback_without_partial_batch() -> None:
    evidence = evaluate_case_exceeding_feature_cell_cap_by_one()
    assert evidence.risk_source == "deterministic_fallback"
    assert evidence.fallback_reason == "risk_feature_budget_exceeded"
    assert all(value == 1.0 for value in evidence.segment_risk_upper_bounds)


def test_corridor_risk_segment_cap_is_exact_and_cap_plus_one_stops_before_features() -> None:
    assert evaluate_corridor_with_segments(PROFILE.max_corridor_risk_segments).segment_count == PROFILE.max_corridor_risk_segments
    with pytest.raises(LeggedSQPResourceCutoffV1):
        evaluate_corridor_with_segments(PROFILE.max_corridor_risk_segments + 1)
    assert FEATURE_COUNTER.started is False


def test_fallback_reserve_scales_with_segment_count_and_memory_boundary() -> None:
    reserve = risk_fallback_aggregation_reserve_v1(PROFILE.max_corridor_risk_segments)
    assert reserve.bytes == RISK_FALLBACK_AGGREGATION_BASE_BYTES_V1 + PROFILE.max_corridor_risk_segments * RISK_FALLBACK_AGGREGATION_PER_SEGMENT_BYTES_V1
    assert fallback_with_memory_bytes(reserve.bytes).completed
    with pytest.raises(LeggedSQPResourceCutoffV1):
        fallback_with_memory_bytes(reserve.bytes - 1)


@pytest.mark.parametrize("source", ["corridor", "candidate"])
def test_risk_feature_allocations_use_the_shared_memory_ledger(source: str) -> None:
    result = evaluate_with_exact_shared_memory_boundary(source)
    assert result.audit.memory_ledger_identity is REQUEST_MEMORY_LEDGER
    with pytest.raises(LeggedSQPResourceCutoffV1):
        evaluate_one_byte_over_shared_memory_boundary(source)
    assert FEATURE_COUNTER.partial_publication_count == 0


def test_under_cap_but_slow_feature_extraction_switches_to_complete_fallback_before_l2_reserve() -> None:
    evidence, audit = evaluate_with_feature_clock_crossing_phase_cutoff()
    assert evidence.fallback_reason == "risk_feature_phase_overrun"
    assert all(value == 1.0 for value in evidence.segment_risk_upper_bounds)
    assert audit.remaining_s > audit.downstream_reserve_s
```

- [ ] **Step 3: Run RED tests**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_legged_rolling_risk.py -q -x
```

Expected: import failure for the new risk surface.

- [ ] **Step 4: Implement immutable feature extraction and strict model Protocol**

```python
@dataclass(frozen=True, slots=True)
class LeggedRiskFeaturesV1:
    segment_index: int
    slope_margin_min_deg: float
    confidence_min: float
    confidence_margin_min: float
    confidence_variation: float
    clearance_min_m: float
    elevation_range_m: float
    slope_variation_deg: float
    terrain_edge_distance_m: float
    max_speed_mps: float
    max_abs_accel_mps2: float
    max_abs_yaw_rate_radps: float
    max_curvature_proxy: float
    control_variation: float
    missing_feature_count: int
    feature_schema_id: str = LEGGED_RISK_FEATURE_SCHEMA_V1


@dataclass(frozen=True, slots=True)
class LeggedRiskModelOutputV1:
    segment_risk_upper_bound: float
    terrain_risk: float
    clearance_risk: float
    confidence_risk: float
    kinematic_aggressiveness_risk: float
    model_uncertainty: float
    ood_detected: bool


@dataclass(frozen=True, slots=True)
class RiskContextV1:
    risk_source: str
    model_id: str
    feature_schema_id: str
    calibration_id: str
    snapshot_hash: str
    fallback_reason: str | None


@dataclass(slots=True)
class LeggedRiskInferenceLedgerV1:
    request_hash: str
    profile_hash: str
    max_total_s: float
    charged_inference_s: float
    sent_batch_count: int
    ledger_id: str = LEGGED_RISK_INFERENCE_LEDGER_V1


class LeggedRiskModelV1(Protocol):
    model_id: str
    feature_schema_id: str
    calibration_id: str
    deterministic: bool

    def predict_upper_bound(self, features: LeggedRiskFeaturesV1) -> LeggedRiskModelOutputV1:
        raise NotImplementedError
```

The inference ledger is intentionally mutable: `create(request_hash, profile)` is the sole constructor, fixes `profile_hash/max_total_s`, starts both counters at zero, and is never exported in a public outcome. `remaining_s` is decimal12 `max(0,max_total-charged)`; `_reserve_slot_v1(slot_s, slot_end_monotonic_s)` verifies planner-thread ownership and exact request/profile identity, then increments charge/count once before transport send and records the frozen slot endpoint. Copy/deepcopy/pickle are rejected so a helper cannot fork the request budget. It is distinct from, but always travels with, the single Task 1 request-memory ledger.

Freeze feature extraction before any model/fallback call. A corridor risk segment is one consecutive `cells_0p5m` edge and `segment_count=len(cells_0p5m)-1` must be in `1..profile.max_corridor_risk_segments`; a longer guide stops before feature allocation as a typed resource failure. A candidate risk segment is one canonical SQP segment and is already bounded by `profile.max_segments`. Corridor pose samples are start/mid/end cell centers with the edge tangent; candidate samples come from `sample_legged_body_segment_v1()` including both endpoints. At every pose use the margin-expanded body center plus four corners as the fixed risk support set. Enumerate the sorted unique conservative rectangle cells and their in-bounds eight-neighborhood for slope/confidence/variation features. Separately enumerate, in row-major order, every in-bounds cell square intersecting the closed axis-aligned `RISK_DISTANCE_CLIP_M_V1=2.0` box around any of the five support points; deduplicate the union and charge every visited cell to the same feature-cell cap/memory ledger. No model-facing feature may depend on model output or mutable telemetry.

Terrain formulas are exact: `slope_margin_min_deg=min(30.0-slope_deg)`、`confidence_min=min(confidence)`、`confidence_margin_min=min(confidence-profile.min_publish_confidence)` and `confidence_variation=max(confidence)-min(confidence)` over observed footprint/neighborhood cells; `elevation_range_m=max(elevation)-min(elevation)` and `slope_variation_deg=max(slope)-min(slope)` over the same sorted expanded set. `clearance_min_m` is the minimum exact point-to-unsafe-cell-square or map-boundary distance over the five support points and the frozen distance-window cells, clipped above at exactly `RISK_DISTANCE_CLIP_M_V1`; unsafe is the Task 2 hard-view complement. `terrain_edge_distance_m` is the minimum point-to-edge distance within that same window to the outer map boundary or to a shared cell edge whose two four-neighbor hard-view pass/fail bits differ, with edges enumerated by `(min_y,min_x,orientation)`, likewise clipped above at 2.0 m. An empty local unsafe/edge set with a farther map boundary deterministically yields 2.0 m; no feature scans the full map or claims an exact distance beyond the clip. If any required footprint cell is out of bounds/unobserved, use conservative zero slope/confidence margins、clearance and terrain-edge distance, zero variation placeholders, and set `missing_feature_count=4`; otherwise it is `0`. Partial availability never silently drops a cell. Checked-in goldens cover the nearest edge outside the eight-neighborhood、exactly at the clip and no edge inside the clip.

For a candidate segment: `max_speed_mps=max(abs(v_start),abs(v_end))`、`max_abs_accel_mps2=abs(a)`、`max_abs_yaw_rate_radps=abs(omega)`、`max_curvature_proxy=abs(omega)/max(max_speed_mps,1.0e-6)`; segment 0 has `control_variation=0.0`, later segments use `abs(a_i-a_prev)/max(max_accel,max_decel)+abs(omega_i-omega_prev)/max_yaw_rate`. For a corridor edge, speed is `min(0.3,max_forward_speed)`、acceleration is zero、duration is physical edge length/speed; segment 0 yaw-rate/control-variation are zero, later yaw-rate is wrapped tangent change divided by the mean adjacent duration and control variation uses the same normalized yaw-rate delta. Curvature uses the same `1.0e-6` denominator floor. Every output scalar is decimal12 canonicalized; segment order and feature-schema bytes are covered by checked-in golden fixtures and exact/`nextafter` boundary tests.

`risk_fallback_aggregation_reserve_v1(segment_count)` validates the bounded exact count, then returns outward-rounded `seconds=BASE_S + segment_count*PER_SEGMENT_S` and overflow-safe `bytes=BASE_BYTES + segment_count*PER_SEGMENT_BYTES` using the four Task 1 constants. Feature extraction requires exact finite nonnegative `feature_reserve_s/downstream_reserve_s`, with `feature_reserve_s>=fallback_reserve.seconds`; before starting it gives deadline precedence, then requires strict `deadline.remaining_s > feature_reserve_s + downstream_reserve_s`. Equality raises the typed resource cutoff before any feature record. Seal `feature_cutoff=deadline.deadline_monotonic_s-downstream_reserve_s-fallback_reserve.seconds`. It checks the shared deadline and that cutoff before/after each segment and every 256 pose/cell records. Precompute overflow-safe upper counts and bytes before allocation using `RISK_FEATURE_BASE_BYTES_V1=4096`、`RISK_FEATURE_POSE_RECORD_BYTES_V1=96`、`RISK_FEATURE_CELL_RECORD_BYTES_V1=16` and `RISK_FEATURE_OUTPUT_RECORD_BYTES_V1=128`; cumulative pose samples and visited feature cells must stay within `profile.max_risk_feature_pose_samples/max_risk_feature_cells`.

Before any feature/fallback allocation, determine enough bytes for the retained input identity、feature scratch and complete output. Corridor evaluation reserves that exact scope directly from the shared ledger and may shrink/release it after corridor-guide construction. Post-SQP candidate/repair evaluation must instead validate the exact pre-reserved `risk_phase` token and may never charge again or shrink its sealed bytes: after normal/fallback aggregation it transfers the entire handle to owner `risk_evidence` and returns `LeggedRiskEvaluationV1(evidence,new_token)`, making the input handle stale. Keeping the sealed capacity live makes later composite admission byte-exact even if fallback used less scratch. If normal feature capacity cannot be obtained but the complete fallback fits the same scope, emit fallback without starting extraction; if neither fits, raise typed resource cutoff. A count-cap、memory-cap or feature-cutoff hit discards all partial features and does not send a partial batch; within the count-scaled reserve it emits a complete per-segment all-max deterministic fallback (`missing_feature_count=4`, risk components/upper bound `1.0`) with reason `risk_feature_budget_exceeded` or `risk_feature_phase_overrun`. Every failed path releases provisional tokens. Global deadline expiry remains timeout rather than fallback. Immediately after normal/fallback aggregation, recheck deadline then strict `remaining_s > downstream_reserve_s` before any worker admission or return to L2.

`predict_upper_bound()` 的 `NotImplementedError` 是 Protocol 的抽象方法体，不是运行时 fallback 或延期实现；实际模型必须覆盖该方法。The evaluator accepts external model output only when IDs match, `deterministic is True`, all scalars are exact finite floats in `[0,1]`, `ood_detected is False`, and deadline remains open; otherwise it records one stable fallback reason and creates a separately marked internal fallback output. The aggregate auditor distinguishes those two sources, so an OOD-triggered fallback may preserve `ood_detected=True` without accepting the rejected external prediction.

The planner thread never invokes that Protocol directly. A provider may own one prestarted isolated worker whose code/model/calibration hashes are sealed into the provider token. At `provider.plan()` entry create exactly one `LeggedRiskInferenceLedgerV1` bound to the request/profile hashes with `charged_inference_s=0.0`; it is planner-thread-confined, passed with the exact memory ledger through corridor/candidate/repair evaluation, and never stored on the provider/runner. Before any send compute `slot_s=min(profile.max_risk_batch_inference_s, ledger.remaining_s)` in decimal12 and the exact count-scaled fallback reserve. The caller supplies a finite nonnegative frozen downstream reserve: corridor calls use the worst complete post-solver tail; candidate/repair calls use the admitted actual L2 tail. Admission requires `slot_s>0` and strict `deadline.remaining_s > downstream_reserve_s + slot_s + fallback_reserve.seconds`; equality skips without charging. This post-worker tail is reserved even for an expected-valid model because output audit may force fallback.

If admitted, freeze `slot_start=clock()` and `slot_end=slot_start+slot_s`, atomically reserve/charge that entire slot before send, and require `slot_end < deadline.deadline_monotonic_s-downstream_reserve_s-fallback_reserve.seconds`. The worker gets at most that slot. An early valid result is held in the planner-thread transport envelope until the same frozen `slot_end`; the planner checks the shared deadline while waiting and never starts another risk call or downstream phase early. A timeout、partial output or exit is cut at the same endpoint, and no future is joined afterward. Thus fast and slow responses inside one admitted slot consume the same wall phase as well as the same logical charge; scheduler latency cannot change the next call's eligibility. Missing runner or any pre-send failure consumes zero and uses immediate fallback. Output audit and complete deterministic fallback run after the slot under a cutoff that preserves `downstream_reserve_s`; if that aggregation tail or its shared-memory reservation overruns, raise typed resource cutoff. After transport/fallback, give global deadline precedence and again require strict `remaining_s > downstream_reserve_s` before starting L2. A new request starts fresh, and an untrusted model cannot block the shared deadline or directly create a new hard/L2 failure category.

- [ ] **Step 5: Implement deterministic fallback and stable keys**

Use only already extracted finite features:

```python
def deterministic_segment_risk_v1(features: LeggedRiskFeaturesV1, profile: LeggedRollingBodySQPProfileV1) -> float:
    slope = 1.0 - _clamp01(features.slope_margin_min_deg / 30.0)
    confidence = 1.0 - _clamp01(features.confidence_margin_min)
    clearance = 1.0 / (1.0 + max(0.0, features.clearance_min_m))
    aggression = max(
        _clamp01(features.max_speed_mps / max(profile.max_forward_speed_mps, profile.max_reverse_speed_mps)),
        _clamp01(features.max_abs_accel_mps2 / max(profile.max_linear_accel_mps2, profile.max_linear_decel_mps2)),
        _clamp01(features.max_abs_yaw_rate_radps / profile.max_yaw_rate_radps),
    )
    missing = _clamp01(features.missing_feature_count / 4.0)
    return max(1.0e-12, min(1.0, max(slope, confidence, clearance, aggression, missing)))


def aggregate_segment_risk_v1(outputs: tuple[LeggedRiskModelOutputV1, ...], *, candidate_hash: str, context: RiskContextV1) -> LeggedRiskEvidenceV1:
    if type(outputs) is not tuple or not outputs:
        raise ValueError("segment risk outputs must be nonempty exact tuple")
    audited = tuple(_audit_segment_risk_output(value) for value in outputs)
    values = tuple(value.segment_risk_upper_bound for value in audited)
    cumulative = 1.0 - prod(1.0 - value for value in values)
    return _build_evidence(
        segment_risk_upper_bounds=values,
        max_segment_risk_upper_bound=max(values),
        cumulative_route_risk=min(1.0, max(0.0, cumulative)),
        terrain_risk=max(value.terrain_risk for value in audited),
        clearance_risk=max(value.clearance_risk for value in audited),
        confidence_risk=max(value.confidence_risk for value in audited),
        kinematic_aggressiveness_risk=max(value.kinematic_aggressiveness_risk for value in audited),
        model_uncertainty=max(value.model_uncertainty for value in audited),
        ood_detected=any(value.ood_detected for value in audited),
        candidate_hash=candidate_hash,
        context=context,
    )
```

The deterministic fallback records `risk_source=LEGGED_DETERMINISTIC_RISK_SOURCE_V1` and `model_id=LEGGED_DETERMINISTIC_RISK_MODEL_V1`, then creates a complete `LeggedRiskModelOutputV1` per segment, not only one scalar: its component fields are the corresponding normalized slope/clearance/confidence/aggressiveness proxies, `model_uncertainty=1.0`, and `ood_detected=True` only when OOD caused rejection. Thus every frozen evidence field has one auditable source. `legged_selection_key_v1()` returns the exact tuple from the spec; it never multiplies risk by `ObjectiveProfileV2.risk_weight`.

- [ ] **Step 6: Run GREEN and mutation tests**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_legged_rolling_risk.py tests/test_v2_legged_rolling_safety.py -q
```

Expected: all pass, including model enabled/disabled/OOD/nonfinite permutations with identical hard-view digest.

- [ ] **Step 7: Review and commit Task 3**

```powershell
git -C path-planner diff --check
git -C path-planner add src/path_planner/v2/legged_rolling_risk.py tests/test_v2_legged_rolling_risk.py
git -C path-planner commit -m "feat: add isolated legged route risk evaluation"
git add path-planner
git commit -m "build: integrate legged route risk evaluation"
```

---

### Task 4: Generate and Risk-Order up to Three Topologically Distinct Global Corridors

**Files:**
- Create: `path-planner/src/path_planner/v2/legged_rolling_corridors.py`
- Create: `path-planner/tests/test_v2_legged_rolling_corridors.py`

**Interfaces:**
- Consumes: `LeggedHardSafetyViewV1`、`LeggedConservativeHierarchyV1`、start/mission-goal poses、profile、risk evaluator/runner、the exact request-scoped risk and memory ledgers、caller-frozen worst-case feature/downstream reserves、`ResourceBudgetV2` and one `PlanningDeadlineV2`。
- Produces: internal `GeneratedLeggedCorridorsV1(intents,retained_memory_tokens)` from `generate_legged_corridors_v1(view, hierarchy, start_state, mission_goal_state, profile, risk_runner, risk_ledger, memory_ledger, feature_reserve_s, downstream_reserve_s, resource_budget, deadline)`；`intents` is the public-semantic `tuple[GlobalCorridorIntentV1,...]`, while the sorted nonserializable token tuple owns retained paths/guides until provider scope cleanup。

- [ ] **Step 1: Write RED topology, quota, and deterministic-order tests**

```python
def test_generator_returns_at_most_three_unique_topologies() -> None:
    corridors = generate_case("three_passages")
    assert 1 <= len(corridors) <= 3
    assert len({value.topology_signature for value in corridors}) == len(corridors)
    assert all(value.executable is False for value in corridors)
    assert all(value.validation_level is ValidationLevelV2.L1 for value in corridors)


def test_near_duplicate_same_signature_does_not_consume_quota() -> None:
    corridors = generate_case("duplicate_upper_passage")
    assert tuple(value.topology_signature for value in corridors) == EXPECTED_DISTINCT_SIGNATURES


def test_zero_overlap_duplicate_signature_is_penalized_then_other_topology_is_found() -> None:
    corridors, audit = generate_case_with_audit("two_disjoint_lanes_same_signature_before_lower_passage")
    assert len({value.topology_signature for value in corridors}) == len(corridors)
    assert LOWER_PASSAGE_SIGNATURE in {value.topology_signature for value in corridors}
    assert audit.attempted_path_penalty_count >= 2


def test_equal_cost_ties_are_byte_deterministic() -> None:
    runs = [encode_corridors(generate_case("symmetric")) for _ in range(5)]
    assert len(set(runs)) == 1


@pytest.mark.parametrize("case", ["ray_vertex_touch", "ray_collinear_overlap", "ray_endpoint_on_level"])
def test_topology_ray_degeneracies_use_the_frozen_symbolic_rule(case: str) -> None:
    assert topology_signature_for(case) == EXPECTED_SIGNATURES[case]


def test_topology_signature_degeneracy_rule_is_cross_process_stable() -> None:
    assert run_signature_fixture_in_fresh_processes("ray_degeneracies", count=5) == EXPECTED_SIGNATURE_BYTES
```

- [ ] **Step 2: Write RED fail-closed refinement and first-order tests**

```python
def test_coarse_path_that_cannot_refine_to_fine_is_rejected() -> None:
    assert generate_case("coarse_open_fine_blocked") == ()


def test_risk_guide_precedes_time_energy_and_length() -> None:
    corridors = generate_case("short_risky_long_safe")
    assert corridors[0].risk_guide.max_upper_bound < corridors[1].risk_guide.max_upper_bound


def test_budget_and_deadline_have_stable_distinct_failures() -> None:
    with pytest.raises(LeggedCorridorBudgetExceededV1):
        generate_with_budget(max_expanded_states=1)
    with pytest.raises(LeggedPlanningDeadlineExpiredV1):
        generate_with_expired_deadline()


def test_one_corridor_ledger_is_shared_across_attempts_and_all_scales() -> None:
    result = generate_with_audited_budget_boundary()
    assert result.audit.candidate_attempt_count == 2
    assert result.audit.expanded_state_count == EXPECTED_SCALE4_PLUS_SCALE2_PLUS_SCALE1_TOTAL
    assert result.audit.ledger_reset_count == 0


def test_fine_corridor_risk_segment_cap_boundary_is_fail_closed() -> None:
    assert len(generate_with_exact_fine_edge_count(PROFILE.max_corridor_risk_segments)) == 1
    with pytest.raises(LeggedCorridorBudgetExceededV1):
        generate_with_exact_fine_edge_count(PROFILE.max_corridor_risk_segments + 1)
    assert RISK_FEATURE_COUNTER.started is False


def test_topology_crossing_test_cap_and_mid_loop_deadline_are_bounded() -> None:
    assert generate_topology_at_crossing_cap(PROFILE.max_topology_crossing_tests).audit.topology_crossing_test_count == PROFILE.max_topology_crossing_tests
    with pytest.raises(LeggedCorridorBudgetExceededV1):
        generate_topology_at_crossing_cap(PROFILE.max_topology_crossing_tests + 1)
    with pytest.raises(LeggedPlanningDeadlineExpiredV1):
        generate_topology_with_clock_crossing_mid_256_test_batch()
    assert PARTIAL_INTENT_SINK == []


@pytest.mark.parametrize(
    "phase",
    ["scale4_queue", "stale_heap_drain", "scale4_neighbors", "scale2_refine", "scale1_refine", "reconstruct", "component_rows", "component_bfs"],
)
def test_every_large_corridor_loop_observes_deadline_inside_256_work_batch(phase: str) -> None:
    with pytest.raises(LeggedPlanningDeadlineExpiredV1):
        generate_large_case_with_clock_expiring_mid_phase(phase)
    assert PARTIAL_INTENT_SINK == []
```

- [ ] **Step 3: Run RED tests**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_legged_rolling_corridors.py -q -x
```

Expected: import failure for the new corridor module.

- [ ] **Step 4: Implement stable A* and fail-closed refinement**

Implement eight-neighbor A* with fixed neighbor order:

```python
NEIGHBORS_V1 = (
    (-1, 0, 1.0), (0, -1, 1.0), (1, 0, 1.0), (0, 1, 1.0),
    (-1, -1, SQRT_2), (1, -1, SQRT_2), (-1, 1, SQRT_2), (1, 1, SQRT_2),
)
```

Queue key is `(f,g,y,x,insertion_index)`; diagonal moves forbid corner cutting. Generate at scale 4, then refine within a deterministic one-parent-cell band at scale 2 and scale 1. Every refined fine cell must pass the hard view. Start/goal are never snapped outside their original fine cells.

Freeze the base search math: edge cost is physical step length (`scale*0.5` or `sqrt(2)*scale*0.5` metres); heuristic is the matching admissible octile distance; no learned risk enters `g` or `h`. For attempt `j>0`, entering an interior scale-4 cell used by any prior attempted coarse path—not only accepted signatures—adds `exclusion_penalty_m = (j + 1) * (map_diagonal_m + first_attempt_length_m + 1.0)` per recorded occurrence; endpoints are exempt. Record the first/each coarse attempt before signature/refinement acceptance, so a duplicate signature with zero overlap to the accepted path is itself penalized on the next attempt instead of repeating to the cap. Refinement minimizes the same physical length inside the fixed parent band. A guide uses `guide_length_m` from the fine polyline, `estimated_time_s=guide_length_m/profile.max_forward_speed_mps`, `estimated_energy=guide_length_m`, and the Task 3 corridor-feature risk aggregate before the final lexicographic sort. All arithmetic is finite binary64 followed by the shared decimal12 canonicalizer before hashing.

Create one `LeggedCorridorResourceLedgerV1` before the first A* call and pass it by identity through every scale-4 rerun and every scale-2/scale-1 refinement. It holds the exact Task 1 memory-ledger object and delegates every live reservation/release to it; it does not keep an independent ceiling or resettable peak. `candidate_attempt_count` increments before each coarse A* and is capped by `profile.max_corridor_candidates`; `expanded_state_count` increments on each non-stale queue pop across all scales/attempts and is capped by `request.resource_budget.max_expanded_states`; `emitted_route_state_count` counts every reconstructed/refined cell cumulatively and is capped by `max_route_states`; `topology_crossing_test_count` increments before each exact component×path-segment reference-ray test across all attempts and is capped by `profile.max_topology_crossing_tests`. Exact-cap work is allowed; the next increment fails before mutation. Deterministic memory accounting uses v1 constants `QUEUE_RECORD_BYTES=96`、`SEARCH_RECORD_BYTES=64`、`COMPONENT_RECORD_BYTES=24`、`PATH_CELL_BYTES=16`; search scratch is released when an attempt ends, while accepted hierarchy/path/intent records stay live in the shared ledger until provider cleanup. The same deadline callback runs before each phase and after every bounded batch: 256 non-stale queue pops、256 neighbor relaxations、256 emitted/reconstructed/refined route cells、256 component-grid row cells、256 component-BFS queue pops and 256 topology crossing tests. It also runs after every helper return and before publishing a complete path/signature, so no coarse search、both refinement levels、reconstruction or component flood fill can cross the deadline merely because another counter has not advanced. No counter resets after a duplicate signature、failed refinement or accepted corridor; budget/deadline exhaustion before a complete signature discards partial intents、releases provisional tokens and raises the frozen typed failure.

Immediately after a complete fine path is reconstructed and before geometry hashing or risk extraction, require `1 <= len(cells_0p5m)-1 <= profile.max_corridor_risk_segments`. The exact cap passes; cap+1 raises `LeggedCorridorBudgetExceededV1` before a feature/fallback record is allocated. Compute the corridor feature/fallback reserve from that actual bounded edge count, while still preserving the caller's worst complete post-solver downstream reserve. This makes the Task 3 per-segment fallback finite and prevents `ResourceBudgetV2.max_route_states` from silently expanding model-facing work beyond the specialized cap.

A* checkpoint clarification is normative: maintain a separate `raw_heap_pop_count` that increments on every `heappop` before stale detection and invokes the shared deadline callback every 256 raw pops. `expanded_state_count` still increments only for non-stale pops and remains the semantic resource counter；raw-pop batching is a deadline mechanism, not a second expansion budget. The stale-drain regression fills the heap with over 256 obsolete entries, expires the scripted clock mid-drain and requires timeout before the next live expansion or neighbor relaxation.

- [ ] **Step 5: Implement blocked components and reference-ray signature**

Label blocked scale-4 components in row-major order using four-connectivity, with the same 256-work deadline batches for row scanning and BFS queue pops. For each component choose its `(min_y,min_x)` cell center and a positive-X reference ray. Convert the path polyline to exact doubled-integer scale-4 cell-center coordinates; never use floating intersection arithmetic. The frozen degeneracy rule is the positive infinitesimal ray `y=origin_y+ε`: horizontal/collinear segments contribute zero; an upward segment is eligible exactly when `y0<=origin_y<y1`, a downward segment exactly when `y1<=origin_y<y0`. For either direction reorder the endpoints as lower `(xl,yl)` and upper `(xu,yu)`, set `D=yu-yl>0`, `A0=D*(xl-origin_x)+(xu-xl)*(origin_y-yl)` and `Aeps=xu-xl`; the symbolic intersection is on the positive ray iff `(A0,Aeps)` is lexicographically greater than `(0,0)`. Upward eligible crossings contribute `+1`, downward `-1`; exact pair `(0,0)` indicates a path through the blocked origin line and rejects that attempted path as corrupt/corner-cut geometry rather than guessing. This defines vertex touch、endpoint-on-level and collinear overlap cases without platform `float` behavior. Serialize only nonzero `(component_id,crossing_number)` pairs; hash the canonical tuple. Every attempted coarse path receives the deterministic path-cell exclusion penalty described above, and A* reruns until `profile.max_corridors` unique signatures、`profile.max_corridor_candidates` total attempts、shared-ledger exhaustion or a proven no-path result.

```python
def corridor_order_key_v1(intent: GlobalCorridorIntentV1) -> tuple[object, ...]:
    guide = intent.risk_guide
    return (
        guide.max_upper_bound,
        guide.cumulative,
        guide.estimated_time_s,
        guide.estimated_energy,
        guide.guide_length_m,
        intent.topology_signature,
        intent.corridor_hash,
    )
```

- [ ] **Step 6: Bind intent identity and sort exactly once**

`corridor_geometry_hash` covers all three scale paths、signature、source ID and snapshot/profile/controller hashes, excluding risk、order index、memory tokens and mutable telemetry. Build it immediately after fine refinement and use it as the corridor-risk feature batch identity. After complete risk-guide evidence exists, build `corridor_hash` over geometry hash plus that evidence while excluding `corridor_index`. Sort by the approved key, then assign `corridor_index=0..N-1` as the resulting attempt order; the index is observational and cannot create a hash/sort cycle. Return one `GeneratedLeggedCorridorsV1` whose token order matches intents exactly; provider registers those handles in its request scope, processes `bundle.intents` in index order without reranking, and can therefore release every retained allocation on continue/return/exception without putting a token in the public intent.

- [ ] **Step 7: Run GREEN, property, and resource tests**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_legged_rolling_corridors.py tests/test_v2_legged_rolling_safety.py tests/test_v2_hierarchy.py -q
```

Expected: all pass for 1/2/3/no-corridor fixtures, rotated/symmetric maps, signature duplicates, expansion caps and deadline boundaries.

- [ ] **Step 8: Review and commit Task 4**

```powershell
git -C path-planner diff --check
git -C path-planner add src/path_planner/v2/legged_rolling_corridors.py tests/test_v2_legged_rolling_corridors.py
git -C path-planner commit -m "feat: add deterministic legged global corridors"
git add path-planner
git commit -m "build: integrate legged global corridors"
```

---

### Task 5: Select Rolling Subgoals and Freeze the `(x,y,yaw,v)/(a,omega,dt)` Model and Initializer

**Files:**
- Create: `path-planner/src/path_planner/v2/legged_body_kinematics.py`
- Create: `path-planner/src/path_planner/v2/legged_rolling_subgoals.py`
- Create: `path-planner/src/path_planner/v2/legged_body_sqp_initialization.py`
- Create: `path-planner/tests/test_v2_legged_body_kinematics.py`
- Create: `path-planner/tests/test_v2_legged_rolling_subgoals.py`
- Create: `path-planner/tests/test_v2_legged_body_sqp_initialization.py`

**Interfaces:**
- Consumes: exact body state、profile、one `GlobalCorridorIntentV1`、mission goal、hard view、deadline and the exact request memory ledger for planning-time materialization。
- Produces: `integrate_legged_body_segment_v1()`、`legged_body_segment_jacobian_v1()`、`sample_legged_body_segment_v1()`、`legged_body_relative_energy_v1()`、`select_legged_rolling_subgoal_v1(..., *, memory_ledger) -> LeggedRollingSubgoalV1`、`initialize_legged_body_sqp_v1(..., *, memory_ledger)`；the two retained planning objects carry single-use internal memory tokens excluded from semantic hashes/codecs。

- [ ] **Step 1: Write RED exact-dynamics and near-zero tests**

```python
def test_zero_yaw_rate_matches_constant_acceleration_line() -> None:
    end = integrate_legged_body_segment_v1(
        LeggedBodyStateV1(1.0, 2.0, 0.0, 0.4),
        a_mps2=0.2,
        omega_radps=0.0,
        duration_s=2.0,
    )
    assert end == LeggedBodyStateV1(2.2, 2.0, 0.0, 0.8)


def test_nonzero_yaw_rate_matches_high_precision_reference() -> None:
    end = integrate_legged_body_segment_v1(START, a_mps2=0.3, omega_radps=0.4, duration_s=1.5)
    assert end.x_m == pytest.approx(REFERENCE_X, abs=1.0e-12)
    assert end.y_m == pytest.approx(REFERENCE_Y, abs=1.0e-12)
    assert end.yaw_rad == pytest.approx(START.yaw_rad + 0.6, abs=1.0e-15)
    assert end.v_mps == pytest.approx(START.v_mps + 0.45, abs=1.0e-15)


def test_near_zero_omega_is_continuous_and_finite() -> None:
    left = integrate_legged_body_segment_v1(START, 0.1, -1.0e-10, 1.0)
    zero = integrate_legged_body_segment_v1(START, 0.1, 0.0, 1.0)
    right = integrate_legged_body_segment_v1(START, 0.1, 1.0e-10, 1.0)
    assert max(abs(left.x_m - zero.x_m), abs(right.x_m - zero.x_m)) < 1.0e-9
    assert all(isfinite(value) for value in astuple(left) + astuple(right))


@pytest.mark.parametrize("h", [nextafter(1.0e-2, -inf), 1.0e-2, nextafter(1.0e-2, inf)])
def test_kernel_branch_boundary_matches_high_precision_value_and_derivatives(h: float) -> None:
    assert _a_kernel(h) == pytest.approx(HIGH_PRECISION_A(h), abs=1.0e-12)
    assert _b_kernel(h) == pytest.approx(HIGH_PRECISION_B(h), abs=1.0e-12)
    assert _kernel_derivatives(h) == pytest.approx(HIGH_PRECISION_DERIVATIVES(h), abs=1.0e-11)
```

- [ ] **Step 2: Write RED rolling-goal and single-initializer tests**

```python
def test_intermediate_goal_uses_corridor_tangent_and_not_mission_heading() -> None:
    subgoal = select_legged_rolling_subgoal_v1(START, FAR_MISSION_GOAL, CORRIDOR, VIEW, PROFILE, DEADLINE, memory_ledger=MEMORY_LEDGER)
    assert subgoal.is_mission_goal is False
    assert subgoal.distance_from_start_m == pytest.approx(10.0)
    assert subgoal.pose.heading_rad == pytest.approx(CORRIDOR_TANGENT)
    assert subgoal.terminal_speed_reference_mps == min(0.3, PROFILE.max_forward_speed_mps)
    assert subgoal.terminal_speed_is_hard is False


def test_final_goal_inside_lookahead_preserves_exact_mission_pose() -> None:
    subgoal = select_legged_rolling_subgoal_v1(START, NEAR_MISSION_GOAL, CORRIDOR, VIEW, PROFILE, DEADLINE, memory_ledger=MEMORY_LEDGER)
    assert subgoal.is_mission_goal is True
    assert subgoal.pose == NEAR_MISSION_GOAL
    assert subgoal.terminal_speed_reference_mps == 0.0
    assert subgoal.terminal_speed_is_hard is True
    assert subgoal.mission_goal_hash == mission_goal_hash_v1(NEAR_MISSION_GOAL)
    assert subgoal.corridor_hash == CORRIDOR.corridor_hash
    assert subgoal.rolling_goal_hash == legged_rolling_goal_hash_v1(subgoal)


def test_initializer_is_single_byte_deterministic_and_starts_exactly_at_request() -> None:
    guesses = [initialize_legged_body_sqp_v1(REQUEST, PROFILE, CORRIDOR, SUBGOAL, DEADLINE, memory_ledger=fresh_memory_ledger()) for _ in range(5)]
    assert len({encode_initial_guess_semantics_v1(value) for value in guesses}) == 1
    assert guesses[0].states[0] == REQUEST.current_body_state
    assert len(guesses[0].a_mps2) == len(guesses[0].states) - 1
    assert len(guesses[0].omega_radps) == len(guesses[0].states) - 1
    assert len(guesses[0].duration_s) == len(guesses[0].states) - 1
    assert all(value.memory_token.owner == "initial_guess" for value in guesses)


@pytest.mark.parametrize("max_segments", [1, 48])
def test_initializer_segment_cap_always_keeps_n_plus_one_states(max_segments: int) -> None:
    guess = initialize_case(max_segments=max_segments, max_route_states=max_segments + 1)
    assert 1 <= len(guess.duration_s) <= max_segments
    assert len(guess.states) == len(guess.duration_s) + 1


def test_initializer_straight_and_reverse_stop_match_checked_in_golden_bytes() -> None:
    assert encode_initial_guess(initialize_case(case="straight_10m")) == STRAIGHT_10M_GOLDEN
    reverse = initialize_case(case="reverse_start_then_forward")
    assert encode_initial_guess(reverse) == REVERSE_STOP_GOLDEN
    assert any(state.v_mps == 0.0 for state in reverse.states[1:-1])


def test_zero_length_goal_uses_rotation_or_hold_without_zero_duration_segment() -> None:
    rotation = initialize_case(case="same_position_different_heading_stopped")
    hold = initialize_case(case="already_at_goal_stopped")
    assert all(value > 0.0 for value in rotation.duration_s + hold.duration_s)
    assert all(value == 0.0 for value in rotation.a_mps2 + hold.a_mps2)
    assert hold.states[0] == hold.states[-1]


@pytest.mark.parametrize("speed", [-1.0e-6, -PROFILE.max_reverse_speed_mps])
def test_reverse_brake_respects_min_and_max_duration_with_exact_stop(speed: float) -> None:
    guess = initialize_case(current_speed_mps=speed, reverse_enabled=True)
    stop_index = next(index for index, state in enumerate(guess.states) if index > 0 and state.v_mps == 0.0)
    assert all(PROFILE.min_segment_duration_s <= dt <= PROFILE.max_segment_duration_s for dt in guess.duration_s[:stop_index])
    assert all(0.0 < value <= PROFILE.max_linear_decel_mps2 for value in guess.a_mps2[:stop_index])


def test_forward_backward_speed_pass_never_rewrites_request_start_speed() -> None:
    guess = initialize_case(current_speed_mps=0.61, case="long_stop_at_goal")
    assert guess.states[0] == REQUEST_WITH_061_MPS.current_body_state
    assert guess.states[0].v_mps == 0.61


def test_too_short_to_brake_fails_instead_of_lowering_fixed_start_speed() -> None:
    with pytest.raises(LeggedSQPInitializationFailedV1):
        initialize_case(current_speed_mps=0.61, case="goal_1cm_ahead_requires_stop")
```

- [ ] **Step 3: Run RED tests**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_legged_body_kinematics.py tests/test_v2_legged_rolling_subgoals.py tests/test_v2_legged_body_sqp_initialization.py -q -x
```

Expected: collection fails only for absent modules.

- [ ] **Step 4: Implement stable closed-form propagation**

Use normalized time `s=t/T` and complex displacement:

```python
SERIES_KERNEL_MAX_ABS_H_V1 = 1.0e-2


def _a_kernel(h: float) -> complex:
    if abs(h) <= SERIES_KERNEL_MAX_ABS_H_V1:
        return sum((1j * h) ** n / (factorial(n) * (n + 1)) for n in range(9))
    return (exp(1j * h) - 1.0) / (1j * h)


def _b_kernel(h: float) -> complex:
    if abs(h) <= SERIES_KERNEL_MAX_ABS_H_V1:
        return sum((1j * h) ** n / (factorial(n) * (n + 2)) for n in range(9))
    return (exp(1j * h) * (1.0 - 1j * h) - 1.0) / (h * h)


def integrate_legged_body_segment_v1(
    start: LeggedBodyStateV1,
    a_mps2: float,
    omega_radps: float,
    duration_s: float,
) -> LeggedBodyStateV1:
    state = _exact_body_state(start)
    a = _exact_finite_float(a_mps2, "a_mps2")
    omega = _exact_finite_float(omega_radps, "omega_radps")
    duration = _exact_positive_float(duration_s, "duration_s")
    h = omega * duration
    displacement = exp(1j * state.yaw_rad) * duration * (
        state.v_mps * _a_kernel(h) + a * duration * _b_kernel(h)
    )
    return LeggedBodyStateV1(
        x_m=_positive_zero(state.x_m + displacement.real),
        y_m=_positive_zero(state.y_m + displacement.imag),
        yaw_rad=_positive_zero(state.yaw_rad + h),
        v_mps=_positive_zero(state.v_mps + a * duration),
    )
```

`legged_body_segment_jacobian_v1()` implements derivatives of the same A/B kernels, with the same closed `SERIES_KERNEL_MAX_ABS_H_V1` series branch、term count and fixed variable order `(x,y,yaw,v,a,omega,dt)`; the direct and series formulas and their first derivatives must match checked-in high-precision goldens at the threshold and its two adjacent binary64 values. Compare the analytic state Jacobian against central finite differences only as an additional test. Public headings remain one deterministic unwrapped sequence; wrap only error comparisons.

- [ ] **Step 5: Implement bounded sampling, costs, and direction rules**

Sampling step count is `max(1, ceil(point_speed_upper*duration_s / 0.25))`, where `point_speed_upper=max(|v_start|,|v_end|)+outward_rounded_margin_body_radius*|omega|`; `MAX_SEGMENT_SAMPLE_COUNT_V1=4096`, and exceeding it is a typed resource failure rather than coarsening. Samples include both endpoints at exact `t=i*duration/step_count`. Relative energy is a reporting proxy:

```python
energy = translation_m + 0.2 * abs(delta_yaw_rad) + 0.1 * integral_abs_accel + 0.05 * duration_s
```

The initializer rejects any segment whose speed endpoints exceed profile bounds or whose endpoints have opposite nonzero signs. A transition from positive to negative motion requires one state with exact `v_mps==0.0`; when `reverse_enabled=False`, all states require `v_mps>=0.0`.

- [ ] **Step 6: Implement rolling-goal selection and deterministic resampling**

Walk the fine corridor polyline from the request start by `lookahead_nominal_m`. If the remaining mission-path length is `<=lookahead_max_m`, use the exact mission pose. Otherwise select the first interpolated point at nominal lookahead, use the forward tangent, then scan backward in stable fine-cell order until `view.validate_body_pose(..., memory_ledger=memory_ledger)` proves the full stationary body pose; reject if no point remains at or beyond `lookahead_min_m`. Reserve bounded scan/output bytes before materialization, release every rejected-pose scratch token, and transfer one retained token into exactly one `LeggedRollingSubgoalV1`. Compute `rolling_goal_hash` only after every semantic invariant passes; the memory token is request-bound/nonserializable and excluded, and no downstream function accepts a naked pose in its place.

Freeze initializer construction as follows. Let the available segment cap be `min(profile.max_segments, request.resource_budget.max_route_states - 1)`; `max_route_states<2` is a typed initialization/resource failure before allocation. Precompute exact state/control/sample/connector bytes from that cap, reserve them through `memory_ledger` before allocation, and on success transfer only the exact retained initial-guess bytes into its internal token; all helper scratch and all failed paths release their tokens. For current `v<0`, compute `minimum_stop_time=abs(v)/max_linear_decel_mps2`, `M=max(1,ceil(minimum_stop_time/max_segment_duration_s))`, `total_stop_time=max(minimum_stop_time,M*min_segment_duration_s)`, uniform `dt=total_stop_time/M` and constant `a=-v/total_stop_time`. Reserve `M` prefix segments, integrate them with `omega=0`, and force the final prefix knot to exact `v=0.0`; this handles tiny negative speed by lowering acceleration and long stops by splitting while every `dt` stays in bounds. Sample and hard-check the full prefix using the same ledger/deadline checkpoints. Build a deterministic straight connector from the stopped pose back to the original fine-corridor start, then append the original corridor polyline; the Task 7 trust tube explicitly includes this checked stop-prefix/connector before the normal corridor band. The forward suffix speed pass starts at that exact zero knot, so no negative-to-positive adjacent endpoints exist. Reserve terminal zero-speed rotation capacity when the final required yaw differs from the last corridor tangent after canonical wrapping. For positive augmented arc length, the remaining translational count is `N=min(ceil(augmented_corridor_arc_length_m/0.5), remaining_cap)` and must be at least one; place its `N+1` points at exact arc fractions `k/N` with stable segment interpolation. For zero arc and zero current speed, skip translation: emit only the terminal rotation when heading differs, or one positive-duration `v=a=omega=0` hold of `min_segment_duration_s` when pose/heading already match. A zero-arc request with nonzero speed must form a valid checked braking/rejoin path or fail initialization; it never fabricates a zero-distance velocity change.

The initial-guess cruise target is `min(0.3, profile.max_forward_speed_mps)`—an initializer reference, never a hard minimum. Intermediate subgoals target that speed; a true mission goal targets `0.0`. For the forward suffix set `v_suffix[0]=0.0` after a reverse prefix, otherwise copy the nonnegative request speed exactly. On translational arc increments `ds[k]`, compute node speeds with one forward pass `v[k+1]=min(v_cruise,sqrt(v[k]^2+2*max_linear_accel_mps2*ds[k]))`. Pin index `0` as immutable: the backward braking pass updates only `k=N-1..1` with `v[k]=min(v[k],sqrt(v[k+1]^2+2*max_linear_decel_mps2*ds[k]))`, then separately checks the first span against the unchanged `v[0]`. If that span or the complete remaining distance cannot meet the requested terminal speed under the exact deceleration bound, initialization fails; it never lowers the request start speed. Raw `dt[k]=2*ds[k]/(v[k]+v[k+1])`; a zero denominator is a typed initialization failure. If raw `dt<min_segment_duration_s`, merge with the following span (or previous at the end) and recompute; if raw `dt>max_segment_duration_s`, split the span when cap remains; if neither operation is possible, fail rather than clamp and silently change the trapezoid. Every merge/split reruns the frozen-start speed pass and preserves `states[0]` byte-for-byte. Unwrap tangent yaw in forward order, set `a=(v[k+1]-v[k])/dt[k]` and `omega=(yaw[k+1]-yaw[k])/dt[k]`; bounds are checked but exact multiple-shooting equality is left to SQP. For terminal rotation, choose the minimum decimal12 `dt` in `[min_segment_duration_s,max_segment_duration_s]` satisfying both `omega=delta_yaw/dt` absolute rate and the Task 7 adjacent yaw-slew inequality against the preceding segment; use deterministic bisection with an upward-rounded feasible endpoint. Split a large delta into the minimum equal-angle segment count only when one segment cannot satisfy maximum duration/rate and cap remains; otherwise fail typed. States are always exactly segment count + 1, and no random seed is consumed. Checked-in golden bytes cover straight、turn、reverse-brake/stop、fixed-positive-start braking、too-short-to-brake failure、small-angle minimum-duration and cap-boundary cases.

- [ ] **Step 7: Run GREEN and numerical regressions**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_legged_body_kinematics.py tests/test_v2_legged_rolling_subgoals.py tests/test_v2_legged_body_sqp_initialization.py -q
```

Expected: all analytic-reference、Jacobian、wrap、reverse、stop、lookahead and resource cases pass.

- [ ] **Step 8: Review and commit Task 5**

```powershell
git -C path-planner diff --check
git -C path-planner add src/path_planner/v2/legged_body_kinematics.py src/path_planner/v2/legged_rolling_subgoals.py src/path_planner/v2/legged_body_sqp_initialization.py tests/test_v2_legged_body_kinematics.py tests/test_v2_legged_rolling_subgoals.py tests/test_v2_legged_body_sqp_initialization.py
git -C path-planner commit -m "feat: add legged rolling body initialization"
git add path-planner
git commit -m "build: integrate legged body initialization"
```

---

### Task 6: Canonicalize and Strictly Serialize Private Candidates and Public L2 Trajectories

**Files:**
- Create: `path-planner/src/path_planner/v2/legged_body_sqp_serialization.py`
- Create: `path-planner/tests/test_v2_legged_body_sqp_serialization.py`
- Modify: `path-planner/tests/test_v2_serialization.py`
- Modify: `path-planner/tests/test_v2_wheel_sqp_serialization.py`

**Interfaces:**
- Consumes: request/profile/corridor/subgoal/internal SQP states and controls、kinematic propagation、canonical JSON；planning-time calls also consume the exact memory ledger and a correctly owned pre-reserved phase token。
- Consumes the Task 1 `CanonicalLeggedBodyCandidateV1` contract and single `canonicalize_legged_body_scalar_v1()` primitive; produces `CanonicalLeggedBodySegmentV1`、the only candidate materializer、candidate/segment/trajectory/request/profile hashes、strict checkpoint/memory-token-capable encode/decode、`DecodedLeggedBodyCandidateV1(candidate,candidate_bytes_token,decoded_candidate_token)` and public trajectory promotion/projection helpers。

- [ ] **Step 1: Write RED canonical-byte and identity tests**

```python
def test_candidate_round_trip_is_byte_stable_and_fresh() -> None:
    candidate = materialize_canonical_legged_candidate_v1(RAW_RESULT, request=REQUEST, profile=PROFILE, corridor=CORRIDOR, rolling_goal=SUBGOAL)
    encoded = encode_legged_body_candidate_v1(candidate)
    decoded = decode_legged_body_candidate_v1(encoded)
    assert decoded == candidate
    assert decoded is not candidate
    assert encode_legged_body_candidate_v1(decoded) == encoded


@pytest.mark.parametrize("mutation", ["request_hash", "profile_hash", "controller_hash", "snapshot_hash", "corridor_hash", "rolling_goal_hash"])
def test_candidate_rejects_every_identity_drift(mutation: str) -> None:
    with pytest.raises(LeggedBodyCodecErrorV1):
        decode_legged_body_candidate_v1(mutate_encoded_candidate(VALID_BYTES, mutation))


def test_candidate_canonicalizes_negative_zero_and_half_even() -> None:
    candidate = decode_legged_body_candidate_v1(encode_legged_body_candidate_v1(make_negative_zero_candidate()))
    assert all(copysign(1.0, value) == 1.0 for value in candidate_float_values(candidate) if value == 0.0)
    assert canonicalize_legged_body_scalar_v1(1.2345678901235) == 1.234567890124


def test_request_start_is_the_only_unquantized_state_anchor() -> None:
    request = make_request(current_body_state=LeggedBodyStateV1(0.123456789012345, 0.0, 0.0, 0.4))
    candidate = materialize_case(request=request)
    assert candidate.states[0] == request.current_body_state
    assert candidate.states[1].x_m == canonicalize_legged_body_scalar_v1(candidate.states[1].x_m)
```

- [ ] **Step 2: Write RED strict-JSON and public-promotion tests**

```python
@pytest.mark.parametrize("payload", [DUPLICATE_KEY_BYTES, BOM_BYTES, NAN_BYTES, INFINITY_BYTES, EXTRA_KEY_BYTES, MISSING_KEY_BYTES])
def test_codec_rejects_noncanonical_json(payload: bytes) -> None:
    with pytest.raises(LeggedBodyCodecErrorV1):
        decode_legged_body_candidate_v1(payload)


def test_only_l2_receipt_can_promote_public_timed_trajectory() -> None:
    with pytest.raises(ValueError, match="L2"):
        promote_legged_body_trajectory_v1(CANDIDATE, FAILED_RECEIPT, RISK)
    trajectory = promote_legged_body_trajectory_v1(CANDIDATE, PASSED_RECEIPT, RISK)
    assert trajectory.validation_level is ValidationLevelV2.L2
    assert trajectory.source_candidate_hash == CANDIDATE.candidate_hash


@pytest.mark.parametrize("operation", ["materialize", "encode", "decode"])
def test_codec_phase_cutoff_aborts_without_publishing_partial_bytes(operation: str) -> None:
    with pytest.raises(LeggedSQPResourceCutoffV1):
        run_codec_with_checkpoint_crossing_cutoff(operation)
    assert PARTIAL_OUTPUT_SINK == []


def test_provider_codec_rejects_stale_or_wrong_owner_tail_token_without_recharging() -> None:
    for token in (STALE_ENCODE_TOKEN, WRONG_REQUEST_ENCODE_TOKEN, WRONG_BYTES_ENCODE_TOKEN):
        with pytest.raises((LeggedBodyCodecErrorV1, LeggedSQPResourceCutoffV1)):
            encode_legged_body_candidate_v1(CANDIDATE, checkpoint=noop, memory_ledger=MEMORY_LEDGER, memory_token=token, output_memory_token=OUTPUT_TOKEN)
    assert MEMORY_LEDGER.peak_bytes == PEAK_BEFORE_CODEC


def test_provider_decode_keeps_encoded_bytes_and_uses_distinct_pre_reserved_object_token() -> None:
    decoded = decode_legged_body_candidate_with_token_v1(
        VALID_BYTES,
        checkpoint=noop,
        memory_ledger=MEMORY_LEDGER,
        memory_token=ENCODED_BYTES_TOKEN,
        decoded_output_memory_token=DECODED_OBJECT_TOKEN,
    )
    assert decoded.candidate is not CANDIDATE
    assert decoded.candidate_bytes_token.bytes == ENCODED_BYTES_TOKEN.bytes
    assert decoded.decoded_candidate_token.bytes == DECODED_OBJECT_TOKEN.bytes
    assert MEMORY_LEDGER.peak_bytes == PEAK_BEFORE_CODEC
```

- [ ] **Step 3: Run RED tests**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_legged_body_sqp_serialization.py tests/test_v2_serialization.py tests/test_v2_wheel_sqp_serialization.py -q -x
```

Expected: only new imports/tests fail; generic and wheel codecs stay green.

- [ ] **Step 4: Implement decimal12 half-even canonicalization and hashes**

Use the already audited wheel-codec conversion rule `Decimal(format(normalized, ".17g")).quantize(Decimal("1e-12"), rounding=ROUND_HALF_EVEN)`, then convert to exact finite float and normalize signed zero and unwrapped yaw. Do not use `Decimal.from_float()` because that would make the frozen tie fixture round differently from the repository's decimal12 contract. The sole state exception is `states[0]`: copy `request.current_body_state` exactly as the identity anchor even when it is not decimal12; require exact equality on decode/L2. `materialize_canonical_legged_candidate_v1(..., *, checkpoint=None, memory_ledger=None, memory_token=None)` invokes the optional checkpoint before/after every segment and before hash/return. Pure offline calls require both memory arguments `None`; provider/solver calls require both exact, validate and fill the already-reserved solver-candidate capacity without a second charge, and leave ownership in the enclosing feasible optimization result's `solver_candidate_token` rather than pretending that the token is stored in the semantic candidate. Canonicalize every control/duration and every later state, replay each segment from the prior stored state, and require declared/replayed end state to match within `profile.hard_constraint_tolerance`; never trust solver endpoint fields independently.

Hash domains are distinct bytes prefixes. Candidate hash covers segment hashes and all request/profile/controller/snapshot/corridor/corridor-geometry/subgoal/solver/SQP-geometry/canonicalization/preloaded-backend IDs and backend version hash, plus repair lineage. Public trajectory hash additionally covers L2 receipt and risk-evidence hash. Each own hash field is excluded only while computing itself.

- [ ] **Step 5: Implement strict loader and fresh-object round trip**

```python
def _strict_json_load_v1(encoded: bytes) -> dict[str, object]:
    if type(encoded) is not bytes or encoded.startswith(codecs.BOM_UTF8):
        raise LeggedBodyCodecErrorV1("encoded candidate must be BOM-free bytes")
    value = json.loads(
        encoded.decode("utf-8", errors="strict"),
        object_pairs_hook=_reject_duplicate_keys,
        parse_constant=_reject_constant,
    )
    if type(value) is not dict:
        raise LeggedBodyCodecErrorV1("top-level payload must be object")
    return value
```

`encode_legged_body_candidate_v1()` and `decode_legged_body_candidate_v1()` take optional keyword-only checkpoint/memory arguments. Pure calls omit all memory arguments. Provider encode requires the exact ledger、`memory_token` for scratch and distinct live `output_memory_token` for retained canonical bytes; it validates both capacities, fills the already-reserved output scope and releases scratch without changing `peak_bytes`. Calls check before work、before/after each segment and every 64 scalar/key records、before internal canonical re-encode and before return. Provider supplies a closure that gives global deadline precedence and then enforces the current encode-phase cutoff; default `None` preserves pure codec use. Bytes are assembled privately and returned only after the final checkpoint, so an abort cannot leak a prefix；the raw solver candidate and its separate token remain live. The pure decoder returns only a fresh candidate. The provider-only `decode_legged_body_candidate_with_token_v1()` requires both the retained encoded-bytes handle and a distinct pre-reserved decoded-object output handle, then returns exact `DecodedLeggedBodyCandidateV1(candidate,candidate_bytes_token,decoded_candidate_token)` after transferring both to new owners and making both input handles stale. It never stores a fresh object under the bytes-only handle and never discards the bytes required by L2. Provider/L2 must use the returned handles; no token is implicitly borrowed or looked up by owner. Every decoder validates exact key sets、exact primitive types、finite values、hashes and semantic relations, then requires `encode(decoded)==input`; whitespace or numeric spellings that are semantically equal but noncanonical are rejected. The decoder allows the first state to be a finite non-decimal12 anchor, but the L2/API boundary must match it exactly to the request whose exact bytes participate in `request_hash`; no later state receives that exception.

- [ ] **Step 6: Run GREEN and cross-codec regressions**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_legged_body_sqp_serialization.py tests/test_v2_serialization.py tests/test_v2_wheel_sqp_serialization.py tests/test_v2_wheel_sqp_contracts.py -q
```

Expected: all pass and no existing canonical bytes change.

- [ ] **Step 7: Review and commit Task 6**

```powershell
git -C path-planner diff --check
git -C path-planner add src/path_planner/v2/legged_body_sqp_serialization.py tests/test_v2_legged_body_sqp_serialization.py tests/test_v2_serialization.py tests/test_v2_wheel_sqp_serialization.py
git -C path-planner commit -m "feat: add canonical legged body trajectory codec"
git add path-planner
git commit -m "build: integrate legged body trajectory codec"
```

---

### Task 7: Implement the Deterministic Direct Multiple-Shooting SQP and Resource Admission

**Files:**
- Create: `path-planner/src/path_planner/v2/legged_body_sqp_solver.py`
- Create: `path-planner/tests/test_v2_legged_body_sqp_solver.py`

**Interfaces:**
- Consumes: request/profile/hard view/corridor/subgoal/initial guess/deadline、the exact request memory ledger、a preloaded exact `LeggedSLSQPBackendV1`、optional one repair constraint and `LeggedBodyL2ReserveModelV1`。
- Produces: `LeggedSLSQPBackendV1.load()` for request-external optional dependency readiness and `solve_legged_body_sqp_v1(request, profile, view, corridor, rolling_goal, initial_guess, deadline, *, backend, reserve_model, memory_ledger, repair_constraint) -> LeggedBodySQPOptimizationResultV1` whose feasible branch contains the exact Task 1/6 `CanonicalLeggedBodyCandidateV1`、fixed layout pack/unpack、constraint/Jacobian functions and typed backend/resource/deadline results。

- [ ] **Step 1: Write RED layout, feasibility, and no-randomness tests**

```python
def test_layout_is_fixed_states_then_controls_then_durations() -> None:
    layout = LeggedBodySQPLayoutV1(segment_count=3)
    assert layout.state_slice == slice(0, 16)
    assert layout.accel_slice == slice(16, 19)
    assert layout.omega_slice == slice(19, 22)
    assert layout.duration_slice == slice(22, 25)
    assert layout.variable_count == 25


def test_straight_corridor_solves_and_preserves_exact_start() -> None:
    result = solve_case("straight")
    assert result.status is LeggedBodySQPStatusV1.FEASIBLE
    assert result.candidate is not None
    assert result.candidate.states[0] == REQUEST.current_body_state


def test_solver_does_not_use_rng_or_multistart(monkeypatch) -> None:
    monkeypatch.setattr(numpy.random, "default_rng", forbidden_call)
    result = solve_case("straight")
    assert result.status is LeggedBodySQPStatusV1.FEASIBLE
    assert result.solve_attempt_count == 1


def test_adjacent_yaw_acceleration_exact_boundary_passes_and_nextafter_fails() -> None:
    assert solve_case("yaw_slew_boundary").status is LeggedBodySQPStatusV1.FEASIBLE
    assert solve_case("yaw_slew_nextafter").status is LeggedBodySQPStatusV1.INFEASIBLE


def test_mission_goal_requires_exact_zero_terminal_speed() -> None:
    assert solve_case("mission_terminal_zero").status is LeggedBodySQPStatusV1.FEASIBLE
    assert solve_case("mission_terminal_nonzero").status is LeggedBodySQPStatusV1.INFEASIBLE


def test_merit_and_analytic_jacobian_match_checked_in_golden() -> None:
    problem = make_problem("turn_with_duration_variables")
    assert problem.objective(problem.x0) == MERIT_GOLDEN
    assert tuple(problem.objective_jacobian(problem.x0)) == MERIT_JACOBIAN_GOLDEN
    assert problem.objective_jacobian(problem.x0) == pytest.approx(central_difference(problem.objective, problem.x0), abs=1.0e-7)


def test_collision_sample_dimension_is_fixed_from_profile_bounds_not_current_iterate() -> None:
    problem = make_problem("slow_short_initializer")
    assert problem.segment_sample_counts == EXPECTED_PROFILE_BOUND_COUNTS
    faster_longer_iterate = mutate_within_bounds(problem.x0, speed="max", duration="max")
    assert len(problem.collision_constraints(faster_longer_iterate)) == len(problem.collision_constraints(problem.x0))
    assert problem.collision_constraints(faster_longer_iterate)[KNOWN_COLLISION_INDEX] < 0.0


def test_frozen_broadphase_contains_every_potentially_violating_unsafe_cell() -> None:
    problem = make_problem("unsafe_cells_on_buffer_edges")
    assert problem.broadphase_audit == BROADPHASE_GOLDEN
    assert problem.closest_unsafe_cell(TIE_SAMPLE) == Cell(TIE_EXPECTED_Y, TIE_EXPECTED_X)


def test_support_point_inside_unsafe_square_has_negative_sdf_and_exit_jacobian() -> None:
    value, jac = unsafe_square_sdf_and_jacobian(POINT_INSIDE_CENTER_TIE, UNSAFE_CELL)
    assert value == -0.25
    assert tuple(jac) == EXPECTED_X_NEGATIVE_FACE_EXIT_JACOBIAN
    repaired = solve_case("support_point_initially_inside_repair_cell")
    assert repaired.audit.repair_constraint_jacobian != ZERO_VECTOR


@pytest.mark.parametrize("cap", ["broadphase_cells", "unsafe_records", "distance_evaluations", "memory"])
def test_sqp_broadphase_caps_stop_before_backend_and_partial_problem(cap: str) -> None:
    result, audit = solve_with_tiny_broadphase_cap(cap)
    assert result.status is LeggedBodySQPStatusV1.RESOURCE_EXHAUSTED
    assert audit.backend_called is False
    assert audit.partial_problem_published is False


def test_sqp_broadphase_checks_deadline_inside_wide_candidate_region() -> None:
    result = solve_with_clock_crossing_mid_broadphase_batch()
    assert result.status is LeggedBodySQPStatusV1.DEADLINE_EXPIRED


def test_distance_callback_crossing_solver_cutoff_preserves_tail_before_global_deadline() -> None:
    result, audit = solve_with_clock_crossing_cutoff_mid_distance_batch()
    assert audit.global_deadline_expired is False
    assert result.status is LeggedBodySQPStatusV1.RESOURCE_EXHAUSTED
    assert result.post_solver_tail_token is None
```

- [ ] **Step 2: Write RED deadline, optional-backend, numeric, and reserve tests**

```python
def test_missing_scipy_backend_load_is_typed_unsupported_without_importing_old_planners(monkeypatch) -> None:
    monkeypatch.setattr(importlib, "import_module", missing_scipy_only)
    backend = LeggedSLSQPBackendV1.load()
    result = solve_case("straight", backend=backend)
    assert result.status is LeggedBodySQPStatusV1.BACKEND_UNAVAILABLE
    assert result.reason_code == "legged_body_profile_unsupported"


def test_cold_backend_import_happens_before_request_deadline_exists() -> None:
    provider = build_provider_with_import_clock_advance(10.0)
    outcome = plan_with_fresh_request_clock(provider)
    assert provider.backend.scipy_version == "1.18.0"
    assert outcome.telemetry.elapsed_s < REQUEST.replan_period_s


def test_reserve_rejects_before_backend_call(monkeypatch) -> None:
    monkeypatch.setattr(solver, "_run_slsqp_v1", forbidden_call)
    result = solve_case("straight", remaining_s=1.0e-9)
    assert result.status is LeggedBodySQPStatusV1.RESOURCE_EXHAUSTED
    assert result.reason_code == "legged_resource_budget_exceeded"


def test_expiry_after_backend_return_cannot_be_success() -> None:
    result = solve_with_backend_that_advances_clock_past_deadline()
    assert result.status is LeggedBodySQPStatusV1.DEADLINE_EXPIRED
    assert result.candidate is None


def test_materialization_crossing_tail_cutoff_is_resource_not_candidate() -> None:
    result = solve_with_clock_crossing_cutoff_during_materialization()
    assert result.status is LeggedBodySQPStatusV1.RESOURCE_EXHAUSTED
    assert result.candidate is None


def test_solver_cutoff_preserves_tail_before_global_deadline() -> None:
    result, audit = solve_with_clock_crossing_solver_cutoff_only()
    assert audit.global_deadline_expired is False
    assert result.status is LeggedBodySQPStatusV1.RESOURCE_EXHAUSTED
    assert result.reason_code == "legged_resource_budget_exceeded"


def test_evaluation_cap_stops_before_the_first_over_cap_callback() -> None:
    result, audit = solve_with_callback_cap(7)
    assert result.status is LeggedBodySQPStatusV1.RESOURCE_EXHAUSTED
    assert audit.value_callback_count == 7
    assert audit.backend_callback_count_attempted == 8


def test_l2_tail_reserve_exact_boundary_is_not_admitted() -> None:
    reserve = RESERVE_MODEL.assess(PROFILE, snapshot_cell_count=VIEW.snapshot_cell_count, snapshot_canonical_byte_count=VIEW.snapshot_canonical_byte_count, segment_count=48, risk_segment_count=48, encoded_scalar_count=340)
    memory_boundary = PROFILE.solver_memory_reservation_bytes + reserve.tail_bytes
    assert RESERVE_MODEL.admit(deadline=deadline_with_remaining(nextafter(reserve.tail_time_s, inf)), memory_ledger=ledger_with_available(memory_boundary + 1), reserve=reserve).accepted
    assert not RESERVE_MODEL.admit(deadline=deadline_with_remaining(reserve.tail_time_s), memory_ledger=ledger_with_available(memory_boundary + 1), reserve=reserve).accepted
    assert not RESERVE_MODEL.admit(deadline=deadline_with_remaining(nextafter(reserve.tail_time_s, inf)), memory_ledger=ledger_with_available(memory_boundary), reserve=reserve).accepted


def test_feasible_solver_transfers_candidate_and_remaining_tail_without_second_peak_charge() -> None:
    result, ledger = solve_with_memory_audit("straight")
    assert result.solver_candidate_token.bytes == result.reserve_audit.solver_candidate_bytes
    assert result.post_solver_tail_token.bytes == result.reserve_audit.tail_bytes - result.reserve_audit.solver_candidate_bytes
    before = ledger.peak_bytes
    phase_tokens = split_post_solver_tail_v1(result.solver_candidate_token, result.post_solver_tail_token, result.reserve_audit, memory_ledger=ledger)
    assert phase_tokens.solver_candidate.bytes == result.reserve_audit.solver_candidate_bytes
    assert sum(token.bytes for token in phase_tokens.ordered()) == result.reserve_audit.tail_bytes
    assert ledger.peak_bytes == before
    with pytest.raises(ValueError, match="stale"):
        ledger.release(result.post_solver_tail_token)


def test_candidate_representation_peak_is_exactly_pre_reserved() -> None:
    result, ledger = solve_encode_and_fresh_decode_with_memory_audit("straight")
    reserve = result.reserve_audit
    assert reserve.candidate_bytes == reserve.solver_candidate_bytes + reserve.encoded_candidate_bytes + reserve.decoded_candidate_bytes
    assert result.audit.simultaneous_candidate_representations == ("solver_object", "canonical_bytes", "fresh_decoded_object")
    assert ledger.peak_bytes == result.audit.pre_admitted_peak_bytes
    with pytest.raises(LeggedSQPResourceCutoffV1):
        solve_encode_and_fresh_decode_with_memory_limit(result.audit.pre_admitted_peak_bytes - 1)


def test_solver_cutoff_reserves_mandatory_risk_features_before_complete_l2() -> None:
    result, audit = solve_then_advance_feature_clock_to_reserved_bound()
    assert result.status is LeggedBodySQPStatusV1.FEASIBLE
    assert audit.risk_feature_phase_completed is True
    assert audit.remaining_before_l2_s > audit.l2_time_s


def test_l2_tail_includes_two_full_large_snapshot_hash_passes_at_exact_boundary() -> None:
    reserve = assess_large_snapshot_tail(cell_count=PROFILE.max_snapshot_cells)
    assert reserve.snapshot_hash_pass_count == 2
    assert not admit_with_remaining(reserve.tail_time_s).accepted
    assert admit_with_remaining(nextafter(reserve.tail_time_s, inf)).accepted
    result = run_l2_hash_clock_consuming_reserved_two_pass_work()
    assert result.audit.snapshot_hash_pass_count == 2


@pytest.mark.parametrize(
    ("scipy_status", "success", "expected_status", "expected_reason"),
    SCIPY_STATUS_MAPPING_CASES,
)
def test_scipy_status_mapping_is_total_and_exact(
    scipy_status: int,
    success: bool,
    expected_status: LeggedBodySQPStatusV1,
    expected_reason: str | None,
) -> None:
    result = solve_with_backend_result(status=scipy_status, success=success)
    assert (result.status, result.reason_code) == (expected_status, expected_reason)


def test_repair_adds_only_the_frozen_cell_phase_clearance_constraints() -> None:
    original, repaired = build_problem_pair_with_valid_repair()
    assert repaired.base_constraint_hash == original.base_constraint_hash
    assert repaired.repair_constraint_count == 5 * repaired.body_support_point_count
    assert repaired.removed_or_relaxed_constraint_count == 0


def test_repair_extra_constraints_are_in_the_combined_cap_and_memory_admission() -> None:
    assert build_repair_problem_at_combined_constraint_cap().combined_constraint_count == PROFILE.max_sqp_support_constraints
    result, audit = build_repair_problem_one_over_combined_constraint_cap()
    assert result.status is LeggedBodySQPStatusV1.RESOURCE_EXHAUSTED
    assert audit.backend_called is False
```

- [ ] **Step 3: Run RED tests**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_legged_body_sqp_solver.py -q -x
```

Expected: import failure for solver/layout.

- [ ] **Step 4: Implement exact layout, bounds, equality and inequality constraints**

For `N` segments, decision order is `q[0:N+1]` row-major `(x,y,yaw,v)`, then `a[0:N]`, `omega[0:N]`, `dt[0:N]`. Equalities enforce exact request start and `q[k+1]=integrate(q[k],a[k],omega[k],dt[k])`. Bounds enforce profile speed/yaw/duration limits. Acceleration semantics follow motion direction without crossing zero inside a segment: forward uses `-max_linear_decel_mps2 <= a <= max_linear_accel_mps2`; reverse uses `-max_linear_accel_mps2 <= a <= max_linear_decel_mps2`; a zero-speed segment chooses the sign of its nonzero endpoint, and a zero-to-zero rotation requires `a==0`. Inequalities enforce terminal Euclidean position error `<=profile.profile.goal_position_tolerance_m`、wrapped absolute heading error `<=profile.profile.goal_heading_tolerance_rad`、corridor band、sampled oriented-body signed-distance margin、no forbidden reverse and no in-segment sign crossing. When `rolling_goal.terminal_speed_is_hard=True`, add exact `q[N].v_mps==rolling_goal.terminal_speed_reference_mps==0.0`; when false, the hashed cruise reference remains initializer-only and adds no minimum-speed feasibility constraint.

For every adjacent planned pair, freeze the yaw-slew hard constraint as:

```python
tau_s = 0.5 * (duration_s[k] + duration_s[k + 1])
abs(omega_radps[k + 1] - omega_radps[k]) / tau_s <= profile.max_yaw_accel_radps2
```

The approved request has no measured pre-horizon yaw rate, so the planner does not invent a `omega[-1]` or claim the controller handoff transient satisfies this bound; it applies to all transitions represented inside the published trajectory. The first segment still obeys the absolute yaw-rate bound, and external tracking/handoff remains outside planner authority.

Freeze the SQP geometry approximation under `LEGGED_BODY_SQP_GEOMETRY_V1`: each normal forward knot and every fixed collision sample is monotonically assigned by the initializer to one fine-corridor polyline segment, and its center must remain within `CORRIDOR_TRUST_RADIUS_M=1.0`. For a negative initial speed, the analytically sampled reverse-stop prefix and straight rejoin connector from Task 5 form an additional sealed initial trust polyline and must independently pass the same hard clearance constraints; once rejoined, assignment cannot return to that prefix. Build the exact tensor body support lattice over the margin-expanded rectangle with `nx=max(2,ceil((body_length+2*margin)/0.25)+1)`、`ny=max(2,ceil((body_width+2*margin)/0.25)+1)` and row-major local `(y,x)` order, so maximum spacing is `BODY_SUPPORT_SPACING_M=0.25` and all corners/edges are present.

Collision-vector dimension is frozen once at problem construction and never recomputed from an iterate. Let `body_radius=nextafter(hypot((body_length+2*margin)/2,(body_width+2*margin)/2),inf)` and `point_speed_bound=nextafter(max(max_forward_speed_mps,max_reverse_speed_mps)+body_radius*max_yaw_rate_radps,inf)`. For every motion segment use `segment_step_count=max(1,ceil(point_speed_bound*profile.max_segment_duration_s/MAX_SUPPORT_POINT_MOTION_M))` and `segment_sample_count=segment_step_count+1`, with `MAX_SUPPORT_POINT_MOTION_M=0.25`; sample phases are exactly `i/segment_step_count`. Speed、yaw-rate and duration bounds then prove that any feasible iterate moves a support point by at most 0.25 m between adjacent samples, even when the initializer was slower/shorter. At each sample, every support point must have exact point-to-unsafe-cell-square/map-boundary distance at least `required_clearance=nextafter(0.5*hypot(BODY_SUPPORT_SPACING_M,BODY_SUPPORT_SPACING_M)+MAX_SUPPORT_POINT_MOTION_M+profile.continuous_separation_epsilon_m,inf)`. “Unsafe” is the Task 2 hard-view complement, including unknown、obstacle、nontraversable、slope、confidence and out-of-bounds.

Freeze a finite broad phase before constructing SLSQP. For each motion segment, take the union of all fine-corridor/reverse-prefix polyline segments assigned to its fixed samples, form their axis-aligned bounds, and expand each side by `nextafter(CORRIDOR_TRUST_RADIUS_M+body_radius+required_clearance,inf)`. Enumerate every in-bounds 0.5 m cell square intersecting that conservative box in `(y,x)` order, query the immutable hard view, retain every unsafe cell, then deduplicate per motion segment in `(y,x)` order; map-boundary distance is a separate analytic candidate. Any unsafe cell outside the box is provably farther than `required_clearance` from every support point of an iterate satisfying its trust constraint, so omitting it cannot turn an infeasible final candidate feasible. The set and order never depend on the current iterate.

Before allocating lattice/sample/constraint/broad-phase arrays, compute all counts with overflow-safe integers. Support points must not exceed `profile.max_body_support_points`; `base_support_constraint_count=sum(segment_sample_count)*support_point_count`; for a repair, `repair_extra_constraint_count=REPAIR_PHASE_SAMPLE_COUNT_V1*support_point_count`, otherwise zero; `combined_support_constraint_count=base+repair_extra` must not exceed `profile.max_sqp_support_constraints`. Cumulative enumerated broad-phase cells and retained per-segment unsafe-cell records must stay within their profile caps. Per value/Jacobian callback, `base_distance_evaluation_count=sum(segment_sample_count*support_point_count*max(1,len(segment_unsafe_cells)))` and `combined_distance_evaluation_count=base_distance_evaluation_count+repair_extra_constraint_count` must not exceed `profile.max_sqp_distance_evaluations_per_callback`; each repair inequality performs one additional exact failing-cell distance evaluation even when that cell is absent from the base set. Account `SQP_SUPPORT_POINT_RECORD_BYTES_V1=64`、`SQP_SUPPORT_CONSTRAINT_RECORD_BYTES_V1=96`、`SQP_BROADPHASE_CELL_RECORD_BYTES_V1=16`、`SQP_UNSAFE_CELL_RECORD_BYTES_V1=32`、one separate `SQP_REPAIR_CELL_RECORD_BYTES_V1=32` when repairing, plus combined packed decision/value/Jacobian buffers through the exact shared memory ledger and `profile.solver_memory_reservation_bytes`. Equality at a byte/count cap is allowed; the next record fails before mutation as `legged_resource_budget_exceeded`. A single unified checkpoint runs before/after reservation、every 256 enumerated broad-phase cells、every bounded segment batch and every 256 distance evaluations inside every value/Jacobian callback; its order is global-deadline expiry first, then `deadline.remaining_s <= tail_reserve_s` solver cutoff, then work. Thus one large callback cannot consume the reserved post-solver tail while the global deadline remains open. Initial and repair problems both perform the complete admission before backend invocation; no partial problem escapes.

Distance constraints use signed distance. For an unsafe closed square `[xmin,xmax]×[ymin,ymax]`, let `ox=max(xmin-x,0,x-xmax)` and `oy=max(ymin-y,0,y-ymax)`: outside SDF is `hypot(ox,oy)` with its analytic outward gradient; on/inside, SDF is `-min(x-xmin,xmax-x,y-ymin,ymax-y)`. Equal inside-face distances choose x before y and negative-side face before positive-side face, yielding a nonzero deterministic exit gradient even at the square center. Map SDF is `min(x-map_xmin,map_xmax-x,y-map_ymin,map_ymax-y)` inside the closed map and the negative Euclidean distance to it outside, with the same x-before-y/negative-before-positive tie order. The hard constraint is `min(map_sdf, unsafe_square_sdf for frozen segment-local unsafe cells)-required_clearance >= 0`, with analytic chain rule through body pose. Select a nearest obstacle tie by `(sdf,cell_y,cell_x,axis_branch,sign_branch)`. The same fixed sample count、support order、broad-phase set/order、nearest tie and Jacobian branch are used in checked-in outside/edge/inside/center goldens; finite differences are only an independent check. These conservative SQP samples are feasibility constraints but still do not grant L2.

The SQP scalar merit is only a search mechanism. Freeze `MERIT_SMOOTH_EPS=1.0e-6`; `smooth_abs(z)=sqrt(z*z+eps*eps)-eps` and `smooth_norm(dx,dy)=sqrt(dx*dx+dy*dy+eps*eps)-eps`. For fixed `N`, compute once: `time_scale=N*profile.max_segment_duration_s`, `accel_scale=max(profile.max_linear_accel_mps2,profile.max_linear_decel_mps2)`, `omega_scale=profile.max_yaw_rate_radps`, and `energy_scale=max(1.0, rolling_goal.distance_from_start_m + 0.2*pi + 0.1*accel_scale*time_scale + 0.05*time_scale)`. Then:

```python
normalized_time = sum(dt[k] for k in range(N)) / time_scale
normalized_energy = sum(
    smooth_norm(x[k + 1] - x[k], y[k + 1] - y[k])
    + 0.2 * smooth_abs(yaw[k + 1] - yaw[k])
    + 0.1 * smooth_abs(a[k]) * dt[k]
    + 0.05 * dt[k]
    for k in range(N)
) / energy_scale
control_slew = (
    sum(
        ((a[k + 1] - a[k]) / (2.0 * accel_scale)) ** 2
        + ((omega[k + 1] - omega[k]) / (2.0 * omega_scale)) ** 2
        for k in range(N - 1)
    ) / max(1, N - 1)
)
corridor_deviation = sum(
    squared_distance_to_frozen_assigned_segment(q[k], assignment[k])
    for k in range(N + 1)
) / ((N + 1) * CORRIDOR_TRUST_RADIUS_M ** 2)
merit = (
    normalized_time
    + normalized_energy
    + 1.0e-4 * control_slew
    + 1.0e-3 * corridor_deviation
)
```

Assignments and segment endpoints are fixed by the initializer, and closest-point projection clamps with branch order interior→start→end and an exact left-segment tie. The objective and `objective_jacobian` use precisely the same branches and smooth primitives; normalization never depends on the current iterate. It never includes hard-violation slack in a publishable candidate and never replaces the system lexicographic selection key.

- [ ] **Step 5: Implement lazy SLSQP adapter and deterministic status mapping**

```python
def _run_slsqp_v1(
    problem: LeggedBodySQPProblemV1,
    deadline: PlanningDeadlineV2,
    backend: LeggedSLSQPBackendV1,
    *,
    tail_reserve_s: float,
) -> OptimizeResult:
    def callback(_: np.ndarray) -> None:
        if deadline.expired:
            raise LeggedSQPDeadlineExpiredV1("planning deadline expired")
        if deadline.remaining_s <= tail_reserve_s:
            raise LeggedSQPResourceCutoffV1("solver reached reserved L2 tail")

    return backend.minimize(
        problem.objective,
        problem.x0,
        method="SLSQP",
        jac=problem.objective_jacobian,
        bounds=problem.bounds,
        constraints=problem.constraints,
        callback=callback,
        options={"maxiter": problem.profile.max_sqp_iterations, "ftol": problem.profile.sqp_ftol, "disp": False},
    )
```

`LeggedSLSQPBackendV1.load()` is called while constructing the explicit provider, before any request or `PlanningDeadlineV2`; it lazily imports `scipy.optimize`, requires exactly `scipy==1.18.0`, seals backend/module/version hashes, and returns a typed unavailable backend on failure. The solver never imports. Provider/API token checks prevent backend replacement after deadline creation.

Set BLAS/OpenMP thread-count guards to one inside the execution environment used by tests/runner; do not mutate global environment during import. Compute `tail_reserve_s` once and seal `solver_cutoff_monotonic_s = deadline.deadline_monotonic_s - tail_reserve_s` in the problem audit. Every objective、objective-Jacobian、constraint、constraint-Jacobian and callback first gives global expiry precedence, then stops with typed resource cutoff when `deadline.remaining_s <= tail_reserve_s`; do not create a second `PlanningDeadlineV2`. Thus a long SLSQP evaluation cannot bypass the reserved tail checkpoint. A deterministic wrapper increments `value_callback_count` before each objective or constraint-value call; the call that would exceed `max_sqp_function_evaluations` raises the typed resource stop before evaluating user math. Jacobian calls have a separate audit count but share deadline/nonfinite checks. Do not trust SciPy `nfev` as the authority; compare it only as telemetry.

Freeze backend result mapping after typed exception precedence (`deadline` before `resource`) and before candidate materialization:

| exact SciPy 1.18 status | required `success` | v1 result | reason |
|---:|:---:|---|---|
| `0` | `True` | `FEASIBLE` only after all independent residual checks | `None` |
| `9` | `False` | `RESOURCE_EXHAUSTED` | `legged_resource_budget_exceeded` |
| `4`, `8` | `False` | `INFEASIBLE` | `legged_sqp_infeasible` |
| `1`, `2`, `3`, `5`, `6`, `7` | `False` | `NUMERIC_FAILURE` | `legged_sqp_numeric_contract_failed` |

An unavailable sealed backend maps to `BACKEND_UNAVAILABLE/legged_body_profile_unsupported`. A global deadline exception or expiry maps to `DEADLINE_EXPIRED/planning_deadline_expired`; the typed reserved-tail/evaluation cutoff or terminal SciPy iteration-limit status `9` maps to `RESOURCE_EXHAUSTED/legged_resource_budget_exceeded`. Terminal status `1` is illegal reverse-communication leakage, not evidence of budget exhaustion, and maps to numeric failure. Any non-exact status/success type、status outside `0..9`、`success != (status == 0)`、nonfinite `x/fun/jac`、wrong vector length、reported/authoritative count overrun or status-0 hard residual above `profile.hard_constraint_tolerance` is the defined “status drift” and maps to `NUMERIC_FAILURE/legged_sqp_numeric_contract_failed`. Recheck global deadline then solver cutoff, independently recompute every hard residual, then call the Task 6 materializer exactly once with a checkpoint that preserves the sealed complete tail before/after every segment; a cutoff abort maps to resource exhaustion and publishes no candidate. Only the fully materialized canonical object may populate the feasible result.

- [ ] **Step 6: Implement L2-tail reserve and one repair constraint**

Freeze `LeggedBodyL2ReserveModelV1` as `legged_body_l2_reserve_model/v1`; despite the retained v1 name it reserves the complete mandatory post-solver encode → candidate-risk-feature/fallback aggregation → L2 tail, with these audit constants:

```text
TAIL_BASE_S=0.002
TAIL_PER_INTERVAL_S=2.0e-7
TAIL_PER_CELL_S=5.0e-8
TAIL_PER_ENCODED_SCALAR_S=1.0e-7
TAIL_SNAPSHOT_HASH_PASS_COUNT=2
TAIL_PER_SNAPSHOT_CELL_PASS_S=2.0e-8
TAIL_PER_SNAPSHOT_BYTE_PASS_S=1.0e-10
TAIL_RISK_BASE_S=0.001
TAIL_RISK_PER_POSE_SAMPLE_S=1.0e-7
TAIL_RISK_PER_CELL_S=2.0e-8
TAIL_BASE_BYTES=1_048_576
TAIL_INTERVAL_BYTES=48
TAIL_STREAMING_CELL_BYTES=16
TAIL_STREAMING_CELL_CAP=65_536
TAIL_SNAPSHOT_HASH_SCRATCH_BYTES=SNAPSHOT_HASH_SCRATCH_BYTES_V2
TAIL_CANDIDATE_OBJECT_BASE_BYTES=4_096
TAIL_CANDIDATE_OBJECT_PER_SCALAR_BYTES=64
TAIL_ENCODED_CANDIDATE_BASE_BYTES=4_096
TAIL_ENCODED_SCALAR_BYTES=16
TAIL_L2_DECODED_OBJECT_BASE_BYTES=4_096
TAIL_L2_DECODED_OBJECT_PER_SCALAR_BYTES=64
TAIL_RISK_POSE_SAMPLE_BYTES=32
TAIL_RISK_STREAMING_CELL_BYTES=16
TAIL_RISK_STREAMING_CELL_CAP=65_536
```

`assess(profile, *, snapshot_cell_count, snapshot_canonical_byte_count, segment_count, risk_segment_count: int | None = None, encoded_scalar_count)` first sets `effective_risk_segment_count=segment_count if risk_segment_count is None else risk_segment_count`, validates all bounded counts, invokes the single Task 3 `risk_fallback_aggregation_reserve_v1(effective_risk_segment_count)`, then uses the profile's worst-case L2 interval/cell caps、worst risk-feature pose/cell caps、actual snapshot/segment/risk-segment/encoded-scalar counts and outward `nextafter(...,+inf)` arithmetic. It returns separately sealed `encode_time_s`、`risk_feature_time_s` (including that count-scaled fallback/aggregation reserve)、`l2_time_s`、`tail_time_s=sum(...)` and byte fields `encode_scratch_bytes`、`solver_candidate_bytes`、`encoded_candidate_bytes`、`decoded_candidate_bytes`、`candidate_bytes`、`risk_feature_bytes`、`l2_bytes`、`tail_bytes=sum(...)`. Freeze `solver_candidate_bytes=TAIL_CANDIDATE_OBJECT_BASE_BYTES+encoded_scalar_count*TAIL_CANDIDATE_OBJECT_PER_SCALAR_BYTES`、`encoded_candidate_bytes=TAIL_ENCODED_CANDIDATE_BASE_BYTES+encoded_scalar_count*TAIL_ENCODED_SCALAR_BYTES`、`decoded_candidate_bytes=solver_candidate_bytes` and `candidate_bytes=sum(the three)`；the conservative v1 peak intentionally admits all three simultaneous representations rather than relying on Python garbage collection. The L2 byte field separately includes `TAIL_L2_DECODED_OBJECT_BASE_BYTES+encoded_scalar_count*TAIL_L2_DECODED_OBJECT_PER_SCALAR_BYTES` for its independent fresh decode. `l2_time_s` explicitly includes exactly two deadline-aware snapshot hash passes as `TAIL_SNAPSHOT_HASH_PASS_COUNT*(snapshot_cell_count*TAIL_PER_SNAPSHOT_CELL_PASS_S+snapshot_canonical_byte_count*TAIL_PER_SNAPSHOT_BYTE_PASS_S)`; L2 bytes include one bounded `TAIL_SNAPSHOT_HASH_SCRATCH_BYTES` buffer because passes are serial, never the caller-owned snapshot bytes. Tail bytes otherwise are base + worst L2 intervals + both streaming-cell caps + risk pose/output records + count-scaled fallback bytes + all explicit candidate representations. `assess_risk_only(profile, *, risk_segment_count)` returns the identical risk-feature time/byte component for corridor admission, so 48-segment candidates and 2048-edge corridors cannot diverge in reserve math.

`admit()` consumes the exact request memory ledger. With a nonzero ceiling, its currently available bytes after retained hierarchy/corridor records must be strictly greater than `profile.solver_memory_reservation_bytes + tail_bytes`; time admission requires `deadline.remaining_s > tail_time_s`. It atomically reserves one complete solver+tail token before problem/backend allocation. Before backend work it splits out the sealed tail; the materializer fills that tail's pre-reserved solver-candidate child. On feasible completion it releases only solver scratch and returns the live `solver_candidate_token` plus the remaining `post_solver_tail_token` in `LeggedBodySQPOptimizationResultV1`; their exact sum is `tail_bytes`. Failures release every provisional token. `split_post_solver_tail_v1()` accepts both handles and creates exact encode-scratch、encoded-bytes、decoded-object、risk-feature/output and L2 children while preserving the existing solver-candidate child, so provider phases never reserve the same tail twice or infer ownership from ledger state. All three candidate representations remain explicitly accounted until L2 merges/releases them; retaining conservative capacity is required even if a local reference is dropped early. Stale/wrong-request/wrong-byte tokens are typed internal identity failures and are released fail-closed. `max_memory_bytes==0` means no ceiling but retains overflow-safe accounting. The backend receives `solver_cutoff = global_deadline - tail_time_s` and cannot run into the mandatory feature/L2 tail. Candidate risk worker admission, which happens after feature extraction, preserves at least the sealed `l2_time_s`; corridor risk calls use `assess_risk_only()` for their own feature phase while preserving the worst complete candidate post-solver `tail_time_s`. Starting initial solve or repair at equality fails before backend invocation. Feature work that reaches its count/time/memory phase cutoff takes the Task 3 all-max fallback path rather than borrowing L2 reserve.

A repair constraint is one exact `LeggedBodyRepairConstraintV1` bound to the failed complete validation-input/request/profile/controller/snapshot/corridor/rolling-goal/risk/source-candidate/counterexample/validation-record lineage and normalized segment-phase interval. Freeze `REPAIR_PHASE_SAMPLE_COUNT_V1=5`; phases are exactly `p_i=p0+(p1-p0)*i/4` for `i=0..4`. At each phase, replay the affected analytic segment and add, for every Task 7 support-lattice point, the same branch-ordered point-to-square inequality against the exact failing cell with required distance equal to the normal SQP clearance threshold plus `profile.repair_clearance_m`. Its analytic Jacobian uses the same replay and distance branches. The repair problem retains byte-identical base bounds/equalities/inequalities, corridor assignments, merit and resource limits; it only appends these `5*support_point_count` inequalities. Those extra records are included in the combined count、Jacobian bytes、solver reservation and shared request-memory admission described above; a base problem at the cap cannot bypass it. Cell-less、zero-width/out-of-range phase、nonlocalized、identity、numeric、kinematic、resource or deadline failures cannot create a repair. The solver verifies every constraint/hash, requires the repaired output to carry that exact constraint hash, and rejects any removed/relaxed base constraint or second repair attempt.

- [ ] **Step 7: Run GREEN and focused numerical suite**

Install/verify the explicit numerical layer in the existing D-drive environment; keep the pip cache on D:

```powershell
$env:PIP_CACHE_DIR = "D:/CodexDownloads/pip-cache"
D:/conda_envs/lunar-explorer/python.exe -m pip install -e ".[dev,legged-sqp]"
D:/conda_envs/lunar-explorer/python.exe -c "import scipy; assert scipy.__version__ == '1.18.0'"
```

The ordinary package import and non-numerical contract suite must remain runnable without importing SciPy; the `BACKEND_UNAVAILABLE` test injects that environment deterministically. Real straight/turn/conformance/formal success tests require the verified extra and must fail preflight rather than silently skip when the formal runner is used.

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_legged_body_sqp_solver.py tests/test_v2_legged_body_kinematics.py tests/test_v2_legged_body_sqp_initialization.py -q
```

Expected: feasible/infeasible/iteration/evaluation/memory/numeric/backend/deadline/repair cases all map to exact statuses and reasons.

- [ ] **Step 8: Review and commit Task 7**

```powershell
git -C path-planner diff --check
git -C path-planner add src/path_planner/v2/legged_body_sqp_solver.py tests/test_v2_legged_body_sqp_solver.py
git -C path-planner commit -m "feat: add deterministic legged body SQP solver"
git add path-planner
git commit -m "build: integrate legged body SQP solver"
```

---

### Task 8: Independently Validate the Canonical Continuous Body Envelope and Produce the First L2 Counterexample

**Files:**
- Create: `path-planner/src/path_planner/v2/legged_body_sqp_validation.py`
- Create: `path-planner/tests/test_v2_legged_body_sqp_validation.py`
- Modify: `path-planner/tests/test_v2_route_validation.py`
- Modify: `path-planner/tests/test_v2_legged_oracle.py`

**Interfaces:**
- Consumes: canonical candidate bytes、request/profile/hard view、exact corridor/rolling subgoal、risk evidence、deadline、the exact request memory ledger、correctly owned pre-reserved L2/candidate-bytes/risk-evidence tokens and resource caps。
- Produces: `validate_legged_body_trajectory_l2_v1(candidate_bytes, request, profile, view, corridor, rolling_goal, risk_evidence, deadline, *, memory_ledger, memory_token, candidate_representation_tokens, risk_evidence_token, repair_constraint=None, prior_l2_failure=None, prior_candidate=None, prior_risk=None, repair_context_token=None) -> LeggedBodyTrajectoryL2ResultV1`、`LeggedBodyL2CounterexampleV1`、passed receipt and exact `repair_constraint_from_legged_counterexample_v1(...) -> LeggedRepairPreparationV1(constraint,memory_token)`；`candidate_representation_tokens` is the exact ordered internal bundle of solver-object、encoded-bytes and provider-fresh-decoded handles。

- [ ] **Step 1: Write RED adversarial continuous-sweep tests**

```python
def test_safe_nodes_but_colliding_between_nodes_is_rejected() -> None:
    result = validate_case("thin_mid_segment_obstacle")
    assert result.passed is False
    assert result.counterexample.reason_code == "terrain_hard_obstacle"
    assert result.counterexample.segment_index == 0


def test_rotating_body_corner_collision_is_rejected() -> None:
    result = validate_case("rotation_corner_clip")
    assert result.passed is False
    assert result.counterexample.cell == EXPECTED_CORNER_CELL


def test_boundary_touch_unknown_and_low_confidence_fail_closed() -> None:
    assert [validate_case(name).counterexample.reason_code for name in CASES] == [
        "terrain_out_of_bounds",
        "terrain_unknown",
        "terrain_confidence_below_profile",
    ]
```

- [ ] **Step 2: Write RED identity, canonicalization, caps, and counterexample-order tests**

```python
def test_validator_decodes_fresh_candidate_and_rejects_hash_drift() -> None:
    result = validate_legged_body_trajectory_l2_v1(MUTATED_BYTES, REQUEST, PROFILE, VIEW, CORRIDOR, SUBGOAL, RISK, DEADLINE, memory_ledger=MEMORY_LEDGER, memory_token=L2_TOKEN, candidate_representation_tokens=REPRESENTATION_TOKENS, risk_evidence_token=RISK_TOKEN)
    assert result.passed is False
    assert result.counterexample.reason_code == "legged_identity_mismatch"


@pytest.mark.parametrize("mode", ["missing_constraint", "wrong_constraint", "stale_counterexample"])
def test_repaired_candidate_requires_exact_prior_constraint_lineage(mode: str) -> None:
    result = validate_repaired_case(mode)
    assert result.passed is False
    assert result.counterexample.reason_code == "legged_identity_mismatch"


def test_first_counterexample_is_stable_by_segment_time_reason_and_cell() -> None:
    results = [validate_case("multiple_collisions").counterexample for _ in range(5)]
    assert len(set(results)) == 1
    assert (
        results[0].segment_index,
        results[0].segment_phase_start,
        results[0].segment_phase_end,
        results[0].reason_code,
        results[0].cell,
    ) == (0, 0.25, 0.375, "terrain_unknown", Cell(4, 3))


def test_raw_candidate_safe_but_decimal12_candidate_unsafe_is_rejected() -> None:
    result = validate_case("canonicalization_moves_corner_into_obstacle")
    assert result.passed is False
    assert result.counterexample.reason_code == "terrain_hard_obstacle"
    assert result.counterexample.cell == CANONICALIZED_COLLISION_CELL


@pytest.mark.parametrize("cap", ["interval", "cell", "memory"])
def test_l2_cap_exhaustion_never_returns_pass(cap: str) -> None:
    result = validate_with_tiny_cap(cap)
    assert result.passed is False
    assert result.counterexample.reason_code == "legged_resource_budget_exceeded"


@pytest.mark.parametrize("mutation", ["duration_below_min", "duration_above_max", "segment_count_over_profile"])
def test_l2_independently_rejects_duration_and_segment_cap_drift(mutation: str) -> None:
    result = validate_rehashed_candidate_mutation(mutation)
    assert result.passed is False
    assert result.counterexample.reason_code == "legged_kinematic_constraint_failed"


def test_l2_rejects_nonzero_terminal_speed_at_mission_subgoal() -> None:
    result = validate_rehashed_candidate_mutation("mission_terminal_nonzero")
    assert result.passed is False
    assert result.counterexample.reason_code == "legged_kinematic_constraint_failed"


def test_validated_costs_are_recomputed_and_public_tamper_is_rejected() -> None:
    result = validate_case("safe_turn")
    assert result.receipt.validated_relative_energy == EXPECTED_RECOMPUTED_ENERGY
    assert result.receipt.cost_breakdown_hash == legged_cost_breakdown_hash_v1(EXPECTED_COST)
    assert public_postcondition(rehash_tampered_energy_and_cost(result.trajectory)) is False


def test_curved_segment_distance_is_analytic_arc_not_endpoint_chord() -> None:
    result = validate_case("constant_speed_half_circle")
    segment = result.trajectory.segments[0]
    chord = hypot(segment.end_state.x_m-segment.start_state.x_m, segment.end_state.y_m-segment.start_state.y_m)
    expected_arc = canonicalize_legged_body_scalar_v1(0.5*(abs(segment.start_state.v_mps)+abs(segment.end_state.v_mps))*segment.duration_s)
    assert segment.distance_m == expected_arc
    assert segment.distance_m > chord
    assert result.receipt.validated_distance_m == expected_arc


def test_prior_failure_risk_lineage_cannot_be_rehashed_into_repair_authority() -> None:
    failed = validate_case("localized_collision")
    forged = rehash_with_different_risk_evidence(failed)
    with pytest.raises(ValueError, match="validation input"):
        repair_constraint_from_legged_counterexample_v1(forged, candidate=CANDIDATE, corridor=CORRIDOR, rolling_goal=SUBGOAL, risk=RISK, view=VIEW, profile=PROFILE, memory_ledger=MEMORY_LEDGER, repair_context_token=REPAIR_CONTEXT_TOKEN)


@pytest.mark.parametrize("phase", ["mid_replay", "mid_subdivision", "after_last_interval_before_receipt"])
def test_l2_deadline_expiry_never_reaches_receipt_or_repair(phase: str) -> None:
    result = validate_with_clock_expiring_at(phase)
    assert result.passed is False
    assert result.counterexample.reason_code == "planning_deadline_expired"
    assert result.counterexample.repairable is False


def test_subdivision_stops_only_at_frozen_depth_and_returns_dyadic_phase() -> None:
    result = validate_case("depth_boundary_collision", max_subdivision_depth=3)
    assert (result.counterexample.segment_phase_start, result.counterexample.segment_phase_end) == (0.375, 0.5)
```

- [ ] **Step 3: Run RED tests**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_legged_body_sqp_validation.py tests/test_v2_route_validation.py tests/test_v2_legged_oracle.py -q -x
```

Expected: only new validation imports fail; old static legged oracle remains green.

- [ ] **Step 4: Pin identities, decode, and independently replay dynamics**

At entry, capture exact request/profile/controller/snapshot/corridor/rolling-goal/risk/candidate tokens, validate the exact pre-reserved L2、all three ordered candidate-representation handles and retained risk-evidence handle without recharging, build `validation_input_hash` from those identities plus `LEGGED_BODY_L2_VALIDATOR_V1`, and check the shared deadline. Decode candidate bytes to one additional fresh object inside the pre-reserved L2 decode capacity; require `1<=segment_count<=profile.max_segments`, then replay every segment with `integrate_legged_body_segment_v1()` while checking the deadline before/after each segment. Verify exact request-start equality、start continuity、`profile.min_segment_duration_s<=duration_s<=profile.max_segment_duration_s`、strictly increasing canonical time、direction-aware acceleration bounds、speed/reverse limits、absolute yaw-rate limits、the exact adjacent-pair yaw-slew formula from Task 7、Euclidean/wrapped rolling-goal tolerances from the nested `PlatformProfileV2`、all candidate/corridor/subgoal/risk hashes and exact mission terminal `v==0.0` when `terminal_speed_is_hard`. Every success/failure record and counterexample carries that same input hash and full rolling-goal/risk lineage. Recompute the deadline-checked snapshot raw-layer hash before and after terrain work. Do not call solver constraint functions or trust its collision samples. The absence of a pre-horizon yaw-rate field is recorded as the same declared controller-handoff boundary, not silently treated as zero. A success merges/transfers all three representation handles plus risk and L2 handles into one `promotion_token`; a localized repairable failure merges/transfers those same candidate/risk capacities plus exact failed-record bytes from the L2 child into one `repair_context_token` and releases unused L2 scratch; every nonrepairable failure releases all handles. No live prior object exists outside an accounting token.

Independently recompute each segment's traveled distance as the analytic integral `integral_0^dt abs(v0+a*t) dt`. The already rechecked no-sign-crossing constraint makes this exactly `0.5*(abs(v_start)+abs(v_end))*duration_s`; pure zero-speed rotation is zero. Never substitute endpoint Euclidean chord length. Recompute `legged_body_relative_energy_v1()`、duration and control slew (`abs(delta_a)/max_accel_decel + abs(delta_omega)/max_yaw_rate`, first segment zero) in candidate order. Canonicalize each segment scalar to decimal12, sum those canonical segment values in order, then canonicalize each total. Construct the only allowed `CostBreakdownV2` as `distance_weight*distance`、`risk_weight*cumulative_route_risk`、`energy_weight*relative_energy`、`time_weight*duration`, canonicalize components, and hash it with `legged_cost_breakdown_hash_v1()`. Promotion never accepts caller/solver segment distance/energy or public cost fields; it creates every `TimedBodySegmentV1.distance_m/relative_energy` and the receipt/plan cost only from these L2-validated totals.

- [ ] **Step 5: Implement conservative interval subdivision**

For interval `[t0,t1]`, evaluate exact endpoint/midpoint body states. Include the safety margin in an outward-rounded body radius, then bound every body point's motion by:

```python
half_length_m = nextafter(0.5 * profile.body_length_m + profile.safety_margin_m, inf)
half_width_m = nextafter(0.5 * profile.body_width_m + profile.safety_margin_m, inf)
body_radius_m = nextafter(hypot(half_length_m, half_width_m), inf)
point_speed_upper = nextafter(max(abs(v0), abs(vmid), abs(v1)) + body_radius_m * abs(omega_radps), inf)
expansion_m = nextafter(point_speed_upper * (t1 - t0) / 2.0, inf)
```

Enumerate the midpoint rectangle with `conservative_body_pose_cells(mid_pose, view.snapshot.geometry, length_m=profile.body_length_m, width_m=profile.body_width_m, safety_margin_m=profile.safety_margin_m, expansion_m=expansion_m, checkpoint=deadline_checkpoint, max_candidate_cells=profile.max_body_pose_cells)`. Check deadline before each segment、before each interval pop、before every 256-cell batch、after each helper return and after the final interval. If every cell passes, the interval is proven safe. If any cell fails and `depth<profile.max_l2_subdivision_depth`, bisect deterministically left then right; there is no implementation-chosen minimum time/phase width. At the exact depth limit emit the sorted first failing `(segment,time_start,time_end,reason,cell)`; if binary64 cannot represent a strict midpoint earlier, fail conservatively with a nonrepairable numeric counterexample. Segment phases are exact dyadic fractions derived from integer `(depth,index)` and only then decimal12 encoded. Count every interval/cell before processing and fail before exceeding caps. Deadline expiry returns a nonrepairable `planning_deadline_expired` record and takes precedence over cap/collision evidence.

- [ ] **Step 6: Grant L2 only after all bindings pass**

```python
receipt = LeggedBodyL2ReceiptV1(
    validator_id=LEGGED_BODY_L2_VALIDATOR_V1,
    passed=True,
    validation_input_hash=validation_input_hash,
    candidate_hash=candidate.candidate_hash,
    request_hash=candidate.request_hash,
    profile_hash=candidate.profile_hash,
    controller_capability_hash=candidate.controller_capability_hash,
    terrain_snapshot_hash=candidate.terrain_snapshot_hash,
    rolling_goal_hash=rolling_goal.rolling_goal_hash,
    risk_evidence_hash=risk_evidence_hash_v1(risk),
    checked_interval_count=interval_count,
    checked_cell_count=cell_count,
    validated_distance_m=validated.distance_m,
    validated_relative_energy=validated.relative_energy,
    validated_duration_s=validated.duration_s,
    validated_control_slew=validated.control_slew,
    cost_breakdown_hash=legged_cost_breakdown_hash_v1(validated.cost_breakdown),
    repair_applied=candidate.repair_applied,
)
```

Before building totals/receipt, check the deadline again and require the candidate's repair pair to be exactly `(False,None)` or `(True,<valid SHA-256>)`. An initial candidate requires `repair_constraint/prior_l2_failure/prior_candidate/prior_risk/repair_context_token` all `None`. A repaired candidate requires all five exact objects/handle keyword-only. Independently recompute the prior `validation_input_hash` from the supplied current request/profile/controller/snapshot/corridor/rolling-goal/risk objects and prior source candidate, then recompute `prior_l2_failure.validation_record_hash`; require `passed=False`、no receipt/trajectory、one self-consistent counterexample、the same live context token and exact equality of every full-lineage field/hash before trusting its localized interval. Then require the constraint hash to equal `candidate.repair_constraint_hash`. `repair_constraint_from_legged_counterexample_v1(failed_validation, *, candidate, corridor, rolling_goal, risk, view, profile, memory_ledger, repair_context_token)` accepts the whole exact failed record plus every independent input and live ownership handle needed to recompute both hashes; it derives source hashes itself and accepts only a localized terrain cell/segment-phase counterexample. Before materializing the constraint it requires that the sealed context capacity includes `REPAIR_CONSTRAINT_BASE_BYTES_V1+REPAIR_CONSTRAINT_CELL_BYTES_V1`, transfers the whole handle to owner `repair_context_with_constraint`, and returns `LeggedRepairPreparationV1(constraint,new_token)`, making the input handle stale without charging again. Substituting and rehashing prior risk or rolling-goal evidence is rejected. Identity/numeric/kinematic/resource/deadline failures are nonrepairable. Counterexample absolute times remain audit evidence, while the normalized phase interval is the only interval transferred into a re-solved variable-duration problem.

Repaired-validation ownership is single-use and explicit: after the prior failure/constraint lineage is fully revalidated, a pass releases the old `repair_context_with_constraint` handle before returning the new-candidate promotion token；any repaired failure (including a geometrically repairable-looking second counterexample) is final, releases both the old context and every new candidate/risk/L2 handle, and returns no repair-context token. Expiry or exception before that terminal transfer leaves both handle sets registered in the same attempt scope for cleanup. Thus the authorized one-repair lineage is audited but never retained in the public plan, and no old context can be spent twice.

After all intervals, recompute snapshot/cost/record hashes, check deadline immediately before receipt creation and again before `promote_legged_body_trajectory_v1()`. Promotion creates the public segments/cost from validated values and then seals `validation_record_hash`; expiry at either checkpoint returns timeout and destroys the provisional receipt. Risk/model output never changes an interval pass/fail.

- [ ] **Step 7: Run GREEN and old-oracle regressions**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_legged_body_sqp_validation.py tests/test_v2_route_validation.py tests/test_v2_legged_oracle.py tests/test_v2_legged_provider.py -q
```

Expected: all pass; old static crawl validator IDs and decisions are unchanged.

- [ ] **Step 8: Review and commit Task 8**

```powershell
git -C path-planner diff --check
git -C path-planner add src/path_planner/v2/legged_body_sqp_validation.py tests/test_v2_legged_body_sqp_validation.py tests/test_v2_route_validation.py tests/test_v2_legged_oracle.py
git -C path-planner commit -m "feat: add continuous legged body trajectory L2"
git add path-planner
git commit -m "build: integrate legged body trajectory L2"
```

---

### Task 9: Orchestrate Corridors, SQP, Risk, L2, One Repair, and First-Pass Return

**Files:**
- Create: `path-planner/src/path_planner/v2/providers/legged_rolling.py`
- Modify: `path-planner/src/path_planner/v2/providers/__init__.py`
- Create: `path-planner/tests/test_v2_legged_rolling_provider.py`

**Interfaces:**
- Consumes: audited specialized profile、optional prestarted bounded risk inference runner、optional derived-map cache and a frozen `LeggedRollingTrustedOpsV1` containing the exact Task 2–8 callables；raw model objects never enter the planner thread。
- Produces: `LeggedRollingBodySQPProviderV1.create(...)` which preloads/seals backend and optional risk worker before requests, exact read-only `.profile/.backend/.risk_runner/.derived_map_cache/.trusted_ops` fields, and internal `plan(request, deadline, *, memory_ledger, output_escrow) -> LeggedProviderResultV1` with exact failure taxonomy/telemetry and API-owned ledger/escrow identity；only Task 10 claims, unwraps and publicizes the public outcome。

- [ ] **Step 1: Write RED state-machine and first-pass tests**

```python
def test_provider_returns_first_l2_success_and_never_runs_later_corridors(monkeypatch) -> None:
    calls: list[str] = []
    provider = make_provider_with_scripted_corridors(calls, outcomes=("l2_fail", "l2_pass", "forbidden"))
    provider_result = provider.plan(REQUEST, DEADLINE, memory_ledger=API_OWNED_MEMORY_LEDGER, output_escrow=API_OWNED_OUTPUT_ESCROW)
    outcome = provider_result.outcome
    assert type(outcome) is LeggedRollingPlanV1
    assert provider_result.output_token is not None
    assert outcome.global_corridor_intent.corridor_index == 1
    assert calls == ["corridor:0", "corridor:1"]
    assert outcome.selection_policy == "first_l2_in_order/v1"
    assert outcome.global_optimality_claimed is False


@pytest.mark.parametrize("first_failure", ["subgoal_unavailable", "initialization_failed", "sqp_infeasible"])
def test_corridor_local_failure_advances_to_next_corridor(first_failure: str) -> None:
    outcome = run_scripted_provider((first_failure, "l2_pass"))
    assert type(outcome) is LeggedRollingPlanV1
    assert outcome.global_corridor_intent.corridor_index == 1


def test_each_corridor_gets_at_most_one_localized_repair() -> None:
    outcome, telemetry = run_scripted_provider(("localized_fail", "localized_fail_again"))
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "legged_repair_l2_rejected"
    assert telemetry.repair_attempt_count == 1
    assert telemetry.l2_attempt_count == 2


def test_provider_never_returns_partial_candidate() -> None:
    outcome = run_scripted_provider(("sqp_partial",))
    assert type(outcome) is PlanningFailureV2
    assert not hasattr(outcome, "route")


def test_provider_uses_only_its_sealed_runner_cache_and_trusted_ops() -> None:
    provider = make_provider()
    assert provider.risk_runner is EXPECTED_RUNNER
    assert provider.derived_map_cache is EXPECTED_CACHE
    assert provider.trusted_ops is EXPECTED_TRUSTED_OPS
    assert provider.provider_token == legged_rolling_provider_token_v1(provider)
```

- [ ] **Step 2: Write RED risk isolation, mission-complete, and failure-precedence tests**

```python
def test_model_toggle_can_change_corridor_order_but_not_hard_or_l2_answers() -> None:
    # This scripted fixture makes every corridor fail L2, so early-return cannot
    # hide a candidate from either side of the paired isolation comparison.
    fallback = run_provider_all_l2_fail(model=None)
    learned = run_provider_all_l2_fail(model=DETERMINISTIC_MODEL)
    assert fallback.audit.hard_feasibility_digest == learned.audit.hard_feasibility_digest
    assert fallback.audit.l2_by_risk_neutral_geometry == learned.audit.l2_by_risk_neutral_geometry


def test_mission_complete_only_for_true_final_goal() -> None:
    intermediate = run_success(far_goal=True)
    final = run_success(far_goal=False)
    assert intermediate.mission_complete is False
    assert final.mission_complete is True
    assert final.rolling_goal_state == final.mission_goal_state
    assert final.local_body_trajectory.segments[-1].end_state.v_mps == 0.0


def test_deadline_beats_late_success_and_internal_failure() -> None:
    assert run_with_late_solver_success().reason_code == "planning_deadline_expired"
    assert run_with_exception_after_expiry().reason_code == "planning_deadline_expired"


def test_expiry_after_l2_before_decision_hash_cannot_publish_success() -> None:
    outcome, audit = run_with_clock_expiring_during_plan_build()
    assert outcome.reason_code == "planning_deadline_expired"
    assert audit.decision_hash_published is False


def test_provider_expiry_after_plan_build_before_scope_detach_reclaims_output() -> None:
    provider_result, audit = run_provider_with_clock_expiring_at_final_detach_checkpoint()
    assert provider_result.outcome.reason_code == "planning_deadline_expired"
    assert provider_result.output_token is None
    assert audit.memory_live_bytes_after_provider_scope == 0


@pytest.mark.parametrize("speed", [nextafter(PROFILE.max_forward_speed_mps, inf), -0.1])
def test_invalid_initial_motion_is_unsafe_start_not_sqp_infeasible(speed: float) -> None:
    outcome = run_request(current_speed_mps=speed, reverse_enabled=False)
    assert outcome.reason_code == "legged_start_invalid"
    assert outcome.category is FailureCategoryV2.UNSAFE_START


@pytest.mark.parametrize("phase", ["before_encode", "before_risk_features", "before_l2"])
def test_post_solver_stage_reserve_equality_stops_before_next_phase(phase: str) -> None:
    outcome, audit = run_with_clock_at_exact_phase_reserve(phase)
    assert outcome.reason_code == "legged_resource_budget_exceeded"
    assert audit.phase_started[phase] is False


def test_encode_overrun_stops_before_risk_and_l2() -> None:
    outcome, audit = run_with_slow_codec_crossing_encode_cutoff()
    assert outcome.reason_code == "legged_resource_budget_exceeded"
    assert audit.risk_feature_started is False
    assert audit.l2_started is False


@pytest.mark.parametrize(
    ("fault", "expected_reason"),
    [
        ("corridor_cap", "legged_corridor_budget_exceeded"),
        ("sqp_resource_cutoff", "legged_resource_budget_exceeded"),
    ],
)
def test_provider_preserves_corridor_vs_sqp_resource_taxonomy(fault: str, expected_reason: str) -> None:
    result = run_provider_with_injected_resource_fault(fault)
    assert result.outcome.reason_code == expected_reason
    assert result.outcome.category is FailureCategoryV2.RESOURCE_LIMIT


def test_feature_overrun_falls_back_and_preserves_l2_start_reserve() -> None:
    outcome, audit = run_with_slow_under_cap_risk_features()
    assert type(outcome) is LeggedRollingPlanV1
    assert outcome.risk_evidence.fallback_reason == "risk_feature_phase_overrun"
    assert audit.remaining_before_l2_s > audit.l2_reserve_s


@pytest.mark.parametrize("phase", ["encode_admission", "risk_admission", "l2_admission"])
def test_failed_first_corridor_scope_releases_tail_before_second_corridor(phase: str) -> None:
    outcome, audit = run_two_corridors(first_fails_at=phase, second="l2_pass")
    assert type(outcome) is LeggedRollingPlanV1
    assert audit.live_bytes_before_second == audit.retained_request_baseline_bytes
    assert audit.stale_token_count == 0


@pytest.mark.parametrize("branch", ["hierarchy_resource", "corridor_deadline", "memory_identity", "codec_identity", "risk_resource", "l2_nonrepairable", "repair_final_failure"])
def test_request_scope_maps_typed_failures_and_closes_every_token(branch: str) -> None:
    outcome, audit = run_scoped_failure_branch(branch)
    assert outcome.reason_code == EXPECTED_REASON_BY_BRANCH[branch]
    assert audit.memory_live_bytes_after_scope == 0
    assert audit.double_release_count == 0


def test_provider_snapshot_rehash_scratch_is_charged_beside_api_postcondition_reserve() -> None:
    admitted, admitted_audit = run_with_provider_hash_memory_limit(API_POSTCONDITION_RESERVE_BYTES + SNAPSHOT_HASH_SCRATCH_BYTES_V2)
    assert admitted_audit.provider_hash_scratch_ledger_id == admitted_audit.api_postcondition_ledger_id
    assert admitted_audit.peak_bytes >= API_POSTCONDITION_RESERVE_BYTES + SNAPSHOT_HASH_SCRATCH_BYTES_V2
    rejected, rejected_audit = run_with_provider_hash_memory_limit(API_POSTCONDITION_RESERVE_BYTES + SNAPSHOT_HASH_SCRATCH_BYTES_V2 - 1)
    assert rejected.outcome.reason_code == "legged_resource_budget_exceeded"
    assert rejected_audit.hard_view_hash_started is False
```

- [ ] **Step 3: Run RED tests**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_legged_rolling_provider.py -q -x
```

Expected: provider module is absent.

- [ ] **Step 4: Freeze the exact call-argument catalog, then implement one scoped serial state machine**

The next fenced block is a non-executable call-argument catalog only: it freezes complete positional/keyword arguments、phase order and branch reasons that the sole scoped state machine below must splice into its named steps. Its `try/for/return` layout is not a second control-flow implementation and must not be copied as code；where presentation shorthand conflicts with the scoped state machine, the scoped state machine is authoritative.

```text
ops = self.trusted_ops
try:
    view = ops.build_hard_safety_view(request.terrain_snapshot, profile, deadline, memory_ledger=memory_ledger, hash_scratch_token=provider_hash_scratch)
except LeggedMapProfileMismatchV1:
    return _provider_failure_v1(request, profile, "legged_map_profile_mismatch", telemetry)
request_hash = ops.request_hash(request, terrain_snapshot_hash=view.snapshot_hash)
memory_ledger = ops.require_injected_memory_ledger(memory_ledger, request_hash, ops.profile_hash(profile), request.resource_budget)
risk_ledger = LeggedRiskInferenceLedgerV1.create(request_hash, profile)
start_check = view.validate_body_pose(request.current_body_state, deadline, memory_ledger=memory_ledger)
if not start_check.passed:
    return _provider_failure_v1(request, profile, "legged_start_invalid", telemetry, start_check)
start_motion_check = ops.validate_initial_motion(request.current_body_state, profile)
if not start_motion_check.passed:
    return _provider_failure_v1(request, profile, "legged_start_invalid", telemetry, start_motion_check)
goal_body_state = LeggedBodyStateV1(
    request.mission_goal_state.x_m,
    request.mission_goal_state.y_m,
    request.mission_goal_state.heading_rad,
    0.0,
)
goal_check = view.validate_body_pose(goal_body_state, deadline, memory_ledger=memory_ledger)
if not goal_check.passed:
    return _provider_failure_v1(request, profile, "legged_goal_invalid", telemetry, goal_check)
hierarchy = ops.build_hierarchy(
    view,
    deadline=deadline,
    memory_ledger=memory_ledger,
    cache=self.derived_map_cache,
)
worst_post_solver_reserve = self.reserve_model.assess(
    profile,
    snapshot_cell_count=view.snapshot_cell_count,
    snapshot_canonical_byte_count=view.snapshot_canonical_byte_count,
    segment_count=profile.max_segments,
    risk_segment_count=profile.max_segments,
    encoded_scalar_count=encoded_scalar_count_for_segments_v1(profile.max_segments),
)
worst_corridor_risk_reserve = self.reserve_model.assess_risk_only(
    profile,
    risk_segment_count=profile.max_corridor_risk_segments,
)
corridor_bundle = ops.generate_corridors(
    view,
    hierarchy,
    request.current_body_state,
    request.mission_goal_state,
    profile,
    self.risk_runner,
    risk_ledger,
    memory_ledger,
    worst_corridor_risk_reserve.risk_feature_time_s,
    worst_post_solver_reserve.tail_time_s,
    request.resource_budget,
    deadline,
)
corridors = corridor_bundle.intents
provider_scope.register_all(corridor_bundle.retained_memory_tokens)
if not corridors:
    return _provider_failure_v1(request, profile, "legged_no_global_corridor", telemetry)

# The following phase-order sketch is the body of scoped `_attempt_one_corridor_v1`;
# the executable caller/ownership operations are frozen immediately after this block.
for corridor, corridor_memory_token in zip(corridors, corridor_bundle.retained_memory_tokens, strict=True):
    if deadline.expired:
        raise LeggedPlanningDeadlineExpiredV1("planning deadline expired")
    try:
        subgoal = ops.select_rolling_subgoal(request.current_body_state, request.mission_goal_state, corridor, view, profile, deadline, memory_ledger=memory_ledger)
    except LeggedLocalSubgoalUnavailableV1 as exc:
        _require_open_legged_deadline_v1(deadline)
        record_corridor_rejection("legged_local_subgoal_unavailable", exc)
        continue
    try:
        initial = ops.initialize_sqp(request, profile, corridor, subgoal, deadline, memory_ledger=memory_ledger)
    except LeggedSQPInitializationFailedV1 as exc:
        _require_open_legged_deadline_v1(deadline)
        record_corridor_rejection("legged_sqp_initialization_failed", exc)
        continue
    result = ops.solve_sqp(
        request,
        profile,
        view,
        corridor,
        subgoal,
        initial,
        deadline,
        backend=self.backend,
        reserve_model=self.reserve_model,
        memory_ledger=memory_ledger,
        repair_constraint=None,
    )
    if result.candidate is None:
        record_solver_rejection(result)
        continue
    post_solver_reserve = self.reserve_model.assess(
        profile,
        snapshot_cell_count=view.snapshot_cell_count,
        snapshot_canonical_byte_count=view.snapshot_canonical_byte_count,
        segment_count=len(result.candidate.duration_s),
        risk_segment_count=len(result.candidate.duration_s),
        encoded_scalar_count=encoded_scalar_count_for_segments_v1(len(result.candidate.duration_s)),
    )
    phase_tokens = split_post_solver_tail_v1(
        result.solver_candidate_token,
        result.post_solver_tail_token,
        post_solver_reserve,
        memory_ledger=memory_ledger,
    )
    if not admit_post_solver_phase_v1(
        "encode",
        required_time_s=post_solver_reserve.tail_time_s,
        required_bytes=post_solver_reserve.tail_bytes,
        deadline=deadline,
        memory_ledger=memory_ledger,
        memory_tokens=(phase_tokens.solver_candidate, phase_tokens.encode_scratch, phase_tokens.encoded_candidate, phase_tokens.decoded_candidate, phase_tokens.risk_phase, phase_tokens.l2_phase),
    ):
        record_corridor_rejection("legged_resource_budget_exceeded")
        continue
    def codec_checkpoint() -> None:
        if deadline.expired:
            raise LeggedPlanningDeadlineExpiredV1("planning deadline expired")
        if deadline.remaining_s <= post_solver_reserve.risk_feature_time_s + post_solver_reserve.l2_time_s:
            raise LeggedSQPResourceCutoffV1("encode reached reserved risk/L2 tail")

    candidate_bytes = ops.encode_candidate(
        result.candidate,
        checkpoint=codec_checkpoint,
        memory_ledger=memory_ledger,
        memory_token=phase_tokens.encode_scratch,
        output_memory_token=phase_tokens.encoded_candidate,
    )
    decoded_candidate = ops.decode_candidate_with_token(
        candidate_bytes,
        checkpoint=codec_checkpoint,
        memory_ledger=memory_ledger,
        memory_token=phase_tokens.encoded_candidate,
        decoded_output_memory_token=phase_tokens.decoded_candidate,
    )
    canonical_candidate = decoded_candidate.candidate
    candidate_representation_tokens = LeggedCandidateRepresentationTokensV1(
        solver_candidate=phase_tokens.solver_candidate,
        encoded_candidate=decoded_candidate.candidate_bytes_token,
        decoded_candidate=decoded_candidate.decoded_candidate_token,
    )
    if not admit_post_solver_phase_v1(
        "risk_features",
        required_time_s=post_solver_reserve.risk_feature_time_s + post_solver_reserve.l2_time_s,
        required_bytes=post_solver_reserve.candidate_bytes + post_solver_reserve.risk_feature_bytes + post_solver_reserve.l2_bytes,
        deadline=deadline,
        memory_ledger=memory_ledger,
        memory_tokens=(*candidate_representation_tokens.ordered(), phase_tokens.risk_phase, phase_tokens.l2_phase),
    ):
        record_corridor_rejection("legged_resource_budget_exceeded")
        continue
    risk_evaluation = ops.evaluate_candidate_risk(
        canonical_candidate,
        view,
        profile,
        runner=self.risk_runner,
        ledger=risk_ledger,
        memory_ledger=memory_ledger,
        memory_token=phase_tokens.risk_phase,
        feature_reserve_s=post_solver_reserve.risk_feature_time_s,
        downstream_reserve_s=post_solver_reserve.l2_time_s,
        deadline=deadline,
    )
    risk = risk_evaluation.evidence
    risk_evidence_token = risk_evaluation.memory_token
    if not admit_post_solver_phase_v1(
        "l2",
        required_time_s=post_solver_reserve.l2_time_s,
        required_bytes=post_solver_reserve.candidate_bytes + post_solver_reserve.risk_feature_bytes + post_solver_reserve.l2_bytes,
        deadline=deadline,
        memory_ledger=memory_ledger,
        memory_tokens=(*candidate_representation_tokens.ordered(), risk_evidence_token, phase_tokens.l2_phase),
    ):
        record_corridor_rejection("legged_resource_budget_exceeded")
        continue
    l2 = ops.validate_l2(
        candidate_bytes,
        request,
        profile,
        view,
        corridor,
        subgoal,
        risk,
        deadline,
        memory_ledger=memory_ledger,
        memory_token=phase_tokens.l2_phase,
        candidate_representation_tokens=candidate_representation_tokens,
        risk_evidence_token=risk_evidence_token,
        repair_constraint=None,
        prior_l2_failure=None,
    )
    if l2.passed:
        plan_build = ops.build_plan(
            request,
            profile,
            corridor,
            subgoal,
            l2,
            risk,
            telemetry,
            canonical_candidate_bytes=candidate_bytes,
            deadline=deadline,
            memory_ledger=memory_ledger,
            memory_tokens=(l2.promotion_token, corridor_memory_token),
        )
        if deadline.expired:
            raise LeggedPlanningDeadlineExpiredV1("planning deadline expired")
        return plan_build
    if l2.counterexample is not None and l2.counterexample.repairable:
        repair_preparation = ops.build_repair_constraint(
            l2,
            candidate=canonical_candidate,
            corridor=corridor,
            rolling_goal=subgoal,
            risk=risk,
            memory_ledger=memory_ledger,
            repair_context_token=l2.repair_context_token,
            view=view,
            profile=profile,
        )
        repair = repair_preparation.constraint
        repair_context_token = repair_preparation.memory_token
        repaired = ops.solve_sqp(
            request,
            profile,
            view,
            corridor,
            subgoal,
            initial,
            deadline,
            backend=self.backend,
            reserve_model=self.reserve_model,
            memory_ledger=memory_ledger,
            repair_constraint=repair,
        )
        repaired_outcome = ops.validate_repaired_candidate_once(
            repaired,
            request,
            profile,
            view,
            corridor,
            subgoal,
            repair,
            l2,
            self.risk_runner,
            risk_ledger,
            memory_ledger,
            telemetry,
            deadline,
            prior_candidate=canonical_candidate,
            prior_risk=risk,
            repair_context_token=repair_context_token,
        )
        if type(repaired_outcome) is LeggedRollingPlanBuildV1:
            return repaired_outcome

return _provider_failure_from_attempts_v1(request, profile, attempt_records, telemetry)
```

The preceding non-executable catalog supplies arguments to both provider-level and single-corridor steps；it defines no scope、loop or return semantics. The following decomposition is the sole executable ownership/taxonomy/control-flow contract:

```python
def plan(self, request, deadline, *, memory_ledger, output_escrow) -> LeggedProviderResultV1:
    telemetry = LeggedRollingTelemetryBuilderV1.start(request, self.profile, deadline)
    try:
        ops = self.trusted_ops
        with LeggedRequestScopeV1(memory_ledger, owner="provider") as provider_scope:
            result = _plan_in_request_scope_v1(
                request,
                self.profile,
                deadline,
                ops=ops,
                backend=self.backend,
                risk_runner=self.risk_runner,
                derived_map_cache=self.derived_map_cache,
                reserve_model=self.reserve_model,
                memory_ledger=memory_ledger,
                provider_scope=provider_scope,
                telemetry=telemetry,
            )
            _require_open_legged_deadline_v1(deadline)
            if result.output_token is not None:
                output_escrow.deposit_from(provider_scope, result.output_token)
            return result
    except (LeggedPlanningDeadlineExpiredV1, LeggedSQPDeadlineExpiredV1):
        return LeggedProviderResultV1(_provider_failure_v1(request, self.profile, "planning_deadline_expired", telemetry), None, None)
    except LeggedMapProfileMismatchV1:
        if deadline.expired:
            return LeggedProviderResultV1(_provider_failure_v1(request, self.profile, "planning_deadline_expired", telemetry), None, None)
        return LeggedProviderResultV1(_provider_failure_v1(request, self.profile, "legged_map_profile_mismatch", telemetry), None, None)
    except LeggedCorridorBudgetExceededV1:
        if deadline.expired:
            return LeggedProviderResultV1(_provider_failure_v1(request, self.profile, "planning_deadline_expired", telemetry), None, None)
        return LeggedProviderResultV1(_provider_failure_v1(request, self.profile, "legged_corridor_budget_exceeded", telemetry), None, None)
    except LeggedSQPResourceCutoffV1:
        if deadline.expired:
            return LeggedProviderResultV1(_provider_failure_v1(request, self.profile, "planning_deadline_expired", telemetry), None, None)
        return LeggedProviderResultV1(_provider_failure_v1(request, self.profile, "legged_resource_budget_exceeded", telemetry), None, None)
    except (LeggedPlanningMemoryIdentityErrorV1, LeggedBodyCodecErrorV1):
        if deadline.expired:
            return LeggedProviderResultV1(_provider_failure_v1(request, self.profile, "planning_deadline_expired", telemetry), None, None)
        return LeggedProviderResultV1(_provider_failure_v1(request, self.profile, "legged_identity_mismatch", telemetry), None, None)


def _plan_in_request_scope_v1(request, profile, deadline, *, ops, backend, risk_runner, derived_map_cache, reserve_model, memory_ledger, provider_scope, telemetry) -> LeggedProviderResultV1:
    profile_hash = ops.profile_hash(profile)
    ops.require_provisional_memory_ledger(memory_ledger, profile_hash, request.resource_budget)
    provider_hash_scratch = memory_ledger.reserve("provider_snapshot_hash_scratch", SNAPSHOT_HASH_SCRATCH_BYTES_V2)
    provider_scope.register(provider_hash_scratch)
    view = ops.build_hard_safety_view(
        request.terrain_snapshot,
        profile,
        deadline,
        memory_ledger=memory_ledger,
        hash_scratch_token=provider_hash_scratch,
    )
    if view.profile_hash != profile_hash:
        raise LeggedPlanningMemoryIdentityErrorV1("hard-view profile hash differs from sealed trusted op")
    provider_scope.release(provider_hash_scratch)
    request_hash = ops.request_hash(request, terrain_snapshot_hash=view.snapshot_hash)
    ops.require_injected_memory_ledger(
        memory_ledger,
        request_hash,
        profile_hash,
        request.resource_budget,
    )
    risk_ledger = LeggedRiskInferenceLedgerV1.create(request_hash, profile)
    start_check = view.validate_body_pose(..., memory_ledger=memory_ledger)
    _require_open_legged_deadline_v1(deadline)
    if not start_check.passed or not ops.validate_initial_motion(...).passed:
        return LeggedProviderResultV1(_provider_failure_v1(..., "legged_start_invalid", ...), None, None)
    goal_check = view.validate_body_pose(..., memory_ledger=memory_ledger)
    _require_open_legged_deadline_v1(deadline)
    if not goal_check.passed:
        return LeggedProviderResultV1(_provider_failure_v1(..., "legged_goal_invalid", ...), None, None)
    hierarchy = ops.build_hierarchy(..., cache=derived_map_cache, memory_ledger=memory_ledger)
    provider_scope.register(hierarchy.memory_token)
    worst_post_solver_reserve = reserve_model.assess(...)
    worst_corridor_risk_reserve = reserve_model.assess_risk_only(...)
    corridor_bundle = ops.generate_corridors(..., risk_runner=risk_runner, memory_ledger=memory_ledger, risk_ledger=risk_ledger)
    provider_scope.register_all(corridor_bundle.retained_memory_tokens)
    _require_open_legged_deadline_v1(deadline)
    if not corridor_bundle.intents:
        return LeggedProviderResultV1(_provider_failure_v1(..., "legged_no_global_corridor", ...), None, None)
    for corridor, corridor_token in zip(corridor_bundle.intents, corridor_bundle.retained_memory_tokens, strict=True):
        _require_open_legged_deadline_v1(deadline)
        with provider_scope.corridor_attempt(corridor.corridor_hash, corridor_token) as attempt_scope:
            attempt = _attempt_one_corridor_v1(
                request,
                profile,
                view,
                corridor,
                deadline,
                ops=ops,
                backend=backend,
                risk_runner=risk_runner,
                reserve_model=reserve_model,
                risk_ledger=risk_ledger,
                memory_ledger=memory_ledger,
                attempt_scope=attempt_scope,
                telemetry=telemetry,
            )
            if attempt.plan_build is not None:
                _require_open_legged_deadline_v1(deadline)
                attempt_scope.handoff_to(provider_scope, attempt.plan_build.memory_token)
                return LeggedProviderResultV1(attempt.plan_build.plan, attempt.plan_build.canonical_candidate_bytes, attempt.plan_build.memory_token)
        _require_open_legged_deadline_v1(deadline)  # deadline wins over local/init/rejection mapping
        record_corridor_rejection(attempt.rejection)
    _require_open_legged_deadline_v1(deadline)
    return LeggedProviderResultV1(_provider_failure_from_attempts_v1(...), None, None)


def _attempt_one_corridor_v1(request, profile, view, corridor, deadline, *, ops, backend, risk_runner, reserve_model, risk_ledger, memory_ledger, attempt_scope, telemetry) -> LeggedCorridorAttemptResultV1:
    try:
        subgoal = ops.select_rolling_subgoal(...)
    except LeggedLocalSubgoalUnavailableV1 as exc:
        _require_open_legged_deadline_v1(deadline)
        return LeggedCorridorAttemptResultV1(rejection=local_subgoal_rejection(exc))
    attempt_scope.register(subgoal.memory_token)
    try:
        initial = ops.initialize_sqp(...)
    except LeggedSQPInitializationFailedV1 as exc:
        _require_open_legged_deadline_v1(deadline)
        return LeggedCorridorAttemptResultV1(rejection=initialization_rejection(exc))
    attempt_scope.register(initial.memory_token)
    result = ops.solve_sqp(...)
    if result.candidate is None:
        _require_open_legged_deadline_v1(deadline)
        return LeggedCorridorAttemptResultV1(rejection=solver_rejection(result))
    attempt_scope.register_all((result.solver_candidate_token, result.post_solver_tail_token))
    phase_tokens = split_post_solver_tail_v1(result.solver_candidate_token, result.post_solver_tail_token, ...)
    attempt_scope.replace_many((result.solver_candidate_token, result.post_solver_tail_token), phase_tokens.ordered())
    candidate_bytes = ops.encode_candidate(...)
    attempt_scope.forget_consumed(phase_tokens.encode_scratch)  # encoder already released it
    decoded = ops.decode_candidate_with_token(...)
    attempt_scope.replace_many(
        (phase_tokens.encoded_candidate, phase_tokens.decoded_candidate),
        (decoded.candidate_bytes_token, decoded.decoded_candidate_token),
    )
    candidate_representation_tokens = LeggedCandidateRepresentationTokensV1(
        solver_candidate=phase_tokens.solver_candidate,
        encoded_candidate=decoded.candidate_bytes_token,
        decoded_candidate=decoded.decoded_candidate_token,
    )
    risk_eval = ops.evaluate_candidate_risk(...)
    attempt_scope.replace(phase_tokens.risk_phase, risk_eval.memory_token)
    l2 = ops.validate_l2(...)
    l2_inputs = (*candidate_representation_tokens.ordered(), risk_eval.memory_token, phase_tokens.l2_phase)
    if l2.passed:
        attempt_scope.replace_many(l2_inputs, (l2.promotion_token,))
        plan_build = ops.build_plan(
            ...,
            canonical_candidate_bytes=candidate_bytes,
            memory_ledger=memory_ledger,
            memory_tokens=(l2.promotion_token, attempt_scope.corridor_memory_token),
        )
        attempt_scope.replace_many((l2.promotion_token, attempt_scope.corridor_memory_token), (plan_build.memory_token,))
        return LeggedCorridorAttemptResultV1(plan_build=plan_build)
    if l2.counterexample is not None and l2.counterexample.repairable:
        attempt_scope.replace_many(l2_inputs, (l2.repair_context_token,))
        repair_attempt = _run_exactly_one_scoped_repair_or_rejection_v1(..., attempt_scope=attempt_scope)
        _require_exact_type(repair_attempt, LeggedCorridorAttemptResultV1, "repair_attempt")
        return repair_attempt
    attempt_scope.forget_consumed_many(l2_inputs)  # L2 already released every input
    return LeggedCorridorAttemptResultV1(rejection=l2_rejection(l2))
```

`LeggedRequestScopeV1` keeps an explicit ordered registry of every live handle handed to it; it never discovers ownership by querying the ledger. `register/replace/replace_many/adopt/detach/release` mirror ledger reserve/transfer/merge operations, reject `None` except in the explicitly empty non-feasible branch, and `__exit__` releases still-owned handles in descending token-ID order. The child scope adopts that corridor's retained token and every subgoal/initializer/tail/repair handle；any rejection、phase-admission false or exception closes it before the next corridor. The repair helper performs the same explicit register/replace sequence, transfers the repair-context handle through constraint/SQP/codec/risk/L2, and always returns exact `LeggedCorridorAttemptResultV1`：its mutually exclusive branch contains either one `LeggedRollingPlanBuildV1` or one token-free rejection, never a naked plan-build object. Success merges the selected corridor handle into plan-build ownership before it can leave the attempt scope；unselected corridor handles remain in provider scope and are released there. This wrapper gives deadline precedence, maps every typed resource/codec exception before Task 10's broad catch, and makes early returns mechanically safe rather than relying on prose cleanup.

`LeggedRollingTrustedOpsV1` is a frozen exact-type container for hard-view/hierarchy/corridor/subgoal/initializer/solver/codec/risk/L2/repair/promotion and all identity/cost hash callables; `create()` rejects any function whose module、qualname、code hash or object identity differs from the approved imports, and `plan()` invokes only `self.trusted_ops`. `validate_legged_initial_motion_v1()` accepts `0 <= v <= max_forward_speed_mps`; negative speed additionally requires `reverse_enabled=True` and `abs(v) <= max_reverse_speed_mps`. Exact boundary values pass and `nextafter` outside fails before corridor/SQP work. `split_post_solver_tail_v1()` independently recomputes the reserve, validates the feasible result's exact solver-candidate plus remaining-tail handles, and splits them once into solver-candidate、encode-scratch、encoded-candidate、decoded-candidate、risk-phase and L2-phase children whose bytes match the sealed fields and sum to `tail_bytes`. `admit_post_solver_phase_v1()` never re-reserves memory: it always gives deadline precedence, validates the exact ordered live-child composite and requires `sum(token.bytes)==required_bytes`, then applies the strict remaining-time condition；equality returns the stable resource failure before mutation. Encode releases only scratch；decode transfers distinct encoded/object handles；risk evidence transfers the risk child；L2 merges all three candidate representations plus risk/L2 children into a success promotion token, into a repair-context token only for a localized repairable failure, or releases all children for every nonrepairable failure. All helper calls map exceptions to the frozen reasons, with timeout precedence checked immediately before mapping.

`build_legged_rolling_plan_v1(..., deadline, memory_ledger, memory_tokens=(l2_promotion_token,selected_corridor_token)) -> LeggedRollingPlanBuildV1` checks global expiry before work、every 64 bounded segment/evidence/telemetry hash records、before decision hash and immediately before return. It validates and merges both handles with exact public-plan materialization capacity, returns the immutable plan plus one output token and never publishes a partial plan；therefore public `global_corridor` cannot outlive released backing ownership. The repair builder and `validate_repaired_candidate_once()` both require the same live `repair_context_token` from the preceding `l2` object, exact `prior_candidate=canonical_candidate`、`prior_risk=risk` and the same request risk/memory ledgers；those prior objects are mandatory independent inputs for recomputing the failed validation-input/record hashes and are never recovered from a closure or trusted as hash strings alone. The builder only validates the live token；the repaired helper retains it through encode/SQP/L2 and releases it exactly once after final success/failure. It applies the identical pre-encode、pre-feature、post-feature/pre-L2 admissions, then repeats encode → fresh decode → risk evaluation → full L2 (passing exact `repair` and `prior_l2_failure=l2` keyword-only) → the same selected-corridor-aware plan-build, rechecks expiry before returning, and cannot recurse into a second repair. Solver calls validate but do not invalidate the initial guess's live memory token, so the same immutable guess may seed the one authorized repair；provider releases it when that corridor ends. The provider never creates the request ledger and never returns a naked public plan: it uses the single Task 10-injected ledger and returns `LeggedProviderResultV1`; provider scopes release hierarchy、unselected corridors、risk、solver、codec and L2 work tokens exactly once while detaching at most one output handle to the API root scope. The loop never processes corridors concurrently and never reevaluates an accepted route for improvement. `mission_complete=True` additionally requires `subgoal.is_mission_goal`、exact requested mission pose identity、L2 goal tolerance and exact terminal `v==0.0`.

Plan-build signature clarification: its exact signature also requires `canonical_candidate_bytes` and `LeggedRollingPlanBuildV1` returns those same validated bytes beside `plan/memory_token`. The output token covers the immutable plan, selected-corridor backing and retained bytes through the API postcondition；no public field exposes the private bytes, and Task 10 discards them before publicize.

Scope/phase-token clarification is normative: no `None` handle is ever registered. `forget_consumed/forget_consumed_many` may be called only when a named callee contract has already invalidated or released the exact registered handles；each method requires a matching ledger terminal receipt, proves every token is stale and `live_bytes` was already debited or transferred, then removes registry entries without a second ledger release. A still-live、wrong-ledger、wrong-owner or receipt-less handle raises `LeggedPlanningMemoryIdentityErrorV1` and remains registered for fail-closed cleanup. Thus encoder-released scratch and nonrepairable-L2 inputs cannot be released twice or silently forgotten. `split_post_solver_tail_v1()` has one return type only—`LeggedPostSolverPhaseTokensV1(solver_candidate,encode_scratch,encoded_candidate,decoded_candidate,risk_phase,l2_phase)`—and its `.ordered()` tuple is used identically by tests, provider and repair helper.

`_run_exactly_one_scoped_repair_or_rejection_v1()` likewise has one return type only: exact `LeggedCorridorAttemptResultV1(plan_build=<LeggedRollingPlanBuildV1>,rejection=None)` on repaired L2 success or `LeggedCorridorAttemptResultV1(plan_build=None,rejection=<record>)` on every final failure. It never returns a naked plan-build object；the outer corridor loop therefore uses the same `attempt.plan_build` branch for initial and repaired success.

Registry methods never repeat callee ledger mutations: `register/adopt/detach` change only scope ownership after validating a live same-ledger handle；`replace/replace_many` require the exact ledger split/merge/transfer receipt proving old handles stale and new handles live；only `release` invokes `ledger.release()` itself. This pre/postcondition supersedes any earlier shorthand that scope methods “mirror” ledger operations.

Cross-scope correction is stricter: successful code never composes a standalone `detach` followed by `adopt/register`. Attempt→provider uses only atomic `attempt_scope.handoff_to(provider_scope, token)`；provider→caller uses only `output_escrow.deposit_from(provider_scope, token)`；API immediately uses `api_scope.claim_from_escrow(...)` before any deadline/result branch. Bare `detach/adopt` remain internal primitives for constructing the locked transaction and are not callable by planning code.

The trusted-ops seal includes exact `request_hash`、`profile_hash`、`require_provisional_memory_ledger`、`require_injected_memory_ledger` and `validate_initial_motion` callables in addition to terrain/SQP/codec/L2 functions. Executable provider code invokes all five only through `ops`；module-global rebinding therefore cannot alter ledger identity admission or initial-motion validation outside the API's provider token.

- [ ] **Step 5: Build stable telemetry, cost reporting, and decision hash**

Telemetry extends `SearchTelemetryV2` with corridor count/order/attempts、SQP iteration/evaluation、L2 interval/cell、repair、risk fallback、cache hit and decision hash. `elapsed_s`、`cache_hit`、worker/runtime utilization and the `decision_hash` field itself are observational/self-referential and excluded from decision-hash input; semantic counters、ordered identities、outcome、risk evidence and route hashes remain included. `CostBreakdownV2` is copied only from the L2-validated cost totals/hash in the receipt; provider code never independently recomputes or accepts solver cost. The authoritative selection tuple remains a separate exact telemetry field.

Failure evidence flattens first counterexample and resource ledger into sorted scalar `(key,value)` details. It does not attach any corridor/candidate object or encoded bytes.

Test/formal isolation uses a non-authoritative `risk_neutral_geometry_digest`: canonical request/profile/controller/snapshot identities、`corridor_geometry_hash` plus decoded `(states,a,omega,duration)` bytes, explicitly excluding risk-bearing corridor hash、risk guide、risk/model IDs、receipt and decision hash. It is never published as candidate identity. Paired risk tests compare L2 verdicts only for the same risk-neutral geometry key and use fixtures/corpora that evaluate the complete fixed candidate universe, so normal first-success early return cannot create a false mismatch or hide one.

- [ ] **Step 6: Enforce stable failure selection after all corridors fail**

Choose the final reason by a fixed severity/order table, not exception arrival time: deadline → resource → identity/numeric/internal → repair L2 → candidate L2 → SQP infeasible/init → subgoal → no corridor. Preserve the first deterministic evidence within the selected class.

- [ ] **Step 7: Run GREEN and component regressions**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_legged_rolling_provider.py tests/test_v2_legged_rolling_corridors.py tests/test_v2_legged_body_sqp_solver.py tests/test_v2_legged_body_sqp_validation.py tests/test_v2_legged_rolling_risk.py -q
```

Expected: all pass for success, every failure reason, first-pass stop, repair cap, model fallback, cache equivalence and deadline injection.

- [ ] **Step 8: Review and commit Task 9**

```powershell
git -C path-planner diff --check
git -C path-planner add src/path_planner/v2/providers/legged_rolling.py src/path_planner/v2/providers/__init__.py tests/test_v2_legged_rolling_provider.py
git -C path-planner commit -m "feat: orchestrate legged rolling body planning"
git add path-planner
git commit -m "build: integrate legged rolling body provider"
```

---

### Task 10: Add the Capability-Specific Trusted API and PPO Goal Adapter Without Touching `plan_v2()`

**Files:**
- Create: `path-planner/src/path_planner/v2/legged_rolling_api.py`
- Create: `path-planner/src/path_planner/v2/adapters/legged_rolling_target.py`
- Modify: `path-planner/src/path_planner/v2/adapters/__init__.py`
- Modify: `path-planner/src/path_planner/v2/__init__.py`
- Create: `path-planner/tests/test_v2_legged_rolling_api.py`
- Modify: `path-planner/tests/test_v2_api.py`
- Modify: `path-planner/tests/test_v2_ppo_target_adapter.py`

**Interfaces:**
- Consumes: exact request/profile registry/provider map/clock and existing `PpoTargetV2`；adapter 的非 PPO 参数显式包含 `request_id`、current `PoseStateV2`、authoritative `TerrainSnapshotV2`、current speed、profile/controller IDs、period and budgets。
- Produces: public `plan_legged_rolling_v1()`、`build_legged_rolling_request_from_ppo_v1()` and `plan_legged_rolling_ppo_target_v1()`。

- [ ] **Step 1: Write RED opt-in and no-fallback API tests**

```python
def test_new_capability_is_exported_only_from_v2_namespace() -> None:
    import path_planner
    import path_planner.v2 as v2

    assert "plan_legged_rolling_v1" not in vars(path_planner)
    assert callable(v2.plan_legged_rolling_v1)


def test_unresolved_profile_or_provider_never_calls_old_planners(monkeypatch) -> None:
    monkeypatch.setattr(old_legged_provider, "plan", forbidden_call)
    monkeypatch.setattr(old_wheel_provider, "plan", forbidden_call)
    monkeypatch.setattr(AStarPlanner, "plan", forbidden_call)
    monkeypatch.setattr(HybridAStarPlanner, "plan", forbidden_call)
    outcome = plan_legged_rolling_v1(REQUEST, registry=EMPTY_REGISTRY, providers={})
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "legged_body_profile_unsupported"

    missing_provider = plan_legged_rolling_v1(REQUEST, registry=REGISTRY, providers={})
    assert type(missing_provider) is PlanningFailureV2
    assert missing_provider.reason_code == "legged_body_profile_unsupported"


def test_generic_plan_v2_behavior_and_signature_do_not_change() -> None:
    assert inspect.signature(plan_v2) == EXPECTED_EXISTING_PLAN_V2_SIGNATURE
    assert plan_v2(EXISTING_REQUEST, registry=EXISTING_REGISTRY, providers=EXISTING_PROVIDERS) == EXISTING_OUTCOME
```

- [ ] **Step 2: Write RED deadline, identity-seal, snapshot, and PPO-boundary tests**

```python
@pytest.mark.parametrize(("timeout_s", "period_s", "expected"), [(1.0, 0.2, 0.2), (0.1, 0.5, 0.1)])
def test_api_creates_one_minimum_deadline(timeout_s: float, period_s: float, expected: float) -> None:
    clock = ScriptedClock(100.0)
    provider = make_exact_capturing_provider(PROFILE)
    plan_legged_rolling_v1(make_request(timeout_s=timeout_s, replan_period_s=period_s), registry=REGISTRY, providers={PROFILE_ID: provider}, monotonic_clock=clock)
    assert provider.deadline.started_monotonic_s == 100.0
    assert provider.deadline.deadline_monotonic_s == pytest.approx(100.0 + expected)


def test_late_success_is_replaced_by_timeout() -> None:
    outcome = plan_with_provider_that_returns_success_after_deadline()
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "planning_deadline_expired"


def test_ppo_adapter_uses_only_target_pose_and_explicit_current_speed() -> None:
    request = build_legged_rolling_request_from_ppo_v1(
        TARGET,
        request_id=REQUEST_ID,
        current_body_pose=START,
        terrain_snapshot=AUTHORITATIVE_TERRAIN,
        current_speed_mps=0.4,
        platform_profile_id=PROFILE_ID,
        controller_capability_id=CONTROLLER_ID,
        objective_profile=OBJECTIVE,
        resource_budget=BUDGET,
        timeout_s=0.5,
        replan_period_s=0.2,
        determinism_seed=7,
    )
    assert request.current_body_state == LeggedBodyStateV1(START.x_m, START.y_m, START.heading_rad, 0.4)
    assert request.mission_goal_state == PoseStateV2(TARGET.x_m, TARGET.y_m, TARGET.theta_rad)
    assert request.terrain_snapshot is AUTHORITATIVE_TERRAIN
    assert dict(request.terrain_snapshot.provenance.details)["controller_capability_hash"] == CONTROLLER_HASH


def test_legged_ppo_adapter_does_not_accept_provenance_rewriting_observation_wrapper() -> None:
    assert "observed_terrain" not in inspect.signature(build_legged_rolling_request_from_ppo_v1).parameters


@pytest.mark.parametrize("mutation", ["segment_distance", "segment_relative_energy", "cost_component", "cost_hash"])
def test_api_postcondition_rejects_rehashed_public_cost_tamper(mutation: str) -> None:
    outcome = plan_with_provider_returning_tampered_success(mutation)
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code == "legged_identity_mismatch"


def test_api_independently_seals_snapshot_and_request_hash_before_provider_dispatch() -> None:
    outcome, audit = plan_with_provider_returning_self_consistent_forged_hashes()
    assert outcome.reason_code == "legged_identity_mismatch"
    assert audit.api_expected_snapshot_hash == snapshot_hash(AUTHORITATIVE_TERRAIN)
    assert audit.api_expected_request_hash == legged_rolling_request_hash_v1(REQUEST, terrain_snapshot_hash=audit.api_expected_snapshot_hash)


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        ("provider_plan_method", "legged_body_profile_unsupported"),
        ("l2_validator", "legged_body_profile_unsupported"),
        ("candidate_codec", "legged_body_profile_unsupported"),
        ("request_hash_callable", "legged_body_profile_unsupported"),
        ("promotion_callable", "legged_body_profile_unsupported"),
    ],
)
def test_api_rejects_provider_or_trusted_callable_rebinding(mutation: str, expected_reason: str) -> None:
    outcome = plan_with_rebound_provider_surface(mutation)
    assert outcome.reason_code == expected_reason


def test_api_snapshot_preflight_can_expire_inside_ultrawide_row_before_dispatch() -> None:
    outcome, provider = plan_ultrawide_snapshot_with_expiring_clock()
    assert outcome.reason_code == "planning_deadline_expired"
    assert provider.plan_call_count == 0


def test_api_provider_and_postcondition_share_the_only_request_memory_ledger() -> None:
    outcome, audit = plan_with_memory_identity_audit()
    assert type(outcome) is LeggedRollingPlanV1
    assert audit.ledger_create_count == 1
    assert audit.provider_ledger_id == audit.postcondition_ledger_id == audit.api_ledger_id
    assert audit.live_bytes_after_publicize == 0


def test_api_postcondition_reserve_one_byte_short_rejects_before_provider() -> None:
    outcome, audit = plan_with_memory_limit(API_POSTCONDITION_RESERVE_BYTES - 1)
    assert outcome.reason_code == "legged_resource_budget_exceeded"
    assert audit.provider_call_count == 0


@pytest.mark.parametrize(
    ("fault", "expected_reason"),
    [
        ("corridor_cap", "legged_corridor_budget_exceeded"),
        ("sqp_resource_cutoff", "legged_resource_budget_exceeded"),
    ],
)
def test_public_api_preserves_provider_resource_reason(fault: str, expected_reason: str) -> None:
    outcome = plan_legged_rolling_v1(request_for_provider_resource_fault(fault), registry=REGISTRY, providers=SEALED_PROVIDERS)
    assert outcome.reason_code == expected_reason
    assert outcome.category is FailureCategoryV2.RESOURCE_LIMIT


@pytest.mark.parametrize("branch", ["tampered_success", "postcondition_deadline", "provider_exception"])
def test_api_scope_releases_output_and_postcondition_tokens_on_every_failure(branch: str) -> None:
    outcome, audit = plan_api_failure_with_memory_audit(branch)
    assert type(outcome) is PlanningFailureV2
    assert audit.live_bytes_after_scope == 0
    assert audit.double_release_count == 0


@pytest.mark.parametrize("boundary", ["attempt_to_provider", "provider_to_escrow", "escrow_to_api"])
def test_atomic_output_handoff_has_no_unowned_window_under_base_exception(boundary: str) -> None:
    with pytest.raises(KeyboardInterrupt):
        run_with_interrupt_at_handoff_boundary(boundary)
    assert HANDOFF_AUDIT.unowned_token_count == 0
    assert HANDOFF_AUDIT.live_bytes_after_root_scope == 0
```

- [ ] **Step 3: Run RED tests**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_legged_rolling_api.py tests/test_v2_api.py tests/test_v2_ppo_target_adapter.py -q -x
```

Expected: new API/adapter imports fail; old API tests remain green.

- [ ] **Step 4: Implement exact preflight, provider seal, and one deadline**

```python
def plan_legged_rolling_v1(
    request: LeggedRollingRequestV1,
    *,
    registry: LeggedRollingProfileRegistryV1,
    providers: Mapping[str, LeggedRollingBodySQPProviderV1],
    monotonic_clock: MonotonicClockV2 = monotonic,
) -> LeggedRollingOutcomeV1:
    _require_exact_type(request, LeggedRollingRequestV1, "request")
    _require_exact_type(registry, LeggedRollingProfileRegistryV1, "registry")
    if not isinstance(providers, Mapping) or not callable(monotonic_clock):
        raise TypeError("providers must be Mapping and monotonic_clock must be callable")
    started = _exact_finite_timestamp(monotonic_clock(), "started_monotonic_s")
    deadline = PlanningDeadlineV2(
        started_monotonic_s=started,
        deadline_monotonic_s=started + min(request.timeout_s, request.replan_period_s),
        _monotonic_clock=monotonic_clock,
    )
    if deadline.expired:
        return _api_timeout(request, None, "profile_resolution")
    profile = registry.resolve(request.platform_profile_id)
    if deadline.expired:
        return _api_timeout(request, profile, "profile_resolution")
    if profile is None:
        return _api_failure(request, None, "legged_body_profile_unsupported", "profile_resolution")
    if request.controller_capability_id != profile.controller_capability_id:
        if deadline.expired:
            return _api_timeout(request, profile, "controller_resolution")
        return _api_failure(request, profile, "legged_body_profile_unsupported", "controller_resolution")
    provider = providers.get(profile.profile.profile_id)
    if type(provider) is not LeggedRollingBodySQPProviderV1 or provider.profile != profile:
        if deadline.expired:
            return _api_timeout(request, profile, "provider_resolution")
        return _api_failure(request, profile, "legged_body_profile_unsupported", "provider_resolution")
    try:
        provider_token = legged_rolling_provider_token_v1(provider)
    except LeggedProviderSealErrorV1:
        if deadline.expired:
            return _api_timeout(request, profile, "provider_resolution")
        return _api_failure(request, profile, "legged_body_profile_unsupported", "provider_resolution")
    provider_plan = provider.plan
    if 0 < request.resource_budget.max_memory_bytes < SNAPSHOT_HASH_SCRATCH_BYTES_V2:
        if deadline.expired:
            return _api_timeout(request, profile, "snapshot_preflight")
        return _api_failure(request, profile, "legged_resource_budget_exceeded", "snapshot_preflight")
    try:
        audit_legged_snapshot_preflight_v1(request.terrain_snapshot, profile, deadline)
        snapshot_audit = snapshot_hash_audit_with_checkpoints_v2(
            request.terrain_snapshot,
            checkpoint=lambda: _require_open_legged_deadline_v1(deadline),
            limits=LEGGED_SNAPSHOT_METADATA_LIMITS_V1,
        )
        expected_snapshot_hash = snapshot_audit.snapshot_hash
        expected_request_hash = legged_rolling_request_hash_v1(
            request,
            terrain_snapshot_hash=expected_snapshot_hash,
        )
    except LeggedPlanningDeadlineExpiredV1:
        return _api_timeout(request, profile, "snapshot_preflight")
    except LeggedMapProfileMismatchV1:
        return _api_failure(request, profile, "legged_map_profile_mismatch", "snapshot_preflight")
    request_token = legged_rolling_request_token_v1(
        request,
        profile,
        expected_snapshot_hash=expected_snapshot_hash,
        expected_request_hash=expected_request_hash,
    )
    profile_hash = legged_rolling_profile_hash_v1(profile)
    memory_ledger = LeggedPlanningMemoryLedgerV1.create(expected_request_hash, profile_hash, request.resource_budget)
    postcondition_reserve = assess_legged_api_postcondition_v1(profile)
    try:
        with LeggedRequestScopeV1(memory_ledger, owner="api") as api_scope:
            _require_open_legged_deadline_v1(deadline)
            postcondition_token = memory_ledger.reserve("api_postcondition_scratch", postcondition_reserve.bytes)
            api_scope.register(postcondition_token)
            output_escrow = LeggedOutputEscrowV1.create(memory_ledger, api_scope)
            api_scope.register_escrow(output_escrow)
            try:
                provider_result = provider_plan(request, deadline, memory_ledger=memory_ledger, output_escrow=output_escrow)
            except (KeyboardInterrupt, SystemExit, MemoryError):
                raise
            except Exception as exc:
                if deadline.expired:
                    return _api_timeout(request, profile, "provider_completion")
                return _api_internal_failure(request, profile, type(exc).__name__)
            _require_exact_type(provider_result, LeggedProviderResultV1, "provider_result")
            if provider_result.output_token is not None:
                api_scope.claim_from_escrow(output_escrow, provider_result.output_token)
            elif not output_escrow.empty:
                raise LeggedPlanningMemoryIdentityErrorV1("failure result left output in escrow")
            if deadline.expired:
                return _api_timeout(request, profile, "provider_completion")
            postcondition = _postcondition_or_failure(
                request,
                profile,
                provider,
                provider_result,
                request_token,
                provider_token,
                expected_snapshot_hash,
                expected_request_hash,
                deadline,
                memory_ledger=memory_ledger,
                postcondition_memory_token=postcondition_token,
            )
            if type(postcondition.outcome) is PlanningFailureV2:
                return postcondition.outcome
            api_scope.replace(provider_result.output_token, postcondition.validated_output_token)
            api_scope.release(postcondition_token)
            _require_open_legged_deadline_v1(deadline)
            public_plan = postcondition.outcome
            api_scope.publicize(postcondition.validated_output_token)
            return public_plan
    except (LeggedPlanningDeadlineExpiredV1, LeggedSQPDeadlineExpiredV1):
        return _api_timeout(request, profile, "api_scope")
    except LeggedSQPResourceCutoffV1:
        if deadline.expired:
            return _api_timeout(request, profile, "api_scope")
        return _api_failure(request, profile, "legged_resource_budget_exceeded", "api_scope")
    except (LeggedPlanningMemoryIdentityErrorV1, LeggedBodyCodecErrorV1):
        if deadline.expired:
            return _api_timeout(request, profile, "api_scope")
        return _api_failure(request, profile, "legged_identity_mismatch", "api_scope")
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except Exception as exc:
        if deadline.expired:
            return _api_timeout(request, profile, "api_scope")
        return _api_internal_failure(request, profile, type(exc).__name__)
```

`legged_rolling_provider_token_v1()` seals exact provider class/profile、the exact bound class `plan` code/object identity、preloaded backend ID/version hash、SQP geometry ID、reserve-model ID、cache policy ID、optional risk-runner/model/calibration hashes and every `LeggedRollingTrustedOpsV1` callable's module/qualname/code hash/object identity. The API captures the bound method once and the postcondition rechecks the complete provider/trusted-ops token after return. Provider code uses only that frozen ops object, so module-global/class/instance rebinding cannot silently authorize an alternate codec、hash、L2 or promotion path. No branch calls generic `plan_v2()`、legacy static legged provider、wheel provider、Hybrid A* or v1 A*.
Every `_api_failure*` constructor sets `PlanningFailureV2.platform_kind=PlatformKindV2.LEGGED`, including unresolved profile/provider cases, so it respects the shared failure dataclass invariant without rewriting the frozen specialized reason to `platform_profile_unresolved`.

Freeze `LeggedAPIPostconditionReserveV1` with `API_POSTCONDITION_BASE_BYTES_V1=65_536`、`API_POSTCONDITION_PER_SEGMENT_BYTES_V1=512`、`API_POSTCONDITION_PER_SCALAR_BYTES_V1=192` and `API_POSTCONDITION_CANONICAL_BUFFER_COUNT_V1=2`. `assess_legged_api_postcondition_v1(profile)` uses `profile.max_segments` and the same `encoded_scalar_count_for_segments_v1()` overflow-safe formula to cover strict JSON parsing、one fresh candidate object、canonical re-encode buffers、segment cost/hash scratch and the postcondition wrapper. The API reserves this complete scratch before provider dispatch；a one-byte-short bounded ledger fails without calling provider. The initial snapshot preflight uses only the frozen `TAIL_SNAPSHOT_HASH_SCRATCH_BYTES` streaming buffer and checks a nonzero request memory ceiling against that fixed peak before starting, because the request hash needed to create the ledger does not exist yet. No later API/provider allocation is exempt from the single ledger.

Before dispatch the root API scope also registers an empty `LeggedOutputEscrowV1` and passes it with the ledger. Provider success deposits atomically；on return the API exact-type checks the wrapper then atomically claims the matching token before checking late deadline or any identity field. Provider failure requires an empty escrow. Therefore every base-exception/timeout path is owned by attempt scope、provider scope、escrow or API scope at all times；there is no return-stack interval in which a live output handle is unregistered.

- [ ] **Step 5: Implement lightweight postcondition and PPO adapter**

For a success, API requires exact `LeggedProviderResultV1` with exact `LeggedRollingPlanV1`、nonempty canonical candidate bytes and one same-ledger output token；a failure requires exact `PlanningFailureV2` with both internal fields `None`. Under the pre-reserved postcondition token it checks exact equality to independently sealed `expected_snapshot_hash/expected_request_hash`、matching remaining IDs/hashes、global `executable=False`、trajectory L2、receipt/candidate/risk/rolling-goal/validation-input hash agreement、correct `mission_complete` and canonical candidate `decode→encode` byte equality. Without repeating the terrain sweep, it independently recomputes each public segment's analytic traveled distance `0.5*(abs(v_start)+abs(v_end))*duration` under the rechecked no-sign-crossing rule、relative energy、receipt distance/energy/time/slew totals、the objective-weighted `CostBreakdownV2` and `cost_breakdown_hash`；endpoint chord distance is never authoritative, and any rehashed segment/cost drift is `legged_identity_mismatch`. Recheck request/provider/trusted-ops/snapshot tokens before and after codec/cost work and check deadline between bounded segments and immediately before return；expiry always wins over identity failure. Only after every check passes does `_postcondition_or_failure()` transfer the output token to owner `api_validated_public_output` and return `LeggedAPIPostconditionResultV1(plan,new_token)`；failure leaves all registered handles for root-scope cleanup. The API then releases scratch, performs the final deadline check and calls `api_scope.publicize(new_token)`；the private candidate bytes disappear with the internal wrapper and the ledger has zero live bytes on public return.

The PPO adapter constructs only mission pose from `PpoTargetV2`; request ID、current body pose/speed、the original authoritative snapshot、profile/controller IDs、period and budgets remain explicit non-PPO caller inputs. It exact-type checks and passes the snapshot object unchanged, returns the specialized outcome unchanged, and never creates `ObservedTerrainInputV2`、canonical-observation provenance or a route projection back into PPO map truth. Existing `build_ppo_request_v2()`/`plan_ppo_target_v2()` remain unchanged for legacy capabilities.

- [ ] **Step 6: Run GREEN and old API regressions**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_legged_rolling_api.py tests/test_v2_api.py tests/test_v2_ppo_target_adapter.py tests/test_v2_legged_provider.py tests/test_v2_wheel_provider.py -q
```

Expected: all pass; old signatures, type aliases, default dispatch and expected outcomes are unchanged.

- [ ] **Step 7: Review and commit Task 10**

```powershell
git -C path-planner diff --check
git -C path-planner add src/path_planner/v2/legged_rolling_api.py src/path_planner/v2/adapters/legged_rolling_target.py src/path_planner/v2/adapters/__init__.py src/path_planner/v2/__init__.py tests/test_v2_legged_rolling_api.py tests/test_v2_api.py tests/test_v2_ppo_target_adapter.py
git -C path-planner commit -m "feat: expose opt-in legged rolling planner API"
git add path-planner
git commit -m "build: integrate legged rolling planner API"
```

---

### Task 11: Freeze Adversarial Conformance, Golden Bytes, Cache Equivalence, and Cross-Process Determinism

**Files:**
- Create: `path-planner/tests/fixtures/legged_rolling_conformance_v1.json`
- Create: `path-planner/tests/test_v2_legged_rolling_conformance.py`
- Modify only if a RED test exposes a real defect: the exact Task 1–10 source/test file responsible for that defect.

**Interfaces:**
- Consumes: public API and all frozen IDs/codec/hash fields。
- Produces: checked-in small conformance corpus and a deterministic semantic digest used by formal benchmark tests。

- [ ] **Step 1: Write the exact conformance fixture and schema test**

The fixture contains only small synthetic proxy maps and expected semantics, never physical obstacle claims:

```json
{
  "schema_id": "legged_rolling_conformance_corpus/v1",
  "capability_id": "legged_multicorridor_rolling_body_sqp/v1",
  "cases": [
    {"case_id": "flat_straight_200ms", "expected": "success", "period_s": 0.2},
    {"case_id": "flat_turn_500ms", "expected": "success", "period_s": 0.5},
    {"case_id": "localized_repair_once", "expected_repair_count": 1, "period_s": 0.5},
    {"case_id": "low_confidence_strip", "expected_reason": "legged_no_global_corridor", "period_s": 0.5},
    {"case_id": "slope_nextafter_30", "expected_reason": "legged_no_global_corridor", "period_s": 0.5},
    {"case_id": "two_corridors_first_l2_rejected", "expected_selected_corridor": 1, "period_s": 0.5},
    {"case_id": "unknown_strip", "expected_reason": "legged_no_global_corridor", "period_s": 0.5}
  ]
}
```

Test exact keys、unique sorted case IDs、valid periods、synthetic provenance and a checked-in SHA-256 of canonical fixture bytes.

- [ ] **Step 2: Add metamorphic safety/risk/identity tests**

```python
def test_risk_model_matrix_never_changes_hard_or_candidate_l2_matrix() -> None:
    audits = [run_corpus(model=value) for value in (None, VALID_MODEL, OOD_MODEL, NONFINITE_MODEL)]
    assert len({audit.hard_feasibility_digest for audit in audits}) == 1
    assert len({audit.risk_neutral_geometry_l2_matrix_digest for audit in audits}) == 1


def test_cache_matrix_changes_no_semantic_field() -> None:
    disabled = run_corpus(cache=None)
    enabled = run_corpus(cache=LeggedDerivedMapCacheV1())
    assert disabled.decision_digest == enabled.decision_digest


@pytest.mark.parametrize("identity", ["profile", "controller", "snapshot", "candidate", "solver_geometry", "backend", "risk", "validator"])
def test_each_identity_mutation_fails_closed(identity: str) -> None:
    outcome = run_identity_mutation(identity)
    assert type(outcome) is PlanningFailureV2
    assert outcome.reason_code in {"legged_map_profile_mismatch", "legged_identity_mismatch"}
```

`run_corpus()` has two outputs: normal public-API outcomes, which retain first-L2 early return, and an isolation-only matrix over the fixture's checked-in canonical candidate bytes. The latter invokes hard/L2 validation for every fixed candidate under paired risk evidence, keys rows by the Task 9 risk-neutral geometry digest, and is never fed back into provider selection.

- [ ] **Step 3: Add subprocess determinism and golden-byte tests**

Run the corpus in fresh processes for `PYTHONHASHSEED=0` and `1`, with cache off/on and each risk mode held constant across the compared process pair. Compare normalized corridor order/signatures、rolling goal、candidate bytes、L2 receipt/counterexample/repair、risk evidence、success/failure reason and decision hash. Exclude elapsed time and cache-hit fields. Do not require different risk modes to have the same decision hash or candidate identity; their isolation comparison uses the fixed, complete risk-neutral geometry/L2 matrix from Step 2.

```python
def test_cross_process_semantic_bytes_are_identical() -> None:
    outputs = [run_subprocess(seed, cache) for seed in (0, 1) for cache in (False, True)]
    assert len({value["semantic_sha256"] for value in outputs}) == 1
```

- [ ] **Step 4: Run conformance RED/GREEN and fix only demonstrated defects**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest -q -p no:cacheprovider tests/test_v2_legged_rolling_conformance.py --basetemp D:/xunce/tmp/lbsqp_conformance
```

Expected: all fixture、metamorphic、golden and subprocess tests pass. Any failure must first remain as a focused RED test, then receive the smallest source fix and affected regression run.

- [ ] **Step 5: Run complete focused capability suite**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest -q -p no:cacheprovider tests/test_v2_legged_rolling_contracts.py tests/test_v2_legged_rolling_safety.py tests/test_v2_legged_rolling_risk.py tests/test_v2_legged_rolling_corridors.py tests/test_v2_legged_body_kinematics.py tests/test_v2_legged_rolling_subgoals.py tests/test_v2_legged_body_sqp_initialization.py tests/test_v2_legged_body_sqp_serialization.py tests/test_v2_legged_body_sqp_solver.py tests/test_v2_legged_body_sqp_validation.py tests/test_v2_legged_rolling_provider.py tests/test_v2_legged_rolling_api.py tests/test_v2_legged_rolling_conformance.py --basetemp D:/xunce/tmp/lbsqp_focused
```

Expected: all pass with no unregistered warning or nondeterministic retry.

- [ ] **Step 6: Review and commit Task 11**

```powershell
git -C path-planner diff --check
git -C path-planner add tests/fixtures/legged_rolling_conformance_v1.json tests/test_v2_legged_rolling_conformance.py
git -C path-planner commit -m "test: freeze legged rolling planner conformance"
git add path-planner
git commit -m "build: integrate legged rolling conformance"
```

If conformance exposed source fixes, add only their exact paths to the nested commit and list each RED test in the commit body.

---

### Task 12: Implement Formal Benchmark Rows, Aggregation, Thresholds, and Independent-Oracle Audit

**Files:**
- Create: `path-planner/src/path_planner/v2/legged_rolling_benchmark.py`
- Create: `path-planner/tests/test_v2_legged_rolling_benchmark.py`

**Interfaces:**
- Consumes: immutable formal result rows produced by actual public API calls and independent oracle labels。
- Produces: `LeggedRollingFormalRowV1`、`LeggedRollingFormalSummaryV1`、`aggregate_legged_rolling_formal_v1()` and exact blockers。

- [ ] **Step 1: Write RED pass-boundary aggregation tests**

```python
def test_formal_summary_passes_exact_approved_boundaries() -> None:
    summary = aggregate_legged_rolling_formal_v1(make_boundary_rows())
    assert summary.evidence_coverage_complete is True
    assert summary.oracle_execution_complete is True
    assert summary.oracle_unsafe_false_positive_count == 0
    assert summary.success_without_complete_l2_count == 0
    assert summary.hard_violation_count == 0
    assert summary.oracle_reachable_success_ratio == 0.99
    assert summary.risk_isolation_violation_count == 0
    assert summary.late_success_count == 0
    assert summary.partial_trajectory_count == 0
    assert summary.executable_corridor_count == 0
    assert summary.determinism_match_ratio == 1.0
    assert summary.periods_reported_s == (0.2, 0.5)
    assert summary.passed is True
```

- [ ] **Step 2: Write RED one-mutation-per-gate tests**

```python
@pytest.mark.parametrize(
    ("mutation", "blocker"),
    [
        ("unsafe_false_positive", "legged_body_sqp_unsafe_false_positive"),
        ("success_without_l2", "legged_body_sqp_complete_l2_gate_failed"),
        ("hard_violation", "legged_body_sqp_hard_violation"),
        ("reachable_0_989", "legged_body_sqp_reachable_success_gate_failed"),
        ("risk_changes_l2", "legged_body_sqp_risk_isolation_gate_failed"),
        ("late_success", "legged_body_sqp_late_success_gate_failed"),
        ("partial_trajectory", "legged_body_sqp_partial_trajectory_gate_failed"),
        ("executable_corridor", "legged_body_sqp_corridor_authority_gate_failed"),
        ("decision_hash_drift", "legged_body_sqp_determinism_gate_failed"),
        ("missing_200ms_rows", "legged_body_sqp_period_coverage_incomplete"),
        ("empty_reachable_denominator", "legged_body_sqp_reachable_denominator_empty"),
        ("reachable_denominator_below_minimum", "legged_body_sqp_reachable_coverage_insufficient"),
        ("empty_unsafe_denominator", "legged_body_sqp_unsafe_denominator_empty"),
        ("missing_unsafe_boundary_class", "legged_body_sqp_unsafe_boundary_coverage_incomplete"),
        ("missing_strong_join", "legged_body_sqp_oracle_join_incomplete"),
        ("duplicate_strong_join", "legged_body_sqp_oracle_join_not_unique"),
        ("unapproved_oracle_manifest", "legged_body_sqp_oracle_independence_unproven"),
        ("low_oracle_budget", "legged_body_sqp_oracle_budget_contract_failed"),
        ("mixed_oracle_budget", "legged_body_sqp_oracle_budget_contract_failed"),
        ("oracle_timeout_or_unknown", "legged_body_sqp_oracle_execution_incomplete"),
        ("oracle_receipt_hash_drift", "legged_body_sqp_oracle_execution_incomplete"),
        ("oracle_receipt_hash_cycle_domain", "legged_body_sqp_oracle_execution_incomplete"),
    ],
)
def test_each_formal_mutation_has_one_stable_blocker(mutation: str, blocker: str) -> None:
    assert blocker in aggregate_legged_rolling_formal_v1(make_mutated_rows(mutation)).blockers
```

- [ ] **Step 3: Write RED source-independence and metric-definition tests**

Formal labels must declare:

```text
oracle_source_id != provider_source_id
source_independent = true
oracle_uses_subject_corridors = false
oracle_uses_subject_sqp = false
oracle_uses_subject_risk_model = false
oracle_uses_subject_l2_validator = false
hard_capability_boundary_id = legged_body_hard_capability_boundary/v1
oracle_repository_commit = <40 lowercase hex>
oracle_tree_hash = <sha256>
oracle_container_image_digest = sha256:<64 lowercase hex>
oracle_entrypoint_hash = <sha256>
oracle_review_record_hash = <sha256>
oracle_high_budget_contract_id = legged_body_independent_oracle_high_budget/v1
oracle_run_config_hash = <sha256>
oracle_run_id = <sha256>
oracle_execution_receipt_hash = <sha256>
oracle_max_expanded_states_per_case = 50000000
oracle_timeout_s_per_case = 300.0
oracle_max_memory_bytes_per_case = 8589934592
```

These declarations are not accepted on self-assertion alone. A separate immutable `independent_oracle_manifest.json` is a required trust-root input; config pins the approved source ID、repository commit、tree/container/entrypoint hashes、independent review-record hash、high-budget contract ID、the three exact per-case limits and canonical run-config hash. The runner hashes that manifest, requires exact equality to the pins, requires all implementation hashes to differ from the subject parent/nested tree and provider/validator source hashes, and records the reviewed manifest in `manifest.json`. This proves the artifact is the approved independent implementation; a different string ID alone never passes.

Require one immutable oracle execution receipt with an acyclic hash DAG. First hash the implementation-manifest subobject and run config. Next compute schedule payload hashes and body/rolling label payload hashes over canonical headers/rows that explicitly exclude only the outer-link fields `oracle_execution_receipt_hash` and `oracle_manifest_envelope_hash`; run/config IDs and every semantic label remain included. The receipt then binds those payload hashes、implementation-manifest hash、run-config/run IDs、case count and per-status counts, and only then receives `oracle_execution_receipt_hash`. Finally the outer reviewed envelope hashes implementation manifest + run config + receipt; label-file headers may reference the resulting receipt/envelope hashes, but those link fields never feed back into the payload hashes. Raw five-input byte hashes are separately frozen by the formal runner and are not claimed as receipt inputs. Tests recompute every domain independently and reject either omitted semantic output or accidental inclusion of a back-reference.

Every scheduled case must have exactly one terminal proof status `proved_reachable` or `proved_unreachable`; `timeout_count==resource_exhaustion_count==unknown_count==crash_count==0`, `completed_case_count==scheduled_case_count`, and every label row repeats the same run ID/config hashes while its file header carries the single receipt/envelope hashes. `proved_unreachable` is accepted only as the independent oracle's completed exhaustive proof under the pinned config, never as an alias for budget exhaustion. Low-budget、mixed-budget、partial/mixed-run or receipt/hash drift blocks before any reachable denominator is computed.

Freeze the one-to-one joins. Request-level rolling-goal labels and period schedules join on exactly `(scenario_id,request_id,period_s,determinism_seed,request_hash,terrain_snapshot_hash,profile_hash,controller_capability_hash,hard_capability_boundary_id,oracle_run_id,oracle_run_config_hash)`. Candidate-level body-L2 labels add `risk_neutral_geometry_digest`. Each schedule row must have exactly one actual public-API result and one rolling-goal label; each public success and every fixed risk-isolation candidate must have exactly one body-L2 label. Duplicate、missing、extra、weak-join or mixed-manifest/config/run/receipt rows block aggregation. Labels never import provider outcome、subject L2 receipt or runtime.

Reject a label set if it uses the old static foothold oracle to claim rolling-body controller feasibility. It may only label the same map/body-envelope/kinematic hard boundary; dynamic stability and real-controller tracking remain out of formal scope. Evidence coverage additionally requires at least 10,000 unique body-L2 rows、at least 1,000 independently safe rows and at least 100 independently unsafe rows for each of `unknown`、`hard_obstacle`、`not_traversable`、`slope_nextafter_30`、`confidence_below_profile`、`out_of_bounds`、`mid_segment_collision` and `rotation_corner_collision`; at least 1,000 unique rolling-goal labels; at least 100 oracle-reachable request rows for each `0.2` and `0.5` period; complete paired risk matrices and complete determinism repeats. A row has one primary boundary class so counts cannot be duplicated across classes. These are evidence-integrity preconditions, not relaxed performance thresholds.

- [ ] **Step 4: Run RED tests**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_legged_rolling_benchmark.py -q -x
```

Expected: new benchmark module is absent.

- [ ] **Step 5: Implement exact row and aggregation contracts**

Every result row contains the complete strong join key、oracle manifest/run-config/run/execution-receipt hashes、exact IDs/hashes、scenario/period、oracle labels、actual provider outcome、complete L2、hard violations、mission/rolling goal、corridor executable flag、risk-isolation paired digest、decision digest、runtime and component telemetry. Ratios require the explicit coverage and completed high-budget execution preconditions above, not merely a nonempty aggregate denominator. Runtime p50/p95/p99 use nearest-rank on finite nonnegative values and are reported separately for `0.2` and `0.5`; the design does not add a percentile threshold beyond no late success.

Hard gates are exactly:

```python
passed = (
    evidence_coverage_complete is True
    and oracle_execution_complete is True
    and oracle_unsafe_false_positive_count == 0
    and success_without_complete_l2_count == 0
    and hard_violation_count == 0
    and oracle_reachable_success_ratio >= 0.99
    and risk_isolation_violation_count == 0
    and late_success_count == 0
    and partial_trajectory_count == 0
    and executable_corridor_count == 0
    and determinism_match_ratio == 1.0
    and periods_reported_s == (0.2, 0.5)
)
```

- [ ] **Step 6: Run GREEN and boundary regressions**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_v2_legged_rolling_benchmark.py tests/test_v2_legged_rolling_conformance.py -q
```

Expected: all exact-boundary、one-epsilon、empty-denominator、source-identity and nearest-rank tests pass.

- [ ] **Step 7: Review and commit Task 12**

```powershell
git -C path-planner diff --check
git -C path-planner add src/path_planner/v2/legged_rolling_benchmark.py tests/test_v2_legged_rolling_benchmark.py
git -C path-planner commit -m "feat: add legged rolling formal metrics"
git add path-planner
git commit -m "build: integrate legged rolling formal metrics"
```

---

### Task 13: Add the D-Drive Formal Runner, Run Whole-Branch Regression, and Hand Off Without Publishing

**Files:**
- Create: `scripts/run_xunce_path_v2_legged_body_sqp_formal.py`
- Create: `scripts/verify_xunce_path_v2_legged_body_sqp_formal.py`
- Create: `configs/xunce_path_v2_legged_body_sqp_formal_v1.json`
- Create: `tests/test_xunce_path_v2_legged_body_sqp_formal.py`
- Modify: `configs/stage_registry.json`
- Modify: `docs/xunce-stage-documentation-index.md`
- Modify only exact source/test files required by new focused RED tests for Critical/Important review findings.
- Write runtime artifacts only under a fresh child of `D:/xunce/out/path_v2/lbsqp`.

**Interfaces:**
- Consumes: five absolute D-drive formal input files, pinned oracle trust-root fields and the public `plan_legged_rolling_v1()` implementation。
- Produces: resumable formal results, exact blockers, eight canonical artifacts and a non-publishing handoff。

- [ ] **Step 1: Write RED config, registry, missing-input, and self-oracle tests**

The config requires exactly these independent inputs:

```text
independent_oracle_manifest_json        exactly 1 reviewed implementation+run-config+execution-receipt envelope
independent_body_l2_labels_jsonl       >= 10,000 rows
independent_rolling_goal_labels_jsonl  >= 1,000 rows
period_200ms_schedules_jsonl           >= 100 rows
period_500ms_schedules_jsonl           >= 100 rows
```

```python
def test_stage_uses_short_d_drive_root_and_exact_config() -> None:
    entry = stage_entry("xunce-path-v2-legged-body-sqp-formal")
    assert entry["default_config"] == "configs/xunce_path_v2_legged_body_sqp_formal_v1.json"
    assert entry["default_output_root"] == "D:/xunce/out/path_v2/lbsqp"


def test_formal_preflight_requires_exact_preloaded_scipy_backend(monkeypatch) -> None:
    monkeypatch.setattr(runner, "load_legged_backend", wrong_or_missing_backend)
    summary = runner.run_legged_body_sqp_formal(CONFIG, OUT, REPO_ROOT, execute=True)
    assert summary["status"] == "blocked"
    assert "legged_body_sqp_backend_not_ready" in summary["blockers"]


def test_missing_inputs_is_blocked_not_passed(tmp_path) -> None:
    summary = runner.run_legged_body_sqp_formal(CONFIG, tmp_path / "out", REPO_ROOT, execute=False)
    assert summary["status"] == "blocked"
    assert summary["formal_metrics_status"] == "not_evaluated"
    assert summary["blockers"][0] == "legged_body_sqp_independent_formal_inputs_missing"


def test_subject_cannot_label_itself(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(runner, "_execute_provider_case", forbidden_call)
    inputs = write_inputs(tmp_path, oracle_source_id="legged-body-sqp-subject/v1", provider_source_id="legged-body-sqp-subject/v1")
    summary = runner.run_legged_body_sqp_formal(config_for(inputs), tmp_path / "out", REPO_ROOT, execute=True)
    assert summary["status"] == "blocked"
    assert "legged_body_sqp_oracle_provider_identity_not_independent" in summary["blockers"]


def test_independent_verifier_rejects_subject_aggregator_only_tamper(tmp_path, monkeypatch) -> None:
    run_root = write_complete_boundary_run(tmp_path)
    monkeypatch.setattr(runner, "aggregate_legged_rolling_formal_v1", forged_passing_aggregate)
    runner.write_subject_summary(run_root)
    verified = run_verifier_subprocess(run_root)
    assert verified["passed"] is False
    assert "legged_body_sqp_subject_summary_mismatch" in verified["blockers"]


@pytest.mark.parametrize("mutation", ["low_budget", "mixed_budget", "timeout", "resource_exhausted", "unknown", "receipt_hash_drift", "receipt_back_reference_in_payload_domain"])
def test_oracle_execution_envelope_blocks_before_subject_execution(mutation: str, monkeypatch) -> None:
    monkeypatch.setattr(runner, "_execute_provider_case", forbidden_call)
    summary = runner.run_legged_body_sqp_formal(config_with_oracle_mutation(mutation), OUT, REPO_ROOT, execute=True)
    assert summary["status"] == "blocked"
    assert summary["formal_metrics_status"] == "not_evaluated"
```

- [ ] **Step 2: Write RED immutable-input and actual-execution tests**

Reject relative/C-drive paths、non-UTF-8、BOM、duplicate JSON keys、hash mismatch、wrong row count、duplicate request or strong-join key、mixed identities、wrong capability/profile/controller/solver/SQP-geometry/backend/reserve/validator/risk IDs、unapproved oracle source/commit/tree/container/entrypoint/review hash、wrong/mixed high-budget contract or run-config hash、limits other than exactly `(50000000,300.0,8589934592)`、execution-receipt/run ID drift、any timeout/resource/unknown/crash/incomplete count、wrong parent/nested commit and input bytes that change between audit and execution. The execution receipt is embedded in and hashed by the single reviewed oracle manifest envelope, so the formal input count remains exactly five. Schedule rows contain only request terrain/start/goal/period/seed and strong join identities—no oracle labels and never trusted provider success、runtime、trajectory hash or L2 result. Oracle labels live only in their independently produced files and must join one-to-one after actual execution.

- [ ] **Step 3: Run RED root tests**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_xunce_path_v2_legged_body_sqp_formal.py -q -x
```

Expected: runner/config/stage failures because root files are absent.

- [ ] **Step 4: Implement the resumable phases and real provider execution**

Freeze phases:

```text
preflight
hard_safety_audit
trajectory_l2_audit
rolling_goal_reachability
period_200ms
period_500ms
risk_isolation
determinism
aggregate
```

Read every input once as immutable bytes, validate header/rows hash and exact keys, first verify the oracle implementation manifest、run config and embedded execution receipt against config pins and subject source/tree hashes, and require completed high-budget status before any subject call. Freeze the five input byte hashes before any request; re-read/re-hash before aggregation. `period_200ms` and `period_500ms` call the public API with exact `replan_period_s`; no benchmark-only longer timeout or hidden solver tolerance. Join actual outputs to independent labels only by the Task 12 strong keys including run/config identity and require complete one-to-one coverage before metrics. Risk isolation reruns the same audited requests with deterministic fallback and an approved deterministic model, then validates the complete fixed canonical candidate corpus under both evidence modes and compares hard/L2 verdicts by the Task 9 risk-neutral geometry key. It never compares model-dependent candidate hashes and never treats first-success early stopping as evidence that an unevaluated candidate passed or failed. After the subject aggregate, invoke the second verifier in a fresh process; that script parses frozen bytes/results itself and must not import `path_planner.v2.legged_rolling_benchmark` or the runner aggregate.

Determinism runs semantic cases under `PYTHONHASHSEED=0,1` and outer worker counts `1,4`; the SQP process remains single-threaded. Reassemble by `(period_s,scenario_id,seed,worker)` before digesting. Runtime and utilization are excluded from semantic hashes.

- [ ] **Step 5: Write canonical artifacts only through artifact helpers**

Use `scripts/xunce_artifact_io.py` and `scripts/xunce_artifact_paths.py`; write exactly:

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
  "changes_plan_v2_semantics": false,
  "uses_old_legged_runtime_fallback": false,
  "claims_dynamic_stability": false
}
```

Register the stage and documentation index; do not modify old Gate 4/Gate 6 configs、cases、metrics or historical artifacts.

- [ ] **Step 6: Run GREEN root tests and dry-run registration**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_xunce_path_v2_legged_body_sqp_formal.py -q
D:/conda_envs/lunar-explorer/python.exe scripts/run_stage.py --stage xunce-path-v2-legged-body-sqp-formal --dry-run
```

Expected: tests pass; dry run resolves the exact config/root and performs no provider execution or artifact overwrite.

- [ ] **Step 7: Commit root formal orchestration separately**

```powershell
git diff --check
git add scripts/run_xunce_path_v2_legged_body_sqp_formal.py scripts/verify_xunce_path_v2_legged_body_sqp_formal.py configs/xunce_path_v2_legged_body_sqp_formal_v1.json tests/test_xunce_path_v2_legged_body_sqp_formal.py configs/stage_registry.json docs/xunce-stage-documentation-index.md
git commit -m "feat: add legged body SQP formal gate"
```

- [ ] **Step 8: Run exact nested full regression in a fresh D-drive temp root**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest -q -p no:cacheprovider path-planner/tests --basetemp D:/xunce/tmp/lbsqp_nested_full
```

Expected: all required tests pass; only previously documented optional-environment skips remain. Record fresh counts rather than copying prior reports.

- [ ] **Step 9: Run root formal-runner and inherited isolation guards**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest -q -p no:cacheprovider tests/test_xunce_path_v2_legged_body_sqp_formal.py tests/test_xunce_path_v2_gate_benchmark.py tests/test_xunce_path_v2_g0_baseline_and_isolation.py --basetemp D:/xunce/tmp/lbsqp_root
```

Run the checked-in Gate 0 auditor under a fresh D-drive root and require no new failed node ID beyond its inherited allowlist. The new capability must not change old PPO/default-path results.

- [ ] **Step 10: Audit formal inputs or emit the exact blocked evidence**

If any independent input is missing, run once with `--no-execute` under a fresh root:

```powershell
$blockedRoot = "D:/xunce/out/path_v2/lbsqp/blocked_$([DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ', [Globalization.CultureInfo]::InvariantCulture))"
if (Test-Path -LiteralPath $blockedRoot) { throw "blocked evidence root must be fresh" }
D:/conda_envs/lunar-explorer/python.exe scripts/run_xunce_path_v2_legged_body_sqp_formal.py `
  --config configs/xunce_path_v2_legged_body_sqp_formal_v1.json `
  --output-root $blockedRoot `
  --no-execute
if ($LASTEXITCODE -ne 1) { throw "blocked audit must exit 1" }
```

Require `status=blocked`、`formal_metrics_status=not_evaluated`、first blocker `legged_body_sqp_independent_formal_inputs_missing` and no pass claim. Do not synthesize replacement labels from the subject planner.

If all inputs exist and audit cleanly, execute the nine phases under a new timestamp/parent/nested-hash directory. Never overwrite, delete, rename or promote an old failed/blocked run.

- [ ] **Step 11: Verify formal gates and artifacts independently**

For a pass claim require all Task 12 gates、all evidence-coverage minima and unique strong joins、both periods、eight artifacts、strict UTF-8/finite JSON、oracle implementation/review/run-config/high-budget/execution-receipt pins、zero timeout/resource/unknown/crash、complete schedule counts、input hashes/commits and all eight publishing flags false. Run `verify_xunce_path_v2_legged_body_sqp_formal.py` in a fresh process to independently recompute the receipt bindings、joins、denominators、nearest-rank metrics and gates from frozen input bytes plus `results.jsonl`; require exact structured agreement with subject summary. Static tests forbid imports of the subject benchmark aggregator/runner, and the deliberate subject-only tamper fixture must be caught.

If any gate fails, retain the failed evidence and keep the capability opt-in/unpromoted. Do not loosen safety/L2、increase request deadline、add Hybrid/static fallback or run later corridors after first success to force a pass.

- [ ] **Step 12: Obtain four fresh independent reviews**

Review package includes approved spec、this plan、parent/nested merge-base diffs、tests、conformance bytes、approved oracle implementation manifest with repository commit/tree/container/entrypoint/review/run-config/high-budget/execution-receipt hashes and completion counts、formal input headers/hashes/strong-join coverage、results/summary/routing/manifest/report and exact status. Require:

1. spec/contract coverage;
2. module boundaries and code quality;
3. adversarial continuous L2/repair/identity correctness;
4. benchmark independence、metrics、artifacts and thresholds.

Every Critical/Important finding gets a new focused failing test, minimal fix, affected focused suite, complete Steps 8–11 rerun and fresh re-review. Minor findings enter `review.json`/report ledger.

- [ ] **Step 13: Finish without publishing side effects**

Run strict UTF-8 scan、`git diff --check`、parent/nested status and exact gitlink check. Final handoff reports commits、test counts、formal pass/failed/blocked status、blockers and D-drive artifact path. It does not push、open a PR、publish checkpoint、replace default policy、connect executor、start canary or make the new profile default.

---

## Plan Self-Review

- Spec coverage: Task 1 freezes request/profile/controller/output/failure identities; Task 2 covers the single 0.5m truth, confidence/provenance hard gate and conservative `1.0/2.0m` layers; Task 3 covers learned-risk isolation、fallback and approved union/lexicographic semantics; Task 4 covers three topology-distinct corridors and stable order; Task 5 covers rolling goals and `(x,y,yaw,v)/(a,omega,dt)`; Tasks 6–8 cover canonical fresh-object replay、SQP、continuous L2 and one repair; Task 9 covers first-L2 return and stable failures; Task 10 covers dedicated API/PPO boundary; Tasks 11–13 cover determinism、formal gates and non-publishing release boundary.
- Scope consistency: old `LeggedProfileV2`、`plan_v2()`、`PlanningSuccessV2`、static footstep oracle and legacy CLI stay untouched in semantics. Shared source refactors are limited to the platform-neutral rectangle delegate and streaming snapshot-hash implementation；both preserve the full legacy call/acceptance domain and are guarded by exact wheel/generic golden regressions, while legged-only caps remain at the opt-in boundary.
- Type consistency: request uses `LeggedBodyStateV1` and mission `PoseStateV2`; private solver output remains `CanonicalLeggedBodyCandidateV1`; only passed `LeggedBodyL2ReceiptV1` promotes `TimedBodyTrajectoryV1`; outer success is `LeggedRollingPlanV1` while failure reuses `PlanningFailureV2`.
- Identity consistency: request/profile/controller/snapshot/corridor/subgoal/candidate/risk/validation-input/validation-record/trajectory/decision hashes are defined before consumers and checked at every promotion/repair boundary；provider/trusted-callable tokens and the internal canonical candidate bytes are independently rechecked by the API before publicize.
- Deadline consistency: only Task 10 creates the absolute deadline；all lower modules consume it. Snapshot chunks、hierarchy、A*/refinement/component/signature loops、risk slots、SQP callbacks、codec、L2、plan build and API postcondition all have bounded in-loop checkpoints；provider/L2/API recheck expiry before success, so late success is impossible.
- Safety authority: safety view and L2 never call the risk model; the model receives derived immutable features only. Coarse layers、corridors、SQP samples and optimizer success cannot grant L2.
- Controller boundary: no type or task contains footstep、contact force、gait phase、support polygon、MPC/WBC or joint command. Formal oracle labels only the declared hard body-envelope/kinematic boundary.
- Determinism: exact neighbor/component/signature order、single initializer、fixed SQP layout/options、half-even codec、risk full-slot scheduling、counterexample order、one repair、first-pass stop and subprocess matrix are explicit.
- Resource behavior: generic request budgets and downward-tightenable specialized caps bound snapshot/hierarchy、corridor/component/topology、risk、SQP broadphase/callback、codec and L2 work. Task 10 creates one request ledger shared through provider and postcondition；attempt scopes release on every branch, the reserve covers all simultaneous candidate representations plus two snapshot passes, selected-corridor/L2 ownership merges into one output token, and only the validated API boundary may publicize it.
- Formal evidence: subject code never creates oracle labels or optima. Missing/invalid independent input produces blocked/not-evaluated, never a fabricated pass.
- Compatibility: optional SciPy absence affects only explicit new capability. Existing default A*、Hybrid A*、Scout SQP work-in-progress、old static legged、hopper、PPO baseline and Gate 0–6 remain isolated.

---

## Execution Handoff

Plan execution has two supported modes:

1. **Subagent-Driven (recommended):** use `superpowers:subagent-driven-development`; dispatch a fresh bounded implementation worker per Task 1–13, then perform specification-compliance and code-quality review before each task commit. Shared files and nested gitlink integration stay serial under the main agent.
2. **Inline Execution:** use `superpowers:executing-plans`; execute the same tasks serially in this task with RED/GREEN batches and review checkpoints.

Neither mode authorizes publishing、default replacement、executor、canary、push or release. Before execution, finish or pause the concurrent Wheel-SQP task and obtain a clean, reviewed ownership boundary for every shared file.
