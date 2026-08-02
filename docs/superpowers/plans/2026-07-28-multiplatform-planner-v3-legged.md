# Multiplatform Planner v3 Legged Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现只输出固定机体参考点 x、y、z、yaw 与时序的足式 C++20 平台层，在解析地形阈值和机体几何范围内直接发布权威参考，并明确不保证足步可行。

**Architecture:** `LeggedLatticeAdapter` 在 `(ix, iy, iyaw)` 格点上搜索，同时为每个状态传播连续高度可达区间；候选经 xy 凸走廊、z 区间和 yaw 区间的完整积空间验证，再生成 yaw 独立的机体样条和机体系速度时序。任何连续处理失败都整段回退到认证机体运动原语。

**Tech Stack:** C++20、CMake 3.28+、Eigen 5.0.1、OSQP 1.0.0（`LPP_V3_ENABLE_OSQP=ON` 时）、GoogleTest 1.17.0、vcpkg manifest。

## Global Constraints

- 只在 `path-planner/cpp/**` 和本计划列出的测试 fixture 中实现；不调用现有 Python 规划代码。
- 命名空间固定为 `lunar::planning::v3`。
- 本分卷链接 `lpp_v3_contracts` 与 `lpp_v3_common`，产出
  `lpp_v3_legged`、`lpp_v3_legged_tests` 和 `lpp_v3_legged_benchmark`。
- `lpp_v3_legged_tests` 必须通过 `gtest_discover_tests(... TEST_PREFIX
  "lpp_v3_legged.")` 注册；所有过滤式 CTest gate 使用该前缀并带
  `--no-tests=error`。
- 规划状态和参考不使用四元数；roll、pitch 只能作为诊断包络。
- 参考点必须是能力 profile 固定的 `reference_point_id`，不得使用随腿姿变化的瞬时真实质心。
- 不搜索、不序列化足端位置、步态、接触序列或接触力。
- 每个输出必须固定：

```text
feasibility_scope = body_geometry_and_terrain_thresholds_only
footstep_feasibility_guaranteed = false
```

- 足式参考直接权威生效，不增加下游预接受门控。
- 硬可行性只使用地图已知性、机体碰撞和解析坡度、粗糙度、台阶、沟隙、净空、高度区间阈值。
- 主搜索代价是预计执行时间；其他代价只在时间等价池内比较。
- yaw 独立于路径切向，允许侧向、对角和原地转向。
- 走廊只有在完整 `xy × z × yaw` 笛卡尔积通过验证时才有效。
- 连续处理失败必须整段回退到认证机体原语。
- 承诺末端为零平移速度、零 yaw 角速度的安全停止锚点。
- 算法不读取 1 秒墙钟截止。
- 构建目录固定为 `D:/xunce/build/path-planner-v3/windows-msvc-debug`。

---

### Task 1: 足式能力视图和机体运动原语

**Files:**
- Create: `path-planner/cpp/include/lunar_path_planner/v3/legged/legged_capability.hpp`
- Create: `path-planner/cpp/include/lunar_path_planner/v3/legged/body_motion_primitive.hpp`
- Create: `path-planner/cpp/src/legged/legged_capability.cpp`
- Create: `path-planner/cpp/src/legged/body_motion_primitive.cpp`
- Create: `path-planner/cpp/tests/unit/legged/body_motion_primitive_test.cpp`
- Modify: `path-planner/cpp/src/legged/CMakeLists.txt`
- Modify: `path-planner/cpp/tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `SafetyCapabilityProfile`, `ContentRef`, `PoseXyzYaw`,
  `DurationNanoseconds` and `SecondaryCostVector` from the shared targets.
- Produces:

```cpp
enum class BodyMotionKind {
  kForward,
  kBackward,
  kLateralLeft,
  kLateralRight,
  kDiagonal,
  kSpin,
  kCoupledTranslationYaw,
};

struct BodyMotionPrimitive {
  PrimitiveId id;
  BodyMotionKind kind;
  PoseXyzYaw relative_end;
  DurationNanoseconds nominal_duration;
  BodyFrameVelocityEnvelope nominal_envelope;
  std::vector<double> normalized_samples;
  SecondaryCostVector secondary_costs;
};

class BodyMotionPrimitiveCatalog {
 public:
  static Result<BodyMotionPrimitiveCatalog> Create(
      const SafetyCapabilityProfile&);
  std::span<const BodyMotionPrimitive> ordered_primitives() const noexcept;
  const ContentRef& content_ref() const noexcept;
};
```

- [ ] **Step 1: 写出固定参考点缺失时拒绝能力的测试**

```cpp
TEST(BodyMotionPrimitiveTest, RejectsMissingFixedReferencePoint) {
  auto profile = MakeLeggedCapabilityFixture();
  profile.reference_point_id.clear();

  const auto result = BodyMotionPrimitiveCatalog::Create(profile);

  ASSERT_FALSE(IsOk(result));
  EXPECT_EQ(std::get<Error>(result).code,
            ErrorCode::kInvalidArgument);
}
```

- [ ] **Step 2: 写出能力必须含侧移和自旋原语的测试**

```cpp
TEST(BodyMotionPrimitiveTest, RequiresLateralAndSpinPrimitives) {
  auto profile = MakeLeggedCapabilityWithoutSpin();
  const auto result = BodyMotionPrimitiveCatalog::Create(profile);
  EXPECT_FALSE(IsOk(result));
}
```

- [ ] **Step 3: 构建并确认失败**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_legged_tests --parallel
```

Expected: 编译失败，提示 `BodyMotionPrimitiveCatalog` 未定义。

- [ ] **Step 4: 实现严格能力转换和稳定排序**

验证所有速度、持续时间、z 变化和 yaw 变化有限且在 profile 硬上限内。目录按 `BodyMotionKind` 固定次序，再按 `PrimitiveId` 升序。
在 `src/legged/CMakeLists.txt` 首次创建 `lpp_v3_legged`，公开链接
`lpp_v3_contracts` 与 `lpp_v3_common`；在测试 CMake 中把本分卷测试源汇总为
`lpp_v3_legged_tests`。后续任务只向这两个既有目标追加源文件。

- [ ] **Step 5: 增加“不得携带足端字段”的合同测试**

在 C++ 类型层通过编译时概念检查：

```cpp
template <class T>
concept HasFootstepMember = requires(T value) {
  value.footsteps;
};

static_assert(!HasFootstepMember<BodyMotionPrimitive>);
static_assert(!HasFootstepMember<LeggedBodyReference>);
```

- [ ] **Step 6: 运行能力与原语测试**

Run:

```powershell
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R '^lpp_v3_legged\.' --no-tests=error --output-on-failure
```

Expected: 全部通过。

- [ ] **Step 7: 提交足式能力和原语**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/legged/legged_capability.hpp cpp/include/lunar_path_planner/v3/legged/body_motion_primitive.hpp cpp/src/legged/legged_capability.cpp cpp/src/legged/body_motion_primitive.cpp cpp/src/legged/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/tests/unit/legged/body_motion_primitive_test.cpp
git -C path-planner commit -m "feat(v3): add legged body motion primitives"
```

---

### Task 2: 解析地形评估和高度区间传播

**Files:**
- Create: `path-planner/cpp/include/lunar_path_planner/v3/legged/legged_terrain.hpp`
- Create: `path-planner/cpp/include/lunar_path_planner/v3/legged/height_interval.hpp`
- Create: `path-planner/cpp/src/legged/legged_terrain.cpp`
- Create: `path-planner/cpp/src/legged/height_interval.cpp`
- Create: `path-planner/cpp/tests/unit/legged/legged_terrain_test.cpp`
- Create: `path-planner/cpp/tests/unit/legged/height_interval_test.cpp`
- Modify: `path-planner/cpp/src/legged/CMakeLists.txt`
- Modify: `path-planner/cpp/tests/CMakeLists.txt`

**Interfaces:**
- Consumes: immutable terrain layers and `LeggedCapabilityView`.
- Produces:

```cpp
struct HeightInterval {
  double min_m;
  double max_m;
};

struct LeggedTerrainEvaluation {
  bool hard_feasible;
  HeightInterval body_height_interval;
  Vec3 fitted_normal;
  double plane_residual_m;
  DurationNanoseconds terrain_scaled_duration;
  SecondaryCostVector secondary_costs;
  std::vector<ValidationReason> rejection_reasons;
};

class LeggedTerrainEvaluator {
 public:
  LeggedTerrainEvaluation EvaluatePose(
      const Pose2d&,
      const BodyCollisionEnvelope&) const;

  std::optional<HeightInterval> PropagateEdgeInterval(
      const HeightInterval& source,
      const BodyMotionPrimitive& edge,
      std::span<const LeggedTerrainEvaluation> samples) const;
};
```

- [ ] **Step 1: 写出未知和低置信单元硬拒绝测试**

```cpp
TEST(LeggedTerrainTest, UnknownOrLowConfidenceIsHardInfeasible) {
  const auto evaluator = MakeTerrainEvaluator();
  EXPECT_FALSE(evaluator.EvaluatePose(
      UnknownPose(), MakeBodyEnvelope()).hard_feasible);
  EXPECT_FALSE(evaluator.EvaluatePose(
      LowConfidencePose(), MakeBodyEnvelope()).hard_feasible);
}
```

- [ ] **Step 2: 写出高度区间断开时边不可达的测试**

```cpp
TEST(HeightIntervalTest, RejectsDisconnectedEdgeIntervals) {
  const HeightInterval source{0.40, 0.50};
  const auto result = MakeTerrainEvaluator().PropagateEdgeInterval(
      source,
      MakeVerticalLimitedPrimitive(/*max_delta_z=*/0.10),
      MakeSamplesWithRequiredHeight(0.70, 0.80));

  EXPECT_FALSE(result.has_value());
}
```

- [ ] **Step 3: 构建并确认失败**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_legged_tests --parallel
```

Expected: 编译失败，提示地形评估或高度区间类型未定义。

- [ ] **Step 4: 实现全部解析硬阈值**

按固定顺序验证：

```text
known_and_confident
body_collision_and_clearance
slope
roughness_and_plane_residual
step_height
gap_width
terrain_normal_change
body_height_interval
```

任何硬阈值失败都不得进入软代价。

- [ ] **Step 5: 实现区间交、Minkowski 高度变化和连续传播**

```cpp
reachable = Intersect(
    terrain_interval,
    Expand(source_interval,
           -primitive.max_down_delta_m,
           primitive.max_up_delta_m));
```

沿所有原语样本依次传播；任一步为空即拒绝整条边。

- [ ] **Step 6: 增加台阶、沟隙和顶部净空边界测试**

每个阈值分别测试“等于上限通过”和“超过一个机器可表示增量失败”，避免 `<`/`<=` 漂移。

- [ ] **Step 7: 运行地形和区间测试**

Run:

```powershell
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R '^lpp_v3_legged\.' --no-tests=error --output-on-failure
```

Expected: 全部通过。

- [ ] **Step 8: 提交地形和高度区间**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/legged/legged_terrain.hpp cpp/include/lunar_path_planner/v3/legged/height_interval.hpp cpp/src/legged/legged_terrain.cpp cpp/src/legged/height_interval.cpp cpp/src/legged/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/tests/unit/legged/legged_terrain_test.cpp cpp/tests/unit/legged/height_interval_test.cpp
git -C path-planner commit -m "feat(v3): evaluate legged terrain height intervals"
```

---

### Task 3: `SE(2.5D)` 足式格点与搜索

**Files:**
- Create: `path-planner/cpp/include/lunar_path_planner/v3/legged/legged_lattice.hpp`
- Create: `path-planner/cpp/include/lunar_path_planner/v3/legged/legged_search.hpp`
- Create: `path-planner/cpp/src/legged/legged_lattice.cpp`
- Create: `path-planner/cpp/src/legged/legged_search.cpp`
- Create: `path-planner/cpp/tests/unit/legged/legged_lattice_test.cpp`
- Create: `path-planner/cpp/tests/integration/legged/legged_search_test.cpp`
- Modify: `path-planner/cpp/src/legged/CMakeLists.txt`
- Modify: `path-planner/cpp/tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `RunAraStar`, `CandidateRanker`, Tasks 1–2.
- Produces:

```cpp
struct LeggedLatticeState {
  std::int32_t ix;
  std::int32_t iy;
  std::int32_t iyaw;
  HeightInterval reachable_z;
};

struct LeggedLatticeEdge {
  LeggedLatticeState source;
  LeggedLatticeState target;
  PrimitiveId primitive_id;
  DurationNanoseconds transition_time;
  SecondaryCostVector secondary_costs;
};

class LeggedLatticeAdapter {
 public:
  StateKey Key(const LeggedLatticeState&) const noexcept;
  std::vector<SearchTransition<LeggedLatticeState, LeggedLatticeEdge>> Expand(
      const LeggedLatticeState&) const;
  bool HardFeasible(
      const SearchTransition<LeggedLatticeState, LeggedLatticeEdge>&) const;
  DurationNanoseconds TransitionTime(
      const SearchTransition<LeggedLatticeState, LeggedLatticeEdge>&) const noexcept;
  DurationNanoseconds AdmissibleTimeHeuristic(
      const LeggedLatticeState&,
      const SearchProblem<LeggedLatticeState>&) const noexcept;
  SecondaryCostVector SecondaryCosts(
      const SearchTransition<LeggedLatticeState, LeggedLatticeEdge>&) const noexcept;
  bool IsTerminal(
      const LeggedLatticeState&,
      const SearchProblem<LeggedLatticeState>&) const noexcept;
};

Result<LeggedDiscretePlan> PlanLeggedDiscrete(
    const LeggedPlanningProblem&,
    const BodyMotionPrimitiveCatalog&,
    const AraStarConfig&);
```

- [ ] **Step 1: 写出 yaw 与切向独立的侧移测试**

```cpp
TEST(LeggedLatticeTest, LateralMoveKeepsBodyYaw) {
  const auto edge = FindPrimitiveEdge(
      MakeLeggedAdapter(), BodyMotionKind::kLateralLeft);

  EXPECT_NE(edge.source.iy, edge.target.iy);
  EXPECT_EQ(edge.source.iyaw, edge.target.iyaw);
}
```

- [ ] **Step 2: 写出原地转向零平移测试**

```cpp
TEST(LeggedLatticeTest, SpinChangesYawWithoutTranslation) {
  const auto edge = FindPrimitiveEdge(
      MakeLeggedAdapter(), BodyMotionKind::kSpin);
  EXPECT_EQ(edge.source.ix, edge.target.ix);
  EXPECT_EQ(edge.source.iy, edge.target.iy);
  EXPECT_NE(edge.source.iyaw, edge.target.iyaw);
}
```

- [ ] **Step 3: 构建并确认失败**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_legged_tests --parallel
```

Expected: 编译失败，提示足式格点接口未定义。

- [ ] **Step 4: 实现状态键与高度区间 dominance**

离散键只含 `(ix, iy, iyaw)`；同一键的标签若具有更小时间且可达 z 区间包含另一标签，则支配后者。不得把 z 压成单个离散高度。

- [ ] **Step 5: 实现时间启发式和候选排序**

平移和 yaw 最短时间取能力上限下界组合；不得假设 yaw 等于运动方向。ARA* 后按共享时间等价池比较能耗、风险和平滑性。

- [ ] **Step 6: 验证终端 z 区间与安全停止锚点**

终端 pose 必须具有非空高度区间；取：

```cpp
z_ref = std::clamp(
    capability.preferred_body_height_m,
    interval.min_m,
    interval.max_m);
```

并附加零平移、零 yaw 角速度锚点条件。

- [ ] **Step 7: 运行格点与搜索测试**

Run:

```powershell
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R '^lpp_v3_legged\.' --no-tests=error --output-on-failure
```

Expected: 全部通过。

- [ ] **Step 8: 提交足式格点与搜索**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/legged/legged_lattice.hpp cpp/include/lunar_path_planner/v3/legged/legged_search.hpp cpp/src/legged/legged_lattice.cpp cpp/src/legged/legged_search.cpp cpp/src/legged/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/tests/unit/legged/legged_lattice_test.cpp cpp/tests/integration/legged/legged_search_test.cpp
git -C path-planner commit -m "feat(v3): search legged SE2.5D body poses"
```

---

### Task 4: 足式积空间走廊

**Files:**
- Create: `path-planner/cpp/include/lunar_path_planner/v3/legged/legged_corridor.hpp`
- Create: `path-planner/cpp/src/legged/legged_corridor.cpp`
- Create: `path-planner/cpp/tests/unit/legged/legged_corridor_test.cpp`
- Modify: `path-planner/cpp/src/legged/CMakeLists.txt`
- Modify: `path-planner/cpp/tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `BuildConvexCorridor`, `LeggedDiscretePlan`, height intervals and body envelope.
- Produces:

```cpp
struct LeggedCorridorSection {
  ConvexPolygon2d xy;
  HeightInterval z;
  YawInterval yaw;
  TerrainNormalEnvelope terrain_normal;
};

struct LeggedCorridor {
  std::vector<LeggedCorridorSection> sections;
};

Result<LeggedCorridor> BuildLeggedCorridor(
    const LeggedCorridorRequest&);
```

- [ ] **Step 1: 写出中心线安全但角点碰撞时拒绝积空间的测试**

```cpp
TEST(LeggedCorridorTest, RejectsUnsafeCartesianProductCorner) {
  const auto result = BuildLeggedCorridor(
      MakeCenterSafeButYawHeightCornerUnsafeRequest());

  EXPECT_FALSE(IsOk(result));
  EXPECT_EQ(std::get<Error>(result).code,
            ErrorCode::kInvalidArgument);
}
```

- [ ] **Step 2: 写出可通过分裂修复的测试**

```cpp
TEST(LeggedCorridorTest, SplitsYawIntervalWhenWholeProductIsUnsafe) {
  const auto result = BuildLeggedCorridor(
      MakeSplittableYawCorridorRequest());

  ASSERT_TRUE(IsOk(result));
  const auto& corridor = std::get<LeggedCorridor>(result);
  EXPECT_GT(corridor.sections.size(), 1u);
  EXPECT_TRUE(AllSectionProductsCertified(corridor));
}
```

- [ ] **Step 3: 构建并确认失败**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_legged_tests --parallel
```

Expected: 编译失败，提示 `BuildLeggedCorridor` 未定义。

- [ ] **Step 4: 实现 xy 基础走廊和 z/yaw 收紧**

从共享二维半平面走廊开始，结合每段高度可达交集、地形法向、顶部/侧向净空和 yaw 全范围机体扫掠进行收紧。

- [ ] **Step 5: 实现固定上限的分裂策略**

先分裂 yaw，再分裂 z；每次按区间中点确定性分裂，最多 `max_product_splits`。达到上限仍不能认证时失败并触发原语回退。

- [ ] **Step 6: 运行积空间走廊测试**

Run:

```powershell
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R '^lpp_v3_legged\.' --no-tests=error --output-on-failure
```

Expected: 全部通过。

- [ ] **Step 7: 提交足式走廊**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/legged/legged_corridor.hpp cpp/src/legged/legged_corridor.cpp cpp/src/legged/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/tests/unit/legged/legged_corridor_test.cpp
git -C path-planner commit -m "feat(v3): certify legged pose corridors"
```

---

### Task 5: yaw 独立机体样条优化

**Files:**
- Create: `path-planner/cpp/include/lunar_path_planner/v3/legged/legged_spline_optimizer.hpp`
- Create: `path-planner/cpp/src/legged/legged_spline_optimizer.cpp`
- Create: `path-planner/cpp/tests/unit/legged/legged_spline_optimizer_test.cpp`
- Modify: `path-planner/cpp/src/legged/CMakeLists.txt`
- Modify: `path-planner/cpp/tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `BoundedQpSolver`, `LeggedDiscretePlan`, `LeggedCorridor`.
- Produces:

```cpp
struct LeggedSplineRequest {
  const LeggedDiscretePlan& discrete_plan;
  const LeggedCorridor& corridor;
  LeggedSplineConfig config;
  std::span<const FrozenControlPoint> committed_points;
};

struct LeggedSplineResult {
  std::optional<ClampedCubicBSplinePath> body_spline;
  DurationNanoseconds estimated_execution_time;
  OptimizationTermination termination;
};

LeggedSplineResult OptimizeLeggedBodySpline(
    const LeggedSplineRequest&,
    BoundedQpSolver&);
```

- [ ] **Step 1: 写出侧移时 yaw 不对齐切向的测试**

```cpp
TEST(LeggedSplineOptimizerTest, PreservesIndependentYawForLateralMotion) {
  const auto result = OptimizeLeggedBodySpline(
      MakeLateralMotionRequest(), MakeQpSolver());

  ASSERT_TRUE(result.body_spline.has_value());
  EXPECT_NEAR(EvaluateYaw(*result.body_spline, 0.5), 0.0, 1e-8);
  EXPECT_NEAR(EvaluatePathTangentYaw(*result.body_spline, 0.5),
              std::numbers::pi / 2.0, 1e-3);
}
```

- [ ] **Step 2: 写出已承诺控制点不可修改测试**

冻结 x、y、z、yaw 四个分量，优化后逐位比较。

- [ ] **Step 3: 构建并确认失败**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_legged_tests --parallel
```

Expected: 编译失败，提示 `OptimizeLeggedBodySpline` 未定义。

- [ ] **Step 4: 实现 x/y/z/yaw 四维夹持三次 B 样条**

目标函数分别惩罚位置二阶差分、yaw 二阶差分和离散原语偏离；不得加入 yaw 与 xy 切向相等约束。

- [ ] **Step 5: 实现固定轮数 SCP 与积空间约束**

每个控制点及验证采样点必须落入同一 `LeggedCorridorSection` 的 xy、z、yaw 范围；机体外形和 terrain-normal envelope 在每轮后完整复核。

- [ ] **Step 6: 执行时间等价条件和整段失败**

若估算时长超过离散基线加 \(\Delta T_{\mathrm{eq}}\)，或任何非线性复核失败，返回空 spline。

- [ ] **Step 7: 运行足式样条测试**

Run:

```powershell
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R '^lpp_v3_legged\.' --no-tests=error --output-on-failure
```

Expected: 全部通过。

- [ ] **Step 8: 提交机体样条优化**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/legged/legged_spline_optimizer.hpp cpp/src/legged/legged_spline_optimizer.cpp cpp/src/legged/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/tests/unit/legged/legged_spline_optimizer_test.cpp
git -C path-planner commit -m "feat(v3): optimize legged body reference splines"
```

---

### Task 6: 机体系速度包络时序化

**Files:**
- Create: `path-planner/cpp/include/lunar_path_planner/v3/legged/legged_timing.hpp`
- Create: `path-planner/cpp/src/legged/legged_timing.cpp`
- Create: `path-planner/cpp/tests/unit/legged/legged_timing_test.cpp`
- Modify: `path-planner/cpp/src/legged/CMakeLists.txt`
- Modify: `path-planner/cpp/tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `GeometricPath`, `BodyFrameVelocityEnvelope`, terrain limits.
- Produces:

```cpp
struct LeggedTimingResult {
  MonotoneTimeScaling time_scaling;
  DurationNanoseconds duration;
  TimingDiagnostics diagnostics;
};

Result<LeggedTimingResult> ParameterizeLeggedBodyPath(
    const GeometricPath&,
    const BodyFrameVelocityEnvelope&,
    const LeggedTimingConfig&);
```

- [ ] **Step 1: 写出侧向速度受 lateral 而非 forward 上限约束的测试**

```cpp
TEST(LeggedTimingTest, LateralPathUsesLateralVelocityLimit) {
  auto limits = MakeVelocityEnvelope();
  limits.forward_mps.upper = 1.0;
  limits.lateral_mps.upper = 0.2;

  const auto result = ParameterizeLeggedBodyPath(
      MakePureLateralPath(), limits, MakeTimingConfig());

  ASSERT_TRUE(IsOk(result));
  EXPECT_GE(std::get<LeggedTimingResult>(result).duration.value,
            std::chrono::seconds{5});
}
```

- [ ] **Step 2: 写出竖直和 yaw 上限共同收紧测试**

```cpp
TEST(LeggedTimingTest, UsesVerticalAndYawBounds) {
  const auto result = ParameterizeLeggedBodyPath(
      MakeRisingTurningPath(), MakeVelocityEnvelope(),
      MakeTimingConfig());
  ASSERT_TRUE(IsOk(result));
  EXPECT_TRUE(AllBodyFrameRatesInsideEnvelope(
      std::get<LeggedTimingResult>(result)));
}
```

- [ ] **Step 3: 构建并确认失败**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_legged_tests --parallel
```

Expected: 编译失败，提示足式时序函数未定义。

- [ ] **Step 4: 沿 s 计算世界系导数并旋转到机体系**

```cpp
const Eigen::Rotation2Dd world_to_body(-yaw);
const Eigen::Vector2d body_xy =
    world_to_body * Eigen::Vector2d(dx_ds, dy_ds);
```

分别计算前/后、左右、竖直和 yaw 对 \(\dot s\) 的上限。

- [ ] **Step 5: 实现前向/后向可达传播**

使用固定最大采样数和加减速上限生成单调 `s(t)`；安全锚点处 \(\dot s=0\)，因此平移和 yaw 角速度均为零。

- [ ] **Step 6: 运行足式时序测试**

Run:

```powershell
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R '^lpp_v3_legged\.' --no-tests=error --output-on-failure
```

Expected: 全部通过。

- [ ] **Step 7: 提交足式时序化**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/legged/legged_timing.hpp cpp/src/legged/legged_timing.cpp cpp/src/legged/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/tests/unit/legged/legged_timing_test.cpp
git -C path-planner commit -m "feat(v3): parameterize legged body timing"
```

---

### Task 7: 足式参考流水线与有限保证

**Files:**
- Create: `path-planner/cpp/include/lunar_path_planner/v3/legged/legged_planner.hpp`
- Create: `path-planner/cpp/src/legged/legged_planner.cpp`
- Create: `path-planner/cpp/tests/integration/legged/legged_planner_test.cpp`
- Modify: `path-planner/cpp/src/legged/CMakeLists.txt`
- Modify: `path-planner/cpp/tests/CMakeLists.txt`

**Interfaces:**
- Consumes: Tasks 1–6、已经通过共享语义校验的 `PlanningRequest` 和
  `ResolvedTerminalSet`；不定义第二套上下文类型。
- Produces:

```cpp
class LeggedPlanner {
 public:
  Result<LeggedBodyReference> Plan(
      const PlanningRequest&,
      const ResolvedTerminalSet&) noexcept;
};
```

实现先从 `PlanningRequest.current_state` 提取并校验
`WheeledOrLeggedState`；地图、能力、算法配置和可选固定学习代价均只使用 request
内已经解析且不可变的对象。平台层不得重新按 handle 查询内容。

- [ ] **Step 1: 写出输出强制有限保证字段测试**

```cpp
TEST(LeggedPlannerTest, AlwaysDeclaresLimitedFeasibilityScope) {
  const auto result = MakeLeggedPlanner().Plan(
      MakeLeggedPlanningRequest(),
      MakeResolvedTerminalSet(TerminalKind::kGoal));

  ASSERT_TRUE(IsOk(result));
  const auto& reference = std::get<LeggedBodyReference>(result);
  EXPECT_EQ(reference.feasibility_scope,
            "body_geometry_and_terrain_thresholds_only");
  EXPECT_FALSE(reference.footstep_feasibility_guaranteed);
}
```

- [ ] **Step 2: 写出 spline 失败后整段原语回退测试**

```cpp
TEST(LeggedPlannerTest, SplineFailureReturnsPrimitiveChainOnly) {
  const auto result = MakePlannerWithFailingQp().Plan(
      MakeLeggedPlanningRequest(),
      MakeResolvedTerminalSet(TerminalKind::kGoal));

  ASSERT_TRUE(IsOk(result));
  EXPECT_TRUE(std::holds_alternative<ValidatedPrimitiveChain>(
      std::get<LeggedBodyReference>(result).geometric_path));
}

TEST(LeggedPlannerTest, FinalReferenceHasStableIdentityOriginAndJcsHash) {
  const auto request = MakeLeggedPlanningRequest();
  const auto terminal =
      MakeResolvedTerminalSet(TerminalKind::kGoal);
  const auto first = MakeLeggedPlanner().Plan(request, terminal);
  const auto second = MakeLeggedPlanner().Plan(request, terminal);
  ASSERT_TRUE(IsOk(first));
  ASSERT_TRUE(IsOk(second));
  const auto& a = std::get<LeggedBodyReference>(first);
  const auto& b = std::get<LeggedBodyReference>(second);
  EXPECT_FALSE(a.reference_id.empty());
  EXPECT_EQ(a.reference_id, b.reference_id);
  EXPECT_EQ(a.reference_time_origin, request.request_time);
  EXPECT_EQ(a.reference_hash, CanonicalReferenceHash(a));
  EXPECT_EQ(a.reference_hash, b.reference_hash);
}
```

- [ ] **Step 3: 构建并确认失败**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_legged_tests --parallel
```

Expected: 编译失败，提示 `LeggedPlanner` 未定义。

- [ ] **Step 4: 实现连续处理事务和原语回退**

样条候选只有在走廊、时序、碰撞、地形和安全锚点全部验证后才进入 `LeggedBodyReference`；否则重建完整 `ValidatedPrimitiveChain` 并重新时序、验证。

- [ ] **Step 5: 固定 `reference_point_id` 和诊断包络**

输出参考点必须逐字等于能力 profile；roll/pitch 只从地形法向和能力边界计算诊断包络，不输出权威 roll/pitch 曲线。

全部几何、时序、terrain 和 anchor 终验通过后才构造 wire DTO：

- `reference_id` 由 `request_id + selected_candidate_id + "LEGGED"` 经稳定 ID
  工厂生成；
- `reference_time_origin = request.request_time`；
- `reference_hash` 用共享 RFC 8785 JCS/SHA-256 对省略该字段的完整
  `LeggedBodyReference` 计算；
- hash 后再执行 semantic validation 和 encode/decode/hash round-trip。

- [ ] **Step 6: 加入 schema 往返和禁止字段测试**

编码后用 Draft 2020-12 schema fixture 验证，并断言 JSON 文档不含：

```text
footsteps
gait
contacts
contact_forces
quaternion
roll_reference
pitch_reference
```

- [ ] **Step 7: 运行足式流水线测试**

Run:

```powershell
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R '^lpp_v3_legged\.' --no-tests=error --output-on-failure
```

Expected: 全部通过。

- [ ] **Step 8: 提交足式流水线**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/legged/legged_planner.hpp cpp/src/legged/legged_planner.cpp cpp/src/legged/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/tests/integration/legged/legged_planner_test.cpp
git -C path-planner commit -m "feat(v3): assemble authoritative legged body references"
```

---

### Task 8: 足式属性测试、故障矩阵和基准 fixture

**Files:**
- Create: `path-planner/cpp/tests/property/legged/legged_reference_property_test.cpp`
- Create: `path-planner/cpp/tests/integration/legged/legged_fault_matrix_test.cpp`
- Create: `path-planner/cpp/benchmarks/legged_planner_benchmark.cpp`
- Create: `path-planner/cpp/benchmarks/fixtures/legged_declared_suite.json`
- Modify: `path-planner/cpp/tests/CMakeLists.txt`
- Modify: `path-planner/cpp/benchmarks/CMakeLists.txt`

**Interfaces:**
- Consumes: `LeggedPlanner`, codec, deterministic fixture generator and benchmark harness.
- Produces: 足式平台验收证据；不改变运行时 API。

- [ ] **Step 1: 写出固定种子属性测试**

```cpp
TEST(LeggedReferencePropertyTest, GeneratedReferencesStayWithinBodyContract) {
  for (std::uint64_t seed = 0; seed < 1000; ++seed) {
    const auto result = PlanGeneratedLeggedCase(seed);
    if (!IsOk(result)) {
      continue;
    }
    const auto& reference = std::get<LeggedBodyReference>(result);
    EXPECT_TRUE(BodyPathInsideCertifiedCorridor(reference));
    EXPECT_TRUE(BodyRatesInsideEnvelope(reference));
    EXPECT_TRUE(EndsAtSafeStopAnchor(reference));
    EXPECT_FALSE(reference.footstep_feasibility_guaranteed);
  }
}
```

- [ ] **Step 2: 建立故障矩阵**

覆盖：

```text
unknown_or_low_confidence
slope_limit
roughness_limit
step_limit
gap_limit
top_clearance
height_interval_disconnect
cartesian_product_corner_unsafe
qp_infeasible
timing_resource_limit
continuous_validation_inconclusive
```

- [ ] **Step 3: 加入字节级确定性测试**

固定输入、配置和线程策略重复规划 100 次，JCS/SHA-256 必须一致；不得按并行完成先后选择候选。

- [ ] **Step 4: 运行足式完整测试**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_legged_tests --parallel
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R '^lpp_v3_legged\.' --no-tests=error --output-on-failure
```

Expected: 全部通过。

- [ ] **Step 5: 建立足式基准**

```cpp
static void BM_LeggedDeclaredSuite(benchmark::State& state) {
  const auto fixture = LoadLeggedBenchmarkFixture();
  LeggedPlanner planner = BuildPlanner(fixture);
  for (auto _ : state) {
    const auto result = planner.Plan(
        fixture.request, fixture.resolved_terminal);
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(BM_LeggedDeclaredSuite)->UseRealTime();
```

- [ ] **Step 6: 导出原始基准 JSON**

Run:

```powershell
Push-Location path-planner/cpp
cmake --preset windows-msvc-release
cmake --build --preset windows-msvc-release --target lpp_v3_legged_benchmark --parallel
Pop-Location
D:/xunce/build/path-planner-v3/windows-msvc-release/benchmarks/lpp_v3_legged_benchmark.exe --benchmark_out=D:/xunce/out/path-planner-v3/legged-benchmark.json --benchmark_out_format=json
```

Expected: 完整运行所有样本，不设置单次 1 秒截止。

- [ ] **Step 7: 提交足式验收资产**

```powershell
git -C path-planner add cpp/tests/property/legged/legged_reference_property_test.cpp cpp/tests/integration/legged/legged_fault_matrix_test.cpp cpp/benchmarks/legged_planner_benchmark.cpp cpp/benchmarks/fixtures/legged_declared_suite.json cpp/tests/CMakeLists.txt cpp/benchmarks/CMakeLists.txt
git -C path-planner commit -m "test(v3): cover legged body planner contracts"
```

---

## Legged Plan Completion Gate

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --parallel
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R '^lpp_v3_legged\.' --no-tests=error --output-on-failure
git -C path-planner status --short
```

预期：

- 所有 `legged_` 测试通过。
- 输出只含固定机体参考点 x、y、z、yaw 与时序。
- yaw 与路径切向独立。
- 完整积空间走廊经过验证。
- 不存在任何足端、步态、接触或四元数权威字段。
- 两个有限保证字段不可省略。
- 连续处理失败整段回退。
- 末端为安全停止锚点。
