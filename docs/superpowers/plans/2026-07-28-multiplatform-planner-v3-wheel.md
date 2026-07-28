# Multiplatform Planner v3 Wheeled Platform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现支持前进、倒车和原地旋转的轮式 C++20 平台层，输出经过连续验证的类型化路径与时序参考，并在任一连续处理失败时完整回退到认证运动原语。

**Architecture:** `WheelLatticeAdapter` 把版本化运动原语目录接入共享 ARA*，以预计执行时间为主目标；离散候选经过轮式走廊收紧、固定迭代 SCP、路径时序化和完整扫掠验证。最终输出 `WheeledReference`，但原子 bundle 组装由集成计划负责。

**Tech Stack:** C++20、CMake 3.28+、Eigen 5.0.1、OSQP 1.0.0（`LPP_V3_ENABLE_OSQP=ON` 时）、GoogleTest 1.17.0、vcpkg manifest。

## Global Constraints

- 只在 `path-planner/cpp/**` 和本计划列出的 C++ 测试 fixture 中实现；不调用现有 Python 规划代码。
- 命名空间固定为 `lunar::planning::v3`。
- 本分卷链接 `lpp_v3_contracts` 与 `lpp_v3_common`，产出
  `lpp_v3_wheel`、`lpp_v3_wheel_tests` 和 `lpp_v3_wheel_benchmark`。
- `lpp_v3_wheel_tests` 必须通过 `gtest_discover_tests(... TEST_PREFIX
  "lpp_v3_wheel.")` 注册；所有过滤式 CTest gate 使用该前缀并带
  `--no-tests=error`。
- 轮式平台允许前进、倒车和原地旋转，不声明 Ackermann 可行。
- 离散状态固定为 `(ix, iy, iyaw, motion_mode)`；速度不得加入格点状态。
- 主搜索代价固定为预计执行时间；能耗、风险和平滑性只在时间等价池内比较。
- `DriveSegment` 的 `s(t)` 始终从 0 单调增加到 1；倒车通过负的车体前向速度表达。
- 前进、倒车和自旋模式切换点必须满足线速度与 yaw 角速度同时为零。
- 任何走廊、平滑、时序或连续验证失败均整段回退，不允许半验证混合路径。
- 每个可激活承诺前缀终止于零速度安全停止锚点。
- 算法只受迭代、扩展、候选和内存上限约束，不读取 1 秒墙钟截止。
- 构建目录固定为 `D:/xunce/build/path-planner-v3/windows-msvc-debug`；不得把依赖和大型构建产物写入 C 盘工作树。

---

### Task 1: 轮式能力视图与运动原语目录

**Files:**
- Create: `path-planner/cpp/include/lunar_path_planner/v3/wheel/wheel_capability.hpp`
- Create: `path-planner/cpp/include/lunar_path_planner/v3/wheel/wheel_primitive.hpp`
- Create: `path-planner/cpp/src/wheel/wheel_capability.cpp`
- Create: `path-planner/cpp/src/wheel/wheel_primitive.cpp`
- Create: `path-planner/cpp/tests/unit/wheel/wheel_primitive_test.cpp`
- Modify: `path-planner/cpp/src/wheel/CMakeLists.txt`
- Modify: `path-planner/cpp/tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `SafetyCapabilityProfile`, `ContentRef`, `PoseXyzYaw`,
  `DurationNanoseconds` and `SecondaryCostVector` from the shared targets.
- Produces:

```cpp
enum class WheelMotionMode {
  kStart,
  kForward,
  kReverse,
  kSpinClockwise,
  kSpinCounterClockwise,
};

enum class WheelPrimitiveKind {
  kDriveLine,
  kDriveArc,
  kSpin,
  kModeSwitch,
};

struct WheelPrimitive {
  PrimitiveId id;
  WheelPrimitiveKind kind;
  WheelMotionMode source_mode;
  WheelMotionMode target_mode;
  PoseXyzYaw relative_end;
  DurationNanoseconds nominal_duration;
  SecondaryCostVector secondary_costs;
  WheelSweepDescriptor sweep;
};

class WheelPrimitiveCatalog {
 public:
  static Result<WheelPrimitiveCatalog> Create(
      const SafetyCapabilityProfile& profile);
  std::span<const WheelPrimitive> Outgoing(WheelMotionMode mode) const noexcept;
  const ContentRef& content_ref() const noexcept;
};
```

- [ ] **Step 1: 写出目录拒绝无倒车或无自旋认证原语的失败测试**

```cpp
TEST(WheelPrimitiveCatalogTest, RejectsIncompleteRequiredModes) {
  auto profile = MakeWheelCapabilityFixture();
  profile.wheel().primitive_specs.erase(
      std::ranges::find_if(
          profile.wheel().primitive_specs,
          [](const auto& p) { return p.kind == WheelPrimitiveKind::kSpin; }));

  const auto result = WheelPrimitiveCatalog::Create(profile);

  ASSERT_FALSE(IsOk(result));
  EXPECT_EQ(std::get<Error>(result).code,
            ErrorCode::kInvalidArgument);
}
```

- [ ] **Step 2: 构建测试并确认因类型尚未定义而失败**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_wheel_tests --parallel
```

Expected: 编译失败，提示 `WheelPrimitiveCatalog` 或相关枚举未定义。

- [ ] **Step 3: 定义模式、原语、扫掠描述和只读目录头文件**

在 `wheel_primitive.hpp` 写入上述接口，并让目录只暴露 `std::span<const WheelPrimitive>`；不得返回可变容器。
在 `src/wheel/CMakeLists.txt` 首次创建 `lpp_v3_wheel`，公开链接
`lpp_v3_contracts` 与 `lpp_v3_common`；在测试 CMake 中把本分卷测试源汇总为
`lpp_v3_wheel_tests`。后续任务只向这两个既有目标追加源文件。

- [ ] **Step 4: 实现 profile 到目录的严格转换**

实现时逐项验证：

```cpp
if (!has_forward || !has_reverse || !has_spin_cw || !has_spin_ccw) {
  return Error{ErrorCode::kInvalidArgument,
               "/content/motion_primitives",
               "INCOMPLETE_WHEEL_PRIMITIVE_CATALOG"};
}
if (spec.nominal_duration.value.count() <= 0) {
  return Error{ErrorCode::kInvalidArgument,
               "/content/motion_primitives/nominal_duration_ns",
               "INVALID_WHEEL_PRIMITIVE_DURATION"};
}
```

wire 的 `nominal_duration_ns` 在 codec 层无损解析为
`DurationNanoseconds nominal_duration`；目录构造不得再引入浮点秒字段或做
纳秒→秒→纳秒往返。

模式切换原语的 `relative_end` 必须为零位移，并显式包含停止、启动或换向持续时间。
序列化到共享 `ValidatedPrimitiveChain` 时映射为
`PrimitiveKind::kStopAndSwitch` / wire `STOP_AND_SWITCH`，不得伪装成零长度 drive
或丢弃其 `validation_ref`。

- [ ] **Step 5: 增加目录确定性和 hash 一致性测试**

```cpp
TEST(WheelPrimitiveCatalogTest, StableOrderDoesNotDependOnInputOrder) {
  auto a = MakeWheelCapabilityFixture();
  auto b = a;
  std::ranges::reverse(b.wheel().primitive_specs);

  const auto catalog_a =
      std::get<WheelPrimitiveCatalog>(WheelPrimitiveCatalog::Create(a));
  const auto catalog_b =
      std::get<WheelPrimitiveCatalog>(WheelPrimitiveCatalog::Create(b));

  EXPECT_EQ(PrimitiveIds(catalog_a), PrimitiveIds(catalog_b));
  EXPECT_EQ(catalog_a.content_ref(), catalog_b.content_ref());
}
```

- [ ] **Step 6: 运行轮式目录测试**

Run:

```powershell
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R '^lpp_v3_wheel\.' --no-tests=error --output-on-failure
```

Expected: 全部通过。

- [ ] **Step 7: 提交目录**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/wheel/wheel_capability.hpp cpp/include/lunar_path_planner/v3/wheel/wheel_primitive.hpp cpp/src/wheel/wheel_capability.cpp cpp/src/wheel/wheel_primitive.cpp cpp/src/wheel/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/tests/unit/wheel/wheel_primitive_test.cpp
git -C path-planner commit -m "feat(v3): add certified wheel primitive catalog"
```

---

### Task 2: `SE(2)` 模式格点适配器

**Files:**
- Create: `path-planner/cpp/include/lunar_path_planner/v3/wheel/wheel_lattice.hpp`
- Create: `path-planner/cpp/src/wheel/wheel_lattice.cpp`
- Create: `path-planner/cpp/tests/unit/wheel/wheel_lattice_test.cpp`
- Modify: `path-planner/cpp/src/wheel/CMakeLists.txt`
- Modify: `path-planner/cpp/tests/CMakeLists.txt`

**Interfaces:**
- Consumes:

```cpp
template <class State, class Edge, class Adapter>
SearchResult<State, Edge> RunAraStar(
    const Adapter& adapter,
    const SearchProblem<State>& problem,
    const AraStarConfig& config);
```

- Produces:

```cpp
struct WheelLatticeState {
  std::int32_t ix;
  std::int32_t iy;
  std::int32_t iyaw;
  WheelMotionMode motion_mode;
};

struct WheelLatticeEdge {
  WheelLatticeState source;
  WheelLatticeState target;
  PrimitiveId primitive_id;
  DurationNanoseconds transition_time;
  SecondaryCostVector secondary_costs;
};

StateKey EncodeWheelStateKey(
    std::int32_t ix,
    std::int32_t iy,
    std::int32_t iyaw,
    WheelMotionMode mode) noexcept;

class WheelLatticeAdapter {
 public:
  StateKey Key(const WheelLatticeState&) const noexcept;
  std::vector<SearchTransition<WheelLatticeState, WheelLatticeEdge>> Expand(
      const WheelLatticeState&) const;
  bool HardFeasible(
      const SearchTransition<WheelLatticeState, WheelLatticeEdge>&) const;
  DurationNanoseconds TransitionTime(
      const SearchTransition<WheelLatticeState, WheelLatticeEdge>&) const noexcept;
  DurationNanoseconds AdmissibleTimeHeuristic(
      const WheelLatticeState&,
      const SearchProblem<WheelLatticeState>&) const noexcept;
  SecondaryCostVector SecondaryCosts(
      const SearchTransition<WheelLatticeState, WheelLatticeEdge>&) const noexcept;
  bool IsTerminal(
      const WheelLatticeState&,
      const SearchProblem<WheelLatticeState>&) const noexcept;
};
```

- [ ] **Step 1: 写出速度不属于格点键的测试**

```cpp
TEST(WheelLatticeTest, StateKeyContainsPoseAndModeOnly) {
  const WheelLatticeState state{3, 7, 5, WheelMotionMode::kReverse};
  const auto key = MakeAdapter().Key(state);

  EXPECT_EQ(key, EncodeWheelStateKey(
                     3, 7, 5, WheelMotionMode::kReverse));
}
```

- [ ] **Step 2: 写出倒车和自旋均可扩展的测试**

```cpp
TEST(WheelLatticeTest, ExpandsReverseAndSpinEdges) {
  const auto edges = MakeAdapter().Expand(
      WheelLatticeState{10, 10, 0, WheelMotionMode::kStart});

  EXPECT_TRUE(ContainsTargetMode(edges, WheelMotionMode::kReverse));
  EXPECT_TRUE(ContainsTargetMode(edges, WheelMotionMode::kSpinClockwise));
  EXPECT_TRUE(ContainsTargetMode(
      edges, WheelMotionMode::kSpinCounterClockwise));
}
```

- [ ] **Step 3: 构建并确认测试失败**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_wheel_tests --parallel
```

Expected: 编译失败，提示 `WheelLatticeAdapter` 未定义。

- [ ] **Step 4: 实现位姿量化、模式键和稳定扩展顺序**

扩展顺序固定为：

```cpp
constexpr std::array kModeOrder{
    WheelMotionMode::kForward,
    WheelMotionMode::kReverse,
    WheelMotionMode::kSpinClockwise,
    WheelMotionMode::kSpinCounterClockwise,
};
```

同一模式内按 `PrimitiveId` 升序。边界外、未知区域、硬地形超限和扫掠碰撞边在 `HardFeasible` 中过滤。

- [ ] **Step 5: 实现时间下界启发式**

```cpp
const double translation_lb_s =
    xy_distance_m / capability_.max_abs_drive_speed_mps;
const double rotation_lb_s =
    yaw_distance_rad / capability_.max_spin_rate_radps;
return SecondsToDuration(std::max(translation_lb_s, rotation_lb_s));
```

不得加入能耗、风险或平滑性。

- [ ] **Step 6: 增加模式切换时间测试**

```cpp
TEST(WheelLatticeTest, ReverseSwitchIncludesStopAndRestartTime) {
  const auto edge = FindEdge(
      MakeAdapter(),
      WheelMotionMode::kForward,
      WheelMotionMode::kReverse);

  EXPECT_EQ(edge.transition_time,
            MakeCapability().forward_to_reverse_switch_time);
  EXPECT_EQ(edge.source.ix, edge.target.ix);
  EXPECT_EQ(edge.source.iy, edge.target.iy);
}
```

- [ ] **Step 7: 运行格点测试**

Run:

```powershell
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R '^lpp_v3_wheel\.' --no-tests=error --output-on-failure
```

Expected: 全部通过。

- [ ] **Step 8: 提交格点适配器**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/wheel/wheel_lattice.hpp cpp/src/wheel/wheel_lattice.cpp cpp/src/wheel/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/tests/unit/wheel/wheel_lattice_test.cpp
git -C path-planner commit -m "feat(v3): add wheeled SE2 mode lattice"
```

---

### Task 3: 轮式搜索、时间等价池和分段

**Files:**
- Create: `path-planner/cpp/include/lunar_path_planner/v3/wheel/wheel_search.hpp`
- Create: `path-planner/cpp/src/wheel/wheel_search.cpp`
- Create: `path-planner/cpp/tests/integration/wheel/wheel_search_test.cpp`
- Modify: `path-planner/cpp/src/wheel/CMakeLists.txt`
- Modify: `path-planner/cpp/tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `RunAraStar`, `CandidateRanker::BuildTimeEquivalentPool`, `ResolvedTerminalSet`, `WheelLatticeAdapter`.
- Produces:

```cpp
struct WheelDiscreteSegment {
  WheelMotionMode mode;
  std::vector<WheelLatticeEdge> edges;
  DurationNanoseconds expected_time;
  SecondaryCostVector secondary_costs;
};

struct WheelDiscretePlan {
  std::vector<WheelDiscreteSegment> segments;
  SecondaryCostVector total_secondary_costs;
  SafeStopAnchor safe_stop_anchor;
  SearchDiagnostics diagnostics;
};

Result<WheelDiscretePlan> PlanWheelDiscrete(
    const WheelPlanningProblem& problem,
    const WheelPrimitiveCatalog& primitives,
    const AraStarConfig& search_config);
```

- [ ] **Step 1: 写出最短时间优先于低能耗的测试**

```cpp
TEST(WheelSearchTest, TimeDominatesEnergyOutsideEquivalencePool) {
  auto problem = MakeTwoRouteWheelProblem(
      /*fast_time_s=*/10.0, /*fast_energy=*/20.0,
      /*slow_time_s=*/10.6, /*slow_energy=*/1.0,
      /*time_equivalence_s=*/0.5);

  const auto result = PlanWheelDiscrete(
      problem, MakePrimitiveCatalog(), MakeAraConfig());

  ASSERT_TRUE(IsOk(result));
  const auto& plan = std::get<WheelDiscretePlan>(result);
  EXPECT_EQ(plan.segments.front().expected_time.value,
            std::chrono::seconds{10});
}
```

- [ ] **Step 2: 写出等价池内能耗打破并列的测试**

```cpp
TEST(WheelSearchTest, EnergyBreaksTieInsideTimeEquivalencePool) {
  auto problem = MakeTwoRouteWheelProblem(
      10.0, 20.0, 10.4, 1.0, 0.5);

  const auto result = PlanWheelDiscrete(
      problem, MakePrimitiveCatalog(), MakeAraConfig());

  ASSERT_TRUE(IsOk(result));
  EXPECT_EQ(std::get<WheelDiscretePlan>(result)
                .total_secondary_costs.energy,
            1.0);
}
```

- [ ] **Step 3: 构建并确认测试失败**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_wheel_tests --parallel
```

Expected: 编译失败，提示 `PlanWheelDiscrete` 未定义。

- [ ] **Step 4: 实现 ARA* 调用和候选池排序**

只把 `TransitionTime` 传入主搜索键；对已完成候选调用：

```cpp
const auto pool = CandidateRanker::BuildTimeEquivalentPool(
    candidates, problem.algorithm.time_equivalence_tolerance);
const auto selected = StableLexicographicMinimum(
    pool, &Candidate::energy, &Candidate::risk,
    &Candidate::smoothness, &Candidate::stable_id);
```

- [ ] **Step 5: 按模式边界切分离散边**

同方向行驶边保留在同一 `WheelDiscreteSegment`。前进/倒车、自旋方向或行驶/自旋切换必须创建新分段。

- [ ] **Step 6: 验证末端安全停止锚点**

如果终端候选没有静态地形安全锚点或无法附加确定性减速边，返回 `NO_KNOWN_SAFE_ROUTE`，不得发布移动中终点。

- [ ] **Step 7: 运行搜索集成测试**

Run:

```powershell
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R '^lpp_v3_wheel\.' --no-tests=error --output-on-failure
```

Expected: 全部通过。

- [ ] **Step 8: 提交搜索与分段**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/wheel/wheel_search.hpp cpp/src/wheel/wheel_search.cpp cpp/src/wheel/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/tests/integration/wheel/wheel_search_test.cpp
git -C path-planner commit -m "feat(v3): select and segment wheel lattice plans"
```

---

### Task 4: 轮式走廊收紧与解析扫掠验证

**Files:**
- Create: `path-planner/cpp/include/lunar_path_planner/v3/wheel/wheel_corridor.hpp`
- Create: `path-planner/cpp/include/lunar_path_planner/v3/wheel/wheel_sweep_validator.hpp`
- Create: `path-planner/cpp/src/wheel/wheel_corridor.cpp`
- Create: `path-planner/cpp/src/wheel/wheel_sweep_validator.cpp`
- Create: `path-planner/cpp/tests/unit/wheel/wheel_corridor_test.cpp`
- Create: `path-planner/cpp/tests/unit/wheel/wheel_sweep_validator_test.cpp`
- Modify: `path-planner/cpp/src/wheel/CMakeLists.txt`
- Modify: `path-planner/cpp/tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `CorridorResult BuildConvexCorridor(const CorridorRequest&)`, wheel collision envelope and primitive sweeps.
- Produces:

```cpp
struct WheelCorridorRequest {
  std::span<const WheelLatticeEdge> edges;
  const SafeProjection& projection;
  const WheelCollisionEnvelope& envelope;
  WheelCorridorConfig config;
};

CorridorResult BuildWheelCorridor(const WheelCorridorRequest&);

class WheelSweepValidator {
 public:
  ValidationReport ValidatePrimitiveChain(
      const ValidatedPrimitiveChain&) const;
  ValidationReport ValidateSpline(
      const ClampedCubicBSplinePath&,
      const MonotoneTimeScaling&) const;
  ValidationReport ValidateSpin(
      const Vec3& fixed_position,
      const PiecewiseCubicScalarTrajectory& yaw) const;
};
```

- [ ] **Step 1: 写出非圆形足迹自旋扫掠失败测试**

```cpp
TEST(WheelSweepValidatorTest, RejectsCornerCollisionDuringSpin) {
  auto scene = MakeNarrowSpinScene();
  auto yaw = MakeYawLaw(0.0, std::numbers::pi / 2.0);

  const auto result =
      scene.validator.ValidateSpin(scene.position, yaw);

  EXPECT_FALSE(result.ok());
  EXPECT_TRUE(result.Contains("/segments", "SWEPT_COLLISION"));
}
```

- [ ] **Step 2: 写出相邻走廊无重叠时失败的测试**

```cpp
TEST(WheelCorridorTest, FailsClosedWhenSectionsDoNotOverlap) {
  const auto result = BuildWheelCorridor(
      MakeDisconnectedCorridorRequest());

  EXPECT_FALSE(result.ok());
  EXPECT_EQ(result.reason,
            CorridorFailureReason::kNoConservativeOverlap);
}
```

- [ ] **Step 3: 构建并确认失败**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_wheel_tests --parallel
```

Expected: 编译失败，提示轮式走廊或扫掠验证器未定义。

- [ ] **Step 4: 实现非圆形足迹、yaw 和曲率收紧**

每个二维凸截面按对应 yaw 区间验证完整足迹顶点；自旋使用固定位置下完整 yaw 区间的旋转包络，倒车使用相同车体外形但按倒车原语扫掠。

- [ ] **Step 5: 实现直线、圆弧和样条验证入口**

直线和圆弧使用解析包围；B 样条使用控制点凸包快速排除与有界自适应细分确认。达到细分深度仍不确定时返回失败。

- [ ] **Step 6: 增加未知单元和低净空失败测试**

```cpp
TEST(WheelSweepValidatorTest, UnknownCellIsNeverClearance) {
  auto scene = MakeSceneWithUnknownCellUnderArc();
  EXPECT_FALSE(
      scene.validator.ValidatePrimitiveChain(scene.chain).ok());
}
```

- [ ] **Step 7: 运行走廊与扫掠测试**

Run:

```powershell
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R '^lpp_v3_wheel\.' --no-tests=error --output-on-failure
```

Expected: 全部通过。

- [ ] **Step 8: 提交轮式走廊与验证器**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/wheel/wheel_corridor.hpp cpp/include/lunar_path_planner/v3/wheel/wheel_sweep_validator.hpp cpp/src/wheel/wheel_corridor.cpp cpp/src/wheel/wheel_sweep_validator.cpp cpp/src/wheel/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/tests/unit/wheel/wheel_corridor_test.cpp cpp/tests/unit/wheel/wheel_sweep_validator_test.cpp
git -C path-planner commit -m "feat(v3): validate wheeled corridors and sweeps"
```

---

### Task 5: 固定迭代轮式 B 样条 SCP

**Files:**
- Create: `path-planner/cpp/include/lunar_path_planner/v3/wheel/wheel_spline_optimizer.hpp`
- Create: `path-planner/cpp/src/wheel/wheel_spline_optimizer.cpp`
- Create: `path-planner/cpp/tests/unit/wheel/wheel_spline_optimizer_test.cpp`
- Modify: `path-planner/cpp/src/wheel/CMakeLists.txt`
- Modify: `path-planner/cpp/tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `BoundedQpSolver`, `CorridorResult`, `WheelDiscreteSegment`.
- Produces:

```cpp
struct WheelSplineOptimizationRequest {
  const WheelDiscreteSegment& discrete_segment;
  const CorridorResult& corridor;
  const WheelCapabilityView& capability;
  WheelSplineConfig config;
  std::span<const FrozenControlPoint> committed_points;
};

struct WheelSplineOptimizationResult {
  std::optional<ClampedCubicBSplinePath> spline;
  OptimizationTermination termination;
  DurationNanoseconds estimated_execution_time;
};

WheelSplineOptimizationResult OptimizeWheelSpline(
    const WheelSplineOptimizationRequest&,
    BoundedQpSolver&);
```

- [ ] **Step 1: 写出已承诺控制点冻结测试**

```cpp
TEST(WheelSplineOptimizerTest, NeverMovesCommittedControlPoints) {
  auto request = MakeWheelSplineRequest();
  request.committed_points = FreezeFirstTwoControlPoints(request);

  const auto result = OptimizeWheelSpline(request, MakeQpSolver());

  ASSERT_TRUE(result.spline.has_value());
  EXPECT_EQ(result.spline->control_points[0],
            request.committed_points[0].value);
  EXPECT_EQ(result.spline->control_points[1],
            request.committed_points[1].value);
}
```

- [ ] **Step 2: 写出时间超出等价容差时拒绝样条的测试**

```cpp
TEST(WheelSplineOptimizerTest, RejectsSlowerThanDiscreteTolerance) {
  auto request = MakeWheelSplineRequest();
  request.config.time_equivalence_tolerance = 100ms;
  auto solver = MakeQpSolverReturningLongDetour();

  const auto result = OptimizeWheelSpline(request, solver);

  EXPECT_FALSE(result.spline.has_value());
  EXPECT_EQ(result.termination,
            OptimizationTermination::kTimeToleranceExceeded);
}
```

- [ ] **Step 3: 构建并确认失败**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_wheel_tests --parallel
```

Expected: 编译失败，提示 `OptimizeWheelSpline` 未定义。

- [ ] **Step 4: 实现夹持三次 B 样条初始化**

端点重复四次，内部 knot 按离散弧长归一化；控制点初值来自原语端点和必要的圆弧中间点。前进 yaw 为切向，倒车 yaw 为切向加 \(\pi\)，并全程解缠。

- [ ] **Step 5: 实现固定轮数 SCP**

每轮 QP 包含：

```cpp
objective =
    config.path_deviation_weight * path_deviation +
    config.second_difference_weight * curvature_smoothness;
constraints = {
    corridor_half_planes,
    trust_region,
    endpoint_equalities,
    frozen_control_points,
    linearized_curvature_bound,
};
```

只运行 `config.max_scp_iterations`，OSQP 只运行固定最大迭代数并使用固定容差；不得读取墙钟。

- [ ] **Step 6: 对最终样条重新执行非线性约束检查**

线性化 QP 成功不等于候选有效。曲率、走廊、端点、yaw 关系或时间条件任一失败时，返回无 spline。

- [ ] **Step 7: 运行样条优化测试**

Run:

```powershell
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R '^lpp_v3_wheel\.' --no-tests=error --output-on-failure
```

Expected: 全部通过。

- [ ] **Step 8: 提交样条优化**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/wheel/wheel_spline_optimizer.hpp cpp/src/wheel/wheel_spline_optimizer.cpp cpp/src/wheel/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/tests/unit/wheel/wheel_spline_optimizer_test.cpp
git -C path-planner commit -m "feat(v3): optimize wheel splines with bounded SCP"
```

---

### Task 6: 行驶与自旋时序化

**Files:**
- Create: `path-planner/cpp/include/lunar_path_planner/v3/wheel/wheel_timing.hpp`
- Create: `path-planner/cpp/src/wheel/wheel_timing.cpp`
- Create: `path-planner/cpp/tests/unit/wheel/wheel_timing_test.cpp`
- Modify: `path-planner/cpp/src/wheel/CMakeLists.txt`
- Modify: `path-planner/cpp/tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `GeometricPath`, `WheelCapabilityView`, terrain speed-limit layer.
- Produces:

```cpp
struct WheelTimingResult {
  MonotoneTimeScaling time_scaling;
  DurationNanoseconds duration;
  TimingDiagnostics diagnostics;
};

Result<WheelTimingResult> ParameterizeWheelDrive(
    const GeometricPath&,
    DriveDirection,
    const WheelTimingLimits&,
    const WheelTimingConfig&);

Result<PiecewiseCubicScalarTrajectory> ParameterizeWheelSpin(
    double yaw_start_rad,
    double yaw_end_unwrapped_rad,
    const SpinTimingLimits&);
```

- [ ] **Step 1: 写出倒车 `s` 单调且有符号速度为负的测试**

```cpp
TEST(WheelTimingTest, ReverseKeepsSIncreasingAndBodySpeedNegative) {
  const auto result = ParameterizeWheelDrive(
      MakeStraightPath(), DriveDirection::kReverse,
      MakeTimingLimits(), MakeTimingConfig());

  ASSERT_TRUE(IsOk(result));
  const auto& timing = std::get<WheelTimingResult>(result);
  EXPECT_TRUE(IsMonotoneNondecreasing(timing.time_scaling));
  EXPECT_LT(DerivedBodyForwardSpeed(
                timing.time_scaling, DriveDirection::kReverse, 500ms),
            0.0);
}
```

- [ ] **Step 2: 写出终点制动约束测试**

```cpp
TEST(WheelTimingTest, EndsAtZeroSpeedAtSafeStopAnchor) {
  const auto result = ParameterizeWheelDrive(
      MakeStraightPath(), DriveDirection::kForward,
      MakeTimingLimits(), MakeTimingConfig());

  ASSERT_TRUE(IsOk(result));
  const auto& timing = std::get<WheelTimingResult>(result);
  EXPECT_NEAR(EvaluateDerivative(
                  timing.time_scaling,
                  timing.time_scaling.segments.back().end_offset),
              0.0, 1e-10);
}
```

- [ ] **Step 3: 构建并确认失败**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_wheel_tests --parallel
```

Expected: 编译失败，提示时序函数未定义。

- [ ] **Step 4: 实现自适应 s 采样和局部上限**

每点计算：

```cpp
v_limit = MinFinite({
    capability.forward_or_reverse_speed_limit,
    YawRateToPathSpeedLimit(curvature),
    LateralAccelerationToSpeedLimit(curvature),
    terrain_speed_limit,
    clearance_speed_limit,
    traction_speed_limit,
});
```

在曲率、坡度或净空变化超过配置阈值时细分，且不得超过 `max_timing_samples`。

- [ ] **Step 5: 实现前向加速和后向制动传播**

先前向传播可达速度平方上界，再从终端零速度向后传播制动上界；取两者最小值并转换为连续分段三次 `s(t)`。

- [ ] **Step 6: 实现自旋梯形或受限 S 曲线**

起止 yaw 角速度为零；小角度旋转退化为无匀速平台的三角速度曲线。输出仍统一编码为 `PiecewiseCubicScalarTrajectory`。

- [ ] **Step 7: 增加模式边界双零速测试**

```cpp
TEST(WheelTimingTest, DriveSpinBoundaryHasZeroLinearAndYawRates) {
  const auto reference = BuildDriveThenSpinFixture();
  EXPECT_DOUBLE_EQ(reference.drive_end_linear_speed(), 0.0);
  EXPECT_DOUBLE_EQ(reference.drive_end_yaw_rate(), 0.0);
  EXPECT_DOUBLE_EQ(reference.spin_start_yaw_rate(), 0.0);
}
```

- [ ] **Step 8: 运行时序测试**

Run:

```powershell
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R '^lpp_v3_wheel\.' --no-tests=error --output-on-failure
```

Expected: 全部通过。

- [ ] **Step 9: 提交时序化**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/wheel/wheel_timing.hpp cpp/src/wheel/wheel_timing.cpp cpp/src/wheel/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/tests/unit/wheel/wheel_timing_test.cpp
git -C path-planner commit -m "feat(v3): parameterize wheel drive and spin timing"
```

---

### Task 7: 轮式参考流水线与整段回退

**Files:**
- Create: `path-planner/cpp/include/lunar_path_planner/v3/wheel/wheel_planner.hpp`
- Create: `path-planner/cpp/src/wheel/wheel_planner.cpp`
- Create: `path-planner/cpp/tests/integration/wheel/wheel_planner_test.cpp`
- Modify: `path-planner/cpp/src/wheel/CMakeLists.txt`
- Modify: `path-planner/cpp/tests/CMakeLists.txt`

**Interfaces:**
- Consumes: Tasks 1–6、已经通过共享语义校验的 `PlanningRequest` 和
  `ResolvedTerminalSet`；不定义第二套上下文类型。
- Produces:

```cpp
class WheelPlanner {
 public:
  Result<WheeledReference> Plan(
      const PlanningRequest&,
      const ResolvedTerminalSet&) noexcept;
};
```

实现先从 `PlanningRequest.current_state` 提取并校验
`WheeledOrLeggedState`；地图、能力、算法配置和可选固定学习代价均只使用 request
内已经解析且不可变的对象。平台层不得重新按 handle 查询内容。

- [ ] **Step 1: 写出走廊失败后整段使用原语链的测试**

```cpp
TEST(WheelPlannerTest, CorridorFailureReturnsOnlyPrimitiveChains) {
  auto planner = MakeWheelPlannerWithFailingCorridorBuilder();
  const auto result = planner.Plan(
      MakeWheelPlanningRequest(),
      MakeResolvedTerminalSet(TerminalKind::kGoal));

  ASSERT_TRUE(IsOk(result));
  const auto& reference = std::get<WheeledReference>(result);
  for (const auto& segment : reference.segments) {
    if (const auto* drive = std::get_if<DriveSegment>(&segment)) {
      EXPECT_TRUE(
          std::holds_alternative<ValidatedPrimitiveChain>(
              drive->geometric_path));
    }
  }
}
```

- [ ] **Step 2: 写出不允许零长度 Drive 代替 Spin 的测试**

```cpp
TEST(WheelPlannerTest, RepresentsInPlaceTurnAsSpinSegment) {
  const auto result = MakeWheelPlanner().Plan(
      MakeSpinOnlyPlanningRequest(),
      MakeResolvedTerminalSet(TerminalKind::kGoal));

  ASSERT_TRUE(IsOk(result));
  const auto& reference = std::get<WheeledReference>(result);
  ASSERT_EQ(reference.segments.size(), 1u);
  EXPECT_TRUE(std::holds_alternative<SpinSegment>(
      reference.segments.front()));
}

TEST(WheelPlannerTest, FinalReferenceHasStableIdentityOriginAndJcsHash) {
  const auto request = MakeWheelPlanningRequest();
  const auto terminal =
      MakeResolvedTerminalSet(TerminalKind::kGoal);
  const auto first = MakeWheelPlanner().Plan(request, terminal);
  const auto second = MakeWheelPlanner().Plan(request, terminal);
  ASSERT_TRUE(IsOk(first));
  ASSERT_TRUE(IsOk(second));
  const auto& a = std::get<WheeledReference>(first);
  const auto& b = std::get<WheeledReference>(second);
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
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_wheel_tests --parallel
```

Expected: 编译失败，提示 `WheelPlanner` 未定义。

- [ ] **Step 4: 实现按分段的连续处理事务**

对每个行驶段：

```cpp
auto candidate = TryBuildSmoothDriveSegment(discrete_segment);
if (!IsOk(candidate) ||
    !validator_.Validate(std::get<DriveSegment>(candidate)).ok()) {
  candidate = BuildValidatedPrimitiveDriveSegment(discrete_segment);
}
if (!IsOk(candidate) ||
    !validator_.Validate(std::get<DriveSegment>(candidate)).ok()) {
  return Error{ErrorCode::kNumericalFailure,
               "/platform_reference",
               "NO_VALIDATED_WHEEL_REFERENCE"};
}
```

不得保留失败样条的任何局部片段。

- [ ] **Step 5: 实现模式切换和相对时间拼接**

所有段的 `TimeInterval` 连续无重叠。切换等待时间由零速度边或自旋段显式表达；不得把等待时间藏入诊断字段。

- [ ] **Step 6: 验证安全停止锚点和数学真值**

最终检查：

- `P(s)` 与 `s(t)` 定义域完整。
- 派生速度缓存若存在，与导数一致。
- 最后一个段的线速度和 yaw 角速度为零。
- `safe_stop_anchor` 与末端位姿一致。

全部分段终验通过后才构造 wire DTO：

- `reference_id` 由 `request_id + selected_candidate_id + "WHEELED"` 经稳定
  ID 工厂生成，不读取墙钟或随机完成顺序；
- `reference_time_origin = request.request_time`，所有段保持相对纳秒偏移；
- `reference_hash` 使用共享 RFC 8785 JCS/SHA-256，对省略该字段的完整
  `WheeledReference` 计算；
- 计算 hash 后再次执行 semantic validation 和 encode/decode/hash round-trip。

- [ ] **Step 7: 运行轮式流水线测试**

Run:

```powershell
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R '^lpp_v3_wheel\.' --no-tests=error --output-on-failure
```

Expected: 全部通过。

- [ ] **Step 8: 提交轮式流水线**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/wheel/wheel_planner.hpp cpp/src/wheel/wheel_planner.cpp cpp/src/wheel/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/tests/integration/wheel/wheel_planner_test.cpp
git -C path-planner commit -m "feat(v3): assemble validated wheeled references"
```

---

### Task 8: 轮式属性测试、故障矩阵和基准 fixture

**Files:**
- Create: `path-planner/cpp/tests/property/wheel/wheel_reference_property_test.cpp`
- Create: `path-planner/cpp/tests/integration/wheel/wheel_fault_matrix_test.cpp`
- Create: `path-planner/cpp/benchmarks/wheel_planner_benchmark.cpp`
- Create: `path-planner/cpp/benchmarks/fixtures/wheel_declared_suite.json`
- Modify: `path-planner/cpp/tests/CMakeLists.txt`
- Modify: `path-planner/cpp/benchmarks/CMakeLists.txt`

**Interfaces:**
- Consumes: `WheelPlanner`, response codec, deterministic fixture generator and benchmark harness.
- Produces: 轮式平台验收证据；不改变运行时 API。

- [ ] **Step 1: 写出 1000 个固定种子参考不变量属性测试**

```cpp
TEST(WheelReferencePropertyTest, GeneratedReferencesRespectSegmentRules) {
  for (std::uint64_t seed = 0; seed < 1000; ++seed) {
    const auto result = PlanGeneratedWheelCase(seed);
    if (!IsOk(result)) {
      continue;
    }
    const auto& reference = std::get<WheeledReference>(result);
    EXPECT_TRUE(AllTimeIntervalsContiguous(reference));
    EXPECT_TRUE(AllDriveScalingsMonotone(reference));
    EXPECT_TRUE(AllModeSwitchesHaveDoubleZeroSpeed(reference));
    EXPECT_TRUE(EndsAtSafeStopAnchor(reference));
  }
}
```

- [ ] **Step 2: 加入确定性字节级回归测试**

固定同一请求、配置和线程策略重复规划 100 次，对 JCS 编码响应计算 SHA-256，要求全部 hash 相同。

- [ ] **Step 3: 加入故障矩阵**

至少覆盖：

```text
unknown_cell
non_circular_spin_collision
corridor_disconnected
qp_infeasible
qp_iteration_limit
timing_sample_limit
continuous_validation_inconclusive
search_resource_limit_with_incumbent
search_resource_limit_without_incumbent
```

每个 fixture 必须断言产生完整原语回退、继续活动承诺或明确无安全参考，不得产生半验证样条。

- [ ] **Step 4: 构建并运行轮式完整测试**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_wheel_tests --parallel
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R '^lpp_v3_wheel\.' --no-tests=error --output-on-failure
```

Expected: 全部通过。

- [ ] **Step 5: 建立不带运行时截止的轮式基准**

```cpp
static void BM_WheelDeclaredSuite(benchmark::State& state) {
  const auto fixture = LoadWheelBenchmarkFixture();
  WheelPlanner planner = BuildPlanner(fixture);
  for (auto _ : state) {
    const auto result = planner.Plan(
        fixture.request, fixture.resolved_terminal);
    benchmark::DoNotOptimize(result);
  }
}
BENCHMARK(BM_WheelDeclaredSuite)->UseRealTime();
```

fixture 记录地图大小、分辨率、障碍密度、起终距离、线程数和能力/config hash，但不传入 `PlannerV3::Plan`。

- [ ] **Step 6: 运行基准并导出 JSON**

Run:

```powershell
Push-Location path-planner/cpp
cmake --preset windows-msvc-release
cmake --build --preset windows-msvc-release --target lpp_v3_wheel_benchmark --parallel
Pop-Location
D:/xunce/build/path-planner-v3/windows-msvc-release/benchmarks/lpp_v3_wheel_benchmark.exe --benchmark_out=D:/xunce/out/path-planner-v3/wheel-benchmark.json --benchmark_out_format=json
```

Expected: 生成逐样本原始耗时；本任务不以单次 1 秒中断规划。

- [ ] **Step 7: 提交轮式验收资产**

```powershell
git -C path-planner add cpp/tests/property/wheel/wheel_reference_property_test.cpp cpp/tests/integration/wheel/wheel_fault_matrix_test.cpp cpp/benchmarks/wheel_planner_benchmark.cpp cpp/benchmarks/fixtures/wheel_declared_suite.json cpp/tests/CMakeLists.txt cpp/benchmarks/CMakeLists.txt
git -C path-planner commit -m "test(v3): cover wheeled planner invariants and faults"
```

---

## Wheel Plan Completion Gate

完成本分卷后必须同时满足：

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --parallel
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R '^lpp_v3_wheel\.' --no-tests=error --output-on-failure
git -C path-planner status --short
```

预期：

- 所有 `wheel_` 测试通过。
- 轮式参考只包含合法 `DriveSegment` 与 `SpinSegment`。
- 倒车 `s(t)` 单调且车体前向速度为负。
- 模式切换双零速。
- 连续处理失败只产生完整认证原语回退。
- 末端为安全停止锚点。
- `git status` 仅包含后续计划尚未提交的明确文件。
