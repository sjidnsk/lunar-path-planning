# 飞跃式平台路径规划 v3 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在全新 C++20 核心中实现飞跃式平台的下一安全着陆域规划、唯一纯弹道飞跃边界、连续飞行管和确定性落点范围认证，并以无执行器副作用的状态机约束飞跃承诺。

**Architecture:** 飞跃平台不复用轮式/足式二维走廊。实现分成四条可独立验证的链：地形安全着陆域与惰性有向图、有限瞄准点与一维飞行时间求解、碰撞/姿态/落点集合物理认证、最终引用与承诺状态机。所有候选先通过便宜筛选，只有候选路线的第一跳进入完整认证；只有全部子证书同时有效时才构造 `HopperReference`。

**Tech Stack:** C++20、CMake 3.28+、Eigen 5.0.1、nlohmann_json 3.12.0、GoogleTest 1.17.0、CTest；Google Benchmark 仅由集成性能计划使用。

## Global Constraints

- 所有新增在线代码位于 `path-planner/cpp/**`，公共命名空间为 `lunar::planning::v3`。
- 这是 clean-room C++ 实现；不得读取、移植、包装或依赖仓库中的既有 Python 路径规划实现。
- 依赖 `lpp_v3_contracts` 与 `lpp_v3_common`；本计划产出静态库 `lpp_v3_hopper` 和测试程序 `lpp_v3_hopper_tests`。
- `lpp_v3_hopper_tests` 使用 `gtest_discover_tests(... TEST_PREFIX
  "lpp_v3_hopper.")`；所有过滤式 CTest gate 必须带 `--no-tests=error`。
- 公共 wire schema 以 `path-planner/schemas/v3/common.schema.json`、`planning-request.schema.json`、`planning-response.schema.json`、`reference-bundle.schema.json`、`references/hopper.schema.json` 为唯一外部字段依据。
- 物理模型必须是常量局部重力下的纯弹道质心平移：`p(t)=p0+v0*t+0.5*g*t^2`；飞行姿态控制不得改变质心轨迹。
- `NextLandingRegion` 是带完整 yaw 区间的单个确定性凸安全域；`JumpBoundary` 只含一组唯一名义发射边界；控制器不得在区域内另选目标。
- `PredictedLandingFootprint ⊕ safe_margin` 必须完全包含于 `NextLandingRegion`，否则不得产生新的飞跃引用。
- 地图未知、数值无法判定、区间细分耗尽或任一子认证失败都按不可认证处理；不得输出部分认证引用。
- 图节点数、出度、瞄准点数、完整认证尝试数、求根迭代数和碰撞细分深度均由 `PlannerAlgorithmConfig` 的正整数资源上限控制。
- 运行时不得读取 `BenchmarkProfile`，不得设置 1 秒截止，也不得根据墙钟剩余时间改变搜索、认证、降级或候选排序。
- 固定请求、地图、能力、算法配置和线程策略时，节点 ID、候选 ID、排序、图重规划和最终序列化结果必须确定。
- 本计划只实现纯函数式规划和状态迁移，不连接 executor，不发送命令，不替换默认策略，不发布 checkpoint，不启动 canary。
- 构建产物写入 `D:/xunce/build/path-planner-v3/<preset>`；不得把构建缓存或实验输出写入仓库。

## Prerequisite Interfaces

本计划消费合同与共享核心计划冻结的以下接口；实现任务不得另造同义 DTO：

```cpp
namespace lunar::planning::v3 {

struct Error;
template <class T>
using Result = std::variant<T, Error>;

struct ValidationReport;
struct HopperState;
struct GoalRegion;
struct ContentRef;
struct SafetyCapabilityProfile;
struct PlannerAlgorithmConfig;
struct PreviousExecutionContext;
struct CircularYawInterval;
struct LandingPlane;
struct NextLandingRegion;
struct JumpBoundary;
struct PredictedLandingFootprint;
struct CertifiedFlightTube;
struct AttitudeBoundary;
struct HopperReference;
struct NominalAimPoint;
struct FutureRoutePreview;
struct ReferenceBundle;
struct PlanningResponse;
struct ConvexPolygonUv;
struct GridGeometry;
struct Cell;
struct AxisAlignedBox3;
struct RotationVectorBall;
struct SymmetricScalarInterval;
struct ConvexPolytope3;
struct ClockStamp;
struct DurationNanoseconds;
struct ResolvedTerminalSet;
struct ValidationContext;
struct SecondaryCostVector;
using StateKey = std::uint64_t;

template <class State, class Edge>
struct SearchTransition;

template <class State>
struct SearchProblem;

enum class PlanningOutcome : std::uint8_t;
enum class ExecutionDirective : std::uint8_t;
enum class TerminalKind : std::uint8_t;
enum class FutureViability : std::uint8_t;

class ImmutableMapSnapshot;

Result<HopperReference> decode_hopper_reference(
    const nlohmann::json& value);
void encode_json(
    nlohmann::json& value, const HopperReference& reference);
ValidationReport validate(
    const HopperReference& reference,
    const ValidationContext& context);

}  // namespace lunar::planning::v3
```

Schema 字段必须保持以下已冻结表示：

```text
HopperState
  position_m
  orientation_body_to_frame
  linear_velocity_mps
  angular_velocity_radps
  error_bounds

CircularYawInterval
  representation = "canonical_ccw"
  start_rad in [-pi, pi)
  span_rad in [0, 2*pi]
  closed = true

LandingPlane
  origin_m
  normal
  basis_u
  basis_v
  residual_bound_m

NextLandingRegion
  region_id
  frame_id
  landing_plane
  convex_polygon
    winding = "CCW"
    vertices_uv
  allowed_yaw_interval
  terrain_certification_ref
  inward_safety_margin_m
```

所有平面多边形顶点按 `winding="CCW"` 编码。`normal,basis_u,basis_v`
必须构成单位正交右手基。所有合同、搜索和发布持续时间使用
`DurationNanoseconds`；弹道求根器可把已校验时长局部转换为有限非负
`double` 秒参与数值计算，但不得把该近似值直接序列化或作为跨模块时间身份。

## File Structure

```text
path-planner/cpp/
├── include/lunar_path_planner/v3/hopper/
│   ├── hopper_config.hpp
│   ├── landing_geometry.hpp
│   ├── swept_footprint.hpp
│   ├── safe_pose_mask.hpp
│   ├── landing_region.hpp
│   ├── aim_point_generator.hpp
│   ├── landing_graph.hpp
│   ├── ballistic_kinematics.hpp
│   ├── ballistic_time_solver.hpp
│   ├── flight_tube_certifier.hpp
│   ├── attitude_certifier.hpp
│   ├── landing_set_propagator.hpp
│   ├── hop_certifier.hpp
│   ├── hopper_planner.hpp
│   └── hopper_commitment_state_machine.hpp
├── src/hopper/
│   ├── CMakeLists.txt
│   ├── hopper_config.cpp
│   ├── landing_geometry.cpp
│   ├── swept_footprint.cpp
│   ├── safe_pose_mask.cpp
│   ├── landing_region.cpp
│   ├── aim_point_generator.cpp
│   ├── landing_graph.cpp
│   ├── ballistic_kinematics.cpp
│   ├── ballistic_time_solver.cpp
│   ├── flight_tube_certifier.cpp
│   ├── attitude_certifier.cpp
│   ├── landing_set_propagator.cpp
│   ├── hop_certifier.cpp
│   ├── hopper_planner.cpp
│   └── hopper_commitment_state_machine.cpp
└── tests/hopper/
    ├── CMakeLists.txt
    ├── hopper_test_fixtures.hpp
    ├── hopper_test_fixtures.cpp
    ├── landing_geometry_test.cpp
    ├── swept_footprint_test.cpp
    ├── safe_pose_mask_test.cpp
    ├── landing_region_test.cpp
    ├── aim_point_generator_test.cpp
    ├── landing_graph_test.cpp
    ├── ballistic_kinematics_test.cpp
    ├── ballistic_time_solver_test.cpp
    ├── flight_tube_certifier_test.cpp
    ├── attitude_certifier_test.cpp
    ├── landing_set_propagator_test.cpp
    ├── hop_certifier_test.cpp
    ├── hopper_planner_test.cpp
    ├── hopper_commitment_state_machine_test.cpp
    └── hopper_scenario_matrix_test.cpp
```

文件按单一职责拆分。`hopper_planner.cpp` 只编排，不实现几何、求根或碰撞公式；`hop_certifier.cpp` 只聚合证书和执行最终不变量。

`hopper_test_fixtures.hpp/.cpp` 是测试专用 fixture 库。每个任务必须在这两个文件中同步增加该任务测试片段使用的全部 `make_*`、事件构造、几何断言和 canonical hash helper；fixture 使用显式固定数值，不调用生产规划器生成期望值。

---

### Task 1: 冻结飞跃配置视图和着陆平面/yaw 基础运算

**Files:**
- Create: `path-planner/cpp/include/lunar_path_planner/v3/hopper/hopper_config.hpp`
- Create: `path-planner/cpp/include/lunar_path_planner/v3/hopper/landing_geometry.hpp`
- Create: `path-planner/cpp/src/hopper/hopper_config.cpp`
- Create: `path-planner/cpp/src/hopper/landing_geometry.cpp`
- Create: `path-planner/cpp/src/hopper/CMakeLists.txt`
- Create: `path-planner/cpp/tests/hopper/landing_geometry_test.cpp`
- Create: `path-planner/cpp/tests/hopper/hopper_test_fixtures.hpp`
- Create: `path-planner/cpp/tests/hopper/hopper_test_fixtures.cpp`
- Create: `path-planner/cpp/tests/hopper/CMakeLists.txt`
- Modify: `path-planner/cpp/CMakeLists.txt`
- Modify: `path-planner/cpp/tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `SafetyCapabilityProfile`, `PlannerAlgorithmConfig`, schema-derived `CircularYawInterval` 和 `LandingPlane`。
- Produces:

```cpp
namespace lunar::planning::v3 {

struct HopperPlannerLimits final {
  enum class FutureRouteAuthority {
    kMissionPreviewOnly
  };
  std::size_t yaw_partition_count;
  std::size_t support_direction_count;
  std::size_t maximum_landing_regions;
  std::size_t landing_region_maximum_vertices;
  std::size_t landing_region_inflation_iterations;
  std::size_t landing_region_maximum_split_depth;
  std::size_t maximum_graph_nodes;
  std::size_t maximum_graph_out_degree;
  std::size_t maximum_nominal_aim_points_per_region;
  std::size_t maximum_full_certification_attempts;
  std::size_t maximum_interval_subdivision_depth;
  std::size_t maximum_root_iterations;
  std::size_t maximum_collision_subdivision_depth;
  std::size_t maximum_flight_tube_sections;
  FutureRouteAuthority future_route_authority{
      FutureRouteAuthority::kMissionPreviewOnly};
};

struct HopperCapabilityView final;

Result<HopperCapabilityView> bind_hopper_capability(
    const SafetyCapabilityProfile& profile,
    const ResolvedCapabilityBindings& bindings);
Result<HopperPlannerLimits> bind_hopper_limits(
    const PlannerAlgorithmConfig& config);

ValidationReport validate_circular_yaw_interval(
    const CircularYawInterval& interval);
ValidationReport validate_landing_plane(const LandingPlane& plane);

struct ConvexPolygon2d final {
  std::vector<Eigen::Vector2d> vertices_ccw;
};

double canonical_yaw(double yaw_rad);
double yaw_interval_end_unwrapped(const CircularYawInterval& interval);
bool yaw_interval_contains(
    const CircularYawInterval& interval, double yaw_rad);
Eigen::Vector3d plane_uv_to_world(
    const LandingPlane& plane, const Eigen::Vector2d& uv);
Eigen::Vector2d world_to_plane_uv(
    const LandingPlane& plane, const Eigen::Vector3d& point_m);

}  // namespace lunar::planning::v3
```

- `HopperCapabilityView` 必须是对已验证 profile 和 request-fixed bindings 的只读值
  视图，至少包含非圆形着陆足迹、机体碰撞包络、地形硬阈值、发射速度/飞行时间/
  冲量/着陆边界、重力模型、任意轴姿态能力、误差模型、执行器/冲量 profile 和
  机体旋转包络。能量只来自解析代价模型，不能成为未声明的硬边界。

- [ ] **Step 1: 写配置必填字段、圆周区间和平面基失败测试**

```cpp
#include <gtest/gtest.h>
#include "lunar_path_planner/v3/hopper/hopper_config.hpp"
#include "lunar_path_planner/v3/hopper/landing_geometry.hpp"

namespace lunar::planning::v3 {

TEST(LandingGeometry, CanonicalCcwIntervalCrossesPiWithoutWrapFlag) {
  const CircularYawInterval interval{
      .start_rad = 3.0,
      .span_rad = 0.4,
  };
  EXPECT_TRUE(yaw_interval_contains(interval, 3.1));
  EXPECT_TRUE(yaw_interval_contains(interval, -3.1));
  EXPECT_FALSE(yaw_interval_contains(interval, 0.0));
}

TEST(LandingGeometry, RejectsNonRightHandedPlaneBasis) {
  LandingPlane plane = make_xy_landing_plane();
  plane.basis_v = Eigen::Vector3d{0.0, -1.0, 0.0};
  EXPECT_FALSE(validate_landing_plane(plane).ok());
}

TEST(HopperConfig, RejectsZeroResourceCapsAndMissingHardLimits) {
  PlannerAlgorithmConfig algorithm = make_valid_algorithm_config();
  algorithm.hopper.maximum_root_iterations = 0;
  EXPECT_FALSE(IsOk(bind_hopper_limits(algorithm)));

  SafetyCapabilityProfile capability = make_valid_hopper_capability();
  capability.hopper.gravity_model.reset();
  EXPECT_FALSE(IsOk(bind_hopper_capability(
      capability, make_resolved_capability_bindings())));
}

}  // namespace lunar::planning::v3
```

- [ ] **Step 2: 配置并运行测试，确认目标与源文件尚不存在而失败**

Run:

```powershell
Push-Location path-planner/cpp
cmake --preset windows-msvc-debug
Pop-Location
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_hopper_tests --parallel
```

Expected: build fails because hopper headers and `lpp_v3_hopper` target do not exist.

- [ ] **Step 3: 添加 `lpp_v3_hopper` 目标和严格配置绑定**

在 `src/hopper/CMakeLists.txt` 建立：

```cmake
add_library(lpp_v3_hopper STATIC
  hopper_config.cpp
  landing_geometry.cpp
)
target_compile_features(lpp_v3_hopper PUBLIC cxx_std_20)
target_include_directories(lpp_v3_hopper
  PUBLIC "${PROJECT_SOURCE_DIR}/include"
)
target_link_libraries(lpp_v3_hopper
  PUBLIC lpp_v3_contracts lpp_v3_common Eigen3::Eigen
)
```

在 `tests/hopper/CMakeLists.txt` 建立：

```cmake
add_executable(lpp_v3_hopper_tests
  hopper_test_fixtures.cpp
  landing_geometry_test.cpp
)
target_link_libraries(lpp_v3_hopper_tests
  PRIVATE lpp_v3_hopper GTest::gtest_main
)
include(GoogleTest)
gtest_discover_tests(
  lpp_v3_hopper_tests
  TEST_PREFIX "lpp_v3_hopper."
)
```

`bind_hopper_limits()` 对每个算法上限执行 `>0` 校验；
`bind_hopper_capability()` 还要求 `minimum_flight_time >= 0`、
`maximum_flight_time > 0` 且 `minimum <= maximum`，并对所有物理硬边界、重力
有效范围和已解析模型绑定执行完整性及有限性校验。不得填充宽松默认值。

- [ ] **Step 4: 实现 canonical yaw 和正交右手平面基运算**

核心实现采用：

```cpp
double canonical_yaw(const double yaw_rad) {
  constexpr double kTwoPi = 2.0 * std::numbers::pi;
  double value = std::fmod(yaw_rad + std::numbers::pi, kTwoPi);
  if (value < 0.0) {
    value += kTwoPi;
  }
  return value - std::numbers::pi;
}

bool yaw_interval_contains(
    const CircularYawInterval& interval, const double yaw_rad) {
  constexpr double kTwoPi = 2.0 * std::numbers::pi;
  constexpr double kAngleTolerance = 1e-12;
  if (!validate_circular_yaw_interval(interval).ok()) {
    return false;
  }
  if (interval.span_rad >= kTwoPi - kAngleTolerance) {
    return true;
  }
  double delta = canonical_yaw(yaw_rad) - interval.start_rad;
  if (delta < 0.0) {
    delta += kTwoPi;
  }
  return delta <= interval.span_rad;
}
```

`validate_landing_plane()` 检查所有量有限、法向和两基向量单位化误差在 capability-independent codec tolerance 内、三者正交、`basis_u.cross(basis_v).dot(normal)>0`、残差非负。坐标转换只能使用该显式平面基，不能隐式选择世界 x/y。

- [ ] **Step 5: 运行飞跃基础测试并确认通过**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_hopper_tests --parallel
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -C Debug -R '^lpp_v3_hopper\.(LandingGeometry|HopperConfig)' --no-tests=error --output-on-failure
```

Expected: all `LandingGeometry` and `HopperConfig` tests pass.

- [ ] **Step 6: 提交配置和几何基础**

```powershell
git -C path-planner add cpp/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/include/lunar_path_planner/v3/hopper/hopper_config.hpp cpp/include/lunar_path_planner/v3/hopper/landing_geometry.hpp cpp/src/hopper/CMakeLists.txt cpp/src/hopper/hopper_config.cpp cpp/src/hopper/landing_geometry.cpp cpp/tests/hopper/CMakeLists.txt cpp/tests/hopper/hopper_test_fixtures.hpp cpp/tests/hopper/hopper_test_fixtures.cpp cpp/tests/hopper/landing_geometry_test.cpp
git -C path-planner commit -m "feat(planner-v3): add hopper geometry contracts"
```

---

### Task 2: 对完整 yaw 区间构造非圆形着陆足迹保守外包

**Files:**
- Create: `path-planner/cpp/include/lunar_path_planner/v3/hopper/swept_footprint.hpp`
- Create: `path-planner/cpp/src/hopper/swept_footprint.cpp`
- Create: `path-planner/cpp/tests/hopper/swept_footprint_test.cpp`
- Modify: `path-planner/cpp/tests/hopper/hopper_test_fixtures.hpp`
- Modify: `path-planner/cpp/tests/hopper/hopper_test_fixtures.cpp`
- Modify: `path-planner/cpp/src/hopper/CMakeLists.txt`
- Modify: `path-planner/cpp/tests/hopper/CMakeLists.txt`

**Interfaces:**
- Consumes: schema/core 的 `CircularYawInterval`、Task 1 的内部 `ConvexPolygon2d`，以及 Task 1 的 yaw 运算。
- Produces:

```cpp
namespace lunar::planning::v3 {

struct SupportHalfspace2 final {
  Eigen::Vector2d outward_unit_normal;
  double upper_bound_m;
};

struct SweptFootprintEnvelope final {
  std::vector<SupportHalfspace2> halfspaces;
  ConvexPolygon2d vertices_ccw;
  CircularYawInterval certified_yaw_interval;
};

Result<SweptFootprintEnvelope>
outer_approximate_rotated_footprint(
    const ConvexPolygon2d& body_frame_footprint,
    const CircularYawInterval& yaw_interval,
    std::span<const Eigen::Vector2d> support_directions,
    double deterministic_margin_m);

bool polygon_is_contained(
    const ConvexPolygon2d& polygon,
    const SweptFootprintEnvelope& envelope,
    double tolerance_m);

}  // namespace lunar::planning::v3
```

- [ ] **Step 1: 写稠密 yaw 采样点必须全部落入保守外包的测试**

```cpp
TEST(SweptFootprint, ContainsEveryRotatedRectangleVertexAcrossYawInterval) {
  const ConvexPolygon2d rectangle =
      make_rectangle_polygon(1.2, 0.6);
  const CircularYawInterval yaw{
      .start_rad = -0.35,
      .span_rad = 1.4,
  };
  const auto directions = make_uniform_unit_directions(32);
  const auto result = outer_approximate_rotated_footprint(
      rectangle, yaw, directions, 0.08);
  ASSERT_TRUE(IsOk(result));
  const auto& envelope =
      std::get<SweptFootprintEnvelope>(result);

  for (int index = 0; index <= 400; ++index) {
    const double theta =
        yaw.start_rad + yaw.span_rad * static_cast<double>(index) / 400.0;
    const ConvexPolygon2d rotated = rotate_polygon(rectangle, theta);
    EXPECT_TRUE(polygon_is_contained(rotated, envelope, 1e-10));
  }
}

TEST(SweptFootprint, RejectsTooFewOrNonUnitSupportDirections) {
  const auto invalid = std::array{
      Eigen::Vector2d{1.0, 0.0},
      Eigen::Vector2d{2.0, 0.0},
  };
  EXPECT_FALSE(IsOk(outer_approximate_rotated_footprint(
      make_rectangle_polygon(1.0, 0.5),
      make_full_yaw_interval(), invalid, 0.0)));
}
```

- [ ] **Step 2: 运行定向测试确认失败**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_hopper_tests --parallel
```

Expected: build fails because `swept_footprint.hpp` and its implementation are absent.

- [ ] **Step 3: 实现每个支撑方向上的解析最大值**

对每个单位方向 \(n\) 和足迹顶点 \(v\)，使用

\[
n^\mathsf T R(\theta)v=a\cos\theta+b\sin\theta
\]

在区间端点以及区间内所有 `atan2(b,a)+2*k*pi` 驻点求最大值。每个方向的上界再加 `deterministic_margin_m`；半平面交使用稳定方向顺序，空集或无界结果返回验证错误。

关键函数必须是有限枚举：

```cpp
double max_rotated_projection(
    const Eigen::Vector2d& normal,
    const Eigen::Vector2d& vertex,
    const CircularYawInterval& yaw_interval) {
  const double a = normal.dot(vertex);
  const double b = normal.dot(
      Eigen::Vector2d{-vertex.y(), vertex.x()});
  double maximum = std::max(
      a * std::cos(yaw_interval.start_rad) +
          b * std::sin(yaw_interval.start_rad),
      a * std::cos(yaw_interval_end_unwrapped(yaw_interval)) +
          b * std::sin(yaw_interval_end_unwrapped(yaw_interval)));
  for (const double theta : stationary_angles_in_interval(
           std::atan2(b, a), yaw_interval)) {
    maximum = std::max(
        maximum, a * std::cos(theta) + b * std::sin(theta));
  }
  return maximum;
}
```

- [ ] **Step 4: 实现半平面交和保守性后验检查**

半平面交生成 CCW 顶点后，再对原足迹每个顶点和所有解析驻点执行包含检查。任何超过数值容差的点都使函数失败，不能通过扩大未记录裕量修补。

- [ ] **Step 5: 运行外包测试并确认通过**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_hopper_tests --parallel
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -C Debug -R '^lpp_v3_hopper\.SweptFootprint' --no-tests=error --output-on-failure
```

Expected: every dense sampled rotation is contained; invalid support directions are rejected.

- [ ] **Step 6: 提交 yaw 扫掠足迹外包**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/hopper/swept_footprint.hpp cpp/src/hopper/swept_footprint.cpp cpp/tests/hopper/swept_footprint_test.cpp cpp/tests/hopper/hopper_test_fixtures.hpp cpp/tests/hopper/hopper_test_fixtures.cpp cpp/src/hopper/CMakeLists.txt cpp/tests/hopper/CMakeLists.txt
git -C path-planner commit -m "feat(planner-v3): certify hopper yaw swept footprint"
```

---

### Task 3: 构造已知地形安全位姿掩码和确定性种子

**Files:**
- Create: `path-planner/cpp/include/lunar_path_planner/v3/hopper/safe_pose_mask.hpp`
- Create: `path-planner/cpp/src/hopper/safe_pose_mask.cpp`
- Create: `path-planner/cpp/tests/hopper/safe_pose_mask_test.cpp`
- Modify: `path-planner/cpp/tests/hopper/hopper_test_fixtures.hpp`
- Modify: `path-planner/cpp/tests/hopper/hopper_test_fixtures.cpp`
- Modify: `path-planner/cpp/src/hopper/CMakeLists.txt`
- Modify: `path-planner/cpp/tests/hopper/CMakeLists.txt`

**Interfaces:**
- Consumes: `ImmutableMapSnapshot`、`HopperCapabilityView`、Task 2 的 `SweptFootprintEnvelope`。
- Produces:

```cpp
namespace lunar::planning::v3 {

enum class LandingCellState : std::uint8_t {
  kUnsafe = 0,
  kSafe = 1,
};

struct SafePoseMask final {
  GridGeometry geometry;
  std::vector<LandingCellState> cells;
  std::vector<double> clearance_m;
  CircularYawInterval certified_yaw_interval;
  ContentRef source_snapshot_ref;
};

struct LandingSeed final {
  Cell cell;
  double clearance_m;
  std::uint64_t stable_id;
};

Result<SafePoseMask> build_safe_pose_mask(
    const ImmutableMapSnapshot& map,
    const HopperCapabilityView& capability,
    const SweptFootprintEnvelope& footprint_envelope);

std::vector<LandingSeed> select_landing_seeds(
    const SafePoseMask& mask,
    std::size_t maximum_seed_count);

bool is_landing_cell_safe(
    const SafePoseMask& mask, Cell cell) noexcept;

}  // namespace lunar::planning::v3
```

- [ ] **Step 1: 写未知单元、硬阈值和稳定种子排序测试**

```cpp
TEST(SafePoseMask, UnknownCellRemainsUnsafeAfterFootprintErosion) {
  auto map = make_flat_known_map(9, 9, 0.25);
  map.set_known(Cell{4, 4}, false);
  const auto snapshot = freeze_map_fixture(map);
  const auto mask = build_safe_pose_mask(
      *snapshot, make_hopper_capability_view(),
      make_test_swept_footprint(0.20));
  ASSERT_TRUE(IsOk(mask));
  const auto& safe_mask = std::get<SafePoseMask>(mask);
  EXPECT_EQ(safe_mask.cells[
                safe_mask.geometry.linear(Cell{4, 4})],
            LandingCellState::kUnsafe);
}

TEST(SafePoseMask, RejectsSlopeRoughnessAndClearanceViolations) {
  auto map = make_flat_known_map(7, 7, 0.25);
  map.set_slope_deg(Cell{2, 2}, 31.0);
  map.set_roughness_m(Cell{3, 3}, 0.12);
  map.set_overhead_clearance_m(Cell{4, 4}, 0.20);
  const auto snapshot = freeze_map_fixture(map);
  const auto mask = build_safe_pose_mask(
      *snapshot, make_hopper_capability_view(),
      make_test_swept_footprint(0.20));
  ASSERT_TRUE(IsOk(mask));
  const auto& safe_mask = std::get<SafePoseMask>(mask);
  EXPECT_FALSE(is_landing_cell_safe(safe_mask, Cell{2, 2}));
  EXPECT_FALSE(is_landing_cell_safe(safe_mask, Cell{3, 3}));
  EXPECT_FALSE(is_landing_cell_safe(safe_mask, Cell{4, 4}));
}

TEST(SafePoseMask, SeedOrderUsesClearanceThenStableCellId) {
  const SafePoseMask mask = make_symmetric_safe_pose_mask();
  const auto seeds = select_landing_seeds(mask, 2);
  ASSERT_EQ(seeds.size(), 2U);
  EXPECT_GT(seeds[0].clearance_m, 0.0);
  EXPECT_LT(seeds[0].stable_id, seeds[1].stable_id);
}
```

- [ ] **Step 2: 运行测试确认缺少掩码实现**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_hopper_tests --parallel
```

Expected: build fails because safe-pose-mask declarations are absent.

- [ ] **Step 3: 实现硬地形过滤和完整足迹侵蚀**

每个候选中心必须同时满足：

- `known=true` 且所有扫掠足迹覆盖单元均已知；
- 障碍、坡度、粗糙度、平面残差和顶/侧净空满足 capability 硬阈值；
- 足迹扫掠包络与确定性位置裕量完整落在安全单元集合；
- 所用地图层来自 `SafePoseMask::source_snapshot_ref` 的同一快照。

发布前的凸着陆域面积还必须不小于
`landing_terrain_thresholds.minimum_landing_region_area_m2`；掩码中的单个
安全 cell 或退化多边形不能冒充“着陆范围”。

使用栅格 supercover 枚举足迹覆盖单元，不允许仅检查中心或多边形顶点。

- [ ] **Step 4: 实现整数栅格 EDT 和稳定种子选择**

实现精确或保守欧氏距离变换，种子排序键固定为：

```cpp
const auto seed_order = [](const LandingSeed& lhs,
                           const LandingSeed& rhs) {
  return std::tuple{-lhs.clearance_m, lhs.stable_id} <
         std::tuple{-rhs.clearance_m, rhs.stable_id};
};
```

种子选取达到 `maximum_seed_count` 后停止；不得按线程完成顺序截断。

- [ ] **Step 5: 运行掩码测试并确认通过**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_hopper_tests --parallel
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -C Debug -R '^lpp_v3_hopper\.SafePoseMask' --no-tests=error --output-on-failure
```

Expected: unknown and所有硬阈值违规单元均为 unsafe，种子顺序稳定。

- [ ] **Step 6: 提交安全位姿掩码**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/hopper/safe_pose_mask.hpp cpp/src/hopper/safe_pose_mask.cpp cpp/tests/hopper/safe_pose_mask_test.cpp cpp/tests/hopper/hopper_test_fixtures.hpp cpp/tests/hopper/hopper_test_fixtures.cpp cpp/src/hopper/CMakeLists.txt cpp/tests/hopper/CMakeLists.txt
git -C path-planner commit -m "feat(planner-v3): build hopper safe pose mask"
```

---

### Task 4: 生成单平面保守凸着陆域

**Files:**
- Create: `path-planner/cpp/include/lunar_path_planner/v3/hopper/landing_region.hpp`
- Create: `path-planner/cpp/src/hopper/landing_region.cpp`
- Create: `path-planner/cpp/tests/hopper/landing_region_test.cpp`
- Modify: `path-planner/cpp/tests/hopper/hopper_test_fixtures.hpp`
- Modify: `path-planner/cpp/tests/hopper/hopper_test_fixtures.cpp`
- Modify: `path-planner/cpp/src/hopper/CMakeLists.txt`
- Modify: `path-planner/cpp/tests/hopper/CMakeLists.txt`

**Interfaces:**
- Consumes: Task 3 的 `SafePoseMask` 与 `LandingSeed`、共享地图高程/法向只读视图。
- Produces:

```cpp
namespace lunar::planning::v3 {

struct TerrainCertification final {
  std::string certification_id;
  ContentRef source_snapshot_ref;
  double maximum_slope_deg;
  double maximum_roughness_m;
  double maximum_plane_residual_m;
  double minimum_clearance_m;
};

struct TerrainCertifiedLandingRegion final {
  std::string region_id;
  LandingPlane landing_plane;
  ConvexPolygon2d vertices_uv_ccw;
  CircularYawInterval allowed_yaw_interval;
  TerrainCertification terrain_certification;
  double inward_safety_margin_m;
  Cell source_seed;
};

class LandingRegionGenerator final {
 public:
  LandingRegionGenerator(
      HopperCapabilityView capability,
      HopperPlannerLimits limits);

  std::vector<TerrainCertifiedLandingRegion> generate(
      const ImmutableMapSnapshot& map,
      const SafePoseMask& mask,
      std::span<const LandingSeed> seeds) const;
};

ValidationReport validate_terrain_certified_region(
    const TerrainCertifiedLandingRegion& region,
    const ImmutableMapSnapshot& map,
    const SafePoseMask& mask,
    const HopperCapabilityView& capability);

TerrainCertifiedLandingRegion simplify_region_inward(
    const TerrainCertifiedLandingRegion& region,
    std::size_t maximum_vertex_count);

}  // namespace lunar::planning::v3
```

- [ ] **Step 1: 写含洞、窄颈、多平面和向内简化测试**

```cpp
TEST(LandingRegion, ConvexInflationNeverBridgesUnsafeHole) {
  auto fixture = make_safe_mask_with_center_hole();
  const auto regions = make_region_generator().generate(
      *fixture.map, fixture.mask, fixture.seeds);
  ASSERT_FALSE(regions.empty());
  for (const auto& region : regions) {
    EXPECT_TRUE(validate_terrain_certified_region(
        region, *fixture.map, fixture.mask, fixture.capability).ok());
    EXPECT_FALSE(region_polygon_contains_cell(
        region, Cell{5, 5}, fixture.mask.geometry));
  }
}

TEST(LandingRegion, SplitsRegionWhenOnePlaneCannotCertifyWholeArea) {
  auto fixture = make_two_plane_landing_fixture();
  const auto regions = make_region_generator().generate(
      *fixture.map, fixture.mask, fixture.seeds);
  ASSERT_GE(regions.size(), 2U);
  for (const auto& region : regions) {
    EXPECT_LE(region.landing_plane.residual_bound_m,
              fixture.capability.maximum_plane_residual_m);
  }
}

TEST(LandingRegion, VertexSimplificationIsAnInwardSubset) {
  const auto original = make_dense_convex_region();
  const auto simplified = simplify_region_inward(original, 8);
  EXPECT_LE(simplified.vertices_uv_ccw.size(), 8U);
  EXPECT_TRUE(polygon_is_subset(
      simplified.vertices_uv_ccw, original.vertices_uv_ccw));
}
```

- [ ] **Step 2: 运行测试确认缺少着陆域生成器**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_hopper_tests --parallel
```

Expected: build fails because `LandingRegionGenerator` is undefined.

- [ ] **Step 3: 实现有界半平面膨胀和原栅格复核**

从每个 seed 单元的内接多边形开始，按最近 unsafe 单元的稳定 ID 顺序添加分离半平面。每轮必须：

1. 与地图边界相交；
2. 保持 seed 在多边形内部；
3. 使用 supercover 枚举多边形覆盖单元；
4. 拒绝任何包含 unsafe/unknown 单元的扩张；
5. 在 `landing_region_maximum_vertices` 与 `landing_region_inflation_iterations` 内结束。

不得对安全掩码的连通分量直接取凸包。最终多边形再次在原始地图层和完整 yaw 扫掠足迹上验证。

- [ ] **Step 4: 实现单平面拟合、残差认证和确定性分裂**

用 Eigen 对区域覆盖单元的三维点做最小二乘平面拟合，法向朝平台声明的地表外侧。若最大绝对残差超过硬阈值，按最大残差点与 seed 的确定性分离方向把候选拆成两个子掩码并重新膨胀；达到 `landing_region_maximum_split_depth` 或 `maximum_landing_regions` 仍不能形成单平面时丢弃该候选。

平面基构造使用固定参考轴：

```cpp
const Eigen::Vector3d reference =
    std::abs(normal.z()) < 0.9
        ? Eigen::Vector3d::UnitZ()
        : Eigen::Vector3d::UnitX();
const Eigen::Vector3d basis_u =
    reference.cross(normal).normalized();
const Eigen::Vector3d basis_v =
    normal.cross(basis_u).normalized();
```

- [ ] **Step 5: 实现向内顶点简化和稳定 region ID**

只允许用相邻顶点 chord 替换凸多边形边界，因此新多边形是原区域子集。每次候选简化后重新执行栅格、平面和 yaw 足迹验证。`region_id` 由 snapshot ref、seed stable ID、量化后的平面和 CCW 顶点内容哈希生成。

- [ ] **Step 6: 运行着陆域测试并确认通过**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_hopper_tests --parallel
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -C Debug -R '^lpp_v3_hopper\.LandingRegion' --no-tests=error --output-on-failure
```

Expected: 所有着陆域均避开洞和未知区域，跨平面区域被分裂，顶点简化保持向内。

- [ ] **Step 7: 提交地形认证着陆域**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/hopper/landing_region.hpp cpp/src/hopper/landing_region.cpp cpp/tests/hopper/landing_region_test.cpp cpp/tests/hopper/hopper_test_fixtures.hpp cpp/tests/hopper/hopper_test_fixtures.cpp cpp/src/hopper/CMakeLists.txt cpp/tests/hopper/CMakeLists.txt
git -C path-planner commit -m "feat(planner-v3): generate certified hopper landing regions"
```

---

### Task 5: 为每个着陆域生成有限、确定且严格位于内部的瞄准点

**Files:**
- Create: `path-planner/cpp/include/lunar_path_planner/v3/hopper/aim_point_generator.hpp`
- Create: `path-planner/cpp/src/hopper/aim_point_generator.cpp`
- Create: `path-planner/cpp/tests/hopper/aim_point_generator_test.cpp`
- Modify: `path-planner/cpp/tests/hopper/hopper_test_fixtures.hpp`
- Modify: `path-planner/cpp/tests/hopper/hopper_test_fixtures.cpp`
- Modify: `path-planner/cpp/src/hopper/CMakeLists.txt`
- Modify: `path-planner/cpp/tests/hopper/CMakeLists.txt`

**Interfaces:**
- Consumes: `TerrainCertifiedLandingRegion`、`GoalRegion`、`HopperCapabilityView` 中的名义着陆机体中心法向偏置、任务方向提示和 `HopperPlannerLimits::maximum_nominal_aim_points_per_region`。
- Produces:

```cpp
namespace lunar::planning::v3 {

enum class AimPointSource : std::uint8_t {
  kChebyshevCenter,
  kGoalProjection,
  kMissionDirectionInset,
  kSupportDirectionInset,
};

struct AimPointCandidate final {
  std::string aim_point_id;
  std::string region_id;
  std::string frame_id;
  Eigen::Vector2d position_uv;
  Eigen::Vector3d position_m;
  AimPointSource source;
  double minimum_boundary_distance_m;
};

class AimPointGenerator final {
 public:
  AimPointGenerator(
      HopperCapabilityView capability,
      HopperPlannerLimits limits);

  std::vector<AimPointCandidate> generate(
      const TerrainCertifiedLandingRegion& region,
      const GoalRegion& goal,
      const std::optional<Eigen::Vector3d>& mission_direction_frame) const;
};

bool point_strictly_inside_region(
    const AimPointCandidate& point,
    const TerrainCertifiedLandingRegion& region);

}  // namespace lunar::planning::v3
```

- [ ] **Step 1: 写候选有限性、内部裕量、去重和稳定顺序测试**

```cpp
TEST(AimPointGenerator, ProducesFiniteStableInteriorCandidates) {
  const auto region = make_rectangular_landing_region();
  const auto goal = make_goal_region_near_region_edge();
  AimPointGenerator generator(
      make_hopper_capability_view(),
      make_hopper_limits(
          {.maximum_nominal_aim_points_per_region = 4}));

  const auto first = generator.generate(
      region, goal, Eigen::Vector3d{1.0, 0.0, 0.0});
  const auto second = generator.generate(
      region, goal, Eigen::Vector3d{1.0, 0.0, 0.0});

  ASSERT_LE(first.size(), 4U);
  ASSERT_FALSE(first.empty());
  EXPECT_EQ(first, second);
  for (const auto& point : first) {
    EXPECT_TRUE(point.position_m.allFinite());
    EXPECT_GE(point.minimum_boundary_distance_m,
              region.inward_safety_margin_m);
    EXPECT_TRUE(point_strictly_inside_region(point, region));
  }
}

TEST(AimPointGenerator, DeduplicatesCoincidentCenterAndGoalProjection) {
  const auto region = make_rectangular_landing_region();
  const auto goal = make_goal_centered_on_region(region);
  const auto points = AimPointGenerator(
      make_hopper_capability_view(),
      make_hopper_limits(
          {.maximum_nominal_aim_points_per_region = 8})).generate(
          region, goal, std::nullopt);
  EXPECT_EQ(count_quantized_unique_points(points), points.size());
}
```

- [ ] **Step 2: 运行测试确认缺少瞄准点生成器**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_hopper_tests --parallel
```

Expected: build fails because `AimPointGenerator` is undefined.

- [ ] **Step 3: 实现小规模凸多边形 Chebyshev 中心**

把 CCW 多边形转换为归一化内向半平面 `a_i.dot(x) >= b_i`，对所有两条边和等距约束的有限组合求候选 `(x,y,r)`，保留满足全部半平面的最大 `r`。并列时按量化 `(x,y)` 字典序选择。若数值退化，则使用已认证 seed 在平面上的投影，仍必须满足内部裕量。

- [ ] **Step 4: 实现目标投影、任务方向内点和固定方向内点**

目标投影使用凸多边形最近点后沿 Chebyshev 中心方向内缩至安全裕量；任务方向点由中心沿投影后的任务方向与区域边界相交并内缩；固定方向按 `support_direction_count` 的稳定角序生成。所有点经过同一个 `point_strictly_inside_region()` 硬检查。`position_m` 必须由 `plane_uv_to_world()` 再沿 plane normal 加 capability 中认证的名义着陆机体中心偏置，不能把地面接触点误当成质心落点。

候选排序键固定为：

```cpp
return std::tuple{
    static_cast<std::uint8_t>(candidate.source),
    -candidate.minimum_boundary_distance_m,
    quantize(candidate.position_uv.x()),
    quantize(candidate.position_uv.y())};
```

排序后按量化坐标去重，再截断至配置上限。

- [ ] **Step 5: 运行瞄准点测试并确认通过**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_hopper_tests --parallel
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -C Debug -R '^lpp_v3_hopper\.AimPointGenerator' --no-tests=error --output-on-failure
```

Expected: 候选数量有界、顺序可重复、全部严格位于着陆域内部。

- [ ] **Step 6: 提交有限瞄准点生成器**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/hopper/aim_point_generator.hpp cpp/src/hopper/aim_point_generator.cpp cpp/tests/hopper/aim_point_generator_test.cpp cpp/tests/hopper/hopper_test_fixtures.hpp cpp/tests/hopper/hopper_test_fixtures.cpp cpp/src/hopper/CMakeLists.txt cpp/tests/hopper/CMakeLists.txt
git -C path-planner commit -m "feat(planner-v3): generate bounded hopper aim points"
```

---

### Task 6: 实现有向惰性着陆域图和第一跳认证循环

**Files:**
- Create: `path-planner/cpp/include/lunar_path_planner/v3/hopper/landing_graph.hpp`
- Create: `path-planner/cpp/src/hopper/landing_graph.cpp`
- Create: `path-planner/cpp/tests/hopper/landing_graph_test.cpp`
- Modify: `path-planner/cpp/tests/hopper/hopper_test_fixtures.hpp`
- Modify: `path-planner/cpp/tests/hopper/hopper_test_fixtures.cpp`
- Modify: `path-planner/cpp/src/hopper/CMakeLists.txt`
- Modify: `path-planner/cpp/tests/hopper/CMakeLists.txt`

**Interfaces:**
- Consumes: Task 4 的着陆域、Task 5 的瞄准点、共享 `RunAraStar<State, Edge>()` 和预计时间下界。
- Produces:

```cpp
namespace lunar::planning::v3 {

enum class LandingEdgeState : std::uint8_t {
  kCheapPossible,
  kCertifiedNextHop,
  kRejected,
};

struct LandingGraphNode final {
  std::string node_id;
  TerrainCertifiedLandingRegion region;
  std::vector<AimPointCandidate> aim_points;
};

struct LandingGraphEdge final {
  std::string edge_id;
  std::size_t from_node;
  std::size_t to_node;
  LandingEdgeState state;
  DurationNanoseconds lower_bound_execution_time;
  double estimated_energy_lower_bound_j;
};

struct LandingGraphSearchState final {
  std::size_t node_index;
};

struct LandingGraphSearchEdge final {
  std::size_t edge_index;
};

class LandingGraphAdapter final {
 public:
  LandingGraphAdapter(
      std::span<const LandingGraphNode> nodes,
      std::span<const LandingGraphEdge> edges,
      std::span<const std::size_t> terminal_nodes);

  StateKey Key(const LandingGraphSearchState& state) const;
  std::vector<SearchTransition<
      LandingGraphSearchState, LandingGraphSearchEdge>>
  Expand(const LandingGraphSearchState& state) const;
  bool HardFeasible(const SearchTransition<
      LandingGraphSearchState, LandingGraphSearchEdge>& transition) const;
  DurationNanoseconds TransitionTime(const SearchTransition<
      LandingGraphSearchState, LandingGraphSearchEdge>& transition) const;
  DurationNanoseconds AdmissibleTimeHeuristic(
      const LandingGraphSearchState& state,
      const SearchProblem<LandingGraphSearchState>& problem) const;
  SecondaryCostVector SecondaryCosts(const SearchTransition<
      LandingGraphSearchState, LandingGraphSearchEdge>& transition) const;
  bool IsTerminal(
      const LandingGraphSearchState& state,
      const SearchProblem<LandingGraphSearchState>& problem) const;
};

struct LazyNextHopSelection final {
  std::size_t edge_index;
  std::string selected_aim_point_id;
  HopperReference certified_next_hop;
  FutureRoutePreview future_preview;
  std::size_t full_certification_attempt_count;
  enum class IncumbentStatus {
    kAllCompetitiveFirstEdgesRuledOut,
    kResourceLimitedCompetitiveEdgesRemain
  } incumbent_status;
  std::size_t remaining_competitive_first_edge_count;
};

struct FirstEdgeCertificationResult final {
  struct CertifiedCandidate {
    HopperReference reference;
    DurationNanoseconds expected_execution_time;
    SecondaryCostVector secondary_costs;
  };
  std::optional<CertifiedCandidate> certified_candidate;
  std::string rejection_reason;
};

class NextHopCertificationOracle {
 public:
  virtual ~NextHopCertificationOracle() = default;
  virtual FirstEdgeCertificationResult certify_first_edge(
      const LandingGraphNode& from,
      const LandingGraphNode& to) = 0;
};

class LazyLandingGraphPlanner final {
 public:
  Result<LazyNextHopSelection> select(
      std::vector<LandingGraphNode> nodes,
      std::vector<LandingGraphEdge> edges,
      std::size_t start_node,
      std::span<const std::size_t> terminal_nodes,
      NextHopCertificationOracle& oracle,
      const HopperPlannerLimits& limits) const;
};

}  // namespace lunar::planning::v3
```

- [ ] **Step 1: 写认证失败后标记边并恢复搜索的测试**

```cpp
class RecordingOracle final : public NextHopCertificationOracle {
 public:
  std::vector<std::string> attempted_edge_ids;

  FirstEdgeCertificationResult certify_first_edge(
      const LandingGraphNode& from,
      const LandingGraphNode& to) override {
    attempted_edge_ids.push_back(from.node_id + "->" + to.node_id);
    if (to.node_id == "safe_b") {
      return FirstEdgeCertificationResult{
          .certified_candidate = make_valid_certified_first_edge_candidate(),
          .rejection_reason = "",
      };
    }
    return FirstEdgeCertificationResult{
        .certified_candidate = std::nullopt,
        .rejection_reason = "fixture_rejected",
    };
  }
};

TEST(LandingGraph, RejectsFailedFirstEdgeAndResumesDeterministically) {
  auto fixture = make_two_route_landing_graph();
  RecordingOracle oracle;
  const auto selection = LazyLandingGraphPlanner{}.select(
      fixture.nodes, fixture.edges, fixture.start_node,
      fixture.terminal_nodes, oracle, fixture.limits);
  ASSERT_TRUE(IsOk(selection));
  const auto& selected =
      std::get<LazyNextHopSelection>(selection);
  EXPECT_EQ(oracle.attempted_edge_ids,
            (std::vector<std::string>{"start->cheap_a", "start->safe_b"}));
  EXPECT_EQ(selected.future_preview.authority,
            FutureRoutePreview::Authority::
                kNonAuthoritativeMissionPreview);
}

TEST(LandingGraph, MarksSafeDeadEndAsMetadataWithoutInventingOutcome) {
  auto fixture = make_safe_dead_end_landing_graph();
  RecordingOracle oracle;
  const auto selection = LazyLandingGraphPlanner{}.select(
      fixture.nodes, fixture.edges, fixture.start_node,
      fixture.terminal_nodes, oracle, fixture.limits);
  ASSERT_TRUE(IsOk(selection));
  const auto& selected =
      std::get<LazyNextHopSelection>(selection);
  EXPECT_EQ(selected.future_preview.reason_code, "SAFE_DEAD_END");
  EXPECT_EQ(selected.future_preview.future_viability,
            FutureViability::kNoCertifiedContinuation);
  EXPECT_EQ(selected.future_preview.authority,
            FutureRoutePreview::Authority::
                kNonAuthoritativeMissionPreview);
}
```

- [ ] **Step 2: 运行测试确认惰性图尚未实现**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_hopper_tests --parallel
```

Expected: build fails because landing graph types and planner are absent.

- [ ] **Step 3: 实现有界有向图构建和便宜筛选**

按稳定 region ID 顺序建立节点。每个节点只保留距离、正飞行时间、速度和粗略
弹道时间下界通过的最近 `maximum_graph_out_degree` 条有向边；全图不超过
`maximum_graph_nodes`。能量下界只作为稳定次级排序，不得过滤边。便宜筛选只能
产生 `kCheapPossible`，不得产生物理安全声明。

- [ ] **Step 4: 实现 ARA* adapter 和时间优先排序**

`LandingGraphAdapter` 的方法名和 `DurationNanoseconds` 返回类型必须逐字满足共享
`PlatformSearchAdapter` concept。主边代价为发射准备、飞行时间下界和着陆稳定时间下界之和；
启发式是到任一终端的时间下界。解析秒值转纳秒时向下取整以维持 admissibility，并用检查加法拒绝
int64 溢出。能量和着陆余量仅作为时间等价池次级键。
`LandingEdgeState::kRejected` 的边不展开。搜索调用必须是：

```cpp
const auto search_result =
    RunAraStar<LandingGraphSearchState, LandingGraphSearchEdge>(
        adapter, search_problem, algorithm_config.ara_star);
```

- [ ] **Step 5: 实现只认证候选路线第一条边的恢复循环**

每轮 ARA* 返回候选 region 路线后，只把尚未尝试的第一条边及目标节点内的有限
瞄准点集交给 oracle。维护真实预计执行时间最小的完整认证 incumbent；认证成功后
不得立即返回，而要继续认证所有
`first_edge_time_lower_bound <= incumbent_time + delta_t_equivalence` 的竞争第一边。
只有这些边均被认证或拒绝，才标记
`kAllCompetitiveFirstEdgesRuledOut` 并按真实预计时间及时间等价次级键返回。
若 `maximum_full_certification_attempts` 先触发，可返回完整认证 incumbent，但必须
标记 `kResourceLimitedCompetitiveEdgesRemain`、记录剩余竞争边数，且不得声称
时间最优或 ARA* 次优界；没有完整 incumbent 时不得发布任何边。

后续路线仅映射为：

```cpp
FutureRoutePreview{
    .candidate_region_ids = future_region_ids,
    .authority = FutureRoutePreview::Authority::
        kNonAuthoritativeMissionPreview,
    .future_viability =
        first_hop_reaches_terminal
            ? FutureViability::kViable
            : (future_region_ids.empty()
                   ? FutureViability::kNoCertifiedContinuation
                   : FutureViability::kUnknown),
    .reason_code =
        !first_hop_reaches_terminal && future_region_ids.empty()
            ? "SAFE_DEAD_END"
            : "FUTURE_ROUTE_NON_AUTHORITATIVE",
};
```

- [ ] **Step 6: 运行惰性图测试并确认通过**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_hopper_tests --parallel
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -C Debug -R '^lpp_v3_hopper\.LandingGraph' --no-tests=error --output-on-failure
```

Expected: 失败边被稳定拒绝后恢复搜索，只认证第一跳，future preview 始终非权威。

- [ ] **Step 7: 提交惰性着陆域图**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/hopper/landing_graph.hpp cpp/src/hopper/landing_graph.cpp cpp/tests/hopper/landing_graph_test.cpp cpp/tests/hopper/hopper_test_fixtures.hpp cpp/tests/hopper/hopper_test_fixtures.cpp cpp/src/hopper/CMakeLists.txt cpp/tests/hopper/CMakeLists.txt
git -C path-planner commit -m "feat(planner-v3): add lazy hopper landing graph"
```

---

### Task 7: 实现常量局部重力下的纯弹道运动学

**Files:**
- Create: `path-planner/cpp/include/lunar_path_planner/v3/hopper/ballistic_kinematics.hpp`
- Create: `path-planner/cpp/src/hopper/ballistic_kinematics.cpp`
- Create: `path-planner/cpp/tests/hopper/ballistic_kinematics_test.cpp`
- Modify: `path-planner/cpp/tests/hopper/hopper_test_fixtures.hpp`
- Modify: `path-planner/cpp/tests/hopper/hopper_test_fixtures.cpp`
- Modify: `path-planner/cpp/src/hopper/CMakeLists.txt`
- Modify: `path-planner/cpp/tests/hopper/CMakeLists.txt`

**Interfaces:**
- Consumes: `HopperState`、能力中的常量局部重力模型及其 frame/时空有效范围、Task 5 的名义瞄准点。
- Produces:

```cpp
namespace lunar::planning::v3 {

struct SpatialValidityRegion final {
  AxisAlignedBox3 bounds_m;
};

struct TimeValidityInterval final {
  ClockStamp valid_from;
  ClockStamp valid_until;
};

struct GravityModelView final {
  std::string model_id;
  std::string frame_id;
  Eigen::Vector3d nominal_acceleration_mps2;
  AxisAlignedBox3 acceleration_error_mps2;
  SpatialValidityRegion spatial_validity;
  TimeValidityInterval time_validity;
};

struct BallisticState final {
  Eigen::Vector3d position_m;
  Eigen::Vector3d velocity_mps;
};

struct NominalBallisticArc final {
  std::string aim_point_id;
  Eigen::Vector3d launch_position_m;
  Eigen::Vector3d aim_position_m;
  Eigen::Vector3d gravity_mps2;
  Eigen::Vector3d launch_velocity_mps;
  Eigen::Vector3d landing_velocity_mps;
  double flight_time_s;
};

Result<NominalBallisticArc> make_nominal_ballistic_arc(
    const Eigen::Vector3d& launch_position_m,
    std::string_view launch_frame_id,
    const AimPointCandidate& aim_point,
    const GravityModelView& gravity,
    double flight_time_s);

BallisticState evaluate_ballistic_state(
    const NominalBallisticArc& arc, double time_s);

double ballistic_apex_time_s(const NominalBallisticArc& arc);

}  // namespace lunar::planning::v3
```

- [ ] **Step 1: 写端点、速度和纯弹道不变量测试**

```cpp
TEST(BallisticKinematics, ArcHitsAimPointAtRequestedFlightTime) {
  const Eigen::Vector3d p0{0.0, 0.0, 1.0};
  const auto aim = make_nominal_aim_point(
      Eigen::Vector3d{4.0, -2.0, 0.5});
  const auto gravity = make_gravity_model(
      Eigen::Vector3d{0.0, 0.0, -1.62});
  const auto arc = make_nominal_ballistic_arc(
      p0, "map", aim, gravity, 3.0);
  ASSERT_TRUE(IsOk(arc));
  const auto& nominal_arc =
      std::get<NominalBallisticArc>(arc);

  const auto start = evaluate_ballistic_state(nominal_arc, 0.0);
  const auto finish = evaluate_ballistic_state(nominal_arc, 3.0);
  EXPECT_TRUE(start.position_m.isApprox(p0, 1e-12));
  EXPECT_TRUE(finish.position_m.isApprox(aim.position_m, 1e-12));
  EXPECT_TRUE(finish.velocity_mps.isApprox(
      nominal_arc.launch_velocity_mps +
          gravity.nominal_acceleration_mps2 * 3.0,
      1e-12));
}

TEST(BallisticKinematics, RejectsInvalidGravityFrameAndNonPositiveTime) {
  auto gravity = make_gravity_model(Eigen::Vector3d{0.0, 0.0, -1.62});
  gravity.frame_id = "different_frame";
  EXPECT_FALSE(IsOk(make_nominal_ballistic_arc(
      Eigen::Vector3d::Zero(), "map", make_nominal_aim_point(
          Eigen::Vector3d::Ones()), gravity, 1.0)));
  gravity.frame_id = "map";
  EXPECT_FALSE(IsOk(make_nominal_ballistic_arc(
      Eigen::Vector3d::Zero(), "map", make_nominal_aim_point(
          Eigen::Vector3d::Ones()), gravity, 0.0)));
}
```

- [ ] **Step 2: 运行测试确认弹道运动学缺失**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_hopper_tests --parallel
```

Expected: build fails because ballistic kinematics declarations are absent.

- [ ] **Step 3: 实现唯一发射速度和着陆速度公式**

```cpp
const Eigen::Vector3d displacement =
    aim_point.position_m - launch_position_m;
const Eigen::Vector3d launch_velocity =
    displacement / flight_time_s -
    0.5 * gravity.nominal_acceleration_mps2 * flight_time_s;
const Eigen::Vector3d landing_velocity =
    launch_velocity +
    gravity.nominal_acceleration_mps2 * flight_time_s;
```

输入必须有限、`flight_time_s>0`、frame 一致且名义轨迹落在重力模型时空有效范围内。函数不得接受飞行中推力或质心控制输入。

- [ ] **Step 4: 实现状态求值和最高点事件**

`evaluate_ballistic_state()` 严格使用二次位置与一次速度。最高点相对着陆平面法向求解 `n.dot(v0+g*t)=0`，仅当解位于 `[0,T]` 时返回内部事件；否则最高高度候选只取端点。

- [ ] **Step 5: 运行弹道运动学测试并确认通过**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_hopper_tests --parallel
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -C Debug -R '^lpp_v3_hopper\.BallisticKinematics' --no-tests=error --output-on-failure
```

Expected: 起点、终点和速度公式通过，非法时间、坐标系或重力有效范围被拒绝。

- [ ] **Step 6: 提交纯弹道运动学**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/hopper/ballistic_kinematics.hpp cpp/src/hopper/ballistic_kinematics.cpp cpp/tests/hopper/ballistic_kinematics_test.cpp cpp/tests/hopper/hopper_test_fixtures.hpp cpp/tests/hopper/hopper_test_fixtures.cpp cpp/src/hopper/CMakeLists.txt cpp/tests/hopper/CMakeLists.txt
git -C path-planner commit -m "feat(planner-v3): add pure ballistic kinematics"
```

---

### Task 8: 对有限瞄准点执行有界一维飞行时间求解

**Files:**
- Create: `path-planner/cpp/include/lunar_path_planner/v3/hopper/ballistic_time_solver.hpp`
- Create: `path-planner/cpp/src/hopper/ballistic_time_solver.cpp`
- Create: `path-planner/cpp/tests/hopper/ballistic_time_solver_test.cpp`
- Modify: `path-planner/cpp/tests/hopper/hopper_test_fixtures.hpp`
- Modify: `path-planner/cpp/tests/hopper/hopper_test_fixtures.cpp`
- Modify: `path-planner/cpp/src/hopper/CMakeLists.txt`
- Modify: `path-planner/cpp/tests/hopper/CMakeLists.txt`

**Interfaces:**
- Consumes: Task 7 的纯弹道函数、能力中的正飞行时间、发射速度、可解析冲量和
  着陆速度硬边界，以及姿态最小所需时间的只读输入。
- Produces:

```cpp
namespace lunar::planning::v3 {

struct FlightTimeSearchInput final {
  Eigen::Vector3d launch_position_m;
  std::vector<AimPointCandidate> aim_points;
  GravityModelView gravity;
  double minimum_attitude_time_s;
  double launch_preparation_time_s;
  double landing_settle_time_s;
};

struct BallisticCandidate final {
  std::string candidate_id;
  NominalBallisticArc arc;
  double expected_execution_time_s;
  double estimated_energy_j;
  double landing_margin_m;
  double nominal_minimum_clearance_m;
};

struct BallisticSolveDiagnostics final {
  std::size_t aim_point_count;
  std::size_t interval_count;
  std::size_t subdivision_count;
  std::size_t root_iteration_count;
  std::string termination_reason;
};

struct BallisticSolveResult final {
  std::vector<BallisticCandidate> candidates;
  BallisticSolveDiagnostics diagnostics;
};

bool ballistic_candidate_order(
    const BallisticCandidate& lhs,
    const BallisticCandidate& rhs);

class BallisticTimeSolver final {
 public:
  BallisticTimeSolver(
      HopperCapabilityView capability,
      HopperPlannerLimits limits);

  BallisticSolveResult solve(
      const FlightTimeSearchInput& input) const;
};

}  // namespace lunar::planning::v3
```

- [ ] **Step 1: 写边界可行、无解、资源上限和稳定排序测试**

```cpp
TEST(BallisticTimeSolver, FindsShortestFeasibleTimeForFiniteAimSet) {
  const auto input = make_ballistic_time_search_input();
  const auto result = make_ballistic_time_solver().solve(input);
  ASSERT_FALSE(result.candidates.empty());
  EXPECT_TRUE(std::is_sorted(
      result.candidates.begin(), result.candidates.end(),
      ballistic_candidate_order));
  EXPECT_TRUE(all_candidates_satisfy_hard_ballistic_limits(
      result.candidates, make_hopper_capability_view()));
}

TEST(BallisticTimeSolver, ReturnsNoCandidateWhenLandingSpeedIsImpossible) {
  auto capability = make_hopper_capability_view();
  capability.maximum_landing_speed_mps = 0.01;
  const BallisticTimeSolver solver{
      capability, make_hopper_limits()};
  const auto result = solver.solve(
      make_ballistic_time_search_input());
  EXPECT_TRUE(result.candidates.empty());
  EXPECT_EQ(result.diagnostics.termination_reason,
            "no_feasible_flight_time");
}

TEST(BallisticTimeSolver, ResourceLimitNeverReturnsUncheckedInterval) {
  auto limits = make_hopper_limits();
  limits.maximum_interval_subdivision_depth = 1;
  limits.maximum_root_iterations = 1;
  const auto result = BallisticTimeSolver{
      make_hopper_capability_view(), limits}.solve(
          make_boundary_sensitive_time_search_input());
  EXPECT_TRUE(all_returned_candidates_are_pointwise_verified(result));
}
```

- [ ] **Step 2: 运行测试确认一维求解器缺失**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_hopper_tests --parallel
```

Expected: build fails because `BallisticTimeSolver` is undefined.

- [ ] **Step 3: 实现解析初始时间区间和断点**

对每个瞄准点，由能力 min/max 正飞行时间、发射速度、可解析冲量、着陆速度、
向下横截性和 `minimum_attitude_time_s` 计算保守初始区间。最高点只作为重力
有效域和后续碰撞认证的切分事件，不形成高度硬阈值；能量不参与区间排除。
收集约束导数为零、分母边界、最高点进入/离开 `[0,T]` 的解析断点；排序、量化
去重后形成有限闭区间列表。

- [ ] **Step 4: 实现固定深度区间排除和有界 Brent/二分**

每个约束提供三值区间判定：

```cpp
enum class IntervalFeasibility : std::uint8_t {
  kAllFeasible,
  kAllInfeasible,
  kIndeterminate,
};
```

`kAllInfeasible` 立即丢弃；`kIndeterminate` 在未达到 `maximum_interval_subdivision_depth` 时二分；达到深度仍不能证明时丢弃。边界根最多执行 `maximum_root_iterations`，每个最终 `T` 再调用点值硬约束验证。

- [ ] **Step 5: 实现时间主目标和时间等价池次级排序**

候选先按 `expected_execution_time_s` 求 \(T_{\min}\)，只保留
`T <= T_min + delta_t_equivalence_s` 的等价池参与能量、着陆余量和净空排序。候选 ID 由 aim ID、量化 T 和重力模型 ID 哈希生成；最终稳定键为：

```cpp
return std::tuple{
    candidate.expected_execution_time_s,
    candidate.estimated_energy_j,
    -candidate.landing_margin_m,
    -candidate.nominal_minimum_clearance_m,
    candidate.candidate_id};
```

- [ ] **Step 6: 运行一维时间求解测试并确认通过**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_hopper_tests --parallel
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -C Debug -R '^lpp_v3_hopper\.BallisticTimeSolver' --no-tests=error --output-on-failure
```

Expected: 最短可行候选稳定返回，无解和区间无法证明时不产生候选。

- [ ] **Step 7: 提交有界飞行时间求解器**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/hopper/ballistic_time_solver.hpp cpp/src/hopper/ballistic_time_solver.cpp cpp/tests/hopper/ballistic_time_solver_test.cpp cpp/tests/hopper/hopper_test_fixtures.hpp cpp/tests/hopper/hopper_test_fixtures.cpp cpp/src/hopper/CMakeLists.txt cpp/tests/hopper/CMakeLists.txt
git -C path-planner commit -m "feat(planner-v3): solve bounded hopper flight time"
```

---

### Task 9: 对完整弹道构造连续碰撞证书和 `CertifiedFlightTube`

**Files:**
- Create: `path-planner/cpp/include/lunar_path_planner/v3/hopper/flight_tube_certifier.hpp`
- Create: `path-planner/cpp/src/hopper/flight_tube_certifier.cpp`
- Create: `path-planner/cpp/tests/hopper/flight_tube_certifier_test.cpp`
- Modify: `path-planner/cpp/tests/hopper/hopper_test_fixtures.hpp`
- Modify: `path-planner/cpp/tests/hopper/hopper_test_fixtures.cpp`
- Modify: `path-planner/cpp/src/hopper/CMakeLists.txt`
- Modify: `path-planner/cpp/tests/hopper/CMakeLists.txt`

**Interfaces:**
- Consumes: `NominalBallisticArc`、`ImmutableMapSnapshot`、状态/重力确定性误差集、机体碰撞外形和任意姿态旋转外包。
- Produces:

```cpp
namespace lunar::planning::v3 {

struct FlightTubeCertificationInput final {
  NominalBallisticArc arc;
  const ImmutableMapSnapshot* map;
  TerrainCertifiedLandingRegion source_region;
  TerrainCertifiedLandingRegion target_region;
  AxisAlignedBox3 initial_position_error_m;
  AxisAlignedBox3 initial_velocity_error_mps;
  AxisAlignedBox3 launch_execution_velocity_error_mps;
  AxisAlignedBox3 gravity_error_mps2;
  ConvexPolytope3 arbitrary_attitude_body_envelope;
  ContentRef source_snapshot_ref;
  ContentRef body_rotation_envelope_ref;
  ContentRef error_model_ref;
};

struct FlightTubeCertificationDiagnostics final {
  std::size_t time_slab_count;
  std::size_t overlapped_cell_count;
  std::size_t subdivision_count;
  double minimum_clearance_m;
  std::string rejection_reason;
};

struct FlightTubeCertificationResult final {
  std::optional<CertifiedFlightTube> certified_tube;
  FlightTubeCertificationDiagnostics diagnostics;
};

class FlightTubeCertifier final {
 public:
  FlightTubeCertifier(
      HopperCapabilityView capability,
      HopperPlannerLimits limits);

  FlightTubeCertificationResult certify(
      const FlightTubeCertificationInput& input) const;
};

}  // namespace lunar::planning::v3
```

- [ ] **Step 1: 写最高点碰撞、未知单元和有序 tube section 测试**

```cpp
TEST(FlightTubeCertifier, DetectsObstacleBetweenClearEndpoints) {
  auto fixture = make_apex_obstacle_flight_fixture();
  const auto result = make_flight_tube_certifier().certify(fixture.input);
  EXPECT_FALSE(result.certified_tube.has_value());
  EXPECT_EQ(result.diagnostics.rejection_reason,
            "continuous_swept_volume_intersects_obstacle");
}

TEST(FlightTubeCertifier, RejectsAnyOverlappedUnknownCell) {
  const auto fixture = make_unknown_apex_flight_fixture();
  const auto result = make_flight_tube_certifier().certify(fixture.input);
  EXPECT_FALSE(result.certified_tube.has_value());
  EXPECT_EQ(result.diagnostics.rejection_reason,
            "flight_tube_overlaps_unknown_cell");
}

TEST(FlightTubeCertifier, EmitsOrderedConservativeSections) {
  const auto fixture = make_clear_flight_fixture();
  const auto result = make_flight_tube_certifier().certify(fixture.input);
  ASSERT_TRUE(result.certified_tube.has_value());
  EXPECT_TRUE(time_sections_are_ordered_and_cover_closed_interval(
      result.certified_tube->sections,
      0.0, fixture.input.arc.flight_time_s));
  EXPECT_TRUE(dense_uncertainty_samples_are_inside_tube(
      fixture.input, *result.certified_tube, 2000));
}
```

- [ ] **Step 2: 运行测试确认飞行管认证器缺失**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_hopper_tests --parallel
```

Expected: build fails because `FlightTubeCertifier` is undefined.

- [ ] **Step 3: 实现确定性位置包络和事件时间分段**

对任意 `t∈[ta,tb]`，把发射执行速度误差并入初始速度误差，使用区间算术外包：

\[
p_0+\delta p_0+(v_0+\delta v_0)t+
\frac12(g+\delta g)t^2
\]

再与任意姿态机体外包做 Minkowski 和。初始分段点包含发射、最高点、弹道水平投影穿越栅格边界的解析根和着陆；全部时间量化、排序、去重。

- [ ] **Step 4: 实现水平 supercover 和一般重力自适应覆盖**

重力与地图竖直轴平行时，对每个时间段的水平线性投影使用 supercover DDA，包含边界接触单元。一般重力下水平投影是二次曲线；用曲线对 chord 的解析偏差上界判断是否细分。达到 `maximum_collision_subdivision_depth` 仍不能证明 cell 覆盖完整时返回 `collision_projection_inconclusive`。

- [ ] **Step 5: 实现每个重叠单元的解析最小竖直净空**

对固定时间段和 cell 的保守最高地形，净空函数是二次多项式下界。检查段端点及导数为零且位于段内的时刻，求最小值。自由飞行区间内 cell 未知、净空非正或区间算术结果包含零都拒绝候选。

接触例外只允许 `t=0` 的已认证起跳接触和 `t=T` 的目标着陆接触：相应接触单元必须完全属于 `source_region` 或 `target_region` 的已认证地面平面和安全足迹，着陆接触法向必须向下横截，且不得与障碍/未知/非目标地形相交。`minimum_certified_clearance_m` 统计两端接触之外自由飞行段的最小净空；接触由 launch/landing certification 单独记录，不能用接触例外放宽内部时间区间。

- [ ] **Step 6: 构造 schema 一致的凸多面体 sections**

每个时间段输出一个或多个 H-representation 凸包围体；总数超过 `maximum_flight_tube_sections` 时拒绝候选：

```cpp
CertifiedFlightTubeSection{
    .time_interval = make_relative_time_interval_ns(
        time_begin_s, time_end_s),
    .envelope = make_axis_aligned_h_polytope(
        conservative_position_and_body_bounds),
};
```

顶层 `CertifiedFlightTube` 必须写入 `frame_id`、map snapshot ref、body rotation envelope ref、error model ref 和全段最小认证净空。section 必须覆盖整个 `[0,T]`，不允许有空隙。

- [ ] **Step 7: 运行飞行管测试并确认通过**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_hopper_tests --parallel
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -C Debug -R '^lpp_v3_hopper\.FlightTubeCertifier' --no-tests=error --output-on-failure
```

Expected: 中间碰撞和未知单元被拒绝，成功 tube 有序覆盖全飞行时间且包含误差采样。

- [ ] **Step 8: 提交连续碰撞认证**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/hopper/flight_tube_certifier.hpp cpp/src/hopper/flight_tube_certifier.cpp cpp/tests/hopper/flight_tube_certifier_test.cpp cpp/tests/hopper/hopper_test_fixtures.hpp cpp/tests/hopper/hopper_test_fixtures.cpp cpp/src/hopper/CMakeLists.txt cpp/tests/hopper/CMakeLists.txt
git -C path-planner commit -m "feat(planner-v3): certify hopper continuous flight tube"
```

---

### Task 10: 用保守任意轴能力包络认证着陆姿态可达性

**Files:**
- Create: `path-planner/cpp/include/lunar_path_planner/v3/hopper/attitude_certifier.hpp`
- Create: `path-planner/cpp/src/hopper/attitude_certifier.cpp`
- Create: `path-planner/cpp/tests/hopper/attitude_certifier_test.cpp`
- Modify: `path-planner/cpp/tests/hopper/hopper_test_fixtures.hpp`
- Modify: `path-planner/cpp/tests/hopper/hopper_test_fixtures.cpp`
- Modify: `path-planner/cpp/src/hopper/CMakeLists.txt`
- Modify: `path-planner/cpp/tests/hopper/CMakeLists.txt`

**Interfaces:**
- Consumes: `HopperState::orientation_body_to_frame`、`angular_velocity_radps`
  及其确定性误差集合、着陆平面/allowed yaw、能力中的保守任意轴角速度/
  角加速度和可选收紧表。
- Produces:

```cpp
namespace lunar::planning::v3 {

class AttitudeTighteningTable;

struct ArbitraryAxisAttitudeCapability final {
  double maximum_angular_speed_radps;
  double maximum_angular_acceleration_radps2;
  double maximum_initial_angular_speed_radps;
  DurationNanoseconds minimum_settle_guard;
  ContentRef source_safety_capability_ref;
  std::optional<ContentRef> attitude_tightening_table_ref;
  std::shared_ptr<const AttitudeTighteningTable>
      resolved_attitude_tightening_table;
  HopperErrorBounds certified_state_error_bounds;
};

struct AttitudeCertificationInput final {
  Eigen::Quaterniond initial_orientation_body_to_frame;
  Eigen::Vector3d initial_angular_velocity_radps;
  RotationVectorBall initial_orientation_error_set;
  DeterministicVectorSet3 initial_angular_velocity_error_set_radps;
  LandingPlane landing_plane;
  CircularYawInterval allowed_yaw_interval;
  double flight_time_s;
};

struct AttitudeCertificationResult final {
  std::optional<AttitudeBoundary> boundary;
  double shortest_rotation_angle_rad;
  double certified_required_rotation_time_s;
  double worst_case_initial_angular_speed_radps;
  double effective_maximum_speed_radps;
  double effective_maximum_acceleration_radps2;
  std::string rejection_reason;
};

class AttitudeCertifier final {
 public:
  explicit AttitudeCertifier(
      ArbitraryAxisAttitudeCapability capability);

  Result<double> conservative_required_time_s(
      const Eigen::Quaterniond& initial_orientation_body_to_frame,
      const Eigen::Vector3d& initial_angular_velocity_radps,
      const RotationVectorBall& initial_orientation_error_set,
      const DeterministicVectorSet3&
          initial_angular_velocity_error_set_radps,
      const LandingPlane& landing_plane,
      const CircularYawInterval& allowed_yaw_interval) const;

  AttitudeCertificationResult certify(
      const AttitudeCertificationInput& input) const;
};

double shortest_quaternion_angle_rad(
    const Eigen::Quaterniond& from,
    const Eigen::Quaterniond& to);

double bang_bang_rotation_time_s(
    double angle_rad,
    double maximum_speed_radps,
    double maximum_acceleration_radps2);

}  // namespace lunar::planning::v3
```

`ArbitraryAxisAttitudeCapability` 只能在请求校验阶段由
`HopperCapability.attitude_envelope`、同一 profile 的误差边界和可选
`attitude_tightening_table_ref` 构造；在线认证不得按字符串再加载查表。
`resolved_attitude_tightening_table` 必须与该 `ContentRef` 的 ID、revision 和
hash 完全匹配。

- [ ] **Step 1: 写四元数最短角、三角/梯形时间和收紧表测试**

```cpp
TEST(AttitudeCertifier, QuaternionSignDoesNotChangeShortestAngle) {
  const Eigen::Quaterniond identity = Eigen::Quaterniond::Identity();
  const Eigen::Quaterniond target{
      Eigen::AngleAxisd(0.7, Eigen::Vector3d::UnitY())};
  EXPECT_NEAR(shortest_quaternion_angle_rad(identity, target),
              shortest_quaternion_angle_rad(
                  identity,
                  Eigen::Quaterniond{-target.w(), -target.x(),
                                     -target.y(), -target.z()}),
              1e-12);
}

TEST(AttitudeCertifier, BangBangTimeUsesTriangularAndTrapezoidalCases) {
  EXPECT_NEAR(bang_bang_rotation_time_s(0.25, 1.0, 2.0),
              2.0 * std::sqrt(0.25 / 2.0), 1e-12);
  EXPECT_NEAR(bang_bang_rotation_time_s(2.0, 1.0, 2.0),
              2.5, 1e-12);
}

TEST(AttitudeCertifier, LookupCanOnlyTightenBaseEnvelope) {
  auto capability = make_attitude_capability();
  capability = apply_tightening_lookup(
      capability,
      AttitudeLookupEntry{
          .maximum_speed_radps = 2.0,
          .maximum_acceleration_radps2 = 4.0});
  EXPECT_LE(capability.maximum_angular_speed_radps,
            make_attitude_capability().maximum_angular_speed_radps);
  EXPECT_LE(capability.maximum_angular_acceleration_radps2,
             make_attitude_capability().maximum_angular_acceleration_radps2);
}

TEST(LandingGraph, SelectsBestActualTimeAcrossCompetitiveLowerBounds) {
  auto fixture = make_competitive_first_edge_graph(
      /*first_lower_bound_s=*/1.0, /*first_actual_s=*/10.0,
      /*second_lower_bound_s=*/2.0, /*second_actual_s=*/3.0);
  const auto selection = SelectWithTimedOracle(fixture);
  ASSERT_TRUE(IsOk(selection));
  EXPECT_EQ(std::get<LazyNextHopSelection>(selection).edge_index,
            fixture.second_edge_index);
}

TEST(HopperCapabilityBinding, RejectsAnyLookupEntryThatWidensBaseEnvelope) {
  auto profile = make_hopper_profile_with_tightening_table();
  profile.resolved_attitude_tightening_table->entries.front()
      .maximum_speed_radps =
      profile.content.attitude_envelope.maximum_angular_speed_radps + 0.1;
  const auto result =
      bind_hopper_capability(profile, make_resolved_capability_bindings());
  EXPECT_FALSE(IsOk(result));
  EXPECT_EQ(std::get<Error>(result).field_path,
            "attitude_tightening_table.entries[0].maximum_speed_radps");
}
```

- [ ] **Step 2: 写近零发射角速度和着陆前稳定余量失败测试**

```cpp
TEST(AttitudeCertifier, RejectsLaunchAngularVelocityOutsideNearZeroBound) {
  auto input = make_attitude_certification_input();
  input.initial_angular_velocity_radps =
      Eigen::Vector3d{0.0, 0.0, 0.2};
  const auto result = AttitudeCertifier{
      make_attitude_capability()}.certify(input);
  EXPECT_FALSE(result.boundary.has_value());
  EXPECT_EQ(result.rejection_reason,
            "launch_angular_velocity_outside_certified_bound");
}

TEST(AttitudeCertifier, RequiresRotationSettleAndGuardBeforeLanding) {
  auto input = make_attitude_certification_input();
  input.flight_time_s = 0.05;
  const auto result = AttitudeCertifier{
      make_attitude_capability()}.certify(input);
  EXPECT_FALSE(result.boundary.has_value());
  EXPECT_EQ(result.rejection_reason,
             "insufficient_attitude_settle_time");
}

TEST(AttitudeCertifier, AdverseInitialSpinAndErrorIncreaseRequiredTime) {
  auto input = make_attitude_certification_input();
  input.initial_angular_velocity_radps =
      -0.08 * target_shortest_rotation_axis(input);
  input.initial_angular_velocity_error_set_radps =
      make_euclidean_ball_vector_set(0.02);
  const auto required = AttitudeCertifier{
      make_attitude_capability()}.conservative_required_time_s(
          input.initial_orientation_body_to_frame,
          input.initial_angular_velocity_radps,
          input.initial_orientation_error_set,
          input.initial_angular_velocity_error_set_radps,
          input.landing_plane,
          input.allowed_yaw_interval);
  ASSERT_TRUE(IsOk(required));
  EXPECT_GT(std::get<double>(required),
            required_time_from_rest_for_same_nominal_angle(input));
}
```

- [ ] **Step 3: 运行测试确认姿态认证器缺失**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_hopper_tests --parallel
```

Expected: build fails because attitude certification functions are absent.

- [ ] **Step 4: 实现目标姿态集合和最短四元数角**

由着陆平面法向确定机体着陆 up 轴，yaw 使用 `CircularYawInterval`。有限候选取区间起点、中心、终点和使最短角驻定的解析 yaw；每个候选四元数归一化并固定 `w>=0` 的 canonical sign。

```cpp
const double dot = std::clamp(
    std::abs(from.normalized().dot(to.normalized())), 0.0, 1.0);
return 2.0 * std::acos(dot);
```

- [ ] **Step 5: 实现保守 bang-bang 时间和只收紧能力表**

令切换角为 \(\theta_s=\omega_{\max}^2/\alpha_{\max}\)：

```cpp
if (angle_rad <= square(maximum_speed_radps) /
                         maximum_acceleration_radps2) {
  return 2.0 * std::sqrt(
      angle_rad / maximum_acceleration_radps2);
}
return 2.0 * maximum_speed_radps /
           maximum_acceleration_radps2 +
       (angle_rad -
        square(maximum_speed_radps) /
            maximum_acceleration_radps2) /
           maximum_speed_radps;
```

静止起步公式只是内部子步骤。对每个目标 yaw 候选，认证器先计算

```cpp
const double omega0_bound =
    initial_angular_velocity_radps.norm() +
    maximum_vector_norm(initial_angular_velocity_error_set_radps);
const double angle_bound = std::min(
    std::numbers::pi,
    nominal_shortest_angle_rad + initial_orientation_error_set.radius_rad);
const double brake_time =
    omega0_bound / effective_maximum_acceleration_radps2;
const double adverse_brake_angle =
    square(omega0_bound) /
    (2.0 * effective_maximum_acceleration_radps2);
const double required_time =
    brake_time +
    bang_bang_rotation_time_s(
        std::min(std::numbers::pi,
                 angle_bound + adverse_brake_angle),
        effective_maximum_speed_radps,
        effective_maximum_acceleration_radps2);
```

`omega0_bound` 超过 `maximum_initial_angular_speed_radps` 时直接拒绝。上述先按
最坏方向制动、再从静止旋转的上界必须用于 flight-time 可行性判断；不得用忽略
初始角速度的静止公式认证边界。

请求绑定/semantic validation 阶段逐条证明全部表项不大于基础包络；任一字段
试图放宽就拒绝整个 `SafetyCapabilityProfile`，不得把违规表带入规划调用。
在线认证仍逐字段取 `min(base, lookup)`，只作为已校验内容遭内存破坏或实现回归
时的防御性二次保护，不能把非法表降级为可接受输入。

- [ ] **Step 6: 构造不含完整姿态轨迹的 `AttitudeBoundary`**

只有
`rotation_time + minimum_settle_guard <= flight_time`
且发射/着陆角速度边界满足时输出 boundary。boundary 只记录初始误差、近零角速度界、目标姿态集合、着陆角速度界和稳定余量；不得输出姿态时间序列。

- [ ] **Step 7: 运行姿态认证测试并确认通过**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_hopper_tests --parallel
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -C Debug -R '^lpp_v3_hopper\.AttitudeCertifier' --no-tests=error --output-on-failure
```

Expected: 最短角和旋转时间正确，查表只能收紧，角速度或稳定时间不足时拒绝。

- [ ] **Step 8: 提交姿态能力包络认证**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/hopper/attitude_certifier.hpp cpp/src/hopper/attitude_certifier.cpp cpp/tests/hopper/attitude_certifier_test.cpp cpp/tests/hopper/hopper_test_fixtures.hpp cpp/tests/hopper/hopper_test_fixtures.cpp cpp/src/hopper/CMakeLists.txt cpp/tests/hopper/CMakeLists.txt
git -C path-planner commit -m "feat(planner-v3): certify hopper attitude envelope"
```

---

### Task 11: 传播确定性误差并构造落点最坏情况外包

**Files:**
- Create: `path-planner/cpp/include/lunar_path_planner/v3/hopper/landing_set_propagator.hpp`
- Create: `path-planner/cpp/src/hopper/landing_set_propagator.cpp`
- Create: `path-planner/cpp/tests/hopper/landing_set_propagator_test.cpp`
- Modify: `path-planner/cpp/tests/hopper/hopper_test_fixtures.hpp`
- Modify: `path-planner/cpp/tests/hopper/hopper_test_fixtures.cpp`
- Modify: `path-planner/cpp/src/hopper/CMakeLists.txt`
- Modify: `path-planner/cpp/tests/hopper/CMakeLists.txt`

**Interfaces:**
- Consumes: 初始位置/速度、重力、发射执行和着陆平面的确定性集合，Task 7 的名义弹道，以及 schema 的显式 bounded-set discriminator。
- Produces:

```cpp
namespace lunar::planning::v3 {

struct LandingPlaneUncertainty final {
  AxisAlignedBox3 origin_error_m;
  RotationVectorBall normal_error;
  SymmetricScalarInterval residual_error_m;
};

struct LandingSetPropagationInput final {
  NominalBallisticArc arc;
  LandingPlane landing_plane;
  AxisAlignedBox3 initial_position_error_m;
  AxisAlignedBox3 initial_velocity_error_mps;
  AxisAlignedBox3 gravity_error_mps2;
  AxisAlignedBox3 launch_execution_velocity_error_mps;
  LandingPlaneUncertainty landing_plane_error;
  CircularYawInterval certified_landing_yaw_interval;
  ContentRef source_error_model_ref;
};

struct LandingSetPropagationDiagnostics final {
  std::size_t impact_root_iteration_count;
  std::size_t support_direction_count;
  double worst_case_downward_normal_speed_mps;
  std::string rejection_reason;
};

struct LandingSetPropagationResult final {
  std::optional<PredictedLandingFootprint> footprint;
  LandingSetPropagationDiagnostics diagnostics;
};

class LandingSetPropagator final {
 public:
  LandingSetPropagator(
      HopperCapabilityView capability,
      HopperPlannerLimits limits);

  LandingSetPropagationResult propagate(
      const LandingSetPropagationInput& input) const;
};

ValidationReport validate_landing_containment(
    const PredictedLandingFootprint& footprint,
    const NextLandingRegion& region,
    const AttitudeBoundary& attitude_boundary);

}  // namespace lunar::planning::v3
```

- [ ] **Step 1: 写零误差点落地、误差扩张和向下横截性测试**

```cpp
TEST(LandingSetPropagator, ZeroErrorCollapsesToNominalImpact) {
  const auto input = make_zero_error_landing_set_input();
  const auto result = make_landing_set_propagator().propagate(input);
  ASSERT_TRUE(result.footprint.has_value());
  EXPECT_TRUE(polygon_contains_uv(
      result.footprint->convex_center_landing_polygon.vertices_uv,
      world_to_plane_uv(input.landing_plane,
                        input.arc.aim_position_m)));
  const auto expected_impact_offset =
      exact_duration_from_seconds(input.arc.flight_time_s);
  EXPECT_EQ(
      result.footprint->landing_time_window.start_offset.value,
      expected_impact_offset.value);
  EXPECT_EQ(
      result.footprint->landing_time_window.end_offset.value,
      expected_impact_offset.value);
}

TEST(LandingSetPropagator, LargerDeterministicErrorsCannotShrinkOuterSet) {
  const auto small = make_landing_set_propagator().propagate(
      make_landing_set_input_with_error_scale(0.5));
  const auto large = make_landing_set_propagator().propagate(
      make_landing_set_input_with_error_scale(1.0));
  ASSERT_TRUE(small.footprint.has_value());
  ASSERT_TRUE(large.footprint.has_value());
  EXPECT_TRUE(polygon_is_subset(
      small.footprint->convex_center_landing_polygon.vertices_uv,
      large.footprint->convex_center_landing_polygon.vertices_uv));
}

TEST(LandingSetPropagator, RejectsTangencyOrPossibleMultipleImpactRoots) {
  const auto result = make_landing_set_propagator().propagate(
      make_tangent_plane_impact_input());
  EXPECT_FALSE(result.footprint.has_value());
  EXPECT_EQ(result.diagnostics.rejection_reason,
            "downward_transversality_not_proven");
}
```

- [ ] **Step 2: 运行测试确认落点集合传播器缺失**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_hopper_tests --parallel
```

Expected: build fails because `LandingSetPropagator` is undefined.

- [ ] **Step 3: 实现着陆平面相交时间区间求根**

建立包含全部确定性集合的有符号接触距离区间函数
`D(t)=n·(p_center(t)-plane_origin)-certified_landing_center_normal_offset`
。先在名义飞行时间邻域按解析可达时间界建立 bracket，再用区间二分收缩至配置容差或 `maximum_root_iterations`。仅当：

```cpp
upper_bound(normal_velocity_interval_at_impact) <
    -capability.minimum_downward_impact_speed_mps
```

时证明向下横截；可能切触、无 bracket、多根或迭代耗尽均拒绝。

- [ ] **Step 4: 实现完整误差和时间区间上的位置/速度传播**

用 interval arithmetic 传播 `p0`、`v0`、launch execution error、`g` 和 `[T-,T+]`。每个运算保持向外舍入；结果转换到含误差的着陆平面基时继续外包，不以名义平面投影替代。

- [ ] **Step 5: 用固定支撑方向生成有限顶点凸外包**

按 `support_direction_count` 生成稳定平面方向，对每个方向求位置集合支撑上界，交成 H-polygon，再转换为不超过 `landing_region_maximum_vertices` 的 CCW 顶点；若需要减点，只能向外保守并把增加量记录到 `outer_approximation_margin_m`。

`PredictedLandingFootprint` 同时写入 landing plane、time window、velocity bounds、landing yaw interval、source error model ref 和外包裕量。它是着陆时间窗与线速度边界的唯一真值；`HopperReference` 顶层不得复制。`validate_landing_containment()` 还必须用 canonical CCW 圆周区间运算验证 `footprint landing yaw ⊆ target attitude yaw ⊆ landing region yaw`。

- [ ] **Step 6: 实现 Minkowski 裕量后的完整区域包含检查**

对 footprint 每个顶点和 `NextLandingRegion` 每条内向单位半平面，检查：

```cpp
normal.dot(vertex_uv) >=
    boundary_offset +
    region.inward_safety_margin_m +
    footprint.outer_approximation_margin_m;
```

所有量必须在同一 `LandingPlane` 基中；平面 ID、frame 或基不一致直接失败，不允许静默转换。

- [ ] **Step 7: 运行落点集合测试并确认通过**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_hopper_tests --parallel
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -C Debug -R '^lpp_v3_hopper\.LandingSetPropagator' --no-tests=error --output-on-failure
```

Expected: 误差增大不缩小外包，切触/多根被拒绝，成功结果包含完整时间和速度边界。

- [ ] **Step 8: 提交确定性落点范围传播**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/hopper/landing_set_propagator.hpp cpp/src/hopper/landing_set_propagator.cpp cpp/tests/hopper/landing_set_propagator_test.cpp cpp/tests/hopper/hopper_test_fixtures.hpp cpp/tests/hopper/hopper_test_fixtures.cpp cpp/src/hopper/CMakeLists.txt cpp/tests/hopper/CMakeLists.txt
git -C path-planner commit -m "feat(planner-v3): propagate deterministic hopper landing set"
```

---

### Task 12: 聚合全部子证书并生成唯一 `HopperReference`

**Files:**
- Create: `path-planner/cpp/include/lunar_path_planner/v3/hopper/hop_certifier.hpp`
- Create: `path-planner/cpp/src/hopper/hop_certifier.cpp`
- Create: `path-planner/cpp/tests/hopper/hop_certifier_test.cpp`
- Modify: `path-planner/cpp/tests/hopper/hopper_test_fixtures.hpp`
- Modify: `path-planner/cpp/tests/hopper/hopper_test_fixtures.cpp`
- Modify: `path-planner/cpp/src/hopper/CMakeLists.txt`
- Modify: `path-planner/cpp/tests/hopper/CMakeLists.txt`

**Interfaces:**
- Consumes: Tasks 7–11、完整 `HopperState`、起跳/目标着陆域、目标域的有限瞄准点、不可变地图和 schema codec/validator。
- Produces:

```cpp
namespace lunar::planning::v3 {

struct HopCertificationContext final {
  ClockStamp reference_time_origin;
  HopperState launch_state;
  const ImmutableMapSnapshot* map;
  HopperCapabilityView capability;
  HopperPlannerLimits limits;
  std::string source_request_id;
  ContentRef source_error_model_ref;
};

struct HopCertificationDiagnostics final {
  std::size_t ballistic_candidate_count;
  std::size_t fully_attempted_candidate_count;
  std::size_t certified_candidate_count;
  std::vector<std::string> candidate_rejection_reasons;
  std::string selected_candidate_id;
};

class HopCertifier final : public NextHopCertificationOracle {
 public:
  explicit HopCertifier(HopCertificationContext context);

  FirstEdgeCertificationResult certify_first_edge(
      const LandingGraphNode& from,
      const LandingGraphNode& to) override;

  const HopCertificationDiagnostics& diagnostics() const noexcept;

 private:
  HopCertificationContext context_;
  HopCertificationDiagnostics diagnostics_;
};

ValidationReport validate_final_hopper_reference(
    const HopperReference& reference,
    const HopCertificationContext& context);

}  // namespace lunar::planning::v3
```

- [ ] **Step 1: 写任一缺失子证书都不产生引用的参数化测试**

```cpp
class HopCertifierFailureTest
    : public ::testing::TestWithParam<HopFailureMode> {};

TEST_P(HopCertifierFailureTest, NeverLeaksPartiallyCertifiedReference) {
  auto fixture = make_hop_certification_fixture();
  apply_failure_mode(fixture, GetParam());
  HopCertifier certifier(fixture.context);
  const auto result = certifier.certify_first_edge(
      fixture.from_node, fixture.to_node);
  EXPECT_FALSE(result.certified_candidate.has_value());
  EXPECT_FALSE(result.rejection_reason.empty());
}

INSTANTIATE_TEST_SUITE_P(
    RequiredCertificates,
    HopCertifierFailureTest,
    ::testing::Values(
        HopFailureMode::kTerrain,
        HopFailureMode::kLaunchBoundary,
        HopFailureMode::kFlightTube,
        HopFailureMode::kAttitude,
        HopFailureMode::kLandingVelocity,
        HopFailureMode::kLandingSetContainment,
        HopFailureMode::kPostLandingHold));
```

- [ ] **Step 2: 写成功引用只能含一个发射命令且字段来源一致的测试**

```cpp
TEST(HopCertifier, SuccessfulReferenceContainsOneUniqueJumpBoundary) {
  const auto fixture = make_hop_certification_fixture();
  HopCertifier certifier(fixture.context);
  const auto result = certifier.certify_first_edge(
      fixture.from_node, fixture.to_node);
  ASSERT_TRUE(result.certified_candidate.has_value());
  const HopperReference& reference =
      result.certified_candidate->reference;

  EXPECT_FALSE(reference.jump_boundary.boundary_id.empty());
  EXPECT_TRUE(is_exactly_zero(
      reference.ground_hold_anchor.hold_state.linear_velocity_mps));
  EXPECT_TRUE(is_exactly_zero(
      reference.ground_hold_anchor.hold_state.angular_velocity_radps));
  EXPECT_EQ(reference.jump_boundary.gravity_model_ref.id,
            fixture.context.capability.gravity_model.model_id);
  EXPECT_EQ(reference.next_landing_region.region_id,
            fixture.to_node.region.region_id);
  EXPECT_TRUE(validate_landing_containment(
      reference.predicted_landing_footprint,
      reference.next_landing_region,
      reference.attitude_boundary).ok());
  EXPECT_TRUE(validate_final_hopper_reference(
      reference, fixture.context).ok());
}

TEST(HopCertifier, NominalAimPointIsExplanatoryNotAlternateCommand) {
  const auto reference = make_fully_certified_hopper_reference();
  nlohmann::json encoded;
  encode_json(encoded, reference);
  EXPECT_NE(reference.nominal_aim_point.aim_point_id,
            reference.jump_boundary.boundary_id);
  EXPECT_FALSE(encoded.contains("selectable_aim_points"));
  EXPECT_FALSE(encoded.contains("jump_boundaries"));
}

TEST(HopCertifier, RejectsLandingYawOutsideAttitudeOrTerrainInterval) {
  auto reference = make_fully_certified_hopper_reference();
  reference.predicted_landing_footprint.landing_yaw_interval =
      make_canonical_yaw_interval(1.0, 1.0);
  reference.attitude_boundary.target_attitude_set.allowed_yaw_interval =
      make_canonical_yaw_interval(1.2, 0.2);
  EXPECT_FALSE(validate_final_hopper_reference(
      reference, make_hop_validation_context()).ok());

  reference = make_fully_certified_hopper_reference();
  reference.attitude_boundary.target_attitude_set.allowed_yaw_interval =
      make_canonical_yaw_interval(2.0, 0.5);
  reference.next_landing_region.allowed_yaw_interval =
      make_canonical_yaw_interval(-0.2, 0.4);
  EXPECT_FALSE(validate_final_hopper_reference(
      reference, make_hop_validation_context()).ok());
}
```

- [ ] **Step 3: 运行测试确认最终认证聚合器缺失**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_hopper_tests --parallel
```

Expected: build fails because `HopCertifier` and final validation are absent.

- [ ] **Step 4: 实现边内所有瞄准点的弹道候选生成**

先调用 `AttitudeCertifier::conservative_required_time_s()`，把初始姿态/角速度
及其确定性误差的最坏方向制动时间纳入安全飞行时间下界，再把目标节点内的全部
有限瞄准点交给 `BallisticTimeSolver`。候选按预计执行时间排序，并保留时间等价池；
完整认证尝试不超过配置上限。

- [ ] **Step 5: 为每个候选顺序执行完整物理认证**

对每个候选严格执行：

1. 重新验证目标 `TerrainCertifiedLandingRegion` 与当前 map snapshot/capability；
2. 验证正飞行时间、名义发射速度、执行误差和 actuator/impulse profile；
3. 调用 `FlightTubeCertifier`；
4. 调用 `AttitudeCertifier`；
5. 调用 `LandingSetPropagator`；
6. 验证着陆速度、时间窗、向下横截性和 post-landing stable hold；
7. 调用 `validate_landing_containment()`，同时验证位置外包包含和两级 yaw
   子集关系。

任何一步失败只记录稳定 reason code 并转向下一个候选；不得保留该候选的部分输出。

- [ ] **Step 6: 在完整认证候选中执行最终时间等价池排序**

主键仍是预计执行时间。只有满足
`time <= certified_time_min + delta_t_equivalence_s`
的完整认证候选才比较能量、landing margin、tube minimum clearance 和稳定 candidate ID。

- [ ] **Step 7: 一次性构造完整 schema DTO 并终验**

构造内容必须包括：

```cpp
HopperReference reference{
    .reference_id = make_reference_id(selected),
    .reference_hash = "",
    .reference_time_origin =
        context_.reference_time_origin,
    .translation_model =
        HopperReference::TranslationModel::
            kPureBallisticNoInflightTranslationControl,
    .ground_hold_anchor = make_certified_ground_hold_anchor(
        context_.launch_state, selected.source_region),
    .next_landing_region = make_next_landing_region(
        selected.target_region),
    .jump_boundary = make_unique_jump_boundary(selected),
    .predicted_landing_footprint = selected.landing_set,
    .certified_flight_tube = selected.flight_tube,
    .attitude_boundary = selected.attitude_boundary,
    .nominal_aim_point = NominalAimPoint{
        .authority = NominalAimPoint::Authority::
            kNonAuthoritativeExplanatory,
        .position_m = to_contract_vec3(
            selected.aim_point.position_m),
    },
    .physical_certification_ref =
        make_physical_certification_ref(selected),
    .future_route_preview = make_empty_non_authoritative_preview(),
};
reference.reference_hash =
    canonical_hopper_reference_hash(reference);
```

`make_certified_ground_hold_anchor()` 只接受当前已认证着陆地面上的静止状态，并
强制线速度/角速度逐分量为零；它保存起跳锁定前的 committed hold，不是发射命令。
`make_unique_jump_boundary()` 只写一组 `nominal_launch_state`，误差容差只进入
`allowed_launch_state_error_set`，并固定
`ballistic_time_origin = BALLISTIC_LAUNCH_EVENT`。tube section offset 从 0 连续
覆盖到 `ballistic_flight_time`，landing window 包含同一 `T`。对 DTO 依次执行
`validate_final_hopper_reference()`、合同层 `validate(reference, context)` 和
encode/decode round-trip；任一失败不返回 reference。

- [ ] **Step 8: 运行最终认证测试并确认通过**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_hopper_tests --parallel
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -C Debug -R '^lpp_v3_hopper\.HopCertifier' --no-tests=error --output-on-failure
```

Expected: 所有注入失败都无引用，成功引用只有一个边界命令且通过完整终验。

- [ ] **Step 9: 提交最终飞跃认证器**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/hopper/hop_certifier.hpp cpp/src/hopper/hop_certifier.cpp cpp/tests/hopper/hop_certifier_test.cpp cpp/tests/hopper/hopper_test_fixtures.hpp cpp/tests/hopper/hopper_test_fixtures.cpp cpp/src/hopper/CMakeLists.txt cpp/tests/hopper/CMakeLists.txt
git -C path-planner commit -m "feat(planner-v3): aggregate certified hopper reference"
```

---

### Task 13: 实现无副作用的飞跃承诺状态机

**Files:**
- Create: `path-planner/cpp/include/lunar_path_planner/v3/hopper/hopper_commitment_state_machine.hpp`
- Create: `path-planner/cpp/src/hopper/hopper_commitment_state_machine.cpp`
- Create: `path-planner/cpp/tests/hopper/hopper_commitment_state_machine_test.cpp`
- Modify: `path-planner/cpp/tests/hopper/hopper_test_fixtures.hpp`
- Modify: `path-planner/cpp/tests/hopper/hopper_test_fixtures.cpp`
- Modify: `path-planner/cpp/src/hopper/CMakeLists.txt`
- Modify: `path-planner/cpp/tests/hopper/CMakeLists.txt`

**Interfaces:**
- Consumes: `PreviousExecutionContext`、bundle/boundary ID 和控制器已观测事件；不消费 executor handle。
- Produces:

```cpp
namespace lunar::planning::v3 {

enum class HopperCommitmentState : std::uint8_t {
  kGroundHold,
  kJumpReady,
  kJumpCommitted,
  kInFlight,
  kLandedHold,
  kEmergencyDelegated,
};

enum class HopperCommitmentEventType : std::uint8_t {
  kPublishCertifiedCandidate,
  kWithdrawCandidate,
  kLockJumpBoundary,
  kDetectLaunch,
  kDetectStableLanding,
  kInvalidateCommittedAction,
};

enum class HopperCommitmentScope : std::uint8_t {
  kGroundHold,
  kJumpBoundary,
  kEmergencyDelegated,
};

struct HopperCommitmentSnapshot final {
  HopperCommitmentState state;
  std::optional<std::string> active_bundle_id;
  std::optional<std::string> active_boundary_id;
  HopperCommitmentScope commitment_scope;
};

struct HopperCommitmentEvent final {
  HopperCommitmentEventType type;
  std::optional<std::string> bundle_id;
  std::optional<std::string> boundary_id;
};

struct HopperCommitmentTransition final {
  HopperCommitmentSnapshot next;
  bool accepted;
  ReasonCode reason_code;
};

class HopperCommitmentStateMachine final {
 public:
  HopperCommitmentTransition transition(
      const HopperCommitmentSnapshot& current,
      const HopperCommitmentEvent& event) const;

  bool may_publish_new_jump(
      const HopperCommitmentSnapshot& current) const noexcept;
};

}  // namespace lunar::planning::v3
```

- [ ] **Step 1: 写正常状态序列和 bundle view 语义测试**

```cpp
TEST(HopperCommitmentStateMachine, FollowsCertifiedJumpLifecycle) {
  HopperCommitmentStateMachine machine;
  auto state = make_ground_hold_snapshot();

  state = require_accepted(machine.transition(
      state, publish_candidate_event("bundle-1", "boundary-1")));
  EXPECT_EQ(state.state, HopperCommitmentState::kJumpReady);
  EXPECT_EQ(state.commitment_scope,
            HopperCommitmentScope::kGroundHold);

  state = require_accepted(machine.transition(
      state, lock_boundary_event("bundle-1", "boundary-1")));
  EXPECT_EQ(state.state, HopperCommitmentState::kJumpCommitted);
  EXPECT_EQ(state.commitment_scope,
            HopperCommitmentScope::kJumpBoundary);

  state = require_accepted(machine.transition(
      state, detect_launch_event("boundary-1")));
  EXPECT_EQ(state.state, HopperCommitmentState::kInFlight);

  state = require_accepted(machine.transition(
      state, detect_stable_landing_event()));
  EXPECT_EQ(state.state, HopperCommitmentState::kLandedHold);
  EXPECT_TRUE(machine.may_publish_new_jump(state));
}
```

- [ ] **Step 2: 写锁定后替换、飞行中重定向和未稳定着陆失败测试**

```cpp
TEST(HopperCommitmentStateMachine, RejectsReplacementAfterBoundaryLock) {
  const auto committed = make_jump_committed_snapshot(
      "bundle-1", "boundary-1");
  const auto result = HopperCommitmentStateMachine{}.transition(
      committed,
      publish_candidate_event("bundle-2", "boundary-2"));
  EXPECT_FALSE(result.accepted);
  EXPECT_EQ(result.reason_code,
            "COMMITTED_JUMP_CANNOT_BE_REPLACED");
}

TEST(HopperCommitmentStateMachine, InvalidationDelegatesToEmergencyState) {
  const auto in_flight = make_in_flight_snapshot(
      "bundle-1", "boundary-1");
  const auto result = HopperCommitmentStateMachine{}.transition(
      in_flight, invalidate_committed_action_event());
  ASSERT_TRUE(result.accepted);
  EXPECT_EQ(result.next.state,
            HopperCommitmentState::kEmergencyDelegated);
  EXPECT_FALSE(HopperCommitmentStateMachine{}.may_publish_new_jump(
      result.next));
}

TEST(HopperCommitmentStateMachine, LaunchCannotSkipBoundaryLock) {
  const auto ready = make_jump_ready_snapshot(
      "bundle-1", "boundary-1");
  const auto result = HopperCommitmentStateMachine{}.transition(
      ready, detect_launch_event("boundary-1"));
  EXPECT_FALSE(result.accepted);
}
```

- [ ] **Step 3: 运行测试确认状态机缺失**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_hopper_tests --parallel
```

Expected: build fails because commitment state types and reducer are absent.

- [ ] **Step 4: 实现显式 transition table**

使用 `switch(current.state)` 和每个状态允许事件白名单实现纯函数 reducer。事件中的 bundle/boundary ID 必须与当前 snapshot 一致。拒绝转换返回原 snapshot，不得隐式修正 ID 或跨越状态。

- [ ] **Step 5: 固化替换与承诺边界规则**

- `kGroundHold`/`kLandedHold` 可发布认证候选；
- `kJumpReady` 可撤回或用新完整认证候选替换；
- `kJumpCommitted` 只能接受匹配 boundary 的 launch 或失效事件；
- `kInFlight` 只能接受稳定着陆或失效事件；
- `kEmergencyDelegated` 不产生正常规划转换；
- 稳定 `kLandedHold` 后才可发布下一跳。

`kGroundHold`/`kJumpReady`/`kLandedHold` 的 runtime commitment scope 为
`kGroundHold`；lock 成功后 `kJumpCommitted`/`kInFlight` 为
`kJumpBoundary`；失效后为 `kEmergencyDelegated`。scope 是既有执行上下文的
状态，不会把 immutable bundle 的 committed-prefix component 从 ground hold
改写成 jump。状态机不调用控制器，不改变地图，不发送 `JumpBoundary`。

- [ ] **Step 6: 运行状态机测试并确认通过**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_hopper_tests --parallel
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -C Debug -R '^lpp_v3_hopper\.HopperCommitmentStateMachine' --no-tests=error --output-on-failure
```

Expected: 正常生命周期通过，所有越级、错 ID、锁后替换和飞行重定向被拒绝。

- [ ] **Step 7: 提交飞跃承诺状态机**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/hopper/hopper_commitment_state_machine.hpp cpp/src/hopper/hopper_commitment_state_machine.cpp cpp/tests/hopper/hopper_commitment_state_machine_test.cpp cpp/tests/hopper/hopper_test_fixtures.hpp cpp/tests/hopper/hopper_test_fixtures.cpp cpp/src/hopper/CMakeLists.txt cpp/tests/hopper/CMakeLists.txt
git -C path-planner commit -m "feat(planner-v3): add hopper commitment state machine"
```

---

### Task 14: 编排着陆域、惰性图和物理认证为平台规划器

**Files:**
- Create: `path-planner/cpp/include/lunar_path_planner/v3/hopper/hopper_planner.hpp`
- Create: `path-planner/cpp/src/hopper/hopper_planner.cpp`
- Create: `path-planner/cpp/tests/hopper/hopper_planner_test.cpp`
- Modify: `path-planner/cpp/tests/hopper/hopper_test_fixtures.hpp`
- Modify: `path-planner/cpp/tests/hopper/hopper_test_fixtures.cpp`
- Modify: `path-planner/cpp/src/hopper/CMakeLists.txt`
- Modify: `path-planner/cpp/tests/hopper/CMakeLists.txt`

**Interfaces:**
- Consumes: 共享层已经校验的 `PlanningRequest` 和 `ResolvedTerminalSet`、
  Tasks 1–12。Task 13 承诺状态由总集成层在调用本平台规划器之前处理，平台层
  不重复仲裁 execution directive。
- Produces:

```cpp
namespace lunar::planning::v3 {

struct HopperPlanningDiagnostics final {
  std::size_t safe_pose_cell_count;
  std::size_t landing_region_count;
  std::size_t graph_node_count;
  std::size_t graph_edge_count;
  std::size_t full_certification_attempt_count;
  std::string graph_termination_reason;
  std::string physical_rejection_reason;
  bool safe_dead_end;
};

struct HopperPlanResult final {
  std::optional<HopperReference> reference;
  std::optional<Error> failure;
  HopperPlanningDiagnostics diagnostics;
};

class HopperPlanner final {
 public:
  HopperPlanResult Plan(
      const PlanningRequest& request,
      const ResolvedTerminalSet& resolved_terminal) const;
};

}  // namespace lunar::planning::v3
```

`HopperPlanResult` 必须恰好携带 `reference` 或 `failure` 之一。它是内部平台
结果，不携带 wire `PlanningOutcome`、`ExecutionDirective` 或第二套 reason code；
总集成层将 `Error` 和 diagnostics 映射到响应仲裁器。

- [ ] **Step 1: 写成功、无安全域和前沿结果测试**

```cpp
TEST(HopperPlanner, ReturnsFullyCertifiedNextHopForGoalRegion) {
  const auto request = make_hopper_request();
  const auto terminal = make_resolved_terminal_set(TerminalKind::kGoal);
  const auto result = HopperPlanner{}.Plan(request, terminal);
  ASSERT_TRUE(result.reference.has_value());
  EXPECT_FALSE(result.failure.has_value());
  EXPECT_TRUE(validate_landing_containment(
      result.reference->predicted_landing_footprint,
      result.reference->next_landing_region,
      result.reference->attitude_boundary).ok());
}

TEST(HopperPlanner, UsesSafeFrontierOutcomeWithoutAuthorizingUnknownTail) {
  const auto request = make_hopper_request();
  const auto terminal =
      make_resolved_terminal_set(TerminalKind::kSafeFrontier);
  const auto result = HopperPlanner{}.Plan(request, terminal);
  ASSERT_TRUE(result.reference.has_value());
  EXPECT_FALSE(result.failure.has_value());
  ASSERT_TRUE(result.reference->future_route_preview.has_value());
  EXPECT_EQ(
      result.reference->future_route_preview->authority,
      FutureRoutePreview::Authority::
          kNonAuthoritativeMissionPreview);
}

TEST(HopperPlanner, NoLandingRegionProducesNoJumpBoundary) {
  auto request = make_hopper_request_with_map(
      make_no_safe_landing_snapshot());
  const auto terminal = make_resolved_terminal_set(TerminalKind::kGoal);
  const auto result = HopperPlanner{}.Plan(request, terminal);
  EXPECT_FALSE(result.reference.has_value());
  ASSERT_TRUE(result.failure.has_value());
  EXPECT_EQ(result.failure->code, ErrorCode::kNoKnownSafeRoute);
}
```

- [ ] **Step 2: 写结果 one-of 和无 wire 仲裁字段测试**

```cpp
template <class T>
concept HasExecutionDirective = requires(T value) {
  value.execution_directive;
};

TEST(HopperPlanner, PlatformResultHasExactlyOnePayloadAndNoDirective) {
  static_assert(!HasExecutionDirective<HopperPlanResult>);
  const auto request = make_hopper_request_with_map(
      make_no_safe_landing_snapshot());
  const auto terminal = make_resolved_terminal_set(TerminalKind::kGoal);
  const auto result = HopperPlanner{}.Plan(request, terminal);
  EXPECT_NE(result.reference.has_value(), result.failure.has_value());
}
```

- [ ] **Step 3: 运行测试确认平台编排器缺失**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_hopper_tests --parallel
```

Expected: build fails because `HopperPlanner` is undefined.

- [ ] **Step 4: 实现入口守卫和地形着陆域阶段**

入口先调用 `bind_hopper_capability(profile, request.capability_bindings)`、
`bind_hopper_limits()`，验证 platform
variant、frame、重力有效范围和 request 中已经解析内容的 ID/revision/hash。
`PlannerV3Impl` 在调用前已经处理 `JUMP_COMMITTED`、`IN_FLIGHT` 和
`EMERGENCY_DELEGATED`；`HopperPlanner` 不读取活动 bundle，也不产生 directive。

可规划状态把 goal/capability 允许 yaw 的交集按 `yaw_partition_count` 确定性切成闭区间；未给 goal yaw 时从完整圆周开始。对每个区间依次执行 yaw 扫掠足迹、safe pose mask、seed、`LandingRegionGenerator`，并保留区间端点来源。region 为空时不构造 `HopCertifier`。

- [ ] **Step 5: 实现节点、终端和惰性认证编排**

从当前稳定地面位姿构造只读 start node；其余节点来自 `TerrainCertifiedLandingRegion`。目标 region 与 `ResolvedTerminalSet` 的安全交集确定 terminal nodes；未知 tail 只进入非权威任务预览。

为每个目标节点生成瞄准点，建立有向图，以当前 request/map/capability 构造 `HopCertifier`，调用 `LazyLandingGraphPlanner::select()`。

- [ ] **Step 6: 绑定 future preview 并重新执行最终验证**

惰性图返回完整第一跳引用后，将共享合同 `FutureRoutePreview` 写入 schema `future_route_preview`；其 C++ 类型固定 authority 为 `kNonAuthoritativeMissionPreview`。随后再次执行合同层与 `validate_final_hopper_reference()`。`SAFE_DEAD_END` 只进入 diagnostics 和 preview reason code，不生成新的 `PlanningOutcome` 枚举值。

- [ ] **Step 7: 实现平台失败到共享 `ErrorCode` 的稳定映射**

- 无安全着陆域或无已认证路径：`ErrorCode::kNoKnownSafeRoute`；
- 输入 platform/frame/content 不一致：`ErrorCode::kInvalidArgument`；
- 数值认证失败：`ErrorCode::kNumericalFailure`；
- 资源耗尽且没有完整引用：`ErrorCode::kResourceLimit`；
- 资源耗尽但已有完整引用：返回完整 reference，并把稳定终止原因写入
  diagnostics。

活动 bundle 的继续、地面 hold 和 `NO_SAFE_PLANNER_REFERENCE` 选择全部留给
Integration Task 3；平台错误不得覆盖仍有效的已承诺跳。

- [ ] **Step 8: 运行平台编排测试并确认通过**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_hopper_tests --parallel
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -C Debug -R '^lpp_v3_hopper\.HopperPlanner' --no-tests=error --output-on-failure
```

Expected: goal/前沿成功结果通过；无域不泄漏边界；平台结果 one-of 成立且不含 execution directive。

- [ ] **Step 9: 提交飞跃平台编排器**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/hopper/hopper_planner.hpp cpp/src/hopper/hopper_planner.cpp cpp/tests/hopper/hopper_planner_test.cpp cpp/tests/hopper/hopper_test_fixtures.hpp cpp/tests/hopper/hopper_test_fixtures.cpp cpp/src/hopper/CMakeLists.txt cpp/tests/hopper/CMakeLists.txt
git -C path-planner commit -m "feat(planner-v3): orchestrate certified hopper planning"
```

---

### Task 15: 完成 schema、场景矩阵、故障和确定性验收

**Files:**
- Create: `path-planner/cpp/tests/hopper/hopper_scenario_matrix_test.cpp`
- Modify: `path-planner/cpp/tests/hopper/hopper_test_fixtures.hpp`
- Modify: `path-planner/cpp/tests/hopper/hopper_test_fixtures.cpp`
- Modify: `path-planner/cpp/tests/hopper/CMakeLists.txt`

**Interfaces:**
- Consumes: 完整 `lpp_v3_hopper`、合同层 `encode_json()`、`decode_hopper_reference()` 和 `validate()`。
- Produces: 飞跃平台的完整回归门；不产出运行时新类型。

- [ ] **Step 1: 写 schema round-trip 和字段权威性测试**

```cpp
TEST(HopperScenarioMatrix, CertifiedReferenceRoundTripsThroughWireSchema) {
  const auto request = make_hopper_request();
  const auto terminal = make_resolved_terminal_set(TerminalKind::kGoal);
  const HopperReference original =
      require_reference(HopperPlanner{}.Plan(request, terminal));
  nlohmann::json encoded;
  encode_json(encoded, original);
  const auto decoded = decode_hopper_reference(encoded);
  ASSERT_TRUE(IsOk(decoded));
  const auto& decoded_reference =
      std::get<HopperReference>(decoded);
  EXPECT_EQ(canonical_contract_hash(original),
            canonical_contract_hash(decoded_reference));
  EXPECT_TRUE(validate(
      decoded_reference, make_validation_context()).ok());
}

TEST(HopperScenarioMatrix, WireReferenceContainsNoSelectableAlternateCommand) {
  nlohmann::json encoded;
  encode_json(encoded, make_fully_certified_hopper_reference());
  EXPECT_TRUE(encoded.contains("jump_boundary"));
  EXPECT_FALSE(encoded.contains("jump_boundaries"));
  EXPECT_FALSE(encoded.contains("selectable_aim_points"));
  EXPECT_EQ(encoded.at("future_route_preview")
                .at("authority").get<std::string>(),
            "NON_AUTHORITATIVE_MISSION_PREVIEW");
}
```

- [ ] **Step 2: 写规范场景矩阵**

用 GoogleTest 参数化固定 fixtures：

```cpp
INSTANTIATE_TEST_SUITE_P(
    RequiredTerrainAndPhysicsCases,
    HopperScenarioMatrixTest,
    ::testing::Values(
        HopperScenario::kConcaveMask,
        HopperScenario::kMaskWithHole,
        HopperScenario::kNarrowNeck,
        HopperScenario::kNonCircularFootprint,
        HopperScenario::kYawIntervalCrossingPi,
        HopperScenario::kMultipleLandingPlanes,
        HopperScenario::kObstacleAtApex,
        HopperScenario::kUnknownAlongFlight,
        HopperScenario::kLowClearance,
        HopperScenario::kArbitraryAxisAttitudeTightening,
        HopperScenario::kLandingSetOutsideRegion,
        HopperScenario::kSafeDeadEnd));
```

每个 fixture 声明唯一预期 reference/failure one-of、共享 `ErrorCode`、内部稳定
诊断码，以及成功时的最小认证净空、位置包含和 yaw 子集关系。wire outcome 与
directive 只在 Integration 场景矩阵断言。

- [ ] **Step 3: 写资源、数值和版本故障矩阵**

覆盖：

- map snapshot/layer 版本不一致；
- capability 或 gravity model 版本改变；
- 非有限输入；
- region/graph/aim/root/collision 上限命中；
- 平面拟合和 root 不收敛；
- tube 碰撞区间无法判定。

每个失败断言 `reference == nullopt` 且 `failure` 有值；若资源上限命中前已经
形成完整认证候选，则允许返回该完整 reference。活动 committed jump 与
boundary ID 一致性由 Hopper Task 13 和 Integration Task 3 单独覆盖。

- [ ] **Step 4: 写多线程调度无关的确定性回归**

对同一不可变输入分别使用配置允许的单线程和固定四线程验证器各运行 100 次。比较 canonical contract hash、selected candidate ID、region/edge/aim ID、终止原因和资源计数；所有结果必须一致。

```cpp
for (int run = 0; run < 100; ++run) {
  const auto result = HopperPlanner{}.Plan(request, terminal);
  EXPECT_EQ(canonical_result_hash(result), expected_hash);
  EXPECT_EQ(result.diagnostics.full_certification_attempt_count,
            expected_attempt_count);
}
```

- [ ] **Step 5: 运行完整飞跃测试**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_hopper_tests --parallel
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -C Debug -R '^lpp_v3_hopper\.' --no-tests=error --output-on-failure
```

Expected: all hopper tests pass.

- [ ] **Step 6: 执行无墙钟分支和旧实现依赖静态审计**

Run:

```powershell
rg -n "steady_clock|system_clock|deadline|remaining_wall|sleep_for" path-planner/cpp/include/lunar_path_planner/v3/hopper path-planner/cpp/src/hopper
rg -n "src/path_planner|PYTHONPATH|pybind11" path-planner/cpp/include/lunar_path_planner/v3/hopper path-planner/cpp/src/hopper
```

Expected: both commands return no matches. Duration fields such as flight time and validity intervals remain allowed because they are physical/reference semantics, not planner wall-clock cutoffs.

- [ ] **Step 7: 运行全 C++ 回归和差异检查**

Run:

```powershell
ctest --preset windows-msvc-debug --output-on-failure
git diff --check
```

Expected: all C++ tests pass and `git diff --check` produces no output.

- [ ] **Step 8: 提交飞跃平台验收矩阵**

```powershell
git -C path-planner add cpp/tests/hopper/hopper_scenario_matrix_test.cpp cpp/tests/hopper/hopper_test_fixtures.hpp cpp/tests/hopper/hopper_test_fixtures.cpp cpp/tests/hopper/CMakeLists.txt
git -C path-planner commit -m "test(planner-v3): verify hopper certification matrix"
```

---

## Final Review Gate

实现者在交给集成计划前必须逐项记录证据：

- `NextLandingRegion` 是单个确定性凸域，平面基和 `canonical_ccw` yaw 区间通过 schema 校验。
- 非圆形着陆足迹在完整 yaw 区间上保守侵蚀，未知、洞、窄颈和多平面场景没有被凸包跨越。
- 惰性图只把第一条边升级为 `CERTIFIED_NEXT_HOP`；future preview 永远非权威。
- 每条候选只有飞行时间 `T` 是连续变量，质心全程纯弹道。
- `CertifiedFlightTube` 连续覆盖 `[0,T]`，包含任意姿态机体包络和全部确定性误差。
- 姿态查表只能收紧基础任意轴能力；引用中没有完整姿态轨迹。
- 落点集合通过向下横截性、时间区间、速度边界和区域包含检查。
- 任一子认证失败都没有 `HopperReference` 或新 `JumpBoundary`。
- `JUMP_COMMITTED`/`IN_FLIGHT` 后不产生替换引用；下一跳只在稳定 `LANDED_HOLD` 后规划。
- 在线代码没有 executor 副作用、旧实现依赖或基于 1 秒墙钟的分支。

计划执行完成后，由总集成计划把 `HopperPlanResult.reference` 原子封装为
`ReferenceBundle`，把 `failure`/diagnostics 映射为统一规划尝试，执行 bundle
ID/子组件 ID 一致性验证，并运行三平台统一 P95 性能实验；本计划不在飞跃模块
内部实现这些跨平台职责。
