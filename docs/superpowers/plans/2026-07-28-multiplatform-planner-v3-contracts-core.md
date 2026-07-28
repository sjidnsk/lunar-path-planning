# Multiplatform Planner v3 Contracts and Shared Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立可由轮式、足式和飞跃式平台共同复用的 C++20 强类型合同、不可变地图、安全投影、目标解析、ARA*、候选排序、凸走廊、固定学习代价、确定性缓存和有界 QP 基础设施。

**Architecture:** 在 `path-planner/cpp/` 内建立 clean-room C++ 子工程，内部只传递强类型不可变对象，JSON 只存在于进程边界。共享核心按“合同校验 → 安全投影 → 目标/安全前沿解析 → 平台 adapter 搜索 → 时间等价候选排序 → 共享走廊/有界 QP”组织；平台层只依赖这些接口，不反向依赖任何现有 Python 实现。

**Tech Stack:** C++20、CMake 3.28+、vcpkg manifest、Eigen 5.0.1、nlohmann_json 3.12.0、double-conversion 3.4.0、PicoSHA2 1.0.1、GoogleTest 1.17.0、Google Benchmark 1.9.5、可选 OSQP 1.0.0、CTest。

## Global Constraints

- 所有新增 C++ 文件必须位于 `path-planner/cpp/**`；公共命名空间固定为 `lunar::planning::v3`。
- CMake 最低版本固定为 `3.28`，语言标准固定为 C++20，禁止依赖编译器扩展。
- vcpkg builtin baseline 固定为 `56bb2411609227288b70117ead2c47585ba07713`。
- 依赖版本固定为 Eigen `5.0.1`、nlohmann_json `3.12.0`、double-conversion `3.4.0`、PicoSHA2 `1.0.1`、GoogleTest `1.17.0`、Google Benchmark `1.9.5`；OSQP `1.0.0` 是默认关闭的可选 QP backend。
- 所有构建、测试和基准产物必须写入 `D:/xunce/build/path-planner-v3/<preset>`，不得写入仓库或 C 盘大目录。
- 语言无关合同只消费 `path-planner/schemas/v3/**`；运行时不得从磁盘加载 schema。
- schema `$id` 和 `schema_version` 必须逐文件使用
  `path-planner/schemas/v3/**` 中已冻结的常量；平台引用 ID 分别是
  `urn:lunar-path-planner:v3:reference:wheeled|legged|hopper`，不得用字符串
  模板臆造。
- JSON 中纳秒时刻和纳秒时长使用十进制字符串；C++ 内部映射为 `std::chrono::nanoseconds`，不得经过 `double`。
- 不修改 `path-planner/src/**`、`src/lunar_exploration_ppo/**` 或其他现有 Python 实现。
- 不复制、调用或引用任何旧版规划实现、轨迹、代价或输出作为正确性 oracle。
  Python 侧只可提供按 v3 合同独立编写的 schema/golden fixture 与隔离适配测试。
- 不连接 executor，不替换 default grid A*，不发布 checkpoint，不启动 canary。
- `P95 < 1 s` 仅是固定基准上的实验指标；任何搜索、QP、缓存或降级逻辑不得读取墙钟剩余时间。
- 所有算法以迭代数、扩展数、候选数、内存或图规模上限终止；资源上限必须来自 `PlannerAlgorithmConfig`。
- 学习代价只能修改有界软代价；硬可行域必须完全由解析能力和地图快照决定。
- 固定输入、配置、线程策略和依赖版本必须产生确定结果；并行结果必须按稳定 ID 归并。
- 每项任务必须先写失败测试，再写最小实现；每项任务通过自身测试后单独提交。
- 每个 GoogleTest 目标必须用 `gtest_discover_tests` 注册稳定
  `lpp_v3_<target>.` 前缀；任何带 `-R` 的 CTest gate 必须同时带
  `--no-tests=error`，禁止以零测试冒充通过。

---

## File Structure

```text
path-planner/
├── cpp/
│   ├── CMakeLists.txt
│   ├── CMakePresets.json
│   ├── vcpkg.json
│   ├── cmake/
│   │   ├── CompilerWarnings.cmake
│   │   └── ProjectOptions.cmake
│   ├── include/lunar_path_planner/v3/
│   │   ├── contracts/
│   │   │   ├── base_types.hpp
│   │   │   ├── status.hpp
│   │   │   ├── state.hpp
│   │   │   ├── profiles.hpp
│   │   │   ├── planning_request.hpp
│   │   │   ├── platform_reference.hpp
│   │   │   ├── reference_bundle.hpp
│   │   │   └── planning_response.hpp
│   │   ├── codec/
│   │   │   ├── json_codec.hpp
│   │   │   └── semantic_validator.hpp
│   │   ├── crypto/
│   │   │   └── sha256.hpp
│   │   ├── map/
│   │   │   ├── immutable_snapshot.hpp
│   │   │   └── safe_projection.hpp
│   │   ├── goal/
│   │   │   └── terminal_resolver.hpp
│   │   ├── search/
│   │   │   ├── ara_star.hpp
│   │   │   ├── platform_adapter.hpp
│   │   │   └── candidate_ranker.hpp
│   │   ├── corridor/
│   │   │   └── convex_corridor.hpp
│   │   ├── cost/
│   │   │   └── learned_cost_snapshot.hpp
│   │   ├── optimization/
│   │   │   └── bounded_qp_solver.hpp
│   │   └── cache/
│   │       └── deterministic_cache.hpp
│   ├── src/
│   │   ├── contracts/
│   │   ├── codec/
│   │   ├── crypto/
│   │   ├── map/
│   │   ├── goal/
│   │   ├── search/
│   │   ├── corridor/
│   │   ├── cost/
│   │   ├── optimization/
│   │   └── cache/
│   ├── tests/
│   │   ├── contracts/
│   │   ├── codec/
│   │   ├── crypto/
│   │   ├── map/
│   │   ├── goal/
│   │   ├── search/
│   │   ├── corridor/
│   │   ├── cost/
│   │   ├── optimization/
│   │   ├── cache/
│   │   ├── integration/
│   │   └── fixtures/
│   └── benchmarks/
│       └── shared_core_benchmark.cpp
└── schemas/v3/
    ├── common.schema.json
    ├── planning-request.schema.json
    ├── planning-response.schema.json
    ├── reference-bundle.schema.json
    ├── safety-capability-profile.schema.json
    ├── planner-algorithm-config.schema.json
    ├── benchmark-profile.schema.json
    ├── benchmark-report.schema.json
    └── references/
        ├── wheeled.schema.json
        ├── legged.schema.json
        └── hopper.schema.json
```

`ReferenceBundleBuilder` 不在本分卷实现；集成分卷将消费本计划产生的强类型 `ReferenceBundle`、`SemanticValidator` 和平台引用数据结构，并负责原子 bundle 封装。

### Target Graph

```text
lpp_v3_contracts
    ├── lpp_v3_codec
    ├── lpp_v3_map
    ├── lpp_v3_search
    ├── lpp_v3_cost
    └── lpp_v3_optimization

lpp_v3_map + lpp_v3_search
    ├── lpp_v3_goal
    └── lpp_v3_corridor

上述 target 由后续 lpp_v3_wheel、lpp_v3_legged、lpp_v3_hopper 和
lpp_v3_api 链接。
```

---

### Task 1: Bootstrap the C++20, CMake, vcpkg, Test, and Benchmark Baseline

**Files:**

- Create: `path-planner/cpp/CMakeLists.txt`
- Create: `path-planner/cpp/CMakePresets.json`
- Create: `path-planner/cpp/vcpkg.json`
- Create: `path-planner/cpp/cmake/CompilerWarnings.cmake`
- Create: `path-planner/cpp/cmake/ProjectOptions.cmake`
- Create: `path-planner/cpp/tests/toolchain/smoke_test.cpp`
- Create: `path-planner/cpp/benchmarks/shared_core_benchmark.cpp`

**Interfaces:**

- Consumes: CMake `>=3.28`；`VCPKG_ROOT` 指向检出到固定 baseline 的 vcpkg。
- Produces: `lpp_v3_project_options`、`lpp_v3_project_warnings`、CTest 和 benchmark 构建开关，以及 D 盘 preset。

- [ ] **Step 1: Write the toolchain smoke test before any build target exists**

```cpp
#include <Eigen/Core>
#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

TEST(ToolchainSmoke, UsesPinnedDependenciesAndCpp20) {
  static_assert(__cplusplus >= 202002L);
  Eigen::Vector2d value{1.0, 2.0};
  nlohmann::json payload{{"x", value.x()}, {"y", value.y()}};
  EXPECT_EQ(payload.at("x").get<double>(), 1.0);
  EXPECT_EQ(payload.at("y").get<double>(), 2.0);
}
```

- [ ] **Step 2: Verify configuration fails because the CMake project is absent**

Run:

```powershell
cmake -S path-planner/cpp -B D:/xunce/build/path-planner-v3/windows-msvc-debug
```

Expected: FAIL because `path-planner/cpp/CMakeLists.txt` does not yet exist.

- [ ] **Step 3: Pin the vcpkg registry and dependency versions**

Create `path-planner/cpp/vcpkg.json` with:

```json
{
  "$schema": "https://raw.githubusercontent.com/microsoft/vcpkg-tool/main/docs/vcpkg.schema.json",
  "name": "lunar-path-planner-v3",
  "version-string": "0.1.0",
  "builtin-baseline": "56bb2411609227288b70117ead2c47585ba07713",
  "dependencies": [
    {"name": "eigen3", "version>=": "5.0.1"},
    {"name": "nlohmann-json", "version>=": "3.12.0"},
    {"name": "double-conversion", "version>=": "3.4.0"},
    {"name": "picosha2", "version>=": "1.0.1"},
    {"name": "gtest", "version>=": "1.17.0"},
    {"name": "benchmark", "version>=": "1.9.5"}
  ],
  "features": {
    "qp": {
      "description": "Build the bounded OSQP backend",
      "dependencies": [
        {"name": "osqp", "version>=": "1.0.0"}
      ]
    }
  },
  "overrides": [
    {"name": "eigen3", "version": "5.0.1"},
    {"name": "nlohmann-json", "version": "3.12.0"},
    {"name": "double-conversion", "version": "3.4.0"},
    {"name": "picosha2", "version": "1.0.1"},
    {"name": "gtest", "version": "1.17.0"},
    {"name": "benchmark", "version": "1.9.5"},
    {"name": "osqp", "version": "1.0.0"}
  ]
}
```

- [ ] **Step 4: Add D-drive CMake configure, build, and test presets**

Create `path-planner/cpp/CMakePresets.json` with a hidden base preset and these concrete presets:

```json
{
  "version": 6,
  "cmakeMinimumRequired": {"major": 3, "minor": 28, "patch": 0},
  "configurePresets": [
    {
      "name": "base",
      "hidden": true,
      "generator": "Ninja",
      "binaryDir": "D:/xunce/build/path-planner-v3/${presetName}",
      "cacheVariables": {
        "CMAKE_TOOLCHAIN_FILE": "$env{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake",
        "CMAKE_CXX_STANDARD": "20",
        "CMAKE_CXX_STANDARD_REQUIRED": "ON",
        "CMAKE_CXX_EXTENSIONS": "OFF",
        "LPP_V3_BUILD_TESTS": "ON",
        "LPP_V3_BUILD_BENCHMARKS": "ON",
        "LPP_V3_ENABLE_OSQP": "OFF"
      }
    },
    {
      "name": "windows-msvc-debug",
      "inherits": "base",
      "condition": {"type": "equals", "lhs": "${hostSystemName}", "rhs": "Windows"},
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug",
        "VCPKG_TARGET_TRIPLET": "x64-windows"
      }
    },
    {
      "name": "linux-gcc-debug",
      "inherits": "base",
      "condition": {"type": "equals", "lhs": "${hostSystemName}", "rhs": "Linux"},
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug",
        "VCPKG_TARGET_TRIPLET": "x64-linux"
      }
    },
    {
      "name": "windows-msvc-release",
      "inherits": "base",
      "condition": {"type": "equals", "lhs": "${hostSystemName}", "rhs": "Windows"},
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release",
        "LPP_V3_BUILD_TESTS": "ON",
        "VCPKG_TARGET_TRIPLET": "x64-windows"
      }
    }
  ],
  "buildPresets": [
    {"name": "windows-msvc-debug", "configurePreset": "windows-msvc-debug"},
    {"name": "linux-gcc-debug", "configurePreset": "linux-gcc-debug"},
    {"name": "windows-msvc-release", "configurePreset": "windows-msvc-release"}
  ],
  "testPresets": [
    {
      "name": "windows-msvc-debug",
      "configurePreset": "windows-msvc-debug",
      "output": {"outputOnFailure": true}
    },
    {
      "name": "linux-gcc-debug",
      "configurePreset": "linux-gcc-debug",
      "output": {"outputOnFailure": true}
    },
    {
      "name": "windows-msvc-release",
      "configurePreset": "windows-msvc-release",
      "output": {"outputOnFailure": true}
    }
  ]
}
```

- [ ] **Step 5: Add strict compiler options and exact package checks**

The top-level `CMakeLists.txt` must contain the following baseline:

```cmake
cmake_minimum_required(VERSION 3.28)
project(lunar_path_planner_v3 VERSION 0.1.0 LANGUAGES CXX)

option(LPP_V3_BUILD_TESTS "Build C++ tests" ON)
option(LPP_V3_BUILD_BENCHMARKS "Build C++ benchmarks" ON)
option(LPP_V3_ENABLE_OSQP "Build the optional OSQP backend" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 5.0.1 EXACT CONFIG REQUIRED)
find_package(nlohmann_json 3.12.0 EXACT CONFIG REQUIRED)

add_library(lpp_v3_project_options INTERFACE)
target_compile_features(lpp_v3_project_options INTERFACE cxx_std_20)
add_library(lpp_v3_project_warnings INTERFACE)

include(cmake/ProjectOptions.cmake)
include(cmake/CompilerWarnings.cmake)
lpp_v3_set_project_options(lpp_v3_project_options)
lpp_v3_set_project_warnings(lpp_v3_project_warnings)

if(LPP_V3_BUILD_TESTS)
  enable_testing()
  find_package(GTest 1.17.0 EXACT CONFIG REQUIRED)
  add_executable(lpp_v3_toolchain_tests tests/toolchain/smoke_test.cpp)
  target_link_libraries(
    lpp_v3_toolchain_tests
    PRIVATE Eigen3::Eigen nlohmann_json::nlohmann_json
            GTest::gtest_main lpp_v3_project_options lpp_v3_project_warnings)
  include(GoogleTest)
  gtest_discover_tests(lpp_v3_toolchain_tests)
endif()

if(LPP_V3_BUILD_BENCHMARKS)
  find_package(benchmark 1.9.5 EXACT CONFIG REQUIRED)
  add_executable(
    lpp_v3_shared_core_benchmark benchmarks/shared_core_benchmark.cpp)
  target_link_libraries(
    lpp_v3_shared_core_benchmark
    PRIVATE benchmark::benchmark_main lpp_v3_project_options)
endif()
```

`CompilerWarnings.cmake` must enable `/W4 /WX` for MSVC and
`-Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Werror` for GCC/Clang.

- [ ] **Step 6: Configure the pinned D-drive build**

Run:

```powershell
$env:VCPKG_ROOT='D:\CodexDownloads\vcpkg'
git -C $env:VCPKG_ROOT rev-parse HEAD
cmake --preset windows-msvc-debug -S path-planner/cpp
```

Expected: the Git command prints
`56bb2411609227288b70117ead2c47585ba07713`; CMake reports the exact dependency
versions and writes only below
`D:/xunce/build/path-planner-v3/windows-msvc-debug`.

- [ ] **Step 7: Build and run the smoke test**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --parallel
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug --no-tests=error --output-on-failure
```

Expected: `lpp_v3_toolchain_tests` passes.

- [ ] **Step 8: Commit the build baseline**

```powershell
git -C path-planner add cpp/CMakeLists.txt cpp/CMakePresets.json cpp/vcpkg.json cpp/cmake cpp/tests/toolchain cpp/benchmarks
git -C path-planner commit -m "build: add planner v3 C++ toolchain baseline"
```

---

### Task 2: Add Strong Common Types, States, Profiles, Requests, and Statuses

**Files:**

- Create: `path-planner/cpp/include/lunar_path_planner/v3/contracts/base_types.hpp`
- Create: `path-planner/cpp/include/lunar_path_planner/v3/contracts/status.hpp`
- Create: `path-planner/cpp/include/lunar_path_planner/v3/contracts/state.hpp`
- Create: `path-planner/cpp/include/lunar_path_planner/v3/contracts/profiles.hpp`
- Create: `path-planner/cpp/include/lunar_path_planner/v3/contracts/planning_request.hpp`
- Create: `path-planner/cpp/src/contracts/base_types.cpp`
- Create: `path-planner/cpp/tests/contracts/planning_request_test.cpp`
- Modify: `path-planner/cpp/CMakeLists.txt`

**Interfaces:**

- Consumes: `Eigen::Vector2d` and `Eigen::Vector3d` only at numerical boundaries.
- Produces: `Result<T>`、`ContentRef`、`WheeledOrLeggedState`、`HopperState`、`SafetyCapabilityProfile`、`PlannerAlgorithmConfig`、`PlanningRequest`。

- [ ] **Step 1: Write contract tests for platform-specific state separation and exact nanoseconds**

```cpp
#include <chrono>
#include <type_traits>
#include <variant>
#include <gtest/gtest.h>
#include "lunar_path_planner/v3/contracts/planning_request.hpp"

namespace lpp = lunar::planning::v3;

template <class T>
concept HasOrientationBodyToFrame = requires(T state) {
  state.orientation_body_to_frame;
};

TEST(PlanningRequestContract, WheelAndLegStateDoesNotContainQuaternion) {
  static_assert(
      !HasOrientationBodyToFrame<lpp::WheeledOrLeggedState>);
}

TEST(PlanningRequestContract, HopperCarriesQuaternionAndAngularVelocity) {
  lpp::HopperState state{};
  state.orientation_body_to_frame = {1.0, 0.0, 0.0, 0.0};
  state.angular_velocity_radps = {0.0, 0.0, 0.1};
  EXPECT_DOUBLE_EQ(state.orientation_body_to_frame.w, 1.0);
}

TEST(PlanningRequestContract, TimestampDoesNotRoundThroughDouble) {
  const lpp::ClockStamp stamp{
      .clock_id = "mission-clock",
      .tick = std::chrono::nanoseconds{9'007'199'254'740'993LL}};
  EXPECT_EQ(stamp.tick.count(), 9'007'199'254'740'993LL);
}

TEST(PlanningRequestContract, ResultHasOneUniformErrorAlternative) {
  static_assert(
      std::variant_size_v<lpp::Result<int>> == 2U);
  static_assert(std::same_as<
      std::variant_alternative_t<1, lpp::Result<int>>, lpp::Error>);
}
```

- [ ] **Step 2: Build the test to verify the contract headers are missing**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_contracts_tests
```

Expected: FAIL because the contract headers and target do not exist.

- [ ] **Step 3: Define the common value and error types**

Implement these exact public types:

```cpp
namespace lunar::planning::v3 {

using Identifier = std::string;
using Sha256Digest = std::string;
using RequestId = Identifier;
using GoalId = Identifier;
using FrameId = Identifier;
using ClockId = Identifier;
using BundleId = Identifier;
using ReferenceId = Identifier;
using ComponentId = Identifier;
using SegmentId = Identifier;
using PrimitiveId = Identifier;
using ReferencePointId = Identifier;
using LandingRegionId = Identifier;
using JumpBoundaryId = Identifier;
using ReasonCode = std::string;

struct Vec2 final { double x{}; double y{}; };
struct Vec3 final { double x{}; double y{}; double z{}; };
struct Quaternion final { double w{1.0}; double x{}; double y{}; double z{}; };
struct Interval final { double lower{}; double upper{}; };
struct DurationNanoseconds final { std::chrono::nanoseconds value{}; };
struct ClockStamp final {
  std::string clock_id;
  std::chrono::nanoseconds tick{};
};

struct ContentRef final {
  Identifier id;
  std::uint32_t revision{};
  Sha256Digest content_hash;
  auto operator<=>(const ContentRef&) const = default;
};

enum class ErrorCode {
  kInvalidArgument,
  kSchemaMismatch,
  kStaleInput,
  kInconsistentSnapshot,
  kResourceLimit,
  kNumericalFailure,
  kNoKnownSafeRoute
};

struct Error final {
  ErrorCode code;
  std::string field_path;
  std::string message;
};

template <class T>
using Result = std::variant<T, Error>;

template <class T>
[[nodiscard]] bool IsOk(const Result<T>& value) noexcept {
  return std::holds_alternative<T>(value);
}

}  // namespace lunar::planning::v3
```

`Result<T>` is the only result alias in the C++ v3 core. Do not add a
two-parameter `Result<T, E>` variant in a platform layer.

- [ ] **Step 4: Define state and deterministic-error structures**

```cpp
struct AxisAlignedBox3 final {
  Vec3 center;
  Vec3 half_extent;
};

struct EuclideanBall3 final {
  Vec3 center;
  double radius{};
};

using DeterministicVectorSet3 =
    std::variant<AxisAlignedBox3, EuclideanBall3>;

struct SymmetricScalarInterval final {
  double center{};
  double half_width{};
};

struct RotationVectorBall final {
  double radius_rad{};
};

struct WheeledOrLeggedErrorBounds final {
  DeterministicVectorSet3 position_bound_m;
  SymmetricScalarInterval yaw_bound_rad;
  DeterministicVectorSet3 linear_velocity_bound_mps;
  SymmetricScalarInterval yaw_rate_bound_radps;
};

struct WheeledOrLeggedState final {
  Vec3 position_m;
  double yaw_rad{};
  Vec3 linear_velocity_mps;
  double yaw_rate_radps{};
  WheeledOrLeggedErrorBounds error_bounds;
};

struct HopperErrorBounds final {
  DeterministicVectorSet3 position_bound_m;
  RotationVectorBall orientation_bound;
  DeterministicVectorSet3 linear_velocity_bound_mps;
  DeterministicVectorSet3 angular_velocity_bound_radps;
};

struct HopperState final {
  Vec3 position_m;
  Quaternion orientation_body_to_frame;
  Vec3 linear_velocity_mps;
  Vec3 angular_velocity_radps;
  HopperErrorBounds error_bounds;
};

enum class PlatformType { kWheeled, kLegged, kHopper };
using PlatformState = std::variant<WheeledOrLeggedState, HopperState>;
```

- [ ] **Step 5: Define profiles and request ownership**

`PlanningRequest` must hold immutable shared ownership and must not contain a
runtime deadline:

```cpp
struct CircularYawInterval final {
  enum class Representation { kCanonicalCcw };
  Representation representation{Representation::kCanonicalCcw};
  double start_rad{};
  double span_rad{};
  bool closed{true};
};

struct LandingPlane final {
  Vec3 origin_m;
  Vec3 normal;
  Vec3 basis_u;
  Vec3 basis_v;
  double residual_bound_m{};
};

struct ConvexPolygonUv final {
  enum class Winding { kCcw };
  std::vector<Vec2> vertices_uv;
  Winding winding{Winding::kCcw};
};

struct PointGoal final {
  Vec3 position_m;
  double position_tolerance_m{};
};

struct PlanarRegionGoal final {
  LandingPlane plane;
  ConvexPolygonUv polygon;
  double normal_tolerance_m{};
};

using GoalTarget = std::variant<PointGoal, PlanarRegionGoal>;
using MetadataValue = std::variant<std::string, double, bool>;

struct MetadataEntry final {
  std::string key;
  MetadataValue value;
};

struct GoalRegion final {
  std::string goal_id;
  GoalTarget target;
  std::optional<CircularYawInterval> optional_yaw_interval;
  std::optional<Vec3> mission_direction_hint;
  std::vector<MetadataEntry> task_metadata;
};

struct ResourceCaps final {
  std::size_t maximum_expanded_states{};
  std::size_t maximum_reopened_states{};
  std::size_t maximum_generated_candidates{};
  std::size_t maximum_open_states{};
  std::size_t maximum_memory_bytes{};
};

struct AraStarConfig final {
  double initial_epsilon{};
  double epsilon_decrement{};
  double target_epsilon{};
  ResourceCaps resource_caps;
};

struct PlannerAlgorithmConfig final {
  ContentRef content_ref;
  DurationNanoseconds time_equivalence_tolerance;
  DurationNanoseconds max_input_skew;
  std::string error_bound_model_id;
  std::size_t projection_cache_capacity{};
  AraStarConfig ara_star;
  WheeledAlgorithmConfig wheeled;
  LeggedAlgorithmConfig legged;
  HopperAlgorithmConfig hopper;
  LearnedCostPolicy learned_cost_policy;
  std::optional<ContentRef> learned_cost_model_ref;
  DeterministicExecutionConfig deterministic_execution;
};

struct ArbitraryAxisAttitudeEnvelope final {
  double maximum_angular_speed_radps{};
  double maximum_angular_acceleration_radps2{};
  double maximum_initial_angular_speed_radps{};
  DurationNanoseconds minimum_settle_guard;
};

struct HopperLandingTerrainThresholds final {
  double maximum_slope_rad{};
  double maximum_roughness_m{};
  double maximum_plane_residual_m{};
  double minimum_overhead_clearance_m{};
  double minimum_lateral_clearance_m{};
  double minimum_landing_region_area_m2{};
};

struct HopperCapability final {
  enum class TranslationModel {
    kPureBallisticNoInflightTranslationControl
  };
  FrameId frame_id;
  BodyConvexPolytope collision_envelope;
  ContentRef motion_model_ref;
  ContentRef analytic_cost_model_ref;
  ContentRef gravity_model_ref;
  HopperLandingTerrainThresholds landing_terrain_thresholds;
  HopperLaunchLimits launch_limits;
  ArbitraryAxisAttitudeEnvelope attitude_envelope;
  std::optional<ContentRef> attitude_tightening_table_ref;
  HopperErrorBounds certified_state_error_bounds;
  TranslationModel translation_model{
      TranslationModel::kPureBallisticNoInflightTranslationControl};
};

using SafetyCapabilityContent =
    std::variant<WheeledCapability, LeggedCapability, HopperCapability>;

struct SafetyCapabilityProfile final {
  ContentRef content_ref;
  SafetyCapabilityContent content;
};

class ImmutableMapSnapshot;
class LearnedCostSnapshot;
class MotionModel;
class AnalyticCostModel;
class GravityModel;
class DeterministicErrorModel;
class ActuatorOrImpulseProfile;
class BodyRotationEnvelope;
class AttitudeTighteningTable;

template <class T>
struct ResolvedBinding final {
  ContentRef content_ref;
  std::shared_ptr<const T> object;
};

struct ResolvedCapabilityBindings final {
  ResolvedBinding<MotionModel> motion_model;
  ResolvedBinding<AnalyticCostModel> analytic_cost_model;
  std::optional<ResolvedBinding<GravityModel>> gravity_model;
  std::optional<ResolvedBinding<DeterministicErrorModel>> error_model;
  std::optional<ResolvedBinding<ActuatorOrImpulseProfile>>
      actuator_or_impulse_profile;
  std::optional<ResolvedBinding<BodyRotationEnvelope>>
      body_rotation_envelope;
  std::optional<ResolvedBinding<AttitudeTighteningTable>>
      attitude_tightening_table;
};

struct TimeExecutionCursor final {
  DurationNanoseconds offset;
  std::optional<SegmentId> segment_id;
};

enum class JumpExecutionState {
  kGroundHold,
  kJumpReady,
  kJumpCommitted,
  kInFlight,
  kLandedHold
};

struct JumpExecutionCursor final {
  JumpExecutionState jump_state{};
  std::optional<JumpBoundaryId> boundary_id;
};

using ExecutionCursor =
    std::variant<TimeExecutionCursor, JumpExecutionCursor>;

struct TimeCommitBoundary final {
  DurationNanoseconds committed_until_offset;
};

struct JumpCommitBoundary final {
  JumpBoundaryId boundary_id;
  bool locked{};
};

using CommitBoundary =
    std::variant<TimeCommitBoundary, JumpCommitBoundary>;

enum class ControllerStatus {
  kReady,
  kExecuting,
  kHolding,
  kCommitted,
  kInFlight,
  kFault
};

struct PreviousExecutionContext final {
  ContentRef active_bundle_ref;
  std::string active_bundle_handle;
  CommitBoundary commit_boundary;
  ExecutionCursor execution_cursor;
  ControllerStatus controller_status{};
  ContentRef source_map_snapshot_ref;
  ContentRef source_capability_ref;
};

struct LearnedCostSnapshotBinding final {
  ContentRef snapshot_ref;
  std::string registry_handle;
  std::shared_ptr<const LearnedCostSnapshot> resolved_snapshot;
};

struct PlanningRequest final {
  RequestId request_id;
  ClockStamp request_time;
  ClockStamp state_time;
  FrameId frame_id;
  PlatformType platform_type;
  PlatformState current_state;
  GoalRegion goal;
  std::shared_ptr<const ImmutableMapSnapshot> map_snapshot;
  std::shared_ptr<const SafetyCapabilityProfile> safety_capability;
  std::shared_ptr<const PlannerAlgorithmConfig> algorithm_config;
  ResolvedCapabilityBindings capability_bindings;
  std::optional<PreviousExecutionContext> previous_execution_context;
  std::optional<LearnedCostSnapshotBinding> learned_cost_snapshot;
};

class ContractObjectRegistry;

struct ReferenceActivationContext final {
  const PlanningRequest& request;
  const ContractObjectRegistry& registry;
};
```

Before defining the top-level profile structs, implement every profile/config
member type named by the frozen schemas
`safety-capability-profile.schema.json` and
`planner-algorithm-config.schema.json` one-to-one. In particular:

- wheeled capability includes the non-circular extruded footprint, separate
  forward/reverse/spin limits and certified motion primitives;
- legged capability includes a fixed body reference point, body convex
  collision polytope, terrain thresholds, body-velocity limits and body
  primitives, but no footstep/contact-force fields;
- hopper capability uses
  `PURE_BALLISTIC_NO_INFLIGHT_TRANSLATION_CONTROL` and the certified
  conservative arbitrary-axis envelope; an optional lookup has
  `TIGHTEN_ONLY` authority;
- request decoding resolves the profile's direct refs and the Hopper motion
  model's fixed gravity/error/actuator/body-rotation dependencies into
  `ResolvedCapabilityBindings`; platform code never loads a model from a bare
  ID during `Plan`;
- algorithm config contains the frozen wheel/leg/hopper sub-configs, learned
  soft-cost policy and deterministic-execution fields, and contains no
  benchmark target or runtime deadline.

- [ ] **Step 6: Add the contracts library and test target**

Add `lpp_v3_contracts` and `lpp_v3_contracts_tests` to CMake. Public include
paths must expose only `path-planner/cpp/include`; source directories must
remain private.

- [ ] **Step 7: Run the contract tests**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_contracts_tests
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R "^lpp_v3_contracts\." --no-tests=error --output-on-failure
```

Expected: all contract tests pass with `/WX`.

- [ ] **Step 8: Commit the strong input contracts**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/contracts cpp/src/contracts cpp/tests/contracts cpp/CMakeLists.txt
git -C path-planner commit -m "feat: add planner v3 strong input contracts"
```

---

### Task 3: Add Typed Platform References, Reference Bundles, and Planning Responses

**Files:**

- Create: `path-planner/cpp/include/lunar_path_planner/v3/contracts/platform_reference.hpp`
- Create: `path-planner/cpp/include/lunar_path_planner/v3/contracts/reference_bundle.hpp`
- Create: `path-planner/cpp/include/lunar_path_planner/v3/contracts/planning_response.hpp`
- Create: `path-planner/cpp/tests/contracts/reference_bundle_test.cpp`
- Modify: `path-planner/cpp/CMakeLists.txt`

**Interfaces:**

- Consumes: Task 2 `Vec2`、`Vec3`、`Quaternion`、`Interval`、`ContentRef`。
- Produces: data-only `WheeledReference`、`LeggedBodyReference`、`HopperReference`、`ReferenceBundle`、`PlanningResponse`；不产生 `ReferenceBundleBuilder`。

- [ ] **Step 1: Write tests for typed variants and orthogonal response states**

```cpp
#include <gtest/gtest.h>
#include "lunar_path_planner/v3/contracts/planning_response.hpp"

namespace lpp = lunar::planning::v3;

TEST(ReferenceBundleContract, PlatformReferenceIsTypeSafeVariant) {
  lpp::ReferenceBundle bundle{};
  bundle.platform_reference = lpp::WheeledReference{};
  EXPECT_TRUE(std::holds_alternative<lpp::WheeledReference>(
      bundle.platform_reference));
}

TEST(PlanningResponseContract, ActivationRequiresBundleAtSemanticLayer) {
  lpp::PlanningResponse response{
      .planning_outcome = lpp::PlanningOutcome::kNewReferenceReady,
      .execution_directive = lpp::ExecutionDirective::kActivateNewBundle};
  EXPECT_FALSE(response.new_reference_bundle.has_value());
}
```

- [ ] **Step 2: Build to verify the output contract headers are missing**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_contracts_tests
```

Expected: FAIL because `planning_response.hpp` is absent.

- [ ] **Step 3: Define wheel and leg reference data without controller commands**

```cpp
struct TimeInterval final {
  DurationNanoseconds start_offset;
  DurationNanoseconds end_offset;
};

struct PoseXyzYaw final {
  Vec3 position_m;
  double yaw_rad{};
};

struct CubicPolynomialSegment final {
  DurationNanoseconds start_offset;
  DurationNanoseconds end_offset;
  std::array<double, 4> coefficients;
};

struct PiecewiseCubicScalarTrajectory final {
  std::string value_semantics;
  std::vector<CubicPolynomialSegment> segments;
};

struct MonotoneTimeScaling final {
  std::vector<CubicPolynomialSegment> segments;
};

struct DerivedKinematicCaches final {
  double consistency_tolerance{};
  PiecewiseCubicScalarTrajectory signed_body_forward_speed_mps;
  PiecewiseCubicScalarTrajectory yaw_rate_radps;
};

struct ClampedCubicBSplinePath final {
  std::vector<double> knots;
  std::vector<PoseXyzYaw> control_points;
};

enum class PrimitiveKind {
  kDriveForward,
  kDriveReverse,
  kSpinCw,
  kSpinCcw,
  kStopAndSwitch,
  kBodyTranslation,
  kBodySpin,
  kBodyCoupled
};

struct ValidatedPrimitive final {
  PrimitiveId primitive_id;
  PrimitiveId capability_primitive_id;
  PrimitiveKind primitive_kind;
  PoseXyzYaw start_pose;
  PoseXyzYaw end_pose;
  DurationNanoseconds nominal_duration;
  ContentRef validation_ref;
};

struct ValidatedPrimitiveChain final {
  std::vector<ValidatedPrimitive> primitives;
};

using GeometricPath =
    std::variant<ClampedCubicBSplinePath, ValidatedPrimitiveChain>;

enum class DriveDirection { kForward, kReverse };

struct DriveSegment final {
  SegmentId segment_id;
  TimeInterval time_interval;
  DriveDirection direction;
  GeometricPath geometric_path;
  MonotoneTimeScaling time_scaling;
  std::optional<DerivedKinematicCaches> derived_caches;
};

struct SpinSegment final {
  SegmentId segment_id;
  TimeInterval time_interval;
  Vec3 fixed_position_m;
  PiecewiseCubicScalarTrajectory unwrapped_yaw_rad;
};

using WheeledSegment = std::variant<DriveSegment, SpinSegment>;

struct SafeStopAnchor final {
  std::string anchor_id;
  PoseXyzYaw pose;
  double target_linear_velocity_mps{0.0};
  double target_yaw_rate_radps{0.0};
  ContentRef terrain_certification_ref;
};

struct WheeledReference final {
  ReferenceId reference_id;
  Sha256Digest reference_hash;
  ClockStamp reference_time_origin;
  std::vector<WheeledSegment> segments;
  SafeStopAnchor safe_stop_anchor;
};

struct BodyFrameVelocityEnvelope final {
  Interval forward_mps;
  Interval lateral_mps;
  Interval vertical_mps;
  Interval yaw_rate_radps;
};

struct TerrainNormalEnvelope final {
  double maximum_normal_deviation_rad{};
  ContentRef source_terrain_certification_ref;
};

struct RollPitchDiagnosticEnvelope final {
  Interval roll_rad;
  Interval pitch_rad;
};

struct LeggedBodyReference final {
  ReferenceId reference_id;
  Sha256Digest reference_hash;
  ReferencePointId reference_point_id;
  ClockStamp reference_time_origin;
  GeometricPath geometric_path;
  MonotoneTimeScaling time_scaling;
  BodyFrameVelocityEnvelope body_frame_velocity_envelope;
  TerrainNormalEnvelope terrain_normal_envelope;
  RollPitchDiagnosticEnvelope roll_pitch_diagnostic_envelope;
  SafeStopAnchor safe_stop_anchor;
  std::string feasibility_scope{
      "body_geometry_and_terrain_thresholds_only"};
  bool footstep_feasibility_guaranteed{false};
};
```

- [ ] **Step 4: Define hopper reference data as deterministic sets**

```cpp
struct Vector3Bounds final {
  Vec3 lower;
  Vec3 upper;
};

struct Halfspace3 final {
  Vec3 normal;
  double offset_m{};
};

struct ConvexPolytope3 final {
  enum class Representation { kHalfspaceIntersection };
  Representation representation{Representation::kHalfspaceIntersection};
  std::vector<Halfspace3> halfspaces;
};

struct NextLandingRegion final {
  LandingRegionId region_id;
  FrameId frame_id;
  LandingPlane landing_plane;
  ConvexPolygonUv convex_polygon;
  CircularYawInterval allowed_yaw_interval;
  ContentRef terrain_certification_ref;
  double inward_safety_margin_m{};
};

struct HopperKinematicState final {
  Vec3 position_m;
  Quaternion orientation_body_to_frame;
  Vec3 linear_velocity_mps;
  Vec3 angular_velocity_radps;
};

struct GroundHoldAnchor final {
  Identifier anchor_id;
  HopperKinematicState hold_state;
  HopperErrorBounds allowed_hold_state_error_set;
  ContentRef terrain_certification_ref;
};

struct JumpBoundary final {
  enum class LockEvent { kJumpBoundaryLock };
  enum class BallisticTimeOrigin { kBallisticLaunchEvent };
  JumpBoundaryId boundary_id;
  LockEvent lock_event{LockEvent::kJumpBoundaryLock};
  HopperKinematicState nominal_launch_state;
  HopperErrorBounds allowed_launch_state_error_set;
  ContentRef gravity_model_ref;
  BallisticTimeOrigin ballistic_time_origin{
      BallisticTimeOrigin::kBallisticLaunchEvent};
  DurationNanoseconds ballistic_flight_time;
  ContentRef actuator_or_impulse_profile_ref;
};

struct PredictedLandingFootprint final {
  LandingPlane landing_plane;
  ConvexPolygonUv convex_center_landing_polygon;
  TimeInterval landing_time_window;
  Vector3Bounds landing_velocity_bounds;
  CircularYawInterval landing_yaw_interval;
  ContentRef source_error_model_ref;
  double outer_approximation_margin_m{};
};

struct FlightTubeSection final {
  TimeInterval time_interval;
  ConvexPolytope3 envelope;
};

struct CertifiedFlightTube final {
  FrameId frame_id;
  std::vector<FlightTubeSection> sections;
  ContentRef source_map_snapshot_ref;
  ContentRef body_rotation_envelope_ref;
  ContentRef error_model_ref;
  double minimum_certified_clearance_m{};
};

struct TargetAttitudeSet final {
  Quaternion nominal_orientation_body_to_frame;
  RotationVectorBall orientation_error_set;
  CircularYawInterval allowed_yaw_interval;
};

struct AttitudeBoundary final {
  enum class TranslationAuthority { kNone };
  RotationVectorBall initial_orientation_error_set;
  DeterministicVectorSet3 initial_angular_velocity_error_set_radps;
  TargetAttitudeSet target_attitude_set;
  DeterministicVectorSet3 landing_angular_velocity_bounds_radps;
  DurationNanoseconds settle_guard;
  TranslationAuthority center_of_mass_translation_authority{
      TranslationAuthority::kNone};
  ContentRef certification_ref;
};

struct NominalAimPoint final {
  enum class Authority { kNonAuthoritativeExplanatory };
  Authority authority{Authority::kNonAuthoritativeExplanatory};
  Vec3 position_m;
};

enum class FutureViability {
  kViable,
  kUnknown,
  kNoCertifiedContinuation
};

struct FutureRoutePreview final {
  enum class Authority { kNonAuthoritativeMissionPreview };
  Authority authority{Authority::kNonAuthoritativeMissionPreview};
  FutureViability future_viability{FutureViability::kUnknown};
  std::string reason_code;
  std::vector<LandingRegionId> candidate_region_ids;
};

struct HopperReference final {
  enum class TranslationModel {
    kPureBallisticNoInflightTranslationControl
  };
  ReferenceId reference_id;
  Sha256Digest reference_hash;
  ClockStamp reference_time_origin;
  TranslationModel translation_model{
      TranslationModel::kPureBallisticNoInflightTranslationControl};
  GroundHoldAnchor ground_hold_anchor;
  NextLandingRegion next_landing_region;
  JumpBoundary jump_boundary;
  PredictedLandingFootprint predicted_landing_footprint;
  CertifiedFlightTube certified_flight_tube;
  AttitudeBoundary attitude_boundary;
  NominalAimPoint nominal_aim_point;
  ContentRef physical_certification_ref;
  std::optional<FutureRoutePreview> future_route_preview;
};

using PlatformReference =
    std::variant<WheeledReference, LeggedBodyReference, HopperReference>;
```

Constants frozen by schema are represented by the type and codec, not free-form
strings:

- `JumpBoundary::lock_event = JUMP_BOUNDARY_LOCK`;
- hopper translation model =
  `PURE_BALLISTIC_NO_INFLIGHT_TRANSLATION_CONTROL`;
- `AttitudeBoundary::center_of_mass_translation_authority = NONE`;
- `NominalAimPoint::authority = NON_AUTHORITATIVE_EXPLANATORY`;
- `FutureRoutePreview::authority = NON_AUTHORITATIVE_MISSION_PREVIEW`.

`CertifiedFlightTube` uses ordered H-representation halfspaces, not
axis-aligned radii. `SemanticValidator` must require complete section coverage
of the ballistic interval and enforce
`PredictedLandingFootprint ⊕ inward_safety_margin_m ⊆ NextLandingRegion`.
着陆时间窗与线速度边界仅以 `PredictedLandingFootprint` 中的字段为真值，
顶层 `HopperReference` 不复制。圆周区间还必须验证
`footprint.landing_yaw_interval ⊆
attitude_boundary.target_attitude_set.allowed_yaw_interval ⊆
next_landing_region.allowed_yaw_interval`。

- [ ] **Step 5: Define bundle identity and orthogonal response enums**

```cpp
template <class Content>
struct InlineComponent final {
  ComponentId component_id;
  Sha256Digest component_hash;
  Content content;
};

struct RouteSkeletonContent final {
  ReferenceId source_reference_id;
  Sha256Digest source_reference_hash;
  std::vector<PoseXyzYaw> waypoints;
  std::vector<Vec3> unresolved_tail;
};

struct TimeViewSelector final { TimeInterval time_interval; };
struct SegmentViewSelector final {
  std::size_t first_segment_index{};
  std::size_t past_last_segment_index{};
};
struct GroundHoldViewSelector final {
  Identifier anchor_id;
};
struct JumpViewSelector final {
  enum class Scope { kNextHop, kFutureMissionPreview };
  JumpBoundaryId boundary_id;
  Scope scope{Scope::kNextHop};
};
using ReferenceViewSelector = std::variant<
    TimeViewSelector,
    SegmentViewSelector,
    GroundHoldViewSelector,
    JumpViewSelector>;

struct ReferenceViewContent final {
  enum class Role { kCommittedPrefix, kPreview };
  Role role{};
  ReferenceId source_reference_id;
  Sha256Digest source_reference_hash;
  ReferenceViewSelector selector;
};

enum class InvalidationCondition {
  kMapSafetyRevisionChanged,
  kStateDeviationExceeded,
  kCapabilityRevisionChanged,
  kExecutionCursorPastReplacementBoundary,
  kJumpBoundaryLocked,
  kJumpLaunched
};

struct ReferenceValidity final {
  ClockStamp valid_from;
  std::optional<ClockStamp> valid_until;
  ContentRef required_map_snapshot_ref;
  ContentRef required_capability_ref;
  std::variant<WheeledOrLeggedErrorBounds, HopperErrorBounds>
      allowed_state_deviation;
  std::vector<InvalidationCondition> invalidation_conditions;
};

struct ValidationSummary final {
  bool hard_constraints_passed{true};
  bool continuous_validation_passed{true};
  std::vector<ContentRef> certificate_refs;
  std::vector<std::string> warning_codes;
};

enum class GenerationMode {
  kSmoothedSplineReference,
  kValidatedPrimitiveChainReference,
  kCertifiedBallisticReference
};

struct GenerationEvidence final {
  std::string selected_candidate_id;
  GenerationMode generation_mode{};
  std::string termination_reason;
  std::vector<ContentRef> evidence_refs;
  std::optional<ContentRef> learned_cost_snapshot_ref;
};

struct ReferenceBundle final {
  BundleId bundle_id;
  std::uint32_t bundle_revision{};
  Sha256Digest bundle_hash;
  std::optional<BundleId> supersedes_bundle_id;
  RequestId source_request_id;
  ContentRef source_map_snapshot_ref;
  ContentRef source_safety_capability_ref;
  ContentRef source_algorithm_config_ref;
  PlatformType platform_type{};
  PlatformReference platform_reference;
  InlineComponent<RouteSkeletonContent> route_skeleton;
  InlineComponent<ReferenceViewContent> committed_prefix;
  InlineComponent<ReferenceViewContent> preview;
  ReferenceValidity validity;
  ValidationSummary validation_summary;
  GenerationEvidence generation_evidence;
};

enum class PlanningOutcome {
  kNewReferenceReady,
  kSafeFrontierReferenceReady,
  kNoKnownSafeRoute,
  kGoalInfeasible,
  kInvalidRequest,
  kStaleInput,
  kNumericalFailure,
  kResourceLimit,
  kActiveReferenceInvalidated
};

enum class ExecutionDirective {
  kActivateNewBundle,
  kContinueActiveBundle,
  kHoldStationary,
  kContinueCommittedJump,
  kNoSafePlannerReference
};

enum class LearnedCostUsage {
  kDisabled,
  kUsedBoundedSoftCost,
  kFellBackToAnalytic
};

struct SecondaryCosts final {
  std::optional<double> energy;
  std::optional<double> nonfatal_risk;
  std::optional<double> smoothness;
};

struct CallDiagnostics final {
  DurationNanoseconds api_latency;
  std::string termination_reason;
  std::optional<double> final_epsilon;
  std::optional<DurationNanoseconds> expected_execution_time;
  std::optional<SecondaryCosts> secondary_costs;
  std::uint64_t expanded_state_count{};
  std::uint64_t reopened_state_count{};
  std::uint64_t candidate_count{};
  bool resource_limit_hit{};
  LearnedCostUsage learned_cost_usage{LearnedCostUsage::kDisabled};
  std::optional<ContentRef> learned_cost_snapshot_ref;
  std::vector<std::string> message_codes;
};

struct PlanningResponse final {
  RequestId request_id;
  ClockStamp response_time;
  PlanningOutcome planning_outcome{};
  ExecutionDirective execution_directive{};
  ReasonCode reason_code;
  std::optional<ContentRef> active_bundle_ref;
  std::optional<ReferenceBundle> new_reference_bundle;
  CallDiagnostics call_diagnostics;
};
```

- [ ] **Step 6: Run output contract tests**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_contracts_tests
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R "^lpp_v3_contracts\." --no-tests=error --output-on-failure
```

Expected: typed variant tests pass; the deliberately incomplete response remains
constructible so `SemanticValidator` can report all errors in one report.

- [ ] **Step 7: Commit the output data contracts**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/contracts cpp/tests/contracts cpp/CMakeLists.txt
git -C path-planner commit -m "feat: add planner v3 typed output contracts"
```

---

### Task 4: Implement Strict JSON Codecs and `SemanticValidator`

**Files:**

- Create: `path-planner/cpp/include/lunar_path_planner/v3/codec/json_codec.hpp`
- Create: `path-planner/cpp/include/lunar_path_planner/v3/codec/jcs_canonicalizer.hpp`
- Create: `path-planner/cpp/include/lunar_path_planner/v3/codec/semantic_validator.hpp`
- Create: `path-planner/cpp/include/lunar_path_planner/v3/crypto/sha256.hpp`
- Create: `path-planner/cpp/src/codec/json_codec.cpp`
- Create: `path-planner/cpp/src/codec/jcs_canonicalizer.cpp`
- Create: `path-planner/cpp/src/codec/semantic_validator.cpp`
- Create: `path-planner/cpp/src/crypto/sha256.cpp`
- Create: `path-planner/cpp/tests/codec/json_codec_test.cpp`
- Create: `path-planner/cpp/tests/codec/jcs_canonicalizer_test.cpp`
- Create: `path-planner/cpp/tests/codec/semantic_validator_test.cpp`
- Create: `path-planner/cpp/tests/crypto/sha256_test.cpp`
- Create: `path-planner/cpp/tests/fixtures/minimal_wheeled_request.json`
- Modify: `path-planner/cpp/CMakeLists.txt`

**Interfaces:**

- Consumes: Task 2 and Task 3 contract types；schema files under `path-planner/schemas/v3/**`。
- Produces: `JsonCodec`、`JcsCanonicalizer`、`Sha256Hex`、`ContractObjectRegistry`、`SemanticValidator`、`ValidationReport`。

- [ ] **Step 1: Write failing codec tests for schema IDs and lossless nanoseconds**

```cpp
#include <gtest/gtest.h>
#include "lunar_path_planner/v3/codec/json_codec.hpp"

namespace lpp = lunar::planning::v3;

TEST(JsonCodec, RejectsWrongSchemaVersion) {
  const std::string payload = R"({
    "schema_version":"path-planner-v3-planning-request/v0"
  })";
  lpp::EmptyContractObjectRegistry registry;
  const auto result = lpp::JsonCodec::DecodePlanningRequest(payload, registry);
  ASSERT_FALSE(lpp::IsOk(result));
  EXPECT_EQ(std::get<lpp::Error>(result).code,
            lpp::ErrorCode::kSchemaMismatch);
}

TEST(JsonCodec, NanosecondsRoundTripAsDecimalString) {
  const lpp::DurationNanoseconds value{
      std::chrono::nanoseconds{9'007'199'254'740'993LL}};
  const auto encoded = lpp::JsonCodec::EncodeDuration(value);
  ASSERT_TRUE(lpp::IsOk(encoded));
  EXPECT_EQ(std::get<std::string>(encoded), "\"9007199254740993\"");
  const auto decoded =
      lpp::JsonCodec::DecodeDuration(std::get<std::string>(encoded));
  ASSERT_TRUE(lpp::IsOk(decoded));
  EXPECT_EQ(std::get<lpp::DurationNanoseconds>(decoded).value.count(),
             value.value.count());
}

TEST(JsonCodec, RejectsNonCanonicalOrNegativeDurationNanoseconds) {
  for (const std::string_view text :
       {"\"00\"", "\"01\"", "\"-0\"", "\"+1\"", "\"-1\""}) {
    EXPECT_FALSE(lpp::IsOk(lpp::JsonCodec::DecodeDuration(text)));
  }
}
```

- [ ] **Step 2: Write failing semantic tests for cross-field invariants**

```cpp
TEST(SemanticValidator, RejectsPlatformStateVariantMismatch) {
  lpp::PlanningRequest request = MakeMinimalWheeledRequest();
  request.current_state = lpp::HopperState{};
  const auto report = lpp::SemanticValidator{}.Validate(request);
  EXPECT_FALSE(report.ok());
  EXPECT_TRUE(report.Contains("state", "platform_state_mismatch"));
}

TEST(SemanticValidator, RejectsActivationWithoutBundle) {
  const lpp::PlanningResponse response{
      .planning_outcome = lpp::PlanningOutcome::kNewReferenceReady,
      .execution_directive = lpp::ExecutionDirective::kActivateNewBundle};
  const auto report = lpp::SemanticValidator{}.Validate(response);
  EXPECT_TRUE(report.Contains(
      "new_reference_bundle", "activation_requires_validated_bundle"));
}

TEST(SemanticValidator, RejectsZeroBallisticFlightTime) {
  auto response = MakeValidHopperActivationResponse();
  hopper_reference(response).jump_boundary.ballistic_flight_time =
      DurationNanoseconds{std::chrono::nanoseconds{0}};
  const auto report =
      MakeValidator().ValidateForActivation(
          response, MakeReferenceActivationContext());
  EXPECT_TRUE(report.Contains(
      "jump_boundary.ballistic_flight_time",
      "positive_flight_time_required"));
}

TEST(SemanticValidator, RejectsBundleProvenanceMismatch) {
  auto response = MakeValidHopperActivationResponse();
  response.new_reference_bundle->validity.required_map_snapshot_ref =
      MakeOtherMapRef();
  const auto report =
      MakeValidator().ValidateForActivation(
          response, MakeReferenceActivationContext());
  EXPECT_TRUE(report.Contains(
      "validity.required_map_snapshot_ref",
      "bundle_map_provenance_mismatch"));
}

TEST(SemanticValidator, RejectsDiscontinuousGroundHoldToLaunch) {
  auto response = MakeValidHopperActivationResponse();
  hopper_reference(response).jump_boundary.nominal_launch_state.position_m.x +=
      1.0;
  const auto report =
      MakeValidator().ValidateForActivation(
          response, MakeReferenceActivationContext());
  EXPECT_TRUE(report.Contains(
      "jump_boundary.nominal_launch_state",
      "ground_hold_launch_discontinuity"));
}
```

- [ ] **Step 3: Build to verify codec and validator symbols are absent**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_codec_tests
```

Expected: FAIL because `JsonCodec` and `SemanticValidator` do not exist.

- [ ] **Step 4: Define the registry and codec boundary**

```cpp
enum class ContractObjectKind {
  kMotionModel,
  kAnalyticCostModel,
  kGravityModel,
  kDeterministicErrorModel,
  kActuatorOrImpulseProfile,
  kBodyRotationEnvelope,
  kAttitudeTighteningTable,
  kCertification
};

struct CertificationProvenance final {
  ContentRef source_map_snapshot_ref;
  ContentRef source_safety_capability_ref;
  ContentRef source_algorithm_config_ref;
  std::vector<ContentRef> input_refs;
  std::string certification_purpose;
};

class ImmutableContractObject {
 public:
  virtual ~ImmutableContractObject() = default;
  virtual ContentRef content_ref() const = 0;
  virtual ContractObjectKind kind() const = 0;
};

class CertificationObject : public ImmutableContractObject {
 public:
  virtual const CertificationProvenance& provenance() const = 0;
};

class ContractObjectRegistry {
 public:
  virtual ~ContractObjectRegistry() = default;
  virtual std::shared_ptr<const ImmutableMapSnapshot> FindMapSnapshot(
      const ContentRef& snapshot_ref,
      std::string_view immutable_data_handle) const = 0;
  virtual std::shared_ptr<const SafetyCapabilityProfile> FindSafetyCapability(
      const ContentRef& ref) const = 0;
  virtual std::shared_ptr<const PlannerAlgorithmConfig> FindAlgorithmConfig(
      const ContentRef& ref) const = 0;
  virtual std::shared_ptr<const LearnedCostSnapshot> FindLearnedCost(
      const ContentRef& snapshot_ref,
      std::string_view registry_handle) const = 0;
  virtual Result<ResolvedCapabilityBindings> ResolveCapabilityBindings(
      const SafetyCapabilityProfile& profile) const = 0;
  virtual Result<std::shared_ptr<const ImmutableContractObject>> Resolve(
      const ContentRef& ref,
      ContractObjectKind expected_kind) const = 0;
};

class EmptyContractObjectRegistry final : public ContractObjectRegistry {
 public:
  std::shared_ptr<const ImmutableMapSnapshot> FindMapSnapshot(
      const ContentRef&, std::string_view) const override {
    return nullptr;
  }
  std::shared_ptr<const SafetyCapabilityProfile> FindSafetyCapability(
      const ContentRef&) const override {
    return nullptr;
  }
  std::shared_ptr<const PlannerAlgorithmConfig> FindAlgorithmConfig(
      const ContentRef&) const override {
    return nullptr;
  }
  std::shared_ptr<const LearnedCostSnapshot> FindLearnedCost(
      const ContentRef&, std::string_view) const override {
    return nullptr;
  }
  Result<ResolvedCapabilityBindings> ResolveCapabilityBindings(
      const SafetyCapabilityProfile&) const override {
    return Error{ErrorCode::kMissingRegistryObject};
  }
  Result<std::shared_ptr<const ImmutableContractObject>> Resolve(
      const ContentRef&, ContractObjectKind) const override {
    return Error{ErrorCode::kMissingRegistryObject};
  }
};

class JsonCodec final {
 public:
  static Result<PlanningRequest> DecodePlanningRequest(
      std::string_view payload, const ContractObjectRegistry& registry);
  static Result<PlanningResponse> DecodePlanningResponse(
      std::string_view payload,
      const ReferenceActivationContext& activation_context);
  static std::string EncodePlanningRequest(const PlanningRequest& request);
  static std::string EncodePlanningResponse(const PlanningResponse& response);
  static Result<std::string> EncodeDuration(DurationNanoseconds value);
  static Result<DurationNanoseconds> DecodeDuration(
      std::string_view json_string);
};
```

The decoder must reject duplicate keys, unknown required-enum values, non-finite
numbers, missing registry objects and any top-level key not declared by the
corresponding schema. It must never load a schema file at runtime.

- [ ] **Step 5: Parse nanoseconds with `std::from_chars`**

Use an integer-only helper:

```cpp
Result<std::chrono::nanoseconds> ParseNanoseconds(
    const nlohmann::json& value,
    std::string_view field_path,
    bool allow_negative) {
  if (!value.is_string()) {
    return Error{ErrorCode::kInvalidArgument, std::string(field_path),
                 "nanoseconds must be a decimal string"};
  }
  const std::string& text = value.get_ref<const std::string&>();
  const bool negative = !text.empty() && text.front() == '-';
  const std::size_t first_digit = negative ? 1U : 0U;
  const bool canonical_zero = text == "0";
  const bool canonical_nonzero =
      first_digit < text.size() &&
      text[first_digit] >= '1' && text[first_digit] <= '9' &&
      std::all_of(
          text.begin() + static_cast<std::ptrdiff_t>(first_digit + 1U),
          text.end(),
          [](const char c) { return c >= '0' && c <= '9'; });
  if ((!canonical_zero && !canonical_nonzero) ||
      (negative && !allow_negative)) {
    return Error{ErrorCode::kInvalidArgument, std::string(field_path),
                 "nanoseconds must use the canonical decimal form"};
  }
  std::int64_t count{};
  const auto [end, ec] =
      std::from_chars(text.data(), text.data() + text.size(), count);
  if (ec != std::errc{} || end != text.data() + text.size()) {
    return Error{ErrorCode::kInvalidArgument, std::string(field_path),
                 "invalid nanoseconds decimal string"};
  }
  return std::chrono::nanoseconds{count};
}
```

Call this helper with `allow_negative = true` only for
`DecimalNanoseconds` time points. All durations, offsets, statistics and
other `NonNegativeDecimalNanoseconds` fields pass `false`; their typed wrappers
must never be constructed from a negative count.

- [ ] **Step 6: Implement RFC 8785 JCS and SHA-256 as shared services**

Expose one canonical implementation:

```cpp
class JcsCanonicalizer final {
 public:
  static Result<std::string> Canonicalize(
      const nlohmann::json& value);
};

Result<Sha256Digest> Sha256Hex(std::string_view canonical_utf8);
```

The canonicalizer must reject non-I-JSON input, preserve Unicode without
normalization, sort object names by UTF-16 code units, use the RFC 8785 string
escaping rules, and serialize finite IEEE-754 binary64 values with ECMAScript
shortest formatting. Configure `double-conversion` with decimal exponent
boundaries `-6` and `21`, lowercase `e`, an explicit plus sign for positive
exponents, and unique zero. `Sha256Hex` uses pinned PicoSHA2 and returns exactly
64 lowercase hexadecimal characters.

Tests must include RFC 8785 Appendix B number vectors, non-BMP UTF-16 property
ordering, C0 escaping, negative zero, duplicate-key rejection at the JSON
decoder, NaN/infinity rejection, and standard SHA-256 empty/`abc` vectors.
Neither builder nor codec may contain a second canonicalization or hash
implementation.

- [ ] **Step 7: Implement accumulated semantic validation**

```cpp
struct ValidationIssue final {
  std::string field_path;
  std::string reason_code;
  std::string message;
};

struct ValidationReport final {
  std::vector<ValidationIssue> issues;
  [[nodiscard]] bool ok() const noexcept { return issues.empty(); }
  [[nodiscard]] bool Contains(
      std::string_view field_path, std::string_view reason_code) const;
};

class SemanticValidator final {
 public:
  ValidationReport Validate(const PlanningRequest& request) const;
  ValidationReport Validate(const PlanningResponse& response) const;
  ValidationReport Validate(const PlatformReference& reference) const;
  ValidationReport Validate(const ReferenceBundle& bundle) const;
  ValidationReport Validate(
      const SafetyCapabilityProfile& capability) const;
  ValidationReport Validate(
      const PlannerAlgorithmConfig& config) const;
  ValidationReport ValidateForActivation(
      const ReferenceBundle& bundle,
      const ReferenceActivationContext& context) const;
  ValidationReport ValidateForActivation(
      const PlanningResponse& response,
      const ReferenceActivationContext& context) const;
};
```

The implementation must accumulate issues in stable `field_path` order and
enforce at least:

- `current_state` and safety-capability variants match `platform_type`;
- all floating-point values are finite;
- intervals have `lower <= upper`;
- capability platform and frame match the request;
- every `ContentRef` has revision in `[1, 2147483647]` and a lower-case
  64-hex JCS/SHA-256 hash;
- resource caps obey their frozen positive/non-negative schema bounds;
- `initial_epsilon >= target_epsilon >= 1`;
- `epsilon_decrement > 0`;
- `time_equivalence_tolerance.value.count() >= 0`;
- clock IDs match before comparing `request_time`, `state_time`, map source time
  and response time;
- `CircularYawInterval` has canonical start in `[-pi,pi)` and span in
  `[0,2*pi]`;
- hopper quaternions are unit and canonical-sign, landing-plane
  normal/basis vectors are unit orthogonal right-handed, UV polygons are
  strictly convex CCW, and plane residual bounds are non-negative;
- hopper flight-tube sections are finite, ordered and continuously cover the
  strictly positive ballistic interval relative to
  `BALLISTIC_LAUNCH_EVENT`; capability minimum flight time is not greater than
  its positive maximum, the boundary duration lies in that interval, and the
  footprint's canonical landing time window contains
  the boundary's nominal impact time and its velocity bounds are ordered; the
  margin-expanded predicted footprint is contained in `NextLandingRegion`;
  `footprint.landing_yaw_interval` is a subset of the target-attitude yaw
  interval, which is itself a subset of the landing-region yaw interval;
- bundle component IDs/hashes bind their exact inline content, selectors refer
  only to the same platform reference, and `platform_type` matches the variant;
- activation requires
  `response.request_id == context.request.request_id ==
  bundle.source_request_id`; platform, frame and reference-time provenance also
  match the request;
- activation joins bundle source map to validity map and Hopper tube map,
  source capability to validity capability, and source algorithm config to the
  request; all nested refs resolve with exact ID/revision/hash;
- Hopper activation joins boundary gravity, actuator/impulse,
  body-rotation-envelope and footprint/tube error-model refs to the request's
  frozen `ResolvedCapabilityBindings`;
- every terrain, physical, attitude and validation-summary certificate resolves
  as `ContractObjectKind::kCertification`; its `CertificationProvenance`
  repeats the exact map/capability/config refs and contains the fixed
  gravity/error/actuator/body-envelope inputs used by the candidate;
- the request Hopper state set is contained in the ground-hold set; ground-hold
  and launch nominal position/orientation are continuous, while velocity/error
  changes are certified by the fixed actuator/impulse profile;
- launch/landing speed, impulse, downward-impact, clearance, attitude-envelope
  and settle-guard limits are satisfied after deterministic error expansion;
- a Hopper `GroundHoldViewSelector` appears only in a committed-prefix view,
  resolves to the same reference's `ground_hold_anchor`, and that anchor has
  exactly zero linear/angular velocity plus a valid terrain certificate;
- activation directives and READY outcomes carry `new_reference_bundle`;
- non-activation directives cannot carry a new bundle;
- continue directives carry `active_bundle_ref`, while stationary hold carries
  neither a new bundle nor a READY outcome.
- per-call learned usage is joined to
  `call_diagnostics.learned_cost_snapshot_ref`: `DISABLED` forbids it and
  `USED_BOUNDED_SOFT_COST`/`FELL_BACK_TO_ANALYTIC` require the exact request
  ref; only an activated `USED` candidate copies it into generation evidence.

`Validate(const ReferenceBundle&)` and `Validate(const PlanningResponse&)` are
local checks only. Only `ValidateForActivation(..., context)` may authorize
`ACTIVATE_NEW_BUNDLE`; callers must not substitute a successful local report.

- [ ] **Step 8: Add schema parity tests without adding schema I/O to production**

Compile the absolute schema root into `lpp_v3_codec_tests` only:

```cmake
target_compile_definitions(
  lpp_v3_codec_tests
  PRIVATE LPP_V3_SCHEMA_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/../schemas/v3")
```

The test must parse every schema with nlohmann_json and assert exact `$id` and
`schema_version` constants, including the three files under `references/`.

- [ ] **Step 9: Run codec and semantic tests**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_codec_tests
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R "^lpp_v3_codec\." --no-tests=error --output-on-failure
```

Expected: malformed JSON, wrong schema versions, lossy timestamps and invalid
state combinations are rejected; valid fixtures round-trip canonically.

- [ ] **Step 10: Commit the boundary codec and validator**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/codec cpp/include/lunar_path_planner/v3/crypto cpp/src/codec cpp/src/crypto cpp/tests/codec cpp/tests/crypto cpp/tests/fixtures/minimal_wheeled_request.json cpp/CMakeLists.txt
git -C path-planner commit -m "feat: add planner v3 JSON contract validation"
```

---

### Task 5: Build the Immutable, Atomically Consistent Map Snapshot

**Files:**

- Create: `path-planner/cpp/include/lunar_path_planner/v3/map/immutable_snapshot.hpp`
- Create: `path-planner/cpp/src/map/immutable_snapshot.cpp`
- Create: `path-planner/cpp/tests/map/immutable_snapshot_test.cpp`
- Modify: `path-planner/cpp/include/lunar_path_planner/v3/contracts/planning_request.hpp`
- Modify: `path-planner/cpp/src/codec/semantic_validator.cpp`
- Modify: `path-planner/cpp/CMakeLists.txt`

**Interfaces:**

- Consumes: Task 2 `Result<T>`、`Vec2`、`Vec3`、`ClockStamp`。
- Produces: `ImmutableMapSnapshot::Create`、`GridGeometry`、typed read-only layer views。

- [ ] **Step 1: Write failing tests for copy-on-publication and layer consistency**

```cpp
#include <gtest/gtest.h>
#include "lunar_path_planner/v3/map/immutable_snapshot.hpp"

namespace lpp = lunar::planning::v3;

TEST(ImmutableMapSnapshot, CopiesMutableInputBeforePublication) {
  auto input = MakeMinimalMapInput();
  auto result = lpp::ImmutableMapSnapshot::Create(input);
  ASSERT_TRUE(lpp::IsOk(result));
  const auto snapshot =
      std::get<std::shared_ptr<const lpp::ImmutableMapSnapshot>>(result);
  input.elevation_m[0] = 99.0F;
  EXPECT_FLOAT_EQ(snapshot->ElevationMeters()[0], 0.0F);
}

TEST(ImmutableMapSnapshot, RejectsDuplicateLayerKind) {
  auto input = MakeMinimalMapInput();
  input.layer_manifest.push_back(input.layer_manifest.front());
  const auto result = lpp::ImmutableMapSnapshot::Create(input);
  EXPECT_FALSE(lpp::IsOk(result));
}

TEST(ImmutableMapSnapshot, ExposesFrameNormalsAndPerLayerIdentity) {
  const auto result =
      lpp::ImmutableMapSnapshot::Create(MakeMinimalMapInput());
  ASSERT_TRUE(lpp::IsOk(result));
  const auto snapshot =
      std::get<std::shared_ptr<const lpp::ImmutableMapSnapshot>>(result);
  EXPECT_EQ(snapshot->frame_id(), "map");
  EXPECT_FLOAT_EQ(snapshot->SurfaceNormals().z[0], 1.0F);
  ASSERT_TRUE(snapshot->LayerIdentity(
      lpp::LayerKind::kTerrainNormal).has_value());
  EXPECT_EQ(snapshot->LayerIdentity(
                lpp::LayerKind::kTerrainNormal)->get().revision,
            7U);
}
```

- [ ] **Step 2: Build to verify the immutable snapshot API is missing**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_map_tests
```

Expected: FAIL because `ImmutableMapSnapshot` is not defined.

- [ ] **Step 3: Define exact map geometry, manifest, and storage input**

```cpp
enum class LayerKind {
  kKnownMask,
  kElevation,
  kTerrainNormal,
  kRoughness,
  kHardObstacle,
  kConfidence,
  kEsdf,
  kStaticSpeedLimit
};

struct MapBounds final {
  Vec3 minimum_m;
  Vec3 maximum_m;
};

struct GridGeometry final {
  std::size_t width{};
  std::size_t height{};
  double resolution_m{};
  Vec2 origin_m;
  std::string frame_id;
  [[nodiscard]] std::size_t CellCount() const noexcept {
    return width * height;
  }
};

struct LayerManifestEntry final {
  LayerKind layer_kind{};
  ContentRef content_ref;
};

struct SurfaceNormalGridView final {
  std::span<const float> x;
  std::span<const float> y;
  std::span<const float> z;
};

struct MapSnapshotInput final {
  ContentRef snapshot_ref;
  std::uint32_t map_revision{};
  std::string immutable_data_handle;
  ClockStamp source_time;
  MapBounds bounds;
  GridGeometry geometry;
  std::vector<LayerManifestEntry> layer_manifest;
  std::vector<std::uint8_t> known_mask;
  std::vector<float> elevation_m;
  std::vector<float> normal_x;
  std::vector<float> normal_y;
  std::vector<float> normal_z;
  std::vector<float> roughness_m;
  std::vector<std::uint8_t> hard_obstacle_mask;
  std::vector<float> confidence;
  std::vector<float> esdf_m;
  std::vector<float> static_speed_limit_mps;
};
```

- [ ] **Step 4: Implement private immutable ownership**

```cpp
class ImmutableMapSnapshot final {
 public:
  static Result<std::shared_ptr<const ImmutableMapSnapshot>> Create(
      const MapSnapshotInput& input);

  [[nodiscard]] const ContentRef& snapshot_ref() const noexcept;
  [[nodiscard]] std::uint32_t map_revision() const noexcept;
  [[nodiscard]] std::string_view frame_id() const noexcept;
  [[nodiscard]] ClockStamp source_time() const noexcept;
  [[nodiscard]] const MapBounds& bounds() const noexcept;
  [[nodiscard]] const GridGeometry& geometry() const noexcept;
  [[nodiscard]] std::span<const LayerManifestEntry> layer_manifest()
      const noexcept;
  [[nodiscard]] std::optional<std::reference_wrapper<const ContentRef>>
  LayerIdentity(LayerKind kind) const noexcept;
  [[nodiscard]] std::string_view LayerManifestHash() const noexcept;
  [[nodiscard]] std::span<const std::uint8_t> KnownMask() const noexcept;
  [[nodiscard]] std::span<const float> ElevationMeters() const noexcept;
  [[nodiscard]] SurfaceNormalGridView SurfaceNormals() const noexcept;
  [[nodiscard]] std::span<const float> RoughnessMeters() const noexcept;
  [[nodiscard]] std::span<const std::uint8_t>
  HardObstacleMask() const noexcept;
  [[nodiscard]] std::span<const float> Confidence() const noexcept;
  [[nodiscard]] std::span<const float> EsdfMeters() const noexcept;
  [[nodiscard]] std::span<const float>
  StaticSpeedLimitMps() const noexcept;

 private:
  struct Storage;
  explicit ImmutableMapSnapshot(std::shared_ptr<const Storage> storage);
  std::shared_ptr<const Storage> storage_;
};
```

`Create` must copy every input vector exactly once; require unique layer kinds
and exact `ContentRef` identities; require the mandatory known/elevation/
terrain-normal/roughness/hard-obstacle/confidence layers; require
`width*height` values per present layer; reject non-finite values or non-unit
normals instead of repairing them; compute the canonical manifest hash; and
publish only `std::shared_ptr<const ImmutableMapSnapshot>`.

- [ ] **Step 5: Extend semantic validation with snapshot and staleness rules**

Add stable issues for:

- request frame differs from map frame;
- snapshot `ContentRef` or a layer `ContentRef` conflicts with the registry;
- `source_time` and `state_time` use different clocks or exceed configured skew;
- snapshot or capability shared pointer is null;
- unknown cells contain interpolated hard-feasible flags.

The validator must not mutate or repair the snapshot.

- [ ] **Step 6: Run immutable-map tests**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_map_tests lpp_v3_codec_tests
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R "^lpp_v3_(map|codec)\." --no-tests=error --output-on-failure
```

Expected: all layer-size, identity, frame, normal, elevation/roughness,
finite-value and lifetime cases pass.

- [ ] **Step 7: Commit the immutable snapshot**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/map cpp/src/map cpp/tests/map cpp/include/lunar_path_planner/v3/contracts/planning_request.hpp cpp/src/codec/semantic_validator.cpp cpp/CMakeLists.txt
git -C path-planner commit -m "feat: add immutable planner map snapshots"
```

---

### Task 6: Pin and Bound the Optional `LearnedCostSnapshot`

**Files:**

- Create: `path-planner/cpp/include/lunar_path_planner/v3/cost/learned_cost_snapshot.hpp`
- Create: `path-planner/cpp/src/cost/learned_cost_snapshot.cpp`
- Create: `path-planner/cpp/tests/cost/learned_cost_snapshot_test.cpp`
- Modify: `path-planner/cpp/include/lunar_path_planner/v3/contracts/profiles.hpp`
- Modify: `path-planner/cpp/CMakeLists.txt`

**Interfaces:**

- Consumes: Task 2 `Result<T>`、`ContentRef`、`Interval` and Task 5 immutable map identity。
- Produces: immutable `LearnedCostSnapshot` and
  `ResolvedSoftCost ComposeBoundedSoftCost(...)`；不加载模型、不执行推理、不改变硬可行域。

- [ ] **Step 1: Write the failing immutability and fallback tests**

```cpp
#include <limits>
#include <gtest/gtest.h>
#include "lunar_path_planner/v3/cost/learned_cost_snapshot.hpp"

namespace lpp = lunar::planning::v3;

TEST(LearnedCostSnapshot, CopiesPinnedCorrectionValues) {
  auto input = MakeValidLearnedCostInput();
  auto result = lpp::LearnedCostSnapshot::Create(input, 4U);
  ASSERT_TRUE(lpp::IsOk(result));
  const auto snapshot =
      std::get<std::shared_ptr<const lpp::LearnedCostSnapshot>>(result);
  input.energy_correction[0] = 99.0F;
  EXPECT_FLOAT_EQ(snapshot->EnergyCorrection()[0], 0.1F);
}

TEST(LearnedCostSnapshot, InvalidSnapshotFallsBackToAnalyticCost) {
  auto input = MakeValidLearnedCostInput();
  input.nonfatal_risk_correction[1] =
      std::numeric_limits<float>::quiet_NaN();
  const auto snapshot = lpp::LearnedCostSnapshot::Create(input, 4U);
  ASSERT_FALSE(lpp::IsOk(snapshot));
  const std::array<float, 4> analytic_energy{
      1.0F, 2.0F, 3.0F, 4.0F};
  const std::array<float, 4> analytic_risk{
      0.1F, 0.2F, 0.3F, 0.4F};
  const auto map = MakeMinimalMapSnapshot();
  const auto resolved = lpp::ComposeBoundedSoftCost(
      analytic_energy, analytic_risk, nullptr, *map,
      MakeAlgorithmConfig());
  EXPECT_EQ(resolved.source, lpp::SoftCostSource::kAnalyticOnly);
  EXPECT_EQ(resolved.energy,
            std::vector<float>(
                analytic_energy.begin(), analytic_energy.end()));
  EXPECT_EQ(resolved.nonfatal_risk,
            std::vector<float>(
                analytic_risk.begin(), analytic_risk.end()));
}
```

- [ ] **Step 2: Add the missing cost test target and confirm red**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_cost_tests
```

Expected: FAIL because the learned-cost header and `lpp_v3_cost_tests` do not
exist.

- [ ] **Step 3: Define the snapshot input and immutable API**

```cpp
struct LearnedCostSnapshotInput final {
  ContentRef snapshot_ref;
  ContentRef model_ref;
  ContentRef source_map_snapshot_ref;
  std::uint32_t source_map_revision{};
  std::string frame_id;
  std::string feature_contract_id;
  std::string output_contract_id;
  ClockStamp generated_at;
  Interval certified_energy_correction_bounds;
  Interval certified_nonfatal_risk_correction_bounds;
  std::vector<float> energy_correction;
  std::vector<float> nonfatal_risk_correction;
};

class LearnedCostSnapshot final {
 public:
  static Result<std::shared_ptr<const LearnedCostSnapshot>> Create(
      const LearnedCostSnapshotInput& input,
      std::size_t expected_cell_count);

  [[nodiscard]] const ContentRef& snapshot_ref() const noexcept;
  [[nodiscard]] const ContentRef& model_ref() const noexcept;
  [[nodiscard]] const ContentRef& source_map_snapshot_ref() const noexcept;
  [[nodiscard]] std::uint32_t source_map_revision() const noexcept;
  [[nodiscard]] const std::string& frame_id() const noexcept;
  [[nodiscard]] const Interval&
  certified_energy_correction_bounds() const noexcept;
  [[nodiscard]] const Interval&
  certified_nonfatal_risk_correction_bounds() const noexcept;
  [[nodiscard]] std::span<const float> EnergyCorrection() const noexcept;
  [[nodiscard]] std::span<const float>
  NonfatalRiskCorrection() const noexcept;

 private:
  struct Storage;
  explicit LearnedCostSnapshot(std::shared_ptr<const Storage> storage);
  std::shared_ptr<const Storage> storage_;
};
```

- [ ] **Step 4: Implement the exact snapshot admission checks**

`Create` must copy the vector before publication and reject the snapshot unless
all of these are true:

- snapshot/model/map/frame/feature/output identifiers are valid;
- `expected_cell_count > 0` and both vectors have exactly that many entries;
- both certified intervals are finite and ordered;
- every correction is finite and lies inside its certified interval.

On failure, return `ErrorCode::kInvalidArgument`; do not clamp an invalid
snapshot into validity.

- [ ] **Step 5: Define analytic-only fallback and bounded composition**

```cpp
enum class SoftCostSource {
  kAnalyticOnly,
  kAnalyticPlusPinnedLearned
};

struct ResolvedSoftCost final {
  std::vector<float> energy;
  std::vector<float> nonfatal_risk;
  SoftCostSource source{SoftCostSource::kAnalyticOnly};
  std::optional<std::string> fallback_reason_code;
};

ResolvedSoftCost ComposeBoundedSoftCost(
    std::span<const float> analytic_energy,
    std::span<const float> analytic_nonfatal_risk,
    const std::shared_ptr<const LearnedCostSnapshot>& learned,
    const ImmutableMapSnapshot& map,
    const PlannerAlgorithmConfig& config);
```

The function must use analytic-only output with a stable reason code when the
snapshot is null, map identity/revision/frame differs, model ref is not the
configured fixed ref, shape differs, or any composed value is non-finite. A
valid correction is bounded by both its certified interval and
`LearnedCostPolicy` maximum absolute energy/nonfatal-risk correction before it
is added to finite analytic secondary cost. Expected execution time is not an
argument, so learning cannot change the primary time ordering. The function has
no hard-mask argument and therefore cannot make a hard-infeasible cell feasible.

- [ ] **Step 6: Add valid, mismatch, bounds, and determinism cases**

Extend the test file to cover:

```cpp
EXPECT_EQ(ComposeForMap("map-a", 1U).source,
          lpp::SoftCostSource::kAnalyticPlusPinnedLearned);
EXPECT_EQ(ComposeForMap("map-a", 2U).fallback_reason_code,
          "learned_map_revision_mismatch");
EXPECT_EQ(ComposeTwiceWithSameInput().first,
          ComposeTwiceWithSameInput().second);
```

Also test invalid `ContentRef`, wrong vector lengths, unordered bounds,
out-of-range values, infinite analytic inputs, policy-bound tightening and an
unconfigured model ref.

- [ ] **Step 7: Build and run the cost tests**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_cost_tests
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R "^lpp_v3_cost\." --no-tests=error --output-on-failure
```

Expected: immutable publication, strict admission, analytic fallback, bounded
composition and repeatability tests pass.

- [ ] **Step 8: Commit the learned-cost boundary**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/cost cpp/src/cost cpp/tests/cost cpp/include/lunar_path_planner/v3/contracts/profiles.hpp cpp/CMakeLists.txt
git -C path-planner commit -m "feat: add pinned bounded learned cost snapshots"
```

---

### Task 7: Compute Shared Static Safety Projection and Safe-Stop Anchors

**Files:**

- Create: `path-planner/cpp/include/lunar_path_planner/v3/map/safe_projection.hpp`
- Create: `path-planner/cpp/src/map/safe_projection.cpp`
- Create: `path-planner/cpp/tests/map/safe_projection_test.cpp`
- Modify: `path-planner/cpp/include/lunar_path_planner/v3/contracts/profiles.hpp`
- Modify: `path-planner/cpp/CMakeLists.txt`

**Interfaces:**

- Consumes: Task 5 `ImmutableMapSnapshot`、Task 2
  `SafetyCapabilityProfile`/`PlannerAlgorithmConfig` and Task 6
  `LearnedCostSnapshot`。
- Produces: `Result<SafeProjection> BuildSafeProjection(const SafeProjectionRequest&)` and `Result<SafeStopAnchor> ResolveSafeStopAnchor(...)`。

- [ ] **Step 1: Write failing safety-property tests**

```cpp
TEST(SafeProjection, UnknownCellsNeverBecomeHardFeasible) {
  auto snapshot = MakeMapWithUnknownCenterCell();
  const auto result = lpp::BuildSafeProjection(
      MakeSafeProjectionRequest(snapshot, MakeWheelCapability()));
  ASSERT_TRUE(lpp::IsOk(result));
  const auto& projection = std::get<lpp::SafeProjection>(result);
  EXPECT_FALSE(projection.HardFeasible(Cell{1, 1}));
}

TEST(SafeProjection, RetainsImmutableTerrainForHopperCertification) {
  const auto snapshot = MakeMapWithTiltedLandingPatch();
  const auto result = lpp::BuildSafeProjection(
      MakeSafeProjectionRequest(snapshot, MakeHopperCapability()));
  ASSERT_TRUE(lpp::IsOk(result));
  const auto& projection = std::get<lpp::SafeProjection>(result);
  EXPECT_EQ(projection.source_map().get(), snapshot.get());
  EXPECT_FLOAT_EQ(projection.ElevationMeters()[0],
                  snapshot->ElevationMeters()[0]);
  EXPECT_FLOAT_EQ(projection.RoughnessMeters()[0],
                  snapshot->RoughnessMeters()[0]);
  EXPECT_FLOAT_EQ(projection.SurfaceNormals().z[0],
                  snapshot->SurfaceNormals().z[0]);
}

TEST(SafeProjection, LearnedCostCannotChangeHardMask) {
  auto request = MakeSafeProjectionRequest();
  const auto baseline = lpp::BuildSafeProjection(request);
  request.learned_cost = MakeExtremeButBoundedLearnedCost();
  const auto learned = lpp::BuildSafeProjection(request);
  EXPECT_EQ(std::get<lpp::SafeProjection>(baseline).hard_feasible_mask(),
            std::get<lpp::SafeProjection>(learned).hard_feasible_mask());
}

TEST(SafeStopAnchor, RequiresZeroSpeedCertifiedTerrain) {
  const auto result = lpp::ResolveSafeStopAnchor(
      MakeProjectionWithOnlyUnsafeStopCells(), MakeMovingState());
  EXPECT_FALSE(lpp::IsOk(result));
}
```

- [ ] **Step 2: Build to verify the safety projection API is missing**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_map_tests
```

Expected: FAIL because `BuildSafeProjection` is undefined.

- [ ] **Step 3: Define projection layers and query methods**

```cpp
struct Cell final {
  std::int32_t x{};
  std::int32_t y{};
  auto operator<=>(const Cell&) const = default;
};

struct SafeProjectionRequest;

class SafeProjection final {
 public:
  GridGeometry geometry;
  std::vector<std::uint8_t> known_mask;
  std::vector<std::uint8_t> hard_feasible_mask;
  std::vector<float> esdf_clearance_m;
  std::vector<float> analytic_time_cost_s;
  std::vector<float> analytic_energy;
  std::vector<float> analytic_nonfatal_risk;
  std::vector<float> resolved_energy;
  std::vector<float> resolved_nonfatal_risk;
  std::vector<float> conservative_speed_limit_mps;
  std::vector<std::int32_t> connected_component;
  std::vector<std::uint8_t> safe_stop_candidate_mask;

  [[nodiscard]] const std::shared_ptr<const ImmutableMapSnapshot>&
  source_map() const noexcept;
  [[nodiscard]] std::span<const float> ElevationMeters() const noexcept;
  [[nodiscard]] std::span<const float> RoughnessMeters() const noexcept;
  [[nodiscard]] SurfaceNormalGridView SurfaceNormals() const noexcept;
  [[nodiscard]] bool InBounds(Cell cell) const noexcept;
  [[nodiscard]] bool Known(Cell cell) const noexcept;
  [[nodiscard]] bool HardFeasible(Cell cell) const noexcept;
  [[nodiscard]] float ClearanceMeters(Cell cell) const noexcept;

 private:
  friend Result<SafeProjection> BuildSafeProjection(
      const SafeProjectionRequest&);
  std::shared_ptr<const ImmutableMapSnapshot> source_map_;
};

struct SafeProjectionRequest final {
  std::shared_ptr<const ImmutableMapSnapshot> map;
  std::shared_ptr<const SafetyCapabilityProfile> capability;
  std::shared_ptr<const PlannerAlgorithmConfig> algorithm_config;
  std::shared_ptr<const LearnedCostSnapshot> learned_cost;
};

Result<SafeProjection> BuildSafeProjection(
    const SafeProjectionRequest& request);
```

- [ ] **Step 4: Implement hard filtering before every soft cost**

First derive platform-specific projection rules with a closed variant visit:

```cpp
Result<SafetyProjectionLimits> ResolveProjectionLimits(
    const SafetyCapabilityProfile& capability);
```

This visitor must read wheel slope/clearance/braking limits, legged terrain and
body-clearance thresholds, or hopper landing-clearance/body-envelope limits
from the matching frozen capability variant. It must reject a variant/platform
mismatch and must not invent a permissive default for a missing hard limit.

For each cell, apply this exact order:

```cpp
const auto limits = std::get<SafetyProjectionLimits>(
    ResolveProjectionLimits(*request.capability));
const bool hard_feasible =
    KnownAndPlatformStaticallyFeasible(
        index, *request.map, *request.capability, limits);

projection.hard_feasible_mask[index] =
    static_cast<std::uint8_t>(hard_feasible);
projection.analytic_time_cost_s[index] =
    hard_feasible ? AnalyticTraversalTime(index, request) : infinity;
```

Unknown or inconclusive geometry must always yield `hard_feasible=false`.
After all analytic costs are fixed, call `ComposeBoundedSoftCost` for
energy/nonfatal-risk and store its resolved secondary-cost vectors/source
diagnostic. Never pass the hard mask or analytic execution time into that
composition function.

- [ ] **Step 5: Compute ESDF and connected components deterministically**

Implement row-major, stable-order distance transform and flood fill:

- initialize obstacle, unknown and out-of-bounds cells at distance zero;
- perform the exact same horizontal and vertical passes on every platform;
- label components by scanning `y` then `x`;
- expand four-neighbors in the fixed order left, right, down, up;
- never use unordered-container iteration to assign component IDs.

- [ ] **Step 6: Separate static stop candidates from current-state braking**

```cpp
Result<SafeStopAnchor> ResolveSafeStopAnchor(
    const SafeProjection& projection,
    const WheeledOrLeggedState& state,
    const SafetyCapabilityProfile& capability);
```

`safe_stop_candidate_mask` may contain only analytically certified terrain.
`ResolveSafeStopAnchor` must additionally check current speed, maximum
certified stopping primitive or wheel braking bound, and available path
distance. It must return the schema-derived `SafeStopAnchor` with exactly zero
linear speed and yaw rate plus its terrain certification ref.
“能够制动”不得写成静态地图单元属性。

- [ ] **Step 7: Run projection and safe-stop tests**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_map_tests
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R "^lpp_v3_map\." --no-tests=error --output-on-failure
```

Expected: unknown, obstacle, excess-slope, insufficient-clearance,
insufficient-braking-distance and capability-variant mismatch cases are
rejected; elevation/roughness/normal views remain available for hopper landing
and attitude certification.

- [ ] **Step 8: Commit safety projection**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/map cpp/src/map cpp/tests/map cpp/include/lunar_path_planner/v3/contracts/profiles.hpp cpp/CMakeLists.txt
git -C path-planner commit -m "feat: add shared planner safety projection"
```

---

### Task 8: Resolve a Known-Safe Goal or a Safe Frontier Terminal Set

**Files:**

- Create: `path-planner/cpp/include/lunar_path_planner/v3/goal/terminal_resolver.hpp`
- Create: `path-planner/cpp/src/goal/terminal_resolver.cpp`
- Create: `path-planner/cpp/tests/goal/terminal_resolver_test.cpp`
- Modify: `path-planner/cpp/CMakeLists.txt`

**Interfaces:**

- Consumes: Task 2 `GoalRegion` and Task 7 `SafeProjection`。
- Produces:
  `Result<ResolvedTerminalSet> ResolveTerminal(const TerminalResolutionRequest&)`。
- This module does not generate a second “global geometric route.” The
  non-executable unresolved tail is visualization/mission intent only.

- [ ] **Step 1: Write failing tests for goal, known infeasible, and frontier cases**

```cpp
#include <gtest/gtest.h>
#include "lunar_path_planner/v3/goal/terminal_resolver.hpp"

namespace lpp = lunar::planning::v3;

TEST(TerminalResolver, UsesGoalCellsInTheStartSafeComponent) {
  const auto result = lpp::ResolveTerminal(MakeReachableGoalRequest());
  ASSERT_TRUE(lpp::IsOk(result));
  const auto& terminals = std::get<lpp::ResolvedTerminalSet>(result);
  EXPECT_EQ(terminals.kind, lpp::TerminalKind::kGoal);
  EXPECT_FALSE(terminals.candidates.empty());
  EXPECT_FALSE(terminals.unresolved_tail.has_value());
}

TEST(TerminalResolver, ReportsFullyKnownBlockedGoalAsInfeasible) {
  const auto result = lpp::ResolveTerminal(MakeFullyKnownBlockedGoalRequest());
  ASSERT_TRUE(lpp::IsOk(result));
  EXPECT_EQ(std::get<lpp::ResolvedTerminalSet>(result).kind,
            lpp::TerminalKind::kGoalInfeasible);
}

TEST(TerminalResolver, StopsAtKnownSafeFrontierBeforeUnknownSpace) {
  const auto result = lpp::ResolveTerminal(MakeUnknownSeparatedGoalRequest());
  ASSERT_TRUE(lpp::IsOk(result));
  const auto& terminals = std::get<lpp::ResolvedTerminalSet>(result);
  ASSERT_EQ(terminals.kind, lpp::TerminalKind::kSafeFrontier);
  ASSERT_TRUE(terminals.unresolved_tail.has_value());
  EXPECT_FALSE(terminals.unresolved_tail->executable);
  for (const auto& candidate : terminals.candidates) {
    EXPECT_TRUE(candidate.requires_zero_speed);
  }
}
```

- [ ] **Step 2: Add the goal test target and confirm red**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_goal_tests
```

Expected: FAIL because the terminal resolver is absent.

- [ ] **Step 3: Define terminal resolution data without execution commands**

```cpp
enum class TerminalKind {
  kGoal,
  kSafeFrontier,
  kGoalInfeasible,
  kNoKnownSafeRoute
};

struct TerminalCandidate final {
  std::string stable_id;
  Cell cell;
  Vec3 position_m;
  std::optional<CircularYawInterval> yaw_interval;
  bool requires_zero_speed{};
  double clearance_m{};
};

struct UnresolvedTailPreview final {
  std::vector<Vec3> intent_polyline_m;
  bool executable{false};
  std::string reason_code;
};

struct ResolvedTerminalSet final {
  TerminalKind kind{TerminalKind::kNoKnownSafeRoute};
  std::vector<TerminalCandidate> candidates;
  std::optional<UnresolvedTailPreview> unresolved_tail;
  std::string reason_code;
};

struct TerminalResolutionRequest final {
  const SafeProjection& projection;
  const GoalRegion& goal_region;
  Cell start_cell;
  const SafetyCapabilityProfile& capability;
  const PlannerAlgorithmConfig& algorithm_config;
};

Result<ResolvedTerminalSet> ResolveTerminal(
    const TerminalResolutionRequest& request);
```

- [ ] **Step 4: Rasterize the convex goal region in stable row-major order**

Visit `GoalRegion::target` in stable row-major order:

- for `PointGoal`, include cells whose 3-D surface point is within
  `position_tolerance_m`;
- for `PlanarRegionGoal`, project each surface point into the frozen
  `LandingPlane` basis, apply the CCW UV half-space tests, and require the
  normal residual to be within `normal_tolerance_m`.

Record separately:

- all in-bounds goal cells;
- goal cells whose map value is known;
- hard-feasible goal cells in the start connected component.

Sort every resulting cell list by `(y, x)`. Do not infer traversability for
unknown or out-of-bounds cells.

- [ ] **Step 5: Implement the exact goal outcome decision table**

Apply these branches in order:

1. invalid start cell or start not hard-feasible:
   `kNoKnownSafeRoute/start_not_hard_feasible`;
2. at least one hard-feasible goal cell in the start component:
   `kGoal/known_safe_goal_reachable`;
3. every in-bounds goal cell is known and none is hard-feasible:
   `kGoalInfeasible/goal_fully_known_and_infeasible`;
4. a hard-feasible goal cell exists only in another known component:
   `kNoKnownSafeRoute/goal_in_disconnected_known_component`;
5. otherwise run safe-frontier extraction.

The goal branches must not silently convert a known collision into a frontier
goal.

- [ ] **Step 6: Extract and cluster safe-frontier cells deterministically**

A frontier cell must:

- belong to the start hard-feasible connected component;
- have a four-neighbor that is unknown;
- be set in `safe_stop_candidate_mask`;
- retain the configured footprint/error clearance after erosion.

Flood-fill frontier cells in row-major seed order using neighbors
left/right/down/up. For each cluster, choose the representative by the tuple:

```cpp
std::tuple{
    DistanceToGoalCentroid(cell),
    -projection.ClearanceMeters(cell),
    cell.y,
    cell.x};
```

Keep at most
`algorithm_config.ara_star.resource_caps.maximum_generated_candidates`
representatives and assign stable IDs
from `(map snapshot identity, y, x)`.

- [ ] **Step 7: Emit only a non-executable unresolved tail**

For `kSafeFrontier`, create a straight mission-intent polyline from each chosen
frontier representative toward the goal centroid, clipped at the known/unknown
boundary. Set `executable=false` unconditionally. If no frontier survives,
return `kNoKnownSafeRoute/no_certified_safe_frontier`.

- [ ] **Step 8: Run goal-resolution tests**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_goal_tests
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R "^lpp_v3_goal\." --no-tests=error --output-on-failure
```

Expected: reachable goal, fully known infeasible goal, disconnected known goal,
unknown-separated goal, deterministic clustering and no-frontier cases pass.

- [ ] **Step 9: Commit terminal resolution**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/goal cpp/src/goal cpp/tests/goal cpp/CMakeLists.txt
git -C path-planner commit -m "feat: add goal and safe frontier resolution"
```

---

### Task 9: Implement the Deterministic Generic ARA* Adapter Contract

**Files:**

- Create: `path-planner/cpp/include/lunar_path_planner/v3/search/platform_adapter.hpp`
- Create: `path-planner/cpp/include/lunar_path_planner/v3/search/ara_star.hpp`
- Create: `path-planner/cpp/tests/search/ara_star_test.cpp`
- Modify: `path-planner/cpp/CMakeLists.txt`

**Interfaces:**

- Consumes: Task 2 `AraStarConfig`/resource limits and Task 8 terminal sets。
- Produces the exact generic entry point:

```cpp
SearchResult<State, Edge> RunAraStar(
    const Adapter&, const SearchProblem<State>&, const AraStarConfig&);
```

- The adapter methods are named exactly
  `Key`、`Expand`、`HardFeasible`、`TransitionTime`、
  `AdmissibleTimeHeuristic`、`SecondaryCosts` and `IsTerminal`。

- [ ] **Step 1: Write a failing synthetic-graph adapter test**

```cpp
#include <gtest/gtest.h>
#include "lunar_path_planner/v3/search/ara_star.hpp"

namespace lpp = lunar::planning::v3;

TEST(AraStar, ImprovesTimeBoundWithoutUsingWallClock) {
  const TinyGraphAdapter adapter = MakeTinyGraphAdapter();
  const lpp::SearchProblem<TinyState> problem{
      .start = TinyState{0},
      .limits = MakeSearchLimits(100U, 100U, 4U)};
  const lpp::AraStarConfig config{
      .initial_epsilon = 2.5,
      .epsilon_decrement = 0.5,
      .target_epsilon = 1.0};

  const auto result =
      lpp::RunAraStar<TinyState, TinyEdge>(adapter, problem, config);

  ASSERT_EQ(result.status, lpp::SearchStatus::kSolved);
  ASSERT_FALSE(result.candidates.empty());
  EXPECT_EQ(result.candidates.front().total_time.value,
            std::chrono::seconds{3});
  EXPECT_DOUBLE_EQ(result.achieved_epsilon, 1.0);
}

TEST(AraStar, FiltersTransitionBeforeAddingAnyCost) {
  const auto result = RunGraphContainingCheaperHardInvalidEdge();
  EXPECT_EQ(result.candidates.front().edge_ids,
            (std::vector<std::string>{"safe-a", "safe-b"}));
}
```

- [ ] **Step 2: Add the search test target and confirm red**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_search_tests
```

Expected: FAIL because the generic search header is absent.

- [ ] **Step 3: Define state keys, transitions, and secondary costs**

```cpp
using StateKey = std::uint64_t;

struct SecondaryCostVector final {
  double energy{};
  double risk{};
  double smoothness{};

  SecondaryCostVector& operator+=(const SecondaryCostVector& rhs);
};

template <class State, class Edge>
struct SearchTransition final {
  std::string stable_edge_id;
  State successor;
  Edge edge;
};

template <class State>
struct SearchProblem final {
  State start;
  ResourceCaps limits;
};
```

- [ ] **Step 4: Encode the exact adapter concept**

```cpp
template <class Adapter, class State, class Edge>
concept PlatformSearchAdapter =
    requires(const Adapter& adapter,
             const State& state,
             const SearchProblem<State>& problem,
             const SearchTransition<State, Edge>& transition) {
      { adapter.Key(state) } -> std::same_as<StateKey>;
      { adapter.Expand(state) }
          -> std::same_as<std::vector<SearchTransition<State, Edge>>>;
      { adapter.HardFeasible(transition) } -> std::same_as<bool>;
      { adapter.TransitionTime(transition) }
          -> std::same_as<DurationNanoseconds>;
      { adapter.AdmissibleTimeHeuristic(state, problem) }
          -> std::same_as<DurationNanoseconds>;
      { adapter.SecondaryCosts(transition) }
          -> std::same_as<SecondaryCostVector>;
      { adapter.IsTerminal(state, problem) } -> std::same_as<bool>;
    };
```

Every wheel, legged, and hopper adapter in later plan volumes must satisfy this
concept; shared search must not inspect platform state fields.

- [ ] **Step 5: Define the result and exact public function template**

```cpp
enum class SearchStatus {
  kSolved,
  kNoPath,
  kResourceLimit,
  kInvalidAdapterCost
};

template <class State, class Edge>
struct SearchPath final {
  std::string stable_path_id;
  std::vector<State> states;
  std::vector<Edge> edges;
  std::vector<std::string> edge_ids;
  DurationNanoseconds total_time;
  SecondaryCostVector secondary_costs;
  bool fully_hard_validated{};
};

template <class State, class Edge>
struct SearchResult final {
  SearchStatus status{SearchStatus::kNoPath};
  std::vector<SearchPath<State, Edge>> candidates;
  double achieved_epsilon{};
  std::size_t expansions{};
  std::size_t generated_states{};
  std::string reason_code;
};

template <class State, class Edge, class Adapter>
  requires PlatformSearchAdapter<Adapter, State, Edge>
SearchResult<State, Edge> RunAraStar(
    const Adapter& adapter,
    const SearchProblem<State>& problem,
    const AraStarConfig& config);
```

- [ ] **Step 6: Implement one bounded ARA* improve-path pass**

Validate every duration as non-negative and use checked int64-nanosecond
addition for path accumulation. Convert to `long double` seconds only when
forming the weighted ARA* priority; never round-trip an accumulated cost through
floating point. For each expansion:

1. call `Expand`;
2. sort transitions by
   `(adapter.Key(successor), stable_edge_id)`;
3. call `HardFeasible` before time or secondary-cost accumulation;
4. reject negative duration, duration overflow or non-finite/negative secondary cost;
5. relax `g_time` and predecessor by stable key.

Order OPEN by:

```cpp
std::tuple{
    ToPrioritySeconds(g_time) +
        epsilon * ToPrioritySeconds(admissible_time_heuristic),
    g_time.value.count(),
    adapter.Key(state),
    stable_insertion_sequence};
```

Use ordered maps/sets for CLOSED and INCONS.

- [ ] **Step 7: Add bounded anytime improvement and candidate reconstruction**

Start at `initial_epsilon`; after each solution, move INCONS to OPEN, recompute
priorities and subtract `epsilon_decrement` without passing
`target_epsilon`. Stop only on one of:

- `target_epsilon` reached and improve-path completes;
- `maximum_expanded_states`, `maximum_open_states` or
  `maximum_generated_candidates` reached;
- OPEN is empty.

Collect each distinct fully hard-validated terminal path by stable edge-ID hash,
then sort by `(total_time.value.count(), stable_path_id)`. Do not include
`steady_clock`, deadlines, cancellation by elapsed time or OS scheduling state
in this header.

- [ ] **Step 8: Add determinism and resource-bound tests**

Run the same graph with every permutation of neighbor insertion order and
assert identical serialized `SearchResult`. Add tests for:

- equal-priority key tie;
- zero and negative transition times;
- NaN heuristic/secondary cost;
- expansion bound before a solution;
- best-so-far solution retained when a later ARA* pass hits a resource bound.

- [ ] **Step 9: Run the search tests**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_search_tests
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R "^lpp_v3_search\." --no-tests=error --output-on-failure
```

Expected: synthetic optimality, hard-filter order, resource limits and
permutation determinism pass.

- [ ] **Step 10: Commit generic ARA***

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/search cpp/tests/search cpp/CMakeLists.txt
git -C path-planner commit -m "feat: add deterministic generic ARA star"
```

---

### Task 10: Build the Time-Equivalent Fully Validated Candidate Pool

**Files:**

- Create: `path-planner/cpp/include/lunar_path_planner/v3/search/candidate_ranker.hpp`
- Create: `path-planner/cpp/src/search/candidate_ranker.cpp`
- Create: `path-planner/cpp/tests/search/candidate_ranker_test.cpp`
- Modify: `path-planner/cpp/CMakeLists.txt`

**Interfaces:**

- Consumes: Task 9 fully validated search candidates。
- Produces: `CandidateRanker::BuildTimeEquivalentPool`。

- [ ] **Step 1: Write a failing test that preserves the fastest time band**

```cpp
using namespace std::chrono_literals;

TEST(CandidateRanker, SecondaryCostsOnlyRankInsideTimeEquivalentPool) {
  const std::vector<lpp::CandidateScore> candidates{
      MakeCandidate("fast-risky", 10'000ms, 8.0, 5.0, 2.0, true),
      MakeCandidate("near-efficient", 10'030ms, 2.0, 1.0, 1.0, true),
      MakeCandidate("too-slow", 10'070ms, 0.1, 0.1, 0.1, true),
      MakeCandidate("invalid", 9'000ms, 0.0, 0.0, 0.0, false)};

  const auto result =
      lpp::CandidateRanker::BuildTimeEquivalentPool(
          candidates, lpp::DurationNanoseconds{50ms});

  ASSERT_TRUE(lpp::IsOk(result));
  const auto& pool = std::get<lpp::TimeEquivalentPool>(result);
  EXPECT_EQ(pool.minimum_validated_time.value, 10s);
  ASSERT_EQ(pool.ordered_candidates.size(), 2U);
  EXPECT_EQ(pool.ordered_candidates[0].stable_candidate_id,
            "near-efficient");
}
```

- [ ] **Step 2: Confirm the ranker symbols are absent**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_search_tests
```

Expected: FAIL because `CandidateRanker` is not defined.

- [ ] **Step 3: Define score and pool contracts**

```cpp
struct CandidateScore final {
  std::string stable_candidate_id;
  DurationNanoseconds total_time;
  double energy{};
  double risk{};
  double smoothness{};
  bool fully_hard_validated{};
};

struct TimeEquivalentPool final {
  DurationNanoseconds minimum_validated_time;
  DurationNanoseconds tolerance;
  std::vector<CandidateScore> ordered_candidates;
};

class CandidateRanker final {
 public:
  static Result<TimeEquivalentPool> BuildTimeEquivalentPool(
      std::span<const CandidateScore> candidates,
      DurationNanoseconds time_equivalence_tolerance);
};
```

- [ ] **Step 4: Implement filter, band, and lexicographic rank in that order**

The implementation must:

1. reject negative tolerance;
2. discard every candidate with `fully_hard_validated=false`;
3. reject negative/overflowing time and non-finite or negative secondary scores;
4. compute `T_min` from validated candidates only;
5. keep exactly `T <= T_min + tolerance` using checked nanosecond addition;
6. sort the pool by
   `(energy, risk, smoothness, total_time.value.count(), stable_candidate_id)`.

Energy/risk/smoothness must never compensate for leaving the time-equivalent
band or violating hard feasibility.

- [ ] **Step 5: Add edge and repeatability tests**

Cover empty validated input, exact tolerance boundary, duplicate stable IDs,
NaN, negative metrics, and every permutation of the same candidate list.
Duplicate IDs must return `ErrorCode::kInvalidArgument`.

- [ ] **Step 6: Run the ranker tests**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_search_tests
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R "^lpp_v3_search\." --no-tests=error --output-on-failure
```

Expected: the pool membership is time-driven and ranking is deterministic.

- [ ] **Step 7: Commit time-equivalent ranking**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/search/candidate_ranker.hpp cpp/src/search/candidate_ranker.cpp cpp/tests/search/candidate_ranker_test.cpp cpp/CMakeLists.txt
git -C path-planner commit -m "feat: add time equivalent candidate ranking"
```

---

### Task 11: Generate a Bounded Shared Convex Corridor

**Files:**

- Create: `path-planner/cpp/include/lunar_path_planner/v3/corridor/convex_corridor.hpp`
- Create: `path-planner/cpp/src/corridor/convex_corridor.cpp`
- Create: `path-planner/cpp/tests/corridor/convex_corridor_test.cpp`
- Modify: `path-planner/cpp/CMakeLists.txt`

**Interfaces:**

- Consumes: Task 7 hard-safe projection and a fully validated discrete
  wheel/legged centerline from Task 9。
- Produces the exact entry point
  `CorridorResult BuildConvexCorridor(const CorridorRequest&)`。
- Hopper references do not consume this 2-D corridor.

- [ ] **Step 1: Write failing success and all-or-fallback tests**

```cpp
#include <gtest/gtest.h>
#include "lunar_path_planner/v3/corridor/convex_corridor.hpp"

namespace lpp = lunar::planning::v3;

TEST(ConvexCorridor, CoversCenterlineWithOverlappingCertifiedCells) {
  const auto result =
      lpp::BuildConvexCorridor(MakeStraightKnownSafeCorridorRequest());
  ASSERT_EQ(result.status, lpp::CorridorStatus::kCertified);
  EXPECT_FALSE(result.cells.empty());
  EXPECT_EQ(result.fallback, lpp::CorridorFallback::kNone);
}

TEST(ConvexCorridor, ReturnsNoPartialCorridorWhenCertificationFails) {
  const auto result =
      lpp::BuildConvexCorridor(MakePinchedCorridorRequest());
  EXPECT_EQ(result.status, lpp::CorridorStatus::kFallbackRequired);
  EXPECT_TRUE(result.cells.empty());
  EXPECT_EQ(result.fallback,
            lpp::CorridorFallback::kUseDiscreteValidatedPrimitives);
}
```

- [ ] **Step 2: Add the corridor target and confirm red**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_corridor_tests
```

Expected: FAIL because `BuildConvexCorridor` is absent.

- [ ] **Step 3: Define half-plane, tightening, request, and result types**

```cpp
struct HalfPlane2 final {
  Vec2 outward_unit_normal;
  double upper_offset_m{};
};

struct CorridorTightening final {
  double footprint_support_radius_m{};
  double tracking_error_bound_m{};
  double additional_margin_m{};
  std::vector<HalfPlane2> platform_half_planes;
};

struct ConvexCorridorCell final {
  std::string stable_cell_id;
  double centerline_s_begin{};
  double centerline_s_end{};
  std::vector<HalfPlane2> half_planes;
};

struct CorridorRequest final {
  const SafeProjection& projection;
  std::span<const PathControlPoint> validated_centerline;
  CorridorTightening tightening;
  std::size_t max_planes{};
  std::size_t max_iterations{};
};

enum class CorridorStatus {
  kCertified,
  kFallbackRequired,
  kInvalidRequest
};

enum class CorridorFallback {
  kNone,
  kUseDiscreteValidatedPrimitives
};

struct CorridorResult final {
  CorridorStatus status{CorridorStatus::kInvalidRequest};
  CorridorFallback fallback{
      CorridorFallback::kUseDiscreteValidatedPrimitives};
  std::vector<ConvexCorridorCell> cells;
  std::string reason_code;
  std::size_t iterations{};
};

CorridorResult BuildConvexCorridor(const CorridorRequest& request);
```

- [ ] **Step 4: Seed corridor cells from the certified ESDF**

Walk centerline control points in ascending `s`. At each uncovered point:

1. map the point to a hard-feasible grid cell;
2. compute
   `effective_margin = footprint + tracking_error + additional_margin`;
3. require `esdf_clearance_m > effective_margin`;
4. seed an axis-aligned convex box with half-width
   `esdf_clearance_m - effective_margin`.

Reject unsorted/duplicate `s`, non-finite values, zero resource bounds or a
centerline point outside the hard-feasible mask.

- [ ] **Step 5: Inflate with deterministic obstacle-separating planes**

For the current seed, enumerate forbidden cells inside the bounded search
window, sort by `(squared_distance, y, x)`, and add the separating half-plane
from the seed centroid toward each forbidden-cell center. Normalize plane
coefficients and canonicalize negative zero. Apply `platform_half_planes` in
their canonical `(normal.x, normal.y, offset)` order. Stop at
`max_planes`/`max_iterations`; never stop on elapsed time.

- [ ] **Step 6: Certify every cell and adjacent overlap**

Before publishing:

- sample every covered grid cell and require hard feasibility plus the effective
  margin;
- require every centerline point in its assigned convex cell;
- require a non-empty intersection between each adjacent corridor-cell pair;
- require all coefficients finite and all cells bounded.

If any check fails, discard the entire vector and return
`kUseDiscreteValidatedPrimitives`; never return a certified prefix plus an
uncertified suffix.

- [ ] **Step 7: Add deterministic tightening and resource-limit tests**

Test wheel-radius tightening, legged body-envelope tightening, reversed
forbidden-cell insertion, plane-limit failure, iteration-limit failure,
unknown-cell contact and repeatable cell IDs. Do not test footstep feasibility
here; that belongs to the legged platform layer.

- [ ] **Step 8: Run corridor tests**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_corridor_tests
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R "^lpp_v3_corridor\." --no-tests=error --output-on-failure
```

Expected: certified open-space corridors and atomic discrete-path fallback
cases pass.

- [ ] **Step 9: Commit the shared corridor**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/corridor cpp/src/corridor cpp/tests/corridor cpp/CMakeLists.txt
git -C path-planner commit -m "feat: add bounded shared convex corridor"
```

---

### Task 12: Add a Reusable Bounded QP Interface and Optional OSQP Backend

**Files:**

- Create: `path-planner/cpp/include/lunar_path_planner/v3/optimization/bounded_qp_solver.hpp`
- Create: `path-planner/cpp/src/optimization/bounded_qp_solver.cpp`
- Create: `path-planner/cpp/src/optimization/osqp_bounded_qp_solver.cpp`
- Create: `path-planner/cpp/tests/optimization/bounded_qp_solver_test.cpp`
- Create: `path-planner/cpp/tests/optimization/osqp_bounded_qp_solver_test.cpp`
- Modify: `path-planner/cpp/CMakeLists.txt`
- Modify: `path-planner/cpp/CMakePresets.json`

**Interfaces:**

- Consumes: Eigen 5.0.1 and, only when enabled, OSQP 1.0.0。
- Produces: backend-neutral `BoundedQpSolver` and optional
  `OsqpBoundedQpSolver`。
- QP termination is iteration/tolerance bounded. There is no planner time
  deadline or OSQP wall-time limit.

- [ ] **Step 1: Write a failing backend-neutral interface test**

```cpp
#include <gtest/gtest.h>
#include "lunar_path_planner/v3/optimization/bounded_qp_solver.hpp"

namespace lpp = lunar::planning::v3;

TEST(BoundedQpSolver, RejectsProblemWithInconsistentDimensions) {
  DeterministicFakeQpSolver solver;
  auto problem = MakeUnitBoxQp();
  problem.upper_bounds.conservativeResize(2);
  const auto result = solver.Solve(problem, MakeQpSettings());
  ASSERT_FALSE(lpp::IsOk(result));
  EXPECT_EQ(std::get<lpp::Error>(result).code,
            lpp::ErrorCode::kInvalidArgument);
}
```

- [ ] **Step 2: Add the backend-neutral target and confirm red**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_optimization_tests
```

Expected: FAIL because the QP interface does not exist.

- [ ] **Step 3: Define sparse problem, settings, termination, and solution**

```cpp
using SparseQpMatrix =
    Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

struct SparseQpProblem final {
  SparseQpMatrix hessian_upper_triangle;
  Eigen::VectorXd gradient;
  SparseQpMatrix constraints;
  Eigen::VectorXd lower_bounds;
  Eigen::VectorXd upper_bounds;
};

struct BoundedQpSettings final {
  std::size_t max_iterations{};
  double absolute_tolerance{};
  double relative_tolerance{};
  bool polish{};
};

enum class QpTermination {
  kSolved,
  kMaxIterations,
  kPrimalInfeasible,
  kDualInfeasible,
  kNumericalFailure
};

struct QpSolution final {
  QpTermination termination{QpTermination::kNumericalFailure};
  Eigen::VectorXd primal;
  double objective{};
  double primal_residual{};
  double dual_residual{};
  std::size_t iterations{};
};

class BoundedQpSolver {
 public:
  virtual ~BoundedQpSolver() = default;
  virtual Result<QpSolution> Solve(
      const SparseQpProblem& problem,
      const BoundedQpSettings& settings) const = 0;
};
```

- [ ] **Step 4: Implement common validation before backend dispatch**

Add a non-virtual helper used by every backend. It must check dimensions,
finite coefficients, `lower <= upper`, symmetric upper-triangular Hessian
convention, positive `max_iterations`, positive finite tolerances and a
positive-semidefinite Hessian using deterministic Eigen factorization.

- [ ] **Step 5: Add the optional OSQP class without enabling the feature**

```cpp
#if defined(LPP_V3_HAS_OSQP)
class OsqpBoundedQpSolver final : public BoundedQpSolver {
 public:
  Result<QpSolution> Solve(
      const SparseQpProblem& problem,
      const BoundedQpSettings& settings) const override;
};
#endif
```

In CMake:

```cmake
if(LPP_V3_ENABLE_OSQP)
  find_package(osqp 1.0.0 EXACT CONFIG REQUIRED)
  add_library(
    lpp_v3_osqp_backend
    src/optimization/osqp_bounded_qp_solver.cpp)
  target_compile_definitions(
    lpp_v3_osqp_backend PUBLIC LPP_V3_HAS_OSQP=1)
  target_link_libraries(
    lpp_v3_osqp_backend
    PUBLIC lpp_v3_optimization osqp::osqp)
endif()
```

- [ ] **Step 6: Map bounded settings to OSQP deterministically**

Set `max_iter`, `eps_abs`, `eps_rel` and polishing from
`BoundedQpSettings`. Disable OSQP time-limit termination, adaptive parameter
changes that depend on wall time, verbose output and implicit warm starts.
Map every OSQP exit status to `QpTermination`; return the iteration count and
residuals. Never convert `kMaxIterations` into `kSolved`.

- [ ] **Step 7: Add OSQP-on configure/build/test presets**

Add `windows-msvc-debug-osqp` and `linux-gcc-debug-osqp` configure presets that
inherit their non-OSQP counterpart and set:

```json
{
  "LPP_V3_ENABLE_OSQP": "ON",
  "VCPKG_MANIFEST_FEATURES": "qp"
}
```

Each preset retains
`D:/xunce/build/path-planner-v3/${presetName}` as its binary directory.

- [ ] **Step 8: Run backend-neutral tests with OSQP off**

Run:

```powershell
cmake --preset windows-msvc-debug -S path-planner/cpp
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_optimization_tests
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R "^lpp_v3_optimization\." --no-tests=error --output-on-failure
```

Expected: interface validation and fake-backend tests pass without installing or
linking OSQP.

- [ ] **Step 9: Configure the optional feature and run exact-solution tests**

Run:

```powershell
cmake --preset windows-msvc-debug-osqp -S path-planner/cpp
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug-osqp --target lpp_v3_osqp_tests
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug-osqp -R "^lpp_v3_osqp\." --no-tests=error --output-on-failure
```

Expected: convex unit-box solution, infeasible problem and forced
`max_iterations=1` status mapping pass against OSQP 1.0.0.

- [ ] **Step 10: Commit the optional bounded QP layer**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/optimization cpp/src/optimization cpp/tests/optimization cpp/CMakeLists.txt cpp/CMakePresets.json
git -C path-planner commit -m "feat: add bounded QP interface and OSQP backend"
```

---

### Task 13: Add a Capacity-Bounded Deterministic Shared Cache

**Files:**

- Create: `path-planner/cpp/include/lunar_path_planner/v3/cache/deterministic_cache.hpp`
- Create: `path-planner/cpp/tests/cache/deterministic_cache_test.cpp`
- Modify: `path-planner/cpp/CMakeLists.txt`

**Interfaces:**

- Consumes: immutable shared values from Tasks 5–7。
- Produces: `DeterministicCache<Key, Value>` and
  `SafeProjectionCacheKey`；cache admission happens only after semantic
  validation.

- [ ] **Step 1: Write failing insertion-order and non-LRU tests**

```cpp
#include <gtest/gtest.h>
#include "lunar_path_planner/v3/cache/deterministic_cache.hpp"

namespace lpp = lunar::planning::v3;

TEST(DeterministicCache, FinalKeySetDoesNotDependOnInsertionOrder) {
  lpp::DeterministicCache<int, std::string> forward{2U};
  lpp::DeterministicCache<int, std::string> reverse{2U};
  for (int key : {1, 2, 3}) {
    forward.Publish(key, std::make_shared<const std::string>(
                             std::to_string(key)));
  }
  for (int key : {3, 2, 1}) {
    reverse.Publish(key, std::make_shared<const std::string>(
                             std::to_string(key)));
  }
  EXPECT_EQ(forward.Keys(), reverse.Keys());
  EXPECT_EQ(forward.Keys(), (std::vector<int>{1, 2}));
}

TEST(DeterministicCache, ReadDoesNotChangeEvictionOrder) {
  auto cache = MakeTwoEntryCache();
  ASSERT_NE(cache.Get(2), nullptr);
  cache.Publish(0, std::make_shared<const Value>(Value{}));
  EXPECT_EQ(cache.Keys(), (std::vector<int>{0, 1}));
}

TEST(SafeProjectionCacheKey, LearnedCostUsesFullContentIdentity) {
  const auto base = MakeProjectionCacheKeyWithLearnedCost(
      ContentRef{.id = "learned-1", .revision = 1,
                 .content_hash = HashOf("revision-1")});
  auto new_revision = base;
  new_revision.learned_cost_snapshot_ref->revision = 2;
  auto new_hash = base;
  new_hash.learned_cost_snapshot_ref->content_hash = HashOf("other-content");
  EXPECT_NE(base, new_revision);
  EXPECT_NE(base, new_hash);
}
```

- [ ] **Step 2: Add the cache test target and confirm red**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_cache_tests
```

Expected: FAIL because the deterministic cache header is absent.

- [ ] **Step 3: Define a total-order projection cache key**

```cpp
struct SafeProjectionCacheKey final {
  std::string map_snapshot_id;
  std::string map_revision;
  std::string layer_manifest_hash;
  ContentRef safety_capability_ref;
  ContentRef algorithm_config_ref;
  std::optional<ContentRef> learned_cost_snapshot_ref;
  std::string frame_id;
  std::uint64_t resolution_ieee754_bits{};
  std::string error_bound_model_id;

  auto operator<=>(const SafeProjectionCacheKey&) const = default;
};
```

Construct `resolution_ieee754_bits` using `std::bit_cast<std::uint64_t>` after
normalizing `-0.0` to `0.0`. Include every field that can change hard or soft
projection output; never key by raw pointer address.
In particular, the learned snapshot key is the complete ID/revision/hash
triple. Add a regression that keeps the same snapshot ID while changing only
revision, and another that changes only content hash; both must miss and build
a distinct projection. `std::nullopt` remains the analytic-cost baseline key.

- [ ] **Step 4: Implement immutable publication and deterministic eviction**

```cpp
template <class Key, class Value>
  requires std::totally_ordered<Key>
class DeterministicCache final {
 public:
  explicit DeterministicCache(std::size_t capacity);

  [[nodiscard]] std::shared_ptr<const Value> Get(const Key& key) const;
  void Publish(const Key& key, std::shared_ptr<const Value> value);
  [[nodiscard]] std::vector<Key> Keys() const;

 private:
  std::size_t capacity_;
  mutable std::shared_mutex mutex_;
  std::map<Key, std::shared_ptr<const Value>> entries_;
};
```

`Publish` inserts/replaces under an exclusive lock and, when over capacity,
erases `std::prev(entries_.end())`: the lexicographically greatest key. `Get`
uses a shared lock and does not update recency or counters. Reject zero capacity
at construction and null values at publication.

- [ ] **Step 5: Add concurrency and complete-key tests**

Run fixed sets of concurrent publications in varied scheduling orders, join all
threads, and assert the same lowest `capacity` keys and immutable values.
Individually mutate every `SafeProjectionCacheKey` field and assert a cache
miss. Add an integration assertion that `SemanticValidator` runs before cache
lookup and final request/bundle validation runs after cache lookup.

- [ ] **Step 6: Run cache tests**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_cache_tests
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R "^lpp_v3_cache\." --no-tests=error --output-on-failure
```

Expected: insertion permutation, read behavior, concurrent publication and
complete-key tests pass under ThreadSanitizer on the Linux CI preset.

- [ ] **Step 7: Commit deterministic caching**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/cache cpp/tests/cache cpp/CMakeLists.txt
git -C path-planner commit -m "feat: add deterministic shared planner cache"
```

---

### Task 14: Prove Shared-Core Composition and Measure the One-Second Metric

**Files:**

- Create: `path-planner/cpp/tests/integration/shared_core_pipeline_test.cpp`
- Create: `path-planner/cpp/tests/fixtures/shared_core_case_a.json`
- Create: `path-planner/cpp/tests/fixtures/shared_core_case_a.expected.json`
- Modify: `path-planner/cpp/benchmarks/shared_core_benchmark.cpp`
- Modify: `path-planner/cpp/CMakeLists.txt`

**Interfaces:**

- Consumes: Tasks 4–13 with a test-only fake platform adapter。
- Produces: the aggregate interface target `lpp_v3_common`, a deterministic
  shared-core evidence test and a benchmark result；it does not create
  `ReferenceBundleBuilder`, platform references or executor calls.

- [ ] **Step 1: Write the failing end-to-end shared-core test**

```cpp
TEST(SharedCorePipeline, ProducesTheSameValidatedArtifactsFromFixedInput) {
  const auto request = LoadCaseARequest();
  const auto first = RunSharedCoreWithFakeAdapter(request);
  const auto second = RunSharedCoreWithFakeAdapter(request);

  ASSERT_TRUE(first.validation.ok());
  ASSERT_EQ(first.search.status, lpp::SearchStatus::kSolved);
  ASSERT_FALSE(first.pool.ordered_candidates.empty());
  EXPECT_EQ(CanonicalSharedArtifactJson(first),
            CanonicalSharedArtifactJson(second));
  EXPECT_EQ(CanonicalSharedArtifactJson(first),
            LoadExpectedCaseAJson());
}
```

The test helper must explicitly call:

1. `SemanticValidator::Validate`;
2. deterministic cache lookup/publication for `BuildSafeProjection`;
3. `ResolveTerminal`;
4. `RunAraStar`;
5. `CandidateRanker::BuildTimeEquivalentPool`;
6. `BuildConvexCorridor` for the fake wheel/legged centerline;
7. final shared-artifact invariant validation.

- [ ] **Step 2: Add the aggregate shared-core and integration targets, then confirm red**

Add the stable platform-facing aggregate after all component targets exist:

```cmake
add_library(lpp_v3_common INTERFACE)
target_link_libraries(
  lpp_v3_common
  INTERFACE
    lpp_v3_contracts
    lpp_v3_codec
    lpp_v3_map
    lpp_v3_cost
    lpp_v3_goal
    lpp_v3_search
    lpp_v3_corridor
    lpp_v3_optimization
    lpp_v3_cache)
```

Wheel, legged and hopper plans link `lpp_v3_contracts` plus
`lpp_v3_common`; they do not enumerate or fork the component libraries.

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_integration_tests
```

Expected: FAIL until the fixture loader and shared-core test helper are added.

- [ ] **Step 3: Add a tiny hand-auditable canonical fixture**

Create a UTF-8 JSON case with:

- one immutable `8 x 8` known map and an unknown strip;
- one fixed wheeled capability/config revision;
- one optional valid pinned learned-cost snapshot;
- one safe-frontier terminal;
- two fully validated time-equivalent candidate paths;
- one certified corridor.

The expected JSON must contain stable IDs, terminal kind, candidate order,
hard-mask hash, learned-cost source and corridor plane coefficients. It must not
contain timestamps sampled at test runtime, pointer values or unordered-map
iteration output.

- [ ] **Step 4: Run the integration test twice from clean processes**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_integration_tests
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R "^lpp_v3_integration\." --no-tests=error --repeat until-fail:2 --output-on-failure
```

Expected: both processes emit byte-identical canonical expected artifacts.

- [ ] **Step 5: Implement the fixed-profile latency benchmark**

In `shared_core_benchmark.cpp`, construct immutable inputs outside the timed
region, then benchmark complete shared-core calls for fixed small/medium/large
profiles and cache-hit/cache-miss variants. Use Google Benchmark manual timing
only around the planner call and fixed arguments:

```cpp
BENCHMARK_REGISTER_F(SharedCoreBenchmark, MediumCacheMiss)
    ->Iterations(1000)
    ->Unit(benchmark::kMillisecond);
```

The planner itself must not receive the timer or a deadline. Record map size,
candidate count, resource limits, dependency/build IDs and benchmark-profile
schema version as counters/JSON metadata.

- [ ] **Step 6: Run repetitions and compute P95 outside planning**

Run:

```powershell
cmake --preset windows-msvc-release -S path-planner/cpp
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-release --target lpp_v3_shared_core_benchmark
D:/xunce/build/path-planner-v3/windows-msvc-release/lpp_v3_shared_core_benchmark.exe --benchmark_repetitions=30 --benchmark_out=D:/xunce/build/path-planner-v3/windows-msvc-release/shared_core_benchmark.json --benchmark_out_format=json
```

Use a benchmark-only postprocessor in the same target to sort recorded sample
durations and write P50/P95/P99 plus `p95_latency_target_met` into the
`benchmark-report.schema.json` representation. The result is an experimental
metric on the named hardware/profile, not a runtime cutoff and not a safety
claim.

- [ ] **Step 7: Run the complete default-off verification suite**

Run:

```powershell
cmake --preset windows-msvc-debug -S path-planner/cpp
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --parallel
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug --no-tests=error --output-on-failure
```

Expected: contracts, codec, map, cost, goal, search, corridor, optimization,
cache and integration targets all pass with OSQP disabled.

- [ ] **Step 8: Verify no Python or executor coupling entered the C++ core**

Run:

```powershell
rg -n "path-planner/src|src/lunar_exploration_ppo|executor|ReferenceBundleBuilder|steady_clock|system_clock" path-planner/cpp/include path-planner/cpp/src
```

Expected: no Python path, executor, `ReferenceBundleBuilder` or algorithm-side
clock reference. Any `steady_clock` occurrence is confined to
`path-planner/cpp/benchmarks/**`.

- [ ] **Step 9: Commit shared-core evidence and benchmark**

```powershell
git -C path-planner add cpp/tests/integration cpp/tests/fixtures/shared_core_case_a.json cpp/tests/fixtures/shared_core_case_a.expected.json cpp/benchmarks/shared_core_benchmark.cpp cpp/CMakeLists.txt
git -C path-planner commit -m "test: verify planner v3 shared core composition"
```

---

## Implementation Handoff

- Execute tasks in numeric order. Task 6 must precede Task 7 so learned soft
  cost is pinned before static projection composition.
- Platform plan volumes may implement wheel/legged/hopper adapters only through
  the seven `PlatformSearchAdapter` methods; they may not fork ARA*.
- The integration volume owns `ReferenceBundleBuilder`; this volume only defines
  its data inputs and validates the resulting response at the contract boundary.
- No task in this plan authorizes executor connection, default-planner
  replacement, checkpoint publication or canary execution.
