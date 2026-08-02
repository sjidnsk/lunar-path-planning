# Multiplatform Planner v3 Integration and Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将共享核心、轮式、足式和飞跃式平台层组装成单一 C++20 `PlannerV3`，实现原子 ReferenceBundle、承诺状态机、严格 wire codec、可选 Python 诊断适配和固定基准上的 P95 延迟验收。

**Architecture:** `PlannerV3Impl` 先完成合同、快照和既有执行上下文校验，再解析终端并直接分派至对应平台 planner；`ReferenceBundleBuilder` 只接受已经完整验证的平台参考。执行 directive 与规划 outcome 由独立状态仲裁器产生；schema、C++ codec、golden fixtures、系统故障矩阵和 benchmark report 共同形成验收闭环。

**Tech Stack:** C++20、CMake 3.28+、Eigen 5.0.1、nlohmann_json 3.12.0、GoogleTest 1.17.0、Google Benchmark 1.9.5、pybind11 3.0.1（可选）、Python 3.12 + pytest/jsonschema（仅合同测试）。

**执行顺序：** 先执行 Task 2（bundle builder），再执行 Task 3（仲裁器），然后
执行 Task 1（统一编排），最后按 Task 4–8 顺序执行。Task 1 明确消费 Task 2/3
产物，不能在依赖尚未建立时单独编译。

## Global Constraints

- 实现路径固定为 `path-planner/cpp/**`；机器 schema 固定为 `path-planner/schemas/v3/**`。
- 命名空间固定为 `lunar::planning::v3`。
- `PlannerV3::Plan` 不接收 deadline、剩余时长或 BenchmarkProfile。
- `bundle_id` 是唯一激活权威；子组件不可独立激活或跨 bundle 混用。
- `HOLD_STATIONARY` 不生成新 bundle。
- `SAFE_DEAD_END` 只能是 `reason_code`，不是 `planning_outcome`。
- BenchmarkProfile 只进入 benchmark runner，不进入运行时 planner。
- 学习代价只使用请求开始前固定的 ready snapshot。
- 固定输入、配置和线程策略必须产生稳定候选和稳定 JCS/SHA-256。
- 不替换现有默认 A*，不连接 executor，不修改用户已有的 `path_planner_adapter.py` 或 `env.py` 改动。
- Python 适配仅为 opt-in 诊断投影视图，不成为 C++ 热路径依赖。
- API、schema codec、Python projection 和 benchmark 测试分别使用稳定 CTest
  前缀 `lpp_v3_api.`、`lpp_v3_schema_codec.`、
  `lpp_v3_python_projection.`、`lpp_v3_benchmark.`；任何过滤式 gate 必须带
  `--no-tests=error`。
- 构建与 benchmark 输出分别写 `D:/xunce/build/path-planner-v3/windows-msvc-debug` 和 `D:/xunce/out/path-planner-v3`。

---

### Task 1: 统一 `PlannerV3Impl` 编排与平台分派

**Files:**
- Create: `path-planner/cpp/include/lunar_path_planner/v3/api/planner_v3.hpp`
- Create: `path-planner/cpp/src/api/planner_v3.cpp`
- Create: `path-planner/cpp/tests/integration/api/planner_dispatch_test.cpp`
- Modify: `path-planner/cpp/src/api/CMakeLists.txt`
- Modify: `path-planner/cpp/tests/CMakeLists.txt`

**Interfaces:**
- Consumes:

```cpp
class WheelPlanner;
class LeggedPlanner;
class HopperPlanner;
class SemanticValidator;
class SafeProjectionCache;
struct ReferenceActivationContext;
Result<ReferenceBundle> BuildReferenceBundle(
    const BundleBuildRequest&,
    const ReferenceActivationContext&);
PlanningResponse ArbitratePlanningResponse(
    const PlanningAttempt&,
    const std::optional<PreviousExecutionContext>&,
    const PlatformState&);
```

- Produces:

```cpp
class PlannerV3 {
 public:
  virtual ~PlannerV3() = default;
  [[nodiscard]] virtual PlanningResponse Plan(
      const PlanningRequest&) noexcept = 0;
};

class PlannerV3Impl final : public PlannerV3 {
 public:
  PlannerV3Impl(
      const SemanticValidator& semantic_validator,
      const ContractObjectRegistry& contract_registry,
      SafeProjectionCache& projection_cache,
      std::unique_ptr<WheelPlanner> wheel,
      std::unique_ptr<LeggedPlanner> legged,
      std::unique_ptr<HopperPlanner> hopper);

  PlanningResponse Plan(
      const PlanningRequest&) noexcept override;
};
```

`semantic_validator`、`contract_registry` 与 `projection_cache` 是显式非拥有依赖，并且必须比
`PlannerV3Impl` 活得更久。终端解析使用共享层已经冻结的
`ResolveTerminal(const TerminalResolutionRequest&)`；不得再引入未定义的
`SharedPlannerServices` 或第二套 resolver 接口。

- [ ] **Step 1: 写出目标解析后只调用一个平台 planner 的测试**

```cpp
TEST(PlannerDispatchTest, DispatchesDirectlyToSelectedPlatform) {
  auto doubles = MakePlatformPlannerDoubles();
  PlannerV3Impl planner = BuildPlanner(doubles);

  const auto response = planner.Plan(MakeWheelRequest());

  EXPECT_EQ(doubles.wheel->call_count(), 1);
  EXPECT_EQ(doubles.legged->call_count(), 0);
  EXPECT_EQ(doubles.hopper->call_count(), 0);
  EXPECT_EQ(doubles.shared_route_search_call_count(), 0);
  EXPECT_EQ(response.planning_outcome,
            PlanningOutcome::kNewReferenceReady);
}
```

- [ ] **Step 2: 写出合同失败不进入搜索的测试**

```cpp
TEST(PlannerDispatchTest, InvalidRequestNeverCallsPlatformPlanner) {
  auto doubles = MakePlatformPlannerDoubles();
  PlannerV3Impl planner = BuildPlanner(doubles);

  const auto response = planner.Plan(MakeFrameMismatchRequest());

  EXPECT_EQ(response.planning_outcome,
            PlanningOutcome::kInvalidRequest);
  EXPECT_EQ(doubles.total_platform_calls(), 0);
}
```

- [ ] **Step 3: 构建并确认失败**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_api_tests --parallel
```

Expected: 编译失败，提示 `PlannerV3Impl` 未定义。

- [ ] **Step 4: 实现固定编排顺序**

```text
semantic request validation
→ immutable handle/version resolution
→ active execution context validation
→ safe projection lookup/build
→ terminal resolution
→ exactly one platform planner
→ final platform validation
→ bundle construction
→ ValidateForActivation(bundle, request + registry)
→ outcome/directive arbitration
→ ValidateForActivation(response, request + registry)
```

任何阶段失败都不得跳过后续安全检查直接构造 bundle。`PlannerV3Impl` 必须持有
只读 `ContractObjectRegistry`，用原始请求构造 `ReferenceActivationContext`；
只有两次 activation-context 校验都通过，才可返回
`ACTIVATE_NEW_BUNDLE`。

- [ ] **Step 5: 实现异常封闭边界**

`Plan` 标记 `noexcept`；捕获分配失败以外的边界异常并映射为 `NUMERICAL_FAILURE` 或 `INVALID_REQUEST`。无法可靠构造响应的内存耗尽交由进程级故障处理，不返回伪安全响应。

- [ ] **Step 6: 运行分派测试**

Run:

```powershell
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R '^lpp_v3_api\.' --no-tests=error --output-on-failure
```

Expected: 全部通过。

- [ ] **Step 7: 提交统一编排**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/api/planner_v3.hpp cpp/src/api/planner_v3.cpp cpp/src/api/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/tests/integration/api/planner_dispatch_test.cpp
git -C path-planner commit -m "feat(v3): dispatch unified multiplatform planning"
```

---

### Task 2: 原子 `ReferenceBundleBuilder`

**Files:**
- Create: `path-planner/cpp/include/lunar_path_planner/v3/api/reference_bundle_builder.hpp`
- Create: `path-planner/cpp/src/api/reference_bundle_builder.cpp`
- Create: `path-planner/cpp/tests/unit/api/reference_bundle_builder_test.cpp`
- Modify: `path-planner/cpp/src/api/CMakeLists.txt`
- Modify: `path-planner/cpp/tests/CMakeLists.txt`

**Interfaces:**
- Consumes: validated `PlatformReference`, JCS canonicalizer, SHA-256, active bundle context,
  `ReferenceActivationContext`。
- Produces:

```cpp
template <class ContentT>
struct ComponentBuildInput final {
  ComponentId component_id;
  ContentT content;
};

struct BundleBuildRequest {
  BundleId bundle_id;
  std::uint32_t bundle_revision;
  std::optional<BundleId> supersedes_bundle_id;
  RequestId source_request_id;
  ContentRef source_map_snapshot_ref;
  ContentRef source_safety_capability_ref;
  ContentRef source_algorithm_config_ref;
  PlatformReference platform_reference;
  ComponentBuildInput<RouteSkeletonContent> route_skeleton;
  ComponentBuildInput<ReferenceViewContent> committed_prefix;
  ComponentBuildInput<ReferenceViewContent> preview;
  ReferenceValidity validity;
  ValidationSummary validation_summary;
  GenerationEvidence generation_evidence;
  std::optional<ReferenceBundle> active_bundle;
};

Result<ReferenceBundle> BuildReferenceBundle(
    const BundleBuildRequest&,
    const ReferenceActivationContext&);
```

- [ ] **Step 1: 写出相同组件 ID 不同内容 hash 被拒绝的测试**

```cpp
TEST(ReferenceBundleBuilderTest, RejectsReusedIdWithDifferentContent) {
  auto request = MakeBundleBuildRequest();
  request.active_bundle = MakeActiveBundleWithCommittedComponent(
      "committed-7", HashOf("old"));
  request.committed_prefix.component_id = ComponentId{"committed-7"};
  request.committed_prefix.content = MakeDifferentCommittedView();

  const auto result =
      BuildReferenceBundle(request, MakeReferenceActivationContext());

  ASSERT_FALSE(IsOk(result));
  EXPECT_EQ(std::get<Error>(result).code,
            ErrorCode::kInvalidArgument);
}
```

- [ ] **Step 2: 写出 committed/preview 跨平台参考被拒绝的测试**

```cpp
TEST(ReferenceBundleBuilderTest, ViewsMustReferenceSamePlatformPayload) {
  auto request = MakeBundleBuildRequest();
  request.committed_prefix.content.source_reference_hash = HashOf("wheel");
  request.preview.content.source_reference_hash = HashOf("legged");
  EXPECT_FALSE(IsOk(BuildReferenceBundle(
      request, MakeReferenceActivationContext())));
}

TEST(ReferenceBundleBuilderTest, JumpReadyCommitsGroundHoldAndPreviewsBoundary) {
  auto request = MakeHopperBundleBuildRequest();
  request.committed_prefix.content.selector =
      GroundHoldViewSelector{
          .anchor_id =
              hopper_reference(request).ground_hold_anchor.anchor_id};
  request.preview.content.selector =
      JumpViewSelector{
          .boundary_id =
              hopper_reference(request).jump_boundary.boundary_id,
          .scope = JumpViewSelector::Scope::kNextHop};

  const auto result =
      BuildReferenceBundle(request, MakeReferenceActivationContext());
  ASSERT_TRUE(IsOk(result));
  const auto& bundle = std::get<ReferenceBundle>(result);
  EXPECT_TRUE(std::holds_alternative<GroundHoldViewSelector>(
      bundle.committed_prefix.content.selector));
  EXPECT_TRUE(std::holds_alternative<JumpViewSelector>(
      bundle.preview.content.selector));
}

TEST(ReferenceBundleBuilderTest, ComputesCanonicalBundleHashLast) {
  const auto context = MakeReferenceActivationContext();
  const auto result =
      BuildReferenceBundle(MakeBundleBuildRequest(), context);
  ASSERT_TRUE(IsOk(result));
  const auto& bundle = std::get<ReferenceBundle>(result);
  EXPECT_EQ(bundle.bundle_hash,
            CanonicalBundleHashOmittingBundleHash(bundle));
  EXPECT_TRUE(
      SemanticValidator{}.ValidateForActivation(bundle, context).ok());
}

TEST(ReferenceBundleBuilderTest, RejectsMismatchedProvenanceAndLaunchContinuity) {
  auto request = MakeHopperBundleBuildRequest();
  request.validity.required_map_snapshot_ref = MakeOtherMapRef();
  EXPECT_FALSE(IsOk(BuildReferenceBundle(
      request, MakeReferenceActivationContext())));

  request = MakeHopperBundleBuildRequest();
  hopper_reference(request).jump_boundary.nominal_launch_state.position_m.x +=
      1.0;
  EXPECT_FALSE(IsOk(BuildReferenceBundle(
      request, MakeReferenceActivationContext())));
}
```

- [ ] **Step 3: 构建并确认失败**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_api_tests --parallel
```

Expected: 编译失败，提示 `BuildReferenceBundle` 未定义。

- [ ] **Step 4: 实现 JCS/SHA-256 组件 hash**

调用方的稳定 ID 生成器先把 `bundle_id` 和三个 `component_id` 写入
`BundleBuildRequest`；builder 不生成 ID，也不使用系统时间或随机完成顺序。
builder 对每个 `ComponentBuildInput.content` 应用 JCS/SHA-256，生成完整
`InlineComponent`；计算 hash 时不包含将被写入的 `component_hash`。
写 hash 前后都必须执行带上下文的 provenance/capability/continuity join；
局部 `Validate(bundle)` 不能替代 `ValidateForActivation`。

- [ ] **Step 5: 实现旧承诺组件复用**

仅在 ID、hash、规范化内容和平台 payload hash 全部相同时保留旧 committed component；preview 可使用新 ID。
Hopper 新 bundle 必须把 `GroundHoldViewSelector` 放在 committed prefix，把
同一 reference 的 `JumpViewSelector(kNextHop)` 放在 preview；两者的
anchor/boundary ID 和 source reference hash 均由 builder 终验。

- [ ] **Step 6: 实现 bundle revision 与 supersedes**

同 bundle ID 的内容修订必须严格递增；替换不同 bundle 时设置 `supersedes_bundle_id`。`schema_version` 不参与实例 revision。
在平台 reference、三个 component hash、revision/supersedes、validity、validation
summary 和 generation evidence 全部固定后，builder 最后对“省略顶层
`bundle_hash` 的完整 bundle”执行共享 JCS/SHA-256 并写入 `bundle_hash`。随后
重新计算一次、执行 semantic validation 与 encode/decode/hash round-trip；禁止
用原始 JSON 字节、组件 hash 拼接或第二套 canonicalizer 代替。

- [ ] **Step 7: 运行 bundle builder 测试**

Run:

```powershell
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R '^lpp_v3_api\.' --no-tests=error --output-on-failure
```

Expected: 全部通过。

- [ ] **Step 8: 提交 bundle builder**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/api/reference_bundle_builder.hpp cpp/src/api/reference_bundle_builder.cpp cpp/src/api/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/tests/unit/api/reference_bundle_builder_test.cpp
git -C path-planner commit -m "feat(v3): build atomic reference bundles"
```

---

### Task 3: outcome/directive 仲裁并消费既有飞跃承诺状态

**Files:**
- Create: `path-planner/cpp/include/lunar_path_planner/v3/api/execution_arbitrator.hpp`
- Create: `path-planner/cpp/src/api/execution_arbitrator.cpp`
- Create: `path-planner/cpp/tests/unit/api/execution_arbitrator_test.cpp`
- Modify: `path-planner/cpp/src/api/CMakeLists.txt`
- Modify: `path-planner/cpp/tests/CMakeLists.txt`

**Interfaces:**
- Consumes: bundle builder 已经验证的结果、可选 previous execution context、当前
  state，以及 Hopper 分卷 Task 13 已交付的
  `HopperCommitmentSnapshot`/`HopperCommitmentStateMachine`；不得创建第二套 reducer。
- Produces:

```cpp
struct PlanningAttempt final {
  std::optional<ReferenceBundle> validated_new_bundle;
  std::optional<Error> failure;
  ReasonCode reason_code;
  CallDiagnostics call_diagnostics;
};

PlanningResponse ArbitratePlanningResponse(
    const PlanningAttempt&,
    const std::optional<PreviousExecutionContext>&,
    const PlatformState&);
```

`PlanningAttempt` 必须恰好携带 `validated_new_bundle` 或 `failure` 之一。
`reason_code` 使用合同层大写开放码；内部诊断字符串必须先映射，不能直接写入
wire response。

- [ ] **Step 1: 建立完整 outcome/directive 参数化测试矩阵**

```cpp
struct AllowedCombination {
  PlanningOutcome outcome;
  ExecutionDirective directive;
  bool requires_bundle;
};

class ResponseCombinationTest
    : public ::testing::TestWithParam<AllowedCombination> {};

TEST_P(ResponseCombinationTest, AcceptsOnlyDeclaredCombination) {
  const auto response = MakeResponse(GetParam());
  EXPECT_EQ(ValidateResponseCombination(response).ok(), true);
}
```

矩阵逐字复制接口 schema 第 9 节。

- [ ] **Step 2: 写出 HOLD 无 bundle 测试**

```cpp
TEST(ExecutionArbitratorTest, StationaryHoldNeverCreatesBundle) {
  const auto response = ArbitratePlanningResponse(
      MakeFailedAttempt(), std::nullopt,
      MakeSafeStationaryWheelState());

  EXPECT_EQ(response.execution_directive,
            ExecutionDirective::kHoldStationary);
  EXPECT_FALSE(response.new_reference_bundle.has_value());
}
```

- [ ] **Step 3: 写出飞跃锁定后拒绝替换测试**

```cpp
TEST(ExecutionArbitratorTest, CommittedJumpContinuesPinnedActiveBundle) {
  const auto previous = MakeCommittedJumpExecutionContext(
      "bundle-1", "boundary-1");
  const auto response = ArbitratePlanningResponse(
      MakeFailedAttempt(), previous, MakeInFlightHopperState());

  EXPECT_EQ(response.execution_directive,
            ExecutionDirective::kContinueCommittedJump);
  ASSERT_TRUE(response.active_bundle_ref.has_value());
  EXPECT_EQ(response.active_bundle_ref->id, previous.active_bundle_ref.id);
  EXPECT_FALSE(response.new_reference_bundle.has_value());
}
```

- [ ] **Step 4: 构建并确认失败**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_api_tests --parallel
```

Expected: 编译失败，提示仲裁器未定义。

- [ ] **Step 5: 实现正常仲裁优先序**

```text
validated new bundle -> ACTIVATE_NEW_BUNDLE
valid committed jump -> CONTINUE_COMMITTED_JUMP
valid active ground bundle -> CONTINUE_ACTIVE_BUNDLE
safe stationary state -> HOLD_STATIONARY
otherwise -> NO_SAFE_PLANNER_REFERENCE
```

若 active reference 已失效，不得选择 `CONTINUE_ACTIVE_BUNDLE`。

- [ ] **Step 6: 接入飞跃承诺状态快照**

仲裁器只读取 Hopper Task 13 的单一真值：

```text
JUMP_COMMITTED / IN_FLIGHT
  + matching bundle_id and boundary_id
  -> CONTINUE_COMMITTED_JUMP

GROUND_HOLD / JUMP_READY / LANDED_HOLD
  -> may activate a newly validated bundle according to normal priority

EMERGENCY_DELEGATED
  -> NO_SAFE_PLANNER_REFERENCE
```

bundle/boundary ID 不匹配或 committed action 已失效时，不得自行回退为
`JUMP_READY`；状态转换仍由 `HopperCommitmentStateMachine::transition` 独占。

- [ ] **Step 7: 运行仲裁和状态机测试**

Run:

```powershell
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R '^lpp_v3_api\.' --no-tests=error --output-on-failure
```

Expected: 全部通过。

- [ ] **Step 8: 提交仲裁与状态机**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/api/execution_arbitrator.hpp cpp/src/api/execution_arbitrator.cpp cpp/src/api/CMakeLists.txt cpp/tests/CMakeLists.txt cpp/tests/unit/api/execution_arbitrator_test.cpp
git -C path-planner commit -m "feat(v3): arbitrate execution directives"
```

---

### Task 4: JSON Schema、C++ codec 与 golden fixture 一致性

**Files:**
- Modify: `path-planner/cpp/include/lunar_path_planner/v3/codec/json_codec.hpp`
- Modify: `path-planner/cpp/src/codec/json_codec.cpp`
- Create: `path-planner/cpp/tests/contract/schema_codec_conformance_test.cpp`
- Create: `path-planner/tests/test_v3_schema.py`
- Create: `path-planner/tests/fixtures/v3/schema/valid/wheel-request.json`
- Create: `path-planner/tests/fixtures/v3/schema/valid/legged-request.json`
- Create: `path-planner/tests/fixtures/v3/schema/valid/hopper-request.json`
- Create: `path-planner/tests/fixtures/v3/schema/valid/wheel-response.json`
- Create: `path-planner/tests/fixtures/v3/schema/valid/legged-response.json`
- Create: `path-planner/tests/fixtures/v3/schema/valid/hopper-response.json`
- Create: `path-planner/tests/fixtures/v3/schema/invalid/runtime-deadline.json`
- Create: `path-planner/tests/fixtures/v3/schema/invalid/platform-state-mismatch.json`
- Create: `path-planner/tests/fixtures/v3/schema/invalid/hold-with-bundle.json`
- Create: `path-planner/tests/fixtures/v3/schema/invalid/component-hash-conflict.json`
- Create: `path-planner/tests/fixtures/v3/schema/invalid/zero-ballistic-flight-time.json`
- Create: `path-planner/tests/fixtures/v3/schema/invalid/activation-provenance-mismatch.json`
- Modify: `path-planner/pyproject.toml`
- Modify: `path-planner/cpp/CMakeLists.txt`
- Modify: `path-planner/cpp/tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `path-planner/schemas/v3/**`, registry interfaces and all DTOs.
- Produces:

```cpp
Result<PlanningRequest> JsonCodec::DecodePlanningRequest(
    std::string_view payload,
    const ContractObjectRegistry& registry);
Result<PlanningResponse> JsonCodec::DecodePlanningResponse(
    std::string_view payload,
    const ReferenceActivationContext& activation_context);
std::string JsonCodec::EncodePlanningRequest(
    const PlanningRequest&);
std::string JsonCodec::EncodePlanningResponse(
    const PlanningResponse&);
```

本任务扩展共享合同分卷已经建立的 `JsonCodec`，不创建第二套 request/response codec。

- [ ] **Step 1: 在 dev dependency 中加入 Draft 2020-12 validator**

```toml
[project.optional-dependencies]
dev = [
  "pytest>=8",
  "jsonschema>=4.23,<5",
]
```

- [ ] **Step 2: 写出 Python schema 注册与 fixture 参数化测试**

```python
from jsonschema import Draft202012Validator


def test_all_v3_schema_documents_are_valid(schema_store):
    for schema in schema_store.values():
        Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize("fixture_path", valid_fixture_paths())
def test_valid_v3_fixture(fixture_path, validator_for_fixture):
    validator_for_fixture(fixture_path).validate(load_json(fixture_path))
```

invalid fixtures 必须断言至少一个结构错误。

- [ ] **Step 3: 运行 Python 测试并确认 codec fixture 尚不完整**

Run:

```powershell
$env:PYTHONPATH = "path-planner/src"
python -m pytest -q path-planner/tests/test_v3_schema.py
```

Expected: FAIL，指出 fixture 或 validator 映射尚未完成。

- [ ] **Step 4: 实现 request codec 的三层失败报告**

分别报告：

```text
JSON structure
typed decode
semantic validation / registry resolution
```

纳秒十进制字符串映射 `std::chrono::nanoseconds`；未知字段、错误 discriminator、hash 冲突、四元数非单位和 interval 逆序全部拒绝。

- [ ] **Step 5: 实现 response codec 与 JCS hash 复算**

编码前重新计算子组件 hash，并验证 READY/ACTIVATE/bundle 组合。解码待激活响应必须
使用 `ReferenceActivationContext` 完成 request ID、map/capability/config、证书、
Hopper 模型和连续性 join；非 ACTIVATE directive 不编码 `new_reference_bundle`。

- [ ] **Step 6: 写出 C++ golden fixture 往返测试**

```cpp
TEST(SchemaCodecConformanceTest, ValidFixturesRoundTripCanonically) {
  for (const auto& path : ValidFixturePaths()) {
    const auto document = LoadJson(path);
    const auto encoded = RoundTripTypedDocument(document, MakeRegistry());
    ASSERT_TRUE(IsOk(encoded)) << path;
    EXPECT_EQ(JcsCanonicalize(
                  nlohmann::json::parse(std::get<std::string>(encoded))),
              JcsCanonicalize(document));
  }
}
```

`RoundTripTypedDocument` 是测试辅助函数：按 `schema_version` 分派到共享
`JsonCodec` 的强类型 decode/encode 路径，并返回 `Result<std::string>`；不得直接返回原 JSON。

- [ ] **Step 7: 运行 Python 与 C++ 合同测试**

Run:

```powershell
$env:PYTHONPATH = "path-planner/src"
python -m pytest -q path-planner/tests/test_v3_schema.py
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_schema_codec_tests --parallel
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R '^lpp_v3_schema_codec\.' --no-tests=error --output-on-failure
```

Expected: 全部通过。

- [ ] **Step 8: 提交 codec 与 fixtures**

```powershell
git -C path-planner add pyproject.toml tests/test_v3_schema.py tests/fixtures/v3 cpp/include/lunar_path_planner/v3/codec cpp/src/codec cpp/tests/contract/schema_codec_conformance_test.cpp cpp/tests/CMakeLists.txt
git -C path-planner commit -m "feat(v3): enforce schema and codec conformance"
```

---

### Task 5: 可选 pybind11 诊断适配

**Files:**
- Create: `path-planner/cpp/include/lunar_path_planner/v3/bindings/python_projection.hpp`
- Create: `path-planner/cpp/src/bindings/python_projection.cpp`
- Create: `path-planner/cpp/src/bindings/module.cpp`
- Create: `path-planner/cpp/tests/unit/bindings/python_projection_test.cpp`
- Create: `src/lunar_exploration_ppo/integrations/path_planner_v3_adapter.py`
- Create: `tests/ppo_highres_frontier/test_path_planner_v3_adapter.py`
- Modify: `path-planner/cpp/CMakeLists.txt`
- Modify: `path-planner/cpp/vcpkg.json`

**Interfaces:**
- Consumes: `PlanningResponse`; pybind11 3.0.1 only when `LPP_V3_BUILD_PYTHON=ON`.
- Produces:

```cpp
struct PpoPathProjection {
  std::vector<GridCell> path_cells;
  double path_length_m;
  double final_yaw_rad;
  bool executable_reference_present;
  std::string source_bundle_id;
};

Result<PpoPathProjection> ProjectForPpoDiagnostics(
    const PlanningResponse&,
    const GridGeometry&);
```

- [ ] **Step 1: 写出 route skeleton 不可作为执行 path 的测试**

```cpp
TEST(PythonProjectionTest, NeverProjectsRouteSkeletonAsExecutablePath) {
  auto response = MakeResponseWithSkeletonButNoActivatedBundle();
  const auto result =
      ProjectForPpoDiagnostics(response, MakeGridGeometry());
  ASSERT_TRUE(IsOk(result));
  const auto& projection = std::get<PpoPathProjection>(result);
  EXPECT_FALSE(projection.executable_reference_present);
  EXPECT_TRUE(projection.path_cells.empty());
}
```

- [ ] **Step 2: 写出轮/足参考投影和 hopper 无伪路径测试**

轮式/足式从权威平台几何参考采样 `path_cells`；hopper 只投影当前点和下一着陆区域诊断，不伪造地面连续路径。

- [ ] **Step 3: 构建并确认失败**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_binding_tests --parallel
```

Expected: 编译失败，提示 projection 未定义。

- [ ] **Step 4: 实现纯 C++ projection**

projection 不调用 Python，也不修改规划结果；相同 bundle 和 grid geometry 必须生成相同输出。

- [ ] **Step 5: 增加 opt-in pybind target**

在 `vcpkg.json` 建立 `python` feature：

```json
{
  "python": {
    "description": "Build the optional Python diagnostics binding",
    "dependencies": [
      {"name": "pybind11", "version>=": "3.0.1"}
    ]
  }
}
```

同时在根 `overrides` 中固定 `{"name": "pybind11", "version": "3.0.1"}`。
CMake 仅在 `LPP_V3_BUILD_PYTHON=ON` 时构建模块。

- [ ] **Step 6: 新建独立 Python adapter**

```python
class PathPlannerV3DiagnosticAdapter:
    def project(self, response: object, grid: object) -> dict[str, object]:
        projection = _path_planner_v3.project_for_ppo(response, grid)
        return {
            "path_cells": projection.path_cells,
            "path_length_m": projection.path_length_m,
            "final_yaw_rad": projection.final_yaw_rad,
            "executable_reference_present": projection.executable_reference_present,
        }
```

不得修改或替换既有 adapter/default A*。

- [ ] **Step 7: 运行 C++ 和 Python 适配测试**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_binding_tests --parallel
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R '^lpp_v3_python_projection\.' --no-tests=error --output-on-failure
$env:PYTHONPATH = "src"
python -m pytest -q tests/ppo_highres_frontier/test_path_planner_v3_adapter.py
```

Expected: 全部通过。

- [ ] **Step 8: 提交可选适配**

```powershell
git -C path-planner add cpp/include/lunar_path_planner/v3/bindings cpp/src/bindings cpp/tests/unit/bindings/python_projection_test.cpp cpp/CMakeLists.txt cpp/vcpkg.json
git -C path-planner commit -m "feat(v3): expose optional Python diagnostic projection"
git add src/lunar_exploration_ppo/integrations/path_planner_v3_adapter.py tests/ppo_highres_frontier/test_path_planner_v3_adapter.py
git commit -m "feat: add opt-in planner v3 diagnostic adapter"
```

---

### Task 6: 跨平台系统、故障和确定性测试

**Files:**
- Create: `path-planner/cpp/tests/integration/system/multiplatform_scenario_test.cpp`
- Create: `path-planner/cpp/tests/integration/system/failure_injection_test.cpp`
- Create: `path-planner/cpp/tests/property/system/determinism_test.cpp`
- Create: `path-planner/cpp/tests/fixtures/system/wheel_scenarios.json`
- Create: `path-planner/cpp/tests/fixtures/system/legged_scenarios.json`
- Create: `path-planner/cpp/tests/fixtures/system/hopper_scenarios.json`
- Modify: `path-planner/cpp/tests/CMakeLists.txt`

**Interfaces:**
- Consumes: 完整 `PlannerV3` API。
- Produces: 三平台系统验收证据。

- [ ] **Step 1: 建立三平台场景矩阵**

每个 fixture 明确：

```text
platform_type
map_snapshot_ref
safety_capability_ref
algorithm_config_ref
initial_state
goal_region
expected_outcome
expected_directive
required_reason_codes
```

- [ ] **Step 2: 写出前沿、目标障碍和未知区语义测试**

```cpp
TEST(MultiplatformScenarioTest, KnownObstacleGoalIsNotFrontier) {
  const auto response = PlanFixture("goal_in_known_obstacle");
  EXPECT_EQ(response.planning_outcome,
            PlanningOutcome::kGoalInfeasible);
}
```

未知区阻隔可返回 `SAFE_FRONTIER_REFERENCE_READY`；未知区本身不得出现在权威参考。

- [ ] **Step 3: 注入合同、数值和资源故障**

覆盖地图层版本错配、旧 capability、旧 active bundle、学习快照 hash 错、QP 不收敛、搜索扩展上限、区间细分上限和缓存版本错。

- [ ] **Step 4: 写出多线程确定性测试**

```cpp
TEST(DeterminismTest, FixedThreadPolicyProducesStableResponseHash) {
  const auto fixture = LoadSystemFixture("complex_all_candidates");
  std::set<Sha256Digest> hashes;
  for (int i = 0; i < 100; ++i) {
    hashes.insert(HashResponse(Plan(fixture)));
  }
  EXPECT_EQ(hashes.size(), 1u);
}
```

- [ ] **Step 5: 构建并运行系统测试**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_system_tests --parallel
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R '^lpp_v3_api\.' --no-tests=error --output-on-failure
```

Expected: 全部通过。

- [ ] **Step 6: 运行现有 Python 非 Drake 回归**

Run:

```powershell
Push-Location path-planner
$env:PYTHONPATH = "src"
python -m pytest -m "not drake" -q
Pop-Location
```

Expected: 现有默认 planner 回归全部通过。

- [ ] **Step 7: 提交系统验收**

```powershell
git -C path-planner add cpp/tests/integration/system cpp/tests/property/system cpp/tests/fixtures/system cpp/tests/CMakeLists.txt
git -C path-planner commit -m "test(v3): verify multiplatform system contracts"
```

---

### Task 7: Benchmark runner、报告和 P95 验收

**Files:**
- Create: `path-planner/cpp/benchmarks/planner_api_benchmark.cpp`
- Create: `path-planner/cpp/benchmarks/benchmark_report.cpp`
- Create: `path-planner/cpp/benchmarks/benchmark_report.hpp`
- Create: `path-planner/cpp/benchmarks/fixtures/declared_benchmark_profile.json`
- Create: `path-planner/cpp/tests/unit/benchmark/benchmark_report_test.cpp`
- Modify: `path-planner/cpp/benchmarks/CMakeLists.txt`
- Modify: `path-planner/cpp/tests/CMakeLists.txt`

**Interfaces:**
- Consumes: `BenchmarkProfile`,完整 `PlannerV3`, JCS codec.
- Produces:

```cpp
struct BenchmarkSample {
  PlatformType platform_type;
  DurationNanoseconds api_latency;
  ModuleTimingBreakdown modules;
  PlanningOutcome outcome;
  SearchTerminationReason termination_reason;
};

BenchmarkReport BuildBenchmarkReport(
    const BenchmarkProfile&,
    std::span<const BenchmarkSample>);
```

- [ ] **Step 1: 写出分位数计算测试**

```cpp
TEST(BenchmarkReportTest, UsesNearestRankPercentiles) {
  const auto samples = MakeThreePlatformLatencySamplesMilliseconds(
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20});
  const auto report = BuildBenchmarkReport(
      MakeBenchmarkProfile(), samples);
  const auto& wheeled =
      FindPlatformResult(report, PlatformType::kWheeled);
  EXPECT_EQ(wheeled.latency.p95, 19ms);
  EXPECT_EQ(wheeled.latency.p99, 20ms);
}
```

- [ ] **Step 2: 写出 BenchmarkProfile 从不传给 planner 的测试**

使用 spy `PlannerV3`，断言每次调用只收到 `PlanningRequest`，profile 仅由 runner 决定 fixture 次序、预热和采样。

- [ ] **Step 3: 构建并确认失败**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --target lpp_v3_benchmark_tests --parallel
```

Expected: 编译失败，提示 `BuildBenchmarkReport` 未定义。

- [ ] **Step 4: 实现计时边界**

每个样本计时：

```text
typed request ready
→ validation
→ planning
→ final certification
→ response codec completes
```

不包含进程启动、原始建图、磁盘模型加载和网络传输。冷启动使用单独样本数组。

- [ ] **Step 5: 实现报告统计和三配置引用**

报告为每个平台 result 记录对应 SafetyCapability 的 ID/revision/hash，并在顶层记录共享 AlgorithmConfig、BenchmarkProfile 的 ID/revision/hash；`algorithm_config_ref` 必须从 profile content 原样复制。同时记录 mean、median、P95、P99、max、样本数、冷启动和模块分解。profile suite 与 report result 都必须恰好各含一个 `WHEELED`、`LEGGED`、`HOPPER`；重复或缺失平台、sample 与 suite capability 不匹配均拒绝生成报告。

```cpp
platform_result.p95_latency_target_met =
    platform_result.latency.p95 < std::chrono::seconds(1);
```

该字段只在全部规划调用自然结束后计算。

- [ ] **Step 6: 运行三平台固定基准**

Run:

```powershell
Push-Location path-planner/cpp
cmake --preset windows-msvc-release
cmake --build --preset windows-msvc-release --target lpp_v3_api_benchmark --parallel
Pop-Location
D:/xunce/build/path-planner-v3/windows-msvc-release/benchmarks/lpp_v3_api_benchmark.exe --profile path-planner/cpp/benchmarks/fixtures/declared_benchmark_profile.json --output D:/xunce/out/path-planner-v3/benchmark-report.json
```

Expected:

- 生成符合 `path-planner-v3-benchmark-report/v1` 的报告。
- 每个平台分别报告 P95。
- 不存在单次调用 1 秒截断。

- [ ] **Step 7: 验证 benchmark report schema**

Run:

```powershell
$env:PYTHONPATH = "path-planner/src"
python -m pytest -q path-planner/tests/test_v3_schema.py -k benchmark_report
```

Expected: PASS。

- [ ] **Step 8: 提交 benchmark runner**

```powershell
git -C path-planner add cpp/benchmarks/planner_api_benchmark.cpp cpp/benchmarks/benchmark_report.cpp cpp/benchmarks/benchmark_report.hpp cpp/benchmarks/fixtures/declared_benchmark_profile.json cpp/benchmarks/CMakeLists.txt cpp/tests/unit/benchmark/benchmark_report_test.cpp cpp/tests/CMakeLists.txt
git -C path-planner commit -m "perf(v3): report declared multiplatform API latency"
```

---

### Task 8: 最终验收、安装与文档

**Files:**
- Create: `path-planner/cpp/cmake/PlannerV3Config.cmake.in`
- Create: `path-planner/cpp/docs/build-and-test.md`
- Create: `path-planner/cpp/docs/interface-and-safety-boundary.md`
- Modify: `path-planner/cpp/CMakeLists.txt`
- Modify: `path-planner/README.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: 所有实现、测试和 schema。
- Produces: 可安装 `lunar_path_planner_v3` CMake package 与最终开发者说明。

- [ ] **Step 1: 增加安装和 find_package 烟测**

安装 targets：

```text
LunarPathPlannerV3::contracts
LunarPathPlannerV3::common
LunarPathPlannerV3::wheel
LunarPathPlannerV3::legged
LunarPathPlannerV3::hopper
LunarPathPlannerV3::api
```

创建临时 consumer CMake 项目，使用 `find_package(LunarPathPlannerV3 CONFIG REQUIRED)` 编译最小程序。

- [ ] **Step 2: 运行 Debug 和 Release 全量测试**

Run:

```powershell
Push-Location path-planner/cpp
cmake --preset windows-msvc-debug
cmake --build --preset windows-msvc-debug --parallel
ctest --preset windows-msvc-debug --output-on-failure
cmake --preset windows-msvc-release
cmake --build --preset windows-msvc-release --parallel
ctest --preset windows-msvc-release --output-on-failure
Pop-Location
```

Expected: 两种配置全部通过。

- [ ] **Step 3: 运行 Python schema 与既有回归**

Run:

```powershell
$env:PYTHONPATH = "path-planner/src"
python -m pytest -q path-planner/tests/test_v3_schema.py
Push-Location path-planner
python -m pytest -m "not drake" -q
Pop-Location
```

Expected: 全部通过。

- [ ] **Step 4: 运行三平台 benchmark**

Run:

```powershell
D:/xunce/build/path-planner-v3/windows-msvc-release/benchmarks/lpp_v3_api_benchmark.exe --profile path-planner/cpp/benchmarks/fixtures/declared_benchmark_profile.json --output D:/xunce/out/path-planner-v3/benchmark-report.json
```

报告必须如实显示每个平台的 `p95_latency_target_met`；未达 P95 指标时不得修改 planner 语义或提前返回。

- [ ] **Step 5: 更新边界文档**

文档明确：

- C++ v3 为 opt-in clean-room implementation。
- 现有默认 A* 未替换。
- 未连接 executor。
- 足式不保证足步可行。
- 飞跃发射后不可重定向。
- 1 秒是基准 P95，不是 runtime deadline。

- [ ] **Step 6: 检查仓库差异范围**

Run:

```powershell
git -C path-planner status --short
git status --short
```

Expected: 不包含对既有 dirty `path_planner_adapter.py`、`env.py` 或其他用户文件的覆盖。

- [ ] **Step 7: 提交安装和文档**

```powershell
git -C path-planner add cpp/cmake/PlannerV3Config.cmake.in cpp/docs cpp/CMakeLists.txt README.md
git -C path-planner commit -m "docs(v3): finalize build and safety boundaries"
git add README.md path-planner
git commit -m "docs: register multiplatform planner v3"
```

---

## Integration Plan Completion Gate

最终必须运行：

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --parallel
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug --no-tests=error --output-on-failure
$env:PYTHONPATH = "path-planner/src"
python -m pytest -q path-planner/tests/test_v3_schema.py
Push-Location path-planner
python -m pytest -m "not drake" -q
Pop-Location
D:/xunce/build/path-planner-v3/windows-msvc-release/benchmarks/lpp_v3_api_benchmark.exe --profile path-planner/cpp/benchmarks/fixtures/declared_benchmark_profile.json --output D:/xunce/out/path-planner-v3/benchmark-report.json
```

验收：

- 三个平台完整系统测试通过。
- schema、C++ codec 和 golden fixtures 一致。
- outcome/directive 组合全部受控。
- 只有 READY/ACTIVATE 响应携带新 bundle。
- `HOLD_STATIONARY` 不携带 bundle。
- 固定输入的响应 hash 稳定。
- 现有 Python 非 Drake 回归通过。
- 三个平台分别报告稳态 P95；P95 小于 1 秒作为实验目标如实判定。
- 无 1 秒 runtime cutoff、无 default policy 替换、无 executor 连接。
