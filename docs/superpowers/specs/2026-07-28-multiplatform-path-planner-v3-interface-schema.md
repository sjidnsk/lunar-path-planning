# 多平台路径规划 v3 接口 Schema

## 1. 状态与权威关系

- 状态：与《多平台路径规划 v3 设计规范》同步的接口冻结稿。
- 规范代际：`v3`。
- 首个 wire schema 修订：`v1`。
- 机器可读权威：`path-planner/schemas/v3/`。
- C++ 域模型权威：本文件定义的类型关系和 C++ 头文件实现。
- JSON Schema 负责线格式结构校验；C++ semantic validator 负责跨字段、几何、时序和安全语义。
- 运行时热路径只使用强类型 C++ 对象，不在搜索、优化或验证循环中解释 JSON Schema。

算法规范与接口规范的关系：

1. 算法和安全含义由 `2026-07-28-multiplatform-path-planner-v3-design.md` 定义。
2. 本文件把抽象对象冻结为可序列化字段、C++ 类型和状态组合。
3. `path-planner/schemas/v3/*.schema.json` 是跨进程、fixture 和回放使用的线格式。
4. 若机器 schema 与本文件冲突，必须停止集成并修复两者，不能静默选择一侧。

---

## 2. Schema 文件

```text
path-planner/schemas/v3/
├── README.md
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

所有 JSON Schema 文件必须：

- 使用 JSON Schema Draft 2020-12。
- 使用稳定 URN `$id`。
- 通过 `const schema_version` 冻结消息版本。
- 对对象使用 `additionalProperties: false`；组合对象使用 `unevaluatedProperties: false`。
- 对数组长度、字符串长度、图节点数和多项式段数设置显式上限。
- 用 `oneOf` 与 `const` discriminator 表示平台联合类型。
- 不允许 `NaN`、正负无穷、重复对象键或未知字段。

Schema 版本字符串：

```text
path-planner-v3-planning-request/v1
path-planner-v3-reference-bundle/v1
path-planner-v3-planning-response/v1
path-planner-v3-wheeled-reference/v1
path-planner-v3-legged-body-reference/v1
path-planner-v3-hopper-reference/v1
path-planner-v3-safety-capability-profile/v1
path-planner-v3-planner-algorithm-config/v1
path-planner-v3-benchmark-profile/v1
path-planner-v3-benchmark-report/v1
```

`v3` 表示产品合同代际；末尾 `v1` 表示该消息的线格式修订。实例自身的修改次数使用 `revision` 或 `bundle_revision`，不得命名为 `schema_version`。

---

## 3. C++ 命名空间与公共 API

```cpp
namespace lunar::planning::v3 {

struct PlanningRequest;
struct PlanningResponse;
struct ReferenceActivationContext;
struct ContractIssue;
struct Error;
template <class T>
using Result = std::variant<T, Error>;
class ImmutableMapSnapshot;
class SafetyCapabilityProfile;
class PlannerAlgorithmConfig;
class LearnedCostEvaluator;
class LearnedCostSnapshot;
class ContractObjectRegistry;

class PlannerV3 {
 public:
  virtual ~PlannerV3() = default;

  [[nodiscard]] virtual PlanningResponse Plan(
      const PlanningRequest& request) noexcept = 0;
};

}  // namespace lunar::planning::v3
```

公共 `Plan` 接口：

- 不接收 deadline、剩余时长或 BenchmarkProfile。
- 输入对象及其引用的地图、能力、算法配置和学习代价快照在调用期间不可变。
- 普通合同错误通过 `PlanningResponse` 返回，不抛出异常。
- 进程资源损坏等无法构造响应的错误由进程边界处理，不伪装成安全规划结果。

JSON 边界接口：

```cpp
namespace lunar::planning::v3 {

class JsonCodec final {
 public:
  static Result<PlanningRequest> DecodePlanningRequest(
      std::string_view payload,
      const ContractObjectRegistry& registry);
  static Result<PlanningResponse> DecodePlanningResponse(
      std::string_view payload,
      const ReferenceActivationContext& activation_context);
  static std::string EncodePlanningRequest(
      const PlanningRequest& request);
  static std::string EncodePlanningResponse(
      const PlanningResponse& response);
};

}  // namespace lunar::planning::v3
```

`immutable_data_handle` 等 wire 字段只是 registry key。C++ 域对象必须把它解析成
`std::shared_ptr<const ImmutableMapSnapshot>` 等受控不可变句柄；不得把内存地址序列化。

---

## 4. 基础类型

### 4.1 强类型标识和内容引用

```cpp
using Identifier = std::string;
using Sha256Digest = std::string;

struct ContentRef {
  std::string id;
  std::uint32_t revision;
  Sha256Digest content_hash;
};

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
```

`Identifier` 的类型层保持轻量，语义层按字段角色区分；codec/validator 拒绝空串、
超长值和非法字符。`Sha256Digest` 在 C++ 中保存 wire 的 64 字符小写十六进制串，
构造时必须完成长度和字符校验。

`ContentRef` 用于：

- `SafetyCapabilityProfile`。
- `PlannerAlgorithmConfig`。
- `BenchmarkProfile`。
- 运动原语集。
- 学习模型快照。
- 误差模型。

相同 ID 与 revision 必须对应相同 hash。发现不同内容时返回合同错误，不能覆盖 registry 中的既有对象。

### 4.2 时间

C++ 使用：

```cpp
struct DurationNanoseconds {
  std::chrono::nanoseconds value;
};

struct ClockStamp {
  ClockId clock_id;
  std::chrono::nanoseconds tick;
};

struct TimeInterval {
  DurationNanoseconds start_offset;
  DurationNanoseconds end_offset;
};
```

wire 格式中的 int64 纳秒值使用十进制字符串，避免 JavaScript/IEEE-754 对超过 \(2^{53}-1\) 的整数失真。codec 必须拒绝：

- 前导正号。
- 负持续时间。
- `JumpBoundary.ballistic_flight_time` 和能力的 `maximum_flight_time` 为零。
- 溢出 `std::int64_t` 的值。
- 混用 clock domain 的比较。

轮式和足式参考中的时间偏移均相对 `reference_time_origin`。Hopper 的
`CertifiedFlightTube` 截面和 `PredictedLandingFootprint.landing_time_window`
是唯一例外：它们显式相对 `JumpBoundary.ballistic_time_origin =
BALLISTIC_LAUNCH_EVENT`，因此在 `JUMP_READY` 等待多久都不会改变弹道合同。

### 4.3 向量、四元数和确定性误差集合

```cpp
struct Vec2 { double x; double y; };
struct Vec3 { double x; double y; double z; };
struct Quaternion { double w; double x; double y; double z; };

struct AxisAlignedBox3 {
  Vec3 center;
  Vec3 half_extent;
};

struct EuclideanBall3 {
  Vec3 center;
  double radius;
};

using DeterministicVectorSet3 =
    std::variant<AxisAlignedBox3, EuclideanBall3>;

struct SymmetricScalarInterval {
  double center;
  double half_width;
};

struct RotationVectorBall {
  double radius_rad;
};

struct WheeledOrLeggedErrorBounds {
  DeterministicVectorSet3 position_bound_m;
  SymmetricScalarInterval yaw_bound_rad;
  DeterministicVectorSet3 linear_velocity_bound_mps;
  SymmetricScalarInterval yaw_rate_bound_radps;
};

struct HopperErrorBounds {
  DeterministicVectorSet3 position_bound_m;
  RotationVectorBall orientation_bound;
  DeterministicVectorSet3 linear_velocity_bound_mps;
  DeterministicVectorSet3 angular_velocity_bound_radps;
};
```

所有集合尺度必须非负且有限；状态误差通常以零为中心，若非零则中心也参与确定性传播。四元数必须满足单位范数容差，并通过规范化符号规则得到稳定编码。

### 4.4 yaw 圆周区间

```cpp
struct CircularYawInterval {
  enum class Representation { kCanonicalCcw };
  Representation representation;
  double start_rad;
  double span_rad;
  bool closed;
};
```

规范编码：

- `start_rad` 归一化到 \([-\pi,\pi)\)。
- `span_rad` 位于 \([0,2\pi]\)，沿逆时针方向解释，端点闭合。
- `span_rad = 0` 是单一 yaw；`span_rad = 2\pi` 是完整圆周，因此没有零宽/整圆歧义。
- wire 还固定 `representation = "canonical_ccw"` 与 `closed = true`；它们由 codec 注入并校验。

### 4.5 平面与凸多边形

```cpp
struct LandingPlane {
  Vec3 origin_m;
  Vec3 normal;
  Vec3 basis_u;
  Vec3 basis_v;
  double residual_bound_m;
};

struct ConvexPolygonUv {
  enum class Winding { kCcw };
  std::vector<Vec2> vertices_uv;
  Winding winding;
};

struct PlanarConvexRegion {
  LandingPlane plane;
  ConvexPolygonUv polygon;
};
```

semantic validator 必须验证：

- `normal`、`basis_u`、`basis_v` 单位化。
- 三个轴两两正交且 `basis_u × basis_v` 与 normal 同向。
- 顶点逆时针、无重复、无自交且严格凸。
- 顶点数与 schema 上限一致。

### 4.6 枚举与开放诊断码

```cpp
enum class PlatformType { kWheeled, kLegged, kHopper };
enum class DriveDirection { kForward, kReverse };

enum class PrimitiveKind {
  kDriveForward,
  kDriveReverse,
  kSpinCw,
  kSpinCcw,
  kStopAndSwitch,
  kBodyTranslation,
  kBodySpin,
  kBodyCoupled,
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
  kActiveReferenceInvalidated,
};

enum class ExecutionDirective {
  kActivateNewBundle,
  kContinueActiveBundle,
  kHoldStationary,
  kContinueCommittedJump,
  kNoSafePlannerReference,
};

enum class FutureViability {
  kViable,
  kUnknown,
  kNoCertifiedContinuation,
};

enum class JumpViewScope {
  kNextHop,
  kFutureMissionPreview,
};

enum class ReferenceViewRole {
  kCommittedPrefix,
  kPreview,
};

enum class InvalidationCondition {
  kMapSafetyRevisionChanged,
  kStateDeviationExceeded,
  kCapabilityRevisionChanged,
  kExecutionCursorPastReplacementBoundary,
  kJumpBoundaryLocked,
  kJumpLaunched,
};

enum class GenerationMode {
  kSmoothedSplineReference,
  kValidatedPrimitiveChainReference,
  kCertifiedBallisticReference,
};

enum class LearnedCostUsage {
  kDisabled,
  kUsedBoundedSoftCost,
  kFellBackToAnalytic,
};

using ReasonCode = std::string;
```

`ReasonCode`、termination reason 与 warning/message code 是受
`^[A-Z][A-Z0-9_]*$`、长度上限和版本治理约束的开放代码，不把每个诊断原因编译进 wire 枚举。

---

## 5. 状态、目标与请求

### 5.1 平台状态

```cpp
struct WheeledOrLeggedState {
  Vec3 position_m;
  double yaw_rad;
  Vec3 linear_velocity_mps;
  double yaw_rate_radps;
  WheeledOrLeggedErrorBounds error_bounds;
};

struct HopperState {
  Vec3 position_m;
  Quaternion orientation_body_to_frame;
  Vec3 linear_velocity_mps;
  Vec3 angular_velocity_radps;
  HopperErrorBounds error_bounds;
};

using PlatformState =
    std::variant<WheeledOrLeggedState, HopperState>;
```

请求的 `platform_type` 必须与状态 variant 一致：

- `WHEELED`、`LEGGED` -> `WheeledOrLeggedState`。
- `HOPPER` -> `HopperState`。

### 5.2 目标区域

```cpp
struct PointGoal {
  Vec3 position_m;
  double position_tolerance_m;
};

struct PlanarRegionGoal {
  LandingPlane plane;
  ConvexPolygonUv polygon;
  double normal_tolerance_m;
};

using MetadataValue = std::variant<std::string, double, bool>;

struct MetadataEntry {
  std::string key;
  MetadataValue value;
};

using PositionGoal =
    std::variant<PointGoal, PlanarRegionGoal>;

struct GoalRegion {
  GoalId goal_id;
  PositionGoal target;
  std::optional<CircularYawInterval> optional_yaw_interval;
  std::optional<Vec3> mission_direction_hint;
  std::vector<MetadataEntry> task_metadata;
};
```

位置容差非负、平面/多边形一致、metadata key 唯一和 mission direction 非零由 semantic validator 检查。

### 5.3 学习代价快照

```cpp
struct LearnedOutputBound {
  std::string output_name;
  double lower;
  double upper;
};

struct LearnedCostSnapshotBinding {
  ContentRef snapshot_ref;
  std::string registry_handle;
  std::shared_ptr<const LearnedCostSnapshot> resolved_snapshot;
};
```

wire 请求只传 `snapshot_ref`、本地 `registry_handle`、`ready_before_request = true` 和
`hard_feasibility_authority = "NONE"`。只有解码前已经 ready、registry 内容与 ref hash
一致的不可变 `LearnedCostSnapshot` 才能解析成上面的运行时 binding。
`LearnedCostSnapshot` 内部固定 model ref、输入合同 hash、有限输出边界和只读 evaluator；
请求中没有 binding 时，整次调用固定使用解析代价；不得在调用中等待模型加载或按完成先后切换。

### 5.4 已解析能力依赖

能力 profile 的 `ContentRef` 不是可直接计算的模型。解码请求时必须一次性解析并
固定全部依赖：

```cpp
class MotionModel;
class AnalyticCostModel;
class GravityModel;
class DeterministicErrorModel;
class ActuatorOrImpulseProfile;
class BodyRotationEnvelope;
class AttitudeTighteningTable;
enum class ContractObjectKind {
  kMotionModel,
  kAnalyticCostModel,
  kGravityModel,
  kDeterministicErrorModel,
  kActuatorOrImpulseProfile,
  kBodyRotationEnvelope,
  kAttitudeTighteningTable,
  kCertification,
};

struct CertificationProvenance {
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

template <class T>
struct ResolvedBinding {
  ContentRef content_ref;
  std::shared_ptr<const T> object;
};

struct ResolvedCapabilityBindings {
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

class ContractObjectRegistry {
 public:
  virtual ~ContractObjectRegistry() = default;
  virtual Result<ResolvedCapabilityBindings> ResolveCapabilityBindings(
      const SafetyCapabilityProfile& profile) const = 0;
  virtual Result<std::shared_ptr<const ImmutableContractObject>> Resolve(
      const ContentRef& ref,
      ContractObjectKind expected_kind) const = 0;
};
```

`ResolveCapabilityBindings` 必须检查 profile 的直接引用，并从固定 motion model
解析 Hopper 的误差模型、执行器/冲量 profile 和机体旋转包络依赖。每个
`ResolvedBinding.content_ref` 与对象的 ID、revision、JCS/SHA-256 必须完全一致。
平台 planner 只消费这些只读绑定，不根据裸 ID 自行加载或选择模型。

### 5.5 既有执行上下文

```cpp
enum class JumpExecutionState {
  kGroundHold,
  kJumpReady,
  kJumpCommitted,
  kInFlight,
  kLandedHold,
};

struct TimeExecutionCursor {
  DurationNanoseconds offset;
  std::optional<SegmentId> segment_id;
};

struct JumpExecutionCursor {
  JumpExecutionState jump_state;
  std::optional<JumpBoundaryId> boundary_id;
};

using ExecutionCursor =
    std::variant<TimeExecutionCursor, JumpExecutionCursor>;

struct TimeCommitBoundary {
  DurationNanoseconds committed_until_offset;
};

struct JumpCommitBoundary {
  JumpBoundaryId boundary_id;
  bool locked;
};

using CommitBoundary =
    std::variant<TimeCommitBoundary, JumpCommitBoundary>;

enum class ControllerStatus {
  kReady,
  kExecuting,
  kHolding,
  kCommitted,
  kInFlight,
  kFault,
};

struct PreviousExecutionContext {
  ContentRef active_bundle_ref;
  std::string active_bundle_handle;
  CommitBoundary commit_boundary;
  ExecutionCursor execution_cursor;
  ControllerStatus controller_status;
  ContentRef source_map_snapshot_ref;
  ContentRef source_capability_ref;
};
```

### 5.6 `PlanningRequest`

```cpp
struct PlanningRequest {
  RequestId request_id;
  ClockStamp request_time;
  ClockStamp state_time;
  PlatformType platform_type;
  FrameId frame_id;
  PlatformState current_state;
  GoalRegion goal;
  std::shared_ptr<const ImmutableMapSnapshot> map_snapshot;
  std::shared_ptr<const SafetyCapabilityProfile> safety_capability;
  std::shared_ptr<const PlannerAlgorithmConfig> algorithm_config;
  ResolvedCapabilityBindings capability_bindings;
  std::optional<PreviousExecutionContext> previous_execution_context;
  std::optional<LearnedCostSnapshotBinding> learned_cost_snapshot;
};
```

`learned_cost_snapshot == std::nullopt` 是合法解析基线。请求不得包含 runtime deadline 或 BenchmarkProfile。
飞跃 `SafetyCapabilityProfile` 必须携带机器 schema 定义的
`landing_terrain_thresholds`；缺失坡度、粗糙度、单平面残差、顶/侧净空或
最小着陆域面积任一硬阈值时，请求校验失败，不能使用默认值补齐。

`capability_bindings` 是 C++ 运行时字段，不出现在 wire request。Hopper 请求必须
解析出 gravity、error model、actuator/impulse profile 和 body rotation envelope；
可选姿态表若存在也必须在请求进入规划前解析并证明只能收紧基础包络。

```cpp
struct ReferenceActivationContext {
  const PlanningRequest& request;
  const ContractObjectRegistry& registry;
};
```

任何新 bundle 的解码、终验和激活都必须携带该上下文；无上下文的
`Validate(ReferenceBundle)` 只能做局部结构检查，不能授予执行权。

---

## 6. 几何路径与时序

### 6.1 几何路径联合

```cpp
struct PoseXyzYaw {
  double x_m;
  double y_m;
  double z_m;
  double yaw_unwrapped_rad;
};

struct ClampedCubicBSplinePath {
  static constexpr std::uint32_t kDegree = 3;
  static constexpr double kParameterStart = 0.0;
  static constexpr double kParameterEnd = 1.0;
  std::vector<double> knots;
  std::vector<PoseXyzYaw> control_points;
};

struct ValidatedPrimitive {
  PrimitiveId primitive_id;
  PrimitiveId capability_primitive_id;
  PrimitiveKind primitive_kind;
  PoseXyzYaw start_pose;
  PoseXyzYaw end_pose;
  DurationNanoseconds nominal_duration;
  ContentRef validation_ref;
};

struct ValidatedPrimitiveChain {
  std::vector<ValidatedPrimitive> primitives;
};

using GeometricPath =
    std::variant<ClampedCubicBSplinePath, ValidatedPrimitiveChain>;
```

该联合同时表达：

- 经过连续优化和验证的三次 B 样条。
- 连续处理失败后完整回退的认证运动原语链。

原语实例由 `capability_primitive_id` 指向已固定的能力 profile，`validation_ref`
固定其扫掠/几何证书。控制器只能使用与 bundle 的
`source_safety_capability_ref` 和各 `validation_ref` hash 一致的本地内容。

### 6.2 分段三次标量函数

```cpp
struct CubicPolynomialSegment {
  DurationNanoseconds start_offset;
  DurationNanoseconds end_offset;
  std::array<double, 4> coefficients;
};

struct PiecewiseCubicScalarTrajectory {
  std::string value_semantics;
  std::vector<CubicPolynomialSegment> segments;
};

struct MonotoneTimeScaling {
  std::vector<CubicPolynomialSegment> segments;
};
```

对段内秒数 \(\tau\)：

\[
f(\tau)=c_0+c_1\tau+c_2\tau^2+c_3\tau^3.
\]

`PiecewiseCubicScalarTrajectory` 用于：

- `SpinSegment` 的 `yaw(t)`。
- 需要传输的其他一维时间函数。

`MonotoneTimeScaling` 专用于 `s(t)`。codec 固定 wire 的
`representation = "piecewise_cubic"`、`value_semantics = "path_parameter_s"`、
`monotonicity = "nondecreasing"`、`start_value = 0` 和 `end_value = 1`。
semantic validator 必须检查段连续、时间严格递增、定义域完整，并证明 `s(t)` 全程单调不减。

---

## 7. 平台参考联合

```cpp
struct SafeStopAnchor {
  Identifier anchor_id;
  PoseXyzYaw pose;
  double target_linear_velocity_mps;
  double target_yaw_rate_radps;
  ContentRef terrain_certification_ref;
};

struct Interval {
  double lower;
  double upper;
};

struct BodyFrameVelocityEnvelope {
  Interval forward_mps;
  Interval lateral_mps;
  Interval vertical_mps;
  Interval yaw_rate_radps;
};

struct TerrainNormalEnvelope {
  double maximum_normal_deviation_rad;
  ContentRef source_terrain_certification_ref;
};

struct RollPitchDiagnosticEnvelope {
  Interval roll_rad;
  Interval pitch_rad;
};

using PlatformReference = std::variant<
    WheeledReference,
    LeggedBodyReference,
    HopperReference>;
```

### 7.1 轮式

```cpp
struct DerivedKinematicCaches {
  double consistency_tolerance;
  PiecewiseCubicScalarTrajectory signed_body_forward_speed_mps;
  PiecewiseCubicScalarTrajectory yaw_rate_radps;
};

struct DriveSegment {
  SegmentId segment_id;
  TimeInterval time_interval;
  DriveDirection direction;
  GeometricPath geometric_path;
  MonotoneTimeScaling time_scaling;
  std::optional<DerivedKinematicCaches> derived_caches;
};

struct SpinSegment {
  SegmentId segment_id;
  TimeInterval time_interval;
  Vec3 fixed_position_m;
  PiecewiseCubicScalarTrajectory unwrapped_yaw_rad;
};

using WheeledSegment = std::variant<DriveSegment, SpinSegment>;

struct WheeledReference {
  ReferenceId reference_id;
  Sha256Digest reference_hash;
  ClockStamp reference_time_origin;
  std::vector<WheeledSegment> segments;
  SafeStopAnchor safe_stop_anchor;
};
```

### 7.2 足式

```cpp
struct LeggedBodyReference {
  ReferenceId reference_id;
  Sha256Digest reference_hash;
  ReferencePointId reference_point_id;
  ClockStamp reference_time_origin;
  GeometricPath geometric_path;
  MonotoneTimeScaling time_scaling;
  BodyFrameVelocityEnvelope velocity_envelope;
  TerrainNormalEnvelope terrain_normal_envelope;
  RollPitchDiagnosticEnvelope roll_pitch_diagnostic_envelope;
  SafeStopAnchor safe_stop_anchor;
  static constexpr std::string_view kFeasibilityScope =
      "body_geometry_and_terrain_thresholds_only";
  static constexpr bool kFootstepFeasibilityGuaranteed = false;
};
```

`SafeStopAnchor` 的 wire 目标线速度和 yaw 角速度恒为 0；
`RollPitchDiagnosticEnvelope.authority` 恒为 `NON_AUTHORITATIVE_DIAGNOSTIC`。
schema 必须拒绝足端、步态、接触序列和接触力字段。

### 7.3 飞跃式

```cpp
struct NextLandingRegion {
  LandingRegionId region_id;
  FrameId frame_id;
  LandingPlane landing_plane;
  ConvexPolygonUv convex_polygon;
  CircularYawInterval allowed_yaw_interval;
  ContentRef terrain_certification_ref;
  double inward_safety_margin_m;
};

struct HopperKinematicState {
  Vec3 position_m;
  Quaternion orientation_body_to_frame;
  Vec3 linear_velocity_mps;
  Vec3 angular_velocity_radps;
};

struct GroundHoldAnchor {
  Identifier anchor_id;
  HopperKinematicState hold_state;
  HopperErrorBounds allowed_hold_state_error_set;
  ContentRef terrain_certification_ref;
};

struct JumpBoundary {
  enum class LockEvent { kJumpBoundaryLock };
  enum class BallisticTimeOrigin { kBallisticLaunchEvent };
  JumpBoundaryId boundary_id;
  LockEvent lock_event;
  HopperKinematicState nominal_launch_state;
  HopperErrorBounds allowed_launch_state_error_set;
  ContentRef gravity_model_ref;
  BallisticTimeOrigin ballistic_time_origin;
  DurationNanoseconds ballistic_flight_time;
  ContentRef actuator_or_impulse_profile_ref;
};

struct Vector3Bounds {
  Vec3 lower;
  Vec3 upper;
};

struct PredictedLandingFootprint {
  LandingPlane landing_plane;
  ConvexPolygonUv convex_center_landing_polygon;
  TimeInterval landing_time_window;
  Vector3Bounds landing_velocity_bounds;
  CircularYawInterval landing_yaw_interval;
  ContentRef source_error_model_ref;
  double outer_approximation_margin_m;
};

struct Halfspace3 {
  Vec3 normal;
  double offset_m;
};

struct ConvexPolytope3 {
  enum class Representation { kHalfspaceIntersection };
  Representation representation;
  std::vector<Halfspace3> halfspaces;
};

struct FlightTubeSection {
  TimeInterval time_interval;
  ConvexPolytope3 envelope;
};

struct CertifiedFlightTube {
  FrameId frame_id;
  std::vector<FlightTubeSection> sections;
  ContentRef source_map_snapshot_ref;
  ContentRef body_rotation_envelope_ref;
  ContentRef error_model_ref;
  double minimum_certified_clearance_m;
};

struct TargetAttitudeSet {
  Quaternion nominal_orientation_body_to_frame;
  RotationVectorBall orientation_error_set;
  CircularYawInterval allowed_yaw_interval;
};

struct AttitudeBoundary {
  enum class TranslationAuthority { kNone };
  RotationVectorBall initial_orientation_error_set;
  DeterministicVectorSet3 initial_angular_velocity_error_set_radps;
  TargetAttitudeSet target_attitude_set;
  DeterministicVectorSet3 landing_angular_velocity_bounds_radps;
  DurationNanoseconds settle_guard;
  TranslationAuthority center_of_mass_translation_authority;
  ContentRef certification_ref;
};

struct NominalAimPoint {
  enum class Authority { kNonAuthoritativeExplanatory };
  Authority authority;
  Vec3 position_m;
};

struct FutureRoutePreview {
  enum class Authority { kNonAuthoritativeMissionPreview };
  Authority authority;
  FutureViability future_viability;
  ReasonCode reason_code;
  std::vector<LandingRegionId> candidate_region_ids;
};

struct HopperReference {
  enum class TranslationModel {
    kPureBallisticNoInflightTranslationControl,
  };
  ReferenceId reference_id;
  Sha256Digest reference_hash;
  ClockStamp reference_time_origin;
  TranslationModel translation_model;
  GroundHoldAnchor ground_hold_anchor;
  NextLandingRegion next_landing_region;
  JumpBoundary jump_boundary;
  PredictedLandingFootprint predicted_landing_footprint;
  CertifiedFlightTube certified_flight_tube;
  AttitudeBoundary attitude_boundary;
  NominalAimPoint nominal_aim_point;
  ContentRef physical_certification_ref;
  std::optional<FutureRoutePreview> future_route_preview;
  static constexpr std::string_view kTranslationModel =
      "PURE_BALLISTIC_NO_INFLIGHT_TRANSLATION_CONTROL";
};
```

codec 还固定以下 wire 常量：`lock_event = "JUMP_BOUNDARY_LOCK"`、
`AttitudeBoundary.center_of_mass_translation_authority = "NONE"`、
`NominalAimPoint.authority = "NON_AUTHORITATIVE_EXPLANATORY"` 和
`FutureRoutePreview.authority = "NON_AUTHORITATIVE_MISSION_PREVIEW"`。它们不形成第二条可执行命令。

必须验证：

\[
\operatorname{PredictedLandingFootprint}
\oplus \operatorname{safe\_margin}
\subseteq \operatorname{NextLandingRegion}.
\]

着陆时间窗与线速度边界只有一个 canonical source：
`PredictedLandingFootprint.landing_time_window` 和
`PredictedLandingFootprint.landing_velocity_bounds`；`HopperReference` 顶层不重复。
圆周区间按 canonical CCW 集合语义额外验证：

\[
\operatorname{footprint.landing\_yaw}
\subseteq
\operatorname{attitude.target.allowed\_yaw}
\subseteq
\operatorname{region.allowed\_yaw}.
\]

任一 yaw 子集关系不成立时，codec 后的语义校验必须拒绝整个 reference。

---

## 8. `ReferenceBundle`

```cpp
struct RouteSkeletonContent {
  ReferenceId source_reference_id;
  Sha256Digest source_reference_hash;
  std::vector<PoseXyzYaw> waypoints;
  std::vector<Vec3> unresolved_tail;
};

struct TimeViewSelector { TimeInterval time_interval; };
struct SegmentViewSelector {
  std::uint32_t first_segment_index;
  std::uint32_t past_last_segment_index;
};
struct GroundHoldViewSelector { Identifier anchor_id; };
struct JumpViewSelector {
  JumpBoundaryId boundary_id;
  JumpViewScope scope;
};

using ReferenceViewSelector = std::variant<
    TimeViewSelector,
    SegmentViewSelector,
    GroundHoldViewSelector,
    JumpViewSelector>;

struct ReferenceViewContent {
  ReferenceViewRole role;
  ReferenceId source_reference_id;
  Sha256Digest source_reference_hash;
  ReferenceViewSelector selector;
};

using AllowedStateDeviation =
    std::variant<WheeledOrLeggedErrorBounds, HopperErrorBounds>;

struct ReferenceValidity {
  ClockStamp valid_from;
  std::optional<ClockStamp> valid_until;
  ContentRef required_map_snapshot_ref;
  ContentRef required_capability_ref;
  AllowedStateDeviation allowed_state_deviation;
  std::vector<InvalidationCondition> invalidation_conditions;
};

struct ValidationSummary {
  bool hard_constraints_passed;
  bool continuous_validation_passed;
  std::vector<ContentRef> certificate_refs;
  std::vector<std::string> warning_codes;
};

struct GenerationEvidence {
  Identifier selected_candidate_id;
  GenerationMode generation_mode;
  std::string termination_reason;
  std::vector<ContentRef> evidence_refs;
  std::optional<ContentRef> learned_cost_snapshot_ref;
};

template <class Payload>
struct InlineComponent {
  ComponentId component_id;
  Sha256Digest component_hash;
  Payload content;
};

struct ReferenceBundle {
  BundleId bundle_id;
  std::uint32_t bundle_revision;
  Sha256Digest bundle_hash;
  std::optional<BundleId> supersedes_bundle_id;
  RequestId source_request_id;
  ContentRef source_map_snapshot_ref;
  ContentRef source_safety_capability_ref;
  ContentRef source_algorithm_config_ref;
  PlatformType platform_type;
  PlatformReference platform_reference;
  InlineComponent<RouteSkeletonContent> route_skeleton;
  InlineComponent<ReferenceViewContent> committed_prefix;
  InlineComponent<ReferenceViewContent> preview;
  ReferenceValidity validity;
  ValidationSummary validation_summary;
  GenerationEvidence generation_evidence;
};
```

规则：

- wire 固定 `RouteSkeletonContent.authority = "NON_AUTHORITATIVE"`、
  `ValidationSummary.hard_constraints_passed = true` 和
  `continuous_validation_passed = true`。
- `bundle_id` 是唯一激活权威。
- `bundle_revision` 是实例修订，不是 schema 版本。
- 子组件内联，不能独立激活。
- 相同 `component_id` 必须具有相同 JCS/SHA-256 hash 和相同规范化内容。
- `committed_prefix` 与 `preview` 只能引用同一 `platform_reference` 的索引或参数区间。
- `GroundHoldViewSelector` 只允许用于 Hopper 的 committed prefix，且
  `anchor_id` 必须命中同一 reference 的 `ground_hold_anchor`；anchor 的线速度和
  角速度必须逐分量为 0。
- Hopper 在 `JUMP_READY` 时 committed prefix 选择 ground-hold anchor，preview
  使用 `JumpViewSelector(scope = NEXT_HOP)` 选择同一 reference 的唯一边界；
  锁定状态只更新 `PreviousExecutionContext.commit_boundary`，不改写 bundle。
- `source_map_snapshot_ref == validity.required_map_snapshot_ref`；
  `source_safety_capability_ref == validity.required_capability_ref`；
  `source_algorithm_config_ref` 等于激活上下文中请求的算法配置。
- `response.request_id == context.request.request_id ==
  bundle.source_request_id`，platform、frame 与 reference-time provenance 也与该
  请求一致。
- Hopper tube 的 `source_map_snapshot_ref` 等于 bundle 源地图；
  boundary 的 gravity ref 等于已解析能力 gravity ref；footprint 与 tube 的
  error-model ref 相同且等于本次固定 error-model binding；执行器/冲量和机体旋转
  包络 ref 也分别等于已解析能力绑定。
- `ballistic_flight_time > 0` 且位于能力的
  `[minimum_flight_time, maximum_flight_time]` 内。tube 必须以 launch event 为
  0 连续覆盖 `[0,T]`，landing window 必须包含名义冲击偏移 `T`。
- 请求的 Hopper 当前状态集合包含于 ground-hold 允许集合；anchor 名义位置/姿态
  与 launch boundary 连续，线/角速度跃迁和误差集合映射由同一个已解析
  actuator/impulse profile 认证。
- 发射、着陆、净空和姿态边界满足激活能力：名义及误差后的发射速度/冲量不超限，
  名义着陆速度和 footprint 速度边界满足着陆速度与向下横截性，settle guard
  不小于能力下限，起始/着陆角速度集合包含于任意轴包络，可选表只能收紧。
- terrain、physical、attitude 和 validation-summary certificate 必须解析为
  `kCertification` 对象；其 `CertificationProvenance` 的 map/capability/config
  和固定模型 input refs 必须与激活上下文一致。
- `generation_evidence` 是生成该参考的可追踪证据；它不等同本次 API 调用的 `call_diagnostics`。
- `learned_cost_usage = USED_BOUNDED_SOFT_COST` 时，
  `generation_evidence.learned_cost_snapshot_ref` 必须存在并逐字段等于请求绑定；
  `DISABLED` 或 `FELL_BACK_TO_ANALYTIC` 时该字段必须不存在。
- Runtime bundle 记录 SafetyCapability 与 AlgorithmConfig；不记录 BenchmarkProfile。

激活校验接口固定为：

```cpp
class SemanticValidator final {
 public:
  ValidationReport Validate(const PlanningRequest& request) const;
  ValidationReport Validate(const PlanningResponse& response) const;
  ValidationReport Validate(const ReferenceBundle& bundle) const;
  ValidationReport ValidateForActivation(
      const ReferenceBundle& bundle,
      const ReferenceActivationContext& context) const;
  ValidationReport ValidateForActivation(
      const PlanningResponse& response,
      const ReferenceActivationContext& context) const;
};
```

前三个 overload 只做局部合同校验；只有 `ValidateForActivation` 可以授予新 bundle
执行权。任何 join、模型解析、连续性或能力边界失败都使
`ACTIVATE_NEW_BUNDLE` 非法。

---

## 9. `PlanningResponse`

```cpp
struct SecondaryCosts {
  std::optional<double> energy;
  std::optional<double> nonfatal_risk;
  std::optional<double> smoothness;
};

struct CallDiagnostics {
  DurationNanoseconds api_latency;
  std::string termination_reason;
  std::optional<double> final_epsilon;
  std::optional<DurationNanoseconds> expected_execution_time;
  std::optional<SecondaryCosts> secondary_costs;
  std::uint64_t expanded_state_count;
  std::uint64_t reopened_state_count;
  std::uint64_t candidate_count;
  bool resource_limit_hit;
  LearnedCostUsage learned_cost_usage;
  std::optional<ContentRef> learned_cost_snapshot_ref;
  std::vector<std::string> message_codes;
};

struct PlanningResponse {
  RequestId request_id;
  ClockStamp response_time;
  PlanningOutcome planning_outcome;
  ExecutionDirective execution_directive;
  ReasonCode reason_code;
  std::optional<ReferenceBundle> new_reference_bundle;
  std::optional<ContentRef> active_bundle_ref;
  CallDiagnostics call_diagnostics;
};
```

允许组合：

| `planning_outcome` | 允许的 `execution_directive` | 新 bundle |
|---|---|---|
| `NEW_REFERENCE_READY` | `ACTIVATE_NEW_BUNDLE` | 必须有 |
| `SAFE_FRONTIER_REFERENCE_READY` | `ACTIVATE_NEW_BUNDLE` | 必须有 |
| `NO_KNOWN_SAFE_ROUTE` | `CONTINUE_ACTIVE_BUNDLE` / `HOLD_STATIONARY` / `CONTINUE_COMMITTED_JUMP` / `NO_SAFE_PLANNER_REFERENCE` | 必须无 |
| `GOAL_INFEASIBLE` | `CONTINUE_ACTIVE_BUNDLE` / `HOLD_STATIONARY` / `CONTINUE_COMMITTED_JUMP` / `NO_SAFE_PLANNER_REFERENCE` | 必须无 |
| `INVALID_REQUEST` | `CONTINUE_ACTIVE_BUNDLE` / `HOLD_STATIONARY` / `CONTINUE_COMMITTED_JUMP` / `NO_SAFE_PLANNER_REFERENCE` | 必须无 |
| `STALE_INPUT` | `CONTINUE_ACTIVE_BUNDLE` / `HOLD_STATIONARY` / `CONTINUE_COMMITTED_JUMP` / `NO_SAFE_PLANNER_REFERENCE` | 必须无 |
| `NUMERICAL_FAILURE` | `CONTINUE_ACTIVE_BUNDLE` / `HOLD_STATIONARY` / `CONTINUE_COMMITTED_JUMP` / `NO_SAFE_PLANNER_REFERENCE` | 必须无 |
| `RESOURCE_LIMIT` | `CONTINUE_ACTIVE_BUNDLE` / `HOLD_STATIONARY` / `CONTINUE_COMMITTED_JUMP` / `NO_SAFE_PLANNER_REFERENCE` | 必须无 |
| `ACTIVE_REFERENCE_INVALIDATED` | `HOLD_STATIONARY` / `CONTINUE_COMMITTED_JUMP` / `NO_SAFE_PLANNER_REFERENCE` | 必须无 |

额外前置条件：

- `CONTINUE_ACTIVE_BUNDLE`：活动承诺仍有效，并携带匹配的 `active_bundle_ref`。
- `HOLD_STATIONARY`：当前平台已经静止且当前位姿安全；不创建新 bundle。
- `CONTINUE_COMMITTED_JUMP`：飞跃状态为 `JUMP_COMMITTED` 或 `IN_FLIGHT`，并携带匹配的 `active_bundle_ref`。
- `NO_SAFE_PLANNER_REFERENCE`：不存在以上任何安全动作。
- `reason_code = SAFE_DEAD_END` 是任务诊断，可伴随一个物理安全的 READY 结果；它不是 outcome。
- `CallDiagnostics.learned_cost_usage` 描述本次调用：`DISABLED` 时 snapshot ref
  必须无；`USED_BOUNDED_SOFT_COST` 或 `FELL_BACK_TO_ANALYTIC` 时必须携带本次请求
  固定的 snapshot ref，即使本次没有生成 bundle。只有 `USED` 且激活新 bundle
  时，generation evidence 才复制同一 ref；fallback 生成的解析候选不复制。

---

## 10. Benchmark Schema

`BenchmarkProfile` 不是 `PlanningRequest` 的一部分。它只固定实验环境：

- 硬件、操作系统、编译器、优化级别和线程数。
- 地图集、场景混合和样本数。
- 三个平台共享且内容寻址的 `algorithm_config_ref`。
- 恰好一个 `WHEELED`、一个 `LEGGED` 和一个 `HOPPER` suite；每个 suite
  携带对应平台的 `safety_capability_ref`。
- 预热与冷启动规则。
- API 计时边界。
- `p95_latency_target_ns = "1000000000"`，且它是严格上界。

`BenchmarkReport` 必须记录：

- 顶层共享的 `algorithm_config_ref` 与 `benchmark_profile_ref`。
- 恰好一个 `WHEELED`、一个 `LEGGED` 和一个 `HOPPER` result；每个 result
  携带对应平台的 `safety_capability_ref`。
- 样本数。
- mean、median、P95、P99、最大值。
- 冷启动结果。
- 模块耗时分解。
- 每个平台的 `p95_latency_target_ns` 与 `p95_latency_target_met`。

BenchmarkProfile 和墙钟观测不得传入 `PlannerV3::Plan`，也不得改变候选、状态或降级行为。

---

## 11. 规范化、hash 与兼容性

### 11.1 规范化

所有 `content_hash` 和 `component_hash`：

1. 对对应 JSON 值应用 RFC 8785 JCS。
2. 对规范化 UTF-8 字节计算 SHA-256。
3. 以 64 个小写十六进制字符传输。

hash 输入不包含承载该 hash 的字段本身。

### 11.2 向后兼容

同一消息的 `.../v1` 内：

- 不得删除或改名字段。
- 不得改变字段单位或语义。
- 不得扩大枚举后假设旧消费者自动安全处理。
- 只可在明确可选、旧消费者可忽略且不影响安全的扩展容器中增加诊断字段。

硬合同发生不兼容变化时，新增消息修订；不得复用原 schema_version。

### 11.3 验证顺序

```text
UTF-8 / JSON / I-JSON
→ Draft 2020-12 结构校验
→ JSON codec 强类型解码
→ 数值与单位校验
→ 跨字段和 variant 校验
→ 几何、时序、状态机和包含关系校验
→ registry 版本/hash/句柄解析
→ 进入规划
```

任一步失败都不得进入搜索或生成可激活 bundle。
