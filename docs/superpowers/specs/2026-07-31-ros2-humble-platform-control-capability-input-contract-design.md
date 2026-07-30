# ROS 2 Humble 平台控制单位能力输入合同

日期：2026-07-31

## 1. 目的

本文定义其他负责平台控制的单位必须向本项目提供哪些静态平台能力数据，
使本项目能够：

1. 维护平台相关的地图约束；
2. 在地图上进行自主探索决策；
3. 调用 Path Planner v3 生成时间参数化参考轨迹；
4. 向平台控制系统交付可执行的规划结果和安全诊断。

本文描述的是“外部单位向本项目交付的能力资料”，不是要求外部单位直接编写
v3 `SafetyCapabilityProfile`。

本项目负责把这些资料校验、施加项目安全策略并转换成内部 v3 配置，包括：

- v3 `ContentRef`、revision 和内容哈希；
- 项目设置的规划误差界；
- 搜索、优化和时间参数化算法配置；
- 内部不可变对象注册和引用绑定。

## 2. 与其他输入的边界

平台控制单位不需要重复提供已经由其他合同负责的输入。

| 内容 | 责任方 | 是否属于本文 |
|---|---|---:|
| 机体几何和机械运动极限 | 平台控制/平台集成单位 | 是 |
| 可稳定执行的运动形式 | 平台控制单位 | 是 |
| 时间参数化轨迹接收约定 | 平台控制单位 | 是 |
| 当前位姿、速度和协方差 | 定位系统 | 否 |
| 定位状态有效性 | 定位系统 | 否 |
| 规划误差界 | 本项目根据状态和配置计算 | 否 |
| 全局图和局部图 | 建图/融合系统 | 否 |
| 地图 `confidence` | 本项目根据地图质量信息计算 | 否 |
| 传感器原始数据和 SLAM | 传感器、建图和融合系统 | 否 |
| 自主探索观察范围模型 | 传感器集成或任务系统 | 否 |
| 任务目标、禁区和价值区域 | 任务系统 | 否 |
| A* 分辨率、权重和搜索预算 | 本项目 | 否 |
| 月面重力模型 | 环境模型 | 否 |

平台定位与运动状态继续由以下接口提供：

- `/localization/odometry`：`nav_msgs/msg/Odometry`；
- `/localization/status`：定位状态；
- TF：`map → odom → base_link`。

## 3. 交付方式

平台能力是低频静态资料，不要求平台控制单位发布高频 Topic。

建议交付一个版本化能力资料包：

```text
platform-capability-source/
├── platform_capability.yaml
├── geometry/
│   └── collision_envelope.*
├── motion/
│   ├── motion_model.*
│   ├── certified_primitives.*
│   └── swept_geometry.*
├── interface/
│   └── trajectory_acceptance.md
└── evidence/
    └── parameter_sources.md
```

外部资料包可以使用 YAML、JSON 和受控文档；本项目将其转换成版本化 v3 JSON。

机器可读数值统一使用：

- 长度：m；
- 速度：m/s；
- 加速度：m/s²；
- 角度：rad；
- 角速度：rad/s；
- 角加速度：rad/s²；
- 时间：s 或整数 ns；
- 冲量：N·s；
- 面积：m²。

## 4. 所有平台必须提供的公共内容

### 4.1 平台标识

| 字段 | 格式 | 定义 |
|---|---|---|
| `platform_id` | 非空字符串 | 平台型号或实例的稳定标识 |
| `platform_type` | 枚举 | `WHEELED`、`LEGGED` 或 `HOPPER` |
| `capability_version` | 字符串 | 本次能力资料版本 |
| `base_frame_id` | 字符串 | 机体几何和运动方向的参考坐标系，默认 `base_link` |

能力版本发生以下任一变化时必须升级：

- 碰撞包络变化；
- 速度或加速度限制变化；
- 支持的运动形式变化；
- 底层控制器升级导致跟踪能力变化；
- 平台载荷变化导致安全能力变化。

### 4.2 机体碰撞包络

平台控制单位必须提供保守覆盖平台本体的碰撞包络。

轮式平台提供：

```yaml
collision_envelope:
  type: EXTRUDED_CONVEX_POLYGON
  vertices_xy_m:
    - [x0, y0]
    - [x1, y1]
    - [x2, y2]
  minimum_z_m: <required>
  maximum_z_m: <required>
```

约束：

- 顶点定义于 `base_link`；
- 顶点按逆时针排列；
- 多边形必须严格凸；
- `minimum_z_m <= maximum_z_m`。

足式和飞跃式平台提供三维凸包：

```yaml
collision_envelope:
  type: CONVEX_POLYTOPE
  halfspaces:
    - normal: [nx, ny, nz]
      offset_m: <required>
```

每个半空间满足：

```text
normal · point <= offset_m
```

外部单位应说明包络是否包含机械臂、天线、支架等附属结构。

### 4.3 可执行运动形式

平台控制单位必须声明底层控制系统能够稳定跟踪的运动形式，而不是只提供机械极限。

每个运动形式至少包含：

```yaml
motion_kind: <required>
relative_end_pose:
  position_m: [x, y, z]
  yaw_rad: <required>
nominal_duration_s: <required>
tracking_validation:
  source: <document-or-test-id>
  status: CERTIFIED | TESTED | ESTIMATED
```

扫掠几何可以采用两种方式：

1. 平台控制单位直接提供并确认；
2. 本项目根据碰撞包络和运动轨迹生成，平台控制单位审核确认。

未确认的运动形式不得加入部署用 v3 运动原语目录。

### 4.4 轨迹接收约定

v3 输出的是：

```text
几何路径 P(s) + 单调时间缩放 s(t)
```

平台控制单位必须提供：

| 内容 | 定义 |
|---|---|
| 接收的轨迹表示 | 能否接收连续时间轨迹，或需要本项目离散采样 |
| 接收坐标系 | `map`、`odom` 或其他约定 |
| 时间基准 | ROS time、steady time 或任务时钟 |
| 支持的运动模式 | 前进、倒退、旋转、横移或飞跃 |
| 停止要求 | 轨迹段切换和末端停止条件 |
| 跟踪允许偏差 | 控制单位能够稳定保证的轨迹跟踪范围 |
| 拒绝条件 | 控制器无法接受轨迹时的错误码和反馈方式 |

跟踪允许偏差是控制系统性能资料，不等同于定位误差界。本项目仍自行设置规划误差界。

### 4.5 参数依据

每项能力必须标明来源：

```yaml
source:
  source_kind: CERTIFIED | TESTED | MANUFACTURER_SPEC | ESTIMATED
  document_id: <required>
  revision: <required>
  description: <required>
```

部署配置不得使用没有标识来源的宽松默认值。

## 5. 轮式平台交付内容

当前项目轮式规划固定采用滑移转向语义，不支持 Ackermann 或全向轮声明。

### 5.1 必须提供的几何和地形能力

| 内容 | 格式/单位 | 定义 |
|---|---|---|
| 机体凸包和高度范围 | m | 平台本体碰撞包络 |
| `minimum_clearance_m` | m，非负 | 平台要求保持的最小机械净空 |
| `maximum_slope_rad` | rad，`[0, π/2]` | 可规划通过的最大地形坡度 |
| `maximum_obstacle_height_m` | m，非负 | 可跨越的最大离散障碍高度 |
| `maximum_drive_curvature_per_m` | `1/m`，正数 | 非原地旋转行驶的最大曲率 |

### 5.2 必须提供的运动硬极限

| 内容 | 单位 |
|---|---:|
| 最大前进速度 | m/s |
| 最大倒退速度 | m/s |
| 最大原地旋转角速度 | rad/s |
| 最大前进加速度 | m/s² |
| 最大制动减速度 | m/s² |
| 最大偏航角加速度 | rad/s² |
| 最大横向加速度 | m/s² |

以上均为正数幅值。

### 5.3 必须确认的运动形式

当前 v3 轮式规划至少需要：

- 前进直线；
- 前进左、右弧线；
- 倒退直线；
- 倒退左、右弧线；
- 顺时针原地旋转；
- 逆时针原地旋转；
- 停止和运动模式切换。

如果底层控制系统不支持其中某项，平台控制单位必须明确提出；本项目不得通过伪造原语绕过。

## 6. 足式平台交付内容

当前 v3 足式规划只保证机体几何和地形阈值，不保证足端落脚点可行。

### 6.1 必须提供的机体能力

| 内容 | 格式/单位 | 定义 |
|---|---|---|
| 固定机体参考点 | 标识和定义 | 规划轨迹所表示的机体点 |
| 三维机体凸包 | 半空间 | 机体级碰撞包络 |
| 最大地形坡度 | rad | 可规划通过的最大坡度 |
| 最大地形粗糙度 | m | 对应本项目粗糙度定义的上限 |
| 最大相邻高程突变 | m | 机体级地形模型允许的台阶高度 |
| 最小机体净空 | m | 机体包络与地形、障碍的最小距离 |
| 最小、最大机体高度 | m | 参考点相对局部地面的高度范围 |

`minimum_confidence` 是本项目地图安全策略，不要求平台控制单位提供。

`maximum_gap_width_m` 当前不要求提供。现有地图适配尚未计算真实沟壑宽度，
部署配置应采用项目的失败关闭设置，不得声称跨沟能力。

### 6.2 必须提供的机体运动极限

平台控制单位必须提供：

- 前进/后退速度区间；
- 左右横向速度区间；
- 机体上下调整速度区间；
- 偏航角速度区间；
- 最大机体线加速度；
- 最大偏航角加速度。

### 6.3 必须确认的机体运动形式

至少包括：

- 前进；
- 后退；
- 向左横移；
- 向右横移；
- 顺时针原地旋转；
- 逆时针原地旋转。

平台控制单位还必须书面确认：

> 底层是否具有步态与足端规划能力，能够接收机体参考轨迹并生成真实足端动作。

如果没有该能力，本项目的足式机体路径不能直接交给执行器。

## 7. 飞跃式平台交付内容

当前 v3 飞跃式规划固定为：

```text
PURE_BALLISTIC_NO_INFLIGHT_TRANSLATION_CONTROL
```

即飞行过程中不具有质心平移控制权。

### 7.1 必须提供的着陆能力

| 内容 | 单位 |
|---|---:|
| 三维机体凸包 | m |
| 最大着陆坡度 | rad |
| 最大着陆面粗糙度 | m |
| 最大着陆面拟合残差 | m |
| 最小顶部净空 | m |
| 最小侧向净空 | m |
| 最小安全着陆区域面积 | m² |

### 7.2 必须提供的发射与着陆极限

| 内容 | 单位 |
|---|---:|
| 最大发射速度 | m/s |
| 最大发射冲量 | N·s |
| 最短飞行时间 | s |
| 最长飞行时间 | s |
| 最大允许着陆速度 | m/s |
| 最小向下碰撞速度约束 | m/s |
| 最小着陆净空 | m |

### 7.3 必须提供的姿态能力

- 最大任意轴角速度；
- 最大任意轴角加速度；
- 发射前最大初始角速度；
- 着陆后最小稳定等待时间；
- 发射执行器或冲量曲线模型；
- 发射与着陆状态切换条件。

重力模型由环境模型维护，不要求平台控制单位提供。

## 8. 本项目负责生成的 v3 内容

平台控制单位提供的是原始能力事实和证据。本项目负责生成：

| v3 内容 | 生成责任 |
|---|---|
| `content_ref`、revision、JCS SHA-256 | 本项目 |
| `motion_model_ref` | 根据控制单位运动模型资料注册 |
| `analytic_cost_model_ref` | 本项目代价模型 |
| `gravity_model_ref` | 本项目环境模型 |
| `certified_state_error_bounds` | 本项目配置的允许误差上界 |
| `motion_primitives` | 根据已确认运动形式构造并校验 |
| 扫掠几何引用 | 本项目生成、控制单位确认，或直接使用控制单位认证结果 |
| v3 schema 和语义校验 | 本项目 |

运行时当前误差界仍由本项目从定位输入计算，并写入
`PlanningRequest.current_state.error_bounds`。

## 9. 验收规则

平台能力资料只有满足以下条件才可用于路径规划：

1. 平台标识、版本和坐标系完整；
2. 碰撞包络闭合、凸且使用 SI 单位；
3. 所有硬限制具有明确来源；
4. 运动原语未超过速度和加速度限制；
5. 平台控制单位确认相应运动形式可稳定跟踪；
6. 轨迹接收、时间基准、停止和拒绝接口已经约定；
7. 本项目生成的 v3 配置通过 schema、内容哈希和语义校验；
8. 示例、估算或未认证参数不得被标记为部署能力；
9. 任一必选能力缺失时失败关闭，不使用宽松默认值。

## 10. 推荐外部交付模板

```yaml
schema_version: platform-control-capability-source/v1

platform:
  platform_id: <required>
  platform_type: WHEELED | LEGGED | HOPPER
  capability_version: <required>
  base_frame_id: base_link

collision_envelope:
  type: <required>
  data: <required>

platform_specific_capability:
  geometry_and_terrain_limits: <required>
  velocity_limits: <required>
  acceleration_limits: <required>
  supported_motion_kinds: <required>
  certified_motion_descriptions: <required>

trajectory_acceptance:
  accepted_representation: <required>
  accepted_frame_id: <required>
  clock_id: <required>
  stop_condition: <required>
  rejection_interface: <required>

provenance:
  source_kind: CERTIFIED | TESTED | MANUFACTURER_SPEC | ESTIMATED
  document_id: <required>
  revision: <required>
```

该模板是外部交付格式，不是 v3 `SafetyCapabilityProfile`。转换和哈希由本项目完成。
