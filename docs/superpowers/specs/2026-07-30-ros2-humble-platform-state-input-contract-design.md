# ROS 2 Humble 平台定位与运动状态外部输入合同设计

日期：2026-07-30
状态：待用户审阅
适用范围：`lunar-path-planning` 在 ROS 2 Humble 中消费的平台定位与运动状态

## 1. 目标与边界

本合同只定义探索决策、Python A*、Hybrid A* 和 C++20 多平台规划所需的最小定位与运动状态输入。

外部定位、SLAM 或状态估计系统负责：

- 处理 IMU、GNSS、轮速、LiDAR、相机等原始输入；
- 完成定位、SLAM、状态融合和重定位；
- 发布平台名义位姿、速度、covariance、定位状态和 TF。

本项目负责：

- 校验外部状态的格式、时间、坐标系和数值；
- 把平台状态转换到全局 `map` 或局部 `odom`；
- 根据 covariance、状态时延、当前速度和平台配置计算规划误差界；
- 生成探索、A*、Hybrid A* 和 v3 所需的起始状态；
- 在定位状态或输入合同不满足时拒绝生成新路线。

本项目不执行定位、SLAM 或多传感器融合，不消费定位所依赖的原始传感器数据。

## 2. 最小外部输入集合

最终只接收：

```text
/localization/odometry
  nav_msgs/msg/Odometry

/localization/status
  lunar_navigation_msgs/msg/LocalizationStatus

/tf
  map -> odom -> base_link
```

不增加独立 Pose、Twist 或误差界 topic。

## 3. 平台状态

### 3.1 Topic

```text
Topic: /localization/odometry
Type:  nav_msgs/msg/Odometry
```

`Odometry` 是唯一的平台位姿与速度输入。

### 3.2 字段合同

| 字段 | 格式/单位 | 合同 |
|---|---|---|
| `header.stamp` | ROS Time | 状态实际对应的融合时刻，不能使用发送完成时刻代替 |
| `header.frame_id` | string | 固定为 `odom` |
| `child_frame_id` | string | 固定为 `base_link` |
| `pose.pose.position` | `x,y,z`，m | 平台参考点在 `odom` 中的位置 |
| `pose.pose.orientation` | quaternion | `base_link` 在 `odom` 中的单位四元数 |
| `pose.covariance` | `float64[36]` | `[x,y,z,roll,pitch,yaw]` 的行主序 `6x6` covariance |
| `twist.twist.linear` | `x,y,z`，m/s | 在 `base_link` 中表达的线速度 |
| `twist.twist.angular` | `x,y,z`，rad/s | 在 `base_link` 中表达的角速度 |
| `twist.covariance` | `float64[36]` | 线速度和角速度的行主序 `6x6` covariance |

不允许：

- `header.frame_id` 或 `child_frame_id` 为空；
- 使用 `map` 作为 Odometry 的 `header.frame_id`；
- 在 pose、twist 或 covariance 中写入 NaN 或无穷；
- 使用零范数或未归一化四元数；
- 使用负 covariance 主对角元素；
- 用命令速度代替状态估计速度。

### 3.3 算法取用

| 模块 | 从 Odometry 取用的字段 |
|---|---|
| 探索决策 | `x,y,yaw` |
| Python A* v1 | `x,y` |
| Hybrid A* | `x,y,yaw` |
| v3 轮式 | 位置、yaw、线速度、`angular.z` |
| v3 足式 | 机体位置、yaw、线速度、`angular.z` |
| v3 飞跃式 | 完整位置、四元数、线速度、角速度 |
| 误差界计算 | pose/twist covariance、速度、状态年龄 |

轮式和足式规划可以忽略 roll、pitch 以及不适用的速度分量，但仍统一消费标准 Odometry。飞跃式规划使用完整 6DoF 状态。

### 3.4 QoS 与频率

```text
推荐频率:    20–50 Hz
reliability: reliable
durability:  volatile
history:     keep_last
depth:       5
```

规划节点只使用通过校验的最新状态，不按队列顺序补处理旧状态。

## 4. 定位状态

### 4.1 Topic

```text
Topic: /localization/status
Type:  lunar_navigation_msgs/msg/LocalizationStatus
```

### 4.2 最小消息定义

```text
std_msgs/Header header

uint8 UNKNOWN=0
uint8 VALID=1
uint8 DEGRADED=2
uint8 INVALID=3
uint8 RELOCALIZING=4

uint8 status
```

不加入字符串说明、source ID、传感器健康、定位残差或误差界。定位系统的详细诊断应通过独立诊断通道提供，不进入规划必选合同。

`header.stamp` 表示该状态判断对应的时刻，`header.frame_id` 固定为 `odom`。它不要求与 Odometry 时间戳完全相同，但两者时间差必须小于配置的状态同步门槛。

### 4.3 状态含义

| 状态 | 外部定义 | 本项目行为 |
|---|---|---|
| `UNKNOWN` | 未知、未初始化或尚未形成可用判断 | 不生成新路线 |
| `VALID` | 定位正常可用 | 正常校验、计算误差界并规划 |
| `DEGRADED` | 定位仍可用，但质量下降 | 继续计算误差界；超过最大门槛时拒绝规划 |
| `INVALID` | 当前定位结果不可用 | 不生成新路线 |
| `RELOCALIZING` | 正在重定位，可能发生全局位姿跳变 | 暂停新规划，等待状态和 TF 稳定 |

状态必须由外部定位、SLAM 或其配套健康监控节点给出。本项目不能根据单条 Odometry 反推出定位系统是否真正收敛或正在重定位。

### 4.4 QoS

```text
发布方式:    状态变化立即发布，并低频周期重发
reliability: reliable
durability:  transient_local
history:     keep_last
depth:       1
```

`transient_local` 保证后启动的规划节点能够立即获得最近的定位状态。

## 5. 坐标变换

### 5.1 TF 树

```text
map -> odom -> base_link
```

外部定位或 SLAM 系统负责：

- 发布 `map -> odom`；
- 发布或协调唯一的 `odom -> base_link`；
- 保证 TF 时间覆盖 Odometry 和地图快照时间。

### 5.2 使用规则

- 全局地图位于 `map`，平台状态通过 `map -> odom` 转换到 `map`；
- 局部地图位于 `odom`，平台状态直接在 `odom` 中使用；
- 必须查询 Odometry 或地图 `header.stamp` 对应时刻的 TF；
- 禁止用当前最新 TF 转换历史状态；
- `map -> odom` 允许因全局校正发生离散变化；
- `odom -> base_link` 必须短期连续；
- 同一个 TF 边只能有一个权威发布者；
- Odometry pose 与相同时刻 `odom -> base_link` 必须在配置容差内一致。

本项目不消费原始传感器数据，因此本合同不要求 `base_link -> lidar/camera/imu` 等传感器 TF。

## 6. 规划误差界

### 6.1 所有权

外部不发布误差界 topic。规划误差界由本项目根据以下输入计算：

- Odometry pose covariance；
- Odometry twist covariance；
- 当前线速度和角速度；
- 状态年龄；
- TF 容差；
- 平台配置中的最小和最大允许误差界。

本项目计算的是规划采用的保守工程边界。除非配置 profile 有独立验证证据，否则不能宣称它是定位系统的严格最坏情况保证。

### 6.2 配置

平台能力配置至少包含：

```yaml
localization_error_bound:
  source_kind: covariance_scaled_engineering_bound/v1
  covariance_sigma_scale: 3.0

  minimum_position_bound_m:
    x: 0.05
    y: 0.05
    z: 0.05
  minimum_yaw_bound_rad: 0.035

  tf_position_tolerance_m: 0.01
  tf_yaw_tolerance_rad: 0.01

  maximum_position_bound_m:
    x: 0.50
    y: 0.50
    z: 0.50
  maximum_yaw_bound_rad: 0.35
```

飞跃式平台还必须配置姿态旋转边界和三轴角速度边界的最小值、最大值及 covariance 映射规则。

### 6.3 计算

逐轴位置边界：

```text
position_bound_i =
  max(
    configured_minimum_i,
    k_sigma * sqrt(pose_variance_i)
      + abs(linear_velocity_i) * state_age
      + tf_position_tolerance
  )
```

yaw 边界：

```text
yaw_bound =
  max(
    configured_minimum_yaw,
    k_sigma * sqrt(yaw_variance)
      + abs(yaw_rate) * state_age
      + tf_yaw_tolerance
  )
```

线速度、yaw rate 和飞跃式角速度边界使用对应 twist covariance 与配置最小值计算。飞跃式姿态边界使用姿态 covariance 形成保守旋转向量球。

计算结果超过配置最大值时必须拒绝生成新路线。禁止把超限结果截断到最大值后继续规划。

### 6.4 使用

| 模块 | 使用方式 |
|---|---|
| 探索决策 | 判断当前状态是否足以支持新目标选择 |
| Python A* v1 | 可选扩大起点安全检查范围，不改变默认 A* 搜索语义 |
| Hybrid A* | 扩大 footprint 与起始姿态安全检查范围 |
| C++ v3 | 写入 `current_state.error_bounds` |

规划 footprint 至少考虑：

```text
planning_footprint =
  physical_footprint
  + localization_error_bound
  + safety_margin
```

当误差界来源是 covariance 缩放或工程假设时，输出必须保留来源标记，不得描述为严格确定性安全保证。未来若连接 executor，v3 严格激活应要求经过验证的误差界 profile；当前主线仍不连接 executor。

## 7. 输入可用性

本项目内部状态可用性由以下条件共同决定：

```text
external localization status
AND Odometry freshness
AND TF availability
AND numeric validation
AND covariance validation
AND computed error bound within configured maximum
```

只有全部通过时才允许生成新路线。

建议初始时间阈值：

| 检查 | 默认值 |
|---|---:|
| 局部规划 Odometry 最大年龄 | `200 ms` |
| v3 新参考生成 Odometry 最大年龄 | `200 ms` |
| 全局探索状态最大年龄 | `1 s` |
| 允许未来时间偏差 | `20 ms` |
| TF 等待上限 | `100 ms` |
| TF 外推 | 默认禁止 |

这些阈值必须配置化，不得写死在算法中。

## 8. 接收校验与失败行为

每次规划前至少校验：

1. Odometry 与 LocalizationStatus 均已接收；
2. status 为 `VALID` 或 `DEGRADED`；
3. 两条消息时间差未超过配置门槛；
4. Odometry 未过期且没有异常未来时间；
5. frame 固定为 `odom -> base_link`；
6. pose、twist 和 covariance 全部有限；
7. 四元数归一化；
8. covariance 对称、半正定，主对角元素非负；
9. 对应时刻 TF 可用；
10. Odometry pose 与 TF 一致；
11. 当前平台状态位于所用地图范围内；
12. 计算误差界未超过平台配置最大值。

失败时：

- 不生成新路线；
- 不覆盖最近一份有效状态缓存；
- 返回稳定失败原因；
- 不直接向平台发送控制或停止命令。

建议失败原因：

```text
localization_status_unusable
odometry_missing
odometry_stale
odometry_future_timestamp
odometry_invalid_numeric
odometry_invalid_covariance
odometry_invalid_quaternion
state_tf_unavailable
state_tf_inconsistent
state_map_time_skew_exceeded
state_outside_map
localization_bound_exceeds_capability
```

## 9. 明确排除的输入

本合同不接收：

- 独立 `geometry_msgs/msg/PoseStamped`；
- 独立 `geometry_msgs/msg/TwistStamped`；
- 外部误差界 topic；
- IMU 原始数据；
- GNSS/GPS 原始数据；
- 轮速编码器；
- LiDAR、相机、点云或深度图；
- SLAM 特征、pose graph 和回环信息；
- 定位滤波器创新残差；
- 车轮或关节状态；
- 加速度；
- 期望速度和控制命令；
- 控制器执行状态；
- 传感器外参 TF。

这些内容由外部定位、融合、SLAM 或平台控制系统处理。

## 10. 本项目内部产物

本项目基于最小外部输入生成：

```text
map-frame platform state
odom-frame platform state
state freshness
state usability
planning localization error bounds
planning start pose
planning start cell
```

这些是内部适配结果，不要求上游重复发布。

## 11. ROS 2 Humble 与当前代码参考

- `nav_msgs`：
  <https://docs.ros.org/en/humble/p/nav_msgs/>
- `tf2_ros`：
  <https://docs.ros.org/en/humble/p/tf2_ros/>
- v3 `PlanningRequest.current_state` 和误差界：
  `path-planner/schemas/v3/common.schema.json`
- v3 请求时刻、frame 和地图绑定：
  `path-planner/schemas/v3/planning-request.schema.json`

本合同不表示当前 Python 或 C++ 代码已经完成 ROS 2 封装。消息包、订阅适配、TF 同步、误差界计算和运行时校验需要后续实施计划。
