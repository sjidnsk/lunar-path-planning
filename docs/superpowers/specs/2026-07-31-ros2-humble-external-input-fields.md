# ROS 2 Humble 外部输入字段总表

## 1. 消息与载体类型

| Topic/载体 | 消息/格式类型 | 定义 |
|---|---|---|
| `/environment/map_global` | `grid_map_msgs/msg/GridMap` | 低频全局多通道地图，`frame_id=map` |
| `/environment/map_local` | `grid_map_msgs/msg/GridMap` | 高频局部多通道地图，`frame_id=odom` |
| `/localization/odometry` | `nav_msgs/msg/Odometry` | 平台位姿、速度及 covariance |
| `/localization/status` | `lunar_navigation_msgs/msg/LocalizationStatus` | 定位状态 |
| `/tf` | `tf2_msgs/msg/TFMessage` | `map -> odom -> base_link` 坐标变换 |
| `/mission/exploration_task` | `lunar_navigation_msgs/msg/ExplorationTask` | 区域探索任务完整状态快照 |
| `science_regions[]` | `lunar_navigation_msgs/msg/ScienceTargetRegion` | 科学目标区域 |
| 平台能力资料包 | `platform-control-capability-source/v1`，YAML/JSON/URDF | 平台控制单位提供的版本化静态能力资料和 URDF 几何来源 |

## 2. 全局图与局部图

### 消息类型

```text
grid_map_msgs/msg/GridMap
```

### GridMap 字段

| 字段 | 格式/范围 | 定义 |
|---|---|---|
| `header.stamp` | ROS Time，非零 | 地图快照对应的融合时刻 |
| `header.frame_id` | 全局图 `map`；局部图 `odom` | 地图坐标系 |
| `info.resolution` | `float64`，m/cell，`>0` | 单元边长 |
| `info.length_x` | `float64`，m，`>0` | x 方向覆盖长度 |
| `info.length_y` | `float64`，m，`>0` | y 方向覆盖长度 |
| `info.pose` | `geometry_msgs/msg/Pose` | 地图几何中心在 `header.frame_id` 中的位姿 |
| `layers` | `string[]`，名称唯一 | 通道名称，顺序与 `data` 一致 |
| `basic_layers` | `string[]` | 判断基础地图数据完整性所依赖的通道 |
| `data` | `std_msgs/msg/Float32MultiArray[]` | 各通道栅格数据 |
| `outer_start_index` | `uint16` | GridMap 循环缓冲区外层起始索引 |
| `inner_start_index` | `uint16` | GridMap 循环缓冲区内层起始索引 |

### 地图通道

| 字段 | 格式/范围 | 定义 |
|---|---|---|
| `elevation` | `float32`，m | 栅格中心相对统一高程基准面的高度 |
| `valid_mask` | `float32`，`0.0/1.0` | 栅格基础环境数据是否有效 |
| `obstacle` | `float32`，`[0,1]` | 物理障碍存在概率或归一化占据置信度 |
| `obstacle_height` | `float32`，m，`>=0` | 障碍物相对局部地面的高度 |
| `observation_age_s` | `float32`，s，`>=0` | 相对 `header.stamp` 距最后一次有效融合观测的时间 |
| `observation_quality` | `float32`，`[0,1]` | 最近一次参与地图更新的有效观测综合质量 |
| `elevation_variance` | `float32`，m²，`>=0` | 高程估计误差方差 |
| `obstacle_variance` | `float32`，`[0,0.25]` | 障碍概率估计不确定度 |
| `observation_count` | `float32` 中的整数，`[0,65535]` | 累计有效融合观测次数，超过上限后饱和 |
| `forbidden` | `float32`，`0.0/1.0` | 任务或安全系统指定的禁入区域 |

## 3. 平台定位与运动状态

### 消息类型

```text
Topic: /localization/odometry
Type:  nav_msgs/msg/Odometry
```

### Odometry 字段

| 字段 | 格式/范围 | 定义 |
|---|---|---|
| `header.stamp` | ROS Time，非零 | 状态实际对应的融合时刻 |
| `header.frame_id` | string，固定 `odom` | 位姿参考坐标系 |
| `child_frame_id` | string，固定 `base_link` | 平台机体坐标系 |
| `pose.pose.position` | `geometry_msgs/msg/Point`，m | 平台参考点在 `odom` 中的位置 |
| `pose.pose.orientation` | 单位 quaternion | `base_link` 在 `odom` 中的姿态 |
| `pose.covariance` | `float64[36]`，有限值，主对角线 `>=0` | `[x,y,z,roll,pitch,yaw]` 行主序 `6x6` covariance |
| `twist.twist.linear` | `geometry_msgs/msg/Vector3`，m/s | 在 `base_link` 中表达的线速度 |
| `twist.twist.angular` | `geometry_msgs/msg/Vector3`，rad/s | 在 `base_link` 中表达的角速度 |
| `twist.covariance` | `float64[36]`，有限值，主对角线 `>=0` | `[vx,vy,vz,wx,wy,wz]` 行主序 `6x6` covariance |

## 4. 定位状态

### 消息类型

```text
Topic: /localization/status
Type:  lunar_navigation_msgs/msg/LocalizationStatus
```

### LocalizationStatus 字段

| 字段 | 格式/范围 | 定义 |
|---|---|---|
| `header.stamp` | ROS Time，非零 | 定位状态判断对应的时刻 |
| `header.frame_id` | string，固定 `odom` | 定位状态坐标系 |
| `status` | `uint8`，`UNKNOWN/VALID/DEGRADED/INVALID/RELOCALIZING` | 当前定位有效性状态 |

### status 枚举

| 字段 | 格式/范围 | 定义 |
|---|---|---|
| `UNKNOWN` | `uint8=0` | 未知、未初始化或尚未形成可用判断 |
| `VALID` | `uint8=1` | 定位正常可用 |
| `DEGRADED` | `uint8=2` | 定位仍可用，但质量下降 |
| `INVALID` | `uint8=3` | 当前定位结果不可用 |
| `RELOCALIZING` | `uint8=4` | 正在重定位，可能发生全局位姿跳变 |

## 5. 坐标变换

### 消息类型

```text
Topic: /tf
Type:  tf2_msgs/msg/TFMessage
```

### TFMessage 字段

| 字段 | 格式/范围 | 定义 |
|---|---|---|
| `transforms[]` | `geometry_msgs/msg/TransformStamped[]` | 坐标变换数组 |
| `transforms[].header.stamp` | ROS Time，非零 | 变换对应的时刻 |
| `transforms[].header.frame_id` | `map` 或 `odom` | 父坐标系 |
| `transforms[].child_frame_id` | `odom` 或 `base_link` | 子坐标系 |
| `transforms[].transform.translation` | `geometry_msgs/msg/Vector3`，m | 子坐标系原点在父坐标系中的位置 |
| `transforms[].transform.rotation` | 单位 quaternion | 子坐标系在父坐标系中的姿态 |

## 6. 任务上下文

### 消息类型

```text
Topic: /mission/exploration_task
Type:  lunar_navigation_msgs/msg/ExplorationTask
```

### ExplorationTask 字段

| 字段 | 格式/范围 | 定义 |
|---|---|---|
| `header.stamp` | ROS Time，非零 | 任务快照开始生效的时间 |
| `header.frame_id` | string，固定 `map` | ROI 和科学目标区域坐标系 |
| `mission_id` | 非空 string | 探索任务全局唯一标识 |
| `revision` | `uint64`，从 `1` 开始 | 同一任务完整快照的严格递增版本 |
| `desired_state` | `uint8`，`ACTIVE/PAUSED/CANCELED` | 任务系统要求的任务状态 |
| `roi_min_x_m` | 有限 `float64`，m，`<roi_max_x_m` | 轴对齐矩形 ROI 最小 x |
| `roi_min_y_m` | 有限 `float64`，m，`<roi_max_y_m` | 轴对齐矩形 ROI 最小 y |
| `roi_max_x_m` | 有限 `float64`，m，`>roi_min_x_m` | 轴对齐矩形 ROI 最大 x |
| `roi_max_y_m` | 有限 `float64`，m，`>roi_min_y_m` | 轴对齐矩形 ROI 最大 y |
| `science_regions` | `ScienceTargetRegion[]`，`0..64` 项 | 软优先探索的科学目标区域 |

### desired_state 枚举

| 字段 | 格式/范围 | 定义 |
|---|---|---|
| `UNKNOWN` | `uint8=0` | 未指定状态，不作为合法运行输入 |
| `ACTIVE` | `uint8=1` | 激活任务 |
| `PAUSED` | `uint8=2` | 暂停生成新的探索目标和路线 |
| `CANCELED` | `uint8=3` | 取消任务 |

## 7. 科学目标区域

### 消息类型

```text
lunar_navigation_msgs/msg/ScienceTargetRegion
```

### ScienceTargetRegion 字段

| 字段 | 格式/范围 | 定义 |
|---|---|---|
| `region_id` | 当前任务内唯一的非空 string | 科学目标区域标识 |
| `objective_id` | 非空 string | 外部科学目标标识；允许多个区域使用同一目标标识 |
| `boundary` | `geometry_msgs/msg/Polygon`，`3..256` 个不同顶点 | `map` 坐标系中的简单非自交多边形 |
| `boundary.points[].x` | 有限 `float32`，m | 多边形顶点 x |
| `boundary.points[].y` | 有限 `float32`，m | 多边形顶点 y |
| `boundary.points[].z` | `float32=0.0` | 二维多边形 z |
| `priority` | 有限 `float32`，`(0,1]` | 只影响候选目标排序的软探索优先级 |

## 8. 平台能力资料包

### 载体类型

```text
Schema: platform-control-capability-source/v1
Format: YAML/JSON + URDF/mesh resources
```

### 公共字段

| 字段 | 格式/范围 | 定义 |
|---|---|---|
| `platform.platform_id` | 非空 string | 平台型号或实例的稳定标识 |
| `platform.platform_type` | `WHEELED/LEGGED/HOPPER` | 平台类型 |
| `platform.capability_version` | 非空 string | 能力资料版本 |
| `platform.base_frame_id` | 非空 string，默认 `base_link` | 几何和运动方向参考坐标系 |

### URDF 几何来源

| 字段/资源 | 格式/范围 | 定义 |
|---|---|---|
| `geometry_source.urdf_file` | 非空 package-relative path，URDF | 平台 `<collision>` 几何、坐标关系和关节限制来源 |
| URDF 引用的 mesh 文件 | 随资料包交付的本地 STL/DAE 等资源 | URDF `<collision>` 中引用的外部几何 |

### 可执行运动描述

| 字段 | 格式/范围 | 定义 |
|---|---|---|
| `motion_kind` | 平台类型对应枚举 | 底层控制系统能够稳定跟踪的运动形式 |
| `relative_end_pose.position_m` | 有限 `float64[3]`，m | 相对运动终点位置 |
| `relative_end_pose.yaw_rad` | 有限 `float64`，rad | 相对运动终点偏航角 |
| `nominal_duration_s` | 有限 `float64`，s，`>0` | 标称运动持续时间 |

## 9. 轮式平台能力字段

### 载体类型

```text
platform-control-capability-source/v1
platform.platform_type: WHEELED
```

| 字段 | 格式/范围 | 定义 |
|---|---|---|
| `minimum_clearance_m` | 有限 `float64`，m，`>=0` | 平台要求保持的最小机械净空 |
| `maximum_slope_rad` | 有限 `float64`，rad，`[0,π/2]` | 可规划通过的最大地形坡度 |
| `maximum_obstacle_height_m` | 有限 `float64`，m，`>=0` | 可跨越的最大离散障碍高度 |
| `maximum_drive_curvature_per_m` | 有限 `float64`，1/m，`>0` | 非原地旋转行驶的最大曲率 |
| `maximum_forward_speed_mps` | 有限 `float64`，m/s，`>0` | 最大前进速度幅值 |
| `maximum_reverse_speed_mps` | 有限 `float64`，m/s，`>0` | 最大倒退速度幅值 |
| `maximum_spin_rate_radps` | 有限 `float64`，rad/s，`>0` | 最大原地旋转角速度幅值 |
| `maximum_forward_acceleration_mps2` | 有限 `float64`，m/s²，`>0` | 最大前进加速度幅值 |
| `maximum_braking_deceleration_mps2` | 有限 `float64`，m/s²，`>0` | 最大制动减速度幅值 |
| `maximum_yaw_acceleration_radps2` | 有限 `float64`，rad/s²，`>0` | 最大偏航角加速度幅值 |
| `maximum_lateral_acceleration_mps2` | 有限 `float64`，m/s²，`>0` | 最大横向加速度幅值 |
| `supported_motion_kinds` | `FORWARD_LINE/FORWARD_ARC_LEFT/FORWARD_ARC_RIGHT/REVERSE_LINE/REVERSE_ARC_LEFT/REVERSE_ARC_RIGHT/SPIN_CW/SPIN_CCW/STOP_AND_SWITCH` | 必须确认可执行的滑移转向运动形式 |

## 10. 足式平台能力字段

### 载体类型

```text
platform-control-capability-source/v1
platform.platform_type: LEGGED
```

| 字段 | 格式/范围 | 定义 |
|---|---|---|
| `reference_point_id` | 非空 string | 机体参考点标识 |
| `reference_point_definition` | 非空 string | 规划轨迹所表示的固定机体点定义 |
| `maximum_slope_rad` | 有限 `float64`，rad，`[0,π/2]` | 可规划通过的最大地形坡度 |
| `maximum_roughness_m` | 有限 `float64`，m，`>=0` | 可规划通过的最大地形粗糙度 |
| `maximum_step_height_m` | 有限 `float64`，m，`>=0` | 机体级地形模型允许的最大相邻高程突变 |
| `minimum_body_clearance_m` | 有限 `float64`，m，`>=0` | 机体包络与地形或障碍的最小距离 |
| `minimum_body_height_m` | 有限 `float64`，m，`<=maximum_body_height_m` | 参考点相对局部地面的最小高度 |
| `maximum_body_height_m` | 有限 `float64`，m，`>=minimum_body_height_m` | 参考点相对局部地面的最大高度 |
| `forward_mps` | 有限下界/上界，m/s | 机体前进和后退速度区间 |
| `lateral_mps` | 有限下界/上界，m/s | 机体左右横移速度区间 |
| `vertical_mps` | 有限下界/上界，m/s | 机体上下调整速度区间 |
| `yaw_rate_radps` | 有限下界/上界，rad/s | 偏航角速度区间 |
| `linear_acceleration_mps2` | 有限 `float64`，m/s²，`>0` | 最大机体线加速度 |
| `yaw_acceleration_radps2` | 有限 `float64`，rad/s²，`>0` | 最大偏航角加速度 |
| `supported_motion_kinds` | `FORWARD/BACKWARD/LATERAL_LEFT/LATERAL_RIGHT/SPIN_CW/SPIN_CCW` | 必须确认可执行的机体运动形式 |

## 11. 飞跃式平台能力字段

### 载体类型

```text
platform-control-capability-source/v1
platform.platform_type: HOPPER
```

| 字段 | 格式/范围 | 定义 |
|---|---|---|
| `maximum_landing_slope_rad` | 有限 `float64`，rad，`[0,π/2]` | 最大安全着陆坡度 |
| `maximum_landing_roughness_m` | 有限 `float64`，m，`>=0` | 最大安全着陆面粗糙度 |
| `maximum_landing_plane_residual_m` | 有限 `float64`，m，`>=0` | 最大着陆面拟合残差 |
| `minimum_overhead_clearance_m` | 有限 `float64`，m，`>=0` | 最小顶部净空 |
| `minimum_lateral_clearance_m` | 有限 `float64`，m，`>=0` | 最小侧向净空 |
| `minimum_landing_region_area_m2` | 有限 `float64`，m²，`>0` | 最小安全着陆区域面积 |
| `maximum_launch_speed_mps` | 有限 `float64`，m/s，`>0` | 最大发射速度 |
| `maximum_launch_impulse_newton_seconds` | 有限 `float64`，N·s，`>0` | 最大发射冲量 |
| `minimum_flight_time_s` | 有限 `float64`，s，`>0` | 最短飞行时间 |
| `maximum_flight_time_s` | 有限 `float64`，s，`>=minimum_flight_time_s` | 最长飞行时间 |
| `maximum_landing_speed_mps` | 有限 `float64`，m/s，`>0` | 最大允许着陆速度 |
| `minimum_downward_impact_speed_mps` | 有限 `float64`，m/s，`>=0` | 最小向下碰撞速度约束 |
| `minimum_landing_clearance_m` | 有限 `float64`，m，`>=0` | 最小着陆净空 |
| `maximum_angular_speed_radps` | 有限 `float64`，rad/s，`>0` | 最大任意轴角速度 |
| `maximum_angular_acceleration_radps2` | 有限 `float64`，rad/s²，`>0` | 最大任意轴角加速度 |
| `maximum_initial_angular_speed_radps` | 有限 `float64`，rad/s，`>=0` | 发射前最大初始角速度 |
| `minimum_settle_guard_s` | 有限 `float64`，s，`>=0` | 着陆后最小稳定等待时间 |
| `actuator_or_impulse_profile.platform_mass_kg` | 有限 `float64`，kg，`>0` | 飞跃动力学计算使用的平台总质量 |
| `actuator_or_impulse_profile.launch_preparation_time_s` | 有限 `float64`，s，`>=0` | 从进入发射准备到允许起跳的持续时间 |
| `actuator_or_impulse_profile.landing_settle_time_s` | 有限 `float64`，s，`>=0` | 着陆后进入稳定状态所需的持续时间 |
| `actuator_or_impulse_profile.nominal_landing_center_normal_offset_m` | 有限 `float64`，m，`>=0` | 标称着陆中心相对着陆面的法向偏移 |
