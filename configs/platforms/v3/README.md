# Path Planner v3 平台能力配置

本目录保存与
`path-planner/schemas/v3/safety-capability-profile.schema.json`
一致的版本化平台能力配置：

| 文件 | 平台类型 | v3 能力范围 |
|---|---|---|
| `wheeled_skid_steer_v3_example_v1.json` | `WHEELED` | 前进、倒退、弧线行驶、原地旋转和时间参数化 |
| `legged_body_v3_example_v1.json` | `LEGGED` | 机体级平移、升降、旋转和时间参数化，不保证足端可行性 |
| `hopper_ballistic_v3_example_v1.json` | `HOPPER` | 纯弹道飞跃、着陆区域、飞行管道和姿态边界 |

## 安全状态

这三份文件是结构有效的示例配置，不是已认证的部署配置。

- 轮式文件中的车体长宽、最高速度、最大坡度和离地间隙参考现有
  `agilex_scout_mini_piper_v1.json`；其余动力学极限、误差界、运动原语及引用仍是示例值。
- 足式和飞跃式文件的数值来自仓库 v3 测试夹具，仅用于固定字段、单位和约束结构。
- `motion_model_ref`、`analytic_cost_model_ref`、`gravity_model_ref`、
  `swept_geometry_ref` 和 `sampled_body_sweep_ref` 的哈希是示例引用。真实运行前必须注册对应的不可变对象。
- 禁止用示例数值启动真实平台或把它们标记为已认证能力。

部署配置应复制为新的版本文件，替换所有示例值和示例引用，增加
`content_ref.revision`，然后重新计算
`content_ref.content_hash`。哈希对象只包含顶层 `content`，算法为
RFC 8785 JCS 后的 SHA-256。

## 为什么保留速度和加速度

v3 输出不是纯几何路径。轮式和足式参考都包含几何路径与单调时间缩放，
因此规划器必须用平台速度和加速度上限完成时间参数化和校验。

这些字段属于平台能力：

- 轮式最大前进、倒退和原地旋转速度；
- 轮式纵向、制动、偏航和横向加速度；
- 足式机体前后、横向、垂直和偏航速度区间；
- 足式机体线加速度和偏航角加速度；
- 飞跃式发射、着陆和姿态角速度边界。

搜索分辨率、原语采样密度、优化迭代数和超时仍属于
`PlannerAlgorithmConfig`，不放入本目录。

## 认证误差界与运行时误差界

配置中的 `certified_state_error_bounds` 是该能力配置允许的认证上界。
它不是定位系统发布的当前误差。

运行时流程为：

1. 本项目从 `nav_msgs/msg/Odometry` 协方差、消息时效、速度和项目阈值计算当前误差界；
2. 当前误差界写入 v3 `PlanningRequest.current_state.error_bounds`；
3. `certified_state_error_bounds` 随平台能力配置进入 v3；
4. v3 在激活和规划时校验当前状态、能力配置和误差模型的兼容性。

因此误差界仍由本项目设定和计算，但 v3 配置必须给出允许范围。

## 坐标系

示例配置使用：

```json
"frame_id": "map"
```

当前 v3 要求能力配置、规划请求和地图快照使用同一个 `frame_id`。
如果规划请求改在 `odom` 中执行，必须生成独立的 `odom` 版本能力对象并重新计算内容哈希，
或者在适配层把局部地图快照统一转换到 `map` 后再调用 v3。

碰撞包络的几何量仍表示平台机体包络；`frame_id` 的一致性要求来自当前 v3
规划合同。

## ROS 2 Humble 选择方式

建议让 ROS 2 参数只负责选择配置文件：

```yaml
path_planner_v3:
  ros__parameters:
    safety_capability_profile_package: lunar_navigation_config
    safety_capability_profile_resource: configs/platforms/v3/wheeled_skid_steer_v3_example_v1.json
    safety_capability_profile_expected_hash: e2518e770e49feed9cef0fa945ee468983bb831bfdaeaf9d81992e48b68b9559
```

节点在生命周期 `configure` 阶段：

1. 通过 ament index 定位包 share 目录；
2. 读取 JSON；
3. 校验 v3 schema；
4. 重新计算 `content` 的 JCS SHA-256；
5. 比对文件内哈希与参数期望哈希；
6. 解析并注册所有 `ContentRef`；
7. 激活后冻结能力配置。

运行中不允许修改能力配置。更换配置需要重新执行生命周期配置过程。

## 不属于 v3 平台能力文件的输入

以下内容由其他合同维护：

- 传感器观察范围：自主探索观察模型；
- 当前位姿、速度和协方差：平台运动状态；
- 当前地图：全局或局部 `grid_map_msgs/msg/GridMap`；
- 搜索预算和离散参数：`PlannerAlgorithmConfig`；
- 任务目标、边界和截止时间：任务上下文。

## 特殊限制

- 轮式配置仅对应当前项目实际支持的滑移转向语义，不声称 Ackermann 可行性。
- 足式配置固定
  `feasibility_scope=body_geometry_and_terrain_thresholds_only`，
  且 `footstep_feasibility_guaranteed=false`。
- 足式 schema 强制要求 `maximum_gap_width_m`。当前正式地图适配尚未计算真实沟壑宽度，
  示例配置将其设为 `0`；不得据此声称跨沟能力。
- 飞跃式配置固定
  `translation_model=PURE_BALLISTIC_NO_INFLIGHT_TRANSLATION_CONTROL`，
  不得声称飞行中具有平移控制能力。
