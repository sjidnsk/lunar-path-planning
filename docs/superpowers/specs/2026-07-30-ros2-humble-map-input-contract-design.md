# ROS 2 Humble 多通道地图外部输入合同设计

日期：2026-07-30
状态：待用户审阅
适用范围：`lunar-path-planning` 在 ROS 2 Humble 中消费的外部地图输入

## 1. 目标与边界

本合同只定义本项目从外部建图、数据融合和安全系统接收的多通道环境地图。区域探索任务和科学目标通过独立任务上下文输入，不再与地图事实层混合发布。

外部系统负责：

- 传感器驱动、标定和时间同步；
- 点云、图像、雷达及其他原始数据处理；
- SLAM、定位、地图融合和物理障碍估计；
- 提供有效、带时间戳的地图事实层和观测质量证据。

本项目负责：

- 维护全局与局部多通道地图的最新有效快照；
- 校验地图几何、数值范围、坐标系、时间和通道合同；
- 根据地图事实层和平台能力派生坡度、崎岖度、置信度、硬约束、通行性和代价；
- 在地图上进行探索决策与路径规划。

本项目不接收原始传感器数据，不执行 SLAM 或多传感器融合，也不把 synthetic terrain proxy 解释为物理障碍真值。

## 2. ROS 2 消息载体

地图数据主体统一使用：

```text
grid_map_msgs/msg/GridMap
```

采用 `GridMap` 的原因：

- 支持同一几何网格上的多个 `float32` 图层；
- 所有图层共享 `header`、分辨率、范围和地图位姿；
- 支持 `basic_layers` 有效性声明；
- 可直接使用 `grid_map_core`、`grid_map_ros` 和 RViz 相关工具。

消息关键字段定义如下：

| 字段 | 合同 |
|---|---|
| `header.stamp` | 本快照对应的融合时刻，不是 ROS 节点发送完成时刻 |
| `header.frame_id` | 全局图固定为 `map`，局部图固定为 `odom` |
| `info.resolution` | 单元边长，单位 `m/cell`，必须大于零 |
| `info.length_x/length_y` | 地图覆盖范围，单位 m，必须大于零 |
| `info.pose` | 地图几何中心在 `header.frame_id` 中的位姿 |
| `layers` | 通道名称数组，名称必须满足本合同 |
| `basic_layers` | 判断单元基础环境数据是否完整所依赖的通道 |
| `data` | 与 `layers` 一一对应的 `Float32MultiArray` |
| `outer_start_index/inner_start_index` | GridMap 循环缓冲区起始索引，订阅方必须正确处理 |

第一版只接收完整地图快照，不定义栅格增量消息。每条 `GridMap` 消息必须自洽，不能引用上一条消息才能恢复完整地图。

第一版以 `(topic_name, header.frame_id, header.stamp)` 作为快照键。同一快照键不得对应不同地图内容；本合同暂不引入自定义 revision 消息。

## 3. 双地图输入

### 3.1 低频全局图

```text
Topic: /environment/map_global
Type:  grid_map_msgs/msg/GridMap
Frame: map
```

全局图用于：

- 全局探索目标生成；
- 已知、未知和覆盖区域维护；
- 全局路径规划；
- 任务派生价值、禁入区和长期地图质量管理。

默认运行建议：

| 属性 | 建议 |
|---|---|
| 空间范围 | 固定任务区域或较大的全局区域 |
| 分辨率 | `0.2–1.0 m/cell`，由任务配置确定 |
| 发布方式 | 地图内容更新时发布，并允许 `0.2–1 Hz` 周期重发 |
| Reliability | `reliable` |
| Durability | `transient_local` |
| History/Depth | `keep_last/1` |

`transient_local` 保证后启动的规划节点能够收到最近一张全局图。

### 3.2 高频局部图

```text
Topic: /environment/map_local
Type:  grid_map_msgs/msg/GridMap
Frame: odom
```

局部图用于：

- 近场障碍和坡度约束；
- 局部路线安全校验；
- 路线阻塞检测；
- 高频局部重规划。

默认运行建议：

| 属性 | 建议 |
|---|---|
| 空间范围 | 以平台为中心的滚动窗口 |
| 分辨率 | `0.05–0.2 m/cell`，由平台速度和传感器能力确定 |
| 发布频率 | 初始建议 `5–10 Hz` |
| Reliability | `reliable` |
| Durability | `volatile` |
| History/Depth | `keep_last/1` |

局部图只保留最新快照。旧局部图不得排队参与规划。

### 3.3 坐标系

坐标树固定为：

```text
map -> odom -> base_link
```

外部定位系统负责发布 `map -> odom` 和 `odom -> base_link`。本项目必须按地图消息的 `header.stamp` 查询 TF，禁止使用当前最新 TF 转换历史地图。

全局图允许随 SLAM 全局校正发生离散位姿变化；局部图位于连续的 `odom` 坐标系，避免全局校正跳变直接破坏近场安全规划。

局部图与全局图重叠时：

- 局部图用于当前近场规划和安全判定；
- 全局图用于探索与全局路线；
- 本项目不把局部图反向融合进全局图；
- 全局地图融合仍属于外部建图系统职责。

## 4. 地图输入通道

全局图和局部图使用以下输入通道。

| 通道名 | 数值格式 | 单位/范围 | 定义 | 数据所有者 |
|---|---|---|---|---|
| `elevation` | `float32` | m | 栅格中心相对统一高程基准面的高度 | 外部建图/融合系统 |
| `valid_mask` | `float32` | `0.0/1.0` | 该格基础环境数据是否有效 | 外部建图/融合系统 |
| `obstacle` | `float32` | `[0,1]` | 物理障碍存在概率或归一化占据置信度 | 外部建图/融合系统 |
| `obstacle_height` | `float32` | m，非负 | 障碍物相对局部地面的高度 | 外部建图/融合系统 |
| `observation_age_s` | `float32` | s，非负 | 相对 `header.stamp`，距该格最后一次有效融合观测的时间 | 外部建图/融合系统 |
| `observation_quality` | `float32` | `[0,1]` | 最近一次实际参与地图更新的有效观测综合质量 | 外部建图/融合系统 |
| `elevation_variance` | `float32` | m²，非负 | 上游高程估计误差方差 | 外部建图/融合系统 |
| `obstacle_variance` | `float32` | `[0,0.25]` | 上游障碍概率估计不确定度，不是障碍概率本身 | 外部建图/融合系统 |
| `observation_count` | `float32` 中的整数 | `[0,65535]` | 累计有效融合观测次数，超过上限后饱和 | 外部建图/融合系统 |
| `forbidden` | `float32` | `0.0/1.0` | 任务或安全系统指定的禁入区域 | 外部任务/安全系统 |

`observation_quality` 应由上游根据实际观测证据给出，可综合：

- 传感器有效状态；
- 点云或图像质量；
- 配准质量；
- 融合残差；
- 遮挡、入射角和有效返回；
- 数据是否实际参与地图更新。

本项目通过位姿、FOV 和量程计算的理论可见性只能用于探索收益预测，不能代替实际 `observation_quality`。

`observation_age_s` 使用相对秒数，避免在 `float32` 图层中保存绝对 ROS 时间造成精度损失：

```text
last_observation_time = header.stamp - observation_age_s
```

`forbidden` 是任务或安全事实，不参与地图数据置信度计算。

在线 ROS 2 区域探索模式下，科学价值不再作为外部地图层输入。外部任务系统通过
`/mission/exploration_task` 中的 `science_regions` 提供矢量科学区域和软优先级，本项目确定性栅格化并在派生全局图中生成 `value`。

如果外部地图仍携带名为 `value` 的层：

- 不把该层作为任务科学价值使用；
- 接受其余满足合同的地图层；
- 忽略外部 `value` 并报告 `EXTERNAL_VALUE_IGNORED`；
- 禁止与任务派生 `value` 求和、取平均或静默覆盖。

## 5. 有效性与 NaN 规则

推荐：

```yaml
basic_layers:
  - elevation
  - valid_mask
  - obstacle
  - obstacle_height
  - observation_age_s
  - observation_quality
```

`GridMap.basic_layers` 只能检查基础层是否为有限值，不能检查 `valid_mask` 的 `0/1` 语义，因此本项目还必须执行显式合同校验。

当 `valid_mask=0.0`：

- 其他事实层和质量层允许为 `NaN`；
- `observation_count` 可以为 `0`；
- 本项目派生的 `confidence`、`slope`、`roughness` 和 `traversability` 在该格必须标记为无效；
- `cost` 可以写入配置的 `blocked_cell_cost`，但必须同时满足 `passable_mask=0` 和 `constraint_reason` 包含 invalid；
- 该格不得进入可通行区域。

当 `valid_mask=1.0`：

- 地图输入通道必须为有限值；
- `observation_age_s >= 0`；
- `observation_quality`、`obstacle` 必须位于 `[0,1]`；
- `obstacle_height >= 0`；
- 其余输入通道必须满足各自范围。

禁止：

- 使用 `-1` 表示 `GridMap` 浮点图层未知值；
- 在 `valid_mask=1.0` 的基础层中保留 NaN 或无穷；
- 把 synthetic terrain proxy 写入在线 `obstacle`；
- 混用不同时间、分辨率、范围或坐标系的图层拼成同一快照。

## 6. 本项目派生层

以下内容不是外部地图输入，由本项目根据事实层、质量证据、平台能力和地图历史计算：

| 派生层 | 定义 |
|---|---|
| `slope` | 从 `elevation + resolution` 计算的坡度，单位 deg |
| `roughness` | 从局部高程变化计算的归一化崎岖度 |
| `confidence` | 地图最终综合置信度，范围 `[0,1]` |
| `passable_mask` | 平台约束下的硬可通行掩膜 |
| `constraint_reason` | invalid/slope/obstacle/height/forbidden 原因位图 |
| `traversability` | 归一化通行性，范围 `[0,1]` |
| `cost` | 非负规划代价 |
| `risk` | 归一化风险诊断，范围 `[0,1]` |
| `coverage_mask` | 已覆盖或已观测区域 |
| `value` | 由任务上下文科学区域栅格化得到的软探索优先级，范围 `[0,1]` |

建议派生地图 topic：

```text
/planning/map_global_derived
/planning/map_local_derived
```

最终 `confidence` 必须发布。其诊断分量可由调试配置控制是否作为附加层发布：

```text
confidence_resolution
confidence_observation
confidence_recency
confidence_consistency
confidence_model
```

这些诊断分量的含义为：

| 分量 | 定义 |
|---|---|
| `confidence_resolution` | 地图分辨率相对规划尺度的适用度 |
| `confidence_observation` | 由外部 `observation_quality` 提供的实际观测质量 |
| `confidence_recency` | 由 `observation_age_s` 计算的时间新鲜度 |
| `confidence_consistency` | 当前快照与历史有效快照的一致性 |
| `confidence_model` | 由概率后验熵、方差和有效观测证据形成的模型明确度 |

综合置信度使用有效分量的归一化加权平均。缺失分量必须同时从分子和分母移除，不能按零分参与融合。

`confidence` 只影响风险与规划代价，不得单独把低置信度格变成硬不可通行格。硬不可通行仍由无效数据、坡度、物理障碍、障碍高度和禁入区决定。

## 7. 接收校验和失败行为

每次接收地图快照时至少校验：

1. topic、`frame_id` 和地图类型匹配；
2. `header.stamp` 非零且未超过配置的新鲜度上限；
3. 分辨率、长度和地图位姿合法；
4. `layers` 名称唯一，`layers` 与 `data` 数量一致；
5. 地图输入通道存在；
6. 每层逻辑尺寸与地图几何一致；
7. `outer_start_index/inner_start_index` 合法；
8. 所有有效格满足数值范围；
9. 全局图可在对应时刻解析 `map -> odom`；
10. 局部图可在对应时刻解析 `odom -> base_link`。

在线任务模式下，接收方还必须检查是否存在外部 `value` 层；该层不导致整张环境地图被拒绝，但必须被忽略并产生稳定诊断。

失败处理：

- 单张新快照校验失败时，不覆盖最近一张有效快照；
- 记录稳定的拒绝原因和 topic/时间信息；
- 局部图超时或无效时，禁止生成新的局部安全路线；
- 全局图超时不自动删除已有路线，但禁止基于过期图生成新的全局探索决策；
- 同一物理区域中，局部图与全局图冲突时，以局部图约束当前近场规划，同时报告冲突，不由本项目执行地图融合。

## 8. ROS 2 Humble 依赖与参考

- `grid_map_msgs/msg/GridMap`：
  <https://docs.ros.org/en/ros2_packages/humble/api/grid_map_msgs/msg/GridMap.html>
- ROS 2 Humble QoS：
  <https://docs.ros.org/en/humble/Concepts/Intermediate/About-Quality-of-Service-Settings.html>
- `nav_msgs`：
  <https://docs.ros.org/en/humble/p/nav_msgs/>

## 9. 当前代码映射

现有代码已具备以下派生能力：

- `dev-platform-constraints/src/dev_platform_constraints/terrain/features.py`：
  从 elevation 派生 slope 和 roughness；
- `dev-platform-constraints/src/dev_platform_constraints/confidence/metrics.py`：
  resolution、observation、recency、consistency 分量及归一化融合；
- `dev-platform-constraints/src/dev_platform_constraints/confidence/bayesian.py`：
  从后验熵派生 model confidence；
- `dev-platform-constraints/src/dev_platform_constraints/mapping/constraints.py`：
  生成 passable mask 和约束原因；
- `dev-platform-constraints/src/dev_platform_constraints/mapping/costmap.py`：
  生成非负 cost 和 traversability。

本合同不表示上述 Python 代码已经完成 ROS 2 封装；ROS 2 消息适配、生命周期、同步、QoS 和运行时校验需要后续实施计划。
