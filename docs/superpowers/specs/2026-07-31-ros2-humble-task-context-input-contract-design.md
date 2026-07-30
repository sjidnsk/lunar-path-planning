# ROS 2 Humble 任务上下文外部输入合同设计

日期：2026-07-31
状态：待用户审阅书面规范
适用范围：`lunar-path-planning` 在 ROS 2 Humble 中消费的区域探索任务上下文

## 1. 目标与边界

本合同定义外部任务系统需要向本项目提供的最小任务上下文：

- 当前执行哪一次区域探索任务；
- 在 `map` 坐标系中的哪个矩形区域探索；
- 任务应当激活、暂停还是取消；
- 哪些科学目标区域应当优先探索。

本项目负责：

- 校验并维护最近一份有效任务快照；
- 根据任务 ROI 限定探索与覆盖统计范围；
- 把科学目标多边形栅格化为任务 `value` 层；
- 在任务激活时生成探索候选、选择目标并调用 v1 或 v3 路径规划；
- 在暂停或取消后停止生成新的探索目标和路线。

本合同不重复承载：

- 全局图、局部图或地图事实层；
- 平台位姿、速度、定位 covariance 或 TF；
- 平台尺寸、运动能力和安全能力；
- A*、Hybrid A* 或 v3 算法调参；
- 控制器状态和路线执行反馈；
- 传感器数据、SLAM 或数据融合结果。

## 2. 按当前项目能力确定输入

当前项目的任务是覆盖优先的自主区域探索，而不是外部指定固定终点的点到点导航：

- `model-explorer` 根据地图自行生成并选择 frontier 目标；
- Python A* v1 和 C++20 v3 只规划到本项目已经选定的单次目标；
- 起始状态来自定位与运动状态输入；
- 禁入区、障碍、坡度和地图有效性来自地图与派生约束；
- 总体成功标准由项目版本化配置固定为覆盖率 `0.99`；
- 最大探索步数、连续无收益和无候选目标属于项目内部终止策略。

因此，任务系统不需要提供起点、逐次目标、规划权重、算法预算或平台参数。

## 3. ROS 2 Topic

```text
Topic: /mission/exploration_task
Type:  lunar_navigation_msgs/msg/ExplorationTask
```

推荐 QoS：

```text
reliability: reliable
durability:  transient_local
history:     keep_last
depth:       1
```

任务消息是完整状态快照，不是增量命令。`transient_local` 保证后启动或重启的规划节点能够立即获得任务系统最近发布的任务状态。

同一时刻只能有一个任务系统作为该 Topic 的权威发布者。

## 4. 消息定义

### 4.1 `ExplorationTask.msg`

```text
uint8 UNKNOWN=0
uint8 ACTIVE=1
uint8 PAUSED=2
uint8 CANCELED=3

std_msgs/Header header

string mission_id
uint64 revision
uint8 desired_state

float64 roi_min_x_m
float64 roi_min_y_m
float64 roi_max_x_m
float64 roi_max_y_m

lunar_navigation_msgs/ScienceTargetRegion[] science_regions
```

### 4.2 `ScienceTargetRegion.msg`

```text
string region_id
string objective_id
geometry_msgs/Polygon boundary
float32 priority
```

第一版通过语义校验限制：

```text
science_regions 最大数量:       64
单个 boundary 最大顶点数:      256
```

ROS `.msg` 本身不表达这些变长数组上限，接收方必须显式校验。

## 5. `ExplorationTask` 字段定义

| 字段 | 格式/范围 | 定义 |
|---|---|---|
| `header.stamp` | ROS Time，非零 | 本任务快照开始生效的时间 |
| `header.frame_id` | 固定为 `map` | ROI 和所有科学目标区域共同使用的坐标系 |
| `mission_id` | 非空 string | 一次探索任务的全局唯一标识 |
| `revision` | `uint64`，从 `1` 开始 | 同一任务完整快照的严格递增版本 |
| `desired_state` | `ACTIVE/PAUSED/CANCELED` | 任务系统要求本项目进入的任务状态 |
| `roi_min_x_m` | 有限 `float64`，m | 轴对齐矩形 ROI 的最小 x |
| `roi_min_y_m` | 有限 `float64`，m | 轴对齐矩形 ROI 的最小 y |
| `roi_max_x_m` | 有限 `float64`，m | 轴对齐矩形 ROI 的最大 x |
| `roi_max_y_m` | 有限 `float64`，m | 轴对齐矩形 ROI 的最大 y |
| `science_regions` | `ScienceTargetRegion[]` | ROI 内需要软优先探索的科学区域；允许为空 |

ROI 必须满足：

```text
roi_min_x_m < roi_max_x_m
roi_min_y_m < roi_max_y_m
```

第一版只支持 `map` 坐标系中的轴对齐矩形 ROI。旋转矩形、任意多边形总任务区、多块不连续总任务区不属于本合同。

## 6. 科学目标区域

### 6.1 字段定义

| 字段 | 格式/范围 | 定义 |
|---|---|---|
| `region_id` | 当前任务内唯一的非空 string | 科学目标区域标识 |
| `objective_id` | 非空 string | 外部科学目标标识；一个目标可以对应多个区域 |
| `boundary` | `geometry_msgs/msg/Polygon` | `map` 坐标系中的简单多边形 |
| `priority` | `float32`，`(0,1]` | 软探索优先级，数值越大越优先 |

### 6.2 多边形规则

- 至少包含三个互不相同的顶点；
- 不重复填写首顶点作为末顶点；
- 所有坐标必须为有限值；
- 所有顶点的 `z` 必须为 `0`；
- 边不得自交；
- 同一任务内 `region_id` 不得重复；
- `objective_id` 可以重复，用于表达同一科学目标的多个空间区域。

科学区域先与任务 ROI 求交：

- 部分超出 ROI：使用交集并产生裁剪诊断；
- 完全位于 ROI 外：拒绝整个任务快照；
- 求交结果退化为线或点：按无有效面积处理并拒绝整个任务快照。

### 6.3 软优先语义

科学目标区域只表示“优先探索”，不表示：

- 必须进入区域内部；
- 必须达到单独的区域覆盖率；
- 可以覆盖禁入、障碍或不可通行约束；
- 未完整覆盖该区域就不能完成总体任务。

第一版不定义科学区域硬必达、分区完成阈值或观测质量门槛。

## 7. 任务生命周期

允许的状态转换为：

```text
新 mission_id
    |
    v
 ACTIVE <----> PAUSED
    |             |
    +------v------+
         CANCELED
```

### 7.1 新任务

- 首条有效快照必须是 `revision=1`；
- 首条有效状态必须是 `ACTIVE`；
- 新任务必须使用从未使用过的新 `mission_id`；
- 当前任务尚未取消时收到另一个 `mission_id`，拒绝新任务，禁止隐式抢占。

### 7.2 暂停

收到合法 `PAUSED` 后：

- 不再生成新的探索目标；
- 不再发起新的 v1/v3 规划请求；
- 保留任务 ROI、科学目标、覆盖历史和地图；
- 暂停前尚未执行的旧路线不得在恢复后直接复用。

`PAUSED` 只约束本项目停止生成新路线。平台减速、制动和安全驻停仍由任务系统与平台控制系统负责，本项目不能仅凭任务状态声称平台已经停止。

### 7.3 恢复

收到合法 `PAUSED -> ACTIVE` 后：

- 使用最新有效地图；
- 使用最新有效定位和运动状态；
- 从当前实际位置重新生成候选目标并规划；
- 不把暂停前的起点、候选集合或旧路线当作当前事实。

### 7.4 取消

收到合法 `CANCELED` 后：

- 终止该任务的目标生成和新路线规划；
- 清除任务派生的 `value` 覆盖层；
- 保留环境地图，不删除地图事实或定位状态；
- 冻结必要的任务审计记录；
- 同一 `mission_id` 不得再次进入 `ACTIVE` 或 `PAUSED`。

## 8. Revision 与不可变内容

同一任务的 `revision` 必须严格递增。

| 情况 | 行为 |
|---|---|
| revision 更大，状态转换合法，静态内容相同 | 接受 |
| revision 相同，完整内容相同 | 作为幂等重复消息忽略 |
| revision 相同，任意内容不同 | 拒绝并报告版本冲突 |
| revision 更小 | 作为乱序旧消息拒绝 |
| revision 更大但静态内容改变 | 拒绝并报告任务内容改变 |

同一 `mission_id` 内以下内容不可修改：

- ROI；
- `science_regions` 数量和顺序无关的语义集合；
- 每个区域的边界；
- `region_id`；
- `objective_id`；
- `priority`。

比较静态任务内容时，接收方按 `region_id` 排序区域，并把多边形统一为逆时针顶点顺序、以字典序最小顶点作为起点，再比较规范化结果。仅数组顺序、多边形绕向或首顶点位置不同，不视为任务内容改变；规范化后的坐标、标识或优先级不同，视为内容改变。

revision 更新只能修改：

- `desired_state`；
- `header.stamp`。

需要修改 ROI、科学区域或优先级时，必须取消旧任务并创建新的 `mission_id`。

同一任务中，新的 `header.stamp` 不得早于最近有效 revision 的时间戳。任务快照是持久状态，不按传感器消息的新鲜度上限拒绝一份仍处于 `ACTIVE` 或 `PAUSED` 的 transient-local 快照。

## 9. 任务与地图的绑定

任务可以先于全局地图到达。

任务通过自身格式校验但尚无可用全局图时，内部进入：

```text
PENDING_MAP
```

此时：

- 保存任务快照；
- 不生成探索目标；
- 不生成规划路线；
- 等待有效 `/environment/map_global`。

全局图到达后必须在几何范围上覆盖完整 ROI。ROI 内允许存在未知或 `valid_mask=0` 的单元；地图几何覆盖不等于所有单元已经被观测。

如果当前全局图不能覆盖完整 ROI：

- 保持 `PENDING_MAP`；
- 报告 `ROI_NOT_COVERED`；
- 不用地图边界静默缩小任务区域。

地图内容、分辨率或几何更新不改变任务 revision。只要新全局图仍覆盖完整 ROI，本项目就根据原始矢量科学区域重新生成任务价值层。

## 10. 科学价值栅格化

对每个全局图单元，使用该单元中心点进行多边形归属判断：

- 单元中心在区域内部：该区域覆盖此单元；
- 单元中心位于区域边界：按区域内部处理；
- 单元中心不在任何区域：任务价值为 `0.0`。

多个科学区域重叠时：

```text
value(cell) =
  max(priority(region) for region covering cell)
```

禁止对重叠优先级求和，避免区域数量改变数值尺度或使结果超过 `[0,1]`。

全局图分辨率或几何变化时，必须从原始多边形重新栅格化，禁止对旧 `value` 栅格反复重采样。

在线 ROS 2 任务模式下：

- `science_regions` 是科学任务价值的唯一外部权威来源；
- 外部 `/environment/map_global` 中的 `value` 层不参与计算；
- 如果外部地图仍携带 `value`，接受其余合法地图层，但忽略该层并报告 `EXTERNAL_VALUE_IGNORED`；
- 本项目在 `/planning/map_global_derived` 中发布任务派生 `value`。

科学价值的硬边界为：

- 不参与 `confidence` 计算；
- 不改变 `valid_mask`；
- 不改变 `passable_mask`；
- 不覆盖 `forbidden`；
- 不覆盖障碍、坡度、障碍高度或平台能力约束；
- 不改变覆盖完成条件。

## 11. 覆盖完成标准

任务系统不逐任务提供 `target_coverage_ratio`。

第一版继续使用项目版本化配置：

```text
success_coverage_rate = 0.99
```

总体覆盖率只统计任务 ROI：

```text
coverage_rate =
  count(observed AND coverable AND inside_roi)
  / count(coverable AND inside_roi)
```

`coverable` 由本项目根据地图、平台能力、安全约束、可达性和观测模型派生，不要求任务系统提供。

科学目标优先级不改变覆盖率的分子、分母或成功阈值。

以下内容仍属于项目内部运行保护和终止原因，而不是任务上下文输入：

- 最大探索步数；
- 连续无覆盖收益步数；
- 无安全可达候选目标；
- 严重安全失败；
- 规划器内部时间、状态数和内存预算。

## 12. 与探索和路径规划的映射

### 12.1 探索决策

本项目从任务上下文获得：

- ROI；
- 派生 `value`；
- 当前任务状态。

`model-explorer` 在 ROI 内生成安全 frontier 候选，并使用科学价值估计：

```text
value_gain =
  sum(value(cell) for visible unknown cells)
```

`value_gain` 只作为候选特征和候选裁剪/排序信号。当前项目的候选优先级合同保持：

```text
candidate_priority_source =
  score_first_gain_value_cost_priority/v1
```

### 12.2 Python A* v1

Python A* v1 不直接消费完整任务上下文。ROS 适配层只向其传入：

- 当前起始栅格；
- `model-explorer` 选择的目标栅格；
- 代价图；
- 可通行掩膜；
- 项目内部算法预算。

`mission_id` 和 revision 由 ROS 编排层与规划请求建立外部关联，不修改 `path-planner-request/v1` 的现有地图搜索语义。

### 12.3 C++20 v3

v3 不直接使用科学多边形决定运动可行性。本项目把选定的探索目标转换成 v3 `GoalRegion`。

允许在 `goal.task_metadata` 中记录：

```text
mission_id
mission_revision
science_region_id
science_objective_id
```

只有目标选择能够明确归属于某一科学区域时，才填写后两个字段。不得伪造区域归属。

科学目标影响“选择去哪里”，平台能力、地图硬约束和 v3 规划算法决定“怎样安全到达”。

## 13. 接收校验

每条任务快照至少校验：

1. Topic 和消息类型正确；
2. `header.frame_id == "map"`；
3. `header.stamp` 非零；
4. `mission_id` 非空；
5. revision 与状态转换合法；
6. ROI 坐标有限且面积为正；
7. 科学区域和顶点数量不超过上限；
8. `region_id`、`objective_id` 非空；
9. `region_id` 在任务内唯一；
10. `priority` 有限且位于 `(0,1]`；
11. 多边形顶点、面积和自交规则合法；
12. 每个科学区域与 ROI 存在有效面积交集；
13. 有全局图时，检查其几何范围是否覆盖完整 ROI。

`UNKNOWN` 是 ROS 消息的零值保护，不是合法运行状态；接收方必须拒绝。

## 14. 失败行为与诊断

任务快照采用原子接收：

- 任意字段或任意科学区域非法时，拒绝整个新快照；
- 不得只接受合法区域并静默丢弃非法区域；
- 已有有效任务时，非法更新不得覆盖最近有效任务；
- 尚无有效任务时，不生成探索目标或路线。

建议稳定诊断码：

```text
TASK_FRAME_INVALID
TASK_STAMP_INVALID
TASK_STATE_INVALID
TASK_ID_CONFLICT
TASK_REVISION_OLD
TASK_REVISION_CONFLICT
TASK_CONTENT_CHANGED
ROI_INVALID
ROI_NOT_COVERED
SCIENCE_REGION_INVALID
SCIENCE_REGION_OUTSIDE_ROI
EXTERNAL_VALUE_IGNORED
```

部分科学区域被 ROI 裁剪但仍有有效面积时，任务可以接受，但必须记录非拒绝性裁剪诊断，并保留原始区域标识。

## 15. 外部责任与本项目责任

| 内容 | 外部任务系统 | 本项目 |
|---|---|---|
| `mission_id` 与任务状态 | 提供 | 校验、维护 |
| 探索 ROI | 提供 | 校验、用于目标生成与覆盖统计 |
| 科学区域边界 | 提供 | 校验、裁剪、栅格化 |
| 科学优先级 | 提供 | 生成 `value` 并用于候选排序 |
| `value` 栅格 | 不提供在线权威值 | 计算并发布 |
| 当前平台状态 | 不通过任务消息提供 | 从定位输入获得 |
| 单次探索目标 | 不提供 | `model-explorer` 选择 |
| 规划路线 | 不提供 | v1/v3 计算 |
| `0.99` 完成阈值 | 不逐任务提供 | 由版本化配置确定 |
| 暂停后的物理驻停 | 与平台控制系统负责 | 停止生成新路线，但不宣称平台已停止 |

## 16. 验收要求

至少覆盖以下测试：

### 16.1 消息与 QoS

- ROS 2 Humble 消息生成、序列化和反序列化；
- `reliable + transient_local + keep_last/1` 重启接收；
- 单一权威发布者配置检查。

### 16.2 状态与 revision

- 所有允许的状态转换；
- `UNKNOWN` 和非法状态转换；
- 幂等重复消息；
- 乱序 revision；
- 同 revision 内容冲突；
- 同任务静态内容改变；
- 活动任务期间新 `mission_id` 冲突；
- `CANCELED` 后禁止重新激活。

### 16.3 ROI 与科学区域

- ROI 零面积、反向边界、NaN 和无穷；
- 多边形顶点不足、重复、非零 z、自交和退化面积；
- 重复 `region_id`；
- 区域部分超出 ROI 的裁剪；
- 区域完全位于 ROI 外；
- 最大区域数和最大顶点数。

### 16.4 栅格化

- 单元中心在区域内、边界上和区域外；
- 重叠区域取最大优先级；
- 无科学区域时全零；
- 地图分辨率或几何变化后从矢量重新栅格化；
- 外部 `value` 被忽略并产生诊断；
- `value` 不改变硬安全约束和覆盖完成条件。

### 16.5 集成

- 任务先于地图到达时进入 `PENDING_MAP`；
- 地图未覆盖完整 ROI 时不规划；
- `ACTIVE` 生成探索候选；
- `PAUSED` 和 `CANCELED` 不生成新路线；
- 恢复后使用最新地图和最新定位；
- v1 只接收内部选定目标；
- v3 `task_metadata` 与任务和科学目标正确绑定。

## 17. 当前代码映射与实施边界

当前代码已经具备以下语义基础：

- `src/lunar_exploration_ppo/configs/stage1.py`：
  矩形 ROI、`success_coverage_rate=0.99`、最大步数和停滞参数；
- `src/lunar_exploration_ppo/env/frontier.py`：
  从低分辨率价值先验计算 `value_gain`，并用于候选优先级；
- `src/lunar_exploration_ppo/env/env.py`：
  覆盖完成、最大步数、停滞和无候选终止；
- `path-planner/src/path_planner/adapters/json_io.py`：
  v1 起点、目标、代价图和可通行掩膜请求；
- `path-planner/schemas/v3/planning-request.schema.json`：
  v3 `GoalRegion` 和规划请求；
- `path-planner/schemas/v3/common.schema.json`：
  v3 `GoalRegion.task_metadata`。

本合同不表示以下 ROS 2 功能已经实现：

- `lunar_navigation_msgs` 消息包；
- `/mission/exploration_task` 订阅器；
- lifecycle 状态机；
- 多边形校验与确定性栅格化；
- 任务派生 `value` 发布；
- v1/v3 ROS 编排和任务元数据绑定；
- 任务诊断或任务结果输出。

这些内容需要在用户复核本规范后进入独立实施计划。
