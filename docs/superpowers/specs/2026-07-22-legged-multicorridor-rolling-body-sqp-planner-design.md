# 足式平台多走廊滚动机体 SQP 路径规划器设计

## 文档状态

- 状态：总体架构、地图/硬约束/风险合同、多走廊滚动 SQP、失败与验收合同均已于 2026-07-22 逐节批准。
- 日期：2026-07-22。
- 工作树：`D:/codex/worktrees/multiplatform-path-planner-v2`。
- 分支：`codex/multiplatform-path-planner-v2`。
- 建议能力 ID：`legged_multicorridor_rolling_body_sqp/v1`。
- 适用对象：以动态行走或小跑方式运动、由外部单位负责稳定控制的四足平台。
- 规划器输出：时间戳机体路径或局部航点，不输出足端轨迹、接触力、步态或关节命令。

本文冻结设计，不授权实现、发布 checkpoint、替换 default policy、连接 executor、启动 canary，亦不改变现有 v1 默认 A*、v2 wheel Hybrid A*、Scout Mini SQP 或足式静态爬行代理。

## 1. 已批准的关键决定

1. 足式平台按 `0.3–0.8m/s` 的常用运行区间设计，规划请求频率为 `2–5Hz`。
2. 四足稳定控制、步态、落足、接触力、MPC/WBC 和关节控制由外部单位负责。
3. 规划器只消费外部单位提供的固定、版本化能力 Profile，不向控制器发起在线轨迹可行性查询。
4. PPO 负责选择任务目标位置和目标朝向；PPO 不提供安全地图，也不绕过规划器硬约束。
5. 地图直接复用公共 `TerrainSnapshotV2`：单一不可变 `0.5m` fine safety anchor，而非独立的全局/局部双真值地图。
6. `1.0m/2.0m` 层只能从 `0.5m` 权威快照保守聚合，用于引导、剪枝和下界估计。
7. 硬安全、预测失败风险、时间/能耗、平滑度按不可交换的字典序处理。
8. 学习模型只提供软风险或启发式；它不能删除硬障碍、修改可通行掩膜、降低坡度门限或授予 L2。
9. 每个不可变快照最多生成三条拓扑不同的确定性全局走廊。
10. 局部机体 SQP 使用状态 `(x,y,yaw,v)`，控制 `(a,omega)`，段持续时间为决策量；不包含足端或接触变量。
11. 多走廊采用 Scout Mini 同构策略：按版本化风险引导键依次求解，第一条通过完整 L2 的局部轨迹立即返回。
12. 全局走廊只是通向任务目标的不可执行意图；唯一可执行载体是完整到达滚动子目标的局部 L2 时间轨迹。
13. 地图更新形成新快照和新请求，不跨快照复用 open/closed、走廊、SQP warm start、反例或旧 L2 证据。

## 2. 问题定义、目标与非目标

### 2.1 问题定义

给定活动四足平台、当前机体状态、PPO 指定的任务目标、不可变已观测 2.5D 地形快照、固定能力 Profile、资源偏好和硬期限，规划器需要：

1. 证明当前快照中存在通向任务目标的全局几何走廊；
2. 沿风险优先的走廊生成一条完整到达滚动子目标的时间戳机体轨迹；
3. 对规范化后的连续机体包络执行 fine-grid L2；
4. 返回全局意图、局部可执行轨迹、风险证据、身份哈希和稳定 telemetry；
5. 无法完成时返回稳定失败，不返回部分可执行轨迹。

### 2.2 目标

- 在静态粗糙地形上提供确定性、可审计的全局连通性和滚动局部机体路径。
- 把安全真值、学习风险和资源目标分离，保证模型故障不改变硬安全结果。
- 与现有多平台公共请求、地图、证据、失败和资源框架保持最大兼容。
- 为外部稳定控制单位提供清晰、版本化、可拒绝的时间戳机体路径接口。

### 2.3 非目标

- 不规划单足落点、足端摆动轨迹、步态相位、接触力或支撑多边形。
- 不声称动力学稳定、实机可执行、抗滑或抗扰性能。
- 不处理移动障碍或时间预测地图。
- 不把当前 `simulation_proxy_static_crawl/v1` 的落足/支撑 oracle 解释为新能力证明。
- 不声称首条返回轨迹是三条走廊中的全局最低风险或全局最优轨迹。
- 不跨请求维护增量 D* 状态、SQP warm start 或局部滚动地图真值。

## 3. 主流与前沿算法对照

足式路径规划没有一个在所有尺度、地形和控制边界上都占优的单一算法。工程主流与研究前沿普遍采用“全局离散搜索 + 局部连续优化/预测控制 + 学习地形代价 + 独立安全验证”的混合结构。

| 类别 | 典型方法 | 优点 | 局限 | 本设计取舍 |
|---|---|---|---|---|
| 栅格图搜索 | A*、Dijkstra、D* Lite、MHA* | 确定性强、可解释、易加入硬掩膜 | 位姿/速度离散化粗，难直接表达连续机体轨迹 | 用于多尺度全局走廊，不直接授权执行 |
| 采样规划 | PRM、RRT、RRT*、BIT* | 适合高维和复杂几何，可渐进改进 | 随机性、窄通道和实时稳定性较弱 | 不作为 v1 权威主链，可作离线 oracle |
| 状态格/Hybrid A* | 运动原语、Hybrid A*、State Lattice | 可把运动学嵌入搜索，工程成熟 | 状态爆炸，足式接触模型难统一 | 保留作参考；新能力采用走廊 + 连续 SQP |
| 落足搜索 | Footstep A*、接触图、MICP | 可显式验证落足与支撑 | 计算昂贵，依赖控制器和接触模型 | 外部稳定控制负责，本规划器不复制 |
| 连续优化 | CHOMP、TrajOpt、SQP、NMPC | 可直接优化时间、控制和平滑度 | 初值敏感，优化成功不等于碰撞安全 | 每条走廊一个确定性 SQP，最终独立 L2 |
| 学习地形代价 | CNN/Transformer traversability、失败概率、学习启发式 | 能表达手工代价难覆盖的地形风险 | OOD、校准和安全证明困难 | 只作软风险与走廊排序，故障时确定性回退 |
| 端到端导航 | 感知到速度/目标的 RL 或 imitation | 反应快，可利用高维感知 | 可解释性和硬安全不足，域外风险高 | 不作为路径安全权威 |

代表性前沿工作包括：

- [ArtPlanner](https://arxiv.org/abs/2303.01420)：用采样可达性抽象结合学习的落足/运动代价，体现“学习代价 + 经典搜索”的实地混合路线。
- [Perceptive Locomotion through Nonlinear Model Predictive Control](https://arxiv.org/abs/2208.08373)：把感知地形直接纳入 NMPC，耦合局部运动与地形约束。
- [Coupled Planning, Estimation, and Control of Quadruped Robot Locomotion](https://arxiv.org/abs/2003.05481)：强调机体、接触和控制联合规划，但计算和接口耦合显著更高。
- [ViPlanner](https://arxiv.org/abs/2310.00982)：展示学习型视觉可通行性与路径规划结合的方向。
- [Traversability-Aware Legged Navigation](https://arxiv.org/abs/2410.10621)：以可通行性估计增强足式导航决策。
- [High-Speed Quadruped Navigation](https://arxiv.org/abs/2506.02835)：代表高速感知导航与学习控制的前沿，但不适合作为当前硬安全合同的直接替代。
- [Footstep Planning on Uneven Terrain with Mixed-Integer Convex Optimization](https://arxiv.org/abs/1612.02109) 与 [Footstep Planning for Legged Robots](https://arxiv.org/abs/1907.08673)：代表显式落足优化与搜索路线。

综合比较后，已批准方案是安全约束混合架构：多尺度确定性多走廊负责全局连通，滚动机体 SQP 负责连续局部轨迹，学习模型只提供软风险，fine-grid L2 独立授予成功。

## 4. 职责与公共输入边界

### 4.1 输入

新能力逻辑输入为：

```text
LeggedRollingRequestV1
  request_id
  platform_profile_id
  controller_capability_id
  current_body_state(x,y,yaw,v)
  mission_goal_state(x,y,yaw)
  terrain_snapshot: TerrainSnapshotV2
  objective_profile
  resource_budget
  timeout_s
  replan_period_s
  determinism_seed
```

`replan_period_s` 必须是精确、有限的内建 `float`，并位于闭区间 `[0.2,0.5]`，对应 `5–2Hz`。平台 Profile 必须绑定机体包络、安全 margin、`0.8m/s` 最大规划速度、加减速、横摆率、横摆加速度、坡度门限、最低发布置信度、滚动前视范围、SQP/L2 资源上限和外部控制器能力身份。`profile.min_publish_confidence` 必须是精确、有限且位于 `[0,1]` 的内建 `float`；其数值由具体控制器能力 Profile 密封，不从本次地图或候选集合动态校准。`0.3m/s` 是常用巡航下界，不是禁止停车的硬最小速度。

### 4.2 PPO 边界

- PPO 只提供任务目标位置与目标朝向，或在候选目标之间给出偏好。
- PPO 不提供 `TerrainSnapshotV2`，不生成 hard mask，不授予可达性或 L2。
- 规划器不替 PPO 改选任务语义目标；不可达时返回失败证据。

### 4.3 外部控制边界

- 外部单位消费 `TimedBodyTrajectoryV1`，自行完成步态、落足、接触力、MPC/WBC 和关节控制。
- 规划器只依赖固定 `controller_capability_id/hash`，不进行在线 feasibility RPC。
- 规划成功仅表示机体路径合同通过，不表示外部控制器已接受或实机稳定。

## 5. 总体架构与模块边界

```text
不可变 TerrainSnapshotV2 + 请求 + 固定能力 Profile
  -> 请求、身份、端点、期限和资源预检
  -> LeggedHardSafetyView
  -> 0.5m 权威层保守聚合 1.0m / 2.0m 引导层
  -> 最多三条确定性、拓扑不同的 GlobalCorridorIntentV1
  -> 风险引导键稳定排序
  -> 沿当前走廊选择 5–20m 滚动子目标
  -> 一个确定性机体轨迹初值
  -> LeggedBodySQPOptimizer
  -> 规范化、风险证据和完整连续机体包络 L2
  -> 可定位反例最多一次确定性修复
  -> 首条通过路线立即返回 LeggedRollingPlanV1
```

模块职责：

- `LeggedHardSafetyView`：把共享地图与固定足式能力 Profile 组合成不可绕过的安全查询，不运行学习模型。
- `LeggedCorridorGenerator`：生成全局几何引导和拓扑签名，不宣称可执行或 L2。
- `LeggedRollingSubgoalSelector`：沿当前走廊选择滚动子目标，并在最终目标进入前视范围时保持真实任务目标。
- `LeggedBodySQPOptimizer`：在单条走廊附近求解机体运动学、时间和近似净空约束；其 success flag 不授予公共成功。
- `LeggedRiskEvaluator`：生成候选级、分段级软风险和模型健康证据，不修改硬安全视图。
- `LeggedBodyTrajectoryL2Validator`：从规范化序列化候选独立重放，唯一有权授予局部轨迹 L2。
- `LeggedRollingPlanAdapter`：组装复合结果或稳定失败，不吸附终点、不修补候选、不把全局走廊标成可执行路线。

## 6. 地图合同

### 6.1 唯一权威快照

足式平台直接复用公共 `TerrainSnapshotV2`：

```text
geometry
  width
  height
  origin
  frame_id
  resolution_m = 0.5

layers
  elevation_m: float64[H,W]
  slope_deg: float64[H,W]
  traversable_mask: bool[H,W]
  hard_obstacle_mask: bool[H,W]
  observed_mask: bool[H,W]
  confidence: float64[H,W] in [0,1]

provenance
  source_kind
  source_id
  source_hash
  physical_obstacle_cells_written
  details
```

几何、provenance 和全部图层共同参与 `terrain_snapshot_hash`。一次请求只读取该不可变快照；快照变化产生新的请求和安全身份。

### 6.2 多尺度派生层

- `2.0m`：全局拓扑和粗走廊引导。
- `1.0m`：中尺度冲突收敛和剪枝。
- `0.5m`：滚动子目标、SQP 地形近似和最终 L2。

粗层必须由 `0.5m` 子单元保守聚合：unknown、硬障碍、不可通行、超坡度或低于硬置信度的任一子单元，不得在粗层被聚合为可授权安全。粗层不单独序列化为安全真值，也不具有 L2 权力。

### 6.3 足式地形语义

公共快照没有独立的台阶、粗糙度和负障碍图层。上游必须把这些足式能力判断保守编码进当前 Profile 专属的 `traversable_mask`，并在 provenance `details` 中绑定：

```text
traversability_derivation_id
controller_capability_id
controller_capability_hash
terrain_feature_source_id
terrain_feature_source_hash
```

缺少上述绑定时，不得声称 `traversable_mask` 已覆盖台阶、粗糙度或负障碍能力。synthetic terrain 只能保持 proxy 语义，不得标成 `physical_obstacle_cells`。

## 7. 硬约束合同

### 7.1 地图硬门

任意发布轨迹的连续机体扫掠必须满足：

```text
in_bounds
&& observed_mask
&& !hard_obstacle_mask
&& traversable_mask
&& slope_deg <= 30.0
&& confidence >= profile.min_publish_confidence
```

- `30.0deg` 保持当前公共硬边界。
- unknown 不得通过插值、学习模型或粗层聚合变成安全。
- 低于 `min_publish_confidence` 的区域 fail closed；阈值以上的置信度差异可以进入软风险。
- 净空由无效 cell 和地图边界确定性派生，不要求保存为独立地图层。

### 7.2 机体运动学与完整性硬门

- 起点必须与请求当前状态精确绑定；滚动子目标满足版本化位置/朝向容差。
- 所有状态、控制、持续时间、成本和证据字段必须有限、规范化、可序列化。
- 时间轴严格单调；每段持续时间为正。
- `v/a/omega`、横摆加速度、停止和方向切换满足固定能力 Profile。
- 连续机体包络及 safety margin 不得扫过地图无效区域。
- 路线必须完整到达本次滚动子目标；不得返回 SQP 中间迭代、部分路径或只通过 L0/L1 的候选。
- 请求、Profile、控制器能力、快照、候选、风险证据、验证器和结果哈希必须一致。

### 7.3 控制职责硬边界

新能力不检查显式足端顺序、支撑多边形、质心投影或接触力。现有 `LeggedProfileV2` 和足式静态 L2 oracle 继续属于 `simulation_proxy_static_crawl/v1`，只能作为历史 benchmark/reference，不能为新机体路径提供动态稳定性声明。

## 8. 风险合同

### 8.1 风险输入

风险评估只消费当前快照和候选轨迹可见信息：

- 坡度及其距硬边界的裕量；
- 最小置信度、置信度裕量和局部波动；
- 到 unknown、障碍、不可通行区和地图边界的净空；
- 局部高程范围、坡度变化和地形边缘；
- 速度、加速度、横摆率、曲率和控制变化；
- 特征缺失、模型不确定性、校准状态和 OOD 状态。

### 8.2 `LeggedRiskEvidenceV1`

```text
segment_risk_upper_bounds[]
max_segment_risk_upper_bound
cumulative_route_risk
terrain_risk
clearance_risk
confidence_risk
kinematic_aggressiveness_risk
model_uncertainty
ood_detected
risk_source
model_id
feature_schema_id
calibration_id
risk_aggregation_id
snapshot_hash
candidate_hash
```

每个 `segment_risk_upper_bounds[i]` 必须是有限的 `[0,1]` 标量。首版聚合固定为：

```text
risk_aggregation_id = independent_segment_union_proxy/v1
cumulative_route_risk = 1 - product(1 - segment_risk_upper_bounds[i])
```

该聚合只是稳定、单调的路线风险 proxy；它不证明分段事件独立，也不得表述为经实机认证的真实失败概率。空分段集合不得形成成功候选。风险值是预测或代理，不是物理安全证明。学习模型不得修改 `TerrainSnapshotV2`、`LeggedHardSafetyView`、硬门限或 L2 结果。

### 8.3 回退与决策顺序

模型缺失、输出非有限、schema 不兼容或 OOD 时：

- 不得把风险写成 `0`；
- 使用坡度、净空、置信度和轨迹激进程度构成的确定性风险代理；
- 记录 `risk_source=deterministic_fallback` 和具体原因；
- 保持硬可行性与 L2 行为不变。

硬失败候选先被过滤。剩余候选和走廊使用以下字典序语义：

```text
(
  max_segment_risk_upper_bound,
  cumulative_route_risk,
  normalized_time,
  normalized_energy,
  smoothness_cost,
  deterministic_tie_break
)
```

当前公共 `CostBreakdownV2` 可继续用于报告，但新能力的选择权威是版本化字典序键，不得把安全或风险重新解释为可由能耗、时间抵消的普通加权和。

## 9. 全局多走廊合同

### 9.1 生成

1. 从 `0.5m` 权威安全视图保守生成 `1.0m/2.0m` 引导层。
2. 在 `2.0m` 层用稳定 tie-break 的确定性二维 A* 生成首条最低引导代价走廊。
3. 把 unknown、硬障碍、不可通行、超坡度、低置信度和越界分量构造成版本化阻塞分量。
4. 用固定参考射线规则计算相对阻塞分量的拓扑签名。
5. 对已接受签名施加确定性排除/惩罚，直到得到三个不同签名、无更多签名或耗尽显式资源。
6. 将每条走廊依次细化并锚定到 `1.0m` 和 `0.5m` 安全视图。

同一拓扑签名的近似重复路径不占用三条配额。全局走廊生成只证明当前快照中的二维连通意图，不证明局部机体轨迹可执行。

### 9.2 排序与返回策略

走廊使用稳定键排序：

```text
(
  risk_guide_max_upper_bound,
  risk_guide_cumulative,
  estimated_time,
  estimated_energy,
  guide_length,
  topology_signature,
  corridor_hash
)
```

按该顺序一次只处理一条走廊。第一条生成局部轨迹并通过完整 L2 的走廊立即返回；后续走廊不再运行。结果必须声明：

```text
selection_policy = first_l2_in_order/v1
global_optimality_claimed = false
```

## 10. 滚动局部机体 SQP

### 10.1 滚动子目标

- 沿当前全局走廊选择 Profile 绑定的 `5–20m` 前视点。
- 若任务最终目标进入前视范围，滚动子目标必须保持真实任务目标及其目标朝向。
- 子目标必须位于 `0.5m` 硬安全视图中，并绑定全局走廊、快照和任务目标哈希。

### 10.2 运动学模型

状态和控制为：

```text
q = (x, y, yaw, v)
u = (a, omega)
dt > 0

dx/dt   = v * cos(yaw)
dy/dt   = v * sin(yaw)
dyaw/dt = omega
dv/dt   = a
```

该模型只表达规划器与外部控制器约定的机体路径包络，不声称四足动力学或接触可行性。

### 10.3 初值与 SQP

- 每条走廊只有一个由走廊切线、当前速度和 Profile 速度包络确定的初值。
- 禁止随机 multi-start。
- 方向切换必须显式经过 `v=0`；是否允许后退由 Profile 固定声明。
- 硬约束包含起点、滚动终点容差、离散动力学一致性、速度/加速度/横摆率/横摆加速度、正持续时间、走廊信赖域、地图边界和保守连续净空。
- SQP 内部 merit 只帮助找到和改善候选，不具有安全授权，也不改变系统级字典序和首条 L2 返回政策。
- restoration 松弛量可以帮助数值恢复，但候选输出前必须全部满足版本化硬残差上限。

### 10.4 规范化、L2 与一次修复

候选先按能力 ID 规范化状态、控制、时间、角度、signed zero 和浮点序列，再序列化并从新对象独立重放。L2 至少检查：

- 请求、Profile、控制器能力、快照、候选和验证器身份；
- 起点、节点连接、时间单调性和解析/离散传播；
- 速度、加速度、横摆率、横摆加速度及模式开关；
- 完整连续机体包络与安全 margin；
- unknown、障碍、不可通行、坡度、置信度和边界；
- 滚动子目标真实位置/朝向容差；
- 风险证据、成本和哈希绑定。

若 L2 仅因一个可定位的扫掠 cell/时间区间失败，可把稳定排序的首个反例转换为当前走廊的新 SQP 硬约束，并允许恰好一次确定性修复。修复后必须重新规范化并从头执行完整 L2。身份、数值、动力学、资源或 deadline 失败不得修复。

## 11. 复合输出合同

### 11.1 `GlobalCorridorIntentV1`

- 从当前状态连通任务最终目标；
- 包含 cell corridor、拓扑签名、风险引导证据、快照/Profile 哈希和走廊哈希；
- 明确 `validation_level != L2`、`executable=false`；
- 不得直接交给外部控制器。

### 11.2 `TimedBodyTrajectoryV1`

每个时间段至少包含：

```text
start_state(x,y,yaw,v,t)
end_state(x,y,yaw,v,t)
a_mps2
omega_radps
duration_s
validation_level = L2
segment_hash
```

路线绑定真实滚动终点、风险证据、完整路线哈希和 L2 receipt。它不包含足端、接触力、步态或动态稳定声明。

### 11.3 `LeggedRollingPlanV1`

```text
request_id
mission_goal_state
rolling_goal_state
global_corridor_intent
local_body_trajectory
mission_complete
terrain_snapshot_hash
platform_profile_hash
controller_capability_hash
risk_evidence
cost_breakdown
telemetry
decision_hash
```

只有任务最终目标进入局部前视范围，且局部 L2 轨迹真实到达该目标时，`mission_complete=true`。因此该结果不能静默伪装成现有“完整到达最终目标”的 `PlanningSuccessV2`；必须使用新的能力专用返回语义或明确版本化包装。

## 12. 失败分类

稳定细因至少包括：

```text
legged_body_profile_unsupported
legged_map_profile_mismatch
legged_start_invalid
legged_goal_invalid
legged_no_global_corridor
legged_corridor_budget_exceeded
legged_local_subgoal_unavailable
legged_sqp_initialization_failed
legged_sqp_infeasible
legged_sqp_numeric_contract_failed
legged_candidate_l2_rejected
legged_repair_l2_rejected
legged_goal_tolerance_exceeded
legged_resource_budget_exceeded
planning_deadline_expired
legged_identity_mismatch
legged_internal_error
```

失败只携带诊断、反例、资源和 telemetry，不携带可执行的部分轨迹。风险模型缺失、OOD 或 schema 不兼容不属于规划失败；它们触发确定性风险回退。

## 13. 期限、资源、缓存与确定性

### 13.1 唯一期限

```text
effective_deadline = min(request.timeout_s, request.replan_period_s)
```

预检、地图派生、走廊、初始化、SQP、风险评估、规范化、L2、修复和编码共享同一 monotonic deadline。deadline 后不得出现 late success。

### 13.2 资源预留

- Profile 密封走廊数、候选数、segment/knot、SQP iteration/evaluation、L2 interval/cell、memory 和编码上限。
- 启动一次 SQP 或修复前，剩余资源必须覆盖最坏完整 L2 和结果编码预留。
- 预留不足时跳过该工作并返回资源/期限失败，不得形成未验证候选。

### 13.3 缓存与快照替换

- 每个请求重建 open/closed、走廊、初值、SQP 和反例状态。
- 只允许缓存由完整 `schema/profile/controller/snapshot` 哈希绑定的只读地图派生数据。
- 缓存开关只能改变耗时和 cache telemetry，不能改变走廊排序、候选、成功路线或失败分类。
- 新快照到达后，旧结果即使随后完成，也不得作为新快照下的有效路线。

### 13.4 确定性

相同输入 bytes、代码版本、快照、Profile、风险模型/回退版本和 seed 必须产生相同：

- 走廊数量、顺序、拓扑签名和哈希；
- 滚动子目标；
- SQP 初值与规范化轨迹；
- L2 结果、首反例和修复选择；
- 风险证据、成功/失败分类和决策哈希。

墙钟耗时和系统利用率不进入决策哈希。风险模型若不能保证批准环境中的确定性推理，必须退回确定性风险代理。

## 14. 测试与正式验收

### 14.1 地图与硬边界

- `30deg` 通过，`nextafter(30deg,+inf)` 拒绝。
- unknown、硬障碍、不可通行、低置信度和越界 fail closed。
- provenance/Profile/controller 绑定缺失或漂移时拒绝。
- 粗走廊可行但 `0.5m` fine anchor 失败时拒绝。
- synthetic proxy 不得升级为物理障碍真值。

### 14.2 多走廊与滚动语义

- 最多三条拓扑不同走廊及稳定排序。
- 相同拓扑近似重复路径不占配额。
- 第一条 L2 通过后不运行后续走廊。
- 全局走廊始终 `executable=false`。
- 局部轨迹完整到达滚动子目标；只有真实到达任务目标时 `mission_complete=true`。

### 14.3 SQP 与对抗 L2

- 节点安全但节点间连续扫掠碰撞。
- 旋转机体角点碰撞、短危险区间和地图边界切触。
- 速度、加速度、横摆率、时间和方向切换边界。
- 规范化前安全、规范化后不安全时拒绝。
- L2 首反例稳定，每走廊最多一次修复。
- interval/cell/memory/deadline cap 到达时 fail closed。

### 14.4 风险隔离

- 学习模型启停、缺失、非有限、OOD 和 fallback 只能改变风险证据或候选顺序。
- 风险模型不得改变 hard feasibility、地图层、门限或 L2。
- fallback 风险不得伪装为零风险。
- 相同模型身份和输入重复运行风险证据一致。

### 14.5 正式门

- oracle unsafe false-positive：`0`。
- 返回局部轨迹完整 L2 通过率：`100%`。
- unknown/障碍/坡度/低置信度违规：`0`。
- oracle-reachable 滚动子目标成功率：`>=99%`。
- 风险隔离违规：`0`。
- late success：`0`。
- 部分可执行轨迹返回：`0`。
- 全局走廊误标可执行：`0`。
- 相同输入的结构化决策字段和路线哈希一致率：`100%`。
- `200ms` 与 `500ms` 两种规划周期分别报告成功率、超时率、p50/p95/p99、走廊尝试数、SQP/L2 开销和 fallback 率。

`oracle-reachable` 必须来自独立、高预算、同硬能力边界的离线 oracle，不能由被测 SQP 或被测风险模型自标注。

## 15. 兼容性与发布边界

- 新能力使用新的 capability/profile/result/risk/validator IDs，保持 opt-in。
- 现有足式 `simulation_proxy_static_crawl/v1`、轮式 Scout Mini SQP、v2 Hybrid A* 和 v1 默认 A* 不变。
- 不发布 checkpoint，不替换 default policy，不连接 executor，不启动 canary。
- 不把风险模型输出写回硬地图，不把 PPO 目标概率写成地图风险。
- 不声明 Ackermann、足式动态稳定、落足可行、接触力可行或实机认证。
- 正式实现与 benchmark 必须使用独立实施计划；本文批准不等于实现或发布授权。

## 16. 未采用方案

### 16.1 直接复用静态落足搜索

当前足式 provider 显式搜索足端和静态支撑，适合历史 proxy 验证，但与“稳定控制由外部单位负责”的职责边界冲突，且不能自然表达 `2–5Hz` 时间戳机体轨迹。

### 16.2 独立 `0.1m` 滚动地图

它能表达更多足尺度细节，但现有公共地图只提供 `0.5m TerrainSnapshotV2`。对 `0.5m` 数据插值不会产生新的安全证据，还会引入跨分辨率身份与时间一致性问题，因此 v1 不采用。

### 16.3 跨快照 D* Lite/SQP warm start

它可能降低平均延迟，但会破坏 Scout Mini 已批准的单快照独立请求、证据绑定和确定性边界。只有未来新能力 revision 能证明等价时才可重新评估。

### 16.4 三条走廊全部优化后选最优

它可能获得更低风险候选，但显著增加延迟和资源波动，不利于 `2–5Hz`。用户已选择风险引导顺序中的第一条 L2 路线立即返回。

### 16.5 纯学习或端到端路线

其 OOD、校准、可解释性和安全授权不足，不适合作为当前 hard mask 与 L2 的替代。

## 17. 结论

足式新能力不是把轮式 SQP 改名，也不是扩展旧静态落足代理。它复用多平台公共地图、身份、资源和 L2 框架，借鉴 Scout Mini 的确定性多走廊、单初值 SQP、一次反例修复和首条通过返回语义，同时把足端稳定控制明确留在外部单位。

最终安全链为：公共 `0.5m TerrainSnapshotV2` 提供唯一地图事实，固定足式能力 Profile 构造硬安全视图，多尺度多走廊提供全局意图，滚动机体 SQP 产生局部时间轨迹，学习模型只提供软风险，独立连续机体包络 L2 授予唯一可执行成功。该结构在不夸大足式控制能力的前提下，兼顾确定性、实时性、风险敏感性和现有多平台模块兼容性。
