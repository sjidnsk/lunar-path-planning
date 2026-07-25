# 多平台能力感知路径规划 v2 设计报告

## 文档状态

- 状态：已批准，进入分阶段 Gate 实施。
- 实施分支：`codex/multiplatform-path-planner-v2`。
- 独立工作树：`D:/codex/worktrees/multiplatform-path-planner-v2`。
- 基线提交：`b635740ee021258ef31811ec87c60add839fc5f9`。
- 适用平台：轮式平台、足式平台、飞跃式平台。
- 版本策略：v1 保持默认；v2 仅显式选择时启用。

本报告冻结已经逐项解释、推荐并由用户批准的设计合同。它描述任务、需求、架构、接口、算法、指标、Gate、风险和重新开发结论；真实执行结果必须写入 D 盘对应阶段的 `report.md`，不能用本设计文档替代运行证据。

## 1. 执行摘要

现有模块不应整体推倒重写，也不能只在现有 A* 接口上增加三个平台名称。推荐采用“保留 v1、并行新增 v2”的渐进式重构：复用已经验证的栅格、Hybrid A*、碰撞/坡度检查和后处理能力；新增统一请求/结果合同、平台能力模型、运动原语提供器、分层搜索、惰性验证、失败证据和 benchmark。

重新开发的必要性来自平台间可达性定义不同：

- 轮式平台的可达性由连续位姿、转向/倒车原语、车体包络、坡度和障碍约束决定。
- 足式平台的可达性由落足点、步长/步高、支撑多边形、机体扫掠和地形接触决定。
- 飞跃式平台的可达性由起跳状态、弹道、全弧线净空、落区概率和着陆姿态决定。

这些差异要求共享框架下的平台专用运动原语，而不是共享一组二维栅格邻接边。结论是“局部重构并扩展”，不是“全量重写”或“维持原状”。

## 2. 范围与非范围

### 2.1 范围

- `path-planner` 包中的 v2 合同、搜索框架和平台插件。
- 轮式、四足静态爬行、月面弹道飞跃三类平台 profile 与安全 oracle。
- 2.5D 地形输入、已观测安全掩膜、确定性 FOV/LOS 观测增益。
- 完整路线的二级安全验证、失败证据、性能 telemetry、缓存和 benchmark。
- PPO 的薄适配层：PPO 提供目标位姿，planner 返回完全验证的路线和统一观测投影。
- 仿真训练和离线评估。

### 2.2 明确非范围

- 不发布 checkpoint。
- 不替换 default policy。
- 不连接 executor，不启动 canary。
- 不修改 PPO 算法、网络、reward 或训练语义。
- 不实现多机器人协同，也不在一次请求中切换平台。
- 不声称 Hybrid A* 满足 Ackermann 运动学。
- 不把 synthetic terrain proxy 表述为物理障碍真值。
- 不把足式或飞跃式的仿真结果宣称为实机认证。
- 首版不维护跨请求增量式 open/closed 搜索树。

## 3. 顶层任务

v2 的任务是：给定一个活动平台、2.5D 已观测环境、起始状态、PPO 指定的目标状态和任务资源偏好，在硬安全约束内返回一条完整、可执行语义明确、全路线已验证的路线；如果不能完成，则返回稳定分类、平台专用原因和可审计证据。

形式化表示：

```text
plan(request, platform_profile, observed_terrain)
  -> CompleteValidatedRoute | PlanningFailure
```

硬约束优先级固定为：

```text
安全可行性
> 完整到达目标
> 资源目标
> 固定资源开销内的探索收益
> 搜索加速与美化
```

任何可选加速器失败都只能降级或关闭，不能降低硬安全和完整路线要求。

## 4. 输入、输出与状态边界

### 4.1 单请求单平台

每个请求必须显式包含一个 `platform_profile_id`。一个请求只能选择 wheel、legged 或 hopper 中的一种平台，禁止在路线中途隐式切换运动模式。

### 4.2 PPO 与 planner 职责

- PPO 负责选目标位置和目标 `theta`。
- planner 按平台能力严格解释并验证目标 `theta`。
- 只有 profile 明确声明独立传感器偏航时，传感器朝向才可与机体朝向解耦。
- planner 不替 PPO 重选语义目标；不可达时返回失败证据。

### 4.3 地图信任边界

- 输入采用 2.5D 地形：高程、坡度、可通行度、硬障碍、已观测状态以及必要的置信信息。
- 接触、停止、落足、起跳、落地和弹道净空只允许依赖已经观测且满足 profile 安全条件的数据。
- 未知区域不得作为接触点或落区。
- 可使用未知区域估计观测收益，但不得据此绕过安全 oracle。

### 4.4 完整性边界

成功只允许返回完整起点到目标路线。部分路径、尚未验证的候选或只验证终点的路径不得标记成功。

## 5. v2 公共合同

### 5.1 请求模型

```text
PlanningRequestV2
  request_id
  platform_profile_id
  start_state
  goal_state
  terrain_snapshot
  objective_profile
  resource_budget
  timeout_s
  accelerator_policy
  determinism_seed
```

`start_state` 与 `goal_state` 至少包含 world `(x, y, theta)`；平台 provider 可以声明额外状态，但必须通过版本化 schema 显式出现。目标到达容差由版本化 platform profile 显式声明，基础 profile 默认位置与朝向容差均为 `0`。成功路线的实际末状态必须位于该容差内，且不得把未被原语 replay 到达的请求目标静默写成路线末状态；朝向比较使用 wrap-safe 最小角差。

### 5.2 成功模型

```text
PlanningSuccessV2
  schema_version
  request_id
  platform_kind
  route
  observation_projection
  cost_breakdown
  validation_evidence
  search_telemetry
  cache_evidence
```

`route` 为带类型的原语序列。公共字段描述起止状态、持续时间、距离、能量、观测贡献和验证层级；平台字段分别承载 wheel motion、leg step 或 ballistic jump 参数。

### 5.3 失败模型

```text
PlanningFailureV2
  schema_version
  request_id
  platform_kind
  category
  reason_code
  evidence
  search_telemetry
```

稳定大类建议冻结为：

- `invalid_request`
- `unsupported_capability`
- `unsafe_start`
- `unsafe_goal`
- `goal_pose_unreachable`
- `no_complete_route`
- `validation_failed`
- `resource_limit`
- `timeout`
- `internal_error`

`reason_code` 记录平台专用细因，例如坡度、足端支撑、弹道净空、着陆概率或目标朝向不可达。上层依赖稳定大类，诊断依赖细因和 evidence。

## 6. 总体架构

```text
PlanningRequestV2
        |
        v
Request/Schema Validation
        |
        v
Platform Profile Registry
        |
        +--> Wheel Primitive Provider
        +--> Legged Primitive Provider
        +--> Hopper Primitive Provider
        |
        v
Shared Fine Safety Anchor
        |
        v
Multi-resolution / Multi-heuristic Search
        |
        v
L0 -> L1 -> L2 Lazy Validation
        |
        v
Complete Typed Route + Observation Projection
```

### 6.1 共享部分

- 请求和结果 schema。
- 坐标、角度、栅格与世界变换。
- fine-grid 安全锚点。
- 搜索队列、确定性合并、资源预算和超时控制。
- 哈希缓存、验证证据、失败分类和 telemetry。
- 观测投影与 benchmark 工具。

### 6.2 平台专用部分

- 状态量化与目标姿态语义。
- 运动原语生成。
- 原语成本和资源模型。
- 接触/扫掠/净空/着陆 oracle。
- 路线反序列化和平台专用证据。

## 7. 共享 fine safety anchor

所有平台和所有搜索分辨率最终绑定同一 fine-grid 安全锚点。当前主线细网格为 `r=0.5m`，层级为 `r/2r/4r = 0.5/1.0/2.0m`。

安全锚点职责：

- 统一 unknown、hard obstacle、slope、observed-safe 与边界语义。
- 为 platform provider 提供不可绕过的局部查询接口。
- 生成可哈希、可缓存的 validation input identity。
- 明确 synthetic proxy provenance，禁止升级为物理真值表述。

粗层只能用于引导、剪枝和下界估计。任何成功路线必须回落到 fine anchor 完成 L2 全路线验证。

## 8. 搜索与验证策略

### 8.1 分层表示

- `4r`：全局拓扑与粗走廊引导。
- `2r`：中尺度候选扩展和冲突收敛。
- `r`：平台原语生成和最终安全验证。

层级不是三个独立安全真值。粗层通过 conservative aggregation 生成提示；fine 层保留最终裁决权。

### 8.2 多启发式

安全核心允许一个 admissible anchor heuristic；可选附加启发式可表达资源、走廊、观测增益和平台偏好。并行或多队列结果必须按固定 key 稳定合并，以保证同输入同输出。

### 8.3 L0/L1/L2 惰性验证

- L0：廉价几何/边界/明显不可行筛选。
- L1：平台局部可行性与近似扫掠/支撑/弹道检查。
- L2：对候选完整原语和整条路线执行 fine-grid 权威 oracle。

只有 L2 全部通过的路线可返回成功。若 L2 否决，搜索必须移除相应边或状态并继续；不能降级返回 L1 结果。

### 8.4 缓存

缓存键至少绑定：

```text
schema_version
platform_profile_hash
terrain_snapshot_hash
primitive_hash
validation_level
objective_profile_hash (仅成本相关缓存)
```

缓存 primitive 与 validation 结果；每个请求重建 open/closed。任何 key 不全、hash 漂移或 schema 不兼容都 fail closed。

## 9. 平台合同

### 9.1 轮式平台

- 类型：差速/滑移转向轮式平台。
- 初始实现复用现有 Hybrid A* 位姿状态、运动原语和碰撞检查。
- `max_traversable_slope_deg=30.0` 保持当前硬边界。
- 支持倒车与否、最小转弯语义、车体包络和代价由 profile 显式声明。
- 轮式首版使用 profile-bound 的版本化相对运动能耗 proxy；该 proxy 不声称物理 Joule，归一化常量不得依赖本次候选集合。
- 不声明 Ackermann feasible。
- v2 轮式路径是 opt-in；v1 默认 A* 不被替换。

### 9.2 足式平台

- 类型：四足静态爬行，`simulation_proxy`。
- 机体包络：`0.60m x 0.40m`。
- 名义足端矩形：`0.70m x 0.50m`。
- 最大水平步长：`0.50m`。
- 最大垂直步高：`0.25m`。
- 最大落足坡度：`25deg`。
- 最小支撑裕度：`0.05m`。
- 局部落足网格：`r/2`。

足式原语至少验证：目标足端已观测安全、步长/步高、局部坡度、支撑多边形、质心投影裕度、机体扫掠、足端顺序和目标机体朝向。首版使用静态稳定近似，不声称动态步态或实机稳定性。

### 9.3 飞跃式平台

- 类型：月面无动力弹道飞跃，`simulation_proxy`。
- 月面重力：`g=1.62m/s^2`。
- 起跳速度集合：`{1.5, 2.0, 2.5, 3.0}m/s`。
- 仰角集合：`{30deg, 45deg, 60deg}`。
- 方位角离散：16 个方向。
- 理论最大水平距离约 `5.6m`。
- 落点一维尺度不确定性：`sigma = 0.05 * range + 0.05m`。
- 最大着陆坡度：`15deg`。
- 安全着陆概率阈值：`>=0.99`。
- 首版不进行中途修正。

飞跃原语至少验证：起跳状态、完整弹道弧线净空、边界、落区全域已观测安全、坡度、概率质量、着陆姿态和后续停止条件。只检查起点/终点或只检查弹道采样中心线均不合格。

## 10. 观测收益合同

所有平台共享一个版本化参考传感器模型，以便公平比较覆盖效率。收益按确定性 FOV/LOS 计算，禁止读取未观测 truth 生成策略可见收益。

- 轮式与足式：沿实际完整路线按固定弧长步长投影观测，并在终点按目标 theta 观测。
- 飞跃式：默认只在起跳与落地状态投影观测。
- 只有 profile 显式允许时，飞跃中观测才可计入，且必须单独版本化和做消融。
- 独立传感器 yaw 只在 profile 声明时允许。

输出同时提供公共 observation projection 和平台原始路线，确保 PPO/评估无需解释三套不同原语格式。

## 11. 目标函数

安全是硬约束，不进入可交换的加权和。对安全可行的完整路线，资源成本默认定义为：

```text
resource_cost = 0.5 * normalized_energy + 0.5 * normalized_time
```

任务 profile 可覆盖权重，但必须记录在请求、结果与报告中。探索收益是次级目标，允许在最小资源路线的固定 20% 资源开销内选择覆盖收益更高的完整路线：

```text
candidate_resource_cost <= 1.20 * minimum_feasible_resource_cost
```

这样形成闭环：安全不可交易；目标必须到达；资源建立主排序；探索只在有限预算内优化。距离、Hybrid A* cost 或 AUC 可保留为诊断，不取代主合同。

## 12. 确定性、并行与超时

- 同一版本、同一输入、同一 profile 和同一 seed 必须返回字节级稳定的结构化决策字段。
- 候选排序、并行 collector、启发式队列和缓存遍历都使用稳定 tie-break key。
- 并行仅优化候选生成/验证，不改变可行性语义。
- Standard p95 目标 `<=250ms`。
- Kilometer p95 目标 `<=750ms`。
- 单请求端到端硬超时 `2s`，包括预处理、搜索、验证和结果编码。
- 超时必须返回 `timeout` 失败及阶段 telemetry，不得返回未完成路线。

Python 是首选实现。只有 Kilometer profile 连续三个正式批次 p95 超过 750ms，且 profiling 证明搜索热路径占比 `>=50%`，才启动 native hot-path Gate；native 实现必须与 Python oracle 等价。

## 13. 指标与正式验收

### 13.1 安全

- oracle false-positive：`0`。
- 每个平台正式 primitive 样本：至少 `10,000`。
- 返回路线全部 L2 验证通过：`100%`。
- 未知接触/落足/落地/弹道净空违规：`0`。

### 13.2 完备性

- oracle-reachable query 成功率：`>=99%`。
- primitive recall：`>=98%`。
- 成功结果必须完整到达目标：`100%`。

### 13.3 质量

- 小型 exact map 上资源成本比：`<=1.10 * optimum`。
- 探索收益候选必须满足资源开销 `<=20%`。
- 覆盖效率相对基线提升：点估计 `>=5%`，95% CI 下界 `>=0`。
- 任务成功率下降不超过 `1` 个百分点。

### 13.4 性能

- Standard 正式场景：每个平台至少 100 episodes，p95 `<=250ms`。
- Kilometer 正式场景：每个平台至少 30 episodes，p95 `<=750ms`。
- 硬超时率和超时阶段必须单独报告。

### 13.5 稳定性

- 相同输入重复运行路线、失败分类和关键 telemetry 一致。
- 多 worker 与单 worker 的语义结果一致。
- 缓存开启/关闭只允许影响耗时和 cache telemetry，不允许改变安全与路线排序。

## 14. Benchmark 与消融矩阵

正式报告至少包含：

- v1 默认 A* / 当前 wheel Hybrid A* opt-in 基线。
- v2 fine-only safe core。
- `+ multi-heuristic`。
- `+ r/2r/4r hierarchy`。
- `+ L0/L1/L2 lazy validation`。
- `+ hash cache`。
- 完整 v2。

每项报告安全、可达成功率、primitive recall、资源成本、覆盖效率、节点扩展数、L0/L1/L2 否决率、cache hit、p50/p95/p99 和超时。可选加速器若不能证明安全等价或造成成功率退化，默认关闭并记录原因。

## 15. 分阶段 Gate

### Gate 0：基线与隔离

- 新分支和独立工作树创建完成。
- 原 Stage6 dirty 工作树不被修改。
- `path-planner` v1 核心测试全绿。
- 记录根仓库 PPO 集成测试的已知、非 v2 基线失败，不擅自吸收或回退用户 Stage6 改动。
- 冻结本设计文档、实施索引和 evidence 目录。

退出条件：v1 核心有可复现绿色基线，所有外部失败都已分类且不会被误算为 v2 回归。

### Gate 1：v2 合同与 fine safety anchor

- 新增 versioned request/success/failure/profile schema。
- 新增共享 fine safety anchor、稳定 reason taxonomy、确定性序列化。
- v1 public API 和默认行为不变。

退出条件：合同测试、unknown/slope/obstacle 边界测试、v1 回归测试全绿。

### Gate 2：轮式 provider

- 将现有 Hybrid A* 适配为 wheel primitive provider。
- 建立 typed wheel route、全路线 L2 validation 和资源成本。
- 不声明 Ackermann，不替换默认 A*。

退出条件：轮式 10,000 primitive 安全审计通过，小图质量与 Standard 性能达到目标或给出明确 blocker。

### Gate 3：可选加速器

- 多启发式、r/2r/4r、惰性验证与缓存逐项接入。
- 每项都有等价性测试、消融和独立开关。

退出条件：安全核心不退化；失败的加速器被关闭，不阻塞后续平台。

### Gate 4：足式 provider

- 四足静态爬行原语、支撑/足端/机体扫掠 oracle。
- 明确 `simulation_proxy`。

退出条件：10,000 primitive 安全审计、可达率/recall、质量与性能证据满足正式门或明确 blocker。

### Gate 5：飞跃 provider

- 离散月面弹道原语、全弧线净空、概率落区与着陆 oracle。
- 明确 `simulation_proxy`，默认无空中观测和中途修正。

退出条件：10,000 primitive 安全审计、99% 落区安全合同、可达率/recall、质量与性能证据满足正式门或明确 blocker。

### Gate 6：PPO adapter、正式 benchmark 与发布判定

- 接入薄适配层，不改变 PPO 算法。
- 运行 Standard 100 / Kilometer 30 episodes 每平台及完整消融。
- 生成结构化 artifact 和人类报告。

退出条件：所有硬安全门通过；性能和收益门给出明确 pass/fail；v2 仍保持 opt-in，禁止自动发布、替换策略或连接 executor。

## 16. Gate 证据与 artifact

默认输出根：

```text
D:/xunce/out/path_v2/<gate_short>/
```

每个正式 Gate 至少生成：

```text
config.json
summary.json
routing.json
manifest.json
report.md
results.jsonl 或 metrics.jsonl
phase-state.jsonl
review.json
```

完整 lineage、hash、seed、profile 与参数放结构化 artifact，不塞入长路径名。新路径优先小于 80 字符；超过 180 字符 warning；达到 240 字符必须缩短。artifact IO 遵循项目既有短路径与原子写入合同。

## 17. 重新开发判定矩阵

| 方案 | 安全表达力 | 复用成本 | 多平台扩展 | v1 风险 | 结论 |
| --- | --- | --- | --- | --- | --- |
| 维持现有模块，仅增加平台枚举 | 低 | 低 | 低 | 表面低、长期高 | 不推荐 |
| 完全推倒重写 | 可高 | 极高 | 高 | 高 | 不推荐 |
| v1 保留、v2 并行渐进重构 | 高 | 中 | 高 | 低 | 推荐并批准 |

进入 v2 的触发条件已经满足：三类平台的状态和安全 oracle 本质不同；现有二维/Hybrid A* 接口不能完整表达足式支撑和弹道概率；同时现有轮式基础具有高复用价值。因此重新开发的是“公共合同、平台能力层和验证框架”，不是重写所有搜索与地图代码。

## 18. 风险与控制

- **安全语义被粗层绕过**：fine anchor 拥有最终裁决权，粗层只做提示。
- **平台抽象泄漏**：公共接口只承载共同语义，原语细节保留在 typed payload。
- **未知区域被当作安全**：所有接触与弹道净空查询 fail closed。
- **飞跃概率模型过度承诺**：标记 simulation proxy，报告参数敏感性，不做实机声明。
- **足式静态模型被误作动态能力**：profile 和 report 固定 capability level。
- **并行导致不确定性**：稳定 key、固定 seed、单/多 worker 等价测试。
- **性能优化污染正确性**：先 safe core，所有 accelerator 可独立关闭并做消融。
- **与 Stage6 用户改动冲突**：独立分支/工作树，后期 adapter 只做显式集成。
- **路径和 artifact 过长**：统一 D 盘短 root 与 canonical 文件名。

## 19. 需求到验证追踪

| 需求 | 主要实现 Gate | 主要证据 |
| --- | --- | --- |
| v1 默认不变、v2 opt-in | 0, 1, 2 | v1 回归、API/default 测试 |
| 单请求单平台 | 1 | schema/property tests |
| unknown 接触 fail closed | 1–5 | fine-anchor 与 oracle tests |
| wheel 位姿可达 | 2 | Hybrid A* provider tests |
| 四足静态稳定 | 4 | foothold/support/sweep oracle |
| 月面弹道安全着陆 | 5 | arc/landing probability oracle |
| 全路线 L2 才成功 | 1–5 | adversarial route validation |
| 资源主目标、探索 20% 预算 | 2–6 | optimality 与 ablation report |
| 确定性和稳定并行 | 1–6 | repeat/multi-worker tests |
| Standard/Kilometer 时延 | 2–6 | benchmark telemetry |
| PPO 只给目标，不改算法 | 6 | adapter contract tests |

## 20. 最终发布结论规则

只有同时满足以下条件，才能把 v2 标记为“内部验证通过”：

1. 三个平台的硬安全门全部通过。
2. v1 默认和既有公共行为未改变。
3. 成功结果全部是完整 L2 路线。
4. 正式可达率、recall、质量、性能和确定性报告齐全。
5. 所有未通过项都在 `routing.json` 中明确为 blocker 或 disabled accelerator。

即使内部验证通过，也不自动授权 checkpoint 发布、default policy 替换、executor 接入、canary 或实机声明。这些仍需要新的明确授权和独立安全流程。
