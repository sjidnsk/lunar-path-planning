# Scout Mini 运动学多走廊 SQP 轮式规划器设计

## 文档状态

- 状态：轮式 Section 1–4 与本书面规格已于 2026-07-22 批准；实现仍须按独立实施计划执行。
- 日期：2026-07-22。
- 工作树：`D:/codex/worktrees/multiplatform-path-planner-v2`。
- 分支：`codex/multiplatform-path-planner-v2`。
- 新能力 ID：`wheel_kinematic_corridor_sqp/v1`。
- 适用对象：当前 Scout Mini 差速/滑移转向平台。
- 版本边界：v1 继续保持默认；新能力仅通过 v2 显式选择。

本文只冻结轮式新算法的设计，不授权实现、发布、checkpoint、default policy 替换、executor、canary 或跨请求搜索状态复用。

## 1. 已批准的关键决定

1. 只设计当前 Scout Mini 差速/滑移转向，不覆盖 Ackermann、全向轮或四轮独立转向。
2. 单次请求只读取静态 2.5D 地形快照；地图变化形成新的独立请求，不复用旧 open/closed、走廊或 SQP warm start。
3. 目标成功容差为位置闭区间 `<=0.25m`、wrap-safe 朝向闭区间 `<=5deg`；路线保存真实 replay 终点，禁止吸附或改写成请求目标。
4. 全局引导最多生成三条确定性、拓扑不同的二维走廊。
5. 连续规划采用运动学直接多重射击 SQP；状态为 `(x,y,theta)`，控制为 `(v,omega)`，持续时间是决策量。
6. 不建立未经标定的轮地摩擦、侧滑、质量或扭矩动力学模型。
7. 第一条完整路线通过独立连续 L2 后立即返回，不等待最优性证明或同请求内的质量改善。
8. 现有 v2 Hybrid A* 只保留为 benchmark/reference，不作为运行时隐式 fallback。

## 2. 目标与非目标

### 2.1 目标

给定 `PlanningRequestV2`、Scout Mini wheel profile 和静态已观测 2.5D 地形，在共享硬期限内返回一条：

- 满足差速运动学；
- 满足速度、角速度、加减速、倒车和原地旋转能力；
- 完整连续车体扫掠已通过 L2；
- 真实终点位于批准容差内；
- 带确定性证据、成本和 telemetry；
- 可由版本化公共合同重放的完整时间轨迹。

若不能完成，返回稳定分类和原因，不返回部分轨迹、SQP 中间迭代或只通过近似约束的候选。

### 2.2 非目标

- 不处理移动障碍或时间预测地图。
- 不宣称完整滑移转向动力学可行或实机认证。
- 不连接控制器或 executor。
- 不用 MPPI、RRT、Hybrid A* 或 State Lattice 作为新能力的运行时备选求解器。
- 不跨请求保存搜索树、走廊、SQP 解或 L2 反例。
- 不改变 v1 默认 A*、PPO、reward、network、checkpoint 或 canary。

## 3. 公共输入与输出

### 3.1 输入

新能力继续接收现有 v2 请求边界：

```text
PlanningRequestV2
  request_id
  platform_profile_id
  start_state(x,y,theta)
  goal_state(x,y,theta)
  terrain_snapshot
  objective_profile
  resource_budget
  timeout_s
  accelerator_policy
  determinism_seed
```

profile 必须精确绑定：Scout Mini 车体尺寸 `0.612m x 0.580m`、安全 margin、差速/滑移转向标识、倒车和原地旋转开关、`v/omega` 上限、加减速上限、坡度硬边界 `30.0deg`、当前 `0.5m` fine safety anchor、数值容差、SQP 上限、走廊/L2 工作量上限、规范化规则和求解器合同 ID。缺字段、身份漂移或不支持的模型必须在求解前失败。

### 3.2 输出

成功结果继续包装为 `PlanningSuccessV2` 和 `TypedRouteV2`，但路线原语使用新的 `WheelKinematicSegmentV2`。每段至少包含：

```text
start_state
end_state
v_mps
omega_radps
duration_s
reverse
turn_in_place
relative_energy
validation_level=L2
segment_hash
```

分段控制为常值，连续轨迹由公开、版本化的差速解析积分唯一确定。用于观测投影的离散 samples 是派生载体，不是碰撞安全权威。成功结果还包含真实终点、完整路线哈希、成本分解、L2 证据、走廊/SQP/L2 telemetry 和缓存证据。

`relative_energy` 继续使用 profile-bound 的 `wheel_relative_motion_energy/v1` 相对能耗 proxy，不声称物理 Joule；归一化常量不得从本次候选集合动态推导。

## 4. 总体架构

```text
请求与静态地形快照
  -> 请求、profile、端点和资源预检
  -> 最多三条确定性二维拓扑走廊
  -> 走廊到带方向/时间初值的确定性转换
  -> 运动学直接多重射击 SQP
  -> 规范化候选
  -> 独立完整连续路线 L2
  -> 首条通过路线立即返回
```

模块边界如下。

### 4.1 `CorridorGenerator`

只生成全局几何引导，不宣称车辆可执行或安全。它只能读取当前 immutable terrain snapshot，并输出带版本化拓扑签名、引导成本和路径哈希的二维 cell corridor。

### 4.2 `TrajectoryInitializer`

把单条二维走廊确定性转换成一组 `(x,y,theta,v,omega,duration)` 初值。它负责前进、倒车和原地旋转模式选择，但不执行安全裁决。

### 4.3 `KinematicSQPOptimizer`

在走廊初值附近求解完整运动学约束问题。它输出候选和求解证据，不具有成功授权。

### 4.4 `WheelTrajectoryL2Validator`

从序列化候选重新解析并重放连续轨迹，权威检查运动学、资源边界和完整旋转矩形扫掠。只有该模块可以授予路线 L2 成功。

### 4.5 `WheelTrajectoryAdapter`

把通过 L2 的候选组装为公共 v2 success，或把失败映射为稳定 failure。它不得修补、吸附或替换候选轨迹。

## 5. 走廊生成

走廊图只包含当前快照中已观测且中心单元不属于 hard obstacle、越界或 `slope>30deg` 的 cell。障碍距离、坡度裕量和路线长度可进入引导代价，但这些代价不能替代真实矩形车体 L2。

走廊生成过程固定为：

1. 用稳定 tie-break 的确定性二维 A* 生成最低引导代价走廊。
2. 把 unknown、hard obstacle、越界和超坡度 cell 的连通分量构造成版本化阻塞分量集合。
3. 对候选路径计算相对这些分量的离散拓扑签名；签名算法和参考射线规则由能力 ID 密封。
4. 对已接受签名施加确定性排除/惩罚并继续搜索，直到得到三个不同签名、无更多签名或达到显式资源边界。
5. 按 `(guide_cost, path_length, topology_signature, path_hash)` 稳定排序。

仅路径不同而拓扑签名相同的近似重复走廊不占用三条配额。走廊生成失败与车辆不可达不是同义词；失败证据必须区分无二维连通走廊、资源耗尽和截止时间到达。

## 6. 确定性初值

每条走廊只产生一个初值，禁止随机 multi-start。初始化器沿走廊切线构造两个局部朝向选择：正向切线和反向切线。动态规划在完整走廊上选择前进、倒车、原地旋转和模式切换序列，并使用 profile-bound 的稳定代价和 tie-break。

初值必须：

- 从请求真实起点开始；
- 保留请求目标，不修改目标姿态；
- 在方向切换处显式经过 `v=0`；
- 为原地旋转显式生成 `v=0, omega!=0` 段；
- 为每段生成正、有限持续时间；
- 把 knot/segment 数量限制在显式资源上限内；
- 生成可供 SQP 使用但不被误记为 L1/L2 的未验证候选。

## 7. 直接多重射击 SQP

第 `k` 段决策变量为状态 `q_k=(x_k,y_k,theta_k)`、控制 `u_k=(v_k,omega_k)` 和持续时间 `dt_k`。状态传播使用分段常值控制下的公开差速解析积分；`omega` 接近零时使用同一版本化连续极限分支，禁止由求解器选择不同公式。

硬约束包括：

- 起点 exact match；
- 末端位置误差 `<=0.25m`；
- wrap-safe 末端朝向误差 `<=5deg`；
- 相邻 shooting node 与解析传播一致；
- `v/omega`、线加速度、角加速度和持续时间边界；
- 倒车和原地旋转开关；
- 地图边界、坡度和连续净空的优化近似约束；
- knot、segment、iteration、memory 和 wall-clock 上限。

优化目标只在可行候选之间引导收敛，包含请求中的相对时间和相对能耗目标，并可加入固定权重的控制变化、走廊偏离和净空正则项。安全不是可交换的软代价。

SQP restoration 可以使用显式松弛量帮助恢复，但候选输出前必须满足 profile 密封的硬约束残差上限。任何非有限数、求解器状态漂移、变量次序漂移或超限都不得形成候选成功。

SQP 后端是内部可替换实现；公共语义只绑定求解器合同 ID、固定变量顺序、固定初值、固定容差、单线程确定性设置和一致的合规测试。后端名称或私有迭代状态不得进入公共路线语义。后端更换若不能在冻结 conformance corpus 上产生相同规范化决策字段，必须使用新的求解器合同或 capability revision，不能在原 ID 下静默漂移。

## 8. 规范化与完整连续 L2

求解器候选先按能力 ID 指定的规则规范化状态、控制、持续时间、角度、signed zero 和浮点序列。规范化后的值才可序列化和验证。规范化可能改变约束结果，因此必须重新执行运动学和完整 L2；规范化前的通过结果不能继承。

L2 从规范化、序列化后的 segments 独立重建，至少检查：

- 请求、profile、terrain snapshot、求解器合同和路线 identity；
- 起点、节点连接、正持续时间和单调时间轴；
- 每段解析传播与声明末状态；
- `v/omega`、加减速、倒车和原地旋转能力；
- 真实 `0.612m x 0.580m` 矩形车体及 safety margin 的完整连续扫掠；
- 扫掠涉及的 unknown、hard obstacle、越界和 `30deg` 坡度边界；
- 真实末端位置/朝向容差；
- 路线、segment、成本和证据 hash/reseal。

连续扫掠不得只检查 shooting nodes 或观测 samples。对每个解析轨迹区间，验证器构造旋转矩形扫掠的保守区间上界：能证明远离危险 cell 的区间直接通过；可能相交的区间按稳定顺序继续细分。达到 profile/request 的 interval、cell、memory、numeric 或 deadline 上限仍不能证明安全时，必须 fail closed。

## 9. L2 反例修复

若候选仅因一个可定位的扫掠 cell/时间区间或净空约束被 L2 否决，验证器输出稳定排序的首个反例。该反例可转换为当前走廊的新增 SQP 约束，并允许恰好一次确定性修复。修复候选必须重新规范化并从头执行完整 L2。

以下情况禁止修复：

- 运动学残差或节点断裂；
- 非有限数或数值合同失败；
- profile、terrain、route、segment 或 solver identity 漂移；
- 资源或截止时间不足；
- 序列化/reseal 不一致；
- 不受支持的能力或求解器状态。

当前走廊修复后仍失败，才处理下一条走廊。任一走廊通过完整 L2 后立即返回，不运行后续走廊或质量改善。

## 10. 截止时间与资源

API 入口创建唯一绝对 monotonic deadline，effective timeout 保持 `min(request.timeout_s,2.0s)`。走廊、初始化、SQP、规范化、L2 和结果编码都接收同一 deadline，并在有界工作单元之间检查。

能力 profile 定义版本化 `L2ReserveModelV1`：根据当前走廊的 segment cap、解析扫掠 broad-phase cell 上界、细分上限和编码上限计算最坏 L2/编码预留。开始一次新的 SQP 或 L2 修复前，剩余资源必须覆盖该预留；不能覆盖时返回 timeout/resource failure，而不是消耗完预算后产生未验证候选。该预留不替代真实 deadline 检查。

每个请求重新生成走廊和 SQP 解。只允许对 immutable terrain 派生数据使用完整 snapshot/profile/schema hash 的只读缓存；缓存开关只能影响耗时与 cache telemetry，不能改变走廊排序、候选、成功路线或失败分类。

## 11. 失败分类

公共稳定大类继续使用现有 v2 taxonomy。新能力至少提供以下细因：

```text
wheel_sqp_profile_unsupported
wheel_sqp_start_invalid
wheel_sqp_goal_invalid
wheel_sqp_no_2d_corridor
wheel_sqp_corridor_budget_exceeded
wheel_sqp_initialization_failed
wheel_sqp_infeasible
wheel_sqp_numeric_contract_failed
wheel_sqp_candidate_l2_rejected
wheel_sqp_repair_l2_rejected
wheel_sqp_goal_tolerance_exceeded
wheel_sqp_resource_budget_exceeded
planning_deadline_expired
wheel_sqp_identity_mismatch
wheel_sqp_internal_error
```

失败只携带审计证据和 telemetry，不携带可执行的部分路线。deadline 到达优先映射为 typed timeout；身份、非有限数和内部合同失败不得降级为普通不可达。

## 12. 确定性与可审计性

相同批准运行环境、代码版本、输入 bytes、profile 和 determinism seed 必须产生相同：

- 走廊数量、顺序、拓扑签名和哈希；
- 初值模式序列；
- 规范化轨迹 segments；
- L2 结果、反例和修复选择；
- 成功/失败分类、原因和路线决策哈希。

墙钟耗时和环境观测 telemetry 不参与决策哈希。求解器使用单线程、固定稀疏/变量顺序、固定容差和稳定 tie-break；并行优化默认不属于本能力。若未来加入并行，必须使用新能力 revision 并证明语义等价。

## 13. 测试策略

### 13.1 数学测试

- 差速解析积分和 `omega->0` 连续极限。
- 前进、倒车、原地旋转和方向切换。
- 角度 wrap、持续时间、速度和加减速边界。
- 规范化前后解析 replay 与残差检查。

### 13.2 模块测试

- 最多三条不同拓扑签名走廊及稳定排序。
- 无近似重复走廊占用配额。
- 动态规划初值的前进/倒车/旋转选择。
- SQP feasible、infeasible、iteration、memory、numeric 和 deadline 结果。
- L2 首反例稳定性和每走廊恰好一次修复。

### 13.3 对抗安全测试

- shooting nodes 均安全但节点之间碰撞。
- 旋转矩形角点扫到障碍、unknown 或边界。
- 单点切触、极短危险区间和角度 wrap 附近碰撞。
- `30deg` 通过、`nextafter(30deg,+inf)` 拒绝。
- 端点安全但中间扫掠不安全。
- interval/cell/memory cap 到达时 fail closed。
- 规范化后从安全变为不安全时拒绝。

### 13.4 端到端测试

- timeout 分别发生在走廊、SQP、规范化、L2、修复和编码阶段。
- deadline 后不得出现 late success。
- 所有失败均无部分路线。
- 真实末端处于 `0.25m/5deg` 边界时通过，边界外一个浮点邻值拒绝。
- v1、当前 v2 Hybrid A*、PPO 和四项发布边界不漂移。

## 14. Benchmark 与验收

正式对照矩阵至少包含：

- v1 默认 A*；
- 当前 v2 wheel Hybrid A*；
- 新 `wheel_kinematic_corridor_sqp/v1` 单走廊；
- 三走廊；
- 三走廊加一次 L2 反例修复。

场景覆盖开阔地、S 形障碍、狭窄通道、死胡同、需要倒车、仅需旋转、多个拓扑绕行、unknown 边界、坡度边界、Standard 和 Kilometer。正式门保持：

- oracle unsafe false-positive：`0`；
- 返回路线完整 L2 通过率：`100%`；
- oracle-reachable 请求成功率：`>=99%`；
- Standard p95：`<=250ms`；
- Kilometer p95：`<=750ms`；
- 单请求端到端硬期限：`<=2s`；
- 小型 exact map 上首条返回路线资源成本：`<=1.10 * optimum`；
- 相同输入重复运行的决策字段和路线哈希一致。

`oracle-reachable` 与 `optimum` 必须来自独立、高预算、同能力边界的离线 oracle/认证标签，不能由被测 SQP 自标注。资源成本比较使用同一个版本化时间/相对能耗定义；无法建立独立标签的行不得进入正式成功率或质量分母。

运行时不等待最优性证明。若首条通过路线的离线质量门不合格，应调整版本化走廊排序、初值或 SQP 合同并重新 benchmark，不能在请求内继续改善后才返回。

## 15. 集成与版本边界

- 新能力使用新的 capability/profile/source/validator IDs 和 typed trajectory segment。
- 现有 `WheelMotionPrimitiveV2`、v2 Hybrid A* profile 和 v1 public API 保持兼容。
- 不允许 SQP 失败后静默切换 Hybrid A*；上层需要 Hybrid A* 时必须显式请求其既有 profile。
- 不发布 checkpoint，不替换 default policy，不连接 executor，不启动 canary。
- 不把 synthetic terrain proxy 升级为物理障碍真值。
- 实现与正式 benchmark 必须另行计划和授权；本文批准不等于运行或发布批准。

## 16. 未采用方案

### 16.1 State Lattice + MHA*

其确定性与可执行性较强，但本轮用户选择连续走廊优化，以直接输出带时间和控制约束的轨迹。

### 16.2 增强 Hybrid A*

它继续作为现有 v2 reference。新能力不继承 continuous pose 到离散 closed key 的 dominance 语义，也不把它作为失败 fallback。

### 16.3 MPPI

MPPI 适合局部跟踪与动态避障，但不作为本静态、完整路线、强确定性和连续 L2 场景的权威全局求解器。
