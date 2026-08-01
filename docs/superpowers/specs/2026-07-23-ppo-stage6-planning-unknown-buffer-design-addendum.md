# Stage 6 规划未知区缓冲与 U74 Warm-start 设计补充

日期：2026-07-23  
状态：用户已批准  
适用范围：PPO High-Resolution Frontier Map Exploration v1 / Stage 6A 单 seed

## 1. 已批准决策

本补充只冻结以下变更：

- 保持车辆物理安全常量不变：
  - `vehicle_radius_m=0.4215874761`
  - `safety_margin_m=0.10`
  - `min_clearance_m=0.5215874761`
  - `traversability_threshold=0.50`
  - `max_traversable_slope_deg=30.0`
- 保持现有 `observed_safe_mask` 的物理语义不变。
- 新增规划专用 `planning_safe_mask`：
  - 以 `observed_safe_mask` 为基础；
  - 排除与任一未知格中心的欧氏距离严格小于 `0.75m` 的栅格；
  - 在 0.5m 栅格上，0.5m 正交邻格和约 0.707m 斜角邻格均被排除，1.0m 邻格不被该规则排除；
  - 只使用当前已观测状态构造，禁止读取隐藏 highres truth。
- episode reset 先执行一次固定的局部安全扫描：
  - range=`0.75m`
  - FOV=`360°`
  - ray step=`1°`
  - 使用与现有传感器相同的 LOS、hard obstacle、坡度和栅格坐标语义；
  - 不计 reward、step 或 stagnation；
  - 随后仍执行原有 initial theta 下的 `20m/90°` exploration scan。
- frontier 候选 endpoint 必须落在 `planning_safe_mask` 内。
- A* 必须把 `planning_safe_mask` 作为搜索时的 `CostGrid.passable_mask`，不是规划完成后才检查。
- 返回的整条 `path_cells` 必须再次逐格核验 `planning_safe_mask`；A* 不进行 footprint 二次膨胀。
- 不新增“无安全候选时原地转向扫描”或“后退观测点”动作；无候选时保持现有 `no_candidate_done`。
- endpoint/path 失败必须持久化可区分的原因，至少区分：
  - endpoint 不满足物理 `observed_safe_mask`
  - endpoint 触发未知区 0.75m 缓冲
  - path 不满足物理 `observed_safe_mask`
  - path 触发未知区 0.75m 缓冲
  - A* 无可达路径
- 不改变 PolicyObservation channel、22 维 frontier feature、网络结构、动作张量或 PPO 超参数。

## 2. reset 局部安全扫描的必要性

只应用 0.75m 未知区缓冲而保留原有 `20m/90°` reset scan 时，16/16 个冻结 Standard validation reset pose 都会因周围仍有未知格而落在规划缓冲区内。

加入 `0.75m/360°` reset 局部安全扫描后的只读可行性诊断为：

```text
reset pose planning-safe:                 16/16
reset pose has planning-safe egress cell: 16/16
has immediately reachable candidate:     11/16
reachable candidates:                    48/120
```

其余 5 个场景按已批准合同进入 `no_candidate_done`，不通过新增 fallback 绕过。

## 3. 两类 mask 的边界

`observed_safe_mask` 回答“基于已经观测到的障碍、坡度、可通行度和真实车辆净空，这个格子是否物理安全”。未知格本身不可通行，但未知格不会按 0.75m 额外扩张。

`planning_safe_mask` 回答“在物理安全基础上，这个格子是否还远离未知区”。其确定性定义为：

```text
planning_safe_mask =
    observed_safe_mask
    AND NOT within_center_distance_lt_0.75m(unknown_mask)
```

未知区缓冲必须在每次 observation 更新后重新计算。它不得覆盖、改写或冒充物理安全结论，也不得把规划风险记录为真实碰撞或真实 obstacle violation。

## 4. reset 顺序与观测语义

reset 固定顺序如下：

1. 创建空的 observed state。
2. 在起点执行 `0.75m/360°/1°` 局部安全扫描。
3. 在相同起点和 initial theta 执行原有 `20m/90°/1°` exploration scan。
4. 重算 `observed_safe_mask`。
5. 从 observed-only 数据重算 `planning_safe_mask`。
6. 生成 frontier 候选和初始 PolicyObservation。

两个扫描都必须有独立、可审计的 diagnostics；不得把两次扫描伪装成一次 20m/90° 扫描。reset 后仍为 `step_count=0`、累计 reward=0。

## 5. U74 Warm-start 合同

父运行固定为：

```text
parent_run_id = s6-standard-single-r1-20260718T220434Z
parent_seed = 20260716
parent_update = 74
parent_checkpoint =
  D:/xunce/out/ppo_frontier/s6-standard-single-r1-20260718T220434Z/
  s6/checkpoints/seed-20260716/update-00000074/checkpoint.pt
parent_checkpoint_sha256 =
  ce9ea8cb047b5acc0ffc4f5d63084f3d54312cf55ecb0b92b8fc20bf43791553
parent_manifest_sha256 =
  6c0e40d2dd9ea489f4c67a6a43868dc80751f07961f5db8ccf5c804213e340c5
parent_complete_sha256 =
  6cd6e6210283fc71a5205a727f82c07ff448458d4ac06ad8080277e2502960ef
parent_policy_state_sha256 =
  c1650b7bed6fa22387ac5b5fdb9d80a6aa0baf3a1d425e868d81586070c5db20
parent_config_sha256 =
  7b1c37673c105c34e7bd71d04d4f508d3a167cdf77426b0b35c7aff86ba8adae
parent_lineage_sha256 =
  c7a0a7bf98b5aa4a39094ec01a2a0031d87ad8fc71060323474b20ebb134530c
```

U74 必须有完整 marker、accepted update transaction 和匹配的 lineage。任何 SHA、update、seed、run、complete marker 或 journal 漂移都必须 fail closed。

该切换不是旧 run 的 exact resume，而是新语义下的新 lineage warm-start：

- 使用新 run id 和新的 effective config hash；
- 第一个 child update 为 U75，最终目标仍为 U100；
- 导入 U74 的 model、optimizer 和 normalization stats；
- 为保持确定性连续性，导入 U74 的 RNG 与 scenario sampler 状态，但只把 sampler 状态用于选择下一批新 reset 场景；
- 严禁导入 U74 的 8 个 active vector-env episode state；
- 不导入父 run 的 validation-best、best record 或 eval metrics；child 只允许用 U80、U90、U100 的新语义 validation 选择自己的 best；
- 严禁复用 U75 attempt1 的 partial rollout、candidate snapshot、old logprob 或任何未 accepted 数据；
- 8 个环境必须按新 reset 和 planning mask 规则全部重新创建并 reset；
- child checkpoint lineage 必须记录父 run、父 update、父 checkpoint/manifest SHA、warm-start schema 和新 effective config SHA；
- U80、U90、U100 validation 及最终 test/unseen/baseline 必须使用新语义；
- 报告必须明确：新语义实际训练 26 updates，前 74 updates 只作为父 checkpoint 预训练来源，不得表述为“100 updates 全部采用 0.75m 规则”。

## 6. 非目标与长期边界

- 不添加 fallback 动作。
- 不改变 reward、done、GAE、PPO loss、网络或动作空间。
- 不修改真实车辆安全常量。
- 不启用 planner footprint inflation。
- 不发布 checkpoint。
- 不替换 default policy。
- 不连接 executor。
- 不启动 canary。
- 只运行 seed `20260716`；额外 seeds 只有用户明确说“追加”后才运行。
- synthetic highres terrain/rocks/craters 仍是 `synthetic_terrain_obstacle_proxy/v1`，`physical_obstacle_cells_written=false`。
