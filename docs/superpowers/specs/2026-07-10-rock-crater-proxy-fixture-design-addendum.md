# 岩石与陨石坑程序化 Proxy Fixture 设计增补

**日期：** 2026-07-10

**状态：** 用户已批准设计，等待书面规格复核

**适用范围：** PPO High-Resolution Frontier Map Exploration v1 的 Smoke、Standard 与 Kilometer 程序化 highres truth proxy

**原算法基线：** `2026-07-09-ppo-highres-frontier-map-exploration-design.md`

## 1. 定位与优先级

本增补只冻结程序化 highres truth proxy 的地形形态与数据生成合同，不改变 PPO 网络、动作空间、reward、A*、传感器、成功谓词或训练预算。原设计仍是唯一算法基线；涉及 proxy fixture 形态时，以本增补为补充约束。

所有生成地形必须继续标记：

```text
synthetic_source_kind=synthetic_terrain_obstacle_proxy/v1
physical_obstacle_cells_written=false
```

岩石、陨石坑及其派生栅格均是确定性程序化 proxy，不得表述为真实月面 0.5m 测量、真实岩块目录、真实陨石坑目录或物理障碍标注。

## 2. 已批准的产品决策

1. 陨石坑采用“形态生成后由坡度与可通行度判定”的语义。
2. 陨石坑本身不得整坑写入 `hard_obstacle`，也不得把坑缘或内壁直接硬编码为障碍。
3. 岩石使用椭圆隆起与 synthetic hard-obstacle core。
4. 对象必须空间零散分布，不能再以抽象直线墙或矩形块作为主要 Smoke 障碍形态。
5. Smoke 固定使用中密度；训练样本均衡混合低、中、高三档密度。
6. validation、test、unseen 与 Kilometer 压力测试按场景数近似等量分层三档密度，任意两档数量差不得超过 1。

## 3. 模块边界

新增内部模块 `env/terrain_proxy.py`，职责限定为：

```text
ProceduralTerrainProxyGenerator.generate(
    base_height,
    geometry,
    start_pose,
    seed,
    density_profile,
) -> TerrainProxyBundle
```

`TerrainProxyBundle` 包含：

- 合成后的 highres `height`；
- 仅岩石 core 写入的 `hard_obstacle`；
- 从最终高度场派生的 `slope_deg`；
- 从坡度派生的 `traversability`；
- 不进入策略观测的 immutable object catalog；
- 生成器版本、seed、密度档、重试次数与各层 hash。

`ScenarioSource` 负责选择场景 seed、密度档、基础高程与起点，然后调用生成器。Sensor、coverage、frontier、planner adapter 与 policy observation 不读取 object catalog。

## 4. 确定性与随机数合同

- 生成器版本固定为 `procedural_lunar_rock_crater_proxy/v1`。
- 使用 NumPy `PCG64`，禁止 Python 进程随机化的 `hash()`。
- 场景 seed 由 canonical UTF-8 字节计算 SHA-256 后截取 128 bit：

```text
generator_version | scenario_key | split | parent_roi | base_seed | density_profile
```

- 相同输入必须逐字节生成相同的 object catalog 和所有 truth layer。
- 不同 `scenario_key` 或 `base_seed` 必须生成不同 catalog hash。
- 程序化 seed 不得造成同一父 ROI 跨 split；Stage 2 的父 ROI/macro-tile 空间分割仍先于场景裁剪和 proxy seed 派生。

## 5. 密度档

计数基准面积为 `4096m²`，即 64m × 64m Smoke。设 `scale = area_m2 / 4096`，缩放后的下界为 `max(1, ceil(base_low * scale))`，上界为 `max(scaled_low, floor(base_high * scale + 0.5))`，再在闭区间内做确定性整数抽样。

| 密度档 | 岩石数量 / 4096m² | 陨石坑数量 / 4096m² |
|---|---:|---:|
| low | 10–18 | 2–3 |
| medium | 24–36 | 4–6 |
| high | 45–64 | 7–10 |

Smoke 使用固定 seed 和 medium 档。训练 sampler 对 low、medium、high 使用 `1/3, 1/3, 1/3`；评估集合按确定性分层分配，避免密度构成随运行 seed 漂移。

若目标计数无法在空间约束内完整放置，生成器不得静默减少数量或放宽安全约束；必须更换确定性 attempt seed，整场最多重试 64 次，之后 fail closed。

## 6. 岩石形态

每个岩石对象包含中心、等效半径、长短轴、朝向和高度。若长短轴比为 `q`、等效半径为 `r`，则半长轴 `a=r*sqrt(q)`、半短轴 `b=r/sqrt(q)`：

- 等效半径：`0.5–1.5m`；
- 长短轴比：`1.0–1.8`；
- 高度：`0.2–1.0m`；
- 朝向：`[-π, π)`；
- 参数在各自范围内连续确定性采样。

在旋转椭圆坐标中定义归一化半径 `ρ`。当 `ρ <= 1` 时，岩石高度增量为：

```text
delta_h = rock_height * (1 - ρ²)²
```

当 `ρ > 1` 时高度增量为 0。`ρ <= 0.85` 的 core 栅格写入 synthetic `hard_obstacle=true`；外围只通过高度、坡度和 traversability 影响安全性。栅格化按 cell center 判断，并保证每个 catalog 岩石至少覆盖一个 core cell。

## 7. 陨石坑形态与安全语义

每个陨石坑包含中心、半径、深度、坑缘高度和坑缘宽度：

- 外形半径 `R`：`2–6m`；
- 深度 `D`：`0.15R–0.30R`；
- 坑缘高度 `H`：`0.05R–0.12R`；
- 归一化坑缘宽度 `W`：`0.12–0.25`。

设 `u = distance / R`。高度增量由 bowl 与 rim 叠加：

```text
bowl(u) = -D * (1 - u²)²,                       0 <= u <= 1
rim(u)  =  H * exp(-((u - 1) / W)²),            0 <= u <= 1.35
delta_h = bowl(u) + rim(u)
```

陨石坑对 `hard_obstacle` 的贡献始终为 false。最终高度场完成后，统一使用 0.5m cell-center spacing、等价于 `numpy.gradient(..., edge_order=1)` 的有限差分计算：

```text
slope_deg = degrees(atan(hypot(dz_dx, dz_dy)))
traversability = clip(1 - slope_deg / 60, 0, 1)
```

安全判定仍只使用既有合同：

```text
traversability >= 0.50
slope_deg <= 30.0
min_clearance_m = 0.5215874761
```

因此坑底、缓坡或局部平坦坑缘可以通行；陡峭内壁和坑缘会由既有阈值自然禁行。

## 8. 空间分布与保护约束

对象中心使用确定性四象限分层拒绝采样。先放置陨石坑，再放置岩石；每轮优先选择当前对象数最少的可用象限，在象限内生成抖动候选并做确定性随机排序，从而避免全部聚集在局部区域。原 Smoke fixture 的抽象直线墙和矩形障碍不再写入最终 truth layer。

必须满足：

- 起点保护半径 `6m`；任何对象外边界不得侵入；
- 对象外边界不得越过地图边界；
- 岩石外边界使用半长轴 `a`，陨石坑外边界使用包含完整 rim 的 `1.35R`；
- rock-rock 中心距离至少为两者半长轴和的 `1.10` 倍；
- crater-crater 中心距离至少为两者 `1.35R` 外边界半径和的 `1.05` 倍；
- rock-crater 不得发生 footprint 重叠；
- medium/high 场景的四个象限均须包含对象，任一象限的对象数不得超过总数的 40%；low 场景至少覆盖三个象限；
- 单对象最多尝试 256 个候选位置，整场最多 64 个 attempt seed。

不得为某个策略预埋隐藏直线路径或使用 policy rollout 结果反向选择对象位置。

## 9. 场景结构验收

每个生成场景必须在进入 sampler 前通过：

1. 起点 cell 在环境执行唯一一次 footprint inflation 后的最终安全 mask 内；planner adapter 不再 inflation；
2. `safe_free_cell_count > 0`；
3. 起点连通的 safe component 占全部 safe-free cells 至少 70%；
4. exact coverable denominator 非零；
5. reset 初始扫描后 coverage `< 0.99`；
6. reset 后至少存在一个 observed-only frontier candidate；
7. 所有 truth layer、catalog 参数和派生指标均 finite；
8. catalog 数量、尺度、间距、象限分布、保护区与 layer hash 全部可复算。

结构验收可读取完整 truth，但只属于离线场景构建与审计，不进入 `ObservationBuilder`、policy 或 baseline 决策。Smoke 固定场景还必须通过 Stage 1 的 10-episode `>=0.99` 闭环；Standard/Kilometer 不得按 PPO 或 baseline 表现筛选场景。

## 10. 遮挡合同

本增补沿用原算法基线的 `two_dimensional_grid_line_of_sight/v1`，不升级为连续 3D DEM visibility。每条 DDA ray 的 blocker 定义保持：

```text
los_blocker_cell =
    hard_obstacle_cell
    OR slope_deg > 30.0
```

对象对应关系为：

- 岩石的 synthetic hard-obstacle core 必须遮挡 LOS；射线包含并 reveal 第一格岩石 blocker，然后停止；其后单元在该 sample 中保持 unknown；
- 陨石坑不写 hard obstacle，但陡峭内壁或坑缘若 `slope_deg > 30.0`，必须以 slope blocker 遮挡 LOS；
- 坑底、缓坡内壁或缓坡坑缘若未超过 slope blocker 阈值，不得仅凭对象语义强制遮挡；
- `traversability < 0.50` 不单独新增 LOS blocker。v1 的 crater traversability 由 slope 单调派生，因此安全阈值与 slope blocker 对齐；
- 高度仍写入 observed highres state，但 v1 不做观测点高度、目标高度与中间地形剖面的连续 3D 插值；低坡但较高地形不会仅因高程形成遮挡，这是必须在报告中披露的 proxy 限制；
- execution sensor 使用 truth blocker 决定实际 reveal；frontier potential-gain 只能使用当前已观测 blocker，禁止利用未观测 rock/crater truth；
- exact coverable mask 可在环境初始化时使用 truth blocker，但只能作为 denominator，不进入 policy observation 或 candidate generation。

岩石与陨石坑的遮挡测试必须使用至少三格共线的独立小地图，验证第一 blocker 可见、其后 cell unknown、相邻无遮挡 ray 仍可见，并分别覆盖 hard-obstacle rock、slope-blocked crater rim 和 non-blocking gentle crater 三种情况。

## 11. Observation 与泄漏边界

- Smoke lowres prior 保持 `constant_neutral/v1`，不得编码 rock/crater catalog、中心、半径、hard mask 或未观测 slope。
- Standard/Kilometer 的真实低分辨率 prior 只来自既有 USGS LRO DEM 管线；程序化 highres 对象不得反向写入 prior。
- rock/crater 信息只能在 sensor LOS 成功 reveal 后进入 observed highres state。
- `ObservationBuilder` 的类型、参数和状态仍不得出现 truth、coverable mask 或 object catalog。
- 修改未观测对象的参数不得改变当前 observation、candidate 或 recommended theta；对象被观测后才允许产生变化。
- 所有 baseline 与 PPO 共用相同场景 truth、sensor、planner、candidate 与预算。

## 12. Provenance 与 hash

每个场景 provenance 至少包含：

```text
proxy_generator_version
synthetic_source_kind
physical_obstacle_cells_written
density_profile
seed_derivation_version
seed_hex
generation_attempt
rock_count
crater_count
object_catalog_sha256
height_sha256
hard_obstacle_sha256
slope_sha256
traversability_sha256
```

object catalog 使用 canonical JSON、稳定字段顺序与十进制有限值编码后计算 SHA-256。场景总 hash 必须绑定 catalog hash、所有 truth layer、prior、geometry、起点与 generator version。

Stage 1 不新增大型 Git artifact；详细 catalog 和可选预览只写 D 盘审计目录。Stage 根的固定 artifact 集不因本增补增加额外文件，必要的聚合 provenance 写入既有 `summary.json` 和 `manifest.json` 绑定的内容。

## 13. TDD 与验收测试

实现前必须先增加失败测试，至少覆盖：

1. Smoke 不再以直线墙/矩形块作为主要障碍，catalog 同时含 rock 与 crater；
2. medium 档计数范围；
3. low/medium/high 的数量区间严格单调且按面积缩放；
4. 相同 key/seed 字节级确定，不同 seed 的 catalog hash 不同；
5. rock 中心高于边缘，椭圆朝向和 core hard mask 正确；
6. crater 中心低于背景、坑缘高于外侧、crater 不写 hard obstacle；
7. crater 陡壁由 slope/traversability 阈值禁行，缓坡区域可通行；
8. 起点 6m 保护区、边界、对象间距和四象限离散分布；
9. 计数无法放置时最多 64 次后 fail closed，不静默降级；
10. catalog/layer/scenario hash 可复算且 mutation 会改变 hash；
11. prior 不含 object geometry；未观测 truth mutation 不改变 observation/candidate；
12. safe component、coverable、初始 coverage 与 reset candidate 结构门；
13. rock hard-obstacle、crater steep-rim 与 gentle-crater 三类 2D LOS 遮挡行为；
14. 原坐标、LOS、single inflation、planner、reward/done、terminal/bootstrap 测试不回归；
15. 10 个连续 Smoke episode 全 finite 且达到既有成功谓词；
16. proxy 标签保持 v1 且 `physical_obstacle_cells_written=false`。

独立 reviewer 必须同时给出规格符合性和代码质量结论，重点审查：形态是否与 catalog 一致、陨石坑是否被偷写 hard obstacle、是否存在 prior/truth 泄漏、场景筛选是否造成策略偏置、hash/provenance 是否完整、以及高密度生成是否存在无界运行或静默降级。

## 14. 非目标

本增补不提供：

- 真实月面岩石/陨石坑统计校准；
- 真实物理碰撞标签；
- rock/crater 语义分割输入通道；
- 新 reward、网络层、动作或候选特征；
- Stage 2 数据 split、PPO 训练或 Kilometer 性能结论。

若未来使用真实岩石/陨石坑目录或物理仿真，必须新增版本和独立证据，不能沿用本 v1 proxy 的结论。
