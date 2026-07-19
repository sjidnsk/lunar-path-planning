# 多平台能力感知路径规划 v2：Gate5A 月面弹道与显式 Proxy Profile 设计

## 1. 文档状态

- 状态：已获用户口头批准，进入书面复核。
- 日期：2026-07-18。
- 分支：`codex/multiplatform-path-planner-v2`。
- 前置父仓提交：`f88055f207ff1ad06a801731ddcfc84016297181`。
- 前置 `path-planner` 提交：`7d5c4dfc8e2a374855e6d8b962b69466c3353265`。
- 上位设计：`docs/superpowers/specs/2026-07-16-multiplatform-path-planner-v2-design.md`。
- 上位计划：`docs/superpowers/plans/2026-07-16-multiplatform-path-planner-v2-implementation.md` 的 Task 10。

本文只细化 Gate5A。上位设计中已经冻结的 Gate0–6、安全、发布和平台边界继续有效；本文不修改其结论。

本文同时替代并澄清上位计划 Task 10 Step 3 中“provider construction 返回
`hopper_proxy_profile_incomplete`”的措辞：Gate5A 不创建 provider，只由
`audit_hopper_profile_v2()` 披露结构缺口；Task11 的 provider 在 `plan()` 入口的
capability preflight 把该缺口映射为 `UNSUPPORTED_CAPABILITY`。provider 构造器不返回
`PlanningOutcomeV2`，Task11 的文件职责和 fail-closed 边界不变。

## 2. 目标与非目标

Gate5A 提供两个可独立验证的基础单元：

1. 无地形副作用的月面同高无动力弹道与概率落区纯函数。
2. 明确披露正式能力参数缺口的 `HopperProfileV2` 与结构审计。

Gate5A 不实现：

- hopper oracle、provider、搜索、完整路线 L2 或 API 接入；
- 正式 Gate5 runner、config、registry 或 evidence；
- 动态修正、空中观测、推进段、非均匀重力、空气阻力或实机模型；
- checkpoint 发布、default policy 替换、executor 接入或 canary；
- `path_planner.v2` 根命名空间导出。

这些集成工作属于 Task 11 / Gate5B。v1 保持默认，v2 继续显式 opt-in。

## 3. 文件范围

生产代码只允许：

- 新建 `path-planner/src/path_planner/v2/ballistics.py`；
- 修改 `path-planner/src/path_planner/v2/profiles.py`。

测试只允许：

- 新建 `path-planner/tests/test_v2_ballistics.py`；
- 修改 `path-planner/tests/test_v2_profiles.py`。

不得提前修改 `contracts.py`、根 `v2/__init__.py`、provider、oracle、API、validation、runner 或 Gate config。增加 `test_v2_profiles.py` 是对上位 Task 10 的测试职责澄清，不扩大生产范围。

## 4. 公共类型

以下类型定义在 `path_planner.v2.ballistics`，均为 exact、`frozen=True`、`slots=True` dataclass。

```python
@dataclass(frozen=True, slots=True)
class BallisticStartV2:
    x_m: float
    y_m: float
    z_m: float

@dataclass(frozen=True, slots=True)
class BallisticSampleV2:
    time_s: float
    x_m: float
    y_m: float
    z_m: float

@dataclass(frozen=True, slots=True)
class LandingCellMassV2:
    cell: Cell
    probability_mass: float
    in_bounds: bool
```

`BallisticStartV2` 是用户批准的显式 3D 起点合同。`PoseStateV2` 仍只表示 `(x,y,theta)`，不得偷加 `z`，也不得由弹道函数隐式解释 heading。

三个 dataclass 的 `__post_init__` 必须逐字段深校验，不能只信任 exact 外层类型。所有数值拒绝
`bool`、NaN 和无穷；规范化为有限内建 `float`，`-0.0` 规范化为 `0.0`。`time_s` 必须非负；
被返回的 `probability_mass` 必须严格大于 0 且不大于 1；`cell` 必须是 exact `Cell`，其
`x/y` 必须是 exact `int` 且不能是 `bool`；`in_bounds` 必须是 exact `bool`。即使 exact
外层对象曾被 `object.__setattr__` 篡改，公共入口也必须稳定拒绝。

`LandingCellMassV2.in_bounds` 只是随结果携带的边界披露值，由经过深审计的 `geometry`
重新计算。Task11 不得把该重复字段当成安全 authority，必须使用同一冻结 geometry 对
`cell` 再次计算 `in_bounds`。

## 5. 弹道采样合同

公开签名保持上位计划不变，并把 `start` 精确冻结为 `BallisticStartV2`：

```python
def sample_ballistic_arc(
    start: BallisticStartV2,
    speed_mps: float,
    elevation_rad: float,
    azimuth_rad: float,
    g_mps2: float,
    dt_s: float,
) -> Sequence[BallisticSampleV2]: ...
```

实际返回 exact `tuple[BallisticSampleV2, ...]`。输入要求：

- `speed_mps > 0`、`g_mps2 > 0`、`dt_s > 0`；
- `0 < elevation_rad < pi/2`；
- `azimuth_rad` 为任意有限弧度值，不改变或量化调用者输入；
- `start` 必须是 exact `BallisticStartV2`。

模型是从 `start.z_m` 起飞并回到同一绝对世界高度的无动力弹道：

```text
t_f = 2 * speed_mps * sin(elevation_rad) / g_mps2
x(t) = start.x_m + speed_mps * cos(elevation_rad) * cos(azimuth_rad) * t
y(t) = start.y_m + speed_mps * cos(elevation_rad) * sin(azimuth_rad) * t
z(t) = start.z_m + speed_mps * sin(elevation_rad) * t - 0.5 * g_mps2 * t^2
```

采样要求：

- 首样本精确为 `(t=0, start.x_m, start.y_m, start.z_m)`；
- 末样本精确使用 `t_f`，并把 `z_m` 规范为 `start.z_m`，避免浮点残差；
- 时间严格递增，相邻时间差不大于 `dt_s`；
- 无 representability repair 时，非整除只缩短最后一个 nominal interval；一旦需要 4-ULP correction、
  bounded bridge/repair 或 endpoint anchor ownership，则以后续精确规则为准；始终不丢 endpoint、不重复末样本；
- `3.0m/s @ 45deg @ g=1.62m/s^2` 的水平距离为 `9/1.62`，约 `5.56m`；
- 从 exact binary64 比值计算 `interval_count = ceil(t_f / dt_s)`；它是最小 interval count，
  `interval_count + 1` 只是最小 sample count，不必然等于实际 sample count；
- 独立冻结 nominal anchor `i * dt_s`。nominal 最多可向下移动 4 ULP；若 binary64 表示无法同时
  满足严格递增和 `gap <= dt_s`，只能插入固定数量的有界局部 repair sample；
- 最后 nominal 与 exact endpoint 碰撞或距离 endpoint 不超过 4 ULP 时，endpoint 拥有该 anchor；任何
  必需 bridge 都必须有界并计入公开 cap；
- 最终实际 sample count 必须满足 `exact lower bound <= actual count <= 100_000`。所有 repair 都计入
  `MAX_BALLISTIC_SAMPLES_V2`，并在 append 或 allocation 前检查，而不仅在 `ceil`、整数转换或初始分配前检查；
- 速度分量、`t_f`、`t_f / dt_s`、水平落点、最高点和每个派生 sample 坐标都必须有限，且
  `t_f > 0`；正的派生 scalar、x/y direction-to-velocity product、x/y velocity-to-displacement product，
  以及被大 origin 吸收的非零 coordinate offset 都必须 fail-closed 地可表示。有限输入导致上溢、下溢或
  不可表示结果时稳定抛出 `ValueError`，不得泄漏 `OverflowError`、生成 `inf` 或产生重复端点。

`BallisticSampleV2` 不重复保存速度。Task11 可从冻结输入和时间解析速度；Gate5A 不提前定义着陆停止模型。

## 6. 一维正态区间质量

```python
def normal_interval_mass(
    lo: float,
    hi: float,
    mean: float,
    sigma: float,
) -> float: ...
```

合同如下：

- 全部输入为有限实数且拒绝 `bool`；
- `sigma > 0`、`lo <= hi`；
- `lo == hi` 精确返回 `0.0`；
- 使用数值稳定的分段标准正态计算，不采样、不依赖随机数或 SciPy；同侧尾部必须使用
  `erfc`/生存函数差，跨均值区间才可使用 `erf`/CDF 组合，禁止直接用两个接近 1 的 CDF
  相减；
- 仅为抵消浮点舍入把结果夹到 `[0,1]`，不得掩盖非法输入；
- 对称区间、平移和镜像输入必须给出数学等价结果。

## 7. 二维概率落区

公开签名精确解释为：

```python
def landing_zone_cells(
    mean_xy: WorldPoint,
    sigma_m: float,
    probability_threshold: float,
    geometry: FineGridGeometryV2,
) -> Sequence[LandingCellMassV2]: ...
```

实际返回 exact tuple。`mean_xy`、`geometry` 必须是 exact 类型。入口必须逐字段深审计：
`mean_xy.x/y` 是有限实数且拒绝 `bool`；geometry 的 `width/height` 是非 bool 的 exact 正
`int`，`origin` 是 exact 二元 tuple 且元素是有限实数、`resolution_m` 是 exact `float 0.5`、
`frame_id` 是 exact 非空 `str`。这些字段必须能重建出等价的 canonical
`FineGridGeometryV2`。`sigma_m > 0`，且 `0 < probability_threshold < 1`。

概率模型固定为 x/y 独立、同方差、未条件化的二维高斯：

```text
P(cell) = P(x_lo <= X < x_hi) * P(y_lo <= Y < y_hi)
```

每个 cell 的世界边界由 `geometry.origin` 与 `geometry.resolution_m` 外推。候选域是整个整数 cell 平面 `Cell(x,y), x,y in Z`，不是仅地图内 cell。

返回集合定义为全局排序后的最短前缀，使未归一化累计质量首次达到或超过 `probability_threshold`。全局排序 key 固定为：

```text
(-probability_mass, squared_distance(cell_center, mean_xy), cell.y, cell.x)
```

其中地图外 cell 的中心仍按相同网格外推。要求：

- 选中的地图外 cell 必须保留，`in_bounds=False`；
- 不裁剪、clamp 或丢弃地图外概率；
- 不按地图内总质量重新归一化；
- 不把“单个 cell 质量达到 0.99”误作“落区累计质量达到 0.99”；
- 等质量时必须按固定 key 决定顺序；
- 累计质量必须使用 `math.fsum`，禁止依赖迭代顺序的 naive `sum`；
- 只有同时满足以下三个终止不变量才能返回：`fsum(prefix) >= threshold`、删除末 cell 后的
  `fsum(prefix[:-1]) < threshold`、所有未见 cell 的严格质量上界小于末 cell 的
  `probability_mass`；严格小于同时排除遗漏同质量 tie 对前缀顺序的影响；
- 实现必须用确定性扩展和解析尾界证明上述不变量。
  landing candidate traversal 只遍历同心正方形 perimeter；每个 cell 恰好访问一次，禁止重复 full-square
  rescan。对 N 个已评估 candidate，生成/访问工作为 Theta(N)，保留的逐 cell state 为 O(N)（排序仍可为
  O(N log N)）；
  `MAX_LANDING_ZONE_CANDIDATES_V2 = 1_000_000` 统计进入生成、质量求值、排序、visited 或
  frontier 任一状态的不同 cell 总数；
  每个 cell 只计一次，任何追加或扩容前检查，不得只统计 selected cells，也不得保留超过
  上限的逐 cell 辅助状态；
- 若在上限内无法同时证明终止不变量并达到阈值，则稳定抛出 `ValueError`，不能静默截断或
  返回 partial zone。所有 world/index 转换、cell 边界、中心、距离和质量中间量也必须保持
  有限且可表示，否则在整数转换或容器扩展前稳定拒绝。

因此，99% 落区只要有任何 selected cell 越界，Task11 就能 fail closed；边界外概率不会被重新条件化成地图内的“安全 100%”。

## 8. Hopper Profile 合同

`profiles.py` 新增：

```python
HOPPER_LUNAR_BALLISTIC_CAPABILITY_REVISION_V2 = (
    "simulation_proxy_lunar_ballistic/v1"
)
HOPPER_PROXY_PROFILE_INCOMPLETE_REASON_V2 = "hopper_proxy_profile_incomplete"
```

`HopperProfileV2` 为 exact、frozen、slotted dataclass。公共字段名、顺序、类型和默认值精确冻结为：

```python
@dataclass(frozen=True, slots=True)
class HopperProfileV2:
    profile: PlatformProfileV2
    gravity_mps2: float = 1.62
    launch_speeds_mps: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0)
    launch_elevations_rad: tuple[float, ...] = (pi / 6.0, pi / 4.0, pi / 3.0)
    azimuth_direction_count: int = 16
    landing_sigma_range_scale: float = 0.05
    landing_sigma_offset_m: float = 0.05
    max_landing_slope_deg: float = 15.0
    landing_probability_threshold: float = 0.99
    midcourse_correction_enabled: bool = False
    inflight_observation_enabled: bool = False
    body_envelope_radius_m: float | None = None
    launch_reference_height_m: float | None = None
    arc_clearance_margin_m: float | None = None
    landing_footprint_radius_m: float | None = None
    stop_condition: str | None = None
    energy_model: str | None = None
```

`profile` 必须是 exact `PlatformProfileV2`，且必须通过逐字段重建或等价深审计；只检查外层
exact 类型不合格。它必须满足：

- `profile_id`、`capability_revision`、`schema_version` 是 exact 内建非空 `str`；
- `platform_kind` 是 exact `PlatformKindV2`；
- `simulation_proxy` 是 exact `bool`；
- slope 与两个 goal tolerance 是 exact 内建有限 `float`；
- `platform_kind is PlatformKindV2.HOPPER`；
- `simulation_proxy is True`；
- capability revision 精确等于上述版本；
- `max_traversable_slope_deg == 30.0`；
- goal position/heading tolerance 均为 `0.0`。

已批准的 proxy 参数作为 exact defaults：

```text
gravity_mps2 = 1.62
launch_speeds_mps = (1.5, 2.0, 2.5, 3.0)
launch_elevations_rad = (pi/6, pi/4, pi/3)
azimuth_direction_count = 16
landing_sigma_range_scale = 0.05
landing_sigma_offset_m = 0.05
max_landing_slope_deg = 15.0
landing_probability_threshold = 0.99
midcourse_correction_enabled = False
inflight_observation_enabled = False
```

这些字段不仅是默认值，也是不可覆盖的冻结合同：各 float 必须是 exact 内建 `float` 且与
canonical 值精确相等；速度/仰角必须是 exact `tuple`，其元素必须是 exact 内建 `float`
且逐值相等；方向数必须是 exact `int` 16 且不能是 `bool`；两个开关必须是 exact `bool`
`False`。容器替换、派生类型、bool 冒充数字或任一 `nextafter` 漂移都必须拒绝。这些字段
不得被解释为实机认证。

六个未获正式批准的能力字段保持 nullable 且无安全默认值：

```text
body_envelope_radius_m: float | None = None
launch_reference_height_m: float | None = None
arc_clearance_margin_m: float | None = None
landing_footprint_radius_m: float | None = None
stop_condition: str | None = None
energy_model: str | None = None
```

四个数值若显式提供必须是有限 exact float 且严格大于 0；两个字符串若显式提供必须是
exact、非空、匹配 `/v[1-9][0-9]*$` 的版本化 proxy model ID。`0.0`、空字符串、bool、
派生伪类型、NaN 和无穷均不能把缺口变成完整能力。

构造 incomplete `HopperProfileV2` 是合法的，因为默认 repo profile 必须能显式表达“尚未冻结”。构造函数不得把 `None` 替换为零值或隐藏默认模型。

## 9. Profile 结构审计

`profiles.py` 新增 exact、frozen、slotted `HopperProfileAuditV2`：

```python
@dataclass(frozen=True, slots=True)
class HopperProfileAuditV2:
    complete: bool
    reason_code: str | None
    missing_fields: tuple[str, ...]
```

以及：

```python
def audit_hopper_profile_v2(profile: HopperProfileV2) -> HopperProfileAuditV2: ...
```

`audit_hopper_profile_v2()` 只接受 exact `HopperProfileV2`，并重复执行上述逐字段深审计，
不能信任冻结对象未被篡改。`HopperProfileAuditV2.__post_init__` 也必须冻结自洽不变量：
`complete` 为 exact `bool`；`missing_fields` 为 exact tuple、只含下列 canonical 字段且无重复
并保持固定顺序；`complete=True` 当且仅当 reason 为 `None` 且缺项为空；
`complete=False` 当且仅当 reason 为固定 incomplete code 且缺项非空。伪造或自相矛盾的
audit 对象必须稳定拒绝。

缺口顺序固定为：

```text
body_envelope_radius_m
launch_reference_height_m
arc_clearance_margin_m
landing_footprint_radius_m
stop_condition
energy_model
```

任一字段为 `None` 时：

- `complete=False`；
- `reason_code="hopper_proxy_profile_incomplete"`；
- `missing_fields` 只包含实际缺失项并保持上述顺序。

六项结构齐全时，audit 只返回 `complete=True`、`reason_code=None`、空缺项。这个结果只证明结构齐全，不证明 model ID 已被 Task11 provider 支持，也不构成安全、性能、Gate pass 或实机声明。Task11 必须再次校验受支持的 stop/energy model ID，并在 incomplete 时返回：

```text
FailureCategoryV2.UNSUPPORTED_CAPABILITY
reason_code = hopper_proxy_profile_incomplete
```

Gate5A 不创建 provider，因此不伪造 PlanningOutcome。

## 10. 连续安全边界

Gate5A 的 sample tuple 是确定性物理轨迹表示，不是安全证明。Task11 必须：

- 对每对相邻 sample 做连续 XY supercover/capsule 检查；
- 使用 `body_envelope_radius_m + arc_clearance_margin_m` 扩张，而非只查中心线；
- 对整个时间区间使用保守高度下界检查地形净空；
- 检查全弧边界、完整概率落区、着陆 footprint、坡度、目标 theta 与停止条件；
- unknown、越界或未支持模型一律 fail closed；
- 禁止用更密采样替代连续保守包络证明。

本文不提前冻结 Task11 的具体 supercover 实现，但冻结“只查 sample 点不合格”。

## 11. 错误处理与确定性

- 三个纯函数无文件 IO、随机数、全局可变状态或 terrain truth 访问。
- 无效调用使用稳定 `TypeError`/`ValueError`；不把编程错误转换成空结果。
- 所有返回容器为 tuple，稳定顺序由合同定义。
- 先做深层输入、派生有限性和资源上限校验，再做整数转换或分配大容器。
- 不使用平台相关哈希顺序、集合迭代顺序或非确定性并行。
- canonical serializer 已支持 dataclass；Gate5A 不修改 serialization。

## 12. TDD 与验收矩阵

### 12.1 弹道

- RED 覆盖首尾精确、同高着陆、时间单调、最后短区间、四象限方位与重复字节稳定。
- 验证 `3m/s @ 45deg` 水平距离 `9/1.62`。
- 参数化拒绝 bool、NaN、无穷、非正 speed/g/dt、无效仰角、超样本上限，以及会让
  `t_f`、exact-rational lower bound、sample count、落点或最高点不可表示的极端有限组合。
- 覆盖 exact binary64 lower bound、4-ULP nominal downward repair、endpoint anchor ownership、有界 bridge，
  以及每次 repair append/allocation 前的 public-cap fail-closed 检查。
- 覆盖正派生 scalar、x/y direction-to-velocity product、x/y velocity-to-displacement product 与大 origin
  吸收非零 coordinate offset 时的 representability guard。
- 覆盖 exact 外层 `BallisticStartV2` 被篡改内部字段的 deep-validation 拒绝路径。

### 12.2 正态质量

- 覆盖零宽区间、对称性、镜像/平移等价、已知区间质量和结果边界；特别覆盖标准正态
  `[8,9]` 与 `[9,10]` 尾区间，防止直接 CDF/`erf` 相减变成错误零质量。
- 拒绝反向区间、非正 sigma、bool 与非有限输入。

### 12.3 落区

- 覆盖非零 origin、0.5m fine geometry、mean 位于 cell 中心/边界、对称 tie-break 与稳定前缀。
- 验证二维 cell mass 精确等于两个一维 interval mass 的乘积。
- 验证累计未条件化质量达到阈值，且去掉末 cell 后低于阈值。
- 验证地图边缘 mean 会保留 `in_bounds=False` cell，且不重新归一化。
- 极小 sigma 与接近 1 的合法阈值必须确定性处理：能在合同上限内证明终止不变量时成功，
  只有实际触发候选上限或派生不可表示时才稳定拒绝；另用专门 fixture 证明候选上限
  fail closed、无静默截断。
- 覆盖 `math.fsum` 累计、严格 unseen-mass 终止证明、forged exact `WorldPoint`/geometry 与
  candidate cap 对全部逐-cell 状态的计数。
- 覆盖只访问同心 square perimeter、每 cell 一次、无 repeated full-square rescan，以及 N candidate 的
  Theta(N) generation/visit 与 O(N) retained per-cell state 回归。

### 12.4 Profile

- 覆盖 exact HOPPER、simulation proxy、固定 capability revision、固定月面参数与两项 `False` 声明。
- 默认六项均为 `None`；audit 给出完整稳定缺口顺序。
- 参数化每个字段单独补齐/缺失及非法零值、字符串、bool、NaN、无穷、派生类型。
- 参数化拒绝每个冻结 proxy 字段的漂移、非 exact 容器/元素、bool/派生类型，并覆盖被篡改的
  exact `PlatformProfileV2`/`HopperProfileV2` 与不自洽 `HopperProfileAuditV2`。
- 全量显式 test fixture 只能得到“结构完整”，不能产生 provider、Gate pass 或实机声明。

### 12.5 回归

- Gate5A focused：`test_v2_ballistics.py`、`test_v2_profiles.py`、terrain/contracts/serialization 基础测试。
- nested `path-planner/tests` 全量必须保持现有通过集合，新增 v2 不得增加 fail/error/skip。
- 父仓只集成 nested gitlink；不触碰 C 盘 Stage6 dirty 工作树。

## 13. 完成条件

Gate5A 只有在以下条件全部满足时完成：

1. 本文经书面复核批准并有详细实施计划。
2. RED tests 先证明生产表面缺失或合同不满足。
3. GREEN 只改批准范围，pure functions 与 profile audit 全绿。
4. fresh spec/quality review 均通过。
5. nested 全量回归无新增 fail/error/skip，父仓与 nested clean。
6. 未创建 Gate5 passed 证据，未把 test fixture 参数提升为正式能力。
7. v1 默认、v2 opt-in 及 checkpoint/default/executor/canary 四项硬边界不变。
