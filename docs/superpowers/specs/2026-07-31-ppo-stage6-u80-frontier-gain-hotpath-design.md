# Stage 6 U80 候选增益热路径严格等价提速设计

日期：2026-07-31  
状态：用户已批准方案方向，待书面规格复核  
适用范围：PPO High-Resolution Frontier Map Exploration / Stage 6 / `FrontierGenerator.extract()`

## 1. 背景与基准

本设计以 checkpoint `s6-standard-single-r1-20260724T000124Z` 的 Update 80 为主基准。该 checkpoint 的 validation best record 保持在 U80；后续 U90、U100 验证没有替换它。

已复现的 U80 worker 0 状态为：

- scenario：`train/scenario-0287`
- step：`42`
- coverage：约 `0.687482`
- frontier segment：`112`
- regular segment：`0`
- 最终有效候选：`315`
- 内部 gain 评估：`5290`
- 不同空间落点：`5023`
- 纯 `FrontierGenerator.extract()`：约 `19.7863s`

每个 gain 评估使用 `20m / 90° / 1°` 几何，即 91 条射线。U80 worker 0 因此执行约 48.1 万条 gain 射线。当前实现先由 `ray_cells_from_world()` 物化每条射线的完整 `tuple[CellXY, ...]`，随后调用方才在 observed blocker 处停止，造成不必要的尾部遍历、对象分配和 tuple 分配。

运行时最小实验仅把 tuple 接口替换为 iterator，U80 worker 0 从约 `19.7863s` 降到 `13.3508s`，约 `1.482x`，完整 action-set 签名保持一致。该结果证明射线提前终止方向成立，但尚未达到本设计的目标。

## 2. 目标与非目标

### 2.1 目标

1. 保持候选生成结果严格等价。
2. 融合 gain 专用整数 DDA，避免先物化完整射线 tuple。
3. 使用扁平整数索引聚合 footprint，只在返回结果时构造 `CellXY`。
4. 在一次 `extract()` 内复用有界 scratch memory，避免为 5290 次 gain 重复分配整图标记数组。
5. U80 全部 8 个 checkpoint 环境状态逐项通过等价验证。
6. U80 worker 0 纯 `extract()` 至少达到 `2.0x`，目标为 `2–3x`；同时报告 8 个状态逐项耗时和最慢状态，不用单个样本替代整体结论。

### 2.2 非目标

- 不减少 frontier segment、anchor、候选池或每段最多 3 个候选的合同。
- 不改变 `frontier_top_m=1024`。
- 不改变 20m、90°、1° gain 几何。
- 不改变候选 cell、recommended theta、22 维 features、priority、mask、排序或 diagnostics。
- 不改变 planning-safe、坡度、clearance、traversability、fallback 或 start-fallback 语义。
- 不改变公共 `ray_cells_from_world()` / `iter_ray_cells_from_world()` API。
- 不引入 Numba、SciPy、OpenCV、Cython 或其他运行时依赖。
- 不训练 PPO，不发布 checkpoint，不替换 default policy，不连接 executor，不启动 canary。

## 3. 设计

### 3.1 Gain workspace

为一次 `FrontierGenerator.extract()` 创建私有、局部的 gain workspace：

- 与地图大小一致的一维标记数组；
- 当前 gain 触及的 flat index 列表；
- 明确的 reset 操作，只清除上一候选触及的位置，不扫描整张地图；
- workspace 不进入 `FrontierActionSet`、checkpoint、episode state 或公共 API。

workspace 只在同步的单次 `extract()` 调用栈内使用，不作为模块全局状态，也不跨环境或线程共享。公共 `estimate_observed_only_gain()` 使用独立局部 workspace，保持可重入性。

### 3.2 融合整数 DDA

gain 私有核心直接执行与 `iter_ray_cells_from_world()` 相同的状态机：

- 使用相同的 `math.cos()`、`math.sin()`、step 选择和 world boundary 计算；
- 使用相同的 `min(t_max_x, t_max_y)` 与 `next_distance > range_m` 判断；
- 使用相同的 `math.isclose(..., rel_tol=0.0, abs_tol=1e-12)` 对角 tie 规则；
- blocker cell 自身进入 footprint，且仅当 `ray_cell_index > 0` 时终止；
- origin blocker 不阻挡下一格；
- 地图边界、射线顺序和浮点运算顺序保持不变。

每访问一个格子，将 `flat_index = y * width + x` 写入 workspace。只有首次触及时加入 touched 列表，从而等价于原 `set[CellXY]` 去重。

### 3.3 精确结果构造

flat index 按整数升序排列。对于固定 width，这与原来的 `(cell.y, cell.x)` 排序完全一致。随后：

1. 按原 observed mask 生成 `visible_unknown`；
2. 按原公式映射 low-resolution prior；
3. 按原顺序累加 `value_gain`；
4. 构造一次 `visible_unknown` 和 `visible_footprint` 的 `CellXY` tuple；
5. 返回原 `GainEstimate` 类型。

不得改变 float 累加顺序，不得用向量求和替代逐格 Python `float(...)` 累加，以免产生 bit-level 漂移。

### 3.4 调用边界

`FrontierGenerator.extract()` 构造一次 workspace，并传给 regular、irregular、fallback 和 start-fallback 的 gain 调用。现有公开函数签名保持不变；私有函数可以增加显式可选 workspace 参数，以便测试和独立调用继续工作。

## 4. 正确性与 TDD

实施前先增加失败测试，要求生产 gain 热路径：

1. 不调用会物化完整 tuple 的 `ray_cells_from_world()`；
2. 在近距离 blocker 后不继续消费射线尾部；
3. 一次 `extract()` 只创建一个 workspace，并在候选之间复用；
4. workspace 不泄漏上一个候选的 footprint。

差分 oracle 使用冻结的旧 gain 实现，覆盖：

- 随机 observed map；
- 地图边缘与角落；
- axial、diagonal 和 `1e-12` tie；
- origin blocker、第一格 blocker 和远端 blocker；
- 无 blocker、全 observed、局部 unknown；
- regular、irregular、fallback 和 start-fallback；
- 重复调用与不同地图尺寸。

逐项比较：

- `visible_unknown`
- `visible_footprint`
- `potential_gain_cells`
- `value_gain`（要求精确相等）
- candidate cells
- frontier features 原始 bytes
- candidate/frontier masks 原始 bytes
- diagnostics
- 完整 action-set 签名

## 5. U80 基准与验收

基准必须先完成 scenario/state 恢复，再单独计时纯 `extract()`，不得把 DEM、catalog、checkpoint 或场景构建时间混入。

主基准读取 U80 checkpoint 的 8 个 `vector_env_states[*].episode_state`。每个状态记录：

- scenario、step、coverage、segment count、candidate count；
- warmup 次数、repeat 次数和原始 samples；
- baseline/optimized median；
- speedup；
- 完整 action-set 签名比较。

性能验收：

- U80 worker 0 median speedup `>= 2.0x`；
- 8 个状态全部严格等价；
- 报告 8 个状态的 median、最大延迟和逐状态 speedup；
- 若 worker 0 未达到 `2.0x`，停止继续叠加猜测式优化，重新 profile 并提出一个新的单一假设；
- 不把微基准倍数直接宣称为整轮 PPO update 倍数。

基准 artifact 写入 `D:/xunce/review/` 下的新短目录，不写入仓库。

## 6. 风险与控制

- **DDA 漂移：** 用旧实现差分 oracle 和边界 fixture 锁定每格顺序。
- **workspace 污染：** 每次 gain 后按 touched index 精确清理，并用交替大/小 footprint 测试。
- **浮点漂移：** 保持角度、DDA 和 value gain 的计算与累加顺序。
- **工作区已有改动：** 只对当前任务涉及的 frontier 测试、frontier 源码、基准脚本和文档做增量修改，不覆盖现有用户改动。
- **U80 代表性：** 以完整 8-state 报告为结论，worker 0 只作为明确的高压力门槛。

## 7. 完成边界

只有在严格等价测试、U80 8-state 基准、受影响回归和 diff/UTF-8 检查全部通过后，才可声称“候选生成热路径提速”。未达到 `2.0x` 时应报告实验结果与剩余热点，不得把约 `1.5x` 描述为目标已完成。
