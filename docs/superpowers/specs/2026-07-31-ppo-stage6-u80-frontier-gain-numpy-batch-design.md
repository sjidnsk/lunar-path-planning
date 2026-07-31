# Stage 6 U80 Frontier Gain NumPy Per-Segment Batch DDA Design

## 目标

在完全不修改生产 `src/lunar_exploration_ppo/env/frontier.py` 的前提下，构建第三版独立原型，验证“每个 frontier segment 内批量执行 gain DDA”能否在保持最终候选输出严格等价的同时，显著快于 R2 Python 融合 DDA。

R3 只批量化 gain DDA。它不修改 heading target 搜索、候选池、候选姿态、候选剪枝、排序、top-3/top-m、feature、mask、diagnostics、PPO、reward、network、Hybrid A* 或 synthetic terrain。

## 已冻结基线

- 主输入固定为 U80 checkpoint：
  `D:/xunce/out/ppo_frontier/s6-standard-single-r1-20260724T000124Z/s6/checkpoints/seed-20260716/update-00000080/checkpoint.pt`。
- checkpoint SHA-256 固定为：
  `35e04c86f9f973af028fb08f0175d42ab45378d09e1f2b96ee6aad6e4c12b5b5`。
- 生产 `frontier.py` 冻结 SHA-256 为：
  `2fff448478730c5059b558fbfc771c6dae775a4c2dd7a2346293dc8c12eceeaf`。
- R2 review root 固定为：
  `D:/xunce/review/s6-u80-frontier-hotpath-prototype-r2/`。
- R2 全 8 状态中位加速为 `4.823902939105883x`；优化后状态中位耗时为 `3.8273730500077363s`。
- R2 worker-1 耗时为 `5.833743199997116s`，对应 210 个 segment、10,150 次 gain 评估和 580 个最终候选。

## 输出位置与隔离边界

R3 原型、测试和报告只写入：

`D:/xunce/review/s6-u80-frontier-hotpath-prototype-r3-numpy-r1/`

原型通过独立 benchmark 进程中的内存绑定或独立 generator wrapper 运行。不得把 R3 callable 导入生产模块，不得覆盖、格式化或替换生产源码。结果审核完成后继续等待用户明确授权，不能自动实施生产替换。

## 方案选择

### 采用：每 segment 批量 gain DDA

每个 segment 保持现有候选池生成顺序。R2 heading helper 逐候选计算 `recommended_theta` 和 `redirected`，随后把该 segment 的全部候选一次性交给 NumPy batch gain evaluator。

典型 worker-1 segment 中位包含 49 个候选。批量数组的主形状为：

`candidate_count × 91 rays`

Python 只保留约 30–60 次 DDA step 循环；每一步使用 NumPy ufunc 同时更新整个二维射线阵列。

### 不采用：跨 segment 全局批量

跨 segment 批量需要先重构全部候选收集、heading binding 和结果回填，内存峰值也会显著增加。它超出本次最小原型范围。

### 不采用：同时修改 heading 或候选剪枝

heading 在 R2 profile 中约占 10%，gain DDA 约占 83%。同时改 heading 或候选剪枝会破坏单变量归因，并显著提高严格等价风险。因此 R3 保持 R2 heading 实现和所有候选选择语义不变。

## 数据流

1. Python 按现有逻辑生成一个 segment 的候选 pool。
2. R2 heading helper 按原顺序计算每个候选的 theta 和 redirected 状态。
3. Python 使用 `math.sin`、`math.cos` 为每个候选的 91 条射线生成方向值，避免改用 NumPy 三角函数造成末位差异。
4. NumPy batch evaluator 初始化 `[N, 91]` 的 current cell、step、`t_max`、`t_delta`、active 和 ray index 数组。
5. 每个 DDA step 先记录当前 active cell，再应用 blocker、range 和 bounds 停止语义，最后按 tie/x/y 分支更新下一步状态。
6. 每个可见 cell 编码为复合整数键：
   `candidate_index * map_cell_count + flat_cell_index`。
7. 对本 segment 的全部有效键执行一次 `np.unique()`，得到按 candidate 和 flat index 升序排列的唯一 footprint。
8. 通过 `np.bincount()` 计算整数 footprint count 和 potential gain count。
9. `value_gain` 不使用 NumPy浮点归约；它按 candidate、flat升序执行与 Python reference 相同的串行 double 累加。
10. 将紧凑 gain 结果绑定回原候选对象，继续使用现有排序、稳定 tie-break 和 top-3 逻辑。

## 批量 workspace

每个 segment 的 workspace 只保存当前 DDA step，不保存完整 `[N, 91, S]` 浮点状态。

主要数组包括：

- `current_x/current_y: int32[N, 91]`
- `step_x/step_y: int8[N, 91]`
- `t_max_x/t_max_y: float64[N, 91]`
- `t_delta_x/t_delta_y: float64[N, 91]`
- `active/in_bounds/tie/blocked: bool[N, 91]`
- 每一步收集的复合 visible keys

workspace 应在同一 extract 内按最大 segment 容量复用。禁止每个 DDA step 重建全部工作数组；可使用原地更新和 `out=` 的操作应优先使用。

预计典型 segment 峰值为数 MB。R3 gate 要求相对 R2 单 worker 新增峰值内存不超过 64 MB。

## 严格等价合同

### DDA 几何

- range 保持 `20.0m`。
- FOV 保持 `90.0°`。
- ray step 保持 `1.0°`，共 91 条射线。
- origin cell 必须加入 footprint，且 origin blocker 不停止射线。
- 非 origin blocker cell 必须先加入 footprint，再停止射线。
- 只有 `next_distance > range_m` 才停止；相等时继续。
- 对角 tie 必须同时更新 x/y。
- 越界 cell 不得记录。

### 浮点

- theta 和所有 DDA 距离使用 `float64`。
- 三角函数继续调用 Python `math.sin/math.cos`。
- 不使用默认 `np.isclose()`；tie 必须复刻 `rel_tol=0.0, abs_tol=1e-12`，并处理相等值与非有限值。
- 计算 boundary、差值、除法和累计 `t_max` 时不得做代数化简或改变操作顺序。
- 不允许 fast-math、近似角度离散化或容差放宽。

### footprint 与 value gain

- 唯一 footprint 的 canonical 顺序为 flat index 升序，即 `(y, x)` 升序。
- potential gain 只统计未观测 cell。
- `value_gain` 必须按 canonical flat 顺序逐项把 prior `float32` 转为 Python double 语义后累加。
- 禁止使用可能改变浮点归约顺序的 `np.bincount(weights=...)`、`np.add.reduceat()` 或并行 reduction。

### 最终输出

必须逐字段保持：

- candidate cell
- recommended theta
- candidate ordering
- frontier features bytes
- candidate mask bytes
- frontier/observed-safe/reachable mask bytes
- canonical diagnostics

公开 `estimate_observed_only_gain()` 的 materialized `GainEstimate` 行为不属于 R3 替换范围，仍以当前生产实现为 reference。

## 错误处理与停止条件

- shape、dtype、C-contiguous、geometry 或 finite 校验失败时立即报错，不静默复制或回退。
- NumPy batch evaluator 出现任何差分失败时保存最小失败 case，并停止性能测试。
- 如果 `np.unique()`、数组分配或 inactive lane 计算使 worker-1 不能达到最低性能门，则记录实际结果并停止该路线；不能通过改变候选语义获取速度。
- 原型性能失败不影响 R2，R2 保持可复核和可运行。

## 验证计划

### RED/GREEN 自测

先写 batch API 自测并确认在 callable 未实现时失败。实现后要求：

- 禁止调用 R2 标量 gain callable 时 batch evaluator 仍能完成；
- 连续运行大/小 segment 不发生 workspace 污染；
- blocker origin、第一格、远端、边界、轴向和对角 case 精确相等。

### 射线级差分

至少验证 10,000 条确定性射线，逐格比较 batch DDA 与当前 Python iterator 的 cell 序列，包括近 tie、地图边缘、不同 resolution 和 range boundary。

### gain 差分

至少 160 个确定性随机/边界地图，要求：

- `potential_gain_cells` 精确相等；
- `footprint_count` 精确相等；
- `value_gain` 精确相等；
- batch 内候选顺序不影响结果。

### U80 性能与结果

1. worker-0 先执行 R2/R3 各 1 次 warmup 和 3 次交错计时；完整 action-set 必须相等。
2. worker-0 的 R3/R2 增量加速低于 `1.2x` 时停止全状态扩展。
3. 通过 worker-0 后，对 U80 全 8 状态执行 R2/R3 各 1 次 warmup 和 3 次交错计时。
4. worker-1 的 R3/R2 增量加速必须至少 `1.3x`。
5. 全 8 状态的 R3/R2 状态加速中位数必须至少 `1.4x`。
6. 8/8 action-set 必须精确相等。

性能计时只覆盖 `FrontierGenerator.extract()`，排除 checkpoint、catalog、scenario 和 episode-state 恢复。报告不得把 microbenchmark 写成整 PPO update 加速结论。

## 预期结果与决策规则

R3 的保守目标是把 worker-1 从 R2 的 `5.834s` 降到 `3.2–4.5s`；较好结果是 `2.5–3.2s`。

- 如果 R3 达到全部严格等价门和性能门，可向用户提交报告，并等待是否授权生产替换。
- 如果严格等价通过但性能门失败，保留报告，停止 NumPy 路线，优先评估 C++ batch DDA。
- 如果严格等价失败，不进行性能结论或生产替换。

## 非目标

- 不减少每 segment 候选数。
- 不改变每候选姿态数。
- 不引入候选上界剪枝。
- 不修改 heading target 算法。
- 不引入 C++、Cython、Numba、pybind11、Torch extension 或 GPU。
- 不训练 PPO，不发布 checkpoint，不替换 default policy，不连接 executor，不启动 canary。
- 不宣称整 update、训练吞吐或端到端任务耗时已经提升。
