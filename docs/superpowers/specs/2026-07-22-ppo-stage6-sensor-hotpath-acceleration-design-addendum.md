# Stage 6 传感器热路径源码提速设计补充

日期：2026-07-22  
状态：用户已批准方案 A  
适用阶段：PPO High-Resolution Frontier Map Exploration v1 / Stage 6A

## 1. 背景与冻结恢复点

正式运行 `s6-standard-single-r1-20260718T220434Z` 在 Update 47--49 的同口径耗时代理均值为 105.05 分钟、中位数为 107.64 分钟。运行时 GPU 利用率约 30%，而多个环境 worker 持续占用 CPU；`SpawnVectorEnv` 在每轮动作后同步等待所有 worker，因此最慢环境决定整批推进速度。

源码路径确认主要热区为：

1. `LunarExplorationEnv.step()` 对 A* 路径每 1m 构造传感器姿态；
2. `SensorUpdater.reveal()` 对每个姿态执行 91 条射线；
3. 每条射线以纯 Python DDA 遍历 0.5m 栅格，并在循环中反复创建 `CellXY`、`WorldXY` 和射线 tuple；
4. 同步 vector env 把单个长路径 worker 的耗时传播到全部 8 个 worker。

代表性 256×256 基准在 128 个姿态、11,648 条射线、593,536 次栅格访问下，中位耗时为 `0.9124088999815285s`。

用户批准后，旧 runner 已在 Update 50 尚未 accepted 时停止。冻结恢复点为：

```text
latest_complete_update = 49
checkpoint_sha256 = cc23ed757dd48b8ad3c98fe95bfbe044562f6f3ff22048ee29337e62aa773066
manifest_sha256 = a78c37fe3f96efb011df29c3f100cd9af9737ab1bb01a45c7de80cc97700991b
discarded_partial_update = 50 attempt 1
```

## 2. 目标与非目标

目标：在不改变任何环境、传感器、PPO、采样顺序或复现合同的前提下，降低 Standard v1 rollout 的 CPU 热路径耗时，并从 Update 49 完整 checkpoint 继续训练。

非目标：

- 不改变 20m、90°、1° 射线合同；
- 不改变 1m 路径观测步长；
- 不改为异步 collector；
- 不改变 8-env 同步推理、transition 顺序、RNG 消耗或 checkpoint schema；
- 不改变 A*、frontier、reward、terminal、网络、PPO 或训练配置；
- 不引入 Numba、SciPy、OpenCV、Cython 或新运行时依赖；
- 不发布 checkpoint，不替换 default policy，不连接 executor，不启动 canary。

## 3. 方案 A

### 3.1 融合 LOS DDA 热路径

`SensorUpdater` 在构造时一次性冻结射线角偏移。`reveal()` 使用内部融合遍历器，直接维护整数 `(x, y)`、DDA 边界距离和数组索引；不再为每条射线先构造完整 `tuple[CellXY, ...]`，也不在每个格子上构造世界坐标对象。

可见集合以扁平整数索引聚合，全部姿态完成后再按 `(y, x)` 顺序构造唯一一次 `CellXY` 输出。阻挡格自身仍可见；仅当 `ray_cell_index > 0` 且格子是 hard obstacle 或坡度大于 30° 时停止该射线。原点阻挡不遮挡下一格的既有语义必须保持。

精确距离判定继续使用与旧实现等价的 cell-center `math.hypot(...) <= range_m` 语义；DDA 的 `1e-12` tie 判定、地图边界和原始 sub-cell pose 语义保持不变。公开函数 `ray_cells_from_world()` 的输入输出合同保持不变。

诊断字段必须逐项精确相等：

```text
sample_count
ray_count
cell_visit_count
unique_visible_cell_count
duplicate_cell_visits
sample_sources
sample_headings
```

### 3.2 线性路径姿态构造

`build_sensor_poses()` 用单调递增 segment cursor 代替每个采样距离从路径起点重新线性搜索 segment。采样距离、插值、切线、endpoint theta、路径长度和诊断字段必须逐项不变。

### 3.3 暂不采用的方案

异步 collector 可能进一步隐藏慢 worker，但会改变推理批次、动作采样 RNG 顺序和恢复状态机，风险高于本阶段收益。降低射线密度或增大路径观测步长会改变已批准算法合同。两者均不进入本补充范围。

## 4. TDD 与等价性证明

实现必须先增加失败测试并确认 RED，再写生产代码。

1. 冻结旧实现为测试侧独立 reference oracle；生产代码不得调用该 oracle。
2. 覆盖随机地图、连续 heading、sub-cell origin、地图边缘、hard rock、陡坡 crater、原点 blocker、精确 20m 边界和重复姿态。
3. 对 reference 与新实现逐项比较：visible cells、newly observed cells、完整 `SensorDiagnostics`、observed mask、height、obstacle、slope、traversability、observed safe mask。
4. 路径姿态覆盖直线、折线、对角、零长度路径和跨 segment 的 1m 采样；结果必须逐项相等。
5. 运行现有 Stage 1 sensor/LOS、Stage 2 catalog、Stage 4 collector 和 Stage 6 env/recovery 回归。
6. 使用冻结代表性 workload 重跑性能基准；语义等价通过后，最低接受 `1.5×`，目标 `2×+`。性能门写入 D 盘审计 artifact，不作为跨机器 CI 的易抖动单元测试。

## 5. 恢复与 lineage

源码通过 TDD、控制器验证和 fresh 规格/质量双审后，生成新的 Stage 6 source-repair amendment、review package、launch authorization 和 execution identity。任何 Critical/Important finding 必须修复并重审。

恢复必须：

- 使用相同正式 run id；
- 只从 Update 49 最后完整 checkpoint 加载；
- 下一次 transaction 为 Update 50 的新 attempt；
- 保持 policy、optimizer、normalizer、RNG、scenario sampler 和 8 个 vector-env episode state 来自 Update 49；
- 不接受或复用旧 Update 50 attempt 1 的部分 rollout；
- 在新 resource segment 中继续 append-only 审计；
- 启动前验证源码 hash、review hash、checkpoint hash 和授权 hash。

恢复后以前 3 个 accepted update 为初始速度观察窗；以新 source 下的同口径 resource sample proxy 与 U47--49 基线比较。若没有达到 1.5×，停止继续叠加猜测式修复，回到 profiling 与单一新假设。

## 6. 验收结论边界

源码提速只证明运行效率与语义等价，不构成 PPO 性能优势。Stage 6A 仍是 `single_seed_system_closure/v1`，额外 seeds 只有用户明确说“追加”才运行，且不阻塞 Stage 6/7/8。所有 synthetic terrain 仍标记为 proxy，`physical_obstacle_cells_written=false`。
