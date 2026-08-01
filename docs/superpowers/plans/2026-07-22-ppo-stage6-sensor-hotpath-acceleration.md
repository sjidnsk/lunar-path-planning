# Stage 6 传感器热点精确语义提速实施计划

> **执行约束：** 本计划是已获批“方案 A”的实施细化。Stage 6 仍执行原 Goal 的 TDD、fresh implementer、fresh 规格/质量双审和人工 Gate；实现者不得提交，Stage 6 只在最终人工批准后生成唯一提交。

## 目标

在不改变 PPO、环境、LOS、路径、奖励、终止、候选、随机数、worker 顺序或 checkpoint 语义的前提下，消除 Standard 训练中 `SensorUpdater.reveal` 和 `build_sensor_poses` 的纯 Python 分配/重复搜索热点。以 Update 49 的完整 checkpoint 为唯一恢复点，丢弃未完成的 Update 50 attempt1；通过语义差分、回归、性能、独立审查和 ordinal6 fail-closed 恢复授权后，从 Update 50 attempt2 继续同一 run。

## 冻结输入与成功标准

- formal run：`s6-standard-single-r1-20260718T220434Z`
- seed：`20260716`
- last accepted update：`49`
- checkpoint SHA-256：`cc23ed757dd48b8ad3c98fe95bfbe044562f6f3ff22048ee29337e62aa773066`
- manifest SHA-256：`a78c37fe3f96efb011df29c3f100cd9af9737ab1bb01a45c7de80cc97700991b`
- Update 50 attempt1：仅有 pre，明确废弃；下一次为 attempt2、resource `segment_index=7`
- 代表负载基线：256×256、128 poses、11,648 rays、593,536 visits，median `0.9124088999815285s`
- 性能硬门：同机同负载 median 至少 1.5×；目标 2×以上；低于 1.5×不得恢复正式训练
- 精确性硬门：reference/optimized 的 visible/newly-observed cell、顺序、全部 diagnostics、observed state 数组逐项相同；现有测试和完整 Stage 6 focused tests 通过

## 不变合同

- 传感器仍为 20m、90°、1°；路径采样仍为 1m。
- 保留 DDA 的入格顺序、角点 `1e-12` tie、精确 range cutoff、起点格不阻挡、后续 hard-obstacle/坡度阻挡、map-edge 行为。
- `visible_cells` 和 `newly_observed_cells` 仍按 `(y,x)` 稳定排序。
- 不引入 Numba、SciPy、OpenCV、Cython或新依赖；不启用异步 collector。
- 不更改 frontier、A*、reward、done、GAE、PPO、网络、checkpoint schema 或训练超参数。
- 正式 runner 停止期间才能改源码；验证完成前不得启动重复 runner。

## Task 1：建立独立 reference、RED 测试与可复现 benchmark

**文件：**

- 新增：`tests/ppo_highres_frontier/test_stage6_sensor_acceleration.py`
- 新增：`scripts/benchmark_ppo_stage6_sensor_hotpath.py`

**步骤：**

1. 在测试中内嵌冻结的 legacy reference 实现，不调用待优化私有函数，覆盖：随机 pose/heading、四象限、轴向、角点 tie、sub-cell origin、map edge、range 边界、起点 blocker、hard rock、陡坡、重复可见格和多 pose。
2. 比较 reference 与 production 的 `ObservationDelta`、全部 diagnostics 和所有 observed-state arrays；固定 seed，至少 100 个小图随机 case，并加入一组 Standard 尺寸代表 case。
3. 加 RED 结构测试：在 `reveal` 期间将公共 `ray_cells_from_world` 替换为抛错函数，要求优化路径不再按 ray 构造 tuple/`CellXY`；公共函数本身仍保留并继续通过既有 API 测试。
4. 加 `build_sensor_poses` 长折线路径等价测试，冻结 pose 坐标、heading、source 与 diagnostics。
5. benchmark 脚本只写 D 盘 review 目录，记录环境、warmup、repeat、样本规模、baseline/optimized samples、median、speedup 和语义校验摘要；默认不写仓库。
6. 先运行目标测试并保存 RED 证据；RED 必须因尚未启用 fused hot path 失败，而非 fixture/导入错误。

## Task 2：实现 fused LOS DDA 与单调路径游标

**文件：**

- 修改：`src/lunar_exploration_ppo/env/sensor_model.py`
- 修改：`src/lunar_exploration_ppo/env/action_execution.py`
- 修改：`tests/ppo_highres_frontier/test_stage6_sensor_acceleration.py`

**步骤：**

1. `SensorUpdater.__init__` 预计算固定 ray offsets（弧度）；不预计算与 heading 有关的方向。
2. 在 `reveal` 内直接执行与公共 `ray_cells_from_world` 完全相同的 DDA 标量状态机：整数 `x/y`、预取 geometry 标量、width/height、origin、resolution、range，以及 truth blocker arrays。cell-center 门必须继续调用标量 `math.hypot(...) > range_m`，不得换成平方距离比较。
3. 以 flat integer index 聚合可见格；只有结束时按 flat index（等价 `(y,x)`）创建一次 `CellXY` tuple。cell center range 检查保持原来的 `>` 语义；不得用近似视锥或栅格化替代。
4. 保持 visits 只统计 range 内格、blocker cell 可见且终止该 ray、origin cell 不阻挡、duplicate 计算不变。
5. `build_sensor_poses` 使用只向前移动的 `segment_index` 游标定位累计距离，保持 `1e-12` 比较、零/末端行为和输出顺序。
6. 依次运行 RED→GREEN 测试、现有 sensor/path 测试、随机差分测试；任何差异先最小化反例再修复，禁止放宽断言。
7. 运行同机 benchmark；若 median speedup <1.5×，只做一次 profile，提出单一新假设并重新审查，不叠加猜测式优化。

## Task 3：新增 ordinal6 fail-closed 源码提速恢复桥

**文件：**

- 修改：`src/lunar_exploration_ppo/workflows/stage6_source_repair.py`
- 修改：`src/lunar_exploration_ppo/workflows/stage6.py`
- 修改：`src/lunar_exploration_ppo/ppo/standard_training.py`
- 修改：`scripts/create_ppo_stage6_source_repair_amendment.py`
- 修改：`tests/ppo_highres_frontier/test_stage6_source_repair.py`
- 修改：`tests/ppo_highres_frontier/test_stage6_workflow.py`
- 修改：`tests/ppo_highres_frontier/test_stage6_training.py`

**步骤：**

1. 先写 RED fixture，构造 ordinal1–5 后的 accepted Update1–49、Update50 attempt1 pre、停止证据和 Update49 完整 checkpoint。
2. 新增唯一 artifact `source-repair-sensor-acceleration.json`，schema `stage6_source_repair_sensor_acceleration/v1`，repair ordinal 6；固定 parent 为 ordinal5，last accepted=49，first resumed=50，next transaction=`0049:20260716:update:050`。
3. ordinal6 必须绑定：Update49 checkpoint/manifest/complete/latest、accepted journal/metrics/resource prefix、Update50 partial-pre discard 证据、R18 停止证据、优化后的 source set、差分结果、benchmark、fresh 双审和 review manifest。
4. 扩展 `Stage6SourceRepairContext`、summary、immutable bindings、protected checkpoints、`require_current`、load/create/preview/publish 和命令行结果；存在 ordinal6 时 `protected_checkpoint_updates` 必须含 49。
5. 同步扩展所有消费方：Stage 6 optional manifest artifact chain、terminal/recovery embedded evidence、manifest graph、`build_stage6_acceptance_artifacts()` 的 summary v6 验证，以及 production/test source-set。新设计补充、实施计划、sensor/action 源码、新差分测试和 benchmark 脚本必须进入受审 source identity，不能成为未绑定旁路。
6. fail closed 测试覆盖：错误 update/SHA、缺失或增长前缀、partial attempt 未声明废弃、旧进程仍活跃、benchmark <1.5×、语义差分失败、source/review 漂移、ordinal5 parent 漂移、重复 publish 和发布后篡改。
7. ordinal6 publish 前只允许 dry-run；publish 后 runner 再次校验 authorization、execution identity、source repair context 和精确 Update49 checkpoint。

## Task 4：集成验证、fresh 双审与恢复授权

**大型证据目录：** `D:/xunce/review/s6-sensor-acceleration-r1-<timestamp>/`

**步骤：**

1. 运行格式/静态检查、目标测试、Stage1 sensor/path 回归、Stage6 source-repair/workflow/training focused suite，以及各子项目分进程基线；保存命令、退出码和日志 hash。
2. benchmark 至少 warmup 3、repeat 7；记录 before/after 原始 samples，不只记录汇总；确认 GPU/RSS/D盘硬门未被改变。
3. fresh 规格 reviewer 审查：算法/LOS/恢复状态是否逐条满足本计划和主设计；fresh 质量 reviewer 审查：正确性、可维护性、性能证据、Windows spawn/路径安全和 fail-closed。
4. Critical/Important 必须由 fresh implementer 修复并重审；Minor 可记录但不得伪装为已修复。实现者和 reviewer 均不得提交。
5. 生成 prospective diff、source hashes、测试/benchmark/review manifest 和新的 launch authorization；先 dry-run ordinal6，再 publish 一次并记录 SHA-256。
6. 启动前再次确认无旧 runner/worker、latest=49、checkpoint hashes 精确匹配、下一资源段=7；从同 run-id 启动 Update50 attempt2。
7. 恢复 `ppo-stage-6` 每约30分钟 heartbeat；自动化只做一次只读快照，不在回合内 sleep，不启动重复 runner。

## Task 5：恢复后的验收边界

1. 首个 accepted Update50 后比较同口径耗时与 U47–49 baseline（mean 105.05、median 107.64 分钟），同时报告环境/场景波动，不能把微基准倍数直接宣称为整 update 倍数。
2. 若出现语义、lineage、mask、snapshot、nonfinite 或 checkpoint 漂移，立即停门；若只是未达到整 update 预期提速，保持训练状态并基于 profile 决定是否另立方案。
3. 继续 seed `20260716 ×100`；不运行额外 seeds，除非用户明确说“追加”。
4. Stage 6 机器验收与 fresh 双审完成后，提交完整 Gate 证据等待人工批准；获批前不 commit，不进入 Stage 7。

## 声明边界

- 本次改动只能声称“精确语义实现提速”和“同一单 seed 训练从 Update49 可恢复继续”。
- 微基准通过不等于 PPO 已建立性能优势；单 seed 也不能支持多 seed 性能结论。
- 不发布 checkpoint、不替换 default policy、不连接 executor、不启动 canary；highres terrain/障碍物仍仅为 synthetic proxy。
