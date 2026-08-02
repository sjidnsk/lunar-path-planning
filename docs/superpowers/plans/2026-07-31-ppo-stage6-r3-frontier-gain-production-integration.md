# Stage 6 R3 Frontier Gain Production Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将已通过 U80 严格等价、性能和内存门槛的 R3 每 segment NumPy 批量 gain DDA 设为 irregular frontier 候选生成的默认生产实现。

**Architecture:** 公开 `estimate_observed_only_gain()` 及标量 `_estimate_observed_only_gain_with_blockers()` 保持完整 `GainEstimate` 行为。新增仅供 irregular 候选路径使用的紧凑 gain summary 与批量 evaluator；`_irregular_scored_candidates()` 继续按原顺序生成 pool、theta 和 redirected，只把逐候选标量 gain loop 替换为一次 per-segment batch 调用。

**Tech Stack:** Python 3.12、NumPy 2.2.6、pytest、Stage 6 U80 checkpoint；不增加依赖。

## Global Constraints

- 不修改 heading target、候选池、候选姿态、剪枝、稳定排序、top-3/top-m、feature、mask 或 diagnostics 语义。
- 不修改公开单点 gain API 的 materialized footprint 合同。
- 不训练 PPO、不发布 checkpoint、不替换 policy、不连接 executor、不启动 canary。
- 当前工作区已有改动均视为用户改动，只允许对目标位置做小范围补丁。
- 性能计时仅覆盖 `FrontierGenerator.extract()`；不得宣称整个 PPO update 已获得相同比例加速。

---

### Task 1: 建立生产 batch gain RED 合同

**Files:**
- Modify: `tests/ppo_highres_frontier/test_stage2_frontier.py`

**Interfaces:**
- Consumes: 公开标量 `estimate_observed_only_gain()` 作为独立 reference。
- Produces: `_estimate_observed_only_gain_batch_with_blockers(state, prior, cells, thetas, blocker, **geometry)` 的行为合同。

- [ ] 新增一个确定性小地图测试，覆盖空 batch、多个候选、轴向/对角射线、blocker 和 value gain。
- [ ] 对每个候选把 batch 结果的 `potential_gain_cells`、footprint count 和 `value_gain` 与公开标量 reference 精确比较。
- [ ] 运行该测试并确认因 batch callable 不存在而失败。

### Task 2: 迁移 R3 evaluator 并设为默认

**Files:**
- Modify: `src/lunar_exploration_ppo/env/frontier.py`

**Interfaces:**
- Produces: `_CompactGainEstimate` 和 `_estimate_observed_only_gain_batch_with_blockers(...) -> tuple[_CompactGainEstimate, ...]`。
- Updates: `FrontierGenerator._irregular_scored_candidates()` 默认使用 batch evaluator。

- [ ] 从已验证 R3 原型迁移 `float64` DDA、Python `math.sin/math.cos`、`1e-12` tie、composite flat key、整数 bincount 和 canonical Python double 累加。
- [ ] 保留标量完整 gain evaluator，不改变公开 API。
- [ ] 在 irregular method 中先按原顺序收集 theta/redirected，再执行一次 batch，并按原顺序绑定结果。
- [ ] 运行 Task 1 测试并确认转绿。

### Task 3: 收敛既有 irregular 回归测试

**Files:**
- Modify: `tests/ppo_highres_frontier/test_stage2_frontier.py`

**Interfaces:**
- Consumes: 新 batch callable。
- Produces: pool 每候选只评估一次、redirect 保留、108 候选池和 fallback gain binding 的生产回归证据。

- [ ] 将只为注入 synthetic gain 而 monkeypatch 标量内部 callable 的测试改为注入 batch callable，保持原行为断言不变。
- [ ] 增加断言证明 irregular 默认路径只调用一次 batch 且不会调用标量 evaluator。
- [ ] 运行所有 Stage 2 frontier 测试并修复仅由默认调用边界变化导致的失败。

### Task 4: 时间受限的严格等价与性能验收

**Files:**
- Reuse: `D:/xunce/review/s6-u80-frontier-hotpath-prototype-r3-numpy-r1/prototype_benchmark.py`
- Create: `D:/xunce/review/s6-u80-frontier-hotpath-production-r3-r1/`

**Interfaces:**
- Produces: 生产标量 reference 与生产默认 R3 的定向差分，以及 worker 1 单状态 action signature 和一次性能结果。

- [ ] 运行生产 batch/scalar 定向等价测试和 irregular 候选回归测试。
- [ ] 运行 Stage 2 frontier 测试文件以及相关 smoke 测试。
- [ ] 仅恢复 U80 最慢的 worker 1 状态，生产默认 R3 执行一次 `extract()`，并与冻结 R3 prototype signature 精确比较。
- [ ] 记录 worker 1 单次 `extract()` 时间；不运行全 8 状态、不做 3 次重复、不运行 PPO update。
- [ ] 校验 checkpoint SHA、目标文件 diff、所有 JSON finite 和报告 UTF-8。

### Task 5: 交付边界

**Files:**
- Create: `D:/xunce/review/s6-u80-frontier-hotpath-production-r3-r1/report.md`

**Interfaces:**
- Produces: 默认切换结果、测试证据、性能边界和回退说明。

- [ ] 新鲜重跑目标测试和验证命令后再报告完成。
- [ ] 只提交本任务相关的计划、生产热点和直接测试改动；不纳入工作区其他改动。
- [ ] 不自动 push、创建 PR、训练或启动任何运行流量。
