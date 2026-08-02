# Stage 6 U80 Frontier Gain NumPy Batch Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在生产源码保持冻结的条件下，构建并运行每 segment 批量 gain DDA 的独立 R3 NumPy 原型，提交严格等价与 R2 增量性能报告。

**Architecture:** 复制 R2 benchmark harness 到新的 D 盘 review root，保留 R2 heading helper 和所有候选选择语义。R3 只用一个 prototype-only `_irregular_scored_candidates` wrapper 收集同一 segment 的 cell/theta，再调用 NumPy batch gain evaluator，最后复用生产排序和候选对象合同。

**Tech Stack:** Python 3.12、NumPy 2.2.6、PyTorch checkpoint loader、现有 Stage 6 U80 fixture；不增加依赖，不修改生产源码。

## Global Constraints

- R3 root 固定为 `D:/xunce/review/s6-u80-frontier-hotpath-prototype-r3-numpy-r1/`。
- U80 checkpoint 和 SHA-256 必须匹配已批准规格。
- `src/lunar_exploration_ppo/env/frontier.py` SHA-256 必须始终为 `2fff448478730c5059b558fbfc771c6dae775a4c2dd7a2346293dc8c12eceeaf`。
- heading 使用 R2 integer-DDA helper；不改 heading、候选池、剪枝、排序、top-m 或输出字段。
- 仅在独立进程内绑定 prototype callable；不训练 PPO、不写 checkpoint、不替换 policy。
- 大型原型和运行 artifact 只写 D 盘。

---

### Task 1: 建立 R3 harness 与 RED 合同

**Files:**
- Create: `D:/xunce/review/s6-u80-frontier-hotpath-prototype-r3-numpy-r1/prototype_benchmark.py`
- Create at runtime: `D:/xunce/review/s6-u80-frontier-hotpath-prototype-r3-numpy-r1/input-audit.json`

**Interfaces:**
- Consumes: R2 harness、冻结 checkpoint 和生产 reference callable。
- Produces: `prototype_batch_gain_for_segment(...)` 与 `prototype_irregular_scored_candidates_r3(...)` 的测试入口。

- [ ] 复制 R2 harness 到新的单一 D 盘 root，修改 schema/root 名称，不覆盖 R2。
- [ ] 加入 `--r3-self-test`，先引用尚不存在的 `prototype_batch_gain_for_segment()`。
- [ ] 运行 `--r3-self-test`，确认因缺少 batch callable 出现预期 NameError。
- [ ] 运行 `--audit-only`，确认 checkpoint、U80 state count 和生产源码哈希。

### Task 2: 实现 NumPy per-segment batch gain

**Files:**
- Modify: `D:/xunce/review/s6-u80-frontier-hotpath-prototype-r3-numpy-r1/prototype_benchmark.py`

**Interfaces:**
- Produces: `prototype_batch_gain_for_segment(state, prior, cells, thetas, blocker, **geometry) -> list[PrototypeCompactGain]`。

- [ ] 使用 Python `math.sin/math.cos` 生成 `[N,91]` 方向数组，建立 current/step/t-max/t-delta/active workspace。
- [ ] 用 30–60 次 step loop 和 NumPy mask 原地更新 DDA；先记录 blocker cell，再终止对应 ray。
- [ ] 使用 composite key 和单次 `np.unique()` 合并本 segment footprint。
- [ ] 用整数 bincount 计算 count；按 canonical flat 顺序串行累计 `value_gain`。
- [ ] GREEN 运行 `--r3-self-test`，并禁止测试路径调用 R2 标量 gain。

### Task 3: 绑定现有 irregular candidate 流并完成差分

**Files:**
- Modify: `D:/xunce/review/s6-u80-frontier-hotpath-prototype-r3-numpy-r1/prototype_benchmark.py`
- Create at runtime: `ray-differential-results.json`
- Create at runtime: `gain-differential-results.json`

**Interfaces:**
- Produces: 与生产 `_irregular_scored_candidates()` 返回结构相同的 R3 method。

- [ ] 逐句复制当前 method 的 pool、local theta、R2 heading、redirected 和稳定排序逻辑，只把标量 gain loop 替换为单次 batch gain。
- [ ] 运行至少 10,000 条确定性单射线序列差分。
- [ ] 运行至少 160 个 batch gain 地图差分，精确比较 potential、footprint count 和 value gain。
- [ ] 运行 segment 级差分，比较 R2/R3 返回 cell、theta、redirected 和 gain 消费字段。
- [ ] 任一失败时写 failure artifact 并停止，不进入性能测试。

### Task 4: 执行 worker-0 与全 8 状态性能门

**Files:**
- Modify: `D:/xunce/review/s6-u80-frontier-hotpath-prototype-r3-numpy-r1/prototype_benchmark.py`
- Create at runtime: `worker-0-results.json`
- Create at runtime: `state-progress.json`
- Create at runtime: `results.json`

**Interfaces:**
- Consumes: R2 scalar prototype、R3 batch prototype、完整 action-set signature。
- Produces: R2/R3 各 1 次 warmup 和 3 次交错 samples。

- [ ] worker-0 比较 R2/R3 完整 action-set；若不等价或增量加速低于 `1.2x` 则停止。
- [ ] 通过后运行全 8 状态，逐状态写 progress，支持从已完成 worker 恢复。
- [ ] 要求 worker-1 增量至少 `1.3x`、8 状态中位增量至少 `1.4x`、8/8 输出精确相等。
- [ ] 记录 R3 峰值 RSS 增量；单 worker 不得超过 64 MB。

### Task 5: 报告与完成前验证

**Files:**
- Create at runtime: `D:/xunce/review/s6-u80-frontier-hotpath-prototype-r3-numpy-r1/report.md`

**Interfaces:**
- Produces: 可审核 R2/R3 性能、严格等价、内存和生产冻结证据。

- [ ] 以 UTF-8 写入 report，明确区分候选生成 microbenchmark 与整 PPO update。
- [ ] 新鲜重跑 self-test、射线差分和 gain 差分。
- [ ] 校验 JSON 全部 finite、prototype SHA、checkpoint SHA 和 8-state signatures。
- [ ] 比较实验前后 `frontier.py` SHA-256，必须完全一致。
- [ ] 暂停并提交报告；用户明确授权前不得替换生产代码。
