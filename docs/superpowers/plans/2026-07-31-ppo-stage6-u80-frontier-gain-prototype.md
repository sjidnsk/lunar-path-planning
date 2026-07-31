# Stage 6 U80 Frontier Gain Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在完全不修改生产 `frontier.py` 的前提下，构建并运行严格等价的融合整数 DDA 原型，向用户提交 U80 性能与结果对照，随后暂停等待生产替换授权。

**Architecture:** 原型和 benchmark harness 只写入 `D:/xunce/review/s6-u80-frontier-hotpath-prototype-r1/`。harness 在独立进程内先调用当前生产 gain 得到 baseline，再临时绑定原型 callable 得到 optimized；每个状态比较完整 `FrontierActionSet`，进程结束后模块状态自然丢弃，仓库源码不发生变化。

**Tech Stack:** Python 3.12、NumPy、PyTorch checkpoint loader、现有 `lunar_exploration_ppo` 类型与 Stage 6 U80 checkpoint；不增加依赖。

## Global Constraints

- 主输入固定为 `D:/xunce/out/ppo_frontier/s6-standard-single-r1-20260724T000124Z/s6/checkpoints/seed-20260716/update-00000080/checkpoint.pt`。
- checkpoint SHA-256 必须等于 `35e04c86f9f973af028fb08f0175d42ab45378d09e1f2b96ee6aad6e4c12b5b5`。
- 不修改、覆盖或格式化 `src/lunar_exploration_ppo/env/frontier.py` 及任何生产源码。
- 不修改候选 cell、theta、features、mask、排序、diagnostics、20m/90°/1° gain 几何或安全合同。
- 不训练 PPO，不写 checkpoint，不替换 policy，不连接 executor，不启动 canary。
- 大型结果与临时源码只写 D 盘 review root。
- 原型报告完成后暂停；用户再次明确确认前不得实施生产替换。

---

### Task 1: 冻结原型输入与输出签名

**Files:**
- Create: `D:/xunce/review/s6-u80-frontier-hotpath-prototype-r1/prototype_benchmark.py`
- Create at runtime: `D:/xunce/review/s6-u80-frontier-hotpath-prototype-r1/input-audit.json`

**Interfaces:**
- Consumes: U80 checkpoint、Stage 6 config、`StandardTrainingEnv.import_episode_state()`。
- Produces: `action_set_signature(action_set) -> dict[str, object]` 与已验证的 8 个 episode states。

- [ ] **Step 1: 创建明确的 D 盘 review root**

使用 PowerShell `New-Item -ItemType Directory -Force` 只创建上述单一目录；不得清理或覆盖其他 review root。若目录已存在且含非本次文件，创建带时间戳的新 root 并在报告中记录实际路径。

- [ ] **Step 2: 写入 checkpoint 审计代码**

原型入口必须在 `torch.load()` 前后验证文件存在、SHA-256、`update_step == 80`、8 个 `vector_env_states`，并验证每项恰好包含 `episode_state` 和 `sampler_state`。不满足时以非零状态退出，不运行基准。

- [ ] **Step 3: 实现完整 action-set 签名**

签名必须分别记录 `cells`、`frontier_features` bytes SHA-256、三个 frontier mask bytes SHA-256、candidate mask bytes SHA-256、数组 dtype/shape 和 canonicalized diagnostics；不得只比较一个拼接总哈希，以便定位任何漂移。

- [ ] **Step 4: 验证输入审计**

运行：

```powershell
$env:PYTHONPATH='src'
D:/conda_envs/lunar-explorer/python.exe D:/xunce/review/s6-u80-frontier-hotpath-prototype-r1/prototype_benchmark.py --audit-only
```

预期：退出码 0，并生成 UTF-8 `input-audit.json`，其中 checkpoint/update/state count 全部匹配冻结值。

### Task 2: 实现不落入生产源码的融合 DDA 原型

**Files:**
- Modify: `D:/xunce/review/s6-u80-frontier-hotpath-prototype-r1/prototype_benchmark.py`
- Create at runtime: `D:/xunce/review/s6-u80-frontier-hotpath-prototype-r1/differential-results.json`

**Interfaces:**
- Consumes: 当前生产 `_estimate_observed_only_gain_with_blockers()` 作为 reference oracle。
- Produces: `PrototypeGainWorkspace` 和 `prototype_estimate_observed_only_gain_with_blockers(...) -> GainEstimate`。

- [ ] **Step 1: 先写原型自测并验证 RED**

在原型脚本中加入自测：临时禁止 `frontier_module.ray_cells_from_world` 后调用 prototype；在 prototype 尚未定义时必须因缺少 callable 失败。该测试证明后续实现不会偷偷调用 tuple 物化 API。

- [ ] **Step 2: 实现局部 workspace**

`PrototypeGainWorkspace` 持有一维 `np.bool_` marks 和 Python `list[int]` touched。每个 gain 开始时清除上一轮 touched 对应 marks 并清空列表；flat index 首次出现时置位并追加。workspace 只由单个 benchmark 调用栈拥有。

- [ ] **Step 3: 实现融合整数 DDA**

逐句复刻生产 `iter_ray_cells_from_world()` 的 cos/sin、step、boundary、`next_distance > range_m`、`math.isclose(..., abs_tol=1e-12)`、blocker-cell-visible 和 origin-not-blocking 语义。flat index 排序后按原循环顺序构造 `CellXY` 并逐格累加 value gain。

- [ ] **Step 4: 运行 RED 后的 GREEN 自测**

运行：

```powershell
$env:PYTHONPATH='src'
D:/conda_envs/lunar-explorer/python.exe D:/xunce/review/s6-u80-frontier-hotpath-prototype-r1/prototype_benchmark.py --self-test
```

预期：退出码 0；tuple API 禁用时 prototype 仍工作，连续大/小 footprint 不发生 workspace 污染。

- [ ] **Step 5: 运行差分 oracle**

使用固定 seed 生成不少于 128 个 gain case，覆盖地图边缘、无 blocker、origin/第一格/远端 blocker、axial/diagonal heading、随机 observed/unknown mask 和重复 workspace。每例要求 `GainEstimate` 四个字段精确相等；任一失败时保存最小 case 并停止性能测试。

### Task 3: 执行 U80 worker 0 性能与完整结果门

**Files:**
- Modify: `D:/xunce/review/s6-u80-frontier-hotpath-prototype-r1/prototype_benchmark.py`
- Create at runtime: `D:/xunce/review/s6-u80-frontier-hotpath-prototype-r1/worker-0-results.json`

**Interfaces:**
- Consumes: Task 1 的 action-set 签名和 Task 2 的 prototype callable。
- Produces: baseline/optimized 原始 samples、median、speedup、逐字段等价结论。

- [ ] **Step 1: 恢复 worker 0 且排除场景构建时间**

先创建并 `import_episode_state()`，完成后才开始计时。记录 scenario、step、coverage、segment count 和 candidate count，但不把 checkpoint、catalog、DEM 或 state restore 计入 `extract()`。

- [ ] **Step 2: 交替运行 baseline 与 optimized**

至少执行 1 次不计时 warmup，并交替收集各 3 次 timed samples。baseline 使用进程启动时保存的生产 callable；optimized 只在内存中临时绑定 prototype callable。每次运行都生成完整 action-set 签名。

- [ ] **Step 3: 验证 worker 0 门**

要求所有 optimized 签名与对应 baseline 签名逐字段一致，且 `baseline_median / optimized_median >= 2.0`。若不等价或未达 2.0，写出实际结果并停止，不运行全 8-state 扩展。

### Task 4: 执行 U80 全 8-state 对照并生成报告

**Files:**
- Modify: `D:/xunce/review/s6-u80-frontier-hotpath-prototype-r1/prototype_benchmark.py`
- Create at runtime: `D:/xunce/review/s6-u80-frontier-hotpath-prototype-r1/results.json`
- Create at runtime: `D:/xunce/review/s6-u80-frontier-hotpath-prototype-r1/report.md`

**Interfaces:**
- Consumes: 已通过的 worker 0 门。
- Produces: 用户可复核的最终原型证据，不产生生产源码改动。

- [ ] **Step 1: 对 8 个状态逐项运行**

每个状态至少收集 baseline/optimized 各 3 个 timed samples，记录逐状态 median、speedup、候选数、segment 数和完整等价结果。为避免运行顺序偏差，按状态索引奇偶交替 A/B 顺序。

- [ ] **Step 2: 写入严格 UTF-8 结果**

`results.json` 保存原始 samples、环境与 Python/NumPy 版本、checkpoint/source/prototype SHA-256、逐字段签名和聚合统计。所有 JSON 数值必须 finite，写入后重新以 UTF-8 读取验证。

- [ ] **Step 3: 生成人类可读报告**

`report.md` 必须明确区分 worker 0、8-state median、最慢状态和整 update 未测；列出是否达到 `2.0x`、是否严格等价、剩余热点以及所有生产/训练动作均未执行。

- [ ] **Step 4: 验证仓库源码未变化**

运行：

```powershell
git diff -- src/lunar_exploration_ppo/env/frontier.py
git status --short
```

将实验前保存的 `frontier.py` SHA-256 与实验后重新计算值比较，必须完全一致。现有其他 dirty 文件只报告、不修改。

- [ ] **Step 5: 暂停并请求用户确认**

向用户提供 `report.md`、`results.json` 和原型源码的绝对链接。不得修改生产源码、创建生产实现提交或启动训练；等待用户明确答复是否接受结果并授权替换。
