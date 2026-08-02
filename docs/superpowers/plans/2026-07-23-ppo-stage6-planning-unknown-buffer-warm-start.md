# Stage 6 规划未知区缓冲与 U74 Warm-start 实施计划

> **执行约束：** 本计划是用户已批准设计的实施细化。使用 strict TDD、fresh implementer、主 agent 独立验证、fresh 规格审查和 fresh 质量审查。Critical/Important 必须修复并重审。实现者和 reviewer 不得 stage、commit、发布 source-repair、生成正式 launch authorization 或启动训练；Stage 6 Gate 获用户批准前不创建 Git 提交。

## 目标

在不改变 PPO 网络、动作张量和物理安全常量的前提下，新增 0.75m 未知区规划缓冲与 0.75m/360° reset 局部安全扫描。停止旧 run 在 U74 之后的旧语义训练，从完整 U74 checkpoint 建立新 lineage，清空旧 rollout 和 active episode state，并从 U75 继续到 U100。

## 冻结输入

- 设计补充：`docs/superpowers/specs/2026-07-23-ppo-stage6-planning-unknown-buffer-design-addendum.md`
- parent run：`s6-standard-single-r1-20260718T220434Z`
- parent seed：`20260716`
- parent update：`74`
- checkpoint SHA：`ce9ea8cb047b5acc0ffc4f5d63084f3d54312cf55ecb0b92b8fc20bf43791553`
- manifest SHA：`6c0e40d2dd9ea489f4c67a6a43868dc80751f07961f5db8ccf5c804213e340c5`
- complete SHA：`6cd6e6210283fc71a5205a727f82c07ff448458d4ac06ad8080277e2502960ef`
- policy-state SHA：`c1650b7bed6fa22387ac5b5fdb9d80a6aa0baf3a1d425e868d81586070c5db20`
- parent config SHA：`7b1c37673c105c34e7bd71d04d4f508d3a167cdf77426b0b35c7aff86ba8adae`
- parent lineage SHA：`c7a0a7bf98b5aa4a39094ec01a2a0031d87ad8fc71060323474b20ebb134530c`
- discarded partial：U75 attempt1 及其所有未 accepted 数据
- child first update：U75
- child final update：U100

## Task 1：TDD 实现 observed-only planning mask

**主要文件：**

- `src/lunar_exploration_ppo/env/map_state.py`
- `src/lunar_exploration_ppo/env/env.py`
- `src/lunar_exploration_ppo/env/frontier.py`
- `src/lunar_exploration_ppo/integrations/path_planner_adapter.py`
- 对应 env/frontier/planner tests

**步骤：**

1. 先写 RED：
   - 0.5m 正交未知邻格排除；
   - 约 0.707m 斜角未知邻格排除；
   - 1.0m 邻格不因 0.75m 规则排除；
   - exactly `<0.75m` 语义；
   - 修改未观测 truth 不改变 mask、candidate 或 observation；
   - observation 更新后 planning mask 确定性扩张；
   - 不改变 observed-safe 物理 mask。
2. 实现独立的 observed-only planning mask 构造，不复用会额外改变 ROI boundary 语义的物理 clearance helper。
3. frontier endpoint 和 reachability component 使用 planning mask。
4. A* 的输入 `passable_mask` 使用 planning mask；返回后逐格复核。
5. 保留 physical/post-observation safety 检查并为物理与未知区规划失败写不同 reason。

## Task 2：TDD 实现双 reset scan

**主要文件：**

- `src/lunar_exploration_ppo/configs/stage1.py`
- `src/lunar_exploration_ppo/configs/stage6.py`
- `src/lunar_exploration_ppo/env/standard_training.py`
- `src/lunar_exploration_ppo/env/env.py`
- `src/lunar_exploration_ppo/env/sensor_model.py`（仅在现有接口不足时最小修改）
- 对应 config/env/sensor/Standard tests

**步骤：**

1. 先写 RED，冻结 `0.75m/360°/1°` local safety scan 和现有 `20m/90°/1°` exploration scan 的顺序。
2. 证明两次扫描都不计 reward、step 或 stagnation。
3. 分别持久化两次 scan diagnostics。
4. 证明 local scan 使用 LOS 遮挡语义且不泄漏扫描范围/FOV 外 truth。
5. 运行冻结 16 场景 reset probe，验收：
   - 16/16 reset pose planning-safe；
   - 16/16 至少一个 planning-safe egress cell；
   - 无安全候选仍为 `no_candidate_done`，无 fallback。

## Task 3：TDD 实现 U74 child-lineage warm-start

**主要文件：**

- `src/lunar_exploration_ppo/ppo/standard_training.py`
- `src/lunar_exploration_ppo/workflows/stage6.py`
- `src/lunar_exploration_ppo/workflows/stage6_source_repair.py`
- 新增 `src/lunar_exploration_ppo/workflows/stage6_planning_warm_start.py`
- `scripts/create_ppo_stage6_source_repair_amendment.py`
- 新增 `scripts/create_ppo_stage6_planning_warm_start.py`
- `scripts/run_ppo_stage6_standard.py`
- 对应 checkpoint/workflow/source-repair/training tests

**步骤：**

1. 先写 RED fixture，包含完整 accepted U74、U75 attempt1 partial、父 checkpoint payload 和新 child root。
2. 新增职责独立的 fail-closed warm-start artifact/config contract；旧 run 的 ordinal1–6 source-repair 链保持只读历史，不得把 child 伪装成旧 run 的 ordinal7 exact resume。新 contract 绑定：
   - 已批准设计与计划；
   - 新源代码和测试；
   - U74 checkpoint/manifest/complete/journal；
   - discarded U75 attempt1；
   - 新 effective config hash；
   - fresh 双审和 launch authorization。
3. 只导入 model、optimizer、normalization stats、RNG 和 scenario sampler state。
4. 明确拒绝导入 `vector_env_states`、父 validation-best、best record 和 eval metrics；child 8 env 全部新建并 reset，best 只由 U80/U90/U100 新语义 validation 产生。
5. child journal 首事务必须是 U75；禁止 U74 重新训练、U75 attempt1 重放或从 U76 跳过。
6. checkpoint lineage 必须可反向验证到 parent U74。
7. parent run/source-repair 历史保持 immutable；新语义不得伪装为旧 run exact resume。

## Task 4：主验证与 fresh 双审

1. 主 agent 独立审阅 diff、实现者报告和 TDD RED/GREEN 证据。
2. 分进程运行：
   - planning mask / frontier / planner focused tests；
   - reset sensor / env tests；
   - checkpoint / warm-start / source-repair / workflow / training tests；
   - Stage 1–6 受影响回归；
   - `git diff --check`、UTF-8 和 import-boundary checks。
3. 生成只含本任务改动的 prospective review package。
4. 派 fresh 规格 reviewer；只有 C0/I0 后再派不同的 fresh 质量 reviewer。
5. 任一 Critical/Important 由 fresh fixer 按 TDD 修复并重审。

## Task 5：受控创建 child run 并恢复训练

1. 审核通过后生成新短 run id、effective config、warm-start artifact、source identity、machine preflight 和 launch authorization。
2. dry-run 验证：
   - 唯一 parent 为 U74；
   - U74 SHA/marker/journal 精确匹配；
   - U75 attempt1 明确废弃；
   - child 首事务为 U75；
   - 8 env 没有导入 parent episode state；
   - 新 reset probe 和 planning mask probe 通过；
   - 没有旧 runner/worker。
3. 只启动一个 child runner，验证 root PID、8 workers、GPU、D 盘、RSS 和首条 U75 pre artifact。
4. 更新并恢复现有 `ppo-stage-6` 30 分钟自动化，使其动态读取 child run 状态；不得创建重复 automation。
5. 后续只在 validation、性能显著变化、资源异常、runner 退出或 U100 终态时通知。

## 验收边界

- 物理 `observed_safe_mask` 与 0.5215874761m 常量未改变。
- planning mask 完全 observed-only，正交和斜角邻格合同通过。
- 候选 endpoint、A* 搜索和全部 path cells 使用同一 planning mask。
- reset 局部扫描与 exploration scan 均可审计。
- 无 fallback 动作。
- U74 只作为父 checkpoint；新语义从 U75 开始，U100 结束。
- final report 明确“74 个父语义 updates + 26 个新语义 updates”。
- 不发布 checkpoint、不替换 default policy、不连接 executor、不启动 canary、不运行额外 seed。
