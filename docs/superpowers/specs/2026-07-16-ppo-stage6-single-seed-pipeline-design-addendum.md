# PPO Stage 6 单 Seed 管线设计补充

**状态：** 用户已于 2026-07-16 批准。

**适用范围：** 本补充覆盖原设计中 Stage 6 的训练 seed 数量、Stage 7 的 checkpoint 来源，以及 Standard 场景起点安全合同。其余网络、PPO、reward、planner、observation、评估公平性和资源门保持不变。

## 1. Stage 6A：单 Seed 系统闭环

Stage 6A 只训练 seed `20260716`，仍完整执行 100 次 PPO update。每 10 次 update 执行 16 episode deterministic validation；保存 latest、update 50/100 periodic 和 validation best。validation best 冻结后执行 64 test、64 unseen，并在相同场景、预算、候选、planner、sensor 和 seed 合同下执行四个 baseline。

Stage 6A 的通过含义仅为：训练、恢复、checkpoint、评估、公平性、泄漏审计、资源审计和复现链完整。单 seed 结果只能作为描述性潜力证据，不得形成跨 seed 稳健性或论文级性能结论；未击败 `gain_over_cost_frontier` 仍不阻塞系统验收。

## 2. 可选 Stage 6B：额外 Seeds

额外 3–5 个独立训练 seeds 不是 Stage 1–8 的关键路径。只有用户后续明确说“追加”并指定范围后才允许启动；不得由指标、脚本或 agent 自动触发。

Stage 6B 不阻塞、不撤销也不延迟 Stage 6 Gate、Stage 7 或 Stage 8。若执行，它使用独立 run-id、配置、lineage、artifact 和审查，只生成额外的多-seed性能补充，不改写已经通过的 Stage 6A 证据。

## 3. Stage 7 绑定

Stage 7 直接使用 Stage 6A seed `20260716` 的 frozen validation-best checkpoint，完成 Kilometer eval-only 压力测试。Stage 7 不等待 Stage 6B，也不得因 Stage 6B 未执行而失败。

## 4. Standard deterministic exact-safe 起点

Standard 场景起点在 proxy 生成前确定，不能在环境 reset 时跳过坏场景，也不能在 collector 中重采样以隐藏目录缺陷。

起点选择按以下固定顺序执行：

1. 从真实 4m DEM window 确定性上采样得到 0.5m base height。
2. 使用 `max_traversable_slope_deg=30.0`、`traversability_threshold=0.50` 和 `min_clearance_m=0.5215874761` 计算 base exact-safe mask，且只做一次 clearance inflation。
3. 候选仍限制为 4m prior 的内圈低分辨率格中心映射到 0.5m cell center。
4. 优先从 `prior_traversability >= 0.75` 且 exact-safe 的候选中按既有 `start_pose_seed` 选择。
5. 优先集合为空时，只在 exact-safe 的内圈低分辨率格中心中回退；禁止回退到任意格。
6. exact-safe候选为空时 fail closed，不跳过场景。
7. 选定起点继续作为 6m proxy start-protection center；最终 proxy truth 中该起点必须再次通过相同安全合同。

该选择发生在场景生成边界，不进入 `PolicyObservation`，不得向 policy 泄漏 truth 或 `coverable_mask`。

## 5. 失败运行处置

`s6-standard-r2-20260716T133254Z` 绑定旧 5-seed配置和旧起点算法，且在 update 1 的 worker 5 第五个场景 `train/scenario-0506` 失败。该 run 永久只读，不恢复、不覆盖。修复后的 Stage 6A 必须使用新的唯一 run-id 和重新冻结的 source/config/environment lineage。

## 6. 验收

- 1064 个 Standard catalog 记录均存在 deterministic exact-safe 低分辨率中心候选。
- 回归场景 `train/scenario-0506` 的最终 proxy 起点位于 exact safe mask。
- 训练状态机恰好包含 100 个 update、10 次 validation 和 1 个完成 seed。
- final checkpoint 选择、test/unseen、四 baseline、资源/数学/checkpoint/lineage audit 全部执行。
- summary/report 明确标记 `single_seed_system_closure/v1` 和“无跨 seed 性能结论”。
- Stage 6 Gate 明确授权 Stage 7，不依赖可选 Stage 6B。

