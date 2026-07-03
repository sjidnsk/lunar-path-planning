# Agent Notes

本文件只保存 agent 长期操作规则、当前主线关键合同和最近阶段摘要。禁止继续把每个阶段完整说明追加到 AGENTS.md；完整计划放在 `docs/superpowers/plans/`，真实结果放在 `outputs/.../report.md`，文件职责见 `docs/xunce-stage-documentation-index.md`。

## 沟通语言

- 默认使用中文回答。
- 用户明确要求其他语言时，按用户指定语言回答。

## 文件管理与危险操作

- 默认禁止批量删除文件或目录。
- 禁止使用 `del /s`、`rd /s`、`rmdir /s`、`Remove-Item -Recurse`、`rm -rf`。
- 需要删除文件时，只能一次删除一个明确路径的文件。
- 递归移动、批量移动、大范围覆盖都按危险操作处理；执行前必须确认目标路径明确、范围受控。
- 允许读取文件、搜索文本、列目录、查看 git 状态、运行测试和 dry-run 命令。

## UTF-8 编码与中文路径/中文文档

- 所有包含中文的 `.md`、`.json`、`.py`、`.txt` 文件必须按 UTF-8 写入。
- 不要把 PowerShell / cmd 终端显示结果当作中文是否正确的唯一依据；终端可能因为 code page 显示成 mojibake。
- 禁止从乱码终端输出中复制中文再回写到文件。
- 禁止用容易受终端编码影响的方式写入中文内容，例如 `echo 中文 > file`、`cat > file`、未显式指定编码的 `Set-Content` / `Out-File`。
- 修改中文文档时优先使用 `apply_patch`；如果 shell 通道已经污染中文，改用 Python `Path(...).write_text(text, encoding="utf-8")` 或 PowerShell `Set-Content -Encoding utf8NoBOM`。
- 在脚本或测试里写入中文锚点时，优先使用 Python Unicode escape，例如 `"\u4e2d\u6587\u8bf4\u660e"` 表示 `中文说明`。
- 写入或修改中文路径/中文文档后，必须用 Python 显式 UTF-8 读取验证，不要只看 PowerShell 输出。
- 如果看到明显 mojibake 片段，不要继续复制或提交；先用 `Path(...).read_text(encoding="utf-8")` 检查文件真实内容。
- 测试文件中不得把 mojibake 字符串作为期望值；需要检查中文内容时，必须使用真实 UTF-8 中文或 Unicode escape。

## 下载与临时资源位置

- 默认把下载资源、数据集、模型、缓存、导出文件等保存到 `D:\CodexDownloads`。
- 不要默认把大文件、临时下载、数据集、模型、构建产物、Node cache、训练输出放在 C 盘。
- 小型测试产物、pytest cache、项目 ignored output 目录可以按工具默认行为生成，但不应提交进 Git。
- 如果由于工具限制只能写入 C 盘，必须在回复中说明完整路径、用途、是否可删除和建议删除时机。

## Git 与用户改动保护

- 默认认为工作区已有改动可能来自用户或其他任务。
- 禁止执行 `git reset --hard`、`git checkout -- <path>`、大范围 revert、批量清理未跟踪文件。
- 提交时只纳入当前任务相关文件。
- 无关 dirty 文件应忽略；若无关改动影响当前任务，先说明风险再继续。

## Subagent-Driven 使用偏好

- 多文件实现、重构、测试补齐或文档治理任务，先判断是否适合 `superpowers:subagent-driven-development`。
- 适合条件：任务可拆为相对独立模块，且需要主 agent 做架构把关、代码审查和最终整合。
- 不适合条件：改动强耦合在少数共享文件、问题尚未定位、或用户明确要求不要使用 subagent。
- 无论是否使用 subagent，主 agent 对范围、边界、测试和最终验收负责。

## 当前长期硬边界

- 不发布 checkpoint。
- 不替换 default policy。
- 不连接 executor。
- 不启动 canary。
- synthetic terrain 只能标记为 proxy，不得写成或宣称 `physical_obstacle_cells`。
- default A* 不被替换；Hybrid A* 仍是 opt-in path-cost / pose-planner source。
- Hybrid A* 不宣称 Ackermann feasible。
- 平台对齐硬坡度阈值保持 `max_traversable_slope_deg=30.0`。

## 当前主线合同

- `coverage_source=endpoint_theta_slope_obstacle_los/v1`
- `path_cost_source=hybrid_astar_pose_path/v1`
- `synthetic_source_kind=synthetic_terrain_obstacle_proxy/v1`
- `action_space_type=hybrid_discrete_xy_continuous_theta/v1`
- `hybrid_astar_candidate_eval_workers=4` 是 Stage26 synthetic terrain policy-signal 诊断默认并行 collector 设定。
- 当前阶段是 Stage26.8M `generalized_resumable_training_pipeline`；把 26.8D/8H/8I 的局部恢复器统一成可配置、可恢复、可审计的训练级 pipeline，状态文件放在 D 盘 output root。

## Stage26.7C-26.7H Main-Coverable Efficiency And Eval Binding

- Stage26.7C 已把成功判据切到 `main_coverable_cells/v1` 与单位路程主覆盖率；`hybrid_astar_path_cost_delta` 只作为诊断字段。
- Stage26.7D 修复 synthetic credit sampler 的 continuous theta reachability 与 behavior point/theta logprob 合同，不改 reward、network 或 Hybrid A* 搜索语义。
- Stage26.7G 已证明 PPO update 稳定性应使用 policy-vs-policy KL；behavior-policy KL 只作 off-policy diagnostic。
- Stage26.7H 当前只修 post-update eval binding：区分 inference fields missing、explicit selected theta unreachable 与 valid coverage-efficiency result。
- Stage26.8 将 AUC 降级为早期覆盖节奏诊断，使用 `main_coverage_per_100m_delta` 作为 synthetic terrain multi-seed pilot 的主指标。
- Stage26.8A 只扩大 horizon 预算，不同时扩 seed；若 H12/H16/H20 都为 0，则回到 policy signal / credit target 诊断。
- Stage26.8B 区分“synthetic sidecar 未加载”和“长 horizon 末端无 Hybrid A* 可达候选”；只有已达最小训练样本、lineage/safety 干净的 `no_hybrid_reachable_candidate_terminal` 可作为自然终止。
- Stage26.8C 不复用旧 H16 failed root；H12 只作为 baseline audit，H16/H20 写入新 root 并继续以 `main_coverage_per_100m_delta` 为主判据。
- Stage26.8D 不新增算法能力，只提供可恢复实验流水线；Stage26.8C partial artifacts 只能只读 carryover，failed/incomplete root 不得当作成功。
- Stage26.8F 不推进 Stage26.8D job，不运行 Stage26.1/26.2/26.3；它只读已完成 Stage26.3 pre/post artifacts，并把 H16/H20 pending job 标记为 source incomplete。
- Stage26.8G 不改 reward、PPO、network、Hybrid A* 或 synthetic terrain；它只新增 `scenario_diversity_source=synthetic_roi_start_seed_matrix/v1` 与可审计 scenario fixture。
- Stage26.8M 是后续长时间 PPO update-strength / sample-count 实验的首选通用可恢复 runner；它只编排 `collector -> update -> eval_pre -> eval_post -> aggregate`，状态文件放在 D 盘 output root，不改 reward、network、Hybrid A* 或 synthetic terrain。

## Stage26.3 Synthetic Terrain Post-Update Eval

- Stage26.3 对 Stage26.2 experimental checkpoint 做 Stage21.5 pre/post eval。
- 强 join key 包含 `synthetic_terrain_hash`；弱 join 不得用来声称动作概率或轨迹变化。
- 结果显示动作概率与 selected `(x,y,theta)` 基本不变，进入 Stage26.4。

## Stage26.4 Synthetic Policy Update Signal Strength Repair

- Stage26.4 使用 worker=4 collector，扩充到 16 条 trainable transition 并跑多个 update combo。
- best combo 为 policy-amplified depth；概率有变化但未跨过离散候选选择边界。
- 该阶段是离线诊断，不是性能结论。

## Stage26.5 Synthetic Discrete Margin Crossing Calibration

- Stage26.5 跳过 Stage26.4A 串并行等价验收，直接审计 Stage26.4 worker=4 结果。
- 诊断结论：best synthetic candidate 没有被采样成 trainable selected action，未获得直接 PPO credit。
- 同时发现 candidate feature 缺少 synthetic LOS / hard obstacle / Hybrid A* path-cost 候选级信号。

## Stage26.6 Synthetic Exploration Credit Assignment

- Stage26.6 给 8 维候选输入槽写入 synthetic/Hybrid/coverage 语义图，保持网络结构不变。
- Stage21.1 新增 `synthetic_credit_mixture_policy/v1`，让 synthetic credit target 成为真实 selected action 并获得 direct PPO credit。
- `old_log_prob` 使用 behavior total logprob，同时保留 `old_policy_*` 与 `old_behavior_*` 审计字段。
