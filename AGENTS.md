# Agent Notes

## 沟通语言

- 默认使用中文回答，除非用户明确要求使用其他语言。

## Notion 导出规则

- `D:\codex\project\lunar-path-planning\docs\月面巡视探索.md` 是 Notion 页面“月面巡视探索”的单向导出副本。
- Notion 为主源，本地 Markdown 文件为副本。
- Notion 内容更新后，需要重新导出覆盖本地文件。
- 本地文件应保持 UTF-8 编码。

## 项目目标优先级

- 探索任务的最高目标是整场任务最终覆盖率超过 99%。当覆盖率尚未达到 99% 时，后续方案、reward 调整、审计和阶段推进应优先解释并解决覆盖不足问题。
- 路径成本、soft risk exposure、coverage efficiency 是重要约束和次级优化目标，但不能替代 99% 覆盖率这个主指标。不要因为路径更短或风险代理更低，就把低覆盖方案描述为任务成功。
- 风险系统的职责是过滤不允许走的路径；reward 的职责是在允许路径中选择更高覆盖、更低成本的行动。hard risk violation 必须保持为 0；soft risk 只应在可接受范围内约束探索，不应把策略压成过度保守。
- 路径成本优化的目标是在保持或推进 99% 覆盖率的前提下降低绕路。若出现“覆盖率明显不足但路径成本很好”的结果，应优先视为覆盖目标未完成，而不是成功收敛。
- 阶段报告和下一阶段计划必须明确区分：最终覆盖率、覆盖增量、路径成本、单位路径覆盖效率、hard risk violation 和 soft risk exposure。结论排序应以“是否接近或达到 99% 覆盖率”为第一判断，再讨论成本和风险是否可接受。

## Goal 模式规则

- Goal 模式的 prompt 不能超过 4000 字符。
- 当用户要求“下一阶段 goal 模式 prompt”或“goal 模式完整 prompt”时，输出应是可直接粘贴执行的 `/goal` 文本，而不是泛泛路线图。
- Goal prompt 必须基于当前仓库状态、最新 evidence/root、readiness blocker 和已实现/未完成边界来写；不要脱离当前证据重新发散方案。
- Goal prompt 默认包含：背景、目标、范围、验收标准、验证命令、产物路径、非目标。
- 每次编写下一阶段 Goal prompt 时，必须把“更新项目文档”写入范围和验收标准；已知文件时明确列出，例如 `docs/算法设计与系统架构报告.md` 与 `docs/superpowers/specs/`。
- Goal prompt 应保留关键 scope guards，例如不启动 PPO、不修改 network/action space/default A*、不宣称 Ackermann-feasible trajectory、不把 IRIS/GCS 诊断当训练放行。
- 若用户明确要求 4000 字符以内，应从一开始按该预算压缩，优先保留验收门禁、关键 artifact、验证命令和非目标，删减重复解释。

## 开发环境

- 项目开发和验证默认使用 Conda 环境 `lunar-explorer`：`/home/kai/anaconda3/envs/lunar-explorer`。

## Stage 19 / Stage 20 证据边界

- Stage 19 的职责是 evaluator / critic preflight：把 Stage 18.11 的 reward-rerank oracle 诊断结果整理成人工审查证据，不训练模型。
- 当前 practical target 为 `candidate_count=36`、`path_cost_weight=0.1`；判断 99% 覆盖目标时必须使用 capped final coverage，raw coverage 超过 1.0 只能作为饱和诊断。
- 固定 Xunce checkpoint 是否真正学会 oracle 行为必须单独说明；当 `xunce_checkpoint_advantage_established=false` 时，不得把 oracle 结果写成 Xunce 模型优势。
- Stage 19 通过后的下一跳可以是 `stage20_reward_rerank_oracle_preference_dataset_preparation`，但 `stage20_authorized=false`。Stage 20 仍是偏好证据准备和人工审查，不是 PPO 训练、checkpoint 发布、default policy 替换、executor 连接或 canary 启动。
