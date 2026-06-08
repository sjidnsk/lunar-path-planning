# Agent Notes

## 沟通语言

- 默认使用中文回答，除非用户明确要求使用其他语言。

## Notion 导出规则

- `D:\codex\project\lunar-path-planning\docs\月面巡视探索.md` 是 Notion 页面“月面巡视探索”的单向导出副本。
- Notion 为主源，本地 Markdown 文件为副本。
- Notion 内容更新后，需要重新导出覆盖本地文件。
- 本地文件应保持 UTF-8 编码。

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
