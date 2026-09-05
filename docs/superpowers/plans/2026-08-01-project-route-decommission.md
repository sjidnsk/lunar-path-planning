# Project Route Decommission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将用户明确退役的 Xunce Stage18--26、path-feedback、`model-explorer` 与 `visual-workbench` 从“当前主线”中移出，并生成一份不会误伤 Stage 6、G1/G2/G3、v3、平台约束或默认 A* 的精确清理清单。

**Architecture:** 先以只读依赖审计为事实来源，生成可追溯的退役清单；随后只修改项目导航和生命周期描述，不删除源码、子模块或输出。物理删除被拆为负责人手工执行的最后一步，因为该仓库禁止批量/递归删除，且 Stage 6 对 `path-planner` 的默认 `AStarPlanner` 有运行时身份绑定。

**Tech Stack:** Markdown、Git submodule、Python/pytest 静态引用检查、PowerShell。

## Global Constraints

- 用户明确保留：高分辨率前沿 PPO / Stage 6、中期缩减规模双门槛（G1/G2/G3）、多平台路径规划 v3、`dev-platform-constraints`、以及 Stage 6 默认的 grid `AStarPlanner`。
- 用户明确退役：Xunce Stage18--26、path-feedback / 早期策略实验、`model-explorer` 子模块、`visual-workbench` 子模块。
- `src/lunar_exploration_ppo/integrations/path_planner_adapter.py` 直接导入 `path_planner.search.AStarPlanner`；不得删除 `path-planner` 子模块、`path_planner.core`、`path_planner.search` 或 Stage 6 的 planner identity 代码。
- 默认 grid A* 不被替换；Hybrid A* 和 v3 不得被描述为 Stage 6 的默认运行时 planner。
- 不执行 `Remove-Item -Recurse`、`rm -rf`、`del /s`、`rd /s`、`rmdir /s` 或任何批量/递归删除。
- 不移动、不删除历史 D 盘 outputs；不修改用户现有未跟踪的文档、外部输入目录或 `.obj` 文件。
- 所有新增/修改中文文档使用 UTF-8，并以 Python 显式 UTF-8 读取验证。

---

### Task 1: 生成退役边界与最小留存清单

**Files:**
- Create: `docs/route-retirement-manifest.md`
- Modify: `docs/project-route-map.md`
- Test: `README.md`, `AGENTS.md`, `configs/stage_registry.json`, `.gitmodules`

**Interfaces:**
- Consumes: 用户的保留/退役路线决定，以及 `rg`、Git submodule 和 stage registry 的静态审计结果。
- Produces: 一份含 `retire_now`、`retain_for_active_route`、`manual_confirmation_required` 三类条目的 Markdown 清单；后续文档更新只能引用该清单，不自行扩大删除范围。

- [ ] **Step 1: 记录每条退役路线的静态边界**

运行：

```powershell
rg -l "stage1[89]|stage2[0-6]|path_feedback" scripts configs tests docs src
git submodule status
git ls-files -s | Select-String '^160000'
```

期望：列出 Xunce/root path-feedback 文件族和两个待退役 gitlink；不把 `path-planner` 或 `dev-platform-constraints` 列入输出。

- [ ] **Step 2: 记录保留主线对旧路线的直接引用**

运行：

```powershell
rg -n "from path_planner|import path_planner|AStarPlanner|model_explorer|visual-workbench|path_feedback" src scripts configs tests README.md AGENTS.md
```

期望：明确 Stage 6 的 `path_planner.search.AStarPlanner` 依赖；任何 `model_explorer`、`visual-workbench` 或 path-feedback 命中都必须写入 `manual_confirmation_required`，不能只靠名称判断。

- [ ] **Step 3: 写入 `docs/route-retirement-manifest.md`**

文档必须有以下固定章节：

```markdown
## 保留边界
## 退役源码与注册表候选
## 退役子模块候选
## 可再生本地产物候选
## 需要人工确认的交叉依赖
## 删除前检查与逐路径执行规则
```

每个条目必须包含：路径或文件族、分类、保留/退役理由、直接引用证据、物理删除是否允许。`outputs/pytest-*` 只作为可再生临时产物候选；`outputs/path_feedback_batch_*` 必须先保留 `summary.json`、`manifest.json`、`routing.json`、`report.md` 的索引。

- [ ] **Step 4: 把生命周期决定回写到路线图**

在 `docs/project-route-map.md` 中把 Stage18--26、path-feedback、`model-explorer` 和 `visual-workbench` 的状态改为“用户确认退役”，同时新增“默认路径规划算法”行：`src/lunar_exploration_ppo/integrations/path_planner_adapter.py` 通过 `path_planner.search.AStarPlanner` 使用默认 grid A*。

- [ ] **Step 5: 验证清单的编码与交叉引用**

运行：

```powershell
@'
from pathlib import Path
for name in ('docs/route-retirement-manifest.md', 'docs/project-route-map.md'):
    text = Path(name).read_text(encoding='utf-8')
    assert '保留边界' in text or '路线总览' in text
    assert '\ufffd' not in text
print('UTF8 and anchors verified')
'@ | python -
```

期望：命令打印 `UTF8 and anchors verified`。

### Task 2: 修正当前主线的文档入口

**Files:**
- Modify: `README.md`
- Modify: `AGENTS.md`
- Modify: `docs/xunce-stage-documentation-index.md`
- Test: `docs/route-retirement-manifest.md`

**Interfaces:**
- Consumes: Task 1 的三类退役清单。
- Produces: 三个入口对当前主线的一致说明；历史 Stage26 仍可由退役清单定位，但不再显示为 current route。

- [ ] **Step 1: 更新 README 的中英文当前路线**

将“当前路线”改为同一组保留路线：

```text
高分辨率前沿 PPO / Stage 6
中期缩减规模双门槛（G1/G2/G3）
多平台路径规划 v3
平台约束
默认 grid A* 路径规划
```

明确写出：Stage18--26、path-feedback、`model-explorer` 与 `visual-workbench` 已退役，完整边界见 `docs/route-retirement-manifest.md`。不得删掉默认 A*、Hybrid A* opt-in 和不连接 executor 的安全边界。

- [ ] **Step 2: 收缩 AGENTS.md 中的已退役 Stage26 叙述**

保留项目操作规则、默认 A*、安全阈值、Stage 6/v3/平台约束边界；将“当前主线合同”和所有 Stage26 专属的长期阶段摘要替换为一段退役说明及 `docs/route-retirement-manifest.md` 链接。不得把 retired 路线的安全边界误写成已解除。

- [ ] **Step 3: 把 Xunce Documentation Index 改为历史索引**

把 `## Current Stage26 Mainline` 改为 `## Retired Xunce Historical Line`，并把“Current next route”替换为“不可作为新开发入口；参见 retirement manifest”。保留每个历史 plan/report 的定位规则，避免破坏可追溯性。

- [ ] **Step 4: 验证入口一致性**

运行：

```powershell
rg -n "Stage26.*(当前主线|Current Route|current route)|repair_stage26" README.md AGENTS.md docs/xunce-stage-documentation-index.md
rg -n "AStarPlanner|default grid A\*|默认.*A\*" README.md AGENTS.md docs/project-route-map.md
git diff --check
```

期望：第一条命令没有把 Stage26 标为当前入口的命中；第二条命令在至少一个 README/AGENTS/路线图位置确认默认 A*；`git diff --check` 无错误。

### Task 3: 为物理清理准备人工执行包

**Files:**
- Modify: `docs/route-retirement-manifest.md`
- Test: `git status --short`, `git diff --check`

**Interfaces:**
- Consumes: Task 1 的清单和 Task 2 的文档入口。
- Produces: 负责人可逐项批准的手工删除顺序；不由 agent 执行删除。

- [ ] **Step 1: 为每个待删除顶层目标记录前置检查**

文档按以下顺序列出，并为每项要求 `Test-Path`、`git ls-files --error-unmatch`、`git check-ignore -v` 和 `rg` 交叉引用检查：

```text
outputs/pytest-*
outputs/debug-*
outputs/path_feedback_batch_*
model-explorer (gitlink)
visual-workbench (gitlink)
Stage18--26 root scripts/configs/tests/docs
```

`outputs/path_feedback_batch_*`、gitlink 和 Stage18--26 必须标为“负责人确认后才可执行”；不得把通配符转换为递归删除命令。

- [ ] **Step 2: 明确子模块退役的 Git 顺序**

在 manifest 中写明：先确认父仓库不存在保留主线引用，再移除 `.gitmodules` 的单个条目、对应 gitlink 和仅属于该子模块的文档引用；最后才由负责人删除本地子模块工作目录。`path-planner` 和 `dev-platform-constraints` 不得出现在该步骤。

- [ ] **Step 3: 验证手工包没有授权过宽命令**

运行：

```powershell
rg -n "Remove-Item -Recurse|rm -rf|del /s|rd /s|rmdir /s" docs/route-retirement-manifest.md docs/superpowers/plans/2026-08-01-project-route-decommission.md
git diff --check
```

期望：第一条命令只有“禁止使用”的规则文本，不包含可执行的批量删除命令；`git diff --check` 无错误。

## Execution Handoff

本计划的 Task 1--3 可以通过 Subagent-Driven 执行并完成非破坏性退役。物理删除、子模块移除和大量 stage 文件移除需要负责人对 manifest 的精确路径清单再次确认；这不是 agent 的默认执行权限。
