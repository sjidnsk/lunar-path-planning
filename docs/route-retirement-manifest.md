# 退役路线清单与人工清理边界

## 决策与审计快照

本清单记录用户于 2026-08-01 确认的路线退役决定，并把“路线退役”与“允许物理删除”严格分开。退役范围为：Xunce Stage18--25 演进链、Xunce Stage26 synthetic terrain / PPO diagnostics、path-feedback / 早期策略实验，以及 `model-explorer`、`visual-workbench` 子模块。清单仅定义后续人工清理边界；提交本清单不授权或执行任何源码、子模块或产物的物理清理。

只读审计给出的 Stage18--25 候选计数为：57 个 runner、64 个 config、59 个 test、78 个 plan doc；Stage26 候选计数为：38 个 runner、39 个 config、37 个 test、34 个 plan doc。这些是待逐路径核验的文件族计数，不是 glob 删除授权。

## 保留边界

| 路径或文件族 | 生命周期类别 | 保留路线依赖结果 | 物理删除是否允许 |
| --- | --- | --- | --- |
| 高分辨率前沿 PPO / Stage6：`src/lunar_exploration_ppo/`、Stage6 config、runner 与测试 | 保留的当前路线 | 用户确认保留；不得由本清单影响 | 不允许 |
| 中期缩减规模双门槛 G1/G2/G3 | 保留的交付链 | 直接导入 `scripts/xunce_artifact_io.py` 与 `scripts/xunce_artifact_paths.py` | 不允许 |
| 多平台路径规划 v3 | 保留的 opt-in 路线 | 用户确认保留；不连接 executor | 不允许 |
| `dev-platform-constraints/` | 保留的子模块 | 用户确认保留的平台约束 | 不允许 |
| `path-planner/` 与其 Python A* 实现 | 默认规划算法依赖 | `src/lunar_exploration_ppo/integrations/path_planner_adapter.py` 导入 `path_planner.search.AStarPlanner` | 不允许 |
| `scripts/xunce_artifact_io.py`、`scripts/xunce_artifact_paths.py` | 保留的共享基础设施 | G1/G2/G3 直接导入 | 不允许 |

默认 grid A* 保持为高分辨率前沿 PPO 使用的默认算法；Hybrid A* 仍仅为 opt-in 的 path-cost / pose-planner source，且不宣称 Ackermann feasible。不得改变 PPO、默认 policy、安全阈值、executor 或 checkpoint 发布边界。

## 退役源码与注册表候选

| 路径或文件族 | 生命周期类别 | 保留路线依赖结果 | 物理删除是否允许 |
| --- | --- | --- | --- |
| Stage18--25：57 runners（`scripts/run_xunce_stage18_*` 至相关 Stage25 runner） | 用户确认退役；源码候选 | 仍须逐路径核验 Stage6、G1/G2/G3、v3、平台约束和默认 A* 无引用 | 不允许；宽路径族必须人工逐路径执行 |
| Stage18--25：64 configs、59 tests、78 plan docs | 用户确认退役；配置/测试/文档候选 | 须核验注册表、文档入口及 retained-route 引用；不可只删其中一种文件族 | 不允许；宽路径族必须人工逐路径执行 |
| Stage26：38 runners、39 configs、37 tests、34 plan docs | 用户确认退役；synthetic terrain / PPO diagnostics 候选 | 须核验共享 artifact helper、注册表、Stage6/G1/G2/G3 与默认 A* 边界无引用 | 不允许；宽路径族必须人工逐路径执行 |
| path-feedback / 早期策略实验：`scripts/run_*path_feedback*`、`configs/path_feedback_*` 及对应测试/文档/注册项 | 用户确认退役；实验路线候选 | 须核验保留路线不读取其源码、配置或最小证据索引 | 不允许；宽路径族必须人工逐路径执行 |
| `configs/stage_registry.json` 中上述路线的注册项 | 退役后的注册表候选 | 只能在逐项核验 runner、config、测试和 retained-route 引用后移除 | 不允许；不得批量改写或删除 |

## 退役子模块候选

| 路径或文件族 | 生命周期类别 | 保留路线依赖结果 | 物理删除是否允许 |
| --- | --- | --- | --- |
| `model-explorer` gitlink 与 `.gitmodules` 对应 stanza | 用户确认退役；子模块候选 | 必须先完成 retained-route reference check，确认 Stage6、G1/G2/G3、v3、平台约束、默认 A* 均无引用 | 不允许；gitlink 与 stanza 仅可在检查后由人工逐项处理 |
| `visual-workbench` gitlink 与 `.gitmodules` 对应 stanza | 用户确认退役；子模块候选 | 必须先完成 retained-route reference check；不得影响 `dev-platform-constraints/` | 不允许；gitlink 与 stanza 仅可在检查后由人工逐项处理 |

两个子模块的本地工作目录需要另行获得明确的人工操作确认；本清单不授权 agent 删除、移动或递归清理其工作目录。

## 可再生本地产物候选

| 路径或文件族 | 生命周期类别 | 保留路线依赖结果 | 物理删除是否允许 |
| --- | --- | --- | --- |
| `outputs/pytest-*` | 高置信度可再生产物 | 确认没有运行中的 pytest，且无保留路线将旧目录作为生产输入后可清理 | 不允许；必须由人工按明确目录逐路径处理 |
| `outputs/path_feedback_batch_*` | 退役实验产物，需保留最小索引 | 清理前须为每个目录索引 `summary.json`、`manifest.json`、`routing.json`、`report.md` 与 checkpoint pointer | 不允许；不得自动删除 |
| `build/`、`.pytest_cache/` | 可再生本地产物 | 仍需确认没有并发构建或保留路线运行依赖 | 不允许；须人工逐路径处理 |

## 需要人工确认的交叉依赖

1. 对每个 Stage18--26 与 path-feedback 候选，检查 import、动态加载、`configs/stage_registry.json`、文档入口和测试 fixture，确认未被 Stage6、G1/G2/G3、v3、`dev-platform-constraints` 或默认 grid A* 使用。
2. `path-planner` 及 Python `AStarPlanner` 为保留依赖，任何 retirement/delete candidate 均不得包含它们；同时保持 `PathPlannerAdapter` 的 import/call 边界有效。
3. `scripts/xunce_artifact_io.py` 和 `scripts/xunce_artifact_paths.py` 必须保留，因为 G1/G2/G3 直接导入；清理 Xunce 文件时不得将共享 helper 误归类。
4. 子模块处理前先检查父仓库 `.gitmodules`、gitlink、保留路线引用和本地未提交状态；本地目录清理另行人工确认。
5. 用户拥有的未跟踪文档及 `hello.obj`、`schema_codec_conformance_test.obj` 等 `.obj` 文件必须保留，不得纳入候选。

## 删除前检查与逐路径执行规则

1. 先在只读模式生成并复核精确路径清单，记录每一项的生命周期、依赖检查结果和最小保留证据。
2. 每一项必须经人工确认后单独执行；禁止 glob 删除、递归删除、批量移动或大范围覆盖。
3. 对 `outputs/path_feedback_batch_*`，先完成五类最小索引和 checkpoint pointer 记录，再由人工决定归档或清理大文件。
4. 对 `outputs/pytest-*`，先确认无测试进程占用；如遇权限或 ACL 问题，停止并人工处理，不得接管权限或强删。
5. 清理后复查保留路线的 import、注册表、默认 grid A* 与 `PathPlannerAdapter` 边界；任何失败均停止后续操作并恢复到人工审查。

## 人工物理清理执行包

本执行包只规定负责人在另一次明确授权后如何生成、核验和逐项批准精确路径；它不是删除授权，也不包含实际删除命令。禁止递归、批量或通配符删除命令，包括 `Remove-Item -Recurse`、`rm -rf`、`del /s`、`rd /s` 与 `rmdir /s`。即使某项通过所有检查，也必须在负责人针对该单一路径再次确认后，才可由负责人选择受控的人工处理方式。

### 通用逐路径核验协议

1. 负责人先以只读方式枚举候选族，并将结果展开为单个、可复制的精确路径；不得把目录名、glob、正则或文件族名称直接当作处理目标。
2. 对清单中的每一个精确路径分别记录以下四项只读结果：`Test-Path -LiteralPath <exact-path>`、`git ls-files --error-unmatch -- <exact-path>`、`git check-ignore -v -- <exact-path>`，以及以精确路径或其唯一文件名进行的 `rg` 交叉引用检查。交叉引用范围至少覆盖保留路线、`configs/stage_registry.json`、`.gitmodules`（如适用）、脚本、配置、测试和文档入口。
3. 预期之外的结果包括但不限于：路径不存在或类型与清单不符、被 Git 跟踪状态不符合候选记录、意外被忽略、`rg` 找到保留路线或未审查的引用、无法读取，或检查结果彼此矛盾。任一项出现预期之外的结果，立即停止并保留该路径，不得推断其可处理性。
4. 每个通过核验的路径仍须附上生命周期理由、四项检查记录、负责人逐路径确认和处理后的复核记录。不得将一个目录或一个文件族的一次确认扩展到其他路径。

### 目标族与前置检查（固定顺序）

1. **`outputs/pytest-*`**：先枚举为单个精确目录或文件路径，并确认没有运行中的 pytest 使用该路径。对每个精确路径执行通用逐路径核验协议；`rg` 特别检查其是否被保留路线作为输入或证据引用。仅在负责人逐路径确认后，才可人工处理。
2. **`outputs/debug-*`**：先枚举为单个精确目录或文件路径。对每个精确路径执行通用逐路径核验协议，并记录其是否仍关联活跃调试、复现实验或保留路线文档。任何活跃引用、用途不明或检查异常都要求停止并保留该路径。
3. **`outputs/path_feedback_batch_*`**：不得根据名称直接处理。负责人必须先为每个精确目录建立最小索引，分别记录 `summary.json`、`manifest.json`、`routing.json`、`report.md` 是否存在及其可读性，并记录 checkpoint pointer 的位置、指向对象和保留需求；之后才对目录内每个拟处理的精确路径执行通用逐路径核验协议。只有负责人额外确认后，才可逐项人工处理。
4. **`model-explorer` gitlink**：先以精确 gitlink 路径、对应 `.gitmodules` stanza 及父仓库引用分别建项；对每项执行通用逐路径核验协议，并完成保留路线引用核验。只有负责人额外确认后，才能进入下面的子模块 Git 顺序。
5. **`visual-workbench` gitlink**：先以精确 gitlink 路径、对应 `.gitmodules` stanza 及父仓库引用分别建项；对每项执行通用逐路径核验协议，并完成保留路线引用核验。只有负责人额外确认后，才能进入下面的子模块 Git 顺序。
6. **Xunce Stage18--26 的 root scripts/configs/tests/docs 文件族**：先将 root `scripts/`、`configs/`、`tests/` 和 `docs/` 中的候选展开为单个精确文件路径；不得根据阶段名或通配符直接处理。每个精确路径都执行通用逐路径核验协议，`rg` 必须额外核验 Stage6、G1/G2/G3、v3、默认 grid A*、`dev-platform-constraints`、共享 artifact helper、注册表及文档入口。只有负责人额外确认后，才能逐项人工处理；任一引用或不确定性都使该路径保留。

### 待退役子模块的 Git 顺序

对 `model-explorer` 与 `visual-workbench`，必须按以下顺序分别处理，且每一步均需保留审计记录：

1. 完成保留路线引用核验，确认 Stage6、G1/G2/G3、v3、默认 grid A*、共享基础设施及平台约束均不依赖该子模块。
2. 获得负责人对该单一子模块的额外确认。
3. 仅修改该子模块对应的单个 `.gitmodules` stanza、单个 gitlink 与其专属父仓库引用；不得顺带修改其他子模块或不相关配置。
4. 复核 `.gitmodules`、git 索引、保留路线引用和父仓库状态，任何异常均停止并保留现场供人工审查。
5. 最后才由负责人另行决定本地工作目录的处理；本执行包不授权 agent 对本地目录执行删除、移动或重命名。

`path-planner` 与 `dev-platform-constraints` 明确不在上述子模块退役或物理清理流程中，必须保持原状。
