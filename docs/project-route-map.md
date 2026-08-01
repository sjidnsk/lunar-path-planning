# 项目路线图与精简边界

> 审计快照：2026-08-01，Git `HEAD=9ca435e`。本文件根据当前工作树、`configs/stage_registry.json`、Git 最近提交和已有文档编制；它用于分类和清理决策，不替代阶段报告、运行证据或安全合同。

## 先读这份表

“旧”不等于“可删”。本仓库至少存在四种不同状态：

| 状态 | 含义 | 处理原则 |
| --- | --- | --- |
| 当前实现 | 最近仍有源码、测试或交付链改动 | 保留在主工作区。 |
| 当前支撑 | 被当前路线的接口、子模块或证据链使用 | 保留；不能按最后修改日期清理。 |
| 历史可复现 | 已完成的研究/诊断阶段，仍有注册入口或可能被后续路线读取 | 先保留源码、配置、测试和最小证据，再决定是否移入 archive。 |
| 明确终止/阻塞 | 文档或程序明确标注为 superseded、invalid、excluded 或 blocked | 只清理对应的运行产物；保留最小说明和可追溯记录。 |

任何没有明确状态的路线都标为“待负责人确认”，不得仅因文件较旧而删除。

## 路线总览

| 路线 | 当前判断 | 主要位置 | 证据与边界 |
| --- | --- | --- | --- |
| 高分辨率前沿 PPO / Stage 6 | **当前实现** | `src/lunar_exploration_ppo/`、`configs/ppo_highres_frontier_stage6_v1.json`、`scripts/*stage6*`、`tests/ppo_highres_frontier/` | 2026-08-01 仍有 Stage 6 与双门槛工作流提交。Stage 6 是新 PPO 探索任务，设计规格明确其不替代当时的 Stage26 主线。 |
| 中期缩减规模双门槛（G1/G2/G3） | **当前实现/交付链** | `scripts/*midterm*`、`src/lunar_exploration_ppo/eval/midterm_reduced.py`、`independent/g2_t2_producer/`、`docs/xunce-midterm-dual-gate-runbook.md` | 用于受限规模的证据交付；不是默认 planner 或策略替换路线。 |
| 多平台路径规划 v3 | **当前实现，opt-in** | `path-planner/cpp/`、`path-planner/schemas/v3/`、`docs/superpowers/specs/2026-07-28-multiplatform-path-planner-v3-*.md` | v3 是 clean-room C++20 路线；默认 grid A* 保持不变，不连接 executor。 |
| 默认路径规划算法 | **当前支撑，必须保留** | `src/lunar_exploration_ppo/integrations/path_planner_adapter.py`、`path-planner` Python 实现 | `PathPlannerAdapter` 导入并调用 `path_planner.search.AStarPlanner`；default grid A* 是高分辨率前沿 PPO 的默认规划算法，不得列入退役或删除候选。Hybrid A* 仍是 opt-in path-cost / pose-planner source，不宣称 Ackermann feasible。 |
| Xunce Stage26 synthetic terrain / PPO 诊断 | **用户确认退役** | `scripts/run_xunce_stage26_*`、`configs/xunce_stage26_*`、`tests/test_xunce_stage26_*`、相关 plans/specs | 用户确认退役；源码清理须等待退役清单完成及人工确认，不得批量删除或影响共享 artifact helper、默认 A*、Stage6、G1/G2/G3、v3 或平台约束。 |
| Xunce Stage18--25 演进链 | **用户确认退役** | 对应的 `run_xunce_*`、`configs/xunce_*`、`tests/test_xunce_*` 与阶段 plans | 用户确认退役；先做逐路径 retained-route 引用核验和最小证据审计，再由人工确认后处理。 |
| path-feedback / 早期策略实验 | **用户确认退役** | `model-explorer/` 子模块、`scripts/run_*path_feedback*`、`configs/path_feedback_*` | 用户确认退役；先做静态引用、最小再现和 `outputs/path_feedback_batch_*` 索引审计，禁止自动删除。 |
| `model-explorer` 与 `visual-workbench` 子模块 | **用户确认退役** | 父仓库 gitlink 与 `.gitmodules` stanzas | 仅在 retained-route reference checks 通过后才可人工逐项移除 gitlink/stanza；本地工作目录需要单独的人工确认。 |
| 平台约束 | **当前支撑** | `dev-platform-constraints/` 子模块 | 用户确认保留；不得因 `visual-workbench` 退役而改变或删除。 |

## 注册表与输出根的现实边界

`configs/stage_registry.json` 当前有 **142** 个可执行 stage；其引用的脚本与默认配置均存在。

| 注册表分组 | 数量 | 含义 |
| --- | ---: | --- |
| Stage26 | 38 | synthetic terrain、恢复执行、路径治理及后续诊断。 |
| Stage21--25 | 46 | PPO、theta、坡度/障碍与 Hybrid A* 演进链。 |
| Stage18--20 | 11 | 较早研究与证据链。 |
| path planner v2/v3 gates | 7 | 多平台规划验收链。 |
| 其他 Xunce 与非 Xunce | 40 | 早期比较、治理和基础校验。 |

默认输出根也反映了历史迁移尚未完成，而非自动说明路线无用：

| 输出根类型 | stage 数量 | 建议 |
| --- | ---: | --- |
| `D:/xunce/out/...` 短路径 | 20 | 新路线的首选位置。 |
| `D:/CodexDownloads/lunar-path-planning/...` 长历史路径 | 77 | 作为 legacy input 只读兼容；不要因路径长而移动或删除。 |
| 仓库内 `outputs/...` | 45 | 应迁移“新运行”的默认 root，而不是直接删掉对应 runner。 |

## 已明确终止、失效或阻塞的内容

下列项目有明确文字或机器状态，和“仅仅较旧”的模块不同：

| 项目 | 明确状态 | 应保留什么 | 可清理对象 |
| --- | --- | --- | --- |
| `a_gcs_ws-2.0.1` 执行层参考工程 | 已从父仓库运行依赖中排除 | README 中的一行历史说明 | 若其副本出现在本机其他位置，应由负责人单独确认后处理；当前仓库不含该目录。 |
| Stage 6 R1 `s6-standard-single-r1-20260718T062833Z` | 失败证据，不可恢复、不可描述为成功 | 最小失败证据与 lineage | 该 run 的可再生大产物可在证据导出后单独归档；不得把它当作可续跑 checkpoint。 |
| pre-rock/crater R3 包与 D 盘运行 | 被 2026-07-10 计划明确标为 superseded historical evidence；已中止的 pre-rock/crater R3 包不可复用 | 说明、hash、替代 run 的链接 | 旧包和旧 D 盘运行是强候选，但需先核对最新 rock/crater 包的完整性。 |
| anchor projection 的两项旧输入 | 仅当 `anchor_projection_candidate_generation_summary` 存在时，被标记 `superseded_by` | 当前 candidate summary 和兼容逻辑 | 不要删除 fallback loader；只把旧输入视为可选历史证据。 |
| Path v2 Gate5 / Gate6 | 分别为 `blocked_profile_freeze`、`blocked_formal_inputs_missing` | runner、config、阻塞原因和最小输入说明 | 不是“废弃代码”；缺少 profile 或正式输入前不能运行，也不应悄然删除。 |

## 中间产物盘点

### 仓库内 `outputs/`

`outputs/` 被 `.gitignore` 排除，且没有 Git 历史。当前顶层有 **681** 个目录：

| 类别 | 数量 | 判断 | 清理优先级 |
| --- | ---: | --- | --- |
| `outputs/pytest-*` | 574 | pytest `--basetemp` 产生的测试临时树；测试和计划只把这些名字当作输出位置，不把旧目录作为生产输入 | **高**：确认没有运行中的 pytest 后，可按明确清单人工清理。 |
| `outputs/path_feedback_batch_*` | 100 | 实验结果，可能保存 summary、manifest、routing 或 report | 中：先提取最小证据索引，再归档或删除大文件。 |
| `outputs/xunce_*` | 1 | 早期 Xunce 结果 | 中：与相应注册 stage/报告一起判断。 |
| `outputs/debug-*`、`_xunce_dynamic_validation_work`、`platform_validation` | 6 | 调试或校验残留 | 高到中：先确认没有被现有测试/脚本显式读取。 |

无法在本次只读审计中完成精确总字节数：递归枚举在 `outputs/pytest-stage18-4e` 遇到访问拒绝。不要通过改变 ACL、接管目录或强删来绕过这一点；先确认该目录是否仍被进程占用或受特殊权限保护。

### 其他本地生成物

| 位置 | 判断 | 默认动作 |
| --- | --- | --- |
| `build/` | Python/C++ 构建产物，已忽略 | 可重建；进入清理清单。 |
| `.pytest_cache/` | pytest 缓存 | 可重建；进入清理清单。 |
| `.ua/` | 本地代码知识图谱，约 25 MiB，已由本地 Git exclude 排除 | 默认保留，便于理解项目；需要空间时再单独重建。 |
| `v/cache/` | 来源未确认的本地缓存 | 暂不操作。 |
| `hello.obj`、`schema_codec_conformance_test.obj` | 未跟踪的 COFF/MSVC 对象文件 | 暂不操作；它们可能来自本地 C++ 编译验证。 |
| 未跟踪的 `docs/uncertainty-aware-2p5d-traversability-model.md` 与 `docs/外部输入/` | 用户已有未提交内容 | 严格保护，不纳入清理。 |

## 已确认的路线分界与待办

用户已确认 Stage18--25、Stage26、path-feedback / 早期策略实验，以及 `model-explorer`、`visual-workbench` 为退役路线。Stage26 的表述统一为：**user-confirmed retired, source cleanup pending the manifest and manual confirmation**。这项决定不授权自动清理，也不改变 Stage6、G1/G2/G3、v3、平台约束、默认 grid A* 或 Hybrid A* 的安全边界。

README 的中英文入口与阶段索引已经同步为当前保留主线；后续人工清理前仍须把它们作为文档交叉依赖逐项复核。所有 Stage18--26 源码、配置、测试、plans 和注册表条目都必须在退役清单、引用检查和人工确认完成后，按明确路径处理。

## 建议的精简顺序

1. **先删可再生测试临时树**：由负责人确认后，按 `outputs/pytest-*` 的明确目录清单人工处理；不碰 `path_feedback_batch_*`。
2. **为实验结果建最小索引**：对每个保留的 `path_feedback_batch_*` 只保留 `summary.json`、`manifest.json`、`routing.json`、`report.md`、配置 hash 和必要 checkpoint 指针；大 trace/checkpoint 移到明确的 D 盘归档根。
3. **给 stage 注册表增加生命周期标签**：建议后续增加 `active`、`historical_repro`、`blocked`、`retired`，并让 `run_stage --list` 能按标签筛选。这样“可运行”与“当前推荐运行”不再混淆。
4. **完成退役清单、保留路线引用核验与额外人工确认后再做源码收缩**：先生成跨 scripts/configs/tests/imports 的依赖清单，再把确认退役的路线移到单独 archive/tag；不要只删除 runner 或 config 的一半。
5. **保持文档入口同步**：README 的中英文当前路线和 `xunce-stage-documentation-index.md` 已完成本轮同步；后续每次物理清理或注册表变更都须在同一批次更新这些入口，并以此文件作为项目级导航补充。

## 本次审计不做的事

- 不删除、移动、覆盖任何已有文件或外部 D 盘 artifact。
- 不把 Stage26、path-feedback 或 submodule 仅因较旧而称为废弃。
- 不改变默认 A*、Hybrid A*、PPO、安全阈值、executor 或 checkpoint 发布边界。
