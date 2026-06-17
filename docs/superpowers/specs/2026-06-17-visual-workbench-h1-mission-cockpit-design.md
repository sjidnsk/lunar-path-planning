# Visual Workbench H1 Mission Cockpit Design

## Summary

`visual-workbench` 的下一阶段正式方向是 **H1 任务驾驶舱**。

它不是继续扩展一个普通 artifact dashboard，而是把前端做成面向月面巡视任务的展示与交互工作台。首屏必须同时完成两件事：

1. 让观看者一眼理解“当前月面巡视任务走到哪里了”。
2. 让操作者能直接进行地图回放、证据追踪和受控验证。

本设计继承上一版 mission-control redesign 的任务优先导航、深浅分区、渐进证据面板和研发/演示模式；同时吸收草图中的最终选择：

- **H1 为主结构**：任务驾驶舱，展示与交互同屏。
- **吸收 H3**：中央地图回放加入时间轴和图层控制。
- **吸收 B**：证据链作为右侧工具区和底部抽屉，而不是独立顶层页面。

## Product Stance

H1 的产品定位是 **mission-first interactive evidence cockpit**。

核心问题：

> 当前月面巡视任务走到哪里了？为什么可信？下一步能做什么？

因此首页不再以工具名称或原始数据表作为第一入口。用户首先看到任务阶段、当前判断、地图回放和下一步动作；低层 artifact 证据在明确交互后出现。

## Layout

H1 首屏由五个区域组成。

### 1. Left Mission Rail

左侧任务阶段导航保留六个任务阶段：

1. 环境测绘
2. 目标捕获
3. 路线制导
4. 可达确认
5. 风险复核
6. 任务简报

每个阶段显示：

- 阶段序号；
- 阶段名称；
- 一行阶段含义；
- 状态标签：已完成、当前、待验证、阻塞。

阶段导航的职责是任务叙事，不再承担工具导航职责。

### 2. Top Mission Status

顶部区域回答首屏主问题：

- 项目与子系统标识；
- 当前任务标题；
- 当前阶段摘要；
- API 状态；
- 研发模式 / 演示视图切换；
- 任务态势 KPI。

建议 KPI：

- 证据完整度；
- 关键 artifact 数；
- 地图回放速度或当前帧；
- 剩余风险等级。

这些 KPI 只表达 artifact 和前端派生状态，不得包装成训练放行、性能证明或发布证明。

### 3. Main Mission Map Replay

中央深色地图是主交互区域。

必须支持：

- 地形 / 代价 / 通行边界底图；
- raw path、smoothed path、optimized path 图层；
- blocked/passability 图层；
- 目标点或当前选中路径段；
- 图层开关；
- 路径回放时间轴。

地图不只是装饰。点击目标点、路径段或时间轴节点时，应更新右侧阶段判断和底部证据链抽屉。

### 4. Right Judgment And Tools Panel

右侧浅色面板同时承担展示解释和研发入口。

默认展示：

- 当前阶段判断；
- 可信度摘要；
- 风险 / 未完成边界；
- 下一步动作。

固定工具入口：

- Evidence Trace：查看证据链、schema 覆盖、缺失项和原始 artifact。
- Map Replay：聚焦地图图层、路径段、目标点和回放状态。
- Validate：只允许 dry-run / validate，不允许 full run。

同一工具按钮再次点击时应关闭对应面板；工具按钮必须有可访问的 expanded / pressed 状态。

### 5. Bottom Evidence Chain Drawer

底部证据链抽屉是 H1 的关键补强。它用于在不打断主地图体验的前提下展示 artifact-first 可信边界。

抽屉内容包括：

- 当前选中地图对象或路径段；
- 支撑该判断的 artifact；
- schema 流程；
- 缺失 schema 或 pending 证据；
- 下一步安全动作；
- 禁止动作提示，例如 full run / PPO / training。

抽屉默认可以是收起或半展开状态，但应能从 Evidence Trace 或地图选择中打开。

## Interaction Model

### Stage Selection

点击任务阶段后：

- 更新当前阶段；
- 更新地图焦点；
- 更新右侧阶段判断；
- 清空或重置当前打开的工具面板；
- 保留 API 状态和模式切换不变。

### Map Interaction

点击地图目标点、路径段或时间轴节点后：

- 高亮选中对象；
- 更新右侧判断摘要；
- 打开或刷新底部证据链抽屉；
- 展示对应 artifact/schema 证据；
- 不触发任何训练、完整 run 或外部副作用。

### Evidence Trace

Evidence Trace 用于解释“为什么可信”，而不是简单显示 JSON。

它应展示：

- 相关 artifact 列表；
- schema coverage；
- status / reason code；
- 缺失项；
- 原始内容入口；
- 当前结论可支持和不可支持的边界。

### Validate

Validate 是受控研发入口。

允许：

- dry-run；
- validate；
- 展示 stdout/stderr；
- 展示解析后的 JSON；
- 展示失败恢复建议。

禁止：

- full run；
- PPO；
- training；
- staged release；
- 修改 network/action space/default A*；
- 把诊断结果包装为发布或训练放行证明。

## Research Mode And Presentation Mode

H1 保留双模式，但两种模式使用同一份数据。

### Research Mode

研发模式显示：

- artifact id；
- schema version；
- source path；
- raw evidence；
- stderr；
- command result；
- missing evidence；
- reason code。

### Presentation Mode

演示模式隐藏或弱化：

- 长路径；
- stderr；
- raw JSON；
- 低层诊断细节；
- 内部命令文本。

但演示模式不得改写指标含义，不得隐藏关键风险，不得把 pending/blocked 状态讲成 completed。

## Data Model

第一阶段继续使用现有 API，由前端派生 H1 状态：

```text
/api/artifacts
  -> schema detector
  -> mission stage grouping
  -> stage status
  -> map replay model
  -> evidence chain model
  -> cockpit view model
```

现有 raw artifact endpoints 继续支持地图和证据链按需加载。

可选后续 API：

- `GET /api/mission/stages`
- `GET /api/mission/stages/{stage_id}`
- `GET /api/artifacts/{artifact_id}/related`
- `GET /api/replay/current`

这些不是 H1 第一阶段必须项。除非前端派生逻辑明显重复或性能不足，否则不新增后端 API。

## Component Boundaries

建议组件边界：

- `MissionStageRail`
  - 渲染六阶段导航；
  - 只关心 stage state 和 selection。

- `MissionStatusHeader`
  - 渲染首屏问题、当前摘要、API 状态和模式切换。

- `MissionKpiStrip`
  - 渲染证据完整度、artifact 数、回放帧、风险等级。

- `MissionMapReplay`
  - 渲染地图、图层、路径和时间轴；
  - 输出 selected map object。

- `MissionJudgmentPanel`
  - 渲染阶段判断、风险、下一步动作。

- `StageTools`
  - 渲染 Evidence Trace / Map Replay / Validate；
  - 管理 active tool state 的可访问表现。

- `EvidenceChainDrawer`
  - 渲染选中对象、artifact、schema flow、缺失项和禁止动作。

- `ValidatePanel`
  - 封装 dry-run / validate 调用和结果展示。

`App.tsx` 只负责加载数据、管理选择状态和组合布局。

## Visual System

H1 使用深浅分区混合：

- 深色区域：地图、阶段 rail、回放控制；
- 浅色区域：判断、证据、工具、验证结果；
- 蓝色：当前阶段或选中对象；
- 绿色：通过或可追踪；
- 琥珀色：待确认或缺失；
- 红色：阻塞或禁止动作。

要求：

- 使用 lucide 图标，不使用 emoji 作为结构图标；
- 交互控件最小高度不低于 44px；
- active / hover / focus / disabled 状态清晰；
- 地图和状态不能只靠颜色表达；
- 卡片圆角控制在 8px 左右；
- 避免一屏全是同一种蓝色或深色块。

## Responsiveness

桌面优先，但必须适配 375、768、1024、1440 宽度。

建议布局：

- ≥1200px：左 rail + 中央地图 + 右侧判断 + 底部抽屉。
- 768-1199px：rail 可上移或收缩，地图和右侧面板上下堆叠。
- <768px：阶段导航转为横向或紧凑列表，地图优先显示，工具和证据抽屉折叠。

不得出现水平滚动；地图、代码块、路径文本必须有明确的 overflow 策略。

## Accessibility

必须满足：

- 所有按钮有可读 accessible name；
- 展开/折叠控件使用 `aria-expanded`；
- 当前阶段使用 `aria-current` 或等价状态；
- 工具按钮使用 pressed/expanded 状态；
- 键盘可访问所有主要控件；
- focus ring 明显；
- 颜色不是唯一状态表达；
- 减少动效时，回放和状态切换仍可理解。

## Testing

前端测试至少覆盖：

- 六个任务阶段按顺序出现；
- 选择阶段会更新标题、地图焦点和右侧判断；
- 地图回放图层控件可切换并保持布局稳定；
- 点击时间轴或路径段会更新证据链抽屉；
- Evidence Trace 可打开和关闭；
- Validate 文案和行为保持 dry-run / validate only；
- 演示模式隐藏低层诊断，但不隐藏关键风险；
- 375、768、1024、1440 下无水平溢出；
- focus 状态可见。

后端测试继续覆盖：

- artifact allowlist；
- schema detection；
- raw reads；
- command allowlist；
- full run rejection。

## Verification

下一阶段实现完成后，至少运行：

```powershell
cd visual-workbench
python -m pytest
cd web
npm test -- --run
npm run build
cd ..\..
python -m pytest tests\test_visual_workbench_boundary.py tests\test_path_planner_algorithm_report.py
```

还应运行浏览器 smoke check：

- 375px；
- 768px；
- 1024px；
- 1440px；
- 检查无水平滚动；
- 检查地图非空；
- 检查工具按钮可打开/关闭；
- 检查证据链抽屉可更新。

## Non-Goals

H1 阶段不得：

- 启动 PPO；
- 启动 training；
- 启动 staged release；
- 触发 full run；
- 修改 network/action space/default A*；
- 替代 `dev-platform-constraints`、`model-explorer`、`path-planner` 的职责；
- 宣称 Ackermann-feasible trajectory；
- 把 IRIS/GCS/path-planner 诊断当作训练放行、发布证明或性能证明；
- 为视觉效果牺牲 artifact-first traceability。

## Acceptance Criteria

H1 spec 对应的下一阶段实现可验收条件：

- 首屏明确呈现 H1 任务驾驶舱，而不是普通 artifact dashboard；
- 左侧任务阶段、中央地图回放、右侧阶段判断、底部证据链抽屉同时成立；
- 地图回放具备图层控制和时间轴；
- Evidence Trace / Map Replay / Validate 保留为阶段内工具；
- 证据链能解释当前判断的 artifact 支撑和缺口；
- Validate 仍被限制在 dry-run / validate；
- 演示模式和研发模式共享同一数据语义；
- 响应式和可访问性要求通过验证；
- 项目文档同步更新，至少包括 `visual-workbench/README.md` 和相关 `docs/superpowers/specs/` / `docs/superpowers/plans/` 文件。
