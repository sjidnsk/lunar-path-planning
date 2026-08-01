# 迅策中期缩减规模双门槛实验操作手册

## 1. 用途与证据边界

本手册用于执行 `midterm_reduced_w8x3_update80/v1` 实验链。它规定准备、
正式运行、恢复、复算和结果交付顺序，不宣称任何尚未完成的 G1、G2、G3
或总体门槛结果。

设计与实施依据：

- [实验设计规格](superpowers/specs/2026-07-26-midterm-dual-gate-experiment-design.md)
- [实施计划](superpowers/plans/2026-07-26-midterm-dual-gate-experiment.md)
- [G1 场景源物化计划](superpowers/plans/2026-07-27-midterm-g1-scenario-source-materialization.md)

所有正式结论必须带“缩减规模”或 `reduced` 限定。
`final_threshold_reduced_gate_passed` 只表示本次缩减规模实验达到 99%/1 s
数值门槛，不等于项目全规模完成验收。

## 2. 冻结合同

| 项目 | 正式合同 | 判定方式 |
| --- | --- | --- |
| checkpoint | Stage 6 `update 80` | 路径、checkpoint SHA-256、policy-state SHA-256 均须一致 |
| G1 | Test-Q24 与 Unseen-24 各 24 个 episode | 8 workers，每条 lane 固定 3 个场景，不运行 64 后截断 |
| G2 | `3 × 43 × 5 = 645` 次正式调用 | wheel、legged、hopper；每平台 43 个请求、每请求 5 次 |
| G3 | `10 + 3 + 3` | 10 个 wheel 闭环；3 个 legged、3 个 hopper 接口回放 |
| 覆盖率中期门槛 | `coverage >= 0.80` | inclusive，即等于 0.80 通过 |
| 覆盖率完成时门槛 | `coverage >= 0.99` | inclusive，即等于 0.99 通过 |
| 规划中期门槛 | `time <= 2000 ms` | inclusive，即等于 2000 ms 通过 |
| 规划完成时门槛 | `time <= 1000 ms` | inclusive，即等于 1000 ms 通过 |

G1 的每个正式 split 还要求平均覆盖率达到相应门槛，且至少 23/24 个
episode 达到相应门槛；Test 与 Unseen 分开统计并分别通过。安全违规和
掩膜外动作必须为 0。

G2 的正式统计只使用 645 条 `formal_sample=true` 原始记录。每个平台的
Standard 规模为 165 次、Kilometer 规模为 50 次；warm-up、cold-start 和
worker-1 诊断不进入正式分布。平台×规模、平台×规模×结果类型以及
平台×规模×请求类别的所有正式分组都必须分别过线。中期判定要求各组的
mean、P95 和 max 均不超过 2000 ms；完成时判定要求各组的 mean、P95
不超过 1000 ms、至少 95% 样本不超过 1000 ms，且没有样本超过
2000 ms。43 个请求中每平台 38 个可达请求的 5 次重复必须全部成功、
L2 复核有效且语义一致；不可达请求必须保持结构化未完成语义。

G3 不替代 G1 或 G2。总体双门槛只有在 G1、G2 和 G3 对应门槛全部满足时
才能通过。

## 3. 两个独立时钟

1. **就绪/实施时钟**：包括 runner 实现与验证、独立输入生产、Hopper
   正式 simulation-proxy 参数审批、覆盖缓存生成、场景冻结、缺陷修复和
   readiness probe。该时钟不计入“一天正式实验窗口”。
2. **正式执行时钟**：只有全部 readiness 条件满足后才开始。无失败重跑
   时预计 14～18 小时，连续排期上限为 20 小时。该窗口不包括外部输入
   生产、代码修复、Test-C24 确认分支或学术图形制作。

若 20 小时内尚未形成完整证据，状态保持 `incomplete` 或 `blocked`，不得
用部分样本形成通过结论。

## 4. 正式开始前的硬门

以下各项必须同时成立：

- 实现测试和兼容性测试已通过。
- update80 checkpoint 文件存在，且 checkpoint SHA-256 为
  `35e04c86f9f973af028fb08f0175d42ab45378d09e1f2b96ee6aad6e4c12b5b5`。
- policy-state SHA-256 为
  `3123e6adde99be41e3bd5cc2f3843068e5416892395926d759c2c6a243ffd381`。
- 覆盖缓存完整，场景源和场景清单已经按内容寻址冻结，所有清单与掩膜
  哈希可复核。
- G2 每平台至少有 3334 个独立 primitive labels、1 个独立 small-map
  optimum，以及固定的 33 个 Standard 和 10 个 Kilometer 正式请求。
- oracle、provider、来源证明和实现哈希相互独立且已审批。
- Hopper 正式 simulation-proxy 参数记录存在、有效，并与 G2 execution
  bundle 和 artifact-bound O2 approval 精确绑定。
- G1 Validation3 dry-run 已通过。
- G2 worker-1/4 语义和只读静态缓存诊断已通过。
- 机器环境、线程、电源模式和后台负载预检通过。

独立输入缺失、Hopper 参数记录缺失或无效、审批绑定漂移，均必须
fail closed 为 `blocked`，正式样本数为 0；不得调用 provider 后再补审批。
Validation、Gate5B test fixture、自标注输入或缩小请求集不能替代上述条件。

## 5. 当前已冻结输入示例

以下路径仅是 2026-07-27 已物化输入的只读示例，不是运行结果，也不保证
未来仍未漂移。每次使用前必须重新核对文件和 SHA-256；不得覆盖这些目录。

| 输入 | 示例路径 | 示例 SHA-256 |
| --- | --- | --- |
| 完整 coverage manifest | `D:/xunce/out/mid_dual/g1-cache-full-20260727a/coverage-cache-manifest.json` | `f3308c2b349895343d7a21cf46de32341f8d45a88c454830604cbfd1d78be31c` |
| policy-blind 场景源 manifest | `D:/xunce/inputs/mid_dual/scenario-sources/9be80fecf8a3cb87/manifest.json` | `eeeb7d4e0540ce0fc61b9b440cc37187b29c3bccbabdf082963b78349119a67c` |
| 冻结场景 manifest | `D:/xunce/inputs/mid_dual/scenarios/37a3e639b4decfaf/manifest.json` | `37a3e639b4decfaf660ff66408c5c7109808c5f44aa22c18e98cd901050bb54a` |
| G2 execution bundle manifest | `D:/xunce/inputs/mid_dual/g2/g2t2-candidate-9b9a19e2ca96992b14e4040f-exec-v1/manifest.json` | `3ee4b73069a75454e5ee49199895f0f93e3d79a58814cb5f13fa99ef9918c5af` |

静态核对示例：

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath `
  'D:/xunce/out/mid_dual/g1-cache-full-20260727a/coverage-cache-manifest.json', `
  'D:/xunce/inputs/mid_dual/scenario-sources/9be80fecf8a3cb87/manifest.json', `
  'D:/xunce/inputs/mid_dual/scenarios/37a3e639b4decfaf/manifest.json', `
  'D:/xunce/inputs/mid_dual/g2/g2t2-candidate-9b9a19e2ca96992b14e4040f-exec-v1/manifest.json'
```

## 6. 场景源物化与场景冻结

本次正式链若复用第 5 节示例，只进行哈希复核，不重新生成。需要独立复现
时，必须选择新的、未存在的 D 盘输出基目录。

### 6.1 物化 policy-blind 场景源

```powershell
$Py = 'D:/conda_envs/lunar-explorer/python.exe'
$CoverageManifest = 'D:/xunce/out/mid_dual/g1-cache-full-20260727a/coverage-cache-manifest.json'
$CoverageManifestSha256 = 'f3308c2b349895343d7a21cf46de32341f8d45a88c454830604cbfd1d78be31c'
$NewSourceOutputBase = '<NEW_ABSOLUTE_D_SOURCE_OUTPUT_BASE>'

& $Py scripts/prepare_xunce_mid_dual_scenario_sources.py `
  --config configs/xunce_mid_dual_scenario_sources_v1.json `
  --coverage-manifest $CoverageManifest `
  --coverage-manifest-sha256 $CoverageManifestSha256 `
  --output-root $NewSourceOutputBase `
  --execute
```

`<...>` 是必须替换的占位符，禁止按字面执行。完成后，工具在新基目录下
发布内容寻址子目录；将其绝对路径记为 `$ScenarioSourceRoot`，并验证其中
`manifest.json` 后再继续。

### 6.2 冻结 Test-Q24、Test-C24、Unseen-24 与 G3 cohort

```powershell
$ScenarioSourceRoot = '<ABSOLUTE_VERIFIED_SCENARIO_SOURCE_ROOT>'
$NewScenarioOutputBase = '<NEW_ABSOLUTE_D_SCENARIO_OUTPUT_BASE>'

& $Py scripts/freeze_xunce_mid_dual_scenarios.py `
  --config configs/xunce_mid_dual_scenario_freeze_v1.json `
  --descriptor-catalog "$ScenarioSourceRoot/descriptors.jsonl" `
  --source-manifest "$ScenarioSourceRoot/source-manifest.json" `
  --coverage-manifest $CoverageManifest `
  --coverage-manifest-sha256 $CoverageManifestSha256 `
  --reconstruction-index "$ScenarioSourceRoot/reconstruction-index.jsonl" `
  --output-root $NewScenarioOutputBase `
  --execute
```

冻结选择只允许读取静态环境描述，不得读取 checkpoint、策略动作、覆盖率
结果、规划成功率、奖励或运行时间。正式结果产生后，清单不得重选。

## 7. 唯一正式命令顺序

命令顺序固定为：

```text
scenario freeze
→ G1 preflight
→ G1 Validation3 dry-run
→ G2 preflight
→ G2 diagnostic
→ formal source freeze
→ G1 formal
→ G2 formal（独占）
→ G3 preflight/formal
→ aggregate
→ 从已验证 aggregate 生成结果表和学术图形
```

### 7.1 设置路径和新运行标识

```powershell
$Py = 'D:/conda_envs/lunar-explorer/python.exe'
$ScenarioBundleRoot = 'D:/xunce/inputs/mid_dual/scenarios/37a3e639b4decfaf'
$ScenarioManifest = "$ScenarioBundleRoot/manifest.json"
$G2InputBundle = 'D:/xunce/inputs/mid_dual/g2/g2t2-candidate-9b9a19e2ca96992b14e4040f-exec-v1'

$G1RunId = '<NEW_G1_RUN_ID>'
$G2RunId = '<NEW_G2_RUN_ID>'
$G3RunId = '<NEW_G3_RUN_ID>'
$AggregateRunId = '<NEW_AGGREGATE_RUN_ID>'
```

四个占位符都必须替换为新的稳定 ID。允许字符为字母、数字、下划线和
连字符；G1/G2/G3 还允许点号。不得指向已有的 finalized root。首次开始
前执行：

```powershell
$NewRoots = @(
  "D:/xunce/out/mid_dual/g1/$G1RunId",
  "D:/xunce/out/mid_dual/g2/$G2RunId",
  "D:/xunce/out/mid_dual/g3/$G3RunId",
  "D:/xunce/out/mid_dual/aggregate/$AggregateRunId"
)
if ($NewRoots | Where-Object { Test-Path -LiteralPath $_ }) {
  throw '至少一个新运行目录已经存在；请更换对应 run ID，不得覆盖。'
}
```

### 7.2 G1 preflight 与 Validation3 dry-run

同一 `$G1RunId` 从 preflight 形成可恢复 phase 前缀，再由 dry-run 和 formal
继续；不是覆盖已有 artifact。

```powershell
& $Py scripts/run_xunce_mid_dual_g1_coverage.py `
  --config configs/xunce_mid_dual_g1_coverage_v1.json `
  --scenario-manifest $ScenarioManifest `
  --run-id $G1RunId `
  --mode preflight

& $Py scripts/run_xunce_mid_dual_g1_coverage.py `
  --config configs/xunce_mid_dual_g1_coverage_v1.json `
  --scenario-manifest $ScenarioManifest `
  --run-id $G1RunId `
  --mode dry-run
```

预期只接受 `p01`、`p02`，不产生 Test-Q24 或 Unseen-24 正式行。若
checkpoint、manifest、缓存或 Validation3 任一不一致，停止并按 `blocked`
处理。

### 7.3 G2 preflight 与 diagnostic

同一 `$G2RunId` 依次完成输入验证、cold/warm-up 和 worker-1 语义对照；
这些行都不是 645 个正式样本。

```powershell
& $Py scripts/run_xunce_mid_dual_g2_planning_time.py `
  --config configs/xunce_mid_dual_g2_planning_time_v1.json `
  --input-bundle $G2InputBundle `
  --run-id $G2RunId `
  --mode preflight

& $Py scripts/run_xunce_mid_dual_g2_planning_time.py `
  --config configs/xunce_mid_dual_g2_planning_time_v1.json `
  --input-bundle $G2InputBundle `
  --run-id $G2RunId `
  --mode diagnostic
```

若独立输入、artifact-bound O2 approval 或 Hopper 参数绑定缺失/无效，runner
必须写出 0 正式行的 terminal `blocked` root。修复后必须换新
`$G2RunId`，不能复用已 finalized 的 blocked root。

### 7.4 Formal source freeze

G1 dry-run 和 G2 diagnostic 均通过后：

1. 停止修改正式链依赖的源码、配置、冻结输入和审批文件。
2. 记录主仓库 HEAD、Path Planner 子模块 HEAD 和精确 dirty inventory。
3. 重新核对 checkpoint、policy-state、场景 manifest、G2 bundle manifest
   以及审批绑定。
4. 确认 G1/G2 的 effective config 与既有 phase 前缀一致。
5. 任何字节发生变化都停止正式运行；不得让 formal 阶段悄悄接受新输入。

只读记录命令：

```powershell
git rev-parse HEAD
git status --short
git -C path-planner rev-parse HEAD
git -C path-planner status --short
Get-FileHash -Algorithm SHA256 -LiteralPath $ScenarioManifest, "$G2InputBundle/manifest.json"
```

runner 会把所需源码、配置、环境和 manifest 哈希写入 lineage/evidence
artifact。工作区不要求无 dirty 项，但正式依赖的字节必须从此保持冻结，
且所需 dirty 源文件必须被逐字节快照绑定。

### 7.5 G1 formal

```powershell
& $Py scripts/run_xunce_mid_dual_g1_coverage.py `
  --config configs/xunce_mid_dual_g1_coverage_v1.json `
  --scenario-manifest $ScenarioManifest `
  --run-id $G1RunId `
  --mode formal
```

原正式阶段设计为 `p03 Test-Q24 → p04 Unseen-24 → p05 replay3 →
p06 recompute → p07 finalize`。2026-07-27 用户明确取消 replay3；本次正式收尾
不得启动 p05 场景重跑，只允许基于原始 Test-Q24、只读 Unseen 父结果和三条显式
修复结果执行独立离线复算与图表生成。Test-Q24 未达到中期门槛时，runner 必须在
Unseen-24 前停止并形成诚实的 `failed` 结果，不得自动运行 Test-C24。

### 7.6 G2 formal：必须独占运行

只有 G1 正式进程已完全退出，且没有训练、pytest、G3、其他 formal runner
或其他 CPU/GPU/磁盘高负载任务时，才允许启动：

```powershell
& $Py scripts/run_xunce_mid_dual_g2_planning_time.py `
  --config configs/xunce_mid_dual_g2_planning_time_v1.json `
  --input-bundle $G2InputBundle `
  --run-id $G2RunId `
  --mode formal
```

G2 formal 使用 4 workers，计时函数为 `time.perf_counter_ns()`。每条 raw
row 保存 input validation、platform instantiation、search、complete route
validation、result assembly 五段整数纳秒、精确总纳秒及唯一换算的毫秒值。
进程启动、输入文件读取、外部队列等待、绘图和日志落盘不进入单次规划
计时。禁止与任何训练、G1 或重负载任务并行。

### 7.7 G3 preflight 与 formal

先把已完成且 manifest 验证通过的 G1、G2 root 填入：

```powershell
$G1Root = '<COMPLETED_G1_ROOT>'
$G2Root = '<COMPLETED_G2_ROOT>'

& $Py scripts/run_xunce_mid_dual_g3_closed_loop.py `
  --config configs/xunce_mid_dual_g3_closed_loop_v1.json `
  --frozen-bundle-root $ScenarioBundleRoot `
  --g1-root $G1Root `
  --g2-root $G2Root `
  --run-id $G3RunId `
  --mode preflight

& $Py scripts/run_xunce_mid_dual_g3_closed_loop.py `
  --config configs/xunce_mid_dual_g3_closed_loop_v1.json `
  --frozen-bundle-root $ScenarioBundleRoot `
  --g1-root $G1Root `
  --g2-root $G2Root `
  --run-id $G3RunId `
  --mode formal
```

合格的 G3 preflight 不创建正式 root；formal 要求该 root 尚不存在。若
preflight 因输入问题写出 terminal blocked root，修复后必须换新
`$G3RunId`。G3 的 10 个 wheel 闭环和 3+3 接口回放必须分开记录；后两者
不得进入 wheel 覆盖率统计。

### 7.8 Aggregate 独立复算

```powershell
$G3Root = '<COMPLETED_G3_ROOT>'

& $Py scripts/run_xunce_mid_dual_aggregate.py `
  --config configs/xunce_mid_dual_aggregate_v1.json `
  --g1-root $G1Root `
  --g2-root $G2Root `
  --g3-root $G3Root `
  --run-id $AggregateRunId `
  --mode create
```

aggregate 只以三门的 raw rows 和 manifest 为统计真值，独立重算覆盖率、
均值、P50、P95、P99、样本数、超时数、场景通过比例和分组完整性；
`summary.json` 与 `report.md` 只用于一致性反查。任何逐行数据、manifest、
门内 summary 或 report 不一致时，总体状态必须为 `blocked`。

## 8. Test-C24 独立确认分支

仅当 Test-Q24 形成有效 `failed` 结果，并且修复只使用 Train 和 Validation
完成后，才允许启用 Test-C24：

1. 不再调试或重跑 Test-Q24。
2. 保持 update80、场景规模、覆盖分母和阈值不变。
3. 形成绑定失败 parent run 的独立 repair-lineage JSON；当前代码或配置
   lineage 必须与失败版本不同。
4. 使用新的 run ID 和预先冻结、未在 Test-Q24 使用的 Test-C24。

```powershell
$TestCRunId = '<NEW_TEST_C_RUN_ID>'
$RepairLineage = '<ABSOLUTE_REPAIR_LINEAGE_JSON>'

& $Py scripts/run_xunce_mid_dual_g1_coverage.py `
  --config configs/xunce_mid_dual_g1_coverage_v1.json `
  --scenario-manifest $ScenarioManifest `
  --run-id $TestCRunId `
  --mode test-c-confirmation `
  --repair-lineage $RepairLineage
```

repair-lineage 必须包含精确字段 `schema_version`、`repair_id`、
`parent_run_id`、`code_sha256` 和 `config_sha256`，并通过 runner 校验。
Test-C24 是失败后的独立确认结果，不计入首轮 48 个 G1 episode，也不得
冒充原 Test-Q24 通过。

## 9. 中断与恢复

通用规则：

- 每个 phase 完成后才接受其整批 rows，并追加 `phase-state.jsonl`。
- 只接受连续、完整、哈希有效且 effective config 完全一致的 phase 前缀。
- 不完整 phase 必须整阶段重新运行；不得拼接局部统计或手工补行。
- 已 finalized 的 `passed`、`failed` 或 `blocked` root 均只读保留，不
  覆盖、不移动、不删除。
- manifest/输入/代码哈希漂移、重复 ID、样本缺失、平台错配、split 泄漏、
  NaN 或 inf 均应阻塞。

恢复命令：

- G1：对未 finalized 的同一 root，使用原 `$G1RunId`、原 manifest 和
  `--mode formal` 重新执行。
- G2：对未 finalized 的同一 root，保持独占环境，使用原 `$G2RunId`、
  原 input bundle 和 `--mode formal` 重新执行；`job-state.jsonl` 只恢复
  哈希一致的正式 schedule。
- G3：仅对未 finalized 的原 root 使用显式 `resume`：

```powershell
& $Py scripts/run_xunce_mid_dual_g3_closed_loop.py `
  --config configs/xunce_mid_dual_g3_closed_loop_v1.json `
  --frozen-bundle-root $ScenarioBundleRoot `
  --g1-root $G1Root `
  --g2-root $G2Root `
  --run-id $G3RunId `
  --mode resume
```

- Aggregate：仅对未 finalized 的原 root 使用显式 `resume`：

```powershell
& $Py scripts/run_xunce_mid_dual_aggregate.py `
  --config configs/xunce_mid_dual_aggregate_v1.json `
  --g1-root $G1Root `
  --g2-root $G2Root `
  --g3-root $G3Root `
  --run-id $AggregateRunId `
  --mode resume
```

若恢复校验不通过，保留原目录作为失败证据，调查原因并为新 attempt 使用
新 run ID；不得清理旧 root 后伪装成首次运行。

## 10. `blocked`、`failed` 与 `incomplete`

| 状态 | 含义 | 允许的后续动作 |
| --- | --- | --- |
| `incomplete` / `interrupted` | phase 尚未形成完整、可接受证据 | 输入和 lineage 未漂移时按第 9 节恢复 |
| `blocked` | 证据资格、输入、审批、结构或完整性不成立 | 先解决 blocker；若 root 已 finalized，使用新 run ID |
| `failed` | 证据完整且正式数值未达到门槛 | 保留有效失败结果；不得删慢样本或缩规模，按冻结诊断/修复流程进入新版本 |
| `passed` | 对应缩减规模门槛和证据合同均满足 | 仍必须保留 `reduced` 限定并通过 aggregate |

`blocked` 不是“指标失败”，报告应写“证据未就绪”；`failed` 也不能通过
删除异常值、合并平台、混合 Test/Unseen 或更换分母改写为通过。

## 11. 原始证据、复算与学术图形

每门至少保留：

- `config.json`
- `results.jsonl`
- `summary.json`
- `routing.json`
- `manifest.json`
- `phase-state.jsonl`
- `report.md`
- 相应 input、lineage、environment、coverage、timing、cache 或 replay audit

学术图形只能在 aggregate manifest 验证通过后从已验证原始数据生成，不得
手工录入或平滑掉慢样本。正式交付至少应清楚展示：

- G1 Test-Q24 与 Unseen-24 分开的场景覆盖率分布、均值/置信区间以及
  0.80、0.99 门槛线；
- G2 按平台和 Standard/Kilometer 分面的正式时间分布、P95 以及
  1000 ms、2000 ms 门槛线；
- G3 wheel 闭环覆盖率、与 G1 配对差值，以及 legged/hopper 接口回放
  完整性；
- 每幅图的实际样本数、单位、统计方法、`reduced` 限定和数据 manifest
  SHA-256。

图形应采用色盲友好配色、可辨识线型、可打印字号，优先输出矢量
PDF/SVG，并附 600 dpi 的 TIFF/PNG。绘图后端和样式需在生成前另行冻结；
绘图不进入 G2 计时，也不能改变任何门槛判定。

## 12. 禁止性声明

本实验链只产生计算评价证据：

- 不得宣称完成实物平台或硬件实验。
- legged 和 hopper 只能标记为 simulation proxy；不得宣称动态步态、
  真实机器人稳定性、实物弹跳可行性或物理障碍证据。
- 不发布 checkpoint，不替换 default policy，不连接 executor，不启动
  canary。
- 默认 A* 不被替换；Path Planner v2 只用于 G2 和显式接口 replay。
- 不得把 synthetic terrain proxy 写成物理地形真值。
- 不得用本缩减规模通过结论替代全规模项目验收。

## 13. 最终交付检查

- [ ] G1 每个正式 split 恰为 24；不是 21，也不是从 64 个结果中截取。
- [ ] G2 恰为 645 个正式调用；warm-up、cold-start、worker-1 不混入。
- [ ] G3 恰为 10 wheel + 3 legged + 3 hopper，且两类统计分离。
- [ ] 0.80/0.99、2000/1000 ms 均按 inclusive boundary 判定。
- [ ] Test 与 Unseen、平台与规模均分开判定。
- [ ] 独立输入、Hopper 参数和 O2 approval 都已验证且 hash-bound。
- [ ] G2 正式计时期间无训练、G1 或其他重负载并发。
- [ ] 只从 raw rows + manifest 独立复算，门内 summary/report 仅用于反查。
- [ ] 原始 rows、所有 audit、manifest、report 和学术图形可追溯。
- [ ] `blocked`、`failed`、`incomplete` 未混淆。
- [ ] 所有通过字段和图注保留 `reduced` 限定。
- [ ] 无硬件/实物、checkpoint 发布、默认策略替换、executor 或 canary
  越界声称。
