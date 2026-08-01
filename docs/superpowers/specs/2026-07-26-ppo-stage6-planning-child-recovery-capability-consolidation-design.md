# Stage 6 Planning-Child Recovery Capability Consolidation 设计

日期：2026-07-26  
状态：方案 B 与书面 spec 均已获用户批准，进入 implementation plan  
适用范围：Stage 6 planning-safety child，单 seed `20260716`

## 1. 决策摘要

停止继续按单个 review finding 为现有恢复链叠加门禁。对
planning-child source-repair 恢复路径做一次有边界的收敛：

1. 保留真正保护授权、历史数据和单 runner 的门禁。
2. 把 parent amendment、continuation、正式 journal、checkpoint 和
   当前授权的重型校验集中到唯一启动入口。
3. 启动入口生成不可变 `PlanningChildRecoveryCapability`。
4. workflow、training backend、machine verifier 和 terminal recovery
   只消费该 capability，不再独立重读、重哈希、重新推导或硬编码
   U84/U85/U86、attempt、transaction key 和 resource segment。
5. 运行期间复用已有 run lease、execution capability 和 input pin；
   每步资源轮询不得重新验证整个 source-repair 图。
6. 保持同一正式 run，不重算 U75..U85，不改写任何历史 artifact。

这不是放宽 fail-closed，而是将同一事实改为只有一个所有者。

## 2. 正式基线

实现和验证必须从正式 artifact 动态复核下列基线，不得只信本文：

```text
formal run id =
  s6-standard-single-r1-20260724T000124Z
stage root =
  D:/xunce/out/ppo_frontier/
  s6-standard-single-r1-20260724T000124Z/s6
accepted updates = U75..U85
latest complete update = U85 attempt2
U85 checkpoint SHA-256 =
  7dddb04da7911b7f6705d9dcd60edbeac02614e2a13911bc5ccf52be7880d3b1
parent planning-child amendment SHA-256 =
  64896928a294d77229d0ea6d665e98827f51f8f2b2a6f08b1a8f155ccbcbf9ac
resource segments present = 1..4
resource segment5 = absent
U86 resource/job/training rows = absent
formal continuation artifact = absent
formal runner/replay = 0
```

当前工作区中的 continuation R5/R1 是暂停的未发布原型，不构成正式
合同，也不得据此推断 D 盘状态。它可以在严格 TDD 中被修改或替换，
但不得通过 reset、clean 或覆盖方式破坏其他已有改动。

## 3. 范围与非目标

### 3.1 本次范围

- planning-child parent amendment 与 successor continuation 的加载和验证。
- 当前授权、execution identity、immutable source/config binding。
- append-only journal 的启动快照和 accepted tail 生产语义验证。
- latest complete checkpoint 的 production snapshot 验证。
- restart cursor、checkpoint lineage epoch 和 retention pin 的唯一推导。
- workflow、training backend、machine verifier、terminal recovery 的消费接口。
- continuation preview、write-once publish 和 formal zero-write dry-run。

### 3.2 非目标

- 不改 PPO、GAE、reward、网络、optimizer、action tensor 或候选特征。
- 不改 0.75m/360° planning-safety 规则和 U85 LOS-heading 修复。
- 不改训练 update 数量、validation 计划或 seed。
- 不重写 Stage 6 通用 authorization、run lease、input pin 框架。
- 不把本次工作扩展为整个 Stage 6 或整个仓库的通用重构。
- 不发布 checkpoint，不替换 default policy，不连接 executor，
  不启动 canary，不进入 Stage 7。
- synthetic 仍只能标记为 proxy。

## 4. 门禁分类

### 4.1 必须保留

| 门禁 | 唯一所有者 |
|---|---|
| 单 formal runner / run lease | Stage 6 execution framework |
| 当前 C0/I0 authorization 与 source identity | Stage 5/6 authorization |
| parent 与 continuation 的 canonical、write-once、路径安全 | 各 artifact codec |
| U75..U85 历史 bytes 和 frozen prefix 不可改 | recovery capability issuer |
| production training/validation 语义 | 现有共享 production validator |
| latest complete checkpoint 完整性 | 现有 production checkpoint inspector |
| 正式 journal append-only 和事务顺序 | recovery issuer + 正常 transaction writer |
| checkpoint lineage epoch | recovery capability 中的 lineage policy |
| 被 immutable artifact 物理引用的 checkpoint retention | recovery capability |
| 运行时 source/artifact currentness | 既有 execution capability/input pin |

### 4.2 必须合并

- parent `require_current()` 与 continuation `require_current()`。
- parent `append_only_prefixes` 与 restart `validated_restart_prefixes`。
- workflow、backend、machine、terminal 各自的 resume boundary 推导。
- 多处 parent/continuation/current authorization 的嵌套字段重复校验。
- machine context 和 terminal capability 对同一 accepted tail 的重复重放。
- 多处对 U84/U85/U86、attempt 和 segment 的硬编码比较。

### 4.3 必须移除或停止调用

- consumer 层直接调用重型 `verify_append_only_prefixes()`。
- consumer 层直接重新打开五个 journal 或 checkpoint bundle。
- backend 每次 update/finalize 重建整个 source-repair 图。
- per-step/resource poll 重哈希 amendment、continuation、journal 或 checkpoint。
- terminal recovery 自建第二套 planning-child capability。
- 仅由文件名、warm lineage 或固定整数判断 planning-child profile。

旧检查只有在下列三项同时成立后才可移除：新唯一所有者覆盖原检查的全部
接受与拒绝样例、对应 RED/GREEN 已持久化、主 agent 独立回归通过。

## 5. 目标架构

### 5.1 静态 artifact codec

现有两个文件保留各自单一职责：

```text
stage6_planning_child_source_repair.py
  只负责 parent amendment 的 schema、canonical bytes、路径安全、
  origin/current 绑定和 parent lineage 描述。

stage6_planning_child_source_repair_continuation.py
  只负责 successor continuation 的 schema、canonical bytes、
  parent artifact binding、当前授权 binding、review evidence 和
  write-once preview/publish。
```

它们不得读取动态 training/resource/job tail 来决定当前 restart cursor。

正式 continuation 尚未发布，因此实现应使用新的明确 schema：

```text
canonical filename =
  planning-child-source-repair-continuation.json
schema_version =
  stage6_planning_child_source_repair_continuation/v2
mode =
  write_once_reviewed_source_epoch_successor/v1
```

未发布的原型 v1 不提供兼容入口，加载时必须 fail closed。

continuation v2 只保存不可变桥接事实：

- parent amendment 完整 file binding 与 canonical SHA。
- 新 authorization、execution identity 和 immutable source-set binding。
- fresh spec/quality review evidence 的完整 file binding。
- 发布时 accepted anchor 的 production-validated 摘要。
- 发布时五个 journal 的 canonical prefix seal。
- U86+ lineage epoch 描述。
- canonical SHA 和创建时间。

它不得复制 parent artifact 已经承诺的完整 origin/current 子图，也不得把
发布时的 `next_attempt` 或 `next_resource_segment` 当作永久运行时常量。

### 5.2 唯一 recovery issuer

新增聚焦模块：

```text
src/lunar_exploration_ppo/workflows/
  stage6_planning_child_recovery.py
```

公开接口：

```python
class Stage6PlanningChildRecoveryError(RuntimeError):
    ...

@dataclass(frozen=True, slots=True)
class PlanningChildResumeCursor:
    last_accepted_update: int
    next_update: int
    next_attempt: int
    next_transaction_key: str
    next_resource_segment_index: int
    latest_complete_checkpoint_update: int
    pending_pre_attempt: int | None

@dataclass(frozen=True, slots=True)
class PlanningChildLineageEpoch:
    first_update: int
    last_update: int | None
    artifact_sha256: str
    execution_identity_sha256: str

@dataclass(frozen=True, slots=True)
class PlanningChildRecoveryAnchor:
    formal_run_id: str
    seed: int
    stage_root: Path
    parent_artifact_sha256: str
    input_snapshot_sha256: str
    accepted_anchor: Mapping[str, object]
    journal_prefixes: Mapping[str, Mapping[str, object]]

@dataclass(frozen=True, slots=True)
class PlanningChildRecoveryCapability:
    formal_run_id: str
    seed: int
    stage_root: Path
    parent_artifact_sha256: str
    continuation_artifact_sha256: str | None
    input_snapshot_sha256: str
    capability_sha256: str
    resume_cursor: PlanningChildResumeCursor
    lineage_epochs: tuple[PlanningChildLineageEpoch, ...]
    protected_checkpoint_updates: tuple[int, ...]
    input_pin_requests: tuple[tuple[str, Path], ...]
    acceptance_binding: Mapping[str, object]

    def checkpoint_lineage_for_update(
        self,
        update: int,
    ) -> Mapping[str, object]:
        ...

def inspect_planning_child_recovery_anchor(
    *,
    stage_root: Path,
    parent_artifact_path: Path,
) -> PlanningChildRecoveryAnchor:
    ...

def issue_planning_child_recovery_capability(
    *,
    stage_root: Path,
    parent_artifact_path: Path,
    continuation_artifact_path: Path | None,
    current_execution_identity: Mapping[str, object],
    current_verified_review_authorization: Mapping[str, object],
    current_immutable_bindings: Mapping[str, Mapping[str, object]],
) -> PlanningChildRecoveryCapability:
    ...
```

`PlanningChildRecoveryCapability` 的方法必须是纯内存操作，不得访问文件系统。
所有 nested mapping 在构造时转换为不可变副本，所有 sequence 转为 tuple；
调用方不得持有可变 payload 的别名。

`PlanningChildRecoveryAnchor` 只证明 continuation preview 所需的正式 accepted
anchor 和 journal prefix，不携带启动权限，也不得传给 backend。anchor
inspector 与 capability issuer 必须复用同一个内部 bound-bytes production
validator，不能演化成两套语义。

### 5.3 单次、绑定 bytes 的启动验证

issuer 每个 workflow/recovery/verifier 进程只执行一次：

1. 安全读取 parent、`continuation_artifact_path` 非空时的 continuation、
   authorization、config、
   lineage、五个 journal 和 latest checkpoint bundle。
2. 每个文件形成 path、size、SHA 的 immutable payload binding。
3. 所有 JSONL 和 checkpoint 验证只消费已读 bytes；验证过程中不得重新开路径。
4. 调用唯一共享的 production training/validation validator。
5. 调用 production complete-checkpoint inspector。
6. 验证当前 journal 是 artifact prefix 的合法扩展。
7. 从同一组 bound rows 推导 accepted tail、pending pre 和 resume cursor。
8. 从 artifact chain 推导 lineage epochs 和 retention pins。
9. 生成 canonical `input_snapshot_sha256` 与 `capability_sha256`。
10. 在返回前重新确认全部 immutable input pin 仍与启动快照一致。

预期的正式首次结果是：

```text
last accepted = U85
next update = U86
next attempt = 1
next transaction key = 0011:20260716:update:086
next resource segment = 5
```

这些值必须来自正式 rows，不得由 consumer 写死。

### 5.4 restart cursor 语义

每次新进程启动都重新发行 capability，因此合法 partial tail 必须动态派生：

| 当前 durable tail | 新进程 cursor |
|---|---|
| 无 segment5 | U86 attempt1，新 segment5 |
| 仅 segment5 start | U86 attempt1，新 segment6 |
| segment5 + U86 attempt1 pre | U86 attempt2，新 segment6 |
| U86 accepted | U87 attempt1，下一新 segment |

同一进程启动后，cursor 只约束第一个 remaining transaction 和该进程新建的
resource segment。第一个事务进入正常 durable transaction 生命周期后，
不得继续用冻结 cursor 阻止 U87+。

### 5.5 lineage 与 retention

lineage epoch 从 artifact chain 关系推导，而不是由消费者写死：

```text
U75..U84 -> planning warm-start/origin epoch
U85      -> parent planning-child amendment epoch
U86+     -> continuation current-source epoch
```

`checkpoint_lineage_for_update()` 是唯一 checkpoint lineage 入口。
workflow、backend、machine verifier 和 terminal verifier 不得各自拼装。

retention pin 是所有 immutable chain 物理 checkpoint 引用的并集。当前正式
chain 的结果必须精确为 `(84, 85)`；只要 parent/continuation 的启动验证仍
要求读取对应 bundle，retention 就不得删除它们。未来 artifact 若新增物理
checkpoint 引用，才允许由同一并集规则扩展该 tuple。

### 5.6 运行时轻量守卫

启动后：

- 每次 mutation boundary 使用既有 execution capability、run lease 和
  input pin currentness。
- 在写 update post/accepted 前执行 `rehash=True` 的 immutable input 检查。
- per-step/resource poll 只执行 run lease 和已发行 capability token 的
  轻量检查，不读取 parent、continuation、journal 或 checkpoint。
- 正常 update 的 receipt、resource 和 checkpoint 完整性继续由已有
  durable transaction 逻辑负责。
- 若 immutable source/artifact 在 rollout 中发生变化，下一 mutation
  boundary 必须在写入正式结果前 fail closed。

不新增第二个通用 execution capability 框架。

## 6. 消费者职责

### 6.1 Stage 6 workflow

- 启动时调用 issuer 一次。
- 将同一个 capability 实例传给 backend、machine binding 和 terminal
  preparation。
- 仅检查 profile 组合合法性和 runner lease。
- manifest/evidence graph 记录 capability digest 与 artifact chain binding。
- 不重新推导 accepted tail、resume cursor 或 lineage。

### 6.2 StandardProductionBackend

- 构造时接收 `PlanningChildRecoveryCapability`。
- `prepare_seed()` 仅把第一个 remaining transaction 和新 resource segment
  与 `resume_cursor` 比较一次。
- `run_update()`、`finalize()` 不调用 parent/continuation 重型验证。
- checkpoint 写入只调用 capability 的 lineage 纯函数。
- retention 使用 capability 的 `protected_checkpoint_updates`。

### 6.3 Machine verifier

独立 machine verification 进程可从正式目录发行一次新的 capability，
然后用其 acceptance binding 验证完整 parent74 + child26 图。它不得再维护
`_validate_planning_child_machine_context()` 的第二套 source-repair 语义。

### 6.4 Terminal recovery

移除 planning-child 专用的第二套 terminal capability 推导。terminal 流程：

1. 独立发行一次 `PlanningChildRecoveryCapability`。
2. 运行正常 terminal semantic validator。
3. 将 semantic result 与 capability digest 绑定。
4. receipt commit 前后只验证 terminal evidence handle、run lease 和 input pin。

terminal 不得重新读取五个 journal 来重建同一个 source-repair context。

## 7. 错误处理与安全

- 所有正式路径必须是 canonical absolute plain path，拒绝 symlink、
  reparse point、hardlink/path escape 和非预期 root。
- 只捕获并归一化明确的 domain exception；禁止新增宽泛
  `except Exception`。
- checkpoint 只通过现有 restricted decoder 和 production inspector。
- snapshot 读取、验证、二次 currentness 检查期间若任何 bytes 漂移，
  issuer 必须失败且零写入。
- continuation preview 必须只消费 `PlanningChildRecoveryAnchor` 且零写入；
  正式 continuation 被加载并绑定当前授权后才能发行 launch capability。
- continuation publish 必须经 `ArtifactStore` 原子独占写入，已存在文件无论
  bytes 是否相同都不得在正式流程中重复发布。
- 正式 runner/replay 非零时拒绝 preview、publish、dry-run 和 launch。
- capability 发行失败不得创建 resource segment 或 update pre。

## 8. TDD 与验收

### 8.1 capability 单元测试

必须先取得当前代码 RED，再实现 GREEN：

- 正式形状 parent + continuation + U85 accepted 正例。
- segment5 absent/start/pre 三种 cursor 派生。
- U86 accepted 后动态派生 U87。
- journal prefix 改写、截断、重排、重复 accepted、事务间断均失败。
- authorization、identity、parent/continuation、checkpoint 任一 drift 失败。
- 验证只使用 bound bytes；snapshot 后路径变化不能改变派生结果。
- issuer 返回后 capability 方法零文件读取。
- U84/U85 retention pin 与三个 lineage epoch 正确。

### 8.2 artifact 测试

- continuation v2 preview 零写入。
- v2 publish 原子独占且只能一次。
- v1、错误 parent SHA、旧 authorization、错误 review binding 均失败。
- continuation payload 不重复 parent 完整子图。
- publication anchor 来自 production-validated
  `PlanningChildRecoveryAnchor`，该对象不可启动训练。

### 8.3 consumer 集成测试

- workflow 每进程只发行一次 capability。
- backend 首事务正确接受 U86 attempt1/segment5。
- partial segment/pre restart 正确接受 attempt/segment 的动态结果。
- 第一个事务后 U87+ 不受冻结 U86 cursor 阻塞。
- per-step poll 对 parent/continuation/journal/checkpoint 的 secure read
  次数为零。
- mutation boundary 的 pinned source 漂移仍在正式写入前失败。
- machine verifier 和 terminal verifier 使用同一 acceptance binding，
  不再自行推导 U84/U85/U86。
- classic source-repair 与 planning-child profile 继续互斥。

### 8.4 回归与正式零写入验证

- focused planning-child/recovery/workflow/training/terminal 测试全绿。
- 相关完整回归全绿，不保留 expected failure。
- `py_compile` 覆盖全部修改文件。
- 测试 basetemp 位于 D 盘并禁用新增 pytest cache。
- formal preview/dry-run 前后对 stage root 和 authorization evidence 做
  完整文件集合、size、SHA 快照比较，结果必须完全相同。
- formal runner、replay、pytest 在发布前均为 0。

## 9. 实施与审查顺序

1. 主 agent 写 implementation plan，不实施正式操作。
2. 恢复同一 Plato，按一个紧耦合任务执行严格 RED/GREEN；不得并行派第二个
   implementer。
3. 主 agent 独立审 diff、RED/GREEN、focused/full 和 formal zero-write。
4. 按用户既定要求并行恢复同一 Jason 做规格复审、同一 Hegel 做质量/安全
   复审。
5. 任一 Critical/Important 退回同一 Plato 修复并做 scoped re-review。
6. 双审 C0/I0 后才生成绑定当前树的新 identity/authorization。
7. formal preview 后单次发布 continuation v2。
8. 重新读取正式 bytes 并执行 zero-write recovery dry-run。
9. 确认唯一 runner 条件后，从动态 cursor 恢复 U86 并继续到 U100。

本次设计批准不构成 identity、authorization、publish、launch、Stage 6 Gate
或 Stage 7 授权。

## 10. 完成定义

只有同时满足以下条件，方案 B 实现才完成：

- 重型 planning-child 恢复事实只有 issuer 一个所有者。
- 四个 consumer 不再重读或独立推导同一 source-repair 状态。
- 所有必要安全门禁仍有明确所有者和覆盖测试。
- 当前两个 partial-U86 RED 转为 GREEN，且没有新增 C/I。
- focused/full regression、formal zero-write 和并行双审均通过。
- 正式 continuation 仍未发布、runner 仍未启动，直到另行满足发布顺序。
