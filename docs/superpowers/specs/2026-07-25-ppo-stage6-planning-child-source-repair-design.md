# Stage 6 Planning Child 同 Run Source-Repair Amendment 设计

日期：2026-07-25  
状态：用户已批准方案 A  
适用范围：Stage 6A planning-safety child，单 seed `20260716`

## 1. 决策摘要

对已经停止的 planning-safety child：

```text
run_id = s6-standard-single-r1-20260724T000124Z
accepted child updates = U75..U84
failed transaction = U85 attempt1 pre only
last complete checkpoint = U84
checkpoint SHA-256 =
  302d7d3a6af1b61b2ffd19f9984d2078293833b88b45237dcc6976a018656062
```

采用“同一 child run 的独立、不可变 source-repair amendment”：

- 不创建 grandchild run。
- 不重算 U75..U84。
- 不改写现有 `lineage_audit.json`。
- 不改写、覆盖或重新打包 U84 checkpoint。
- 不复用 parent run 的 ordinal1..6 source-repair 链。
- 新 amendment 同时证明旧 child 起源身份与当前修复后身份。
- 当前身份按实际运行时内容寻址，不把 Git HEAD 单独当成源码身份。
- 方案 A 必须额外绑定正式 Python 实际加载的 legacy `path_planner` 运行时源码闭包；未被 Stage6 使用的 `path_planner.v2` 不纳入也不得被导入。
- amendment 发布后，唯一允许的下一事务是 U85 attempt2。
- 下一 resource segment 由已有 artifacts 动态推导；本次冻结快照推导结果为 `segment_index=2`。
- 只有 U85 修复 fresh 规格/质量双审均为 C0/I0、新授权通过、amendment formal preview/publish 和恢复 dry-run 全部通过后，才能启动一个 runner。

## 2. 设计目标

### 2.1 功能目标

1. 允许同一 planning child 在经过审查的源码修复后，从最后完整 U84 checkpoint 精确恢复。
2. 保留 U75..U84 的合法训练历史、validation 和 lineage。
3. 将 U85 及之后的 checkpoint、manifest、acceptance 和最终 Gate 证据绑定到当前源码身份。
4. 对旧身份、当前身份、warm-start 起源、U84 checkpoint、accepted prefix、失败 attempt 和下一事务提供可重放的 fail-closed 证明。

### 2.2 非目标

- 不改变 PPO、GAE、reward、网络、动作空间或 optimizer 配置。
- 不改变已审查的 planning-safe 规则、0.75m buffer 或 reset 双扫描。
- 不再引入 frontier 算法语义；U85 LOS-heading 修复由独立 TDD 与双审闭合。
- 不发布 checkpoint，不替换 default policy，不连接 executor，不启动 canary。
- 不运行额外 seed。
- 不把 synthetic terrain 描述为真实物理障碍。

## 3. 冻结起点

amendment 创建时必须重新读取并冻结以下事实，不能仅信任命令行或本设计文档：

| 事实 | 期望值 |
|---|---|
| child stage root | `D:/xunce/out/ppo_frontier/s6-standard-single-r1-20260724T000124Z/s6` |
| child run id | `s6-standard-single-r1-20260724T000124Z` |
| seed | `20260716` |
| effective config SHA | `d7f1b82985e919d3d350fa569bc624bd6967f696875fff890ee4bf676e728bf9` |
| origin `lineage_audit.json` SHA | `5172495848d6544e21ab546b7311cf833baf7fa3793343463000b5f8e15a844d` |
| parent warm-start update | U74 |
| accepted child prefix | U75..U84，连续 10 条 |
| latest complete child update | U84 |
| U84 checkpoint SHA | `302d7d3a6af1b61b2ffd19f9984d2078293833b88b45237dcc6976a018656062` |
| U84 manifest SHA | `2ccb5a64deb3f5983f449150985b7e23b5b44e70d661c539e4b324b04843b2e0` |
| U84 complete SHA | `9a09d920035fdcbcefa62a496852cf5eaceaf1ae6994e3ee14ba30a03ff8bb84` |
| U84 policy-state SHA | `3d82be1a6fbff6b41825c74849dca8b5901ccb895044ff83f563b46315243b66` |
| U84 transaction key | `0009:20260716:update:084` |
| failed transaction key | `0010:20260716:update:085` |
| failed attempt | `1`，仅有 `pre` |
| next attempt | `2` |
| completed resource segment | `segment_index=1` |
| next resource segment | 动态推导为 `segment_index=2` |

任何缺行、重复 accepted、U85 post/accepted、更新号间断、checkpoint 不完整、SHA 漂移或已有 formal runner/replay 都必须拒绝创建或加载 amendment。

## 4. 新 artifact 合同

### 4.1 文件与 schema

```text
canonical name = planning-child-source-repair.json
schema_version = stage6_planning_child_source_repair/v1
mode = same_run_exact_resume_after_reviewed_source_repair/v1
```

artifact 位于既有 child stage root。它不是 parent ordinal source-repair 文件，也不使用 `source-repair-*.json` 命名。

### 4.2 顶层字段

```json
{
  "schema_version": "stage6_planning_child_source_repair/v1",
  "mode": "same_run_exact_resume_after_reviewed_source_repair/v1",
  "formal_run_id": "s6-standard-single-r1-20260724T000124Z",
  "seed": 20260716,
  "origin": {},
  "current": {},
  "accepted_prefix": {},
  "resume_point": {},
  "immutable_inputs": {},
  "append_only_prefixes": {},
  "review_evidence": {},
  "created_at_utc": "...",
  "canonical_sha256": "..."
}
```

`canonical_sha256` 使用删除该字段后的 canonical JSON bytes 计算；正式文件仍以换行结尾。

### 4.3 `origin`

`origin` 必须绑定：

- 原 `lineage_audit.json` 的 path、size、SHA 和 schema。
- 原 U75 launch authorization 的 path、size、SHA。
- 原 execution identity 的 canonical SHA。
- 原 immutable source/config bindings。
- U74 planning warm-start artifact 的 path、size、SHA 和关键 payload。
- parent U74 checkpoint/manifest/complete/policy-state/config/lineage SHA。

原 lineage 继续代表“这个 child 如何从 parent U74 诞生”，永远不改写。

### 4.4 `current`

`current` 必须绑定：

- fresh 修复后源码树 identity。
- 正式 Python 实际解析到的 legacy `path_planner` runtime source-set identity。
- 新 launch authorization 的 path、size、SHA。
- 新 execution identity canonical SHA。
- 当前 immutable source/config bindings。
- U85 修复实现报告、fresh spec review 和 fresh quality review 的 path、size、SHA。
- formal test/replay 证据的 path、size、SHA。

新授权必须在 U85 修复双审 C0/I0 后生成；旧授权不能为当前源码背书。

### 4.5 `accepted_prefix`

必须记录并验证：

- `first_update=75`
- `last_update=84`
- `receipt_count=10`
- 10 个连续 transaction key 与 accepted receipt。
- U80 validation 的既有记录仅作为 child 历史，不重新运行。
- 最新完整 checkpoint 固定为 U84。

accepted prefix 的证明来自 checkpoint index、job state、training metrics、resource audit 和 checkpoint bundle 的交叉验证，不能只看一个文件。

### 4.6 `resume_point`

```json
{
  "last_complete_update": 84,
  "next_update": 85,
  "abandoned_attempt": 1,
  "next_attempt": 2,
  "abandoned_attempt_has_pre_only": true,
  "last_resource_segment_index": 1,
  "next_resource_segment_index": 2,
  "next_transaction_key": "0010:20260716:update:085"
}
```

`segment_id` 不在 preview 时猜测。实际 launch 创建 segment2 后，由 launch refresh/PID record 绑定新 `segment_id`；第一条正式 resource row 必须为 U85 attempt2 pre。

### 4.7 immutable 与 append-only 输入

以下输入必须做完整 bytes/size/SHA 验证：

- `config.json`
- `lineage_audit.json`
- planning warm-start artifact
- U84 `checkpoint.pt`
- U84 `manifest.json`
- U84 `complete.json`
- 新 review authorization
- 双审和 formal replay/test 证据
- legacy `path_planner` runtime source-set 的每个成员文件

以下 journal 在恢复后会追加，因此 amendment 绑定创建时的 prefix size/SHA：

- `checkpoints/index.jsonl`
- `job-state.jsonl`
- `resource_audit.jsonl`
- `training_metrics.jsonl`
- `validation_metrics.jsonl`（若存在）
- `checkpoint_audit.jsonl`（若存在）
- `math_audit.jsonl`（若存在）

恢复前要求整文件与冻结 prefix 完全相等；恢复后只允许尾部追加，冻结 prefix 的任意 byte 改写、截断或重排都 fail closed。

## 5. 运行时身份与 lineage

### 5.1 `lineage_audit.json`

现有 v3 lineage 文件保持字节级不变：

```text
SHA-256 =
5172495848d6544e21ab546b7311cf833baf7fa3793343463000b5f8e15a844d
```

恢复路径不得调用“用当前身份重写 lineage”的逻辑。它必须：

1. 验证现有 lineage 与 amendment 的 `origin` 完全一致。
2. 验证当前授权/身份与 amendment 的 `current` 完全一致。
3. 将 amendment 作为 origin 与 current 之间的独立桥接证据。

### 5.2 checkpoint lineage

- 加载 U84 时，按历史 planning warm-start lineage 验证；不能要求 U84 含有尚未存在的 amendment。
- 从 U85 开始写出的 checkpoint lineage，必须保留原 planning warm-start lineage，并新增：

```json
{
  "planning_child_source_repair": {
    "schema_version": "stage6_planning_child_source_repair/v1",
    "artifact_sha256": "...",
    "origin_lineage_sha256": "517249...",
    "last_origin_update": 84,
    "first_repaired_update": 85,
    "current_execution_identity_sha256": "..."
  }
}
```

- U85..U100 的每个 checkpoint 都必须能反向验证到 parent U74、child U75..U84 和当前 amendment。

### 5.3 实际 planner 运行时依赖身份

正式 Python 的 editable 解析是运行时事实，不能用主仓库中的 submodule
指针代替。创建、加载、授权和启动时必须从与 runner 相同的 Python
解释器解析：

```text
lunar_exploration_ppo -> 当前 Stage6 worktree
path_planner -> 解释器实际 editable package root
```

amendment 的 `current` 新增：

```json
{
  "planner_runtime_identity": {
    "schema_version": "stage6_path_planner_legacy_runtime_source_set/v1",
    "distribution_name": "path-planner",
    "distribution_version": "0.1.0",
    "package_root": ".../src/path_planner",
    "source_set_sha256": "...",
    "paths": [
      {
        "path": "search/astar.py",
        "size_bytes": 8579,
        "sha256": "..."
      }
    ],
    "resolved_symbols": {
      "AStarPlanner": "search/astar.py",
      "Cell": "core/models.py",
      "CostGrid": "core/models.py",
      "GridSpec": "core/models.py",
      "NeighborPolicy": "core/models.py",
      "PlanRequest": "core/models.py"
    },
    "forbidden_loaded_prefix": "path_planner.v2"
  }
}
```

运行时 source-set 是确定排序后的 legacy Python 源码成员集合：

- `path_planner/__init__.py`
- `path_planner/platform.py`
- `path_planner/core/**/*.py`
- `path_planner/postprocess/**/*.py`
- `path_planner/regions/**/*.py`
- `path_planner/search/**/*.py`
- `path_planner/trajectory/**/*.py`

只收集 `.py`，排除 `__pycache__`、测试、构建产物和整个
`path_planner/v2`。排除 v2 不是忽略依赖：导入
`PathPlannerAdapter` 后若发现任何 `path_planner.v2` 模块已加载，必须
fail closed。`AStarPlanner` 及 adapter 直接导入的模型符号必须解析到
上述 source-set 内，不能来自另一个 editable root。

身份以成员相对路径、size 和 SHA-256 的 canonical 聚合为准。Git HEAD、
submodule gitlink 和 editable repository commit 只作诊断字段，不能替代
bytes identity，也不能因为不相关 v2 文件或主仓库其他路径变化而改变
Stage6 身份。

amendment context 和 Stage6 input pins 必须持有这些外部文件的快照：

- preview、publish、workflow entry、launch 前逐文件重验；
- 每个 update 边界通过既有 input-pin 生命周期重验；
- 成员新增、删除、bytes 变化、editable root 漂移或符号解析漂移均拒绝；
- U85 恢复后若该闭包变化，必须停止并创建下一份 fresh 双审 amendment，
  不能覆盖本 amendment。

## 6. 工作流与训练集成

### 6.1 独立模块

新增：

```text
src/lunar_exploration_ppo/workflows/stage6_planning_child_source_repair.py
```

核心接口：

```python
class Stage6PlanningChildSourceRepairError(RuntimeError): ...

@dataclass(frozen=True, slots=True)
class PlanningChildSourceRepairContext:
    ...
    def require_current(self, label: str) -> None: ...
    def verify_append_only_prefixes(self, label: str) -> None: ...
    def checkpoint_lineage_for_update(self, update: int) -> dict[str, object]: ...

def build_planning_child_source_repair_artifact(...) -> dict[str, object]: ...
def validate_planning_child_source_repair_artifact(...) -> PlanningChildSourceRepairContext: ...
def load_planning_child_source_repair_artifact(...) -> tuple[PlanningChildSourceRepairContext, str]: ...
def publish_planning_child_source_repair_artifact(...) -> Path: ...
```

该模块不得导入训练 backend，避免循环依赖。

### 6.2 CLI

新增：

```text
scripts/create_ppo_stage6_planning_child_source_repair.py
```

默认只 preview；只有显式 `--publish` 才可通过 `ArtifactStore` 原子独占写入。重复发布、已存在不同 bytes、路径逃逸、reparse point、活动 formal runner/replay 或输入漂移均拒绝。

runner 新增显式参数：

```text
--planning-child-source-repair <absolute path>
```

提供 planning warm-start 时，该参数可以省略于首次 U75 启动，但对于已存在 U75..U84 且当前身份与 origin 身份不同的恢复必须提供。它不能与 parent classic source-repair ordinal 链混用。

### 6.3 workflow

`stage6.py` 需要：

- 加载并 pin amendment。
- 将 amendment 中的 planner runtime source-set 每个文件作为一等 input pin。
- 在 preview、protected workflow、machine acceptance 和 recovery authorization verification 中传递 context。
- manifest/evidence graph 绑定新 artifact。
- 对 planning warm-start + planning-child amendment 使用独立分支。
- 继续拒绝 planning child 携带 parent classic source-repair artifacts。

### 6.4 training backend

`standard_training.py` 需要：

- 接受独立 `_planning_child_source_repair` context。
- 允许它与 `_planning_warm_start` 同时存在。
- 继续禁止它与 classic `_source_repair` 同时存在。
- 加载 U84 时使用 historical lineage。
- 写 U85+ checkpoint 时加入 current amendment binding。
- `_persist_execution_identity()` 验证旧 lineage，不改写旧 lineage。
- acceptance artifact 同时报告 warm-start 与 child amendment，而不是把 amendment伪装为 parent source-repair。

## 7. 授权与发布顺序

严格顺序：

1. U85 修复主验证通过。
2. fresh spec review PASS C0/I0。
3. 不同 fresh quality review PASS C0/I0。
4. 生成绑定当前 tree/source-set 的新 launch authorization 和 execution identity。
   该绑定同时包含 Stage6 repo source-set 和 planner runtime source-set；
   主仓库 HEAD 仅为诊断。
5. preview amendment；不得写 child root。
6. fresh 审核 preview payload 与 frozen inputs。
7. 单次 `--publish` 写入 `planning-child-source-repair.json`。
8. 重新读取正式 bytes，核对 path/size/SHA/canonical SHA。
9. formal recovery dry-run 验证 U84、U85 attempt2、segment2 和全部 lineage。
10. 确认 formal runner 数量为 0 后，只启动一个 runner。
11. 启动后只读确认 segment2 与 U85 attempt2 pre；不允许 U85 attempt1 post 或 U86 跳步。

## 8. Fail-closed 条件

以下任一条件必须拒绝 preview、publish 或 resume：

- run id、seed、config、warm-start、origin lineage 或 U84 checkpoint 不匹配。
- U75..U84 不是连续 10 条 accepted receipt。
- U85 attempt1 存在 post/accepted。
- next attempt 不是 2，或下一事务不是 U85。
- next resource segment 不是从 journal 动态推导。
- frozen append-only prefix 被改写或截断。
- 新授权不绑定当前源码 identity。
- planner editable root、legacy runtime source-set 或 resolved symbol 漂移。
- `path_planner.v2` 在 Stage6 adapter 导入闭包中出现。
- spec/quality 任一有 Critical/Important。
- amendment 已存在且 bytes 不同。
- 路径逃逸、symlink/reparse point 或非 canonical root。
- formal runner/replay 已活动或检测到重复 runner。
- 尝试重写旧 lineage 或 U84 checkpoint。

## 9. 验收标准

实现必须以测试证明：

1. 正常 fixture 能构建、preview、publish、加载并重验 amendment。
2. origin/current 身份分离，且旧 lineage bytes 不变。
3. current 身份同时绑定 Stage6 repo source-set 与实际 planner runtime source-set。
4. U84 使用 historical lineage；U85 使用 amendment lineage。
5. U75..U84 accepted prefix 任一行漂移都会失败。
6. U85 attempt1 只有 pre；出现 post/accepted 都失败。
7. segment index 动态推导为 2；硬编码或错序失败。
8. append-only journal 允许合法尾部追加，但拒绝 prefix 改写/截断。
9. planner legacy runtime 成员、bytes、root 或 resolved symbol 任一漂移都会失败；不相关 v2 dirty 不进入身份。
10. planning warm-start + child amendment 合法；classic source-repair + child amendment 非法。
11. preview 不写文件；publish 原子独占；重复不同内容拒绝。
12. machine acceptance/manifest/checkpoint/final controller 能重放 parent74 + child26 的完整证据。
13. formal dry-run 后没有写入训练 transaction，没有启动 runner。

## 10. 长期边界

- 本 amendment 只授权从 U84 到 U85 attempt2 的当前 reviewed repair，不是通用“任意源码可续训”开关。
- 后续再发生源码修复时，必须新增另一份独立、不可变且 fresh 双审的 amendment；不得覆盖本文件。
- Stage6 Gate 前不 stage、不 commit、不进入 Stage7。
- 最终报告必须明确：U75..U84 来自 child origin identity，U85..U100 来自本 amendment 绑定的 current identity；两段共享同一 run id，但源码 lineage 可审计分段。
