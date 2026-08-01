# Stage 6 Planning Child Source Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute this plan task-by-task. Every implementation task uses strict superpowers:test-driven-development. Before any completion claim, use superpowers:verification-before-completion.

**Goal:** 在不改写既有 planning child lineage、U84 checkpoint 或 U75..U84 accepted 历史的前提下，新增不可变 source-repair amendment，使同一 run 能在 fresh 双审和新授权后从 U84 精确恢复 U85 attempt2。

**Architecture:** 新增职责独立的 `stage6_planning_child_source_repair.py`，把 origin identity、current identity、实际 Python 加载的 legacy `path_planner` runtime source-set、accepted prefix、U84 checkpoint、U85 failed attempt 和 append-only journal prefix 绑定到一个原子发布 artifact。workflow 与 training backend 显式传递该 context；U84 使用 historical lineage 验证，U85+ checkpoint 添加 amendment lineage。旧 `lineage_audit.json` 始终只读且字节不变。

**Tech Stack:** Python 3.11、dataclass、canonical JSON、ArtifactStore、secure path/input pinning、pytest、PowerShell、现有 Stage6 workflow/training/checkpoint/recovery contracts。

## Global Constraints

- 用户已批准设计：`docs/superpowers/specs/2026-07-25-ppo-stage6-planning-child-source-repair-design.md`。
- formal run 固定为 `s6-standard-single-r1-20260724T000124Z`，seed 固定为 `20260716`。
- U75..U84 已 accepted；U85 attempt1 只有 pre；最后完整 checkpoint 固定为 U84。
- 不创建 grandchild，不重算 U75..U84，不覆盖 U84，不改写 `lineage_audit.json`。
- 不复用 parent ordinal1..6 source-repair 链。
- 双审 C0/I0、新授权、formal preview/publish 和 dry-run 之前不得启动 runner。
- 不发布 checkpoint、不替换 default policy、不连接 executor、不启动 canary、不运行额外 seed。
- Stage6 Gate 用户批准前不 stage、不 commit；每个任务只记录候选文件和证据。
- 保留用户和其他任务的 dirty changes；禁止 reset、checkout、批量清理或大范围 revert。
- Git HEAD 仅作诊断；正式 current identity 必须按 Stage6 repo source-set 与实际 planner runtime source-set 的 bytes 寻址。
- 正式 Python 当前把 `path_planner` 解析到外部 editable root；不得用主仓库 submodule gitlink 冒充运行时依赖身份。
- `path_planner.v2` 不属于 Stage6 A* 合同，不纳入 source-set；若 adapter 导入后实际加载了它则 fail closed。
- 所有正式 Python 命令使用 `D:\conda_envs\lunar-explorer\python.exe`。
- pytest 使用 `-p no:cacheprovider`，不删除已有 cache。

---

## Task 1：冻结 fixture 与核心 amendment 合同

**Files:**

- Create: `src/lunar_exploration_ppo/workflows/stage6_planning_child_source_repair.py`
- Create: `tests/ppo_highres_frontier/test_stage6_planning_child_source_repair.py`
- Reference: `src/lunar_exploration_ppo/workflows/stage6_planning_warm_start.py`
- Reference: `src/lunar_exploration_ppo/workflows/stage6_source_repair.py`
- Reference: `src/lunar_exploration_ppo/utils/artifact_io.py`
- Reference: `src/lunar_exploration_ppo/utils/stage6_input_pinning.py`

### Step 1：写最小 frozen fixture helper

在新测试文件中构造短路径 `tmp_path / "s6"`，包含：

- canonical `config.json` 与 origin `lineage_audit.json`。
- planning warm-start artifact 与 origin authorization fixture。
- 连续 U75..U84 的 checkpoint index、job state、training metrics、resource accepted rows。
- U84 `checkpoint.pt`、`manifest.json`、`complete.json`。
- U85 attempt1 segment1 pre，且无 post/accepted。
- fresh spec/quality PASS C0/I0 与 current authorization fixture。

fixture helper 必须按生产 schema 写真实字段，不用 monkeypatch 跳过 validator。

### Step 2：写 RED 合同测试

至少新增：

```python
def test_build_child_source_repair_binds_origin_current_and_exact_u85_attempt2(...):
    ...

def test_child_source_repair_rejects_rewriting_origin_lineage(...):
    ...

def test_child_source_repair_rejects_non_contiguous_accepted_prefix(...):
    ...

def test_child_source_repair_rejects_u85_attempt1_post_or_accept(...):
    ...

def test_child_source_repair_derives_next_resource_segment_from_journal(...):
    ...

def test_child_source_repair_rejects_spec_or_quality_important_findings(...):
    ...
```

运行：

```powershell
D:\conda_envs\lunar-explorer\python.exe -m pytest -p no:cacheprovider tests/ppo_highres_frontier/test_stage6_planning_child_source_repair.py -q
```

预期 RED：模块不存在或公开接口不存在；不能以 fixture/schema 错误代替目标失败。

### Step 3：实现 schema、canonical hash 与 context

实现：

```python
PLANNING_CHILD_SOURCE_REPAIR_NAME: Final = (
    "planning-child-source-repair.json"
)
PLANNING_CHILD_SOURCE_REPAIR_SCHEMA: Final = (
    "stage6_planning_child_source_repair/v1"
)
PLANNING_CHILD_SOURCE_REPAIR_MODE: Final = (
    "same_run_exact_resume_after_reviewed_source_repair/v1"
)

class Stage6PlanningChildSourceRepairError(RuntimeError):
    pass

@dataclass(frozen=True, slots=True)
class PlanningChildSourceRepairContext:
    artifact: Mapping[str, object]
    artifact_path: Path | None
    artifact_sha256: str | None
    formal_run_id: str
    seed: int
    origin_lineage_sha256: str
    current_execution_identity_sha256: str
    last_origin_update: int
    first_repaired_update: int
    next_attempt: int
    next_resource_segment_index: int

    def require_current(self, label: str) -> None: ...
    def verify_append_only_prefixes(self, label: str) -> None: ...
    def checkpoint_lineage_for_update(
        self, update: int
    ) -> dict[str, object]: ...
```

实现 canonical JSON、SHA、path/size/SHA binding、review verdict validator。禁止使用 hardcoded“验证成功”旁路。

### Step 4：实现 accepted prefix 与 resume point 交叉验证

实现内部纯函数，分别解析：

- checkpoint index 的 U75..U84 receipt。
- resource 的同 update/attempt pre/post accepted。
- job/training update 序列。
- U84 checkpoint bundle 完整性。
- U85 attempt1 pre-only。
- resource lifecycle 的下一 segment index。

同一个事实至少由两个独立 artifact 对齐；错误消息必须指出 artifact、update、attempt 或 segment。

### Step 5：GREEN 与负例

运行：

```powershell
D:\conda_envs\lunar-explorer\python.exe -m pytest -p no:cacheprovider tests/ppo_highres_frontier/test_stage6_planning_child_source_repair.py -q
```

预期：Task 1 全绿。

### Step 6：记录候选文件

```powershell
git diff --check -- src/lunar_exploration_ppo/workflows/stage6_planning_child_source_repair.py tests/ppo_highres_frontier/test_stage6_planning_child_source_repair.py
git status --short -- src/lunar_exploration_ppo/workflows/stage6_planning_child_source_repair.py tests/ppo_highres_frontier/test_stage6_planning_child_source_repair.py
```

不得 stage/commit。

---

## Task 2：实现 append-only prefix pinning 与原子 preview/publish

**Files:**

- Modify: `src/lunar_exploration_ppo/workflows/stage6_planning_child_source_repair.py`
- Modify: `tests/ppo_highres_frontier/test_stage6_planning_child_source_repair.py`
- Create: `scripts/create_ppo_stage6_planning_child_source_repair.py`
- Create: `tests/ppo_highres_frontier/test_stage6_planning_child_source_repair_cli.py`

### Step 1：写 prefix mutation RED

新增测试：

```python
def test_child_repair_allows_only_tail_append_after_frozen_prefix(...):
    ...

def test_child_repair_rejects_prefix_byte_mutation(...):
    ...

def test_child_repair_rejects_prefix_truncation(...):
    ...

def test_child_repair_preview_writes_nothing(...):
    ...

def test_child_repair_publish_is_atomic_and_exclusive(...):
    ...

def test_child_repair_rejects_path_escape_and_reparse_point(...):
    ...
```

RED 命令：

```powershell
D:\conda_envs\lunar-explorer\python.exe -m pytest -p no:cacheprovider tests/ppo_highres_frontier/test_stage6_planning_child_source_repair.py tests/ppo_highres_frontier/test_stage6_planning_child_source_repair_cli.py -q
```

### Step 2：实现 prefix snapshot

immutable input 使用完整 bytes/size/SHA。

append-only input 记录：

```json
{
  "path": "resource_audit.jsonl",
  "prefix_size_bytes": 123,
  "prefix_sha256": "...",
  "line_count": 22
}
```

加载时在恢复前验证完整文件等于 prefix；运行中验证前 `prefix_size_bytes` 完全相同，并允许其后追加。不得仅比较整文件当前 SHA。

### Step 3：实现 CLI

CLI 接口：

```python
def build_parser() -> argparse.ArgumentParser: ...
def main(argv: Sequence[str] | None = None) -> int: ...
```

参数：

```text
--stage-root
--run-id
--review-authorization
--planning-warm-start
--spec-review
--quality-review
--implementation-report
--exact-replay-evidence
--output
--publish
```

默认输出 canonical preview JSON 到 stdout，不写文件。`--publish` 使用 `ArtifactStore` 对 canonical output 做原子独占写入；已存在相同 bytes 返回幂等成功还是拒绝，必须与现有 ArtifactStore 合同一致；已存在不同 bytes 必须拒绝。

### Step 4：GREEN 与脚本 import 边界

```powershell
D:\conda_envs\lunar-explorer\python.exe -m pytest -p no:cacheprovider tests/ppo_highres_frontier/test_stage6_planning_child_source_repair.py tests/ppo_highres_frontier/test_stage6_planning_child_source_repair_cli.py -q
D:\conda_envs\lunar-explorer\python.exe -m py_compile scripts/create_ppo_stage6_planning_child_source_repair.py src/lunar_exploration_ppo/workflows/stage6_planning_child_source_repair.py
```

### Step 5：记录候选文件

运行 `git diff --check` 和 scoped `git status`；不得 stage/commit。

---

## Task 2A：绑定实际 legacy path-planner 运行时源码闭包

**Files:**

- Modify: `src/lunar_exploration_ppo/workflows/stage6_planning_child_source_repair.py`
- Modify: `src/lunar_exploration_ppo/workflows/stage6.py`
- Modify: `tests/ppo_highres_frontier/test_stage6_planning_child_source_repair.py`
- Modify: `tests/ppo_highres_frontier/test_stage6_workflow.py`

### Step 1：写 runtime identity RED

至少新增：

```python
def test_planner_runtime_identity_binds_actual_editable_legacy_source_set(...):
    ...

def test_planner_runtime_identity_rejects_member_bytes_or_membership_drift(...):
    ...

def test_planner_runtime_identity_rejects_resolved_symbol_outside_package_root(...):
    ...

def test_planner_runtime_identity_rejects_loaded_path_planner_v2(...):
    ...

def test_unrelated_repo_head_or_path_planner_v2_dirty_does_not_change_identity(...):
    ...

def test_stage6_input_pins_include_every_planner_runtime_source_member(...):
    ...
```

fixture 使用临时 fake editable package root，不能修改或 monkeypatch 正式
`D:/codex/worktrees/.../path-planner` 文件。RED 必须来自 runtime identity
字段/validator/input pins 尚未实现。

### Step 2：实现 deterministic legacy source-set

实现内容寻址 identity：

- 从当前解释器实际解析 `path_planner` package root。
- 仅确定排序收集设计中列出的 legacy `.py` 成员；成员新增/删除也改变 identity。
- 每行记录 package-relative path、size、SHA-256。
- canonical 聚合 `source_set_sha256` 不含机器绝对路径。
- 绝对 editable/package root 单独记录并在同机恢复时重验。
- `AStarPlanner`、`Cell`、`CostGrid`、`GridSpec`、`NeighborPolicy`、
  `PlanRequest` 的实际 source file 必须属于该集合。
- adapter 导入后 `sys.modules` 不得含 `path_planner.v2` 前缀。
- Git HEAD、tree、dirty 只作诊断，绝不能替代 bytes identity。

禁止递归包含 `path_planner/v2`，也禁止通过 pin 整个外部 repository 阻塞
不相关任务。

### Step 3：接入 amendment 与 input pins

- `current.planner_runtime_identity` 成为必需字段。
- build/validate/load/`require_current()` 均重验。
- `planning_child_source_repair_input_pin_requests()` 返回每个 runtime source
  成员的绝对路径。
- Stage6 evidence/input-pin 生命周期在 workflow entry、launch 前和 update
  边界持续验证。
- 错误必须区分 root、membership、bytes、resolved-symbol 和 forbidden-v2
  drift。

### Step 4：GREEN

```powershell
D:\conda_envs\lunar-explorer\python.exe -m pytest -p no:cacheprovider tests/ppo_highres_frontier/test_stage6_planning_child_source_repair.py tests/ppo_highres_frontier/test_stage6_workflow.py -q
```

随后用正式 Python 做一次只读 probe，证明：

- `lunar_exploration_ppo` 指向当前 Stage6 worktree。
- `path_planner` 指向实际 editable root。
- source-set 中包含实际 A* 与模型文件。
- 未加载 `path_planner.v2`。
- probe 不写 child root、不启动 runner/replay。

### Step 5：记录证据

运行 scoped `git diff --check` 和文件 SHA；不得 stage/commit。

---

## Task 3：把 amendment 接入 runner 与 protected workflow

**Files:**

- Modify: `scripts/run_ppo_stage6_standard.py`
- Modify: `src/lunar_exploration_ppo/workflows/stage6.py`
- Modify: `src/lunar_exploration_ppo/workflows/stage6_terminal_recovery.py`
- Modify: `tests/ppo_highres_frontier/test_stage6_execution.py`
- Modify: `tests/ppo_highres_frontier/test_stage6_workflow.py`
- Modify: `tests/ppo_highres_frontier/test_stage6_planning_warm_start.py`

### Step 1：写参数传播 RED

新增测试证明：

- runner 接受 `--planning-child-source-repair`。
- path 从 CLI 传到 protected workflow、preview、machine acceptance 和 recovery verifier。
- 已有 U75..U84 且当前身份不同，没有 amendment 时 fail closed。
- planning warm-start + child amendment 合法。
- planning warm-start + parent classic source-repair 仍非法。
- child amendment + classic source-repair 非法。

RED 命令：

```powershell
D:\conda_envs\lunar-explorer\python.exe -m pytest -p no:cacheprovider tests/ppo_highres_frontier/test_stage6_execution.py tests/ppo_highres_frontier/test_stage6_workflow.py tests/ppo_highres_frontier/test_stage6_planning_warm_start.py -q
```

### Step 2：扩展公开 workflow 签名

所有 Stage6 preview/run/protected entrypoint 增加：

```python
planning_child_source_repair_path: str | Path | None = None
```

内部传递：

```python
planning_child_source_repair_context: object | None
planning_child_source_repair_sha256: str | None
```

加载函数必须在进入训练前完成 secure read、schema validation、origin/current binding 和 `require_current("workflow-entry")`。

### Step 3：扩展 `_Stage6EvidenceHandle`

将 amendment path、bytes、SHA 作为一等 evidence handle；close/revalidation 语义与 warm-start artifact 一致。禁止只在 CLI 检查一次后丢弃。

### Step 4：修改 recovery authorization binding

`_verify_stage6_recovery_authorization_bindings()` 的 planning child 分支必须：

1. 从 origin lineage 取旧 authorization/identity。
2. 从 amendment 取 current authorization/identity。
3. 验证两者与 artifact 中固定 SHA。
4. 验证当前授权绑定当前 source identity。
5. 不要求旧 lineage 等于当前 identity。

其他 profile 的行为不变。

### Step 5：GREEN

```powershell
D:\conda_envs\lunar-explorer\python.exe -m pytest -p no:cacheprovider tests/ppo_highres_frontier/test_stage6_execution.py tests/ppo_highres_frontier/test_stage6_workflow.py tests/ppo_highres_frontier/test_stage6_planning_warm_start.py -q
```

不得为通过测试 monkeypatch 掉 production validator。

---

## Task 4：训练 backend、checkpoint lineage 与 acceptance

**Files:**

- Modify: `src/lunar_exploration_ppo/ppo/standard_training.py`
- Modify: `tests/ppo_highres_frontier/test_stage6_training.py`
- Modify: `tests/ppo_highres_frontier/test_stage6_execution.py`
- Modify: `tests/ppo_highres_frontier/test_stage6_planning_warm_start.py`

### Step 1：写 lineage 分段 RED

新增测试：

```python
def test_planning_child_repair_loads_u84_with_historical_lineage(...):
    ...

def test_planning_child_repair_writes_u85_with_amendment_lineage(...):
    ...

def test_planning_child_repair_never_rewrites_origin_lineage_audit(...):
    ...

def test_planning_child_repair_requires_u85_attempt2_segment2(...):
    ...

def test_stage6_acceptance_reports_warm_start_and_child_repair_separately(...):
    ...

def test_planning_child_repair_rejects_classic_source_repair_context(...):
    ...
```

RED：

```powershell
D:\conda_envs\lunar-explorer\python.exe -m pytest -p no:cacheprovider tests/ppo_highres_frontier/test_stage6_training.py tests/ppo_highres_frontier/test_stage6_execution.py tests/ppo_highres_frontier/test_stage6_planning_warm_start.py -q
```

### Step 2：扩展 backend constructor

新增独立参数：

```python
_planning_child_source_repair: object | None = None
_planning_child_source_repair_sha256: str | None = None
```

合同：

```python
if child_repair is not None and planning_warm_start is None:
    raise ...
if child_repair is not None and classic_source_repair is not None:
    raise ...
```

在 update transaction、checkpoint 写入和 terminal acceptance 前重复 `require_current()` 与 prefix 验证。

### Step 3：修改 `_persist_execution_identity`

planning child repair 分支：

```python
existing_lineage_bytes = active_input_pin.read_bytes(
    stage_root / "lineage_audit.json",
    label="planning child repair origin lineage",
)
context.verify_origin_lineage(existing_lineage_bytes)
context.verify_current_identity(current_identity)
return existing_lineage_payload
```

不得调用 write/replace。测试要记录 mtime/bytes/SHA 均不变。

### Step 4：修改 checkpoint lineage

- U84 load：`context.expected_historical_checkpoint_lineage(84)`。
- U85+ write：warm-start lineage 与 `context.checkpoint_lineage_for_update(update)` 合并。
- 所有 update 必须满足 `85 <= update <= 100`。
- U84 payload、manifest、complete 不得写入。

### Step 5：修改 acceptance

`build_stage6_acceptance_artifacts()` 增加：

```python
planning_child_source_repair_binding: Mapping[str, object] | None = None
```

planning profile 允许 warm-start + child repair；输出分别包含：

```json
{
  "planning_warm_start": {...},
  "planning_child_source_repair": {...}
}
```

不得把 child amendment 写到 classic `source_repair_binding`。

### Step 6：GREEN

```powershell
D:\conda_envs\lunar-explorer\python.exe -m pytest -p no:cacheprovider tests/ppo_highres_frontier/test_stage6_training.py tests/ppo_highres_frontier/test_stage6_execution.py tests/ppo_highres_frontier/test_stage6_planning_warm_start.py -q
```

---

## Task 5：manifest、machine acceptance 与 final-controller 证据

**Files:**

- Modify: `src/lunar_exploration_ppo/workflows/stage6.py`
- Modify: `src/lunar_exploration_ppo/workflows/stage6_terminal_recovery.py`
- Modify: `tests/ppo_highres_frontier/test_stage6_workflow.py`
- Modify: `tests/ppo_highres_frontier/test_stage6_training.py`
- Modify only if required by reviewed child final-controller contract:
  - `.superpowers/sdd/stage6-final-controller-runbook.md`
  - `.superpowers/sdd/stage6-final-evidence-matrix.md`
  - `.superpowers/sdd/stage6-final-review-dispatch-templates.md`

### Step 1：写 evidence graph RED

测试 machine acceptance：

- 读取 origin lineage、warm-start 和 child amendment 三层证据。
- parent74 + child26 语义不变。
- child receipts 最终仍为 26，不伪称 amendment 后重新训练 26 条。
- U80 属于 origin child segment；U90/U100 属于 repaired segment。
- amendment 缺失、SHA 漂移或 current identity 漂移时 fail closed。

### Step 2：实现 manifest/evidence binding

将 `planning-child-source-repair.json` 加入 canonical manifest artifact graph、review evidence 和 terminal acceptance。仅在该 profile 存在时要求；普通 Stage6 和首次 planning warm-start 不受影响。

### Step 3：协调 final-controller 文档

先检查 Locke 的 R6 文档修复范围与 SHA。若其合同尚未覆盖同-run child amendment，只添加最小 addendum；不得覆盖其并行改动。任何文档变更仍需 fresh 文档规格/质量双审 C0/I0。

### Step 4：GREEN

```powershell
D:\conda_envs\lunar-explorer\python.exe -m pytest -p no:cacheprovider tests/ppo_highres_frontier/test_stage6_workflow.py tests/ppo_highres_frontier/test_stage6_training.py tests/ppo_highres_frontier/test_stage6_execution.py -q
```

---

## Task 6：主 agent 独立验证

### Step 1：审 diff 与范围

```powershell
git diff --check
git status --short
git diff -- src/lunar_exploration_ppo/workflows/stage6_planning_child_source_repair.py scripts/create_ppo_stage6_planning_child_source_repair.py scripts/run_ppo_stage6_standard.py src/lunar_exploration_ppo/workflows/stage6.py src/lunar_exploration_ppo/workflows/stage6_terminal_recovery.py src/lunar_exploration_ppo/ppo/standard_training.py tests/ppo_highres_frontier/test_stage6_planning_child_source_repair.py tests/ppo_highres_frontier/test_stage6_planning_child_source_repair_cli.py tests/ppo_highres_frontier/test_stage6_execution.py tests/ppo_highres_frontier/test_stage6_workflow.py tests/ppo_highres_frontier/test_stage6_training.py tests/ppo_highres_frontier/test_stage6_planning_warm_start.py
```

确认未修改 frontier/reward/PPO 语义，未覆盖 Hooke 或 Locke 的不相干改动。

### Step 2：focused tests

```powershell
D:\conda_envs\lunar-explorer\python.exe -m pytest -p no:cacheprovider tests/ppo_highres_frontier/test_stage6_planning_child_source_repair.py tests/ppo_highres_frontier/test_stage6_planning_child_source_repair_cli.py tests/ppo_highres_frontier/test_stage6_planning_warm_start.py tests/ppo_highres_frontier/test_stage6_execution.py tests/ppo_highres_frontier/test_stage6_workflow.py tests/ppo_highres_frontier/test_stage6_training.py -q
```

### Step 3：相关回归

```powershell
D:\conda_envs\lunar-explorer\python.exe -m pytest -p no:cacheprovider tests/ppo_highres_frontier/test_stage6_review_authorization.py tests/ppo_highres_frontier/test_stage6_source_repair.py tests/ppo_highres_frontier/test_stage6_preflight.py -q
```

### Step 4：真实 child 只读 preview

使用正式 child root 和 fresh review artifacts 运行 CLI，不带 `--publish`。验收：

- stdout schema 正确。
- origin lineage SHA 为 `517249...`。
- U84 checkpoint SHA 为 `302d7...`。
- accepted prefix 为 10。
- next 为 U85 attempt2。
- next segment index 为 2。
- child root 中没有新文件。
- formal runner/replay 数量仍为 0。

### Step 5：记录验证证据

写入新的短路径：

```text
D:/xunce/review/s6-planning-child-repair-<timestamp>/
```

正式 JSON 通过 ArtifactStore 原子独占写入。不得写训练 root，直到双审完成。

---

## Task 7：fresh 规格与质量双审

### Step 1：fresh spec review

派一个未参与实现的 fresh reviewer，只读检查：

- 设计全部字段和 fail-closed 条件。
- Stage6 repo source-set 与实际 planner runtime source-set 的双内容身份。
- Git HEAD 只作诊断，未把不相关 merge/v2 dirty 误纳入授权。
- old lineage/U84 不写。
- U75..U84 prefix 与 U85 attempt2/segment2。
- historical/current checkpoint lineage 分段。
- preview/publish/runner 顺序。

Critical/Important 必须回到唯一 fixer 严格 TDD 修复，并由新的 fresh spec re-review 复核。

### Step 2：fresh quality review

仅 spec C0/I0 后，派不同 fresh quality reviewer，检查：

- path security、TOCTOU 和 input pinning。
- 外部 editable root、legacy runtime membership、resolved symbol 与 forbidden-v2 校验。
- append-only prefix 生命周期。
- ArtifactStore 原子独占。
- context 生命周期与重复 revalidation。
- failure messages、类型和边界。
- 测试是否真实经过 production 路径。

Critical/Important 必须修复并 fresh 重审。

### Step 3：冻结双审 artifact

将 reviewer 原样 Markdown 持久化到新的 review root，记录 path/size/SHA。双审 C0/I0 前不得生成恢复授权、publish amendment 或启动 runner。

---

## Task 8：新授权、单次 publish 与 formal dry-run

### Step 1：生成新 current authorization

授权必须绑定：

- 当前 HEAD 诊断值与按 bytes 固定的 Stage6 reviewed source-set。
- 实际 planner editable root 的 legacy runtime source-set SHA 和全部成员。
- U85 修复后的 frontier/test SHA。
- child amendment 实现/test SHA。
- fresh spec/quality C0/I0 artifact SHA。
- formal config SHA `d7f1...`。
- warm-start artifact SHA。
- U84 checkpoint SHA。

### Step 2：重新 preview

用正式授权运行 amendment CLI preview；将 stdout canonical bytes 持久化到 review root，比较两次 preview bytes 完全一致。

### Step 3：单次 publish

显式 `--publish` 写：

```text
D:/xunce/out/ppo_frontier/s6-standard-single-r1-20260724T000124Z/s6/planning-child-source-repair.json
```

立即 secure read 并记录正式 path/size/SHA/canonical SHA。禁止覆盖或第二次不同内容 publish。

### Step 4：formal recovery dry-run

dry-run 必须证明：

- origin lineage bytes/SHA 不变。
- U84 checkpoint bundle bytes/SHA 不变。
- accepted U75..U84 连续 10 条。
- U85 attempt1 pre-only。
- first new transaction 为 U85 attempt2。
- next resource segment index 为 2。
- historical/current identity 分段正确。
- dry-run 不写 resource/job/training/checkpoint row。
- formal runner 数量为 0。

若失败，先系统性诊断；不得猜测重启或修改 artifact 绕过。

---

## Task 9：精确恢复 U85 attempt2

### Step 1：启动前只读验收

- 无 `run_ppo_stage6_standard.py` formal runner。
- 无 replay/debug runner。
- D 盘、Windows commit/pagefile、GPU、RSS 正常。
- authorization、identity、warm-start、amendment 和 U84 checkpoint 均 current。
- planner runtime root/source-set/resolved symbols 均 current，且未加载 v2。

### Step 2：只启动一个 runner

使用同一 run id、同一 child root、同一 effective config，显式传：

```text
--planning-warm-start <frozen warm artifact>
--planning-child-source-repair <child root/planning-child-source-repair.json>
--review-authorization <fresh authorization>
```

不得创建新 run root。

### Step 3：启动后一次只读快照

确认：

- 1 root + 8 workers。
- 新 resource `segment_index=2`。
- 第一条新 resource row 为 U85 attempt2 pre。
- 没有 U85 attempt1 post/accepted。
- 没有 U86 pre。
- stderr 为空或无异常。

成功后更新既有 `ppo-stage-6` automation 的 PID、authorization、identity、amendment SHA 和监控合同；不得创建重复 automation。

---

## Task 10：U85..U100 与 Stage6 收尾

1. U85 accepted 后核对 checkpoint lineage 含 planning warm-start + child amendment。
2. U90/U100 validation 使用修复后 identity。
3. U100 正常终止后按已 fresh 双审 C0/I0 的 child final-controller 文档执行。
4. machine verifier 验证：
   - parent U1..U74 只作 warm-start 来源；
   - child receipts U75..U100 共 26；
   - child origin identity U75..U84；
   - repaired identity U85..U100；
   - child validation U80/U90/U100 共 3；
   - final evaluations 10，episodes 640。
5. machine verifier 通过后依次 fresh 最终规格与质量审查。
6. 双审 C0/I0 后写 child Stage6 `review.json`，状态停在 `awaiting_human_approval`。
7. 不写 `approval.json`/`gate.json`，不 stage/commit，不进入 Stage7。
8. Gate 证据送达用户且状态正确后，删除 `ppo-stage-6` automation。

## Plan Self-Review Checklist

- [ ] 每条设计决策都有实现任务和测试。
- [ ] 不存在未决标记、空白实现说明或被下放的设计决策。
- [ ] origin/current、warm-start/amendment、historical/current lineage 类型一致。
- [ ] append-only journal 没有被错误当成运行期完整 SHA immutable。
- [ ] U84 load 与 U85+ write 的 lineage 语义分开。
- [ ] next attempt 和 segment 由 artifact 动态推导并在正式实例冻结。
- [ ] 没有 commit、publish checkpoint、Stage7 或额外 seed 步骤。
- [ ] 实施与 reviewer 写范围互不重叠，主 agent 负责整合与最终验收。
