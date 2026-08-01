# Stage 6 Planning-Child Recovery Capability Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Stage 6 planning-child 的 parent amendment、continuation、正式 journals、checkpoint 与当前授权集中到一个启动期 recovery capability，消除消费者层重复校验和硬编码恢复边界，在不改写 U75–U85 的前提下安全恢复 U86。

**Architecture:** parent amendment 与 continuation 仅作为静态 write-once artifact codec；新增 `stage6_planning_child_recovery.py` 作为唯一重型恢复事实所有者。该模块从同一组绑定 bytes 一次性验证正式输入，派生不可变 `PlanningChildRecoveryCapability`；workflow、backend、machine verifier 与 terminal recovery 只消费这一 capability。运行期继续使用现有 run lease、execution capability 和 input pin，不能在 per-step poll 重放 source-repair 图。

**Tech Stack:** Python 3.11、frozen/slots dataclass、`MappingProxyType`、现有 `ArtifactStore`、Stage 6 input pin/run lease/execution capability、pytest、PowerShell。

## Global Constraints

- 设计依据：
  `docs/superpowers/specs/2026-07-26-ppo-stage6-planning-child-recovery-capability-consolidation-design.md`。
- 唯一正式 run：
  `s6-standard-single-r1-20260724T000124Z`；正式 stage root：
  `D:/xunce/out/ppo_frontier/s6-standard-single-r1-20260724T000124Z/s6`。
- 正式状态必须保持 U75–U85 accepted；U85 attempt2 checkpoint SHA-256 必须为
  `7dddb04da7911b7f6705d9dcd60edbeac02614e2a13911bc5ccf52be7880d3b1`。
- parent planning-child amendment SHA-256 必须为
  `64896928a294d77229d0ea6d665e98827f51f8f2b2a6f08b1a8f155ccbcbf9ac`。
- 实现及双审 C0/I0 前：formal runner、replay、formal preview、publish、
  dry-run、identity/authorization 生成均禁止；不得创建 segment5 或 U86 pre。
- 不改 PPO、GAE、reward、网络、optimizer、action tensor、候选特征、
  0.75m/360° planning-safety 或 U85 LOS-heading 语义。
- 不发布 checkpoint、不替换 default policy、不连接 executor、不启动 canary、
  不进入 Stage 7；synthetic 仍仅为 proxy。
- 保持同一正式 run；不得重跑 U75，不得改写 U75–U85、parent amendment、
  U84/U85 checkpoint 或既有 lineage 的任何 bytes。
- 工作区已有大量其他改动。禁止 reset、clean、checkout、批量删除或大范围回退。
- 本计划是一个紧耦合实现任务，只恢复同一 Plato
  `019f9b0c-f893-72e0-bfb5-a6500aa9d97b`；不得派第二个 implementer。
- 用户已选定 subagent-driven execution；计划完成后无需再次询问执行方式。
- 项目 Gate 禁止本阶段擅自提交。即使执行技能默认建议 commit，也不得
  `git add`、`git commit` 或 push；只记录精确 diff、测试与 SHA。
- 所有 pytest 使用 `D:/xunce/tmp/pytest-stage6-recovery-capability` 下的明确
  basetemp，并加 `-p no:cacheprovider`；不得删除已有 cache。
- 所有中文修改用 UTF-8；最终用
  `D:/conda_envs/lunar-explorer/python.exe` 显式 UTF-8 读取复核。

## Allowed Write Set

实现者仅可创建或修改以下文件；需要新增其他文件时必须先停下并由主 agent
批准：

- Create:
  - `src/lunar_exploration_ppo/workflows/stage6_planning_child_recovery.py`
  - `tests/ppo_highres_frontier/test_stage6_planning_child_recovery.py`
  - `.superpowers/sdd/stage6-planning-child-recovery-capability-consolidation-task1-report.md`
- Modify:
  - `src/lunar_exploration_ppo/workflows/stage6_planning_child_source_repair.py`
  - `src/lunar_exploration_ppo/workflows/stage6_planning_child_source_repair_continuation.py`
  - `src/lunar_exploration_ppo/workflows/stage6.py`
  - `src/lunar_exploration_ppo/ppo/standard_training.py`
  - `src/lunar_exploration_ppo/workflows/stage6_terminal_recovery.py`
  - `scripts/create_ppo_stage6_planning_child_source_repair_continuation.py`
  - `tests/ppo_highres_frontier/test_stage6_planning_child_source_repair.py`
  - `tests/ppo_highres_frontier/test_stage6_planning_child_source_repair_continuation.py`
  - `tests/ppo_highres_frontier/test_stage6_planning_child_source_repair_cli.py`
  - `tests/ppo_highres_frontier/test_stage6_workflow.py`
  - `tests/ppo_highres_frontier/test_stage6_training.py`
  - `tests/ppo_highres_frontier/test_stage6_terminal_recovery.py`

---

## Task 1: Implement the Consolidated Recovery Capability as One Tightly Coupled Change

**Owner:** same Plato only.

### Task 1.1: Freeze the local and formal baseline

- [ ] Capture the allowed-file baseline without touching unrelated changes:

```powershell
git status --short -- `
  src/lunar_exploration_ppo/workflows/stage6_planning_child_recovery.py `
  src/lunar_exploration_ppo/workflows/stage6_planning_child_source_repair.py `
  src/lunar_exploration_ppo/workflows/stage6_planning_child_source_repair_continuation.py `
  src/lunar_exploration_ppo/workflows/stage6.py `
  src/lunar_exploration_ppo/ppo/standard_training.py `
  src/lunar_exploration_ppo/workflows/stage6_terminal_recovery.py `
  scripts/create_ppo_stage6_planning_child_source_repair_continuation.py `
  tests/ppo_highres_frontier/test_stage6_planning_child_recovery.py `
  tests/ppo_highres_frontier/test_stage6_planning_child_source_repair.py `
  tests/ppo_highres_frontier/test_stage6_planning_child_source_repair_continuation.py `
  tests/ppo_highres_frontier/test_stage6_planning_child_source_repair_cli.py `
  tests/ppo_highres_frontier/test_stage6_workflow.py `
  tests/ppo_highres_frontier/test_stage6_training.py `
  tests/ppo_highres_frontier/test_stage6_terminal_recovery.py
```

- [ ] Confirm no formal runner, replay, or pytest is active. Count only Python
  processes whose direct script basename is
  `run_ppo_stage6_standard.py`, `replay_update85.py`, or `pytest`/`py.test`;
  do not count the inspecting PowerShell process.
- [ ] Read the formal stage root and record, but do not alter:
  accepted update, latest complete checkpoint, resource segment count,
  absence of continuation, absence of segment5, and absence of U86 rows.
- [ ] Record SHA-256/size for the parent amendment, U84/U85 checkpoint bundles,
  five journals, and every current authorization input in the implementer report.
- [ ] Run the currently failing partial-U86 test and preserve the real RED:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
D:/conda_envs/lunar-explorer/python.exe -m pytest `
  tests/ppo_highres_frontier/test_stage6_planning_child_source_repair_continuation.py::test_continuation_derives_partial_u86_production_resume_boundary `
  -vv -p no:cacheprovider `
  --basetemp D:/xunce/tmp/pytest-stage6-recovery-capability/red-existing
```

Expected RED: current backend rejects the legal segment5-start and
segment5-plus-U86-pre shapes with stale resource/boundary drift errors.

### Task 1.2: Add the recovery issuer contract as RED tests

**Create:** `tests/ppo_highres_frontier/test_stage6_planning_child_recovery.py`

- [ ] Reuse existing fixture builders rather than copying their production
  semantics:
  - continuation helpers from
    `test_stage6_planning_child_source_repair_continuation.py`;
  - journal/resource/checkpoint helpers from `test_stage6_workflow.py`;
  - production checkpoint fixture generation from existing Stage 6 tests.
- [ ] Add these exact tests first:

```python
def test_recovery_anchor_is_preview_only_and_cannot_launch_training() -> None:
    ...

def test_recovery_capability_derives_initial_u86_attempt1_segment5() -> None:
    ...

def test_recovery_capability_derives_segment5_start_as_u86_attempt1_segment6() -> None:
    ...

def test_recovery_capability_derives_segment5_pre_as_u86_attempt2_segment6() -> None:
    ...

def test_recovery_capability_derives_u86_accepted_as_u87_attempt1() -> None:
    ...

def test_recovery_capability_rejects_rewrite_truncation_reorder_duplicate_and_gap() -> None:
    ...

def test_recovery_capability_rejects_identity_authorization_artifact_and_checkpoint_drift() -> None:
    ...

def test_recovery_capability_uses_one_bound_snapshot_and_methods_do_no_io() -> None:
    ...

def test_recovery_capability_deep_freezes_nested_payloads() -> None:
    ...

def test_recovery_capability_derives_three_lineage_epochs_and_pins_u84_u85() -> None:
    ...
```

- [ ] In the no-I/O test, patch only the secure read seam after issuance. Calls
  to `checkpoint_lineage_for_update()` and reads of `resume_cursor`,
  `acceptance_binding`, `protected_checkpoint_updates`, and
  `input_pin_requests` must succeed with all filesystem open/read seams set to
  fail.
- [ ] In the bound-snapshot test, mutate a fixture path only after the bytes have
  been captured. The in-flight bound validator must use the original bytes;
  the final currentness check must detect the mutation and issue no capability.
- [ ] In the immutability test, attempt nested mapping/list mutation from both
  caller-owned inputs and returned fields. Neither route may alter the
  capability digest or behavior.
- [ ] Run only the new module and confirm import/API RED:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
D:/conda_envs/lunar-explorer/python.exe -m pytest `
  tests/ppo_highres_frontier/test_stage6_planning_child_recovery.py `
  -vv -p no:cacheprovider `
  --basetemp D:/xunce/tmp/pytest-stage6-recovery-capability/red-issuer
```

### Task 1.3: Implement the one-shot recovery issuer

**Create:** `src/lunar_exploration_ppo/workflows/stage6_planning_child_recovery.py`

- [ ] Implement these public types and signatures exactly:

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

- [ ] Use one private immutable bound-input representation and one internal
  validator for both public entrypoints:

```python
def _validate_bound_recovery_snapshot(
    snapshot: _BoundRecoverySnapshot,
    *,
    launch_bindings: _LaunchBindings | None,
) -> _ValidatedRecoverySnapshot:
    ...
```

`launch_bindings=None` may produce only `PlanningChildRecoveryAnchor`.
Issuing a launch capability requires a valid continuation plus all launch
bindings.

- [ ] Capture each input exactly once as canonical absolute path, bytes, size and
  SHA-256. Reject symlink/reparse point, unexpected hardlink/path escape,
  non-canonical JSON/JSONL and schema mismatch using existing path/artifact
  utilities; do not add a second generic secure-I/O framework.
- [ ] Validate training/validation through
  `validate_standard_training_validation_rows`, checkpoint/journal joins through
  `verify_journal_checkpoint_bindings` and
  `completed_transaction_keys_from_journal`, resource rows through
  `validate_resource_lifecycle_rows`, and latest complete checkpoint through
  `inspect_complete_checkpoint_snapshot`. Do not reimplement these semantics.
- [ ] Use the existing restricted checkpoint decoder. Do not call
  `torch.load(..., weights_only=False)` and do not add a fallback unrestricted
  decoder.
- [ ] Derive cursor only from the bound rows:
  - no segment5 → U86 attempt1, segment5;
  - segment5 start only → U86 attempt1, segment6;
  - segment5 plus U86 attempt1 pre → U86 attempt2, segment6;
  - U86 accepted → U87 attempt1 and the next new segment.
- [ ] Detect incomplete/foreign pending attempts, duplicate accepted rows,
  non-contiguous accepted updates, transaction-key mismatch and checkpoint
  mismatch before constructing any public object.
- [ ] Derive lineage epochs from artifact relations:
  - planning warm-start/origin epoch ends U84;
  - parent amendment epoch is U85;
  - continuation epoch starts U86 with no fixed end.
- [ ] Derive protected checkpoint updates as the sorted unique union of physical
  checkpoint references. For the current chain it must equal `(84, 85)`.
- [ ] Deep-freeze all returned nested values. Implement one recursive helper:
  mappings become `MappingProxyType`, lists/tuples become tuples, and scalar
  JSON values are retained. Never store a caller-owned mutable alias.
- [ ] Compute `input_snapshot_sha256` from canonical bound-input identities and
  `capability_sha256` from the complete immutable capability payload.
- [ ] Perform a final currentness comparison against every captured immutable
  input immediately before return. Any drift raises
  `Stage6PlanningChildRecoveryError` and writes nothing.
- [ ] Run the new test module to GREEN:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
D:/conda_envs/lunar-explorer/python.exe -m pytest `
  tests/ppo_highres_frontier/test_stage6_planning_child_recovery.py `
  -vv -p no:cacheprovider `
  --basetemp D:/xunce/tmp/pytest-stage6-recovery-capability/green-issuer
```

### Task 1.4: Reduce parent and continuation modules to static artifact codecs

**Modify:**

- `src/lunar_exploration_ppo/workflows/stage6_planning_child_source_repair.py`
- `src/lunar_exploration_ppo/workflows/stage6_planning_child_source_repair_continuation.py`
- `scripts/create_ppo_stage6_planning_child_source_repair_continuation.py`
- the three source-repair/continuation test files listed in Allowed Write Set.

- [ ] Add a bytes-consuming static parent validation seam. It may preserve the
  published parent v1 schema and existing builder/publisher, but it must not read
  dynamic training/resource/job tails or derive a restart cursor.
- [ ] Keep legacy public entrypoints only where existing non-planning-child
  callers still require them. New recovery consumers must not call:
  `require_current()`, `verify_append_only_prefixes()`,
  `validate_restart_tail()`, `effective_next_update`,
  `effective_next_attempt`, `effective_next_transaction_key`, or
  `effective_next_resource_segment_index`.
- [ ] Define a static continuation validation result:

```python
@dataclass(frozen=True, slots=True)
class ValidatedPlanningChildContinuation:
    formal_run_id: str
    seed: int
    artifact_path: Path | None
    artifact_sha256: str
    parent_artifact_sha256: str
    current_execution_identity_sha256: str
    current_authorization_binding: Mapping[str, object]
    current_immutable_bindings_sha256: str
    review_evidence: Mapping[str, Mapping[str, object]]
    accepted_anchor: Mapping[str, object]
    journal_prefixes: Mapping[str, Mapping[str, object]]
    lineage_epoch: Mapping[str, object]
    input_pin_requests: tuple[tuple[str, Path], ...]
```

- [ ] Change continuation schema/mode to:

```python
PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_SCHEMA = (
    "stage6_planning_child_source_repair_continuation/v2"
)
PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_MODE = (
    "write_once_reviewed_source_epoch_successor/v1"
)
```

- [ ] Continuation v2 top-level fields must be exactly:
  `schema_version`, `mode`, `formal_run_id`, `seed`, `parent`, `current`,
  `review_evidence`, `accepted_anchor`, `journal_prefixes`, `lineage_epoch`,
  `created_at_utc`, `canonical_sha256`.
- [ ] `parent` stores only the canonical parent file binding and parent canonical
  SHA. `current` stores only the current execution-identity SHA, current
  authorization file binding and canonical immutable-bindings SHA. Full nested
  parent/current graphs must not be copied.
- [ ] Change
  `build_planning_child_source_repair_continuation_artifact(...)` to require:

```python
anchor: PlanningChildRecoveryAnchor
```

It must not reopen journals/checkpoints or issue a launch capability.
- [ ] Change validation/load to return
  `ValidatedPlanningChildContinuation`, not
  `PlanningChildSourceRepairContext`. Remove production use of
  `load_planning_child_source_repair_chain`; no compatibility loader for
  unpublished v1 is permitted.
- [ ] Keep publish write-once and atomic-exclusive through `ArtifactStore`.
  Existing output, even byte-identical, must fail in formal publish mode.
- [ ] Update the CLI preview to call
  `inspect_planning_child_recovery_anchor(...)`, build v2 in memory, print/write
  preview only to the explicitly supplied review output, and perform no formal
  stage mutation. Formal publish remains a separate explicit mode.
- [ ] Convert the continuation tests to assert:
  - v2 preview is zero-write;
  - v1 fails closed;
  - parent SHA/current auth/review binding drift fails;
  - payload does not duplicate parent graph;
  - anchor cannot launch training;
  - publish is atomic and exclusive.
- [ ] Re-run focused artifact tests:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
D:/conda_envs/lunar-explorer/python.exe -m pytest `
  tests/ppo_highres_frontier/test_stage6_planning_child_source_repair.py `
  tests/ppo_highres_frontier/test_stage6_planning_child_source_repair_continuation.py `
  tests/ppo_highres_frontier/test_stage6_planning_child_source_repair_cli.py `
  -vv -p no:cacheprovider `
  --basetemp D:/xunce/tmp/pytest-stage6-recovery-capability/green-codecs
```

### Task 1.5: Make workflow and backend consume one capability

**Modify:**

- `src/lunar_exploration_ppo/workflows/stage6.py`
- `src/lunar_exploration_ppo/ppo/standard_training.py`
- `tests/ppo_highres_frontier/test_stage6_workflow.py`
- `tests/ppo_highres_frontier/test_stage6_training.py`

- [ ] Add RED integration tests with these exact names:

```python
def test_planning_child_workflow_issues_one_recovery_capability_per_process() -> None:
    ...

def test_planning_child_backend_consumes_capability_for_first_resume_boundary() -> None:
    ...

def test_planning_child_backend_releases_frozen_cursor_after_first_transaction() -> None:
    ...

def test_planning_child_runtime_poll_does_not_reopen_recovery_inputs() -> None:
    ...

def test_planning_child_mutation_boundary_rehashes_pinned_inputs() -> None:
    ...

def test_planning_child_backend_retention_pins_current_u84_u85_chain() -> None:
    ...
```

- [ ] Replace `_load_planning_child_source_repair_for_workflow()` with a startup
  path that loads the static continuation and calls
  `issue_planning_child_recovery_capability()` exactly once.
- [ ] Replace `_require_planning_child_source_repair_current()` calls with:
  - existing execution capability/run lease checks at operation boundaries;
  - existing input pin `require_current(..., rehash=False)` at light polls;
  - existing input pin `require_current(..., rehash=True)` immediately before
    post/accepted mutation.
- [ ] Pass the same `PlanningChildRecoveryCapability` object to
  `StandardProductionBackend`, machine evidence binding, and terminal
  preparation. Do not rebuild it inside the same process.
- [ ] Make `_stage6_planning_checkpoint_lineage_for_update()` delegate only to
  `capability.checkpoint_lineage_for_update(update)` for planning-child.
- [ ] Make planning-child acceptance/manifest evidence bind
  `capability_sha256`, `input_snapshot_sha256`, parent/continuation SHA and the
  capability acceptance binding. Do not reconstruct a second nested context.
- [ ] `StandardProductionBackend` constructor accepts
  `PlanningChildRecoveryCapability | None`. Planning-child profile requires it;
  classic source-repair and ordinary runs reject it.
- [ ] In `prepare_seed()`, compare only the first remaining transaction with
  `resume_cursor` once and mark that check consumed. `run_update()` and
  `finalize()` must not call parent/continuation heavy validators.
- [ ] After the first transaction begins its normal durable lifecycle, U87+
  proceeds through the ordinary state machine; a frozen U86 cursor cannot block
  later updates.
- [ ] Retention receives
  `capability.protected_checkpoint_updates`. Replace
  `test_planning_child_real_retention_keeps_u84_and_removes_superseded_u85`
  with an expectation that the current chain retains both U84 and U85.
- [ ] Preserve classic source-repair behavior and all existing generic
  checkpoint-retention safeguards.
- [ ] Turn the existing two-case RED
  `test_continuation_derives_partial_u86_production_resume_boundary` GREEN
  through capability-derived values, not relaxed comparison.
- [ ] Run focused workflow/backend tests:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
D:/conda_envs/lunar-explorer/python.exe -m pytest `
  tests/ppo_highres_frontier/test_stage6_planning_child_recovery.py `
  tests/ppo_highres_frontier/test_stage6_planning_child_source_repair_continuation.py::test_continuation_derives_partial_u86_production_resume_boundary `
  tests/ppo_highres_frontier/test_stage6_workflow.py `
  tests/ppo_highres_frontier/test_stage6_training.py `
  -vv -p no:cacheprovider `
  --basetemp D:/xunce/tmp/pytest-stage6-recovery-capability/green-consumers
```

### Task 1.6: Consolidate machine and terminal recovery semantics

**Modify:**

- `src/lunar_exploration_ppo/workflows/stage6.py`
- `src/lunar_exploration_ppo/workflows/stage6_terminal_recovery.py`
- `tests/ppo_highres_frontier/test_stage6_workflow.py`
- `tests/ppo_highres_frontier/test_stage6_terminal_recovery.py`

- [ ] Add RED tests:

```python
def test_planning_child_machine_verifier_uses_capability_acceptance_binding() -> None:
    ...

def test_planning_child_terminal_recovery_uses_capability_acceptance_binding() -> None:
    ...

def test_machine_and_terminal_reject_different_recovery_capability_digest() -> None:
    ...

def test_terminal_commit_does_not_reopen_planning_child_journals() -> None:
    ...
```

- [ ] Independent machine/terminal processes may each issue one capability from
  formal inputs at startup. Inside one process, all machine/terminal stages must
  share that issued object and its `acceptance_binding`.
- [ ] Remove planning-child source-repair semantics from
  `_validate_planning_child_machine_context()` and
  `_validate_planning_child_resource_boundary()`; retain only generic machine
  graph/resource semantic validation not owned by the issuer.
- [ ] Remove these second-capability constructs from terminal recovery:
  `_PlanningChildTerminalCapabilityState`,
  `_PlanningChildTerminalCapability`,
  `_planning_child_terminal_capability_state`,
  `_issue_planning_child_terminal_capability`, and
  `_require_planning_child_terminal_capability`.
- [ ] Bind the normal terminal semantic result to
  `PlanningChildRecoveryCapability.capability_sha256` and
  `acceptance_binding`. The terminal evidence handle, run lease and input pin
  remain the mutation-time guards.
- [ ] Do not reread five journals, parent amendment, continuation or checkpoint
  during receipt commit. Terminal code may read only its normal terminal
  evidence inputs after startup issuance.
- [ ] Ensure classic source-repair and non-planning-child terminal paths remain
  unchanged.
- [ ] Run focused machine/terminal tests:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
D:/conda_envs/lunar-explorer/python.exe -m pytest `
  tests/ppo_highres_frontier/test_stage6_workflow.py `
  tests/ppo_highres_frontier/test_stage6_terminal_recovery.py `
  -vv -p no:cacheprovider `
  --basetemp D:/xunce/tmp/pytest-stage6-recovery-capability/green-terminal
```

### Task 1.7: Prove duplicate heavy gates are gone

- [ ] Add or update static assertions so planning-child consumers contain no
  calls to:
  `verify_append_only_prefixes`, `validate_restart_tail`, parent/continuation
  `require_current`, or `effective_next_*`.
- [ ] Assert consumer resume-boundary logic contains no hardcoded
  `84`, `85`, `86`, attempt2 or segment5 constants. Artifact codec constants and
  test fixtures may retain historical numbers.
- [ ] Assert `run_update()`, `finalize()`, resource poll and terminal commit do
  not call secure-read seams for parent, continuation, journals or checkpoint.
- [ ] Assert no new broad `except Exception` was added in the allowed source
  diff.
- [ ] Run the static search and record every remaining hit with its owner:

```powershell
rg -n `
  "verify_append_only_prefixes|validate_restart_tail|effective_next_|_PlanningChildTerminalCapability|_issue_planning_child_terminal_capability" `
  src/lunar_exploration_ppo/workflows/stage6.py `
  src/lunar_exploration_ppo/ppo/standard_training.py `
  src/lunar_exploration_ppo/workflows/stage6_terminal_recovery.py
```

Expected result: zero consumer hits, except a deliberately retained compatibility
definition that has no production caller and is explicitly justified in the
report.

### Task 1.8: Run focused and full relevant regression once

- [ ] Compile every changed source/script:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
D:/conda_envs/lunar-explorer/python.exe -m py_compile `
  src/lunar_exploration_ppo/workflows/stage6_planning_child_recovery.py `
  src/lunar_exploration_ppo/workflows/stage6_planning_child_source_repair.py `
  src/lunar_exploration_ppo/workflows/stage6_planning_child_source_repair_continuation.py `
  src/lunar_exploration_ppo/workflows/stage6.py `
  src/lunar_exploration_ppo/ppo/standard_training.py `
  src/lunar_exploration_ppo/workflows/stage6_terminal_recovery.py `
  scripts/create_ppo_stage6_planning_child_source_repair_continuation.py
```

- [ ] Run the complete relevant suite once after focused GREEN:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
D:/conda_envs/lunar-explorer/python.exe -m pytest `
  tests/ppo_highres_frontier/test_stage6_planning_child_recovery.py `
  tests/ppo_highres_frontier/test_stage6_planning_child_source_repair.py `
  tests/ppo_highres_frontier/test_stage6_planning_child_source_repair_continuation.py `
  tests/ppo_highres_frontier/test_stage6_planning_child_source_repair_cli.py `
  tests/ppo_highres_frontier/test_stage6_workflow.py `
  tests/ppo_highres_frontier/test_stage6_training.py `
  tests/ppo_highres_frontier/test_stage6_terminal_recovery.py `
  -q -p no:cacheprovider `
  --basetemp D:/xunce/tmp/pytest-stage6-recovery-capability/full
```

- [ ] Do not start a second full pytest while this one is active. If it yields,
  monitor the same PID until completion.
- [ ] Any failure must be classified against the approved design. Fix only a
  demonstrated relevant regression; do not add a new gate merely to satisfy a
  brittle historical assertion.

### Task 1.9: Write the implementer report without staging or committing

**Create:**
`.superpowers/sdd/stage6-planning-child-recovery-capability-consolidation-task1-report.md`

- [ ] Record:
  - exact allowed diff;
  - preserved pre-existing changes;
  - each RED command/failure and corresponding GREEN;
  - focused/full/compile commands with pass counts and durations;
  - remaining static-search hits and ownership;
  - formal baseline hashes before and after read-only checks;
  - source/test/report SHA-256 values;
  - explicit confirmation of zero identity/auth/preview/publish/runner/replay;
  - explicit confirmation that no file was staged or committed.
- [ ] End the report with a concise handoff:
  `READY_FOR_MAIN_AGENT_VERIFICATION` only if all Task 1 checks pass.

---

## Task 2: Main-Agent Independent Verification

**Owner:** main agent; Plato must be stopped or waiting.

- [ ] Review the allowed diff line by line against the approved design.
- [ ] Confirm the write set contains no unauthorized file and no unrelated
  overwrite.
- [ ] Independently verify:
  - one issuer owns all heavy recovery facts;
  - anchor cannot launch;
  - continuation is static v2 and write-once;
  - capability is deeply immutable and its methods do no I/O;
  - cursor handles all four durable-tail states;
  - backend checks the first transaction once, then ordinary lifecycle resumes;
  - machine and terminal consume capability acceptance binding;
  - runtime poll is light while mutation boundary rehashes pinned inputs;
  - retention pins exactly U84/U85;
  - no unrestricted checkpoint load.
- [ ] Re-run the new recovery test module, the existing partial-U86 test,
  compile command, and the full relevant regression using fresh D basetemp
  suffixes under `.../main-validation`.
- [ ] Take a read-only before/after full file-set/size/SHA snapshot of the formal
  stage root and current authorization evidence. Any drift blocks review.
- [ ] Confirm formal runner/replay/pytest count is zero.
- [ ] If Critical/Important defect is found, send one consolidated finding set
  back to the same Plato. Do not dispatch a new implementer.

---

## Task 3: Parallel Fresh Review After Main Verification

**Owners:** same Jason for spec; same Hegel for quality/security.

- [ ] Only after Task 2 passes, dispatch both reviews in parallel:
  - Jason `019f991a-3082-7980-96aa-90471049b65f` checks exact conformance to
    the approved design and this plan.
  - Hegel `019f9b7e-eefc-74f3-867f-6872d3dc116a` checks code quality,
    immutability, secure bound-byte handling, checkpoint safety, runtime
    complexity, tests and production recoverability.
- [ ] Reviews are read-only and independent. They may run tests under different
  D basetemp paths but may not edit source or formal artifacts.
- [ ] Persist their raw Markdown verbatim as new review artifacts with SHA-256.
- [ ] Any Critical/Important finding returns as one consolidated R1 brief to the
  same Plato; after the scoped fix, main-agent verification and both scoped
  re-reviews repeat.
- [ ] Do not generate identity/auth, preview, publish or start a runner until
  both reviews are C0/I0.

---

## Task 4: Formal Continuation and U86 Recovery

**Owner:** main agent, only after Task 3 C0/I0.

- [ ] Reconfirm formal runner/replay/pytest zero and unchanged U75–U85/history.
- [ ] Generate one new execution identity and authorization bound to the exact
  reviewed tree, report and both C0/I0 review artifacts.
- [ ] Call `inspect_planning_child_recovery_anchor()` on formal bound bytes.
- [ ] Run formal continuation v2 preview with zero stage-root writes; verify the
  preview binds:
  - parent amendment full file binding/SHA;
  - new identity/authorization;
  - both review artifacts;
  - accepted U85 anchor and five journal prefix seals;
  - U86+ lineage epoch.
- [ ] Compare formal stage-root and authorization snapshots before/after preview;
  they must be identical.
- [ ] Publish
  `planning-child-source-repair-continuation.json` exactly once via
  `ArtifactStore` atomic exclusive write.
- [ ] Reload formal bytes and issue
  `PlanningChildRecoveryCapability`; expected cursor must be dynamically
  U86 attempt1 and next resource segment5 for the current state.
- [ ] Run one zero-write recovery dry-run. Confirm no formal artifact changes,
  no duplicate runner, and no partial transaction write.
- [ ] Launch exactly one formal recovery runner from accepted U85. The first new
  durable transaction must be U86 attempt1 in the capability-derived segment.
- [ ] Resume ordinary monitoring through U100. Do not modify the Goal for every
  healthy non-validation update.
- [ ] At U100, use only a planning-child-corrected final-controller runbook that
  has separately passed C0/I0 review; otherwise stop at the terminal state and
  notify the user. Do not enter Stage 7.

## Definition of Done

- [ ] Heavy planning-child recovery validation has one owner.
- [ ] Parent/continuation codecs contain no dynamic restart-cursor ownership.
- [ ] Workflow/backend/machine/terminal do not independently reopen or derive
  the same recovery state.
- [ ] Current partial-U86 RED is GREEN without relaxing fail-closed semantics.
- [ ] Capability cursor, lineage, retention, immutability and no-I/O behavior are
  covered by tests.
- [ ] Compile, focused, full relevant regression and formal read-only snapshot
  checks pass.
- [ ] Main-agent verification, Jason spec review and Hegel quality/security
  review are C0/I0.
- [ ] Only then is continuation v2 published once and U86 formal training
  resumed.
