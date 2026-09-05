# Project Backup And Route Retirement Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不删除任何源码、子模块或运行产物的前提下，为 `6a4c2dd` 建立可独立恢复的 Git 备份，生成逐路径退役候选清单，并冻结 Stage6、G1/G2/G3、v3、平台约束和默认 A* 的验收基线。

**Architecture:** 本计划是物理精简前的非破坏性准备阶段。先对父仓库和四个子模块制作 bundle/tag，再在独立 worktree 中增加只读审计工具；工具以当前 Git 树、已校验知识图谱和明确的保留优先规则生成 D 盘 manifest。任何物理删除都拆入后续独立计划，并以本计划产出的精确路径清单和负责人批准为输入。

**Tech Stack:** Git、Git bundle、Git worktree、PowerShell、Python 3.12、pytest、JSON、Understand Anything knowledge graph。

## Global Constraints

- 基线父仓库提交固定为 `6a4c2dd0352fd6c1918a5eef39c9783b9d3c5c65`（短 SHA `6a4c2dd`）；后续命令不得改用浮动分支名作为备份基线。
- 当前四个 gitlink 固定为：`path-planner=2f6378d3c47da027c0d4146d94cab881b8f2a594`、`model-explorer=b547a997d94ad199c822136d1ae345e180b87ca7`、`dev-platform-constraints=61e9fa8afd09db83632456bdcf181c222ee13513`、`visual-workbench=9acc4ce83fc2884221a9759fa00824692eab3315`。
- 备份根固定为 `D:/CodexDownloads/lunar-path-planning-backups/2026-08-01-6a4c2dd`；审计输出根固定为 `D:/xunce/out/route_retire_preflight`。
- 知识图谱只读输入固定为 `D:/codex/project/lunar-path-planning/.ua/knowledge-graph.json`；验收基线为 252 个文件、1480 个节点、2710 条边和 7 个架构层。
- 保留：`src/lunar_exploration_ppo/`、Stage6、G1/G2/G3、完整 `path-planner/`、完整 `dev-platform-constraints/`、v3、`scripts/xunce_artifact_io.py`、`scripts/xunce_artifact_paths.py`、默认 `AStarPlanner`。
- 退役候选：Xunce Stage18--26、path-feedback / 早期策略实验、`model-explorer`、`visual-workbench`；“退役”不等于本计划授权删除。
- 禁止递归、批量和通配符删除；禁止 `Remove-Item -Recurse`、`rm -rf`、`del /s`、`rd /s`、`rmdir /s`。本计划不执行任何删除。
- 不修改 `configs/stage_registry.json`、`.gitmodules` 或 gitlink；它们只作为审计输入。
- 不移动、不删除历史 D 盘 outputs；不接管 `outputs/pytest-stage18-4e` 的 ACL。
- 不暂存或提交现有未跟踪的 `docs/uncertainty-aware-2p5d-traversability-model.md`、`docs/外部输入/`、`hello.obj`、`schema_codec_conformance_test.obj` 或旧的未跟踪计划文件。
- 不发布 checkpoint、不替换 default policy、不连接 executor、不启动 canary；`max_traversable_slope_deg=30.0` 保持不变。

---

## Planned File Structure

| Path | Responsibility |
|---|---|
| `configs/route_retirement_policy_v1.json` | 唯一的保留、人工复核、退役候选匹配策略；保留规则优先于退役规则。 |
| `scripts/audit_route_retirement_preflight.py` | 只读枚举 Git、图谱、子模块和 outputs 元数据，生成精确候选与交叉引用报告；不包含删除功能。 |
| `tests/test_route_retirement_preflight.py` | 验证规则优先级、图谱合同、精确路径输出、权限错误处理和无删除能力。 |
| `D:/xunce/out/route_retire_preflight/summary.json` | 总体状态、基线提交、候选计数、阻塞项和下一步 route。 |
| `D:/xunce/out/route_retire_preflight/candidate-manifest.json` | 排序后的单一精确路径记录，供后续删除计划逐项引用。 |
| `D:/xunce/out/route_retire_preflight/reference-audit.json` | 保留路线到退役候选的 Git/rg/图谱引用边。 |
| `D:/xunce/out/route_retire_preflight/backup-manifest.json` | 父仓库与四个子模块 bundle 的路径、大小、SHA-256 和 `git bundle verify` 状态。 |
| `D:/xunce/out/route_retire_preflight/baseline-tests/` | 清理前的 Stage6、G1/G2/G3、v3 和平台约束测试基线。 |
| `D:/xunce/out/route_retire_preflight/report.md` | 面向负责人的批准清单；不得写入自动删除命令。 |

---

### Task 1: Freeze A Recoverable Repository Snapshot

**Files:**
- Create externally: `D:/CodexDownloads/lunar-path-planning-backups/2026-08-01-6a4c2dd/*.bundle`
- Create externally: `D:/CodexDownloads/lunar-path-planning-backups/2026-08-01-6a4c2dd/bundle-hashes.txt`
- Test: parent repository and four submodule repositories

**Interfaces:**
- Consumes: parent commit `6a4c2dd` and the four fixed gitlink SHAs from Global Constraints.
- Produces: five verified bundles plus tag `archive/pre-route-retirement-20260801-6a4c2dd`; Task 3 records their hashes in `backup-manifest.json`.

- [ ] **Step 1: Verify the exact snapshot inputs**

Run from `C:/Users/77634/.codex/worktrees/ca49/lunar-path-planning`:

```powershell
$baselineCommit = '6a4c2dd0352fd6c1918a5eef39c9783b9d3c5c65'
git show --no-patch --oneline $baselineCommit
git submodule status
git diff --cached --quiet
```

Expected: the parent resolves to the committed route-documentation change; the four gitlinks match Global Constraints; the index has no staged changes. Existing unrelated untracked files are allowed and must remain untouched.

- [ ] **Step 2: Create the exact D-drive backup directory**

```powershell
$backupRoot = 'D:\CodexDownloads\lunar-path-planning-backups\2026-08-01-6a4c2dd'
if (Test-Path -LiteralPath $backupRoot) { throw "Backup root already exists: $backupRoot" }
New-Item -ItemType Directory -Path $backupRoot | Out-Null
```

Expected: one new empty directory at the exact D-drive path. Do not silently select another root.

- [ ] **Step 3: Create one bundle per Git repository**

```powershell
$backupRoot = 'D:\CodexDownloads\lunar-path-planning-backups\2026-08-01-6a4c2dd'
git bundle create "$backupRoot\lunar-path-planning.bundle" --all
git -C path-planner bundle create "$backupRoot\path-planner.bundle" --all
git -C model-explorer bundle create "$backupRoot\model-explorer.bundle" --all
git -C dev-platform-constraints bundle create "$backupRoot\dev-platform-constraints.bundle" --all
git -C visual-workbench bundle create "$backupRoot\visual-workbench.bundle" --all
```

Expected: five non-empty bundle files. A parent bundle alone is insufficient because it stores only submodule gitlink SHAs, not submodule objects.

- [ ] **Step 4: Verify every bundle and record hashes**

```powershell
$backupRoot = 'D:\CodexDownloads\lunar-path-planning-backups\2026-08-01-6a4c2dd'
Get-ChildItem -LiteralPath $backupRoot -Filter '*.bundle' | ForEach-Object {
  git bundle verify $_.FullName
  if ($LASTEXITCODE -ne 0) { throw "Bundle verification failed: $($_.FullName)" }
}
$requiredHeads = @{
  'lunar-path-planning.bundle' = '6a4c2dd0352fd6c1918a5eef39c9783b9d3c5c65'
  'path-planner.bundle' = '2f6378d3c47da027c0d4146d94cab881b8f2a594'
  'model-explorer.bundle' = 'b547a997d94ad199c822136d1ae345e180b87ca7'
  'dev-platform-constraints.bundle' = '61e9fa8afd09db83632456bdcf181c222ee13513'
  'visual-workbench.bundle' = '9acc4ce83fc2884221a9759fa00824692eab3315'
}
foreach ($bundleName in $requiredHeads.Keys) {
  $bundleHeads = git bundle list-heads "$backupRoot\$bundleName"
  if ($bundleHeads -notmatch $requiredHeads[$bundleName]) {
    throw "Required commit missing from bundle: $bundleName"
  }
}
Get-ChildItem -LiteralPath $backupRoot -Filter '*.bundle' |
  Sort-Object Name |
  Get-FileHash -Algorithm SHA256 |
  ForEach-Object { "{0}  {1}" -f $_.Hash.ToLowerInvariant(), $_.Path } |
  Set-Content -LiteralPath "$backupRoot\bundle-hashes.txt" -Encoding utf8NoBOM
```

Expected: five successful `git bundle verify` results, every fixed parent/gitlink commit present in its corresponding bundle, and five SHA-256 lines.

- [ ] **Step 5: Create the immutable rollback tag**

```powershell
$baselineCommit = '6a4c2dd0352fd6c1918a5eef39c9783b9d3c5c65'
$tagName = 'archive/pre-route-retirement-20260801-6a4c2dd'
git rev-parse --verify "refs/tags/$tagName"
```

Expected before creation: exit code 1. If the tag exists, verify it points to the same commit; otherwise stop instead of overwriting it.

```powershell
git tag -a $tagName $baselineCommit -m 'Archive before physical route retirement'
git show --no-patch --decorate $tagName
```

Expected: the tag resolves to the exact baseline commit.

- [ ] **Step 6: Push the rollback tag only after explicit external-write approval**

```powershell
git push origin refs/tags/archive/pre-route-retirement-20260801-6a4c2dd
```

Expected: the remote accepts the single annotated tag. If push approval is unavailable, retain the verified local tag and five bundles and mark `remote_tag_status=not_pushed` in Task 3 rather than guessing permission.

### Task 2: Create An Isolated Preflight Worktree

**Files:**
- Create externally: `D:/CodexDownloads/lunar-path-planning-worktrees/route-decommission-preflight/`
- Test: Git worktree and branch metadata

**Interfaces:**
- Consumes: baseline commit and verified bundles from Task 1.
- Produces: clean branch `codex/route-decommission-preflight` for Tasks 3--5; user-owned untracked files remain only in the original worktree.

- [ ] **Step 1: Invoke the worktree safety skill**

At execution time, read and follow `superpowers:using-git-worktrees` before creating the worktree. Confirm the target does not already exist:

```powershell
$worktreeRoot = 'D:\CodexDownloads\lunar-path-planning-worktrees\route-decommission-preflight'
if (Test-Path -LiteralPath $worktreeRoot) { throw "Worktree target already exists: $worktreeRoot" }
```

- [ ] **Step 2: Create the cleanup-preflight branch from the frozen commit**

```powershell
New-Item -ItemType Directory -Path 'D:\CodexDownloads\lunar-path-planning-worktrees' -Force | Out-Null
git worktree add 'D:\CodexDownloads\lunar-path-planning-worktrees\route-decommission-preflight' -b codex/route-decommission-preflight 6a4c2dd0352fd6c1918a5eef39c9783b9d3c5c65
```

Expected: a clean worktree on `codex/route-decommission-preflight` at `6a4c2dd`.

- [ ] **Step 3: Initialize submodules without changing their recorded commits**

```powershell
git -C 'D:\CodexDownloads\lunar-path-planning-worktrees\route-decommission-preflight' submodule update --init --recursive
git -C 'D:\CodexDownloads\lunar-path-planning-worktrees\route-decommission-preflight' submodule status
```

Expected: all four gitlinks match Global Constraints exactly.

### Task 3: Add Deterministic Retirement-Preflight Tooling

**Files:**
- Create: `configs/route_retirement_policy_v1.json`
- Create: `scripts/audit_route_retirement_preflight.py`
- Create: `tests/test_route_retirement_preflight.py`

**Interfaces:**
- Consumes: repo root, policy JSON, knowledge graph, backup root and baseline commit.
- Produces: `CandidateRecord`, `GraphSummary`, `build_manifest(repo_root, policy_path, graph_path, backup_root)`, and CLI artifacts under the exact D-drive output root.

- [ ] **Step 1: Write the classification-policy test first**

Create `tests/test_route_retirement_preflight.py` with these first contracts:

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_route_retirement_preflight import classify_path, load_policy


def test_protected_rules_win_over_retirement_patterns() -> None:
    policy = load_policy(Path("configs/route_retirement_policy_v1.json"))
    assert classify_path("scripts/xunce_artifact_io.py", policy).classification == "protected"
    assert classify_path("scripts/run_xunce_mid_dual_g3_closed_loop.py", policy).classification == "protected"
    assert classify_path("path-planner/src/path_planner/search/astar.py", policy).classification == "protected"
    assert classify_path("dev-platform-constraints/src/dev_platform_constraints/core/contracts.py", policy).classification == "protected"


def test_retired_and_manual_review_boundaries_are_distinct() -> None:
    policy = load_policy(Path("configs/route_retirement_policy_v1.json"))
    assert classify_path("scripts/run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke.py", policy).classification == "retire_candidate"
    assert classify_path("scripts/run_path_feedback_validation.py", policy).classification == "retire_candidate"
    assert classify_path("model-explorer", policy).classification == "manual_review"
    assert classify_path("visual-workbench", policy).classification == "manual_review"
    assert classify_path("configs/stage_registry.json", policy).classification == "manual_review"
```

- [ ] **Step 2: Run the focused test and confirm red state**

Run:

```powershell
python -m pytest tests/test_route_retirement_preflight.py -q
```

Expected: collection fails because `audit_route_retirement_preflight` does not exist.

- [ ] **Step 3: Create the exact policy schema**

Create `configs/route_retirement_policy_v1.json` with these top-level fields and precedence:

```json
{
  "schema_version": "route_retirement_policy/v1",
  "classification_precedence": ["protected", "manual_review", "retire_candidate", "unclassified"],
  "protected_globs": [
    "src/lunar_exploration_ppo/**",
    "configs/ppo_highres_frontier_stage*.json",
    "scripts/*ppo_stage6*.py",
    "scripts/run_ppo_stage6*.py",
    "tests/ppo_highres_frontier/**",
    "scripts/run_xunce_mid_dual_*.py",
    "scripts/xunce_mid_dual_*.py",
    "configs/xunce_mid_dual_*.json",
    "tests/test_xunce_mid_dual_*.py",
    "independent/g2_t2_producer/**",
    "path-planner",
    "path-planner/**",
    "dev-platform-constraints",
    "dev-platform-constraints/**",
    "configs/platforms/**",
    "scripts/xunce_artifact_io.py",
    "scripts/xunce_artifact_paths.py",
    "scripts/xunce_platform_contract.py"
  ],
  "manual_review_globs": [
    ".gitmodules",
    "configs/stage_registry.json",
    "model-explorer",
    "visual-workbench",
    "scripts/bootstrap_env.py",
    "scripts/bootstrap_ubuntu_conda.sh",
    "scripts/bootstrap_windows_conda.ps1",
    "scripts/run_platform_smoke.py",
    "scripts/run_platform_validation_matrix.py",
    "tests/test_platform_stage_runner.py",
    "tests/test_platform_validation_matrix.py"
  ],
  "retire_candidate_globs": [
    "scripts/run_xunce_stage1[89]_*.py",
    "scripts/run_xunce_stage2[0-6]_*.py",
    "configs/xunce_stage1[89]_*.json",
    "configs/xunce_stage2[0-6]_*.json",
    "tests/test_xunce_stage1[89]_*.py",
    "tests/test_xunce_stage2[0-6]_*.py",
    "scripts/*path_feedback*",
    "configs/path_feedback_*",
    "tests/*path_feedback*",
    "docs/**/*path-feedback*",
    "docs/superpowers/plans/*xunce-stage1[89]*",
    "docs/superpowers/plans/*xunce-stage2[0-6]*"
  ],
  "forbidden_delete_roots": [
    ".",
    "src/lunar_exploration_ppo",
    "path-planner",
    "dev-platform-constraints"
  ]
}
```

- [ ] **Step 4: Implement the read-only audit interfaces**

Create `scripts/audit_route_retirement_preflight.py` with these exact public contracts:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

Classification = Literal["protected", "manual_review", "retire_candidate", "unclassified"]


@dataclass(frozen=True)
class CandidateRecord:
    path: str
    classification: Classification
    matched_rules: tuple[str, ...]
    tracked: bool
    ignored: bool


@dataclass(frozen=True)
class GraphSummary:
    git_commit_hash: str
    analyzed_files: int
    nodes: int
    edges: int
    layers: int


```

Implement the callable interfaces below exactly:

| Callable | Required behavior |
|---|---|
| `load_policy(path: Path) -> Mapping[str, Any]` | UTF-8 read, schema/version validation, immutable mapping result. |
| `classify_path(path: str, policy: Mapping[str, Any]) -> CandidateRecord` | Normalize to a POSIX relative path, apply precedence, and return every matching rule. |
| `validate_graph(path: Path) -> GraphSummary` | Read-only JSON validation against the frozen graph counts and commit. |
| `list_tracked_paths(repo_root: Path) -> tuple[str, ...]` | Return sorted `git ls-files -z` paths without ignored or untracked files. |
| `audit_references(repo_root: Path, records: Sequence[CandidateRecord], graph_path: Path) -> dict[str, Any]` | Combine deterministic `rg` evidence and graph 1-hop edges, keyed by exact candidate path. |
| `build_manifest(repo_root: Path, policy_path: Path, graph_path: Path, backup_root: Path) -> dict[str, Any]` | Combine classification, references, bundle verification/hashes and blocking reasons into a stable sorted manifest. |

Implementation rules:

- Normalize all repository paths to forward-slash relative paths before matching.
- Use `PurePosixPath.match` or `fnmatch.fnmatchcase`; evaluate categories strictly in `classification_precedence` order.
- Use `git ls-files -z` for tracked files and `git check-ignore -z --stdin` for ignored-state evidence.
- Parse the knowledge graph read-only; require exactly 252 analyzed files, 1480 nodes, 2710 edges and 7 layers for this baseline.
- Record graph commit `1682d7f9f755eb59c85e6d0ceaaa6f6d5d203dfd` and whether it is an ancestor of the parent baseline; do not mislabel graph drift as a deletion candidate.
- Treat `PermissionError` while enumerating ignored outputs as a stable `permission_denied` record; never change ACLs.
- Expose exactly these CLI arguments: required `--repo-root`, `--policy`, `--knowledge-graph`, `--backup-root`, `--output-root`, `--baseline-commit`; optional `--baseline-tests-root` for hashing JUnit results after Task 5.
- The CLI may write JSON/Markdown only below the caller-provided output root. It must not expose a delete, move, unlink, rmdir, git-rm or cleanup mode.

- [ ] **Step 5: Add graph and no-delete contract tests**

Extend the test file:

```python
import inspect

from audit_route_retirement_preflight import build_manifest, validate_graph


def test_validates_the_frozen_knowledge_graph() -> None:
    summary = validate_graph(Path("D:/codex/project/lunar-path-planning/.ua/knowledge-graph.json"))
    assert (summary.analyzed_files, summary.nodes, summary.edges, summary.layers) == (252, 1480, 2710, 7)


def test_audit_module_has_no_delete_capability() -> None:
    import audit_route_retirement_preflight as module

    source = inspect.getsource(module).lower()
    for forbidden in ("unlink(", "rmdir(", "remove-item", "rm -rf", "git rm", "shutil.rmtree"):
        assert forbidden not in source
```

- [ ] **Step 6: Run focused tests to green**

```powershell
python -m pytest tests/test_route_retirement_preflight.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit the audit tooling only**

```powershell
git add -- configs/route_retirement_policy_v1.json scripts/audit_route_retirement_preflight.py tests/test_route_retirement_preflight.py
git diff --cached --check
git diff --cached --name-only
git commit -m "chore(routes): add retirement preflight audit"
```

Expected staged scope: exactly the three files listed above.

### Task 4: Generate And Validate The Exact Candidate Manifest

**Files:**
- Create externally: `D:/xunce/out/route_retire_preflight/{summary.json,candidate-manifest.json,reference-audit.json,backup-manifest.json,report.md}`
- Test: `tests/test_route_retirement_preflight.py`

**Interfaces:**
- Consumes: Task 1 bundles, Task 3 policy/tooling, current clean worktree and knowledge graph.
- Produces: sorted exact-path records and blocking references; later physical-removal plans may consume only records with `classification=retire_candidate`, `blocking_references=[]`, and explicit owner approval.

- [ ] **Step 1: Run the audit in the isolated worktree**

```powershell
python scripts/audit_route_retirement_preflight.py `
  --repo-root . `
  --policy configs/route_retirement_policy_v1.json `
  --knowledge-graph D:/codex/project/lunar-path-planning/.ua/knowledge-graph.json `
  --backup-root D:/CodexDownloads/lunar-path-planning-backups/2026-08-01-6a4c2dd `
  --output-root D:/xunce/out/route_retire_preflight `
  --baseline-commit 6a4c2dd0352fd6c1918a5eef39c9783b9d3c5c65
```

Expected: five artifacts are written under the exact D-drive root; repository files are unchanged.

- [ ] **Step 2: Validate manifest determinism**

```powershell
Get-FileHash -Algorithm SHA256 D:\xunce\out\route_retire_preflight\candidate-manifest.json
python scripts/audit_route_retirement_preflight.py `
  --repo-root . `
  --policy configs/route_retirement_policy_v1.json `
  --knowledge-graph D:/codex/project/lunar-path-planning/.ua/knowledge-graph.json `
  --backup-root D:/CodexDownloads/lunar-path-planning-backups/2026-08-01-6a4c2dd `
  --output-root D:/xunce/out/route_retire_preflight `
  --baseline-commit 6a4c2dd0352fd6c1918a5eef39c9783b9d3c5c65
Get-FileHash -Algorithm SHA256 D:\xunce\out\route_retire_preflight\candidate-manifest.json
```

Expected: both hashes are identical.

- [ ] **Step 3: Verify protected boundaries and unresolved references**

```powershell
@'
import json
from pathlib import Path

root = Path("D:/xunce/out/route_retire_preflight")
manifest = json.loads((root / "candidate-manifest.json").read_text(encoding="utf-8"))
records = manifest["records"]
by_path = {row["path"]: row for row in records}

assert by_path["scripts/xunce_artifact_io.py"]["classification"] == "protected"
assert by_path["scripts/xunce_artifact_paths.py"]["classification"] == "protected"
assert by_path["path-planner"]["classification"] == "protected"
assert by_path["dev-platform-constraints"]["classification"] == "protected"
assert by_path["model-explorer"]["classification"] == "manual_review"
assert by_path["visual-workbench"]["classification"] == "manual_review"
assert all(row["path"] not in {"hello.obj", "schema_codec_conformance_test.obj"} for row in records)
print("RETIREMENT_MANIFEST_BOUNDARIES_OK")
'@ | python -
```

Expected: `RETIREMENT_MANIFEST_BOUNDARIES_OK`.

- [ ] **Step 4: Confirm the audit changed no repository files**

```powershell
git status --short
git diff --check
```

Expected: clean worktree after the Task 3 commit; all new audit artifacts live on D drive.

### Task 5: Freeze Retained-Route Test Baselines

**Files:**
- Create externally: `D:/xunce/out/route_retire_preflight/baseline-tests/*.xml`
- Create externally: `D:/CodexDownloads/lunar-path-planning-builds/planner-v3-preflight/`
- Test: retained parent and submodule suites listed below

**Interfaces:**
- Consumes: clean preflight worktree and protected paths from the candidate manifest.
- Produces: machine-readable baseline results that every later removal plan must reproduce or improve.

- [ ] **Step 1: Run default A* and Stage6 adapter tests**

```powershell
python -m pytest `
  tests/ppo_highres_frontier/test_stage1_smoke_env.py `
  tests/ppo_highres_frontier/test_stage6_planning_unknown_buffer.py `
  -q --junitxml=D:/xunce/out/route_retire_preflight/baseline-tests/stage6-planner.xml
```

Expected: pass; tests exercise `PathPlannerAdapter` and default `AStarPlanner` binding.

- [ ] **Step 2: Run G1/G2/G3 delivery-chain tests**

```powershell
python -m pytest `
  tests/test_xunce_mid_dual_aggregate.py `
  tests/test_xunce_mid_dual_g1_coverage.py `
  tests/test_xunce_mid_dual_g2_planning_time.py `
  tests/test_xunce_mid_dual_g3_closed_loop.py `
  -q --junitxml=D:/xunce/out/route_retire_preflight/baseline-tests/mid-dual.xml
```

Expected: pass with no missing artifact-helper import.

- [ ] **Step 3: Run parent platform-contract tests**

```powershell
python -m pytest `
  tests/test_platform_validation_matrix.py `
  tests/test_platform_stage_runner.py `
  -q --junitxml=D:/xunce/out/route_retire_preflight/baseline-tests/platform-parent.xml
```

Expected: pass. Any visual-workbench-specific assertion is recorded in `reference-audit.json` as a later decoupling requirement, not silently removed here.

- [ ] **Step 4: Run retained Python submodule suites**

```powershell
python -m pytest path-planner/tests -q --junitxml=D:/xunce/out/route_retire_preflight/baseline-tests/path-planner-python.xml
python -m pytest dev-platform-constraints/tests -q --junitxml=D:/xunce/out/route_retire_preflight/baseline-tests/dev-platform-constraints.xml
```

Expected: both suites pass. A `dev-platform-constraints` test that explicitly depends on `model-explorer` becomes a blocking decoupling item for the follow-on submodule plan.

- [ ] **Step 5: Configure, build and test planner v3 on D drive**

```powershell
$buildRoot = 'D:\CodexDownloads\lunar-path-planning-builds\planner-v3-preflight'
cmake -S path-planner/cpp -B $buildRoot -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build $buildRoot --config Release
ctest --test-dir $buildRoot -C Release --output-on-failure
```

Expected: configure, build and ctest all succeed. Do not create the C++ build tree on C drive.

- [ ] **Step 6: Record baseline status in `summary.json`**

Re-run the audit CLI with `--baseline-tests-root D:/xunce/out/route_retire_preflight/baseline-tests` so it hashes the JUnit XML files and writes `baseline_status=passed` only when every required file exists and reports zero failures.

### Task 6: Owner Approval Gate And Follow-On Plan Split

**Files:**
- Read: `D:/xunce/out/route_retire_preflight/report.md`
- Read: `D:/xunce/out/route_retire_preflight/candidate-manifest.json`
- Read: `docs/route-retirement-manifest.md`
- No repository modifications

**Interfaces:**
- Consumes: verified backup, deterministic manifest and passing retained-route baselines.
- Produces: an explicit owner decision for four separate implementation plans; no decision is inferred from the existing “retired” label.

- [ ] **Step 1: Review blockers and exact counts**

```powershell
Get-Content -LiteralPath 'D:\xunce\out\route_retire_preflight\report.md' -Raw -Encoding utf8
```

Expected: the report lists exact candidate counts, protected exceptions, cross-route references, permission errors, backup hashes and baseline results.

- [ ] **Step 2: Stop and request explicit approval for each follow-on plan**

The four follow-on plans are deliberately independent:

1. `path-feedback-model-visual-retirement`: decouple retained platform scripts/tests, remove parent path-feedback entry points and registry records, then remove only the `model-explorer` and `visual-workbench` gitlinks/stanzas; local submodule directories remain untouched.
2. `xunce-stage26-retirement`: remove exact Stage26 runner/config/test/doc records while preserving shared artifact helpers, Stage6, G1/G2/G3, v3 and default A*.
3. `xunce-stage18-25-retirement`: process the exact manifest one path at a time after all retained references are resolved; preserve blocked v2/v3 gates and shared helpers.
4. `retired-runtime-artifact-cleanup`: index path-feedback evidence, then present individual explicit output paths for human cleanup; handle `outputs/pytest-stage18-4e` permission denial without ACL changes.

Each follow-on requires a new `superpowers:writing-plans` document containing the exact paths approved from `candidate-manifest.json`. Do not begin one plan merely because another was approved.

## Final Verification For This Plan

- [ ] Run the focused audit tests:

```powershell
python -m pytest tests/test_route_retirement_preflight.py -q
```

- [ ] Verify repository scope and formatting:

```powershell
git diff --check
git status --short
```

- [ ] Verify all bundles and external artifacts:

```powershell
Get-ChildItem -LiteralPath 'D:\CodexDownloads\lunar-path-planning-backups\2026-08-01-6a4c2dd' -Filter '*.bundle' |
  ForEach-Object { git bundle verify $_.FullName; if ($LASTEXITCODE -ne 0) { throw $_.FullName } }
Get-ChildItem -LiteralPath 'D:\xunce\out\route_retire_preflight' -File
```

- [ ] Confirm no physical cleanup occurred:

```powershell
git diff --diff-filter=D --name-only 6a4c2dd0352fd6c1918a5eef39c9783b9d3c5c65..HEAD
```

Expected for this plan: no deleted repository paths. The only repository commit after the baseline is the three-file audit-tooling commit; bundles, manifests, test reports and build trees are external D-drive artifacts.

## Execution Handoff

This plan ends at the owner approval gate. It intentionally does not remove files, submodules, registry entries or outputs. Once approved, execute it either with fresh task agents and review gates or inline with checkpointed batches; physical removal begins only in the four exact-path follow-on plans.
