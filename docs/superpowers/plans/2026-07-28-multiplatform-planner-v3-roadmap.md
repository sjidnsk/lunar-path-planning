# Multiplatform Planner v3 Master Implementation Roadmap

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按可独立评审、可独立测试的阶段实现多平台路径规划 v3 C++20 核心、三平台算法、原子接口和性能验收。

**Architecture:** v3 是 `path-planner/cpp/` 下的 clean-room C++20 子工程。实现顺序固定为 schema/合同 → 共享核心 → 轮式与足式 → 飞跃式 → 统一编排与性能；每个分卷独立提交并通过自己的 completion gate 后才能进入依赖它的分卷。

**Tech Stack:** C++20、CMake 3.28+、vcpkg 2026.04.27 baseline、Eigen 5.0.1、nlohmann_json 3.12.0、double-conversion 3.4.0、PicoSHA2 1.0.1、GoogleTest 1.17.0、Google Benchmark 1.9.5、OSQP 1.0.0（可选 feature）、pybind11 3.0.1（可选 feature）、Python 3.12 合同测试。

## Global Constraints

- 设计规范：`docs/superpowers/specs/2026-07-28-multiplatform-path-planner-v3-design.md`。
- 接口规范：`docs/superpowers/specs/2026-07-28-multiplatform-path-planner-v3-interface-schema.md`。
- 机器 schema：`path-planner/schemas/v3/**`。
- C++ 实现：`path-planner/cpp/**`，命名空间 `lunar::planning::v3`。
- 不调用、复制或移植现有 Python 规划实现；Python 只用于合同 fixture 和可选诊断适配。
- 不替换现有默认 A*，不连接 executor，不发布 checkpoint，不启动 canary。
- 安全、候选和降级结果不受 1 秒墙钟截止影响。
- 1 秒只用于固定 `BenchmarkProfile` 上的稳态 API 延迟 P95 验收。
- 依赖、编译和 benchmark 大文件写入 D 盘。
- 每个分卷完成后先 review 和运行 completion gate，再进入下一依赖阶段。

---

## 分卷与依赖

| 顺序 | 分卷 | 可独立交付物 | 依赖 |
|---|---|---|---|
| 0 | Interface Schema | Draft 2020-12 schema、C++ 映射和状态组合 | 无 |
| 1 | Contracts & Core | CMake、合同、验证、地图、安全投影、ARA*、走廊、QP、缓存 | 0 |
| 2A | Wheeled | 前进/倒车/自旋格点、路径、时序、验证、回退 | 1 |
| 2B | Legged | 高度区间、机体格点、积空间走廊、路径、时序、有限保证 | 1 |
| 3 | Hopper | 着陆域图、纯弹道、飞行管、姿态和落点认证 | 1 |
| 4 | Integration & Performance | 分派、bundle、承诺状态机、codec、适配、系统测试、P95 | 2A、2B、3 |

```text
Interface Schema
      ↓
Contracts & Core
   ↙    ↓     ↘
Wheel  Legged  Hopper
   ↘    ↓     ↙
Integration & Performance
```

---

### Task 1: 冻结 schema 与执行基线

**Files:**
- Verify: `path-planner/schemas/v3/**`
- Verify: `docs/superpowers/specs/2026-07-28-multiplatform-path-planner-v3-interface-schema.md`
- Verify: `docs/superpowers/plans/2026-07-28-multiplatform-planner-v3-contracts-core.md`
- Verify: `docs/superpowers/plans/2026-07-28-multiplatform-planner-v3-wheel.md`
- Verify: `docs/superpowers/plans/2026-07-28-multiplatform-planner-v3-legged.md`
- Verify: `docs/superpowers/plans/2026-07-28-multiplatform-planner-v3-hopper.md`
- Verify: `docs/superpowers/plans/2026-07-28-multiplatform-planner-v3-integration-performance.md`

**Interfaces:**
- Consumes: 已确认 v3 设计规范。
- Produces: 后续实现不得自行改名的 schema version、C++ 类型和分卷边界。

- [ ] **Step 1: 校验全部 JSON 文档语法**

Run:

```powershell
@'
from pathlib import Path
import json

for path in sorted(Path("path-planner/schemas/v3").rglob("*.json")):
    json.loads(path.read_text(encoding="utf-8"))
    print(path.as_posix())
'@ | python -
```

Expected: 列出全部 schema 且退出码为 0。

- [ ] **Step 2: 校验本地 `$ref` 和稳定 `$id` 唯一**

Run:

```powershell
@'
from pathlib import Path
import json

root = Path("path-planner/schemas/v3")
documents = {
    path: json.loads(path.read_text(encoding="utf-8"))
    for path in root.rglob("*.json")
}
by_id = {}
for path, doc in documents.items():
    assert doc["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert doc["$id"] not in by_id, f"duplicate $id: {doc['$id']}"
    by_id[doc["$id"]] = doc

def resolve_pointer(document, pointer):
    current = document
    if not pointer:
        return current
    assert pointer.startswith("/"), pointer
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        current = current[int(token)] if isinstance(current, list) else current[token]
    return current

ref_count = 0
def walk(value, owner):
    global ref_count
    if isinstance(value, dict):
        if "$ref" in value:
            ref_count += 1
            base, marker, fragment = value["$ref"].partition("#")
            target = owner if not base else by_id[base]
            resolve_pointer(target, fragment if marker else "")
        for child in value.values():
            walk(child, owner)
    elif isinstance(value, list):
        for child in value:
            walk(child, owner)

for document in documents.values():
    walk(document, document)
print(f"validated schemas={len(documents)} refs={ref_count}")
'@ | python -
```

Expected: 输出 schema 与已解析 `$ref` 数量且退出码为 0；所有 URN 和 JSON Pointer 都在本地闭合。

- [ ] **Step 3: 检查禁止语义**

Run:

```powershell
@'
from pathlib import Path
import json

root = Path("path-planner/schemas/v3")
documents = {
    path.name: json.loads(path.read_text(encoding="utf-8"))
    for path in root.rglob("*.json")
}

keys = set()
strings = set()
def walk(value):
    if isinstance(value, dict):
        keys.update(value.keys())
        for child in value.values():
            walk(child)
    elif isinstance(value, list):
        for child in value:
            walk(child)
    elif isinstance(value, str):
        strings.add(value)

for document in documents.values():
    walk(document)

assert not {"deadline", "timeout", "remaining_wall_time"} & keys
assert "SAFE_PARTIAL" not in strings
assert "flight_corridor_3d" not in strings
legged = documents["legged.schema.json"]
assert (
    legged["properties"]["footstep_feasibility_guaranteed"]["const"]
    is False
)
print("validated forbidden runtime semantics")
'@ | python -
```

Expected: 输出 `validated forbidden runtime semantics`；基准阈值仍只通过 `p95_latency_target_ns` 表达。

- [ ] **Step 4: 记录基线提交**

```powershell
git -C path-planner add schemas/v3
git -C path-planner commit -m "docs(v3): define multiplatform planner interface schemas"
git add path-planner docs/superpowers/specs/2026-07-28-multiplatform-path-planner-v3-interface-schema.md docs/superpowers/plans/2026-07-28-multiplatform-planner-v3-*.md
git commit -m "docs: add multiplatform planner v3 implementation roadmap"
```

---

### Task 2: 执行 Contracts & Core 分卷

**Files:**
- Read and execute: `docs/superpowers/plans/2026-07-28-multiplatform-planner-v3-contracts-core.md`

**Interfaces:**
- Consumes: Task 1 schema。
- Produces: `lpp_v3_contracts`、`lpp_v3_common`、共享 ARA*、走廊、QP、验证和缓存接口。

- [ ] **Step 1: 按 contracts/core 分卷逐任务执行**

使用独立子 agent 实现一个任务，主 agent 在每次提交前做规范和代码质量两轮 review。

- [ ] **Step 2: 运行 core completion gate**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --parallel
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R "^lpp_v3_(contracts|codec|map|cost|goal|search|corridor|optimization|cache|integration)\." --no-tests=error --output-on-failure
```

Expected: core 分卷全部测试通过。

- [ ] **Step 3: 审计依赖方向**

Run:

```powershell
$sharedDirs = @(
  "path-planner/cpp/include/lunar_path_planner/v3/contracts",
  "path-planner/cpp/include/lunar_path_planner/v3/codec",
  "path-planner/cpp/include/lunar_path_planner/v3/crypto",
  "path-planner/cpp/include/lunar_path_planner/v3/map",
  "path-planner/cpp/include/lunar_path_planner/v3/goal",
  "path-planner/cpp/include/lunar_path_planner/v3/search",
  "path-planner/cpp/include/lunar_path_planner/v3/corridor",
  "path-planner/cpp/include/lunar_path_planner/v3/cost",
  "path-planner/cpp/include/lunar_path_planner/v3/optimization",
  "path-planner/cpp/include/lunar_path_planner/v3/cache",
  "path-planner/cpp/src/contracts",
  "path-planner/cpp/src/codec",
  "path-planner/cpp/src/crypto",
  "path-planner/cpp/src/map",
  "path-planner/cpp/src/goal",
  "path-planner/cpp/src/search",
  "path-planner/cpp/src/corridor",
  "path-planner/cpp/src/cost",
  "path-planner/cpp/src/optimization",
  "path-planner/cpp/src/cache"
)
rg -n '#include\s*[<"]lunar_path_planner/v3/(wheel|legged|hopper)/' $sharedDirs
if ($LASTEXITCODE -eq 1) {
  Write-Output "validated shared-to-platform dependency direction"
  exit 0
}
exit $LASTEXITCODE
```

Expected: 输出 `validated shared-to-platform dependency direction`；共享核心不得依赖平台层。

---

### Task 3: 并行执行 Wheel 与 Legged 分卷

**Files:**
- Read and execute: `docs/superpowers/plans/2026-07-28-multiplatform-planner-v3-wheel.md`
- Read and execute: `docs/superpowers/plans/2026-07-28-multiplatform-planner-v3-legged.md`

**Interfaces:**
- Consumes: Task 2 共享接口。
- Produces: `WheelPlanner::Plan(...)` 与 `LeggedPlanner::Plan(...)`。

- [ ] **Step 1: 创建两个互不修改共享文件的执行队列**

Wheel 只修改 `wheel/**` 和 wheel 测试；Legged 只修改 `legged/**` 和 legged 测试。若发现必须修改共享接口，暂停两个队列，由主 agent 在 core 分卷单独提交接口变更。

- [ ] **Step 2: 完成 Wheel completion gate**

Run:

```powershell
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R "^lpp_v3_wheel\." --no-tests=error --output-on-failure
```

Expected: 全部通过。

- [ ] **Step 3: 完成 Legged completion gate**

Run:

```powershell
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R "^lpp_v3_legged\." --no-tests=error --output-on-failure
```

Expected: 全部通过。

- [ ] **Step 4: 验证平台互不依赖**

Run:

```powershell
rg -n "#include <lunar_path_planner/v3/legged/" path-planner/cpp/src/wheel path-planner/cpp/include/lunar_path_planner/v3/wheel
rg -n "#include <lunar_path_planner/v3/wheel/" path-planner/cpp/src/legged path-planner/cpp/include/lunar_path_planner/v3/legged
```

Expected: 两条命令均无匹配。

---

### Task 4: 执行 Hopper 分卷

**Files:**
- Read and execute: `docs/superpowers/plans/2026-07-28-multiplatform-planner-v3-hopper.md`

**Interfaces:**
- Consumes: Task 2 共享接口。
- Produces: `HopperPlanner::Plan(...)`、下一着陆范围、唯一 `JumpBoundary` 和完整物理认证。

- [ ] **Step 1: 按 hopper 分卷逐任务执行**

每次提交必须保留“未知或数值不确定即拒绝”和“只授权第一跳”两条硬边界。

- [ ] **Step 2: 运行 hopper completion gate**

Run:

```powershell
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug -R "^lpp_v3_hopper\." --no-tests=error --output-on-failure
```

Expected: 全部通过。

- [ ] **Step 3: 检查纯弹道和唯一命令**

Run:

```powershell
rg -n "in_flight_translation_control|select_point_inside_landing_region|redirect_jump" path-planner/cpp/include path-planner/cpp/src
```

Expected: 无可执行能力匹配。

---

### Task 5: 执行 Integration & Performance 分卷

**Files:**
- Read and execute: `docs/superpowers/plans/2026-07-28-multiplatform-planner-v3-integration-performance.md`

该分卷内部按 `Task 2 → Task 3 → Task 1 → Task 4…8` 执行，确保 bundle builder
和仲裁接口先于统一编排器存在。

**Interfaces:**
- Consumes: Tasks 2–4 的平台 planner。
- Produces: `PlannerV3`、bundle、codec、状态机、可选适配、系统测试和 benchmark report。

- [ ] **Step 1: 完成统一 API、bundle 和状态机**

执行集成分卷 Tasks 1–3，先通过所有 outcome/directive 和 jump commitment 测试。

- [ ] **Step 2: 完成 schema codec 与可选诊断适配**

执行集成分卷 Tasks 4–5；现有默认 adapter 保持不变。

- [ ] **Step 3: 完成系统测试和 benchmark**

执行集成分卷 Tasks 6–8，并生成：

```text
D:/xunce/out/path-planner-v3/benchmark-report.json
```

- [ ] **Step 4: 运行全量 completion gate**

Run:

```powershell
cmake --build D:/xunce/build/path-planner-v3/windows-msvc-debug --parallel
ctest --test-dir D:/xunce/build/path-planner-v3/windows-msvc-debug --no-tests=error --output-on-failure
$env:PYTHONPATH = "path-planner/src"
python -m pytest -q path-planner/tests/test_v3_schema.py
Push-Location path-planner
python -m pytest -m "not drake" -q
Pop-Location
```

Expected: 全部通过。

---

### Task 6: 最终规范覆盖审计

**Files:**
- Verify: `docs/superpowers/specs/2026-07-28-multiplatform-path-planner-v3-design.md`
- Verify: `docs/superpowers/specs/2026-07-28-multiplatform-path-planner-v3-interface-schema.md`
- Verify: `D:/xunce/out/path-planner-v3/benchmark-report.json`

**Interfaces:**
- Consumes: 完整实现与验收结果。
- Produces: 可进入部署前独立安全 review 的实现基线；不自动连接 executor。

- [ ] **Step 1: 对照设计规范逐节标记实现和测试**

生成本地 review 表，至少映射 17 个主章节到 C++ 文件、测试名和提交 hash。任何未覆盖条款阻止完成。

- [ ] **Step 2: 检查禁止项**

Run:

```powershell
rg -n "runtime_deadline|remaining_wall_time|SAFE_PARTIAL|flight_corridor_3d|footstep_feasibility_guaranteed\\s*=\\s*true" path-planner/cpp path-planner/schemas/v3
```

Expected: 无匹配。

- [ ] **Step 3: 检查 Git 范围和子模块状态**

Run:

```powershell
git -C path-planner status --short
git status --short
git diff --submodule=log -- path-planner
```

Expected: 不存在意外未提交实现文件；根仓库只记录已 review 的子模块提交和相关文档。

- [ ] **Step 4: 记录最终实现提交**

仅在全部 completion gate 通过后提交根仓库的子模块指针与最终文档；不得把 benchmark 是否达标误写为 executor 接入授权。

---

## Execution Handoff

推荐使用 Subagent-Driven 执行：

1. 每个 task 使用新 subagent。
2. 主 agent 先做规范符合性 review，再做代码质量 review。
3. 每个 task 通过自己的测试并提交后才进入下一 task。
4. Wheel 与 Legged 只有在不修改共享文件时可以并行。
5. Integration 必须等待三个平台分卷全部通过。
