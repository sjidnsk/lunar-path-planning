# 多平台能力感知路径规划 v2 Gate5A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变 v1 默认行为的前提下，为显式 opt-in 的 `path_planner.v2` 增加确定性月面同高弹道、未条件化二维高斯落区，以及明确保留六项能力缺口的 `HopperProfileV2` 结构审计。

**Architecture:** `path_planner.v2.ballistics` 只提供无 IO、无随机数、无 terrain truth 的纯数学类型与函数；`path_planner.v2.profiles` 冻结批准的 lunar simulation-proxy 参数，并把未批准参数保持为 `None`。本计划不创建 hopper provider/oracle，不接 API/validation/runner；Gate5B 在独立设计稿批准后再计划，因为 jump primitive、连续净空、多跳搜索、目标 theta、stop/energy 模型尚未冻结。

**Tech Stack:** Python 3.12、标准库 `dataclasses`/`math`、项目既有 `Cell`/`WorldPoint`/`FineGridGeometryV2`/canonical JSON、pytest。

## Global Constraints

- 工作树固定为 `D:/codex/worktrees/multiplatform-path-planner-v2`，父仓分支固定为 `codex/multiplatform-path-planner-v2`。
- 父仓前置提交固定为 `cf2a6a20b51870e7161267baec075184f05f6418`；nested `path-planner` 前置提交固定为 `7d5c4dfc8e2a374855e6d8b962b69466c3353265`。
- 已批准规格固定为 `docs/superpowers/specs/2026-07-18-multiplatform-path-planner-v2-gate5a-design.md`；任何实现歧义先回到该规格，不得自行扩大能力。
- 保护 `C:/Users/77634/.codex/worktrees/ca49/lunar-path-planning` Stage6 dirty 工作树：不读取、不修改、不在其中运行命令。
- 所有 nested Python/pytest 命令必须从本工作树的 `path-planner` 根运行；涉及子进程的测试还必须把 `PYTHONPATH` 精确绑定到本工作树的 `path-planner/src`，不得回退到 C 盘 editable 安装。
- v1 保持默认；v2 只能显式 opt-in。本计划不得修改 v1 `AStarPlanner`、CLI、`PathPlannerAdapter`、default policy 或 executor。
- 四项发布边界固定为 `publishes_checkpoint=false`、`replaces_default_policy=false`、`connects_real_executor=false`、`starts_online_canary=false`。
- synthetic terrain 只能称为 proxy；不得生成或宣称 `physical_obstacle_cells`。
- `max_traversable_slope_deg=30.0` 保持平台对齐硬边界；本计划不修改 wheel、legged、Hybrid A*、PPO、reward 或 network。
- Gate5A 生产范围只有 `path-planner/src/path_planner/v2/ballistics.py` 与 `path-planner/src/path_planner/v2/profiles.py`；测试范围只有 `path-planner/tests/test_v2_ballistics.py` 与 `path-planner/tests/test_v2_profiles.py`。
- 不修改 `contracts.py`、根 `path_planner.v2.__init__`、provider、oracle、API、validation、serialization、runner、config、registry 或正式 Gate evidence。
- Gate5A 不生成 `passed` evidence；正式 Gate5 仍受 `freeze_hopper_simulation_proxy_profile_parameters` 阻塞。
- 所有包含中文的文档继续以 UTF-8 保存；写后用 Python `read_text(encoding="utf-8")` 验证且不得包含 U+FFFD。
- nested 全量允许保留既有 17 项 `pydrake` skip，但不得增加 fail/error/skip；v2 子集不得新增 skip。
- PPO Stage1 的 13 项继承失败 allowlist 可保留；后续实际失败 nodeid 必须是该 allowlist 的子集，且不得新增 error 或 skip。既有失败消失不是 v2 回归。

---

## Spec Coverage Map

- 规格 §2–3（目标、非目标与文件范围）：由 `Global Constraints`、`File Structure` 和最终 staged-scope 审计锁定。
- 规格 §4–5（公共类型与弹道采样）：由 Task 1 的 RED/GREEN 数据合同、数值边界、样本上限与确定性测试覆盖。
- 规格 §6–7（一维正态质量与二维未条件化落区）：由 Task 2 的尾部稳定性、全局排序、最短前缀、越界质量保留、未见单格上界与资源上限测试覆盖。
- 规格 §8（Hopper Profile）：由 Task 3 的冻结常量、深层 base-profile 审计、六项显式 `None` 缺口和 test-only 标识测试覆盖。
- 规格 §9（Profile 结构审计）：由 Task 4 的 canonical missing 顺序、complete-only 语义、自洽 audit 值对象和 forged-object 测试覆盖。
- 规格 §10（连续安全边界）：本计划只保留 `minimum_clearance_m=None`，不实现 jump primitive、terrain truth、连续 capsule 净空或 landing feasibility；这些能力必须进入另行批准的 Gate5B 规格与计划。
- 规格 §11（错误处理与确定性）：贯穿 Task 1–4 的 exact-type、finite、deep trust-boundary、稳定错误消息、不可变返回值和 byte-stability 测试。
- 规格 §12（TDD 与验收矩阵）：由各任务 RED→GREEN 提交、Task 4 聚焦测试及 Task 5 v2/full regression 落地。
- 规格 §13（完成条件）：由 Task 5 的范围、禁区、D 盘临时目录、PPO allowlist、父仓 gitlink-only 集成和最终 clean-tree 审计覆盖。

---

## File Structure

- Create `path-planner/src/path_planner/v2/ballistics.py`：3D 起点/采样/落区 dataclass、同高弹道、稳定正态区间质量、全平面概率落区。
- Create `path-planner/tests/test_v2_ballistics.py`：纯数学、数值稳定、深层信任边界、资源上限与确定性测试。
- Modify `path-planner/src/path_planner/v2/profiles.py`：冻结 hopper lunar proxy 参数、nullable 六缺口、结构审计。
- Modify `path-planner/tests/test_v2_profiles.py`：hopper profile/audit 的 exact、forged-object、负向与结构完整性测试。
- Track `path-planner` gitlink in the parent repository only after all nested Gate5A tests pass.

Gate5B 不纳入本计划。它必须先另行冻结 `HopperJumpPrimitiveV2` replay、launch z 与 terrain elevation、连续 capsule 净空、landing reason 映射、heading/azimuth、multi-hop search、stop/energy model 及 API failure precedence。

---

### Task 1: Exact ballistic data contracts and same-height arc sampling

**Files:**
- Create: `path-planner/tests/test_v2_ballistics.py`
- Create: `path-planner/src/path_planner/v2/ballistics.py`

**Interfaces:**
- Consumes: no Gate5A implementation; only Python standard library.
- Produces: `MAX_BALLISTIC_SAMPLES_V2`, `BallisticStartV2`, `BallisticSampleV2`, `LandingCellMassV2`, and `sample_ballistic_arc(start, speed_mps, elevation_rad, azimuth_rad, g_mps2, dt_s) -> tuple[BallisticSampleV2, ...]`.
- Preserves: `PoseStateV2` remains `(x_m, y_m, heading_rad)` and is not imported into `ballistics.py`.

- [ ] **Step 1: Write the RED arc/data-contract test file**

Create `path-planner/tests/test_v2_ballistics.py` with this initial content:

```python
from dataclasses import FrozenInstanceError, fields
from math import nextafter, pi, sin

import pytest

from path_planner.core.models import Cell
from path_planner.v2.ballistics import (
    MAX_BALLISTIC_SAMPLES_V2,
    BallisticSampleV2,
    BallisticStartV2,
    LandingCellMassV2,
    sample_ballistic_arc,
)


def test_ballistic_dataclasses_freeze_exact_public_fields() -> None:
    assert tuple(field.name for field in fields(BallisticStartV2)) == (
        "x_m", "y_m", "z_m"
    )
    assert tuple(field.name for field in fields(BallisticSampleV2)) == (
        "time_s", "x_m", "y_m", "z_m"
    )
    assert tuple(field.name for field in fields(LandingCellMassV2)) == (
        "cell", "probability_mass", "in_bounds"
    )
    start = BallisticStartV2(-0.0, 1.0, 2.0)
    assert start.x_m == 0.0
    assert not hasattr(start, "__dict__")
    with pytest.raises(FrozenInstanceError):
        start.x_m = 1.0
    with pytest.raises(TypeError, match="cell.*exact Cell"):
        LandingCellMassV2(object(), 0.5, True)
    with pytest.raises(TypeError, match="in_bounds.*exact bool"):
        LandingCellMassV2(Cell(0, 0), 0.5, 1)


def test_sample_ballistic_arc_has_exact_endpoints_and_lunar_range() -> None:
    start = BallisticStartV2(10.0, -2.0, 4.0)
    samples = sample_ballistic_arc(start, 3.0, pi / 4.0, 0.0, 1.62, 0.7)
    flight_time = 2.0 * ((3.0 * sin(pi / 4.0)) / 1.62)

    assert type(samples) is tuple
    assert samples[0] == BallisticSampleV2(0.0, 10.0, -2.0, 4.0)
    assert samples[-1].time_s == flight_time
    assert samples[-1].x_m == pytest.approx(10.0 + 9.0 / 1.62)
    assert samples[-1].y_m == pytest.approx(-2.0, abs=1.0e-15)
    assert samples[-1].z_m == start.z_m
    assert all(left.time_s < right.time_s for left, right in zip(samples, samples[1:]))
    assert all(
        right.time_s - left.time_s <= nextafter(0.7, float("inf"))
        for left, right in zip(samples, samples[1:])
    )


def test_sample_ballistic_arc_shortens_only_final_interval() -> None:
    start = BallisticStartV2(0.0, 0.0, 0.0)
    samples = sample_ballistic_arc(start, 1.5, pi / 6.0, 0.0, 1.62, 0.4)
    interior = tuple(sample.time_s for sample in samples[:-1])
    assert interior == tuple(index * 0.4 for index in range(len(interior)))
    assert samples[-1].time_s - samples[-2].time_s <= nextafter(0.4, float("inf"))

    two_samples = sample_ballistic_arc(start, 1.5, pi / 6.0, 0.0, 1.62, 99.0)
    assert len(two_samples) == 2
    assert two_samples[0].time_s == 0.0
    assert two_samples[-1].z_m == 0.0


@pytest.mark.parametrize(
    ("azimuth", "expected_signs"),
    [
        (0.0, (1, 0)),
        (pi / 2.0, (0, 1)),
        (pi, (-1, 0)),
        (3.0 * pi / 2.0, (0, -1)),
    ],
)
def test_sample_ballistic_arc_rotates_cardinal_azimuths(
    azimuth: float,
    expected_signs: tuple[int, int],
) -> None:
    end = sample_ballistic_arc(
        BallisticStartV2(0.0, 0.0, 1.0), 2.0, pi / 4.0, azimuth, 1.62, 0.25
    )[-1]
    dx, dy = end.x_m, end.y_m
    if expected_signs[0] == 0:
        assert dx == pytest.approx(0.0, abs=1.0e-15)
    else:
        assert dx * expected_signs[0] > 0.0
    if expected_signs[1] == 0:
        assert dy == pytest.approx(0.0, abs=1.0e-15)
    else:
        assert dy * expected_signs[1] > 0.0


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("speed_mps", True, TypeError),
        ("speed_mps", 0.0, ValueError),
        ("speed_mps", float("nan"), ValueError),
        ("g_mps2", 0.0, ValueError),
        ("g_mps2", float("inf"), ValueError),
        ("dt_s", -1.0, ValueError),
        ("elevation_rad", 0.0, ValueError),
        ("elevation_rad", pi / 2.0, ValueError),
        ("azimuth_rad", float("-inf"), ValueError),
    ],
)
def test_sample_ballistic_arc_rejects_invalid_inputs(field, value, error) -> None:
    values = {
        "speed_mps": 3.0,
        "elevation_rad": pi / 4.0,
        "azimuth_rad": 0.0,
        "g_mps2": 1.62,
        "dt_s": 0.25,
    }
    values[field] = value
    with pytest.raises(error, match=field):
        sample_ballistic_arc(BallisticStartV2(0.0, 0.0, 0.0), **values)


def test_sample_ballistic_arc_checks_derived_limits_before_allocation() -> None:
    start = BallisticStartV2(0.0, 0.0, 0.0)
    assert MAX_BALLISTIC_SAMPLES_V2 == 100_000
    with pytest.raises(ValueError, match="sample_count"):
        sample_ballistic_arc(start, 3.0, pi / 4.0, 0.0, 1.62, 1.0e-12)
    with pytest.raises(ValueError, match="finite"):
        sample_ballistic_arc(start, 1.0e308, pi / 4.0, 0.0, 1.0e-308, 1.0)


def test_sample_ballistic_arc_reaudits_forged_exact_start() -> None:
    start = BallisticStartV2(0.0, 0.0, 0.0)
    object.__setattr__(start, "z_m", float("nan"))
    with pytest.raises(ValueError, match="z_m.*finite"):
        sample_ballistic_arc(start, 3.0, pi / 4.0, 0.0, 1.62, 0.25)
```

- [ ] **Step 2: Run the focused test to prove RED**

Run from `path-planner`:

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest -p no:cacheprovider tests/test_v2_ballistics.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'path_planner.v2.ballistics'`; no production file has been created yet.

- [ ] **Step 3: Commit the RED contract**

Run from `path-planner`:

```powershell
git add -- tests/test_v2_ballistics.py
git diff --cached --check
git -c user.name=Codex -c user.email=codex@local commit -m "test: freeze lunar ballistic arc contract"
```

Expected: the commit contains only `tests/test_v2_ballistics.py` and remains intentionally RED for the missing module.

- [ ] **Step 4: Implement exact dataclasses and stable same-height sampling**

Create `path-planner/src/path_planner/v2/ballistics.py` with this Task 1 content:

```python
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, cos, isfinite, pi, sin
from numbers import Real

from path_planner.core.models import Cell


MAX_BALLISTIC_SAMPLES_V2 = 100_000


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    try:
        normalized = float(value)
    except (OverflowError, RuntimeError, ValueError):
        raise ValueError(f"{name} must be finite") from None
    if not isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return 0.0 if normalized == 0.0 else normalized


def _positive_real(value: object, name: str) -> float:
    normalized = _finite_real(value, name)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _derived_finite(value: float, name: str) -> float:
    if type(value) is not float or not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return 0.0 if value == 0.0 else value


def _exact_cell(value: object) -> Cell:
    if type(value) is not Cell:
        raise TypeError("cell must be exact Cell")
    if type(value.x) is not int or type(value.y) is not int:
        raise TypeError("cell coordinates must be exact int values")
    return Cell(value.x, value.y)


@dataclass(frozen=True, slots=True)
class BallisticStartV2:
    x_m: float
    y_m: float
    z_m: float

    def __post_init__(self) -> None:
        for name in ("x_m", "y_m", "z_m"):
            object.__setattr__(self, name, _finite_real(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class BallisticSampleV2:
    time_s: float
    x_m: float
    y_m: float
    z_m: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_s", _finite_real(self.time_s, "time_s"))
        if self.time_s < 0.0:
            raise ValueError("time_s must be nonnegative")
        for name in ("x_m", "y_m", "z_m"):
            object.__setattr__(self, name, _finite_real(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class LandingCellMassV2:
    cell: Cell
    probability_mass: float
    in_bounds: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "cell", _exact_cell(self.cell))
        mass = _finite_real(self.probability_mass, "probability_mass")
        if not 0.0 < mass <= 1.0:
            raise ValueError("probability_mass must be in (0, 1]")
        object.__setattr__(self, "probability_mass", mass)
        if type(self.in_bounds) is not bool:
            raise TypeError("in_bounds must be exact bool")


def sample_ballistic_arc(
    start: BallisticStartV2,
    speed_mps: float,
    elevation_rad: float,
    azimuth_rad: float,
    g_mps2: float,
    dt_s: float,
) -> tuple[BallisticSampleV2, ...]:
    if type(start) is not BallisticStartV2:
        raise TypeError("start must be exact BallisticStartV2")
    audited_start = BallisticStartV2(start.x_m, start.y_m, start.z_m)
    speed = _positive_real(speed_mps, "speed_mps")
    elevation = _finite_real(elevation_rad, "elevation_rad")
    azimuth = _finite_real(azimuth_rad, "azimuth_rad")
    gravity = _positive_real(g_mps2, "g_mps2")
    dt = _positive_real(dt_s, "dt_s")
    if not 0.0 < elevation < pi / 2.0:
        raise ValueError("elevation_rad must be in (0, pi/2)")

    horizontal_speed = _derived_finite(speed * cos(elevation), "horizontal speed")
    vertical_speed = _derived_finite(speed * sin(elevation), "vertical speed")
    vx = _derived_finite(horizontal_speed * cos(azimuth), "x velocity")
    vy = _derived_finite(horizontal_speed * sin(azimuth), "y velocity")
    vertical_time_scale = _derived_finite(
        vertical_speed / gravity,
        "vertical speed / gravity",
    )
    flight_time = _derived_finite(2.0 * vertical_time_scale, "flight time")
    if flight_time <= 0.0:
        raise ValueError("flight time must be positive")
    interval_ratio = _derived_finite(flight_time / dt, "flight time / dt_s")
    if interval_ratio <= 0.0:
        raise ValueError("flight time / dt_s must be positive")
    if interval_ratio > float(MAX_BALLISTIC_SAMPLES_V2 - 1):
        raise ValueError(
            f"sample_count must not exceed {MAX_BALLISTIC_SAMPLES_V2}"
        )

    horizontal_dx = _derived_finite(vx * flight_time, "landing x displacement")
    horizontal_dy = _derived_finite(vy * flight_time, "landing y displacement")
    landing_x = _derived_finite(audited_start.x_m + horizontal_dx, "landing x_m")
    landing_y = _derived_finite(audited_start.y_m + horizontal_dy, "landing y_m")
    apex_height = _derived_finite(
        vertical_time_scale * (0.5 * vertical_speed),
        "apex height",
    )
    _derived_finite(audited_start.z_m + apex_height, "apex z_m")

    try:
        interval_count = ceil(interval_ratio)
    except (OverflowError, ValueError):
        raise ValueError("sample_count must be finite") from None
    sample_count = interval_count + 1
    if sample_count > MAX_BALLISTIC_SAMPLES_V2:
        raise ValueError(
            f"sample_count must not exceed {MAX_BALLISTIC_SAMPLES_V2}"
        )

    samples = [
        BallisticSampleV2(
            0.0,
            audited_start.x_m,
            audited_start.y_m,
            audited_start.z_m,
        )
    ]
    for index in range(1, interval_count):
        time_s = _derived_finite(float(index) * dt, "sample time_s")
        if not samples[-1].time_s < time_s < flight_time:
            raise ValueError("sample times must remain strictly interior")
        fraction = _derived_finite(time_s / flight_time, "sample time fraction")
        x_m = _derived_finite(
            audited_start.x_m + horizontal_dx * fraction,
            "sample x_m",
        )
        y_m = _derived_finite(
            audited_start.y_m + horizontal_dy * fraction,
            "sample y_m",
        )
        z_m = _derived_finite(
            audited_start.z_m
            + 4.0 * apex_height * fraction * (1.0 - fraction),
            "sample z_m",
        )
        samples.append(BallisticSampleV2(time_s, x_m, y_m, z_m))

    samples.append(
        BallisticSampleV2(
            flight_time,
            landing_x,
            landing_y,
            audited_start.z_m,
        )
    )
    return tuple(samples)
```

- [ ] **Step 5: Run Task 1 tests and the adjacent contracts**

Run from `path-planner`:

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest -p no:cacheprovider tests/test_v2_ballistics.py tests/test_v2_contracts.py tests/test_v2_serialization.py -q
```

Expected: exit code `0`; every selected test passes; no selected test is skipped.

- [ ] **Step 6: Commit the Task 1 implementation**

Run from `path-planner`:

```powershell
git add -- src/path_planner/v2/ballistics.py
git diff --cached --check
git -c user.name=Codex -c user.email=codex@local commit -m "feat: add deterministic lunar ballistic arc"
```

Expected: the commit contains only `src/path_planner/v2/ballistics.py`; nested status is clean and the Task 1 RED commit is now GREEN.

---
### Task 2: Stable normal mass and globally ordered unconditioned landing zone

**Files:**
- Modify: `path-planner/tests/test_v2_ballistics.py`
- Modify: `path-planner/src/path_planner/v2/ballistics.py`

**Interfaces:**
- Consumes: Task 1 dataclasses and `_finite_real`/`_derived_finite` validation helpers.
- Produces: `MAX_LANDING_ZONE_CANDIDATES_V2`, `normal_interval_mass(lo, hi, mean, sigma) -> float`, and `landing_zone_cells(mean_xy, sigma_m, probability_threshold, geometry) -> tuple[LandingCellMassV2, ...]`.
- Probability semantics: independent isotropic x/y Gaussian over the entire integer cell plane, raw cell mass, deterministic shortest global prefix, OOB retained, no clipping and no in-bounds renormalization.

- [ ] **Step 1: Extend the RED imports and landing-distribution tests**

Replace the import block at the top of `path-planner/tests/test_v2_ballistics.py` with:

```python
from dataclasses import FrozenInstanceError, fields
from math import fsum, nextafter, pi, sin

import pytest

import path_planner.v2.ballistics as ballistics_module
from path_planner.core.models import Cell, WorldPoint
from path_planner.v2.ballistics import (
    MAX_BALLISTIC_SAMPLES_V2,
    MAX_LANDING_ZONE_CANDIDATES_V2,
    BallisticSampleV2,
    BallisticStartV2,
    LandingCellMassV2,
    landing_zone_cells,
    normal_interval_mass,
    sample_ballistic_arc,
)
from path_planner.v2.serialization import canonical_json_bytes
from path_planner.v2.terrain import FineGridGeometryV2
```

Then append these tests:

```python
@pytest.mark.parametrize(
    ("lo", "hi", "expected"),
    [
        (-1.0, 1.0, 0.6826894921370859),
        (8.0, 9.0, 6.219831985865866e-16),
        (9.0, 10.0, 1.1285122074236006e-19),
    ],
)
def test_normal_interval_mass_is_stable_in_center_and_far_tail(
    lo: float,
    hi: float,
    expected: float,
) -> None:
    assert normal_interval_mass(lo, hi, 0.0, 1.0) == pytest.approx(
        expected, rel=1.0e-14, abs=0.0
    )


def test_normal_interval_mass_zero_width_symmetry_and_translation() -> None:
    assert normal_interval_mass(2.0, 2.0, 1.0, 0.5) == 0.0
    left = normal_interval_mass(-1.5, -0.5, -1.0, 0.25)
    right = normal_interval_mass(0.5, 1.5, 1.0, 0.25)
    translated = normal_interval_mass(8.5, 9.5, 9.0, 0.25)
    assert left == right == translated


@pytest.mark.parametrize(
    ("values", "error", "message"),
    [
        ((1.0, 0.0, 0.0, 1.0), ValueError, "lo.*hi"),
        ((0.0, 1.0, 0.0, 0.0), ValueError, "sigma"),
        ((0.0, 1.0, 0.0, True), TypeError, "sigma"),
        ((0.0, float("inf"), 0.0, 1.0), ValueError, "hi.*finite"),
    ],
)
def test_normal_interval_mass_rejects_invalid_contracts(values, error, message) -> None:
    with pytest.raises(error, match=message):
        normal_interval_mass(*values)


def _landing_key(item: LandingCellMassV2, geometry: FineGridGeometryV2, mean: WorldPoint):
    center_x = geometry.origin[0] + (item.cell.x + 0.5) * geometry.resolution_m
    center_y = geometry.origin[1] + (item.cell.y + 0.5) * geometry.resolution_m
    distance_sq = (center_x - mean.x) ** 2 + (center_y - mean.y) ** 2
    return (-item.probability_mass, distance_sq, item.cell.y, item.cell.x)


def test_landing_zone_is_shortest_global_raw_mass_prefix() -> None:
    geometry = FineGridGeometryV2(5, 5, origin=(-1.25, -1.25))
    mean = WorldPoint(0.0, 0.0)
    threshold = 0.99
    zone = landing_zone_cells(mean, 0.2, threshold, geometry)

    assert type(zone) is tuple
    assert zone == tuple(sorted(zone, key=lambda item: _landing_key(item, geometry, mean)))
    masses = tuple(item.probability_mass for item in zone)
    assert fsum(masses) >= threshold
    assert fsum(masses[:-1]) < threshold
    for item in zone:
        x_lo = geometry.origin[0] + item.cell.x * geometry.resolution_m
        y_lo = geometry.origin[1] + item.cell.y * geometry.resolution_m
        expected = normal_interval_mass(x_lo, x_lo + 0.5, mean.x, 0.2) * normal_interval_mass(
            y_lo, y_lo + 0.5, mean.y, 0.2
        )
        assert item.probability_mass == expected
        assert item.in_bounds is geometry.in_bounds(item.cell)


def test_landing_zone_boundary_ties_use_distance_y_x_order() -> None:
    geometry = FineGridGeometryV2(4, 4)
    zone = landing_zone_cells(WorldPoint(1.0, 1.0), 0.05, 0.99, geometry)
    assert tuple(item.cell for item in zone) == (
        Cell(1, 1),
        Cell(2, 1),
        Cell(1, 2),
        Cell(2, 2),
    )


def test_landing_zone_retains_oob_mass_without_renormalizing() -> None:
    geometry = FineGridGeometryV2(1, 1)
    zone = landing_zone_cells(WorldPoint(0.5, 0.5), 0.4, 0.99, geometry)
    in_bounds_mass = fsum(item.probability_mass for item in zone if item.in_bounds)
    assert any(item.in_bounds is False for item in zone)
    assert in_bounds_mass < 0.99
    assert fsum(item.probability_mass for item in zone) >= 0.99


def test_landing_zone_is_byte_stable_and_recomputes_disclosure_flags() -> None:
    geometry = FineGridGeometryV2(3, 3, origin=(-0.5, -0.5))
    first = landing_zone_cells(WorldPoint(0.25, 0.25), 0.17, 0.99, geometry)
    second = landing_zone_cells(WorldPoint(0.25, 0.25), 0.17, 0.99, geometry)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert tuple(item.in_bounds for item in first) == tuple(
        geometry.in_bounds(item.cell) for item in first
    )


def test_landing_zone_handles_legal_extremes_or_hits_public_cap(monkeypatch) -> None:
    geometry = FineGridGeometryV2(2, 2)
    one = landing_zone_cells(
        WorldPoint(0.25, 0.25),
        1.0e-6,
        nextafter(1.0, 0.0),
        geometry,
    )
    assert one == (LandingCellMassV2(Cell(0, 0), 1.0, True),)

    assert MAX_LANDING_ZONE_CANDIDATES_V2 == 1_000_000
    monkeypatch.setattr(ballistics_module, "MAX_LANDING_ZONE_CANDIDATES_V2", 9)
    with pytest.raises(ValueError, match="candidate.*9"):
        landing_zone_cells(WorldPoint(0.25, 0.25), 5.0, 0.99, geometry)


def test_landing_zone_reaudits_forged_exact_outer_objects() -> None:
    mean = WorldPoint(0.25, 0.25)
    object.__setattr__(mean, "x", float("nan"))
    with pytest.raises(ValueError, match="mean_xy.x.*finite"):
        landing_zone_cells(mean, 0.1, 0.99, FineGridGeometryV2(2, 2))

    geometry = FineGridGeometryV2(2, 2)
    object.__setattr__(geometry, "width", True)
    with pytest.raises(TypeError, match="geometry.width.*exact int"):
        landing_zone_cells(WorldPoint(0.25, 0.25), 0.1, 0.99, geometry)


@pytest.mark.parametrize(
    ("sigma", "threshold", "error"),
    [
        (0.0, 0.99, ValueError),
        (True, 0.99, TypeError),
        (0.1, 0.0, ValueError),
        (0.1, 1.0, ValueError),
        (0.1, float("nan"), ValueError),
    ],
)
def test_landing_zone_rejects_invalid_probability_inputs(sigma, threshold, error) -> None:
    with pytest.raises(error):
        landing_zone_cells(WorldPoint(0.25, 0.25), sigma, threshold, FineGridGeometryV2(2, 2))
```

- [ ] **Step 2: Run the new tests to prove RED**

Run from `path-planner`:

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest -p no:cacheprovider tests/test_v2_ballistics.py -q
```

Expected: collection fails only because `MAX_LANDING_ZONE_CANDIDATES_V2`, `normal_interval_mass`, and `landing_zone_cells` are absent; no Task 2 production change exists yet.

- [ ] **Step 3: Commit the RED probability contract**

Run from `path-planner`:

```powershell
git add -- tests/test_v2_ballistics.py
git diff --cached --check
git -c user.name=Codex -c user.email=codex@local commit -m "test: freeze lunar landing probability contract"
```

Expected: the commit changes only `tests/test_v2_ballistics.py`.

- [ ] **Step 4: Add stable normal helpers, deep geometry audit, and bounded global-prefix search**

In `path-planner/src/path_planner/v2/ballistics.py`, replace the math import and model imports with:

```python
from math import ceil, cos, erfc, erf, floor, fsum, isfinite, pi, sin, sqrt

from path_planner.core.models import Cell, WorldPoint
from path_planner.v2.terrain import FineGridGeometryV2
```

Add this constant beside `MAX_BALLISTIC_SAMPLES_V2`:

```python
MAX_LANDING_ZONE_CANDIDATES_V2 = 1_000_000
_SQRT_TWO = sqrt(2.0)
```

Append these exact helpers and public functions after `sample_ballistic_arc`:

```python
def normal_interval_mass(lo: float, hi: float, mean: float, sigma: float) -> float:
    lower = _finite_real(lo, "lo")
    upper = _finite_real(hi, "hi")
    center = _finite_real(mean, "mean")
    scale = _positive_real(sigma, "sigma")
    if lower > upper:
        raise ValueError("lo must be less than or equal to hi")
    if lower == upper:
        return 0.0
    lower_delta = _derived_finite(lower - center, "lo - mean")
    upper_delta = _derived_finite(upper - center, "hi - mean")
    lower_z = _derived_finite(lower_delta / scale, "standardized lo")
    upper_z = _derived_finite(upper_delta / scale, "standardized hi")
    if lower_z >= 0.0:
        mass = 0.5 * (
            erfc(lower_z / _SQRT_TWO) - erfc(upper_z / _SQRT_TWO)
        )
    elif upper_z <= 0.0:
        mass = 0.5 * (
            erfc(-upper_z / _SQRT_TWO) - erfc(-lower_z / _SQRT_TWO)
        )
    else:
        mass = 0.5 * (
            erf(upper_z / _SQRT_TWO) - erf(lower_z / _SQRT_TWO)
        )
    mass = _derived_finite(mass, "normal interval mass")
    return min(1.0, max(0.0, mass))


def _audited_world_point(value: object) -> WorldPoint:
    if type(value) is not WorldPoint:
        raise TypeError("mean_xy must be exact WorldPoint")
    return WorldPoint(
        _finite_real(value.x, "mean_xy.x"),
        _finite_real(value.y, "mean_xy.y"),
    )


def _audited_geometry(value: object) -> FineGridGeometryV2:
    if type(value) is not FineGridGeometryV2:
        raise TypeError("geometry must be exact FineGridGeometryV2")
    if type(value.width) is not int or value.width <= 0:
        raise TypeError("geometry.width must be an exact int greater than zero")
    if type(value.height) is not int or value.height <= 0:
        raise TypeError("geometry.height must be an exact int greater than zero")
    if type(value.origin) is not tuple or len(value.origin) != 2:
        raise TypeError("geometry.origin must be an exact two-item tuple")
    origin = (
        _finite_real(value.origin[0], "geometry.origin[0]"),
        _finite_real(value.origin[1], "geometry.origin[1]"),
    )
    if type(value.resolution_m) is not float:
        raise TypeError("geometry.resolution_m must be exact float")
    if value.resolution_m != 0.5:
        raise ValueError("geometry.resolution_m must be exactly 0.5")
    if type(value.frame_id) is not str or not value.frame_id.strip():
        raise TypeError("geometry.frame_id must be exact nonempty str")
    return FineGridGeometryV2(
        width=value.width,
        height=value.height,
        origin=origin,
        frame_id=value.frame_id,
        resolution_m=value.resolution_m,
    )


def _axis_boundary(origin: float, index: int, resolution: float, name: str) -> float:
    try:
        offset = float(index) * resolution
    except OverflowError:
        raise ValueError(f"{name} must be finite") from None
    return _derived_finite(origin + offset, name)


def _axis_interval_mass(
    index: int,
    origin: float,
    resolution: float,
    mean: float,
    sigma: float,
    axis: str,
) -> float:
    lower = _axis_boundary(origin, index, resolution, f"{axis} lower boundary")
    upper = _axis_boundary(origin, index + 1, resolution, f"{axis} upper boundary")
    if upper <= lower:
        raise ValueError(f"{axis} cell boundaries must remain representable")
    return normal_interval_mass(lower, upper, mean, sigma)


def _left_tail(boundary: float, mean: float, sigma: float, name: str) -> float:
    z = _derived_finite((boundary - mean) / sigma, name)
    return 0.5 * erfc(-z / _SQRT_TWO)


def _right_tail(boundary: float, mean: float, sigma: float, name: str) -> float:
    z = _derived_finite((boundary - mean) / sigma, name)
    return 0.5 * erfc(z / _SQRT_TWO)


def _shortest_prefix_length(masses: tuple[float, ...], threshold: float) -> int | None:
    if not masses or fsum(masses) < threshold:
        return None
    lower, upper = 1, len(masses)
    while lower < upper:
        middle = (lower + upper) // 2
        if fsum(masses[:middle]) >= threshold:
            upper = middle
        else:
            lower = middle + 1
    return lower


def landing_zone_cells(
    mean_xy: WorldPoint,
    sigma_m: float,
    probability_threshold: float,
    geometry: FineGridGeometryV2,
) -> tuple[LandingCellMassV2, ...]:
    mean = _audited_world_point(mean_xy)
    audited_geometry = _audited_geometry(geometry)
    sigma = _positive_real(sigma_m, "sigma_m")
    threshold = _finite_real(probability_threshold, "probability_threshold")
    if not 0.0 < threshold < 1.0:
        raise ValueError("probability_threshold must be in (0, 1)")
    cap = MAX_LANDING_ZONE_CANDIDATES_V2
    if type(cap) is not int or cap <= 0:
        raise ValueError("candidate cap must be an exact positive int")

    resolution = audited_geometry.resolution_m
    x_coordinate = _derived_finite(
        (mean.x - audited_geometry.origin[0]) / resolution,
        "mean x grid coordinate",
    )
    y_coordinate = _derived_finite(
        (mean.y - audited_geometry.origin[1]) / resolution,
        "mean y grid coordinate",
    )
    try:
        center_x, center_y = floor(x_coordinate), floor(y_coordinate)
    except (OverflowError, ValueError):
        raise ValueError("mean grid coordinates must be representable") from None

    max_x_mass = _axis_interval_mass(
        center_x, audited_geometry.origin[0], resolution, mean.x, sigma, "x"
    )
    max_y_mass = _axis_interval_mass(
        center_y, audited_geometry.origin[1], resolution, mean.y, sigma, "y"
    )
    evaluated: list[tuple[float, float, int, int, LandingCellMassV2]] = []
    radius = 0
    while True:
        side = 2 * radius + 1
        evaluated_count = side * side
        if evaluated_count > cap:
            raise ValueError(f"landing-zone candidate count exceeds candidate cap {cap}")

        for y_index in range(center_y - radius, center_y + radius + 1):
            for x_index in range(center_x - radius, center_x + radius + 1):
                if radius > 0 and max(
                    abs(x_index - center_x), abs(y_index - center_y)
                ) != radius:
                    continue
                x_mass = _axis_interval_mass(
                    x_index,
                    audited_geometry.origin[0],
                    resolution,
                    mean.x,
                    sigma,
                    "x",
                )
                y_mass = _axis_interval_mass(
                    y_index,
                    audited_geometry.origin[1],
                    resolution,
                    mean.y,
                    sigma,
                    "y",
                )
                mass = _derived_finite(x_mass * y_mass, "cell probability mass")
                x_lower = _axis_boundary(
                    audited_geometry.origin[0], x_index, resolution, "x lower boundary"
                )
                y_lower = _axis_boundary(
                    audited_geometry.origin[1], y_index, resolution, "y lower boundary"
                )
                center_world_x = _derived_finite(
                    x_lower + 0.5 * resolution, "cell center x"
                )
                center_world_y = _derived_finite(
                    y_lower + 0.5 * resolution, "cell center y"
                )
                dx = _derived_finite(center_world_x - mean.x, "cell center dx")
                dy = _derived_finite(center_world_y - mean.y, "cell center dy")
                distance_sq = _derived_finite(dx * dx + dy * dy, "cell distance squared")
                if mass > 0.0:
                    cell = Cell(x_index, y_index)
                    item = LandingCellMassV2(
                        cell=cell,
                        probability_mass=mass,
                        in_bounds=(
                            0 <= x_index < audited_geometry.width
                            and 0 <= y_index < audited_geometry.height
                        ),
                    )
                    evaluated.append((mass, distance_sq, y_index, x_index, item))

        evaluated.sort(key=lambda row: (-row[0], row[1], row[2], row[3]))
        masses = tuple(row[0] for row in evaluated)
        prefix_length = _shortest_prefix_length(masses, threshold)
        if prefix_length is not None:
            prefix = tuple(row[4] for row in evaluated[:prefix_length])
            last_mass = prefix[-1].probability_mass
            x_left = _axis_boundary(
                audited_geometry.origin[0], center_x - radius, resolution, "x tail left"
            )
            x_right = _axis_boundary(
                audited_geometry.origin[0], center_x + radius + 1, resolution, "x tail right"
            )
            y_bottom = _axis_boundary(
                audited_geometry.origin[1], center_y - radius, resolution, "y tail bottom"
            )
            y_top = _axis_boundary(
                audited_geometry.origin[1], center_y + radius + 1, resolution, "y tail top"
            )
            unseen_mass_upper_bound = max(
                _left_tail(x_left, mean.x, sigma, "left x tail") * max_y_mass,
                _right_tail(x_right, mean.x, sigma, "right x tail") * max_y_mass,
                _left_tail(y_bottom, mean.y, sigma, "lower y tail") * max_x_mass,
                _right_tail(y_top, mean.y, sigma, "upper y tail") * max_x_mass,
            )
            if unseen_mass_upper_bound < last_mass:
                if fsum(item.probability_mass for item in prefix[:-1]) >= threshold:
                    raise RuntimeError("landing-zone prefix is not shortest")
                return prefix
        radius += 1
```

- [ ] **Step 5: Run landing tests and deterministic hash-seed replay**

Run from `path-planner`:

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest -p no:cacheprovider tests/test_v2_ballistics.py tests/test_v2_terrain.py tests/test_v2_serialization.py -q
$env:PYTHONHASHSEED = "1"
$one = D:/conda_envs/lunar-explorer/python.exe -c "from path_planner.core.models import WorldPoint; from path_planner.v2.ballistics import landing_zone_cells; from path_planner.v2.serialization import canonical_json_bytes; from path_planner.v2.terrain import FineGridGeometryV2; print(canonical_json_bytes(landing_zone_cells(WorldPoint(0.25,0.25),0.17,0.99,FineGridGeometryV2(3,3,origin=(-0.5,-0.5)))).hex())"
$env:PYTHONHASHSEED = "777"
$two = D:/conda_envs/lunar-explorer/python.exe -c "from path_planner.core.models import WorldPoint; from path_planner.v2.ballistics import landing_zone_cells; from path_planner.v2.serialization import canonical_json_bytes; from path_planner.v2.terrain import FineGridGeometryV2; print(canonical_json_bytes(landing_zone_cells(WorldPoint(0.25,0.25),0.17,0.99,FineGridGeometryV2(3,3,origin=(-0.5,-0.5)))).hex())"
if ($one -cne $two) { throw "landing-zone bytes changed across hash seeds" }
```

Expected: pytest exits `0` with no skips; `$one` and `$two` are byte-for-byte equal.

- [ ] **Step 6: Commit the Task 2 implementation**

Run from `path-planner`:

```powershell
git add -- src/path_planner/v2/ballistics.py
git diff --cached --check
git -c user.name=Codex -c user.email=codex@local commit -m "feat: add deterministic lunar landing distribution"
```

Expected: the commit changes only `src/path_planner/v2/ballistics.py`; both Task 1 and Task 2 tests are GREEN.

---

### Task 3: Frozen Hopper simulation-proxy profile with explicit nullable gaps

**Files:**
- Modify: `path-planner/tests/test_v2_profiles.py`
- Modify: `path-planner/src/path_planner/v2/profiles.py`

**Interfaces:**
- Consumes: existing `PlatformProfileV2`, `PlatformKindV2.HOPPER`, exact profile helpers, and the approved Gate5A dataclass field order.
- Produces: `HOPPER_LUNAR_BALLISTIC_CAPABILITY_REVISION_V2`, `HOPPER_PROXY_PROFILE_INCOMPLETE_REASON_V2`, and `HopperProfileV2`.
- Does not produce: provider, registry entry, root export, supported stop/energy interpretation, PlanningOutcome, or formal profile values for the six nullable fields.

- [ ] **Step 1: Add RED Hopper profile imports, fixture, and frozen-field tests**

Change the first import in `path-planner/tests/test_v2_profiles.py` to:

```python
from dataclasses import FrozenInstanceError, fields
```

Extend the import from `path_planner.v2.profiles` in `path-planner/tests/test_v2_profiles.py` with:

```python
    HOPPER_LUNAR_BALLISTIC_CAPABILITY_REVISION_V2,
    HOPPER_PROXY_PROFILE_INCOMPLETE_REASON_V2,
    HopperProfileV2,
```

Add this fixture helper immediately after `_legged_platform()`:

```python
def _hopper_platform(**overrides) -> PlatformProfileV2:
    values = {
        "profile_id": "hopper-lunar-ballistic-proxy/v1",
        "platform_kind": PlatformKindV2.HOPPER,
        "capability_revision": HOPPER_LUNAR_BALLISTIC_CAPABILITY_REVISION_V2,
        "simulation_proxy": True,
        "max_traversable_slope_deg": 30.0,
        "goal_position_tolerance_m": 0.0,
        "goal_heading_tolerance_rad": 0.0,
    }
    values.update(overrides)
    return PlatformProfileV2(**values)
```

Append these tests at the end of the file:

```python
def test_hopper_profile_freezes_approved_proxy_defaults() -> None:
    hopper = HopperProfileV2(profile=_hopper_platform())
    assert tuple(field.name for field in fields(HopperProfileV2)) == (
        "profile",
        "gravity_mps2",
        "launch_speeds_mps",
        "launch_elevations_rad",
        "azimuth_direction_count",
        "landing_sigma_range_scale",
        "landing_sigma_offset_m",
        "max_landing_slope_deg",
        "landing_probability_threshold",
        "midcourse_correction_enabled",
        "inflight_observation_enabled",
        "body_envelope_radius_m",
        "launch_reference_height_m",
        "arc_clearance_margin_m",
        "landing_footprint_radius_m",
        "stop_condition",
        "energy_model",
    )
    assert HOPPER_LUNAR_BALLISTIC_CAPABILITY_REVISION_V2 == (
        "simulation_proxy_lunar_ballistic/v1"
    )
    assert HOPPER_PROXY_PROFILE_INCOMPLETE_REASON_V2 == (
        "hopper_proxy_profile_incomplete"
    )
    assert hopper.profile.platform_kind is PlatformKindV2.HOPPER
    assert hopper.profile.simulation_proxy is True
    assert hopper.profile.max_traversable_slope_deg == 30.0
    assert hopper.profile.goal_position_tolerance_m == 0.0
    assert hopper.profile.goal_heading_tolerance_rad == 0.0
    assert hopper.gravity_mps2 == 1.62
    assert hopper.launch_speeds_mps == (1.5, 2.0, 2.5, 3.0)
    assert hopper.launch_elevations_rad == (pi / 6.0, pi / 4.0, pi / 3.0)
    assert hopper.azimuth_direction_count == 16
    assert hopper.landing_sigma_range_scale == 0.05
    assert hopper.landing_sigma_offset_m == 0.05
    assert hopper.max_landing_slope_deg == 15.0
    assert hopper.landing_probability_threshold == 0.99
    assert hopper.midcourse_correction_enabled is False
    assert hopper.inflight_observation_enabled is False
    assert not hasattr(hopper, "__dict__")
    with pytest.raises(FrozenInstanceError):
        hopper.gravity_mps2 = 1.0


def test_hopper_profile_keeps_formal_capability_fields_explicitly_nullable() -> None:
    hopper = HopperProfileV2(profile=_hopper_platform())
    assert (
        hopper.body_envelope_radius_m,
        hopper.launch_reference_height_m,
        hopper.arc_clearance_margin_m,
        hopper.landing_footprint_radius_m,
        hopper.stop_condition,
        hopper.energy_model,
    ) == (None, None, None, None, None, None)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"platform_kind": PlatformKindV2.LEGGED}, "HOPPER"),
        ({"simulation_proxy": False}, "simulation_proxy"),
        ({"capability_revision": "simulation_proxy_lunar_ballistic/v2"}, "capability_revision"),
        ({"max_traversable_slope_deg": nextafter(30.0, float("inf"))}, "30.0"),
        ({"goal_position_tolerance_m": nextafter(0.0, 1.0)}, "goal_position_tolerance_m"),
        ({"goal_heading_tolerance_rad": nextafter(0.0, 1.0)}, "goal_heading_tolerance_rad"),
    ],
)
def test_hopper_profile_rejects_incompatible_base_profile(overrides, message) -> None:
    with pytest.raises(ValueError, match=message):
        HopperProfileV2(profile=_hopper_platform(**overrides))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("gravity_mps2", nextafter(1.62, float("inf"))),
        ("launch_speeds_mps", (1.5, 2.0, 2.5, nextafter(3.0, float("inf")))),
        ("launch_elevations_rad", (pi / 6.0, pi / 4.0, nextafter(pi / 3.0, float("inf")))),
        ("azimuth_direction_count", 15),
        ("landing_sigma_range_scale", nextafter(0.05, float("inf"))),
        ("landing_sigma_offset_m", nextafter(0.05, float("inf"))),
        ("max_landing_slope_deg", nextafter(15.0, float("inf"))),
        ("landing_probability_threshold", nextafter(0.99, float("inf"))),
        ("midcourse_correction_enabled", True),
        ("inflight_observation_enabled", True),
    ],
)
def test_hopper_profile_rejects_every_frozen_proxy_field_deviation(field, value) -> None:
    with pytest.raises((TypeError, ValueError), match=field):
        HopperProfileV2(profile=_hopper_platform(), **{field: value})


def test_hopper_profile_rejects_inexact_containers_elements_and_switches() -> None:
    class DerivedFloat(float):
        pass

    with pytest.raises(TypeError, match="gravity_mps2.*exact.*float"):
        HopperProfileV2(profile=_hopper_platform(), gravity_mps2=DerivedFloat(1.62))
    with pytest.raises(TypeError, match="launch_speeds_mps.*exact tuple"):
        HopperProfileV2(profile=_hopper_platform(), launch_speeds_mps=[1.5, 2.0, 2.5, 3.0])
    with pytest.raises(TypeError, match="launch_speeds_mps.*exact float"):
        HopperProfileV2(profile=_hopper_platform(), launch_speeds_mps=(1.5, 2.0, 2.5, 3))
    with pytest.raises(TypeError, match="azimuth_direction_count.*exact int"):
        HopperProfileV2(profile=_hopper_platform(), azimuth_direction_count=True)
    with pytest.raises(TypeError, match="midcourse_correction_enabled.*exact bool"):
        HopperProfileV2(profile=_hopper_platform(), midcourse_correction_enabled=0)


def test_hopper_profile_validates_explicit_formal_capability_fields() -> None:
    hopper = HopperProfileV2(
        profile=_hopper_platform(),
        body_envelope_radius_m=0.25,
        launch_reference_height_m=0.50,
        arc_clearance_margin_m=0.10,
        landing_footprint_radius_m=0.30,
        stop_condition="fixture_stop_proxy/v1",
        energy_model="fixture_energy_proxy/v1",
    )
    assert hopper.body_envelope_radius_m == 0.25
    assert hopper.stop_condition == "fixture_stop_proxy/v1"

    invalid = (
        ("body_envelope_radius_m", 0.0),
        ("launch_reference_height_m", 1),
        ("arc_clearance_margin_m", True),
        ("landing_footprint_radius_m", float("nan")),
        ("stop_condition", "unversioned"),
        ("energy_model", "fixture/v0"),
    )
    for field, value in invalid:
        with pytest.raises((TypeError, ValueError), match=field):
            HopperProfileV2(profile=_hopper_platform(), **{field: value})


def test_hopper_profile_reaudits_forged_exact_base_profile() -> None:
    class DerivedPlatformProfile(PlatformProfileV2):
        pass

    derived = DerivedPlatformProfile(
        profile_id="hopper-derived/v1",
        platform_kind=PlatformKindV2.HOPPER,
        capability_revision=HOPPER_LUNAR_BALLISTIC_CAPABILITY_REVISION_V2,
        simulation_proxy=True,
        max_traversable_slope_deg=30.0,
    )
    with pytest.raises(TypeError, match="exact PlatformProfileV2"):
        HopperProfileV2(profile=derived)

    profile = _hopper_platform()
    object.__setattr__(profile, "max_traversable_slope_deg", True)
    with pytest.raises(TypeError, match="max_traversable_slope_deg.*exact.*float"):
        HopperProfileV2(profile=profile)
```

- [ ] **Step 2: Run Hopper profile tests to prove RED**

Run from `path-planner`:

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest -p no:cacheprovider tests/test_v2_profiles.py -q
```

Expected: collection fails because the two constants and `HopperProfileV2` do not exist; all pre-existing profile tests remain unchanged.

- [ ] **Step 3: Commit the RED Hopper profile contract**

Run from `path-planner`:

```powershell
git add -- tests/test_v2_profiles.py
git diff --cached --check
git -c user.name=Codex -c user.email=codex@local commit -m "test: freeze hopper proxy profile contract"
```

Expected: the commit changes only `tests/test_v2_profiles.py`.

- [ ] **Step 4: Add exact Hopper profile validation helpers and dataclass**

Add these constants after `LEGGED_STATIC_CRAWL_CAPABILITY_REVISION_V2` in `path-planner/src/path_planner/v2/profiles.py`:

```python
HOPPER_LUNAR_BALLISTIC_CAPABILITY_REVISION_V2 = (
    "simulation_proxy_lunar_ballistic/v1"
)
HOPPER_PROXY_PROFILE_INCOMPLETE_REASON_V2 = "hopper_proxy_profile_incomplete"
```

Add these private helpers after `_exact_profile_float`:

```python
def _fixed_exact_float(value: object, name: str, expected: float) -> float:
    normalized = _exact_profile_float(value, name)
    if normalized != expected:
        raise ValueError(f"{name} must be exactly {expected}")
    return expected


def _fixed_exact_float_tuple(
    value: object,
    name: str,
    expected: tuple[float, ...],
) -> tuple[float, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be exact tuple")
    if any(type(item) is not float for item in value):
        raise TypeError(f"{name} elements must be exact float")
    if value != expected:
        raise ValueError(f"{name} must match the frozen proxy values")
    return expected


def _optional_positive_exact_float(value: object, name: str) -> float | None:
    if value is None:
        return None
    normalized = _exact_profile_float(value, name)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive when provided")
    return normalized


def _optional_versioned_proxy_id(value: object, name: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise TypeError(f"{name} must be exact str when provided")
    if not value or value.strip() != value or not re.search(r"/v[1-9][0-9]*$", value):
        raise ValueError(f"{name} must be a nonempty versioned proxy id")
    return value


def _reaudit_hopper_platform_profile(value: object) -> PlatformProfileV2:
    if type(value) is not PlatformProfileV2:
        raise TypeError("profile must be exact PlatformProfileV2")
    profile_id = _exact_profile_string(value.profile_id, "profile_id")
    capability_revision = _exact_profile_string(
        value.capability_revision, "capability_revision"
    )
    schema_version = _exact_profile_string(value.schema_version, "schema_version")
    if type(value.platform_kind) is not PlatformKindV2:
        raise TypeError("platform_kind must be exact PlatformKindV2")
    if type(value.simulation_proxy) is not bool:
        raise TypeError("simulation_proxy must be exact bool")
    audited = PlatformProfileV2(
        profile_id=profile_id,
        platform_kind=value.platform_kind,
        capability_revision=capability_revision,
        simulation_proxy=value.simulation_proxy,
        max_traversable_slope_deg=_exact_profile_float(
            value.max_traversable_slope_deg, "max_traversable_slope_deg"
        ),
        goal_position_tolerance_m=_exact_profile_float(
            value.goal_position_tolerance_m, "goal_position_tolerance_m"
        ),
        goal_heading_tolerance_rad=_exact_profile_float(
            value.goal_heading_tolerance_rad, "goal_heading_tolerance_rad"
        ),
        schema_version=schema_version,
    )
    if audited.platform_kind is not PlatformKindV2.HOPPER:
        raise ValueError("hopper profile requires PlatformKindV2.HOPPER")
    if audited.simulation_proxy is not True:
        raise ValueError("hopper profile requires simulation_proxy=True")
    if audited.capability_revision != HOPPER_LUNAR_BALLISTIC_CAPABILITY_REVISION_V2:
        raise ValueError("capability_revision must be the fixed lunar ballistic proxy")
    if audited.max_traversable_slope_deg != 30.0:
        raise ValueError("hopper profile slope boundary must be exactly 30.0")
    if audited.goal_position_tolerance_m != 0.0:
        raise ValueError("goal_position_tolerance_m must be exactly 0.0")
    if audited.goal_heading_tolerance_rad != 0.0:
        raise ValueError("goal_heading_tolerance_rad must be exactly 0.0")
    return audited
```

Insert this dataclass immediately before `PlatformProfileRegistryV2`:

```python
@dataclass(frozen=True, slots=True)
class HopperProfileV2:
    profile: PlatformProfileV2
    gravity_mps2: float = 1.62
    launch_speeds_mps: tuple[float, ...] = (1.5, 2.0, 2.5, 3.0)
    launch_elevations_rad: tuple[float, ...] = (pi / 6.0, pi / 4.0, pi / 3.0)
    azimuth_direction_count: int = 16
    landing_sigma_range_scale: float = 0.05
    landing_sigma_offset_m: float = 0.05
    max_landing_slope_deg: float = 15.0
    landing_probability_threshold: float = 0.99
    midcourse_correction_enabled: bool = False
    inflight_observation_enabled: bool = False
    body_envelope_radius_m: float | None = None
    launch_reference_height_m: float | None = None
    arc_clearance_margin_m: float | None = None
    landing_footprint_radius_m: float | None = None
    stop_condition: str | None = None
    energy_model: str | None = None

    def __post_init__(self) -> None:
        _reaudit_hopper_platform_profile(self.profile)
        frozen_floats = (
            ("gravity_mps2", 1.62),
            ("landing_sigma_range_scale", 0.05),
            ("landing_sigma_offset_m", 0.05),
            ("max_landing_slope_deg", 15.0),
            ("landing_probability_threshold", 0.99),
        )
        for name, expected in frozen_floats:
            object.__setattr__(
                self,
                name,
                _fixed_exact_float(getattr(self, name), name, expected),
            )
        object.__setattr__(
            self,
            "launch_speeds_mps",
            _fixed_exact_float_tuple(
                self.launch_speeds_mps,
                "launch_speeds_mps",
                (1.5, 2.0, 2.5, 3.0),
            ),
        )
        object.__setattr__(
            self,
            "launch_elevations_rad",
            _fixed_exact_float_tuple(
                self.launch_elevations_rad,
                "launch_elevations_rad",
                (pi / 6.0, pi / 4.0, pi / 3.0),
            ),
        )
        if type(self.azimuth_direction_count) is not int:
            raise TypeError("azimuth_direction_count must be exact int")
        if self.azimuth_direction_count != 16:
            raise ValueError("azimuth_direction_count must be exactly 16")
        for name in (
            "midcourse_correction_enabled",
            "inflight_observation_enabled",
        ):
            value = getattr(self, name)
            if type(value) is not bool:
                raise TypeError(f"{name} must be exact bool")
            if value is not False:
                raise ValueError(f"{name} must be exactly False")
        for name in (
            "body_envelope_radius_m",
            "launch_reference_height_m",
            "arc_clearance_margin_m",
            "landing_footprint_radius_m",
        ):
            object.__setattr__(
                self,
                name,
                _optional_positive_exact_float(getattr(self, name), name),
            )
        for name in ("stop_condition", "energy_model"):
            object.__setattr__(
                self,
                name,
                _optional_versioned_proxy_id(getattr(self, name), name),
            )
```

- [ ] **Step 5: Run all profile tests**

Run from `path-planner`:

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest -p no:cacheprovider tests/test_v2_profiles.py tests/test_v2_contracts.py -q
```

Expected: exit code `0`; all existing wheel/legged tests and new hopper tests pass; no selected test is skipped.

- [ ] **Step 6: Commit the Task 3 implementation**

Run from `path-planner`:

```powershell
git add -- src/path_planner/v2/profiles.py
git diff --cached --check
git -c user.name=Codex -c user.email=codex@local commit -m "feat: add explicit hopper proxy profile gaps"
```

Expected: the commit changes only `src/path_planner/v2/profiles.py`; profile tests are GREEN.

---

### Task 4: Self-consistent Hopper profile structure audit

**Files:**
- Modify: `path-planner/tests/test_v2_profiles.py`
- Modify: `path-planner/src/path_planner/v2/profiles.py`

**Interfaces:**
- Consumes: Task 3 `HopperProfileV2` and its exact deep validation.
- Produces: `HopperProfileAuditV2` and `audit_hopper_profile_v2(profile) -> HopperProfileAuditV2`.
- Meaning: `complete=True` proves only that all six fields are structurally present; it does not approve their values for Gate5B, register supported models, prove safety, or pass Gate5.

- [ ] **Step 1: Add RED structure-audit imports, fixture, and invariant tests**

Extend the profiles import with:

```python
    HopperProfileAuditV2,
    audit_hopper_profile_v2,
```

Add this helper after `_hopper_platform()`:

```python
def _complete_hopper(**overrides) -> HopperProfileV2:
    values = {
        "profile": _hopper_platform(),
        "body_envelope_radius_m": 0.25,
        "launch_reference_height_m": 0.50,
        "arc_clearance_margin_m": 0.10,
        "landing_footprint_radius_m": 0.30,
        "stop_condition": "fixture_stop_proxy/v1",
        "energy_model": "fixture_energy_proxy/v1",
    }
    values.update(overrides)
    return HopperProfileV2(**values)
```

Append these tests:

```python
HOPPER_MISSING_FIELDS = (
    "body_envelope_radius_m",
    "launch_reference_height_m",
    "arc_clearance_margin_m",
    "landing_footprint_radius_m",
    "stop_condition",
    "energy_model",
)


def test_hopper_profile_audit_reports_all_missing_fields_in_canonical_order() -> None:
    audit = audit_hopper_profile_v2(HopperProfileV2(profile=_hopper_platform()))
    assert tuple(field.name for field in fields(HopperProfileAuditV2)) == (
        "complete", "reason_code", "missing_fields"
    )
    assert audit == HopperProfileAuditV2(
        complete=False,
        reason_code=HOPPER_PROXY_PROFILE_INCOMPLETE_REASON_V2,
        missing_fields=HOPPER_MISSING_FIELDS,
    )
    assert not hasattr(audit, "__dict__")
    with pytest.raises(FrozenInstanceError):
        audit.complete = True


@pytest.mark.parametrize("missing_field", HOPPER_MISSING_FIELDS)
def test_hopper_profile_audit_reports_each_actual_missing_subset(missing_field) -> None:
    audit = audit_hopper_profile_v2(_complete_hopper(**{missing_field: None}))
    assert audit.complete is False
    assert audit.reason_code == HOPPER_PROXY_PROFILE_INCOMPLETE_REASON_V2
    assert audit.missing_fields == (missing_field,)


def test_hopper_profile_audit_reports_structural_completion_only() -> None:
    profile = _complete_hopper(
        stop_condition="unsupported_but_structurally_versioned/v7",
        energy_model="unsupported_but_structurally_versioned/v9",
    )
    assert audit_hopper_profile_v2(profile) == HopperProfileAuditV2(True, None, ())


def test_hopper_profile_audit_reaudits_forged_exact_profile() -> None:
    profile = _complete_hopper()
    object.__setattr__(profile, "gravity_mps2", nextafter(1.62, float("inf")))
    with pytest.raises(ValueError, match="gravity_mps2"):
        audit_hopper_profile_v2(profile)


@pytest.mark.parametrize(
    ("values", "error"),
    [
        ((1, None, ()), TypeError),
        ((True, "hopper_proxy_profile_incomplete", ()), ValueError),
        ((False, None, ("energy_model",)), ValueError),
        ((False, "hopper_proxy_profile_incomplete", ()), ValueError),
        ((False, "hopper_proxy_profile_incomplete", ["energy_model"]), TypeError),
        ((False, "hopper_proxy_profile_incomplete", ("unknown",)), ValueError),
        (
            (
                False,
                "hopper_proxy_profile_incomplete",
                ("energy_model", "body_envelope_radius_m"),
            ),
            ValueError,
        ),
    ],
)
def test_hopper_profile_audit_rejects_inconsistent_contracts(values, error) -> None:
    with pytest.raises(error):
        HopperProfileAuditV2(*values)


def test_hopper_profile_audit_requires_exact_hopper_profile() -> None:
    with pytest.raises(TypeError, match="exact HopperProfileV2"):
        audit_hopper_profile_v2(object())
```

- [ ] **Step 2: Run audit tests to prove RED**

Run from `path-planner`:

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest -p no:cacheprovider tests/test_v2_profiles.py -q
```

Expected: collection fails only because `HopperProfileAuditV2` and `audit_hopper_profile_v2` are absent; no Task 4 production change exists yet.

- [ ] **Step 3: Commit the RED audit contract**

Run from `path-planner`:

```powershell
git add -- tests/test_v2_profiles.py
git diff --cached --check
git -c user.name=Codex -c user.email=codex@local commit -m "test: freeze hopper profile audit contract"
```

Expected: the commit changes only `tests/test_v2_profiles.py`.

- [ ] **Step 4: Implement canonical missing-field order and audit invariants**

Add this tuple immediately after the Hopper constants in `path-planner/src/path_planner/v2/profiles.py`:

```python
_HOPPER_FORMAL_CAPABILITY_FIELDS_V2 = (
    "body_envelope_radius_m",
    "launch_reference_height_m",
    "arc_clearance_margin_m",
    "landing_footprint_radius_m",
    "stop_condition",
    "energy_model",
)
```

Insert this code after `HopperProfileV2` and before `PlatformProfileRegistryV2`:

```python
@dataclass(frozen=True, slots=True)
class HopperProfileAuditV2:
    complete: bool
    reason_code: str | None
    missing_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if type(self.complete) is not bool:
            raise TypeError("complete must be exact bool")
        if self.reason_code is not None and type(self.reason_code) is not str:
            raise TypeError("reason_code must be exact str or None")
        if type(self.missing_fields) is not tuple:
            raise TypeError("missing_fields must be exact tuple")
        if any(type(name) is not str for name in self.missing_fields):
            raise TypeError("missing_fields values must be exact str")
        if len(self.missing_fields) != len(set(self.missing_fields)):
            raise ValueError("missing_fields must not contain duplicates")
        try:
            canonical = tuple(
                name
                for name in _HOPPER_FORMAL_CAPABILITY_FIELDS_V2
                if name in self.missing_fields
            )
        except TypeError:
            raise TypeError("missing_fields must contain hashable exact str") from None
        if canonical != self.missing_fields:
            raise ValueError("missing_fields must use canonical fields and order")
        if self.complete:
            if self.reason_code is not None or self.missing_fields:
                raise ValueError("complete audit must have no reason or missing fields")
        elif (
            self.reason_code != HOPPER_PROXY_PROFILE_INCOMPLETE_REASON_V2
            or not self.missing_fields
        ):
            raise ValueError("incomplete audit must have fixed reason and missing fields")


def audit_hopper_profile_v2(profile: HopperProfileV2) -> HopperProfileAuditV2:
    if type(profile) is not HopperProfileV2:
        raise TypeError("profile must be exact HopperProfileV2")
    audited = HopperProfileV2(
        profile=profile.profile,
        gravity_mps2=profile.gravity_mps2,
        launch_speeds_mps=profile.launch_speeds_mps,
        launch_elevations_rad=profile.launch_elevations_rad,
        azimuth_direction_count=profile.azimuth_direction_count,
        landing_sigma_range_scale=profile.landing_sigma_range_scale,
        landing_sigma_offset_m=profile.landing_sigma_offset_m,
        max_landing_slope_deg=profile.max_landing_slope_deg,
        landing_probability_threshold=profile.landing_probability_threshold,
        midcourse_correction_enabled=profile.midcourse_correction_enabled,
        inflight_observation_enabled=profile.inflight_observation_enabled,
        body_envelope_radius_m=profile.body_envelope_radius_m,
        launch_reference_height_m=profile.launch_reference_height_m,
        arc_clearance_margin_m=profile.arc_clearance_margin_m,
        landing_footprint_radius_m=profile.landing_footprint_radius_m,
        stop_condition=profile.stop_condition,
        energy_model=profile.energy_model,
    )
    missing = tuple(
        name
        for name in _HOPPER_FORMAL_CAPABILITY_FIELDS_V2
        if getattr(audited, name) is None
    )
    if missing:
        return HopperProfileAuditV2(
            complete=False,
            reason_code=HOPPER_PROXY_PROFILE_INCOMPLETE_REASON_V2,
            missing_fields=missing,
        )
    return HopperProfileAuditV2(complete=True, reason_code=None, missing_fields=())
```

- [ ] **Step 5: Run the complete profile test file**

Run from `path-planner`:

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest -p no:cacheprovider tests/test_v2_profiles.py -q
```

Expected: exit code `0`; all wheel, legged, hopper profile, and audit cases pass; no test is skipped.

- [ ] **Step 6: Commit the Task 4 implementation**

Run from `path-planner`:

```powershell
git add -- src/path_planner/v2/profiles.py
git diff --cached --check
git -c user.name=Codex -c user.email=codex@local commit -m "feat: add fail-closed hopper profile audit"
```

Expected: the commit changes only `src/path_planner/v2/profiles.py`; every Gate5A RED contract is GREEN.

---

### Task 5: Gate5A scope audit, full regression, PPO allowlist, and parent gitlink integration

**Files:**
- Verify only: the four approved nested files.
- Modify in parent Git index: `path-planner` gitlink only.
- Do not create: `configs/xunce_path_v2_gate5_hopper_v1.json` or `D:/xunce/out/path_v2/g5`.

**Interfaces:**
- Consumes: Tasks 1–4 nested commits and the Gate4 baselines `1524 passed / 17 skipped`, `1368 v2 passed / 0 skipped`, root runner tests `304 passed`, and the PPO Stage1 13-nodeid failure allowlist.
- Produces: one clean parent commit that advances only the `path-planner` gitlink to the verified Gate5A nested HEAD.
- Completion claim: Gate5A foundation implemented; formal Gate5 remains unexecuted and cannot be called passed.

- [ ] **Step 1: Audit exact nested scope and repository identity before broad tests**

Run from the parent worktree:

```powershell
$expected = @(
  "src/path_planner/v2/ballistics.py",
  "src/path_planner/v2/profiles.py",
  "tests/test_v2_ballistics.py",
  "tests/test_v2_profiles.py"
)
$actual = git -C path-planner diff --name-only 7d5c4dfc8e2a374855e6d8b962b69466c3353265..HEAD
if ((Compare-Object ($expected | Sort-Object) ($actual | Sort-Object)).Count -ne 0) {
  throw "Gate5A nested scope differs from the approved four files"
}
git -C path-planner diff --check 7d5c4dfc8e2a374855e6d8b962b69466c3353265..HEAD
git -C path-planner status --short --branch
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
```

Expected: nested scope is exactly the four listed files; nested status is clean; parent branch is `codex/multiplatform-path-planner-v2`; no command reads the protected C-drive worktree.

- [ ] **Step 2: Run the Gate5A focused suite in isolated D-drive temp state**

Run from the parent worktree:

```powershell
$tempRoot = "D:/xunce/tmp/path_v2_g5a_regression"
New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
$env:TEMP = $tempRoot
$env:TMP = $tempRoot
$env:MPLCONFIGDIR = "$tempRoot/mpl"
Push-Location path-planner
try {
  $env:PYTHONPATH = (Resolve-Path "src").Path
  D:/conda_envs/lunar-explorer/python.exe -m pytest -p no:cacheprovider -q `
    tests/test_v2_ballistics.py `
    tests/test_v2_profiles.py `
    tests/test_v2_terrain.py `
    tests/test_v2_contracts.py `
    tests/test_v2_serialization.py `
    tests/test_core_models.py
  if ($LASTEXITCODE -ne 0) { throw "Gate5A focused suite failed" }
} finally {
  Pop-Location
}
```

Expected: exit code `0`; no failures, errors, or skips in the selected suite.

- [ ] **Step 3: Run nested v2 and full regression with frozen count deltas**

The reviewed implementation adds exactly 98 collected passing cases over the Gate4 baseline: all 51 cases in the new `test_v2_ballistics.py`, plus 47 added cases in `test_v2_profiles.py`. The extra cases beyond the original draft are review-driven missing-field, float-boundary, and bounded-prefix-work regressions. Run from the parent worktree:

```powershell
Push-Location path-planner
try {
  $env:PYTHONPATH = (Resolve-Path "src").Path
  $v2Targets = Get-ChildItem -LiteralPath tests -Filter "test_v2_*.py" |
    Sort-Object FullName |
    ForEach-Object { $_.FullName }
  D:/conda_envs/lunar-explorer/python.exe -m pytest -p no:cacheprovider -q $v2Targets
  if ($LASTEXITCODE -ne 0) { throw "nested v2 regression failed" }

  D:/conda_envs/lunar-explorer/python.exe -m pytest -p no:cacheprovider -q tests
  if ($LASTEXITCODE -ne 0) { throw "nested full regression failed" }
} finally {
  Pop-Location
}
```

Expected named-v2 result: `1442 passed`, `0 failed`, `0 errors`, `0 skipped`. Expected full result: `1622 passed`, `17 skipped`, `0 failed`, `0 errors`; every skip remains attributable to optional `pydrake`. Relative to Gate4, the full collected count moves from `1541` (`1524 passed + 17 skipped`) to `1639`, exactly the 98 reviewed additions.

If pytest collection count differs because a RED parameterization was corrected during implementation, update this plan's two deltas to the actual reviewed test design before accepting the run; never weaken the no-new-fail/error/skip rule.

- [ ] **Step 4: Prove Gate5A did not add integration or formal-evidence surfaces**

Run from the parent worktree:

```powershell
$forbidden = @(
  "src/path_planner/v2/__init__.py",
  "src/path_planner/v2/contracts.py",
  "src/path_planner/v2/api.py",
  "src/path_planner/v2/validation.py",
  "src/path_planner/v2/providers",
  "src/path_planner/v2/oracles",
  "src/path_planner/v2/serialization.py"
)
foreach ($path in $forbidden) {
  git -C path-planner diff --quiet 7d5c4dfc8e2a374855e6d8b962b69466c3353265..HEAD -- $path
  if ($LASTEXITCODE -ne 0) { throw "Gate5A changed forbidden nested surface: $path" }
}
if (Test-Path -LiteralPath configs/xunce_path_v2_gate5_hopper_v1.json) {
  throw "Gate5A must not create the Gate5 config"
}
if (Test-Path -LiteralPath D:/xunce/out/path_v2/g5) {
  throw "Gate5A must not create formal Gate5 evidence"
}
```

Expected: every forbidden diff is empty; Gate5 config and formal output root are absent.

- [ ] **Step 5: Integrate only the verified nested gitlink in the parent repository**

Run from the parent worktree:

```powershell
git status --porcelain=v1 --untracked-files=all
git add -- path-planner
$staged = @(git diff --cached --name-only)
if ($staged.Count -ne 1 -or $staged[0] -ne "path-planner") {
  throw "parent integration must stage only the path-planner gitlink"
}
git diff --cached --check
git -c user.name=Codex -c user.email=codex@local commit -m "feat: integrate gate5a lunar ballistics"
```

Expected: the parent commit changes one gitlink and contains no nested file content, runner, config, evidence, checkpoint, policy, executor, or canary change.

- [ ] **Step 6: Run parent runner-contract regression**

Run from the parent worktree:

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest -p no:cacheprovider -q `
  tests/test_xunce_path_v2_g0_baseline_and_isolation.py `
  tests/test_xunce_path_v2_gate_benchmark.py
```

Expected: `304 passed`, `0 failed`, `0 errors`, `0 skipped`; Gate5A has not changed either root test file.

- [ ] **Step 7: Re-run PPO Stage1 and prove its failure set does not expand**

Run from the parent worktree:

```powershell
$repo = (Resolve-Path ".").Path
$ppoRoot = "D:/xunce/tmp/path_v2_g5a_regression/ppo"
New-Item -ItemType Directory -Force -Path $ppoRoot | Out-Null
$env:PYTHONPATH = "$repo/src$([IO.Path]::PathSeparator)$repo/path-planner/src"
D:/conda_envs/lunar-explorer/python.exe -m pytest -p no:cacheprovider -q `
  tests/ppo_highres_frontier/test_stage1_smoke_env.py `
  --junitxml "$ppoRoot/stage1.junit.xml" `
  --basetemp "$ppoRoot/basetemp"
$ppoExit = $LASTEXITCODE
if ($ppoExit -notin @(0, 1)) { throw "PPO Stage1 returned an unexpected pytest exit code" }

@'
import json
import sys
from pathlib import Path

root = Path.cwd()
sys.path.insert(0, str(root / "scripts"))
from run_xunce_path_v2_g0_baseline_and_isolation import parse_junit

config = json.loads(
    (root / "configs/xunce_path_v2_g0_baseline_and_isolation_v1.json").read_text(
        encoding="utf-8"
    )
)
observed = parse_junit(Path("D:/xunce/tmp/path_v2_g5a_regression/ppo/stage1.junit.xml"))
expected = config["ppo_stage1"]["expected"]
allowlist = tuple(row["nodeid"] for row in config["ppo_stage1"]["known_failures"])
assert observed.tests == expected["tests"] == 56
assert observed.passed >= expected["passed"] == 43
assert observed.failures <= expected["failures"] == 13
assert observed.errors == expected["errors"] == 0
assert observed.skipped == expected["skipped"] == 0
assert len(allowlist) == len(set(allowlist)) == 13
assert set(observed.failed_nodeids).issubset(set(allowlist))
print("PPO Stage1 failure set stayed within the frozen 13-nodeid allowlist")
'@ | D:/conda_envs/lunar-explorer/python.exe -
```

Expected: pytest returns `0` or `1`; every remaining failure is allowlisted, and there are no new failure nodeids, errors, or skips. A smaller failure subset is reported as a baseline improvement, not rejected.

- [ ] **Step 8: Perform the final clean-tree and hard-boundary audit**

Run from the parent worktree:

```powershell
git status --short --branch
git -C path-planner status --short --branch
git rev-parse HEAD:path-planner
git -C path-planner rev-parse HEAD
rg -n "publishes_checkpoint|replaces_default_policy|connects_real_executor|starts_online_canary" `
  docs/superpowers/specs/2026-07-18-multiplatform-path-planner-v2-gate5a-design.md `
  docs/superpowers/plans/2026-07-19-multiplatform-path-planner-v2-gate5a.md
```

Expected: parent and nested worktrees are clean; parent gitlink equals nested HEAD; hard boundaries remain explicit; no formal Gate5 route or pass claim exists.

---

## Gate5A Completion Checklist

- [ ] Exact approved four-file nested scope only.
- [ ] RED evidence observed before each production surface exists.
- [ ] All returned containers are exact tuples and all public inputs are deeply re-audited.
- [ ] Arc endpoints are exact, sample allocation is bounded, and all derived values remain finite.
- [ ] Far-tail normal interval mass survives without direct CDF cancellation.
- [ ] Landing zone is a deterministic global shortest raw-mass prefix with OOB cells retained.
- [ ] Hopper frozen proxy fields cannot drift; all six unapproved fields remain nullable by default.
- [ ] Audit structural completion is explicitly weaker than provider support or safety approval.
- [ ] Nested focused/v2/full regressions pass with no enlarged failure or skip set.
- [ ] Parent runner-contract tests remain at 304 passed.
- [ ] PPO Stage1 actual failure nodeids remain a subset of the frozen 13-nodeid allowlist, with no errors or skips.
- [ ] Parent commit changes only the nested gitlink.
- [ ] No Gate5 config, formal artifact, checkpoint publication, default-policy replacement, executor connection, or canary start.
- [ ] Parent and nested worktrees are clean.

After this checklist passes, begin a separate `superpowers:brainstorming` cycle for Gate5B. Do not infer the missing jump/search/stop/energy semantics from Gate5A test fixtures.
