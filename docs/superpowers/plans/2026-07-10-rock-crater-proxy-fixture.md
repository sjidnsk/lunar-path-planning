# Rock and Crater Proxy Fixture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the abstract Smoke obstacle layout with a deterministic, auditable, spatially scattered rock-and-crater high-resolution proxy while preserving the Stage 1 environment, LOS, safety, leakage, artifact, and approval contracts.

**Architecture:** Add a focused `terrain_proxy.py` module that owns deterministic catalog generation, placement, morphology rasterization, derived slope/traversability, and hashes. `ScenarioSource` composes that bundle with the neutral prior and exposes immutable aggregate provenance; the environment, workflow, and gate bind those aggregates without exposing catalog or truth to the policy. Existing 2D DDA LOS remains unchanged: rock hard cores and crater cells above 30 degrees stop rays, while continuous 3D height-profile occlusion remains out of scope.

**Tech Stack:** Python 3.12, NumPy PCG64, Pydantic v2, pytest, existing `ArtifactStore` canonical JSON and Stage 1 DDA/coverage/planner code.

## Global Constraints

- Normative design addendum: `docs/superpowers/specs/2026-07-10-rock-crater-proxy-fixture-design-addendum.md`.
- Original algorithm baseline remains `docs/superpowers/specs/2026-07-09-ppo-highres-frontier-map-exploration-design.md`.
- Keep `synthetic_source_kind=synthetic_terrain_obstacle_proxy/v1` and `physical_obstacle_cells_written=false` exactly.
- Craters never write `hard_obstacle`; rocks write only their `rho <= 0.85` synthetic core.
- LOS stays `two_dimensional_grid_line_of_sight/v1`: blockers are `hard_obstacle` or `slope_deg > 30.0`; first blocker visible, cells behind unknown.
- Smoke prior stays `constant_neutral/v1` and contains no rock/crater geometry.
- Vehicle/safety constants remain `0.4215874761`, `0.10`, `0.5215874761`, traversability `0.50`, slope `30.0`.
- No PPO/network/reward/action/planner semantic change; no Standard/Kilometer implementation in this amendment.
- No checkpoint publication, default policy replacement, executor connection, canary, push, or release.
- Large outputs go under `D:/xunce/out/ppo_frontier`; temporary Git index files stay under `D:/xunce/tmp/ppo_frontier/git-index`.
- Workers do not stage or commit. After machine and independent review pass, the main agent stops at `awaiting_human_approval`; only explicit Stage 1 approval authorizes the single commit `feat: add smoke exploration environment` and gate.

## File Map

- Create `src/lunar_exploration_ppo/env/terrain_proxy.py`: immutable catalog types, seed/count helpers, bounded placement, analytic morphology, layer derivation and hashes.
- Modify `src/lunar_exploration_ppo/configs/stage1.py`: freeze Smoke generator inputs and validate them.
- Modify `configs/ppo_highres_frontier_smoke_v1.json`: declare the frozen generator inputs.
- Modify `src/lunar_exploration_ppo/env/scenario.py`: replace abstract walls/rectangles with the generator bundle; bind catalog and layer hashes into scenario identity.
- Modify `src/lunar_exploration_ppo/env/env.py`: construct the default source from config and expose immutable aggregate morphology metadata.
- Modify `src/lunar_exploration_ppo/workflows/stage1.py`: include aggregate proxy provenance in `summary.json`.
- Modify `src/lunar_exploration_ppo/workflows/stage1_gate.py`: include aggregate proxy provenance in `scenario_data_hash`.
- Modify `src/lunar_exploration_ppo/workflows/stage1_source.py`: add the approved spec, plan, generator, and focused test to the exact reviewed path set.
- Create `tests/ppo_highres_frontier/test_stage1_terrain_proxy.py`: independent generator, morphology, placement, hash, occlusion, and leakage tests.
- Modify `tests/ppo_highres_frontier/test_stage1_smoke_env.py`: update the scenario identity and full Smoke expectations.
- Modify `tests/ppo_highres_frontier/test_stage1_smoke_env_r1.py`: retain approval/gate tests and add provenance drift coverage.
- Modify `.superpowers/sdd/task-2-report.md`: replace the superseded pre-rock/crater snapshot after final verification.

---

### Task 1: Freeze Configuration and Public Internal Types

**Files:**
- Create: `src/lunar_exploration_ppo/env/terrain_proxy.py`
- Modify: `src/lunar_exploration_ppo/configs/stage1.py`
- Modify: `configs/ppo_highres_frontier_smoke_v1.json`
- Test: `tests/ppo_highres_frontier/test_stage1_terrain_proxy.py`

**Interfaces:**
- Produces: `DensityProfile`, `TerrainProxySettings`, `RockProxy`, `CraterProxy`, `TerrainProxyCatalog`, `TerrainProxyBundle`, `TerrainProxyGenerationError`, `derive_proxy_seed`, and `scaled_count_bounds`.
- Consumes: existing `GridGeometry`, `PoseXYTheta`, and `ArtifactStore.canonical_json_bytes`.

- [ ] **Step 1: Write configuration and count-contract tests**

```python
def test_stage1_config_freezes_medium_rock_crater_proxy() -> None:
    config = load_stage1_config(STAGE1_CONFIG)
    assert config.proxy_generator_version == "procedural_lunar_rock_crater_proxy/v1"
    assert config.proxy_density_profile == "medium"
    assert config.proxy_base_seed == 20260710
    assert config.proxy_start_protection_m == 6.0
    assert config.proxy_max_scene_attempts == 64
    assert config.proxy_max_object_attempts == 256


@pytest.mark.parametrize(
    ("profile", "rock", "crater"),
    [("low", (10, 18), (2, 3)), ("medium", (24, 36), (4, 6)), ("high", (45, 64), (7, 10))],
)
def test_density_bounds_at_smoke_area(profile, rock, crater) -> None:
    assert scaled_count_bounds(profile, "rock", 4096.0) == rock
    assert scaled_count_bounds(profile, "crater", 4096.0) == crater
```

- [ ] **Step 2: Run RED tests**

Run:

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/ppo_highres_frontier/test_stage1_terrain_proxy.py -q -x
```

Expected: collection/import failure because `terrain_proxy.py` and the new config fields do not exist.

- [ ] **Step 3: Add exact config fields and immutable type definitions**

Add these exact fields to `Stage1Config` and JSON:

```python
proxy_generator_version: Literal["procedural_lunar_rock_crater_proxy/v1"]
proxy_density_profile: Literal["medium"]
proxy_base_seed: Literal[20260710]
proxy_start_protection_m: Literal[6.0]
proxy_max_scene_attempts: Literal[64]
proxy_max_object_attempts: Literal[256]
```

Define the core types with finite/range validation:

```python
DensityProfile = Literal["low", "medium", "high"]
ObjectKind = Literal["rock", "crater"]


@dataclass(frozen=True, slots=True)
class TerrainProxySettings:
    generator_version: str
    density_profile: DensityProfile
    base_seed: int
    start_protection_m: float
    max_scene_attempts: int
    max_object_attempts: int


@dataclass(frozen=True, slots=True)
class RockProxy:
    center_x_m: float
    center_y_m: float
    equivalent_radius_m: float
    semi_major_m: float
    semi_minor_m: float
    orientation_rad: float
    height_m: float


@dataclass(frozen=True, slots=True)
class CraterProxy:
    center_x_m: float
    center_y_m: float
    radius_m: float
    depth_m: float
    rim_height_m: float
    rim_width_ratio: float


@dataclass(frozen=True, slots=True)
class TerrainProxyCatalog:
    schema_version: str
    generator_version: str
    density_profile: DensityProfile
    seed_hex: str
    generation_attempt: int
    rocks: tuple[RockProxy, ...]
    craters: tuple[CraterProxy, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class TerrainProxyBundle:
    height: np.ndarray
    hard_obstacle: np.ndarray
    slope_deg: np.ndarray
    traversability: np.ndarray
    catalog: TerrainProxyCatalog
    layer_hashes: Mapping[str, str]
```

`TerrainProxyBundle.__post_init__` sets all arrays read-only and wraps `layer_hashes` in `MappingProxyType`.

- [ ] **Step 4: Implement deterministic seed and scaled bounds**

```python
def derive_proxy_seed(*, scenario_key: str, split: str, parent_roi: str, base_seed: int,
                      density_profile: DensityProfile) -> str:
    source = "|".join((GENERATOR_VERSION, scenario_key, split, parent_roi, str(base_seed), density_profile))
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:32]


def scaled_count_bounds(profile: DensityProfile, kind: ObjectKind, area_m2: float) -> tuple[int, int]:
    if not math.isfinite(area_m2) or area_m2 <= 0.0:
        raise ValueError("area_m2 must be positive and finite")
    base_low, base_high = DENSITY_COUNTS[profile][kind]
    scale = area_m2 / 4096.0
    low = max(1, math.ceil(base_low * scale))
    high = max(low, math.floor(base_high * scale + 0.5))
    return low, high
```

- [ ] **Step 5: Run focused GREEN tests**

Run the same command. Expected: all Task 1 tests pass; existing config extra-field and fixed-value drift tests remain green.

Do not stage or commit.

---

### Task 2: Implement Bounded Placement and Analytic Morphology

**Files:**
- Modify: `src/lunar_exploration_ppo/env/terrain_proxy.py`
- Test: `tests/ppo_highres_frontier/test_stage1_terrain_proxy.py`

**Interfaces:**
- Produces: `ProceduralTerrainProxyGenerator.generate`, `rock_height_at`, `rock_is_core`, and `crater_height_at`.
- Consumes: Task 1 types/settings and existing geometry cell-center conversion.

- [ ] **Step 1: Add morphology RED tests independent of rasterization helpers**

```python
def test_analytic_rock_profile_and_core() -> None:
    rock = RockProxy(10.0, 12.0, 1.0, 1.2, 1.0 / 1.2, 0.0, 0.8)
    assert rock_height_at(rock, 10.0, 12.0) == pytest.approx(0.8)
    assert rock_height_at(rock, 11.2, 12.0) == pytest.approx(0.0)
    assert rock_is_core(rock, 10.0, 12.0)
    assert not rock_is_core(rock, 11.1, 12.0)


def test_analytic_crater_has_bowl_rim_and_no_hard_obstacle_semantics() -> None:
    crater = CraterProxy(20.0, 20.0, 4.0, 0.8, 0.32, 0.2)
    center = crater_height_at(crater, 0.0)
    rim = crater_height_at(crater, 4.0)
    outside = crater_height_at(crater, 5.6)
    assert center < 0.0
    assert rim > 0.0
    assert outside == pytest.approx(0.0, abs=1e-12)
```

Add deterministic catalog, count-range, 6m start protection, map boundary, pairwise spacing, quadrant occupancy, and impossible-placement fail-closed tests. Compute distances in the test independently from catalog fields; do not call production validation helpers as the oracle.

- [ ] **Step 2: Run RED tests**

Expected: failures because analytic functions and generator are absent.

- [ ] **Step 3: Implement exact analytic functions**

```python
def rock_height_at(rock: RockProxy, x_m: float, y_m: float) -> float:
    dx, dy = x_m - rock.center_x_m, y_m - rock.center_y_m
    c, s = math.cos(rock.orientation_rad), math.sin(rock.orientation_rad)
    local_x, local_y = c * dx + s * dy, -s * dx + c * dy
    rho2 = (local_x / rock.semi_major_m) ** 2 + (local_y / rock.semi_minor_m) ** 2
    return rock.height_m * (1.0 - rho2) ** 2 if rho2 <= 1.0 else 0.0


def rock_is_core(rock: RockProxy, x_m: float, y_m: float) -> bool:
    dx, dy = x_m - rock.center_x_m, y_m - rock.center_y_m
    c, s = math.cos(rock.orientation_rad), math.sin(rock.orientation_rad)
    local_x, local_y = c * dx + s * dy, -s * dx + c * dy
    return math.hypot(local_x / rock.semi_major_m, local_y / rock.semi_minor_m) <= 0.85


def crater_height_at(crater: CraterProxy, distance_m: float) -> float:
    u = distance_m / crater.radius_m
    bowl = -crater.depth_m * (1.0 - u * u) ** 2 if u <= 1.0 else 0.0
    rim = crater.rim_height_m * math.exp(-((u - 1.0) / crater.rim_width_ratio) ** 2) if u <= 1.35 else 0.0
    return bowl + rim
```

- [ ] **Step 4: Implement deterministic target counts and bounded placement**

Use `np.random.Generator(np.random.PCG64(int(seed_hex, 16)))`. Sample rock/crater target counts once from the base seed. For each scene attempt, derive an attempt-specific SHA-256 seed, place craters first and rocks second, and keep the target counts unchanged. Each candidate must satisfy start protection, boundary, spacing, and quadrant limits. Exhausting 256 candidates restarts the whole scene; exhausting 64 scene attempts raises `TerrainProxyGenerationError` with stable reason `placement_exhausted`.

Sample exact object parameters from these closed ranges:

```text
rock equivalent radius: 0.5–1.5m
rock axis ratio q: 1.0–1.8
rock semi-major: r*sqrt(q)
rock semi-minor: r/sqrt(q)
rock orientation: [-pi, pi)
rock height: 0.2–1.0m
crater radius R: 2–6m
crater depth: 0.15R–0.30R
crater rim height: 0.05R–0.12R
crater rim width ratio: 0.12–0.25
```

Use a 6m start exclusion measured to each object's outer boundary, keep every outer boundary inside the map, use `1.10*(a1+a2)` rock spacing, `1.05*(1.35R1+1.35R2)` crater spacing, and forbid rock/crater footprint overlap. Medium/high scenes occupy all four quadrants with no quadrant above 40% of objects; low scenes occupy at least three quadrants.

The exact entry point is `ProceduralTerrainProxyGenerator(settings: TerrainProxySettings)` with method `generate(base_height: np.ndarray, geometry: GridGeometry, start_pose: PoseXYTheta, *, scenario_key: str, split: str, parent_roi: str) -> TerrainProxyBundle`. The constructor stores only the frozen settings; `generate` executes the bounded scene-attempt loop and returns the first fully validated bundle.

No branch may reduce target counts, relax distance, move the start pose, or inspect policy rollout results.

- [ ] **Step 5: Rasterize and derive layers**

For every cell center, add all crater and rock height deltas to a copied `float64` base height. Set `hard_obstacle[y, x]` only when `rock_is_core(rock, center_x_m, center_y_m)` is true. Derive:

```python
dz_dy, dz_dx = np.gradient(height, geometry.resolution_m, geometry.resolution_m, edge_order=1)
slope_deg = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
traversability = np.clip(1.0 - slope_deg / 60.0, 0.0, 1.0)
```

Create canonical catalog JSON with stable field order, then SHA-256 the catalog and contiguous bytes of every layer. Reject nonfinite arrays or a catalog object whose core cannot rasterize to at least one cell.

- [ ] **Step 6: Run focused GREEN and repeatability tests**

Run:

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/ppo_highres_frontier/test_stage1_terrain_proxy.py -q -x
```

Expected: all generator/morphology tests pass twice with identical hashes; no test writes outside pytest temporary paths.

Do not stage or commit.

---

### Task 3: Integrate Scenario, Occlusion, Provenance, Workflow, and Gate

**Files:**
- Modify: `src/lunar_exploration_ppo/env/scenario.py`
- Modify: `src/lunar_exploration_ppo/env/env.py`
- Modify: `src/lunar_exploration_ppo/workflows/stage1.py`
- Modify: `src/lunar_exploration_ppo/workflows/stage1_gate.py`
- Modify: `src/lunar_exploration_ppo/workflows/stage1_source.py`
- Modify: `tests/ppo_highres_frontier/test_stage1_smoke_env.py`
- Modify: `tests/ppo_highres_frontier/test_stage1_smoke_env_r1.py`
- Test: `tests/ppo_highres_frontier/test_stage1_terrain_proxy.py`

**Interfaces:**
- `ScenarioSource.load(key) -> ScenarioBundle` remains unchanged.
- `ScenarioSource(settings: TerrainProxySettings | None = None)` defaults to the frozen Smoke settings.
- `ScenarioBundle` gains immutable `proxy_catalog: TerrainProxyCatalog` and `proxy_layer_hashes: Mapping[str, str]`.
- `LunarExplorationEnv.proxy_morphology_metadata -> dict[str, object]` exposes only aggregates/hashes.
- `summary.json.proxy_morphology` and gate `scenario_data_hash` bind the same aggregate object.

- [ ] **Step 1: Add scenario/provenance/gate RED tests**

```python
def test_smoke_scenario_contains_auditable_rock_crater_proxy() -> None:
    scenario = ScenarioSource().load("smoke-v1")
    assert scenario.scenario_id == "smoke-v1/procedural-rock-crater/v1"
    assert 24 <= len(scenario.proxy_catalog.rocks) <= 36
    assert 4 <= len(scenario.proxy_catalog.craters) <= 6
    assert scenario.truth.provenance["proxy_generator_version"] == "procedural_lunar_rock_crater_proxy/v1"
    assert scenario.truth.provenance["physical_obstacle_cells_written"] is False
    assert np.count_nonzero(scenario.truth.hard_obstacle) > 0


def test_gate_scenario_identity_hash_binds_proxy_catalog() -> None:
    config = load_stage1_config(STAGE1_CONFIG)
    summary = minimal_valid_stage1_summary_with_proxy_morphology()
    first = Stage1GateBindingVerifier._scenario_identity_hash(summary, config)
    changed = json.loads(json.dumps(summary))
    changed["proxy_morphology"]["object_catalog_sha256"] = "0" * 64
    second = Stage1GateBindingVerifier._scenario_identity_hash(changed, config)
    assert first != second


def minimal_valid_stage1_summary_with_proxy_morphology() -> dict[str, object]:
    return {
        "scenario_id": "smoke-v1/procedural-rock-crater/v1",
        "scenario_hash": "1" * 64,
        "coverage_mask": {"sha256": "2" * 64},
        "proxy_morphology": {
            "proxy_generator_version": "procedural_lunar_rock_crater_proxy/v1",
            "density_profile": "medium",
            "generation_attempt": 0,
            "rock_count": 24,
            "crater_count": 4,
            "object_catalog_sha256": "3" * 64,
            "layer_hashes": {
                "height": "4" * 64,
                "hard_obstacle": "5" * 64,
                "slope": "6" * 64,
                "traversability": "7" * 64,
            },
        },
    }
```

Extract `_scenario_identity_hash(summary, config)` from the current inline gate hash construction so the test exercises the same canonical production path. Define `minimal_valid_stage1_summary_with_proxy_morphology()` in the test file with the exact schema emitted by the workflow. Add assertions that prior channels remain byte-identical when only hidden proxy truth changes, and that object catalog/provenance never appear in `PolicyObservation` or `ObservationBuilder` signatures.

- [ ] **Step 2: Add three independent LOS tests**

Construct small `TruthMap` fixtures directly, without production morphology helpers:

```python
@pytest.mark.parametrize("blocker_kind", ["rock_hard", "crater_steep"])
def test_proxy_blocker_is_visible_and_cell_behind_remains_unknown(blocker_kind) -> None:
    truth = independent_collinear_truth(blocker_kind=blocker_kind)
    state = ObservedMapState.empty(truth.geometry)
    SensorUpdater(fov_deg=0.0).reveal(truth, state, (east_facing_pose(truth),))
    assert state.observed_mask[4, 5]
    assert not state.observed_mask[4, 6]


def test_gentle_crater_does_not_block_2d_los() -> None:
    truth = independent_collinear_truth(blocker_kind="crater_gentle")
    state = ObservedMapState.empty(truth.geometry)
    SensorUpdater(fov_deg=0.0).reveal(truth, state, (east_facing_pose(truth),))
    assert state.observed_mask[4, 6]
```

The rock fixture sets `hard_obstacle=True`; steep crater sets only `slope_deg=31`; gentle crater sets `slope_deg=29` and nonzero height. This proves v1 2D LOS without relying on the generator.

- [ ] **Step 3: Replace the Smoke scenario composition**

Remove the line-wall, high-slope line, and low-traversability rectangle writes. Keep the smooth neutral background height, build `TerrainProxySettings` from frozen Stage 1 values, and call the generator. Set:

```python
scenario_id = "smoke-v1/procedural-rock-crater/v1"
```

Bind generator version, density, seed, attempt, counts, catalog hash, and layer hashes into immutable provenance. Scenario hash input order is: schema/version, geometry, start pose, catalog hash, height, hard obstacle, slope, traversability, and prior bytes.

- [ ] **Step 4: Bind config, workflow, and gate**

When no custom source is injected, `LunarExplorationEnv` constructs `ScenarioSource` with settings from its validated config. Its public aggregate property returns:

```python
{
    "proxy_generator_version": catalog.generator_version,
    "density_profile": catalog.density_profile,
    "generation_attempt": catalog.generation_attempt,
    "rock_count": len(catalog.rocks),
    "crater_count": len(catalog.craters),
    "object_catalog_sha256": catalog.sha256,
    "layer_hashes": dict(self._scenario.proxy_layer_hashes),
}
```

Write this object to `summary["proxy_morphology"]`. Add it verbatim to `Stage1GateBindingVerifier`'s canonical `scenario_record`; reject missing keys, extra keys, non-64-character hashes, count/profile drift, or a summary object inconsistent with the config.

Add these exact paths to `STAGE1_REVIEWED_PATHS`:

```text
docs/superpowers/plans/2026-07-10-ppo-highres-frontier-map-exploration.md
docs/superpowers/specs/2026-07-10-rock-crater-proxy-fixture-design-addendum.md
docs/superpowers/plans/2026-07-10-rock-crater-proxy-fixture.md
src/lunar_exploration_ppo/env/terrain_proxy.py
tests/ppo_highres_frontier/test_stage1_terrain_proxy.py
```

- [ ] **Step 5: Run integration GREEN tests**

Run in separate processes:

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/ppo_highres_frontier/test_stage1_terrain_proxy.py -q
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/ppo_highres_frontier/test_stage1_smoke_env.py -q
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/ppo_highres_frontier/test_stage1_smoke_env_r1.py -q
```

Expected: all pass; the existing approval-before-gate, actual-gate-file replay, no-candidate fail-closed, truth mutation, planner, reward, and terminal tests remain green.

Do not stage or commit.

---

### Task 4: Structural Gates, Full Regression, and Fresh Machine Evidence

**Files:**
- Verify only: all Stage 1 source/config/test files completed by Tasks 1–3.
- Modify: `.superpowers/sdd/task-2-report.md`
- Runtime output: `D:/xunce/out/ppo_frontier/$runId/s1/`, where `$runId` is created by Step 5.

**Interfaces:**
- Consumes the completed Stage 1 source tree.
- Produces a machine-passed D artifact root and canonical implementation report; no review, approval, gate, or checkpoint yet.

- [ ] **Step 1: Assert fixed Smoke structural gates without seed search**

Add tests for final safe start, `reachable_safe/safe_free >= 0.70`, nonzero exact coverable denominator, reset coverage below 0.99, and at least one default observed-only candidate. The test uses the single seed derived from `scenario_key=smoke-v1` and `proxy_base_seed=20260710`; do not sweep seeds or choose a seed from policy/baseline performance.

- [ ] **Step 2: Run the complete Stage 1 suites**

Run `test_stage1_terrain_proxy.py`, `test_stage1_smoke_env.py`, and `test_stage1_smoke_env_r1.py` in independent processes. Any failure gets root-cause debugging; do not hide it by seed replacement, count reduction, or weaker assertions.

- [ ] **Step 3: Run project regressions in separate processes**

```powershell
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/ppo_highres_frontier/test_foundation.py -q
D:/conda_envs/lunar-explorer/python.exe -m pytest tests/test_platform_stage_runner.py tests/test_bootstrap_env.py -q
```

Historical note: this plan originally required a separate `model-explorer` suite. That gitlink is retired and no longer belongs to the parent repository. Current verification runs the retained `path-planner` tests plus the Stage6, G1/G2/G3 and platform contracts.

- [ ] **Step 4: Run static, packaging, and source-identity checks**

Run compileall, `git diff --check` against the prospective tree, AST import-boundary checks, wheel build/install/import from a fresh D root, empty real index check, and prospective-tree computation. Only `integrations/path_planner_adapter.py` may directly import `path_planner`; no `model_explorer`, legacy `scripts/xunce_*`, or `sys.path` dependency is permitted.

- [ ] **Step 5: Produce a unique fresh 10-episode D run**

```powershell
$runId = "s1-rock-crater-final-$((Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ'))"
D:/conda_envs/lunar-explorer/python.exe scripts/run_ppo_stage1_smoke_env.py `
  --config configs/ppo_highres_frontier_smoke_v1.json `
  --run-id $runId
```

Expected: 10 finite episodes, existing `>=0.99` success predicate, exact seven machine artifact files, proxy morphology aggregates/hashes in summary, manifest hashes exact, and no `review.json`, `approval.json`, `gate.json`, `.pt`, `.pth`, or `.ckpt`.

- [ ] **Step 6: Rewrite the canonical implementer report**

Record RED/GREEN commands, final test counts/timings, prospective tree, wheel hashes, D run root, object counts, density, catalog/layer hashes, coverage as both raw fraction and percentage, 2D LOS limitation, and zero forbidden artifacts. Mark all pre-rock/crater R3 packages and D runs as superseded historical evidence.

Do not stage or commit.

---

### Task 5: Independent Review, Human Approval Stop, and Later Gate

**Files:**
- Generate ignored review package under `.superpowers/sdd/`.
- Runtime output after clean review: the Task 4 `$runId` stage root plus `review.json`.

**Interfaces:**
- Fresh reviewer produces both spec-compliance and code-quality verdicts.
- Main agent records the review and stops at `awaiting_human_approval`.

- [ ] **Step 1: Freeze a new review package**

Package the full working tree against Foundation commit `7378737a0d18ce6e4779e30605b557bd1ab25e6e`; include every tracked diff and untracked Stage 1 file. Record package SHA-256, byte/line count, exact path list, and prospective tree. The aborted pre-rock/crater R3 package is invalid and cannot be reused.

- [ ] **Step 2: Dispatch a fresh independent reviewer**

Require complete review of the original design, approved addendum, main plan, this plan, task brief, canonical report, and full diff package. The reviewer must close all historical R1/R2 findings again and inspect morphology, 2D occlusion, no crater hard mask, leakage, placement bounds, provenance/gate binding, runtime bounds, reward/terminal, and scope.

- [ ] **Step 3: Fix and re-review all blockers**

Any Critical or Important finding returns to a fresh implementer with a focused RED test, followed by a newly frozen package and independent re-review. Minor findings are recorded and dispositioned explicitly.

- [ ] **Step 4: Record review and stop for user approval**

After both verdicts are approved, write `review.json`, update state exactly:

```text
machine_passed
-> awaiting_independent_review
-> awaiting_human_approval
```

Verify the Stage root contains machine artifacts plus `review.json` only. Do not create `approval.json`, `gate.json`, commit, push, or enter Stage 2. Present evidence and ask for explicit `批准 Stage 1 Gate`.

- [ ] **Step 5: Execute only after future explicit Stage 1 approval**

The main agent stages only reviewed Stage 1 files, creates the single commit `feat: add smoke exploration environment`, writes the exact human approval artifact, transitions the verified gate through `approved -> next_stage`, and confirms the authorized next stage is `ppo_highres_frontier_stage2_observation_frontier/v1`. This step is not authorized by written-spec approval.
