# Multiscale Planner V3 Latency Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a small reproducible experiment that performs 810 measured `PlannerV3` calls across three map scales, three scenes and three platforms, then generates one scenario overview figure and one latency result figure.

**Architecture:** Extend the existing C++ system fixture with deterministic multiscale maps and platform-correct expected outcomes. A dedicated Release benchmark executable performs `Plan + PlanningResponse JSON serialization`, writes raw JSONL plus summaries, and evaluates the one-second threshold only after all calls finish. A separate Python/Matplotlib script reads those artifacts and renders static PNG figures; plotting never enters the planner hot path.

**Tech Stack:** C++20, MSVC, CMake, Ninja, nlohmann/json, GoogleTest, Python 3, Matplotlib, NumPy.

## Global Constraints

- Map extents are `12 m × 12 m`, `120 m × 80 m`, and `1200 m × 200 m`. Their certified task-grid resolutions are respectively `0.5 m`, `2 m`, and `5 m`.
- Scenes are exactly `G1_LOW_KNOWN`, `G1_MEDIUM_KNOWN`, and `G1_HIGH_FRONTIER`.
- The three scene profiles are derived from the frozen G1 `validation3` records `validation/scenario-0027`, `validation/scenario-0007`, and `validation/scenario-0116`. They must be labelled `g1_validation_reference_scaled/v1`, never exact G1 replay.
- G1-derived obstacle cells retain `synthetic_terrain_obstacle_proxy/v1` provenance and `physical_obstacle_cells_written=false`.
- Ground-platform representative planning distances are approximately `8 m`, `80 m`, and `500 m`; Hopper remains one pure-ballistic next hop at approximately `4 m`, `8 m`, and `10 m`.
- Ground fixture capabilities use certified macro primitives of `1 m`, `4 m`, and `10 m` at increasing scales; primitive duration must respect the unchanged speed limits and sweep validation checks the full segment.
- Each scale/scene/platform combination records one cold start, three unmeasured warmups, and thirty measured calls.
- The timed boundary is immediately before `PlannerV3::Plan()` through completion of `JsonCodec::EncodePlanningResponse()`.
- The `1000 ms` value is a post-run P95 metric only; it must not become a deadline, timeout, cutoff, cancellation condition, search budget, or result selector.
- `G1_LOW_KNOWN` and `G1_MEDIUM_KNOWN` expect `NEW_REFERENCE_READY`; ground `G1_HIGH_FRONTIER` expects `SAFE_FRONTIER_REFERENCE_READY`; Hopper `G1_HIGH_FRONTIER` expects a new reference to a certified landing region in known terrain.
- Preserve the Hopper pure-ballistic/no-inflight-translation contract and deterministic convex landing range.
- Do not change default policy, connect an executor, publish a checkpoint, or start canary execution.
- Build and experiment artifacts go under `D:/xunce/`; repository files contain only source, tests, fixtures and documentation.
- Do not modify or stage unrelated dirty files in the parent worktree.

---

### Task 1: Deterministic Multiscale Environment-Platform Scenarios

**Files:**
- Modify: `path-planner/cpp/tests/fixtures/system/planner_v3_system_fixture.hpp`
- Modify: `path-planner/cpp/tests/fixtures/system/planner_v3_system_fixture.cpp`
- Create: `path-planner/cpp/tests/unit/benchmark/multiscale_scenario_test.cpp`
- Modify: `path-planner/cpp/CMakeLists.txt`

**Interfaces:**
- Produces:

```cpp
enum class MapScale {
  kTenMeter,
  kHundredMeter,
  kThousandMeter,
};

enum class MapScenario {
  kOpenKnown,
  kKnownObstacleAtGoal,
  kUnknownGoalWithSafeFrontier,
  kDetour,
};

enum class ExpectedExperimentOutcome {
  kNewReferenceReady,
  kSafeFrontierReferenceReady,
};

struct ScenarioRegion final {
  enum class Kind {
    kHardObstacle,
    kUnknown,
    kHighSlope,
    kHighRoughness,
  };
  Kind kind{};
  std::vector<Vec2> polygon_xy_m;
};

struct ScenarioDescription final {
  MapScale scale{MapScale::kTenMeter};
  MapScenario scene{MapScenario::kOpenKnown};
  double width_m{};
  double height_m{};
  Vec2 start_xy_m;
  Vec2 goal_xy_m;
  double nominal_plan_distance_m{};
  ExpectedExperimentOutcome expected_outcome{
      ExpectedExperimentOutcome::kNewReferenceReady};
  std::vector<ScenarioRegion> regions;
};

struct SystemScenario final {
  PlanningRequest request;
  std::shared_ptr<const ContractObjectRegistry> registry;
  std::unique_ptr<SafeProjectionCache> projection_cache;
  ScenarioDescription description;
};

SystemScenario MakeMultiscaleSystemScenario(
    PlatformType platform_type,
    MapScale map_scale,
    MapScenario map_scenario);
```

- Existing `MakeSystemScenario(...)` and all current callers retain their current behavior unchanged.

- [x] **Step 1: Write failing scenario tests**

Add table-driven GoogleTests that instantiate all 27 experiment combinations and assert the literal map dimensions and scale-specific resolution, finite start/goal inside bounds, layer array sizes, frozen G1 reference provenance, non-empty proxy obstacle fields, and the expected-outcome matrix. Add one real planner call per combination and assert activation-ready outcome according to the matrix.

- [x] **Step 2: Run tests and observe the missing-type/overload failure**

Run:

```powershell
& 'D:/APP/CMake/bin/cmake.exe' --build 'D:/xunce/build/path-planner-v3/windows-msvc-debug' --target lpp_v3_multiscale_scenario_tests
```

Expected: build fails because `MapScale`, `ScenarioDescription`, and the new test target do not exist.

- [x] **Step 3: Implement deterministic maps and platform-correct goals**

Use rectangular dense maps with literal physical extents/resolution/cell dimensions `(12 m,12 m)/(0.5 m)/(24,24)`, `(120 m,80 m)/(2 m)/(60,40)`, and `(1200 m,200 m)/(5 m)/(240,40)`. Populate known mask, elevation, surface normal, roughness, hard obstacle, confidence, ESDF/static speed where already required by the fixture. Generate:

- `G1_LOW_KNOWN`: use the frozen low-density G1 validation reference seed and obstacle fraction on a fully known map.
- `G1_MEDIUM_KNOWN`: use the frozen medium-density G1 validation reference seed and obstacle fraction on a fully known map.
- `G1_HIGH_FRONTIER`: use the frozen high-density G1 validation reference seed and obstacle fraction in the known region, with an unknown suffix; Wheel/Legged stop at a reachable safe frontier and Hopper lands in a known certified range immediately before the suffix.

Use deterministic rock-core cells or small clusters, protect launch/start and target/landing cells, and write source scenario ID/hash, density profile, scale derivation and synthetic-proxy provenance into experiment artifacts. Do not copy G1 coverage denominators or claim that the scaled maps are exact G1 scenes.

Keep existing capability speed and acceleration limits. Use scale-specific certified ground primitives of `1 m`, `4 m`, and `10 m`, with duration scaled to the speed limits and full-segment sweep validation. Match ground lattice resolution to the task-grid resolution. Use start/goal separation tables that produce ground distances near `8/80/500 m` and Hopper distances near `4/8/10 m`.

- [x] **Step 4: Build and run only the new scenario tests**

Run the build command above, then:

```powershell
& 'D:/xunce/build/path-planner-v3/windows-msvc-debug/tests/lpp_v3_multiscale_scenario_tests.exe'
```

Expected: all table rows pass and every scenario produces its declared outcome.

- [x] **Step 5: Commit Task 1**

```powershell
git -C path-planner add -- cpp/tests/fixtures/system/planner_v3_system_fixture.hpp cpp/tests/fixtures/system/planner_v3_system_fixture.cpp cpp/tests/unit/benchmark/multiscale_scenario_test.cpp cpp/CMakeLists.txt
git -C path-planner commit -m "test: add multiscale planner scenarios"
```

---

### Task 2: Repeated End-to-End Benchmark and Artifact Writer

**Files:**
- Create: `path-planner/cpp/benchmarks/multiscale_experiment.hpp`
- Create: `path-planner/cpp/benchmarks/multiscale_experiment.cpp`
- Create: `path-planner/cpp/benchmarks/multiscale_planner_experiment.cpp`
- Create: `path-planner/cpp/tests/unit/benchmark/multiscale_experiment_test.cpp`
- Modify: `path-planner/cpp/CMakeLists.txt`
- Modify: `path-planner/cpp/docs/build-and-test.md`

**Interfaces:**
- Consumes: `MakeMultiscaleSystemScenario(platform, scale, scene)` and `ScenarioDescription`.
- Produces:

```cpp
struct MultiscaleExperimentConfig final {
  std::size_t warmup_count{3U};
  std::size_t measured_count_per_combination{30U};
  DurationNanoseconds p95_target{
      std::chrono::seconds{1}};
};

struct ExperimentRunPaths final {
  std::filesystem::path manifest_json;
  std::filesystem::path latency_samples_jsonl;
  std::filesystem::path summary_json;
  std::filesystem::path responses_jsonl;
};

Result<ExperimentRunPaths> RunMultiscaleExperiment(
    const MultiscaleExperimentConfig& config,
    const std::filesystem::path& output_root) noexcept;
```

- CLI:

```text
lpp_v3_multiscale_experiment --output-root D:/xunce/out/planner-v3-multiscale
```

- `latency-samples.jsonl` fields: `scale`, `scene`, `platform`, `sample_kind`, `sample_index`, `latency_ns`, `planning_outcome`, `reason_code`, `response_hash`.
- `summary.json` contains `scenario_results`, `scale_platform_results`, `platform_results`, `measured_call_count`, `all_outcomes_correct`, and `all_platform_p95_below_target`; each result includes `sample_count`, `correct_count`, `success_rate`, `p50_ns`, `p95_ns`, `maximum_ns`, and `p95_target_met`.
- `responses.jsonl` contains one representative entry per combination with compact scenario polygons, start/goal, ground reference polyline or Hopper ballistic samples, and Hopper landing polygon.

- [x] **Step 1: Write failing statistics and natural-completion tests**

Test nearest-rank P50/P95/max with hand-derived nanosecond literals. Use a small injected call function to prove all planned calls execute before `p95_target_met` is computed, even when a sample exceeds one second. Assert the default matrix expects exactly `810` measured samples and `27` cold-start samples.

- [x] **Step 2: Run tests and observe the missing runner failure**

Run:

```powershell
& 'D:/APP/CMake/bin/cmake.exe' --build 'D:/xunce/build/path-planner-v3/windows-msvc-debug' --target lpp_v3_multiscale_experiment_tests
```

Expected: build fails because the experiment runner and target do not exist.

- [x] **Step 3: Implement the runner and compact artifact encoding**

For every combination, construct a fresh scenario/planner/cache, time one cold call, perform three warmups, then time thirty calls. End each timed interval only after response JSON serialization. Extract representative platform geometry outside the timed interval. Compute scene/scale/platform statistics directly from raw measured samples; never average percentiles. Write UTF-8 JSON/JSONL after all calls finish.

- [x] **Step 4: Build and run the experiment tests**

Run:

```powershell
& 'D:/APP/CMake/bin/cmake.exe' --build 'D:/xunce/build/path-planner-v3/windows-msvc-debug' --target lpp_v3_multiscale_experiment_tests
& 'D:/xunce/build/path-planner-v3/windows-msvc-debug/tests/lpp_v3_multiscale_experiment_tests.exe'
```

Expected: all focused tests pass.

- [x] **Step 5: Build the Release experiment executable**

Run:

```powershell
& 'D:/APP/CMake/bin/cmake.exe' --build 'D:/xunce/build/path-planner-v3/windows-msvc-release' --target lpp_v3_multiscale_experiment
```

Expected: MSVC C++20 `/W4 /WX` build succeeds.

- [x] **Step 6: Commit Task 2**

```powershell
git -C path-planner add -- cpp/benchmarks/multiscale_experiment.hpp cpp/benchmarks/multiscale_experiment.cpp cpp/benchmarks/multiscale_planner_experiment.cpp cpp/tests/unit/benchmark/multiscale_experiment_test.cpp cpp/CMakeLists.txt cpp/docs/build-and-test.md
git -C path-planner commit -m "feat: add multiscale planner latency experiment"
```

---

### Task 3: Offline Scenario and Latency Figures

**Files:**
- Create: `path-planner/scripts/plot_planner_v3_multiscale.py`
- Create: `path-planner/tests/test_plot_planner_v3_multiscale.py`

**Interfaces:**
- CLI:

```text
python scripts/plot_planner_v3_multiscale.py \
  --input-root D:/xunce/out/planner-v3-multiscale \
  --output-dir D:/xunce/out/planner-v3-multiscale/figures
```

- Produces exactly:
  - `scenario-overview.png`
  - `latency-results.png`

- [x] **Step 1: Write failing plot tests**

Create a tiny literal artifact fixture in pytest temporary storage. Assert that the script parses all required summary/response fields and writes two non-empty PNG files. Add a negative test for a missing `measured_call_count`.

- [x] **Step 2: Run pytest and observe the missing-script failure**

Run:

```powershell
python -m pytest path-planner/tests/test_plot_planner_v3_multiscale.py -q
```

Expected: collection/import fails because the plotting module does not exist.

- [x] **Step 3: Implement two focused Matplotlib figures**

Use the `Agg` backend. `scenario-overview.png` is a `3 × 3` grid ordered by scale columns and G1-derived scene rows, with known terrain, unknown area, synthetic obstacle proxies, start, nominal target, safe frontier, three platform reference styles, Hopper ballistic arc and landing polygon. `latency-results.png` uses a log-millisecond axis, grouped by scale and platform, with P50 circles, P95 squares, maximum whiskers, a `1000 ms` horizontal line, and success-rate annotations.

- [x] **Step 4: Run the focused plot tests**

Run:

```powershell
python -m pytest path-planner/tests/test_plot_planner_v3_multiscale.py -q
```

Expected: all focused tests pass.

- [x] **Step 5: Commit Task 3**

```powershell
git -C path-planner add -- scripts/plot_planner_v3_multiscale.py tests/test_plot_planner_v3_multiscale.py
git -C path-planner commit -m "feat: plot multiscale planner experiment"
```

---

### Task 4: Formal Release Run and Result Handoff

**Files:**
- Create at runtime: `D:/xunce/out/planner-v3-multiscale/manifest.json`
- Create at runtime: `D:/xunce/out/planner-v3-multiscale/latency-samples.jsonl`
- Create at runtime: `D:/xunce/out/planner-v3-multiscale/summary.json`
- Create at runtime: `D:/xunce/out/planner-v3-multiscale/responses.jsonl`
- Create at runtime: `D:/xunce/out/planner-v3-multiscale/figures/scenario-overview.png`
- Create at runtime: `D:/xunce/out/planner-v3-multiscale/figures/latency-results.png`
- Create: `path-planner/cpp/docs/multiscale-latency-experiment.md`

**Interfaces:**
- Consumes the Task 2 executable and Task 3 plotting CLI.
- Produces the final numerical conclusion and links to both PNG figures.

- [x] **Step 1: Run the formal Release experiment**

```powershell
& 'D:/xunce/build/path-planner-v3/windows-msvc-release/benchmarks/lpp_v3_multiscale_experiment.exe' --output-root 'D:/xunce/out/planner-v3-multiscale'
```

Expected: exit code `0`, `measured_call_count=810`, and all four artifact files exist.

- [x] **Step 2: Render the two figures**

```powershell
python path-planner/scripts/plot_planner_v3_multiscale.py --input-root 'D:/xunce/out/planner-v3-multiscale' --output-dir 'D:/xunce/out/planner-v3-multiscale/figures'
```

Expected: both PNG files are non-empty.

- [x] **Step 3: Record only the direct experiment result**

Write `cpp/docs/multiscale-latency-experiment.md` with the exact hardware/build, 810-call matrix, G1 reference provenance, platform P50/P95/max, scale-platform P95 values, correctness rate, and links to the D-drive artifacts. State explicitly that one second was a post-run metric, that scaled maps are G1-derived rather than exact G1 replays, and that Hopper `G1_HIGH_FRONTIER` used a known certified landing range.

- [x] **Step 4: Perform the minimal final verification**

Run only:

```powershell
python -m pytest path-planner/tests/test_plot_planner_v3_multiscale.py -q
& 'D:/xunce/build/path-planner-v3/windows-msvc-release/tests/lpp_v3_multiscale_scenario_tests.exe'
& 'D:/xunce/build/path-planner-v3/windows-msvc-release/tests/lpp_v3_multiscale_experiment_tests.exe'
```

Read `summary.json` and assert `measured_call_count == 810`, `all_outcomes_correct == true`, and each platform `p95_target_met == true`. Check both PNG dimensions and inspect the two images once for visible axes, map features and non-clipped labels.

- [x] **Step 5: Commit the result note and parent submodule pointer**

```powershell
git -C path-planner add -- cpp/docs/multiscale-latency-experiment.md
git -C path-planner commit -m "docs: record multiscale latency experiment"
git add -- path-planner
git commit -m "chore: register multiscale planner experiment"
```
