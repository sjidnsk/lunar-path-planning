# Midterm G1 Scenario Source Materialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Materialize the existing Standard catalog, Stage6 exact coverage cache, policy-blind descriptors, reset evidence, reconstruction masks, and authorization binding required to execute the already-implemented Task 3 scenario freeze.

**Architecture:** A read-only source preparer rebuilds each Test/Unseen/Validation scenario from the canonical Standard catalog, binds its exact Stage6 coverage entry, performs only reset-time environment observation, and publishes a content-addressed source bundle. Task 3 then consumes that immutable bundle; neither preparer nor Task 3 reads a checkpoint, policy, reward, runtime result, or prior evaluation outcome.

**Tech Stack:** Python 3.11, NumPy, existing StandardScenarioCatalog/Factory, Stage6CoverageManifest, LunarExplorationEnv reset path, xunce artifact I/O, pytest.

## Global Constraints

- Candidate splits are exactly Test `150`, Unseen `64`, and Validation `150`; Train is excluded.
- Descriptor selection inputs are only catalog, static truth/cache, exact denominator, and reset-time observed state.
- Forbidden inputs are policy, checkpoint, reward, runtime, planner/evaluation result, and final coverage.
- The Stage6 catalog/hash and each exact coverage entry/key/mask/count must bind one-to-one.
- Source publication is data-first and completion-manifest-last; an existing complete root is immutable.
- All artifacts are UTF-8/canonical or deterministic NPZ and written beneath a new D-drive root.
- No Test/Unseen episode step, PPO inference, training, checkpoint write, or formal gate run occurs.

## Files

- Create `configs/xunce_mid_dual_scenario_sources_v1.json`
- Create `scripts/prepare_xunce_mid_dual_scenario_sources.py`
- Create `tests/test_xunce_mid_dual_scenario_sources.py`
- Modify `scripts/xunce_artifact_io.py`
  - add the missing binary `write_bytes()` primitive used for deterministic NPZ
    publication; existing text/JSON behavior remains unchanged.

## Output

```text
D:/xunce/inputs/mid_dual/scenario-sources/<source-id>/
  manifest.json
  standard-catalog.json
  standard-source.json
  static-truth-index.jsonl
  reset-state-index.jsonl
  descriptors.jsonl
  reconstruction-index.jsonl
  policy-blind-approval.json
  source-manifest.json
  masks/<scenario-hash>.npz
```

## Task S1: Source materializer

**Interfaces:**

- Consumes: exact Stage6 coverage manifest path/SHA, canonical Standard catalog
  built by `build_standard_catalog()`, the frozen safety contract, source config,
  and project authorization SHA.
- Produces: Task 3 `--descriptor-catalog`, `--source-manifest`, and
  `--reconstruction-index` inputs plus a completion manifest.

- [ ] **Step 1: Write RED contract tests**

  Test exact config fields, split exclusion, forbidden-field rejection, stable
  pose/distance bins, exact source-pool hashes, deterministic mask NPZ,
  Stage6 audit/scenario identity binding, content-addressed source ID,
  data-first publication, immutable completed roots, and a no-execute CLI that
  performs zero reads/writes.

- [ ] **Step 2: Run RED**

  ```powershell
  & 'D:/conda_envs/lunar-explorer/python.exe' -m pytest `
    tests/test_xunce_mid_dual_scenario_sources.py -vv -p no:cacheprovider `
    --basetemp 'D:/xunce/tmp/pytest-mid-dual/scenario-sources-red'
  ```

  Expected: fail because the preparer and config do not exist.

- [ ] **Step 3: Implement descriptor construction**

  For each Test/Unseen/Validation record, rebuild the exact scenario, load the
  exact Stage6 mask, reset one local environment without policy inference, and
  record:

  ```text
  slope_p90_deg
  hard_obstacle_fraction
  start_pose_bin
  initial_observed_coverable_fraction
  initial_valid_frontier_count
  coverable_cell_count
  parent_roi
  density_profile
  start_to_farthest_candidate_distance_bin
  ```

  Distance bins are stable rank tertiles within the source split, ordered by
  `(max_reset_candidate_distance_m, scenario_id)`. Retain the raw reset distance
  and candidate-endpoint digest only in `reset-state-index.jsonl`.

- [ ] **Step 4: Implement exact reconstruction evidence**

  Call `Stage6CoverageManifest.scenario_audit()` for every descriptor, load the
  exact bound mask, and publish one deterministic `coverable_mask` NPZ. The
  reconstruction row binds scenario ID/hash, key SHA, mask path/file SHA/size.
  The Standard catalog artifact bytes hash must equal the manifest's
  `catalog_sha256`.

- [ ] **Step 5: Implement source manifest and approval**

  Publish artifact refs for descriptor/Standard catalog, Standard raw-source
  provenance, static-truth index, reset-state index, descriptor-generator source
  SHA, Stage6 manifest, source-pool hashes, and a policy-blind approval bound to
  project authorization
  `720e11ef04ad2b57283421809a077ccf1f0b35167a9f482082241398ad0214d2`.
  The approval declares all five forbidden input classes false and grants only
  source-generation use, not gate passage.

- [ ] **Step 6: Implement recoverable CLI**

  CLI:

  ```text
  --config <repo config>
  --coverage-manifest <absolute D path>
  --coverage-manifest-sha256 <exact>
  --output-root <absolute new D root>
  --execute
  ```

  Without `--execute`, exit before reading the catalog/coverage manifest or
  creating directories. With execute, write all data and masks first, verify
  bytes/hashes/one-to-one joins, then write `manifest.json`. Resume accepts only
  an exact, contiguous prefix with unchanged inputs.
  Binary mask writes must use the additive `xunce_artifact_io.write_bytes()`
  helper; no runner performs direct `Path.open()` or direct artifact writes.

- [ ] **Step 7: Run GREEN and integration**

  ```powershell
  & 'D:/conda_envs/lunar-explorer/python.exe' -m pytest `
    tests/test_xunce_mid_dual_scenario_sources.py `
    tests/test_xunce_mid_dual_scenario_freeze.py -vv -p no:cacheprovider `
    --basetemp 'D:/xunce/tmp/pytest-mid-dual/scenario-sources-green'
  ```

  Expected: all tests pass; fixture source bundle feeds Task 3 builder and
  independently verifies.

- [ ] **Step 8: Commit and later execute**

  Commit only the four Task S1 files after exact diff review. Actual source
  generation happens later, after the full Stage6 coverage cache manifest is
  available and its hash is fixed; it writes a new D root and does not modify
  historical formal outputs.

## Self-Review

- [ ] No checkpoint/policy/result is imported or read.
- [ ] Train records cannot enter descriptors.
- [ ] Every descriptor has one exact Stage6 entry and reconstruction mask.
- [ ] Distance/category selection is deterministic and policy blind.
- [ ] Source-pool and bundle hashes independently recalculate.
- [ ] Approval binds authorization but does not claim a gate pass.
