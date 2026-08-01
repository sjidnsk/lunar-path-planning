# Task 3: Stage26 Synthetic Terrain / PPO Route Cleanup

## Scope and frozen input

- Base commit: 0405e0290c246b60b5fce0283fd2ddbc7144ef73.
- Manifest: D:/xunce/out/route_retire_preflight/candidate-manifest.json.
- Verified SHA-256: da9f23c3af83c3506143338a41e1c1a8f8f09dc4f59d4a4bac5adea160f6371c.
- Exact retired set: 148 tracked Stage26 paths: 39 configs, 38 scripts, 37 tests, and 34 docs.

## Changes

- Removed every manifest-record path with classification=retire_candidate and a
  stage26 path segment, using explicit file paths.
- Removed all 38 Stage26 registry entries from configs/stage_registry.json.
- Removed tests/test_xunce_documentation_boundary_consolidation.py, the
  manifest-external test that directly imported and executed the deleted
  Stage26.5b runner.
- Added tests/test_retired_stage26_cleanup.py. It recomputes the frozen 148
  paths, verifies their category counts and absence from both the worktree and
  tracked files, requires a Stage26-free registry, verifies every registry
  script still exists, and checks retained platform entrypoints do not expose
  Stage26 runners.

## Verification

All commands used D:/conda_envs/lunar-explorer/python.exe with the current
path-planner/src first and src second in PYTHONPATH. Pytest basetemp roots were
under D:/xunce/basetemp/retired-route-task3-*.

- Stage26 absence, retained registry, and preflight contracts: 36 passed.
- Stage6 and G1/G2/G3 lightweight contracts: 297 passed, 2 skipped.
- Shared artifact IO/path and terrain-sidecar callers: 22 passed.
- windows-non-drake platform validation matrix dry-run: passed; it emitted only
  retained platform, Stage15--18, path-planner, and
  dev-platform-constraints commands.
- The final manifest audit found no present or live-tracked member of the 148
  paths, no Stage26 registry key, and no missing registry script.
- configs/stage_registry.json parses as JSON, has a minimal 0-addition /
  228-deletion diff, and git diff --check passes.

## Boundary notes

- Stage18--25 candidates were not deleted or modified.
- Stage6 high-resolution PPO, G1/G2/G3, default grid A*, opt-in
  multi-platform planner v3, dev-platform-constraints, artifact helpers, and
  terrain-sidecar semantics were not changed.
- Historical prose references outside executable entrypoints are intentionally
  left for Task5.
- tests/test_route_retirement_preflight.py retains a pure classification
  sample for a deleted Stage26 runner and does not import it. Two Stage21
  candidate tests still name the removed reward-profile config; they belong to
  the Stage18--25 retirement set and are intentionally left unchanged for
  Task4 rather than deleting that set early.

## Review fix round 1

- Removed the 38 deleted Stage26 keys from the explicit
  tests/test_platform_stage_runner.py dry-run coverage set. The test retains
  its explicit coverage of all other currently registered stages.
- Added the tracked compact fixture
  tests/fixtures/route_retirement_candidates_v1.json. It records the frozen
  source manifest SHA-256 and the exact path lists/counts for path_feedback
  (29), Stage26 (148), and the remaining Stage18--25 Task4 set (250).
- Updated both retired-path-feedback and retired-Stage26 absence audits to
  load the fixture, assert the fixed SHA/count contracts, and no longer depend
  on an uncommitted D: manifest.
- Focused platform-runner, cleanup, and preflight tests: 42 passed.
