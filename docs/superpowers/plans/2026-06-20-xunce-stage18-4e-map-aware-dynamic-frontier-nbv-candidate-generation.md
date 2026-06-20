# Xunce Stage 18.4E Map-Aware Dynamic Frontier-NBV Candidate Generation

## Summary

Stage 18.4E narrows the dynamic rollout bottleneck to candidate generation. The goal is to replace radius/BBox-driven dynamic proposals with map-aware coverage-frontier and undercovered-component proposals while preserving the existing Stage 18.4D dynamic rollout, in-process batch A* validation, model inference, and governance boundaries.

This stage does not train models, change checkpoints, modify action space, change default A*, connect an executor, publish checkpoints, or start online canary traffic.

## Implementation

- Keep `candidate_refresh_mode=dynamic_frontier_nbv_in_process` as the public entrypoint and record the concrete algorithm as `candidate_generation_algorithm_source=map_aware_coverage_frontier_nbv/v1`.
- Generate proposals from coverage frontier, undercovered component centroid/boundary, low-cost bridge, and conservative local families.
- Clip coverage estimates to valid ROI/passable cells and record denominator and ROI-weight provenance.
- Validate proposals through `in_process_path_planner_astar_batch`; only validated, reachable, non-open-grid candidates with finite path cost, path length, and risk enter the action set.
- Select final action candidates with `validated_pareto_diverse` rather than pure coverage-first sorting.
- Treat `risk` as a sidecar/path-cost proxy or candidate scalar with explicit provenance, not as physical executor risk.

## Validation

Recommended checks:

```powershell
D:\conda_envs\lunar-explorer\python.exe -m pytest `
  tests\test_xunce_dynamic_frontier_nbv.py `
  tests\test_xunce_path_planner_astar_batch.py `
  tests\test_xunce_high_fidelity_exploration_coverage_comparison.py `
  tests\test_xunce_stage18_research_evidence_pipeline.py `
  tests\test_platform_stage_runner.py `
  -q --basetemp outputs\pytest-stage18-4e

D:\conda_envs\lunar-explorer\python.exe -m py_compile `
  scripts\xunce_dynamic_frontier_nbv.py `
  scripts\xunce_path_planner_astar_batch.py `
  scripts\xunce_frontier_nbv_validation.py `
  scripts\run_xunce_high_fidelity_exploration_coverage_comparison.py `
  scripts\xunce_stage18_pipeline.py
```

After code validation, rerun a 24x40 dynamic rollout root and compare Xunce vs incumbent by raw covered cells, path cost, risk proxy, coverage per 100m, oracle regret, and candidate generation exhaustion counts.
