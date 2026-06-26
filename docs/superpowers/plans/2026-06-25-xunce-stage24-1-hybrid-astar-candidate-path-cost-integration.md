# Stage24.1 Hybrid A* Candidate Path-Cost Integration

## Goal

Stage24.1 connects the Stage24.0 Scout Mini `(x,y,theta)` Hybrid A* pose
planner to candidate path-cost auditing. The purpose is to let later reward and
PPO stages see the cost of reaching a target observation pose, not only the old
grid A* cost of reaching a target cell.

## Scope

- Add an opt-in helper for candidate-level Hybrid A* pose path-cost evaluation.
- Add a Stage24.1 runner, config, tests, registry entry, and documentation.
- Compare legacy grid A* point path cost with Hybrid A* pose path cost.
- Preserve platform lineage, `max_traversable_slope_deg=30.0`, and hard obstacle
  sources from the Stage23/24 slope-theta line.
- Keep default grid A* unchanged.

## Artifacts

Default output root:

```text
D:\CodexDownloads\lunar-path-planning\stage24_hybrid_astar_pose_planner\outputs\path_feedback_batch_xunce_stage24_1_hybrid_astar_candidate_path_cost_integration_v1
```

Expected artifacts:

- `xunce-stage24-1-summary.json`
- `xunce-stage24-1-candidate-hybrid-path-cost-audit.jsonl`
- `xunce-stage24-1-grid-vs-hybrid-cost-delta-audit.json`
- `xunce-stage24-1-path-cost-source-recommendation.json`
- `xunce-stage24-1-next-stage-routing.json`
- `xunce-stage24-1-report.md`
- `xunce-stage24-1-manifest.json`

## Acceptance Criteria

- Stage24.1 can generate Hybrid A* pose path-cost audit rows for real or bounded
  `(x,y,theta)` candidates.
- Each audit row includes reachability, cost breakdown, pose path hash,
  `hybrid_astar_ackermann_feasible_claimed=false`, legacy grid A* cost, and
  hybrid-vs-grid delta.
- The summary recommends `path_cost_source=hybrid_astar_pose_path/v1` only as a
  later-stage input; it does not replace default A*.
- All publish/default-policy/executor/canary fields stay false or zero.

## Verification

```powershell
python -m pytest tests\test_xunce_stage24_1_hybrid_astar_candidate_path_cost_integration.py tests\test_xunce_stage24_0_hybrid_astar_pose_path_planner_full_foundation.py path-planner\tests\test_hybrid_astar.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage24-1
python -m py_compile scripts\run_xunce_stage24_1_hybrid_astar_candidate_path_cost_integration.py scripts\xunce_hybrid_astar_candidate_path_cost.py path-planner\src\path_planner\search\hybrid_astar.py
python scripts\run_stage.py --stage xunce-stage24-1-hybrid-astar-candidate-path-cost-integration --dry-run
```

## Non-Goals

Stage24.1 does not run PPO, publish checkpoints, replace the default policy,
connect an executor, start canary traffic, introduce continuous theta policy,
perform 3D DEM LOS, or claim performance improvement.
