# Stage23.0 Endpoint Obstacle-Aware Theta Sensor Coverage Contract

## Goal

Stage23.0 introduces a read-only endpoint obstacle-aware theta coverage contract.
The rover still plans to the endpoint `(x,y)` with the existing A* path planner,
then observes from that endpoint along `theta`. The sensor footprint is no
longer allowed to see through 2D obstacles.

## Scope

- Add `scripts/xunce_obstacle_aware_theta_sensor_coverage.py`.
- Add `scripts/run_xunce_stage23_0_endpoint_obstacle_aware_theta_sensor_coverage_contract.py`.
- Register stage id `xunce-stage23-0-endpoint-obstacle-aware-theta-sensor-coverage-contract`.
- Keep Stage22 range/FOV behavior as the no-obstacle baseline.
- Require explicit obstacle sources: `obstacle_cells`, `blocked_cells`,
  `no_go_cells`, or obstacle/blocked rectangles.

## Sensor Model

1. Compute the existing endpoint theta FOV visible cells.
2. For each target cell, draw a 2D grid line from sensor cell to target cell.
3. If the target cell is an obstacle, do not count it as coverage.
4. If an intermediate line cell is an obstacle, mark the target as occluded.
5. Without obstacles, the result must match Stage22 theta FOV exactly.

## Outputs

Default root:

```text
D:\CodexDownloads\lunar-path-planning\stage23_endpoint_obstacle_aware_theta_sensor_coverage\outputs\path_feedback_batch_xunce_stage23_0_endpoint_obstacle_aware_theta_sensor_coverage_contract_v1
```

Artifacts:

- `xunce-stage23-0-summary.json`
- `xunce-stage23-0-obstacle-los-audit.jsonl`
- `xunce-stage23-0-coverage-delta-audit.json`
- `xunce-stage23-0-stage22-compatibility-audit.json`
- `xunce-stage23-0-next-stage-routing.json`
- `xunce-stage23-0-report.md`
- `xunce-stage23-0-manifest.json`

## Routing

- Missing inputs or missing obstacle source:
  `rerun_stage23_0_required_obstacle_sources`
- LOS/hash replay not reproducible:
  `repair_stage23_0_obstacle_los_contract`
- Material occlusion while Stage22 reward remains unobstructed:
  `implement_stage23_1_obstacle_aware_theta_reward_contract`
- Occlusion not material on audited roots:
  `document_obstacle_occlusion_audit_only`
- Any release/default-policy/executor/canary flag enabled:
  `resolve_stage23_0_boundary_rejections`

## Non-Goals

Stage23.0 does not start PPO, publish checkpoints, replace default policy,
connect executors, start canary traffic, implement continuous theta, model
slope/height/3D occlusion, change candidate generation, modify the network, or
change default A*.
