# Stage23.2B Platform Geometry Sensor Contract Alignment

## Goal

Strictly align Stage23 terrain and sensor contracts to the SCOUT MINI + PiPER
platform page. The main behavioral change is that the Stage23 hard slope
obstacle threshold is no longer the old conservative 20 degrees; it is now the
Scout Mini maximum climb capability, 30 degrees.

## Scope

- Add `configs/platforms/agilex_scout_mini_piper_v1.json` as the single platform
  contract source for Scout Mini geometry, drivetrain, climb capability, Livox
  Mid360 fields, and camera calibration gaps.
- Propagate `platform_contract_id`, `platform_contract_hash`,
  `platform_max_climb_deg`, and `max_traversable_slope_deg` through Stage23.1,
  Stage23.2A, high-fidelity obstacle source audit, and sidecars.
- Keep `slope_blocked_cells` as `slope_blocked_as_obstacle_proxy`; they are not
  physical rock, wall, or crack labels.
- Keep 20 degrees only as sensitivity/audit, not as the default hard gate.

## Artifacts

- `scripts/run_xunce_stage23_2b_platform_geometry_sensor_contract_alignment.py`
- `configs/xunce_stage23_2b_platform_geometry_sensor_contract_alignment_v1.json`
- `xunce-stage23-2b-summary.json`
- `xunce-stage23-2b-platform-contract-audit.json`
- `xunce-stage23-2b-slope-threshold-comparison.json`

## Routing

- Boundary violation: `resolve_stage23_2b_boundary_rejections`
- Invalid platform contract: `repair_stage23_2b_platform_contract`
- Failed bounded smoke: `repair_stage23_2b_platform_aligned_smoke`
- 30 degree slope LOS material: `implement_stage23_2_slope_obstacle_aware_theta_reward_contract`
- 30 degree slope LOS not material: `document_platform_aligned_slope_occlusion_audit_only`

## Non-goals

No PPO, checkpoint publication, default-policy replacement, real executor
connection, canary, continuous theta policy, 3D DEM LOS, network change, default
A* change, or physical rock/crack obstacle labeling is authorized by this stage.
