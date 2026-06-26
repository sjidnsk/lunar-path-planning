# Stage23.5A Repair Required Inputs / High-Res ROI Coverage

## Goal

Repair the Stage23.5 input blocker caused by insufficient high-resolution ROI
slice coverage. Stage23.5 required two scenarios for the bounded pre/post
trajectory evaluation, but the Stage23.2B high-res ROI expansion had only one
slice, so the action-probability audit was partial diagnostic evidence.

## Scope

- Read the prior Stage23.5 summary and only proceed if the route is
  `rerun_stage23_5_required_inputs` with scenario/episode-short reason codes.
- Precheck configured USGS 4m GeoTIFF ROI windows.
- Write at least two valid ROI windows into a repaired Stage23.2A config.
- Rerun existing Stage23.2B -> Stage23.2 -> Stage23.3 -> Stage23.4 -> Stage23.5
  under the Stage23.5A output root.
- Preserve 30 degree platform-aligned slope obstacle lineage and
  `coverage_source=endpoint_theta_slope_obstacle_los/v1`.

## Artifacts

Default output root:

```text
D:\CodexDownloads\lunar-path-planning\stage23_slope_obstacle_aware_theta_reward\outputs\path_feedback_batch_xunce_stage23_5a_repair_required_inputs_high_res_roi_coverage_v1
```

Primary outputs:

- `xunce-stage23-5a-summary.json`
- `xunce-stage23-5a-roi-window-selection-audit.json`
- `xunce-stage23-5a-rerun-stage23-2b-summary.json`
- `xunce-stage23-5a-rerun-stage23-2-summary.json`
- `xunce-stage23-5a-rerun-stage23-3-summary.json`
- `xunce-stage23-5a-rerun-stage23-4-summary.json`
- `xunce-stage23-5a-rerun-stage23-5-summary.json`
- `xunce-stage23-5a-next-stage-routing.json`
- `xunce-stage23-5a-report.md`
- `xunce-stage23-5a-manifest.json`

## Routing

- Prior Stage23.5 route is not the required-input blocker:
  `preserve_stage23_5_existing_route`
- Fewer than two valid ROI windows:
  `expand_or_relocate_stage23_high_res_roi_windows`
- Any upstream rerun fails: preserve that stage's repair route.
- Stage23.5 still has scenario/episode-short inputs:
  `rerun_stage23_5_required_inputs`
- Complete Stage23.5 with unchanged actions:
  `repair_stage23_slope_theta_policy_update_signal_strength`
- Changed viewpoint/theta without coverage/AUC lift:
  `repair_stage23_slope_theta_credit_assignment`
- Coverage/AUC lift with no worst-scenario regression:
  `run_stage23_6_slope_obstacle_aware_theta_multi_seed_ppo_pilot`

## Non-Goals

Stage23.5A does not run formal PPO training, publish a checkpoint, replace the
default policy, connect the executor, start canary traffic, modify network or
default A*, change candidate generation, introduce continuous theta, or model
3D LOS. It is an input repair and orchestration stage only.

## Verification

```powershell
python -m pytest tests\test_xunce_stage23_5a_repair_required_inputs_high_res_roi_coverage.py tests\test_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke.py tests\test_xunce_stage23_2a_high_resolution_terrain_data_ingestion.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage23-5a
python -m py_compile scripts\run_xunce_stage23_5a_repair_required_inputs_high_res_roi_coverage.py scripts\run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke.py scripts\run_xunce_stage23_2a_high_resolution_terrain_data_prepare.py
python scripts\run_stage.py --stage xunce-stage23-5a-repair-required-inputs-high-res-roi-coverage --dry-run
```
