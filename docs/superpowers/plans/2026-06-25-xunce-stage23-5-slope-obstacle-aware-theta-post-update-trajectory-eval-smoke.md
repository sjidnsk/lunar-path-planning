# Stage23.5 Slope-Obstacle-Aware Theta Post-Update Trajectory Eval Smoke

## Goal

Run a bounded Stage21.5 pre/post trajectory evaluation for the Stage23.4 experimental-only checkpoint. The evaluation must use the same platform-aligned 30 degree slope-obstacle-aware theta configuration on both sides and only change the checkpoint.

## Scope

- Wrap the existing Stage21.5 runner; do not reimplement trajectory evaluation.
- Require `coverage_source=endpoint_theta_slope_obstacle_los/v1`, `obstacle_occlusion_enabled=true`, `max_traversable_slope_deg=30.0`, 8 theta bins, and 90 degree endpoint FOV.
- Use the Stage23.4 source checkpoint and experimental checkpoint lineage.
- Strong-join pre/post inference rows by `scenario_id + step_index + current_cell + covered_cells_hash + candidate_set_hash`.
- Audit selected viewpoint/theta/cell/action changes, probability deltas, coverage/AUC/path-cost deltas, and safety regressions.

## Outputs

- `xunce-stage23-5-summary.json`
- `xunce-stage23-5-stage21-5-config.json`
- `xunce-stage23-5-stage21-5-summary.json`
- `xunce-stage23-5-slope-theta-action-change-audit.json`
- `xunce-stage23-5-trajectory-delta-audit.json`
- `xunce-stage23-5-next-stage-routing.json`
- `xunce-stage23-5-report.md`
- `xunce-stage23-5-manifest.json`

Default output root:

```text
D:\CodexDownloads\lunar-path-planning\stage23_slope_obstacle_aware_theta_reward\outputs\path_feedback_batch_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke_v1
```

## Routing

- Missing/unready Stage23.4 input: `rerun_stage23_5_required_inputs`
- Missing slope-theta inference fields or unusable strong join: `repair_stage23_5_slope_theta_inference_binding`
- Hard-risk, mask, path-planning, or open-grid regression: `repair_stage23_5_slope_theta_eval_safety_regression`
- Action probabilities and viewpoint unchanged: `repair_stage23_slope_theta_policy_update_signal_strength`
- Viewpoint/theta changed but coverage/AUC did not improve: `repair_stage23_slope_theta_credit_assignment`
- Coverage/AUC improve with no safety regression: `run_stage23_6_slope_obstacle_aware_theta_multi_seed_ppo_pilot`

## Non-Goals

Do not run multi-seed PPO, publish checkpoints, replace the default policy, connect an executor, start canary traffic, modify network/default A*/candidate generation/reward targets, introduce continuous theta, model 3D LOS, or claim final performance.

## Verification

```powershell
python -m pytest tests\test_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke.py tests\test_xunce_stage23_4_slope_obstacle_aware_theta_ppo_update_smoke.py tests\test_xunce_stage21_5_post_update_offline_trajectory_evaluation.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage23-5
python -m py_compile scripts\run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke.py scripts\run_xunce_stage21_5_post_update_offline_trajectory_evaluation.py
python scripts\run_stage.py --stage xunce-stage23-5-slope-obstacle-aware-theta-post-update-trajectory-eval-smoke --dry-run
```
