# Stage23.4 Slope-Obstacle-Aware Theta PPO Update Smoke

## Goal

Verify that the real Stage23.3 slope-obstacle-aware theta Stage21.3 batch can be
consumed by Stage21.4 tiny PPO update and produce a reloadable experimental-only
checkpoint.

Stage23.4 is an offline update-chain smoke only. It does not run trajectory
evaluation and does not claim coverage or AUC improvement.

## Scope

- Use Stage23.3 root:
  `D:\CodexDownloads\lunar-path-planning\stage23_slope_obstacle_aware_theta_reward\outputs\path_feedback_batch_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke_v1`.
- Recheck Stage23.3 status, route, closure counters, safety counters, and
  `coverage_source=endpoint_theta_slope_obstacle_los/v1`.
- Run Stage21.4 under internal root `s21_4` with the same conservative update
  settings as the theta-aware smoke: `learning_rate=2e-6`, `loss_scale=0.25`,
  `value_loss_coefficient=0.1`, `advantage_clip_abs=5.0`.

## Required Contracts

Each Stage21.3 train row must keep:

- `slope_obstacle_aware_theta_reward_contract=true`
- `coverage_source=endpoint_theta_slope_obstacle_los/v1`
- `obstacle_aware_new_visible_cell_count`
- `obstacle_aware_theta_coverage_hash`
- `obstacle_aware_theta_coverage_gain_per_path_cost`
- `obstacle_aware_theta_coverage_denominator_cells`
- `slope_obstacle_source_hash`, `platform_contract_hash`
- `max_traversable_slope_deg=30.0`
- `slope_blocked_source_kind=slope_blocked_as_obstacle_proxy`

The selected action must bind to `info.candidate_viewpoints[action_index]`, and
`xunce_batch` action dimensions must match the viewpoint action count.

## Outputs

Default output root:

`D:\CodexDownloads\lunar-path-planning\stage23_slope_obstacle_aware_theta_reward\outputs\path_feedback_batch_xunce_stage23_4_slope_obstacle_aware_theta_ppo_update_smoke_v1`

Artifacts:

- `xunce-stage23-4-summary.json`
- `xunce-stage23-4-stage21-4-config.json`
- `xunce-stage23-4-stage21-4-summary.json`
- `xunce-stage23-4-slope-theta-ppo-batch-update-audit.json`
- `xunce-stage23-4-loss-gradient-audit.json`
- `xunce-stage23-4-checkpoint-boundary-audit.json`
- `xunce-stage23-4-next-stage-routing.json`
- `xunce-stage23-4-report.md`
- `xunce-stage23-4-manifest.json`

## Verification

```powershell
python -m pytest tests\test_xunce_stage23_4_slope_obstacle_aware_theta_ppo_update_smoke.py tests\test_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke.py tests\test_xunce_stage21_4_tiny_ppo_update_smoke.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage23-4
python -m py_compile scripts\run_xunce_stage23_4_slope_obstacle_aware_theta_ppo_update_smoke.py scripts\run_xunce_stage21_4_tiny_ppo_update_smoke.py
python scripts\run_stage.py --stage xunce-stage23-4-slope-obstacle-aware-theta-ppo-update-smoke --dry-run
```

## Boundaries

Stage23.4 must keep `publishes_checkpoint=false`,
`replaces_default_policy=false`, `connects_real_executor=false`,
`starts_online_canary=false`, and `canary_traffic_fraction=0.0`. Passing only
routes to `run_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke`.

## Result

The bounded real smoke passed with 4 batch rows. Slope/theta contract missing,
viewpoint shape mismatch, action binding mismatch, slope provenance mismatch,
selected mask violation, and reward fallback counts were zero. Stage21.4 loss,
gradient, and component gradient audits were finite; the experimental checkpoint
exists, reloads, remains `experimental_only=true`, and keeps release/executor
boundaries false. The next route is
`run_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke`.
