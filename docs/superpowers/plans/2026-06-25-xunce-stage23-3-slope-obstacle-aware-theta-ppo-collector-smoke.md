# Stage23.3 Slope-Obstacle-Aware Theta PPO Collector Smoke

## Goal

Verify that the real Stage21.1 -> Stage21.2 -> Stage21.3 data path can produce,
reward, and validate platform-aligned 30 degree slope-obstacle-aware
`(x,y,theta)` PPO batches.

Stage23.3 closes the gap left by Stage23.2: Stage23.2 proved the reward/batch
contract by replaying Stage23.0 LOS audit rows, but explicitly did not claim
collector-level `obstacle_aware_new_visible_cell_count` provenance.

## Scope

- Use Stage23.2 root:
  `D:\CodexDownloads\lunar-path-planning\stage23_slope_obstacle_aware_theta_reward\outputs\path_feedback_batch_xunce_stage23_2_slope_obstacle_aware_theta_reward_contract_v1`.
- Inherit the Stage23.2B high-resolution ROI root and pass it to Stage21.1 as
  `source_roi_expansion_root`.
- Run bounded Stage21.1 -> Stage21.2 -> Stage21.3 smoke only.
- Keep `max_traversable_slope_deg=30.0`,
  `slope_blocked_source_kind=slope_blocked_as_obstacle_proxy`, and
  `coverage_source=endpoint_theta_slope_obstacle_los/v1`.

## Required Contracts

Stage21.1 transition `info` must include:

- `candidate_viewpoints`, `candidate_theta_deg`, `selected_viewpoint`,
  `selected_theta_deg`
- `obstacle_aware_new_visible_cell_counts`
- `obstacle_aware_theta_coverage_hashes`
- `obstacle_aware_theta_coverage_gain_per_path_costs`
- `slope_obstacle_source_hash`, `platform_contract_hash`
- `max_traversable_slope_deg=30.0`
- `slope_blocked_source_kind=slope_blocked_as_obstacle_proxy`

Stage21.2 reward rows must use:

- `slope_obstacle_aware_theta_reward_contract=true`
- `coverage_source=endpoint_theta_slope_obstacle_los/v1`
- no point-only or unobstructed theta fallback

Stage21.3 must reject old reward batches when
`require_slope_obstacle_aware_theta_reward_contract=true`.

Stage23.3 also audits the full lineage closure:

- Stage21.1 manifest `model_audit.source_roi_expansion_root` must match the
  Stage23.2B high-resolution ROI root.
- Stage21.1 transition ids, Stage21.2 reward ids, and Stage21.3 batch ids must
  be non-empty and closed; orphan reward or batch rows fail the smoke.

## Outputs

Default output root:

`D:\CodexDownloads\lunar-path-planning\stage23_slope_obstacle_aware_theta_reward\outputs\path_feedback_batch_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke_v1`

Artifacts:

- `xunce-stage23-3-summary.json`
- `xunce-stage23-3-stage21-1-collector-summary.json`
- `xunce-stage23-3-stage21-2-reward-summary.json`
- `xunce-stage23-3-stage21-3-batch-summary.json`
- `xunce-stage23-3-slope-theta-transition-contract-audit.json`
- `xunce-stage23-3-next-stage-routing.json`
- `xunce-stage23-3-report.md`
- `xunce-stage23-3-manifest.json`

## Verification

```powershell
python -m pytest tests\test_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke.py tests\test_xunce_stage23_2_slope_obstacle_aware_theta_reward_contract.py tests\test_xunce_stage21_1_on_policy_ppo_rollout_collector.py tests\test_xunce_stage21_2_coverage_first_ppo_reward_contract.py tests\test_xunce_stage21_3_ppo_batch_validation.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage23-3
python -m py_compile scripts\run_xunce_stage23_3_slope_obstacle_aware_theta_ppo_collector_smoke.py scripts\run_xunce_stage21_1_on_policy_ppo_rollout_collector.py scripts\run_xunce_stage21_2_coverage_first_ppo_reward_contract.py scripts\run_xunce_stage21_3_ppo_batch_validation.py
python scripts\run_stage.py --stage xunce-stage23-3-slope-obstacle-aware-theta-ppo-collector-smoke --dry-run
```

## Result

The bounded real smoke passed with 4 trainable transitions, 4 reward rows, and
4 batch rows. The high-resolution ROI root matched the Stage21.1 manifest, the
transition/reward/batch `transition_id` sets were closed, the slope/theta
contract missing counts were zero, hard-risk/mask/path-planning/open-grid
regressions were zero, and the next route is
`run_stage23_4_slope_obstacle_aware_theta_ppo_update_smoke`.

Stage23.3 does not run PPO update, publish checkpoints, replace default policy,
connect executor, start canary, introduce continuous theta, model 3D LOS, change
network architecture, or change default A*.
