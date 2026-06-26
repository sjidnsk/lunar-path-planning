# Stage26.3 Synthetic Terrain Post-Update Trajectory Eval Smoke

## Goal

Run a bounded Stage21.5 pre/post trajectory evaluation using the Stage26.2
experimental-only checkpoint and the same synthetic rock/pit terrain lineage.
The stage must answer whether the update changes real `(x,y,theta)` selection
and report coverage, AUC, Hybrid A* path cost, coverage-per-100m, and safety
deltas without claiming final performance.

## Scope

- Input root: `D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/outputs/path_feedback_batch_xunce_stage26_2_synthetic_terrain_ppo_update_smoke_v1`
- Wrapper: `scripts/run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke.py`
- Config: `configs/xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke_v1.json`
- Output root: `D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/outputs/path_feedback_batch_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke_v1`

## Contract

- Reuse Stage26.2 source and experimental checkpoints; do not run another PPO update.
- Force `coverage_source=endpoint_theta_slope_obstacle_los/v1`.
- Force `path_cost_source=hybrid_astar_pose_path/v1`.
- Force `synthetic_source_kind=synthetic_terrain_obstacle_proxy/v1`.
- Keep `physical_obstacle_cells_written=false`.
- Strong join pre/post rows by `scenario_id + step_index + current_cell + covered_cells_hash + candidate_set_hash + synthetic_terrain_hash`.
- Use the five-field base key only to diagnose synthetic lineage mismatch, not to claim policy movement.

## Artifacts

- `xunce-stage26-3-summary.json`
- `xunce-stage26-3-stage21-5-config.json`
- `xunce-stage26-3-stage21-5-summary.json`
- `xunce-stage26-3-synthetic-action-change-audit.json`
- `xunce-stage26-3-trajectory-delta-audit.json`
- `xunce-stage26-3-next-stage-routing.json`
- `xunce-stage26-3-report.md`
- `xunce-stage26-3-manifest.json`

## Routing

- Missing or untrusted inputs: `rerun_stage26_3_required_inputs`
- Missing synthetic/Hybrid/path inference fields: `repair_stage26_3_synthetic_inference_binding`
- Synthetic hash/source mismatch: `repair_stage26_3_synthetic_lineage_binding`
- Synthetic-as-physical pollution: `repair_stage26_3_synthetic_source_semantics`
- Safety regression: `repair_stage26_3_synthetic_eval_safety_regression`
- Probability/viewpoint unchanged: `repair_stage26_synthetic_policy_update_signal_strength`
- Probability moved but viewpoint/theta unchanged: `calibrate_stage26_synthetic_discrete_margin_crossing`
- Action changed without coverage/AUC/path-cost improvement: `repair_stage26_synthetic_credit_assignment`
- Improved with clean safety: `run_stage26_4_synthetic_terrain_multi_seed_ppo_pilot`

## Verification

```powershell
python -m pytest tests\test_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke.py tests\test_xunce_stage26_2_synthetic_terrain_ppo_update_smoke.py tests\test_xunce_stage21_5_post_update_offline_trajectory_evaluation.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage26-3
python -m py_compile scripts\run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke.py scripts\run_xunce_stage21_5_post_update_offline_trajectory_evaluation.py
python scripts\run_stage.py --stage xunce-stage26-3-synthetic-terrain-post-update-trajectory-eval-smoke --dry-run
```

## Non-Goals

No PPO update, no multi-seed pilot, no checkpoint publication, no default policy
replacement, no real executor, no canary, no reward/network/default A*/candidate
generation changes, no continuous x/y, no 3D LOS, and no performance claim.
