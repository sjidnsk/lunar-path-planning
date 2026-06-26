# Stage26.2 Synthetic Terrain PPO Update Smoke

## Goal

Verify that the real Stage26.1 synthetic rock/pit terrain PPO batch can be
consumed by Stage21.4 tiny PPO update and saved as a reloadable experimental-only
checkpoint.

## Inputs

- Stage26.1 root:
  `D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/outputs/path_feedback_batch_xunce_stage26_1_synthetic_terrain_collector_smoke_v1`
- Stage21.3 batch under `<Stage26.1 root>/s21_3`.
- Sandbox candidate checkpoint:
  `outputs/path_feedback_batch_xunce_sandbox_candidate_preflight_v1/sandbox_package/xunce-controlled-training-candidate.pt`

## Contract

- `coverage_source=endpoint_theta_slope_obstacle_los/v1`
- `path_cost_source=hybrid_astar_pose_path/v1`
- `synthetic_terrain_model_id=synthetic_rock_pit_terrain/v1`
- `synthetic_source_kind=synthetic_terrain_obstacle_proxy/v1`
- `physical_obstacle_cells_written=false`
- Non-empty `physical_obstacle_cells` payloads are rejected.
- Point-only reward fallback, unobstructed theta fallback, and grid path fallback
  must remain false.
- Stage21.4 source checkpoint SHA must match the Stage26.1 Stage21.1 collector
  checkpoint SHA, preserving behavior-policy / old-log-prob lineage.

## Implementation

- Add `scripts/run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke.py`.
- Add `configs/xunce_stage26_2_synthetic_terrain_ppo_update_smoke_v1.json`.
- Add `tests/test_xunce_stage26_2_synthetic_terrain_ppo_update_smoke.py`.
- Register stage id `xunce-stage26-2-synthetic-terrain-ppo-update-smoke`.
- Reuse Stage21.4 for PPO update. The wrapper performs the synthetic batch audit
  before invoking Stage21.4 and does not change PPO loss logic.

## Outputs

Output root:

`D:/CodexDownloads/lunar-path-planning/stage26_synthetic_terrain_augmentation/outputs/path_feedback_batch_xunce_stage26_2_synthetic_terrain_ppo_update_smoke_v1`

Artifacts:

- `xunce-stage26-2-summary.json`
- `xunce-stage26-2-stage21-4-config.json`
- `xunce-stage26-2-stage21-4-summary.json`
- `xunce-stage26-2-synthetic-ppo-batch-update-audit.json`
- `xunce-stage26-2-loss-gradient-audit.json`
- `xunce-stage26-2-checkpoint-boundary-audit.json`
- `xunce-stage26-2-next-stage-routing.json`
- `xunce-stage26-2-report.md`
- `xunce-stage26-2-manifest.json`

## Routing

- Missing or untrusted inputs: `rerun_stage26_2_required_inputs`
- Batch/action/path-cost contract failure: `repair_stage26_2_synthetic_batch_update_contract`
- Loss, gradient, or KL instability: `repair_stage26_2_synthetic_ppo_update_stability`
- Checkpoint reload or boundary failure: `repair_stage26_2_checkpoint_reload_boundary`
- Passed: `run_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke`

## Verification

```powershell
python -m pytest tests\test_xunce_stage26_2_synthetic_terrain_ppo_update_smoke.py tests\test_xunce_stage26_1_synthetic_terrain_collector_smoke.py tests\test_xunce_stage21_4_tiny_ppo_update_smoke.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage26-2
python -m py_compile scripts\run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke.py scripts\run_xunce_stage21_4_tiny_ppo_update_smoke.py
python scripts\run_stage.py --stage xunce-stage26-2-synthetic-terrain-ppo-update-smoke --dry-run
```

## Non-goals

No Stage21.5 trajectory evaluation, multi-seed PPO, checkpoint publishing,
default policy replacement, executor connection, canary, reward/network/default
A*/candidate-generation change, continuous x/y, 3D LOS, or performance claim.
