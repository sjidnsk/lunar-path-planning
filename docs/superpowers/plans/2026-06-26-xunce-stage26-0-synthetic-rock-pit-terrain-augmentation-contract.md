# Stage26.0 Synthetic Rock/Pit Terrain Augmentation Contract

## Goal

Add a reproducible synthetic terrain augmentation layer on top of existing
sample maps / high-resolution ROI sidecars. The layer creates synthetic rocks
and pits so endpoint theta LOS and Hybrid A* pose path cost face meaningful
local occlusion and detour pressure.

## Contract

- `synthetic_terrain_model_id=synthetic_rock_pit_terrain/v1`.
- `synthetic_terrain_seed` controls all randomness; same seed must reproduce the
  same feature catalog and `synthetic_terrain_hash`.
- Rocks and pits write only synthetic proxy fields such as
  `synthetic_hard_obstacle_cells`, `synthetic_los_blocker_cells`,
  `synthetic_high_risk_cells`, and `synthetic_cost_inflation_cells`.
- The augmentation must not write or imply `physical_obstacle_cells`.
- Synthetic hard obstacles affect Hybrid A* only through opt-in sidecars; default
  A* and existing DEM/slope inputs are not replaced.

## Artifacts

- `xunce-stage26-0-summary.json`
- `xunce-stage26-0-synthetic-feature-catalog.jsonl`
- `xunce-stage26-0-synthetic-obstacle-source.json`
- `xunce-stage26-0-map-augmentation-audit.json`
- `xunce-stage26-0-los-impact-audit.json`
- `xunce-stage26-0-hybrid-path-impact-audit.json`
- `xunce-stage26-0-next-stage-routing.json`
- `xunce-stage26-0-report.md`
- `xunce-stage26-0-manifest.json`

## Verification

```powershell
python -m pytest tests\test_xunce_synthetic_terrain_features.py tests\test_xunce_stage26_0_synthetic_rock_pit_terrain_augmentation_contract.py tests\test_xunce_stage25_0_continuous_theta_hybrid_action_space_foundation.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage26-0
python -m py_compile scripts\xunce_synthetic_terrain_features.py scripts\run_xunce_stage26_0_synthetic_rock_pit_terrain_augmentation_contract.py
python scripts\run_stage.py --stage xunce-stage26-0-synthetic-rock-pit-terrain-augmentation-contract --dry-run
```

## Boundaries

Stage26.0 does not run PPO, publish checkpoints, replace default policy, connect
an executor, start canary traffic, modify the reward target, modify the network,
replace default A*, model 3D LOS, or claim final performance.
