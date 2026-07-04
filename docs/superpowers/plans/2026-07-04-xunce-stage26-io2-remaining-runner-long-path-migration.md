# Stage26.IO2 Remaining Runner Long-Path Migration

## Summary

Stage26.IO2 completes the next artifact-path migration layer after Stage26.IO1.
It extends canonical short artifact aliases beyond Stage21.1/21.2/21.3 and
migrates the remaining active Stage26 update/eval/resume runners onto
`scripts/xunce_artifact_io.py`.

This stage is path and artifact IO governance only. It does not change PPO loss,
reward, network structure, Hybrid A* search semantics, candidate generation, or
synthetic terrain.

## Scope

- Add Stage21.4, Stage21.5, Stage26.2, and Stage26.3 artifact aliases.
- Keep checkpoint `.pt` filenames unchanged; alias only checkpoint audit JSON.
- Migrate Stage21.4/21.5, Stage26.2/26.3, and Stage26.8D/F/G/H/I/O/P runners
  to long-path aware JSON/JSONL/text IO.
- Change new default roots for Stage26.8O and Stage26.8P to short D-drive roots.
- Add an IO2 audit runner that checks alias compatibility, root path budget, and
  high-risk direct artifact IO in migrated runners.

## Output

Default output root:

```text
D:/xunce/out/s26_io2
```

Artifacts:

- `summary.json`
- `path_audit.json`
- `alias_audit.json`
- `runner_static_io_audit.json`
- `routing.json`
- `report.md`
- `manifest.json`

## Routing

- boundary enabled: `resolve_stage26_io2_boundary_rejections`
- alias contract failure: `repair_stage26_io2_alias_contract`
- runner direct artifact IO remains: `repair_stage26_io2_runner_artifact_io_migration`
- short root contract failure: `repair_stage26_io2_short_root_contract`
- all passed: `rerun_stage26_8n_aggressive_update_sweep_with_aligned_planning_proxy`

## Verification

```powershell
python -m pytest tests\test_xunce_artifact_io.py tests\test_xunce_artifact_paths.py tests\test_xunce_stage26_io2_remaining_runner_long_path_migration.py tests\test_xunce_stage21_4_tiny_ppo_update_smoke.py tests\test_xunce_stage21_5_post_update_offline_trajectory_evaluation.py tests\test_xunce_stage26_2_synthetic_terrain_ppo_update_smoke.py tests\test_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke.py tests\test_xunce_stage26_8h_resumable_diverse_scenario_post_update_eval.py tests\test_xunce_stage26_8i_diverse_scenario_policy_signal_strength_repair.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage26-io2

python -m py_compile scripts\xunce_artifact_io.py scripts\xunce_artifact_paths.py scripts\run_xunce_stage26_io2_remaining_runner_long_path_migration.py scripts\run_xunce_stage21_4_tiny_ppo_update_smoke.py scripts\run_xunce_stage21_5_post_update_offline_trajectory_evaluation.py scripts\run_xunce_stage26_2_synthetic_terrain_ppo_update_smoke.py scripts\run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke.py

python scripts\run_stage.py --stage xunce-stage26-io2-remaining-runner-long-path-migration --dry-run
python scripts\run_stage.py --stage xunce-stage26-io2-remaining-runner-long-path-migration
```
