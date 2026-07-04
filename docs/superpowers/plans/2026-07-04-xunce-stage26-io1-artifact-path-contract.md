# Stage26.IO1 Artifact Path Contract And Long-Path Resilience

## Summary

Stage26.IO1 introduces a bounded artifact-path governance layer for Windows runs.
It does not change PPO, rewards, Hybrid A*, candidate generation, or synthetic terrain.

The stage addresses the recurring failure pattern where deep D-drive output roots plus
long artifact filenames make downstream wrappers report missing summaries or batches.
The repair is intentionally staged:

- common long-path aware IO helper;
- canonical short artifact aliases with legacy fallback;
- dual-write for Stage21.1, Stage21.2, and Stage21.3 handoff artifacts;
- short default roots for new Stage26.8M/8N/8Q/IO1 outputs;
- audit-only IO1 stage to validate path budget and alias compatibility.

## Contract

- New output roots use `D:/xunce/out/<stage_short>`.
- Historical `D:/CodexDownloads/...` output roots remain read-only legacy inputs.
- Canonical short files are preferred on read.
- Legacy long filenames remain readable and are still written during migration.
- Large lineage, full stage ids, hashes, and combo ids belong in JSON manifests and
  summaries, not in directory or filename chains.

## Boundary

Stage26.IO1 is path and artifact IO governance only. It must not:

- publish checkpoints;
- replace default policy;
- connect a real executor;
- start canary traffic;
- change reward objectives;
- change network structure;
- change Hybrid A* search semantics;
- change synthetic terrain generation.

## Verification

```powershell
python -m pytest tests\test_xunce_artifact_io.py tests\test_xunce_artifact_paths.py tests\test_xunce_stage26_io1_artifact_path_contract.py tests\test_xunce_stage21_1_on_policy_ppo_rollout_collector.py tests\test_xunce_stage21_2_coverage_first_ppo_reward_contract.py tests\test_xunce_stage21_3_ppo_batch_validation.py tests\test_xunce_stage26_8q_derived_high_res_planning_proxy_alignment.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage26-io1

python -m py_compile scripts\xunce_artifact_io.py scripts\xunce_artifact_paths.py scripts\run_xunce_stage26_io1_artifact_path_contract.py scripts\run_xunce_stage21_1_on_policy_ppo_rollout_collector.py scripts\run_xunce_stage21_2_coverage_first_ppo_reward_contract.py scripts\run_xunce_stage21_3_ppo_batch_validation.py

python scripts\run_stage.py --stage xunce-stage26-io1-artifact-path-contract-and-long-path-resilience --dry-run
python scripts\run_stage.py --stage xunce-stage26-io1-artifact-path-contract-and-long-path-resilience
```
