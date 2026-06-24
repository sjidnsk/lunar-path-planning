# Stage22.2 Theta-Aware Coverage Reward Contract

## Goal

Close the reward-side contract for theta-aware viewpoint actions. The Stage22.1
candidate contract already expresses actions as `(x,y,theta_deg)` viewpoints;
Stage22.2 verifies that coverage reward and PPO batch validation consume that
viewpoint coverage, not legacy point-only coverage.

## Scope

- Stage21.2 supports `require_theta_aware_reward_contract=true`.
- Stage21.2 theta-aware coverage gain comes from
  `theta_new_visible_cell_count / theta_coverage_denominator_cells`.
- Stage21.2 reward rows write `coverage_source=theta_aware_sensor_footprint/v1`
  and `point_only_reward_fallback_used=false`.
- Stage21.3 supports `require_theta_aware_reward_contract=true`.
- Stage21.3 rejects missing theta reward provenance, point-only fallback, or a
  reward viewpoint that does not match the selected transition action.
- Stage22.2 replays sampled Stage22.1 viewpoint candidates and verifies reward
  discrimination across theta.

## Artifacts

Default output root:

`D:\CodexDownloads\lunar-path-planning\stage22_theta_aware_sensor_action_space\outputs\path_feedback_batch_xunce_stage22_2_theta_aware_coverage_reward_contract_v1`

Files:

- `xunce-stage22-2-summary.json`
- `xunce-stage22-2-theta-reward-contract-audit.json`
- `xunce-stage22-2-reward-replay.jsonl`
- `xunce-stage22-2-stage21-2-compatibility-audit.json`
- `xunce-stage22-2-next-stage-routing.json`
- `xunce-stage22-2-report.md`
- `xunce-stage22-2-manifest.json`

## Result

The current Stage22.2 run passed. It replayed 5000 sampled Stage22.1 viewpoint
rows; all reward rows used theta-aware provenance, no point-only fallback was
used, and all 625 same-point groups where theta changed coverage had
discriminating reward values. The route is
`run_stage22_3_theta_aware_ppo_collector_smoke`.

## Boundaries

Stage22.2 does not run PPO, publish checkpoints, replace the default policy,
connect executor, start canary, modify the network, introduce continuous theta
policy, or change default A*.

## Verification

```powershell
python -m pytest tests\test_xunce_stage22_2_theta_aware_coverage_reward_contract.py tests\test_xunce_stage22_1_theta_aware_candidate_viewpoint_generation.py tests\test_xunce_stage21_2_coverage_first_ppo_reward_contract.py tests\test_xunce_stage21_3_ppo_batch_validation.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage22-2
python -m py_compile scripts\run_xunce_stage22_2_theta_aware_coverage_reward_contract.py scripts\run_xunce_stage21_2_coverage_first_ppo_reward_contract.py scripts\run_xunce_stage21_3_ppo_batch_validation.py
python scripts\run_stage.py --stage xunce-stage22-2-theta-aware-coverage-reward-contract --dry-run
```
