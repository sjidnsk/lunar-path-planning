# Stage24.3 Hybrid A* Path-Cost PPO Collector Smoke

## Goal

Complete `hybrid_astar_path_cost_ppo_collector_smoke`: prove that the real
Stage21.1 -> Stage21.2 -> Stage21.3 PPO data path can produce, reward, and gate
`(x,y,theta)` viewpoint batches with Hybrid A* pose path cost provenance.

## Context

Stage24.2 passed as a replay contract. It proved that Stage21.2 and Stage21.3
can consume `path_cost_source=hybrid_astar_pose_path/v1`, but it did not prove
that the Stage21.1 collector writes those fields from source transitions.

## Scope

- Add a Stage24.3 wrapper that runs Stage21.1, Stage21.2, and Stage21.3.
- Add Stage21.1 opt-in support for candidate-level Hybrid A* path cost arrays.
- Prepare a Stage24.3-local safe high-res source root when inherited ROI starts
  are blocked by the 30 degree slope hard gate; keep the original source root
  untouched and write the repaired start-cell provenance into the copied rows.
- In Hybrid path-cost mode, gate PPO sampling by
  `action_mask & hard_risk_clean_mask & hybrid_astar_reachable_flags` so
  trainable actions always have real Hybrid A* pose path cost provenance.
- Keep `coverage_source=endpoint_theta_slope_obstacle_los/v1`.
- Require `path_cost_source=hybrid_astar_pose_path/v1`.
- Keep default grid A* unchanged.

## Non-Goals

- Do not run PPO update.
- Do not publish checkpoint or replace default policy.
- Do not connect an executor or start canary traffic.
- Do not introduce continuous theta policy, 3D DEM LOS, or performance claims.

## Verification

```powershell
python -m pytest tests\test_xunce_stage24_3_hybrid_astar_path_cost_ppo_collector_smoke.py tests\test_xunce_stage24_2_hybrid_astar_reward_path_cost_contract.py tests\test_xunce_stage21_1_on_policy_ppo_rollout_collector.py tests\test_xunce_stage21_2_coverage_first_ppo_reward_contract.py tests\test_xunce_stage21_3_ppo_batch_validation.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage24-3
python -m py_compile scripts\run_xunce_stage24_3_hybrid_astar_path_cost_ppo_collector_smoke.py scripts\run_xunce_stage21_1_on_policy_ppo_rollout_collector.py scripts\run_xunce_stage21_2_coverage_first_ppo_reward_contract.py scripts\run_xunce_stage21_3_ppo_batch_validation.py scripts\xunce_hybrid_astar_candidate_path_cost.py
python scripts\run_stage.py --stage xunce-stage24-3-hybrid-astar-path-cost-ppo-collector-smoke --dry-run
```
