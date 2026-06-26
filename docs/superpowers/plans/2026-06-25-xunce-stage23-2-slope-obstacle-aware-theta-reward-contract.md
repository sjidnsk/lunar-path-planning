# Stage23.2 Slope-Obstacle-Aware Theta Reward Contract

## Goal

Connect the Stage23.2B platform-aligned 30 degree slope obstacle LOS coverage to
Stage21.2 reward rows and Stage21.3 batch gates.

## Contract

- Coverage source: `endpoint_theta_slope_obstacle_los/v1`.
- Default hard slope threshold: `max_traversable_slope_deg=30.0`.
- `20.0` degrees remains sensitivity/audit only.
- Slope blockers are `slope_blocked_as_obstacle_proxy`, not physical rock or
  crack labels.
- Stage21.2 must use `obstacle_aware_new_visible_cell_count` for reward when
  `slope_obstacle_aware_theta_reward_enabled=true`.
- Stage21.3 must reject point-only reward, old unobstructed theta reward, missing
  platform/slope source hashes, and selected-viewpoint mismatch when
  `require_slope_obstacle_aware_theta_reward_contract=true`.

## Scope

Stage23.2 is reward/batch contract work only. It does not run PPO, publish
checkpoints, replace default policy, connect executor, start canary traffic,
introduce continuous theta, model 3D LOS, modify the network, or change default
A*. Existing Stage23.0 LOS rows may be replayed with total visible counts as a
clearly labeled compatibility proxy; Stage23.3 must prove real collector
`obstacle_aware_new_visible_cell_count` provenance.

## Verification

- `python -m pytest tests\test_xunce_stage23_2_slope_obstacle_aware_theta_reward_contract.py tests\test_xunce_stage23_2b_platform_geometry_sensor_contract_alignment.py tests\test_xunce_stage21_2_coverage_first_ppo_reward_contract.py tests\test_xunce_stage21_3_ppo_batch_validation.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage23-2`
- `python -m py_compile scripts\run_xunce_stage23_2_slope_obstacle_aware_theta_reward_contract.py scripts\run_xunce_stage21_2_coverage_first_ppo_reward_contract.py scripts\run_xunce_stage21_3_ppo_batch_validation.py scripts\xunce_obstacle_aware_theta_sensor_coverage.py`
- `python scripts\run_stage.py --stage xunce-stage23-2-slope-obstacle-aware-theta-reward-contract --dry-run`
