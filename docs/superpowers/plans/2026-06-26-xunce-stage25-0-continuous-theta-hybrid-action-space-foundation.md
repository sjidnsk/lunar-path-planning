# Stage25.0 Continuous Theta Hybrid Action Space Foundation

## Goal

Complete `continuous_theta_hybrid_action_space_foundation`: upgrade the action
contract from discrete `(x,y,theta_bin)` viewpoints to discrete candidate point
`(x,y)` plus continuous sampled observation heading `theta_rad`.

## Scope

- Keep `x/y` discrete.
- Use a per-candidate Von Mises theta distribution.
- Preserve Stage24 environment contracts:
  - `coverage_source=endpoint_theta_slope_obstacle_los/v1`
  - `path_cost_source=hybrid_astar_pose_path/v1`
  - `max_traversable_slope_deg=30.0`
  - `default_astar_replaced=false`
  - `hybrid_astar_ackermann_feasible_claimed=false`
- Run the bounded offline chain:
  Stage21.1 collector -> Stage21.2 reward -> Stage21.3 batch ->
  Stage21.4 tiny PPO update -> Stage21.5 pre/post eval.

## Implementation Notes

- The full network emits point logits plus per-candidate
  `theta_mu_sin`, `theta_mu_cos`, and `theta_kappa_raw`.
- PPO uses joint log probability:

```text
old_log_prob = old_point_log_prob + old_theta_log_prob
```

- Stage21.3 must reject continuous theta rows when radians/degrees,
  old sampling logits, theta distribution parameters, or split logprob fields
  cannot be recomputed consistently.
- High-fidelity eval must bind selected continuous theta back to the selected
  base candidate before computing coverage and Hybrid A* pose path cost.

## Boundaries

Stage25.0 does not start formal PPO training, publish checkpoints, replace the
default policy, connect an executor, start canary traffic, continuous-ize
`x/y`, replace the default A*, model 3D DEM LOS, or claim final performance.

## Verification

```powershell
python -m pytest tests\test_xunce_stage25_0_continuous_theta_hybrid_action_space_foundation.py tests\test_xunce_stage24_5a_repair_hybrid_path_inference_binding.py tests\test_xunce_stage21_4_tiny_ppo_update_smoke.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage25-0
python -m py_compile scripts\run_xunce_stage25_0_continuous_theta_hybrid_action_space_foundation.py scripts\run_xunce_stage21_1_on_policy_ppo_rollout_collector.py scripts\run_xunce_stage21_3_ppo_batch_validation.py scripts\run_xunce_stage21_4_tiny_ppo_update_smoke.py
python scripts\run_stage.py --stage xunce-stage25-0-continuous-theta-hybrid-action-space-foundation --dry-run
```
