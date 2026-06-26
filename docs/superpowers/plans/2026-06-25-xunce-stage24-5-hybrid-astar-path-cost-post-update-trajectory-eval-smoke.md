# Stage24.5 Hybrid A* Path-Cost Post-Update Trajectory Eval Smoke

## Goal

Stage24.5 wraps Stage21.5 to compare the Stage24.4 source checkpoint and the
Stage24.4 experimental-only checkpoint on the same bounded trajectory
evaluation. The evaluation must keep:

- `coverage_source=endpoint_theta_slope_obstacle_los/v1`
- `path_cost_source=hybrid_astar_pose_path/v1`
- `max_traversable_slope_deg=30.0`
- `default_astar_replaced=false`
- `hybrid_astar_ackermann_feasible_claimed=false`

The stage answers whether the offline update starts to change selected
`(x,y,theta)` actions and whether coverage, AUC, path cost, or
coverage-per-100m move in the bounded smoke. It is not a performance claim.

## Implementation

- Add `scripts/run_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke.py`.
- Add `configs/xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke_v1.json`.
- Add `tests/test_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke.py`.
- Register `xunce-stage24-5-hybrid-astar-path-cost-post-update-trajectory-eval-smoke`.
- Extend high-fidelity model inference rows with selected Hybrid A* path-cost
  provenance so Stage24.5 can audit pre/post binding.

## Required Audits

- Stage24.4 input passed and routed to Stage24.5.
- Checkpoint exists, reload passed, experimental-only metadata is true, and
  source checkpoint lineage is consistent.
- Pre/post inference rows have the strong key:
  `scenario_id + step_index + current_cell + covered_cells_hash + candidate_set_hash`.
- Pre/post rows include selected viewpoint/theta, action probabilities/logits,
  Hybrid A* path cost, pose path hash, legacy grid cost, grid-vs-hybrid delta,
  and boundary flags.
- Hard risk, mask violation, path planning failure, and open-grid fallback stay
  zero.

## Routing

- Missing inputs or incomplete Stage21.5 execution:
  `rerun_stage24_5_required_inputs`
- Missing Hybrid path-cost inference provenance or strong join:
  `repair_stage24_5_hybrid_path_inference_binding`
- Safety regression:
  `repair_stage24_5_hybrid_path_eval_safety_regression`
- Probability and viewpoint unchanged:
  `repair_stage24_hybrid_path_policy_update_signal_strength`
- Probability changed but viewpoint/theta unchanged:
  `calibrate_stage24_hybrid_path_discrete_margin_crossing`
- Viewpoint/theta changed but coverage/AUC not improved:
  `repair_stage24_hybrid_path_credit_assignment`
- Coverage/AUC improved without worst-case regression:
  `run_stage24_6_hybrid_astar_path_cost_multi_seed_ppo_pilot`

## Verification

```powershell
python -m pytest tests\test_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke.py tests\test_xunce_stage24_4_hybrid_astar_path_cost_ppo_update_smoke.py tests\test_xunce_stage21_5_post_update_offline_trajectory_evaluation.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage24-5
python -m py_compile scripts\run_xunce_stage24_5_hybrid_astar_path_cost_post_update_trajectory_eval_smoke.py scripts\run_xunce_stage21_5_post_update_offline_trajectory_evaluation.py
python scripts\run_stage.py --stage xunce-stage24-5-hybrid-astar-path-cost-post-update-trajectory-eval-smoke --dry-run
```
