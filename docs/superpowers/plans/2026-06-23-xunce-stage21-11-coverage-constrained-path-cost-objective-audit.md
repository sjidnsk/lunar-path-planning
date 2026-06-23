# Stage21.11 Coverage-Constrained Path-Cost Objective Audit

## Summary

Stage21.11 addresses the Stage21.10 result where the repaired PPO update became numerically stable but final coverage and coverage AUC did not improve. The stage is audit-only. It does not run PPO, publish checkpoints, replace the default policy, connect an executor, or start canary traffic.

The next objective is expressed as coverage-constrained path-cost optimization: reach or move toward 99% final coverage first, then prefer lower path cost and better coverage-per-cost among actions with comparable coverage progress.

## Inputs

- Stage21.10 root:
  `D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\outputs\path_feedback_batch_xunce_stage21_10_stage21_9_repaired_multi_seed_ppo_pilot_v1`
- Nested Stage21.6 root from Stage21.10 summary, normally `<Stage21.10 root>\s6`
- Per-seed Stage21.3 trainable batch and return/advantage audit
- Per-seed Stage21.4 loss and gradient audit
- Per-seed Stage21.5 pre/post model inference and trajectory evaluation summaries
- `configs/xunce_stage21_coverage_first_ppo_reward_profile_v1.json`

## Outputs

Default output root:

`D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\outputs\path_feedback_batch_xunce_stage21_11_coverage_constrained_path_cost_objective_audit_v1`

Artifacts:

- `xunce-stage21-11-summary.json`
- `xunce-stage21-11-reward-separation-audit.json`
- `xunce-stage21-11-advantage-separation-audit.json`
- `xunce-stage21-11-action-probability-shift-audit.json`
- `xunce-stage21-11-coverage-cost-objective-recommendation.json`
- `xunce-stage21-11-recommended-stage21-6-config.json`
- `xunce-stage21-11-next-stage-routing.json`
- `xunce-stage21-11-report.md`
- `xunce-stage21-11-manifest.json`

## Audit Rules

Reward separation checks whether the reward proxy and selected action align with high coverage-per-cost candidates in the same candidate set.

Advantage separation checks whether normalized advantage has a positive signal for selected actions with better coverage-per-cost.

Probability shift compares pre/post action probabilities on the same state. Strong binding requires `scenario_id`, `step_index`, `candidate_set_hash`, `covered_cells_hash`, and `current_cell`; weaker binding is diagnostic only.

Objective recommendation keeps 99% final coverage as the first gate and treats path cost as a secondary objective under that coverage constraint.

## Routing

- Missing input or untrusted lineage: `rerun_stage21_11_required_inputs`
- Reward not aligned with coverage-per-cost: `repair_stage21_coverage_constrained_reward_profile`
- Advantage not aligned or too flat: `repair_stage21_return_advantage_credit_assignment`
- Probability barely changes: `calibrate_stage21_policy_update_signal_strength`
- Probability changes but trajectory does not: `repair_stage21_post_update_evaluation_binding`
- All audits pass: `run_stage21_12_coverage_constrained_multi_seed_ppo_pilot`

## Verification

```powershell
python -m pytest tests\test_xunce_stage21_11_coverage_constrained_path_cost_objective_audit.py tests\test_xunce_stage21_10_stage21_9_repaired_multi_seed_ppo_pilot.py tests\test_xunce_stage21_6_multi_seed_ppo_pilot.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage21-11
python -m py_compile scripts\run_xunce_stage21_11_coverage_constrained_path_cost_objective_audit.py scripts\run_xunce_stage21_6_multi_seed_ppo_pilot.py scripts\run_xunce_stage21_4_tiny_ppo_update_smoke.py
python scripts\run_stage.py --stage xunce-stage21-11-coverage-constrained-path-cost-objective-audit --dry-run
```

## Non-Goals

- No formal PPO training.
- No checkpoint publication.
- No default-policy replacement.
- No real executor connection.
- No online canary.
- No network, action-space, or default A* changes.
