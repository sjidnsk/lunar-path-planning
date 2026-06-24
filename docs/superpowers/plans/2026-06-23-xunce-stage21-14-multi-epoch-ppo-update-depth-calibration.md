# Stage21.14 Multi-Epoch PPO Update Depth Calibration

## Summary

Stage21.14 tests whether Stage21.13 failed because the PPO update was too shallow
to change discrete candidate-action selection. Stage21.13 had stable gradients
and observable parameter delta, but raw/capped final coverage and coverage AUC
all stayed flat at `0.0`.

The stage keeps the Stage21.12 coverage-constrained reward v2 profile and the
Stage21.9 loss-scale repair. It changes only PPO update depth for bounded smoke
calibration.

## Inputs

- Stage21.13 root:
  `D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\outputs\path_feedback_batch_xunce_stage21_13_coverage_constrained_multi_seed_ppo_smoke_v1`
- Stage21.12 root:
  `D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\outputs\path_feedback_batch_xunce_stage21_12_coverage_constrained_reward_profile_repair_v1`
- Stage21.12 recommended Stage21.6 config.
- Stage21.10/21.9 repaired Stage21.4 config.
- Reward profile: `configs/xunce_stage21_coverage_constrained_ppo_reward_profile_v2.json`.

## Runner

- Script: `scripts/run_xunce_stage21_14_multi_epoch_ppo_update_depth_calibration.py`
- Config: `configs/xunce_stage21_14_multi_epoch_ppo_update_depth_calibration_v1.json`
- Stage registry id: `xunce-stage21-14-multi-epoch-ppo-update-depth-calibration`

The runner orchestrates existing Stage21.6 and does not reimplement PPO. For
each combo it writes a dedicated Stage21.4 config and points the generated
Stage21.6 config to that file.

## Sweep

Fixed scope:

- Seeds: `2101, 2102, 2103`
- Scenarios: `8`
- Rollout steps: `10`
- Candidate count: `36`
- Proposal pool: `288`
- `loss_scale=0.25`
- `value_loss_coefficient=0.1`
- `advantage_clip_abs=5.0`
- `normalize_minibatch_advantages=true`
- Stage21.4 training clip: `max_grad_norm=1.0`
- Stage21.6 pre-clip audit gate: `25.0`

Priority combos:

1. `e2_lr2e-6_c0p2`
2. `e4_lr2e-6_c0p2`
3. `e8_lr2e-6_c0p2`
4. `e4_lr5e-6_c0p2`
5. `e8_lr5e-6_c0p2`

Each combo has a `7200` second timeout. Runtime blockers are recorded; results
must not be fabricated.
The default `required_core_combo_count` is `3`, so a normal Stage21.14 run stops
after the three `lr=2e-6` core combinations complete. The `lr=5e-6` combinations
are optional and require an explicit config change before execution.

## Outputs

Output root:
`D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\outputs\path_feedback_batch_xunce_stage21_14_multi_epoch_ppo_update_depth_calibration_v1`

Artifacts:

- `xunce-stage21-14-summary.json`
- `xunce-stage21-14-sweep-results.jsonl`
- `xunce-stage21-14-action-rank-shift-audit.json`
- `xunce-stage21-14-recommended-stage21-6-config.json`
- `xunce-stage21-14-next-stage-routing.json`
- `xunce-stage21-14-report.md`
- `xunce-stage21-14-manifest.json`

## Routing

- Missing or untrusted inputs: `rerun_stage21_14_required_inputs`
- Boundary flags open: `resolve_stage21_14_boundary_rejections`
- All combos numerically unstable: `continue_stage21_9_gradient_normalization_loss_scaling_repair`
- Stable but probability/rank/argmax still barely move:
  `calibrate_stage21_policy_update_signal_strength`
- Probability/rank/argmax move but trajectory coverage/AUC stays flat:
  `repair_stage21_return_advantage_credit_assignment`
- Raw and capped coverage/AUC improve with no worst-seed regression:
  `scale_stage21_ppo_pilot_scenarios_and_horizon`

## Boundaries

Stage21.14 is offline smoke calibration only. It does not start formal PPO
training, publish checkpoints, replace the default policy, connect a real
executor, start canary traffic, or change reward target, network, action space,
candidate generation, or default A*.
