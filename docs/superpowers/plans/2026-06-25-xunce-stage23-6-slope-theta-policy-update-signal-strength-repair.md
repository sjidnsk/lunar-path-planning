# Stage23.6 Slope-Theta Policy Update Signal Strength Repair

## Goal

Complete Stage23.6: `repair_stage23_slope_theta_policy_update_signal_strength`.
The stage diagnoses why the 30 degree platform-aligned,
slope-obstacle-aware theta PPO update produced almost no action probability
change in Stage23.5A, and finds a stable offline update configuration that can
make real `(x,y,theta)` choices move.

## Current Evidence

- Stage23.5A repaired the high-resolution ROI input shortage: `slice_count=2`,
  `scenario_count=2`, and Stage23.5 is no longer a partial diagnostic.
- Stage23.5 strong join is available, but
  `mean_abs_probability_delta` is about `5.75e-7`.
- Selected viewpoint, theta, and action did not change; coverage and AUC deltas
  are zero.
- Stage23.4 consumed the slope-obstacle-aware theta batch, but only had 4
  trainable transitions under `epochs=1`, `learning_rate=2e-6`.
- Stage23.4 gradient attribution was value-loss dominated: value/policy
  gradient ratio was about 101.

## Scope

- Reuse existing Stage23.3 collector, Stage23.4 PPO update smoke, and Stage23.5
  post-update evaluator.
- Do not rewrite PPO, reward, trajectory evaluation, candidate generation, or
  default A*.
- Keep `coverage_source=endpoint_theta_slope_obstacle_los/v1` and
  `max_traversable_slope_deg=30.0`.
- Keep all checkpoints experimental-only; do not publish or replace default
  policy.

## Implementation Tasks

- Add `scripts/run_xunce_stage23_6_slope_theta_policy_update_signal_strength_repair.py`.
- Add `configs/xunce_stage23_6_slope_theta_policy_update_signal_strength_repair_v1.json`.
- Add `tests/test_xunce_stage23_6_slope_theta_policy_update_signal_strength_repair.py`.
- Register stage id
  `xunce-stage23-6-slope-theta-policy-update-signal-strength-repair`.
- Update `README.md`, `AGENTS.md`,
  `docs/算法设计与系统架构报告.md`, and the topology-aware coverage spec.

## Execution Design

- Validate Stage23.5A as the required input repair source.
- Run or reuse a Stage23.3 collector pass with `required_scenario_count=2` and
  `rollout_steps=8`; require at least 16 trainable transitions for update
  sweep evidence.
- Run bounded update/eval combos in independent subdirectories:
  - `baseline_current`: `epochs=1`, `lr=2e-6`, `value_coef=0.1`,
    `policy_coef=1.0`, `loss_scale=0.25`.
  - `value_off_depth`: `epochs=4`, `lr=2e-5`, `value_coef=0.0`,
    `policy_coef=1.0`, `loss_scale=0.25`.
  - `policy_amp_depth`: `epochs=8`, `lr=2e-5`, `value_coef=0.02`,
    `policy_coef=2.0`, `loss_scale=0.5`.
- Each combo runs Stage23.4 followed by Stage23.5 using its own Stage23.4 root.

## Required Audits

- Trainable row count and slope/theta reward provenance.
- Action index, viewpoint, action mask, sampling mask, and hard-risk mask
  binding.
- Policy/value/entropy gradient norms and value-to-policy gradient ratio.
- KL, entropy, pre/post clip gradient, parameter delta, checkpoint reload, and
  experimental-only metadata.
- Strong-join probability delta, selected viewpoint/theta/action changes,
  coverage/AUC/path-cost delta, and safety regressions.

## Routing

- Missing or untrusted inputs: `rerun_stage23_6_required_inputs`.
- Collector sample shortage: `expand_stage23_slope_theta_collector_samples`.
- Numeric instability: `repair_stage23_slope_theta_ppo_update_stability`.
- Value loss still dominates: `repair_stage23_policy_value_loss_balance`.
- Probability still too small and action unchanged:
  `increase_stage23_slope_theta_update_strength_or_sample_count`.
- Probability moves but viewpoint/theta remains unchanged:
  `calibrate_stage23_slope_theta_discrete_margin_crossing`.
- Viewpoint/theta changes but coverage/AUC does not improve:
  `repair_stage23_slope_theta_credit_assignment`.
- Coverage/AUC improves and worst scenario does not regress:
  `run_stage23_7_slope_obstacle_aware_theta_multi_seed_ppo_pilot`.

## Verification

```powershell
python -m pytest tests\test_xunce_stage23_6_slope_theta_policy_update_signal_strength_repair.py tests\test_xunce_stage23_5a_repair_required_inputs_high_res_roi_coverage.py tests\test_xunce_stage23_4_slope_obstacle_aware_theta_ppo_update_smoke.py tests\test_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage23-6
python -m py_compile scripts\run_xunce_stage23_6_slope_theta_policy_update_signal_strength_repair.py scripts\run_xunce_stage23_4_slope_obstacle_aware_theta_ppo_update_smoke.py scripts\run_xunce_stage23_5_slope_obstacle_aware_theta_post_update_trajectory_eval_smoke.py scripts\run_xunce_stage21_4_tiny_ppo_update_smoke.py
python scripts\run_stage.py --stage xunce-stage23-6-slope-theta-policy-update-signal-strength-repair --dry-run
```

## Non-Goals

Do not start formal PPO training, publish checkpoints, replace default policy,
connect executor, start canary, modify network/default A*/candidate generation
or reward target, introduce continuous theta, add path-continuous observation,
or implement 3D LOS.
