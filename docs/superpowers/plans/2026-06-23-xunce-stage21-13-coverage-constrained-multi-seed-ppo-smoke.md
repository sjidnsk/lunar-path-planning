# Stage 21.13 Coverage-Constrained Multi-Seed PPO Smoke

## Goal

Execute the Stage21.12 recommended Stage21.6 config with the
coverage-constrained reward v2 profile, then decide whether the repaired reward
signal produces actual final coverage or coverage AUC improvement in a
three-seed offline PPO smoke.

## Implementation

- Added `scripts/run_xunce_stage21_13_coverage_constrained_multi_seed_ppo_smoke.py`.
- Added `configs/xunce_stage21_13_coverage_constrained_multi_seed_ppo_smoke_v1.json`.
- Registered `xunce-stage21-13-coverage-constrained-multi-seed-ppo-smoke` in `configs/stage_registry.json`.
- Added tests in `tests/test_xunce_stage21_13_coverage_constrained_multi_seed_ppo_smoke.py`.

The runner is a thin wrapper around Stage21.6. It requires Stage21.12
`status=passed`, copies the Stage21.12 recommended Stage21.6 config, forces
release/executor/canary boundaries to false, writes it as
`xunce-stage21-13-stage21-6-config.json`, and executes Stage21.6 under the short
nested `s6` directory.

## Routing

- Missing or stale Stage21.12 input: `rerun_stage21_13_required_inputs`.
- Boundary violation: `resolve_stage21_13_boundary_rejections`.
- Unstable KL / entropy / gradient: `continue_stage21_9_gradient_normalization_loss_scaling_repair`.
- No observable parameter delta: `calibrate_stage21_policy_update_signal_strength`.
- Stable update but flat coverage or AUC: `repair_stage21_return_advantage_credit_assignment`.
- Positive raw and capped coverage / AUC across all seeds: `scale_stage21_ppo_pilot_scenarios_and_horizon`.

## Boundaries

Stage21.13 may run local offline PPO updates through Stage21.6, but it is still
a smoke. It does not publish checkpoints, replace the default policy, connect an
executor, start canary traffic, or change the network, action space, or default
A*.

## Verification

```powershell
python -m pytest tests\test_xunce_stage21_13_coverage_constrained_multi_seed_ppo_smoke.py tests\test_xunce_stage21_12_coverage_constrained_reward_profile_repair.py tests\test_xunce_stage21_6_multi_seed_ppo_pilot.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage21-13
python -m py_compile scripts\run_xunce_stage21_13_coverage_constrained_multi_seed_ppo_smoke.py scripts\run_xunce_stage21_6_multi_seed_ppo_pilot.py
python scripts\run_stage.py --stage xunce-stage21-13-coverage-constrained-multi-seed-ppo-smoke --dry-run
```

## Current Result

The current run completed three seeds and 240 trainable transitions under
`D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\outputs\path_feedback_batch_xunce_stage21_13_coverage_constrained_multi_seed_ppo_smoke_v1`.
Lineage and execution boundaries passed, but raw and capped final coverage /
coverage AUC deltas were all `0.0`. The next route is
`repair_stage21_return_advantage_credit_assignment`.
