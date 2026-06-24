# Stage22.4 Theta-Aware PPO Update Smoke

## Summary

Stage22.4 validates that the real theta-aware PPO batch produced by Stage22.3 can be consumed by the existing Stage21.4 tiny PPO update. It performs one bounded offline update and audits loss, gradients, checkpoint reload, and boundary fields.

This stage does not evaluate trajectories and does not claim coverage improvement. Passing only means the `(x,y,theta)` action contract can reach the PPO update path.

## Inputs

- Stage22.3 root: `D:\CodexDownloads\lunar-path-planning\stage22_theta_aware_sensor_action_space\outputs\path_feedback_batch_xunce_stage22_3_theta_aware_ppo_collector_smoke_v1`
- Stage22.3 Stage21.3 batch root: `<stage22_3_root>\s21_3`
- Stage21.4 base config: `configs/xunce_stage21_4_tiny_ppo_update_smoke_v1.json`
- High-fidelity config: `configs/xunce_high_fidelity_exploration_coverage_comparison_stage18_9_strict_v3.json`

## Outputs

Default output root:

`D:\CodexDownloads\lunar-path-planning\stage22_theta_aware_sensor_action_space\outputs\path_feedback_batch_xunce_stage22_4_theta_aware_ppo_update_smoke_v1`

Artifacts:

- `xunce-stage22-4-summary.json`
- `xunce-stage22-4-stage21-4-config.json`
- `xunce-stage22-4-stage21-4-summary.json`
- `xunce-stage22-4-theta-ppo-batch-update-audit.json`
- `xunce-stage22-4-loss-gradient-audit.json`
- `xunce-stage22-4-checkpoint-boundary-audit.json`
- `xunce-stage22-4-next-stage-routing.json`
- `xunce-stage22-4-report.md`
- `xunce-stage22-4-manifest.json`
- Internal Stage21.4 root: `<output-root>\s21_4`

## Contract Checks

- Stage22.3 summary must be `passed` and route to `run_stage22_4_theta_aware_ppo_update_smoke`.
- Every trainable Stage21.3 row must retain theta-aware reward provenance.
- `action_index` must point to `info.candidate_viewpoints[action_index]`, and that viewpoint must match the top-level selected `candidate_viewpoint`.
- `xunce_batch.action_mask`, `candidate_features`, and `context_features` must share the same viewpoint-level action dimension.
- Selected action must be valid in action, sampling, and hard-risk-clean masks.
- Stage21.4 loss, KL, clip fraction, entropy, gradients, and component grad norms must be finite/readable.
- The experimental checkpoint must exist, reload, and be marked `experimental_only=true`.

## Routing

- Input missing or Stage22.3 not passed: `rerun_stage22_4_required_inputs`
- Theta batch/update contract failure: `repair_stage22_4_theta_batch_update_contract`
- Stage21.4 loss/gradient/KL instability: `repair_stage22_4_theta_ppo_update_stability`
- Checkpoint reload or boundary failure: `repair_stage22_4_theta_checkpoint_reload_boundary`
- Contract complete and Stage21.4 passed: `run_stage22_5_theta_aware_post_update_trajectory_eval_smoke`

## Verification

```powershell
python -m pytest tests\test_xunce_stage22_4_theta_aware_ppo_update_smoke.py tests\test_xunce_stage22_3_theta_aware_ppo_collector_smoke.py tests\test_xunce_stage21_4_tiny_ppo_update_smoke.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage22-4
python -m py_compile scripts\run_xunce_stage22_4_theta_aware_ppo_update_smoke.py scripts\run_xunce_stage21_4_tiny_ppo_update_smoke.py
python scripts\run_stage.py --stage xunce-stage22-4-theta-aware-ppo-update-smoke --dry-run
```

## Non-Goals

Stage22.4 does not run Stage21.5, perform multi-seed PPO, publish checkpoints, replace default policy, connect executor, start canary, modify network/default A*/candidate generation/reward target, introduce continuous theta, or claim real performance improvement.
