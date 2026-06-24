# Stage21.18 Iterative PPO Learning Curve Audit

## Goal

Stage21.18 验证 PPO 是否需要多轮稳定小步更新，才能把 Stage21.17 中很小的动作概率变化累积成 rank、argmax、整场覆盖率或 coverage AUC 的真实变化。该阶段不继续盲目放大单次 policy loss。

## Inputs

- Stage21.17 root:
  `D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\outputs\path_feedback_batch_xunce_stage21_17_policy_signal_amplification_value_balance_v1`
- Stage21.17 recommended Stage21.6 config.
- Stage21.17 recommended Stage21.4 config.
- Coverage-constrained reward profile v2 and existing Stage21.1 to Stage21.6 runners.

## Implementation

- Runner: `scripts/run_xunce_stage21_18_iterative_ppo_learning_curve_audit.py`
- Config: `configs/xunce_stage21_18_iterative_ppo_learning_curve_audit_v1.json`
- Registry id: `xunce-stage21-18-iterative-ppo-learning-curve-audit`
- Output root:
  `D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\outputs\path_feedback_batch_xunce_stage21_18_iterative_ppo_learning_curve_audit_v1`

Stage21.18 uses a per-seed checkpoint chain. Round `N` for seed `2101` can only feed round `N+1` for seed `2101`; it cannot use a checkpoint from seed `2102` or `2103`. Each round generates paired Stage21.1 and Stage21.4 configs that point to the same source checkpoint. The lineage audit verifies source sha, experimental checkpoint sha, reload status, and `experimental_only=true`.

## Outputs

- `xunce-stage21-18-summary.json`
- `xunce-stage21-18-round-results.jsonl`
- `xunce-stage21-18-checkpoint-lineage.json`
- `xunce-stage21-18-action-signal-trend.json`
- `xunce-stage21-18-coverage-trend.json`
- `xunce-stage21-18-recommended-stage21-6-config.json`
- `xunce-stage21-18-next-stage-routing.json`
- `xunce-stage21-18-report.md`
- `xunce-stage21-18-manifest.json`

## Routing

- Missing or untrusted inputs: `rerun_stage21_18_required_inputs`
- Checkpoint chain or reload failure: `repair_stage21_iterative_checkpoint_lineage`
- Boundary flag violation: `resolve_stage21_18_boundary_rejections`
- KL, entropy, or gradient instability: `continue_stage21_9_gradient_normalization_loss_scaling_repair`
- Multi-round probability still barely moves: `repair_stage21_policy_update_signal_source`
- Probability accumulates but rank/argmax does not change: `calibrate_stage21_discrete_action_margin_crossing`
- Rank/argmax changes but coverage/AUC does not: `repair_stage21_return_advantage_credit_assignment`
- Coverage/AUC improves and worst seed does not regress: `scale_stage21_ppo_pilot_scenarios_and_horizon`

## Verification

```powershell
python -m pytest tests\test_xunce_stage21_18_iterative_ppo_learning_curve_audit.py tests\test_xunce_stage21_17_policy_signal_amplification_value_balance.py tests\test_xunce_stage21_6_multi_seed_ppo_pilot.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage21-18
python -m py_compile scripts\run_xunce_stage21_18_iterative_ppo_learning_curve_audit.py scripts\run_xunce_stage21_6_multi_seed_ppo_pilot.py scripts\run_xunce_stage21_4_tiny_ppo_update_smoke.py
python scripts\run_stage.py --stage xunce-stage21-18-iterative-ppo-learning-curve-audit --dry-run
```

## Non-Goals

Stage21.18 does not start formal PPO training, publish a checkpoint, replace the default policy, connect a real executor, start canary traffic, change the reward target, change the network, change action space, change candidate generation, or change default A*.
