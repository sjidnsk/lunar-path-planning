# Stage21.17 Policy Signal Amplification Value Balance Plan

## Summary

Stage21.17 addresses the Stage21.16 finding that action-probability movement is
still too small while value loss dominates the policy update gradient. The stage
keeps reward v2, network, action space, candidate generation, and default A*
unchanged. It calibrates Stage21.4 loss balance by lowering
`value_loss_coefficient` and, if needed, increasing a new backward-compatible
`policy_loss_coefficient`.

## Key Changes

- Add `scripts/run_xunce_stage21_17_policy_signal_amplification_value_balance.py`
  and config `configs/xunce_stage21_17_policy_signal_amplification_value_balance_v1.json`.
- Extend Stage21.4 with `policy_loss_coefficient`, default `1.0`, and record it
  in loss, gradient, and summary artifacts.
- Run bounded Stage21.6 sweep combos for value/policy loss balance, one new
  combo per invocation by default.
- Output summary, sweep results, policy-value balance audit, action-signal audit,
  recommended Stage21.6 config, routing, report, and manifest.

## Test Plan

- `python -m pytest tests\test_xunce_stage21_17_policy_signal_amplification_value_balance.py tests\test_xunce_stage21_16_policy_signal_margin_credit_attribution.py tests\test_xunce_stage21_6_multi_seed_ppo_pilot.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage21-17`
- `python -m py_compile scripts\run_xunce_stage21_17_policy_signal_amplification_value_balance.py scripts\run_xunce_stage21_4_tiny_ppo_update_smoke.py scripts\run_xunce_stage21_6_multi_seed_ppo_pilot.py`
- `python scripts\run_stage.py --stage xunce-stage21-17-policy-signal-amplification-value-balance --dry-run`

## Boundaries

Stage21.17 is offline calibration only. It does not publish checkpoints, replace
the default policy, connect an executor, start canary traffic, modify reward
targets, change the network, change action space, change candidate generation,
or change default A*.
