# Stage21.16 Policy Signal Margin And Credit Attribution Plan

Stage21.16 follows Stage21.15, where strong pre/post inference binding was repaired and
720 matched states were available, but PPO still produced tiny probability deltas and no
rank, argmax, selected action, coverage, or AUC change.

The goal is to separate four possible causes:

- update strength is still too small;
- reward and advantage do not separate high coverage-per-cost actions clearly enough;
- the current top action has a large logit/probability margin over the best
  coverage-per-cost action;
- value loss or entropy terms dominate policy-gradient signal.

Implementation:

- Add `scripts/run_xunce_stage21_16_policy_signal_margin_credit_attribution.py`.
- Add `configs/xunce_stage21_16_policy_signal_margin_credit_attribution_v1.json`.
- Register `xunce-stage21-16-policy-signal-margin-credit-attribution`.
- Reuse Stage21.6 for bounded offline PPO smoke combinations.
- Read Stage21.5 pre/post inference with the strong key:
  `scenario_id + step_index + current_cell + covered_cells_hash + candidate_set_hash`.
- Read Stage21.3 reward/advantage artifacts and Stage21.4 loss/gradient artifacts.
- Output update-strength, reward/advantage separation, discrete margin, and
  loss-gradient attribution artifacts.

Boundaries:

- no formal PPO training;
- no checkpoint publication;
- no default-policy replacement;
- no real executor connection;
- no canary traffic;
- no reward target, network, action-space, candidate-generation, or default A* change.

Validation:

```powershell
python -m pytest tests\test_xunce_stage21_16_policy_signal_margin_credit_attribution.py tests\test_xunce_stage21_15_policy_update_signal_strength_calibration.py tests\test_xunce_stage21_6_multi_seed_ppo_pilot.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage21-16
python -m py_compile scripts\run_xunce_stage21_16_policy_signal_margin_credit_attribution.py scripts\run_xunce_stage21_6_multi_seed_ppo_pilot.py scripts\run_xunce_stage21_4_tiny_ppo_update_smoke.py
python scripts\run_stage.py --stage xunce-stage21-16-policy-signal-margin-credit-attribution --dry-run
```
