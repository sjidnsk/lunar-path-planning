# Stage 21.15 Policy Update Signal Strength Calibration

## Goal

Stage21.15 repairs the Stage21.14 evidence gap: pre/post inference rows lacked
strong state identity, so Stage21.14 could not prove whether PPO changed action
probabilities, ranks, or argmax on the same state and candidate set.

## Scope

- Add the high-fidelity inference contract fields required for strong binding:
  `current_cell`, `current_cell_before`, `covered_cells_hash`,
  `candidate_set_hash`, `selected_action_index`, `selected_rank`,
  `selected_probability`, `action_probs`, `logits`, `masked_logits`, `value`,
  `finite_outputs`, and `latency_ms`.
- Add `scripts/run_xunce_stage21_15_policy_update_signal_strength_calibration.py`.
- Add `configs/xunce_stage21_15_policy_update_signal_strength_calibration_v1.json`.
- Register `xunce-stage21-15-policy-update-signal-strength-calibration`.
- Keep the PPO implementation inside Stage21.6; Stage21.15 only orchestrates
  bounded signal calibration sweeps and reads their artifacts.

## Audit Rules

Strong state binding uses only:

```text
scenario_id + step_index + current_cell + covered_cells_hash + candidate_set_hash
```

Weak joins cannot be used to claim policy movement. Missing fields, duplicate
strong keys, or missing post rows route to
`repair_stage21_5_inference_binding_contract`.

## Routing

- Tiny probability shift: `increase_stage21_policy_update_signal_strength`
- Probability shift without rank/argmax movement:
  `calibrate_stage21_discrete_action_margin_crossing`
- Rank/argmax movement without coverage/AUC movement:
  `repair_stage21_return_advantage_credit_assignment`
- Raw and capped coverage/AUC improvement with no worst-seed regression:
  `scale_stage21_ppo_pilot_scenarios_and_horizon`

## Boundaries

Stage21.15 does not publish checkpoints, replace the default policy, connect an
executor, start canary traffic, change reward targets, change network/action
space, change candidate generation, or modify default A*.

## Verification

```powershell
python -m pytest tests\test_xunce_stage21_15_policy_update_signal_strength_calibration.py tests\test_xunce_stage21_14_multi_epoch_ppo_update_depth_calibration.py tests\test_xunce_stage21_6_multi_seed_ppo_pilot.py tests\test_xunce_high_fidelity_exploration_coverage_comparison.py tests\test_platform_stage_runner.py -q --basetemp outputs\pytest-stage21-15
python -m py_compile scripts\run_xunce_stage21_15_policy_update_signal_strength_calibration.py scripts\run_xunce_stage21_6_multi_seed_ppo_pilot.py scripts\run_xunce_high_fidelity_exploration_coverage_comparison.py
python scripts\run_stage.py --stage xunce-stage21-15-policy-update-signal-strength-calibration --dry-run
```
