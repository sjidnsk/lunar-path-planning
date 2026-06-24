# Stage21.19 Policy Update Signal Source Repair

## Goal

Complete `repair_stage21_policy_update_signal_source` after Stage21.18 showed
that three stable iterative PPO rounds still produced very small action
probability movement and no coverage/AUC improvement.

## Scope

- Audit action/log-prob binding, sampling masks, hard-risk masks, and
  `candidate_set_hash`.
- Audit Stage21.4 ratio, clip fraction, advantage scale, and policy loss
  direction.
- Audit policy/value/entropy gradient components and checkpoint parameter delta
  by module.
- Audit xunce-only strong-state pre/post inference movement.
- Run one bounded policy-path diagnostic Stage21.4 smoke.

## Inputs

- Stage21.18 root:
  `D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\outputs\path_feedback_batch_xunce_stage21_18_iterative_ppo_learning_curve_audit_v1`
- Stage21.18 recommended Stage21.6 config:
  `xunce-stage21-18-recommended-stage21-6-config.json`
- Stage21.18 per-round Stage21.1/21.3/21.4/21.5 artifacts.

## Outputs

Default root:

`D:\CodexDownloads\lunar-path-planning\stage21_pure_ppo_coverage_first\outputs\path_feedback_batch_xunce_stage21_19_policy_update_signal_source_repair_v1`

Artifacts:

- `xunce-stage21-19-summary.json`
- `xunce-stage21-19-action-logprob-binding-audit.json`
- `xunce-stage21-19-policy-gradient-path-audit.json`
- `xunce-stage21-19-ratio-clip-advantage-audit.json`
- `xunce-stage21-19-candidate-logit-sensitivity-audit.json`
- `xunce-stage21-19-diagnostic-smoke-summary.json`
- `xunce-stage21-19-recommended-stage21-6-config.json`
- `xunce-stage21-19-next-stage-routing.json`
- `xunce-stage21-19-report.md`
- `xunce-stage21-19-manifest.json`

## Result

Stage21.19 found that the PPO signal source is not the primary blocker:

- `old_log_prob` recompute max error is about `5.17e-7`.
- Action/mask/candidate-set binding passes.
- Policy-head and candidate-encoder parameters both move.
- Strict xunce-only strong binding is available with no duplicate xunce keys.
- A few selected actions change, but final coverage and coverage AUC still do
  not improve.

The next route is:

`repair_stage21_return_advantage_credit_assignment`

## Boundaries

Stage21.19 is offline diagnostic evidence. It does not authorize checkpoint
publication, default-policy replacement, executor connection, canary traffic,
reward-target changes, network changes, action-space changes,
candidate-generation changes, or default A* changes.
