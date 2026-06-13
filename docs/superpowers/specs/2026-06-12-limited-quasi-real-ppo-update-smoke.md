# Limited Quasi-Real PPO Update Smoke v1

## Summary

This stage implements the local quasi-real PPO update smoke after
`ppo_rollout_collector_dry_run_evaluated`. It consumes the 36 train-split
PPO-trainable transitions from
`outputs/path_feedback_batch_quasi_real_ppo_collector_dry_run_v1/`, performs one
small full-batch PPO update from the same experimental checkpoint, and reruns
post-update generated and quasi-real gates.

The implementation is present, but the closure is currently blocked by the
generated sequential post-update gate. Readiness must not advance to
`limited_quasi_real_ppo_update_smoke_evaluated` until that blocker is resolved
or the acceptance contract is explicitly narrowed.

## Artifacts

- Config: `configs/limited_quasi_real_ppo_update_smoke_v1.json`
- Runner: `scripts/run_limited_quasi_real_ppo_update_smoke.py/.sh`
- Closure: `scripts/run_limited_quasi_real_ppo_update_smoke_closure.sh`
- Output root: `outputs/path_feedback_batch_limited_quasi_real_ppo_update_smoke_v1/`
- Summary: `limited-quasi-real-ppo-update-smoke-summary.json`

## Implemented Contract

- Reuses existing `limited-ppo-update-smoke-summary/v1` for PPO math,
  diagnostics, training curves, and experimental checkpoint output.
- Adds `limited-quasi-real-ppo-update-smoke-summary/v1` as the quasi-real wrapper
  summary consumed by readiness.
- Makes the generic PPO smoke trainable filter config-driven while preserving
  the generated default of `controlled_choice_source=policy`.
- Quasi-real config allows optimizer input only when:
  - `split=train`
  - `ppo_trainable=true`
  - `controlled_choice_source in {"policy_teacher_aligned", "policy_safe_disagreement"}`
  - `gate_reason_codes=[]`
- Validation/test, teacher fallback, unsafe/gated disagreement, `none`, and
  `not_scored` samples remain diagnostic-only.

## Current Evidence

The PPO update path itself is clean:

- `input_ppo_trainable_transition_count=36`
- `optimizer_train_transition_count=36`
- `old_log_prob_max_abs_error=0`
- `old_value_max_abs_error=0`
- `parameter_l2_delta≈4.37e-4`
- `approx_kl≈7.7e-5`
- `max_grad_norm_after_clip≈0.95`

Post-update quasi-real gates pass:

- teacher-following: `status=passed`, `teacher_agreement_rate=1.0`
- quasi-real collector: `status=passed`, `ppo_trainable_transition_count=36`,
  `diagnostic_transition_count=72`

Post-update generated sequential gate fails:

- `multi_step_accepted_episode_count_below_threshold`
- `family_with_multi_step_accepted_episode_count_below_threshold`
- `canary_rejected_policy_choice_count_above_threshold`
- `cumulative_path_cost_regression_count_above_threshold`
- `cumulative_risk_regression_count_above_threshold`

A `learning_rate=1e-8` probe produced the same generated sequential failure,
so the blocker is not explained by an overly large single PPO step.

## Readiness Rule

Readiness has a new optional input:

```bash
--limited-quasi-real-ppo-update-smoke-summary \
  outputs/path_feedback_batch_limited_quasi_real_ppo_update_smoke_v1/limited-quasi-real-ppo-update-smoke-summary.json
```

The new status is `limited_quasi_real_ppo_update_smoke_evaluated`, but it should
be emitted only when the wrapper summary passes and all post-update gates are
clean. Current evidence must remain blocked because the generated sequential
gate failed.

## Next Step

Run `Quasi-Real / Generated Sequential Contract Compatibility Diagnosis` before
attempting iterative PPO or guarded rollout. The diagnosis must decide whether
the generated sequential failure is a policy behavior problem to fix, a stale
generated-gate provenance issue, or a contract mismatch that requires a narrower
quasi-real-only acceptance boundary.

Non-goals remain unchanged: no checkpoint publication, no default policy
replacement, no network/action-space/default-A* changes, no distance/path-risk
or source-selection gate relaxation, no Ackermann-feasible trajectory claim, and
no policy performance or formal-training-ready claim.
