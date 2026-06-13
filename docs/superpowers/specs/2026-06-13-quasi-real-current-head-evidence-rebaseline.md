# Quasi-Real Current-HEAD Evidence Rebaseline v1

## Summary

This stage refreshes the quasi-real training-readiness evidence after the
`cc0f627` commit. The previous outputs were behaviorally useful but carried
stale git provenance, so readiness validate-only stopped on provenance blockers.
The stage is an evidence rebaseline, not a new training stage.

## Scope

- Re-run the quasi-real PPO collector dry-run.
- Re-run the limited quasi-real PPO update smoke.
- Re-run quasi-real/generated sequential compatibility diagnosis.
- Re-run generated sequential gate metric/accounting audit.
- Re-run generated sequential long-horizon teacher-skill contract alignment.
- Re-run policy training readiness review with all post-update summaries wired
  explicitly.
- Update README and architecture documentation with the current-HEAD evidence.

## Current-HEAD Evidence

The rebaseline results are:

- quasi-real collector: `status=passed`, `ppo_trainable_transition_count=36`,
  `diagnostic_transition_count=72`
- limited quasi-real PPO smoke: optimizer consumed 36 trainable transitions,
  `old_log_prob_max_abs_error=0.0`, `old_value_max_abs_error=0.0`,
  `parameter_l2_delta=0.00043718616939168335`,
  `approx_kl=0.00007694711530348286`
- compatibility diagnosis: `status=passed`,
  `diagnosis_verdict=pre_existing_generated_sequential_contract_mismatch`,
  `failed_step_count=6`
- accounting audit: `status=passed`, `legacy_mismatch_count=6`,
  `raw_policy_path_cost_regression_count=6`,
  `raw_policy_risk_regression_count=2`,
  `controlled_path_cost_regression_count=0`,
  `controlled_risk_regression_count=0`
- long-horizon contract: `status=passed`,
  `verdict=long_horizon_teacher_skill_contract_aligned`,
  `teacher_equivalent_episode_count=36`,
  `beyond_teacher_episode_count=15`,
  `dominated_raw_choice_count=6`,
  `controlled_regression_episode_count=0`
- readiness: `status=passed`, `reason_codes=[]`,
  `training_readiness_status=limited_quasi_real_ppo_update_smoke_evaluated`,
  `training_blockers=[]`

## Artifacts

- `outputs/path_feedback_batch_quasi_real_ppo_collector_dry_run_v1/`
- `outputs/path_feedback_batch_limited_quasi_real_ppo_update_smoke_v1/`
- `outputs/path_feedback_batch_quasi_real_generated_sequential_contract_compatibility_diagnosis_v1/`
- `outputs/path_feedback_batch_generated_sequential_gate_metric_accounting_audit_v1/`
- `outputs/path_feedback_batch_generated_sequential_long_horizon_teacher_skill_contract_alignment_v1/`
- `outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1/policy-training-readiness-review-summary.json`

## Interpretation

The strict generated sequential raw takeover gate still fails, but accounting
shows no controlled rollout path/risk regression. Long-horizon teacher-skill
alignment proves the controlled multi-step behavior is teacher-equivalent or
better under the current contract. Therefore the limited quasi-real PPO update
smoke can be treated as evaluated, while checkpoint publication and formal
training readiness remain out of scope.

## Next Stage

The next stage is `Quasi-Real Iterative PPO Mini-Loop Stability v1`: repeat the
small quasi-real PPO update over a tightly bounded mini-loop and prove that
teacher-skill alignment, no controlled regression, finite optimizer metrics, and
no publication flags remain stable across iterations.

## Non-Goals

- No iterative PPO in this rebaseline stage.
- No new training data expansion.
- No checkpoint publication or default-policy replacement.
- No network/action-space/default A* change.
- No distance/path-risk/source-selection gate relaxation.
- No Ackermann-feasible trajectory claim.
- No IRIS/GCS/path-planner diagnostic promotion to training release evidence.
