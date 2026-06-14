# Guarded Formal PPO Training Authorization v1

## Summary

This stage adds the offline authorization layer before the next guarded formal
PPO training run. It reads the existing formal PPO evidence chain and turns it
into a training permission package: allowed training input, seeds, budget,
stop conditions, rollback rules, and post-training gates.

It does not run a new PPO optimizer update.

## Inputs

- `outputs/path_feedback_batch_quasi_real_guarded_formal_ppo_preflight_v1/quasi-real-guarded-formal-ppo-preflight-summary.json`
- `outputs/path_feedback_batch_quasi_real_guarded_formal_ppo_rollout_canary_v1/quasi-real-guarded-formal-ppo-rollout-canary-summary.json`
- `outputs/path_feedback_batch_quasi_real_guarded_formal_ppo_stability_holdout_validation_v1/quasi-real-guarded-formal-ppo-stability-holdout-validation-summary.json`
- `outputs/path_feedback_batch_quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout_v1/quasi-real-guarded-formal-ppo-candidate-selection-long-horizon-holdout-summary.json`
- `outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_decision_review_v1/selected-formal-ppo-candidate-promotion-decision-review-summary.json`
- `outputs/path_feedback_batch_guarded_experimental_policy_staged_release_canary_preflight_v1/guarded-experimental-policy-staged-release-canary-preflight-summary.json`

Inputs must be passed, have empty `reason_codes`, and not claim checkpoint
publication, default-policy replacement, performance improvement, or formal
training readiness.

## Outputs

- `configs/guarded_formal_ppo_training_authorization_v1.json`
- `scripts/run_guarded_formal_ppo_training_authorization.py/.sh`
- `scripts/run_guarded_formal_ppo_training_authorization_closure.sh`
- `tests/test_guarded_formal_ppo_training_authorization.py`
- `outputs/path_feedback_batch_guarded_formal_ppo_training_authorization_v1/`
- `formal-ppo-training-authorization-summary.json`
- `formal-ppo-training-input-audit.json`
- `formal-ppo-training-budget-manifest.json`
- `formal-ppo-training-seed-plan.json`
- `formal-ppo-training-stop-condition-manifest.json`
- `formal-ppo-training-rollback-manifest.json`
- `formal-ppo-post-training-gate-plan.json`
- `formal-ppo-training-authorization-readiness-validate-only.json`
- `formal-ppo-training-authorization-report.md`

## Authorization Rules

Training input is authorized only from the formal stability/holdout evidence.
The authorized count must be at least 684 trainable transitions, with matching
optimizer count and unique trainable context count. Validation/test, fallback,
diagnostic, raw-probe rejection, and non-empty gate-reason evidence cannot be
authorized for training.

The training plan is multi-seed and small-step: seeds `[0,1,2,3,4]`, one epoch,
learning rate no larger than `1e-5`, PPO clip ratio `0.2`, discount factor
`0.99`, and max grad norm `1.0`. The checkpoint scope remains experimental.

## Acceptance Gates

- summary `status=passed`, `reason_codes=[]`
- `authorization_verdict=authorized_for_guarded_formal_ppo_training_run`
- `authorized_trainable_transition_count>=684`
- validation/test/fallback/diagnostic trainable counts are 0
- missing observation/log_prob/value, invalid mask, and non-finite reward/return/advantage counts are 0
- controlled safety/contract/path-risk/source-selection regression counts are 0
- seed count is at least 5
- budget, seed, stop-condition, rollback, and post-training gate plans all pass
- readiness advances to `guarded_formal_ppo_training_authorized`

## Non-Goals

- Do not run a new PPO update.
- Do not start an online canary.
- Do not connect a real executor or robot.
- Do not publish or replace a checkpoint.
- Do not replace the default policy.
- Do not modify network, action space, or default A*.
- Do not relax distance, path-risk, or source-selection gates.
- Do not download new raw data.
- Do not claim Ackermann-feasible trajectory.
- Do not treat IRIS/GCS/path-planner diagnostics as training or release evidence.
- Do not claim policy performance improvement or formal training readiness.
