# Guarded Formal PPO Training Run v1

## Summary

This stage executes the first authorized local guarded formal PPO training run.
It consumes the passed authorization package and performs one small full-batch
PPO update for each seed `[0,1,2,3,4]` over the 684 gate-clean train-split
quasi-real transitions. The resulting checkpoints remain experimental
candidates only.

## Inputs

- `outputs/path_feedback_batch_guarded_formal_ppo_training_authorization_v1/formal-ppo-training-authorization-summary.json`
- formal stability/holdout summary referenced by the authorization summary
- trainable steps referenced by the formal stability/holdout summary
- base candidate root `outputs/path_feedback_batch_quasi_real_teacher_distillation_candidate_v1`

The authorization summary must be passed, have empty `reason_codes`, carry
`authorization_verdict=authorized_for_guarded_formal_ppo_training_run`, and not
claim checkpoint publication, default-policy replacement, performance
improvement, or release readiness.

## Outputs

- `configs/guarded_formal_ppo_training_run_v1.json`
- `scripts/run_guarded_formal_ppo_training_run.py/.sh`
- `scripts/run_guarded_formal_ppo_training_run_closure.sh`
- `tests/test_guarded_formal_ppo_training_run.py`
- `outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1/`
- `formal-ppo-training-run-summary.json`
- `formal-ppo-training-run-seed-summaries.jsonl`
- `formal-ppo-training-run-progress.jsonl`
- `formal-ppo-training-run-training-curves.json`
- `formal-ppo-training-run-gate-audit.json`
- `formal-ppo-training-run-rollback-manifest.json`
- `formal-ppo-training-run-readiness-validate-only.json`
- `formal-ppo-training-run-report.md`

## Training Contract

Only train split, gate-clean, PPO-trainable transitions may enter the optimizer.
Validation/test, fallback, diagnostic, raw-probe rejection, non-empty gate
reason, missing observation/log_prob/value, invalid mask, non-finite
reward/return/advantage, or controlled regression evidence is blocked.

The training budget is intentionally small: one epoch, learning rate no larger
than `1e-5`, clip ratio `0.2`, discount factor `0.99`, and max grad norm `1.0`.

## Acceptance Gates

- summary `status=passed`, `reason_codes=[]`
- input authorization status is passed
- `runs_guarded_formal_ppo_training_run=true`
- `runs_new_ppo_update=true`
- `optimizer_train_transition_count=684`
- validation/test/fallback/diagnostic trainable counts are 0
- seed count is at least 5 and every seed passes
- old log_prob/value reconstruction error is `<=1e-4`
- loss, gradient, reward, return, and advantage are finite
- `parameter_l2_delta>0`
- `abs(approx_kl)<=0.25`
- `max_grad_norm_after_clip<=1.0`
- teacher agreement is at least `0.95`
- controlled safety/contract/path-risk/source-selection regression counts are 0
- post-training holdout and canary gates pass
- readiness advances to `guarded_formal_ppo_training_run_evaluated`

## Non-Goals

- Do not start an online canary.
- Do not connect a real executor or robot.
- Do not publish or replace a checkpoint.
- Do not replace the default policy.
- Do not modify network, action space, or default A*.
- Do not relax distance, path-risk, or source-selection gates.
- Do not download new raw data.
- Do not claim Ackermann-feasible trajectory.
- Do not treat IRIS/GCS/path-planner diagnostics as training or release evidence.
- Do not claim deployable policy performance or release readiness.
