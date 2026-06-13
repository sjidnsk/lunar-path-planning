# Quasi-Real Iterative PPO Mini-Loop Stability v1

## Summary

This stage follows the current-HEAD quasi-real evidence rebaseline. The previous
readiness status is `limited_quasi_real_ppo_update_smoke_evaluated`; this stage
checks whether three consecutive tiny quasi-real PPO updates remain stable.

It is still a local experimental stability check. It is not formal PPO rollout,
checkpoint publication, default-policy replacement, or a performance claim.

## Artifacts

- `configs/quasi_real_iterative_ppo_mini_loop_stability_v1.json`
- `scripts/run_quasi_real_iterative_ppo_mini_loop_stability.py`
- `scripts/run_quasi_real_iterative_ppo_mini_loop_stability.sh`
- `scripts/run_quasi_real_iterative_ppo_mini_loop_stability_closure.sh`
- `outputs/path_feedback_batch_quasi_real_iterative_ppo_mini_loop_stability_v1/`

Output files:

- `quasi-real-iterative-ppo-mini-loop-stability-summary.json`
- `quasi-real-iterative-ppo-mini-loop-rounds.jsonl`
- `quasi-real-iterative-ppo-mini-loop-drift-report.json`
- `quasi-real-iterative-ppo-mini-loop-rejection-report.json`

The summary keeps schema `iterative-ppo-mini-loop-stability-summary/v1` so the
existing readiness state `iterative_ppo_mini_loop_stability_evaluated` can be
used without adding a competing quasi-real status.

## Loop Contract

Round 0 starts from
`outputs/path_feedback_batch_quasi_real_teacher_distillation_candidate_v1/`.
Each round runs:

1. quasi-real guarded teacher-following with the round base candidate
2. quasi-real PPO collector dry-run from the teacher-following decisions
3. limited quasi-real PPO update from the collector
4. quasi-real/generated sequential compatibility diagnosis
5. generated sequential gate metric/accounting audit
6. generated sequential long-horizon teacher-skill contract alignment

The updated experimental candidate from one round becomes the base candidate for
the next round.

## Acceptance Gates

- `round_count=3`, `failed_round_count=0`, `stability_passed=true`
- every round optimizer train transition count `>=24`, target `36`
- validation/test/fallback/gated diagnostic transitions never enter optimizer
- old `log_prob` and `value` reconstruction max abs error `<=1e-4`
- loss, gradient, reward, return, and advantage are finite
- `parameter_l2_delta>0`
- `abs(approx_kl)<=0.25`
- `max_grad_norm_after_clip<=1.0`
- `cumulative_parameter_l2_delta<=0.05`
- quasi-real teacher-following passes with `teacher_agreement_rate>=0.9`
- quasi-real collector passes with trainable count `>=24`
- compatibility and accounting summaries pass
- controlled path/risk regression remains `0`
- long-horizon contract passes with
  `verdict=long_horizon_teacher_skill_contract_aligned`
- no checkpoint publication, default-policy replacement, performance claim, or
  formal-training-ready claim
- readiness advances to `iterative_ppo_mini_loop_stability_evaluated`

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_quasi_real_iterative_ppo_mini_loop_stability.py \
  tests/test_limited_quasi_real_ppo_update_smoke.py \
  tests/test_generated_sequential_long_horizon_teacher_skill_contract_alignment.py \
  tests/test_policy_training_readiness_review.py

PYTHON=$P bash scripts/run_quasi_real_iterative_ppo_mini_loop_stability_closure.sh

PYTHON=$P bash scripts/run_policy_training_readiness_review.sh \
  --batch-root outputs/path_feedback_batch_guarded_ppo_rollout_clean_src_v1 \
  --config configs/policy_training_readiness_review_v1.json \
  --iterative-ppo-mini-loop-stability-summary \
    outputs/path_feedback_batch_quasi_real_iterative_ppo_mini_loop_stability_v1/quasi-real-iterative-ppo-mini-loop-stability-summary.json \
  --validate-only

git diff --check
```

## Non-Goals

- No formal PPO rollout.
- No training data expansion.
- No checkpoint publication or default-policy replacement.
- No network/action-space/default A* change.
- No distance/path-risk/source-selection gate relaxation.
- No Ackermann-feasible trajectory claim.
- No IRIS/GCS/path-planner diagnostic promotion to training release evidence.
