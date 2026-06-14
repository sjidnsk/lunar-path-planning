# Return-Aligned Guarded PPO Update Smoke v1

## Goal

Run one tiny local PPO optimizer smoke using the passed return-aligned guarded
multi-step collector evidence. This is not formal PPO training. The purpose is
to prove that multi-step discounted return/advantage can be consumed by the PPO
update path without leaking diagnostic rows or changing the policy contract.

## Inputs

- Return-aligned collector:
  `outputs/path_feedback_batch_return_aligned_guarded_multi_step_ppo_collector_expansion_v1/`
- Original guarded collector episodes:
  `outputs/path_feedback_batch_guarded_ppo_rollout_pilot_v1/pilot/collector/`
- Base checkpoint:
  `outputs/path_feedback_batch_quasi_real_iterative_ppo_mini_loop_stability_v1/round-02/update`

The base checkpoint must match the checkpoint that produced the guarded
collector old `log_prob/value`. Using a different checkpoint is a hard
on-policy violation.

## Implementation

Added:

- `configs/return_aligned_guarded_ppo_update_smoke_v1.json`
- `scripts/run_return_aligned_guarded_ppo_update_smoke.py`
- `scripts/run_return_aligned_guarded_ppo_update_smoke.sh`
- `scripts/run_return_aligned_guarded_ppo_update_smoke_closure.sh`
- `tests/test_return_aligned_guarded_ppo_update_smoke.py`

Updated:

- `scripts/run_limited_ppo_update_smoke.py`
- `scripts/run_policy_training_readiness_review.py`
- `tests/test_limited_ppo_update_smoke.py`
- `tests/test_policy_training_readiness_review.py`
- `README.md`
- `docs/算法设计与系统架构报告.md`

The wrapper materializes optimizer input by joining return-aligned rows to the
original guarded rollout transitions on `(episode_id, step_index)`. The original
rollout supplies `PolicyObservation`, action, old `log_prob`, and old `value`.
The return-aligned row supplies `ppo_return` and `ppo_advantage`.

`run_limited_ppo_update_smoke.py` now supports an opt-in config:

```json
{
  "training": {
    "return_source": "transition_info",
    "return_field": "ppo_return",
    "advantage_field": "ppo_advantage"
  }
}
```

Without that config, the old reward-based discounted return behavior is
unchanged.

## Acceptance Gates

- summary `status=passed`, `reason_codes=[]`
- optimizer trains on at least 24 transitions; current count is 30
- validation/test/source-fallback optimizer count is 0
- old `log_prob/value` max abs error is at most `1e-4`
- loss, gradient, reward, return, and advantage are finite
- `parameter_l2_delta>0`
- `abs(approx_kl)<=0.25`
- `max_grad_norm_after_clip<=1.0`
- post-update gates are actually evaluated:
  raw generalization, generated sequential, generated collector, quasi-real
  teacher-following, quasi-real collector, long-horizon teacher-skill contract,
  and return-aligned replay
- post-update controlled regression count is 0
- post-update quasi-real teacher agreement is at least 0.9
- long-horizon verdict is `long_horizon_teacher_skill_contract_aligned`
- return-aligned replay is passed
- readiness status is `return_aligned_guarded_ppo_update_smoke_evaluated`
- `training_blockers=[]`

## Current Evidence

Latest closure:

- optimizer transition count: `30`
- old `log_prob/value` max abs error: `0.0 / 0.0`
- `parameter_l2_delta=0.00042781765692363765`
- `approx_kl=-0.0008484522695653141`
- `post_update_gates_evaluated=true`
- post-update raw generalization effective status: `passed`
- post-update generated sequential strict status: `failed` with the known
  diagnostic reason codes
  `multi_step_accepted_episode_count_below_threshold`,
  `family_with_multi_step_accepted_episode_count_below_threshold`, and
  `canary_rejected_policy_choice_count_above_threshold`
- post-update generated collector: `status=passed`,
  `ppo_trainable_transition_count=30`
- post-update quasi-real teacher-following: `status=passed`,
  `teacher_agreement_rate=1.0`
- post-update quasi-real collector: `status=passed`,
  `ppo_trainable_transition_count=36`
- post-update long-horizon verdict:
  `long_horizon_teacher_skill_contract_aligned`
- post-update return-aligned replay: `status=passed`,
  `trainable_transition_count=30`
- post-update controlled regression count: `0`
- readiness status: `return_aligned_guarded_ppo_update_smoke_evaluated`

An earlier failed attempt used
`outputs/path_feedback_batch_policy_training_cuda_device_support_v1` as the
base checkpoint and correctly failed with `ppo_update_not_on_collector_policy`.
The corrected base is
`outputs/path_feedback_batch_quasi_real_iterative_ppo_mini_loop_stability_v1/round-02/update`.

## Non-Goals

- No formal PPO rollout.
- No checkpoint publication.
- No default-policy replacement.
- No network, action-space, or default-A* change.
- No distance/path-risk/source-selection gate relaxation.
- No policy performance claim.
- No formal training-ready claim.
- No Ackermann-feasible trajectory claim.
- No IRIS/GCS/path-planner diagnostic treated as training release evidence.
