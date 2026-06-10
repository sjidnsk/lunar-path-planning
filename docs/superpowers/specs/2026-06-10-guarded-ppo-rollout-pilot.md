# Guarded PPO Rollout Pilot v1

## Summary

Current readiness has reached `iterative_ppo_mini_loop_stability_evaluated`.
The system can run three small collect/update/evaluate loops without raw
generalization, sequential canary, or PPO collector regression. The next stage
tests the guarded PPO rollout entry point itself: policy choices are proposed
step by step, existing gates decide whether they may execute, and only accepted
policy-controlled steps become PPO-trainable transitions.

This remains experimental and guarded. It is not a released PPO rollout, not a
default policy replacement, and not a performance claim.

## Interfaces

- Configs:
  - `configs/guarded_ppo_rollout_pilot_v1.json`
  - `configs/guarded_ppo_rollout_update_v1.json`
- Scripts:
  - `scripts/run_guarded_ppo_rollout_pilot.py/.sh`
  - `scripts/run_guarded_ppo_rollout_pilot_closure.sh`
- Evidence root:
  - `outputs/path_feedback_batch_guarded_ppo_rollout_pilot_v1/`
- Readiness:
  - new argument `--guarded-ppo-rollout-pilot-summary`
  - new status `guarded_ppo_rollout_pilot_evaluated`

## Behavior

The pilot uses the current iterative mini-loop final candidate as its base
policy and runs the existing 36 episode / 108 step sequential multi-step
opportunity set. Each step preserves the existing gate: action mask valid,
candidate present, reachable, no replan, no fallback/open-grid, contract safe,
and no path/risk/source-selection regression.

Only `controlled_choice_source=policy` steps that pass every gate are marked
`ppo_trainable=true`. Source-fallback steps remain diagnostic and must never be
optimized. The pilot then runs one tiny on-policy PPO update from the same base
checkpoint and re-runs raw generalization, sequential canary, and collector
checks against the updated experimental candidate.

## Acceptance

- `episode_count=36`, `step_count=108`.
- `ppo_trainable_transition_count>=24`.
- `source_fallback_trainable_count=0`.
- invalid/empty action mask, missing log-prob/value, non-finite reward, and
  state-continuity violation counts are all 0.
- fallback/open-grid, safety, contract, path/risk, and source-selection
  regression counts are all 0.
- old log-prob/value max abs error is `<=1e-4`.
- loss, grad, reward, return, and advantage are finite.
- `parameter_l2_delta>0`, `abs(approx_kl)<=0.25`,
  `max_grad_norm_after_clip<=1.0`.
- post-update raw TEST regression remains 0.
- post-update sequential canary and collector gates remain passed.
- readiness reports `guarded_ppo_rollout_pilot_evaluated` with no blockers.

## Non-goals

- No released PPO policy.
- No default policy replacement.
- No network, action-space, or default-A* change.
- No distance-contract or path/risk/source-selection gate relaxation.
- No Ackermann-feasible trajectory claim.
- No IRIS/GCS/path-planner diagnostic treated as training release evidence.
- No policy performance claim.
