# Iterative PPO Mini-Loop Stability v1

## Summary

Current readiness is `limited_ppo_update_smoke_evaluated`: one tiny PPO update
from the collector policy completed and the post-update raw, sequential, and
collector gates stayed clean. This stage adds a stricter three-round mini-loop:
collect on-policy sequential canary transitions, run one limited PPO update,
evaluate the updated candidate, then use that candidate as the next round's base.

This is still an experimental stability check. It is not formal PPO rollout,
checkpoint publication, default-policy replacement, or a performance claim.

## Artifacts

- `configs/iterative_ppo_mini_loop_stability_v1.json`
- `configs/iterative_ppo_update_step_v1.json`
- `scripts/run_iterative_ppo_mini_loop_stability.py/.sh`
- `scripts/run_iterative_ppo_mini_loop_stability_closure.sh`
- `outputs/path_feedback_batch_iterative_ppo_mini_loop_stability_v1/`

Output files:

- `iterative-ppo-mini-loop-stability-summary.json`
- `iterative-ppo-mini-loop-rounds.jsonl`
- `iterative-ppo-mini-loop-drift-report.json`
- `iterative-ppo-mini-loop-rejection-report.json`
- `final/raw-policy-generalization-evaluation-summary.json`
- `final/sequential/policy-gated-sequential-canary-rollout-summary.json`
- `final/collector/ppo-rollout-collector-summary.json`

## Loop Contract

Round 0 starts from `outputs/path_feedback_batch_limited_ppo_update_smoke_v1/`.
Each round runs:

1. sequential canary using the round base candidate
2. PPO collector from that sequential root and the same base candidate
3. limited PPO update from that collector and base candidate
4. raw generalization, sequential canary, and collector gate on the updated candidate
5. updated candidate becomes the next round base

The collector is on-policy only if old `log_prob` and `value` recomputed from the
round base checkpoint match the transition values within `1e-4`. Source fallback
transitions remain diagnostic only and never enter the optimizer.

## Acceptance Gates

- `round_count=3`, `failed_round_count=0`, `stability_passed=true`
- every round optimizer transition count `>=24`
- source fallback trainable count `0`
- old log-prob/value max abs error `<=1e-4`
- loss, grad, reward, return, and advantage finite
- `parameter_l2_delta>0`
- `max_abs_approx_kl<=0.25`
- `cumulative_parameter_l2_delta<=0.05`
- raw TEST regression remains `0`
- sequential canary remains 36 episode / 108 step, 6 accepted families, multi-step accepted episode count `>=12`, rejected choice `0`
- collector keeps trainable transition count `>=24` and all mask/log_prob/value/reward/state/path/risk/source-selection regressions at `0`
- readiness advances to `iterative_ppo_mini_loop_stability_evaluated`

## Non-Goals

- no formal PPO rollout
- no checkpoint publication or default-policy replacement
- no network/action-space/default-A* change
- no distance/path-risk/source-selection gate relaxation
- no Ackermann-feasible trajectory claim
- no IRIS/GCS/path-planner diagnostic treated as training release evidence
- no policy performance claim
