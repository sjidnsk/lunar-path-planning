# Quasi-Real Guarded PPO Iterative Mini-Loop Stability v1

## Summary

This stage follows the passed `Quasi-Real Trainable Context Expansion v1` and
the passed Scale-512 rerun. It is not formal PPO training and does not download
new raw data. It uses the current 684 train-split, gate-clean, quasi-real
trainable transitions to test whether repeated tiny PPO smoke updates remain
stable across multiple seeds and iterations.

## Inputs

- Expansion summary:
  `outputs/path_feedback_batch_quasi_real_trainable_context_expansion_v1/quasi-real-trainable-context-expansion-summary.json`
- Expansion steps:
  `outputs/path_feedback_batch_quasi_real_trainable_context_expansion_v1/quasi-real-trainable-context-expansion-steps.jsonl`
- Scale-512 rerun summary:
  `outputs/path_feedback_batch_quasi_real_trainable_context_expansion_v1/scale512_rerun/quasi-real-guarded-ppo-scale512-multiseed-preflight-summary.json`
- Base candidate:
  `outputs/path_feedback_batch_quasi_real_teacher_distillation_candidate_v1`

Required evidence:

- `status=passed`, `reason_codes=[]`
- `unique_trainable_context_count=684`
- `ppo_trainable_transition_count=684`
- validation/test/source_fallback/teacher_fallback trainable counts are zero
- controlled regression count is zero
- Scale-512 rerun has `seed_count=3`, `passed_seed_count=3`

## Behavior

The runner executes `seeds=[0,1,2]` and `iteration_count=3`. Each seed starts
from the same experimental base candidate. Each iteration:

1. filters the same 684 train split, controlled-policy, gate-clean transitions;
2. refreshes old `log_prob/value` from the current base candidate;
3. materializes PPO collector episodes;
4. runs one full-batch limited PPO smoke update;
5. records per-iteration metrics;
6. chains the next iteration in that seed from the previous experimental output.

Training parameters remain bounded: `epochs=1`, `learning_rate<=1e-5`,
`clip_ratio=0.2`, `max_grad_norm=1.0`, and discounted multi-step returns remain
the reward/return basis.

## Outputs

- `quasi-real-guarded-ppo-iterative-miniloop-stability-summary.json`
- `iterative-miniloop-iteration-summaries.jsonl`
- `iterative-miniloop-progress.jsonl`
- `iterative-miniloop-readiness-validate-only.json`
- `iterative-miniloop-stability-report.md`

Output root:
`outputs/path_feedback_batch_quasi_real_guarded_ppo_iterative_miniloop_stability_v1/`

## Acceptance Gates

- summary `status=passed`, `reason_codes=[]`
- `input_trainable_transition_count=684`
- `unique_trainable_context_count=684`
- `seed_count=3`, `iteration_count=3`
- `passed_iteration_count=9`, `failed_iteration_count=0`
- every iteration optimizer train count is 684
- validation/test/fallback/teacher_fallback trainable counts are zero
- old log_prob/value max abs error `<=1e-4`
- loss/gradient/reward/return/advantage counters are finite
- `abs(approx_kl)<=0.25`
- `max_grad_norm_after_clip<=1.0`
- minimum teacher agreement `>=0.95`
- controlled regression and behavior drift counts are zero
- readiness advances to
  `quasi_real_guarded_ppo_iterative_miniloop_stability_evaluated`
- docs reflect that this is iterative mini-loop stability, not formal PPO ready

## Non-Goals

- Do not start formal PPO rollout.
- Do not download new raw data.
- Do not publish or replace checkpoints.
- Do not modify network, action space, or default A*.
- Do not relax distance/path-risk/source-selection gates.
- Do not copy samples to fake scale.
- Do not claim Ackermann-feasible trajectory.
- Do not treat IRIS/GCS/path-planner diagnostics as training release evidence.
- Do not claim policy performance or formal training readiness.
