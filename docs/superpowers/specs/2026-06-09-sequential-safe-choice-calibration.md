# Sequential Safe-Choice Calibration and Hard-Negative Refinement v1

## Summary

Sequential canary v1 proved the runner is real sequential evidence: 36 episode /
108 step completed, `state_continuity_violation_count=0`, and
`episode_fallback_count=0`. The blocker is policy behavior: 2 rejected policy
choices produced cumulative path/risk regression, and multi-step accepted
episode coverage stayed too low.

This stage mines those rejected steps into sequence-aware hard-negative
preference samples, also mines missed safe choices from source-aligned steps
when a gate-safe better alternative exists, and retrains an experimental
candidate. It does not relax the sequential gate and does not start formal PPO.

## Artifacts

- `configs/sequential_canary_failure_mining_v1.json`
- `configs/sequential_safe_choice_calibration_candidate_v1.json`
- `scripts/run_sequential_canary_failure_mining.py/.sh`
- `scripts/run_sequential_safe_choice_calibration_candidate.py/.sh`
- `scripts/run_sequential_safe_choice_calibration_closure.sh`
- Baseline failed root:
  `outputs/path_feedback_batch_policy_gated_sequential_canary_rollout_v1/`
- New closure roots:
  `outputs/path_feedback_batch_sequential_safe_choice_clean_src_v1/`,
  `outputs/path_feedback_batch_sequential_safe_choice_candidate_v1/`,
  `outputs/path_feedback_batch_policy_gated_sequential_safe_choice_rollout_v1/`

## Training Signal

The mining step converts `path_cost_regression` and `risk_regression` rejected
sequential choices into `raw_policy_regression_preference_pair` samples with
`sequential_sample_type=sequential_hard_negative_preference_pair`. It also
creates `sequential_missed_safe_choice_preference_pair` samples when a
source-aligned step has a non-source candidate that is action-mask valid,
reachable, no-replan, no-fallback/open-grid, contract-safe, path/risk
non-regressive, and better by path, risk, or utility.

Preferred side is the source-selected or controlled-safe choice. Alternative
side is the raw/policy choice that caused sequential path/risk regression.
Samples preserve `episode_id`, `step_index`, `input_start_cell`, execution
goals, reason codes, and path/risk/utility deltas. Missing context id,
candidate metrics, execution goal, or action-mask validity excludes the sample.

## Current Evidence

The implemented mining on
`outputs/path_feedback_batch_policy_gated_sequential_canary_rollout_v1/`
produces 2 hard-negative preference pairs and 6 missed-safe-choice preference
pairs with `hard_positive_added_count=0` and `exclusion_count=0`.

Balanced sequential calibration uses weight 1.0 for hard-negative,
path-cost-negative, risk-negative, and missed-safe-choice samples. Stronger
2.0/2.0/2.5 hard-negative weighting was rejected by evidence because it removed
regressions but made the policy too conservative.

The rerun sequential safe-choice root
`outputs/path_feedback_batch_policy_gated_sequential_safe_choice_rollout_v1/`
currently reports:

- `episode_count=36`, `step_count=108`, `completed_episode_count=36`
- `policy_takeover_step_count=28`
- `accepted_takeover_step_count=28`
- `accepted_better_step_count=28`
- `accepted_takeover_family_count=6`
- `canary_rejected_policy_choice_count=0`
- `state_continuity_violation_count=0`, `episode_fallback_count=0`
- invalid action mask, fallback/open-grid, safety, contract, path/risk, and
  source-selection regression all 0

The stage still fails the original multi-step coverage gate:

- `multi_step_accepted_episode_count=6`
- `family_with_multi_step_accepted_episode_count=2`
- `next_required_change=sequential_opportunity_distribution_gap_requires_more_episodes`

Diagnosis shows the remaining blocker is not path/risk regression and not seed
selection. In the balanced rerun, most source-aligned later steps in four
families have no gate-safe better alternative to train or select. The next stage
must improve sequential multi-step opportunity generation instead of entering
PPO or relaxing gates.

## Acceptance

- Mining produces at least 2 sequential hard-negative pairs.
- `hard_positive_added_count=0`.
- Candidate remains experimental: no checkpoint publication, no default policy
  replacement, and no performance claim.
- VAL/TEST context leakage count is 0.
- Rerun sequential canary must pass the same 36 episode / 108 step gate with 0
  rejected policy choices, 0 cumulative path/risk regression, and required
  multi-step coverage.
- Current evidence satisfies the safety/regression portion but not the
  multi-step coverage portion, so readiness must remain blocked.
- Readiness may advance to `policy_gated_sequential_safe_choice_calibrated`
  only after the multi-step coverage gate also passes.

## Non-goals

No formal PPO rollout, no PPO parameter update, no network/action-space/default
A* change, no distance-contract relaxation, no Ackermann-feasible trajectory
claim, and no IRIS/GCS/path-planner diagnostic-as-training release.
