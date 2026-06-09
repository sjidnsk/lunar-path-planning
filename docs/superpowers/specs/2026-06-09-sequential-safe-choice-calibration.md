# Sequential Safe-Choice Calibration and Hard-Negative Refinement v1

## Summary

Sequential canary v1 proved the runner is real sequential evidence: 36 episode /
108 step completed, `state_continuity_violation_count=0`, and
`episode_fallback_count=0`. The blocker is policy behavior: 2 rejected policy
choices produced cumulative path/risk regression, and multi-step accepted
episode coverage stayed too low.

This stage mines those rejected steps into sequence-aware hard-negative
preference samples and retrains an experimental candidate. It does not relax the
sequential gate and does not start formal PPO.

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
`sequential_sample_type=sequential_hard_negative_preference_pair`.

Preferred side is the source-selected or controlled-safe choice. Alternative
side is the raw/policy choice that caused sequential path/risk regression.
Samples preserve `episode_id`, `step_index`, `input_start_cell`, execution
goals, reason codes, and path/risk/utility deltas. Missing context id,
candidate metrics, execution goal, or action-mask validity excludes the sample.

## Acceptance

- Mining produces at least 2 sequential hard-negative pairs.
- `hard_positive_added_count=0`.
- Candidate remains experimental: no checkpoint publication, no default policy
  replacement, and no performance claim.
- VAL/TEST context leakage count is 0.
- Rerun sequential canary passes the same 36 episode / 108 step gate with 0
  rejected policy choices and 0 cumulative path/risk regression.
- Readiness may advance only to
  `policy_gated_sequential_safe_choice_calibrated`.

## Non-goals

No formal PPO rollout, no PPO parameter update, no network/action-space/default
A* change, no distance-contract relaxation, no Ackermann-feasible trajectory
claim, and no IRIS/GCS/path-planner diagnostic-as-training release.
