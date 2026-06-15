# Low-Observation Candidate Geometry Improvement v1

## Background

Stage 5B.4 proved that low-observation candidate coverage sources existed but
the old candidate set did not beat teacher: 837 `low_observation_count`
candidates were source-backed, yet all had non-positive coverage advantage.
Stage 5B.3 therefore remained failed with only three safe-better families.

## Implementation

This stage adds a dedicated low-observation geometry pass:

- config: `configs/quasi_real_low_observation_candidate_geometry_v1.json`
- runner: `scripts/run_low_observation_candidate_geometry_improvement.py`
- shell entrypoint: `scripts/run_low_observation_candidate_geometry_improvement.sh`
- output root:
  `outputs/path_feedback_batch_low_observation_candidate_geometry_improvement_v1/`

The runner preserves model-explorer's required four ROI families in the
generated matrix, but applies high-density `start_cells`, `roi_offsets`,
`candidate_count=16`, and `top_k=8` to `low_observation_count`. It then runs
the path-feedback bridge and `model_explorer path-feedback run`, materializes
low-observation overlay/counterfactual rows from the path-feedback summary and
scenario contracts, reruns 5B.4, and reruns 5B.3 with supplemental inputs.

The teacher action is the post-feedback selected cell, matching the executed
controlled choice semantics used by the existing Stage 5A/5B audit chain.

## Current Evidence

Current summary:

```json
{
  "status": "passed",
  "reason_codes": [],
  "source_path_feedback_scenario_count": 63,
  "low_observation_geometry_candidate_count": 441,
  "low_observation_trainable_safe_better_pair_count": 108,
  "missing_counterfactual_source_count": 0,
  "fallback_gain_contamination_count": 0,
  "controlled_regression_count": 0,
  "performance_claimed": false
}
```

The rerun 5B.3 gate now passes:

```json
{
  "status": "passed",
  "safe_better_than_teacher_candidate_count": 723,
  "safe_better_than_teacher_family_count": 4,
  "family_safe_better_counts": {
    "low_observation_count": 108,
    "mixed_risk": 231,
    "rim_or_steep_slope": 186,
    "smooth_high_confidence": 198
  },
  "performance_claimed": false
}
```

## Validation

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
$PY -m pytest tests/test_low_observation_candidate_geometry_improvement.py tests/test_low_observation_counterfactual_opportunity_generation.py tests/test_safe_better_pair_expansion_across_families.py -q
PYTHON=$PY bash scripts/run_low_observation_candidate_geometry_improvement.sh
jq '{status,reason_codes,low_observation_trainable_safe_better_pair_count,performance_claimed}' outputs/path_feedback_batch_low_observation_candidate_geometry_improvement_v1/low-observation-candidate-geometry-summary.json
jq '{status,reason_codes,safe_better_than_teacher_family_count,family_safe_better_counts,performance_claimed}' outputs/path_feedback_batch_safe_better_pair_expansion_across_families_v1/safe-better-pair-expansion-summary.json
git diff --check
```

## Next Required Change

Return to refined coverage-driven PPO improvement using the expanded
four-family safe-better set. The next gate must prove policy action ranking
changes and exploration coverage return improves without fallback dominance,
controlled regression, checkpoint publication, default policy replacement, or
performance claims.

## Non-Goals

- no PPO update in this stage;
- no checkpoint publication;
- no default policy replacement;
- no real executor connection;
- no network/action-space/default-A* change;
- no guard relaxation;
- no Ackermann trajectory claim;
- no IRIS/GCS diagnostic release claim;
- no performance claim.
