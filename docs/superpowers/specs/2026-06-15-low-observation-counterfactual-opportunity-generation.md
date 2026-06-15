# Low-Observation Counterfactual Opportunity Generation v1

## Background

Stage 5B.3 expanded trainable safe-better pairs from 51 to 615, but the
cross-family gate still failed because all 615 pairs came from three families:
`mixed_risk=231`, `rim_or_steep_slope=186`, and
`smooth_high_confidence=198`; `low_observation_count=0`.

This stage targets that missing family only. It does not run PPO and does not
turn a diagnostic artifact into a performance claim.

## Goal

Generate and audit source-backed `low_observation_count` counterfactual
coverage opportunities, then export them as supplemental 5B.3 inputs only when
they are:

- train split for PPO-trainable counting;
- guard-clean and action-mask valid;
- fallback-free;
- controlled-regression-free;
- backed by candidate/counterfactual coverage source;
- positive on candidate-vs-teacher `expected_coverage_rate_delta`.

The pass threshold is at least 8 trainable safe-better low-observation pairs.

## Implemented Artifacts

- `configs/quasi_real_low_observation_counterfactual_opportunity_v1.json`
- `scripts/run_low_observation_counterfactual_opportunity_generation.py`
- `scripts/run_low_observation_counterfactual_opportunity_generation.sh`
- `outputs/path_feedback_batch_low_observation_counterfactual_opportunity_v1/low-observation-counterfactual-opportunity-summary.json`
- `outputs/path_feedback_batch_low_observation_counterfactual_opportunity_v1/low-observation-supplemental-overlay.jsonl`
- `outputs/path_feedback_batch_low_observation_counterfactual_opportunity_v1/low-observation-supplemental-counterfactual-rollouts.jsonl`
- `outputs/path_feedback_batch_low_observation_counterfactual_opportunity_v1/low-observation-family-gap-report.json`
- `outputs/path_feedback_batch_low_observation_counterfactual_opportunity_v1/low-observation-counterfactual-opportunity-report.md`

Stage 5B.3 was also extended with:

- `--supplemental-overlay`
- `--supplemental-counterfactual-rollouts`
- base/supplemental/dedup row-count provenance in
  `safe-better-pair-expansion-summary.json`.

## Current Evidence

The current run is a controlled failure:

```json
{
  "status": "failed",
  "reason_codes": [
    "low_observation_safe_better_pair_count_below_threshold",
    "supplemental_overlay_no_positive_coverage_advantage"
  ],
  "source_overlay_row_count": 5508,
  "low_observation_candidate_count": 837,
  "low_observation_source_available_count": 837,
  "low_observation_trainable_safe_better_pair_count": 0,
  "non_positive_coverage_advantage_count": 837,
  "missing_counterfactual_source_count": 0,
  "fallback_gain_contamination_count": 0,
  "controlled_regression_count": 0,
  "performance_claimed": false
}
```

This means source coverage is present, but every low-observation candidate is
less than or equal to teacher on the current coverage-advantage comparison.

Re-running 5B.3 with the supplemental artifacts preserves the prior gate:
`status=failed`, `safe_better_than_teacher_family_count=3`, and
`family_safe_better_gap_low_observation_count` remains active.

## Validation

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
$PY -m pytest tests/test_low_observation_counterfactual_opportunity_generation.py tests/test_safe_better_pair_expansion_across_families.py -q
PYTHON=$PY bash scripts/run_low_observation_counterfactual_opportunity_generation.sh
PYTHON=$PY bash scripts/run_safe_better_pair_expansion_across_families.sh \
  --supplemental-overlay outputs/path_feedback_batch_low_observation_counterfactual_opportunity_v1/low-observation-supplemental-overlay.jsonl \
  --supplemental-counterfactual-rollouts outputs/path_feedback_batch_low_observation_counterfactual_opportunity_v1/low-observation-supplemental-counterfactual-rollouts.jsonl
```

## Next Required Change

Do not proceed to PPO yet. The next useful target is a low-observation
candidate-generation geometry change: adjust the quasi-real low-observation
start cells, ROI offsets, candidate ranking, or counterfactual action proposal
so that at least 8 train-split low-observation actions beat teacher on real
multi-step coverage return without guard relaxation or fallback contribution.

## Non-Goals

- no PPO update;
- no checkpoint publication;
- no default policy replacement;
- no real executor connection;
- no network/action-space/default-A* change;
- no guard relaxation;
- no Ackermann trajectory claim;
- no IRIS/GCS diagnostic release claim;
- no performance claim.
