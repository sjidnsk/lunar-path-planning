# Canary Opportunity Quality and Multi-Family Safe Choice Expansion v1

## Summary

Canary Diversity v1 proved safe policy changes in 3 scenario families, but it
also exposed that several families did not produce accepted policy takeovers.
This stage separates two causes: no acceptable alternative exists in the
candidate set, or a safe alternative exists but the raw policy stays aligned with
the source-selected candidate.

## Implementation

- Add `policy_canary_opportunity_quality` to batch and single-run scenario set
  validation.
- Add a 12-scenario validation set in `dev-platform-constraints`: two variants
  each for mixed stress detour, near-blocked safe alternative, channel contrast,
  high-risk tradeoff, dense choke safe bypass, and path-complexity benefit.
- Extend the policy-gated canary evaluator with opportunity-quality fields:
  `family_with_acceptable_alternative_count`,
  `missing_acceptable_alternative_families`,
  `source_aligned_with_acceptable_alternative_count`,
  `canary_missed_opportunity_preference_pair_count`,
  `missed_safe_choice_family_count`, and `hard_positive_added_count=0`.
- Add configs:
  `configs/path_feedback_batch_policy_gated_canary_opportunity_quality_v1.json`,
  `configs/policy_gated_canary_opportunity_quality_v1.json`, and
  `configs/canary_missed_opportunity_preference_v1.json`.
- Add `scripts/run_canary_opportunity_quality_closure.sh` for clean SRC,
  candidate, raw-generalization, canary opportunity-quality, and readiness
  closure.
- Update readiness so a passing summary advances to
  `policy_gated_canary_opportunity_quality_evaluated`.

## Evidence Roots

- SRC: `outputs/path_feedback_batch_canary_opportunity_quality_clean_src_v1/`
- CAND: `outputs/path_feedback_batch_canary_opportunity_quality_candidate_v1/`
- CANARY: `outputs/path_feedback_batch_policy_gated_canary_opportunity_quality_v1/`

## Result

The closure passed. The canary summary reports `policy_decision_count=24`,
`canary_opportunity_context_count=24`, `policy_changed_decision_count=10`,
`canary_accepted_policy_choice_count=10`,
`canary_rejected_policy_choice_count=0`, `scenario_family_count=6`,
`accepted_scenario_family_count=5`,
`family_with_acceptable_alternative_count=5`,
`source_aligned_with_acceptable_alternative_count=0`,
`canary_missed_opportunity_preference_pair_count=0`, and
`hard_positive_added_count=0`. Accepted choices cover channel contrast,
high-risk tradeoff, mixed stress detour, near-blocked safe alternative, and
path-complexity benefit. `dense_choke_safe_bypass` remains the only family
without an acceptable alternative.

All controlled/raw regression, invalid action mask, fallback/open-grid, safety,
contract, path/risk, and source-selection regression counts are `0`.
Candidate and checkpoint provenance match the current source state. Readiness
reports `training_readiness_status=policy_gated_canary_opportunity_quality_evaluated`
and `training_blockers=[]`.

## Acceptance

- Clean current provenance and no candidate/checkpoint metadata mismatch.
- Batch `failed_count=0`, fallback/open-grid `0`, safety regression `0`.
- `context_id_missing_count=0`, `legacy_identity_fallback_count=0`.
- `scenario_family_count>=6`.
- `canary_opportunity_context_count>=24`.
- `family_with_acceptable_alternative_count>=5`.
- `accepted_scenario_family_count>=5`.
- `canary_accepted_policy_choice_count>=8`.
- Rejected choices are `0` or all carry reason codes.
- Controlled/raw regression, invalid action mask, fallback, safety, contract,
  path/risk, and source-selection regression counts are all `0`.
- Candidate remains experimental and does not publish a checkpoint, replace the
  default policy, or claim performance.
- Readiness reports `status=passed`,
  `training_readiness_status=policy_gated_canary_opportunity_quality_evaluated`,
  and `training_blockers=[]`.

## Failure Attribution

- Missing acceptable alternatives in a family are attributed to
  `canary_opportunity_generation_gap`.
- Source-aligned decisions with acceptable alternatives are attributed to
  `canary_missed_opportunity_preference_pair` for preference/calibration only.
- `hard_positive_added_count` remains `0`; these records do not become
  `RolloutEpisode` action-label positives.

## Validation

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/kai/anaconda3/envs/lunar-explorer/bin/python -m pytest -q \
  tests/test_policy_gated_canary_rollout.py \
  tests/test_batch_path_feedback_validation.py \
  tests/test_canary_opportunity_quality.py \
  tests/test_policy_training_readiness_review.py

PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_canary_opportunity_quality_closure.sh
```

## Non-Goals

This stage does not start formal PPO rollout, publish or replace a policy,
modify network/action space/default A*, relax the distance contract, claim
Ackermann-feasible trajectory, treat IRIS/GCS/path-planner diagnostics as
training release evidence, or claim policy performance improvement.
