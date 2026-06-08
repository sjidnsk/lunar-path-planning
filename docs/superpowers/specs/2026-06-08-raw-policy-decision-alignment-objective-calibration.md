# Raw Policy Decision Alignment and Objective Calibration v1

## Summary

Controlled scenario-disjoint rollout proved the shadow gate can keep controlled
decisions safe, but it exposed `raw_policy_regression_count=10`: the raw policy
top choice still picks regressive alternatives. This stage converts those raw
failures into pairwise hard-negative preference signals and retrains an
experimental candidate. It does not add action-label positives or publish a
policy.

## Interfaces

- Mining config/script:
  - `configs/raw_policy_regression_mining_v1.json`
  - `scripts/run_raw_policy_regression_mining.py`
  - `scripts/run_raw_policy_regression_mining.sh`
- Candidate config/script:
  - `configs/raw_policy_decision_alignment_candidate_v1.json`
  - `scripts/run_raw_policy_decision_alignment_candidate.py`
  - `scripts/run_raw_policy_decision_alignment_candidate.sh`
- Strict rollout config/script:
  - `configs/raw_policy_strict_rollout_evaluation_v1.json`
  - `scripts/run_raw_policy_strict_rollout_evaluation.py`
  - `scripts/run_raw_policy_strict_rollout_evaluation.sh`
- Closure:
  - `scripts/run_raw_policy_decision_alignment_closure.sh`
- Tests:
  - `tests/test_raw_policy_decision_alignment.py`

## Behavior

Mining reads HOLD `scenario-disjoint-policy-rollout-decisions.jsonl`,
`scenario-disjoint-policy-rollout-evaluation-summary.json`, and HOLD
`path-feedback-summary.json` files. Each raw-regressive decision becomes one
`raw_policy_regression_preference_pair`: source-selected/controlled-safe is the
preferred side; raw-regressive is the alternative side.

The sample records context id, scenario/action ids, path/risk delta, logits or
margin when available, regression reason codes, candidate features, missing
indicators, and sample weight. Missing context id, metrics, action-mask validity,
or provenance sends the record to exclusion. `hard_positive_added_count=0`.

The alignment candidate reuses the existing hybrid trainer only when raw mining
inputs are explicitly configured. Strict rollout uses the scenario-disjoint
evaluator but writes raw-policy summary files and fails if raw regression
remains.

## Outputs

- HOLD:
  - `raw-policy-regression-mining-summary.json`
  - `raw-policy-regression-preference-samples.jsonl`
  - `raw-policy-regression-exclusion-report.json`
  - `raw-policy-strict-rollout-decisions.jsonl`
  - `raw-policy-strict-rollout-regression-report.json`
  - `raw-policy-strict-rollout-evaluation-summary.json`
- CAND:
  - `raw-policy-decision-alignment-candidate-summary.json`
  - `experimental-hybrid-policy-candidate.pt`
  - `experimental-hybrid-policy-candidate-metadata.json`

## Acceptance

- Mining uniquely attributes all raw-regressive decisions; target evidence count
  is 10 on the current HOLD root.
- `hard_positive_added_count=0`; no `RolloutEpisode` action-label positive is
  added.
- New candidate remains experimental, does not publish, does not replace default
  policy, and makes no performance claim.
- Strict rollout requires controlled regression 0, invalid mask/fallback/safety/
  contract/path/risk/source-selection regression all 0, and
  `raw_policy_regression_count` below the configured baseline threshold.
- Readiness may advance to `raw_policy_decision_alignment_evaluated`.

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python
SRC=outputs/path_feedback_batch_clean_head_hybrid_readiness_closure_v1
CAND=outputs/path_feedback_batch_raw_policy_decision_alignment_candidate_v1
HOLD=outputs/path_feedback_batch_scenario_disjoint_policy_candidate_evaluation_v1
PYTHON=$P bash scripts/run_raw_policy_decision_alignment_closure.sh \
  --source-root $SRC \
  --baseline-candidate-root outputs/path_feedback_batch_clean_head_controlled_hybrid_policy_candidate_v1 \
  --candidate-root $CAND \
  --holdout-root $HOLD
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_raw_policy_decision_alignment.py \
  tests/test_policy_training_readiness_review.py
```

## Non-Goals

No formal PPO rollout, checkpoint publication, default policy replacement,
network/action-space/default-A* change, default distance-contract relaxation,
Ackermann-feasible trajectory claim, IRIS/GCS diagnostic training release, or
policy performance claim.
