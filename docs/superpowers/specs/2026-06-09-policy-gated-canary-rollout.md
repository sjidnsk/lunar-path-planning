# Policy-Gated Canary Rollout v1

## Summary

Raw Policy Generalization and Anti-Overfit Closure v1 has moved the candidate to
`raw_policy_generalization_evaluated`: TEST raw regression dropped from the
baseline while controlled regression gates stayed at 0. The next question is not
whether the policy can copy source selection, but whether it can safely choose a
different candidate when a valid alternative exists.

Policy-Gated Canary Rollout v1 is a shadow-only, gated test drive. The policy
may propose a raw top choice, but every changed decision is checked against the
existing action mask, reachability, replan/fallback, safety, contract,
path/risk, and source-selection gates. Accepted changed decisions are evidence
of controlled takeover value; rejected changed decisions remain diagnostic and
fall back to the source-selected candidate.

## Artifacts

- Configs:
  - `configs/path_feedback_batch_policy_gated_canary_rollout_v1.json`
  - `configs/policy_gated_canary_rollout_v1.json`
- Scripts:
  - `scripts/run_policy_gated_canary_rollout.py`
  - `scripts/run_policy_gated_canary_rollout.sh`
- Output root:
  - `outputs/path_feedback_batch_policy_gated_canary_rollout_v1/`
- Output files:
  - `policy-gated-canary-rollout-summary.json`
  - `policy-gated-canary-decisions.jsonl`
  - `policy-gated-canary-rejection-report.json`
  - `policy-gated-canary-opportunity-summary.json`

## Acceptance Gates

- Canary batch `failed_count=0`.
- Canary summary `status=passed` and `reason_codes=[]`.
- `policy_decision_count>0`.
- `canary_opportunity_context_count>0`.
- `policy_changed_decision_count>0`.
- `canary_accepted_policy_choice_count>0`.
- `invalid_action_mask_count=0`.
- `fallback_or_open_grid_count=0`.
- `safety_regression_count=0`.
- `contract_violation_count=0`.
- `path_cost_regression_count=0`.
- `risk_regression_count=0`.
- `source_selection_regression_count=0`.
- Candidate and checkpoint provenance must match the current source state for
  readiness promotion.
- Readiness may advance to `policy_gated_canary_rollout_evaluated`.

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python
SRC=outputs/path_feedback_batch_clean_head_hybrid_readiness_closure_v1
CAND=outputs/path_feedback_batch_raw_policy_generalization_candidate_v1
CANARY=outputs/path_feedback_batch_policy_gated_canary_rollout_v1

PYTHON=$P bash scripts/run_batch_path_feedback_validation.sh \
  --matrix configs/path_feedback_batch_policy_gated_canary_rollout_v1.json

PYTHON=$P bash scripts/run_policy_gated_canary_rollout.sh \
  --source-root $SRC \
  --candidate-root $CAND \
  --batch-root $CANARY \
  --config configs/policy_gated_canary_rollout_v1.json

PYTHON=$P bash scripts/run_policy_training_readiness_review.sh \
  --batch-root $SRC \
  --config configs/policy_training_readiness_review_v1.json \
  --raw-policy-generalization-evaluation-summary \
  $CAND/raw-policy-generalization-evaluation-summary.json \
  --policy-gated-canary-rollout-summary \
  $CANARY/policy-gated-canary-rollout-summary.json

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_policy_gated_canary_rollout.py \
  tests/test_policy_training_readiness_review.py
```

## Non-Goals

Do not start formal PPO rollout, publish or replace a policy, modify
network/action space/default A*, relax the default distance contract, claim
Ackermann-feasible trajectory, treat IRIS/GCS diagnostics as training release
evidence, or claim policy performance improvement.
