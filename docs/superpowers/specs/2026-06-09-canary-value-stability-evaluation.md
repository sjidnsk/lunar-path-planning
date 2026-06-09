# Canary Value/Stability Evaluation v1

## Summary

Current full-family canary evidence proves the experimental raw-policy candidate
can safely choose alternatives in all six canary families. The remaining
question is value and stability: whether safe changes occur often enough across
families, and whether accepted changes are measurably better rather than merely
different.

This stage first refreshes current-HEAD full-family canary evidence, then runs a
new value/stability canary closure. It remains shadow/canary evaluation only.

## Scope

- Add `policy_canary_value_stability` with 6 families x 6 geometry variants.
- Add the batch config
  `configs/path_feedback_batch_policy_gated_canary_value_stability_v1.json`.
- Add the evaluator config `configs/policy_gated_canary_value_stability_v1.json`.
- Add `scripts/run_canary_value_stability_closure.sh`.
- Extend policy-gated canary summary with:
  - `accepted_equal_choice_count`
  - `accepted_better_choice_count`
  - `accepted_better_family_count`
  - `policy_change_rate`
  - `accepted_choice_rate`
  - `accepted_value_delta_summary`
  - `family_value_stability_summary`
  - `canary_value_stability_passed`
- Extend readiness with
  `policy_gated_canary_value_stability_evaluated`.

## Roots

- Refresh/full-family baseline:
  `outputs/path_feedback_batch_policy_gated_canary_full_family_opportunity_v1/`
- Value/stability SRC:
  `outputs/path_feedback_batch_value_stability_clean_src_v1/`
- Value/stability candidate:
  `outputs/path_feedback_batch_value_stability_candidate_v1/`
- Value/stability canary:
  `outputs/path_feedback_batch_policy_gated_canary_value_stability_v1/`

## Acceptance Gates

- Current-HEAD full-family refresh passes with no provenance mismatch.
- Value/stability batch has `failed_count=0`, fallback/open-grid=0, and safety
  regression=0.
- `scenario_family_count=6`.
- `canary_opportunity_context_count>=72`.
- `family_with_acceptable_alternative_count=6`.
- `accepted_scenario_family_count=6`.
- `canary_accepted_policy_choice_count>=24`.
- `canary_rejected_policy_choice_count=0`.
- `accepted_better_choice_count>=8`.
- `accepted_better_family_count>=3`.
- Dense choke accepted count is greater than 0.
- Controlled/raw regression, invalid action mask, fallback, safety, contract,
  path/risk, and source-selection regression are all 0.
- Candidate/checkpoint provenance matches the current source state.
- Readiness becomes `policy_gated_canary_value_stability_evaluated` with
  `training_blockers=[]`.

## Failure Attribution

- Missing safe opportunities:
  `next_required_change=canary_value_opportunity_generation_gap`.
- Safe accepted choices exist but better choices are insufficient:
  `next_required_change=policy_value_alignment_or_objective_refinement_required`.

## Verification

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTHON=$P bash scripts/run_dense_choke_safe_alternative_opportunity_closure.sh
PYTHON=$P bash scripts/run_canary_value_stability_closure.sh

PYTHON=$P bash scripts/run_policy_training_readiness_review.sh \
  --batch-root outputs/path_feedback_batch_value_stability_clean_src_v1 \
  --config configs/policy_training_readiness_review_v1.json \
  --raw-policy-generalization-evaluation-summary outputs/path_feedback_batch_value_stability_candidate_v1/raw-policy-generalization-evaluation-summary.json \
  --policy-gated-canary-rollout-summary outputs/path_feedback_batch_policy_gated_canary_value_stability_v1/policy-gated-canary-rollout-summary.json \
  --validate-only

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_policy_gated_canary_rollout.py \
  tests/test_policy_training_readiness_review.py \
  tests/test_batch_path_feedback_validation.py \
  tests/test_canary_value_stability.py

cd dev-platform-constraints && \
PYTHONPATH=src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q tests/test_npz_validation_maps.py
```

## Non-Goals

- No formal PPO rollout.
- No checkpoint publication or default policy replacement.
- No network/action-space/default-A* change.
- No default distance-contract relaxation.
- No Ackermann-feasible trajectory claim.
- No use of IRIS/GCS/path-planner diagnostics as training release evidence.
- No policy performance claim.
