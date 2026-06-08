# Fresh Holdout Policy Candidate Evaluation v1

## Summary

Source evidence root:
`outputs/path_feedback_batch_hybrid_current_head_readiness_closure_v1/`.

Controlled candidate root:
`outputs/path_feedback_batch_controlled_hybrid_policy_training_candidate_v1/`.

Fresh holdout root:
`outputs/path_feedback_batch_fresh_holdout_policy_candidate_evaluation_v1/`.

The current controlled candidate gate has passed with a local experimental
checkpoint, no publication, no default replacement, and no performance claim.
Fresh Holdout Policy Candidate Evaluation v1 adds an opt-in disjoint
candidate-context holdout gate before any stronger readiness claim.

## Implementation

New artifacts:

- `configs/path_feedback_batch_fresh_holdout_policy_candidate_evaluation_v1.json`
- `configs/fresh_holdout_policy_candidate_evaluation_v1.json`
- `scripts/run_fresh_holdout_policy_candidate_evaluation.py`
- `scripts/run_fresh_holdout_policy_candidate_evaluation.sh`
- `tests/test_fresh_holdout_policy_candidate_evaluation.py`

Readiness review now accepts:

- `--fresh-holdout-policy-candidate-evaluation-summary`

Passing fresh holdout may advance readiness only to
`fresh_holdout_policy_candidate_evaluated`.

## Evidence Contract

The evaluator reuses
`outputs/path_feedback_batch_controlled_hybrid_policy_training_candidate_v1/experimental-hybrid-policy-candidate.pt`
and writes:

- `fresh-holdout-policy-candidate-evaluation-summary.json`
- `fresh-holdout-overlap-report.json`
- `fresh-holdout-candidate-score-report.json`

Accepted records must have identity keys disjoint from both the source root and
the controlled candidate root. Identity prefers an existing context id; otherwise
it uses `(scenario_id, sample_type, source_action_index, policy_target_cell,
execution_goal_cell, target_binding_mode)`. Scenario id overlap is allowed only
as reported metadata and is not a scenario-level generalization claim.

Fresh summary must report:

- `status=passed`
- `reason_codes=[]`
- `fresh_disjoint_context_count > 0`
- `accepted_identity_overlap_count=0`
- `accepted_identity_key_missing_count=0`
- `fallback_or_open_grid_count=0`
- `safety_regression_count=0`
- `contract_violation_count=0`
- path/risk/source-selection regression counts all 0
- `experimental_checkpoint=true`
- `publishes_checkpoint=false`
- `replaces_default_policy=false`
- `performance_claimed=false`

If no disjoint contexts exist, set
`next_required_change=fresh_holdout_scenario_or_candidate_generation_required`.
If quality regressions exist, set
`next_required_change=training_objective_or_sample_weight_refinement_required`.

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python
S=outputs/path_feedback_batch_hybrid_current_head_readiness_closure_v1
H=outputs/path_feedback_batch_fresh_holdout_policy_candidate_evaluation_v1
C=outputs/path_feedback_batch_controlled_hybrid_policy_training_candidate_v1

PYTHON=$P bash scripts/run_batch_path_feedback_validation.sh \
  --matrix configs/path_feedback_batch_fresh_holdout_policy_candidate_evaluation_v1.json

PYTHON=$P bash scripts/run_fresh_holdout_policy_candidate_evaluation.sh \
  --source-root $S \
  --candidate-root $C \
  --batch-root $H \
  --config configs/fresh_holdout_policy_candidate_evaluation_v1.json

PYTHON=$P bash scripts/run_policy_training_readiness_review.sh \
  --batch-root $S \
  --config configs/policy_training_readiness_review_v1.json \
  --fresh-holdout-policy-candidate-evaluation-summary \
    $H/fresh-holdout-policy-candidate-evaluation-summary.json

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  model-explorer/tests/test_model_explorer.py \
  tests/test_fresh_holdout_policy_candidate_evaluation.py \
  tests/test_policy_training_readiness_review.py
```

## Boundaries

This is not formal PPO rollout, not checkpoint publication, not default policy
replacement, not a network/action-space/default-A* change, not a default
distance-contract relaxation, not an Ackermann-feasible trajectory claim, not an
IRIS/GCS diagnostic release gate, and not a policy performance claim.
