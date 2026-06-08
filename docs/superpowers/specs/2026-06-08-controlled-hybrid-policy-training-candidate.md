# Controlled Hybrid Policy Training Candidate v1

## Summary

Source evidence root:
`outputs/path_feedback_batch_hybrid_current_head_readiness_closure_v1/`.

New candidate evidence root:
`outputs/path_feedback_batch_controlled_hybrid_policy_training_candidate_v1/`.

The prior stage closed current-HEAD hybrid dry-run readiness:
`action_label_positive_count=24`, `pairwise_preference_signal_count=54`,
`hybrid_train_signal_count=78`, `hard_positive_added_count=0`, and
`training_readiness_status=hybrid_training_dry_run_completed`.

This stage adds an opt-in local experimental checkpoint candidate plus offline
holdout gates. It does not publish a checkpoint, replace the default policy, or
claim policy performance.

## Implementation

New artifacts:

- `configs/controlled_hybrid_policy_training_candidate_v1.json`
- `configs/controlled_hybrid_policy_holdout_evaluation_v1.json`
- `scripts/run_controlled_hybrid_policy_training_candidate.py`
- `scripts/run_controlled_hybrid_policy_training_candidate.sh`
- `scripts/run_controlled_hybrid_policy_holdout_evaluation.py`
- `scripts/run_controlled_hybrid_policy_holdout_evaluation.sh`
- `tests/test_controlled_hybrid_policy_training_candidate.py`

Readiness review now accepts:

- `--controlled-hybrid-policy-training-candidate-summary`
- `--controlled-hybrid-policy-holdout-evaluation-summary`

Passing controlled candidate and holdout summaries may advance readiness only to
`controlled_hybrid_training_candidate_evaluated`.

## Evidence Contract

Candidate summary must report:

- `status=passed`
- `candidate_training_status=passed`
- `action_label_positive_count=24`
- `pairwise_preference_signal_count=54`
- `hybrid_train_signal_count=78`
- `hard_positive_added_count=0`
- `experimental_checkpoint=true`
- `publishes_checkpoint=false`
- `replaces_default_policy=false`
- `performance_claimed=false`

Holdout summary must report:

- `status=passed`
- `reason_codes=[]`
- `action_mask_invalid_count=0`
- `empty_action_mask_count=0`
- `fallback_or_open_grid_count=0`
- `safety_regression_count=0`
- `contract_violation_count=0`
- path/risk/source-selection regression counts
- `source_selection_agreement_count`
- `preference_margin_improved_count`
- `performance_claimed=false`

If path/risk/source-selection regressions are nonzero, readiness remains
blocked and sets
`next_required_change=training_objective_or_sample_weight_refinement_required`.

## Validation

```bash
SRC=outputs/path_feedback_batch_hybrid_current_head_readiness_closure_v1
ROOT=outputs/path_feedback_batch_controlled_hybrid_policy_training_candidate_v1
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTHON=$PY bash scripts/run_controlled_hybrid_policy_training_candidate.sh \
  --source-root $SRC \
  --output-root $ROOT \
  --config configs/controlled_hybrid_policy_training_candidate_v1.json

PYTHON=$PY bash scripts/run_controlled_hybrid_policy_holdout_evaluation.sh \
  --source-root $SRC \
  --candidate-root $ROOT \
  --config configs/controlled_hybrid_policy_holdout_evaluation_v1.json

PYTHON=$PY bash scripts/run_policy_training_readiness_review.sh \
  --batch-root $SRC \
  --config configs/policy_training_readiness_review_v1.json \
  --hybrid-policy-training-dry-run-summary $SRC/hybrid-policy-training-dry-run-summary.json \
  --controlled-hybrid-policy-training-candidate-summary \
    $ROOT/controlled-hybrid-policy-training-candidate-summary.json \
  --controlled-hybrid-policy-holdout-evaluation-summary \
    $ROOT/controlled-hybrid-policy-holdout-evaluation-summary.json

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest -q \
  tests/test_controlled_hybrid_policy_training_candidate.py \
  tests/test_policy_training_readiness_review.py \
  tests/test_hybrid_policy_training_dry_run.py
```

## Boundaries

This is not formal PPO rollout, not production checkpoint publication, not
default policy replacement, not a network/action-space/default-A* change, not a
default distance-contract relaxation, not an Ackermann-feasible trajectory
claim, and not a policy performance claim.
