# Raw Policy Generalization and Anti-Overfit Closure v1

## Summary

Raw Policy Decision Alignment v1 reduced old HOLD raw regressions, but old HOLD
has now been used for calibration. It is dev evidence, not final
generalization evidence. This stage adds TRAIN/VAL/TEST split closure so the
experimental policy candidate is selected on VAL and accepted only on unseen
TEST.

## Scope

- Add `raw_align_train`, `raw_align_val`, and `raw_align_test` scenario sets.
- Add batch matrices:
  - `configs/path_feedback_batch_raw_policy_generalization_train_v1.json`
  - `configs/path_feedback_batch_raw_policy_generalization_val_v1.json`
  - `configs/path_feedback_batch_raw_policy_generalization_test_v1.json`
- Extend raw regression mining with diagnostic-only mode for VAL/TEST.
- Add candidate training wrapper:
  - `configs/raw_policy_generalization_candidate_v1.json`
  - `scripts/run_raw_policy_generalization_candidate.py`
  - `scripts/run_raw_policy_generalization_candidate.sh`
- Add final evaluator:
  - `configs/raw_policy_generalization_evaluation_v1.json`
  - `scripts/run_raw_policy_generalization_evaluation.py`
  - `scripts/run_raw_policy_generalization_evaluation.sh`
- Add closure wrapper:
  - `scripts/run_raw_policy_generalization_closure.sh`
- Update readiness review with `raw_policy_generalization_evaluated`.

## Evidence Roots

- TRAIN: `outputs/path_feedback_batch_raw_policy_generalization_train_v1/`
- VAL: `outputs/path_feedback_batch_raw_policy_generalization_val_v1/`
- TEST: `outputs/path_feedback_batch_raw_policy_generalization_test_v1/`
- CAND: `outputs/path_feedback_batch_raw_policy_generalization_candidate_v1/`

## Acceptance Gates

- TRAIN/VAL/TEST batch `failed_count=0`.
- fallback/open-grid, safety, provenance, contract, path/risk and
  source-selection regression counts are all 0.
- `context_id_missing_count=0`.
- TRAIN/dev may emit `raw_policy_regression_preference_pair`.
- VAL/TEST emit diagnostics only and must not enter training samples.
- `leaked_context_id_count=0`.
- Candidate remains experimental:
  `publishes_checkpoint=false`, `replaces_default_policy=false`,
  `performance_claimed=false`.
- TEST controlled `regression_count=0`.
- TEST raw regression drops by at least 50% versus baseline.
- `overfit_gap<=0.15`.
- Readiness upgrades only to `raw_policy_generalization_evaluated`.

## Non-Goals

- No formal PPO rollout.
- No checkpoint publication or default policy replacement.
- No network, action-space, default-A*, or default distance-contract change.
- No Ackermann-feasible trajectory claim.
- No IRIS/GCS/path-planner diagnostic treated as training release evidence.
- No policy performance claim.

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python
for S in train val test; do
  PYTHON=$P bash scripts/run_batch_path_feedback_validation.sh \
    --matrix configs/path_feedback_batch_raw_policy_generalization_${S}_v1.json
done

SRC=outputs/path_feedback_batch_clean_head_hybrid_readiness_closure_v1
DEV=outputs/path_feedback_batch_scenario_disjoint_policy_candidate_evaluation_v1
G=outputs/path_feedback_batch_raw_policy_generalization
BASE=outputs/path_feedback_batch_clean_head_controlled_hybrid_policy_candidate_v1

PYTHON=$P bash scripts/run_raw_policy_generalization_closure.sh \
  --source-root $SRC \
  --dev-root $DEV \
  --train-root ${G}_train_v1 \
  --val-root ${G}_val_v1 \
  --test-root ${G}_test_v1 \
  --baseline-candidate-root $BASE \
  --candidate-root ${G}_candidate_v1

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_raw_policy_generalization_anti_overfit.py \
  tests/test_policy_training_readiness_review.py
```
