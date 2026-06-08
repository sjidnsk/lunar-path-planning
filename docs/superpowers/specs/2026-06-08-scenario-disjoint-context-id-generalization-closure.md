# Scenario-Disjoint Context-ID Generalization Closure v1

## Background

Fresh Holdout v1 proved the experimental checkpoint can be evaluated on
candidate/sample identity-disjoint holdout records, but it still reported
scenario overlap. It also depended on legacy identity fallback when stable
context ids were absent. That evidence is useful, but it is not a strict
scenario-level generalization gate.

## Goal

Create an opt-in strict generalization closure:

- Stable `policy-context-id/v1` metadata for path-feedback candidates and all
  derived sample/evaluation artifacts.
- New `holdout` scenario set with disjoint scenario ids and seeds.
- Strict fresh evaluator requiring context-id coverage, zero scenario overlap,
  zero identity overlap, zero legacy fallback, and current git provenance.
- Readiness status `scenario_disjoint_policy_candidate_evaluated` only after
  the strict gate passes.

## Implemented Scope

- Added `model_explorer.policy.context_id` canonical JSON + sha256 helper.
- Added `scenario_seed` and `scenario_variant_id` to generated scenario configs,
  path-feedback manifests, scenario summaries, and candidate context metadata.
- Added `HOLDOUT_VALIDATION_SPECS` and `--scenario-set holdout` support.
- Added strict configs:
  - `configs/path_feedback_batch_scenario_disjoint_policy_candidate_evaluation_v1.json`
  - `configs/scenario_disjoint_policy_candidate_evaluation_v1.json`
- Extended fresh evaluator summaries with:
  - `scenario_disjoint`
  - `context_id_coverage_rate`
  - `context_id_missing_count`
  - `legacy_identity_fallback_count`
  - `identity_overlap_ratio`
  - `candidate_git_current_matches_sources`
  - `checkpoint_metadata_git_current_matches_sources`
- Preserved stable context ids through anchor candidate generation, planner
  validated mining/materialization, preference samples, and unified registry.

## Evidence Roots

Planned clean-HEAD roots:

- SRC: `outputs/path_feedback_batch_clean_head_hybrid_readiness_closure_v1/`
- CAND: `outputs/path_feedback_batch_clean_head_controlled_hybrid_policy_candidate_v1/`
- HOLD: `outputs/path_feedback_batch_scenario_disjoint_policy_candidate_evaluation_v1/`

The root `outputs/tmp_scenario_disjoint_context_id_smoke/` was used only as a
local smoke check and is not acceptance evidence.

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  model-explorer/tests/test_model_explorer.py \
  tests/test_fresh_holdout_policy_candidate_evaluation.py \
  tests/test_policy_training_readiness_review.py \
  tests/test_policy_context_id_contract.py \
  tests/test_scenario_disjoint_policy_candidate_evaluation.py

PYTHON=$P bash scripts/run_batch_path_feedback_validation.sh \
  --matrix configs/path_feedback_batch_scenario_disjoint_policy_candidate_evaluation_v1.json

PYTHON=$P bash scripts/run_fresh_holdout_policy_candidate_evaluation.sh \
  --source-root outputs/path_feedback_batch_clean_head_hybrid_readiness_closure_v1 \
  --candidate-root outputs/path_feedback_batch_clean_head_controlled_hybrid_policy_candidate_v1 \
  --batch-root outputs/path_feedback_batch_scenario_disjoint_policy_candidate_evaluation_v1 \
  --config configs/scenario_disjoint_policy_candidate_evaluation_v1.json
```

## Acceptance Gate

Strict HOLD summary must report:

- `status=passed`
- `reason_codes=[]`
- `scenario_overlap_count=0`
- `identity_overlap_count=0`
- `context_id_missing_count=0`
- `legacy_identity_fallback_count=0`
- `fresh_disjoint_context_count > 0`
- fallback/safety/contract/path/risk/source-selection regression all `0`

Readiness may advance only to
`scenario_disjoint_policy_candidate_evaluated`.

## Non-Goals

No formal PPO rollout, no checkpoint publication, no default policy replacement,
no network/action-space/default-A* changes, no default distance-contract
relaxation, no Ackermann-feasible trajectory claim, no IRIS/GCS diagnostic
training release claim, and no policy performance claim.
