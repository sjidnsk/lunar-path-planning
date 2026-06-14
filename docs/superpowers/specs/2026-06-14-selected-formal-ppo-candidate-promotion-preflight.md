# Selected Formal PPO Candidate Promotion Preflight v1

## Summary

This stage follows the passed `Selected Formal PPO Candidate Multi-Horizon
Shadow Rollout v1`. It keeps the selected experimental candidate fixed and runs
a read-only promotion preflight over checkpoint identity, metadata, hashing,
load/inference behavior, rollback boundaries, and multi-horizon evidence
lineage. It does not run a new PPO update, publish a checkpoint, replace the
default policy, claim policy performance, or claim formal training readiness.

## Inputs

- Multi-horizon root:
  `outputs/path_feedback_batch_selected_formal_ppo_candidate_multihorizon_shadow_rollout_v1/`
- Multi-horizon summary:
  `multihorizon-shadow-rollout-summary.json`
- Selected candidate root from that summary:
  `outputs/path_feedback_batch_quasi_real_guarded_formal_ppo_stability_holdout_validation_v1/epochs1_lr3e-6/seed-00/limited_ppo_update_smoke/`
- Candidate checkpoint and metadata:
  `experimental-hybrid-policy-candidate.pt`,
  `experimental-hybrid-policy-candidate-metadata.json`

## Artifacts

- `configs/selected_formal_ppo_candidate_promotion_preflight_v1.json`
- `scripts/run_selected_formal_ppo_candidate_promotion_preflight.py/.sh`
- `scripts/run_selected_formal_ppo_candidate_promotion_preflight_closure.sh`
- `tests/test_selected_formal_ppo_candidate_promotion_preflight.py`
- `outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_preflight_v1/`

The output root writes:

- `selected-formal-ppo-candidate-promotion-preflight-summary.json`
- `promotion-candidate-manifest.json`
- `checkpoint-hash-audit.json`
- `checkpoint-load-inference-audit.json`
- `rollback-audit.json`
- `promotion-preflight-readiness-validate-only.json`
- `promotion-preflight-report.md`

## Acceptance Gates

- input multi-horizon summary is passed and traceable
- selected seed, budget, and candidate root match the selected evidence
- checkpoint and metadata exist
- checkpoint SHA-256 and size are recorded
- checkpoint loads on CPU
- inference audit covers at least 64 multi-horizon shadow observations
- invalid mask, missing observation, non-finite logits/log_prob/value counts are 0
- log_prob/value reconstruction differences are recorded and explained when the
  shadow records are not an exact current-checkpoint replay
- controlled and family regression counts remain 0
- teacher agreement is at least 0.95
- rollback audit passes and default policy is not replaced
- no new PPO update is run
- no checkpoint publication, default-policy replacement, performance claim, or
  formal-training-ready claim
- readiness status is
  `selected_formal_ppo_candidate_promotion_preflight_evaluated`

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_selected_formal_ppo_candidate_promotion_preflight.py \
  tests/test_selected_formal_ppo_candidate_multihorizon_shadow_rollout.py \
  tests/test_policy_training_readiness_review.py

PYTHON=$P bash scripts/run_selected_formal_ppo_candidate_promotion_preflight_closure.sh

jq '{status,reason_codes,checkpoint_load_passed,inference_audit_count,readiness_status}' \
  outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_preflight_v1/selected-formal-ppo-candidate-promotion-preflight-summary.json

git diff --check
```

## Non-Goals

No new PPO update, no checkpoint publication, no default policy replacement,
no network/action space/default-A* change, no distance/path-risk/source-selection
gate relaxation, no raw-data download, no Ackermann-feasible trajectory claim,
no IRIS/GCS or path-planner diagnostic promotion to training release evidence,
no policy-performance claim, and no formal-training-ready claim.
