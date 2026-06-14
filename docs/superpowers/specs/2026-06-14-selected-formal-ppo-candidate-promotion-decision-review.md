# Selected Formal PPO Candidate Promotion Decision Review v1

## Purpose

This stage reviews the frozen `Selected Formal PPO Candidate Promotion Preflight v1`
evidence and emits a promotion decision for the selected experimental candidate.
It is a decision review only: a passed verdict means the candidate is eligible
for a later guarded release-candidate packaging stage, not that the checkpoint is
published or formal-training-ready.

## Inputs

- Preflight root:
  `outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_preflight_v1/`
- Preflight summary:
  `selected-formal-ppo-candidate-promotion-preflight-summary.json`
- Expected selected candidate:
  seed `0`, budget `epochs1_lr3e-6`
- Expected checkpoint SHA-256:
  `9d9539c685ab965739c91958bf9cbfe90329c460b4bdbcc35881875aa62f0aa2`

The review follows the lineage:

1. promotion preflight summary
2. selected multi-horizon shadow rollout summary
3. formal candidate-selection long-horizon holdout summary
4. formal stability holdout validation summary

## Outputs

- Config:
  `configs/selected_formal_ppo_candidate_promotion_decision_review_v1.json`
- Runner:
  `scripts/run_selected_formal_ppo_candidate_promotion_decision_review.py/.sh`
- Closure:
  `scripts/run_selected_formal_ppo_candidate_promotion_decision_review_closure.sh`
- Tests:
  `tests/test_selected_formal_ppo_candidate_promotion_decision_review.py`
- Output root:
  `outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_decision_review_v1/`

Output files:

- `selected-formal-ppo-candidate-promotion-decision-review-summary.json`
- `evidence-lineage-report.json`
- `checkpoint-identity-audit.json`
- `release-boundary-audit.json`
- `promotion-decision-readiness-validate-only.json`
- `promotion-decision-report.md`

## Acceptance

- Summary `status=passed`, `reason_codes=[]`.
- `decision_verdict=eligible_for_guarded_release_candidate_packaging`.
- Lineage audit passes with 4 source summaries.
- Checkpoint identity audit passes: path, metadata, SHA-256, and file size match
  the preflight manifest and hash audit.
- Release-boundary audit passes: no checkpoint publication, no default-policy
  replacement, no performance claim, and no formal-training-ready claim.
- Readiness accepts
  `--selected-formal-ppo-candidate-promotion-decision-review-summary` and returns
  `selected_formal_ppo_candidate_promotion_decision_review_evaluated`.

## Non-Goals

- Do not run a new PPO update.
- Do not publish or replace any checkpoint/default policy.
- Do not modify network, action space, default A*, or distance/path-risk/source-selection gates.
- Do not download new raw data.
- Do not claim Ackermann-feasible trajectory output.
- Do not treat IRIS/GCS/path-planner diagnostics as training release evidence.
- Do not claim policy performance or formal training readiness.

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_selected_formal_ppo_candidate_promotion_decision_review.py \
  tests/test_selected_formal_ppo_candidate_promotion_preflight.py \
  tests/test_policy_training_readiness_review.py

PYTHON=$P bash scripts/run_selected_formal_ppo_candidate_promotion_decision_review_closure.sh

jq '{status,reason_codes,decision_verdict,readiness_status}' \
  outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_decision_review_v1/selected-formal-ppo-candidate-promotion-decision-review-summary.json
```
