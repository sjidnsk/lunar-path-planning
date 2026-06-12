# Quasi-Real Safe-Alternative Opportunity Diagnosis v1

## Summary

`Quasi-Real Guarded Policy Pilot v1` can run on LOLA quasi-real contexts, but the
current useful blocker is over-conservatism: the policy stays source-aligned on
all 12 quasi-real contexts, while action-mask, fallback/open-grid, safety,
contract, path/risk, and source-selection regressions remain 0.

This stage is diagnostic only. It answers whether quasi-real ROI contexts contain
any top-k alternative that is both gate-safe and better than the source-selected
candidate. If no such alternatives exist, the next change is ROI/start/target
opportunity expansion. If they exist and the policy remains source-aligned, the
next change is quasi-real safe-choice calibration.

## Artifacts

- `configs/quasi_real_safe_alternative_opportunity_diagnosis_v1.json`
- `scripts/run_quasi_real_safe_alternative_opportunity_diagnosis.py`
- `scripts/run_quasi_real_safe_alternative_opportunity_diagnosis.sh`
- `outputs/path_feedback_batch_quasi_real_safe_alternative_opportunity_diagnosis_v1/quasi-real-safe-alternative-opportunity-diagnostics.jsonl`
- `outputs/path_feedback_batch_quasi_real_safe_alternative_opportunity_diagnosis_v1/quasi-real-safe-alternative-opportunity-summary.json`
- `outputs/path_feedback_batch_quasi_real_safe_alternative_opportunity_diagnosis_v1/quasi-real-safe-alternative-opportunity-exclusion-report.json`
- `outputs/path_feedback_batch_quasi_real_safe_alternative_opportunity_diagnosis_v1/quasi-real-safe-alternative-opportunity-report.md`

## Inputs

- `outputs/path_feedback_batch_quasi_real_map_domain_gap_v1/`
- `outputs/path_feedback_batch_quasi_real_guarded_policy_pilot_v1/`
- `outputs/path_feedback_batch_quasi_real_shadow_alignment_candidate_v1/`

## Diagnosis Rules

Each context uses the source-selected candidate as the baseline. Every top-k
alternative is evaluated through this funnel:

```text
candidate present
  -> action mask valid
  -> reachable
  -> no replan
  -> no fallback/open-grid
  -> contract safe
  -> path cost no regression
  -> risk no regression
  -> source-selection no regression
  -> safe alternative
  -> safe-better alternative
```

`safe-better` means no gate regression and at least one value improvement:

- `path_cost_delta <= -0.25`
- `risk_delta <= -0.01`
- `utility_delta >= 0.005`

Context classes:

- `opportunity_missing`
- `safe_alternative_exists_but_not_better`
- `safe_better_opportunity_exists_policy_source_aligned`
- `safe_better_opportunity_policy_selected`
- `bridge_or_feedback_gap`
- `action_mask_or_contract_gap`

Funnel rejection counts are diagnostic and do not by themselves represent system
path/risk regression.

## Readiness

Readiness accepts only the diagnostic status
`quasi_real_safe_alternative_opportunity_diagnosed`. It must not upgrade to a
quasi-real guarded pilot pass or PPO readiness.

Verdicts:

- `quasi_real_safe_alternative_opportunity_gap`
- `acceptable_for_quasi_real_safe_choice_calibration`
- `real_map_bridge_or_feedback_gap`
- `real_map_action_mask_contract_gap`

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_quasi_real_safe_alternative_opportunity_diagnosis.py \
  tests/test_quasi_real_guarded_policy_pilot.py \
  tests/test_quasi_real_shadow_alignment.py \
  tests/test_policy_training_readiness_review.py

PYTHON=$P bash scripts/run_quasi_real_safe_alternative_opportunity_diagnosis.sh \
  --quasi-real-root outputs/path_feedback_batch_quasi_real_map_domain_gap_v1 \
  --guarded-pilot-root outputs/path_feedback_batch_quasi_real_guarded_policy_pilot_v1 \
  --alignment-root outputs/path_feedback_batch_quasi_real_shadow_alignment_candidate_v1 \
  --output-root outputs/path_feedback_batch_quasi_real_safe_alternative_opportunity_diagnosis_v1 \
  --config configs/quasi_real_safe_alternative_opportunity_diagnosis_v1.json
```

## Non-Goals

- Do not run PPO update.
- Do not write PPO transitions.
- Do not execute policy takeover.
- Do not publish or replace checkpoints.
- Do not modify network/action space/default A*.
- Do not relax distance, path-risk, or source-selection gates.
- Do not claim Ackermann-feasible trajectory.
- Do not claim policy performance improvement.
