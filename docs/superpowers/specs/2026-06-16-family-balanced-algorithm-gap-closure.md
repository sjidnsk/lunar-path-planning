# Family-Balanced Algorithm Gap Closure v1

## Background

Stage 14/15/16 closed the front half of checkpoint publication governance, but
Stage 16 intentionally pauses before `default_policy_candidate_sandbox_install_preflight`.
The next model-explorer mainline returns to algorithm evidence because the 5B.7
cost-efficiency candidate is still family-imbalanced:
`low_observation_count=9` versus much larger non-low-observation families.

## Implementation

- Runner: `scripts/run_family_balanced_algorithm_gap_closure.py`
- Shell entrypoint: `scripts/run_family_balanced_algorithm_gap_closure.sh`
- Output root:
  `outputs/path_feedback_batch_family_balanced_algorithm_gap_closure_v1/`

The runner reads Stage 16, Stage 15/14 lineage evidence, Stage 5B.7
cost-efficiency outputs, low-observation geometry outputs, and Stage 5B.3
safe-better four-family outputs. It recomputes family counts from json/jsonl
inputs, keeps 5B.7 cost-efficiency-filtered non-low-observation rows, and
replaces the thin 5B.7 low-observation slice with source-backed
low-observation geometry supplement rows.

## Output Contract

The stage writes:

- `family-balanced-algorithm-gap-closure-summary.json`
- `family-balanced-source-ledger.json`
- `family-balance-audit.json`
- `low-observation-gap-audit.json`
- `family-balanced-safe-better-pairs.jsonl`
- `family-balanced-counterfactual-rollouts.jsonl`
- `family-balanced-coverage-driven-input-manifest.json`
- `family-balanced-release-boundary-audit.json`
- `family-balanced-rejection-report.json`
- `family-balanced-algorithm-gap-closure-report.md`

Passing summary contract:

- `status=passed`
- `reason_codes=[]`
- `gap_closure_verdict=eligible_for_family_balanced_coverage_driven_ppo_rerun`
- `family_balanced_algorithm_gap_closure_passed=true`
- `family_balanced_coverage_driven_ppo_rerun_approved=true`
- `low_observation_gap_closed=true`
- `source_backed_counterfactual_audit_passed=true`
- `family_balance_audit_passed=true`
- `lineage_audit_passed=true`
- `release_boundary_audit_passed=true`
- `next_required_change=family_balanced_coverage_driven_ppo_rerun`
- `checkpoint_publication_approved=false`
- `default_policy_replacement_approved=false`
- `real_executor_connection_approved=false`
- `publishes_checkpoint=false`
- `replaces_default_policy=false`
- `connects_real_executor=false`

Default family-balance gates:

- all four target families are present;
- `low_observation_count >= 32`;
- `minimum_family_pair_count >= 32`;
- `low_observation_count / max_family_count >= 0.15`;
- trainable output rows use `split=train` only.

## Failure Reasons

- `stage16_not_passed`
- `stage16_release_boundary_violation`
- `cost_efficiency_source_not_passed`
- `low_observation_source_missing`
- `low_observation_gap_not_closed`
- `family_balance_below_threshold`
- `counterfactual_source_missing`
- `fallback_gain_contamination`
- `controlled_regression_detected`
- `non_train_split_in_trainable_output`
- `release_boundary_violation`
- `docs_not_updated`

## Verification

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
$PY -m pytest tests/test_family_balanced_algorithm_gap_closure.py \
  tests/test_low_observation_candidate_geometry_improvement.py \
  tests/test_cost_efficiency_aware_coverage_reward_candidate_filter.py -q

PYTHON=$PY bash scripts/run_family_balanced_algorithm_gap_closure.sh

jq '{status,reason_codes,gap_closure_verdict,family_balanced_algorithm_gap_closure_passed,low_observation_gap_closed,next_required_change,checkpoint_publication_approved,default_policy_replacement_approved,real_executor_connection_approved}' \
  outputs/path_feedback_batch_family_balanced_algorithm_gap_closure_v1/family-balanced-algorithm-gap-closure-summary.json

git diff --check
```

## Non-Goals

- Does not continue to Stage 17 default-policy sandbox install preflight.
- Does not run PPO update.
- Does not publish checkpoint.
- Does not replace default policy.
- Does not connect real executor.
- Does not relax guard.
- Does not modify network/action space/default A*.
- Does not claim Ackermann-feasible trajectory or real-world performance.
- Does not treat IRIS/GCS/path-planner diagnostics as release proof.
