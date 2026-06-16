# Family-Balanced Coverage-Driven PPO Rerun v1

## Background

`Family-Balanced Algorithm Gap Closure v1` passed and routed the model-explorer
mainline to `family_balanced_coverage_driven_ppo_rerun`. Its balanced input has
720 safe-better pairs and 720 counterfactual rows across all four target
families, including a repaired low-observation slice.

This stage resumes offline guarded PPO training, but it is not a publication
stage and does not resume default-policy installation.

## Implementation

- Runner: `scripts/run_family_balanced_coverage_driven_ppo_rerun.py`
- Shell entrypoint: `scripts/run_family_balanced_coverage_driven_ppo_rerun.sh`
- Output root:
  `outputs/path_feedback_batch_family_balanced_coverage_driven_ppo_rerun_v1/`

The runner reads the family-balanced gap-closure root and creates:

- `family-balanced-compatible-stage5a2-input/`
- `family-balanced-compatible-coverage-driven-input/`

It materializes `family_balanced_ppo_advantage` from coverage gain, valuable
area, information gain, family-balance weight, and path/risk/energy penalties.
It synthesizes missing old-policy transitions for newly introduced
family-balanced decisions and runs the existing refined coverage-driven PPO
runner with `use_transition_info_advantage=true`.

## Output Contract

The stage writes:

- `family-balanced-coverage-driven-ppo-rerun-summary.json`
- `family-balanced-ppo-source-ledger.json`
- `family-balanced-compatible-input-audit.json`
- `family-balanced-old-transition-materialization-audit.json`
- `family-balanced-ppo-advantage-audit.jsonl`
- `family-balanced-performance-audit.json`
- `family-balanced-release-boundary-audit.json`
- `family-balanced-coverage-driven-ppo-rerun-rejection-report.json`
- `family-balanced-coverage-driven-ppo-rerun-report.md`

Passing summary contract:

- `status=passed`
- `reason_codes=[]`
- `rerun_verdict=eligible_for_family_balanced_shadow_canary_preflight`
- `family_balanced_coverage_driven_ppo_rerun_passed=true`
- `family_balanced_shadow_canary_preflight_approved=true`
- `family_balanced_input_audit_passed=true`
- `old_transition_materialization_audit_passed=true`
- `ppo_update_status=passed`
- `guard_replay_audit_passed=true`
- `coverage_return_improvement>0`
- `cumulative_coverage_rate_delta_improvement>0`
- `valuable_area_covered_improvement>0`
- `coverage_efficiency_regression=false`
- `policy_argmax_changed_count>0`
- `fallback_rate<0.5`
- `fallback_gain_contamination_count=0`
- `controlled_regression_count=0`
- `safe_better_training_family_count=4`
- `low_observation_trainable_transition_count>=32`
- `next_required_change=family_balanced_shadow_canary_preflight`
- release boundaries remain false.

## Failure Reasons

- `family_balanced_gap_closure_not_passed`
- `family_balanced_input_missing`
- `family_balance_regressed`
- `low_observation_transition_gap`
- `old_transition_materialization_failed`
- `family_balanced_advantage_missing`
- `ppo_update_failed`
- `guard_replay_failed`
- `post_update_policy_teacher_equivalent`
- `no_coverage_return_improvement`
- `no_cumulative_coverage_rate_delta_improvement`
- `valuable_coverage_regressed`
- `coverage_efficiency_regression`
- `fallback_dominates`
- `fallback_gain_contamination`
- `controlled_regression_detected`
- `release_boundary_violation`
- `docs_not_updated`

## Verification

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
$PY -m pytest tests/test_family_balanced_coverage_driven_ppo_rerun.py \
  tests/test_family_balanced_algorithm_gap_closure.py \
  tests/test_refined_coverage_driven_ppo_improvement_run.py -q

PYTHON=$PY bash scripts/run_family_balanced_coverage_driven_ppo_rerun.sh

jq '{status,reason_codes,rerun_verdict,family_balanced_coverage_driven_ppo_rerun_passed,coverage_return_improvement,cumulative_coverage_rate_delta_improvement,valuable_area_covered_improvement,coverage_efficiency_regression,next_required_change,checkpoint_publication_approved,default_policy_replacement_approved,real_executor_connection_approved}' \
  outputs/path_feedback_batch_family_balanced_coverage_driven_ppo_rerun_v1/family-balanced-coverage-driven-ppo-rerun-summary.json

git diff --check
```

## Non-Goals

- Does not publish checkpoint.
- Does not replace default policy.
- Does not connect real executor.
- Does not continue Stage 17 default-policy installation.
- Does not modify network/action space/default A*.
- Does not relax guard.
- Does not claim Ackermann-feasible trajectory.
- Does not claim real-world performance.
- Does not treat IRIS/GCS/path-planner diagnostics as release proof.
