# Generated Sequential Long-Horizon Teacher-Skill Contract Alignment v1

## Summary

This stage aligns the generated sequential contract with the teacher-skill
mainline. The mainline is not "choose a different target than the teacher." It
is controlled multi-step behavior that is teacher-equivalent or better under
cumulative return. Beyond-teacher evidence is a value branch and must be proven
by long-horizon return dominance.

The stage does not run PPO, relax generated sequential gates, publish
checkpoints, or claim formal training readiness.

## Inputs

- Generated sequential gate metric/accounting audit summary.
- Quasi-real/generated sequential compatibility diagnosis summary.
- Base and updated generated sequential replay summaries, steps, and rejection
  reports.
- Quasi-real post-update teacher-following summary.
- Quasi-real post-update collector summary.

## Artifacts

- `configs/generated_sequential_long_horizon_teacher_skill_contract_alignment_v1.json`
- `scripts/run_generated_sequential_long_horizon_teacher_skill_contract_alignment.py`
- `scripts/run_generated_sequential_long_horizon_teacher_skill_contract_alignment.sh`
- `scripts/run_generated_sequential_long_horizon_teacher_skill_contract_alignment_closure.sh`
- `outputs/path_feedback_batch_generated_sequential_long_horizon_teacher_skill_contract_alignment_v1/`

Outputs:

- `long-horizon-teacher-skill-contract-summary.json`
- `teacher-vs-policy-return-comparison.jsonl`
- `teacher-equivalent-episode-report.md`
- `beyond-teacher-opportunity-report.md`
- `dominated-raw-choice-diagnostics.jsonl`

## Contract

The runner reuses existing generated sequential episode and step records. It
does not invoke a new planner in v1. For each episode it computes:

- teacher cumulative return
- controlled policy cumulative return
- raw policy diagnostic return

The configurable return includes path cost, risk, safety/contract/source
selection penalties, and progress/terminal proxies. Episode classifications are:

- `teacher_aligned_active_choice`
- `teacher_equivalent_episode`
- `beyond_teacher_episode`
- `dominated_raw_choice`
- `controlled_regression_episode`

Same-as-teacher active choices are positive teacher-skill evidence. Raw rejected
choices remain diagnostic but do not count as controlled cumulative regression.

## Readiness

Readiness accepts
`--generated-sequential-long-horizon-teacher-skill-contract-summary`.

Only `status=passed` with
`verdict=long_horizon_teacher_skill_contract_aligned` can remove the generated
sequential contract-alignment blocker. Failed, inconclusive, missing/stale, or
controlled-regression summaries keep readiness at
`needs_training_contract_refinement`. This stage must not claim formal training
readiness.

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_generated_sequential_long_horizon_teacher_skill_contract_alignment.py \
  tests/test_generated_sequential_gate_metric_accounting_audit.py \
  tests/test_policy_gated_sequential_canary_rollout.py \
  tests/test_quasi_real_generated_sequential_contract_compatibility_diagnosis.py \
  tests/test_policy_training_readiness_review.py

PYTHON=$P bash scripts/run_generated_sequential_long_horizon_teacher_skill_contract_alignment_closure.sh

jq '{status, reason_codes, verdict, teacher_equivalent_episode_count, beyond_teacher_episode_count, dominated_raw_choice_count, controlled_regression_episode_count, recommended_next_action}' \
  outputs/path_feedback_batch_generated_sequential_long_horizon_teacher_skill_contract_alignment_v1/long-horizon-teacher-skill-contract-summary.json

git diff --check
```

## Non-Goals

- No new PPO update.
- No checkpoint publication or default-policy replacement.
- No network/action-space/default A* change.
- No distance/path-risk/source-selection gate relaxation.
- No Ackermann-feasible trajectory claim.
- No IRIS/GCS/path-planner diagnostic promotion to training release evidence.
- No generated sequential gate removal.
