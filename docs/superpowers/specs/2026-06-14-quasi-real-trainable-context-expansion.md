# Quasi-Real Trainable Context Expansion v1

## Summary

This stage follows the passed Horizon-5 expansion and the capacity-blocked
Scale-512 preflight. It does not run formal PPO. It audits and expands the
upstream quasi-real trainable context pool, materializes eligible train split
teacher-distillation raw slices into PPO transitions, and reruns Scale-512 with
at least 512 real, distinct train split contexts.

## Artifacts

- `configs/quasi_real_trainable_context_expansion_v1.json`
- `scripts/run_quasi_real_trainable_context_expansion.py`
- `scripts/run_quasi_real_trainable_context_expansion.sh`
- `scripts/run_quasi_real_trainable_context_expansion_closure.sh`
- `tests/test_quasi_real_trainable_context_expansion.py`
- `outputs/path_feedback_batch_quasi_real_trainable_context_expansion_v1/quasi-real-trainable-context-expansion-summary.json`
- `outputs/path_feedback_batch_quasi_real_trainable_context_expansion_v1/quasi-real-trainable-contexts.jsonl`
- `outputs/path_feedback_batch_quasi_real_trainable_context_expansion_v1/quasi-real-trainable-context-expansion-steps.jsonl`
- `outputs/path_feedback_batch_quasi_real_trainable_context_expansion_v1/quasi-real-trainable-context-capacity-audit.json`
- `outputs/path_feedback_batch_quasi_real_trainable_context_expansion_v1/quasi-real-trainable-context-source-audit.json`
- `outputs/path_feedback_batch_quasi_real_trainable_context_expansion_v1/quasi-real-trainable-context-rejection-report.json`
- `outputs/path_feedback_batch_quasi_real_trainable_context_expansion_v1/quasi-real-trainable-context-expansion-report.md`

## Contract

The runner requires:

- Horizon-5 summary passed with `horizon>=5`, replay stability, teacher
  agreement at least 0.95, and zero controlled regression.
- Scale-512 summary failed only because of
  `insufficient_quasi_real_trainable_capacity`.

Only train split, controlled-policy, gate-clean rows with observation,
log_prob, value, finite reward, finite discounted return, and finite advantage
can be selected. Validation/test, fallback, teacher fallback, raw probe
rejection, non-empty gate reasons, controlled regressions, and
path-planner/IRIS/GCS diagnostics remain diagnostic-only.

Teacher-distillation raw slices are not counted directly. They become trainable
only after reconstructing `PolicyObservation` from the paired path-feedback
candidates and refreshing log_prob/value from the same experimental candidate
checkpoint used by the seed PPO smoke.

The stage never duplicates contexts to reach capacity. Repeated source
`context_id` values are audited separately and do not count toward the selected
unique trainable context pool.

When capacity reaches 512 unique contexts, the runner rebuilds an
`expanded_horizon5/` ledger with horizon at least 5 and multi-step discounted
returns, then reruns Scale-512 preflight. Only a passed Scale-512 rerun can
advance readiness to `quasi_real_guarded_ppo_scale512_multiseed_preflight_evaluated`.

## Current Evidence

The current closure passes:

- `status=passed`
- `reason_codes=[]`
- `source_row_count=1128`
- `materialized_source_row_count=648`
- `materialized_trainable_context_count=648`
- `unique_trainable_context_count=684`
- `ppo_trainable_transition_count=684`
- `duplicate_trainable_context_count=0`
- `duplicate_source_trainable_context_count=126`
- `scale512_status=passed`
- Scale-512 `seed_count=3`, `passed_seed_count=3`
- Scale-512 old log_prob/value max abs error `0.0 / 0.0`
- Scale-512 `seed_max_abs_approx_kl≈1.01e-5`
- Scale-512 `seed_max_grad_norm_after_clip=1.0`
- `readiness_status=quasi_real_guarded_ppo_scale512_multiseed_preflight_evaluated`

This clears the upstream capacity blocker without duplicating contexts or
relaxing gates. It is still a preflight result, not a formal PPO training
completion or performance claim.

## Validation

```bash
P=/home/kai/anaconda3/envs/lunar-explorer/bin/python

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $P -m pytest -q \
  tests/test_quasi_real_trainable_context_expansion.py \
  tests/test_quasi_real_guarded_ppo_scale512_multiseed_preflight.py \
  tests/test_policy_training_readiness_review.py

PYTHON=$P bash scripts/run_quasi_real_trainable_context_expansion_closure.sh

jq '{status, reason_codes, unique_trainable_context_count, ppo_trainable_transition_count, duplicate_trainable_context_count, scale512_status, readiness_status}' \
  outputs/path_feedback_batch_quasi_real_trainable_context_expansion_v1/quasi-real-trainable-context-expansion-summary.json

git diff --check
```

## Non-Goals

No formal PPO, checkpoint publication, default-policy replacement,
network/action-space/default-A* change, gate relaxation, duplicated-context
scale inflation, Ackermann-feasible trajectory claim, policy performance claim,
formal-training-ready claim, or IRIS/GCS/path-planner diagnostic training
release is made by this stage.
