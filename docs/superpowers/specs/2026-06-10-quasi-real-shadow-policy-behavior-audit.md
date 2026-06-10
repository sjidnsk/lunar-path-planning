# Quasi-Real Shadow Policy Behavior Audit v1

## Summary

This stage follows `quasi_real_map_domain_gap_evaluated`. LOLA quasi-real ROI
slices are already bridged into path-feedback and the domain-gap verdict is
`acceptable_for_next_pilot`. The goal is to audit what the current experimental
policy would choose on those quasi-real contexts without taking control,
training, or writing PPO rollout transitions.

## Artifacts

- `configs/quasi_real_shadow_policy_behavior_audit_v1.json`
- `scripts/run_quasi_real_shadow_policy_behavior_audit.py/.sh`
- `scripts/run_quasi_real_shadow_policy_behavior_closure.sh`
- `outputs/path_feedback_batch_quasi_real_shadow_policy_behavior_v1/`
- `quasi-real-shadow-policy-decisions.jsonl`
- `quasi-real-shadow-policy-behavior-summary.json`
- `quasi-real-shadow-policy-rejection-report.json`
- `quasi-real-shadow-policy-group-report.md`

## Behavior

The audit reads the guarded experimental candidate and the quasi-real
path-feedback/domain-gap root. It scores each quasi-real context with the policy
and emits exactly one shadow decision classified as:

- `source_aligned`
- `policy_changed_gate_passed`
- `policy_changed_gate_rejected`
- `not_scored`

Each decision must retain `context_id`, ROI metadata, source and raw policy
action indices, logit margin, action-mask validity, path/risk deltas, and gate
reason codes. The policy does not execute a controlled choice and no PPO update
is run.

## Acceptance

- `status=passed`, `reason_codes=[]`.
- `shadow_context_count>=12`.
- `policy_decision_count == shadow_context_count`.
- `roi_group_count>=4`, with every ROI group covered.
- `context_id_missing_count=0`.
- invalid action mask, fallback/open-grid, safety, contract, path/risk, and
  source-selection regression counts are all `0`.
- `runs_ppo_update=false`.
- `policy_takes_control=false`.
- `publishes_checkpoint=false`.
- `replaces_default_policy=false`.
- `performance_claimed=false`.
- `behavior_verdict=acceptable_for_quasi_real_guarded_pilot`.
- Readiness may advance only to `quasi_real_shadow_policy_behavior_audited`.

## Non-Goals

- No PPO optimizer update.
- No policy takeover or quasi-real guarded pilot.
- No checkpoint publication or default-policy replacement.
- No network/action-space/default-A* change.
- No distance/path-risk/source-selection contract relaxation.
- No Ackermann-feasible trajectory claim.
- No IRIS/GCS/path-planner diagnostic treated as training release evidence.
- No policy performance claim.
