# Xunce Stage 18.4D In-Process Batch A* Validation

## Summary

Stage 18.4D changes the default dynamic rollout validation path from
`PathPlannerRouteAdapter` CLI-per-candidate calls to an in-process batch wrapper
around `path_planner.search.AStarPlanner.plan()`.

The intent is performance and evidence clarity:

- dynamic rollout can regenerate and validate proposal candidates every step;
- each step loads the sidecar/grid once and plans all proposal routes in process;
- `PathPlannerRouteAdapter` remains available for explicit smoke or sampled
  audit, but is not the default full-rollout path;
- batch A* evidence is labeled `in_process_astar_screening`, not full adapter
  evidence.

## Mainline Behavior

Stage 18.4 mainline uses:

```text
candidate_refresh_mode=dynamic_frontier_nbv_in_process
dynamic_candidate_validation_mode=in_process_path_planner_astar_batch
```

Formal rollout candidates still require:

```text
proposal_validated_by_path_feedback=true
proposal_only=false
reachable=true
open_grid_fallback_used=false
finite path_cost/path_length/risk
```

Coverage remains an offline geometric counterfactual and is not planner
validated.

## Evidence Semantics

Batch A* rows use:

```text
planner_validation_backend=in_process_path_planner_astar_batch
validation_evidence_kind=in_process_astar_screening
```

`dynamic_validation_full_adapter_evidence_passed` must remain `false` unless
full `PathPlannerRouteAdapter` validation is explicitly run and passes.
Sample adapter audit is explicitly controlled by:

```text
dynamic_adapter_audit_enabled=true
dynamic_adapter_audit_max_routes=128
```

When enabled, the audit samples batch-A* formal candidates and reruns only those
routes through `PathPlannerRouteAdapter`. The audit may set
`adapter_audit_passed=true`, but it still does not turn batch-A* rollout evidence
into full adapter evidence.

Stage 18.4 reports:

```text
closed_loop_dynamic_rollout_summary
same_candidate_set_policy_selection_summary
in_process_batch_astar_validation_count
validation_evidence_kind_counts
planner_validation_backend_counts
```

## Non-Goals

This stage does not train PPO, actor, critic, evaluator, or reward models. It
does not publish checkpoints, replace default policy, connect executor, start
online canary, modify action space, or claim real executor feasibility.
