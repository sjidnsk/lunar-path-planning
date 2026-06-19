# Xunce Stage 18I.4 In-Process Proposal Validation Closure

## Summary

Stage 18I.4 closes the missing validation link between a newly generated
frontier/NBV proposal cell and a formal candidate action. The default path no
longer creates a temporary path-feedback CLI contract for every scenario. It
constructs an in-memory `ModelExplorerContract` copy, injects proposal cells as
temporary `GoalCandidate` rows, calls
`model_explorer.policy.planning.evaluate_candidate_paths()`, and only copies
candidate-level planner/path-feedback evaluation fields into the formal action
set.

This stage is still offline evidence plumbing. It does not train PPO, publish a
checkpoint, replace the default policy, connect an executor, start an online
canary, or download new map data.

## Validation Contract

The new adapter lives in `scripts/xunce_frontier_nbv_validation.py` as
`validate_candidate_cells()`.

Default validation mode:

```text
in_process_evaluate_candidate_paths
```

Legacy fixture-only mode:

```text
deterministic_exact_cell_match
```

For the in-process mode:

- `reachable=true` on the temporary proposal goal is only a planner-attempt seed.
- Final `reachable` is copied from `PathCandidateEvaluation.to_dict()`.
- Formal candidates must have planner/path-feedback-backed `reachable`,
  `path_cost`, `risk`, `path_length`, and `open_grid_fallback_used=false`.
- Proposal rows that fail validation remain `proposal_only=true` and enter audit
  artifacts only.
- Whole path-feedback summary fields such as `selection_changed` and
  `selected_after_path_feedback` are not research conclusions for Stage 18I.4.

## Provenance Rules

Planner/path-feedback validates candidate feasibility, route cost, route length,
open-grid fallback state, and whatever risk signal is available from the route
evaluation.

Coverage is different. Endpoint, path-line, and ROI-weighted coverage remain
offline geometric counterfactual estimates:

```text
coverage_validated_by_path_feedback=false
coverage_validation_source=offline_geometric_counterfactual_not_path_feedback
```

Risk provenance must be explicit:

```text
risk_source=planner_route_result
risk_source=goal_experimental_fallback
risk_source=sidecar_cost_proxy_no_path_risk
```

## Stage 18I.3 Integration

`scripts/run_xunce_true_frontier_nbv_candidate_source_replacement.py` now reads
each Stage 18A slice contract, sidecar, start cell, and scenario id, then calls
`validate_candidate_cells()`. The formal
`xunce-high-fidelity-path-feedback-audit.json` only contains validated
candidates. Failed proposals are written to proposal validation JSONL and audit
artifacts.

New config fields:

```json
{
  "candidate_validation_mode": "in_process_evaluate_candidate_paths",
  "min_validated_candidates_per_scenario": 3,
  "debug_validation_artifacts": false
}
```

Default output avoids per-scenario temporary contract/manifest/report artifacts
to keep `outputs/` bounded. The CLI path-feedback path is retained only in a
small integration smoke test for schema, sidecar, `PYTHONPATH`, and Windows path
compatibility.

## Acceptance

Stage 18I.3 after Stage 18I.4 should report:

```text
status=passed
comparison_allowed=true
candidate_validation_mode=in_process_evaluate_candidate_paths
valid_candidate_count >= 72
insufficient_validated_candidate_scenario_count=0
formal_candidate_missing_validation_count=0
open_grid_fallback_count=0
fallback_action_index_0_count=0
next_required_change=rerun_true_model_inference_and_binding
```

After this, the project must rerun Stage 18B/G/H/F/C on the new candidate root
before making any statement about Xunce versus incumbent.

## Verification

```powershell
D:\conda_envs\lunar-explorer\python.exe -m pytest `
  tests\test_xunce_frontier_nbv_validation.py `
  tests\test_xunce_frontier_nbv_validation_cli_integration.py `
  tests\test_xunce_true_frontier_nbv_candidate_source_replacement.py `
  tests\test_xunce_high_fidelity_real_map_comparison.py `
  tests\test_xunce_true_incumbent_selection_binding.py `
  tests\test_xunce_risk_coverage_cost_quantization_audit.py `
  tests\test_xunce_oracle_separability_benchmark.py `
  tests\test_platform_stage_runner.py `
  -q

D:\conda_envs\lunar-explorer\python.exe -m py_compile `
  scripts\xunce_frontier_nbv_validation.py `
  scripts\run_xunce_true_frontier_nbv_candidate_source_replacement.py `
  scripts\run_stage.py

git diff --check
```

