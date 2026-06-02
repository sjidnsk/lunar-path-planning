# Core Algorithm Development Chain

Date: 2026-06-02

Status: project-level reference for the next implementation stages.

This document records the algorithm development chain that should guide future
`lunar-path-planning` work after the Current-HEAD evidence refresh, Policy
Decision Robustness v1, and Policy-Robustness Application Smoke v1.

The development endpoint is core algorithm progress, not another standalone
interpretability or audit layer. Audit JSON remains useful as acceptance
evidence, but the next stages must change path-planning or execution-loop
behavior in controlled, testable steps.

## Current Repo Evidence

The current project has three stable boundaries:

- `dev-platform-constraints` exports quasi-real scenarios,
  `model-explorer-contract/v1`, and `path-planner-sidecar/v1` with real
  `cost` and `passable_mask`.
- `path-planner` is the execution evaluator. Its main route remains
  `trajectory_kind=geometric_path`. A*, postprocess, tracking simulation, and
  fixed-corridor optimization are active behavior; IRIS and region-graph
  outputs are still diagnostic.
- `model-explorer` already consumes path-feedback diagnostics as decision,
  rollout, and training-selection signals. It can penalize path failure,
  replan, trajectory-optimization fallback, IRIS fallback, and region-graph
  fallback/disconnect.

The latest evidence root is:

```text
outputs/path_feedback_batch_policy_input
```

At this point, policy-side evidence is strong enough to stop adding separate
explanatory stages:

- batch open-grid fallback count is `0`;
- batch/stability/sample-quality/robustness/application summaries are clean;
- Policy-Robustness Application Smoke v1 records 21 decisions, with 10 changed
  by the selected robustness profile and 11 stable;
- remaining stress exposure is execution-layer dominated: path failure,
  replan, IRIS fallback, region-graph fallback, and region-graph disconnect.

The next work should therefore move these signals upstream into path-planning
algorithm behavior.

## Development Rule

Future `/goal` prompts and implementation plans should reference this document
and choose the first incomplete stage in the chain unless current repo evidence
proves a different stage is blocking.

Do not insert another standalone readiness or explanation phase between the
current state and Stage 1. Readiness checks are acceptance criteria inside each
algorithm stage, not a separate endpoint.

Every stage must preserve these scope guards unless the route contract is
intentionally changed in a later approved design:

- keep `path-planner-route/v1` stable;
- keep top-level `trajectory_kind=geometric_path`;
- keep `model-explorer-contract/v1` and `path-planner-sidecar/v1` stable;
- do not change PPO as the main training path;
- do not change the policy network architecture;
- do not expand the candidate-list action space;
- do not claim Ackermann-feasible trajectories;
- do not treat IRIS or region-graph diagnostics as GCS trajectory proof or
  vehicle-feasibility proof;
- do not use `IrisNp`, `IrisNp2`, `IrisZo`, or C-space IRIS in the current
  workspace-geometric stages.

## Stage 0: Integration / Provenance Gate

Purpose: make the current evidence-producing code part of HEAD before starting
new algorithm work.

Implementation target:

- commit or otherwise archive the current robustness/application smoke config,
  scripts, tests, README updates, and roadmap updates;
- regenerate the evidence root from the committed HEAD;
- verify parent and submodule git provenance is clean.

Acceptance gate:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/kai/anaconda3/envs/lunar-explorer/bin/python -m pytest tests
bash scripts/run_batch_path_feedback_validation.sh --matrix configs/path_feedback_batch_dataset_v1.json --output-root outputs/path_feedback_batch_policy_input
bash scripts/run_path_feedback_stability_analysis.sh --batch-root outputs/path_feedback_batch_policy_input
bash scripts/run_sample_quality_training_application.sh --batch-root outputs/path_feedback_batch_policy_input --config configs/sample_quality_training_application_v1.json --validate-only
bash scripts/run_sample_quality_training_application.sh --batch-root outputs/path_feedback_batch_policy_input --config configs/sample_quality_training_application_v1.json
bash scripts/run_policy_decision_robustness_analysis.sh --batch-root outputs/path_feedback_batch_policy_input --config configs/policy_decision_robustness_v1.json
bash scripts/run_policy_robustness_application_smoke.sh --batch-root outputs/path_feedback_batch_policy_input --robustness-summary outputs/path_feedback_batch_policy_input/policy-decision-robustness-summary.json --config configs/policy_robustness_application_smoke_v1.json
cd model-explorer && PYTHONPATH=src /home/kai/anaconda3/envs/lunar-explorer/bin/python -m model_explorer verify
git diff --check
```

Exit condition:

- all validation commands pass;
- `open_grid_fallback_used_count == 0`;
- source summaries, acceptance metadata, and git provenance are clean.

## Stage 1: Region-Graph-Guided Geometric Search v1

Purpose: make `region_graph_report` influence geometric path generation, rather
than only explaining failures after A*.

Current implementation note: Stage 1 is implemented as an opt-in
`path-planner` backend selected by `--planning-backend region_graph_guided`.
The default backend remains `astar`. The parent batch matrix opts into this
backend through `planner_extra_args` so validation exercises the Stage 1 path
without changing `path-planner` default CLI behavior.

Implementation target:

- add an optional `path-planner` planning backend such as
  `region_graph_guided`;
- when the region graph is connected, derive a waypoint skeleton from region
  centers or region-edge transitions;
- run segment-level A* between skeleton waypoints and stitch the segments into
  a candidate `geometric_path`;
- fall back to existing A* when the graph is disconnected, invalid, or a segment
  fails;
- record backend source, fallback reason, segment count, and comparison against
  baseline A* in optional additive diagnostics.

Implemented diagnostic surface:

- optional top-level `planning_backend_report`;
- `requested_backend`, `selected_backend`, `status`, and `fallback_reason`;
- `segment_count`, `skeleton_cells`, and baseline/candidate path summaries;
- `comparison` and `region_graph_candidate` blocks for path-feedback consumers.

Acceptance criteria:

- existing A* route behavior remains the default;
- route JSON still uses `trajectory_kind=geometric_path`;
- stress or mixed-stress runs show a measurable path-planning behavior change,
  or emit machine-readable blockers such as
  `region_graph_disconnected`, `segment_astar_failed`, or
  `region_graph_candidate_not_better`;
- model-explorer path-feedback can consume the new diagnostics without stable
  contract changes.

Non-goals:

- no GCS solver;
- no vehicle motion-feasibility solver;
- no Ackermann-feasible trajectory claim.

## Stage 2: Workspace IRIS Region Quality v1

Purpose: improve the quality and usability of workspace IRIS regions before
using them as a stronger region graph source.

Current implementation note: Stage 2 keeps IRIS behind the optional
`--drake-iris-regions` diagnostic switch. The batch/single-run validation chain
must use the configured Conda Python consistently so Drake availability is
auditable. Workspace IRIS diagnostics may use merged blocked rectangles and
safe-component domain boxes, but route output remains a geometric path.

Implementation target:

- improve workspace `Iris` seed and domain selection while staying in 2D
  workspace geometry;
- reduce over-fragmented obstacle primitives where appropriate, without hiding
  unsafe cells;
- strengthen IRIS validation for seed containment, unsafe-cell intersection,
  empty regions, and domain bounds;
- make `region_graph_report.region_source = "iris"` reproducible when all
  regions are valid and first-to-last graph connectivity holds.

Acceptance criteria:

- default tests do not require `pydrake`;
- no-Drake behavior falls back cleanly to `grid_box`;
- Drake-enabled validation reduces `iris_fallback_count` and
  `region_graph_fallback_count`, or reports precise machine-readable blockers;
- no C-space IRIS or IRIS NP variants are introduced.

## Stage 3: Sampled Region Path Backend v1

Purpose: generate a geometric path from project-owned region models, not only
from grid A* or postprocess corridors.

Implementation target:

- search `RegionGraph` for a start-to-goal region sequence;
- sample safe points inside each region and across each region edge;
- build a sampled geometric candidate path from that sequence;
- compare sampled-region output against baseline A* and fixed-corridor
  optimization using path cost, path length, high-cost exposure, corridor
  validity, and tracking proxy;
- emit an additive report such as `sampled_region_path_report/v1`.

Acceptance criteria:

- sampled-region output can change the geometric candidate path on selected
  stress fixtures;
- failed sampled-region generation falls back to existing A* / postprocess /
  fixed-corridor behavior;
- model-explorer can include sampled-region source/fallback fields in
  path-feedback diagnostics.

Non-goals:

- no Drake GCS solver yet;
- no top-level route contract change;
- no motion-feasible trajectory claim.

## Stage 4: Drake GCS Geometric Backend v1

Purpose: introduce a real GCS-style geometric backend after project-owned region
models and sampled-region behavior are stable.

Implementation target:

- map `ConvexRegion` and `RegionEdge` into a GCS backend boundary;
- solve for a 2D geometric safe-region path candidate;
- emit additive `gcs_trajectory_report/v1` fields such as backend, status,
  solver status, region source, sampled path, path cost, warnings, and fallback
  status;
- keep the current fixed-corridor optimizer behind `--optimize-trajectory` and
  expose GCS through a separate switch.

Acceptance criteria:

- GCS success produces a comparable geometric candidate path;
- GCS failure follows the existing fallback chain:
  current optimizer -> postprocess smoothed path -> raw A* path;
- no GCS output is labeled as Ackermann-feasible or vehicle-executable.

## Stage 5: Motion-Feasibility Layer v1

Purpose: separate geometric path generation from vehicle motion feasibility.

Implementation target:

- add additive `motion_feasibility_report/v1`;
- begin with `point` and `curvature_bounded` models before any rover-specific
  Ackermann, skid-steer, or differential model;
- report fields such as `motion_model`, `min_turning_radius_m`, `can_reverse`,
  `can_turn_in_place`, `max_heading_change_deg`, `feasibility_status`,
  `violation_indices`, and `fallback_reason`;
- allow downstream consumers to distinguish `not_evaluated`,
  `diagnostic_only`, `feasible`, and `infeasible`.

Acceptance criteria:

- only paths validated by this layer can be described as motion-feasible;
- route compatibility is preserved when the report is absent;
- model-explorer can consume feasibility status as a path-feedback signal.

Non-goals:

- no rover-specific feasibility claim until the motion model is explicitly
  represented and tested;
- no Ackermann-feasible claim for geometric or GCS-only paths.

## Stage 6: Execution-Aware Explorer Loop v1

Purpose: connect improved execution backends back into model-explorer decisions,
rollouts, and training data.

Implementation target:

- allow path-feedback manifests to select the execution backend;
- compare baseline A*, region-guided, sampled-region, GCS, and
  motion-feasibility-aware signals where available;
- record backend source, path candidate source, fallback status, and motion
  feasibility in rollout info;
- keep PPO, network architecture, and candidate-list action space unchanged;
- use improved execution labels as better training data, not as a training
  architecture change.

Acceptance criteria:

- batch/stability/sample-quality/robustness/application validation remains
  clean;
- selected action/cell or path candidate changes are traceable to execution
  backend behavior;
- when execution algorithms do not improve stress failures, blockers are
  attributed back to `path-planner`, not hidden by policy weights.

## Stage 7: Core Algorithm Closure

Purpose: close the full algorithmic loop:

```text
terrain/confidence/costmap
  -> candidate goals
  -> execution-aware path generation
  -> motion feasibility
  -> explorer decision
  -> rollout/training feedback
```

This is the first point where larger contract questions become appropriate:

- whether to expand the action space beyond the candidate list;
- whether target generation should be directly constrained by execution
  feasibility;
- whether to introduce rover-specific motion models;
- whether `trajectory_kind` needs a new value beyond `geometric_path`.

Do not start Stage 7 directly. It depends on the earlier staged behavior and
fallback evidence.
