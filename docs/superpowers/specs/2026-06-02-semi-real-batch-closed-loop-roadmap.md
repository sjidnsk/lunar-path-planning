# Semi-Real Batch Closed-Loop Roadmap

Date: 2026-06-02

## Current State

The repository has completed the semi-real closed-loop path-feedback path and
the dataset/sample-quality audit layer:

- `dev-platform-constraints` exports quasi-real `.npz` scenarios,
  `model-explorer-contract/v1`, and `path-planner-sidecar/v1`.
- `model-explorer` consumes path-feedback summaries through
  `train.system_calibration`, teacher gates, path-feedback gates, and explicit
  `system_calibration.sample_quality.enabled=true` dataset filtering or
  downweighting.
- `path-planner` remains the execution evaluator and returns
  `trajectory_kind=geometric_path`; IRIS and region-graph outputs are
  diagnostics only.
- The current acceptance command is:

```bash
bash scripts/run_path_feedback_validation.sh --scenario-set all --diagnostic-profile all --top-k 3
```

## Selected Next Direction

The first selected direction was option A: batch CLI first.

The completed **Semi-Real Batch Closed-Loop Evaluation v1** turns the current
single-run acceptance gate into a repeatable batch experiment entrypoint while
preserving the existing single-run script and JSON contracts.

Recommended entrypoint:

```bash
bash scripts/run_batch_path_feedback_validation.sh --matrix configs/path_feedback_batch_dataset_v1.json
```

The batch runner should orchestrate the existing
`scripts/run_path_feedback_validation.sh` command instead of replacing it.

Each run should write to an isolated output directory:

```text
outputs/path_feedback_batch/<run_id>/
```

Batch-level outputs should be machine-readable JSON:

```text
outputs/path_feedback_batch/batch-run-index.json
outputs/path_feedback_batch/batch-evaluation-summary.json
```

Implemented entrypoint:

```bash
bash scripts/run_batch_path_feedback_validation.sh --matrix configs/path_feedback_batch_dataset_v1.json --output-root outputs/path_feedback_batch_smoke
```

Batch v1 preserves the single-run script and writes each run to an isolated
directory under the batch output root. It records optional
`sample_quality_profile` metadata for later audit/stability stages, but does not
execute sample-quality filtering or training.

The current implemented direction is **Dataset / Decision Stability v1**. It
consumes a completed batch root and writes stability JSON without rerunning
training or changing the path-feedback contract:

```bash
bash scripts/run_path_feedback_stability_analysis.sh --batch-root outputs/path_feedback_batch_current_analysis
```

Implemented stability outputs:

```text
outputs/path_feedback_batch_current_analysis/batch-stability-summary.json
outputs/path_feedback_batch_current_analysis/dataset-quality-stability-summary.json
outputs/path_feedback_batch_current_analysis/decision-stability-summary.json
```

The analyzer validates `batch-run-index.json`, `batch-evaluation-summary.json`,
source `path-feedback-summary/v1` files, acceptance metadata, open-grid fallback
gates, failed batch runs, and parent/submodule git provenance. Validation
failures remain auditable through machine-readable reason codes; the command
returns nonzero when the input batch is not stability-clean.

The current implemented direction is **Sample-Quality-Aware Training
Application v1**. It consumes the current batch root plus stability summaries and
writes training application audit JSON without running PPO or changing the
network/action space:

```bash
bash scripts/run_sample_quality_training_application.sh --batch-root outputs/path_feedback_batch_training_input --config configs/sample_quality_training_application_v1.json
```

Implemented training application outputs:

```text
outputs/path_feedback_batch_training_input/sample-quality-training-application-summary.json
outputs/path_feedback_batch_training_input/training-selection-stability-summary.json
```

The application summary validates and records `batch-run-index.json`,
`batch-evaluation-summary.json`, `batch-stability-summary.json`,
`dataset-quality-stability-summary.json`, `decision-stability-summary.json`,
source `path-feedback-summary/v1` files, acceptance metadata, and git
provenance. It compares `legacy`, `hard_exclude_open_grid`, and
`soft_downweight_diagnostics` profiles. `open_grid_fallback` is a hard
exclusion for sample-quality-aware profiles; path failure, replan, IRIS
fallback, region-graph fallback, and region-graph disconnect are soft
downweighting signals. The selection stability summary compares legacy and
sample-quality-aware best-run/profile audit choices and explicitly labels the
result as no-training-metric evidence, not a single-run improvement claim.

## Batch v1 Scope

The first batch matrix should support:

- `scenario_set`: `smoke`, `stress`, and `all`;
- `diagnostic_profile`: default/lightweight, `execution`, `iris`, and `all`;
- `top_k`: at least `1` and `3`;
- explicit `run_id` and output root;
- optional `sample_quality_profile` recorded as metadata, without requiring
  training to run in v1.

`batch-run-index.json` should record for each run:

- run id;
- command arguments;
- summary path and report path;
- acceptance metadata;
- parent and submodule git SHAs;
- pass/fail status;
- machine-readable failure reason codes.

`batch-evaluation-summary.json` should aggregate:

- run pass/fail counts;
- `open_grid_fallback_used` gate results;
- scenario group metrics;
- path-planning failure and replan counts;
- IRIS fallback and region-graph fallback/disconnect counts;
- source summary paths for later sample-quality audit consumption.

The default batch runner continues after a run failure, writes both batch JSON
files, and returns a nonzero final exit code when any run fails.

## Follow-On Phases

After Batch Closed-Loop Evaluation v1, continue with:

1. **Dataset / Decision Stability v1**
   - Use batch outputs to identify stable failure patterns across ROI,
     scenario group, reason code, action, source summary, acceptance metadata,
     scenario set, diagnostic profile, and `top_k`.
   - Produce machine-readable stability summaries such as
     `batch-stability-summary/v1`, `dataset-quality-stability-summary/v1`, and
     `decision-stability-summary/v1`.
   - Implemented as
     `scripts/run_path_feedback_stability_analysis.sh --batch-root <batch-root>`.

2. **Sample-Quality-Aware Training Application v1**
   - Compare legacy data selection against explicit sample-quality profiles:
     hard `open_grid_fallback` exclusion and soft downweighting for path
     failure, replan, IRIS fallback, region-graph disconnect, and region-graph
     fallback.
   - Measure whether best-run selection becomes more stable across runs, not
     whether a single metric improves once.
   - Implemented as
     `scripts/run_sample_quality_training_application.sh --batch-root <batch-root> --config configs/sample_quality_training_application_v1.json`.

3. **Policy Decision Robustness v1**
   - Use stable path-feedback and sample-quality signals to improve candidate
     ranking sensitivity to unreachable targets, high replan risk, high path
     cost, and repeated diagnostic warnings.
   - Preserve the candidate-list action space.

4. **Execution-Layer Readiness**
   - Only after the batch and stability stages show sustained diagnostic value,
     evaluate workspace IRIS quality gates, region-graph readiness, GCS backend
     interface design, and a separate motion-feasibility layer.

## Non-Goals

This roadmap does not:

- design a new network architecture;
- expand the action space to the full map;
- replace PPO as the main training path;
- implement GCS graph search or a GCS trajectory backend;
- implement Ackermann, skid-steer, or differential-drive feasibility solving;
- claim that outputs are Ackermann-feasible trajectories;
- use IrisNp, IrisNp2, IrisZo, or C-space IRIS;
- treat IRIS or region-graph diagnostics as GCS trajectory proof or vehicle
  feasibility proof;
- claim quasi-real or mask-stress results are real-world generalization
  performance.
