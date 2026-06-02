# lunar-path-planning

System-level repository for lunar rover autonomous exploration and path planning.

This parent repository coordinates three Git submodules:

- `path-planner`
- `model-explorer`
- `dev-platform-constraints`

`path-planner` is the active in-repository replacement for the former
`a_gcs_ws-2.0.1` execution-layer reference. The old project is intentionally
excluded from the parent repository and is not a runtime dependency.

## Development Status

The three subprojects now form a staged research prototype:

| Subproject | Role | Current status |
|---|---|---|
| `dev-platform-constraints` | Modeling foundation | P0/P1/P2 are runnable: map contracts, terrain features, platform configs, hard constraints, confidence update, coverage-aware goal candidates, sequence scoring, and `model-explorer-contract/v1` reports. |
| `model-explorer` | Decision orchestration | Runnable synthetic decision/benchmark stack exists: contract loading, goal selection, loop/replan reasons, policy experiments, planning-result feedback, CLI-backed `path-planner` route evaluation, and optional system-level semi-real calibration that gates v5 distillation runs with path-feedback diagnostics. |
| `path-planner` | Path execution evaluation | Rebuilt from scratch through Phase 8: platform-aware A*, postprocess corridors, smoothing, curvature checks, trackable path, tracking simulation, fixed-corridor optimization, execution-aware metrics, and optional Drake IRIS/region graph diagnostics. |

Near-term integration focuses on the `dev-platform-constraints -> model-explorer
-> path-planner` JSON loop: generate `model-explorer-contract/v1`, select Top-K
goals, emit `path-planner-request/v1`, consume `path-planner-route/v1`, then feed
reachability, cost, safety, and fallback diagnostics back into goal ranking,
replanning triggers, and smoke/stress experiment reports.

`model-explorer` now has a `path_planner_route` planning backend for this loop.
It runs `path-planner` through the CLI/JSON boundary by default and keeps direct
imports across subprojects out of the decision-layer process. The current smoke
adapter can synthesize an open cost/passability grid when only stable contract
fields are available; real or semi-real experiments should provide actual cost
and hard-constraint arrays from `dev-platform-constraints`.

Semi-real validation starts by exporting sibling JSON artifacts:

```bash
cd dev-platform-constraints
python scripts/generate_npz_validation_maps.py --scenario-set smoke --output-dir data/validation_maps --scenario-config outputs/npz_validation_scenarios.generated.json
PYTHONPATH=src python scripts/export_path_planner_sidecars.py --scenario-config outputs/npz_validation_scenarios.generated.json --output-dir outputs/path_planner_sidecars
```

The export produces paired `model-explorer-contract/v1` and
`path-planner-sidecar/v1` files for the shadow corridor, rock field, and
low-confidence risk-band smoke scenarios. Use `--scenario-set stress` for
near-blocked, high-risk, dense-rock, and mixed reachable/blocked regression scenarios, or
`--scenario-set all` to generate both sets.

Then run `model-explorer` path feedback summary from a manifest that pairs each
contract with its sidecar:

```bash
cd model-explorer
PYTHONPATH=src python -m model_explorer path-feedback run path-feedback.json
```

The summary records target selection before and after path feedback, path
planning failures, replan counts, baseline-vs-feedback path cost deltas,
selection changed counts/rates, safety/optimization fallback indicators, IRIS
status/fallback diagnostics, region graph source/fallback/disconnect
diagnostics, scenario-group aggregates, and whether any open-grid fallback was
used.

## One-Click Semi-Real Closed-Loop Validation

The parent repository provides a reproducible validation entrypoint for the
current semi-real loop:

```bash
bash scripts/run_path_feedback_validation.sh --dry-run
bash scripts/run_path_feedback_validation.sh --top-k 3
bash scripts/run_path_feedback_validation.sh --scenario-set stress --top-k 3
bash scripts/run_path_feedback_validation.sh --scenario-set all --simulate-tracking
bash scripts/run_path_feedback_validation.sh --scenario-set all --diagnostic-profile iris
bash scripts/run_path_feedback_validation.sh --scenario-set all --diagnostic-profile all --top-k 3
bash scripts/run_path_feedback_validation.sh --scenario-set all --diagnostic-profile all --top-k 3 --output-root outputs/path_feedback_validation_next_stage
```

The next-stage acceptance gate is the final command above:
`--scenario-set all --diagnostic-profile all --top-k 3`. It exercises smoke and
stress scenarios, forwards execution and optional workspace IRIS diagnostics,
and still treats IRIS/region-graph output as diagnostic evidence only. It does
not introduce a GCS trajectory backend or a rover motion-feasibility solver.

The script initializes/checks the three submodules, generates the fixed `.npz`
validation maps, exports paired `model-explorer-contract/v1` and
`path-planner-sidecar/v1` JSON files, writes a `path-feedback-manifest/v1`
with `scenario_set`, `diagnostic_profile`, `acceptance_gate`, `top_k`,
planner extra args, and open-grid fallback gate metadata, and runs:

```bash
PYTHONPATH=src python3 -m model_explorer path-feedback validate <manifest>
PYTHONPATH=src python3 -m model_explorer path-feedback run <manifest>
```

The `path-feedback run` command prints a compact stdout summary and writes the
full experiment JSON plus Markdown report to the configured output files. The
JSON summary repeats the same `acceptance_metadata` and records the actual
`open_grid_fallback_used_gate` result. The root script can forward optional execution diagnostics with
`--simulate-tracking`, `--optimize-trajectory`, and `--drake-iris-regions`; the
default remains lightweight and Drake-free. `--diagnostic-profile execution`
forwards tracking simulation plus fixed-corridor optimization, `iris` forwards
optional workspace IRIS diagnostics, and `all` forwards both groups.
The Markdown report now includes Diagnostic Interpretation and Candidate
Diagnostics tables so stress and mixed-stress runs can explain target
replacement, path-planning failure, replan triggers, IRIS fallback, region-graph
fallback/disconnection, and open-grid fallback separately.

By default, generated artifacts are written under
`outputs/path_feedback_validation/`:

- `npz_validation_scenarios.json`
- `path_planner_sidecars/*.contract.json`
- `path_planner_sidecars/*.path-planner-sidecar.json`
- `path-feedback-manifest.json`
- `path-feedback-summary.json`
- `path-feedback-summary.md`

The script fails if the summary does not contain the expected
`path-feedback-summary/v1` shape, the expected scenario set, at least three
evaluated candidates, the core path feedback metrics, selection-change metrics,
scenario/candidate diagnostic interpretation fields, or
`open_grid_fallback_used = false`. The final condition is the credibility gate:
semi-real conclusions must use the sidecar `cost` and `passable_mask`, not the
open-grid smoke fallback. For `stress` and `all`, the script also fails unless
at least one stress scenario produces a path-planning failure or replan
diagnostic, so stress validation cannot silently degrade into another easy smoke
run. Mixed-stress scenarios additionally require at least one reachable
candidate and at least one failure or replan diagnostic.

The summary metric surface is part of the next-stage contract. At minimum it
must retain `selection_changed_rate`, `path_planning_failure_count`,
`replan_count`, `tracking_safety_violation_count`,
`trajectory_optimization_fallback_count`, `region_graph_disconnected_count`,
and `coverage_per_path_cost`, plus IRIS and region-graph status/source/fallback
aggregates for scenario-group comparison.

## Semi-Real Batch Closed-Loop Evaluation v1

The batch entrypoint keeps the single-run validation script as the execution
unit and only adds orchestration, traceability, and aggregation:

```bash
bash scripts/run_batch_path_feedback_validation.sh --matrix configs/path_feedback_batch_dataset_v1.json --validate-only
bash scripts/run_batch_path_feedback_validation.sh --matrix configs/path_feedback_batch_dataset_v1.json --dry-run
bash scripts/run_batch_path_feedback_validation.sh --matrix configs/path_feedback_batch_dataset_v1.json --output-root outputs/path_feedback_batch_smoke
```

The matrix supports `run_id`, `scenario_set`, `diagnostic_profile`, `top_k`,
batch or per-run `output_root`, and optional `sample_quality_profile` metadata.
`sample_quality_profile` is recorded for downstream audit/stability work only;
Batch v1 does not run training, change PPO behavior, or alter any stable JSON
contract.

Each run writes the original single-run artifacts under its isolated output
directory:

- `path-feedback-manifest.json`
- `path-feedback-summary.json`
- `path-feedback-summary.md`
- `maps/`
- `path_planner_sidecars/`

The batch root writes:

- `batch-run-index.json`: run id, command arguments, source paths, acceptance
  metadata, parent/submodule git SHAs, pass/fail status, and machine-readable
  failure reason codes.
- `batch-evaluation-summary.json`: pass/fail counts, open-grid fallback gate
  results, scenario-group aggregates, path failure/replan totals, IRIS fallback
  totals, region-graph fallback/disconnect totals, and source summary paths for
  later sample-quality or stability consumers.

The default batch continues after an individual run fails so the index and
summary remain auditable. Its final exit code is nonzero when any run fails.
This remains a semi-real/quasi-real closed-loop evaluation pipeline only: it
does not implement GCS graph search, a GCS trajectory backend, Ackermann,
skid-steer, or differential-drive feasibility solving, and it does not claim
quasi-real or mask-stress results are real-world generalization performance.

## Dataset / Decision Stability v1

Batch outputs can now be consumed by a stability analysis entrypoint:

```bash
bash scripts/run_path_feedback_stability_analysis.sh --batch-root outputs/path_feedback_batch_current_analysis
```

The analyzer reads `batch-run-index.json`, `batch-evaluation-summary.json`,
and each run's `path-feedback-summary/v1`. It validates source paths, schema
versions, acceptance metadata, open-grid fallback gates, failed batch runs, and
parent/submodule git provenance before writing:

- `batch-stability-summary.json`: `batch-stability-summary/v1`, with run,
  scenario set, diagnostic profile, `top_k`, and scenario-group aggregates for
  pass/fail, open-grid fallback, path failure, replan, IRIS fallback,
  region-graph fallback, and region-graph disconnect.
- `dataset-quality-stability-summary.json`:
  `dataset-quality-stability-summary/v1`, with scenario, group, reason-code,
  and action aggregates for downstream sample-quality/stability consumers.
- `decision-stability-summary.json`: `decision-stability-summary/v1`, with
  target replacement, selection-changed, and failure-source stability across
  runs, profiles, and `top_k`.

Validation failures are machine-readable. Missing source summaries, schema
mismatches, acceptance metadata mismatches, open-grid fallback, and failed batch
runs produce reason codes in the output JSON; the command returns nonzero when
the input batch is not stability-clean. This stage does not run training, change
PPO, alter the network or action space, modify stable path-feedback JSON
fields, implement GCS or vehicle feasibility solving, or claim
Ackermann-feasible or real-world-generalized trajectories.

`model-explorer` can consume the resulting `path-feedback-summary/v1` JSON via
optional `train.system_calibration`. That system summary joins v5
`calibration_recommendation` with teacher gates and path-feedback gates for
source, teacher weight, curriculum profile, and seed. Path failures, replans,
IRIS fallback, region-graph disconnect/fallback, and `open_grid_fallback_used`
are calibration/exclusion/downweighting signals only; they are not evidence of
real-world performance improvement. When explicitly enabled,
`system_calibration.sample_quality` writes `sample_quality_summary` records
with machine-readable filtering/downweighting `reason_codes`; without
`system_calibration.sample_quality.enabled = true`, training data selection
keeps the old behavior. Semi-Real Calibration Dataset Application v2 also writes
`sample_quality_audit_summary`, aggregating records by scenario, group/ROI,
reason code, action, source summary path, acceptance metadata, scenario set,
diagnostic profile, and `top_k`. `open_grid_fallback` is a hard exclusion; path
failure, replan, IRIS fallback, and region-graph disconnect/fallback remain
calibration/downweighting signals only. Quasi-real and mask-stress rows keep
`data_class = quasi_real`, `mask_stress_augmented`, and
`not real-world generalization benchmark` labels.

## Sample-Quality-Aware Training Application v1

Batch and stability outputs can now be consumed by a training application audit
entrypoint:

```bash
bash scripts/run_sample_quality_training_application.sh --batch-root outputs/path_feedback_batch_training_input --config configs/sample_quality_training_application_v1.json
```

The command reads `batch-run-index.json`, `batch-evaluation-summary.json`,
`batch-stability-summary.json`, `dataset-quality-stability-summary.json`,
`decision-stability-summary.json`, and each run's `path-feedback-summary/v1`.
It writes:

- `sample-quality-training-application-summary.json`:
  `sample-quality-training-application-summary/v1`, with legacy,
  `hard_exclude_open_grid`, and `soft_downweight_diagnostics` profile results,
  source summary provenance, acceptance metadata, parent/submodule git SHAs,
  sample keep/downweight/exclude counts, reason-code distributions, and
  scenario/run/group aggregates.
- `training-selection-stability-summary.json`:
  `training-selection-stability-summary/v1`, comparing legacy and
  sample-quality-aware best-run/profile audit selections without treating the
  comparison as a training metric improvement claim.

Validation failures remain machine-readable. Missing source JSON, schema
mismatches, open-grid fallback, failed batch or stability inputs, acceptance
metadata mismatches, and git provenance mismatches are recorded as reason codes;
the command exits nonzero when any such failure is present.

This stage applies quality signals to sample-selection and best-run/profile
audit JSON only. It does not run large-scale training, change PPO behavior,
alter network architecture, expand the action space, implement GCS graph search
or vehicle motion feasibility, claim Ackermann-feasible trajectories, or treat
IRIS/region-graph diagnostics as GCS or vehicle-feasibility proof.

## Policy Decision Robustness v1

Current-HEAD batch, stability, and sample-quality evidence can now be consumed
by a candidate-ranking robustness audit entrypoint:

```bash
bash scripts/run_policy_decision_robustness_analysis.sh --batch-root outputs/path_feedback_batch_policy_input --config configs/policy_decision_robustness_v1.json
```

The command reads `batch-run-index.json`, `batch-evaluation-summary.json`,
`batch-stability-summary.json`, `dataset-quality-stability-summary.json`,
`decision-stability-summary.json`,
`sample-quality-training-application-summary.json`,
`training-selection-stability-summary.json`, and each run's
`path-feedback-summary/v1`. It writes:

- `policy-decision-robustness-summary.json`:
  `policy-decision-robustness-summary/v1`, comparing `legacy`,
  `feedback_aware`, and `sample_quality_aware` audit profiles. The summary
  records source summaries, acceptance metadata, parent/submodule git SHAs,
  candidate ordering before/after profile scoring, selected action/cell,
  rank deltas, score and penalty components, reason-code distributions, and
  scenario/run/group aggregates.
- `policy-decision-selection-comparison-summary.json`:
  `policy-decision-selection-comparison-summary/v1`, comparing profile
  selection stability by scenario without treating the comparison as a training
  metric improvement claim. It always records `no_training_metric_evaluated`.

Validation failures remain machine-readable. Missing source JSON, schema
mismatches, open-grid fallback, failed batch/stability/application inputs,
acceptance metadata mismatches, and git provenance mismatches are recorded as
reason codes; the command exits nonzero when any such failure is present.

This stage only audits candidate sorting robustness. It does not run PPO,
change PPO behavior, alter network architecture, expand the candidate-list
action space, implement GCS graph search or vehicle motion feasibility, claim
Ackermann-feasible trajectories, or treat IRIS/region-graph diagnostics as GCS
or vehicle-feasibility proof.

## Policy-Robustness Application Smoke v1

Policy Decision Robustness v1 outputs can now be applied to lightweight smoke
decision logs without running training:

```bash
bash scripts/run_policy_robustness_application_smoke.sh --batch-root outputs/path_feedback_batch_policy_input --robustness-summary outputs/path_feedback_batch_policy_input/policy-decision-robustness-summary.json --config configs/policy_robustness_application_smoke_v1.json
```

The command reads `policy-decision-robustness-summary/v1`,
`policy-decision-selection-comparison-summary/v1`,
`sample-quality-training-application-summary/v1`,
`training-selection-stability-summary/v1`, and the referenced
`path-feedback-summary/v1` files. It writes:

- `policy-robustness-application-summary.json`:
  `policy-robustness-application-summary/v1`, with the applied robustness
  profile, baseline profile, source summaries, acceptance metadata,
  parent/submodule git SHAs, lightweight episode/decision deltas, selected
  action/cell before and after application, failure/replan exposure,
  sample-quality reason codes, and scenario/group aggregates.
- `policy-robustness-application-comparison-summary.json`:
  `policy-robustness-application-comparison-summary/v1`, comparing legacy and
  robustness-aware smoke decisions with `no_large_scale_training`,
  `no_real_world_performance_claim`, and
  `no_single_metric_improvement_claim` recorded.

Validation failures remain machine-readable. Missing robustness inputs, schema
mismatches, open-grid fallback, failed upstream summaries, acceptance metadata
mismatches, and git provenance mismatches are recorded as reason codes; the
command exits nonzero when any such failure is present.

This stage only applies already-audited robustness decisions to lightweight
smoke decision records. It does not run large-scale training, claim performance
improvement, change PPO behavior, alter network architecture, expand the
candidate-list action space, implement GCS or vehicle feasibility, claim
Ackermann-feasible trajectories, or treat IRIS/region-graph diagnostics as GCS
or vehicle-feasibility proof.

## Core Algorithm Development Chain

The next implementation stages should follow:

```text
A* geometric path baseline
  -> region-graph-guided geometric search
  -> workspace IRIS region quality
  -> sampled region path backend
  -> Drake GCS geometric backend
  -> motion-feasibility layer
  -> execution-aware explorer loop
```

The project-level reference is
`docs/superpowers/specs/2026-06-02-core-algorithm-development-chain.md`.
Future `/goal` prompts should choose the first incomplete stage in that chain
unless current repo evidence proves a different algorithm stage is blocking.

The immediate algorithm target is **Region-Graph-Guided Geometric Search v1**,
after the current robustness/application smoke changes are integrated and the
evidence root is regenerated from the committed HEAD. This target moves
region-graph signals into `path-planner` path generation behavior while keeping
`path-planner-route/v1`, `trajectory_kind=geometric_path`, PPO, network
architecture, and candidate-list action space stable.

## Ubuntu One-Click Conda Setup

Target environment:

- Ubuntu 24.04
- Conda-compatible runtime, such as Miniconda, Mambaforge, or micromamba with a `conda` compatible command
- Python 3.12

Fresh clone:

```bash
git clone --recurse-submodules https://github.com/sjidnsk/lunar-path-planning.git
cd lunar-path-planning
bash scripts/bootstrap_ubuntu_conda.sh --run-validation
```

If the parent repository was cloned without submodules, the script runs:

```bash
git submodule update --init --recursive path-planner model-explorer dev-platform-constraints
```

After setup:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate lunar-explorer
```

The default setup creates or updates a named Conda environment `lunar-explorer`,
checks Python 3.12, and runs import smoke checks for all three modules.
`--run-validation` additionally runs the module test suites and
`model_explorer verify`.

Useful options:

```bash
bash scripts/bootstrap_ubuntu_conda.sh --dry-run
bash scripts/bootstrap_ubuntu_conda.sh --env-prefix "$HOME/conda_envs/lunar-explorer"
bash scripts/bootstrap_ubuntu_conda.sh --conda mamba --run-validation
bash scripts/bootstrap_ubuntu_conda.sh --install-editable
bash scripts/bootstrap_ubuntu_conda.sh --with-training
```

By default, local packages are exposed through per-module `PYTHONPATH` during
validation and are not installed editable. Use `--install-editable` only when
you want package entry points installed into the Conda environment. Use
`--with-training` only when `model-explorer` training paths are needed, because
it installs PyTorch.
