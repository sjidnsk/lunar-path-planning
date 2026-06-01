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
| `model-explorer` | Decision orchestration | Runnable synthetic decision/benchmark stack exists: contract loading, goal selection, loop/replan reasons, policy experiments, planning-result feedback, and CLI-backed `path-planner` route evaluation. The main gap is broadening the semi-real evaluation matrix and interpreting stress diagnostics. |
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
```

The next-stage acceptance gate is the final command above:
`--scenario-set all --diagnostic-profile all --top-k 3`. It exercises smoke and
stress scenarios, forwards execution and optional workspace IRIS diagnostics,
and still treats IRIS/region-graph output as diagnostic evidence only. It does
not introduce a GCS trajectory backend or a rover motion-feasibility solver.

The script initializes/checks the three submodules, generates the fixed `.npz`
validation maps, exports paired `model-explorer-contract/v1` and
`path-planner-sidecar/v1` JSON files, writes a `path-feedback-manifest/v1`, and
runs:

```bash
PYTHONPATH=src python3 -m model_explorer path-feedback validate <manifest>
PYTHONPATH=src python3 -m model_explorer path-feedback run <manifest>
```

The `path-feedback run` command prints a compact stdout summary and writes the
full experiment JSON plus Markdown report to the configured output files. The
root script can forward optional execution diagnostics with
`--simulate-tracking`, `--optimize-trajectory`, and `--drake-iris-regions`; the
default remains lightweight and Drake-free. `--diagnostic-profile execution`
forwards tracking simulation plus fixed-corridor optimization, `iris` forwards
optional workspace IRIS diagnostics, and `all` forwards both groups.

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
or
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
