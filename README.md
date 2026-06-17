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
bash scripts/run_path_feedback_validation.sh --scenario-set all --diagnostic-profile all --top-k 3 --gcs-control-point-candidate --output-root outputs/path_feedback_gcs_control_point_current
```

The standard acceptance gate is
`--scenario-set all --diagnostic-profile all --top-k 3`. It exercises smoke and
stress scenarios, forwards execution plus the existing workspace IRIS and
fixed-sequence GCS geometric/motion/curvature diagnostics, and still treats
IRIS/region-graph/GCS output as additive diagnostic evidence. It does not enable
the control-point terrain-cost GCS candidate unless
`--gcs-control-point-candidate` is supplied, and it does not claim a rover
motion-feasibility solver or Ackermann-feasible trajectory.

The script initializes/checks the three submodules, generates the fixed `.npz`
validation maps, exports paired `model-explorer-contract/v1` and
`path-planner-sidecar/v1` JSON files, writes a `path-feedback-manifest/v1`
with `scenario_set`, `diagnostic_profile`, `acceptance_gate`, `top_k`,
planner extra args, and open-grid fallback gate metadata, and runs:

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTHONPATH=src $PY -m model_explorer path-feedback validate <manifest>
PYTHONPATH=src $PY -m model_explorer path-feedback run <manifest>
```

The `path-feedback run` command prints a compact stdout summary and writes the
full experiment JSON plus Markdown report to the configured output files. The
JSON summary repeats the same `acceptance_metadata` and records the actual
`open_grid_fallback_used_gate` result. The root script defaults to the shared
Conda Python and can be overridden with `PYTHON=/path/to/python`; the chosen
interpreter is recorded in the manifest, summary metadata, and batch index.
The root script can forward optional execution diagnostics with
`--simulate-tracking`, `--optimize-trajectory`, and `--drake-iris-regions`; the
default remains lightweight and Drake-free. `--diagnostic-profile execution`
forwards tracking simulation plus fixed-corridor optimization, `iris` forwards
optional workspace IRIS plus the existing fixed-sequence GCS direction-cone,
geometric-candidate, motion-feasibility, and curvature-constrained diagnostics,
and `all` forwards both execution and `iris` groups. The control-point
terrain-cost path is intentionally outside these profiles: pass
`--gcs-control-point-candidate` explicitly to forward
`pydrake_control_point_direction_cone_program`.

When the control-point flag is present, `path-feedback-summary/v1` adds
`gcs_control_point_*` fields for report/attempt/success counts, backend counts,
candidate selection/fallback reasons, terrain objective sources, sampled
terrain-cost stats, high-cost-exposure delta stats, and per-candidate audit
rows. `pydrake` unavailable remains a `not_evaluated` diagnostic source rather
than a fake success. These fields are additive and do not replace the default
`path-planner-route/v1` reachable/path-cost semantics.
The summary also writes `gcs_control_point_candidate_triage` with schema
`gcs-control-point-candidate-triage-summary/v1` and
`gcs_control_point_candidate_artifacts` with schema
`gcs-control-point-candidate-artifact-index/v1`. The artifact index preserves
stable per-candidate route/request JSON paths so a failed or blocked
control-point route can be replayed by scenario/action. Current semi-real
evidence should be read as candidate-quality triage: smoke scenarios can solve
the control-point program but still be blocked by `cost_dominated` or
`direction_cone_constraint_violation`; stress scenarios with disconnected region
sequences are expected `not_evaluated` evidence, not hidden GCS success. These
triage fields include `calibration_sweep` with schema
`gcs-control-point-candidate-calibration-sweep/v1`; it is a recorded-candidate
diagnostic sweep over terrain objective weight, second-difference weight,
`direction_cone` rho/eta/tolerance, and quality gate thresholds. It requires a
solver rerun before any default update, does not relax the safety gate, and does
not make the control-point candidate the default route replacement.
Use `scripts/export_gcs_control_point_candidate_triage.py` to extract this
triage block from a `path-feedback-summary.json` into standalone JSON and
Markdown review artifacts.
Current triage evidence is rooted at
`outputs/path_feedback_gcs_control_point_triage_current/`: the standalone
triage artifacts are `gcs-control-point-candidate-triage-summary.json` and
`gcs-control-point-candidate-triage-summary.md`. The current run reports 23
candidates, 10 attempted, 10 successful solves, 0 selected, 23 route artifacts,
and fallback counts of `cost_dominated=5`,
`direction_cone_constraint_violation=5`, and `unsupported_route_replacement=13`;
its recorded-candidate calibration sweep keeps `default_change_recommended=false`.
`scripts/run_gcs_control_point_calibration_matrix.py` reruns those per-candidate
request artifacts through an explicit solver parameter matrix and writes both
`gcs-control-point-calibration-matrix-summary/v1` and
`gcs-control-point-targeted-sweep-summary/v1`. The default matrix now covers
single-factor terrain objective weight, second-difference weight, and
`direction_cone` tolerance/rho changes, small joint settings, and opt-in
high-cost exposure proxy entries. The pre-proxy targeted-sweep evidence is rooted at
`outputs/path_feedback_gcs_control_point_targeted_sweep_current/`: 11 matrix
settings, 23 candidates, 253 route artifacts, and
`safety_regression_count=29`. No tested setting increased selected count; wide
`direction_cone` settings convert 14 cases from
`direction_cone_constraint_violation` to `cost_dominated`, while terrain and
tight-cone settings add safety regressions. The result remains
`recommendation=no_default_change_recommended`, so direct parameter comparison
has localized the blocker migration but does not justify a default update.
`scripts/analyze_gcs_control_point_cost_gate.py` decomposes the cost-gate cases
from that targeted sweep into `gcs-control-point-cost-gate-decomposition-summary/v1`.
Current decomposition evidence is rooted at
`outputs/path_feedback_gcs_control_point_cost_gate_decomposition_current/`: 69
cost-gate cases, with `high_cost_exposure_blocked=38`,
`true_cost_dominated=20`, and `safety_regression_excluded=11`. No current case
is primarily classified as `terrain_proxy_mismatch`,
`baseline_overlap_or_duplicate`, or `insufficient_cost_diagnostics`; the
post-sweep blocker is therefore mostly real high-cost exposure or true cost
dominance, not missing diagnostics or a default-parameter opportunity.
The current high-cost exposure proxy generation run is rooted at
`outputs/path_feedback_gcs_control_point_high_cost_proxy_current/`: 13 matrix
settings, 23 candidates, 299 cases, `selected_count=0`,
`safety_regression_count=36`, `high_cost_exposure_proxy_case_count=46`, and
`high_cost_exposure_proxy_evaluated_count=20`. The opt-in
`--gcs-control-point-high-cost-exposure-weight` field is forwarded only with
`--gcs-control-point-candidate`; route JSON records the proxy as
`region_high_cost_exposure_proxy` with the explicit boundary
`proxy_not_continuous_field_integral`. The refreshed cost-gate decomposition
under `outputs/path_feedback_gcs_control_point_high_cost_proxy_current/cost_gate_decomposition/`
still reports `recommendation=no_default_change_recommended`:
`high_cost_exposure_blocked=46`, `true_cost_dominated=24`,
`safety_regression_excluded=14`, and no `terrain_proxy_mismatch` or
`insufficient_cost_diagnostics`. This confirms that the generation-side proxy is
diagnostic and opt-in, not evidence for changing default solver parameters.
The next algorithmic step is therefore not to bypass A* or generate corridors
without a seed path. It is an opt-in channel-aware A* backend: keep the default
A* baseline as the stable route reference, but add a candidate search mode whose
step cost estimates the local corridor/channel quality around each seed cell
using neighborhood mean/max cost, high-cost exposure proxy, clearance or
passable-margin penalty, blocked-nearby penalty, and smoothness/direction proxy.
That backend still emits a seed path; `postprocess` still builds the corridor
around that path; fallback box or workspace IRIS still builds the region
sequence; and control-point GCS remains an additive candidate, not the default
route replacement.
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
candidate and at least one failure, replan, or sampled-region decision
diagnostic.

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
batch or per-run `output_root`, optional `planner_extra_args`, and optional
`sample_quality_profile` metadata. The current Stage 1 matrix opts into
`--planning-backend region_graph_guided` through `planner_extra_args`; the
path-planner CLI default remains `astar`. `sample_quality_profile` is recorded
for downstream audit/stability work only; Batch v1 does not run training, change
PPO behavior, or alter any stable JSON contract.

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
  totals, region-graph fallback/disconnect totals, opt-in control-point GCS
  terrain-cost aggregate fields when present, and source summary paths for later
  sample-quality or stability consumers.

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

## Current Training-Readiness Gate

The active model-explorer gate is **Distance-Contract Relaxation Safety Audit
v1**. It consumes the anchor-projection candidate-generation, evidence-contract,
and policy-readiness summaries to decide whether source-selected projected
anchors beyond the current 2-cell / 1.0 m training distance contract are safe
enough for an explicit opt-in relaxation profile.

The audit keeps provenance, fallback/open-grid, safety, source-selection quality
regression, audit-proxy-positive, and contract-alignment gates intact. A passed
audit summary is not PPO readiness. If any far-distance, not-source-selected, or
unsafe context remains, the expected recommendation is
`keep_current_training_distance_contract`, and PPO remains blocked by
`anchor_projection_nontrainable_contexts_remain`.

Clean validation for
`outputs/path_feedback_batch_anchor_projection_distance_contract_relaxation_safety_audit_v1/`
keeps that recommendation: batch is 8/8 passed, candidate-generation remains
`18 trainable + 60 nontrainable`, the distance audit reports 36 rejected contexts
with 12 source-selected and 24 not-source-selected, and readiness remains
`needs_training_contract_refinement`.

The next active gate is **Anchor Projection Nontrainable Context Reduction /
Source-Selection Candidate Quality v1**. It does not reinterpret the distance
audit as training readiness. Instead it turns the remaining 60 nontrainable
contexts into an explicit accounting report:

- `safe_default_training_conversion_count=0`; the default 2-cell / 1.0 m
  distance contract remains unchanged.
- 6 source-selected near-distance contexts may only be treated as follow-up
  candidates for an explicit opt-in relaxation review.
- The full 60-context blocker is retained until source-selection quality,
  distance rejection, and audit-proxy gates prove otherwise.

The target evidence root is
`outputs/path_feedback_batch_anchor_projection_nontrainable_context_reduction_v1/`.
The new summary is
`anchor-projection-nontrainable-context-reduction-summary/v1` and must keep
`anchor_projection_nontrainable_contexts_remain` when any nontrainable context
remains.

Clean validation for that root was generated from a clean worktree. Batch is
8/8 passed, provenance mismatch counts are 0, and the nontrainable-reduction
summary reports:

- `recommendation=keep_training_blocker_focus_source_selection_candidate_quality`
- `generated_not_source_selected_count=48`
- `distance_contract_rejected_count=36`
- `source_selected_distance_rejected_count=12`
- `not_source_selected_distance_rejected_count=24`
- `source_selection_quality_regression_count=0`
- `positive_training_evidence_contains_audit_proxy_anchor_count=0`
- `candidate_contract_alignment_gap_count=0`

The next implementation gate is **Contract-Aware Trainable Target Generation
v1**. It moves the training contract check upstream into opt-in anchor-projection
candidate generation and source selection. Instead of relaxing the default
2-cell / 1.0 m distance contract, it generates same-action execution substitutes:
the policy action remains the original `top_goals` index, while
`execution_goal_cell` points to the reachable anchor. A candidate is counted as
PPO-consumable only when it is source-selected, contract-safe, free of
source-selection quality regression, and marked `ppo_consumable_action=true`.

New artifacts:

- `configs/path_feedback_batch_anchor_projection_contract_aware_trainable_target_v1.json`
- `configs/anchor_projection_contract_aware_trainable_target_v1.json`
- `scripts/run_anchor_projection_contract_aware_trainable_target.py`
- `scripts/run_anchor_projection_contract_aware_trainable_target.sh`
- `tests/test_anchor_projection_contract_aware_trainable_target.py`
- `docs/superpowers/specs/2026-06-08-anchor-projection-contract-aware-trainable-target.md`

The target evidence root is
`outputs/path_feedback_batch_anchor_projection_contract_aware_trainable_target_v1/`.
The summary
`anchor-projection-contract-aware-trainable-target-summary/v1` reports
`contract_trainable_contrast_count`,
`ppo_consumable_trainable_target_count`, nontrainable/distance/source-selection
deltas against the current 60/36/48 baseline, and `next_required_change`. If no
PPO-consumable target is produced, the summary must set
`next_required_change=action_or_target_contract_change_required` and readiness
must keep `anchor_projection_nontrainable_contexts_remain`.

Current verification expectation for this root is conservative: same-action
substitutes produce `ppo_consumable_trainable_target_count=18` with
`candidate_contract_alignment_gap_count=0`, but
`nontrainable_blocked_target_count` remains 60, so
`nontrainable_blocked_target_count_delta=0`. That means the PPO-consumable
threshold is met but the main success gate is not. The contract-aware summary
therefore must set
`next_required_change=action_or_target_contract_change_required`, and the
readiness review must keep `anchor_projection_nontrainable_contexts_remain`.

The follow-on implementation gate is **Planner-Validated Trainable Target
Mining v1**. It keeps the contract-aware generation path, but moves final
training-positive accounting after planner feedback is available. Each blocked
context receives exactly one final decision:
`selected_default_contract_trainable`,
`selected_planner_validated_distance_exception`,
`rejected_distance_contract`, `rejected_not_source_selected`,
`rejected_quality_regression`, or `rejected_not_ppo_consumable`.

New artifacts:

- `configs/path_feedback_batch_planner_validated_trainable_target_mining_v1.json`
- `configs/planner_validated_trainable_target_mining_v1.json`
- `scripts/run_planner_validated_trainable_target_mining.py`
- `scripts/run_planner_validated_trainable_target_mining.sh`
- `tests/test_planner_validated_trainable_target_mining.py`
- `docs/superpowers/specs/2026-06-08-planner-validated-trainable-target-mining.md`

The target evidence root is
`outputs/path_feedback_batch_planner_validated_trainable_target_mining_v1/`.
The default distance contract remains 2 cells / 1.0 m. The 3 cells / 1.5 m
gate is only an opt-in planner-validated exception and can become positive only
when the same-action substitute is source-selected, reachable, non-replan,
PPO-consumable, and within source-selection path/risk regression gates. A
planner-repaired target that is not source-selected remains diagnostic. If
`planner_validated_trainable_target_count <= 18` or
`nontrainable_blocked_target_count >= 60`, the mining summary must set
`next_required_change=source_selection_or_target_contract_change_required` and
readiness must stay blocked.

Current validation for this root is 8/8 passed with fallback/open-grid,
safety-regression, and provenance mismatch counts all zero. Candidate generation
remains `18 trainable + 60 nontrainable` under the default contract, while the
planner-validated mining summary reports
`planner_validated_trainable_target_count=24`,
`default_contract_trainable_target_count=18`,
`planner_validated_distance_exception_count=6`,
`nontrainable_blocked_target_count=54`,
`nontrainable_blocked_target_count_delta=-6`,
`candidate_contract_alignment_gap_count=0`, and
`next_required_change=null`. The readiness review for this evidence root reports
`training_readiness_status=ready_for_limited_policy_training_dry_run` with
`training_blockers=[]`. This is still a limited dry-run review gate, not PPO
training execution.

The next gate, **Limited Policy Training Dry-Run Input Materialization v1**,
converts only those 24 planner-validated positives into the existing
`RolloutEpisode` / `RolloutTransition` contract and then runs a one-epoch local
`train_policy_on_episodes` smoke pass. New artifacts are:

- `configs/planner_validated_training_input_materialization_v1.json`
- `configs/limited_policy_training_dry_run_v1.json`
- `scripts/run_planner_validated_training_input_materialization.py`
- `scripts/run_planner_validated_training_input_materialization.sh`
- `scripts/run_limited_policy_training_dry_run.py`
- `scripts/run_limited_policy_training_dry_run.sh`
- `tests/test_limited_policy_training_dry_run_input_materialization.py`
- `docs/superpowers/specs/2026-06-08-limited-policy-training-dry-run-input-materialization.md`

Current materialization output under the same evidence root reports
`input_positive_count=24`, `default_contract_positive_count=18`,
`planner_validated_exception_positive_count=6`,
`excluded_nontrainable_count=54`, `invalid_action_mask_count=0`, and
`empty_action_mask_count=0`. The dry-run summary reports
`dry_run_status=passed`, `train_policy_sample_count=24`,
`publishes_checkpoint=false`, and `performance_claimed=false`. This proves the
24 mined samples are consumable by the existing training path; it does not solve
the remaining 54 contract/source-selection blockers and does not publish or
evaluate a trained policy.

The follow-on **Counterfactual Preference Training Samples v1** gate handles the
36 `source_selection_not_selected` contexts as preference evidence rather than
hard positives. These contexts remain `synthetic_projection` and
`ppo_consumable_action=false`, so they are not inserted into the
`RolloutEpisode.action_index` label stream. Instead, the pipeline writes
`counterfactual-preference-training-samples.jsonl`,
`counterfactual-preference-training-summary.json`, and
`counterfactual-preference-exclusion-report.json`, then runs a local pairwise
preference dry-run.

Current results classify all 36 not-source-selected contexts:
`preference_pair_count=24`, split into
`selected_over_alternative_negative_count=12` and
`tradeoff_preference_pair_count=12`; the remaining
`rejected_binding_or_distance_required_count=12` stay excluded for later action
binding / distance-contract work. `hard_positive_added_count=0`. The preference
dry-run reports `preference_dry_run_status=passed`,
`preference_train_sample_count=24`, `publishes_checkpoint=false`, and
`performance_claimed=false`. This trains only the local ranking boundary in a
dry-run; it does not change default PPO training, network architecture, action
space, default A*, or distance contract behavior.

The follow-on **Unified Sample Taxonomy and Residual Boundary Preference v1**
gate completes the sample accounting for all 78 blocked contexts without
relabeling residual diagnostics as PPO action positives. New artifacts are:

- `configs/unified_policy_sample_registry_v1.json`
- `configs/residual_boundary_preference_training_dry_run_v1.json`
- `scripts/run_unified_policy_sample_registry.py`
- `scripts/run_unified_policy_sample_registry.sh`
- `scripts/run_residual_boundary_preference_training_dry_run.py`
- `scripts/run_residual_boundary_preference_training_dry_run.sh`
- `tests/test_unified_policy_sample_registry.py`
- `docs/superpowers/specs/2026-06-08-unified-sample-taxonomy-residual-boundary-preference.md`

Current registry output under the same evidence root reports
`action_label_positive_count=24`, `existing_preference_pair_count=24`,
`boundary_negative_preference_pair_count=12`,
`blocked_target_negative_pair_count=18`,
`residual_trainable_signal_count=30`,
`pairwise_preference_signal_count=54`,
`unified_context_coverage_count=78`, and `hard_positive_added_count=0`.
The 12 boundary-negative samples are the near-corridor synthetic projections
that still require action binding; the 18 blocked-target negative samples are
the dense-rock-choke projections beyond both the default and planner-validated
distance gates. The residual dry-run reports
`residual_preference_dry_run_status=passed`,
`residual_train_sample_count=30`, `publishes_checkpoint=false`, and
`performance_claimed=false`.

This means the 78 contexts now have unified training-signal coverage:
24 existing action-label positives plus 54 pairwise preference/negative signals.
It does **not** mean there are 78 PPO hard positives; the hard-positive stream
remains exactly 24.

The next **Hybrid Training Objective Integration v1** gate consumes these two
training-signal families in one opt-in local dry-run. New artifacts are:

- `configs/hybrid_policy_training_dry_run_v1.json`
- `scripts/run_hybrid_policy_training_dry_run.py`
- `scripts/run_hybrid_policy_training_dry_run.sh`
- `tests/test_hybrid_policy_training_dry_run.py`
- `docs/superpowers/specs/2026-06-08-hybrid-training-objective-integration.md`

The hybrid dry-run keeps hard positives on the existing
`RolloutEpisode.action_index` path and applies pairwise ranking loss only to the
54 preference/negative samples. Current output reports
`dry_run_status=passed`, `action_label_positive_count=24`,
`existing_preference_pair_count=24`, `residual_preference_pair_count=30`,
`pairwise_preference_signal_count=54`, `hybrid_train_signal_count=78`,
`hard_positive_added_count=0`, `invalid_action_mask_count=0`, and
`empty_action_mask_count=0`. It also reports `publishes_checkpoint=false` and
`performance_claimed=false`. Readiness review can now record
`hybrid_training_dry_run_completed`, but this remains a dry-run milestone rather
than formal PPO readiness or a policy performance claim.

The follow-on **Current-HEAD Hybrid Evidence Refresh and Readiness Closure v1**
does not add another sample type. It reruns the complete path-feedback,
sample-registry, limited training, preference, residual, hybrid, and readiness
pipeline under the current repository HEAD so the evidence no longer fails on
`current_git_provenance_mismatch`.

The refreshed evidence root is:

`outputs/path_feedback_batch_hybrid_current_head_readiness_closure_v1/`

The acceptance state for this root is: batch `failed_count=0`,
fallback/open-grid count `0`, safety regression `0`, summary
`reason_codes=[]`, and current git provenance mismatch count `0`. The hybrid
summary must preserve `action_label_positive_count=24`,
`pairwise_preference_signal_count=54`, `hybrid_train_signal_count=78`, and
`hard_positive_added_count=0`. The readiness review may then report
`training_readiness_status=hybrid_training_dry_run_completed`. This closes the
current evidence/readiness mismatch only; it still does not publish a model,
start formal PPO, change the network/action space/default A*, relax the default
distance contract, or claim Ackermann-feasible execution.

The next **Controlled Hybrid Policy Training Candidate v1** gate upgrades the
hybrid dry-run into a strictly local experimental checkpoint candidate. It adds:

- `configs/controlled_hybrid_policy_training_candidate_v1.json`
- `configs/controlled_hybrid_policy_holdout_evaluation_v1.json`
- `scripts/run_controlled_hybrid_policy_training_candidate.py`
- `scripts/run_controlled_hybrid_policy_training_candidate.sh`
- `scripts/run_controlled_hybrid_policy_holdout_evaluation.py`
- `scripts/run_controlled_hybrid_policy_holdout_evaluation.sh`
- `tests/test_controlled_hybrid_policy_training_candidate.py`
- `docs/superpowers/specs/2026-06-08-controlled-hybrid-policy-training-candidate.md`

The candidate training input remains the same 78-signal hybrid set:
`action_label_positive_count=24`, `pairwise_preference_signal_count=54`,
`hybrid_train_signal_count=78`, and `hard_positive_added_count=0`. The new output
root is
`outputs/path_feedback_batch_controlled_hybrid_policy_training_candidate_v1/`.
It may write a local `experimental-hybrid-policy-candidate.pt` plus metadata,
but both candidate and holdout summaries must keep `publishes_checkpoint=false`,
`replaces_default_policy=false`, and `performance_claimed=false`.

The holdout evaluation gate reports action-mask, fallback/open-grid, safety,
contract, path-cost, risk, source-selection, and preference-margin fields. The
hard safety gate requires `action_mask_invalid_count=0`,
`empty_action_mask_count=0`, `fallback_or_open_grid_count=0`,
`safety_regression_count=0`, and `contract_violation_count=0`. If path/risk or
source-selection regressions are present, readiness remains blocked with
`next_required_change=training_objective_or_sample_weight_refinement_required`.
When both candidate and holdout summaries pass without those regressions,
readiness may record `controlled_hybrid_training_candidate_evaluated`; this is
still not formal PPO readiness or a policy performance claim.

The next **Fresh Holdout Policy Candidate Evaluation v1** gate makes the
candidate evaluation stricter. The existing controlled holdout reuses the
current candidate evidence root and is therefore not a fresh/disjoint
generalization check. Fresh holdout adds:

- `configs/path_feedback_batch_fresh_holdout_policy_candidate_evaluation_v1.json`
- `configs/fresh_holdout_policy_candidate_evaluation_v1.json`
- `scripts/run_fresh_holdout_policy_candidate_evaluation.py`
- `scripts/run_fresh_holdout_policy_candidate_evaluation.sh`
- `tests/test_fresh_holdout_policy_candidate_evaluation.py`
- `docs/superpowers/specs/2026-06-08-fresh-holdout-policy-candidate-evaluation.md`

The fresh evidence root is
`outputs/path_feedback_batch_fresh_holdout_policy_candidate_evaluation_v1/`.
The evaluator reuses the controlled candidate checkpoint, but accepts only
candidate/sample context identity keys that are disjoint from both
`outputs/path_feedback_batch_hybrid_current_head_readiness_closure_v1/` and
`outputs/path_feedback_batch_controlled_hybrid_policy_training_candidate_v1/`.
It writes `fresh-holdout-policy-candidate-evaluation-summary.json`,
`fresh-holdout-overlap-report.json`, and
`fresh-holdout-candidate-score-report.json`. Scenario ids may overlap; those
overlaps are reported as `scenario_overlap_count` and must not be described as
scenario-level generalization.

Fresh holdout passes only when `fresh_disjoint_context_count > 0`, accepted
samples have `identity_overlap_count=0` and `identity_key_missing_count=0`, and
fallback/open-grid, safety, contract, path/risk, and source-selection regression
counts are all 0. Passing readiness may advance only to
`fresh_holdout_policy_candidate_evaluated`. If no disjoint samples exist, the
required next change is fresh holdout scenario or candidate generation; if
quality regressions exist, the required next change is training objective or
sample-weight refinement. This remains a candidate-context holdout gate, not
production readiness, not checkpoint publication, and not a policy performance
claim.

**Scenario-Disjoint Context-ID Generalization Closure v1** is the stricter
successor gate. It closes the gap where Fresh Holdout v1 proved only
candidate/sample identity disjointness while still reporting scenario overlap.
The new gate adds:

- `model-explorer/src/model_explorer/policy/context_id.py`
- `configs/path_feedback_batch_scenario_disjoint_policy_candidate_evaluation_v1.json`
- `configs/scenario_disjoint_policy_candidate_evaluation_v1.json`
- `tests/test_policy_context_id_contract.py`
- `tests/test_scenario_disjoint_policy_candidate_evaluation.py`
- `docs/superpowers/specs/2026-06-08-scenario-disjoint-context-id-generalization-closure.md`

New path-feedback candidates carry `policy-context-id/v1` metadata:
`context_id`, `context_id_schema_version`, `context_id_source`, and
`legacy_identity_fallback_used=false`. The id is a sha256 over stable semantic
fields: scenario id/group/seed/variant, diagnostic profile, planning backend,
top-k, sample/candidate role, source action, policy target, execution goal, and
target binding mode. New scenario-disjoint roots must have
`context_id_missing_count=0` and `legacy_identity_fallback_count=0`; old fresh
roots may still use legacy identity fallback only for compatibility.

The `holdout` scenario set is generated by
`dev-platform-constraints/scripts/generate_npz_validation_maps.py` with new
scenario ids and seeds. Strict fresh evaluation uses
`configs/scenario_disjoint_policy_candidate_evaluation_v1.json` and requires
`scenario_overlap_count=0`, `identity_overlap_count=0`,
`candidate_git_current_matches_sources=true`, and
`checkpoint_metadata_git_current_matches_sources=true`. Passing readiness may
advance to `scenario_disjoint_policy_candidate_evaluated`, still only for a
local experimental checkpoint. It does not publish a checkpoint, replace the
default policy, start formal PPO rollout, relax the distance contract, change
network/action space/default A*, or claim Ackermann-feasible trajectory or
policy performance.

**Controlled Scenario-Disjoint Policy Rollout Evaluation v1** is the next
shadow gate after scenario-disjoint candidate evaluation. It adds
`configs/scenario_disjoint_policy_rollout_evaluation_v1.json`,
`scripts/run_scenario_disjoint_policy_rollout_evaluation.py`, the matching
shell wrapper, `tests/test_scenario_disjoint_policy_rollout_evaluation.py`, and
`docs/superpowers/specs/2026-06-08-scenario-disjoint-policy-rollout-evaluation.md`.
The evaluator loads the local experimental checkpoint from the controlled
candidate root and scores HOLD `path-feedback-summary.json` candidates.

Default mode is `shadow_mode=true` and `controlled_selection_mode=false`: the
raw policy top choice is recorded, then action-mask, contract, fallback, safety,
path/risk, and source-selection gates produce the controlled shadow decision.
The HOLD root receives `scenario-disjoint-policy-rollout-decisions.jsonl`,
`scenario-disjoint-policy-rollout-regression-report.json`, and
`scenario-disjoint-policy-rollout-evaluation-summary.json`.

Passing rollout readiness requires `scenario_disjoint_context_count > 0`,
`invalid_action_mask_count=0`, `regression_count=0`, and fallback/open-grid,
safety, contract, path/risk, and source-selection regression counts all 0.
Readiness may then advance to `scenario_disjoint_policy_rollout_evaluated`.
This remains controlled shadow evaluation only: no formal PPO rollout,
checkpoint publication, default policy replacement, distance-contract
relaxation, network/action-space/default-A* change, Ackermann-feasible
trajectory claim, or policy performance claim.

**Raw Policy Decision Alignment and Objective Calibration v1** addresses the
remaining gap exposed by controlled rollout: the controlled gate can keep the
shadow decision safe, but the raw policy top choice may still prefer a
regressive alternative. The new opt-in loop adds
`configs/raw_policy_regression_mining_v1.json`,
`configs/raw_policy_decision_alignment_candidate_v1.json`,
`configs/raw_policy_strict_rollout_evaluation_v1.json`,
`scripts/run_raw_policy_regression_mining.py`,
`scripts/run_raw_policy_decision_alignment_candidate.py`,
`scripts/run_raw_policy_strict_rollout_evaluation.py`, and
`scripts/run_raw_policy_decision_alignment_closure.sh`.

Mining reads HOLD rollout decisions and path-feedback candidates, then converts
raw-regressive choices into `raw_policy_regression_preference_pair` samples:
the source-selected/controlled-safe candidate is preferred, and the raw
regressive candidate is the alternative. These samples are pairwise ranking
signals only; `hard_positive_added_count=0`, and they are never materialized as
`RolloutEpisode` action labels.

The alignment candidate reuses the existing experimental hybrid trainer with
an opt-in raw-regression pairwise input, writing
`raw-policy-decision-alignment-candidate-summary.json`. Strict rollout writes
`raw-policy-strict-rollout-decisions.jsonl`,
`raw-policy-strict-rollout-regression-report.json`, and
`raw-policy-strict-rollout-evaluation-summary.json`. Readiness may advance to
`raw_policy_decision_alignment_evaluated` only when strict rollout passes,
controlled regression remains 0, invalid mask/fallback/safety/contract/path/
risk/source-selection regression are all 0, and `raw_policy_regression_count`
is below the configured baseline threshold. If raw regression does not improve,
the next required change is `policy_objective_or_feature_refinement_required`.

This stage still does not start formal PPO rollout, publish a checkpoint,
replace the default policy, change network/action space/default A*, relax the
distance contract, claim Ackermann-feasible trajectory, or claim policy
performance.

**Raw Policy Generalization and Anti-Overfit Closure v1** adds the next
shadow-only gate after raw alignment. The previous scenario-disjoint HOLD root
is now treated as dev/calibration evidence because it already supplied
raw-regression preference samples. It must not be reused as the final
generalization exam.

The new opt-in closure introduces TRAIN/VAL/TEST scenario sets and matrices:
`raw_align_train`, `raw_align_val`, `raw_align_test`, plus
`configs/path_feedback_batch_raw_policy_generalization_train_v1.json`,
`configs/path_feedback_batch_raw_policy_generalization_val_v1.json`, and
`configs/path_feedback_batch_raw_policy_generalization_test_v1.json`.
TRAIN/dev roots may write `raw_policy_regression_preference_pair` samples.
VAL/TEST use `configs/raw_policy_regression_mining_diagnostic_v1.json` and
write diagnostics only.

The candidate stage is driven by
`configs/raw_policy_generalization_candidate_v1.json` and
`scripts/run_raw_policy_generalization_candidate.py`: it combines TRAIN/dev raw
preferences, rejects any VAL/TEST context leakage, trains experimental local
seed candidates, and selects the best seed by VAL raw-regression count. The
final evaluator is `scripts/run_raw_policy_generalization_evaluation.py` with
`configs/raw_policy_generalization_evaluation_v1.json`; TEST is the only final
acceptance split. Readiness may advance to
`raw_policy_generalization_evaluated` only when TEST controlled regression and
all safety/contract/path/risk/source-selection gates are 0, TEST raw regression
drops by at least 50% versus baseline, and `overfit_gap<=0.15`.

This remains experimental shadow evaluation. It still does not start PPO
rollout, publish or replace a policy, alter network/action space/default A*,
relax the distance contract, claim Ackermann-feasible trajectory, treat
IRIS/GCS diagnostics as training release evidence, or claim performance.

**Policy-Gated Canary Rollout v1** is the next shadow-only gate after raw
generalization. Raw generalization proves the candidate no longer repeats the
known raw-policy regressions on unseen TEST contexts; canary rollout asks a
different question: when a safe alternative exists, can the raw policy choose a
different candidate and still pass every existing gate?

The canary batch uses `configs/path_feedback_batch_policy_gated_canary_rollout_v1.json`
and writes `outputs/path_feedback_batch_policy_gated_canary_rollout_v1/`. The
evaluator is `scripts/run_policy_gated_canary_rollout.py` with
`configs/policy_gated_canary_rollout_v1.json`; it writes
`policy-gated-canary-rollout-summary.json`,
`policy-gated-canary-decisions.jsonl`,
`policy-gated-canary-rejection-report.json`, and
`policy-gated-canary-opportunity-summary.json`.

Canary success requires `policy_decision_count>0`,
`canary_opportunity_context_count>0`, `policy_changed_decision_count>0`,
`canary_accepted_policy_choice_count>0`, and controlled invalid-mask,
fallback/open-grid, safety, contract, path/risk, and source-selection
regression all equal to 0. A source-aligned-only run is not enough; it only
shows the policy copied the teacher. A changed-but-rejected run shows the policy
tried to take over but the gate correctly refused it.

Readiness may advance to `policy_gated_canary_rollout_evaluated` only when the
canary summary passes and candidate/checkpoint provenance matches the current
source state. This remains a gated test drive, not formal PPO rollout or policy
release.

**Canary Diversity and Safe-Takeover Robustness v1** extends that canary from a
single `npz_mixed_stress_detour` exercise into a multi-family closed-course
test. It adds `policy_canary_diversity`, the matrix
`configs/path_feedback_batch_policy_gated_canary_diversity_v1.json`, the
evaluator config `configs/policy_gated_canary_diversity_v1.json`, and the
closure script `scripts/run_canary_diversity_safe_takeover_closure.sh`.

The diversity gate refreshes clean-HEAD SRC and candidate evidence, then writes
`outputs/path_feedback_batch_policy_gated_canary_diversity_v1/`. The summary now
reports per-family opportunity/changed/aligned/accepted/rejected counts,
`accepted_scenario_family_count`,
`accepted_decision_family_distribution`, and `canary_diversity_passed`.
Acceptance requires at least 12 opportunities, at least 5 scenario families,
accepted choices in at least 3 families, at least 4 accepted policy choices,
and all controlled safety/contract/path/risk/source-selection regression gates
at 0. Passing readiness advances only to
`policy_gated_canary_diversity_evaluated`; it is still shadow/canary evidence,
not PPO rollout, policy release, or a performance claim.

Current evidence passes those gates: 12 canary opportunities, 6 changed policy
decisions, 6 accepted policy choices, accepted choices across 3 scenario
families, no rejected choices, no invalid action masks, no fallback/open-grid,
no safety/contract/path/risk/source-selection regression, provenance current,
and readiness `training_readiness_status=policy_gated_canary_diversity_evaluated`.

**Canary Opportunity Quality and Multi-Family Safe Choice Expansion v1** asks a
more precise question: for each scenario family, does the candidate set actually
contain a safe acceptable alternative, and if it does, does the raw policy take
that opportunity or simply stay source-aligned? This separates a scenario/candidate
generation gap from a policy-calibration gap.

This stage adds `policy_canary_opportunity_quality`, the matrix
`configs/path_feedback_batch_policy_gated_canary_opportunity_quality_v1.json`,
the evaluator config `configs/policy_gated_canary_opportunity_quality_v1.json`,
the missed-opportunity preference config
`configs/canary_missed_opportunity_preference_v1.json`, and
`scripts/run_canary_opportunity_quality_closure.sh`. Its canary summary reports
`family_with_acceptable_alternative_count`,
`missing_acceptable_alternative_families`,
`source_aligned_with_acceptable_alternative_count`,
`canary_missed_opportunity_preference_pair_count`,
`missed_safe_choice_family_count`, and `hard_positive_added_count=0`.

Acceptance requires at least 24 canary opportunity contexts, at least 6 scenario
families, at least 5 families with acceptable alternatives, accepted policy
choices in at least 5 families, at least 8 accepted policy choices, no rejected
choice without reason codes, and all controlled/raw invalid-mask, fallback,
safety, contract, path/risk, and source-selection regression gates at 0. Passing
readiness advances only to `policy_gated_canary_opportunity_quality_evaluated`.
It remains an experimental gated canary; it does not start PPO rollout, publish
or replace a policy, alter network/action space/default A*, relax the distance
contract, claim Ackermann-feasible trajectory, or claim performance.

Current evidence passes: 24 canary opportunities across 6 families, 10 changed
policy decisions, 10 accepted choices, 0 rejected choices, 5 accepted families,
5 families with acceptable alternatives, no missed safe-choice preference pairs,
no hard positives added, no controlled/raw regressions, current candidate
provenance match, and readiness
`training_readiness_status=policy_gated_canary_opportunity_quality_evaluated`.
`dense_choke_safe_bypass` remains the only family without an acceptable
alternative and is a scenario/candidate opportunity-generation issue, not a
policy gate regression.

**Dense-Choke Safe Alternative Opportunity Closure v1** closes that last
opportunity gap without relaxing the canary gate. It adds the dedicated
`policy_canary_dense_choke_opportunity` scenario set, the full-family matrix
`configs/path_feedback_batch_policy_gated_canary_full_family_opportunity_v1.json`,
the evaluator config `configs/policy_gated_canary_full_family_opportunity_v1.json`,
`scripts/run_dense_choke_safe_alternative_diagnosis.py/.sh`, and
`scripts/run_dense_choke_safe_alternative_opportunity_closure.sh`.

The dense-choke diagnosis writes
`dense-choke-safe-alternative-diagnosis-summary.json`,
`dense-choke-safe-alternative-diagnosis.md`, and
`outputs/dense_choke_safe_alternative_visual_diagnostics_v1/index.html`. It
reports candidate cell/action/source-action, path/risk delta, action-mask
validity, source binding, and rejection reason counts. If dense choke still has
no acceptable alternative, canary summary now returns
`next_required_change=dense_choke_opportunity_generation_gap`; if it has a safe
alternative but the policy remains source-aligned, the next change is policy
alignment/calibration, not hard-positive expansion.

Full-family acceptance requires 6 scenario families, 6 families with acceptable
alternatives, 6 accepted families, `dense_choke_safe_bypass` acceptable and
accepted counts above 0, at least 12 accepted policy choices, 0 rejected policy
choices, and all controlled/raw invalid-mask, fallback, safety, contract,
path/risk, and source-selection regression gates at 0. Passing readiness can
advance only to `policy_gated_canary_full_family_opportunity_evaluated`. This
is still controlled canary evidence: no formal PPO rollout, no checkpoint
publication or default replacement, no network/action-space/default-A* change,
no distance-contract relaxation, no Ackermann-feasible trajectory claim, and no
performance claim.

Current clean-HEAD closure passes those gates. The full-family root
`outputs/path_feedback_batch_policy_gated_canary_full_family_opportunity_v1/`
reports `policy_decision_count=32`, `canary_opportunity_context_count=32`,
`policy_changed_decision_count=18`, `canary_accepted_policy_choice_count=18`,
`canary_rejected_policy_choice_count=0`, `scenario_family_count=6`,
`family_with_acceptable_alternative_count=6`, `accepted_scenario_family_count=6`,
`dense_choke_acceptable_alternative_count=8`, and
`dense_choke_accepted_policy_choice_count=8`. Candidate/checkpoint provenance
matches the current source, all regression gates remain 0, and readiness under
`outputs/path_feedback_batch_dense_choke_opportunity_clean_src_v1/` is
`training_readiness_status=policy_gated_canary_full_family_opportunity_evaluated`
with `training_blockers=[]`.

**Current-HEAD Evidence Refresh + Canary Value/Stability Evaluation v1** moves
the canary question from “can the policy safely choose a different candidate?”
to “does that safe change appear often enough, across families, and does it
carry measurable path/risk/utility value?” It first requires the current HEAD
full-family canary evidence to be refreshed so provenance no longer reports
`current_git_provenance_mismatch`; only then does it run the new value/stability
closure.

This stage adds the `policy_canary_value_stability` scenario set: 6 scenario
families, 6 geometry variants per family, and two planning backends in the
matrix for 72 canary opportunity contexts. The new artifacts are
`configs/path_feedback_batch_policy_gated_canary_value_stability_v1.json`,
`configs/policy_gated_canary_value_stability_v1.json`, and
`scripts/run_canary_value_stability_closure.sh`. The closure writes SRC,
candidate, and canary roots under
`outputs/path_feedback_batch_value_stability_clean_src_v1/`,
`outputs/path_feedback_batch_value_stability_candidate_v1/`, and
`outputs/path_feedback_batch_policy_gated_canary_value_stability_v1/`.

The canary summary now reports `accepted_equal_choice_count`,
`accepted_better_choice_count`, `accepted_better_family_count`,
`policy_change_rate`, `accepted_choice_rate`, `accepted_value_delta_summary`,
`family_value_stability_summary`, and `canary_value_stability_passed`.
`accepted_better` is stricter than “accepted”: it must first pass all canary
gates and then improve path cost by at least 0.25, risk by at least 0.01, or
utility by at least 0.005. If safe alternatives are missing, the next change is
`canary_value_opportunity_generation_gap`; if safe accepted choices exist but
better choices are insufficient, the next change is
`policy_value_alignment_or_objective_refinement_required`.

Acceptance requires 6 families, at least 72 opportunity contexts, 6 families
with acceptable alternatives, accepted choices in all 6 families, at least 24
accepted choices, at least 8 better choices across at least 3 families, dense
choke accepted count above 0, 0 rejected choices, and controlled/raw
regression, invalid action mask, fallback/open-grid, safety, contract,
path/risk, and source-selection regression all at 0. Passing readiness can
advance only to `policy_gated_canary_value_stability_evaluated`. This remains
experimental canary evidence only: no formal PPO rollout, no checkpoint
publication or default replacement, no network/action-space/default-A* change,
no distance-contract relaxation, no Ackermann-feasible trajectory claim, no
IRIS/GCS/path-planner diagnostic-as-training release, and no performance claim.

**Policy-Gated Sequential Canary Rollout v1** is the next gate toward formal
PPO rollout. Value/stability canary is still a set of independent one-step
questions; sequential canary turns it into short cell-level episodes. Each
episode step must start from the previous step's controlled execution goal. If
the policy choice passes all gates, the next step starts from the policy
execution goal; if it fails, the runner falls back to the source-selected goal
and records the rejection.

This stage adds explicit scenario-spec input for generated NPZ validation maps,
`configs/policy_gated_sequential_canary_rollout_v1.json`,
`scripts/run_policy_gated_sequential_canary_rollout.py/.sh`, and
`scripts/run_policy_gated_sequential_canary_closure.sh`. It writes
`outputs/path_feedback_batch_policy_gated_sequential_canary_rollout_v1/` with
`policy-gated-sequential-canary-episodes.jsonl`,
`policy-gated-sequential-canary-steps.jsonl`,
`policy-gated-sequential-canary-rejection-report.json`, and
`policy-gated-sequential-canary-rollout-summary.json`.

Acceptance requires 36 episodes, 108 steps, at least 30 completed episodes, at
least 24 accepted takeover steps, accepted takeover in all 6 families, at least
12 multi-step accepted episodes, 0 state-continuity violations, 0 episode
fallbacks, 0 rejected policy choices, and cumulative invalid-mask, fallback,
safety, contract, path/risk, and source-selection regression all at 0. Passing
readiness can advance only to
`policy_gated_sequential_canary_rollout_evaluated`. It remains canary/shadow
evidence: no formal PPO rollout, no PPO parameter update, no checkpoint
publication or default policy replacement, no network/action-space/default-A*
change, no distance-contract relaxation, no Ackermann-feasible trajectory claim,
and no performance claim.

**Sequential Safe-Choice Calibration and Hard-Negative Refinement v1** uses the
failed sequential canary root as training evidence instead of weakening the
gate. The sequential runner proved state continuity, but exposed 2 rejected
policy choices and cumulative path/risk regressions. The implemented mining now
uses two non-action-label signals: hard-negative preferences for rejected
path/risk-regressive choices, and missed-safe-choice preferences when a
source-aligned step still has a gate-safe, better alternative. Source or
controlled-safe choices remain preferred over unsafe alternatives; safe-better
alternatives are preferred over overly conservative source-aligned choices.

New artifacts are
`configs/sequential_canary_failure_mining_v1.json`,
`configs/sequential_safe_choice_calibration_candidate_v1.json`,
`scripts/run_sequential_canary_failure_mining.py/.sh`,
`scripts/run_sequential_safe_choice_calibration_candidate.py/.sh`, and
`scripts/run_sequential_safe_choice_calibration_closure.sh`. The failed
baseline remains
`outputs/path_feedback_batch_policy_gated_sequential_canary_rollout_v1/`.
The calibrated closure writes
`outputs/path_feedback_batch_sequential_safe_choice_clean_src_v1/`,
`outputs/path_feedback_batch_sequential_safe_choice_candidate_v1/`, and
`outputs/path_feedback_batch_policy_gated_sequential_safe_choice_rollout_v1/`.

Current evidence is useful but not a readiness pass. Mining from
`outputs/path_feedback_batch_policy_gated_sequential_canary_rollout_v1/`
produces 2 hard-negative pairs and 6 missed-safe-choice pairs with
`hard_positive_added_count=0`; the balanced candidate remains experimental and
removes the sequential safety failure. The rerun sequential canary reports
36 episode / 108 step, 28 accepted-better takeover steps, all 6 families
accepted, 0 rejected choices, 0 state-continuity violations, 0 episode
fallbacks, and invalid-mask/fallback/safety/contract/path/risk/source-selection
regression all at 0. It still fails the original multi-step coverage gate:
`multi_step_accepted_episode_count=6` and
`family_with_multi_step_accepted_episode_count=2`, with
`next_required_change=sequential_opportunity_distribution_gap_requires_more_episodes`.
Readiness must therefore remain blocked; the next stage should generate
stronger sequential multi-step opportunity coverage rather than enter PPO. This
is still calibration and canary evidence only: no formal PPO rollout, no PPO
parameter update, no checkpoint publication or default policy replacement, no
network/action-space or default-A* change, no distance-contract relaxation, no
Ackermann-feasible trajectory claim, and no policy performance claim.

**Sequential Multi-Step Opportunity Generation v1** addresses that remaining
coverage gap. The point is not to make the policy more aggressive or weaken the
gate. It first asks whether each family actually offers enough consecutive
gate-safe and better alternatives, then reuses the same strict sequential gate
to see whether the policy takes those opportunities.

New artifacts are
`configs/sequential_multi_step_opportunity_diagnosis_v1.json`,
`configs/policy_gated_sequential_multi_step_opportunity_rollout_v1.json`,
`configs/path_feedback_batch_sequential_multi_step_opportunity_v1.json`,
`scripts/run_sequential_multi_step_opportunity_diagnosis.py/.sh`, and
`scripts/run_sequential_multi_step_opportunity_closure.sh`. The new scenario
set is `policy_canary_sequential_multi_step_opportunity`: 6 families x 6
variants, with `npz_canary_sequential_multi_step_opportunity_*` template IDs.
The sequential runner now reads `template_scenario_id_prefix` and
`scenario_set` from config, and supports per-episode initial start cells so
family-specific sequential opportunities can be tested without hard-coded
value/stability templates.

The diagnosis writes
`sequential-multi-step-opportunity-diagnosis-summary.json`,
`sequential-multi-step-opportunity-diagnostics.jsonl`, and
`sequential-multi-step-opportunity-exclusion-report.json`. It reports the
opportunity funnel per episode/step: alternative count, action-mask-valid,
reachable, no-replan, no-fallback/open-grid, contract-safe, path/risk
non-regressive, source-selection non-regressive, and safe-better alternative
counts. It separates `opportunity_missing` from
`policy_missed_existing_opportunity`; the former means scenario generation must
be fixed, while the latter means objective/sample weighting should be refined.

Acceptance requires 36 episodes / 108 steps, at least 12 episodes with
step0+step1 multi-step opportunities, all 6 families with multi-step
opportunities, at least 2 such episodes per family, at least 24 safe-better
opportunity steps, 0 opportunity exclusions, and final sequential rollout with
at least 24 accepted takeover steps, at least 12 multi-step accepted episodes,
all 6 families covered, 0 rejected choices, 0 state-continuity violations, 0
episode fallbacks, and all cumulative regression gates at 0. Passing readiness
can advance only to
`policy_gated_sequential_multi_step_opportunity_evaluated`. It remains
canary/shadow evidence: no formal PPO rollout, no PPO parameter update, no
checkpoint publication or default replacement, no network/action-space/default
A* change, no distance-contract relaxation, no Ackermann-feasible trajectory
claim, and no performance claim.

Current closure passes after scenario repair and sequence-aware calibration.
Preflight opportunity evidence is rooted at
`outputs/path_feedback_batch_policy_gated_sequential_multi_step_opportunity_preflight_v1/`
and reports 36 episodes / 108 steps, 57 safe-better opportunity steps, 16
step0+step1 multi-step opportunity episodes, all 6 families covered, and at
least 2 multi-step opportunity episodes per family. The final calibrated
rollout root
`outputs/path_feedback_batch_policy_gated_sequential_multi_step_opportunity_rollout_v1/`
reports 36 policy takeover steps, 36 accepted better steps, 12 multi-step
accepted episodes, 6 accepted families, 0 rejected choices, 0 invalid action
mask, and 0 cumulative path/risk regression. Readiness currently reaches
`policy_gated_sequential_multi_step_opportunity_evaluated` with
`training_blockers=[]` when the closure is run from the current committed HEAD.

## PPO Rollout Collector Dry-Run

The next policy-training boundary is now **PPO rollout collection**, not PPO
parameter update. The repository adds an opt-in dry-run collector that turns
policy-controlled sequential canary takeover steps into existing
`RolloutEpisode/RolloutTransition` records while keeping source-fallback steps
diagnostic-only.

New artifacts:

- `configs/sequential_evidence_consistency_v1.json`
- `configs/ppo_rollout_collector_dry_run_v1.json`
- `scripts/run_sequential_evidence_consistency_check.py/.sh`
- `scripts/run_ppo_rollout_collector_dry_run.py/.sh`
- `scripts/run_ppo_rollout_collector_closure.sh`
- `outputs/path_feedback_batch_ppo_rollout_collector_dry_run_v1/`

Current dry-run evidence materializes 36 PPO-trainable transitions from the
sequential multi-step canary root, with invalid/empty action mask counts,
missing log-prob/value counts, non-finite reward count, and source-fallback
trainable count all at 0. `ppo-rollout-episodes.jsonl` is readable through the
existing rollout IO path and is validated with `validate_rollout_dataset`.
The clean-HEAD closure also refreshes the upstream sequential SRC/CAND/SEQ roots
before collector materialization, so readiness validate-only can consume
matching provenance and reach
`training_readiness_status=ppo_rollout_collector_dry_run_evaluated` with
`training_blockers=[]`.

This stage still does not execute PPO optimizer updates, publish checkpoints,
replace the default policy, change network/action space/default A*, relax the
distance contract, claim Ackermann-feasible trajectories, or treat IRIS/GCS
diagnostics as training release evidence.

## Limited PPO Update Smoke

After `ppo_rollout_collector_dry_run_evaluated`, the next boundary is a tiny
optimizer smoke, not a formal PPO rollout. The new smoke runner loads the same
experimental checkpoint that produced the collector transitions, verifies the
stored `log_prob` and `value` against that checkpoint, filters to
`ppo_trainable=true` policy-controlled transitions, and performs a single
full-batch PPO update with a small learning rate.

New artifacts:

- `configs/limited_ppo_update_smoke_v1.json`
- `scripts/run_limited_ppo_update_smoke.py/.sh`
- `scripts/run_limited_ppo_update_smoke_closure.sh`
- `outputs/path_feedback_batch_limited_ppo_update_smoke_v1/`

The runner writes `limited-ppo-update-smoke-summary.json`,
`limited-ppo-update-training-curves.json`,
`limited-ppo-update-diagnostics.json`,
`experimental-hybrid-policy-candidate.pt`,
`experimental-hybrid-policy-candidate-metadata.json`, and
`raw-policy-generalization-candidate-summary.json`. The updated checkpoint
remains experimental: it is not published, does not replace the default policy,
and does not claim performance.

Acceptance requires the smoke to train only the collector policy's on-policy
transitions, keep source-fallback transitions out of the optimizer, keep
old-log-prob/value reconstruction error below `1e-4`, produce a non-zero but
small parameter delta, keep loss/grad/reward/return/advantage finite, and keep
`approx_kl<=0.25` with clipped gradient norm at or below 1.0. Post-update
raw-policy generalization, sequential canary, collector, and readiness gates
must still pass before readiness may advance to
`limited_ppo_update_smoke_evaluated`.

This remains a local smoke test only: no formal PPO rollout, no released
checkpoint, no default-policy replacement, no network/action-space/default-A*
change, no distance-contract relaxation, no Ackermann-feasible trajectory claim,
and no policy performance claim.

## Iterative PPO Mini-Loop Stability

After `limited_ppo_update_smoke_evaluated`, the next boundary is not a larger
PPO rollout. It is a three-round stability loop that repeats the smallest safe
cycle: collect policy-controlled sequential canary transitions, run one limited
PPO update from the same on-policy checkpoint, then re-run raw generalization,
sequential canary, and collector gates before the updated checkpoint becomes
the next round's base.

New artifacts:

- `configs/iterative_ppo_mini_loop_stability_v1.json`
- `configs/iterative_ppo_update_step_v1.json`
- `scripts/run_iterative_ppo_mini_loop_stability.py/.sh`
- `scripts/run_iterative_ppo_mini_loop_stability_closure.sh`
- `outputs/path_feedback_batch_iterative_ppo_mini_loop_stability_v1/`

The summary files are `iterative-ppo-mini-loop-stability-summary.json`,
`iterative-ppo-mini-loop-rounds.jsonl`,
`iterative-ppo-mini-loop-drift-report.json`, and
`iterative-ppo-mini-loop-rejection-report.json`. Each round must prove the
collector is on-policy for that round's base checkpoint, keep source-fallback
steps out of the optimizer, keep loss/grad/reward/return/advantage finite, keep
`abs(approx_kl)<=0.25`, and keep the cumulative parameter L2 delta within
`0.05`.

Post-update gates remain strict: raw TEST regression stays at 0, sequential
canary still covers 36 episode / 108 step with 6 families and no rejected
choice, and the collector still materializes at least 24 valid PPO-trainable
transitions. Passing readiness may advance only to
`iterative_ppo_mini_loop_stability_evaluated`.

This remains a mini-loop stability check only: no formal PPO rollout, no
checkpoint publication, no default-policy replacement, no network/action-space
or default-A* change, no distance-contract relaxation, no Ackermann-feasible
trajectory claim, and no policy performance claim.

## Guarded PPO Rollout Pilot

After the current-HEAD quasi-real iterative mini-loop reaches
`iterative_ppo_mini_loop_stability_evaluated`, the next boundary is the
Quasi-Real Guarded PPO Rollout Pilot. It is the first stage that treats the
rollout entry point itself as the object under test: the policy proposes each
step, the existing safety/contract/path/risk/source-selection gate decides
whether that choice may execute, and only train-split, policy-controlled,
gate-passed steps become PPO-trainable transitions.

New artifacts:

- `configs/guarded_ppo_rollout_pilot_v1.json`
- `configs/guarded_ppo_rollout_update_v1.json`
- `scripts/run_guarded_ppo_rollout_pilot.py/.sh`
- `scripts/run_guarded_ppo_rollout_pilot_closure.sh`
- `outputs/path_feedback_batch_guarded_ppo_rollout_pilot_v1/`

The pilot reuses the same 36 episode / 108 step sequential multi-step
opportunity set for comparability, then rechecks the updated candidate against
quasi-real teacher-following and quasi-real collector gates. Raw policy probe
rejections stay visible as diagnostics, but they are not counted as controlled
rollout regression after the gate falls back safely. It writes guarded
root-level aliases for the
pilot collector outputs: `guarded-ppo-rollout-episodes.jsonl`,
`guarded-ppo-rollout-transitions.jsonl`,
`guarded-ppo-rollout-reward-audit.json`,
`guarded-ppo-rollout-update-summary.json`,
`guarded-ppo-rollout-rejection-report.json`, and
`guarded-ppo-rollout-pilot-summary.json`.

Passing requires at least 24 PPO-trainable policy-controlled transitions, zero
source-fallback trainable samples, valid action masks, finite log-prob/value and
reward, state continuity, no fallback/open-grid, and zero safety, contract,
path/risk, or source-selection regression. The pilot then performs one tiny
on-policy PPO update from the same checkpoint and re-runs raw generalization,
generated sequential canary, generated collector, quasi-real teacher-following,
and quasi-real collector gates. Readiness may advance only to
`guarded_ppo_rollout_pilot_evaluated` when raw TEST regression and controlled
generated/quasi-real regressions are all zero.

This is still a guarded pilot only: no released PPO policy, no default-policy
replacement, no network/action-space/default-A* change, no distance-contract
relaxation, no Ackermann-feasible trajectory claim, no IRIS/GCS diagnostic as
training release evidence, and no policy performance claim.

## Quasi-Real Guarded PPO Rollout Pilot

`Quasi-Real Guarded PPO Rollout Pilot v1` is the quasi-real counterpart to the
generated guarded pilot. It does not run another PPO update. Instead, it takes
the experimental candidate produced by
`outputs/path_feedback_batch_return_aligned_guarded_ppo_update_smoke_v1/` and
lets it perform horizon-3 guarded rollout over the quasi-real teacher-following
contexts. Each step is a small supervised test drive: the policy proposes an
action, the existing distance/path-risk/source-selection/contract/safety guards
decide whether that action may control the step, and rejected raw probes remain
diagnostic-only after fallback.

New artifacts:

- `configs/quasi_real_guarded_ppo_rollout_pilot_v1.json`
- `scripts/run_quasi_real_guarded_ppo_rollout_pilot.py/.sh`
- `scripts/run_quasi_real_guarded_ppo_rollout_pilot_closure.sh`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_rollout_pilot_v1/`

Current evidence writes
`quasi-real-guarded-ppo-rollout-pilot-summary.json`,
`quasi-real-guarded-ppo-rollout-episodes.jsonl`,
`quasi-real-guarded-ppo-rollout-steps.jsonl`,
`quasi-real-guarded-ppo-rollout-rejection-report.json`, and
`quasi-real-guarded-ppo-rollout-reward-audit.json`. The pilot reports
`status=passed`, `reason_codes=[]`, `episode_count=36`, `step_count=108`,
`ppo_trainable_transition_count=36`, `diagnostic_transition_count=72`,
`controlled_regression_count=0`, `teacher_agreement_rate=1.0`, quasi-real
collector replay `status=passed` with 36 trainable transitions, and
`post_pilot_long_horizon_verdict=long_horizon_teacher_skill_contract_aligned`.

Readiness now accepts
`--quasi-real-guarded-ppo-rollout-pilot-summary` and can advance to
`quasi_real_guarded_ppo_rollout_pilot_evaluated` when the summary is passed,
current provenance matches, validation/test/fallback trainable leakage is zero,
log-prob/value/reward/return/advantage are finite, and no controlled
safety/contract/path-risk/source-selection regression is present.

This remains an evidence-producing guarded rollout pilot only: no formal PPO
rollout, no checkpoint publication, no default-policy replacement, no network
or action-space change, no default-A* change, no gate relaxation, no
Ackermann-feasible trajectory claim, no policy performance claim, and no formal
training-ready claim.

## Quasi-Real Guarded PPO Evidence Freeze

After `quasi_real_guarded_ppo_rollout_pilot_evaluated`, the next boundary is
evidence freeze, not more PPO. The new quasi-real freeze stage packages the
passed guarded rollout pilot into a reproducible audit bundle and makes the
readiness source explicit. This matters because the written
`policy-training-readiness-review-summary.json` under the batch root can become
stale when the worktree changes; the freeze trusts a fresh validate-only run
with the quasi-real pilot summary and records stale written readiness only as a
diagnostic.

New artifacts:

- `configs/quasi_real_guarded_ppo_evidence_freeze_v1.json`
- `scripts/run_quasi_real_guarded_ppo_evidence_freeze.py/.sh`
- `scripts/run_quasi_real_guarded_ppo_evidence_freeze_closure.sh`
- `docs/superpowers/specs/2026-06-14-quasi-real-guarded-ppo-evidence-freeze.md`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_evidence_freeze_v1/`

The freeze writes
`quasi-real-guarded-ppo-evidence-freeze-summary.json`,
`quasi-real-guarded-ppo-evidence-manifest.json`,
`quasi-real-guarded-ppo-readiness-validate-only.json`, and
`quasi-real-guarded-ppo-evidence-freeze-report.md`. Current evidence is
`status=passed`, `reason_codes=[]`, pilot status `passed`, readiness status
`quasi_real_guarded_ppo_rollout_pilot_evaluated`, required artifact missing
count 0, and `stale_written_readiness_summary_detected=true`. The manifest
covers 9 required artifacts with sha256 hashes, including the pilot summary,
episodes, steps, rejection report, reward audit, collector replay summary,
long-horizon summary, return-aligned update smoke summary, and the fresh
readiness validate-only artifact.

The closure may refresh the quasi-real guarded rollout pilot to current
provenance before freezing, but it does not run a new PPO update, publish a
checkpoint, replace the default policy, relax gates, or make performance or
formal-training-ready claims.

## Quasi-Real Guarded PPO Stability Replay

After evidence freeze, the next boundary is stability replay and acceptance
contract refinement. The stage replays the frozen quasi-real guarded PPO rollout
pilot three times with the same experimental checkpoint and quasi-real root,
then compares every replay against the frozen baseline. It is meant to answer
whether the passed guarded rollout is reproducible, not whether the policy is
ready for formal training.

New artifacts:

- `configs/quasi_real_guarded_ppo_stability_replay_v1.json`
- `scripts/run_quasi_real_guarded_ppo_stability_replay.py/.sh`
- `scripts/run_quasi_real_guarded_ppo_stability_replay_closure.sh`
- `docs/superpowers/specs/2026-06-14-quasi-real-guarded-ppo-stability-replay.md`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_stability_replay_v1/`

The stage writes
`quasi-real-guarded-ppo-stability-replay-summary.json`,
`stability-replay-comparison.jsonl`,
`acceptance-contract-refinement.json`,
`stability-replay-progress-events.jsonl`,
`quasi-real-guarded-ppo-stability-readiness-validate-only.json`, and
`stability-replay-report.md`. Current evidence is `status=passed`,
`reason_codes=[]`, `replay_count=3`, `passed_replay_count=3`, readiness status
`quasi_real_guarded_ppo_stability_replay_evaluated`, 36 episodes, 108 steps,
36 trainable transitions, 72 diagnostic transitions, teacher agreement 1.0,
controlled regression count 0, and baseline/replay behavior drift count 0.

The refined acceptance contract lists hard gates such as freeze passed, all
replays passed, split/fallback leakage zero, materialization complete, finite
reward/return/advantage, controlled regression zero, collector replay passed,
long-horizon teacher-skill contract aligned, and readiness validate-only
passed. Validation/test, source fallback, teacher fallback, raw policy probe
rejection, non-empty gate reasons, and IRIS/GCS/path-planner diagnostics remain
diagnostic-only.

This remains an audit and contract-refinement stage only: no new PPO update,
batch expansion, checkpoint publication, default-policy replacement, gate
relaxation, policy performance claim, or formal-training-ready claim is made.

## Quasi-Real Guarded PPO Horizon-5 Batch Expansion

After stability replay, the next boundary is a first-stage longer-horizon and
wider-batch expansion. `Quasi-Real Guarded PPO Horizon-5 Batch Expansion v1`
does not start formal PPO. It takes the passed stability replay evidence,
follows it back through the freeze manifest to the same quasi-real guarded pilot
steps, then deterministically rebuilds 96 guarded episodes with horizon 5.

New artifacts:

- `configs/quasi_real_guarded_ppo_horizon5_batch_expansion_v1.json`
- `scripts/run_quasi_real_guarded_ppo_horizon5_batch_expansion.py/.sh`
- `scripts/run_quasi_real_guarded_ppo_horizon5_batch_expansion_closure.sh`
- `docs/superpowers/specs/2026-06-14-quasi-real-guarded-ppo-horizon5-batch-expansion.md`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_horizon5_batch_expansion_v1/`

The stage writes expanded episodes, expanded steps, reward audit, rejection
report, replay comparison, progress events, readiness validate-only output, and
a markdown report. Current closure evidence is `status=passed`,
`reason_codes=[]`, `horizon=5`, `episode_count=96`, `step_count=480`,
`ppo_trainable_transition_count=162`, `diagnostic_transition_count=318`,
`replay_count=3`, `passed_replay_count=3`, readiness status
`quasi_real_guarded_ppo_horizon5_batch_expansion_evaluated`, teacher agreement
1.0, controlled regression count 0, and behavior drift count 0.

Trainability remains strict: only train split, gate-clean, controlled policy
steps can be PPO-trainable. Validation/test split steps, source fallback,
teacher fallback, raw probe rejection, non-empty gate reasons, and
IRIS/GCS/path-planner diagnostics remain diagnostic-only. Returns and
advantages are recalculated over five-step episodes, so the accounting is
multi-step return aligned rather than single-step greedy.

This is still an expansion audit only: no formal PPO, no new optimizer update,
no checkpoint publication, no default-policy replacement, no gate relaxation,
no performance claim, and no formal-training-ready claim. Formal PPO preflight
still needs a larger evidence gate such as at least 512 trainable transitions
and multi-seed stability.

## Quasi-Real Guarded PPO Scale-512 Multi-Seed Preflight

`Quasi-Real Guarded PPO Scale-512 Multi-Seed Preflight v1` adds the formal PPO
preflight gate after Horizon-5. It is still not formal PPO. It asks whether the
quasi-real guarded rollout evidence is large and diverse enough to justify a
later real training run: at least 512 PPO-trainable transitions, at least 512
unique trainable contexts, horizon at least 5, and three seed-level tiny PPO
smoke checks with finite losses, bounded KL, bounded clipped gradient norm, no
teacher-skill regression, and no controlled rollout regression.

New artifacts:

- `configs/quasi_real_guarded_ppo_scale512_multiseed_preflight_v1.json`
- `scripts/run_quasi_real_guarded_ppo_scale512_multiseed_preflight.py/.sh`
- `scripts/run_quasi_real_guarded_ppo_scale512_multiseed_preflight_closure.sh`
- `docs/superpowers/specs/2026-06-14-quasi-real-guarded-ppo-scale512-multiseed-preflight.md`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_scale512_multiseed_preflight_v1/`

The preflight is intentionally strict. It only counts train split, gate-clean,
controlled-policy transitions with complete observations, log probabilities,
values, finite rewards, finite returns, and finite advantages. Validation/test
split, source fallback, teacher fallback, raw probe rejection, non-empty gate
reason, and path-planner/IRIS/GCS diagnostic rows stay diagnostic-only.

With the current Horizon-5 evidence, the gate is expected to fail for capacity
rather than optimizer instability: Horizon-5 has 162 trainable transitions, but
only 36 unique trainable contexts traced back to the quasi-real input. The
Scale-512 runner therefore reports `insufficient_quasi_real_trainable_capacity`
and skips seed smoke instead of duplicating examples to fake scale.

This is the correct stop signal before formal PPO. The next engineering target
is upstream quasi-real trainable context expansion: mine or generate at least
512 real, distinct train split, gate-clean quasi-real contexts, then rerun this
preflight. No checkpoint is published, no default policy is replaced, no gate is
relaxed, and no formal-training-ready claim is made.

## Quasi-Real Trainable Context Expansion

`Quasi-Real Trainable Context Expansion v1` implements that upstream capacity
audit. It is not formal PPO and does not change the policy. It reads the passed
Horizon-5 evidence and the failed Scale-512 preflight, then scans an explicit
quasi-real source ledger for train split, controlled-policy, gate-clean,
fully-materialized, finite trainable contexts.

New artifacts:

- `configs/quasi_real_trainable_context_expansion_v1.json`
- `scripts/run_quasi_real_trainable_context_expansion.py/.sh`
- `scripts/run_quasi_real_trainable_context_expansion_closure.sh`
- `docs/superpowers/specs/2026-06-14-quasi-real-trainable-context-expansion.md`
- `outputs/path_feedback_batch_quasi_real_trainable_context_expansion_v1/`

The stage writes the selected unique context ledger, rebuilt expansion steps,
capacity audit, source audit, rejection report, markdown report, and, when
capacity is sufficient, an `expanded_horizon5/` ledger for rerunning Scale-512.
It does not count repeated `context_id` values toward capacity. Validation/test
split rows, fallback rows, teacher fallback rows, raw probe rejection rows,
non-empty gate reasons, controlled regressions, and IRIS/GCS/path-planner
diagnostics remain diagnostic-only.

Current closure evidence is passed: `status=passed`, `reason_codes=[]`,
`source_row_count=1128`, `materialized_source_row_count=648`,
`materialized_trainable_context_count=648`,
`unique_trainable_context_count=684`,
`ppo_trainable_transition_count=684`, `duplicate_trainable_context_count=0`,
and `scale512_status=passed`. The runner now materializes train split
teacher-distillation raw slices by reconstructing policy observations from the
paired path-feedback candidates and refreshing log_prob/value from the same
experimental candidate checkpoint used by the PPO smoke.

The Scale-512 rerun also passes with `seed_count=3`, `passed_seed_count=3`,
zero controlled regression, zero old log_prob/value reconstruction error,
`seed_max_abs_approx_kl≈1.01e-5`, and `seed_max_grad_norm_after_clip=1.0`.
Readiness advances to
`quasi_real_guarded_ppo_scale512_multiseed_preflight_evaluated`. This remains a
formal-PPO preflight, not a formal PPO training completion: there is still no
checkpoint publication, default-policy replacement, gate relaxation, policy
performance claim, or formal-training-ready claim.

## Quasi-Real Guarded PPO Iterative Mini-Loop Stability

`Quasi-Real Guarded PPO Iterative Mini-Loop Stability v1` is the next boundary
after the 684-context expansion. It is still not formal PPO and does not require
downloading more raw data first. It asks a narrower question: if the same
experimental base candidate takes three tiny PPO smoke steps per seed, across
seeds `[0,1,2]`, do teacher skill, controlled gates, on-policy reconstruction,
finite loss/gradient/return/advantage accounting, KL, and clipped gradient norm
stay stable?

New artifacts:

- `configs/quasi_real_guarded_ppo_iterative_miniloop_stability_v1.json`
- `scripts/run_quasi_real_guarded_ppo_iterative_miniloop_stability.py/.sh`
- `scripts/run_quasi_real_guarded_ppo_iterative_miniloop_stability_closure.sh`
- `tests/test_quasi_real_guarded_ppo_iterative_miniloop_stability.py`
- `docs/superpowers/specs/2026-06-14-quasi-real-guarded-ppo-iterative-miniloop-stability.md`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_iterative_miniloop_stability_v1/`

The runner reads the current trainable context expansion summary and its
Scale-512 rerun, requires both to be passed, and only admits the 684 train
split, gate-clean, controlled policy transitions into the optimizer. Each seed
starts from the same experimental base candidate; each iteration refreshes
`log_prob/value` from the current base candidate before materializing collector
episodes and running a full-batch PPO smoke update. The next iteration in the
same seed chains from the previous experimental output. A progress JSONL records
seed, iteration, optimizer count, KL, clipped grad norm, teacher agreement, and
controlled regression for every mini-loop step.

Readiness now accepts
`--quasi-real-guarded-ppo-iterative-miniloop-stability-summary` and advances to
`quasi_real_guarded_ppo_iterative_miniloop_stability_evaluated` only when all
3x3 updates pass, optimizer count remains exactly 684, validation/test/fallback
trainable counts are zero, old policy reconstruction stays within `1e-4`, all
numeric counters are finite, teacher agreement stays at least 0.95, controlled
regression and behavior drift remain zero, and publication/performance/formal
ready claims stay false.

## Training Progress Telemetry

Long guarded/iterative closures can spend minutes inside generated sequential
path-feedback, compatibility replay, collector, or PPO update stages without
printing new JSON stdout. `Training Progress Telemetry & Progress Bar v1` adds a
small observability layer for those runs without changing training behavior.

New artifacts:

- `configs/training_progress_telemetry_v1.json`
- `scripts/training_progress.py`
- `outputs/.../training-progress-events.jsonl`
- `outputs/.../training-progress-summary.json`

Supported entry points accept `--progress auto|plain|jsonl|off`. Plain progress
is written to stderr so existing machine-readable JSON stdout remains stable.
Structured events record the run id, stage, status, current/total counters,
round/step indexes, elapsed time, summary path, reason codes, and stage metrics.
The guarded pilot closure writes its progress artifacts under
`outputs/path_feedback_batch_guarded_ppo_rollout_pilot_v1/`.

The progress layer tracks coarse closure stages and inner signals such as
`round 2/3`, `sequential step 2/3`, `ppo epoch 1/1`, trainable/diagnostic
transition counts, raw rejected policy choices, controlled regression counts,
loss, KL, gradient norm, parameter delta, and finite/non-finite counters.
`--progress off` preserves the old behavior and writes no progress artifacts.

This stage is observability only: it does not expand PPO batches, change reward
or gates, alter readiness semantics, publish checkpoints, or claim formal
training readiness.

## Guarded PPO Evidence Freeze

After the guarded pilot closure and progress telemetry pass, the next boundary
is **Guarded PPO Evidence Freeze & Reproducible Closure v1**. This stage does
not run another PPO update. It freezes the evidence package that proves why the
current guarded pilot is accepted: the guarded pilot summary, progress summary,
progress event log, explicit readiness validate-only result, git provenance, and
artifact hashes.

New artifacts:

- `configs/guarded_ppo_evidence_freeze_v1.json`
- `scripts/run_guarded_ppo_evidence_freeze.py/.sh`
- `outputs/path_feedback_batch_guarded_ppo_evidence_freeze_v1/`

The freeze output writes `guarded-ppo-evidence-freeze-summary.json`,
`evidence-manifest.json`, `readiness-final.json`,
`progress-consistency-report.json`, and `reproducibility-report.md`. The
manifest records every required artifact path, existence, schema/status summary,
and sha256 digest. The consistency report explicitly detects stale readiness
drift, for example when an older `policy-training-readiness-review-summary.json`
still reports an iterative status while a fresh validate-only run with the
guarded summary reports `guarded_ppo_rollout_pilot_evaluated`.

Passing freeze requires the guarded pilot summary to remain `passed`, progress
telemetry to remain `passed` with zero failed stages, readiness to return
`guarded_ppo_rollout_pilot_evaluated` with no blockers, and all required
artifacts to exist with non-empty hashes. Stale readiness is diagnostic only;
the final source of truth is the explicit guarded-summary readiness run.

This is evidence packaging only: it does not publish checkpoints, replace the
default policy, claim formal training readiness, relax gates, or change PPO,
network, action space, default A*, reward, collector, or path-risk contracts.

## Policy Training CUDA Device Support

After `guarded_ppo_rollout_pilot_evaluated`, progress telemetry, and evidence
freeze, a later boundary is not a larger rollout or real-map training. It is an opt-in
training-device contract so later larger PPO runs can use GPU compute without
changing existing CPU evidence.

New artifacts:

- `configs/policy_training_cuda_device_support_v1.json`
- `scripts/run_policy_training_cuda_device_support_smoke.py/.sh`
- `outputs/path_feedback_batch_policy_training_cuda_device_support_v1/`

Training configs now support `training.device` with `cpu`, `cuda`, or `auto`.
The default remains CPU. `auto` uses CUDA when available and records
`fallback_to_cpu=true` if it must fall back. Explicit `cuda` fails when CUDA is
unavailable with `cuda_requested_but_unavailable`. Summaries report
`requested_device`, `resolved_device`, `cuda_available`, `cuda_device_name`,
and `fallback_to_cpu`.

The limited PPO update path now moves the network and PPO tensors to the
resolved device, recomputes old log-prob/value on that device, and serializes
checkpoint state dicts back to CPU so artifacts remain portable. Guarded and
iterative summaries preserve update-device provenance from their update step.

Passing the smoke requires at least 24 optimizer transitions, no source-fallback
trainable samples, old log-prob/value max abs error `<=1e-4`, finite
loss/grad/reward/return/advantage, `parameter_l2_delta>0`,
`abs(approx_kl)<=0.25`, `max_grad_norm_after_clip<=1.0`, and a checkpoint that
can be loaded on CPU. Readiness may record
`policy_training_cuda_device_support_evaluated`.

This only accelerates policy training tensor computation. It does not speed up
path planner or evidence generation, does not add real-map data, does not
publish or replace any checkpoint, and does not claim policy performance.

## Quasi-Real Map Domain Gap Evaluation

After `policy_training_cuda_device_support_evaluated`, the next boundary is not
a larger PPO update. It is a quasi-real map domain-gap check: the generated
canary training field must be compared against LOLA south-pole map slices before
real-map data is allowed to influence training.

New artifacts:

- `configs/quasi_real_map_domain_gap_evaluation_v1.json`
- `scripts/run_quasi_real_lola_data_prepare.py/.sh`
- `scripts/run_quasi_real_map_path_feedback_bridge.py/.sh`
- `scripts/run_quasi_real_map_domain_gap_evaluation.py/.sh`
- `scripts/run_quasi_real_map_domain_gap_closure.sh`
- `outputs/path_feedback_batch_quasi_real_map_domain_gap_v1/`

The data prepare step reads
`model-explorer/data/manifests/lunar_south_pole_lro_lola_gdr_875s_20m.json`,
downloads missing ignored raw files into `model-explorer/data/raw/...`, and
validates bytes plus SHA-256. The bridge converts LOLA ROI windows from the
existing quasi-real matrix manifests into path-feedback artifacts with
`map_source.kind=lola_quasi_real_roi`, stable `context_id`, derived cost,
passable mask, and terrain layers. The domain-gap evaluator compares quasi-real
ROI feedback with current generated evidence and reports whether the next step
is acceptable, scenario expansion, or planner/contract triage.

Readiness may advance only to `quasi_real_map_domain_gap_evaluated`. This is
shadow/domain-gap evidence only: no PPO update, no checkpoint release, no
default-policy replacement, no network/action-space/default-A* change, no
distance-contract relaxation, no Ackermann-feasible trajectory claim, and no
policy performance claim.

## Quasi-Real Shadow Policy Behavior Audit

After `quasi_real_map_domain_gap_evaluated`, the next boundary is not real-map
takeover or training. It is a shadow behavior audit on the accepted LOLA ROI
slices: the experimental policy scores each quasi-real context, but source
selection still owns the route and no controlled choice is executed.

New artifacts:

- `configs/quasi_real_shadow_policy_behavior_audit_v1.json`
- `scripts/run_quasi_real_shadow_policy_behavior_audit.py/.sh`
- `scripts/run_quasi_real_shadow_policy_behavior_closure.sh`
- `outputs/path_feedback_batch_quasi_real_shadow_policy_behavior_v1/`

The audit writes one shadow decision per quasi-real context:
`source_aligned`, `policy_changed_gate_passed`,
`policy_changed_gate_rejected`, or `not_scored`. Each record keeps
`context_id`, ROI metadata, source/raw policy actions, logit margin,
action-mask validity, path/risk deltas, and gate reason codes. It never writes
PPO transitions, never lets the policy take control, and never runs an
optimizer update.

Readiness may advance only to `quasi_real_shadow_policy_behavior_audited` when
all quasi-real contexts are scored, at least four ROI groups are covered,
context IDs are complete, action-mask and regression gates remain at 0, and
`behavior_verdict=acceptable_for_quasi_real_guarded_pilot`.

This stage is still an audit: no quasi-real guarded pilot yet, no PPO update,
no checkpoint release, no default-policy replacement, no network/action-space
or default-A* change, no distance/path-risk/source-selection relaxation, no
Ackermann-feasible trajectory claim, and no policy performance claim.

## Quasi-Real Shadow Failure Taxonomy and Anti-Overfit Alignment

The first quasi-real shadow audit exposed one real alignment failure rather than
a planner/bridge failure: `lola_qreal_mixed_risk_test_011` was scored with
source action `1`, while the raw policy preferred action `2`; that alternative
triggered both `path_cost_regression` and `risk_regression`. The correct
response is not to turn that single context into a hard positive or to loosen
the gate. It is treated as a failure seed for a small anti-overfit calibration
loop.

New artifacts:

- `configs/quasi_real_shadow_failure_taxonomy_v1.json`
- `configs/quasi_real_shadow_alignment_splits_v1.json`
- `configs/quasi_real_shadow_alignment_preference_v1.json`
- `configs/quasi_real_shadow_alignment_candidate_v1.json`
- `scripts/run_quasi_real_shadow_failure_taxonomy.py/.sh`
- `scripts/run_quasi_real_shadow_alignment_dataset.py/.sh`
- `scripts/run_quasi_real_shadow_alignment_preference_mining.py/.sh`
- `scripts/run_quasi_real_shadow_alignment_candidate.py/.sh`
- `scripts/run_quasi_real_shadow_alignment_closure.sh`
- `outputs/path_feedback_batch_quasi_real_shadow_failure_taxonomy_v1/`
- `outputs/path_feedback_batch_quasi_real_shadow_alignment_dataset_v1/`
- `outputs/path_feedback_batch_quasi_real_shadow_alignment_preference_v1/`
- `outputs/path_feedback_batch_quasi_real_shadow_alignment_candidate_v1/`

The taxonomy classifies the baseline failure as
`path_risk_joint_regression` and verifies that there is no action-mask,
contract, bridge, or path-feedback gap. The alignment dataset derives disjoint
train/val/holdout quasi-real variants around the failure seed. Preference
mining creates only rule-shaped hard-negative pairwise samples:
source-selected should outrank the raw policy alternative when the alternative
is worse on both path cost and risk. It adds no hard positives and no PPO
transitions.

The alignment candidate starts from the current guarded experimental checkpoint,
performs a small pairwise calibration on the train split, and then shadows the
train/val/holdout variants plus the original 12 ROI contexts. The current
functional closure reports 1 taxonomy failure, 3 quasi-real hard-negative
preference samples, zero split leakage, zero hard positives, zero PPO
transitions, and zero holdout/original ROI path-risk regressions after
calibration. Formal readiness still requires a clean-HEAD evidence refresh
because tracked code and docs changed after the upstream guarded/CUDA/domain-gap
roots were produced.

Readiness may advance only to `quasi_real_shadow_alignment_evaluated` after
clean-HEAD provenance matches every consumed summary. This remains
quasi-real shadow alignment evidence only: no quasi-real policy takeover, no
formal PPO rollout, no checkpoint publication or default-policy replacement, no
network/action-space/default-A* change, no distance/path-risk/source-selection
contract relaxation, no Ackermann-feasible trajectory claim, and no policy
performance claim.

## Current-HEAD Sequential/Guarded Evidence Re-closure

The quasi-real mixed-risk alignment closure is functionally passed, but formal
readiness depends on the upstream generated sequential/guarded evidence being
refreshed under the same HEAD. The current re-closure stage fixes that upstream
evidence boundary instead of adding new quasi-real training samples.

The immediate blocker was sequential opportunity distribution: current-HEAD
probes showed enough single-step safe-better choices, but too few episodes with
two consecutive safe takeovers. The rollout config now pins two additional
episode starts that were verified by focused sequential probes:

- `seq-mixed_stress_detour-b` starts at `[8, 9]`.
- `seq-path_complexity_benefit-b` starts at `[8, 9]`.

The sequential diagnosis summary now includes a per-family opportunity report
covering safe-better steps, policy-used opportunities, policy-missed
opportunities, missing opportunities, accepted takeover steps, multi-step
accepted episodes, and a family-level gap reason. This is intended to prevent
future failures from being misread as a training problem when the actual issue
is missing sequential opportunity geometry.

The re-closure remains evidence hygiene and generated-scenario canary work. It
does not start formal PPO rollout, does not take over on quasi-real maps, does
not publish or replace a checkpoint, does not change network/action-space/default
A*, does not relax distance/path-risk/source-selection contracts, and does not
claim Ackermann-feasible trajectory or policy performance.

## Quasi-Real Guarded Policy Pilot

After `quasi_real_shadow_alignment_evaluated`, the next boundary is a guarded
pilot on LOLA quasi-real ROI contexts. This is the first quasi-real stage where
a policy-changed decision may be accepted as a controlled choice, but only if it
passes the same action-mask, candidate-present, reachable, no-fallback,
contract, path/risk, and source-selection gates. If the policy changes its mind
and fails a gate, the pilot falls back to source-selected and records the exact
reason. This v1 pilot was originally a value-oriented test that expected at
least one accepted changed choice; the later teacher-equivalent roadmap
reclassifies full source alignment as valid behavior when no safe-better
alternative exists.

New artifacts:

- `configs/quasi_real_guarded_policy_pilot_v1.json`
- `scripts/run_quasi_real_guarded_policy_pilot.py/.sh`
- `scripts/run_quasi_real_guarded_policy_pilot_closure.sh`
- `outputs/path_feedback_batch_quasi_real_guarded_policy_pilot_v1/`

The pilot reuses the existing quasi-real shadow scoring and gate logic, then
writes `quasi-real-guarded-policy-decision/v1` records with
`controlled_choice_source=policy|source|source_fallback`. Its summary reports
quasi-real context count, ROI group coverage, changed/pass/rejected counts,
fallback count, all gate regression counters, and
`guarded_pilot_verdict`. Readiness may advance only to
`quasi_real_guarded_policy_pilot_evaluated` for the value-oriented pilot when
the summary is passed, there is at least one gate-passed changed choice,
`policy_changed_gate_rejected_count=0`, all gate regressions are 0, and every
consumed evidence root matches the current HEAD. Teacher-equivalent validation
uses a different success criterion: source-aligned decisions may be correct when
the quasi-real candidate set contains no safe-better opportunity.

This stage is still a guarded pilot, not a quasi-real PPO collector or policy
release. It does not run a PPO optimizer update, does not write PPO
transitions, does not publish or replace a checkpoint, does not change
network/action-space/default A*, does not relax distance/path-risk/source-
selection contracts, and does not claim Ackermann-feasible trajectory or policy
performance.

## Quasi-Real Safe-Alternative Opportunity Diagnosis

`Quasi-Real Guarded Policy Pilot v1` can now expose a useful blocker: the policy
may stay source-aligned on all LOLA quasi-real contexts, with zero safety,
contract, fallback, path/risk, or source-selection regression. That is safe, but
it does not prove the policy can make valuable quasi-real choices. The next
diagnostic asks a narrower question: do the quasi-real ROI contexts actually
contain any top-k alternative that is gate-safe and better than the
source-selected baseline?

New artifacts:

- `configs/quasi_real_safe_alternative_opportunity_diagnosis_v1.json`
- `scripts/run_quasi_real_safe_alternative_opportunity_diagnosis.py/.sh`
- `outputs/path_feedback_batch_quasi_real_safe_alternative_opportunity_diagnosis_v1/`

The diagnosis reads the quasi-real domain-gap root, the failed guarded pilot
root, and the quasi-real shadow alignment candidate root. For every context it
uses the source-selected candidate as baseline and runs each top-k alternative
through a funnel: candidate present, action mask valid, reachable, no replan, no
fallback/open-grid, contract safe, path/risk/source-selection non-regressive,
safe alternative, and safe-better alternative. `safe-better` keeps the existing
canary value/stability rule: no gate regression and path cost improves by at
least `0.25`, or risk improves by at least `0.01`, or utility improves by at
least `0.005`.

The summary classifies each context as `opportunity_missing`,
`safe_alternative_exists_but_not_better`,
`safe_better_opportunity_exists_policy_source_aligned`,
`safe_better_opportunity_policy_selected`, `bridge_or_feedback_gap`, or
`action_mask_or_contract_gap`. Its verdict is intentionally diagnostic:
`quasi_real_safe_alternative_opportunity_gap` means the quasi-real ROI/start/goal
selection needs expansion; `acceptable_for_quasi_real_safe_choice_calibration`
means safe-better opportunities exist and a later calibration stage may use
them; bridge or action-mask/contract gaps stop the pipeline.

Readiness may advance only to
`quasi_real_safe_alternative_opportunity_diagnosed`. This is not a quasi-real
guarded pilot pass and not PPO readiness. The stage does not run a PPO optimizer
update, does not write PPO transitions, does not publish or replace a checkpoint,
does not change network/action-space/default A*, does not relax
distance/path-risk/source-selection gates, and does not claim Ackermann-feasible
trajectory or policy performance.

## Quasi-Real Safe-Better Opportunity Expansion

`Quasi-Real Safe-Alternative Opportunity Diagnosis v1` now gives a concrete
blocker: the 12 LOLA quasi-real contexts cover 4 ROI groups and all system gates
are clean, but no top-k candidate is both gate-safe and better than the
source-selected baseline. In practical terms, the quasi-real road surface is
connected and safe to inspect, but it does not yet contain enough real
"safe lane-change" opportunities for the policy to choose.

The expansion stage adds start-cell aware quasi-real ROI variants:

- `model-explorer/src/model_explorer/data/evaluation_matrix.py` now lets each
  ROI specify an optional `start_cell`, defaulting to `[0, 0]`.
- `scripts/run_quasi_real_map_path_feedback_bridge.py` carries that start cell
  into `LolaSouthPoleRoiConfig`, `current_cell`, slice metadata, and stable
  context IDs.
- `scripts/run_quasi_real_safe_better_opportunity_expansion.py/.sh` generates a
  larger LOLA matrix from neighboring ROI windows and multiple passable starts.
- `scripts/run_quasi_real_safe_better_opportunity_expansion_closure.sh` runs the
  matrix generation, bridge, path-feedback, domain-gap check, opportunity
  diagnosis, and final expansion summary.

The generated matrix is
`model-explorer/data/manifests/lunar_south_pole_lro_lola_safe_better_opportunity_matrix_v1.json`;
the evidence root is
`outputs/path_feedback_batch_quasi_real_safe_better_opportunity_expansion_v1/`.
Final strict diagnosis stays at `top_k=3` and must find at least 8 safe
alternative contexts, at least 4 safe-better contexts, and at least 2 ROI groups
with safe-better opportunity before readiness may advance to
`quasi_real_safe_better_opportunity_expanded`.

This stage still does not train the policy, execute quasi-real takeover, write
PPO transitions, publish or replace a checkpoint, change the network/action
space/default A*, relax distance/path-risk/source-selection gates, or claim
Ackermann-feasible trajectory or policy performance.

## Quasi-Real Teacher-Equivalent Development Direction

The safe-better expansion evidence changes the near-term roadmap. The current
expanded LOLA root contains 108 quasi-real contexts across 4 ROI groups and 9
start cells. The bridge, domain-gap, context-id, action-mask, fallback, safety,
contract, path/risk, and source-selection gates are clean, but the strict
`top_k=3` diagnosis still reports:

```text
safe_alternative_context_count=0
safe_better_opportunity_context_count=0
roi_group_with_safe_better_opportunity_count=0
opportunity_missing_count=108
```

This should not be read as "the policy failed to learn." In quasi-real terrain,
the source-selected teacher may already be the non-regressive choice in the
candidate set. If there is no gate-safe and better alternative, source alignment
is the correct behavior, not over-conservatism. Safe-better opportunity remains
valuable, but it is now an **incremental value branch** for finding teacher blind
spots, not the main prerequisite for proving the policy learned the teacher.

The quasi-real mainline should therefore advance in this order:

1. `Quasi-Real Teacher-Equivalent Validation`: prove the experimental policy
   reproduces source-selected decisions on quasi-real ROI contexts with zero
   gate regression. High source-aligned rate is acceptable and expected.
2. `Quasi-Real Teacher Distillation Robustness`: expand quasi-real ROI
   train/validation/holdout coverage and validate teacher agreement without
   context or slice leakage.
3. `Quasi-Real Guarded Teacher-Following Pilot`: let the policy propose actions
   under the same gates, but treat teacher-aligned controlled steps as valid
   behavior. Changed choices remain guarded and diagnostic.
4. `Quasi-Real PPO Collector Dry-Run`: materialize only contract-valid
   teacher-following or gate-passed policy transitions; source fallback and
   rejected changes stay diagnostic.
5. `Limited Quasi-Real PPO Update Smoke`: run a tiny local PPO update from the
   experimental checkpoint and require generated plus quasi-real gates to remain
   clean.
6. `Broader Real/Quasi-Real Domain Evaluation`: increase ROI area, terrain
   diversity, observation quality variation, and holdout coverage.

The parallel value branches are `Safe-Better Opportunity Search`,
`Source-Selection Blind Spot Mining`, and `Reward/Preference Calibration`.
Finding safe-better choices can improve the policy beyond the teacher, but not
finding them must not block teacher-equivalent validation.

## Quasi-Real Teacher-Equivalent Validation

`Quasi-Real Teacher-Equivalent Validation v1` implements the first mainline
step after the roadmap shift above. It is a shadow-only check over the expanded
LOLA quasi-real root: the policy scores each context, but the teacher/source
selection remains the reference and the policy never takes control.

The key semantic change is that `source_aligned` is treated as normal
teacher-equivalent behavior. `policy_changed_gate_passed` is allowed as a safe
disagreement, but it is not required. `policy_changed_gate_rejected` is an
unsafe disagreement and fails the stage. This keeps the value-oriented
`Quasi-Real Guarded Policy Pilot` intact while giving the quasi-real mainline a
separate way to prove that the policy has learned the teacher when safe-better
alternatives are objectively absent.

New artifacts:

- `configs/quasi_real_teacher_equivalent_validation_v1.json`
- `scripts/run_quasi_real_teacher_equivalent_validation.py/.sh`
- `scripts/run_quasi_real_teacher_equivalent_validation_closure.sh`
- `outputs/path_feedback_batch_quasi_real_teacher_equivalent_validation_v1/`

The validation summary is
`quasi-real-teacher-equivalent-summary.json`. It reports
`teacher_equivalent_context_count`, `policy_decision_count`,
`teacher_aligned_count`, `teacher_agreement_rate`,
`safe_disagreement_count`, `unsafe_disagreement_count`,
`policy_changed_gate_rejected_count`, ROI-group agreement breakdown, and all
gate-regression counters. Readiness accepts
`--quasi-real-teacher-equivalent-validation-summary` and may advance to
`quasi_real_teacher_equivalent_validated` only when coverage is sufficient,
teacher agreement is at least `0.90`, unsafe disagreement is `0`, all
action-mask/fallback/safety/contract/path/risk/source-selection regression
counters are `0`, and provenance matches the current HEAD.

This stage does not run PPO, does not write PPO transitions, does not execute
policy takeover, does not publish or replace a checkpoint, does not change the
network/action space/default A*, does not relax distance/path-risk/source
selection contracts, and does not claim Ackermann-feasible trajectory or policy
performance.

Current execution note: the validation tooling exposed the expected blocker.
The default guarded candidate reached `teacher_agreement_rate=0.8333` with
`unsafe_disagreement_count=18`; all unsafe disagreements carried path-cost
regression, with 5 also carrying risk regression. A shadow-alignment probe
improved agreement to `0.8796` but still missed the teacher-equivalent gate.
This was treated as a teacher-distillation/alignment problem, not as a
safe-better opportunity blocker.

## Quasi-Real Teacher Distillation Robustness

`Quasi-Real Teacher Distillation Robustness v1` closes the teacher-equivalent
alignment gap without PPO, takeover, checkpoint publication, or any gate
relaxation. The stage classifies unsafe quasi-real disagreements, expands the
training signal from "teacher > current raw unsafe choice" to "teacher > every
gate-regressive alternative in the audited quasi-real top-k set", materializes
train/validation/holdout distillation slices, and trains a new experimental
candidate with pairwise preference loss only.

New artifacts:

- `configs/quasi_real_teacher_distillation_taxonomy_v1.json`
- `configs/quasi_real_teacher_distillation_dataset_v1.json`
- `configs/quasi_real_teacher_distillation_preference_v1.json`
- `configs/quasi_real_teacher_distillation_candidate_v1.json`
- `scripts/run_quasi_real_teacher_distillation_taxonomy.py/.sh`
- `scripts/run_quasi_real_teacher_distillation_dataset.py/.sh`
- `scripts/run_quasi_real_teacher_distillation_preference_mining.py/.sh`
- `scripts/run_quasi_real_teacher_distillation_candidate.py/.sh`
- `scripts/run_quasi_real_teacher_distillation_closure.sh`
- `outputs/path_feedback_batch_quasi_real_teacher_distillation_*_v1/`

Current evidence is passed:

- baseline unsafe disagreement classified: `18/18`
- distillation candidate pairs: `216`
- train preference samples: `648`
- `hard_positive_added_count=0`
- `ppo_transition_added_count=0`
- `holdout_leakage_count=0`
- post-distillation 108-context teacher validation:
  `teacher_agreement_rate=1.0`, `unsafe_disagreement_count=0`
- invalid mask, fallback/open-grid, safety, contract, path/risk, and
  source-selection regression counters are all `0`

Readiness accepts `--quasi-real-teacher-distillation-summary` and may advance
to `quasi_real_teacher_distillation_robustness_evaluated` only when the
distillation summary, post-distillation teacher-equivalent validation, leakage
checks, non-goal flags, and git provenance all pass. This is still not formal
PPO rollout, quasi-real policy takeover, checkpoint publication, or a policy
performance claim.

## Quasi-Real Guarded Teacher-Following Pilot

`Quasi-Real Guarded Teacher-Following Pilot v1` is the next quasi-real mainline
stage after teacher distillation. It moves from shadow-only teacher agreement to
a guarded pilot where the policy participates in controlled quasi-real
decisions, while the success definition remains teacher-following rather than
safe-better improvement.

This stage deliberately does not reuse the value-oriented
`Quasi-Real Guarded Policy Pilot` requirement that
`policy_changed_gate_passed_count > 0`. In the current LOLA top-k evidence,
safe-better opportunities are absent, so a source-aligned policy decision is a
valid teacher-following step:

- `source_aligned` -> `controlled_choice_source=policy_teacher_aligned`
- `policy_changed_gate_passed` -> `controlled_choice_source=policy_safe_disagreement`
- `policy_changed_gate_rejected` -> `controlled_choice_source=teacher_fallback`

New artifacts:

- `configs/quasi_real_guarded_teacher_following_pilot_v1.json`
- `scripts/run_quasi_real_guarded_teacher_following_pilot.py/.sh`
- `scripts/run_quasi_real_guarded_teacher_following_closure.sh`
- `outputs/path_feedback_batch_quasi_real_guarded_teacher_following_pilot_v1/`

The pilot passes only when the expanded quasi-real context coverage is intact,
teacher agreement remains at least `0.90`, teacher-following steps are at least
`90`, all unsafe disagreements are zero, and invalid mask, fallback/open-grid,
safety, contract, path/risk, and source-selection regressions remain zero. It
does not write PPO transitions, run PPO updates, publish checkpoints, replace
the default policy, or claim policy performance.

Readiness accepts `--quasi-real-guarded-teacher-following-pilot-summary` and may
advance to `quasi_real_guarded_teacher_following_pilot_evaluated` only when the
teacher-following summary and git provenance pass. Safe-better opportunity
absence remains a value/augmentation branch issue, not a blocker for this
teacher-following mainline.

## Quasi-Real PPO Collector Dry-Run

`Quasi-Real PPO Collector Dry-Run v1` is the next quasi-real mainline stage. It
does not run a PPO optimizer. It materializes the clean guarded
teacher-following decisions into PPO-consumable rollout records while preserving
train/validation/test split isolation.

New artifacts:

- `configs/quasi_real_ppo_collector_dry_run_v1.json`
- `scripts/run_quasi_real_ppo_collector_dry_run.py/.sh`
- `scripts/run_quasi_real_ppo_collector_closure.sh`
- `outputs/path_feedback_batch_quasi_real_ppo_collector_dry_run_v1/`

The collector reads
`outputs/path_feedback_batch_quasi_real_guarded_teacher_following_pilot_v1/`,
derives its candidate and quasi-real roots from the teacher-following summary,
and consumes `quasi-real-guarded-teacher-following-decisions.jsonl`. Only
`train` split decisions with `controlled_choice_source` equal to
`policy_teacher_aligned` or `policy_safe_disagreement` and no gate reason codes
are materialized as PPO-trainable transitions. `validation` and `test` splits,
`teacher_fallback`, `none`, `not_scored`, unsafe disagreement, and any non-empty
gate reason remain diagnostic-only.

Current closure evidence is passed:

- `episode_count=108`, `step_count=108`
- `ppo_trainable_transition_count=36`
- `diagnostic_transition_count=72`
- `source_fallback_trainable_count=0`
- invalid/empty mask, missing log-prob/value, non-finite reward, fallback,
  safety, contract, path/risk, and source-selection regression counters all `0`
- `publishes_checkpoint=false`, `replaces_default_policy=false`,
  `performance_claimed=false`, `formal_training_ready_claimed=false`
- readiness validate-only reaches
  `training_readiness_status=ppo_rollout_collector_dry_run_evaluated`

The next stage is `Limited Quasi-Real PPO Update Smoke`: a tiny local optimizer
smoke from the same experimental checkpoint and these 36 trainable transitions,
with all generated and quasi-real gates still required to remain clean. This
collector stage still does not publish checkpoints, replace the default policy,
change network/action space/default A*, relax distance/path-risk/source-selection
gates, claim Ackermann-feasible trajectories, or promote IRIS/GCS diagnostics to
training release evidence.

### Limited Quasi-Real PPO Update Smoke

The implementation now has a quasi-real wrapper for the next smoke boundary:

- `configs/limited_quasi_real_ppo_update_smoke_v1.json`
- `scripts/run_limited_quasi_real_ppo_update_smoke.py/.sh`
- `scripts/run_limited_quasi_real_ppo_update_smoke_closure.sh`
- `outputs/path_feedback_batch_limited_quasi_real_ppo_update_smoke_v1/`

The wrapper reuses the existing limited PPO update machinery, but makes the
trainable filter explicit for quasi-real data: only `train` split transitions
with `controlled_choice_source` equal to `policy_teacher_aligned` or
`policy_safe_disagreement` and empty gate reasons may enter the optimizer.
The generic generated smoke remains backward-compatible with
`controlled_choice_source=policy`.

Current-HEAD rebaseline confirms the optimizer smoke itself is clean. The
update consumed all 36 quasi-real trainable transitions, reconstructed old
`log_prob` and `value` with zero max error, produced a small non-zero parameter
delta (`~4.37e-4`) and tiny `approx_kl` (`~7.7e-5`), and kept the post-update
quasi-real teacher-following and quasi-real collector gates clean:

- post-update quasi-real teacher-following: `status=passed`,
  `teacher_agreement_rate=1.0`
- post-update quasi-real collector: `status=passed`,
  `ppo_trainable_transition_count=36`, `diagnostic_transition_count=72`

The strict generated sequential post-update gate still fails, so the wrapper
summary remains `status=failed` with
`reason_codes=["limited_quasi_real_ppo_update_post_update_gate_regression"]`.
After the accounting split, the generated sequential failure is no longer a
controlled rollout path/risk regression. It is a raw policy takeover/coverage
contract issue:

- `multi_step_accepted_episode_count_below_threshold`
- `family_with_multi_step_accepted_episode_count_below_threshold`
- `canary_rejected_policy_choice_count_above_threshold`

Current accounting reports `canary_rejected_policy_choice_count=6`,
`raw_policy_path_cost_regression_count=6`,
`raw_policy_risk_regression_count=2`,
`controlled_path_cost_regression_count=0`, and
`controlled_risk_regression_count=0`. No checkpoint is published, no default
policy is replaced, and no performance or formal-training-ready claim is made.

### Quasi-Real / Generated Sequential Contract Compatibility Diagnosis

The next implemented stage is a diagnosis pass, not another PPO update. It
answers whether the generated sequential failure is caused by the tiny
quasi-real PPO update, or whether the quasi-real teacher-following candidate and
the generated sequential canary contract were already incompatible.

New artifacts:

- `configs/quasi_real_generated_sequential_contract_compatibility_diagnosis_v1.json`
- `scripts/run_quasi_real_generated_sequential_contract_compatibility_diagnosis.py/.sh`
- `scripts/run_quasi_real_generated_sequential_contract_compatibility_closure.sh`
- `outputs/path_feedback_batch_quasi_real_generated_sequential_contract_compatibility_diagnosis_v1/`

The diagnosis reads the failed
`outputs/path_feedback_batch_limited_quasi_real_ppo_update_smoke_v1/` root,
copies the quasi-real base candidate checkpoint into
`diagnostic-base-candidate/`, preserves weights, refreshes current git
provenance, and marks the clone as diagnostic, experimental, non-publishing,
and non-default. It then runs generated sequential canary on that diagnostic
base candidate and compares it with the post-update generated sequential replay.

The summary classifies exactly one verdict:

- `ppo_update_induced_generated_regression`
- `pre_existing_generated_sequential_contract_mismatch`
- `stale_or_unreplayable_base_candidate`
- `gate_accounting_or_metric_mismatch`
- `diagnosis_inconclusive`

It writes `failed-step-comparison.jsonl`,
`baseline-vs-updated-sequential-summary.json`, and
`compatibility-diagnosis-report.md`, including failed family/reason counts,
source/raw/policy action indices, logits, selected target cells, and path/risk
delta comparisons. Current-HEAD rebaseline reports `status=passed`,
`failed_step_count=6`, and
`diagnosis_verdict=pre_existing_generated_sequential_contract_mismatch`: the
diagnostic base and updated candidate fail the same generated sequential
contract steps. This stage alone does not advance readiness; it supplies the
origin evidence consumed by the accounting and long-horizon contract stages. It
does not run PPO, publish checkpoints, replace the default policy, relax
generated sequential gates, modify network/action space/default A*, or claim
formal training readiness.

### Generated Sequential Gate Metric / Accounting Audit

`Generated Sequential Gate Metric / Accounting Audit v1` splits the generated
sequential failure into raw policy probe accounting and controlled rollout
accounting. The previous diagnosis showed six failed generated sequential steps,
but those steps fell back to the source action before controlled execution.

New artifacts:

- `configs/generated_sequential_gate_metric_accounting_audit_v1.json`
- `scripts/run_generated_sequential_gate_metric_accounting_audit.py/.sh`
- `scripts/run_generated_sequential_gate_metric_accounting_closure.sh`
- `outputs/path_feedback_batch_generated_sequential_gate_metric_accounting_audit_v1/`

The audit confirms the old accounting mixed raw policy probe rejection with
controlled cumulative path/risk regression. The origin-aware accounting result
is:

- `legacy_mismatch_count=6`
- `raw_policy_path_cost_regression_count=6`
- `raw_policy_risk_regression_count=2`
- `controlled_path_cost_regression_count=0`
- `controlled_risk_regression_count=0`
- corrected shadow `cumulative_path_cost_regression_count=0`
- corrected shadow `cumulative_risk_regression_count=0`

Generated sequential still does **not** pass as a raw takeover gate: the
corrected shadow summary remains failed because raw policy choices are rejected
and multi-step accepted coverage is too low. The diagnosis after origin split is
`pre_existing_generated_sequential_contract_mismatch`, with
`recommended_next_action=generated_sequential_contract_alignment_required`.
The audit by itself only refines the blocker; downstream long-horizon alignment
is what allows readiness to distinguish teacher-skill evidence from raw
takeover failure. This audit does not publish a checkpoint, replace default
policy, run PPO, remove generated sequential from the acceptance contract, or
claim performance.

### Generated Sequential Long-Horizon Teacher-Skill Contract Alignment

`Generated Sequential Long-Horizon Teacher-Skill Contract Alignment v1` is the
next generated-sequential contract stage. It keeps the generated sequential gate
in the acceptance chain, but changes the success question from "did the policy
take a different step?" to "does the controlled multi-step behavior match or
beat the teacher over cumulative return?"

New artifacts:

- `configs/generated_sequential_long_horizon_teacher_skill_contract_alignment_v1.json`
- `scripts/run_generated_sequential_long_horizon_teacher_skill_contract_alignment.py/.sh`
- `scripts/run_generated_sequential_long_horizon_teacher_skill_contract_alignment_closure.sh`
- `outputs/path_feedback_batch_generated_sequential_long_horizon_teacher_skill_contract_alignment_v1/`

The stage writes `long-horizon-teacher-skill-contract-summary.json`,
`teacher-vs-policy-return-comparison.jsonl`,
`teacher-equivalent-episode-report.md`, `beyond-teacher-opportunity-report.md`,
and `dominated-raw-choice-diagnostics.jsonl`. The return accounting compares the
teacher trajectory, controlled policy trajectory, and raw policy diagnostic
trajectory across the configured horizon. Same-as-teacher active choices count
as positive teacher-skill evidence. Beyond-teacher evidence requires cumulative
return dominance, not merely a one-step different target.

Current-HEAD rebaseline writes a passed summary with
`verdict=long_horizon_teacher_skill_contract_aligned`,
`teacher_equivalent_episode_count=36`, `beyond_teacher_episode_count=15`,
`dominated_raw_choice_count=6`, and `controlled_regression_episode_count=0`.
Readiness accepts
`--generated-sequential-long-horizon-teacher-skill-contract-summary`; with the
post-update quasi-real teacher-following summary, post-update collector summary,
limited quasi-real smoke summary, accounting audit summary, and long-horizon
summary all supplied, readiness now reports
`training_readiness_status=limited_quasi_real_ppo_update_smoke_evaluated`,
`training_blockers=[]`, and `reason_codes=[]`.

This does not start iterative PPO. It only establishes a current-HEAD evidence
baseline for the next stage, `Quasi-Real Iterative PPO Mini-Loop Stability v1`.
The stage still does not publish checkpoints, replace the default policy, relax
safety, path-risk, or source-selection gates, or claim formal training
readiness.

### Quasi-Real Iterative PPO Mini-Loop Stability

`Quasi-Real Iterative PPO Mini-Loop Stability v1` is the next quasi-real
training-readiness boundary after the current-HEAD rebaseline. It is separate
from the older generated `Iterative PPO Mini-Loop Stability` stage: the loop
starts from the quasi-real teacher-distillation experimental candidate, uses
quasi-real teacher-following plus quasi-real PPO collector materialization, and
judges generated sequential evidence through the accounting and long-horizon
teacher-skill contract instead of the strict raw takeover gate alone.

New artifacts:

- `configs/quasi_real_iterative_ppo_mini_loop_stability_v1.json`
- `scripts/run_quasi_real_iterative_ppo_mini_loop_stability.py/.sh`
- `scripts/run_quasi_real_iterative_ppo_mini_loop_stability_closure.sh`
- `outputs/path_feedback_batch_quasi_real_iterative_ppo_mini_loop_stability_v1/`

The summary files are
`quasi-real-iterative-ppo-mini-loop-stability-summary.json`,
`quasi-real-iterative-ppo-mini-loop-rounds.jsonl`,
`quasi-real-iterative-ppo-mini-loop-drift-report.json`, and
`quasi-real-iterative-ppo-mini-loop-rejection-report.json`. The summary keeps
the existing readiness schema `iterative-ppo-mini-loop-stability-summary/v1`
so readiness can advance to
`iterative_ppo_mini_loop_stability_evaluated` without introducing a competing
status.

Each of the three rounds runs a bounded chain:

```text
quasi-real teacher-following
  -> quasi-real PPO collector
  -> limited quasi-real PPO update
  -> compatibility diagnosis
  -> accounting audit
  -> long-horizon teacher-skill contract
```

The loop allows the limited quasi-real update wrapper to retain the known strict
generated raw takeover failure only when the per-round compatibility,
accounting, quasi-real gates, and long-horizon contract all pass. Validation/test
split transitions, fallback sources, unsafe disagreements, and non-empty gate
reasons remain diagnostic-only. The drift guard requires finite optimizer
metrics, old `log_prob`/`value` reconstruction error no larger than `1e-4`,
`abs(approx_kl)<=0.25`, `max_grad_norm_after_clip<=1.0`, and cumulative
parameter L2 delta no larger than `0.05`.

This is still a local experimental stability check: no formal PPO rollout, no
checkpoint publication, no default-policy replacement, no network/action-space
or default-A* change, no distance/path-risk/source-selection relaxation, no
Ackermann-feasible trajectory claim, and no formal training-ready claim.

### Quasi-Real Guarded PPO Iterative Mini-Loop Evidence Freeze

The current mini-loop result is now frozen as a local baseline before any larger
PPO stage. The freeze does not rerun training. It packages the passed
`Quasi-Real Guarded PPO Iterative Mini-Loop Stability v1` evidence, progress
telemetry, readiness result, docs, and tests into a SHA256 manifest:

- `configs/quasi_real_guarded_ppo_iterative_miniloop_evidence_freeze_v1.json`
- `scripts/run_quasi_real_guarded_ppo_iterative_miniloop_evidence_freeze.py/.sh`
- `scripts/run_quasi_real_guarded_ppo_iterative_miniloop_evidence_freeze_closure.sh`
- `outputs/path_feedback_batch_quasi_real_guarded_ppo_iterative_miniloop_evidence_freeze_v1/`

The freeze summary is
`quasi-real-guarded-ppo-iterative-miniloop-evidence-freeze-summary.json`; the
manifest is
`quasi-real-guarded-ppo-iterative-miniloop-evidence-manifest.json`. Passing
freeze requires the mini-loop summary to remain `passed`, readiness to remain
`quasi_real_guarded_ppo_iterative_miniloop_stability_evaluated`, 684 trainable
contexts, three seeds, three iterations, nine passed iteration/progress rows,
zero controlled regression, zero behavior drift, and no checkpoint publication
or formal-training claim.

This is a provenance checkpoint: it makes the current evidence package easy to
replay and compare against. It is still not formal PPO training and does not
relax any safety, distance/path-risk, source-selection, network, action-space,
or default-A* boundary.

### Quasi-Real Guarded Formal PPO Preflight

`Quasi-Real Guarded Formal PPO Preflight v1` is the formal PPO admission
precheck after the frozen mini-loop baseline. It still does not start formal PPO
rollout. It asks whether the frozen 684 train split, gate-clean quasi-real
transitions can survive three seed-level full-batch PPO smoke updates with the
same teacher-skill and controlled-gate accounting intact.

New artifacts:

- `configs/quasi_real_guarded_formal_ppo_preflight_v1.json`
- `scripts/run_quasi_real_guarded_formal_ppo_preflight.py/.sh`
- `scripts/run_quasi_real_guarded_formal_ppo_preflight_closure.sh`
- `tests/test_quasi_real_guarded_formal_ppo_preflight.py`
- `docs/superpowers/specs/2026-06-14-quasi-real-guarded-formal-ppo-preflight.md`
- `outputs/path_feedback_batch_quasi_real_guarded_formal_ppo_preflight_v1/`

The current closure passes: `status=passed`, `reason_codes=[]`, 684 input
trainable transitions, 684 optimizer transitions, 684 unique trainable contexts,
three seeds, three passed seed smokes, `teacher_agreement_rate=1.0`, old
`log_prob/value` reconstruction error `0.0/0.0`, max `abs(approx_kl)` about
`1.98e-5`, clipped grad norm at or below `1.0`, and controlled regression count
0. Readiness now accepts
`--quasi-real-guarded-formal-ppo-preflight-summary` and advances to
`quasi_real_guarded_formal_ppo_preflight_evaluated` only for a passed summary
with no publication, default-policy replacement, performance, or formal-ready
claim.

This is a release-style preflight for formal PPO, not formal PPO itself. It
does not publish a checkpoint, replace the default policy, change network/action
space/default A*, relax distance/path-risk/source-selection gates, claim
Ackermann-feasible trajectories, or promote IRIS/GCS/path-planner diagnostics to
training release evidence.

### Quasi-Real Guarded Formal PPO Rollout Canary

`Quasi-Real Guarded Formal PPO Rollout Canary v1` is the next step after the
formal preflight. The preflight was the bench inspection; the canary is a
low-speed guarded road test. It uses the same 684 train split, gate-clean
quasi-real transitions, runs conservative multi-seed PPO canary updates, then
audits teacher agreement, validation/test isolation, controlled regression,
KL, gradients, and rollback metadata before readiness can move forward.

New artifacts:

- `configs/quasi_real_guarded_formal_ppo_rollout_canary_v1.json`
- `scripts/run_quasi_real_guarded_formal_ppo_rollout_canary.py/.sh`
- `scripts/run_quasi_real_guarded_formal_ppo_rollout_canary_closure.sh`
- `tests/test_quasi_real_guarded_formal_ppo_rollout_canary.py`
- `docs/superpowers/specs/2026-06-14-quasi-real-guarded-formal-ppo-rollout-canary.md`
- `outputs/path_feedback_batch_quasi_real_guarded_formal_ppo_rollout_canary_v1/`

The canary summary writes seed summaries, training curves, a progress JSONL,
gate audit, rollback manifest, validate-only readiness output, and a markdown
report. Readiness now accepts
`--quasi-real-guarded-formal-ppo-rollout-canary-summary` and only advances to
`quasi_real_guarded_formal_ppo_rollout_canary_evaluated` when the canary passes
with 684 optimizer transitions, three passed seeds, teacher agreement at least
0.95, zero controlled regression, zero diagnostic/fallback trainable leakage,
finite reward/return/advantage/loss/gradient values, bounded KL and clipped
grad norm, and no publication or formal-ready claim.

This stage still keeps every release guard in place. The canary outputs remain
experimental, do not replace the default policy, do not publish a checkpoint,
do not change network/action space/default A*, do not relax distance/path-risk
or source-selection gates, and do not claim policy performance or formal
training readiness.

### Quasi-Real Guarded Formal PPO Stability & Holdout Validation

`Quasi-Real Guarded Formal PPO Stability & Holdout Validation v1` is the
endurance-and-unfamiliar-road check after the canary. The canary proved one
guarded formal PPO road test can stay clean; this stage repeats the experiment
across a seed/budget matrix and keeps validation/test as holdout diagnostics,
so a lucky single seed cannot masquerade as stable training evidence.

New artifacts:

- `configs/quasi_real_guarded_formal_ppo_stability_holdout_validation_v1.json`
- `scripts/run_quasi_real_guarded_formal_ppo_stability_holdout_validation.py/.sh`
- `scripts/run_quasi_real_guarded_formal_ppo_stability_holdout_validation_closure.sh`
- `tests/test_quasi_real_guarded_formal_ppo_stability_holdout_validation.py`
- `docs/superpowers/specs/2026-06-14-quasi-real-guarded-formal-ppo-stability-holdout-validation.md`
- `outputs/path_feedback_batch_quasi_real_guarded_formal_ppo_stability_holdout_validation_v1/`

The current closure passes with `status=passed`, `reason_codes=[]`, 684 input
trainable transitions, 684 optimizer transitions, 684 unique trainable
contexts, five seeds, six budget settings, and 30/30 passed seed-budget runs.
The held-out diagnostic accounting reports validation/test trainable leakage 0,
validation/test controlled regression 0, family regression 0, teacher agreement
`1.0`, old `log_prob/value` reconstruction error `0.0/0.0`, max
`abs(approx_kl)` about `2.92e-5`, and max clipped grad norm `1.0`. Readiness now
accepts
`--quasi-real-guarded-formal-ppo-stability-holdout-validation-summary` and
advances to `quasi_real_guarded_formal_ppo_stability_holdout_validated` only
for a passed summary with no blockers.

This is still not a release. It freezes the canary as baseline, writes a
stability matrix, holdout audit, family regression report, rollback manifest,
and validate-only readiness result, but it does not publish a checkpoint,
replace the default policy, change network/action space/default A*, relax
distance/path-risk/source-selection gates, download new raw data, claim
Ackermann-feasible trajectories, promote IRIS/GCS/path-planner diagnostics to
training release evidence, claim policy performance, or claim formal training
ready.

### Formal PPO Candidate Selection & Long-Horizon Holdout

`Formal PPO Candidate Selection & Long-Horizon Holdout v1` follows the passed
stability matrix. The stability stage put 30 seed/budget candidates on the
inspection line; this stage picks one auditable experimental candidate by a
deterministic multi-gate rule, then sends it through a longer horizon-10 holdout
audit. It does not run a new PPO update.

New artifacts:

- `configs/quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout_v1.json`
- `scripts/run_quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout.py/.sh`
- `scripts/run_quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout_closure.sh`
- `tests/test_quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout.py`
- `docs/superpowers/specs/2026-06-14-quasi-real-guarded-formal-ppo-candidate-selection-long-horizon-holdout.md`
- `outputs/path_feedback_batch_quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout_v1/`

The current closure passes with `status=passed`, `reason_codes=[]`, selected
candidate seed `0` and budget `epochs1_lr3e-6`, `eligible_candidate_count=30`,
`horizon=10`, 684 long-horizon steps, 68 complete horizon-10 episodes, 4 tail
steps, controlled regression 0, family regression 0, and
`teacher_agreement_rate=1.0`. Readiness now accepts
`--quasi-real-guarded-formal-ppo-candidate-selection-long-horizon-holdout-summary`
and advances to
`quasi_real_guarded_formal_ppo_candidate_selection_long_horizon_holdout_evaluated`
only for a passed summary with no blockers.

This is candidate selection and long-horizon diagnostic holdout, not a release
or formal-training-ready claim. It writes a selection audit, holdout steps and
episodes, return audit, split/family reports, selected-candidate manifest,
rollback manifest, validate-only readiness result, and markdown report. It does
not publish a checkpoint, replace the default policy, run another PPO update,
change network/action space/default A*, relax distance/path-risk/source-selection
gates, download new raw data, claim Ackermann-feasible trajectories, or promote
IRIS/GCS/path-planner diagnostics to training release evidence.

### Selected Formal PPO Candidate Multi-Horizon Shadow Rollout

`Selected Formal PPO Candidate Multi-Horizon Shadow Rollout v1` follows the
passed candidate-selection holdout. The previous stage picked the auditable
experimental candidate; this stage keeps that candidate frozen and runs a
read-only shadow road test at horizons 10, 20, and 30. It is still not another
PPO update.

New artifacts:

- `configs/selected_formal_ppo_candidate_multihorizon_shadow_rollout_v1.json`
- `scripts/run_selected_formal_ppo_candidate_multihorizon_shadow_rollout.py/.sh`
- `scripts/run_selected_formal_ppo_candidate_multihorizon_shadow_rollout_closure.sh`
- `tests/test_selected_formal_ppo_candidate_multihorizon_shadow_rollout.py`
- `docs/superpowers/specs/2026-06-14-selected-formal-ppo-candidate-multihorizon-shadow-rollout.md`
- `outputs/path_feedback_batch_selected_formal_ppo_candidate_multihorizon_shadow_rollout_v1/`

The current closure passes with `status=passed`, `reason_codes=[]`, horizons
`[10,20,30]`, 684 unique trainable contexts, and 2052 shadow trainable
transition records across the three horizon views. Complete episode counts are
68 for horizon 10, 34 for horizon 20, and 22 for horizon 30. Controlled
regression and family regression remain 0, `teacher_agreement_rate=1.0`, and
readiness advances to
`selected_formal_ppo_candidate_multihorizon_shadow_rollout_evaluated`.

This is a shadow rollout, not a release or formal-training-ready claim. It
writes multi-horizon episodes and steps, return audit, rejection report,
family report, validate-only readiness result, and markdown report. It does not
publish a checkpoint, replace the default policy, run another PPO update, change
network/action space/default A*, relax distance/path-risk/source-selection
gates, download new raw data, claim Ackermann-feasible trajectories, or promote
IRIS/GCS/path-planner diagnostics to training release evidence.

### Selected Formal PPO Candidate Promotion Preflight

`Selected Formal PPO Candidate Promotion Preflight v1` follows the passed
multi-horizon shadow rollout. The selected experimental candidate has already
survived the longer read-only road test; this stage is the registration-desk
inspection before any later promotion decision. It verifies that the checkpoint,
metadata, hash, load/inference path, rollback boundary, and shadow-evidence
lineage are coherent on the current HEAD.

New artifacts:

- `configs/selected_formal_ppo_candidate_promotion_preflight_v1.json`
- `scripts/run_selected_formal_ppo_candidate_promotion_preflight.py/.sh`
- `scripts/run_selected_formal_ppo_candidate_promotion_preflight_closure.sh`
- `tests/test_selected_formal_ppo_candidate_promotion_preflight.py`
- `docs/superpowers/specs/2026-06-14-selected-formal-ppo-candidate-promotion-preflight.md`
- `outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_preflight_v1/`

The preflight reads the selected candidate root from
`multihorizon-shadow-rollout-summary.json`, audits
`experimental-hybrid-policy-candidate.pt` and its metadata, records a checkpoint
SHA-256 and size, samples at least 64 multi-horizon shadow observations, reloads
the policy network, and reconstructs finite logits, log-probabilities, and
values. It writes a current-HEAD promotion manifest, checkpoint hash audit,
load/inference audit, rollback audit, readiness validate-only result, and
markdown report. Readiness accepts
`--selected-formal-ppo-candidate-promotion-preflight-summary` and advances to
`selected_formal_ppo_candidate_promotion_preflight_evaluated` only for a passed
summary with no blockers.

Clean-HEAD freeze note: the promotion preflight is frozen only after the stage
code, tests, and docs are committed, the closure is rerun from that clean HEAD,
and readiness is refreshed against the regenerated summary. The frozen evidence
covers checkpoint load, inference, rollback, and readiness audits, but it is
still not a checkpoint release or formal-training-ready claim.

This is still not a release or formal-training-ready claim. It does not publish
a checkpoint, replace the default policy, run another PPO update, change
network/action space/default A*, relax distance/path-risk/source-selection
gates, download new raw data, claim Ackermann-feasible trajectories, or promote
IRIS/GCS/path-planner diagnostics to training release evidence.

### Selected Formal PPO Candidate Promotion Decision Review

`Selected Formal PPO Candidate Promotion Decision Review v1` follows the frozen
promotion preflight. If the preflight is the registration-desk inspection, this
stage is the signing desk: it checks that the evidence chain, checkpoint
identity, and release boundary all agree before deciding whether the selected
experimental candidate may enter guarded release-candidate packaging.

New artifacts:

- `configs/selected_formal_ppo_candidate_promotion_decision_review_v1.json`
- `scripts/run_selected_formal_ppo_candidate_promotion_decision_review.py/.sh`
- `scripts/run_selected_formal_ppo_candidate_promotion_decision_review_closure.sh`
- `tests/test_selected_formal_ppo_candidate_promotion_decision_review.py`
- `docs/superpowers/specs/2026-06-14-selected-formal-ppo-candidate-promotion-decision-review.md`
- `outputs/path_feedback_batch_selected_formal_ppo_candidate_promotion_decision_review_v1/`

The review follows four source summaries: promotion preflight,
multi-horizon shadow rollout, candidate-selection long-horizon holdout, and
formal stability holdout validation. It recomputes checkpoint SHA-256/size,
checks metadata and manifest consistency, and writes an evidence lineage report,
checkpoint identity audit, release-boundary audit, readiness validate-only
result, and markdown report. Readiness accepts
`--selected-formal-ppo-candidate-promotion-decision-review-summary` and advances
to `selected_formal_ppo_candidate_promotion_decision_review_evaluated`.

Current closure result: `status=passed`, `reason_codes=[]`,
`decision_verdict=eligible_for_guarded_release_candidate_packaging`,
lineage/checkpoint/release-boundary audits all passed, and checkpoint SHA-256 is
`9d9539c685ab965739c91958bf9cbfe90329c460b4bdbcc35881875aa62f0aa2`.

This is a decision review, not a release. It does not publish a checkpoint,
replace the default policy, run another PPO update, change network/action
space/default A*, relax distance/path-risk/source-selection gates, download new
raw data, claim Ackermann-feasible trajectories, promote IRIS/GCS/path-planner
diagnostics to training release evidence, or claim formal training readiness.

### Guarded Experimental Policy Release Candidate Packaging

`Guarded Experimental Policy Release Candidate Packaging v1` follows the passed
promotion decision review. If the decision review is the signing desk, this
stage is the sealed evidence bag: it copies the selected experimental checkpoint
into an isolated release-candidate package, writes the manifest, and proves that
the packaged file is byte-for-byte the same candidate before any install or
canary dry-run is allowed.

New artifacts:

- `configs/guarded_experimental_policy_release_candidate_packaging_v1.json`
- `scripts/run_guarded_experimental_policy_release_candidate_packaging.py/.sh`
- `scripts/run_guarded_experimental_policy_release_candidate_packaging_closure.sh`
- `tests/test_guarded_experimental_policy_release_candidate_packaging.py`
- `docs/superpowers/specs/2026-06-14-guarded-experimental-policy-release-candidate-packaging.md`
- `outputs/path_feedback_batch_guarded_experimental_policy_release_candidate_packaging_v1/`

The packager reads the decision review summary, resolves the selected candidate
root, checkpoint, metadata, and source lineage, then writes
`release-candidate-package-manifest.json`, `checkpoint-hash-audit.json`,
`checkpoint-load-audit.json`, `rollback-audit.json`, a readiness validate-only
result, and a markdown report. The load audit reuses multi-horizon observation
samples and checks finite logits, log-probabilities, and values from the packaged
checkpoint. Readiness accepts
`--guarded-experimental-policy-release-candidate-packaging-summary` and advances
to `guarded_experimental_policy_release_candidate_packaging_evaluated`.

Current closure result: `status=passed`, `reason_codes=[]`,
`package_verdict=eligible_for_guarded_install_dry_run`, original and packaged
checkpoint SHA-256 both equal
`9d9539c685ab965739c91958bf9cbfe90329c460b4bdbcc35881875aa62f0aa2`,
`checkpoint_load_sample_count=64`, and rollback audit passed.

This is packaging, not install/canary or release. It does not run another PPO
update, publish a checkpoint, replace the default policy, change network/action
space/default A*, relax distance/path-risk/source-selection gates, download new
raw data, claim Ackermann-feasible trajectories, promote IRIS/GCS/path-planner
diagnostics to training release evidence, or claim formal training readiness.

### Guarded Experimental Policy Install Canary Dry-Run

`Guarded Experimental Policy Install Canary Dry-Run v1` follows the passed
release-candidate package. If packaging is the sealed evidence bag, this stage
is a sandboxed test bench: it points an isolated consumer manifest at the
packaged experimental checkpoint, reloads the model through that sandbox path,
and runs a guarded 64-step canary replay without touching the default policy.

New artifacts:

- `configs/guarded_experimental_policy_install_canary_dry_run_v1.json`
- `scripts/run_guarded_experimental_policy_install_canary_dry_run.py/.sh`
- `scripts/run_guarded_experimental_policy_install_canary_dry_run_closure.sh`
- `tests/test_guarded_experimental_policy_install_canary_dry_run.py`
- `docs/superpowers/specs/2026-06-14-guarded-experimental-policy-install-canary-dry-run.md`
- `outputs/path_feedback_batch_guarded_experimental_policy_install_canary_dry_run_v1/`

The dry-run reads
`guarded-experimental-policy-release-candidate-packaging-summary.json`, checks
the package manifest and checkpoint SHA-256/size, writes an isolated sandbox
manifest, snapshots the default policy boundary, and replays at least 64
multi-horizon holdout steps. The step audit records raw policy action,
controlled action, gate reasons, path/risk deltas, finite log-probability/value
and reward checks, and controlled regression counters. Readiness accepts
`--guarded-experimental-policy-install-canary-dry-run-summary` and advances to
`guarded_experimental_policy_install_canary_dry_run_evaluated`.

Current closure result: `status=passed`, `reason_codes=[]`,
`install_canary_verdict=eligible_for_guarded_shadow_release_trial`,
`canary_step_count=64`, `controlled_regression_count=0`, package and consumer
checkpoint SHA-256 both equal
`9d9539c685ab965739c91958bf9cbfe90329c460b4bdbcc35881875aa62f0aa2`, and the
rollback/default-policy audit passed.

This is still only an install canary dry-run. It does not run another PPO
update, publish a checkpoint, replace the default policy, change network/action
space/default A*, relax distance/path-risk/source-selection gates, download new
raw data, claim Ackermann-feasible trajectories, promote IRIS/GCS/path-planner
diagnostics to training release evidence, or claim formal training readiness.
The next stage is a guarded shadow release trial.

## Return-Aligned Guarded Multi-Step PPO Collector

`Return-Aligned Guarded Multi-Step PPO Collector Expansion v1` upgrades the
guarded PPO evidence from clean individual steps to auditable multi-step return
episodes. It does not run another PPO update. It reads the passed guarded pilot
collector transitions, groups them into the configured horizon, and writes a
separate return-aligned collector package under
`outputs/path_feedback_batch_return_aligned_guarded_multi_step_ppo_collector_expansion_v1/`.

New artifacts:

- `configs/return_aligned_guarded_multi_step_ppo_collector_expansion_v1.json`
- `scripts/run_return_aligned_guarded_multi_step_ppo_collector_expansion.py/.sh`
- `scripts/run_return_aligned_guarded_multi_step_ppo_collector_closure.sh`
- `outputs/.../return-aligned-ppo-episodes.jsonl`
- `outputs/.../return-aligned-ppo-transitions.jsonl`
- `outputs/.../return-aligned-reward-audit.json`
- `outputs/.../return-aligned-rejection-report.json`
- `outputs/.../return-aligned-collector-summary.json`

The current closure passes with `horizon=3`, `episode_count=36`,
`step_count=108`, `trainable_episode_count=31`, and
`trainable_transition_count=30`. Step-level PPO trainability remains strict:
only train-split, policy-controlled, gate-clean transitions stay trainable.
Source fallback and other rejected/gated choices remain diagnostic. Episode
trainability is the return-audit layer: a full-horizon train episode with finite
discounted return, no source fallback, and no controlled safety/contract/path
risk/source-selection regression can count as return-aligned evidence.

The reward audit explicitly reports `teacher_following_return`,
`teacher_equivalent_return`, `safe_better_return`,
`controlled_regression_penalty`, `discounted_episode_return`, and
`advantage_reference_value`. Same-as-teacher choices can be positive evidence;
the collector checks multi-step discounted return rather than one-step greedy
improvement. Readiness accepts
`--return-aligned-guarded-multistep-collector-summary` and advances to
`return_aligned_guarded_multistep_collector_evaluated` when the summary passes
with no leakage, no non-finite reward/return/advantage, and no controlled
regression.

This stage is still evidence expansion only: no formal PPO training, no new PPO
update, no checkpoint publication, no default-policy replacement, no
network/action-space/default-A* change, no gate relaxation, no Ackermann
feasible-trajectory claim, and no formal training-ready claim.

### Return-Aligned Guarded PPO Update Smoke

`Return-Aligned Guarded PPO Update Smoke v1` takes the next narrow step after
the multi-step collector. It does not launch formal PPO training. It joins the
return-aligned audit rows back to the original guarded rollout observations,
actions, old log probabilities, and values, then runs one tiny local PPO update
from the same checkpoint that produced the guarded collector evidence.

New artifacts:

- `configs/return_aligned_guarded_ppo_update_smoke_v1.json`
- `scripts/run_return_aligned_guarded_ppo_update_smoke.py/.sh`
- `scripts/run_return_aligned_guarded_ppo_update_smoke_closure.sh`
- `outputs/path_feedback_batch_return_aligned_guarded_ppo_update_smoke_v1/`
- `outputs/.../optimizer-input/ppo-rollout-episodes.jsonl`
- `outputs/.../return-aligned-guarded-ppo-update-smoke-summary.json`

The optimizer input is deliberately narrow. Only `split=train`,
`controlled_choice_source=policy`, `ppo_trainable=true`, gate-clean,
finite-return rows from the return-aligned collector are materialized. The
original guarded rollout supplies the network-facing observation/action and
old `log_prob/value`; the return-aligned collector supplies `ppo_return` and
`ppo_advantage`. This prevents the update from silently falling back to
one-step reward accounting.

Current closure result:

- return-aligned collector input: `status=passed`, `reason_codes=[]`
- optimizer transition count: `30`
- validation/test/source-fallback optimizer count: `0`
- old `log_prob/value` max abs error: `0.0 / 0.0`
- loss/gradient/reward/return/advantage non-finite count: `0`
- `parameter_l2_delta=0.00042781765692363765`
- `approx_kl=-0.0008484522695653141`
- `max_grad_norm_after_clip<=1.0`
- post-update gates evaluated: `true`
- post-update raw generalization effective status: `passed`,
  `post_update_raw_test_regression_count=0`
- post-update generated sequential strict status: `failed` for the known
  diagnostic reason codes
  `multi_step_accepted_episode_count_below_threshold`,
  `family_with_multi_step_accepted_episode_count_below_threshold`, and
  `canary_rejected_policy_choice_count_above_threshold`; this is not counted as
  a controlled rollout regression
- post-update generated collector: `status=passed`,
  `ppo_trainable_transition_count=30`
- post-update quasi-real teacher-following: `status=passed`,
  `teacher_agreement_rate=1.0`
- post-update quasi-real collector: `status=passed`,
  `ppo_trainable_transition_count=36`
- post-update long-horizon contract:
  `verdict=long_horizon_teacher_skill_contract_aligned`
- post-update return-aligned replay: `status=passed`,
  `trainable_transition_count=30`
- post-update controlled regression count: `0`
- readiness accepts `--return-aligned-guarded-ppo-update-smoke-summary`
- readiness status: `return_aligned_guarded_ppo_update_smoke_evaluated`
- `training_blockers=[]`

This is still an experimental smoke only: no checkpoint publication, no
default-policy replacement, no performance claim, no formal training-ready
claim, no network/action-space/default-A* change, no gate relaxation, and no
Ackermann-feasible trajectory claim.

## Guarded Experimental Policy Shadow Release Trial v1

**Guarded Experimental Policy Shadow Release Trial v1** is the next stage after
the install canary dry-run. It is a shadow-only release trial: the packaged
experimental policy is allowed to produce side-channel decisions and gate audit
rows, but the default policy remains the authoritative output.

The runner consumes the install canary summary and the selected formal PPO
candidate multi-horizon shadow rollout. It writes a runtime manifest, per-step
comparison JSONL, rejection report, risk/reward audit, readiness validate-only
result, markdown report, and
`guarded-experimental-policy-shadow-release-trial-summary.json`.

Current closure result:

- summary `status=passed`, `reason_codes=[]`
- `shadow_release_trial_verdict=eligible_for_guarded_staged_release_preflight`
- `shadow_step_count=2052`
- `unique_shadow_context_count=684`
- `controlled_regression_count=0`
- `shadow_rejection_diagnostic_count=0`
- `shadow_fallback_diagnostic_count=0`
- package/consumer checkpoint SHA-256 remains
  `9d9539c685ab965739c91958bf9cbfe90329c460b4bdbcc35881875aa62f0aa2`
- readiness accepts
  `--guarded-experimental-policy-shadow-release-trial-summary`
- readiness status:
  `guarded_experimental_policy_shadow_release_trial_evaluated`
- `training_blockers=[]`

This remains a shadow release trial, not a staged release or default-policy
replacement. It does not run a new PPO update, publish a checkpoint, replace the
default policy, modify network/action space/default A*, relax gates, download
new raw data, claim Ackermann-feasible trajectory, claim performance
improvement, or claim formal training readiness. The next stage is a guarded
staged release preflight.

## Guarded Experimental Policy Staged Release Preflight v1

**Guarded Experimental Policy Staged Release Preflight v1** audits the release
machinery that would be needed before any guarded staged release. It consumes
the shadow release trial summary and writes a preflight manifest, gate-threshold
audit, kill-switch audit, rollback audit, telemetry audit, readiness
validate-only result, markdown report, and
`guarded-experimental-policy-staged-release-preflight-summary.json`.

This stage deliberately keeps the staged release disabled:

- `staged_release_enabled=false`
- `default_policy_authoritative=true`
- `experimental_control_activation_count=0`
- stage plan only: `stage0_shadow_only`, `stage1_guarded_opt_in`,
  `stage2_limited_guarded_canary`

Acceptance requires the source shadow trial to be passed and current, at least
512 shadow steps and 256 unique contexts, no missing observation/log_prob/value,
no invalid mask, no non-finite logits/value/reward, no controlled
safety/contract/path-risk/source-selection regression, and passed kill-switch,
rollback, and telemetry audits. Readiness accepts
`--guarded-experimental-policy-staged-release-preflight-summary` and advances to
`guarded_experimental_policy_staged_release_preflight_evaluated`.

This remains a release preflight, not an actual staged release. It does not let
the experimental policy take control, run a new PPO update, publish a
checkpoint, replace the default policy, modify network/action space/default A*,
relax gates, download new raw data, claim Ackermann-feasible trajectory, claim
performance improvement, or claim formal training readiness.

## Guarded Experimental Policy Staged Release Trial v1

**Guarded Experimental Policy Staged Release Trial v1** is the first local,
offline staged trial after preflight. It is a tightly guarded handoff rehearsal:
the experimental policy may take control only on a small number of gate-clean
shadow rows, while the default policy remains authoritative and every rejected,
fallback, missing, or non-finite row stays diagnostic-only.

The runner consumes the staged-release preflight summary and the shadow release
step comparison, then writes:

- `guarded-experimental-policy-staged-release-trial-summary.json`
- `staged-release-activation-ledger.jsonl`
- `staged-release-controlled-regression-audit.json`
- `staged-release-fallback-rejection-report.json`
- `staged-release-kill-switch-drill.json`
- `staged-release-rollback-drill.json`
- `staged-release-telemetry-drill.json`
- `staged-release-trial-readiness-validate-only.json`
- `staged-release-trial-report.md`

Acceptance requires `staged_release_enabled=true`,
`default_policy_authoritative=true`, `1 <= experimental_control_activation_count
<= 64`, zero diagnostic/fallback/rejected activations, zero controlled
safety/contract/path-risk/source-selection regression, zero activation-level
missing/non-finite/mask failures, and passed kill-switch, rollback, and
telemetry drills. Readiness accepts
`--guarded-experimental-policy-staged-release-trial-summary` and advances to
`guarded_experimental_policy_staged_release_trial_evaluated`.

This is still local/offline release evidence, not a real staged release. It does
not connect to a real executor, run a new PPO update, publish a checkpoint,
replace the default policy, modify network/action space/default A*, relax
gates, download new raw data, claim Ackermann-feasible trajectory, claim
performance improvement, or claim formal training readiness.

## Guarded Experimental Policy Staged Release Canary Preflight v1

**Guarded Experimental Policy Staged Release Canary Preflight v1** is the next
offline checkpoint after the staged trial. It is a canary launch checklist, not
the launch itself: the stage reads the 64 gate-clean trial activations, selects
at most 16 offline canary-eligible candidates, and verifies limit controls,
kill-switch, rollback, telemetry, automatic downgrade, budget, and operator
approval before any real executor can be connected.

The runner consumes the staged-release trial summary and activation ledger, then
writes:

- `guarded-experimental-policy-staged-release-canary-preflight-summary.json`
- `staged-canary-preflight-manifest.json`
- `staged-canary-eligibility-ledger.jsonl`
- `staged-canary-rejection-report.json`
- `staged-canary-kill-switch-audit.json`
- `staged-canary-rollback-audit.json`
- `staged-canary-telemetry-audit.json`
- `staged-canary-automatic-downgrade-audit.json`
- `staged-canary-budget-audit.json`
- `staged-canary-operator-approval-audit.json`
- `staged-canary-preflight-readiness-validate-only.json`
- `staged-canary-preflight-report.md`

Acceptance requires `staged_canary_enabled=false`,
`connects_real_executor=false`, `default_policy_authoritative=true`,
`canary_traffic_fraction<=0.01`, `1 <= canary_eligible_activation_count <= 16`,
zero diagnostic/fallback/rejected/missing/non-finite/control-regression eligible
rows, zero controlled safety/contract/path-risk/source-selection regression, and
passed kill-switch, rollback, telemetry, automatic downgrade, budget, and
operator approval audits. Readiness accepts
`--guarded-experimental-policy-staged-release-canary-preflight-summary` and
advances to
`guarded_experimental_policy_staged_release_canary_preflight_evaluated`.

This remains offline preflight evidence. It does not start an online canary,
connect a real executor, run new PPO, publish a checkpoint, replace the default
policy, modify network/action space/default A*, relax gates, download new raw
data, claim Ackermann-feasible trajectory, treat IRIS/GCS/path-planner
diagnostics as release evidence, claim policy performance improvement, or claim
formal training readiness.

## Guarded Formal PPO Training Authorization v1

**Guarded Formal PPO Training Authorization v1** is the training-side permission
package after the guarded formal PPO evidence chain and staged canary preflight.
It is a launch authorization for a future controlled PPO training run, not the
run itself: it checks the existing formal preflight, rollout canary,
stability/holdout, candidate selection, promotion decision review, and canary
preflight summaries, then writes the allowed training input, seed plan, budget,
stop conditions, rollback plan, and post-training gate plan.

The runner writes:

- `formal-ppo-training-authorization-summary.json`
- `formal-ppo-training-input-audit.json`
- `formal-ppo-training-budget-manifest.json`
- `formal-ppo-training-seed-plan.json`
- `formal-ppo-training-stop-condition-manifest.json`
- `formal-ppo-training-rollback-manifest.json`
- `formal-ppo-post-training-gate-plan.json`
- `formal-ppo-training-authorization-readiness-validate-only.json`
- `formal-ppo-training-authorization-report.md`

Acceptance requires
`authorization_verdict=authorized_for_guarded_formal_ppo_training_run`,
`authorized_trainable_transition_count>=684`, matching optimizer and unique
trainable context counts, seeds `[0,1,2,3,4]`, zero validation/test/fallback or
diagnostic trainable leakage, zero missing/non-finite/invalid-mask counters,
zero controlled safety/contract/path-risk/source-selection regression, and
passed budget, seed, stop-condition, rollback, and post-training gate manifests.
Readiness accepts `--guarded-formal-ppo-training-authorization-summary` and
advances to `guarded_formal_ppo_training_authorized`.

This is still not formal PPO training. It does not run a new PPO update, start
an online canary, connect a real executor, publish or replace a checkpoint,
replace the default policy, modify network/action space/default A*, relax gates,
download new raw data, claim Ackermann-feasible trajectory, treat
IRIS/GCS/path-planner diagnostics as training or release evidence, claim policy
performance improvement, or claim formal training readiness.

## Guarded Formal PPO Training Run v1

**Guarded Formal PPO Training Run v1** is the first authorized local formal PPO
training pass. It consumes the authorization package above, reads the 684
gate-clean train-split quasi-real transitions from the formal stability/holdout
evidence, and runs one small full-batch PPO update for each seed in
`[0,1,2,3,4]`.

The runner writes:

- `formal-ppo-training-run-summary.json`
- `formal-ppo-training-run-seed-summaries.jsonl`
- `formal-ppo-training-run-progress.jsonl`
- `formal-ppo-training-run-training-curves.json`
- `formal-ppo-training-run-gate-audit.json`
- `formal-ppo-training-run-rollback-manifest.json`
- `formal-ppo-training-run-readiness-validate-only.json`
- `formal-ppo-training-run-report.md`

Acceptance requires the input authorization to be passed, 684 optimizer
transitions, zero validation/test/fallback/diagnostic trainable leakage, all
five seeds passed, finite reward/return/advantage/loss/gradient values, old
`log_prob`/value reconstruction error at or below `1e-4`,
`parameter_l2_delta>0`, `abs(approx_kl)<=0.25`,
`max_grad_norm_after_clip<=1.0`, teacher agreement at least `0.95`, zero
controlled safety/contract/path-risk/source-selection regression, and passed
post-training holdout/canary gate status. Readiness accepts
`--guarded-formal-ppo-training-run-summary` and advances to
`guarded_formal_ppo_training_run_evaluated`.

This stage runs a PPO update, but the resulting checkpoints are still
experimental candidates only. It does not start an online canary, connect a real
executor, publish or replace a checkpoint, replace the default policy, modify
network/action space/default A*, relax gates, download new raw data, claim
Ackermann-feasible trajectory, treat IRIS/GCS/path-planner diagnostics as
training or release evidence, or claim deployable policy performance.

## Guarded Formal PPO Post-Training Stability Replay v1

**Guarded Formal PPO Post-Training Stability Replay v1** replays the five
experimental candidates produced by the formal training run. It does not run a
new PPO update. It checks that the already-trained candidates can be replayed
three times per seed with stable gate accounting, no behavior drift, no
holdout/canary regression, and no controlled regression.

The runner writes:

- `formal-ppo-post-training-stability-replay-summary.json`
- `formal-ppo-post-training-stability-replay-seed-summaries.jsonl`
- `formal-ppo-post-training-stability-replay-progress.jsonl`
- `formal-ppo-post-training-stability-replay-drift-report.jsonl`
- `formal-ppo-post-training-stability-replay-gate-audit.json`
- `formal-ppo-post-training-stability-replay-rollback-manifest.json`
- `formal-ppo-post-training-stability-replay-readiness-validate-only.json`
- `formal-ppo-post-training-stability-replay-report.md`

Acceptance requires 5 seeds, at least 3 replays per seed, at least 15 total
replays, all replays passed, zero missing seed checkpoints, zero replay behavior
drift, 684 optimizer transitions, replay collector trainable count at least
684, zero validation/test/fallback/diagnostic trainable leakage, zero missing
observation/log-prob/value counters, finite reward/return/advantage values,
teacher agreement at least `0.95`, zero controlled safety/contract/path-risk/
source-selection regression, and passed holdout/canary replay gates. Readiness
accepts `--guarded-formal-ppo-post-training-stability-replay-summary` and
advances to `guarded_formal_ppo_post_training_stability_replay_evaluated`.

This stage is replay validation, not release. It does not run another PPO
update, start an online canary, connect a real executor, publish or replace a
checkpoint, replace the default policy, modify network/action space/default A*,
relax gates, download new data, claim Ackermann-feasible trajectory, treat
IRIS/GCS/path-planner diagnostics as release evidence, or claim deployable
policy performance.

## Post-Formal PPO Coverage-First Advancement Chain

Current evidence proves that guarded PPO training can run and remain stable. It
does not yet prove exploration-performance improvement. The formal training
signal remains teacher-following and guard-focused, while `Connect Real
Exploration Coverage Signal v1` now makes the coverage telemetry usable enough
to move from signal audit into coverage-performance evaluation. The project
direction remains coverage-first, not another release-packaging step.

The advancement chain is:

```text
0. Guarded formal PPO training stability baseline
1. Post-training candidate selection / reconciliation
2. Exploration Coverage Signal Audit
3. Exploration Coverage Performance Evaluation
4. Coverage-Aware Reward Refinement
5. Coverage-Driven PPO Improvement Run
5A. Policy Coverage Opportunity / Margin Audit, only if Stage 5 does not improve
5A.1 Candidate-Level Exploration Coverage Materialization
5A.2 Policy-Differentiating Counterfactual Coverage Rollouts
5B. Reward-vs-data repair decision: rerun refined PPO or expand coverage source generation
5B.1 Refined Coverage-Driven PPO Improvement Run
5B.2 Refined Coverage Reward/Margin Tuning
5B.3 Safe-Better Pair Expansion Across Families
5B.4 Low-Observation Counterfactual Opportunity Generation
5B.5 Low-Observation Candidate Geometry Improvement
5B.6 Expanded Four-Family Refined Coverage-Driven PPO Improvement
5B.7 Cost-Efficiency-Aware Coverage Reward / Candidate Filter
6. Shadow / Canary Release Performance Validation
7. Formal Performance Claim / Release Decision
8. Scoped Claim Publication & Evidence Freeze
9. Checkpoint Publication Authorization Preflight
10. Checkpoint Publication Package Preparation
11. Checkpoint Publication Package Verification
12. Checkpoint Publication Sandbox Install Dry-Run Preflight
13. Checkpoint Publication Sandbox Install Dry-Run
14. Checkpoint Publication Sandbox Install Dry-Run Verification
```

Stage 1 must reconcile which selected candidate is authoritative. Existing
promotion evidence selects seed `0`, budget `epochs1_lr3e-6`, with checkpoint
SHA-256 `9d9539c685ab965739c91958bf9cbfe90329c460b4bdbcc35881875aa62f0aa2`,
but any new run rooted at
`outputs/path_feedback_batch_guarded_formal_ppo_training_run_v1/` must explicitly
select from that run's five seeds before evaluation.

Stage 2 verifies that coverage telemetry is real before the project switches
back from teacher-following reward toward the closed-loop reward family based on
coverage gain, path cost, risk, and failure penalties. The audit must check
`initial_coverage_rate`,
`final_coverage_rate`, `coverage_rate_delta`,
`cumulative_coverage_rate_delta`, `expected_coverage_rate_delta`, actual map or
sidecar-backed coverage gain, multi-step state updates, and policy/source/
fallback attribution. If the evidence is all zero, missing, defaulted, or not
attributable, the stage must fail with
`insufficient_exploration_coverage_signal`.

`Exploration Coverage Signal Audit v1` is now implemented by
`scripts/run_exploration_coverage_signal_audit.py` and
`scripts/run_exploration_coverage_signal_audit.sh`. `Connect Real Exploration
Coverage Signal v1` also wires the upstream path-feedback signal through
`scripts/run_quasi_real_trainable_context_expansion.py` and
`scripts/run_quasi_real_guarded_ppo_scale512_multiseed_preflight.py`, so
materialized trainable rows and PPO collector episodes can carry
`initial_coverage_rate`, `final_coverage_rate`, `coverage_rate_delta`,
`cumulative_coverage_rate_delta`, and `coverage_gain_source` instead of default
zeros. For legacy selected-candidate shadow artifacts that were produced before
this wiring, the audit now performs a read-only backfill from existing
path-feedback summaries and rolls those actual deltas into a continuous
multi-step coverage state.

The current audit output at
`outputs/path_feedback_batch_exploration_coverage_signal_audit_v1/` now passes
the signal gate: `status=passed`, `coverage_signal_status=passed`,
`coverage_field_presence_status=passed`,
`actual_coverage_gain_source=path_feedback`,
`nonzero_actual_coverage_delta_count=2052`,
`coverage_delta_default_zero_count=0`,
`expected_actual_coverage_confusion_count=0`,
`multi_step_state_update_verified=true`,
`fallback_coverage_gain_claimed_as_policy_gain_count=0`, and
`controlled_regression_count=0`. This advances the next required change to
`run_exploration_coverage_performance_evaluation`.

Stage 3 is now implemented by
`scripts/run_exploration_coverage_performance_evaluation.py` and
`scripts/run_exploration_coverage_performance_evaluation.sh`, with outputs in
`outputs/path_feedback_batch_exploration_coverage_performance_evaluation_v1/`.
It compares coverage return, cumulative coverage delta, final coverage, valuable
coverage, path-cost/risk/energy efficiency, activation, fallback, teacher
agreement, and controlled regression using the real delta rows from Stage 2.

The current Stage 3 result is intentionally strict:
`status=failed`, `coverage_performance_status=failed`, and
`reason_codes=["coverage_performance_not_improved",
"valuable_coverage_not_improved"]`. The selected seed remains `0` with budget
`epochs1_lr3e-6`; the best available baseline is teacher, derived only as a
`teacher_action_equivalent_from_policy_teacher_aligned_shadow` comparator because
the selected candidate took the same guarded actions as the teacher. Selected PPO
and teacher both have `coverage_return=81.582958126732`,
`cumulative_coverage_rate_delta=89.45694198338`, `valuable_area_covered=0.0`,
`coverage_gain_per_path_cost=0.06605991614`,
`coverage_gain_per_risk=0.216170916939`, `fallback_rate=0.0`, and
`controlled_regression_count=0`. This means the coverage instrument works, but
the PPO candidate has not yet beaten teacher on exploration coverage.

Because Stage 3 did not pass, the next required change is
`coverage_aware_reward_refinement`: improve the reward so the policy is still
guarded by teacher skill but is actually scored on multi-step exploration
coverage gain, valuable coverage, and efficiency. This is still not a release or
performance claim. The intended reward direction is:

```text
reward =
  coverage_gain_bonus
+ valuable_area_bonus
+ information_gain_bonus
+ teacher_skill_retention_bonus
- path_cost_penalty
- risk_penalty
- energy_penalty
- fallback_penalty
- controlled_regression_penalty
```

Stage 4 is now implemented by
`scripts/run_coverage_aware_reward_refinement.py` and
`scripts/run_coverage_aware_reward_refinement.sh`, with outputs in
`outputs/path_feedback_batch_coverage_aware_reward_refinement_v1/`. It joins the
Stage 2 coverage delta audit, selected-candidate shadow steps, and Stage 3
comparison output, then writes `reward-component-audit.jsonl`,
`source-field-audit.json`, `reward-rescore-comparison.json`,
`reward-refinement-rejection-report.json`, and a markdown report.

`Connect Reward Component Source Fields v1` is now implemented by
`scripts/run_connect_reward_component_source_fields.py` and
`scripts/run_connect_reward_component_source_fields.sh`, with outputs in
`outputs/path_feedback_batch_connect_reward_component_source_fields_v1/`. It
does not mutate Stage 2 artifacts; it writes a provenance overlay that connects
2,052/2,052 rows to real `path_feedback` sources. Matching uses direct
`context_id` for 108 rows and `scenario_id + controlled_action_index` for
1,944 rows. The connected sources are
`path_feedback.coverage_rate_delta*path_feedback.candidates.utility` for
`valuable_area_bonus`, `path_feedback.coverage_rate_delta` for
`information_gain_bonus`, and `path_feedback.candidates.risk` for
`risk_penalty`.

After rerunning Stage 4 with that overlay, the current reward refinement output
passes the source-field gate: `status=passed`,
`reward_refinement_status=passed`, `reason_codes=[]`,
`source_field_missing_component_count=0`, and
`component_source_overlay_row_count=2052`. The read-only boundary remains intact:
`runs_new_ppo_update=false`, `publishes_checkpoint=false`,
`replaces_default_policy=false`, `performance_claimed=false`, and
`formal_release_claimed=false`. The selected candidate is still
teacher-equivalent (`selected_teacher_equivalent=true`,
`selected_candidate_performance_improved=false`,
`coverage_aware_reward_improvement=0.0`), so this is a reward-contract pass, not
a PPO performance win.

Stage 5 is now implemented by
`scripts/run_coverage_driven_ppo_improvement_run.py` and
`scripts/run_coverage_driven_ppo_improvement_run.sh`, with outputs in
`outputs/path_feedback_batch_coverage_driven_ppo_improvement_run_v1/`. It
materializes a coverage-aware PPO collector from Stage 4
`reward-component-audit.jsonl`, preserving observations, action masks,
controlled actions, old `log_prob`/`value`, reward components, and provenance.
It then runs one guarded offline PPO update and writes an experimental
checkpoint only for offline evaluation.

The current Stage 5 run is a clean negative result:
`status=failed`,
`coverage_driven_ppo_improvement_status=failed`, and
`reason_codes=["no_coverage_return_improvement"]`. The update consumed
2,052 coverage-aware reward transitions, wrote
`coverage-driven-experimental-policy-candidate.pt`, and changed parameters
(`parameter_l2_delta=0.0004370135471177026`) with finite old-policy checks
(`old_log_prob_max_abs_error=0.0017986297607421875`,
`old_value_max_abs_error=0.002626180648803711`). Guard replay passed on all
2,052 audited rows: `accepted_policy_activation_rate=1.0`,
`fallback_rate=0.0`, `teacher_agreement_rate=1.0`, and
`controlled_regression_count=0`.

That still is not a performance win. The post-update policy remains equivalent
to the frozen pre-improvement selected PPO on coverage return:
`coverage_return=81.582958126732`,
`cumulative_coverage_rate_delta=89.45694198338`, so
`coverage_return_improvement=0.0` and
`cumulative_coverage_rate_delta_improvement=0.0`. Valuable coverage is now
auditable (`valuable_area_covered_improvement=45.824562342498`), but the main
coverage-return gate did not improve. The next required change is therefore
`refine_coverage_reward_or_collect_more_policy_coverage`, not release,
publication, or a performance claim.

The Stage 5 failure now has a concrete diagnostic chain. Replay shows that the
post-update policy did not merely get blocked by the guard: it selected the same
action as the teacher on all 2,052 audited rows
(`raw_policy_action_index == teacher_action_index` and
`controlled_action_index == teacher_action_index` for 2,052/2,052 rows, with
`guard_rejected_action_count=0`). The PPO step changed parameters
(`parameter_l2_delta=0.0004370135471177026`) but did not move the final action
choice away from teacher-equivalent behavior.

Stage 5A is now implemented by
`scripts/run_policy_coverage_opportunity_margin_audit.py` and
`scripts/run_policy_coverage_opportunity_margin_audit.sh`, with outputs in
`outputs/path_feedback_batch_policy_coverage_opportunity_margin_audit_v1/`. It
does not run PPO. It expands the frozen Stage 5 audited data into action-level
candidate rows, scores the pre-improvement selected checkpoint and post-update
experimental checkpoint on the same observations, and writes a summary,
action-level audit JSONL, policy-margin audit, candidate-feature audit,
rejection report, and markdown report.

The current Stage 5A diagnostic passes as a diagnostic but routes to data
repair, not reward tuning: `status=passed`,
`next_required_change=collect_more_policy_differentiating_coverage`, and
`reason_codes=["candidate_coverage_features_missing",
"no_safe_better_than_teacher_alternatives",
"post_update_policy_teacher_equivalent"]`. It audited 2,052 contexts and 5,508
action candidates. The key counts are `safe_better_than_teacher_count=0`,
`safe_better_than_teacher_family_count=0`, `policy_argmax_changed_count=0`,
`post_update_teacher_equal_raw_count=2052`,
`post_update_teacher_equal_controlled_count=2052`,
`teacher_margin_sample_count=2052`, `missing_teacher_signal_count=0`,
`candidate_expected_coverage_nonzero_count=0`,
`candidate_information_gain_nonzero_count=0`,
`candidate_value_nonzero_count=0`, `fallback_gain_contamination_count=0`, and
`controlled_regression_count=0`.

This separates the Stage 5 failure cleanly. The guard is not blocking better
actions, and the PPO update did not change action argmax. More importantly, the
policy still sees no decision-time coverage direction: every audited candidate
has zero `expected_coverage_rate_delta`, zero `information_gain`, and zero
`value`. Actual `coverage_rate_delta` exists only after the selected action is
executed, so the reward can score the taken action after the fact but cannot yet
teach the policy which safe non-teacher candidate should have produced higher
coverage. The next required change is therefore to collect or materialize
policy-differentiating coverage evidence: counterfactual candidate coverage,
nonzero decision-time expected coverage/information/value fields, and explicit
safe-alternative labels.

Stage 5B.1 has now exercised that decision: the data gap is closed well enough
to run refined PPO, but the policy still does not improve coverage return. The
next gate is therefore reward/margin tuning or expanding the safe-better pair
set, followed by another refined run. Any pass still requires positive
`coverage_return_improvement`, positive
`cumulative_coverage_rate_delta_improvement`, positive valuable coverage, no
efficiency regression, `controlled_regression_count=0`, and no fallback gain
contamination.

The Stage 5A diagnostic contract and current result are documented in
`docs/superpowers/specs/2026-06-15-policy-coverage-opportunity-margin-audit.md`.
The Stage 5A.1 materialization contract and current result are documented in
`docs/superpowers/specs/2026-06-15-candidate-level-coverage-materialization.md`.
The Stage 5A.2 counterfactual rollout contract and current result are documented
in
`docs/superpowers/specs/2026-06-15-policy-differentiating-counterfactual-coverage-rollouts.md`.
The Stage 5B.1 refined PPO contract and current failed performance result are
documented in
`docs/superpowers/specs/2026-06-15-refined-coverage-driven-ppo-improvement-run.md`.
The Stage 5B.2 reward/margin tuning contract and current failed performance
result are documented in
`docs/superpowers/specs/2026-06-15-refined-coverage-reward-margin-tuning.md`.

Stage 5A.1 is now implemented by
`scripts/run_candidate_level_coverage_materialization.py` and
`scripts/run_candidate_level_coverage_materialization.sh`, with outputs in
`outputs/path_feedback_batch_candidate_level_coverage_materialization_v1/`. It
does not run PPO. It materializes a candidate-level coverage overlay from the
frozen Stage 5A action audit plus available path-feedback/sidecar evidence, then
reruns Stage 5A with `--candidate-coverage-overlay` to prove whether the policy
can see real decision-time coverage fields.

The current Stage 5A.1 result is a clean negative materialization result:
`status=failed`,
`next_required_change=generate_policy_differentiating_coverage_rollouts`, and
`reason_codes=["candidate_coverage_source_missing",
"counterfactual_candidate_coverage_source_missing",
"no_safe_better_than_teacher_candidate"]`. It audited 2,052 decision contexts
and 5,508 action candidates. The overlay connects 2,052 executed selected/teacher
actions to real path-feedback coverage
(`executed_action_actual_path_feedback`), but 3,456 non-teacher candidate rows
only connect to path/risk/cost source rows without counterfactual coverage
(`candidate_source_without_coverage`). As a result,
`candidate_expected_coverage_nonzero_count=2052`,
`candidate_information_gain_nonzero_count=2052`, `candidate_value_nonzero_count=0`,
`counterfactual_coverage_candidate_count=0`,
`safe_better_than_teacher_candidate_count=0`,
`fallback_gain_contamination_count=0`, and `controlled_regression_count=0`.

The overlay-fed Stage 5A rerun passes as a diagnostic, but still routes to
`collect_more_policy_differentiating_coverage`: it now sees nonzero coverage for
executed teacher-equivalent actions, not for safe non-teacher alternatives. This
means the next bridge is not another PPO update; it is to generate rollout
evidence that records candidate-level counterfactual exploration coverage for
the missing families (`smooth_high_confidence`, `rim_or_steep_slope`,
`low_observation_count`, and `mixed_risk`).

Stage 5A.2 is now implemented by
`scripts/run_policy_differentiating_counterfactual_coverage_rollouts.py` and
`scripts/run_policy_differentiating_counterfactual_coverage_rollouts.sh`, with
outputs in
`outputs/path_feedback_batch_policy_differentiating_counterfactual_coverage_rollouts_v1/`.
It does not run PPO. It reads the Stage 5A action audit, the Stage 5A.1 missing
overlay/source-link audit, the quasi-real path-feedback summary, and the
path-planner sidecars, then computes non-teacher counterfactual coverage from
candidate `diagnostics.expanded_cells`, passable sidecar cells, terrain
confidence, and prior executed coverage state. Missing evidence remains missing;
the script does not fill source gaps with zero.

The current Stage 5A.2 result passes as a routing/data stage:
`status=passed`, `reason_codes=[]`, and
`next_required_change=rerun_coverage_driven_ppo_with_refined_reward_or_advantage`.
It consumed the 3,456 non-teacher candidate rows that Stage 5A.1 could not score
and generated 3,456 counterfactual coverage rows with
`match_method=counterfactual_candidate_expanded_cells_sidecar`. The rerun
overlay has 5,508 rows, Stage 5A.1 rerun passed, and overlay-fed Stage 5A passed.
Key counts are `candidate_expected_coverage_nonzero_count=3690`,
`candidate_information_gain_nonzero_count=3690`,
`candidate_value_nonzero_count=1638`,
`safe_better_than_teacher_candidate_count=51`,
`safe_better_than_teacher_family_count=2`,
`missing_counterfactual_source_count=0`,
`fallback_gain_contamination_count=0`, and
`controlled_regression_count=0`.

This is still not a performance claim. It means the project has finally produced
decision-time evidence that some safe non-teacher candidates can beat the
teacher on counterfactual exploration coverage. The next engineering move is to
rerun coverage-driven PPO with a refined reward/advantage or margin objective
that can use these safe better alternatives; the checkpoint still must not be
published or installed as the default policy.

Stage 5B.1 is now implemented by
`scripts/run_refined_coverage_driven_ppo_improvement_run.py` and
`scripts/run_refined_coverage_driven_ppo_improvement_run.sh`, with outputs in
`outputs/path_feedback_batch_refined_coverage_driven_ppo_improvement_run_v2/`.
This stage does run a guarded offline PPO update, but only as an experimental
refinement. It converts the 51 Stage 5A.2 safe-better non-teacher candidates
into trainable counterfactual advantage and margin records keyed by
`context_id + episode_id + step_index`, recomputes old `log_prob` and `value`
against the selected base checkpoint, and writes a compatibility
`coverage-aware-ppo-batch` for the existing limited PPO update smoke.

The current Stage 5B.1 run proves the refined signal can be consumed, not that
coverage performance improved. Summary status is `failed` with
`reason_codes=["post_update_policy_teacher_equivalent",
"no_coverage_return_improvement",
"no_cumulative_coverage_rate_delta_improvement", "fallback_dominates"]` and
`next_required_change=tune_refined_reward_margin_or_expand_safe_better_pairs`.
The refined batch has `safe_better_training_pair_count=51`,
`refined_trainable_transition_count=51`, and
`counterfactual_advantage_nonzero_count=51`; the PPO update itself passed with
`optimizer_train_transition_count=51`, `old_log_prob_max_abs_error=0.0`,
`old_value_max_abs_error=0.0`, and `parameter_l2_delta=0.0017301534299004352`.
However `policy_argmax_changed_count=0`, `coverage_return_improvement` and
`cumulative_coverage_rate_delta_improvement` are negative against the frozen
baseline, `fallback_rate=1.0`, and the run keeps
`publishes_checkpoint=false`, `replaces_default_policy=false`, and
`performance_claimed=false`.

Stage 5B.2 is now implemented by
`scripts/run_refined_coverage_reward_margin_tuning.py` and
`scripts/run_refined_coverage_reward_margin_tuning.sh`, with outputs in
`outputs/path_feedback_batch_refined_coverage_reward_margin_tuning_v1/`. It
does not change the PPO network or action space. Instead, it reuses the refined
v2 safe-better batch and the existing limited PPO `transition_info` path to
inject explicit `ppo_advantage` and `ppo_return` fields. Four fixed tuning
configs were evaluated: `advantage_x3`, `advantage_x5`, `advantage_x8`, and
`reward_margin_x5`.

The current Stage 5B.2 result is also a controlled failure, but it narrows the
next blocker. Summary status is `failed` with
`reason_codes=["post_update_policy_teacher_equivalent",
"no_coverage_return_improvement",
"no_cumulative_coverage_rate_delta_improvement", "fallback_dominates"]` and
`next_required_change=expand_safe_better_pair_generation_across_families`.
All 4 configs ran PPO updates over the same 51 safe-better pairs, producing
`ppo_advantage_nonzero_count=204` in aggregate. The best config by current
comparison was `advantage_x3`, but it still had `policy_argmax_changed_count=0`,
`coverage_return_improvement=-81.582958126732`,
`cumulative_coverage_rate_delta_improvement=-89.45694198338`, and
`fallback_rate=1.0`. This means reward/margin scaling alone is not enough at
the present sample density; the next useful bridge is to expand safe-better
candidate generation across more scenario families before another PPO
improvement attempt.

Stage 5B.3 is now implemented by
`scripts/run_safe_better_pair_expansion_across_families.py` and
`scripts/run_safe_better_pair_expansion_across_families.sh`, with outputs in
`outputs/path_feedback_batch_safe_better_pair_expansion_across_families_v1/`.
It does not run PPO, publish checkpoints, replace the default policy, connect a
real executor, modify the network/action space/default A*, relax guards, or
claim performance. Instead, it re-audits the Stage 5A.2 candidate overlay and
counterfactual coverage rows, recomputes candidate-vs-teacher coverage
advantage, and emits train-split safe-better pairs plus a family gap report.

The current Stage 5B.3 result is a useful data expansion but still fails the
cross-family gate. It expands trainable safe-better pairs from 51 to 615 with
`missing_counterfactual_source_count=0`,
`fallback_gain_contamination_count=0`, `controlled_regression_count=0`,
`runs_new_ppo_update=false`, `publishes_checkpoint=false`,
`replaces_default_policy=false`, and `performance_claimed=false`. However the
safe-better family count is only 3: `mixed_risk=231`,
`rim_or_steep_slope=186`, and `smooth_high_confidence=198`, while
`low_observation_count=0`. Summary status is therefore `failed` with
`reason_codes=["safe_better_family_count_below_threshold",
"family_safe_better_gap_low_observation_count"]` and
`next_required_change=expand_safe_better_pair_generation_across_families`.
This means the next bridge is not another PPO update yet; it is targeted
low-observation counterfactual opportunity generation that can produce at least
8 trainable safe-better pairs for that family without using placeholder
coverage gain.

Stage 5B.4 `Low-Observation Counterfactual Opportunity Generation v1` is now
implemented by
`scripts/run_low_observation_counterfactual_opportunity_generation.py` and
`scripts/run_low_observation_counterfactual_opportunity_generation.sh`, with
configuration in
`configs/quasi_real_low_observation_counterfactual_opportunity_v1.json` and
outputs in
`outputs/path_feedback_batch_low_observation_counterfactual_opportunity_v1/`.
This stage is a targeted source-backed audit/export bridge: it filters only
`low_observation_count` rows from the candidate-level coverage overlay, keeps
only guard-clean, fallback-free, source-backed candidates whose coverage
advantage over teacher is positive, and writes supplemental overlay plus
counterfactual rollout rows for a follow-up 5B.3 run. It does not run PPO,
publish checkpoints, replace the default policy, connect a real executor,
relax guards, or claim performance.

The current Stage 5B.4 result is a controlled failure that sharpens the next
blocker. From 5508 source overlay rows, it found 837 low-observation candidate
rows and all 837 had source available, but
`low_observation_trainable_safe_better_pair_count=0` and
`non_positive_coverage_advantage_count=837`. Summary status is `failed` with
`reason_codes=["low_observation_safe_better_pair_count_below_threshold",
"supplemental_overlay_no_positive_coverage_advantage"]`,
`missing_counterfactual_source_count=0`, `fallback_gain_contamination_count=0`,
and `performance_claimed=false`. Re-running 5B.3 with the empty supplemental
artifacts preserves the prior controlled failure: 615 trainable pairs across 3
families, with `low_observation_count=0`. The next useful change is therefore
not reward tuning or PPO, but a better low-observation candidate-generation
geometry that can actually propose actions with higher multi-step coverage
return than teacher.

Stage 5B.5 `Low-Observation Candidate Geometry Improvement v1` is now
implemented by
`scripts/run_low_observation_candidate_geometry_improvement.py` and
`scripts/run_low_observation_candidate_geometry_improvement.sh`, with
configuration in
`configs/quasi_real_low_observation_candidate_geometry_v1.json` and outputs in
`outputs/path_feedback_batch_low_observation_candidate_geometry_improvement_v1/`.
It runs a high-density low-observation quasi-real geometry pass, bridges it
through `model_explorer path-feedback run`, materializes source-backed
candidate overlay/counterfactual rows from the path-feedback summary plus
scenario contracts, reruns the 5B.4 low-observation audit, and finally reruns
5B.3 with the supplemental artifacts.

The current Stage 5B.5 result passes the missing-family gate. It processed 63
low-observation path-feedback scenarios, materialized 441 ranked
low-observation candidates, and produced
`low_observation_trainable_safe_better_pair_count=108` with
`missing_counterfactual_source_count=0`,
`fallback_gain_contamination_count=0`, `controlled_regression_count=0`, and
`performance_claimed=false`. The rerun 5B.3 summary now has
`status=passed`, `safe_better_than_teacher_candidate_count=723`, and
`safe_better_than_teacher_family_count=4`, with family counts
`low_observation_count=108`, `mixed_risk=231`,
`rim_or_steep_slope=186`, and `smooth_high_confidence=198`. The next useful
stage is therefore to rerun refined coverage-driven PPO improvement on the
expanded four-family safe-better set; this is still not a performance claim
until PPO evaluation improves exploration coverage without guard or fallback
regression.

Stage 5B.6 `Expanded Four-Family Refined Coverage-Driven PPO Improvement v1`
is now implemented by
`scripts/run_expanded_four_family_refined_coverage_driven_ppo_improvement.py`
and
`scripts/run_expanded_four_family_refined_coverage_driven_ppo_improvement.sh`,
with outputs in
`outputs/path_feedback_batch_expanded_four_family_refined_coverage_driven_ppo_improvement_v1/`.
It explicitly consumes the Stage 5B.3 artifacts
`expanded-safe-better-pairs.jsonl` and
`expanded-counterfactual-coverage-rollouts.jsonl`; it does not fall back to the
old Stage 5A.2 51-pair source. For the 108 new low-observation decisions that
do not exist in the old coverage-aware PPO batch, the runner synthesizes
compatible teacher old-policy transitions, recomputes old `log_prob` and
`value`, and then runs one guarded offline PPO update with transition-info
`ppo_advantage` / `ppo_return`.

The current Stage 5B.6 result is a controlled failure with useful progress:
`safe_better_training_pair_count=723`,
`safe_better_training_family_count=4`,
`counterfactual_advantage_nonzero_count=723`,
`policy_argmax_changed_count=379`,
`coverage_return_improvement=4.178973625356`,
`cumulative_coverage_rate_delta_improvement=0.936713498108`,
`valuable_area_covered_improvement=49.499451984438`,
`fallback_rate=0.283540802213`,
`controlled_regression_count=0`, and
`fallback_gain_contamination_count=0`. It still has
`status=failed` with `reason_codes=["coverage_efficiency_regression"]`, because
the added coverage is bought with much worse path-cost efficiency than the
baseline. The next required change is
`tune_cost_efficiency_aware_reward_or_candidate_filter`. The checkpoint remains
experimental: `publishes_checkpoint=false`, `replaces_default_policy=false`, and
`performance_claimed=false`.

Stage 5B.7 `Cost-Efficiency-Aware Coverage Reward / Candidate Filter v1` is
now implemented by
`scripts/run_cost_efficiency_aware_coverage_reward_candidate_filter.py` and
`scripts/run_cost_efficiency_aware_coverage_reward_candidate_filter.sh`, with
outputs in
`outputs/path_feedback_batch_cost_efficiency_aware_coverage_reward_candidate_filter_v1/`.
This stage keeps the expanded four-family source contract from Stage 5B.3, but
adds an efficiency audit before PPO: candidates remain trainable only when they
are train split, source-backed, fallback-free, guard-clean, coverage-positive,
and have positive cost-efficiency-adjusted advantage. High-coverage but costly
rows are downweighted; low-gain high-cost rows become diagnostic-only. It also
materializes a same-decision-set teacher/pre-improvement metric table so the
post-update policy is compared against teacher behavior on the filtered 621-row
candidate universe instead of the older 2,052-row baseline table.

The current Stage 5B.7 result passes the cost-efficiency gate:
`trainable_pair_count=621`, `safe_better_training_pair_count=621`,
`safe_better_training_family_count=4`, family counts
`low_observation_count=9`, `mixed_risk=231`, `rim_or_steep_slope=186`,
`smooth_high_confidence=195`, `counterfactual_advantage_nonzero_count=621`,
`policy_argmax_changed_count=263`,
`coverage_return_improvement=54.96083408637`,
`cumulative_coverage_rate_delta_improvement=56.92136202257`,
`valuable_area_covered_improvement=25.987354935654`,
`coverage_efficiency_regression=false`, `fallback_rate=0.220611916264`,
`controlled_regression_count=0`, and
`fallback_gain_contamination_count=0`. The summary is `status=passed` with
`reason_codes=[]` and
`next_required_change=shadow_canary_release_performance_validation_preflight`.
This is still not a release: `publishes_checkpoint=false`,
`replaces_default_policy=false`, and `performance_claimed=false`.

Stage 6 `Shadow / Canary Release Performance Validation Preflight v1` is now
implemented by
`scripts/run_shadow_canary_release_performance_validation_preflight.py` and
`scripts/run_shadow_canary_release_performance_validation_preflight.sh`, with
outputs in
`outputs/path_feedback_batch_shadow_canary_release_performance_validation_preflight_v1/`.
It is an offline shadow/canary release preflight over the current 5B.7
candidate. The runner reads the 5B.7 summary, filtered batch, refined
transitions, guard replay audit, performance metric table, stage5a rerun
provenance, compatible performance table, and the formal/replay/selected
summaries. It does not reuse the old Stage 5A.2 51-pair source or an older
selected-formal shadow result as current model evidence.

The current Stage 6 result passes the preflight:
`status=passed`, `reason_codes=[]`,
`next_required_change=formal_performance_claim_release_decision`,
`long_horizon_shadow_passed=true`, horizons `[10,20,30]`,
`coverage_return_improvement=54.96083408637`,
`cumulative_coverage_rate_delta_improvement=56.92136202257`,
`valuable_area_covered_improvement=25.987354935654`,
`coverage_efficiency_regression=false`, `fallback_rate=0.220611916264`,
`safe_better_training_family_count=4`, `controlled_regression_count=0`, and
`fallback_gain_contamination_count=0`. The offline release boundary is also
green: `shadow_policy_takes_control=false`,
`experimental_control_activation_count=0`, `kill_switch_audit_passed=true`,
`rollback_audit_passed=true`, and `telemetry_audit_passed=true`. The detailed
long-horizon artifact includes all/family/scenario/context rollups; local
negative windows remain diagnostic risk evidence, while the pass criterion is
the aggregate all-horizon improvement under guard. This stage still does not
publish a checkpoint, replace the default policy, connect a real executor, or
claim formal performance.

Stage 7 `Formal Performance Claim / Release Decision v1` is now implemented by
`scripts/run_formal_performance_claim_release_decision.py` and
`scripts/run_formal_performance_claim_release_decision.sh`, with outputs in
`outputs/path_feedback_batch_formal_performance_claim_release_decision_v1/`.
It is a decision-review layer, not a training or packaging step. The runner
reads the Stage 6 summary and audits, the 5B.7 cost-efficiency evidence, and the
formal training/replay/selected-candidate summaries, then writes an evidence
ledger, metric-consistency audit, claim-scope audit, release-boundary audit,
provenance audit, decision matrix, scoped claim statement, rejection report, and
markdown report.

The current Stage 7 result is `status=passed` with `reason_codes=[]` and
`decision_verdict=approved_for_scoped_offline_performance_claim`.
It approves only
`scoped_offline_performance_claim_approved=true`, with
`performance_claim_scope=scoped_offline_guarded_shadow_canary_only`.
The approved claim is limited to the current offline guarded shadow/canary
evidence and same-decision-set baseline: `coverage_return_improvement=54.96083408637`,
`cumulative_coverage_rate_delta_improvement=56.92136202257`,
`valuable_area_covered_improvement=25.987354935654`,
`coverage_efficiency_regression=false`, and `fallback_rate=0.220611916264`.
Release actions remain explicitly blocked:
`checkpoint_publication_approved=false`,
`default_policy_replacement_approved=false`,
`real_executor_connection_approved=false`, `publishes_checkpoint=false`,
`replaces_default_policy=false`, and `connects_real_executor=false`.
This decision must not be read as real-world performance, Ackermann-feasible
trajectory, default-policy replacement, or unlimited scenario generalization.

Any future checkpoint publication, default-policy installation, or executor
connection must be handled as a separate authorization stage with its own
evidence and rollback gates.

Stage 8 `Scoped Claim Publication & Evidence Freeze v1` is now implemented by
`scripts/run_scoped_claim_publication_evidence_freeze.py` and
`scripts/run_scoped_claim_publication_evidence_freeze.sh`, with outputs in
`outputs/path_feedback_batch_scoped_claim_publication_evidence_freeze_v1/`.
It is a publication-bundle freeze for the Stage 7 scoped claim, not a training,
checkpoint packaging, default-policy installation, or executor-connection step.
The runner reads the Stage 7 summary, scoped claim statement, decision matrix,
evidence ledger, metric/scope/release/provenance audits, plus the Stage 6,
5B.7, formal training, replay, and selected-candidate summaries. It then writes
a bundle manifest, publication claim text, scope audit, evidence-freeze ledger,
documentation-consistency audit, release-boundary audit, rejection report, and
markdown report.

The Stage 8 pass contract is `publication_verdict=approved_for_scoped_claim_publication`,
`scoped_claim_publication_approved=true`, and
`performance_claim_scope=scoped_offline_guarded_shadow_canary_only`, with
`next_required_change=checkpoint_publication_authorization_preflight`. It may
publish only the current offline guarded shadow/canary scoped claim and evidence
references. It still keeps `checkpoint_publication_approved=false`,
`default_policy_replacement_approved=false`,
`real_executor_connection_approved=false`, `publishes_checkpoint=false`,
`replaces_default_policy=false`, and `connects_real_executor=false`.
Stage 8 must not be read as checkpoint publication, default-policy replacement,
real executor authorization, real-world performance, Ackermann-feasible
trajectory, or unlimited scenario generalization.

Stage 9 `Checkpoint Publication Authorization Preflight v1` is now implemented
by `scripts/run_checkpoint_publication_authorization_preflight.py` and
`scripts/run_checkpoint_publication_authorization_preflight.sh`, with outputs in
`outputs/path_feedback_batch_checkpoint_publication_authorization_preflight_v1/`.
It is an authorization preflight for the selected experimental PPO checkpoint,
not checkpoint publication itself. The runner reads the Stage 8 scoped claim
freeze, Stage 7 formal claim decision, Stage 6 shadow/canary validation, 5B.7
cost-efficiency evidence, formal training/replay summaries, and selected
candidate promotion preflight. It recomputes the selected checkpoint SHA-256 and
size, checks metadata and load evidence, audits lineage and release boundaries,
then writes a candidate manifest, identity audit, metadata audit, load-evidence
audit, lineage audit, release-boundary audit, authorization matrix, rejection
report, and markdown report.

The Stage 9 pass contract is
`authorization_verdict=eligible_for_checkpoint_publication_package_preparation`,
`checkpoint_publication_authorization_preflight_passed=true`, and
`checkpoint_publication_package_preparation_approved=true`, with
`next_required_change=checkpoint_publication_package_preparation`. It preserves
the selected checkpoint identity: seed `0`, budget `epochs1_lr3e-6`, SHA-256
`9d9539c685ab965739c91958bf9cbfe90329c460b4bdbcc35881875aa62f0aa2`.
Release actions remain explicitly closed:
`checkpoint_publication_approved=false`,
`default_policy_replacement_approved=false`,
`real_executor_connection_approved=false`, `publishes_checkpoint=false`,
`replaces_default_policy=false`, and `connects_real_executor=false`.
Stage 9 does not copy the checkpoint to a publication path, publish a
checkpoint, replace the default policy, connect a real executor, run PPO,
modify network/action space/default A*, relax guards, claim Ackermann-feasible
trajectory, or treat IRIS/GCS/path-planner diagnostics as release proof.

Stage 10 `Checkpoint Publication Package Preparation v1` is now implemented by
`scripts/run_checkpoint_publication_package_preparation.py` and
`scripts/run_checkpoint_publication_package_preparation.sh`, with outputs in
`outputs/path_feedback_batch_checkpoint_publication_package_preparation_v1/`.
It prepares an isolated package under
`outputs/path_feedback_batch_checkpoint_publication_package_preparation_v1/checkpoint-publication-package/`
by copying the Stage 9-authorized selected experimental checkpoint and metadata
into the package root, then freezing source/package SHA-256, size, metadata,
lineage, rollback, and release-boundary audits.

The Stage 10 pass contract is
`package_preparation_verdict=prepared_for_checkpoint_publication_package_verification`,
`checkpoint_publication_package_prepared=true`, source and package checkpoint
SHA-256 both equal
`9d9539c685ab965739c91958bf9cbfe90329c460b4bdbcc35881875aa62f0aa2`, and
`next_required_change=checkpoint_publication_package_verification`. Release
actions remain explicitly closed:
`checkpoint_publication_approved=false`,
`default_policy_replacement_approved=false`,
`real_executor_connection_approved=false`, `publishes_checkpoint=false`,
`replaces_default_policy=false`, and `connects_real_executor=false`. Stage 10
does not publish a checkpoint, replace the default policy, connect a real
executor, run PPO, modify network/action space/default A*, relax guards, claim
Ackermann-feasible trajectory, or treat IRIS/GCS/path-planner diagnostics as
release proof.

Stage 11 `Checkpoint Publication Package Verification v1` is now implemented by
`scripts/run_checkpoint_publication_package_verification.py` and
`scripts/run_checkpoint_publication_package_verification.sh`, with outputs in
`outputs/path_feedback_batch_checkpoint_publication_package_verification_v1/`.
It independently verifies the Stage 10 package as a consumable artifact: the
runner rereads the package manifest, recomputes package checkpoint SHA-256 and
size, validates package metadata, performs a `torch.load` state-dict check, and
rechecks lineage, rollback, and release-boundary evidence.

The Stage 11 pass contract is
`verification_verdict=verified_for_checkpoint_publication_sandbox_install_dry_run_preflight`,
`checkpoint_publication_package_verification_passed=true`,
`checkpoint_publication_sandbox_install_dry_run_preflight_approved=true`, and
`next_required_change=checkpoint_publication_sandbox_install_dry_run_preflight`.
The verified package checkpoint SHA-256 remains
`9d9539c685ab965739c91958bf9cbfe90329c460b4bdbcc35881875aa62f0aa2`.
Release actions remain explicitly closed:
`checkpoint_publication_approved=false`,
`default_policy_replacement_approved=false`,
`real_executor_connection_approved=false`, `publishes_checkpoint=false`,
`replaces_default_policy=false`, and `connects_real_executor=false`. Stage 11
does not publish a checkpoint, replace the default policy, connect a real
executor, run PPO, modify network/action space/default A*, relax guards, claim
Ackermann-feasible trajectory, or treat IRIS/GCS/path-planner diagnostics as
release proof.

Stage 12 `Checkpoint Publication Sandbox Install Dry-Run Preflight v1` is now
implemented by
`scripts/run_checkpoint_publication_sandbox_install_dry_run_preflight.py` and
`scripts/run_checkpoint_publication_sandbox_install_dry_run_preflight.sh`, with
outputs in
`outputs/path_feedback_batch_checkpoint_publication_sandbox_install_dry_run_preflight_v1/`.
It is a preflight for a future sandbox install dry-run, not the dry-run itself.
The runner reads the Stage 11 consumer manifest as the authoritative package
path, recomputes package checkpoint SHA-256 and size, checks Stage 11 load
verification evidence, declares only a future sandbox root and planned consumer
checkpoint path, audits optional default-policy read-only boundaries, and
rechecks lineage, rollback, path, and release boundaries.

The Stage 12 pass contract is
`preflight_verdict=eligible_for_checkpoint_publication_sandbox_install_dry_run`,
`checkpoint_publication_sandbox_install_dry_run_preflight_passed=true`,
`checkpoint_publication_sandbox_install_dry_run_approved=true`, and
`next_required_change=checkpoint_publication_sandbox_install_dry_run`. The
package checkpoint SHA-256 remains
`9d9539c685ab965739c91958bf9cbfe90329c460b4bdbcc35881875aa62f0aa2`, and the
planned sandbox checkpoint path remains only a plan. Release actions remain
explicitly closed: `checkpoint_publication_approved=false`,
`default_policy_replacement_approved=false`,
`real_executor_connection_approved=false`, `publishes_checkpoint=false`,
`replaces_default_policy=false`, and `connects_real_executor=false`. Stage 12
does not copy checkpoint files into the sandbox, does not copy to publication,
default, live, release, or executor paths, does not install a policy, does not
run rollout/PPO, does not replace the default policy, does not connect a real
executor, and does not claim Ackermann-feasible trajectory or real-world
performance.

Stage 13 `Checkpoint Publication Sandbox Install Dry-Run v1` is now implemented
by `scripts/run_checkpoint_publication_sandbox_install_dry_run.py` and
`scripts/run_checkpoint_publication_sandbox_install_dry_run.sh`, with outputs in
`outputs/path_feedback_batch_checkpoint_publication_sandbox_install_dry_run_v1/`.
It performs the sandbox install dry-run authorized by Stage 12: the runner
copies the verified package checkpoint and metadata only into the Stage 12
planned sandbox paths, recomputes source and sandbox SHA-256/size, loads the
sandbox checkpoint through the consumer path, and rechecks default-policy,
path, lineage, rollback, and release-boundary evidence.

The Stage 13 pass contract is
`install_dry_run_verdict=installed_in_sandbox_for_checkpoint_publication_sandbox_install_dry_run_verification`,
`checkpoint_publication_sandbox_install_dry_run_passed=true`,
`checkpoint_publication_sandbox_install_dry_run_verification_approved=true`,
and `next_required_change=checkpoint_publication_sandbox_install_dry_run_verification`.
The source and sandbox checkpoint SHA-256 both remain
`9d9539c685ab965739c91958bf9cbfe90329c460b4bdbcc35881875aa62f0aa2`.
Release actions remain explicitly closed:
`checkpoint_publication_approved=false`,
`default_policy_replacement_approved=false`,
`real_executor_connection_approved=false`, `publishes_checkpoint=false`,
`replaces_default_policy=false`, and `connects_real_executor=false`. Stage 13
does not publish a checkpoint, does not copy to publication/default/live/release
or executor paths, does not replace the default policy, does not connect a real
executor, does not run rollout/PPO, and does not claim Ackermann-feasible
trajectory or real-world performance.

Stage 14 `Checkpoint Publication Sandbox Install Dry-Run Verification v1` is now
implemented by
`scripts/run_checkpoint_publication_sandbox_install_dry_run_verification.py` and
`scripts/run_checkpoint_publication_sandbox_install_dry_run_verification.sh`,
with outputs in
`outputs/path_feedback_batch_checkpoint_publication_sandbox_install_dry_run_verification_v1/`.
It is a read-only re-verification of the Stage 13 sandbox install result: the
runner recomputes source and sandbox SHA-256/size, checks the install manifest,
re-loads the sandbox checkpoint from the consumer path, verifies sandbox
metadata, and rechecks lineage, rollback, and release-boundary evidence.

The Stage 14 pass contract is
`verification_verdict=verified_for_checkpoint_publication_sandbox_consumer_smoke_preflight`,
`checkpoint_publication_sandbox_install_dry_run_verification_passed=true`,
`checkpoint_publication_sandbox_consumer_smoke_preflight_approved=true`, and
`next_required_change=checkpoint_publication_sandbox_consumer_smoke_preflight`.
The source and sandbox checkpoint SHA-256 both remain
`9d9539c685ab965739c91958bf9cbfe90329c460b4bdbcc35881875aa62f0aa2`.
Release actions remain explicitly closed:
`checkpoint_publication_approved=false`,
`default_policy_replacement_approved=false`,
`real_executor_connection_approved=false`, `publishes_checkpoint=false`,
`replaces_default_policy=false`, and `connects_real_executor=false`. Stage 14
does not copy, overwrite, move, or delete checkpoint/metadata files, does not
publish a checkpoint, does not replace the default policy, does not connect a
real executor, does not run rollout/PPO, and does not claim Ackermann-feasible
trajectory or real-world performance.

Stage 15 `Checkpoint Publication Sandbox Consumer Replay/Canary v1` consumes the
Stage 14 verified sandbox checkpoint in an isolated consumer harness. It runs CPU
load+forward on deterministic `PolicyObservation` replay rows, records
step-level logits/masked logits/action-mask/selected-action/log-prob/value
telemetry, and audits fallback, default-policy read-only boundary, lineage,
rollback, and release boundary. Its output root is
`outputs/path_feedback_batch_checkpoint_publication_sandbox_consumer_replay_canary_v1/`.
The pass contract advances only to
`default_policy_candidate_authorization_preflight`; checkpoint publication,
default policy replacement, and real executor connection remain false.

Stage 16 `Default Policy Candidate Authorization Preflight v1` reads the Stage
15 consumer replay/canary evidence and checks kill-switch, rollback,
default-policy boundary, path-planner isolation, lineage, and release boundary
before allowing the next gate,
`default_policy_candidate_sandbox_install_preflight`. It is not default policy
replacement and keeps `checkpoint_publication_approved=false`,
`default_policy_replacement_approved=false`,
`real_executor_connection_approved=false`, `publishes_checkpoint=false`,
`replaces_default_policy=false`, and `connects_real_executor=false`.

The current Stage 16 result is `status=passed` with `reason_codes=[]`,
`authorization_verdict=eligible_for_default_policy_candidate_sandbox_install_preflight`,
`default_policy_candidate_authorization_preflight_passed=true`,
`kill_switch_audit_passed=true`, `rollback_audit_passed=true`,
`default_policy_boundary_audit_passed=true`,
`path_planner_isolation_audit_passed=true`, and
`release_boundary_audit_passed=true`. This closes the planned release-governance
front half: the candidate checkpoint has passed sandbox install verification,
sandbox consumer replay/canary, and default-policy-candidate authorization
preflight. The project intentionally pauses further release governance here and
does not proceed to `default_policy_candidate_sandbox_install_preflight` until
the next algorithm gap-closure evidence is stronger.

The next model-explorer mainline target is **Family-Balanced Algorithm Gap
Closure v1**. It resumes algorithm work from the current 5B.7 cost-efficiency
candidate and the Stage 16 sandbox/authorization evidence, but it does not
publish checkpoints, replace the default policy, connect a real executor, run
default-policy installation, relax guards, change network/action space/default
A*, or claim real-world performance. The first gap is family balance:
`low_observation_count` remains thin in the 5B.7 trainable set
(`low_observation_count=9` versus `mixed_risk=231`,
`rim_or_steep_slope=186`, and `smooth_high_confidence=195`). The next algorithm
work should therefore improve low-observation candidate generation and
source-backed counterfactual coverage, rerun coverage-driven PPO only after the
family-balanced inputs are valid, and then validate exploration coverage,
cost-efficiency, fallback contamination, controlled regression, and holdout /
scenario / context-disjoint generalization before any further default-policy
installation preflight.

Stage `Family-Balanced Algorithm Gap Closure v1` materializes this pause as an
auditable bridge. Its runner is
`scripts/run_family_balanced_algorithm_gap_closure.py`, with shell entrypoint
`scripts/run_family_balanced_algorithm_gap_closure.sh` and output root
`outputs/path_feedback_batch_family_balanced_algorithm_gap_closure_v1/`. It
reads Stage 16, Stage 5B.7 cost-efficiency, low-observation geometry, and
Stage 5B.3 four-family safe-better evidence; recomputes family counts from
json/jsonl sources; replaces the thin 5B.7 low-observation slice with
source-backed low-observation geometry supplements; and writes
`family-balanced-safe-better-pairs.jsonl`,
`family-balanced-counterfactual-rollouts.jsonl`, plus a coverage-driven input
manifest. The pass contract is `family_balanced_algorithm_gap_closure_passed=true`,
`low_observation_gap_closed=true`, `family_balance_audit_passed=true`,
`source_backed_counterfactual_audit_passed=true`, and
`next_required_change=family_balanced_coverage_driven_ppo_rerun`.

This stage keeps the release boundary closed:
`checkpoint_publication_approved=false`,
`default_policy_replacement_approved=false`,
`real_executor_connection_approved=false`, `publishes_checkpoint=false`,
`replaces_default_policy=false`, and `connects_real_executor=false`. It does
not run a PPO update, publish a checkpoint, replace the default policy, connect
a real executor, relax guards, modify network/action space/default A*, or claim
Ackermann-feasible trajectory or real-world performance.

Stage `Family-Balanced Coverage-Driven PPO Rerun v1` is the next algorithm
step after the gap-closure bridge. Its runner is
`scripts/run_family_balanced_coverage_driven_ppo_rerun.py`, with shell
entrypoint `scripts/run_family_balanced_coverage_driven_ppo_rerun.sh` and
output root
`outputs/path_feedback_batch_family_balanced_coverage_driven_ppo_rerun_v1/`.
It reads the 720 family-balanced pairs and 720 counterfactual rows from
`outputs/path_feedback_batch_family_balanced_algorithm_gap_closure_v1/`,
materializes a `family-balanced-compatible-stage5a2-input/`, builds a
`family-balanced-compatible-coverage-driven-input/`, synthesizes missing
old-policy transitions with finite old log-prob/value, and then runs an offline
guarded PPO rerun through the refined coverage-driven PPO path using
`family_balanced_ppo_advantage`.

The pass contract is
`family_balanced_coverage_driven_ppo_rerun_passed=true`,
`family_balanced_shadow_canary_preflight_approved=true`,
`family_balanced_input_audit_passed=true`,
`old_transition_materialization_audit_passed=true`,
`ppo_update_status=passed`, `guard_replay_audit_passed=true`,
positive `coverage_return_improvement`,
positive `cumulative_coverage_rate_delta_improvement`,
positive `valuable_area_covered_improvement`,
`coverage_efficiency_regression=false`,
`policy_argmax_changed_count>0`, `fallback_rate<0.5`,
`fallback_gain_contamination_count=0`, `controlled_regression_count=0`,
`safe_better_training_family_count=4`,
`low_observation_trainable_transition_count>=32`, and
`next_required_change=family_balanced_shadow_canary_preflight`.

This rerun writes an experimental checkpoint for subsequent offline replay and
shadow/canary preflight only. It keeps
`checkpoint_publication_approved=false`,
`default_policy_replacement_approved=false`,
`real_executor_connection_approved=false`, `publishes_checkpoint=false`,
`replaces_default_policy=false`, and `connects_real_executor=false`. It does
not resume Stage 17 default-policy installation, publish a checkpoint, replace
the default policy, connect a real executor, relax guards, modify
network/action space/default A*, claim Ackermann-feasible trajectory, or claim
real-world performance.

Stage `Family-Balanced Shadow/Canary Preflight v1` consumes that rerun evidence
without running another PPO update. Its runner is
`scripts/run_family_balanced_shadow_canary_preflight.py`, with shell entrypoint
`scripts/run_family_balanced_shadow_canary_preflight.sh` and output root
`outputs/path_feedback_batch_family_balanced_shadow_canary_preflight_v1/`.
It reads the rerun summary, guard replay audit, refined performance metrics,
same-input compatible performance baseline, family-balanced pair ledger, gap
closure summary, and upstream formal/replay/selected-candidate summaries. It
then writes a long-horizon shadow validation, family generalization audit,
coverage efficiency audit, guard/fallback audit, kill-switch audit, rollback
audit, telemetry audit, lineage audit, and release-boundary audit.

The pass contract is `family_balanced_shadow_canary_preflight_passed=true`,
`family_balanced_formal_performance_claim_release_decision_approved=true`,
`long_horizon_shadow_passed=true`, positive coverage/cumulative/valuable
improvements, `coverage_efficiency_regression=false`,
`family_generalization_audit_passed=true`,
`low_observation_shadow_passed=true`, `fallback_rate<0.5`,
`fallback_gain_contamination_count=0`, `controlled_regression_count=0`,
`kill_switch_audit_passed=true`, `rollback_audit_passed=true`,
`telemetry_audit_passed=true`, `shadow_policy_takes_control=false`,
`experimental_control_activation_count=0`, and
`next_required_change=family_balanced_formal_performance_claim_release_decision`.

This stage remains an offline shadow/canary preflight. It does not publish a
checkpoint, replace the default policy, connect a real executor, continue Stage
17 default-policy installation, relax guards, modify network/action
space/default A*, claim Ackermann-feasible trajectory, or claim real-world
performance.

Stage `Family-Balanced Formal Performance Claim / Release Decision v1` is the
next decision layer after the family-balanced shadow/canary preflight. Its
runner is
`scripts/run_family_balanced_formal_performance_claim_release_decision.py`,
with shell entrypoint
`scripts/run_family_balanced_formal_performance_claim_release_decision.sh` and
output root
`outputs/path_feedback_batch_family_balanced_formal_performance_claim_release_decision_v1/`.
It reads the passed shadow/canary summary plus long-horizon,
family-generalization, coverage-efficiency, guard/fallback, kill-switch,
rollback, telemetry, lineage, and release-boundary audits; then it cross-checks
the family-balanced rerun, gap-closure, formal training, post-training replay,
and selected-candidate evidence.

The pass contract is
`family_balanced_formal_performance_claim_release_decision_passed=true`,
`family_balanced_scoped_offline_performance_claim_approved=true`,
`decision_verdict=approved_for_family_balanced_scoped_offline_performance_claim`,
`allowed_claim_scope=scoped_offline_family_balanced_guarded_shadow_canary_only`,
positive aggregate coverage/cumulative/valuable improvements,
`coverage_efficiency_regression=false`, `fallback_rate<0.5`,
`fallback_gain_contamination_count=0`, `controlled_regression_count=0`,
`family_generalization_audit_passed=true`,
`low_observation_shadow_passed=true`,
`low_observation_limitation_acknowledged=true`, and
`next_required_change=family_balanced_scoped_claim_publication_evidence_freeze`.
The claim statement must explicitly state the low-observation limitation:
current low-observation shadow evidence passes, but low-observation coverage
return does not outperform teacher in the audited evidence set.

This decision stage still keeps the release boundary closed:
`checkpoint_publication_approved=false`,
`default_policy_replacement_approved=false`,
`real_executor_connection_approved=false`, `publishes_checkpoint=false`,
`replaces_default_policy=false`, and `connects_real_executor=false`. It does
not publish a checkpoint, replace the default policy, connect a real executor,
continue Stage 17 default-policy installation, relax guards, modify
network/action space/default A*, claim Ackermann-feasible trajectory, claim
real-world performance, or turn IRIS/GCS/path-planner diagnostics into release
proof.

## Family-Balanced Publication Governance Back Half v1

After `Family-Balanced Formal Performance Claim / Release Decision v1` passes,
model-explorer can resume the release-governance mainline, but only as a
family-balanced evidence chain. The terminal gate for this recovery segment is
the **family-balanced default-policy candidate sandbox install preflight**; it
is not default-policy installation, not checkpoint publication, and not real
executor integration.

The back-half chain is:

```text
family_balanced_scoped_claim_publication_evidence_freeze
  -> family_balanced_checkpoint_publication_authorization_preflight
  -> family_balanced_checkpoint_publication_package_preparation
  -> family_balanced_checkpoint_publication_package_verification
  -> family_balanced_checkpoint_publication_sandbox_install_dry_run_preflight
  -> family_balanced_checkpoint_publication_sandbox_install_dry_run
  -> family_balanced_checkpoint_publication_sandbox_install_dry_run_verification
  -> family_balanced_checkpoint_publication_sandbox_consumer_replay_canary
  -> family_balanced_default_policy_candidate_authorization_preflight
  -> family_balanced_default_policy_candidate_sandbox_install_preflight
```

Each stage writes its own
`outputs/path_feedback_batch_family_balanced_*_v1/` root with summary,
manifest, audit, rejection report, and report artifacts. The chain freezes the
family-balanced scoped claim, authorizes and packages the family-balanced
checkpoint candidate, verifies package identity and loadability, performs only
sandbox dry-run copy/verification, runs isolated consumer replay/canary
telemetry, and then audits kill-switch, rollback, default-policy read-only
boundary, and path-planner isolation for the candidate sandbox install
preflight.

This recovery segment must continue to state 不替换 default policy and 不连接真实执行器.
All summaries keep `checkpoint_publication_approved=false`,
`default_policy_replacement_approved=false`,
`real_executor_connection_approved=false`, `publishes_checkpoint=false`,
`replaces_default_policy=false`, and `connects_real_executor=false`. It does
not run PPO, expand rollout, modify reward, change network/action
space/default A*, relax guards, claim real-world performance, claim
Ackermann-feasible trajectory, or treat IRIS/GCS/path-planner diagnostics as
release proof.

## Global 99% Exploration Coverage Line

`Global 99% Exploration Coverage Line` is a new exploration-performance
roadmap. It is not a current capability claim for the family-balanced policy or
the existing default-policy candidate. The line starts by defining a measurable
contract for 1km x 1km maps and arbitrary ROIs, then adds non-learning and
policy-guided coverage loops before any network architecture upgrade is
justified.

The target metric is 99% coverage of
`reachable_safe_exploration_cells` inside the requested ROI, not 99% of all raw
map cells. If the requested ROI contains unreachable, unsafe, or budget-limited
cells, the benchmark must report explicit infeasibility reason codes instead
of treating the run as a policy success or failure without context.

The development order is:

```text
Global 99% Coverage Benchmark v1
  -> Frontier Coverage Planner Baseline v1
  -> Coverage Memory + Replanning Loop v1
  -> Policy-Guided Global Coverage v1
  -> 99% Multi-Map Generalization v1
  -> Network Architecture Upgrade Readiness Review v1
  -> Global 99 Release Governance Preflight v1
  -> Global 99 Shadow Canary Preflight v1
  -> Global 99 Shadow Canary Replay v1
  -> Global 99 Real Map Preflight v1
  -> Global 99 Real Map Shadow Replay v1
  -> Global 99 Real Map Release Governance Preflight v1
  -> Global 99 Real Map Shadow Canary Preflight v1
  -> Global 99 Real Map Shadow Canary Replay v1
  -> Global 99 Real Map Evidence Refresh / Drift Audit v1
  -> Global 99 Real Map Multi-ROI Generalization v1
  -> Default Policy Candidate Authorization Preflight
  -> Sandbox Candidate Installation Dry Run
  -> Sandbox Consumer Replay / Canary
  -> Controlled Default Policy Candidate Installation Preflight
  -> Network Architecture Upgrade v1 (only if evidence supports it)
  -> 99% Coverage Release Governance v1
```

`巡策` is the short code name for `Topology-Aware Coverage Policy Network
Design`, a separate research track for a future network architecture
breakthrough. Its primary goal is higher exploration coverage and stronger
generalization, with lower parameter count and faster inference as hard
constraints. The design is documented in
`docs/superpowers/specs/2026-06-16-topology-aware-coverage-policy-network-design.md`.
It proposes coverage-memory-aware candidate graph ranking: frontier/waypoint
candidates remain the action set, but candidate relations such as frontier
cluster, BFS distance, ROI overlap, bottleneck risk, revisit cost, budget
fraction, and coverage-memory state can bias a lightweight graph/attention
network. This line is not a default-policy installation path, not a checkpoint
publication path, and not a real-executor path. It should begin with literature
and bottleneck auditing, then an additive observation-contract review, before
any architecture prototype or training attempt.

`Xunce Design Freeze Audit v1` is the Stage 0 evidence gate for this research
line. It verifies that the design documents declare the `巡策` name, the full
0-17 plan-first stage chain, the Global 99 separation, and closed release /
executor / PPO / network-modification boundaries. Run it with:

```bash
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_xunce_design_freeze_audit.sh
```

The output root is
`outputs/path_feedback_batch_xunce_design_freeze_v1/`. A passing summary writes
`next_required_change=current_head_evidence_refresh`, which means the next step
is to refresh Current-HEAD evidence before any literature/bottleneck review or
architecture prototype.

`Xunce Current-HEAD Evidence Refresh v1` is the Stage 1 gate. It consumes the
Stage 0 freeze summary plus Global 99 controlled-installation, real-map replay,
real-map multi-ROI, and network-readiness summaries. It requires the active
checkout to be clean, treats dirty or mismatched declared git provenance as a
blocker, and records legacy missing provenance explicitly rather than silently
claiming it is current. Run it with:

```bash
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_xunce_current_head_evidence_refresh.sh
```

The output root is
`outputs/path_feedback_batch_xunce_current_head_evidence_refresh_v1/`. A passing
summary writes `next_required_change=network_literature_bottleneck_review`.

`Xunce Network Literature Bottleneck Review v1` is the Stage 2 gate. It maps
primary research sources for set ranking, graph attention, routing attention,
sequence memory, modular exploration, and learned-planning diagnostics to the
actual project evidence. It also reads the current architecture inventory
(`mlp_v1`, `mlp_missing_v1`, `candidate_attention_v1`) and Global 99 evidence to
decide whether the current bottleneck is coverage/generalization, fallback /
policy regression, latency/parameter evidence, topology representation, or no
current release-blocking network bottleneck. Run it with:

```bash
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_xunce_network_literature_bottleneck_review.sh
```

The output root is
`outputs/path_feedback_batch_xunce_network_literature_bottleneck_review_v1/`.
Passing does not approve training; it writes
`next_required_change=topology_observation_contract`.

`Xunce Topology Observation Contract v1` is the Stage 3 gate. It defines an
additive contract for candidate topology fields, candidate-edge fields, and a
compact coverage-memory token while preserving `policy-observation/v1.1`,
`action_mask`, candidate order, missing indicators, old checkpoint/scorer
compatibility, and closed release boundaries. Run it with:

```bash
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_xunce_topology_observation_contract.sh
```

The output root is
`outputs/path_feedback_batch_xunce_topology_observation_contract_v1/`. Passing
writes `next_required_change=topology_feature_extraction_audit`. It is still
contract design only, not feature extraction, model implementation, training,
or checkpoint publication.

`Xunce Topology Feature Extraction Audit v1` is the Stage 4 gate. It consumes
the Stage 3 contract and a deterministic synthetic coverage fixture, then
extracts candidate topology fields, candidate-edge fields, and the compact
coverage-memory token using existing Global 99 geometry and frontier helpers.
It audits frontier cluster, ROI group, BFS distance, coverage overlap, revisit,
budget fraction, fallback risk, pairwise candidate relations, memory-token
values, missing indicators, deterministic ordering, and closed boundaries. Run
it with:

```bash
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_xunce_topology_feature_extraction_audit.sh
```

The output root is
`outputs/path_feedback_batch_xunce_topology_feature_extraction_audit_v1/`.
Passing writes `next_required_change=topology_aware_coverage_graph_proto`. It
authorizes only the small prototype stage; it is not a network implementation,
training run, checkpoint publication, default-policy replacement, or executor
connection.

`Xunce Topology Graph Prototype v1` is the Stage 5 gate. It consumes Stage 4
candidate, edge, and memory-token artifacts, builds a small research-only
`topology_aware_coverage_graph_proto_v1` PyTorch module, and audits that the
candidate graph plus memory token can produce finite masked logits, action
probabilities, and a scalar value. The prototype is not registered as a
production trainable/default-policy architecture. Run it with:

```bash
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_xunce_topology_graph_proto.sh
```

The output root is
`outputs/path_feedback_batch_xunce_topology_graph_proto_v1/`. Passing writes
`next_required_change=xunce_proto_mechanism_validation`. It authorizes only
Stage 6 mechanism validation; it does not train, write a checkpoint, publish a
checkpoint, replace default policy, connect an executor, or claim performance.

`Xunce Prototype Mechanism Validation v1` is the Stage 6 gate. It rebuilds the
Stage 5 research prototype and compares full-signal logits against
edge-ablated and memory-ablated forward passes. The audit proves that topology
edges and the memory token can change candidate ranking while preserving action
masking, finite outputs, fallback safety, and closed release/training/executor
boundaries. Run it with:

```bash
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_xunce_proto_mechanism_validation.sh
```

The output root is
`outputs/path_feedback_batch_xunce_proto_mechanism_validation_v1/`. Passing
writes `next_required_change=architecture_contrast_evaluation`. It is still a
mechanism audit, not evidence that the prototype improves coverage and not
approval to train or publish.

`Xunce Architecture Contrast Evaluation v1` is the Stage 7 gate. It compares
`mlp_v1`, `mlp_missing_v1`, `candidate_attention_v1`, and the research-only
`topology_aware_coverage_graph_proto_v1` on the same deterministic topology
fixture. The runner records each architecture's selected candidate, heuristic
ranking quality, parameter count, latency, finite-output behavior, mask
correctness, and closed release/training/executor boundaries. Run it with:

```bash
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_xunce_architecture_contrast_evaluation.sh
```

The output root is
`outputs/path_feedback_batch_xunce_architecture_contrast_evaluation_v1/`.
Passing writes `next_required_change=full_xunce_network_v1_design`. This is a
contrast audit only: it may justify designing the complete Xunce network v1, but
it does not train, publish a checkpoint, register a production architecture,
replace default policy, connect an executor, or claim real-world performance.

`Xunce Full Network v1` is the Stage 8 research-network gate. It implements a
research-only `xunce_full_network_v1` PyTorch module with a candidate graph
encoder, edge encoder, topology-biased message passing, coverage-memory token,
ROI/budget context fusion, masked policy logits, and scalar value head. Run it
with:

```bash
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_xunce_full_network_v1.sh
```

The output root is `outputs/path_feedback_batch_xunce_full_network_v1/`.
Passing writes `next_required_change=full_network_static_contract_validation`.
This implements the complete research network forward contract, but it still
does not train, publish a checkpoint, register a production/default-policy
architecture, replace default policy, connect an executor, or claim performance.

`Xunce Full Network Static Contract Validation v1` is the Stage 9 gate. It runs
deterministic forward cases for shape, action masks, metadata, missing-indicator
defaults, and legacy/additive-observation compatibility. Run it with:

```bash
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_xunce_full_network_static_contract_validation.sh
```

The output root is
`outputs/path_feedback_batch_xunce_full_network_static_contract_validation_v1/`.
Passing writes `next_required_change=full_network_ablation_experiments`. This
is a static contract gate only; it does not prove performance and does not
train, publish, install, or connect an executor.

`Xunce Full Network Ablation Experiments v1` is the Stage 10 gate. It compares a
full `xunce_full_network_v1` forward pass with deterministic ablations for
topology edges, coverage memory, ROI/budget context, and missing indicators.
Run it with:

```bash
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_xunce_full_network_ablation_experiments.sh
```

The output root is
`outputs/path_feedback_batch_xunce_full_network_ablation_experiments_v1/`.
Passing writes `next_required_change=full_network_stress_evaluation`. This is a
module-contribution audit only; it does not train, publish, install, or connect
an executor.

`Xunce Full Network Stress Evaluation v1` is the Stage 11 gate. It stress-tests
candidate count, edge density, missing indicators, numeric range, finite output,
mask preservation, parameter budget, latency budget, and deterministic replay.
Run it with:

```bash
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_xunce_full_network_stress_evaluation.sh
```

The output root is
`outputs/path_feedback_batch_xunce_full_network_stress_evaluation_v1/`. Passing
writes `next_required_change=guarded_training_candidate_preflight`. This is a
stability gate only; it does not train, publish, install, or connect an
executor.

`Xunce Guarded Training Candidate Preflight v1` is the Stage 12 authorization
gate. It reads the full-network, static-contract, ablation, and stress
summaries and decides whether the next stage may run a controlled training
candidate. Run it with:

```bash
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_xunce_guarded_training_candidate_preflight.sh
```

The output root is
`outputs/path_feedback_batch_xunce_guarded_training_candidate_preflight_v1/`.
Passing writes `next_required_change=controlled_training_candidate`. This stage
does not itself run PPO, write checkpoints, publish checkpoints, replace default
policy, or connect an executor.

`Xunce Controlled Training Candidate v1` is the Stage 13 research-training
candidate gate. It reads the Stage 12 preflight summary and runs a tiny,
deterministic, bounded supervised surrogate update for `xunce_full_network_v1`.
Run it with:

```bash
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_xunce_controlled_training_candidate.sh
```

The output root is
`outputs/path_feedback_batch_xunce_controlled_training_candidate_v1/`. Passing
writes `next_required_change=post_training_offline_evaluation`. This stage may
write `xunce-controlled-training-candidate.pt` as a research-only checkpoint
inside the output root, but `publishes_checkpoint=false`,
`runs_new_ppo_update=false`, `ppo_update_executed=false`, and it does not replace
default policy or connect an executor.

`Xunce Post-Training Offline Evaluation v1` is the Stage 14 read-only evaluation
gate. It loads the Stage 13 research checkpoint, compares trained metrics
against a deterministic fresh `xunce_full_network_v1` model, and writes
checkpoint-load and policy-delta audits. Run it with:

```bash
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_xunce_post_training_offline_evaluation.sh
```

The output root is
`outputs/path_feedback_batch_xunce_post_training_offline_evaluation_v1/`.
Passing writes `next_required_change=shadow_replay_validation`. This stage is
read-only with respect to the checkpoint: `checkpoint_read_only=true`,
`runs_new_training_update=false`, `runs_new_ppo_update=false`, and
`publishes_checkpoint=false`.

`Xunce Shadow Replay Validation v1` is the Stage 15 offline replay gate. It
reloads the Stage 13 research checkpoint, replays the same deterministic cases
used by Stage 14, and verifies source-match/determinism before sandbox
packaging. Run it with:

```bash
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_xunce_shadow_replay_validation.sh
```

The output root is
`outputs/path_feedback_batch_xunce_shadow_replay_validation_v1/`. Passing writes
`next_required_change=sandbox_candidate_preflight`. This is still offline
shadow/replay only: `starts_online_canary=false`, `connects_real_executor=false`,
and `publishes_checkpoint=false`.

`Xunce Sandbox Candidate Preflight v1` is the Stage 16 sandbox packaging and
load dry-run gate. It copies the research checkpoint into a sandbox-only output
package, verifies checkpoint hash/load, and writes kill-switch, rollback,
telemetry, and boundary audits. Run it with:

```bash
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_xunce_sandbox_candidate_preflight.sh
```

The output root is
`outputs/path_feedback_batch_xunce_sandbox_candidate_preflight_v1/`. Passing
writes `next_required_change=xunce_release_governance_gate`. This stage creates
a sandbox package only; `publishes_checkpoint=false`,
`replaces_default_policy=false`, and `connects_real_executor=false`.

`Xunce Release Governance Gate v1` is the Stage 17 final research-chain
governance gate. It consumes the sandbox candidate preflight summary, audits
lineage, scope, and closed release boundaries, and writes the final 巡策 research
verdict. Run it with:

```bash
PYTHON=/home/kai/anaconda3/envs/lunar-explorer/bin/python \
  bash scripts/run_xunce_release_governance_gate.sh
```

The output root is
`outputs/path_feedback_batch_xunce_release_governance_gate_v1/`. Passing writes
`next_required_change=xunce_research_track_complete` and
`release_governance_verdict=research_candidate_ready_for_human_governance_review`.
This completes the research evidence chain only; it keeps
`default_policy_replacement_approved=false`, `real_world_release_approved=false`,
`publishes_checkpoint=false`, and `connects_real_executor=false`.

`Xunce High-Fidelity Real-Map ROI Expansion v1` and
`Xunce High-Fidelity Real-Map Policy Comparison v1` extend the completed 巡策
research chain with a larger quasi-real LOLA comparison, not a release. The ROI
expansion runner is `scripts/run_xunce_high_fidelity_real_map_roi_expansion.py`,
with config `configs/xunce_high_fidelity_real_map_roi_expansion_v1.json` and
output root
`outputs/path_feedback_batch_xunce_high_fidelity_real_map_roi_expansion_v1/`.
It expands the evidence matrix from 12 slices / 4 ROI groups to 24 slices / 8
ROI groups, keeps context IDs and sidecar/contract paths auditable, and passes
with `next_required_change=xunce_high_fidelity_real_map_policy_comparison`.

The comparison runner is
`scripts/run_xunce_high_fidelity_real_map_comparison.py`, with config
`configs/xunce_high_fidelity_real_map_comparison_v1.json` and output root
`outputs/path_feedback_batch_xunce_high_fidelity_real_map_comparison_v1/`. It
compares the 巡策 sandbox candidate checkpoint against the incumbent
experimental policy checkpoint in read-only mode. When the expanded evidence is
valid and 巡策 has no worse/regression/fallback blocker while meeting efficiency
budgets, it writes
`next_required_change=xunce_default_policy_candidate_authorization_preflight`.
This still does not approve default-policy replacement, checkpoint publication,
real executor connection, online canary traffic, PPO training, or real-world
performance claims.

The first stage, `Global 99% Coverage Benchmark v1`, is implemented as a
deterministic contract benchmark, not as a frontier planner. Its runner is
`scripts/run_global_99_coverage_benchmark.py`, with shell entrypoint
`scripts/run_global_99_coverage_benchmark.sh`, default config
`configs/global_99_coverage_benchmark_v1.json`, and output root
`outputs/path_feedback_batch_global_99_coverage_benchmark_v1/`. It defines the
ROI contract, coverage denominator, reachable/safe cell ledger, budget model,
failure reason taxonomy, and summary fields such as
`target_coverage_rate=0.99`, `achieved_coverage_rate`,
`reachable_safe_cell_count`, `covered_reachable_safe_cell_count`,
`coverage_target_met`, and `infeasible_reason_codes`. When the benchmark
contract is valid, it sets
`next_required_change=frontier_coverage_planner_baseline`.

Run the first-stage benchmark with:

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest tests/test_global_99_coverage_benchmark.py -q
PYTHON=$PY bash scripts/run_global_99_coverage_benchmark.sh
jq '{status,reason_codes,target_coverage_rate,achieved_coverage_rate,coverage_target_met,next_required_change,default_policy_replacement_approved,real_executor_connection_approved}' \
  outputs/path_feedback_batch_global_99_coverage_benchmark_v1/global-99-coverage-benchmark-summary.json
```

`Frontier Coverage Planner Baseline v1` is implemented as a deterministic
non-learning planner over the same synthetic Global 99 contract. Its runner is
`scripts/run_frontier_coverage_planner_baseline.py`, with shell entrypoint
`scripts/run_frontier_coverage_planner_baseline.sh`, default config
`configs/frontier_coverage_planner_baseline_v1.json`, and output root
`outputs/path_feedback_batch_frontier_coverage_planner_baseline_v1/`. It ignores
source `coverage_events`, computes its own frontier/coverage-map/revisit
penalty/path-budget plan, and writes plan, ledger, budget audit, rejection, and
report artifacts. When the baseline closes cleanly, it sets
`next_required_change=coverage_memory_replanning_loop`.

Run the second-stage baseline with:

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest \
  tests/test_global_99_coverage_benchmark.py \
  tests/test_frontier_coverage_planner_baseline.py -q
PYTHON=$PY bash scripts/run_frontier_coverage_planner_baseline.sh
jq '{status,reason_codes,achieved_coverage_rate,coverage_target_met,next_required_change,publishes_checkpoint,replaces_default_policy,connects_real_executor}' \
  outputs/path_feedback_batch_frontier_coverage_planner_baseline_v1/frontier-coverage-planner-baseline-summary.json
```

`Coverage Memory + Replanning Loop v1` is implemented as a deterministic
memory-backed replanning loop over the same synthetic contract. Its runner is
`scripts/run_coverage_memory_replanning_loop.py`, shell entrypoint is
`scripts/run_coverage_memory_replanning_loop.sh`, default config is
`configs/coverage_memory_replanning_loop_v1.json`, and output root is
`outputs/path_feedback_batch_coverage_memory_replanning_loop_v1/`. It persists
coverage memory snapshots, replans toward still-uncovered frontier cells, writes
trace, snapshot, ledger, budget audit, rejection, and report artifacts, and
sets `next_required_change=policy_guided_global_coverage` when the memory loop
closes cleanly.

Run the third-stage replanning loop with:

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest \
  tests/test_global_99_coverage_benchmark.py \
  tests/test_frontier_coverage_planner_baseline.py \
  tests/test_coverage_memory_replanning_loop.py -q
PYTHON=$PY bash scripts/run_coverage_memory_replanning_loop.sh
jq '{status,reason_codes,achieved_coverage_rate,coverage_target_met,next_required_change,memory_resume_verified,publishes_checkpoint,replaces_default_policy,connects_real_executor}' \
  outputs/path_feedback_batch_coverage_memory_replanning_loop_v1/coverage-memory-replanning-loop-summary.json
```

`Policy-Guided Global Coverage v1` is implemented as a guarded, read-only
policy inference stage over the deterministic synthetic Global 99 contract. Its
runner is `scripts/run_policy_guided_global_coverage.py`, shell entrypoint is
`scripts/run_policy_guided_global_coverage.sh`, default config is
`configs/policy_guided_global_coverage_v1.json`, and output root is
`outputs/path_feedback_batch_policy_guided_global_coverage_v1/`. It loads the
current experimental checkpoint from
`outputs/path_feedback_batch_value_stability_candidate_v1/`, maps frontier
candidates into the `ModelExplorerContract` / `GoalCandidate` policy interface,
blends baseline frontier score with normalized policy logits, and records
policy-score, guard, budget, decision, ledger, memory snapshot, rejection, and
report artifacts. When the guarded policy-guided loop closes cleanly, it sets
`next_required_change=global_99_multi_map_generalization`.

Run the fourth-stage policy-guided loop with:

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest \
  tests/test_global_99_coverage_benchmark.py \
  tests/test_frontier_coverage_planner_baseline.py \
  tests/test_coverage_memory_replanning_loop.py \
  tests/test_policy_guided_global_coverage.py -q
PYTHON=$PY bash scripts/run_policy_guided_global_coverage.sh
jq '{status,reason_codes,achieved_coverage_rate,coverage_target_met,policy_loaded,policy_guidance_applied,policy_scored_candidate_count,policy_guard_fallback_count,next_required_change,publishes_checkpoint,replaces_default_policy,connects_real_executor,runs_new_ppo_update}' \
  outputs/path_feedback_batch_policy_guided_global_coverage_v1/policy-guided-global-coverage-summary.json
```

`99% Multi-Map Generalization v1` is implemented as a deterministic synthetic
multi-map validation matrix. Its runner is
`scripts/run_global_99_multi_map_generalization.py`, shell entrypoint is
`scripts/run_global_99_multi_map_generalization.sh`, default config is
`configs/global_99_multi_map_generalization_v1.json`, and output root is
`outputs/path_feedback_batch_global_99_multi_map_generalization_v1/`. It
validates `open_field`, `corridor`, `rooms`, `blocked_roi`, `unsafe_patch`,
`narrow_passage`, `cells_roi`, and expected-infeasible `budget_limited`
families, then writes scenario, family, policy-vs-baseline, budget, rejection,
manifest, summary, and report artifacts. When the required multi-map scenarios
close cleanly, it sets
`next_required_change=network_architecture_upgrade_readiness_review`.

Run the fifth-stage multi-map validation with:

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest \
  tests/test_global_99_coverage_benchmark.py \
  tests/test_frontier_coverage_planner_baseline.py \
  tests/test_coverage_memory_replanning_loop.py \
  tests/test_policy_guided_global_coverage.py \
  tests/test_global_99_multi_map_generalization.py -q
PYTHON=$PY bash scripts/run_global_99_multi_map_generalization.sh
jq '{status,reason_codes,scenario_count,passed_scenario_count,failed_scenario_count,aggregate_achieved_coverage_rate,min_scenario_achieved_coverage_rate,policy_guidance_applied,policy_scored_candidate_count,next_required_change,publishes_checkpoint,replaces_default_policy,connects_real_executor,runs_new_ppo_update}' \
  outputs/path_feedback_batch_global_99_multi_map_generalization_v1/global-99-multi-map-generalization-summary.json
```

`Network Architecture Upgrade Readiness Review v1` is implemented as an audit
gate, not a network upgrade. Its runner is
`scripts/run_network_architecture_upgrade_readiness_review.py`, shell
entrypoint is `scripts/run_network_architecture_upgrade_readiness_review.sh`,
default config is
`configs/network_architecture_upgrade_readiness_review_v1.json`, and output
root is
`outputs/path_feedback_batch_network_architecture_upgrade_readiness_review_v1/`.
It consumes the multi-map summary, family summary, policy-vs-baseline audit,
and scenario results, then writes readiness summary, evidence audit,
bottleneck attribution, recommendation report, manifest, and rejection report
artifacts. If the synthetic multi-map evidence remains stable, required
scenarios all pass, policy has positive divergence, policy worse count is zero,
controlled regression is zero, and fallback rate is low, the review defers
network changes and sets
`next_required_change=global_99_release_governance_preflight`.

Run the sixth-stage readiness review with:

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest \
  tests/test_global_99_coverage_benchmark.py \
  tests/test_frontier_coverage_planner_baseline.py \
  tests/test_coverage_memory_replanning_loop.py \
  tests/test_policy_guided_global_coverage.py \
  tests/test_global_99_multi_map_generalization.py \
  tests/test_network_architecture_upgrade_readiness_review.py -q
PYTHON=$PY bash scripts/run_network_architecture_upgrade_readiness_review.sh
jq '{status,reason_codes,network_upgrade_recommended,network_upgrade_readiness_decision,next_required_change,policy_guard_fallback_rate,baseline_agreement_rate,policy_better_than_baseline_count,policy_worse_than_baseline_count,controlled_regression_count,modifies_network,runs_new_ppo_update,publishes_checkpoint,replaces_default_policy}' \
  outputs/path_feedback_batch_network_architecture_upgrade_readiness_review_v1/network-architecture-upgrade-readiness-summary.json
```

`Global 99 Release Governance Preflight v1` is implemented as a release
governance audit over the synthetic Global 99 evidence. Its runner is
`scripts/run_global_99_release_governance_preflight.py`, shell entrypoint is
`scripts/run_global_99_release_governance_preflight.sh`, default config is
`configs/global_99_release_governance_preflight_v1.json`, and output root is
`outputs/path_feedback_batch_global_99_release_governance_preflight_v1/`. It
consumes network-readiness, multi-map, and policy-guided summaries, then writes
lineage, scope, boundary, kill-switch, rollback, telemetry, rejection, manifest,
summary, and report artifacts. This stage does not publish a checkpoint, install
or replace default policy, connect an executor, run PPO, modify
network/action-space/default A*, call path-planner, or use NPZ/sidecar maps.
When the governance preflight passes, it sets
`next_required_change=global_99_shadow_canary_preflight`.

Run the seventh-stage release governance preflight with:

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest \
  tests/test_global_99_coverage_benchmark.py \
  tests/test_frontier_coverage_planner_baseline.py \
  tests/test_coverage_memory_replanning_loop.py \
  tests/test_policy_guided_global_coverage.py \
  tests/test_global_99_multi_map_generalization.py \
  tests/test_network_architecture_upgrade_readiness_review.py \
  tests/test_global_99_release_governance_preflight.py -q
PYTHON=$PY bash scripts/run_global_99_release_governance_preflight.sh
jq '{status,reason_codes,release_governance_verdict,next_required_change,release_boundary_audit_passed,kill_switch_audit_passed,rollback_audit_passed,telemetry_audit_passed,publishes_checkpoint,replaces_default_policy,connects_real_executor,runs_new_ppo_update,modifies_network}' \
  outputs/path_feedback_batch_global_99_release_governance_preflight_v1/global-99-release-governance-preflight-summary.json
```

`Global 99 Shadow Canary Preflight v1` is implemented as an offline
shadow/canary eligibility gate, not as an online canary or release. Its runner
is `scripts/run_global_99_shadow_canary_preflight.py`, shell entrypoint is
`scripts/run_global_99_shadow_canary_preflight.sh`, default config is
`configs/global_99_shadow_canary_preflight_v1.json`, and output root is
`outputs/path_feedback_batch_global_99_shadow_canary_preflight_v1/`. It
consumes release-governance, multi-map, and policy-guided summaries, then
writes shadow replay, canary eligibility, boundary, kill-switch, rollback,
telemetry, rejection, manifest, summary, and report artifacts. `canary_mode`
only means eligibility auditing; `canary_traffic_fraction` remains `0.0` and
`starts_online_canary=false`. When this preflight passes, it sets
`next_required_change=global_99_shadow_canary_replay`.

Run the eighth-stage shadow/canary preflight with:

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest \
  tests/test_global_99_coverage_benchmark.py \
  tests/test_frontier_coverage_planner_baseline.py \
  tests/test_coverage_memory_replanning_loop.py \
  tests/test_policy_guided_global_coverage.py \
  tests/test_global_99_multi_map_generalization.py \
  tests/test_network_architecture_upgrade_readiness_review.py \
  tests/test_global_99_release_governance_preflight.py \
  tests/test_global_99_shadow_canary_preflight.py -q
PYTHON=$PY bash scripts/run_global_99_shadow_canary_preflight.sh
jq '{status,reason_codes,shadow_canary_preflight_verdict,next_required_change,shadow_replay_audit_passed,canary_eligibility_audit_passed,boundary_audit_passed,kill_switch_audit_passed,rollback_audit_passed,telemetry_audit_passed,publishes_checkpoint,replaces_default_policy,connects_real_executor,starts_online_canary,canary_traffic_fraction}' \
  outputs/path_feedback_batch_global_99_shadow_canary_preflight_v1/global-99-shadow-canary-preflight-summary.json
```

`Global 99 Shadow Canary Replay v1` is implemented as an offline synthetic
shadow/canary replay stage, not as an online canary or release. Its runner is
`scripts/run_global_99_shadow_canary_replay.py`, shell entrypoint is
`scripts/run_global_99_shadow_canary_replay.sh`, default config is
`configs/global_99_shadow_canary_replay_v1.json`, and output root is
`outputs/path_feedback_batch_global_99_shadow_canary_replay_v1/`. It consumes
shadow/canary preflight evidence and source multi-map artifacts, reruns the
deterministic synthetic multi-map policy-guided matrix, compares replayed
scenario coverage and path cost against source scenario results, and writes
scenario, family, source-match, policy-vs-baseline, boundary, kill-switch,
rollback, telemetry, rejection, manifest, summary, and report artifacts. When
the offline replay is deterministic and boundaries stay closed, it sets
`next_required_change=global_99_real_map_preflight`.

Run the ninth-stage shadow/canary replay with:

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest \
  tests/test_global_99_coverage_benchmark.py \
  tests/test_frontier_coverage_planner_baseline.py \
  tests/test_coverage_memory_replanning_loop.py \
  tests/test_policy_guided_global_coverage.py \
  tests/test_global_99_multi_map_generalization.py \
  tests/test_network_architecture_upgrade_readiness_review.py \
  tests/test_global_99_release_governance_preflight.py \
  tests/test_global_99_shadow_canary_preflight.py \
  tests/test_global_99_shadow_canary_replay.py -q
PYTHON=$PY bash scripts/run_global_99_shadow_canary_replay.sh
jq '{status,reason_codes,shadow_replay_passed,offline_canary_replay_passed,source_match_audit_passed,next_required_change,max_replay_coverage_delta,policy_guard_fallback_rate,publishes_checkpoint,replaces_default_policy,connects_real_executor,starts_online_canary,canary_traffic_fraction}' \
  outputs/path_feedback_batch_global_99_shadow_canary_replay_v1/global-99-shadow-canary-replay-summary.json
```

`Global 99 Real Map Preflight v1` is implemented as a quasi-real / real-map
evidence preflight audit, not as a real-map takeover, online canary, executor
connection, checkpoint publication, or default-policy installation. Its runner
is `scripts/run_global_99_real_map_preflight.py`, shell entrypoint is
`scripts/run_global_99_real_map_preflight.sh`, default config is
`configs/global_99_real_map_preflight_v1.json`, and output root is
`outputs/path_feedback_batch_global_99_real_map_preflight_v1/`. It consumes the
synthetic shadow/canary replay summary plus existing LOLA quasi-real ROI,
domain-gap, path-feedback, and sidecar evidence from
`outputs/path_feedback_batch_quasi_real_map_domain_gap_v1/` and optional
semi-real path-feedback validation context from
`outputs/path_feedback_validation/`. It writes evidence-lineage, domain-gap,
path-feedback, scope, boundary, kill-switch, rollback, telemetry, rejection,
manifest, summary, and report artifacts. When the preflight evidence is stable,
context IDs are present, no open-grid fallback or path-feedback regression is
observed, and all publication/executor/online-canary boundaries stay closed, it
sets `next_required_change=global_99_real_map_shadow_replay`.

Run the tenth-stage real-map preflight with:

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest \
  tests/test_global_99_coverage_benchmark.py \
  tests/test_frontier_coverage_planner_baseline.py \
  tests/test_coverage_memory_replanning_loop.py \
  tests/test_policy_guided_global_coverage.py \
  tests/test_global_99_multi_map_generalization.py \
  tests/test_network_architecture_upgrade_readiness_review.py \
  tests/test_global_99_release_governance_preflight.py \
  tests/test_global_99_shadow_canary_preflight.py \
  tests/test_global_99_shadow_canary_replay.py \
  tests/test_global_99_real_map_preflight.py -q
PYTHON=$PY bash scripts/run_global_99_real_map_preflight.sh
jq '{status,reason_codes,real_map_preflight_verdict,next_required_change,domain_gap_verdict,slice_count,roi_group_count,fallback_or_open_grid_count,boundary_audit_passed,real_world_release_approved,real_world_performance_claimed,replaces_default_policy,connects_real_executor,starts_online_canary,canary_traffic_fraction}' \
  outputs/path_feedback_batch_global_99_real_map_preflight_v1/global-99-real-map-preflight-summary.json
```

`Global 99 Real Map Shadow Replay v1` is implemented as an offline quasi-real
shadow replay over existing LOLA path-feedback evidence, not as an online
canary, executor connection, checkpoint publication, or default-policy
installation. Its runner is
`scripts/run_global_99_real_map_shadow_replay.py`, shell entrypoint is
`scripts/run_global_99_real_map_shadow_replay.sh`, default config is
`configs/global_99_real_map_shadow_replay_v1.json`, and output root is
`outputs/path_feedback_batch_global_99_real_map_shadow_replay_v1/`. It consumes
the real-map preflight summary and the frozen quasi-real path-feedback
manifest, summary, slices, contracts, and sidecars from
`outputs/path_feedback_batch_quasi_real_map_domain_gap_v1/`; it rewrites replay
outputs into the isolated replay root, validates and runs `model_explorer
path-feedback` in offline replay mode, then compares replay and source
scenario evidence by `scenario_id`. This stage permits offline path-feedback /
path-planner route replay only under `path_planner_use_scope =
offline_path_feedback_replay_only`; it keeps `connects_real_executor=false`,
`starts_online_canary=false`, `real_world_release_approved=false`, and
`real_world_performance_claimed=false`. When source-match, context, path
feedback, and boundary audits pass, it sets
`next_required_change=global_99_real_map_release_governance_preflight`.

Run the eleventh-stage real-map shadow replay with:

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest \
  tests/test_global_99_coverage_benchmark.py \
  tests/test_frontier_coverage_planner_baseline.py \
  tests/test_coverage_memory_replanning_loop.py \
  tests/test_policy_guided_global_coverage.py \
  tests/test_global_99_multi_map_generalization.py \
  tests/test_network_architecture_upgrade_readiness_review.py \
  tests/test_global_99_release_governance_preflight.py \
  tests/test_global_99_shadow_canary_preflight.py \
  tests/test_global_99_shadow_canary_replay.py \
  tests/test_global_99_real_map_preflight.py \
  tests/test_global_99_real_map_shadow_replay.py -q
PYTHON=$PY bash scripts/run_global_99_real_map_shadow_replay.sh
jq '{status,reason_codes,real_map_shadow_replay_verdict,next_required_change,source_match_audit_passed,scenario_mismatch_count,max_replay_coverage_delta,max_replay_path_cost_delta_m,open_grid_fallback_used,connects_real_executor,starts_online_canary,canary_traffic_fraction,replaces_default_policy}' \
  outputs/path_feedback_batch_global_99_real_map_shadow_replay_v1/global-99-real-map-shadow-replay-summary.json
```

`Global 99 Real Map Release Governance Preflight v1` is implemented as a
governance audit over quasi-real real-map replay evidence, not as a release or
installation. Its runner is
`scripts/run_global_99_real_map_release_governance_preflight.py`, shell
entrypoint is
`scripts/run_global_99_real_map_release_governance_preflight.sh`, default config
is `configs/global_99_real_map_release_governance_preflight_v1.json`, and
output root is
`outputs/path_feedback_batch_global_99_real_map_release_governance_preflight_v1/`.
It consumes Real Map Shadow Replay, Real Map Preflight, and quasi-real
domain-gap/path-feedback evidence, then writes lineage, shadow replay,
preflight, domain-gap, path-feedback, scope, boundary, kill-switch, rollback,
telemetry, rejection, manifest, summary, and report artifacts. This stage does
not invoke path-planner; it audits the upstream offline route replay scope as
`offline_path_feedback_replay_only`. When governance evidence and boundaries
are valid, it sets
`next_required_change=global_99_real_map_shadow_canary_preflight`.

Run the twelfth-stage real-map release governance preflight with:

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest \
  tests/test_global_99_coverage_benchmark.py \
  tests/test_frontier_coverage_planner_baseline.py \
  tests/test_coverage_memory_replanning_loop.py \
  tests/test_policy_guided_global_coverage.py \
  tests/test_global_99_multi_map_generalization.py \
  tests/test_network_architecture_upgrade_readiness_review.py \
  tests/test_global_99_release_governance_preflight.py \
  tests/test_global_99_shadow_canary_preflight.py \
  tests/test_global_99_shadow_canary_replay.py \
  tests/test_global_99_real_map_preflight.py \
  tests/test_global_99_real_map_shadow_replay.py \
  tests/test_global_99_real_map_release_governance_preflight.py -q
PYTHON=$PY bash scripts/run_global_99_real_map_release_governance_preflight.sh
jq '{status,reason_codes,real_map_release_governance_verdict,next_required_change,source_match_audit_passed,scenario_mismatch_count,open_grid_fallback_used,policy_guard_fallback_rate,release_boundary_audit_passed,connects_real_executor,starts_online_canary,replaces_default_policy,uses_path_planner,audited_path_planner_use_scope}' \
  outputs/path_feedback_batch_global_99_real_map_release_governance_preflight_v1/global-99-real-map-release-governance-preflight-summary.json
```

`Global 99 Real Map Shadow Canary Preflight v1` is implemented as an offline
real-map shadow/canary eligibility gate, not as an online canary. Its runner is
`scripts/run_global_99_real_map_shadow_canary_preflight.py`, shell entrypoint is
`scripts/run_global_99_real_map_shadow_canary_preflight.sh`, default config is
`configs/global_99_real_map_shadow_canary_preflight_v1.json`, and output root
is `outputs/path_feedback_batch_global_99_real_map_shadow_canary_preflight_v1/`.
It consumes real-map release-governance and real-map shadow-replay summaries,
requires source-match evidence, guard fallback, and release boundaries to stay
valid, and keeps `canary_traffic_fraction=0.0` plus
`starts_online_canary=false`. When eligible, it sets
`next_required_change=global_99_real_map_shadow_canary_replay`.

Run the thirteenth-stage real-map shadow/canary preflight with:

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest \
  tests/test_global_99_real_map_shadow_canary_preflight.py -q
PYTHON=$PY bash scripts/run_global_99_real_map_shadow_canary_preflight.sh
jq '{status,reason_codes,real_map_shadow_canary_preflight_verdict,next_required_change,source_match_audit_passed,scenario_mismatch_count,open_grid_fallback_used,policy_guard_fallback_rate,boundary_audit_passed,starts_online_canary,canary_traffic_fraction,connects_real_executor,replaces_default_policy}' \
  outputs/path_feedback_batch_global_99_real_map_shadow_canary_preflight_v1/global-99-real-map-shadow-canary-preflight-summary.json
```

`Global 99 Real Map Shadow Canary Replay v1` is implemented as an offline
real-map shadow/canary replay audit, not as an online canary. Its runner is
`scripts/run_global_99_real_map_shadow_canary_replay.py`, shell entrypoint is
`scripts/run_global_99_real_map_shadow_canary_replay.sh`, default config is
`configs/global_99_real_map_shadow_canary_replay_v1.json`, and output root is
`outputs/path_feedback_batch_global_99_real_map_shadow_canary_replay_v1/`. It
consumes real-map shadow/canary preflight, real-map release-governance, and
real-map shadow-replay summaries, compares deterministic source-match evidence,
fallback, telemetry, rollback, and boundary state, and keeps online canary
traffic at zero. When replay evidence is stable, it sets
`next_required_change=global_99_real_map_evidence_refresh_drift_audit`.

Run the fourteenth-stage real-map shadow/canary replay with:

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest \
  tests/test_global_99_real_map_shadow_canary_replay.py -q
PYTHON=$PY bash scripts/run_global_99_real_map_shadow_canary_replay.sh
jq '{status,reason_codes,real_map_shadow_canary_replay_verdict,next_required_change,source_match_audit_passed,scenario_mismatch_count,max_replay_coverage_delta,max_replay_path_cost_delta_m,open_grid_fallback_used,policy_guard_fallback_rate,boundary_audit_passed,starts_online_canary,canary_traffic_fraction,connects_real_executor,replaces_default_policy}' \
  outputs/path_feedback_batch_global_99_real_map_shadow_canary_replay_v1/global-99-real-map-shadow-canary-replay-summary.json
```

`Global 99 Real Map Evidence Refresh / Drift Audit v1` is implemented as an
evidence refresh and drift gate, not as a new planner run. Its runner is
`scripts/run_global_99_real_map_evidence_refresh_drift_audit.py`, shell
entrypoint is `scripts/run_global_99_real_map_evidence_refresh_drift_audit.sh`,
default config is `configs/global_99_real_map_evidence_refresh_drift_audit_v1.json`,
and output root is
`outputs/path_feedback_batch_global_99_real_map_evidence_refresh_drift_audit_v1/`.
It audits the approved shadow/canary replay summary plus quasi-real
domain-gap/path-feedback manifest, slices, contracts, sidecars, context IDs,
source-match deltas, fingerprints, fallback counters, and boundary state. It
does not rerun PPO, publish checkpoints, replace default policy, connect a real
executor, or call path-planner; the only accepted path-planner evidence is the
upstream offline replay scope. When stable, it sets
`next_required_change=global_99_real_map_multi_roi_generalization`.

Run the fifteenth-stage real-map evidence drift audit with:

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest \
  tests/test_global_99_real_map_evidence_refresh_drift_audit.py -q
PYTHON=$PY bash scripts/run_global_99_real_map_evidence_refresh_drift_audit.sh
jq '{status,reason_codes,evidence_refresh_drift_verdict,next_required_change,source_domain_gap_status,domain_gap_verdict,slice_count,roi_group_count,scenario_id_mismatch_count,missing_contract_count,missing_sidecar_count,context_id_missing_count,legacy_identity_fallback_count,boundary_audit_passed,connects_real_executor,starts_online_canary,replaces_default_policy,uses_path_planner}' \
  outputs/path_feedback_batch_global_99_real_map_evidence_refresh_drift_audit_v1/global-99-real-map-evidence-refresh-drift-audit-summary.json
```

`Global 99 Real Map Multi-ROI Generalization v1` is implemented as a
quasi-real ROI/slice/split matrix audit, not as a policy installation. Its
runner is `scripts/run_global_99_real_map_multi_roi_generalization.py`, shell
entrypoint is `scripts/run_global_99_real_map_multi_roi_generalization.sh`,
default config is `configs/global_99_real_map_multi_roi_generalization_v1.json`,
and output root is
`outputs/path_feedback_batch_global_99_real_map_multi_roi_generalization_v1/`.
It consumes the evidence drift audit and quasi-real domain-gap slices, checks
ROI-group coverage, train/validation/test split coverage, context IDs,
contract/sidecar availability, fallback/regression counters, and boundary
state. When all required ROI groups pass, it sets
`next_required_change=default_policy_candidate_authorization_preflight`.

Run the sixteenth-stage real-map multi-ROI audit with:

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest \
  tests/test_global_99_real_map_multi_roi_generalization.py -q
PYTHON=$PY bash scripts/run_global_99_real_map_multi_roi_generalization.sh
jq '{status,reason_codes,real_map_multi_roi_generalization_verdict,next_required_change,slice_count,roi_group_count,passed_roi_group_count,failed_roi_group_count,passed_required_scenario_count,failed_required_scenario_count,split_coverage_complete,context_id_missing_count,missing_contract_count,missing_sidecar_count,boundary_audit_passed,connects_real_executor,starts_online_canary,replaces_default_policy}' \
  outputs/path_feedback_batch_global_99_real_map_multi_roi_generalization_v1/global-99-real-map-multi-roi-generalization-summary.json
```

`Global 99 Default Policy Candidate Authorization Preflight v1` is implemented
as a candidate authorization gate, not as installation. Its runner is
`scripts/run_global_99_default_policy_candidate_authorization_preflight.py`,
shell entrypoint is
`scripts/run_global_99_default_policy_candidate_authorization_preflight.sh`,
default config is
`configs/global_99_default_policy_candidate_authorization_preflight_v1.json`,
and output root is
`outputs/path_feedback_batch_global_99_default_policy_candidate_authorization_preflight_v1/`.
It consumes the Global 99 multi-ROI summary, verifies candidate/default-policy
read-only boundaries, executor/path-planner isolation, kill-switch, rollback,
telemetry, and release boundaries. Passing this stage only authorizes a sandbox
dry run and sets `next_required_change=sandbox_candidate_installation_dry_run`.

Run the seventeenth-stage authorization preflight with:

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest \
  tests/test_global_99_default_policy_candidate_authorization_preflight.py -q
PYTHON=$PY bash scripts/run_global_99_default_policy_candidate_authorization_preflight.sh
jq '{status,reason_codes,authorization_verdict,next_required_change,source_multi_roi_status,default_policy_candidate_authorization_preflight_passed,candidate_read_only_audit_passed,default_policy_read_only_audit_passed,executor_isolation_audit_passed,path_planner_isolation_audit_passed,kill_switch_audit_passed,rollback_audit_passed,telemetry_audit_passed,boundary_audit_passed,replaces_default_policy,connects_real_executor,default_policy_candidate_installation_approved}' \
  outputs/path_feedback_batch_global_99_default_policy_candidate_authorization_preflight_v1/global-99-default-policy-candidate-authorization-preflight-summary.json
```

`Global 99 Sandbox Candidate Installation Dry Run v1` is implemented as a
sandbox-only install rehearsal. Its runner is
`scripts/run_global_99_sandbox_candidate_installation_dry_run.py`, shell
entrypoint is `scripts/run_global_99_sandbox_candidate_installation_dry_run.sh`,
default config is `configs/global_99_sandbox_candidate_installation_dry_run_v1.json`,
and output root is
`outputs/path_feedback_batch_global_99_sandbox_candidate_installation_dry_run_v1/`.
It consumes the Global 99 authorization preflight, writes a sandbox-only
candidate descriptor, verifies hash/provenance/load/rollback/kill-switch and
telemetry, and keeps default policy untouched. Passing this stage sets
`next_required_change=sandbox_consumer_replay_canary`.

Run the eighteenth-stage sandbox installation dry run with:

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest \
  tests/test_global_99_sandbox_candidate_installation_dry_run.py -q
PYTHON=$PY bash scripts/run_global_99_sandbox_candidate_installation_dry_run.sh
jq '{status,reason_codes,sandbox_installation_verdict,next_required_change,source_authorization_status,sandbox_candidate_installation_dry_run_passed,sandbox_candidate_hash_audit_passed,sandbox_candidate_load_audit_passed,rollback_audit_passed,kill_switch_audit_passed,telemetry_audit_passed,boundary_audit_passed,replaces_default_policy,connects_real_executor,default_policy_candidate_installation_approved}' \
  outputs/path_feedback_batch_global_99_sandbox_candidate_installation_dry_run_v1/global-99-sandbox-candidate-installation-dry-run-summary.json
```

`Global 99 Sandbox Consumer Replay / Canary v1` is implemented as offline
sandbox consumer validation, not an online canary. Its runner is
`scripts/run_global_99_sandbox_consumer_replay_canary.py`, shell entrypoint is
`scripts/run_global_99_sandbox_consumer_replay_canary.sh`, default config is
`configs/global_99_sandbox_consumer_replay_canary_v1.json`, and output root is
`outputs/path_feedback_batch_global_99_sandbox_consumer_replay_canary_v1/`.
It consumes the sandbox installation dry run, generates a sandbox replay trace,
checks candidate load, fallback availability, telemetry, rollback, and boundary
state. Passing this stage sets
`next_required_change=controlled_default_policy_candidate_installation_preflight`.

Run the nineteenth-stage sandbox consumer replay/canary with:

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest \
  tests/test_global_99_sandbox_consumer_replay_canary.py -q
PYTHON=$PY bash scripts/run_global_99_sandbox_consumer_replay_canary.sh
jq '{status,reason_codes,sandbox_consumer_verdict,next_required_change,source_sandbox_install_status,sandbox_consumer_replay_canary_passed,consumer_step_count,fallback_rate,controlled_regression_count,candidate_load_audit_passed,fallback_audit_passed,telemetry_audit_passed,rollback_audit_passed,boundary_audit_passed,replaces_default_policy,connects_real_executor,starts_online_canary,default_policy_candidate_installation_approved}' \
  outputs/path_feedback_batch_global_99_sandbox_consumer_replay_canary_v1/global-99-sandbox-consumer-replay-canary-summary.json
```

`Global 99 Controlled Default Policy Candidate Installation Preflight v1` is
implemented as the final controlled-installation review gate. Its runner is
`scripts/run_global_99_controlled_default_policy_candidate_installation_preflight.py`,
shell entrypoint is
`scripts/run_global_99_controlled_default_policy_candidate_installation_preflight.sh`,
default config is
`configs/global_99_controlled_default_policy_candidate_installation_preflight_v1.json`,
and output root is
`outputs/path_feedback_batch_global_99_controlled_default_policy_candidate_installation_preflight_v1/`.
It consumes sandbox consumer replay/canary evidence and audits review scope,
default-policy read-only status, executor isolation, online-canary disabled
state, kill-switch, rollback, telemetry, and release boundaries. Passing this
stage means the candidate is eligible for controlled installation review only;
`controlled_installation_executed=false` and default policy is not replaced.

Run the final controlled installation preflight with:

```bash
PY=/home/kai/anaconda3/envs/lunar-explorer/bin/python
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $PY -m pytest \
  tests/test_global_99_controlled_default_policy_candidate_installation_preflight.py -q
PYTHON=$PY bash scripts/run_global_99_controlled_default_policy_candidate_installation_preflight.sh
jq '{status,reason_codes,controlled_installation_verdict,next_required_change,source_sandbox_consumer_status,controlled_default_policy_candidate_installation_preflight_passed,review_scope_audit_passed,default_policy_read_only_audit_passed,executor_isolation_audit_passed,online_canary_audit_passed,controlled_installation_executed,replaces_default_policy,connects_real_executor,starts_online_canary}' \
  outputs/path_feedback_batch_global_99_controlled_default_policy_candidate_installation_preflight_v1/global-99-controlled-default-policy-candidate-installation-preflight-summary.json
```

`Network Architecture Upgrade v1` starts only if the readiness review recommends
it because the evidence points to policy regression, excessive guard fallback,
or another network-expression bottleneck. Candidate upgrades include residual
MLP, gated MLP, candidate-set encoder, lightweight attention, and graph/region
encoder variants. If the review does not recommend a network upgrade, the next
step is real-map preflight / real-map shadow replay, not a direct
default-policy replacement.
`99% Coverage Release Governance v1` starts only after the 99% reachable
coverage target is evidence-backed in shadow/canary form.

This line must not disturb the family-balanced publication chain. It does not
replace default policy, connect a real executor, relax guards, modify the
default A* path, change action space, claim Ackermann-feasible trajectory, or
treat IRIS/GCS/path-planner diagnostics as release proof.

## Core Algorithm Development Chain

The next implementation stages should follow:

```text
A* geometric path baseline
  -> opt-in channel-aware A* seed path
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

The immediate algorithm target is **Channel-Aware A* Seed Path v1**. This target
moves the current high-cost exposure evidence upstream into path generation:
instead of finding only the lowest-cost centerline, the opt-in backend searches
for a seed path whose surrounding corridor/channel is lower risk. It must keep
`path-planner-route/v1`, the default A* baseline, `trajectory_kind=geometric_path`,
PPO, network architecture, and candidate-list action space stable.

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
