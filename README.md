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
