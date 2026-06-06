import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


class BatchPathFeedbackValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.script = self.repo_root / "scripts" / "run_batch_path_feedback_validation.sh"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="path-feedback-batch-"))
        self.fake_single_run_log = self.temp_dir / "single-run.log"
        self.fake_single_run = self._write_fake_single_run_script(self.temp_dir / "fake_single_run.sh")

    def _write_matrix(self, payload: dict) -> Path:
        path = self.temp_dir / "matrix.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def _run_batch(self, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["FAKE_SINGLE_RUN_LOG"] = str(self.fake_single_run_log)
        return subprocess.run(
            ["bash", str(self.script), *args],
            cwd=self.repo_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def _write_fake_single_run_script(self, path: Path) -> Path:
        path.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env bash
                set -euo pipefail

                python_bin="${PYTHON:-/home/kai/anaconda3/envs/lunar-explorer/bin/python}"
                output_root=""
                scenario_set=""
                diagnostic_profile=""
                top_k=""
                control_point_requested=0
                while [[ $# -gt 0 ]]; do
                  case "$1" in
                    --output-root)
                      output_root="$2"
                      shift 2
                      ;;
                    --scenario-set)
                      scenario_set="$2"
                      shift 2
                      ;;
                    --diagnostic-profile)
                      diagnostic_profile="$2"
                      shift 2
                      ;;
                    --top-k)
                      top_k="$2"
                      shift 2
                      ;;
                    --simulate-tracking|--optimize-trajectory|--drake-iris-regions|--gcs-trajectory-smoke|--gcs-geometric-candidate|--gcs-motion-feasibility|--gcs-curvature-constrained-candidate)
                      shift
                      ;;
                    --gcs-control-point-candidate)
                      control_point_requested=1
                      shift
                      ;;
                    *)
                      echo "unexpected argument: $1" >&2
                      exit 2
                      ;;
                  esac
                done

                if [[ -z "$output_root" || -z "$scenario_set" || -z "$diagnostic_profile" || -z "$top_k" ]]; then
                  echo "missing required fake single-run argument" >&2
                  exit 2
                fi

                run_id="$(basename "$output_root")"
                if [[ -n "${FAKE_SINGLE_RUN_LOG:-}" ]]; then
                  printf '%s|%s|%s|%s|%s\\n' "$run_id" "$scenario_set" "$diagnostic_profile" "$top_k" "$output_root" >> "$FAKE_SINGLE_RUN_LOG"
                fi

                mkdir -p "$output_root/maps" "$output_root/path_planner_sidecars"
                if [[ "$run_id" == "exit-fail" ]]; then
                  exit 7
                fi

                "$python_bin" - "$output_root" "$scenario_set" "$diagnostic_profile" "$top_k" "$control_point_requested" <<'PY'
                import json
                import sys
                from pathlib import Path

                output_root = Path(sys.argv[1])
                scenario_set = sys.argv[2]
                diagnostic_profile = sys.argv[3]
                top_k = int(sys.argv[4])
                control_point_requested = sys.argv[5] == "1"
                run_id = output_root.name
                open_grid = run_id == "open-grid-fail"

                def group_payload(group):
                    failure_count = top_k if group == "stress" else 0
                    return {
                        "scenario_count": 1,
                        "candidate_count": top_k,
                        "reachable_count": 0 if group == "stress" else top_k,
                        "failure_count": failure_count,
                        "replan_count": failure_count,
                        "selection_changed_count": 1,
                        "iris_report_count": top_k,
                        "iris_fallback_count": top_k,
                        "region_graph_fallback_count": top_k,
                        "region_graph_start_goal_disconnected_count": failure_count,
                    }

                if scenario_set == "all":
                    scenario_groups = {
                        "smoke": group_payload("smoke"),
                        "stress": group_payload("stress"),
                    }
                else:
                    scenario_groups = {scenario_set: group_payload(scenario_set)}

                path_failures = sum(item["failure_count"] for item in scenario_groups.values())
                replans = sum(item["replan_count"] for item in scenario_groups.values())
                iris_fallback = sum(item["iris_fallback_count"] for item in scenario_groups.values())
                region_fallback = sum(item["region_graph_fallback_count"] for item in scenario_groups.values())
                region_disconnect = sum(
                    item["region_graph_start_goal_disconnected_count"]
                    for item in scenario_groups.values()
                )
                sampled_selected = sum(
                    item["candidate_count"] for group, item in scenario_groups.items() if group != "stress"
                )
                sampled_fallback = sum(
                    item["candidate_count"] for group, item in scenario_groups.items() if group == "stress"
                )
                sampled_attempts = sampled_selected * 4 + sampled_fallback * 3
                sampled_rankings = sampled_selected + sampled_fallback
                sampled_anchor_added = sampled_selected
                sampled_anchor_connected = sampled_selected
                sampled_bridge_aware_connector_attempts = sampled_selected
                sampled_bridge_corridor_connector_attempts = sampled_selected
                sampled_connector_attempts = (
                    sampled_rankings
                    + sampled_bridge_aware_connector_attempts
                    + sampled_bridge_corridor_connector_attempts
                )
                sampled_terminal_candidate_count = sampled_selected * 2
                sampled_reachable_component_disconnected = sampled_fallback
                sampled_reachable_component_replacements = sampled_selected
                sampled_reachable_terminal_rescues = sampled_selected
                sampled_proxy_goal_anchor_selected = sampled_selected
                sampled_goal_rescue_candidates = sampled_selected * 2
                sampled_benefit_surface_present = sampled_selected
                sampled_path_duplicates = sampled_fallback
                sampled_baseline_equivalent = sampled_fallback
                sampled_no_quality_gain = sampled_fallback
                sampled_fixture_no_benefit = sampled_fallback
                sampled_candidate_missing_metrics = sampled_fallback
                sampled_constrained_connector_failed = sampled_fallback
                convex_report_count = sampled_selected + sampled_fallback
                convex_region_count_total = convex_report_count * 2
                convex_fallback_used = sampled_fallback
                convex_gcs_ready = convex_report_count
                convex_adjacent_overlaps = convex_report_count
                gcs_report_count = (
                    convex_report_count if diagnostic_profile in {"iris", "all"} else 0
                )
                gcs_success_count = sampled_selected if gcs_report_count else 0
                gcs_collision_count = sampled_fallback if gcs_report_count else 0
                gcs_candidate_report_count = gcs_report_count
                gcs_candidate_attempted_count = gcs_report_count
                gcs_candidate_available_count = gcs_success_count
                gcs_candidate_selected_count = gcs_success_count
                gcs_candidate_collision_count = gcs_collision_count
                gcs_motion_report_count = gcs_report_count
                gcs_motion_evaluated_count = gcs_success_count
                gcs_motion_feasible_count = gcs_success_count
                gcs_motion_infeasible_count = 0
                gcs_motion_diagnostic_only_count = gcs_collision_count
                gcs_curvature_report_count = gcs_report_count
                gcs_curvature_attempted_count = gcs_report_count
                gcs_curvature_available_count = gcs_success_count
                gcs_curvature_selected_count = gcs_success_count
                gcs_curvature_repair_success_count = 0
                gcs_curvature_infeasible_count = 0
                gcs_curvature_diagnostic_only_count = gcs_collision_count
                gcs_control_point_report_count = gcs_report_count if control_point_requested else 0
                gcs_control_point_attempted_count = gcs_control_point_report_count
                gcs_control_point_success_count = gcs_success_count if control_point_requested else 0
                gcs_control_point_selected_count = gcs_success_count if control_point_requested else 0
                gcs_control_point_fallback_count = gcs_collision_count if control_point_requested else 0
                gcs_control_point_terrain_count = gcs_success_count if control_point_requested else 0
                gate = {
                    "status": "failed" if open_grid else "passed",
                    "expected": False,
                    "actual": open_grid,
                    "reason_codes": ["open_grid_fallback_used"] if open_grid else ["open_grid_fallback_not_used"],
                }
                acceptance_metadata = {
                    "schema_version": "path-feedback-acceptance-metadata/v1",
                    "scenario_set": scenario_set,
                    "diagnostic_profile": diagnostic_profile,
                    "acceptance_gate": "semi-real-closed-loop" if scenario_set == "all" and diagnostic_profile == "all" and top_k == 3 else "custom",
                    "top_k": top_k,
                    "python_executable": sys.executable,
                    "planner_extra_args": [],
                    "open_grid_fallback_used": open_grid,
                    "open_grid_fallback_used_gate": gate,
                }
                summary = {
                    "schema_version": "path-feedback-summary/v1",
                    "scenario_count": sum(item["scenario_count"] for item in scenario_groups.values()),
                    "scenario_set": scenario_set,
                    "diagnostic_profile": diagnostic_profile,
                    "acceptance_gate": acceptance_metadata["acceptance_gate"],
                    "top_k": top_k,
                    "candidate_count": sum(item["candidate_count"] for item in scenario_groups.values()),
                    "path_planning_failure_count": path_failures,
                    "replan_count": replans,
                    "iris_fallback_count": iris_fallback,
                    "region_graph_fallback_count": region_fallback,
                    "region_graph_start_goal_disconnected_count": region_disconnect,
                    "region_graph_disconnected_count": region_disconnect,
                    "convex_region_report_count": convex_report_count,
                    "convex_region_count_total": convex_region_count_total,
                    "convex_region_backend_counts": (
                        {"workspace_iris": sampled_selected, "fallback_box": sampled_fallback}
                        if sampled_fallback
                        else {"workspace_iris": sampled_selected}
                    ),
                    "convex_region_fallback_used_count": convex_fallback_used,
                    "convex_region_gcs_ready_count": convex_gcs_ready,
                    "convex_region_blocked_cell_violation_count": 0,
                    "convex_region_coverage_status_counts": {"covered": convex_report_count},
                    "convex_region_gcs_ready_reason_counts": {
                        "convex_region_sequence_ready": convex_report_count
                    },
                    "convex_region_start_contained_count": convex_report_count,
                    "convex_region_goal_contained_count": convex_report_count,
                    "convex_region_adjacent_overlap_count": convex_adjacent_overlaps,
                    "convex_region_portal_count": 0,
                    "convex_region_candidate_audit": [
                        {
                            "scenario_id": "fake",
                            "action_index": index,
                            "backend": "workspace_iris" if index < sampled_selected else "fallback_box",
                            "region_count": 2,
                            "fallback_used": index >= sampled_selected,
                            "coverage_status": "covered",
                            "gcs_ready": True,
                            "gcs_ready_reason": "convex_region_sequence_ready",
                        }
                        for index in range(convex_report_count)
                    ],
                    "gcs_trajectory_report_count": gcs_report_count,
                    "gcs_trajectory_attempted_count": gcs_report_count,
                    "gcs_trajectory_success_count": gcs_success_count,
                    "gcs_trajectory_collision_count": gcs_collision_count,
                    "gcs_trajectory_region_count_total": gcs_report_count * 2,
                    "gcs_trajectory_sample_count_total": gcs_report_count * 5,
                    "gcs_trajectory_backend_counts": (
                        {"pydrake_gcs": gcs_report_count} if gcs_report_count else {}
                    ),
                    "gcs_trajectory_reason_counts": (
                        {
                            "gcs_trajectory_solution_found": gcs_success_count,
                            "sampled_trajectory_collision": gcs_collision_count,
                        }
                        if gcs_report_count
                        else {}
                    ),
                    "gcs_trajectory_result_status_counts": (
                        {
                            "SolutionResult.kSolutionFound": gcs_success_count,
                            "sampled_trajectory_collision": gcs_collision_count,
                        }
                        if gcs_report_count
                        else {}
                    ),
                    "gcs_trajectory_candidate_audit": [
                        {
                            "scenario_id": "fake",
                            "action_index": index,
                            "backend": "pydrake_gcs",
                            "attempted": True,
                            "success": index < gcs_success_count,
                            "reason": (
                                "gcs_trajectory_solution_found"
                                if index < gcs_success_count
                                else "sampled_trajectory_collision"
                            ),
                            "collision_count": 0 if index < gcs_success_count else 1,
                            "region_count": 2,
                        }
                        for index in range(gcs_report_count)
                    ],
                    "gcs_candidate_report_count": gcs_candidate_report_count,
                    "gcs_candidate_attempted_count": gcs_candidate_attempted_count,
                    "gcs_candidate_available_count": gcs_candidate_available_count,
                    "gcs_candidate_selected_count": gcs_candidate_selected_count,
                    "gcs_candidate_collision_count": gcs_candidate_collision_count,
                    "gcs_candidate_fallback_reason_counts": (
                        {"sampled_trajectory_collision": gcs_candidate_collision_count}
                        if gcs_candidate_collision_count
                        else {}
                    ),
                    "gcs_candidate_selection_reason_counts": (
                        {"gcs_candidate_quality_improved": gcs_candidate_selected_count}
                        if gcs_candidate_selected_count
                        else {}
                    ),
                    "gcs_candidate_cost_delta_vs_baseline_negative_count": gcs_candidate_selected_count,
                    "gcs_candidate_cost_delta_vs_baseline_positive_count": 0,
                    "gcs_candidate_cost_delta_vs_baseline_zero_count": 0,
                    "gcs_candidate_audit": [
                        {
                            "scenario_id": "fake",
                            "action_index": index,
                            "attempted": True,
                            "available": index < gcs_candidate_available_count,
                            "selected": index < gcs_candidate_selected_count,
                            "selection_reason": (
                                "gcs_candidate_quality_improved"
                                if index < gcs_candidate_selected_count
                                else None
                            ),
                            "fallback_reason": (
                                None
                                if index < gcs_candidate_available_count
                                else "sampled_trajectory_collision"
                            ),
                            "collision_count": 0 if index < gcs_candidate_available_count else 1,
                            "cost_delta_vs_baseline": (
                                -1.0 if index < gcs_candidate_selected_count else None
                            ),
                        }
                        for index in range(gcs_candidate_report_count)
                    ],
                    "gcs_motion_feasibility_report_count": gcs_motion_report_count,
                    "gcs_motion_feasibility_evaluated_count": gcs_motion_evaluated_count,
                    "gcs_motion_feasibility_feasible_count": gcs_motion_feasible_count,
                    "gcs_motion_feasibility_infeasible_count": gcs_motion_infeasible_count,
                    "gcs_motion_feasibility_diagnostic_only_count": gcs_motion_diagnostic_only_count,
                    "gcs_motion_feasibility_curvature_violation_count": 0,
                    "gcs_motion_feasibility_heading_violation_count": 0,
                    "gcs_motion_feasibility_status_counts": (
                        {
                            "feasible": gcs_motion_feasible_count,
                            "diagnostic_only": gcs_motion_diagnostic_only_count,
                        }
                        if gcs_motion_report_count
                        else {}
                    ),
                    "gcs_motion_feasibility_fallback_reason_counts": (
                        {"gcs_trajectory_failed": gcs_motion_diagnostic_only_count}
                        if gcs_motion_diagnostic_only_count
                        else {}
                    ),
                    "gcs_motion_feasibility_motion_model_counts": (
                        {"curvature_bounded": gcs_motion_report_count} if gcs_motion_report_count else {}
                    ),
                    "gcs_motion_feasibility_audit": [
                        {
                            "scenario_id": "fake",
                            "action_index": index,
                            "evaluated": index < gcs_motion_evaluated_count,
                            "trajectory_source": "gcs_trajectory_sampled_points",
                            "motion_model": "curvature_bounded",
                            "feasibility_status": (
                                "feasible"
                                if index < gcs_motion_evaluated_count
                                else "diagnostic_only"
                            ),
                            "fallback_reason": (
                                None
                                if index < gcs_motion_evaluated_count
                                else "gcs_trajectory_failed"
                            ),
                            "curvature_violation_count": 0,
                            "heading_violation_count": 0,
                            "sample_count": 5 if index < gcs_motion_evaluated_count else 0,
                        }
                        for index in range(gcs_motion_report_count)
                    ],
                    "gcs_curvature_constrained_report_count": gcs_curvature_report_count,
                    "gcs_curvature_constrained_attempted_count": gcs_curvature_attempted_count,
                    "gcs_curvature_constrained_available_count": gcs_curvature_available_count,
                    "gcs_curvature_constrained_selected_count": gcs_curvature_selected_count,
                    "gcs_curvature_constrained_repair_success_count": gcs_curvature_repair_success_count,
                    "gcs_curvature_constrained_infeasible_count": gcs_curvature_infeasible_count,
                    "gcs_curvature_constrained_diagnostic_only_count": gcs_curvature_diagnostic_only_count,
                    "gcs_curvature_constrained_curvature_violation_count_before": 0,
                    "gcs_curvature_constrained_curvature_violation_count_after": 0,
                    "gcs_curvature_constrained_heading_violation_count_before": 0,
                    "gcs_curvature_constrained_heading_violation_count_after": 0,
                    "gcs_curvature_constrained_collision_count": gcs_collision_count,
                    "gcs_curvature_constrained_region_containment_violation_count": 0,
                    "gcs_curvature_constrained_status_before_counts": (
                        {
                            "feasible": gcs_success_count,
                            "diagnostic_only": gcs_collision_count,
                        }
                        if gcs_curvature_report_count
                        else {}
                    ),
                    "gcs_curvature_constrained_status_after_counts": (
                        {
                            "feasible": gcs_success_count,
                            "diagnostic_only": gcs_collision_count,
                        }
                        if gcs_curvature_report_count
                        else {}
                    ),
                    "gcs_curvature_constrained_repair_strategy_counts": (
                        {
                            "none_required": gcs_success_count,
                            "not_attempted": gcs_collision_count,
                        }
                        if gcs_curvature_report_count
                        else {}
                    ),
                    "gcs_curvature_constrained_fallback_reason_counts": (
                        {"gcs_trajectory_failed": gcs_collision_count}
                        if gcs_collision_count
                        else {}
                    ),
                    "gcs_curvature_constrained_audit": [
                        {
                            "scenario_id": "fake",
                            "action_index": index,
                            "attempted": True,
                            "available": index < gcs_success_count,
                            "selected": index < gcs_success_count,
                            "repair_success": False,
                            "repair_strategy": (
                                "none_required" if index < gcs_success_count else "not_attempted"
                            ),
                            "status_before": (
                                "feasible" if index < gcs_success_count else "diagnostic_only"
                            ),
                            "status_after": (
                                "feasible" if index < gcs_success_count else "diagnostic_only"
                            ),
                            "fallback_reason": (
                                None if index < gcs_success_count else "gcs_trajectory_failed"
                            ),
                            "curvature_violation_count_before": 0,
                            "curvature_violation_count_after": 0,
                            "heading_violation_count_before": 0,
                            "heading_violation_count_after": 0,
                            "collision_count": 0 if index < gcs_success_count else 1,
                            "region_containment_violation_count": 0,
                        }
                        for index in range(gcs_curvature_report_count)
                    ],
                    "gcs_control_point_report_count": gcs_control_point_report_count,
                    "gcs_control_point_attempted_count": gcs_control_point_attempted_count,
                    "gcs_control_point_success_count": gcs_control_point_success_count,
                    "gcs_control_point_backend_counts": (
                        {
                            "pydrake_control_point_direction_cone_program": (
                                gcs_control_point_report_count
                            )
                        }
                        if gcs_control_point_report_count
                        else {}
                    ),
                    "gcs_control_point_candidate_selected_count": gcs_control_point_selected_count,
                    "gcs_control_point_candidate_fallback_reason_counts": (
                        {"sampled_trajectory_collision": gcs_control_point_fallback_count}
                        if gcs_control_point_fallback_count
                        else {}
                    ),
                    "gcs_control_point_terrain_objective_source_counts": (
                        {
                            "region_inverse_cost_weighted_passable_cell_centroid": (
                                gcs_control_point_terrain_count
                            ),
                            "not_evaluated": (
                                gcs_control_point_report_count - gcs_control_point_terrain_count
                            ),
                        }
                        if gcs_control_point_report_count
                        else {}
                    ),
                    "gcs_control_point_sampled_terrain_cost_count": gcs_control_point_terrain_count,
                    "gcs_control_point_sampled_terrain_cost_min": (
                        4.0 if gcs_control_point_terrain_count else None
                    ),
                    "gcs_control_point_sampled_terrain_cost_max": (
                        6.0 if gcs_control_point_terrain_count else None
                    ),
                    "gcs_control_point_sampled_terrain_cost_mean": (
                        5.0 if gcs_control_point_terrain_count else None
                    ),
                    "gcs_control_point_high_cost_exposure_delta_count": gcs_control_point_terrain_count,
                    "gcs_control_point_high_cost_exposure_delta_min": (
                        -2.0 if gcs_control_point_terrain_count else None
                    ),
                    "gcs_control_point_high_cost_exposure_delta_max": (
                        0.0 if gcs_control_point_terrain_count else None
                    ),
                    "gcs_control_point_high_cost_exposure_delta_mean": (
                        -1.0 if gcs_control_point_terrain_count else None
                    ),
                    "gcs_control_point_candidate_audit": [
                        {
                            "scenario_id": "fake",
                            "action_index": index,
                            "backend": "pydrake_control_point_direction_cone_program",
                            "attempted": True,
                            "success": index < gcs_control_point_success_count,
                            "candidate_selected": index < gcs_control_point_selected_count,
                            "candidate_fallback_reason": (
                                None
                                if index < gcs_control_point_success_count
                                else "sampled_trajectory_collision"
                            ),
                            "terrain_objective_source": (
                                "region_inverse_cost_weighted_passable_cell_centroid"
                                if index < gcs_control_point_terrain_count
                                else "not_evaluated"
                            ),
                            "sampled_terrain_cost": (
                                5.0 if index < gcs_control_point_terrain_count else None
                            ),
                            "high_cost_exposure_delta_vs_baseline": (
                                -1.0 if index < gcs_control_point_terrain_count else None
                            ),
                        }
                        for index in range(gcs_control_point_report_count)
                    ],
                    "gcs_control_point_candidate_artifacts": {
                        "schema_version": "gcs-control-point-candidate-artifact-index/v1",
                        "artifact_root": str(output_root / "gcs_control_point_candidate_artifacts"),
                        "candidate_count": gcs_control_point_report_count,
                        "route_artifact_count": gcs_control_point_report_count,
                        "entries": [
                            {
                                "scenario_id": "fake",
                                "scenario_group": scenario_set,
                                "action_index": index,
                                "cell": [index, index + 1],
                                "route_artifact": str(
                                    output_root
                                    / "gcs_control_point_candidate_artifacts"
                                    / f"action-{index:03d}"
                                    / "path-planner-route.json"
                                ),
                                "backend": "pydrake_control_point_direction_cone_program",
                                "candidate_selected": index < gcs_control_point_selected_count,
                                "candidate_fallback_reason": (
                                    None
                                    if index < gcs_control_point_success_count
                                    else "sampled_trajectory_collision"
                                ),
                            }
                            for index in range(gcs_control_point_report_count)
                        ],
                    },
                    "gcs_control_point_candidate_triage": {
                        "schema_version": "gcs-control-point-candidate-triage-summary/v1",
                        "candidate_count": gcs_control_point_report_count,
                        "attempted_count": gcs_control_point_attempted_count,
                        "success_count": gcs_control_point_success_count,
                        "selected_count": gcs_control_point_selected_count,
                        "route_artifact_count": gcs_control_point_report_count,
                        "fallback_reason_counts": (
                            {"sampled_trajectory_collision": gcs_control_point_fallback_count}
                            if gcs_control_point_fallback_count
                            else {}
                        ),
                        "terrain_objective_source_counts": (
                            {
                                "region_inverse_cost_weighted_passable_cell_centroid": (
                                    gcs_control_point_terrain_count
                                ),
                                "not_evaluated": (
                                    gcs_control_point_report_count - gcs_control_point_terrain_count
                                ),
                            }
                            if gcs_control_point_report_count
                            else {}
                        ),
                        "blocker_class_counts": (
                            {
                                "selected": gcs_control_point_selected_count,
                                "solver_or_region_sequence_not_successful": gcs_control_point_fallback_count,
                            }
                            if gcs_control_point_report_count
                            else {}
                        ),
                        "sampled_terrain_cost": {
                            "count": gcs_control_point_terrain_count,
                            "min": 4.0 if gcs_control_point_terrain_count else None,
                            "max": 6.0 if gcs_control_point_terrain_count else None,
                            "mean": 5.0 if gcs_control_point_terrain_count else None,
                        },
                        "high_cost_exposure_delta_vs_baseline": {
                            "count": gcs_control_point_terrain_count,
                            "min": -2.0 if gcs_control_point_terrain_count else None,
                            "max": 0.0 if gcs_control_point_terrain_count else None,
                            "mean": -1.0 if gcs_control_point_terrain_count else None,
                        },
                        "candidates": [
                            {
                                "scenario_id": "fake",
                                "action_index": index,
                                "candidate_selected": index < gcs_control_point_selected_count,
                                "candidate_fallback_reason": (
                                    None
                                    if index < gcs_control_point_success_count
                                    else "sampled_trajectory_collision"
                                ),
                                "terrain_objective_weight": (
                                    0.05 if index < gcs_control_point_terrain_count else None
                                ),
                                "sampled_terrain_cost": (
                                    5.0 if index < gcs_control_point_terrain_count else None
                                ),
                                "high_cost_exposure_delta_vs_baseline": (
                                    -1.0 if index < gcs_control_point_terrain_count else None
                                ),
                                "direction_cone_violation_count": 0,
                                "direction_cone_eta": 1.0,
                                "direction_cone_rho_min": 0.025,
                                "direction_cone_tolerance_deg": 45.0,
                                "direction_cone_rho_source_counts": {
                                    "seed_distance_portal_support_min": 1
                                },
                                "second_difference_weight": 0.2,
                                "motion_feasibility_status": (
                                    "feasible"
                                    if index < gcs_control_point_success_count
                                    else "diagnostic_only"
                                ),
                                "route_artifact": str(
                                    output_root
                                    / "gcs_control_point_candidate_artifacts"
                                    / f"action-{index:03d}"
                                    / "path-planner-route.json"
                                ),
                            }
                            for index in range(gcs_control_point_report_count)
                        ],
                    },
                    "sampled_region_path_selected_count": sampled_selected,
                    "sampled_region_path_fallback_count": sampled_fallback,
                    "sampled_region_path_status_counts": {
                        "selected": sampled_selected,
                        "fallback": sampled_fallback,
                    },
                    "sampled_region_path_source_counts": {"grid_box": sampled_selected + sampled_fallback},
                    "sampled_region_path_fallback_reasons": (
                        {"sampled_candidate_not_better": sampled_fallback} if sampled_fallback else {}
                    ),
                    "sampled_region_path_sample_attempt_count": sampled_attempts,
                    "sampled_region_path_candidate_ranking_count": sampled_rankings,
                    "sampled_region_path_anchor_region_added_count": sampled_anchor_added,
                    "sampled_region_path_anchor_region_connected_count": sampled_anchor_connected,
                    "sampled_region_path_anchor_closure_attempt_count": sampled_selected + sampled_fallback,
                    "sampled_region_path_anchor_closure_connected_count": sampled_selected,
                    "sampled_region_path_anchor_closure_status_counts": (
                        {"connected": sampled_selected, "unavailable": sampled_fallback}
                        if sampled_fallback
                        else {"connected": sampled_selected}
                    ),
                    "sampled_region_path_anchor_closure_reason_counts": (
                        {
                            "safe_bridge_found": sampled_selected,
                            "safe_bridge_path_unavailable": sampled_fallback,
                        }
                        if sampled_fallback
                        else {"safe_bridge_found": sampled_selected}
                    ),
                    "sampled_region_path_anchor_closure_connection_kind_counts": {
                        "anchor_region_safe_bridge": sampled_selected,
                    },
                    "sampled_region_path_goal_classification_counts": {
                        "goal_outside_region_coverage": sampled_anchor_added,
                        "covered": sampled_fallback,
                    },
                    "sampled_region_path_connector_attempt_count": sampled_connector_attempts,
                    "sampled_region_path_connector_strategy_counts": {
                        "cost_aware_constrained_astar": sampled_rankings,
                        "bridge_aware_constrained_astar": sampled_bridge_aware_connector_attempts,
                        "bridge_corridor_constrained_astar": sampled_bridge_corridor_connector_attempts,
                    },
                    "sampled_region_path_bridge_aware_connector_attempt_count": (
                        sampled_bridge_aware_connector_attempts
                    ),
                    "sampled_region_path_bridge_aware_connector_available_count": (
                        sampled_bridge_aware_connector_attempts
                    ),
                    "sampled_region_path_bridge_aware_connector_selected_count": (
                        sampled_bridge_aware_connector_attempts
                    ),
                    "sampled_region_path_bridge_aware_connector_rejected_count": 0,
                    "sampled_region_path_bridge_aware_connector_status_counts": (
                        {"available": sampled_bridge_aware_connector_attempts}
                        if sampled_bridge_aware_connector_attempts
                        else {}
                    ),
                    "sampled_region_path_bridge_aware_fallback_reasons": {},
                    "sampled_region_path_bridge_aware_bridge_cell_count": (
                        sampled_bridge_aware_connector_attempts * 3
                    ),
                    "sampled_region_path_bridge_aware_mask_added_cell_count": (
                        sampled_bridge_aware_connector_attempts
                    ),
                    "sampled_region_path_bridge_corridor_connector_attempt_count": (
                        sampled_bridge_corridor_connector_attempts
                    ),
                    "sampled_region_path_bridge_corridor_connector_available_count": (
                        sampled_bridge_corridor_connector_attempts
                    ),
                    "sampled_region_path_bridge_corridor_connector_selected_count": (
                        sampled_bridge_corridor_connector_attempts
                    ),
                    "sampled_region_path_bridge_corridor_connector_rejected_count": 0,
                    "sampled_region_path_bridge_corridor_status_counts": (
                        {"available": sampled_bridge_corridor_connector_attempts}
                        if sampled_bridge_corridor_connector_attempts
                        else {}
                    ),
                    "sampled_region_path_bridge_corridor_fallback_reasons": {},
                    "sampled_region_path_bridge_corridor_radius_counts": (
                        {"1": sampled_bridge_corridor_connector_attempts}
                        if sampled_bridge_corridor_connector_attempts
                        else {}
                    ),
                    "sampled_region_path_bridge_corridor_added_cell_count": (
                        sampled_bridge_corridor_connector_attempts * 4
                    ),
                    "sampled_region_path_terminal_adjusted_count": sampled_selected,
                    "sampled_region_path_terminal_adjustment_candidate_count": sampled_terminal_candidate_count,
                    "sampled_region_path_terminal_adjustment_status_counts": (
                        {"selected": sampled_selected, "not_required": sampled_fallback}
                        if sampled_fallback
                        else {"selected": sampled_selected}
                    ),
                    "sampled_region_path_terminal_adjustment_reason_counts": (
                        {
                            "terminal_adjustment_selected": sampled_selected,
                            "terminal_adjustment_not_required": sampled_fallback,
                        }
                        if sampled_fallback
                        else {"terminal_adjustment_selected": sampled_selected}
                    ),
                    "sampled_region_path_reachable_component_status_counts": (
                        {
                            "adjusted_connected": sampled_reachable_component_replacements,
                            "disconnected": sampled_reachable_component_disconnected,
                        }
                        if sampled_reachable_component_disconnected
                        else {"adjusted_connected": sampled_reachable_component_replacements}
                    ),
                    "sampled_region_path_reachable_component_reason_counts": (
                        {
                            "reachable_component_replacement_selected": sampled_reachable_component_replacements,
                            "target_component_disconnected": sampled_reachable_component_disconnected,
                        }
                        if sampled_reachable_component_disconnected
                        else {"reachable_component_replacement_selected": sampled_reachable_component_replacements}
                    ),
                    "sampled_region_path_reachable_component_disconnected_count": (
                        sampled_reachable_component_disconnected
                    ),
                    "sampled_region_path_reachable_component_replacement_selected_count": (
                        sampled_reachable_component_replacements
                    ),
                    "sampled_region_path_reachable_component_terminal_candidate_count": sampled_selected,
                    "sampled_region_path_reachable_terminal_rescue_count": sampled_reachable_terminal_rescues,
                    "sampled_region_path_proxy_goal_anchor_selected_count": sampled_proxy_goal_anchor_selected,
                    "sampled_region_path_goal_rescue_candidate_count": sampled_goal_rescue_candidates,
                    "sampled_region_path_benefit_surface_present_count": sampled_benefit_surface_present,
                    "sampled_region_path_path_duplicate_with_baseline_count": sampled_path_duplicates,
                    "sampled_region_path_baseline_equivalent_count": sampled_baseline_equivalent,
                    "sampled_region_path_no_quality_gain_count": sampled_no_quality_gain,
                    "sampled_region_path_fixture_no_benefit_surface_count": sampled_fixture_no_benefit,
                    "sampled_region_path_candidate_missing_metrics_count": sampled_candidate_missing_metrics,
                    "sampled_region_path_constrained_connector_failed_count": (
                        sampled_constrained_connector_failed
                    ),
                    "sampled_region_path_complexity_reason_counts": (
                        {
                            "sampled_candidate_has_quality_gain": sampled_selected,
                            "candidate_missing_metrics": sampled_fallback,
                        }
                        if sampled_fallback
                        else {"sampled_candidate_has_quality_gain": sampled_selected}
                    ),
                    "sampled_region_path_execution_tie_break_status_counts": (
                        {"selected": sampled_selected, "fallback": sampled_fallback}
                        if sampled_fallback
                        else {"selected": sampled_selected}
                    ),
                    "sampled_region_path_execution_tie_break_reason_counts": (
                        {
                            "execution_tie_break_improved": sampled_selected,
                            "execution_tie_break_no_alternative": sampled_fallback,
                        }
                        if sampled_fallback
                        else {"execution_tie_break_improved": sampled_selected}
                    ),
                    "sampled_region_path_candidate_audit": [
                        {
                            "scenario_id": run_id,
                            "action_index": 0,
                            "region_source": "grid_box",
                            "status": "fallback" if sampled_fallback else "selected",
                            "fallback_reason": "sampled_candidate_not_better" if sampled_fallback else None,
                            "region_sequence": [0, 1],
                            "start_goal_anchoring": {
                                "start_region_id": 0,
                                "goal_region_id": 1,
                                "goal_classification": "covered" if sampled_fallback else "goal_outside_region_coverage",
                                "goal_anchor_region_added": not sampled_fallback,
                                "goal_anchor_region_connected": not sampled_fallback,
                            },
                            "edge_transition_count": 1,
                            "sample_attempt_count": sampled_attempts,
                            "candidate_ranking_count": sampled_rankings,
                            "candidate_metrics": {"candidate_cost_delta": 1.0 if sampled_fallback else -1.0},
                            "terminal_adjustment_report": {
                                "reason_code": (
                                    "terminal_adjustment_not_required"
                                    if sampled_fallback
                                    else "terminal_adjustment_selected"
                                ),
                                "target_adjusted": not sampled_fallback,
                            },
                            "execution_tie_break": {
                                "reason": (
                                    "execution_tie_break_no_alternative"
                                    if sampled_fallback
                                    else "execution_tie_break_improved"
                                )
                            },
                        }
                    ],
                    "open_grid_fallback_used": open_grid,
                    "open_grid_fallback_used_gate": gate,
                    "acceptance_metadata": acceptance_metadata,
                    "scenario_group_summary": scenario_groups,
                    "scenarios": [],
                }
                manifest = {
                    "schema_version": "path-feedback-manifest/v1",
                    "scenario_set": scenario_set,
                    "diagnostic_profile": diagnostic_profile,
                    "top_k": top_k,
                    "outputs": {
                        "summary": str(output_root / "path-feedback-summary.json"),
                        "report": str(output_root / "path-feedback-summary.md"),
                    },
                }
                (output_root / "path-feedback-summary.json").write_text(
                    json.dumps(summary, indent=2),
                    encoding="utf-8",
                )
                (output_root / "path-feedback-manifest.json").write_text(
                    json.dumps(manifest, indent=2),
                    encoding="utf-8",
                )
                (output_root / "path-feedback-summary.md").write_text(
                    f"# fake report for {run_id}\\n",
                    encoding="utf-8",
                )
                PY
                """
            ),
            encoding="utf-8",
        )
        path.chmod(0o755)
        return path

    def test_matrix_validate_and_dry_run_do_not_write_outputs(self) -> None:
        output_root = self.temp_dir / "batch"
        matrix = self._write_matrix(
            {
                "schema_version": "path-feedback-batch-matrix/v1",
                "output_root": str(output_root),
                "defaults": {
                    "planner_extra_args": ["--planning-backend", "region_graph_guided"],
                },
                "runs": [
                    {
                        "run_id": "smoke-baseline-k1",
                        "scenario_set": "smoke",
                        "diagnostic_profile": "baseline",
                        "top_k": 1,
                        "sample_quality_profile": "audit-only",
                    }
                ],
            }
        )

        validate = self._run_batch("--matrix", str(matrix), "--validate-only")

        self.assertEqual(validate.returncode, 0, validate.stdout + validate.stderr)
        self.assertIn("matrix validated", validate.stdout)
        self.assertFalse(output_root.exists())

        dry_run = self._run_batch("--matrix", str(matrix), "--dry-run")

        self.assertEqual(dry_run.returncode, 0, dry_run.stdout + dry_run.stderr)
        self.assertIn("[DRY RUN]", dry_run.stdout)
        self.assertIn("smoke-baseline-k1", dry_run.stdout)
        self.assertIn("--planning-backend region_graph_guided", dry_run.stdout)
        self.assertFalse(output_root.exists())

    def test_invalid_matrix_is_rejected_before_execution(self) -> None:
        matrix = self._write_matrix(
            {
                "schema_version": "path-feedback-batch-matrix/v1",
                "runs": [
                    {
                        "run_id": "bad-top-k",
                        "scenario_set": "smoke",
                        "diagnostic_profile": "baseline",
                        "top_k": 0,
                    }
                ],
            }
        )

        completed = self._run_batch("--matrix", str(matrix), "--validate-only")

        self.assertEqual(completed.returncode, 2, completed.stdout + completed.stderr)
        self.assertIn("top_k must be a positive integer", completed.stderr)
        self.assertFalse(self.fake_single_run_log.exists())

    def test_batch_calls_single_script_with_independent_output_roots_and_writes_index(self) -> None:
        output_root = self.temp_dir / "batch"
        matrix = self._write_matrix(
            {
                "schema_version": "path-feedback-batch-matrix/v1",
                "output_root": "outputs/ignored-by-cli",
                "runs": [
                    {
                        "run_id": "smoke-baseline-k1",
                        "scenario_set": "smoke",
                        "diagnostic_profile": "baseline",
                        "top_k": 1,
                        "sample_quality_profile": "audit-only",
                    },
                    {
                        "run_id": "all-all-k3",
                        "scenario_set": "all",
                        "diagnostic_profile": "all",
                        "top_k": 3,
                    },
                ],
            }
        )

        completed = self._run_batch(
            "--matrix",
            str(matrix),
            "--output-root",
            str(output_root),
            "--single-run-script",
            str(self.fake_single_run),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertTrue((output_root / "smoke-baseline-k1" / "path-feedback-summary.json").is_file())
        self.assertTrue((output_root / "all-all-k3" / "path-feedback-summary.json").is_file())
        log_lines = self.fake_single_run_log.read_text(encoding="utf-8").splitlines()
        self.assertIn(f"smoke-baseline-k1|smoke|baseline|1|{output_root / 'smoke-baseline-k1'}", log_lines)
        self.assertIn(f"all-all-k3|all|all|3|{output_root / 'all-all-k3'}", log_lines)

        run_index = json.loads((output_root / "batch-run-index.json").read_text(encoding="utf-8"))
        self.assertEqual(run_index["schema_version"], "path-feedback-batch-run-index/v1")
        self.assertEqual(run_index["run_count"], 2)
        first_run = run_index["runs"][0]
        self.assertEqual(first_run["status"], "passed")
        self.assertEqual(first_run["reason_codes"], [])
        self.assertEqual(first_run["command_args"]["scenario_set"], "smoke")
        self.assertEqual(first_run["command_args"]["diagnostic_profile"], "baseline")
        self.assertEqual(first_run["command_args"]["top_k"], 1)
        self.assertEqual(first_run["command_args"]["python_executable"], sys.executable)
        self.assertEqual(first_run["sample_quality_profile"], "audit-only")
        self.assertTrue(first_run["source_paths"]["summary"].endswith("smoke-baseline-k1/path-feedback-summary.json"))
        self.assertTrue(first_run["source_paths"]["report"].endswith("smoke-baseline-k1/path-feedback-summary.md"))
        self.assertEqual(first_run["acceptance_metadata"]["scenario_set"], "smoke")
        self.assertRegex(first_run["git"]["parent"]["sha"], r"^[0-9a-f]{40}$")
        self.assertEqual(
            set(first_run["git"]["submodules"]),
            {"dev-platform-constraints", "model-explorer", "path-planner"},
        )
        for submodule in first_run["git"]["submodules"].values():
            self.assertRegex(submodule["sha"], r"^[0-9a-f]{40}$")

    def test_batch_evaluation_summary_aggregates_run_and_scenario_metrics(self) -> None:
        output_root = self.temp_dir / "batch"
        matrix = self._write_matrix(
            {
                "schema_version": "path-feedback-batch-matrix/v1",
                "output_root": str(output_root),
                "runs": [
                    {
                        "run_id": "smoke-baseline-k1",
                        "scenario_set": "smoke",
                        "diagnostic_profile": "baseline",
                        "top_k": 1,
                    },
                    {
                        "run_id": "all-all-k3",
                        "scenario_set": "all",
                        "diagnostic_profile": "all",
                        "top_k": 3,
                    },
                ],
            }
        )

        completed = self._run_batch(
            "--matrix",
            str(matrix),
            "--single-run-script",
            str(self.fake_single_run),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary = json.loads((output_root / "batch-evaluation-summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["schema_version"], "path-feedback-batch-evaluation-summary/v1")
        self.assertEqual(summary["run_count"], 2)
        self.assertEqual(summary["passed_count"], 2)
        self.assertEqual(summary["failed_count"], 0)
        self.assertEqual(summary["open_grid_fallback_used_count"], 0)
        self.assertEqual(summary["path_planning_failure_count"], 3)
        self.assertEqual(summary["replan_count"], 3)
        self.assertEqual(summary["iris_fallback_count"], 7)
        self.assertEqual(summary["region_graph_fallback_count"], 7)
        self.assertEqual(summary["region_graph_disconnected_count"], 3)
        self.assertEqual(summary["convex_region_report_count"], 7)
        self.assertEqual(summary["convex_region_count_total"], 14)
        self.assertEqual(summary["convex_region_backend_counts"]["workspace_iris"], 4)
        self.assertEqual(summary["convex_region_backend_counts"]["fallback_box"], 3)
        self.assertEqual(summary["convex_region_fallback_used_count"], 3)
        self.assertEqual(summary["convex_region_gcs_ready_count"], 7)
        self.assertEqual(summary["convex_region_blocked_cell_violation_count"], 0)
        self.assertEqual(summary["convex_region_coverage_status_counts"]["covered"], 7)
        self.assertEqual(summary["convex_region_gcs_ready_reason_counts"]["convex_region_sequence_ready"], 7)
        self.assertEqual(summary["convex_region_start_contained_count"], 7)
        self.assertEqual(summary["convex_region_goal_contained_count"], 7)
        self.assertEqual(summary["convex_region_adjacent_overlap_count"], 7)
        self.assertEqual(summary["convex_region_portal_count"], 0)
        self.assertEqual(len(summary["convex_region_candidate_audit"]), 7)
        self.assertEqual(summary["gcs_trajectory_report_count"], 6)
        self.assertEqual(summary["gcs_trajectory_attempted_count"], 6)
        self.assertEqual(summary["gcs_trajectory_success_count"], 3)
        self.assertEqual(summary["gcs_trajectory_collision_count"], 3)
        self.assertEqual(summary["gcs_trajectory_region_count_total"], 12)
        self.assertEqual(summary["gcs_trajectory_sample_count_total"], 30)
        self.assertEqual(summary["gcs_trajectory_backend_counts"]["pydrake_gcs"], 6)
        self.assertEqual(summary["gcs_trajectory_reason_counts"]["gcs_trajectory_solution_found"], 3)
        self.assertEqual(summary["gcs_trajectory_reason_counts"]["sampled_trajectory_collision"], 3)
        self.assertEqual(len(summary["gcs_trajectory_candidate_audit"]), 6)
        self.assertEqual(summary["gcs_candidate_report_count"], 6)
        self.assertEqual(summary["gcs_candidate_attempted_count"], 6)
        self.assertEqual(summary["gcs_candidate_available_count"], 3)
        self.assertEqual(summary["gcs_candidate_selected_count"], 3)
        self.assertEqual(summary["gcs_candidate_collision_count"], 3)
        self.assertEqual(summary["gcs_candidate_fallback_reason_counts"]["sampled_trajectory_collision"], 3)
        self.assertEqual(summary["gcs_candidate_selection_reason_counts"]["gcs_candidate_quality_improved"], 3)
        self.assertEqual(summary["gcs_candidate_cost_delta_vs_baseline_negative_count"], 3)
        self.assertEqual(summary["gcs_candidate_cost_delta_vs_baseline_positive_count"], 0)
        self.assertEqual(summary["gcs_candidate_cost_delta_vs_baseline_zero_count"], 0)
        self.assertEqual(len(summary["gcs_candidate_audit"]), 6)
        self.assertEqual(summary["gcs_motion_feasibility_report_count"], 6)
        self.assertEqual(summary["gcs_motion_feasibility_evaluated_count"], 3)
        self.assertEqual(summary["gcs_motion_feasibility_feasible_count"], 3)
        self.assertEqual(summary["gcs_motion_feasibility_infeasible_count"], 0)
        self.assertEqual(summary["gcs_motion_feasibility_diagnostic_only_count"], 3)
        self.assertEqual(summary["gcs_motion_feasibility_curvature_violation_count"], 0)
        self.assertEqual(summary["gcs_motion_feasibility_heading_violation_count"], 0)
        self.assertEqual(summary["gcs_motion_feasibility_status_counts"]["feasible"], 3)
        self.assertEqual(summary["gcs_motion_feasibility_status_counts"]["diagnostic_only"], 3)
        self.assertEqual(summary["gcs_motion_feasibility_fallback_reason_counts"]["gcs_trajectory_failed"], 3)
        self.assertEqual(summary["gcs_motion_feasibility_motion_model_counts"]["curvature_bounded"], 6)
        self.assertEqual(len(summary["gcs_motion_feasibility_audit"]), 6)
        self.assertEqual(summary["gcs_curvature_constrained_report_count"], 6)
        self.assertEqual(summary["gcs_curvature_constrained_attempted_count"], 6)
        self.assertEqual(summary["gcs_curvature_constrained_available_count"], 3)
        self.assertEqual(summary["gcs_curvature_constrained_selected_count"], 3)
        self.assertEqual(summary["gcs_curvature_constrained_repair_success_count"], 0)
        self.assertEqual(summary["gcs_curvature_constrained_infeasible_count"], 0)
        self.assertEqual(summary["gcs_curvature_constrained_diagnostic_only_count"], 3)
        self.assertEqual(summary["gcs_curvature_constrained_curvature_violation_count_before"], 0)
        self.assertEqual(summary["gcs_curvature_constrained_curvature_violation_count_after"], 0)
        self.assertEqual(summary["gcs_curvature_constrained_collision_count"], 3)
        self.assertEqual(summary["gcs_curvature_constrained_status_after_counts"]["feasible"], 3)
        self.assertEqual(summary["gcs_curvature_constrained_status_after_counts"]["diagnostic_only"], 3)
        self.assertEqual(summary["gcs_curvature_constrained_repair_strategy_counts"]["none_required"], 3)
        self.assertEqual(summary["gcs_curvature_constrained_repair_strategy_counts"]["not_attempted"], 3)
        self.assertEqual(summary["gcs_curvature_constrained_fallback_reason_counts"]["gcs_trajectory_failed"], 3)
        self.assertEqual(len(summary["gcs_curvature_constrained_audit"]), 6)
        self.assertEqual(summary["gcs_control_point_report_count"], 0)
        self.assertEqual(summary["gcs_control_point_attempted_count"], 0)
        self.assertEqual(summary["gcs_control_point_success_count"], 0)
        self.assertEqual(summary["gcs_control_point_backend_counts"], {})
        self.assertEqual(summary["gcs_control_point_candidate_selected_count"], 0)
        self.assertEqual(summary["gcs_control_point_candidate_fallback_reason_counts"], {})
        self.assertEqual(summary["gcs_control_point_terrain_objective_source_counts"], {})
        self.assertEqual(summary["gcs_control_point_sampled_terrain_cost_count"], 0)
        self.assertIsNone(summary["gcs_control_point_sampled_terrain_cost_min"])
        self.assertIsNone(summary["gcs_control_point_sampled_terrain_cost_max"])
        self.assertIsNone(summary["gcs_control_point_sampled_terrain_cost_mean"])
        self.assertEqual(summary["gcs_control_point_high_cost_exposure_delta_count"], 0)
        self.assertIsNone(summary["gcs_control_point_high_cost_exposure_delta_min"])
        self.assertIsNone(summary["gcs_control_point_high_cost_exposure_delta_max"])
        self.assertIsNone(summary["gcs_control_point_high_cost_exposure_delta_mean"])
        self.assertEqual(summary["gcs_control_point_candidate_audit"], [])
        self.assertEqual(summary["sampled_region_path_selected_count"], 4)
        self.assertEqual(summary["sampled_region_path_fallback_count"], 3)
        self.assertEqual(summary["sampled_region_path_status_counts"]["selected"], 4)
        self.assertEqual(summary["sampled_region_path_status_counts"]["fallback"], 3)
        self.assertEqual(summary["sampled_region_path_source_counts"]["grid_box"], 7)
        self.assertEqual(summary["sampled_region_path_fallback_reasons"]["sampled_candidate_not_better"], 3)
        self.assertEqual(summary["sampled_region_path_sample_attempt_count"], 25)
        self.assertEqual(summary["sampled_region_path_candidate_ranking_count"], 7)
        self.assertEqual(summary["sampled_region_path_anchor_region_added_count"], 4)
        self.assertEqual(summary["sampled_region_path_anchor_region_connected_count"], 4)
        self.assertEqual(summary["sampled_region_path_anchor_closure_attempt_count"], 7)
        self.assertEqual(summary["sampled_region_path_anchor_closure_connected_count"], 4)
        self.assertEqual(summary["sampled_region_path_anchor_closure_status_counts"]["connected"], 4)
        self.assertEqual(summary["sampled_region_path_anchor_closure_status_counts"]["unavailable"], 3)
        self.assertEqual(summary["sampled_region_path_anchor_closure_reason_counts"]["safe_bridge_found"], 4)
        self.assertEqual(
            summary["sampled_region_path_anchor_closure_reason_counts"]["safe_bridge_path_unavailable"],
            3,
        )
        self.assertEqual(
            summary["sampled_region_path_anchor_closure_connection_kind_counts"]["anchor_region_safe_bridge"],
            4,
        )
        self.assertEqual(summary["sampled_region_path_goal_classification_counts"]["goal_outside_region_coverage"], 4)
        self.assertEqual(summary["sampled_region_path_goal_classification_counts"]["covered"], 3)
        self.assertEqual(summary["sampled_region_path_connector_attempt_count"], 15)
        self.assertEqual(
            summary["sampled_region_path_connector_strategy_counts"]["cost_aware_constrained_astar"],
            7,
        )
        self.assertEqual(
            summary["sampled_region_path_connector_strategy_counts"]["bridge_aware_constrained_astar"],
            4,
        )
        self.assertEqual(
            summary["sampled_region_path_connector_strategy_counts"]["bridge_corridor_constrained_astar"],
            4,
        )
        self.assertEqual(summary["sampled_region_path_bridge_aware_connector_attempt_count"], 4)
        self.assertEqual(summary["sampled_region_path_bridge_aware_connector_available_count"], 4)
        self.assertEqual(summary["sampled_region_path_bridge_aware_connector_selected_count"], 4)
        self.assertEqual(summary["sampled_region_path_bridge_aware_connector_rejected_count"], 0)
        self.assertEqual(summary["sampled_region_path_bridge_aware_connector_status_counts"]["available"], 4)
        self.assertEqual(summary["sampled_region_path_bridge_aware_bridge_cell_count"], 12)
        self.assertEqual(summary["sampled_region_path_bridge_aware_mask_added_cell_count"], 4)
        self.assertEqual(summary["sampled_region_path_bridge_corridor_connector_attempt_count"], 4)
        self.assertEqual(summary["sampled_region_path_bridge_corridor_connector_available_count"], 4)
        self.assertEqual(summary["sampled_region_path_bridge_corridor_connector_selected_count"], 4)
        self.assertEqual(summary["sampled_region_path_bridge_corridor_connector_rejected_count"], 0)
        self.assertEqual(summary["sampled_region_path_bridge_corridor_status_counts"]["available"], 4)
        self.assertEqual(summary["sampled_region_path_bridge_corridor_radius_counts"]["1"], 4)
        self.assertEqual(summary["sampled_region_path_bridge_corridor_added_cell_count"], 16)
        self.assertEqual(summary["sampled_region_path_terminal_adjusted_count"], 4)
        self.assertEqual(summary["sampled_region_path_terminal_adjustment_candidate_count"], 8)
        self.assertEqual(summary["sampled_region_path_terminal_adjustment_status_counts"]["selected"], 4)
        self.assertEqual(summary["sampled_region_path_terminal_adjustment_status_counts"]["not_required"], 3)
        self.assertEqual(
            summary["sampled_region_path_terminal_adjustment_reason_counts"]["terminal_adjustment_selected"],
            4,
        )
        self.assertEqual(summary["sampled_region_path_reachable_component_status_counts"]["adjusted_connected"], 4)
        self.assertEqual(summary["sampled_region_path_reachable_component_status_counts"]["disconnected"], 3)
        self.assertEqual(
            summary["sampled_region_path_reachable_component_reason_counts"][
                "reachable_component_replacement_selected"
            ],
            4,
        )
        self.assertEqual(
            summary["sampled_region_path_reachable_component_reason_counts"]["target_component_disconnected"],
            3,
        )
        self.assertEqual(summary["sampled_region_path_reachable_component_disconnected_count"], 3)
        self.assertEqual(summary["sampled_region_path_reachable_component_replacement_selected_count"], 4)
        self.assertEqual(summary["sampled_region_path_reachable_component_terminal_candidate_count"], 4)
        self.assertEqual(summary["sampled_region_path_reachable_terminal_rescue_count"], 4)
        self.assertEqual(summary["sampled_region_path_proxy_goal_anchor_selected_count"], 4)
        self.assertEqual(summary["sampled_region_path_goal_rescue_candidate_count"], 8)
        self.assertEqual(summary["sampled_region_path_benefit_surface_present_count"], 4)
        self.assertEqual(summary["sampled_region_path_path_duplicate_with_baseline_count"], 3)
        self.assertEqual(summary["sampled_region_path_baseline_equivalent_count"], 3)
        self.assertEqual(summary["sampled_region_path_no_quality_gain_count"], 3)
        self.assertEqual(summary["sampled_region_path_fixture_no_benefit_surface_count"], 3)
        self.assertEqual(summary["sampled_region_path_candidate_missing_metrics_count"], 3)
        self.assertEqual(summary["sampled_region_path_constrained_connector_failed_count"], 3)
        self.assertEqual(
            summary["sampled_region_path_complexity_reason_counts"]["sampled_candidate_has_quality_gain"],
            4,
        )
        self.assertEqual(summary["sampled_region_path_complexity_reason_counts"]["candidate_missing_metrics"], 3)
        self.assertEqual(
            summary["sampled_region_path_execution_tie_break_reason_counts"]["execution_tie_break_improved"],
            4,
        )
        self.assertEqual(
            summary["sampled_region_path_execution_tie_break_reason_counts"]["execution_tie_break_no_alternative"],
            3,
        )
        self.assertEqual(len(summary["sampled_region_path_candidate_audit"]), 2)
        self.assertEqual(
            summary["sampled_region_path_candidate_audit"][1]["fallback_reason"],
            "sampled_candidate_not_better",
        )
        self.assertEqual(
            summary["sampled_region_path_candidate_audit"][0]["execution_tie_break"]["reason"],
            "execution_tie_break_improved",
        )
        self.assertEqual(summary["scenario_group_summary"]["smoke"]["scenario_count"], 2)
        self.assertEqual(summary["scenario_group_summary"]["stress"]["failure_count"], 3)
        self.assertEqual(
            len(summary["source_summary_paths"]),
            2,
        )
        self.assertTrue(summary["source_summary_paths"][0].endswith("smoke-baseline-k1/path-feedback-summary.json"))

    def test_batch_preserves_and_aggregates_control_point_gcs_opt_in(self) -> None:
        output_root = self.temp_dir / "batch"
        matrix = self._write_matrix(
            {
                "schema_version": "path-feedback-batch-matrix/v1",
                "output_root": str(output_root),
                "runs": [
                    {
                        "run_id": "all-all-control-point",
                        "scenario_set": "all",
                        "diagnostic_profile": "all",
                        "top_k": 3,
                        "planner_extra_args": ["--gcs-control-point-candidate"],
                    },
                ],
            }
        )

        completed = self._run_batch(
            "--matrix",
            str(matrix),
            "--single-run-script",
            str(self.fake_single_run),
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        run_index = json.loads((output_root / "batch-run-index.json").read_text(encoding="utf-8"))
        self.assertEqual(
            run_index["runs"][0]["command_args"]["planner_extra_args"],
            ["--gcs-control-point-candidate"],
        )
        summary = json.loads((output_root / "batch-evaluation-summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["gcs_control_point_report_count"], 6)
        self.assertEqual(summary["gcs_control_point_attempted_count"], 6)
        self.assertEqual(summary["gcs_control_point_success_count"], 3)
        self.assertEqual(
            summary["gcs_control_point_backend_counts"]["pydrake_control_point_direction_cone_program"],
            6,
        )
        self.assertEqual(summary["gcs_control_point_candidate_selected_count"], 3)
        self.assertEqual(
            summary["gcs_control_point_candidate_fallback_reason_counts"]["sampled_trajectory_collision"],
            3,
        )
        self.assertEqual(
            summary["gcs_control_point_terrain_objective_source_counts"][
                "region_inverse_cost_weighted_passable_cell_centroid"
            ],
            3,
        )
        self.assertEqual(summary["gcs_control_point_terrain_objective_source_counts"]["not_evaluated"], 3)
        self.assertEqual(summary["gcs_control_point_sampled_terrain_cost_count"], 3)
        self.assertEqual(summary["gcs_control_point_sampled_terrain_cost_min"], 4.0)
        self.assertEqual(summary["gcs_control_point_sampled_terrain_cost_max"], 6.0)
        self.assertEqual(summary["gcs_control_point_sampled_terrain_cost_mean"], 5.0)
        self.assertEqual(summary["gcs_control_point_high_cost_exposure_delta_count"], 3)
        self.assertEqual(summary["gcs_control_point_high_cost_exposure_delta_min"], -2.0)
        self.assertEqual(summary["gcs_control_point_high_cost_exposure_delta_max"], 0.0)
        self.assertEqual(summary["gcs_control_point_high_cost_exposure_delta_mean"], -1.0)
        self.assertEqual(len(summary["gcs_control_point_candidate_audit"]), 6)
        triage = summary["gcs_control_point_candidate_triage"]
        self.assertEqual(triage["schema_version"], "gcs-control-point-candidate-triage-summary/v1")
        self.assertEqual(triage["candidate_count"], 6)
        self.assertEqual(triage["attempted_count"], 6)
        self.assertEqual(triage["success_count"], 3)
        self.assertEqual(triage["selected_count"], 3)
        self.assertEqual(triage["route_artifact_count"], 6)
        self.assertEqual(triage["fallback_reason_counts"]["sampled_trajectory_collision"], 3)
        self.assertEqual(
            triage["terrain_objective_source_counts"]["region_inverse_cost_weighted_passable_cell_centroid"],
            3,
        )
        self.assertEqual(triage["terrain_objective_source_counts"]["not_evaluated"], 3)
        self.assertEqual(triage["sampled_terrain_cost"]["count"], 3)
        self.assertEqual(triage["sampled_terrain_cost"]["mean"], 5.0)
        self.assertEqual(triage["high_cost_exposure_delta_vs_baseline"]["count"], 3)
        self.assertEqual(triage["high_cost_exposure_delta_vs_baseline"]["mean"], -1.0)
        sweep = triage["calibration_sweep"]
        self.assertEqual(
            sweep["schema_version"],
            "gcs-control-point-candidate-calibration-sweep/v1",
        )
        self.assertFalse(sweep["default_change_recommended"])
        self.assertTrue(sweep["solver_rerun_required"])
        self.assertIn("direction_cone_rho_eta_tolerance", sweep["sweep_dimensions"])
        self.assertEqual(sweep["observed_current_values"]["terrain_objective_weight"], [0.05])
        self.assertEqual(sweep["observed_current_values"]["second_difference_weight"], [0.2])
        self.assertEqual(
            sweep["observed_current_values"]["direction_cone_rho_source_counts"][
                "seed_distance_portal_support_min"
            ],
            6,
        )
        self.assertFalse(sweep["safety_regression_guard"]["terrain_cost_degradation_allowed"])
        self.assertEqual(len(triage["candidates"]), 6)
        artifacts = summary["gcs_control_point_candidate_artifacts"]
        self.assertEqual(artifacts["schema_version"], "gcs-control-point-candidate-artifact-index/v1")
        self.assertEqual(artifacts["candidate_count"], 6)
        self.assertEqual(artifacts["route_artifact_count"], 6)
        self.assertEqual(len(artifacts["entries"]), 6)

    def test_failed_single_run_still_writes_failure_records_and_nonzero_exit(self) -> None:
        output_root = self.temp_dir / "batch"
        matrix = self._write_matrix(
            {
                "schema_version": "path-feedback-batch-matrix/v1",
                "output_root": str(output_root),
                "runs": [
                    {
                        "run_id": "smoke-baseline-k1",
                        "scenario_set": "smoke",
                        "diagnostic_profile": "baseline",
                        "top_k": 1,
                    },
                    {
                        "run_id": "exit-fail",
                        "scenario_set": "stress",
                        "diagnostic_profile": "execution",
                        "top_k": 3,
                    },
                    {
                        "run_id": "open-grid-fail",
                        "scenario_set": "smoke",
                        "diagnostic_profile": "baseline",
                        "top_k": 1,
                    },
                ],
            }
        )

        completed = self._run_batch(
            "--matrix",
            str(matrix),
            "--single-run-script",
            str(self.fake_single_run),
        )

        self.assertNotEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        run_index = json.loads((output_root / "batch-run-index.json").read_text(encoding="utf-8"))
        runs_by_id = {item["run_id"]: item for item in run_index["runs"]}
        self.assertEqual(runs_by_id["smoke-baseline-k1"]["status"], "passed")
        self.assertEqual(runs_by_id["exit-fail"]["status"], "failed")
        self.assertIn("single_run_exit_nonzero", runs_by_id["exit-fail"]["reason_codes"])
        self.assertIn("summary_missing", runs_by_id["exit-fail"]["reason_codes"])
        self.assertEqual(runs_by_id["open-grid-fail"]["status"], "failed")
        self.assertIn("open_grid_fallback_used", runs_by_id["open-grid-fail"]["reason_codes"])

        summary = json.loads((output_root / "batch-evaluation-summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["run_count"], 3)
        self.assertEqual(summary["passed_count"], 1)
        self.assertEqual(summary["failed_count"], 2)
        self.assertEqual(summary["open_grid_fallback_used_count"], 1)
        self.assertIn("exit-fail", summary["failed_run_ids"])
        self.assertIn("open-grid-fail", summary["failed_run_ids"])


class PathFeedbackSingleRunCompatibilityTests(unittest.TestCase):
    def test_single_run_default_dry_run_behavior_is_unchanged(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "run_path_feedback_validation.sh"

        completed = subprocess.run(
            ["bash", str(script), "--dry-run"],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("Output root: " + str(repo_root / "outputs" / "path_feedback_validation"), completed.stdout)
        self.assertIn("Acceptance gate: custom", completed.stdout)
        self.assertIn("Top-K: 3", completed.stdout)
        self.assertIn("Scenario set: smoke", completed.stdout)
        self.assertIn("Diagnostic profile: baseline", completed.stdout)


if __name__ == "__main__":
    unittest.main()
