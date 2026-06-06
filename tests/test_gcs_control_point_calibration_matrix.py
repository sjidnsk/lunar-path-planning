import json
import importlib.util
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


class GcsControlPointCalibrationMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.script = self.repo_root / "scripts" / "run_gcs_control_point_calibration_matrix.py"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="gcs-control-point-calibration-"))

    def test_matrix_runner_rejects_improved_selection_when_safety_regresses(self) -> None:
        triage = self._write_triage()
        matrix = self._write_matrix()
        fake_planner = self._write_fake_planner()
        output_root = self.temp_dir / "matrix-output"

        completed = subprocess.run(
            [
                sys.executable,
                str(self.script),
                "--triage-json",
                str(triage),
                "--matrix-json",
                str(matrix),
                "--output-root",
                str(output_root),
                "--planner-command",
                str(fake_planner),
            ],
            cwd=self.repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        summary_path = output_root / "gcs-control-point-calibration-matrix-summary.json"
        report_path = output_root / "gcs-control-point-calibration-matrix-summary.md"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(
            summary["schema_version"],
            "gcs-control-point-calibration-matrix-summary/v1",
        )
        self.assertEqual(summary["matrix_count"], 2)
        self.assertEqual(summary["candidate_count"], 2)
        self.assertEqual(summary["case_count"], 4)
        self.assertEqual(summary["route_artifact_count"], 4)
        self.assertFalse(summary["default_change_recommended"])
        self.assertEqual(summary["default_change_reason"], "no_matrix_passed_safety_and_blocker_gates")
        self.assertGreater(summary["safety_regression_count"], 0)
        terrain_up = {
            item["matrix_id"]: item for item in summary["matrix_summaries"]
        }["terrain_up"]
        self.assertEqual(terrain_up["selected_count"], 1)
        self.assertEqual(terrain_up["safety_regression_count"], 1)
        self.assertLess(terrain_up["blocker_deltas"]["cost_dominated"], 0)
        report = report_path.read_text(encoding="utf-8")
        self.assertIn("GCS Control-Point Calibration Matrix", report)
        self.assertIn("no_matrix_passed_safety_and_blocker_gates", report)
        self.assertIn("## Blocker Deltas", report)
        self.assertIn("cost_dominated=-1", report)
        self.assertIn("## Safety Regression Cases", report)
        self.assertIn("high_cost_exposure_regression", report)

    def test_default_matrix_contains_targeted_parameter_sweep_coverage(self) -> None:
        module = self._load_script_module()
        matrix = module.DEFAULT_MATRIX
        matrix_ids = {item["matrix_id"] for item in matrix}

        self.assertIn("direction_cone_tight", matrix_ids)
        self.assertIn("direction_cone_wide", matrix_ids)
        self.assertIn("terrain_down", matrix_ids)
        self.assertIn("terrain_mid", matrix_ids)
        self.assertIn("smoothness_down", matrix_ids)
        self.assertIn("smoothness_up", matrix_ids)
        self.assertIn("joint_direction_cone_wide_terrain_down", matrix_ids)
        self.assertIn("high_cost_proxy_low", matrix_ids)
        self.assertIn("joint_direction_cone_wide_high_cost_proxy", matrix_ids)
        self.assertGreaterEqual(len(matrix), 9)
        self.assertTrue(any(item["direction_cone_max_error_deg"] < 45.0 for item in matrix))
        self.assertTrue(any(item["direction_cone_max_error_deg"] > 45.0 for item in matrix))
        self.assertTrue(any(item["direction_cone_seed_rho_ratio"] < 0.05 for item in matrix))
        self.assertTrue(any(item["direction_cone_seed_rho_ratio"] > 0.05 for item in matrix))
        self.assertTrue(any(item["terrain_objective_weight"] < 0.05 for item in matrix))
        self.assertTrue(any(item["terrain_objective_weight"] > 0.05 for item in matrix))
        self.assertTrue(any(item["second_difference_weight"] < 0.2 for item in matrix))
        self.assertTrue(any(item["second_difference_weight"] > 0.2 for item in matrix))
        self.assertTrue(
            any(
                item["direction_cone_max_error_deg"] > 45.0
                and item["terrain_objective_weight"] < 0.05
                for item in matrix
            )
        )
        self.assertTrue(any(item["high_cost_exposure_weight"] > 0.0 for item in matrix))
        self.assertTrue(
            any(
                item["direction_cone_max_error_deg"] > 45.0
                and item["high_cost_exposure_weight"] > 0.0
                for item in matrix
            )
        )

    def test_targeted_sweep_summary_records_transition_classes_and_markdown(self) -> None:
        triage = self._write_targeted_triage()
        matrix = self._write_targeted_matrix()
        fake_planner = self._write_targeted_fake_planner()
        output_root = self.temp_dir / "targeted-output"

        completed = subprocess.run(
            [
                sys.executable,
                str(self.script),
                "--triage-json",
                str(triage),
                "--matrix-json",
                str(matrix),
                "--output-root",
                str(output_root),
                "--planner-command",
                str(fake_planner),
            ],
            cwd=self.repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        status = json.loads(completed.stdout)
        self.assertEqual(status["targeted_sweep_summary"], str(output_root / "gcs-control-point-targeted-sweep-summary.json"))
        summary_path = output_root / "gcs-control-point-targeted-sweep-summary.json"
        report_path = output_root / "gcs-control-point-targeted-sweep-summary.md"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["schema_version"], "gcs-control-point-targeted-sweep-summary/v1")
        self.assertEqual(summary["matrix_count"], 1)
        self.assertEqual(summary["candidate_count"], 5)
        self.assertEqual(summary["case_count"], 5)
        self.assertEqual(summary["selected_count"], 2)
        self.assertEqual(summary["high_cost_exposure_proxy_case_count"], 5)
        self.assertEqual(summary["high_cost_exposure_proxy_evaluated_count"], 4)
        self.assertFalse(summary["default_change_recommended"])
        self.assertEqual(summary["recommendation"], "no_default_change_recommended")
        transition_counts = summary["transition_class_counts"]
        self.assertEqual(transition_counts["direction_cone_fixed_and_selected"], 1)
        self.assertEqual(transition_counts["direction_cone_to_cost_dominated"], 1)
        self.assertEqual(transition_counts["cost_dominated_persistent"], 1)
        self.assertEqual(transition_counts["safety_regression"], 1)
        self.assertEqual(transition_counts["unsupported_not_evaluated"], 1)
        matrix_summary = summary["matrix_summaries"][0]
        self.assertEqual(matrix_summary["selected_count"], 2)
        self.assertEqual(matrix_summary["safety_regression_count"], 1)
        self.assertEqual(matrix_summary["parameters"]["high_cost_exposure_weight"], 0.45)
        self.assertEqual(matrix_summary["transition_class_counts"]["direction_cone_fixed_and_selected"], 1)
        examples = matrix_summary["transition_examples"]
        self.assertIn("direction_cone_to_cost_dominated", examples)
        self.assertEqual(examples["direction_cone_to_cost_dominated"][0]["scenario_id"], "direction-to-cost")
        self.assertEqual(
            [case["transition_class"] for case in summary["cases"] if case["scenario_id"] == "direction-to-selected"],
            ["direction_cone_fixed_and_selected"],
        )
        selected_case = [
            case for case in summary["cases"] if case["scenario_id"] == "direction-to-selected"
        ][0]
        self.assertEqual(selected_case["high_cost_exposure_weight"], 0.45)
        self.assertEqual(selected_case["high_cost_exposure_proxy_source"], "region_high_cost_exposure_proxy")
        self.assertEqual(selected_case["high_cost_exposure_proxy_cost"], 2.5)
        report = report_path.read_text(encoding="utf-8")
        self.assertIn("# GCS Control-Point Targeted Sweep", report)
        self.assertIn("selected_count: 2", report)
        self.assertIn("high_cost_exposure_proxy_case_count: 5", report)
        self.assertIn("high_cost_exposure_weight", report)
        self.assertIn("direction_cone_fixed_and_selected", report)
        self.assertIn("direction_cone_to_cost_dominated", report)
        self.assertIn("unsupported_not_evaluated", report)

    def _load_script_module(self):
        spec = importlib.util.spec_from_file_location("gcs_matrix_script", self.script)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _write_triage(self) -> Path:
        request_a = self.temp_dir / "scenario-a-action-000-request.json"
        request_b = self.temp_dir / "scenario-b-action-001-request.json"
        request_a.write_text(json.dumps({"schema_version": "path-planner-request/v1"}), encoding="utf-8")
        request_b.write_text(json.dumps({"schema_version": "path-planner-request/v1"}), encoding="utf-8")
        triage = self.temp_dir / "triage.json"
        triage.write_text(
            json.dumps(
                {
                    "schema_version": "gcs-control-point-candidate-triage-summary/v1",
                    "candidate_count": 2,
                    "attempted_count": 2,
                    "success_count": 2,
                    "selected_count": 0,
                    "route_artifact_count": 2,
                    "fallback_reason_counts": {
                        "cost_dominated": 1,
                        "direction_cone_constraint_violation": 1,
                    },
                    "candidates": [
                        {
                            "scenario_id": "scenario-a",
                            "action_index": 0,
                            "cell": [4, 5],
                            "candidate_fallback_reason": "cost_dominated",
                            "candidate_selected": False,
                            "attempted": True,
                            "success": True,
                            "sampled_terrain_cost": 10.0,
                            "high_cost_exposure_delta_vs_baseline": 0.0,
                            "direction_cone_violation_count": 0,
                            "direction_cone_risk_flags": [],
                            "motion_feasibility_status": "feasible",
                            "motion_feasibility_curvature_violation_count": 0,
                            "motion_feasibility_heading_violation_count": 0,
                            "request_artifact": str(request_a),
                        },
                        {
                            "scenario_id": "scenario-b",
                            "action_index": 1,
                            "cell": [6, 7],
                            "candidate_fallback_reason": "direction_cone_constraint_violation",
                            "candidate_selected": False,
                            "attempted": True,
                            "success": True,
                            "sampled_terrain_cost": 9.0,
                            "high_cost_exposure_delta_vs_baseline": 0.0,
                            "direction_cone_violation_count": 1,
                            "direction_cone_risk_flags": ["direction_cone_violation"],
                            "motion_feasibility_status": "feasible",
                            "motion_feasibility_curvature_violation_count": 0,
                            "motion_feasibility_heading_violation_count": 0,
                            "request_artifact": str(request_b),
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return triage

    def _write_matrix(self) -> Path:
        matrix = self.temp_dir / "matrix.json"
        matrix.write_text(
            json.dumps(
                {
                    "schema_version": "gcs-control-point-calibration-matrix/v1",
                    "matrix": [
                        {
                            "matrix_id": "baseline",
                            "terrain_objective_weight": 0.05,
                            "second_difference_weight": 0.2,
                            "direction_cone_max_error_deg": 45.0,
                            "direction_cone_rho_floor_m": 0.0001,
                            "direction_cone_seed_rho_ratio": 0.05,
                        },
                        {
                            "matrix_id": "terrain_up",
                            "terrain_objective_weight": 0.08,
                            "second_difference_weight": 0.2,
                            "direction_cone_max_error_deg": 45.0,
                            "direction_cone_rho_floor_m": 0.0001,
                            "direction_cone_seed_rho_ratio": 0.05,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return matrix

    def _write_targeted_triage(self) -> Path:
        candidates = [
            ("direction-to-selected", 0, "direction_cone_constraint_violation", True),
            ("direction-to-cost", 1, "direction_cone_constraint_violation", True),
            ("cost-persistent", 2, "cost_dominated", True),
            ("cost-selected-unsafe", 3, "cost_dominated", True),
            ("unsupported", 4, "unsupported_route_replacement", False),
        ]
        candidate_rows = []
        for scenario_id, action_index, fallback_reason, has_request in candidates:
            request_artifact = self.temp_dir / f"{scenario_id}-action-{action_index:03d}-request.json"
            if has_request:
                request_artifact.write_text(
                    json.dumps({"schema_version": "path-planner-request/v1"}),
                    encoding="utf-8",
                )
            candidate_rows.append(
                {
                    "scenario_id": scenario_id,
                    "action_index": action_index,
                    "cell": [action_index, action_index + 1],
                    "candidate_fallback_reason": fallback_reason,
                    "candidate_selected": False,
                    "attempted": has_request,
                    "success": has_request,
                    "sampled_terrain_cost": 10.0,
                    "high_cost_exposure_delta_vs_baseline": 0.0,
                    "direction_cone_violation_count": (
                        1 if fallback_reason == "direction_cone_constraint_violation" else 0
                    ),
                    "direction_cone_risk_flags": (
                        ["direction_cone_violation"]
                        if fallback_reason == "direction_cone_constraint_violation"
                        else []
                    ),
                    "motion_feasibility_status": "feasible",
                    "motion_feasibility_curvature_violation_count": 0,
                    "motion_feasibility_heading_violation_count": 0,
                    **({"request_artifact": str(request_artifact)} if has_request else {}),
                }
            )
        triage = self.temp_dir / "targeted-triage.json"
        triage.write_text(
            json.dumps(
                {
                    "schema_version": "gcs-control-point-candidate-triage-summary/v1",
                    "candidate_count": len(candidate_rows),
                    "attempted_count": 4,
                    "success_count": 4,
                    "selected_count": 0,
                    "route_artifact_count": len(candidate_rows),
                    "fallback_reason_counts": {
                        "cost_dominated": 2,
                        "direction_cone_constraint_violation": 2,
                        "unsupported_route_replacement": 1,
                    },
                    "candidates": candidate_rows,
                }
            ),
            encoding="utf-8",
        )
        return triage

    def _write_targeted_matrix(self) -> Path:
        matrix = self.temp_dir / "targeted-matrix.json"
        matrix.write_text(
            json.dumps(
                {
                    "schema_version": "gcs-control-point-calibration-matrix/v1",
                    "matrix": [
                        {
                            "matrix_id": "joint_direction_cone_wide_terrain_down",
                            "terrain_objective_weight": 0.03,
                            "second_difference_weight": 0.2,
                            "high_cost_exposure_weight": 0.45,
                            "direction_cone_max_error_deg": 60.0,
                            "direction_cone_rho_floor_m": 0.0001,
                            "direction_cone_seed_rho_ratio": 0.03,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return matrix

    def _write_targeted_fake_planner(self) -> Path:
        script = self.temp_dir / "targeted_fake_path_planner.py"
        script.write_text(
            textwrap.dedent(
                """\
                import json
                import sys
                from pathlib import Path

                args = sys.argv[1:]
                output_json = Path(args[args.index("--output-json") + 1])
                input_path = Path(args[args.index("--input") + 1])
                high_cost_weight = float(args[args.index("--gcs-control-point-high-cost-exposure-weight") + 1])
                scenario_id = input_path.name.split("-action-")[0]
                selected_by_scenario = {
                    "direction-to-selected": True,
                    "direction-to-cost": False,
                    "cost-persistent": False,
                    "cost-selected-unsafe": True,
                }
                fallback_by_scenario = {
                    "direction-to-selected": None,
                    "direction-to-cost": "cost_dominated",
                    "cost-persistent": "cost_dominated",
                    "cost-selected-unsafe": None,
                }
                selected = selected_by_scenario[scenario_id]
                fallback = fallback_by_scenario[scenario_id]
                high_cost_delta = 2.0 if scenario_id == "cost-selected-unsafe" else 0.0
                payload = {
                    "schema_version": "path-planner-route/v1",
                    "gcs_trajectory_attempted": True,
                    "gcs_trajectory_success": True,
                    "gcs_trajectory_backend": "pydrake_control_point_direction_cone_program",
                    "gcs_trajectory_reason": "control_point_direction_cone_solution_found",
                    "gcs_trajectory_collision_count": 0,
                    "gcs_trajectory_cost_summary": {
                        "sampled_terrain_cost": 10.0,
                        "terrain_objective_weight": 0.03,
                        "high_cost_exposure": high_cost_delta,
                        "high_cost_exposure_objective_weight": high_cost_weight,
                        "high_cost_exposure_proxy_cost": 2.5,
                        "high_cost_exposure_proxy_source": "region_high_cost_exposure_proxy",
                    },
                    "gcs_trajectory_constraint_summary": {
                        "evaluated": True,
                        "backend_enforced": True,
                        "violation_count": 0,
                        "risk_flags": [],
                        "max_allowed_direction_error_deg": 60.0,
                        "eta": 1.0,
                        "rho_min": 0.02,
                        "rho_lower_bound_min_m": 0.02,
                        "objective_term_weights": {
                            "control_point_terrain_anchor_quadratic": 0.03,
                            "control_point_second_difference_quadratic": 0.2,
                            "control_point_high_cost_exposure_proxy_quadratic": high_cost_weight,
                        },
                    },
                    "gcs_candidate_selected": selected,
                    "gcs_candidate_fallback_reason": fallback,
                    "gcs_candidate_cost_delta_vs_baseline": -1.0 if selected else 1.0,
                    "gcs_candidate_cost_delta_vs_postprocess": -1.0 if selected else 1.0,
                    "gcs_candidate_cost_summary": {
                        "sampled_terrain_cost": 10.0,
                        "terrain_objective_weight": 0.03,
                        "high_cost_exposure_delta_vs_baseline": high_cost_delta,
                    },
                    "gcs_motion_feasibility_feasibility_status": "feasible",
                    "gcs_motion_feasibility_curvature_violation_count": 0,
                    "gcs_motion_feasibility_heading_violation_count": 0,
                }
                output_json.parent.mkdir(parents=True, exist_ok=True)
                output_json.write_text(json.dumps(payload), encoding="utf-8")
                print(json.dumps({"status": "ok", "output_json": str(output_json)}))
                """
            ),
            encoding="utf-8",
        )
        return script

    def _write_fake_planner(self) -> Path:
        script = self.temp_dir / "fake_path_planner.py"
        script.write_text(
            textwrap.dedent(
                """\
                import json
                import sys
                from pathlib import Path

                args = sys.argv[1:]
                output_json = Path(args[args.index("--output-json") + 1])
                input_path = Path(args[args.index("--input") + 1])
                terrain_weight = float(args[args.index("--gcs-control-point-terrain-weight") + 1])
                high_cost_weight = float(args[args.index("--gcs-control-point-high-cost-exposure-weight") + 1])
                is_scenario_a = "scenario-a" in input_path.name
                selected = is_scenario_a and terrain_weight > 0.05
                fallback = None if selected else ("cost_dominated" if is_scenario_a else "direction_cone_constraint_violation")
                violation_count = 0 if is_scenario_a else 1
                high_cost_delta = 3.0 if selected else 0.0
                payload = {
                    "schema_version": "path-planner-route/v1",
                    "gcs_trajectory_attempted": True,
                    "gcs_trajectory_success": True,
                    "gcs_trajectory_backend": "pydrake_control_point_direction_cone_program",
                    "gcs_trajectory_reason": "control_point_direction_cone_solution_found",
                    "gcs_trajectory_collision_count": 0,
                    "gcs_trajectory_cost_summary": {
                        "sampled_terrain_cost": 10.0 if is_scenario_a else 9.0,
                        "terrain_objective_weight": terrain_weight,
                        "high_cost_exposure": high_cost_delta,
                        "high_cost_exposure_objective_weight": high_cost_weight,
                        "high_cost_exposure_proxy_cost": 2.5,
                        "high_cost_exposure_proxy_source": "region_high_cost_exposure_proxy",
                    },
                    "gcs_trajectory_constraint_summary": {
                        "evaluated": True,
                        "backend_enforced": True,
                        "violation_count": violation_count,
                        "risk_flags": [] if violation_count == 0 else ["direction_cone_violation"],
                        "max_allowed_direction_error_deg": 45.0,
                        "eta": 1.0,
                        "rho_min": 0.025,
                        "rho_lower_bound_min_m": 0.025,
                        "objective_term_weights": {
                            "control_point_terrain_anchor_quadratic": terrain_weight,
                            "control_point_second_difference_quadratic": 0.2,
                            "control_point_high_cost_exposure_proxy_quadratic": high_cost_weight,
                        },
                    },
                    "gcs_candidate_selected": selected,
                    "gcs_candidate_fallback_reason": fallback,
                    "gcs_candidate_cost_delta_vs_baseline": -1.0 if selected else 1.0,
                    "gcs_candidate_cost_delta_vs_postprocess": -1.0 if selected else 1.0,
                    "gcs_candidate_cost_summary": {
                        "sampled_terrain_cost": 10.0 if is_scenario_a else 9.0,
                        "terrain_objective_weight": terrain_weight,
                        "high_cost_exposure_delta_vs_baseline": high_cost_delta,
                    },
                    "gcs_motion_feasibility_feasibility_status": "feasible",
                    "gcs_motion_feasibility_curvature_violation_count": 0,
                    "gcs_motion_feasibility_heading_violation_count": 0,
                }
                output_json.parent.mkdir(parents=True, exist_ok=True)
                output_json.write_text(json.dumps(payload), encoding="utf-8")
                print(json.dumps({"status": "ok", "output_json": str(output_json)}))
                """
            ),
            encoding="utf-8",
        )
        return script
