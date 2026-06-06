import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class GcsControlPointCostGateDecompositionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.script = self.repo_root / "scripts" / "analyze_gcs_control_point_cost_gate.py"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="gcs-cost-gate-decomposition-"))

    def test_decomposes_cost_gate_classes_and_markdown(self) -> None:
        summary_path = self._write_targeted_sweep_summary()
        output_root = self.temp_dir / "decomposition"

        completed = subprocess.run(
            [
                sys.executable,
                str(self.script),
                "--targeted-sweep-summary",
                str(summary_path),
                "--output-root",
                str(output_root),
            ],
            cwd=self.repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        status = json.loads(completed.stdout)
        self.assertEqual(
            status["summary"],
            str(output_root / "gcs-control-point-cost-gate-decomposition-summary.json"),
        )
        summary = json.loads(Path(status["summary"]).read_text(encoding="utf-8"))
        report = Path(status["report"]).read_text(encoding="utf-8")

        self.assertEqual(
            summary["schema_version"],
            "gcs-control-point-cost-gate-decomposition-summary/v1",
        )
        self.assertEqual(summary["source_targeted_sweep_summary"], str(summary_path))
        self.assertEqual(summary["case_count"], 7)
        self.assertFalse(summary["default_change_recommended"])
        classes = summary["cost_gate_class_counts"]
        self.assertEqual(classes["true_cost_dominated"], 1)
        self.assertEqual(classes["high_cost_exposure_blocked"], 1)
        self.assertEqual(classes["terrain_proxy_mismatch"], 1)
        self.assertEqual(classes["baseline_overlap_or_duplicate"], 1)
        self.assertEqual(classes["direction_cone_fixed_but_cost_blocked"], 1)
        self.assertEqual(classes["insufficient_cost_diagnostics"], 1)
        self.assertEqual(classes["safety_regression_excluded"], 1)

        rows = {row["scenario_id"]: row for row in summary["cases"]}
        self.assertEqual(rows["true-cost"]["control_point_terrain_cost"], 8.0)
        self.assertEqual(rows["true-cost"]["baseline_overlap_ratio"], 0.25)
        self.assertEqual(rows["true-cost"]["path_cost_delta"], 3.0)
        self.assertEqual(rows["true-cost"]["request_artifact"], str(self.temp_dir / "true-cost-request.json"))
        self.assertEqual(rows["missing-diagnostics"]["missing_diagnostics"], ["cost_delta", "terrain_cost"])
        self.assertEqual(rows["safe-regression"]["cost_gate_class"], "safety_regression_excluded")
        self.assertEqual(rows["direction-to-cost"]["cost_gate_class"], "direction_cone_fixed_but_cost_blocked")
        self.assertIn("direction_cone", rows["direction-to-cost"])
        self.assertIn("motion_feasibility", rows["direction-to-cost"])

        self.assertIn("# GCS Control-Point Cost-Gate Decomposition", report)
        self.assertIn("## Matrix: matrix-a", report)
        self.assertIn("### Scenario: true-cost", report)
        self.assertIn("true_cost_dominated", report)
        self.assertIn("must remain rejected", report)
        self.assertIn("candidate for terrain proxy or quality-gate review", report)

    def _write_targeted_sweep_summary(self) -> Path:
        cases = [
            self._case(
                "true-cost",
                cost_delta_vs_baseline=3.0,
                cost_delta_vs_postprocess=2.0,
                sampled_terrain_cost=11.0,
                route_payload={
                    "gcs_candidate_baseline_overlap_ratio": 0.25,
                    "gcs_trajectory_cost_summary": {
                        "control_point_terrain_cost": 8.0,
                        "sampled_terrain_cost": 11.0,
                    },
                },
            ),
            self._case(
                "high-exposure",
                cost_delta_vs_baseline=-1.0,
                cost_delta_vs_postprocess=-1.0,
                sampled_terrain_cost=9.0,
                control_point_terrain_cost=8.0,
                high_cost_exposure_delta_vs_baseline=4.0,
            ),
            self._case(
                "terrain-mismatch",
                cost_delta_vs_baseline=-1.0,
                cost_delta_vs_postprocess=-1.0,
                sampled_terrain_cost=2.0,
                control_point_terrain_cost=12.0,
            ),
            self._case(
                "duplicate",
                cost_delta_vs_baseline=0.5,
                cost_delta_vs_postprocess=0.5,
                sampled_terrain_cost=5.0,
                control_point_terrain_cost=5.0,
                baseline_overlap_ratio=1.0,
            ),
            self._case(
                "direction-to-cost",
                baseline_fallback_reason="direction_cone_constraint_violation",
                transition_class="direction_cone_to_cost_dominated",
                cost_delta_vs_baseline=1.0,
                cost_delta_vs_postprocess=1.0,
                sampled_terrain_cost=7.0,
                control_point_terrain_cost=7.0,
                direction_cone_violation_count=0,
                baseline_direction_cone_violation_count=1,
            ),
            self._case("missing-diagnostics"),
            self._case(
                "safe-regression",
                cost_delta_vs_baseline=1.0,
                cost_delta_vs_postprocess=1.0,
                sampled_terrain_cost=9.0,
                control_point_terrain_cost=9.0,
                safety_regression=True,
                safety_regression_reasons=["terrain_cost_regression"],
            ),
            {
                "matrix_id": "matrix-a",
                "scenario_id": "not-cost",
                "action_index": 8,
                "baseline_fallback_reason": "direction_cone_constraint_violation",
                "new_fallback_reason": "direction_cone_constraint_violation",
                "transition_class": "blocker_persistent",
            },
        ]
        summary_path = self.temp_dir / "targeted-sweep-summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "schema_version": "gcs-control-point-targeted-sweep-summary/v1",
                    "matrix_count": 1,
                    "candidate_count": len(cases),
                    "case_count": len(cases),
                    "route_artifact_count": len(cases),
                    "default_change_recommended": False,
                    "recommendation": "no_default_change_recommended",
                    "default_change_reason": "no_matrix_passed_safety_and_blocker_gates",
                    "cases": cases,
                }
            ),
            encoding="utf-8",
        )
        return summary_path

    def _case(
        self,
        scenario_id: str,
        *,
        baseline_fallback_reason: str = "cost_dominated",
        new_fallback_reason: str = "cost_dominated",
        transition_class: str = "cost_dominated_persistent",
        cost_delta_vs_baseline: float | None = None,
        cost_delta_vs_postprocess: float | None = None,
        sampled_terrain_cost: float | None = None,
        control_point_terrain_cost: float | None = None,
        high_cost_exposure_delta_vs_baseline: float | None = 0.0,
        baseline_overlap_ratio: float | None = None,
        direction_cone_violation_count: int = 0,
        baseline_direction_cone_violation_count: int = 0,
        safety_regression: bool = False,
        safety_regression_reasons: list[str] | None = None,
        route_payload: dict | None = None,
    ) -> dict:
        action_index = len(list(self.temp_dir.glob("*-request.json")))
        request_artifact = self.temp_dir / f"{scenario_id}-request.json"
        route_artifact = self.temp_dir / f"{scenario_id}-route.json"
        request_artifact.write_text(json.dumps({"scenario_id": scenario_id}), encoding="utf-8")
        payload = {
            "schema_version": "path-planner-route/v1",
            "gcs_candidate_cost_delta_vs_baseline": cost_delta_vs_baseline,
            "gcs_candidate_cost_delta_vs_postprocess": cost_delta_vs_postprocess,
            "gcs_candidate_baseline_overlap_ratio": baseline_overlap_ratio,
            "gcs_trajectory_collision_count": 0,
            "gcs_trajectory_constraint_summary": {
                "violation_count": direction_cone_violation_count,
                "risk_flags": [],
            },
            "gcs_motion_feasibility_feasibility_status": "feasible",
            "gcs_motion_feasibility_curvature_violation_count": 0,
            "gcs_motion_feasibility_heading_violation_count": 0,
            "gcs_trajectory_cost_summary": {
                "sampled_terrain_cost": sampled_terrain_cost,
                "control_point_terrain_cost": control_point_terrain_cost,
            },
            "gcs_candidate_cost_summary": {
                "high_cost_exposure_delta_vs_baseline": high_cost_exposure_delta_vs_baseline,
            },
        }
        if route_payload:
            payload.update(route_payload)
        route_artifact.write_text(json.dumps(payload), encoding="utf-8")
        return {
            "matrix_id": "matrix-a",
            "parameters": {"matrix_id": "matrix-a"},
            "scenario_id": scenario_id,
            "action_index": action_index,
            "baseline_fallback_reason": baseline_fallback_reason,
            "new_fallback_reason": new_fallback_reason,
            "transition_class": transition_class,
            "cost_delta_vs_baseline": cost_delta_vs_baseline,
            "cost_delta_vs_postprocess": cost_delta_vs_postprocess,
            "sampled_terrain_cost": sampled_terrain_cost,
            "control_point_terrain_cost": control_point_terrain_cost,
            "high_cost_exposure_delta_vs_baseline": high_cost_exposure_delta_vs_baseline,
            "baseline_overlap_ratio": baseline_overlap_ratio,
            "collision_count": 0,
            "direction_cone_violation_count": direction_cone_violation_count,
            "baseline_direction_cone_violation_count": baseline_direction_cone_violation_count,
            "motion_feasibility_status": "feasible",
            "motion_feasibility_curvature_violation_count": 0,
            "motion_feasibility_heading_violation_count": 0,
            "safety_regression": safety_regression,
            "safety_regression_reasons": safety_regression_reasons or [],
            "route_artifact": str(route_artifact),
            "command": [sys.executable, "--input", str(request_artifact)],
        }
