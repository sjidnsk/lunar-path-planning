import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class GcsControlPointTriageExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.script = self.repo_root / "scripts" / "export_gcs_control_point_candidate_triage.py"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="gcs-triage-export-"))

    def test_exports_triage_json_and_markdown(self) -> None:
        summary = self.temp_dir / "path-feedback-summary.json"
        output_json = self.temp_dir / "gcs-triage.json"
        output_markdown = self.temp_dir / "gcs-triage.md"
        summary.write_text(
            json.dumps(
                {
                    "schema_version": "path-feedback-summary/v1",
                    "gcs_control_point_candidate_triage": {
                        "schema_version": "gcs-control-point-candidate-triage-summary/v1",
                        "candidate_count": 1,
                        "attempted_count": 1,
                        "success_count": 1,
                        "selected_count": 0,
                        "route_artifact_count": 1,
                        "fallback_reason_counts": {"cost_dominated": 1},
                        "calibration_sweep": {
                            "schema_version": "gcs-control-point-candidate-calibration-sweep/v1",
                            "mode": "recorded_candidate_gate_diagnostics",
                            "solver_rerun_required": True,
                            "default_change_recommended": False,
                            "default_change_reason": "recorded_candidates_remain_blocked_by_quality_or_direction_cone_gate",
                        },
                        "candidates": [
                            {
                                "scenario_id": "npz_shadow_corridor",
                                "action_index": 0,
                                "candidate_selected": False,
                                "candidate_fallback_reason": "cost_dominated",
                                "route_artifact": "/tmp/path-planner-route.json",
                            }
                        ],
                    },
                }
            ),
            encoding="utf-8",
        )

        result = subprocess.run(
            [
                sys.executable,
                str(self.script),
                "--summary",
                str(summary),
                "--output-json",
                str(output_json),
                "--output-markdown",
                str(output_markdown),
            ],
            cwd=self.repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        exported = json.loads(output_json.read_text(encoding="utf-8"))
        self.assertEqual(
            exported["schema_version"],
            "gcs-control-point-candidate-triage-summary/v1",
        )
        self.assertEqual(exported["candidate_count"], 1)
        report = output_markdown.read_text(encoding="utf-8")
        self.assertIn("# GCS Control-Point Candidate Triage", report)
        self.assertIn("gcs-control-point-candidate-calibration-sweep/v1", report)
        self.assertIn("cost_dominated", report)

    def test_missing_triage_is_an_error(self) -> None:
        summary = self.temp_dir / "path-feedback-summary.json"
        summary.write_text(json.dumps({"schema_version": "path-feedback-summary/v1"}), encoding="utf-8")

        result = subprocess.run(
            [sys.executable, str(self.script), "--summary", str(summary)],
            cwd=self.repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("gcs_control_point_candidate_triage", result.stderr)
