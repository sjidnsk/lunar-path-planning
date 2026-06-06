import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class PathFeedbackValidationScriptTests(unittest.TestCase):
    def test_primary_acceptance_chain_is_labeled_in_dry_run(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "run_path_feedback_validation.sh"
        output_root = Path(tempfile.mkdtemp(prefix="path-feedback-acceptance-")) / "out"

        completed = subprocess.run(
            [
                "bash",
                str(script),
                "--dry-run",
                "--scenario-set",
                "all",
                "--diagnostic-profile",
                "all",
                "--top-k",
                "3",
                "--output-root",
                str(output_root),
            ],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("Acceptance gate: semi-real-closed-loop", completed.stdout)
        self.assertIn("Scenario set: all", completed.stdout)
        self.assertIn("Diagnostic profile: all", completed.stdout)
        self.assertIn("Top-K: 3", completed.stdout)
        self.assertIn("--simulate-tracking", completed.stdout)
        self.assertIn("--optimize-trajectory", completed.stdout)
        self.assertIn("--drake-iris-regions", completed.stdout)
        self.assertIn("--gcs-trajectory-smoke", completed.stdout)
        self.assertIn("--gcs-geometric-candidate", completed.stdout)
        self.assertIn("--gcs-motion-feasibility", completed.stdout)
        self.assertIn("--gcs-curvature-constrained-candidate", completed.stdout)
        self.assertNotIn("--gcs-control-point-candidate", completed.stdout)
        self.assertFalse(output_root.exists())

    def test_control_point_gcs_candidate_is_explicit_opt_in(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "run_path_feedback_validation.sh"
        output_root = Path(tempfile.mkdtemp(prefix="path-feedback-control-point-")) / "out"

        completed = subprocess.run(
            [
                "bash",
                str(script),
                "--dry-run",
                "--scenario-set",
                "all",
                "--diagnostic-profile",
                "all",
                "--top-k",
                "3",
                "--gcs-control-point-candidate",
                "--output-root",
                str(output_root),
            ],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("--gcs-control-point-candidate", completed.stdout)
        self.assertFalse(output_root.exists())

    def test_control_point_calibration_args_are_explicitly_forwarded(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "run_path_feedback_validation.sh"
        output_root = Path(tempfile.mkdtemp(prefix="path-feedback-control-point-calibration-")) / "out"

        completed = subprocess.run(
            [
                "bash",
                str(script),
                "--dry-run",
                "--scenario-set",
                "all",
                "--diagnostic-profile",
                "all",
                "--top-k",
                "3",
                "--gcs-control-point-candidate",
                "--gcs-control-point-terrain-weight",
                "0.08",
                "--gcs-control-point-second-difference-weight",
                "0.35",
                "--gcs-control-point-direction-cone-max-error-deg",
                "35",
                "--gcs-control-point-direction-cone-rho-floor-m",
                "0.04",
                "--gcs-control-point-direction-cone-seed-rho-ratio",
                "0.08",
                "--output-root",
                str(output_root),
            ],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("--gcs-control-point-candidate", completed.stdout)
        self.assertIn("--gcs-control-point-terrain-weight 0.08", completed.stdout)
        self.assertIn("--gcs-control-point-second-difference-weight 0.35", completed.stdout)
        self.assertIn("--gcs-control-point-direction-cone-max-error-deg 35", completed.stdout)
        self.assertIn("--gcs-control-point-direction-cone-rho-floor-m 0.04", completed.stdout)
        self.assertIn("--gcs-control-point-direction-cone-seed-rho-ratio 0.08", completed.stdout)
        self.assertFalse(output_root.exists())

    def test_diagnostic_profiles_are_reflected_in_dry_run_commands(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "run_path_feedback_validation.sh"
        output_root = Path(tempfile.mkdtemp(prefix="path-feedback-profile-")) / "out"

        iris = subprocess.run(
            [
                "bash",
                str(script),
                "--dry-run",
                "--scenario-set",
                "all",
                "--diagnostic-profile",
                "iris",
                "--output-root",
                str(output_root),
            ],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(iris.returncode, 0, iris.stdout + iris.stderr)
        self.assertIn("Diagnostic profile: iris", iris.stdout)
        self.assertIn("--drake-iris-regions", iris.stdout)
        self.assertIn("--gcs-trajectory-smoke", iris.stdout)
        self.assertIn("--gcs-geometric-candidate", iris.stdout)
        self.assertIn("--gcs-motion-feasibility", iris.stdout)
        self.assertIn("--gcs-curvature-constrained-candidate", iris.stdout)
        self.assertNotIn("--gcs-control-point-candidate", iris.stdout)
        self.assertFalse(output_root.exists())

        execution = subprocess.run(
            [
                "bash",
                str(script),
                "--dry-run",
                "--diagnostic-profile",
                "execution",
                "--output-root",
                str(output_root),
            ],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(execution.returncode, 0, execution.stdout + execution.stderr)
        self.assertIn("--simulate-tracking", execution.stdout)
        self.assertIn("--optimize-trajectory", execution.stdout)

    def test_region_graph_guided_backend_is_reflected_in_dry_run_commands(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "run_path_feedback_validation.sh"
        output_root = Path(tempfile.mkdtemp(prefix="path-feedback-backend-")) / "out"

        completed = subprocess.run(
            [
                "bash",
                str(script),
                "--dry-run",
                "--planning-backend",
                "region_graph_guided",
                "--output-root",
                str(output_root),
            ],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("--planning-backend region_graph_guided", completed.stdout)
        self.assertFalse(output_root.exists())

    def test_mixed_stress_gate_accepts_sampled_region_decision_diagnostics(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "run_path_feedback_validation.sh"
        content = script.read_text(encoding="utf-8")

        self.assertIn("mixed_sampled_region_decision_diagnostics", content)
        self.assertIn("sampled_region_path_selected_count", content)
        self.assertIn("sampled-region decision diagnostics", content)

    def test_stress_gate_accepts_sampled_region_decision_diagnostics(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "run_path_feedback_validation.sh"
        content = script.read_text(encoding="utf-8")

        self.assertIn("stress_sampled_region_decision_diagnostics", content)
        self.assertIn("sampled_region_path_reachable_terminal_rescue_count", content)
        self.assertIn("stress scenarios must produce failure, replan, or sampled-region diagnostics", content)

    def test_dry_run_uses_auditable_python_executable_for_all_python_commands(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "run_path_feedback_validation.sh"
        output_root = Path(tempfile.mkdtemp(prefix="path-feedback-python-")) / "out"
        env = os.environ.copy()
        env["PYTHON"] = sys.executable

        completed = subprocess.run(
            [
                "bash",
                str(script),
                "--dry-run",
                "--diagnostic-profile",
                "iris",
                "--output-root",
                str(output_root),
            ],
            cwd=repo_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn(f"Python executable: {sys.executable}", completed.stdout)
        self.assertIn(f"{sys.executable} scripts/generate_npz_validation_maps.py", completed.stdout)
        self.assertIn(f"PYTHONPATH=src {sys.executable} scripts/export_path_planner_sidecars.py", completed.stdout)
        self.assertIn(f"PYTHONPATH=src {sys.executable} -m model_explorer path-feedback validate", completed.stdout)
        self.assertIn(f"PYTHONPATH=src {sys.executable} -m model_explorer path-feedback run", completed.stdout)
        self.assertNotIn(" python3 ", completed.stdout)
        self.assertFalse(output_root.exists())


if __name__ == "__main__":
    unittest.main()
