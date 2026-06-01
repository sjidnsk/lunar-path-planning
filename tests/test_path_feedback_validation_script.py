import subprocess
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


if __name__ == "__main__":
    unittest.main()
