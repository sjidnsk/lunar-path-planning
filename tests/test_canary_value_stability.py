import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class CanaryValueStabilityScenarioTests(unittest.TestCase):
    def test_value_stability_matrix_accepts_new_scenario_set(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        matrix = {
            "schema_version": "path-feedback-batch-matrix/v1",
            "output_root": "outputs/test-canary-value-stability",
            "runs": [
                {
                    "run_id": "value-stability-unit",
                    "scenario_set": "policy_canary_value_stability",
                    "diagnostic_profile": "execution",
                    "top_k": 3,
                }
            ],
        }
        matrix_path = Path(tempfile.mkdtemp(prefix="canary-value-matrix-")) / "matrix.json"
        matrix_path.write_text(json.dumps(matrix, indent=2), encoding="utf-8")

        completed = subprocess.run(
            [
                "bash",
                str(repo_root / "scripts" / "run_batch_path_feedback_validation.sh"),
                "--matrix",
                str(matrix_path),
                "--validate-only",
            ],
            cwd=repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)


if __name__ == "__main__":
    unittest.main()
