import json
import subprocess
import tempfile
import unittest
from pathlib import Path


class DenseChokeSafeAlternativeOpportunityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        self.temp_dir = Path(tempfile.mkdtemp(prefix="dense-choke-opportunity-"))

    def tearDown(self) -> None:
        for path in sorted(self.temp_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        self.temp_dir.rmdir()

    def test_diagnosis_reports_dense_choke_rejection_root_causes(self) -> None:
        batch_root = self.temp_dir / "canary"
        batch_root.mkdir()
        summary = {
            "schema_version": "policy-gated-canary-opportunity-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "canary_opportunity_context_count": 4,
            "opportunities": [
                {
                    "run_id": f"run-{index}",
                    "scenario_id": f"npz_dense_{index}",
                    "scenario_group": "dense_choke_safe_bypass",
                    "source_context_id": f"source-{index}",
                    "alternative_count": 3,
                    "acceptable_alternative_count": 0,
                    "alternatives": [
                        {
                            "action_index": 0,
                            "source_action_index": None,
                            "policy_target_cell": [12, 8],
                            "path_cost_delta": -1.0,
                            "risk_delta": 0.3,
                            "canary_gate_acceptable": False,
                            "canary_gate_rejection_reason_codes": [
                                "invalid_action_mask",
                                "risk_regression",
                            ],
                        },
                        {
                            "action_index": 0,
                            "source_action_index": 0,
                            "policy_target_cell": [12, 8],
                            "path_cost_delta": 2.0,
                            "risk_delta": 0.3,
                            "canary_gate_acceptable": False,
                            "canary_gate_rejection_reason_codes": [
                                "path_cost_regression",
                                "risk_regression",
                            ],
                        },
                        {
                            "action_index": 3,
                            "source_action_index": 0,
                            "policy_target_cell": [12, 8],
                            "path_cost_delta": 2.0,
                            "risk_delta": 0.3,
                            "canary_gate_acceptable": False,
                            "canary_gate_rejection_reason_codes": [
                                "path_cost_regression",
                                "risk_regression",
                            ],
                        },
                    ],
                }
                for index in range(4)
            ],
        }
        (batch_root / "policy-gated-canary-opportunity-summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        completed = subprocess.run(
            [
                "bash",
                str(self.repo_root / "scripts" / "run_dense_choke_safe_alternative_diagnosis.sh"),
                "--batch-root",
                str(batch_root),
            ],
            cwd=self.repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        diagnosis = json.loads(
            (batch_root / "dense-choke-safe-alternative-diagnosis-summary.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(diagnosis["dense_choke_opportunity_context_count"], 4)
        self.assertEqual(diagnosis["dense_choke_alternative_count"], 12)
        self.assertEqual(diagnosis["dense_choke_acceptable_alternative_count"], 0)
        self.assertEqual(
            diagnosis["dense_choke_rejection_reason_counts"],
            {"invalid_action_mask": 4, "path_cost_regression": 8, "risk_regression": 12},
        )
        self.assertEqual(
            diagnosis["next_required_change"],
            "dense_choke_opportunity_generation_gap",
        )
        self.assertTrue(
            (batch_root / "dense-choke-safe-alternative-diagnosis.md").is_file()
        )

    def test_generator_exposes_dense_choke_opportunity_variants(self) -> None:
        completed = subprocess.run(
            [
                "/home/kai/anaconda3/envs/lunar-explorer/bin/python",
                str(
                    self.repo_root
                    / "dev-platform-constraints"
                    / "scripts"
                    / "generate_npz_validation_maps.py"
                ),
                "--scenario-set",
                "policy_canary_dense_choke_opportunity",
                "--dry-run",
            ],
            cwd=self.repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        payload = json.loads(completed.stdout)
        scenarios = payload["scenarios"]
        self.assertGreaterEqual(len(scenarios), 4)
        self.assertEqual({item["scenario_group"] for item in scenarios}, {"dense_choke_safe_bypass"})
        self.assertEqual(len({item["scenario_id"] for item in scenarios}), len(scenarios))
        self.assertEqual(len({item["seed"] for item in scenarios}), len(scenarios))
        self.assertEqual(
            len({item["scenario_variant_id"] for item in scenarios}),
            len(scenarios),
        )

    def test_full_family_config_requires_all_six_families(self) -> None:
        config_path = self.repo_root / "configs" / "policy_gated_canary_full_family_opportunity_v1.json"

        payload = json.loads(config_path.read_text(encoding="utf-8"))

        validation = payload["validation"]
        self.assertEqual(validation["min_scenario_family_count"], 6)
        self.assertEqual(validation["min_family_with_acceptable_alternative_count"], 6)
        self.assertEqual(validation["min_accepted_scenario_family_count"], 6)
        self.assertEqual(validation["min_canary_accepted_policy_choice_count"], 12)

    def test_full_family_closure_script_references_required_roots(self) -> None:
        script = self.repo_root / "scripts" / "run_dense_choke_safe_alternative_opportunity_closure.sh"

        completed = subprocess.run(
            ["bash", "-n", str(script)],
            cwd=self.repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        text = script.read_text(encoding="utf-8")
        self.assertIn("path_feedback_batch_dense_choke_opportunity_clean_src_v1", text)
        self.assertIn("path_feedback_batch_dense_choke_opportunity_candidate_v1", text)
        self.assertIn("path_feedback_batch_policy_gated_canary_full_family_opportunity_v1", text)
        self.assertIn("run_dense_choke_safe_alternative_diagnosis.sh", text)


if __name__ == "__main__":
    unittest.main()
