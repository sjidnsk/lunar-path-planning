import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class XunceFullNetworkStaticContractValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(tempfile.mkdtemp(prefix="xunce-static-contract-"))
        scripts_path = str(Path(__file__).resolve().parents[1] / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.stage8_root = self.repo_root / "outputs" / "stage8"
        self.output_root = self.repo_root / "outputs" / "stage9"
        self.config_path = self.repo_root / "configs" / "xunce_full_network_static_contract_validation_v1.json"
        self._write_stage8()
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.repo_root)

    def test_default_fixture_passes_and_writes_artifacts(self) -> None:
        from scripts.run_xunce_full_network_static_contract_validation import run_xunce_full_network_static_contract_validation

        summary = run_xunce_full_network_static_contract_validation(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["source_full_network_status"], "passed")
        self.assertEqual(summary["architecture"], "xunce_full_network_v1")
        self.assertEqual(summary["next_required_change"], "full_network_ablation_experiments")
        self.assertGreaterEqual(summary["contract_case_count"], 5)
        self.assertTrue(summary["shape_contract_passed"])
        self.assertTrue(summary["mask_contract_passed"])
        self.assertTrue(summary["metadata_contract_passed"])
        self.assertTrue(summary["missing_indicator_contract_passed"])
        self.assertTrue(summary["legacy_observation_compatibility_passed"])
        self.assertTrue(summary["finite_output_contract_passed"])
        self.assertTrue(summary["static_contract_validation_passed"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["modifies_network"])

        for filename in (
            "xunce-full-network-static-contract-validation-summary.json",
            "xunce-full-network-static-contract-validation-manifest.json",
            "xunce-full-network-static-contract-cases.jsonl",
            "xunce-full-network-static-shape-audit.json",
            "xunce-full-network-static-mask-audit.json",
            "xunce-full-network-static-metadata-audit.json",
            "xunce-full-network-static-compatibility-audit.json",
            "xunce-full-network-static-boundary-audit.json",
            "xunce-full-network-static-rejection-report.json",
            "xunce-full-network-static-contract-validation-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_missing_stage8_routes_to_full_network_fix(self) -> None:
        from scripts.run_xunce_full_network_static_contract_validation import run_xunce_full_network_static_contract_validation

        shutil.rmtree(self.stage8_root)
        summary = run_xunce_full_network_static_contract_validation(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_full_network_v1_summary", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_xunce_full_network_v1")

    def test_mask_contract_failure_blocks_ablation(self) -> None:
        from scripts.run_xunce_full_network_static_contract_validation import run_xunce_full_network_static_contract_validation

        self._write_config(mask_case_action_mask=[True, True, True])
        summary = run_xunce_full_network_static_contract_validation(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("static_mask_contract_failed", summary["reason_codes"])
        self.assertFalse(summary["mask_contract_passed"])
        self.assertEqual(summary["next_required_change"], "fix_full_network_static_contract_validation")

    def test_stage8_boundary_violation_fails(self) -> None:
        from scripts.run_xunce_full_network_static_contract_validation import run_xunce_full_network_static_contract_validation

        self._write_stage8(summary_updates={"publishes_checkpoint": True})
        summary = run_xunce_full_network_static_contract_validation(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("static_contract_boundary_violation", summary["reason_codes"])
        self.assertFalse(summary["boundary_audit_passed"])

    def _write_config(self, *, mask_case_action_mask: list[bool] | None = None) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "xunce-full-network-static-contract-validation-config/v1",
            "source_full_network_root": str(self.stage8_root),
            "architecture": "xunce_full_network_v1",
            "candidate_count": 3,
            "edge_count": 3,
            "candidate_feature_count": 4,
            "edge_feature_count": 3,
            "memory_feature_count": 2,
            "context_feature_count": 2,
            "missing_indicator_count": 1,
            "hidden_dim": 8,
            "message_passing_layers": 1,
            "dropout": 0.0,
            "seed": 9,
            "mask_case_action_mask": mask_case_action_mask or [True, False, True],
        }
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_stage8(self, *, summary_updates: dict | None = None) -> None:
        self.stage8_root.mkdir(parents=True, exist_ok=True)
        summary = {
            "schema_version": "xunce-full-network-v1-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "next_required_change": "full_network_static_contract_validation",
            "architecture": "xunce_full_network_v1",
            "full_network_v1_passed": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "runs_new_ppo_update": False,
            "modifies_network": False,
            "modifies_action_space": False,
            "modifies_default_astar": False,
        }
        if summary_updates:
            summary.update(summary_updates)
        self._write_json(self.stage8_root / "xunce-full-network-v1-summary.json", summary)

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
