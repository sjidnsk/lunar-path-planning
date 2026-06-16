import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class XunceFullNetworkAblationExperimentsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(tempfile.mkdtemp(prefix="xunce-ablation-"))
        scripts_path = str(Path(__file__).resolve().parents[1] / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.stage9_root = self.repo_root / "outputs" / "stage9"
        self.output_root = self.repo_root / "outputs" / "stage10"
        self.config_path = self.repo_root / "configs" / "xunce_full_network_ablation_experiments_v1.json"
        self._write_stage9()
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.repo_root)

    def test_default_fixture_passes_and_writes_artifacts(self) -> None:
        from scripts.run_xunce_full_network_ablation_experiments import run_xunce_full_network_ablation_experiments

        summary = run_xunce_full_network_ablation_experiments(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["next_required_change"], "full_network_stress_evaluation")
        self.assertEqual(summary["architecture"], "xunce_full_network_v1")
        self.assertEqual(summary["ablation_case_count"], 5)
        self.assertTrue(summary["topology_ablation_effect_detected"])
        self.assertTrue(summary["memory_ablation_effect_detected"])
        self.assertTrue(summary["context_ablation_effect_detected"])
        self.assertTrue(summary["missing_indicator_ablation_effect_detected"])
        self.assertTrue(summary["mask_preserved"])
        self.assertTrue(summary["finite_outputs_preserved"])
        self.assertTrue(summary["ablation_experiments_passed"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["modifies_network"])

        for filename in (
            "xunce-full-network-ablation-experiments-summary.json",
            "xunce-full-network-ablation-experiments-manifest.json",
            "xunce-full-network-ablation-results.jsonl",
            "xunce-full-network-ablation-module-contribution-audit.json",
            "xunce-full-network-ablation-ranking-delta-audit.json",
            "xunce-full-network-ablation-boundary-audit.json",
            "xunce-full-network-ablation-rejection-report.json",
            "xunce-full-network-ablation-experiments-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_missing_stage9_routes_to_static_contract_fix(self) -> None:
        from scripts.run_xunce_full_network_ablation_experiments import run_xunce_full_network_ablation_experiments

        shutil.rmtree(self.stage9_root)
        summary = run_xunce_full_network_ablation_experiments(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_static_contract_validation_summary", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_full_network_static_contract_validation")

    def test_high_delta_threshold_blocks_stress_evaluation(self) -> None:
        from scripts.run_xunce_full_network_ablation_experiments import run_xunce_full_network_ablation_experiments

        self._write_config(min_logit_delta=100.0)
        summary = run_xunce_full_network_ablation_experiments(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("ablation_module_contribution_too_small", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_full_network_ablation_experiments")

    def test_stage9_boundary_violation_fails(self) -> None:
        from scripts.run_xunce_full_network_ablation_experiments import run_xunce_full_network_ablation_experiments

        self._write_stage9(summary_updates={"publishes_checkpoint": True})
        summary = run_xunce_full_network_ablation_experiments(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("ablation_boundary_violation", summary["reason_codes"])
        self.assertFalse(summary["boundary_audit_passed"])

    def _write_config(self, *, min_logit_delta: float = 1.0e-8) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "xunce-full-network-ablation-experiments-config/v1",
            "source_static_contract_root": str(self.stage9_root),
            "architecture": "xunce_full_network_v1",
            "candidate_count": 5,
            "edge_count": 10,
            "candidate_feature_count": 9,
            "edge_feature_count": 6,
            "memory_feature_count": 7,
            "context_feature_count": 6,
            "missing_indicator_count": 9,
            "hidden_dim": 16,
            "message_passing_layers": 2,
            "dropout": 0.0,
            "seed": 23,
            "min_logit_delta": min_logit_delta,
            "action_mask": [True, False, True, True, True],
        }
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_stage9(self, *, summary_updates: dict | None = None) -> None:
        self.stage9_root.mkdir(parents=True, exist_ok=True)
        summary = {
            "schema_version": "xunce-full-network-static-contract-validation-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "next_required_change": "full_network_ablation_experiments",
            "architecture": "xunce_full_network_v1",
            "static_contract_validation_passed": True,
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
        self._write_json(self.stage9_root / "xunce-full-network-static-contract-validation-summary.json", summary)

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
