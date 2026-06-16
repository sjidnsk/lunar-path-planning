import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class XunceFullNetworkStressEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(tempfile.mkdtemp(prefix="xunce-stress-"))
        scripts_path = str(Path(__file__).resolve().parents[1] / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.stage10_root = self.repo_root / "outputs" / "stage10"
        self.output_root = self.repo_root / "outputs" / "stage11"
        self.config_path = self.repo_root / "configs" / "xunce_full_network_stress_evaluation_v1.json"
        self._write_stage10()
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.repo_root)

    def test_default_fixture_passes_and_writes_artifacts(self) -> None:
        from scripts.run_xunce_full_network_stress_evaluation import run_xunce_full_network_stress_evaluation

        summary = run_xunce_full_network_stress_evaluation(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["next_required_change"], "guarded_training_candidate_preflight")
        self.assertEqual(summary["architecture"], "xunce_full_network_v1")
        self.assertGreaterEqual(summary["stress_case_count"], 3)
        self.assertGreaterEqual(summary["max_candidate_count"], 8)
        self.assertTrue(summary["mask_preserved"])
        self.assertEqual(summary["non_finite_output_count"], 0)
        self.assertTrue(summary["deterministic_replay_passed"])
        self.assertTrue(summary["latency_gate_passed"])
        self.assertTrue(summary["parameter_gate_passed"])
        self.assertEqual(summary["fallback_rate"], 0.0)
        self.assertTrue(summary["stress_evaluation_passed"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["modifies_network"])

        for filename in (
            "xunce-full-network-stress-evaluation-summary.json",
            "xunce-full-network-stress-evaluation-manifest.json",
            "xunce-full-network-stress-case-results.jsonl",
            "xunce-full-network-stress-latency-audit.json",
            "xunce-full-network-stress-determinism-audit.json",
            "xunce-full-network-stress-boundary-audit.json",
            "xunce-full-network-stress-rejection-report.json",
            "xunce-full-network-stress-evaluation-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_missing_stage10_routes_to_ablation_fix(self) -> None:
        from scripts.run_xunce_full_network_stress_evaluation import run_xunce_full_network_stress_evaluation

        shutil.rmtree(self.stage10_root)
        summary = run_xunce_full_network_stress_evaluation(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_ablation_experiments_summary", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_full_network_ablation_experiments")

    def test_parameter_gate_failure_blocks_training_preflight(self) -> None:
        from scripts.run_xunce_full_network_stress_evaluation import run_xunce_full_network_stress_evaluation

        self._write_config(max_parameter_count=1)
        summary = run_xunce_full_network_stress_evaluation(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("stress_parameter_gate_failed", summary["reason_codes"])
        self.assertFalse(summary["parameter_gate_passed"])

    def test_stage10_boundary_violation_fails(self) -> None:
        from scripts.run_xunce_full_network_stress_evaluation import run_xunce_full_network_stress_evaluation

        self._write_stage10(summary_updates={"publishes_checkpoint": True})
        summary = run_xunce_full_network_stress_evaluation(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("stress_boundary_violation", summary["reason_codes"])
        self.assertFalse(summary["boundary_audit_passed"])

    def _write_config(self, *, max_parameter_count: int = 50000) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "xunce-full-network-stress-evaluation-config/v1",
            "source_ablation_root": str(self.stage10_root),
            "architecture": "xunce_full_network_v1",
            "candidate_counts": [2, 5, 8],
            "candidate_feature_count": 9,
            "edge_feature_count": 6,
            "memory_feature_count": 7,
            "context_feature_count": 6,
            "missing_indicator_count": 9,
            "hidden_dim": 16,
            "message_passing_layers": 2,
            "dropout": 0.0,
            "seed": 31,
            "max_parameter_count": max_parameter_count,
            "max_forward_latency_ms": 1000.0,
            "deterministic_tolerance": 1e-12
        }
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_stage10(self, *, summary_updates: dict | None = None) -> None:
        self.stage10_root.mkdir(parents=True, exist_ok=True)
        summary = {
            "schema_version": "xunce-full-network-ablation-experiments-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "next_required_change": "full_network_stress_evaluation",
            "architecture": "xunce_full_network_v1",
            "ablation_experiments_passed": True,
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
        self._write_json(self.stage10_root / "xunce-full-network-ablation-experiments-summary.json", summary)

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
