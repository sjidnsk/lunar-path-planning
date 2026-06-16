import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class XunceControlledTrainingCandidateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(tempfile.mkdtemp(prefix="xunce-controlled-training-"))
        scripts_path = str(Path(__file__).resolve().parents[1] / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.preflight_root = self.repo_root / "outputs" / "stage12"
        self.output_root = self.repo_root / "outputs" / "stage13"
        self.config_path = self.repo_root / "configs" / "xunce_controlled_training_candidate_v1.json"
        self._write_preflight_summary()
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.repo_root)

    def test_default_fixture_runs_bounded_training_and_writes_research_checkpoint(self) -> None:
        from scripts.run_xunce_controlled_training_candidate import run_xunce_controlled_training_candidate

        summary = run_xunce_controlled_training_candidate(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["next_required_change"], "post_training_offline_evaluation")
        self.assertTrue(summary["controlled_training_candidate_passed"])
        self.assertTrue(summary["runs_controlled_training_update"])
        self.assertTrue(summary["loss_decreased"])
        self.assertGreater(summary["loss_before"], summary["loss_after"])
        self.assertGreater(summary["training_step_count"], 0)
        self.assertEqual(summary["non_finite_gradient_count"], 0)
        self.assertTrue(summary["writes_research_checkpoint"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["ppo_update_executed"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])

        for filename in (
            "xunce-controlled-training-candidate-summary.json",
            "xunce-controlled-training-candidate-manifest.json",
            "xunce-controlled-training-loss-audit.json",
            "xunce-controlled-training-gradient-audit.json",
            "xunce-controlled-training-checkpoint-metadata.json",
            "xunce-controlled-training-boundary-audit.json",
            "xunce-controlled-training-rejection-report.json",
            "xunce-controlled-training-candidate-report.md",
            "xunce-controlled-training-candidate.pt",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_missing_preflight_routes_to_preflight_fix(self) -> None:
        from scripts.run_xunce_controlled_training_candidate import run_xunce_controlled_training_candidate

        shutil.rmtree(self.preflight_root)
        summary = run_xunce_controlled_training_candidate(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_training_preflight_summary", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_guarded_training_candidate_preflight")
        self.assertFalse(summary["controlled_training_candidate_passed"])
        self.assertFalse(summary["writes_research_checkpoint"])

    def test_preflight_boundary_violation_blocks_training(self) -> None:
        from scripts.run_xunce_controlled_training_candidate import run_xunce_controlled_training_candidate

        self._write_preflight_summary({"publishes_checkpoint": True})
        summary = run_xunce_controlled_training_candidate(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("controlled_training_source_boundary_violation", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "resolve_xunce_controlled_training_boundary_rejections")
        self.assertFalse(summary["runs_controlled_training_update"])
        self.assertFalse((self.output_root / "xunce-controlled-training-candidate.pt").exists())

    def test_insufficient_loss_improvement_fails_without_publishing_checkpoint(self) -> None:
        from scripts.run_xunce_controlled_training_candidate import run_xunce_controlled_training_candidate

        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        config["min_loss_improvement"] = 100.0
        self.config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = run_xunce_controlled_training_candidate(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("loss_improvement_too_small", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_controlled_training_candidate")
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])

    def _write_config(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "xunce-controlled-training-candidate-config/v1",
            "source_training_preflight_root": str(self.preflight_root),
            "architecture": "xunce_full_network_v1",
            "seed": 17,
            "candidate_feature_count": 8,
            "edge_feature_count": 5,
            "memory_feature_count": 6,
            "context_feature_count": 7,
            "missing_indicator_count": 3,
            "hidden_dim": 32,
            "message_passing_layers": 2,
            "candidate_count": 6,
            "training_step_limit": 4,
            "learning_rate": 0.005,
            "value_loss_weight": 0.1,
            "min_loss_improvement": 1.0e-6,
            "max_gradient_norm": 10.0,
            "write_research_checkpoint": True,
        }
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_preflight_summary(self, updates: dict | None = None) -> None:
        payload = {
            "schema_version": "xunce-guarded-training-candidate-preflight-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "architecture": "xunce_full_network_v1",
            "training_candidate_preflight_passed": True,
            "controlled_training_candidate_authorized": True,
            "next_required_change": "controlled_training_candidate",
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "runs_new_ppo_update": False,
            "modifies_network": False,
            "modifies_action_space": False,
            "modifies_default_astar": False,
            "real_world_release_approved": False,
            "real_world_performance_claimed": False,
        }
        if updates:
            payload.update(updates)
        path = self.preflight_root / "xunce-guarded-training-candidate-preflight-summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
