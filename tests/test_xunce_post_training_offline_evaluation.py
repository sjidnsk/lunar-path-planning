import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch


class XuncePostTrainingOfflineEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(tempfile.mkdtemp(prefix="xunce-post-training-eval-"))
        scripts_path = str(Path(__file__).resolve().parents[1] / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.preflight_root = self.repo_root / "outputs" / "stage12"
        self.training_root = self.repo_root / "outputs" / "stage13"
        self.output_root = self.repo_root / "outputs" / "stage14"
        self.training_config_path = self.repo_root / "configs" / "xunce_controlled_training_candidate_v1.json"
        self.config_path = self.repo_root / "configs" / "xunce_post_training_offline_evaluation_v1.json"
        self._write_preflight_summary()
        self._write_training_config()
        self._run_stage13_training_candidate()
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.repo_root)

    def test_default_fixture_loads_checkpoint_read_only_and_passes(self) -> None:
        from scripts.run_xunce_post_training_offline_evaluation import run_xunce_post_training_offline_evaluation

        summary = run_xunce_post_training_offline_evaluation(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["next_required_change"], "shadow_replay_validation")
        self.assertTrue(summary["checkpoint_loaded"])
        self.assertTrue(summary["checkpoint_read_only"])
        self.assertTrue(summary["post_training_offline_evaluation_passed"])
        self.assertTrue(summary["trained_loss_lower_than_fresh"])
        self.assertTrue(summary["target_probability_improved"])
        self.assertGreater(summary["mean_target_probability_delta"], 0.0)
        self.assertFalse(summary["runs_new_training_update"])
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])

        for filename in (
            "xunce-post-training-offline-evaluation-summary.json",
            "xunce-post-training-offline-evaluation-manifest.json",
            "xunce-post-training-offline-evaluation-results.jsonl",
            "xunce-post-training-checkpoint-load-audit.json",
            "xunce-post-training-policy-delta-audit.json",
            "xunce-post-training-boundary-audit.json",
            "xunce-post-training-rejection-report.json",
            "xunce-post-training-offline-evaluation-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_missing_source_routes_to_training_candidate_fix(self) -> None:
        from scripts.run_xunce_post_training_offline_evaluation import run_xunce_post_training_offline_evaluation

        shutil.rmtree(self.training_root)
        summary = run_xunce_post_training_offline_evaluation(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_controlled_training_summary", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_controlled_training_candidate")

    def test_missing_checkpoint_routes_to_checkpoint_fix(self) -> None:
        from scripts.run_xunce_post_training_offline_evaluation import run_xunce_post_training_offline_evaluation

        (self.training_root / "xunce-controlled-training-candidate.pt").unlink()
        summary = run_xunce_post_training_offline_evaluation(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_research_checkpoint", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_controlled_training_candidate_checkpoint")
        self.assertFalse(summary["checkpoint_loaded"])

    def test_regressed_checkpoint_fails_offline_metric_gate(self) -> None:
        from scripts.xunce_full_network_common import XunceFullNetworkV1
        from scripts.run_xunce_post_training_offline_evaluation import run_xunce_post_training_offline_evaluation

        torch.manual_seed(17)
        model = XunceFullNetworkV1(
            candidate_feature_count=8,
            edge_feature_count=5,
            memory_feature_count=6,
            context_feature_count=7,
            missing_indicator_count=3,
            hidden_dim=32,
            message_passing_layers=2,
            dropout=0.0,
        )
        torch.save(
            {
                "schema_version": "xunce-controlled-training-candidate-checkpoint/v1",
                "model_state_dict": model.state_dict(),
                "metadata": {"architecture": "xunce_full_network_v1"},
            },
            self.training_root / "xunce-controlled-training-candidate.pt",
        )
        summary = run_xunce_post_training_offline_evaluation(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("post_training_metric_regression", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_post_training_offline_evaluation")

    def test_source_boundary_violation_blocks_evaluation(self) -> None:
        from scripts.run_xunce_post_training_offline_evaluation import run_xunce_post_training_offline_evaluation

        summary_path = self.training_root / "xunce-controlled-training-candidate-summary.json"
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["publishes_checkpoint"] = True
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = run_xunce_post_training_offline_evaluation(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("post_training_source_boundary_violation", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "resolve_xunce_post_training_offline_boundary_rejections")
        self.assertFalse(summary["checkpoint_loaded"])

    def _run_stage13_training_candidate(self) -> None:
        from scripts.run_xunce_controlled_training_candidate import run_xunce_controlled_training_candidate

        summary = run_xunce_controlled_training_candidate(
            config_path=self.training_config_path,
            output_root=self.training_root,
            repo_root=self.repo_root,
        )
        self.assertEqual(summary["status"], "passed")

    def _write_config(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "xunce-post-training-offline-evaluation-config/v1",
            "source_controlled_training_root": str(self.training_root),
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
            "evaluation_case_count": 3,
            "min_target_probability_delta": 1.0e-8,
        }
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_training_config(self) -> None:
        self.training_config_path.parent.mkdir(parents=True, exist_ok=True)
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
        self.training_config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_preflight_summary(self) -> None:
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
        path = self.preflight_root / "xunce-guarded-training-candidate-preflight-summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
