import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class XunceSandboxCandidatePreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(tempfile.mkdtemp(prefix="xunce-sandbox-preflight-"))
        scripts_path = str(Path(__file__).resolve().parents[1] / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.preflight_root = self.repo_root / "outputs" / "stage12"
        self.training_root = self.repo_root / "outputs" / "stage13"
        self.post_training_root = self.repo_root / "outputs" / "stage14"
        self.shadow_root = self.repo_root / "outputs" / "stage15"
        self.output_root = self.repo_root / "outputs" / "stage16"
        self.training_config_path = self.repo_root / "configs" / "xunce_controlled_training_candidate_v1.json"
        self.post_training_config_path = self.repo_root / "configs" / "xunce_post_training_offline_evaluation_v1.json"
        self.shadow_config_path = self.repo_root / "configs" / "xunce_shadow_replay_validation_v1.json"
        self.config_path = self.repo_root / "configs" / "xunce_sandbox_candidate_preflight_v1.json"
        self._write_preflight_summary()
        self._write_training_config()
        self._run_stage13()
        self._write_post_training_config()
        self._run_stage14()
        self._write_shadow_config()
        self._run_stage15()
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.repo_root)

    def test_default_fixture_packages_and_loads_sandbox_candidate(self) -> None:
        from scripts.run_xunce_sandbox_candidate_preflight import run_xunce_sandbox_candidate_preflight

        summary = run_xunce_sandbox_candidate_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["next_required_change"], "xunce_release_governance_gate")
        self.assertTrue(summary["sandbox_candidate_preflight_passed"])
        self.assertTrue(summary["sandbox_package_created"])
        self.assertTrue(summary["sandbox_checkpoint_hash_verified"])
        self.assertTrue(summary["sandbox_load_verified"])
        self.assertTrue(summary["kill_switch_audit_passed"])
        self.assertTrue(summary["rollback_audit_passed"])
        self.assertTrue(summary["telemetry_audit_passed"])
        self.assertTrue(summary["default_policy_read_only"])
        self.assertTrue(summary["executor_isolation_passed"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])

        for filename in (
            "xunce-sandbox-candidate-preflight-summary.json",
            "xunce-sandbox-candidate-preflight-manifest.json",
            "xunce-sandbox-package-manifest.json",
            "xunce-sandbox-checkpoint-hash-audit.json",
            "xunce-sandbox-load-audit.json",
            "xunce-sandbox-kill-switch-audit.json",
            "xunce-sandbox-rollback-audit.json",
            "xunce-sandbox-telemetry-audit.json",
            "xunce-sandbox-boundary-audit.json",
            "xunce-sandbox-rejection-report.json",
            "xunce-sandbox-candidate-preflight-report.md",
            "sandbox_package/xunce-controlled-training-candidate.pt",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_missing_shadow_source_routes_to_shadow_fix(self) -> None:
        from scripts.run_xunce_sandbox_candidate_preflight import run_xunce_sandbox_candidate_preflight

        shutil.rmtree(self.shadow_root)
        summary = run_xunce_sandbox_candidate_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_shadow_replay_summary", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_xunce_shadow_replay_validation")

    def test_missing_checkpoint_routes_to_checkpoint_fix(self) -> None:
        from scripts.run_xunce_sandbox_candidate_preflight import run_xunce_sandbox_candidate_preflight

        (self.training_root / "xunce-controlled-training-candidate.pt").unlink()
        summary = run_xunce_sandbox_candidate_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_research_checkpoint", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_controlled_training_candidate_checkpoint")
        self.assertFalse(summary["sandbox_package_created"])

    def test_invalid_checkpoint_routes_to_checkpoint_fix(self) -> None:
        from scripts.run_xunce_sandbox_candidate_preflight import run_xunce_sandbox_candidate_preflight

        (self.training_root / "xunce-controlled-training-candidate.pt").write_text("not a checkpoint", encoding="utf-8")
        summary = run_xunce_sandbox_candidate_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("invalid_research_checkpoint", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_controlled_training_candidate_checkpoint")
        self.assertFalse(summary["sandbox_load_verified"])

    def test_shadow_boundary_violation_blocks_sandbox_preflight(self) -> None:
        from scripts.run_xunce_sandbox_candidate_preflight import run_xunce_sandbox_candidate_preflight

        summary_path = self.shadow_root / "xunce-shadow-replay-validation-summary.json"
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["connects_real_executor"] = True
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = run_xunce_sandbox_candidate_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("sandbox_source_boundary_violation", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "resolve_xunce_sandbox_candidate_boundary_rejections")
        self.assertFalse(summary["sandbox_package_created"])

    def _run_stage13(self) -> None:
        from scripts.run_xunce_controlled_training_candidate import run_xunce_controlled_training_candidate

        summary = run_xunce_controlled_training_candidate(
            config_path=self.training_config_path,
            output_root=self.training_root,
            repo_root=self.repo_root,
        )
        self.assertEqual(summary["status"], "passed")

    def _run_stage14(self) -> None:
        from scripts.run_xunce_post_training_offline_evaluation import run_xunce_post_training_offline_evaluation

        summary = run_xunce_post_training_offline_evaluation(
            config_path=self.post_training_config_path,
            output_root=self.post_training_root,
            repo_root=self.repo_root,
        )
        self.assertEqual(summary["status"], "passed")

    def _run_stage15(self) -> None:
        from scripts.run_xunce_shadow_replay_validation import run_xunce_shadow_replay_validation

        summary = run_xunce_shadow_replay_validation(
            config_path=self.shadow_config_path,
            output_root=self.shadow_root,
            repo_root=self.repo_root,
        )
        self.assertEqual(summary["status"], "passed")

    def _write_config(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "xunce-sandbox-candidate-preflight-config/v1",
            "source_shadow_replay_root": str(self.shadow_root),
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
            "kill_switch_required": True,
            "rollback_required": True,
            "telemetry_required": True,
            "require_executor_isolation": True,
        }
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_shadow_config(self) -> None:
        self.shadow_config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "xunce-shadow-replay-validation-config/v1",
            "source_post_training_root": str(self.post_training_root),
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
            "source_match_probability_tolerance": 1.0e-12,
            "source_match_logit_tolerance": 1.0e-12,
            "source_match_loss_tolerance": 1.0e-12,
        }
        self.shadow_config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_post_training_config(self) -> None:
        self.post_training_config_path.parent.mkdir(parents=True, exist_ok=True)
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
        self.post_training_config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

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
