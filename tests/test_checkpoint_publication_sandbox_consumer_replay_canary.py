import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class CheckpointPublicationSandboxConsumerReplayCanaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts = str(self.repo_root / "scripts")
        model_src = str(self.repo_root / "model-explorer" / "src")
        for path in (scripts, model_src):
            if path not in sys.path:
                sys.path.insert(0, path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="sandbox-consumer-replay-"))
        self.stage14_root = self.temp_dir / "stage14"
        self.stage13_root = self.temp_dir / "stage13"
        self.promotion_root = self.temp_dir / "promotion"
        self.output_root = self.temp_dir / "s15"
        for path in (self.stage14_root, self.stage13_root, self.promotion_root):
            path.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_runs_consumer_replay_canary_without_publication_or_replacement(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_consumer_replay_canary import (
            run_checkpoint_publication_sandbox_consumer_replay_canary,
        )

        self._write_inputs()

        summary = run_checkpoint_publication_sandbox_consumer_replay_canary(
            stage14_root=self.stage14_root,
            stage13_root=self.stage13_root,
            promotion_root=self.promotion_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(
            summary["consumer_replay_canary_verdict"],
            "eligible_for_default_policy_candidate_authorization_preflight",
        )
        self.assertTrue(summary["checkpoint_publication_sandbox_consumer_replay_canary_passed"])
        self.assertTrue(summary["default_policy_candidate_authorization_preflight_approved"])
        self.assertGreaterEqual(summary["consumer_step_count"], 64)
        self.assertEqual(summary["missing_observation_count"], 0)
        self.assertEqual(summary["invalid_action_mask_count"], 0)
        self.assertEqual(summary["empty_action_mask_count"], 0)
        self.assertEqual(summary["non_finite_logits_count"], 0)
        self.assertEqual(summary["non_finite_log_prob_count"], 0)
        self.assertEqual(summary["non_finite_value_count"], 0)
        self.assertEqual(summary["controlled_regression_count"], 0)
        self.assertLess(summary["fallback_rate"], 0.5)
        self.assertTrue(summary["telemetry_audit_passed"])
        self.assertTrue(summary["rollback_audit_passed"])
        self.assertTrue(summary["release_boundary_audit_passed"])
        self.assertEqual(summary["sandbox_consumer_checkpoint_sha256"], self.checkpoint_sha256)
        self.assertEqual(summary["sandbox_consumer_checkpoint_size_bytes"], self.checkpoint_size)
        self.assertEqual(summary["next_required_change"], "default_policy_candidate_authorization_preflight")
        self.assertFalse(summary["checkpoint_publication_approved"])
        self.assertFalse(summary["default_policy_replacement_approved"])
        self.assertFalse(summary["real_executor_connection_approved"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])

        step_rows = [
            json.loads(line)
            for line in Path(summary["consumer_step_results"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(step_rows), summary["consumer_step_count"])
        self.assertTrue(all(row["telemetry_recorded"] for row in step_rows))
        self.assertTrue(all(row["selected_action_mask_valid"] for row in step_rows))

    def test_stage14_failure_blocks_consumer_replay(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_consumer_replay_canary import (
            run_checkpoint_publication_sandbox_consumer_replay_canary,
        )

        self._write_inputs()
        self._patch_stage14_summary({"status": "failed", "reason_codes": ["sandbox_identity_mismatch"]})

        summary = run_checkpoint_publication_sandbox_consumer_replay_canary(
            stage14_root=self.stage14_root,
            stage13_root=self.stage13_root,
            promotion_root=self.promotion_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("stage14_not_passed", summary["reason_codes"])

    def test_invalid_action_mask_blocks_consumer_replay(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_consumer_replay_canary import (
            run_checkpoint_publication_sandbox_consumer_replay_canary,
        )

        self._write_inputs()

        summary = run_checkpoint_publication_sandbox_consumer_replay_canary(
            stage14_root=self.stage14_root,
            stage13_root=self.stage13_root,
            promotion_root=self.promotion_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
            observation_mutator=lambda observation: {
                **observation,
                "action_mask": [False for _ in observation["action_mask"]],
            },
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("invalid_action_mask_detected", summary["reason_codes"])
        self.assertIn("consumer_replay_step_count_insufficient", summary["reason_codes"])
        self.assertIn("telemetry_missing_consumer_decision", summary["reason_codes"])

    def test_missing_sandbox_checkpoint_blocks_consumer_replay(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_consumer_replay_canary import (
            run_checkpoint_publication_sandbox_consumer_replay_canary,
        )

        self._write_inputs()
        self.sandbox_checkpoint_path.unlink()

        summary = run_checkpoint_publication_sandbox_consumer_replay_canary(
            stage14_root=self.stage14_root,
            stage13_root=self.stage13_root,
            promotion_root=self.promotion_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("sandbox_checkpoint_missing", summary["reason_codes"])

    def test_sandbox_identity_mismatch_blocks_consumer_replay(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_consumer_replay_canary import (
            run_checkpoint_publication_sandbox_consumer_replay_canary,
        )

        self._write_inputs()
        manifest_path = self.stage14_root / "checkpoint-publication-sandbox-consumer-verification-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["sandbox_consumer_checkpoint_sha256"] = "wrong"
        self._write_json(manifest_path, manifest)

        summary = run_checkpoint_publication_sandbox_consumer_replay_canary(
            stage14_root=self.stage14_root,
            stage13_root=self.stage13_root,
            promotion_root=self.promotion_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("sandbox_identity_mismatch", summary["reason_codes"])

    def test_non_finite_inference_blocks_consumer_replay(self) -> None:
        import torch

        from scripts.run_checkpoint_publication_sandbox_consumer_replay_canary import (
            run_checkpoint_publication_sandbox_consumer_replay_canary,
        )

        self._write_inputs()
        checkpoint = torch.load(self.sandbox_checkpoint_path, map_location="cpu", weights_only=False)
        first_key = next(iter(checkpoint["model_state_dict"]))
        checkpoint["model_state_dict"][first_key].fill_(float("nan"))
        torch.save(checkpoint, self.sandbox_checkpoint_path)

        summary = run_checkpoint_publication_sandbox_consumer_replay_canary(
            stage14_root=self.stage14_root,
            stage13_root=self.stage13_root,
            promotion_root=self.promotion_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("consumer_inference_non_finite", summary["reason_codes"])

    def test_fallback_dominance_blocks_consumer_replay(self) -> None:
        from scripts.run_checkpoint_publication_sandbox_consumer_replay_canary import (
            run_checkpoint_publication_sandbox_consumer_replay_canary,
        )

        self._write_inputs()

        summary = run_checkpoint_publication_sandbox_consumer_replay_canary(
            stage14_root=self.stage14_root,
            stage13_root=self.stage13_root,
            promotion_root=self.promotion_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
            force_fallback=True,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("fallback_dominates_consumer_replay", summary["reason_codes"])

    def test_docs_spec_and_shell_are_declared(self) -> None:
        shell_path = self.repo_root / "scripts" / "run_checkpoint_publication_sandbox_consumer_replay_canary.sh"
        spec_path = (
            self.repo_root
            / "docs"
            / "superpowers"
            / "specs"
            / "2026-06-16-checkpoint-publication-sandbox-consumer-replay-canary.md"
        )
        self.assertTrue(shell_path.is_file())
        self.assertTrue(spec_path.is_file())

    def _write_inputs(self) -> None:
        import torch
        from model_explorer.policy.architectures import build_policy_network

        observation = self._observation_payload()
        network = build_policy_network(None, observation=self._observation_object(observation), hidden_size=8)
        self.sandbox_checkpoint_path = self.temp_dir / "sandbox" / "experimental-hybrid-policy-candidate.pt"
        self.sandbox_metadata_path = self.temp_dir / "sandbox" / "experimental-hybrid-policy-candidate-metadata.json"
        self.sandbox_checkpoint_path.parent.mkdir(parents=True)
        torch.save(
            {
                "schema_version": "experimental-hybrid-policy-candidate-checkpoint/v1",
                "experimental": True,
                "architecture": "mlp_v1",
                "hidden_size": 8,
                "model_state_dict": network.state_dict(),
            },
            self.sandbox_checkpoint_path,
        )
        self.checkpoint_sha256 = hashlib.sha256(self.sandbox_checkpoint_path.read_bytes()).hexdigest()
        self.checkpoint_size = self.sandbox_checkpoint_path.stat().st_size
        self._write_json(
            self.sandbox_metadata_path,
            {
                "experimental": True,
                "architecture": "mlp_v1",
                "hidden_size": 8,
                "checkpoint_publication_approved": False,
                "default_policy_replacement_approved": False,
                "real_executor_connection_approved": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "performance_claimed": False,
            },
        )
        manifest_path = self.stage14_root / "checkpoint-publication-sandbox-consumer-verification-manifest.json"
        self._write_json(
            manifest_path,
            {
                "sandbox_consumer_checkpoint_path": str(self.sandbox_checkpoint_path),
                "sandbox_consumer_metadata_path": str(self.sandbox_metadata_path),
                "sandbox_consumer_checkpoint_sha256": self.checkpoint_sha256,
                "sandbox_consumer_checkpoint_size_bytes": self.checkpoint_size,
                "source_package_checkpoint_sha256": self.checkpoint_sha256,
                "source_package_checkpoint_size_bytes": self.checkpoint_size,
                "sandbox_consumer_load_reverified": True,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        for filename, payload in {
            "checkpoint-publication-sandbox-consumer-load-reverification-audit.json": {
                "sandbox_consumer_load_reverification_audit_passed": True,
                "tensor_count": 18,
                "non_finite_tensor_count": 0,
            },
            "checkpoint-publication-sandbox-metadata-verification-audit.json": {
                "sandbox_metadata_verification_audit_passed": True,
                "metadata_experimental": True,
            },
            "checkpoint-publication-sandbox-lineage-verification-audit.json": {"lineage_verification_audit_passed": True},
            "checkpoint-publication-sandbox-release-boundary-audit.json": {
                "release_boundary_audit_passed": True,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
            "checkpoint-publication-sandbox-rollback-verification-audit.json": {"rollback_verification_audit_passed": True},
        }.items():
            self._write_json(self.stage14_root / filename, payload)
        self._write_json(
            self.stage14_root / "checkpoint-publication-sandbox-install-dry-run-verification-summary.json",
            {
                "status": "passed",
                "reason_codes": [],
                "checkpoint_publication_sandbox_install_dry_run_verification_passed": True,
                "checkpoint_publication_sandbox_consumer_smoke_preflight_approved": True,
                "next_required_change": "checkpoint_publication_sandbox_consumer_smoke_preflight",
                "consumer_verification_manifest": str(manifest_path),
                "sandbox_consumer_checkpoint_sha256": self.checkpoint_sha256,
                "sandbox_consumer_checkpoint_size_bytes": self.checkpoint_size,
                "checkpoint_publication_approved": False,
                "default_policy_replacement_approved": False,
                "real_executor_connection_approved": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        self._write_json(
            self.stage13_root / "checkpoint-publication-sandbox-install-manifest.json",
            {
                "sandbox_consumer_checkpoint_path": str(self.sandbox_checkpoint_path),
                "sandbox_consumer_metadata_path": str(self.sandbox_metadata_path),
                "sandbox_consumer_checkpoint_sha256": self.checkpoint_sha256,
                "sandbox_consumer_checkpoint_size_bytes": self.checkpoint_size,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        self._write_json(
            self.promotion_root / "checkpoint-load-inference-audit.json",
            {
                "checkpoint_load_passed": True,
                "inference_audit_count": 64,
                "sampled_rows": [{"scenario_id": f"fixture-{index}", "action_index": index % 3} for index in range(64)],
                "invalid_action_mask_count": 0,
                "missing_observation_count": 0,
                "non_finite_logits_count": 0,
                "non_finite_log_prob_count": 0,
                "non_finite_value_count": 0,
            },
        )

    def _observation_payload(self) -> dict:
        from model_explorer.policy.features import CANDIDATE_FEATURE_NAMES, GLOBAL_FEATURE_NAMES, MISSING_INDICATOR_NAMES

        return {
            "candidate_feature_names": list(CANDIDATE_FEATURE_NAMES),
            "candidate_features": [[0.1 + 0.01 * i for i in range(len(CANDIDATE_FEATURE_NAMES))] for _ in range(3)],
            "global_feature_names": list(GLOBAL_FEATURE_NAMES),
            "global_features": [0.2 for _ in range(len(GLOBAL_FEATURE_NAMES))],
            "action_mask": [True, True, True],
            "candidate_cells": [[0, 0], [1, 1], [2, 2]],
            "candidate_missing_feature_names": [[], [], []],
            "candidate_missing_indicator_names": list(MISSING_INDICATOR_NAMES),
            "candidate_missing_indicators": [[0.0 for _ in MISSING_INDICATOR_NAMES] for _ in range(3)],
        }

    def _observation_object(self, payload: dict):
        from model_explorer.policy.rollout_io import _observation_from_dict

        return _observation_from_dict(payload)

    def _patch_stage14_summary(self, updates: dict) -> None:
        path = self.stage14_root / "checkpoint-publication-sandbox-install-dry-run-verification-summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(updates)
        self._write_json(path, payload)

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
