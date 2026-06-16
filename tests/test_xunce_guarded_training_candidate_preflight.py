import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class XunceGuardedTrainingCandidatePreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(tempfile.mkdtemp(prefix="xunce-training-preflight-"))
        scripts_path = str(Path(__file__).resolve().parents[1] / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.full_root = self.repo_root / "outputs" / "stage8"
        self.static_root = self.repo_root / "outputs" / "stage9"
        self.ablation_root = self.repo_root / "outputs" / "stage10"
        self.stress_root = self.repo_root / "outputs" / "stage11"
        self.output_root = self.repo_root / "outputs" / "stage12"
        self.config_path = self.repo_root / "configs" / "xunce_guarded_training_candidate_preflight_v1.json"
        self._write_sources()
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.repo_root)

    def test_default_fixture_passes_and_writes_artifacts(self) -> None:
        from scripts.run_xunce_guarded_training_candidate_preflight import run_xunce_guarded_training_candidate_preflight

        summary = run_xunce_guarded_training_candidate_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["next_required_change"], "controlled_training_candidate")
        self.assertTrue(summary["training_candidate_preflight_passed"])
        self.assertTrue(summary["controlled_training_candidate_authorized"])
        self.assertEqual(summary["training_preflight_verdict"], "eligible_for_controlled_training_candidate")
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])

        for filename in (
            "xunce-guarded-training-candidate-preflight-summary.json",
            "xunce-guarded-training-candidate-preflight-manifest.json",
            "xunce-guarded-training-source-evidence-audit.json",
            "xunce-guarded-training-boundary-audit.json",
            "xunce-guarded-training-rejection-report.json",
            "xunce-guarded-training-candidate-preflight-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_missing_stress_routes_to_stress_fix(self) -> None:
        from scripts.run_xunce_guarded_training_candidate_preflight import run_xunce_guarded_training_candidate_preflight

        shutil.rmtree(self.stress_root)
        summary = run_xunce_guarded_training_candidate_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_full_network_stress_summary", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_full_network_stress_evaluation")

    def test_failed_stress_blocks_training_candidate(self) -> None:
        from scripts.run_xunce_guarded_training_candidate_preflight import run_xunce_guarded_training_candidate_preflight

        self._write_json(
            self.stress_root / "xunce-full-network-stress-evaluation-summary.json",
            self._summary_payload("xunce-full-network-stress-evaluation-summary/v1", "failed", "guarded_training_candidate_preflight", {"stress_evaluation_passed": False}),
        )
        summary = run_xunce_guarded_training_candidate_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("full_network_stress_not_passed", summary["reason_codes"])
        self.assertFalse(summary["controlled_training_candidate_authorized"])

    def test_boundary_violation_fails(self) -> None:
        from scripts.run_xunce_guarded_training_candidate_preflight import run_xunce_guarded_training_candidate_preflight

        self._write_json(
            self.stress_root / "xunce-full-network-stress-evaluation-summary.json",
            self._summary_payload("xunce-full-network-stress-evaluation-summary/v1", "passed", "guarded_training_candidate_preflight", {"stress_evaluation_passed": True, "publishes_checkpoint": True}),
        )
        summary = run_xunce_guarded_training_candidate_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("training_preflight_boundary_violation", summary["reason_codes"])
        self.assertFalse(summary["boundary_audit_passed"])

    def _write_config(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "xunce-guarded-training-candidate-preflight-config/v1",
            "source_full_network_root": str(self.full_root),
            "source_static_contract_root": str(self.static_root),
            "source_ablation_root": str(self.ablation_root),
            "source_stress_root": str(self.stress_root),
            "architecture": "xunce_full_network_v1",
            "require_stress_passed": True,
            "require_boundary_closed": True
        }
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_sources(self) -> None:
        self._write_json(
            self.full_root / "xunce-full-network-v1-summary.json",
            self._summary_payload("xunce-full-network-v1-summary/v1", "passed", "full_network_static_contract_validation", {"full_network_v1_passed": True}),
        )
        self._write_json(
            self.static_root / "xunce-full-network-static-contract-validation-summary.json",
            self._summary_payload("xunce-full-network-static-contract-validation-summary/v1", "passed", "full_network_ablation_experiments", {"static_contract_validation_passed": True}),
        )
        self._write_json(
            self.ablation_root / "xunce-full-network-ablation-experiments-summary.json",
            self._summary_payload("xunce-full-network-ablation-experiments-summary/v1", "passed", "full_network_stress_evaluation", {"ablation_experiments_passed": True}),
        )
        self._write_json(
            self.stress_root / "xunce-full-network-stress-evaluation-summary.json",
            self._summary_payload("xunce-full-network-stress-evaluation-summary/v1", "passed", "guarded_training_candidate_preflight", {"stress_evaluation_passed": True}),
        )

    def _summary_payload(self, schema: str, status: str, next_required_change: str, updates: dict) -> dict:
        payload = {
            "schema_version": schema,
            "status": status,
            "reason_codes": [],
            "next_required_change": next_required_change,
            "architecture": "xunce_full_network_v1",
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "runs_new_ppo_update": False,
            "modifies_network": False,
            "modifies_action_space": False,
            "modifies_default_astar": False,
        }
        payload.update(updates)
        return payload

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
