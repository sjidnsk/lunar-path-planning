import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class ScopedClaimPublicationEvidenceFreezeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts = str(self.repo_root / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="scoped-claim-freeze-"))
        self.stage7_root = self.temp_dir / "stage7"
        self.stage6_root = self.temp_dir / "stage6"
        self.cost_root = self.temp_dir / "cost"
        self.formal_root = self.temp_dir / "formal"
        self.replay_root = self.temp_dir / "replay"
        self.selected_root = self.temp_dir / "selected"
        self.output_root = self.temp_dir / "output"
        for path in (
            self.stage7_root,
            self.stage6_root,
            self.cost_root,
            self.formal_root,
            self.replay_root,
            self.selected_root,
        ):
            path.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_approves_scoped_claim_publication_without_release_actions(self) -> None:
        from scripts.run_scoped_claim_publication_evidence_freeze import (
            run_scoped_claim_publication_evidence_freeze,
        )

        self._write_inputs()

        summary = run_scoped_claim_publication_evidence_freeze(
            stage7_root=self.stage7_root,
            stage6_root=self.stage6_root,
            cost_efficiency_root=self.cost_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["schema_version"], "scoped-claim-publication-evidence-freeze-summary/v1")
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["publication_verdict"], "approved_for_scoped_claim_publication")
        self.assertTrue(summary["scoped_claim_publication_approved"])
        self.assertEqual(summary["performance_claim_scope"], "scoped_offline_guarded_shadow_canary_only")
        self.assertEqual(summary["next_required_change"], "checkpoint_publication_authorization_preflight")
        self.assertFalse(summary["checkpoint_publication_approved"])
        self.assertFalse(summary["default_policy_replacement_approved"])
        self.assertFalse(summary["real_executor_connection_approved"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])

        for field in (
            "bundle_manifest",
            "claim_text",
            "scope_audit",
            "evidence_freeze_ledger",
            "doc_consistency_audit",
            "release_boundary_audit",
            "rejection_report",
            "report",
        ):
            self.assertTrue(Path(summary[field]).is_file(), field)

        claim_text = Path(summary["claim_text"]).read_text(encoding="utf-8")
        self.assertIn("Allowed scoped offline claim", claim_text)
        self.assertIn("Prohibited claims and actions", claim_text)
        self.assertIn("54.96083408637", claim_text)
        self.assertIn("checkpoint publication is not approved", claim_text)

    def test_stage7_failure_blocks_publication(self) -> None:
        from scripts.run_scoped_claim_publication_evidence_freeze import (
            run_scoped_claim_publication_evidence_freeze,
        )

        self._write_inputs()
        self._patch_stage7_summary({"status": "failed", "reason_codes": ["claim_scope_overbroad"]})

        summary = run_scoped_claim_publication_evidence_freeze(
            stage7_root=self.stage7_root,
            stage6_root=self.stage6_root,
            cost_efficiency_root=self.cost_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("stage7_not_passed", summary["reason_codes"])
        self.assertFalse(summary["scoped_claim_publication_approved"])

    def test_rejects_overbroad_claim_scope(self) -> None:
        from scripts.run_scoped_claim_publication_evidence_freeze import (
            run_scoped_claim_publication_evidence_freeze,
        )

        self._write_inputs()
        self._patch_stage7_summary({"performance_claim_scope": "all_real_world_scenarios"})

        summary = run_scoped_claim_publication_evidence_freeze(
            stage7_root=self.stage7_root,
            stage6_root=self.stage6_root,
            cost_efficiency_root=self.cost_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("claim_scope_overbroad", summary["reason_codes"])

    def test_metric_mismatch_blocks_publication(self) -> None:
        from scripts.run_scoped_claim_publication_evidence_freeze import (
            run_scoped_claim_publication_evidence_freeze,
        )

        self._write_inputs()
        self._patch_stage7_summary({"coverage_return_improvement": 999.0})

        summary = run_scoped_claim_publication_evidence_freeze(
            stage7_root=self.stage7_root,
            stage6_root=self.stage6_root,
            cost_efficiency_root=self.cost_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("claim_metric_mismatch", summary["reason_codes"])

    def test_docs_missing_blocks_publication(self) -> None:
        from scripts.run_scoped_claim_publication_evidence_freeze import (
            run_scoped_claim_publication_evidence_freeze,
        )

        self._write_inputs()
        fake_repo = self.temp_dir / "docless_repo"
        fake_repo.mkdir()

        summary = run_scoped_claim_publication_evidence_freeze(
            stage7_root=self.stage7_root,
            stage6_root=self.stage6_root,
            cost_efficiency_root=self.cost_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=fake_repo,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("docs_not_updated", summary["reason_codes"])

    def test_release_boundary_violation_blocks_publication(self) -> None:
        from scripts.run_scoped_claim_publication_evidence_freeze import (
            run_scoped_claim_publication_evidence_freeze,
        )

        self._write_inputs()
        self._patch_stage7_summary({"publishes_checkpoint": True})

        summary = run_scoped_claim_publication_evidence_freeze(
            stage7_root=self.stage7_root,
            stage6_root=self.stage6_root,
            cost_efficiency_root=self.cost_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("release_boundary_violation", summary["reason_codes"])
        self.assertFalse(summary["checkpoint_publication_approved"])

    def test_missing_stage7_artifact_blocks_publication(self) -> None:
        from scripts.run_scoped_claim_publication_evidence_freeze import (
            run_scoped_claim_publication_evidence_freeze,
        )

        self._write_inputs()
        (self.stage7_root / "formal-performance-evidence-ledger.json").unlink()

        summary = run_scoped_claim_publication_evidence_freeze(
            stage7_root=self.stage7_root,
            stage6_root=self.stage6_root,
            cost_efficiency_root=self.cost_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("evidence_freeze_incomplete", summary["reason_codes"])

    def test_prohibited_claim_text_blocks_publication(self) -> None:
        from scripts.run_scoped_claim_publication_evidence_freeze import (
            run_scoped_claim_publication_evidence_freeze,
        )

        self._write_inputs()

        summary = run_scoped_claim_publication_evidence_freeze(
            stage7_root=self.stage7_root,
            stage6_root=self.stage6_root,
            cost_efficiency_root=self.cost_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
            claim_text_override=(
                "Allowed scoped offline claim: the model has real world performance, "
                "Ackermann-feasible trajectory, default policy replacement, and unlimited scenario generalization."
            ),
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("prohibited_claim_present", summary["reason_codes"])

    def test_shell_docs_and_spec_contract_are_declared(self) -> None:
        shell_path = self.repo_root / "scripts" / "run_scoped_claim_publication_evidence_freeze.sh"
        spec_path = (
            self.repo_root
            / "docs"
            / "superpowers"
            / "specs"
            / "2026-06-15-scoped-claim-publication-evidence-freeze.md"
        )
        self.assertTrue(shell_path.is_file())
        self.assertTrue(spec_path.is_file())
        spec_text = spec_path.read_text(encoding="utf-8")
        self.assertIn("scoped-claim-publication-evidence-freeze-summary.json", spec_text)
        self.assertIn("README.md", spec_text)
        self.assertIn("docs/算法设计与系统架构报告.md", spec_text)
        self.assertIn("不发布 checkpoint", spec_text)
        self.assertIn("checkpoint_publication_authorization_preflight", spec_text)

    def _write_inputs(self) -> None:
        metrics = {
            "coverage_return_improvement": 54.96083408637,
            "cumulative_coverage_rate_delta_improvement": 56.92136202257,
            "valuable_area_covered_improvement": 25.987354935654,
            "fallback_rate": 0.220611916264,
            "coverage_efficiency_regression": False,
        }
        stage7_summary = {
            "schema_version": "formal-performance-claim-release-decision-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "decision_verdict": "approved_for_scoped_offline_performance_claim",
            "scoped_offline_performance_claim_approved": True,
            "performance_claimed": True,
            "performance_claim_scope": "scoped_offline_guarded_shadow_canary_only",
            "stage6_long_horizon_shadow_passed": True,
            "metric_consistency_audit_passed": True,
            "claim_scope_audit_passed": True,
            "release_boundary_audit_passed": True,
            "provenance_audit_passed": True,
            "kill_switch_audit_passed": True,
            "rollback_audit_passed": True,
            "telemetry_audit_passed": True,
            "checkpoint_publication_approved": False,
            "default_policy_replacement_approved": False,
            "real_executor_connection_approved": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            **metrics,
            "evidence_ledger": str(self.stage7_root / "formal-performance-evidence-ledger.json"),
            "metric_consistency_audit": str(self.stage7_root / "formal-performance-metric-consistency-audit.json"),
            "claim_scope_audit": str(self.stage7_root / "formal-performance-claim-scope-audit.json"),
            "release_boundary_audit": str(self.stage7_root / "formal-performance-release-boundary-audit.json"),
            "provenance_audit": str(self.stage7_root / "formal-performance-provenance-audit.json"),
            "decision_matrix": str(self.stage7_root / "formal-performance-decision-matrix.json"),
            "scoped_claim_statement": str(self.stage7_root / "formal-performance-scoped-claim-statement.md"),
        }
        self._write_json(self.stage7_root / "formal-performance-claim-release-decision-summary.json", stage7_summary)
        self._write_text(
            self.stage7_root / "formal-performance-scoped-claim-statement.md",
            (
                "# Formal Performance Claim / Release Decision v1\n\n"
                "## Allowed claim\n\n"
                "The current 5B.7 candidate is approved only for a scoped offline guarded shadow/canary performance claim.\n\n"
                "- Coverage return improvement: `54.96083408637`\n"
                "- Cumulative coverage rate delta improvement: `56.92136202257`\n"
                "- Valuable area covered improvement: `25.987354935654`\n"
                "- Fallback rate: `0.220611916264`\n\n"
                "## Prohibited claims\n\n"
                "- Checkpoint publication is not approved.\n"
                "- Default policy replacement is not approved.\n"
                "- Real executor connection is not approved.\n"
                "- Real world performance is not claimed.\n"
                "- Ackermann-feasible trajectory is not claimed.\n"
                "- Unlimited scenario generalization is not claimed.\n"
            ),
        )
        self._write_json(
            self.stage7_root / "formal-performance-evidence-ledger.json",
            {
                "schema_version": "formal-performance-evidence-ledger/v1",
                "all_sources_passed": True,
                "sources": [
                    {"name": "stage6_summary", "passed": True},
                    {"name": "cost_efficiency_summary", "passed": True},
                    {"name": "formal_training_summary", "passed": True},
                    {"name": "post_training_replay_summary", "passed": True},
                    {"name": "selected_candidate_summary", "passed": True},
                ],
            },
        )
        for filename, passed_key in (
            ("formal-performance-metric-consistency-audit.json", "metric_consistency_audit_passed"),
            ("formal-performance-claim-scope-audit.json", "claim_scope_audit_passed"),
            ("formal-performance-release-boundary-audit.json", "release_boundary_audit_passed"),
            ("formal-performance-provenance-audit.json", "provenance_audit_passed"),
        ):
            self._write_json(self.stage7_root / filename, {passed_key: True, "violations": []})
        self._write_json(
            self.stage7_root / "formal-performance-decision-matrix.json",
            {
                "decision_verdict": "approved_for_scoped_offline_performance_claim",
                "scoped_offline_performance_claim_approved": True,
                "checkpoint_publication_approved": False,
                "default_policy_replacement_approved": False,
                "real_executor_connection_approved": False,
            },
        )
        self._write_json(
            self.stage6_root / "shadow-canary-release-performance-validation-preflight-summary.json",
            {
                "schema_version": "shadow-canary-release-performance-validation-preflight-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "long_horizon_shadow_passed": True,
                "kill_switch_audit_passed": True,
                "rollback_audit_passed": True,
                "telemetry_audit_passed": True,
                "checkpoint_publication_approved": False,
                "default_policy_replacement_approved": False,
                "real_executor_connection_approved": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                **metrics,
            },
        )
        self._write_json(
            self.cost_root / "cost-efficiency-aware-coverage-reward-candidate-filter-summary.json",
            {"status": "passed", "reason_codes": [], "publishes_checkpoint": False, "replaces_default_policy": False, "connects_real_executor": False, **metrics},
        )
        self._write_json(
            self.formal_root / "formal-ppo-training-run-summary.json",
            {"status": "passed", "reason_codes": [], "publishes_checkpoint": False, "replaces_default_policy": False},
        )
        self._write_json(
            self.replay_root / "formal-ppo-post-training-stability-replay-summary.json",
            {"status": "passed", "reason_codes": [], "publishes_checkpoint": False, "replaces_default_policy": False},
        )
        self._write_json(
            self.selected_root / "selected-formal-ppo-candidate-promotion-preflight-summary.json",
            {"status": "passed", "reason_codes": [], "publishes_checkpoint": False, "replaces_default_policy": False, "connects_real_executor": False},
        )

    def _patch_stage7_summary(self, updates: dict) -> None:
        path = self.stage7_root / "formal-performance-claim-release-decision-summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(updates)
        self._write_json(path, payload)

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
