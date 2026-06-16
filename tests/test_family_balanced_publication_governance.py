import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import torch


class FamilyBalancedPublicationGovernanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts = str(self.repo_root / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="family-balanced-publication-"))
        self.decision_root = self.temp_dir / "decision"
        self.shadow_root = self.temp_dir / "shadow"
        self.rerun_root = self.temp_dir / "rerun"
        self.gap_root = self.temp_dir / "gap"
        self.outputs = self.temp_dir / "outputs"
        for path in (self.decision_root, self.shadow_root, self.rerun_root, self.gap_root, self.outputs):
            path.mkdir(parents=True)
        self._write_inputs()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_family_balanced_release_governance_reaches_sandbox_install_preflight_only(self) -> None:
        from scripts.run_family_balanced_scoped_claim_publication_evidence_freeze import (
            run_family_balanced_scoped_claim_publication_evidence_freeze,
        )
        from scripts.run_family_balanced_checkpoint_publication_authorization_preflight import (
            run_family_balanced_checkpoint_publication_authorization_preflight,
        )
        from scripts.run_family_balanced_checkpoint_publication_package_preparation import (
            run_family_balanced_checkpoint_publication_package_preparation,
        )
        from scripts.run_family_balanced_checkpoint_publication_package_verification import (
            run_family_balanced_checkpoint_publication_package_verification,
        )
        from scripts.run_family_balanced_checkpoint_publication_sandbox_install_dry_run_preflight import (
            run_family_balanced_checkpoint_publication_sandbox_install_dry_run_preflight,
        )
        from scripts.run_family_balanced_checkpoint_publication_sandbox_install_dry_run import (
            run_family_balanced_checkpoint_publication_sandbox_install_dry_run,
        )
        from scripts.run_family_balanced_checkpoint_publication_sandbox_install_dry_run_verification import (
            run_family_balanced_checkpoint_publication_sandbox_install_dry_run_verification,
        )
        from scripts.run_family_balanced_checkpoint_publication_sandbox_consumer_replay_canary import (
            run_family_balanced_checkpoint_publication_sandbox_consumer_replay_canary,
        )
        from scripts.run_family_balanced_default_policy_candidate_authorization_preflight import (
            run_family_balanced_default_policy_candidate_authorization_preflight,
        )
        from scripts.run_family_balanced_default_policy_candidate_sandbox_install_preflight import (
            run_family_balanced_default_policy_candidate_sandbox_install_preflight,
        )

        stage8 = self.outputs / "stage8"
        stage9 = self.outputs / "stage9"
        stage10 = self.outputs / "stage10"
        stage11 = self.outputs / "stage11"
        stage12 = self.outputs / "stage12"
        stage13 = self.outputs / "stage13"
        stage14 = self.outputs / "stage14"
        stage15 = self.outputs / "stage15"
        stage16 = self.outputs / "stage16"
        final_root = self.outputs / "final"

        freeze = run_family_balanced_scoped_claim_publication_evidence_freeze(
            family_balanced_decision_root=self.decision_root,
            family_balanced_shadow_root=self.shadow_root,
            family_balanced_rerun_root=self.rerun_root,
            family_balanced_gap_root=self.gap_root,
            output_root=stage8,
            repo_root=self.repo_root,
        )
        authorization = run_family_balanced_checkpoint_publication_authorization_preflight(
            stage8_root=stage8,
            family_balanced_decision_root=self.decision_root,
            family_balanced_rerun_root=self.rerun_root,
            output_root=stage9,
            repo_root=self.repo_root,
        )
        package = run_family_balanced_checkpoint_publication_package_preparation(
            stage9_root=stage9,
            stage8_root=stage8,
            family_balanced_rerun_root=self.rerun_root,
            output_root=stage10,
            repo_root=self.repo_root,
        )
        verification = run_family_balanced_checkpoint_publication_package_verification(
            stage10_root=stage10,
            stage9_root=stage9,
            stage8_root=stage8,
            output_root=stage11,
            repo_root=self.repo_root,
        )
        preflight = run_family_balanced_checkpoint_publication_sandbox_install_dry_run_preflight(
            stage11_root=stage11,
            stage10_root=stage10,
            output_root=stage12,
            repo_root=self.repo_root,
        )
        dry_run = run_family_balanced_checkpoint_publication_sandbox_install_dry_run(
            stage12_root=stage12,
            stage11_root=stage11,
            stage10_root=stage10,
            output_root=stage13,
            repo_root=self.repo_root,
        )
        dry_run_verification = run_family_balanced_checkpoint_publication_sandbox_install_dry_run_verification(
            stage13_root=stage13,
            stage12_root=stage12,
            output_root=stage14,
            repo_root=self.repo_root,
        )
        consumer = run_family_balanced_checkpoint_publication_sandbox_consumer_replay_canary(
            stage14_root=stage14,
            stage13_root=stage13,
            output_root=stage15,
            repo_root=self.repo_root,
        )
        default_authorization = run_family_balanced_default_policy_candidate_authorization_preflight(
            stage15_root=stage15,
            output_root=stage16,
            repo_root=self.repo_root,
        )
        final = run_family_balanced_default_policy_candidate_sandbox_install_preflight(
            stage16_root=stage16,
            stage15_root=stage15,
            output_root=final_root,
            repo_root=self.repo_root,
        )

        chain = [
            freeze,
            authorization,
            package,
            verification,
            preflight,
            dry_run,
            dry_run_verification,
            consumer,
            default_authorization,
            final,
        ]
        for summary in chain:
            self.assertEqual(summary["status"], "passed", summary)
            self.assertEqual(summary["reason_codes"], [])
            self.assertFalse(summary["checkpoint_publication_approved"])
            self.assertFalse(summary["default_policy_replacement_approved"])
            self.assertFalse(summary["real_executor_connection_approved"])
            self.assertFalse(summary["publishes_checkpoint"])
            self.assertFalse(summary["replaces_default_policy"])
            self.assertFalse(summary["connects_real_executor"])
            for field in ("manifest", "release_boundary_audit", "rejection_report", "report"):
                self.assertTrue(Path(summary[field]).is_file(), field)

        self.assertEqual(
            final["schema_version"],
            "family-balanced-default-policy-candidate-sandbox-install-preflight-summary/v1",
        )
        self.assertEqual(
            final["preflight_verdict"],
            "eligible_for_family_balanced_default_policy_candidate_sandbox_install_candidate",
        )
        self.assertTrue(final["family_balanced_default_policy_candidate_sandbox_install_preflight_passed"])
        self.assertTrue(final["kill_switch_audit_passed"])
        self.assertTrue(final["rollback_audit_passed"])
        self.assertTrue(final["default_policy_boundary_audit_passed"])
        self.assertTrue(final["path_planner_isolation_audit_passed"])
        self.assertTrue(final["telemetry_audit_passed"])
        self.assertTrue(final["lineage_audit_passed"])
        self.assertEqual(
            final["next_required_change"],
            "family_balanced_default_policy_candidate_sandbox_install_candidate",
        )

    def test_family_balanced_runner_shells_and_specs_are_declared(self) -> None:
        stage_names = (
            "family_balanced_scoped_claim_publication_evidence_freeze",
            "family_balanced_checkpoint_publication_authorization_preflight",
            "family_balanced_checkpoint_publication_package_preparation",
            "family_balanced_checkpoint_publication_package_verification",
            "family_balanced_checkpoint_publication_sandbox_install_dry_run_preflight",
            "family_balanced_checkpoint_publication_sandbox_install_dry_run",
            "family_balanced_checkpoint_publication_sandbox_install_dry_run_verification",
            "family_balanced_checkpoint_publication_sandbox_consumer_replay_canary",
            "family_balanced_default_policy_candidate_authorization_preflight",
            "family_balanced_default_policy_candidate_sandbox_install_preflight",
        )
        for stage in stage_names:
            self.assertTrue((self.repo_root / "scripts" / f"run_{stage}.py").is_file(), stage)
            self.assertTrue((self.repo_root / "scripts" / f"run_{stage}.sh").is_file(), stage)
        spec_texts = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (self.repo_root / "docs" / "superpowers" / "specs").glob("*family-balanced*.md")
        )
        self.assertIn("family-balanced default-policy candidate sandbox install preflight", spec_texts)
        self.assertIn("不替换 default policy", spec_texts)
        self.assertIn("不连接真实执行器", spec_texts)

    def _write_inputs(self) -> None:
        metrics = {
            "coverage_return_improvement": 56.178989747502,
            "cumulative_coverage_rate_delta_improvement": 57.813998506806,
            "valuable_area_covered_improvement": 46.683307708366,
            "coverage_efficiency_regression": False,
            "fallback_rate": 0.294444444444,
            "fallback_gain_contamination_count": 0,
            "controlled_regression_count": 0,
        }
        checkpoint = self.rerun_root / "coverage-driven-experimental-policy-candidate.pt"
        torch.save(
            {
                "schema_version": "family-balanced-test-checkpoint/v1",
                "experimental": True,
                "architecture": "mlp_v1",
                "model_state_dict": {"linear.weight": torch.zeros((1, 1))},
            },
            checkpoint,
        )
        metadata = self.rerun_root / "coverage-driven-experimental-policy-candidate-metadata.json"
        self._write_json(
            metadata,
            {
                "schema_version": "controlled-hybrid-policy-candidate-checkpoint-metadata/v1",
                "experimental": True,
                "checkpoint_path": str(checkpoint),
                "architecture": "mlp_v1",
                "seed": 0,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "performance_claimed": False,
            },
        )
        self._write_json(
            self.decision_root / "family-balanced-formal-performance-claim-release-decision-summary.json",
            {
                "status": "passed",
                "reason_codes": [],
                "decision_verdict": "approved_for_family_balanced_scoped_offline_performance_claim",
                "family_balanced_formal_performance_claim_release_decision_passed": True,
                "family_balanced_scoped_offline_performance_claim_approved": True,
                "allowed_claim_scope": "scoped_offline_family_balanced_guarded_shadow_canary_only",
                "performance_claim_scope": "scoped_offline_family_balanced_guarded_shadow_canary_only",
                "next_required_change": "family_balanced_scoped_claim_publication_evidence_freeze",
                "low_observation_limitation_acknowledged": True,
                **metrics,
                **self._closed(),
            },
        )
        self._write_json(
            self.shadow_root / "family-balanced-shadow-canary-preflight-summary.json",
            {
                "status": "passed",
                "reason_codes": [],
                "family_balanced_shadow_canary_preflight_passed": True,
                "long_horizon_shadow_passed": True,
                "family_generalization_audit_passed": True,
                "low_observation_shadow_passed": True,
                "kill_switch_audit_passed": True,
                "rollback_audit_passed": True,
                "telemetry_audit_passed": True,
                **metrics,
                **self._closed(),
            },
        )
        self._write_json(
            self.rerun_root / "family-balanced-coverage-driven-ppo-rerun-summary.json",
            {
                "status": "passed",
                "reason_codes": [],
                "family_balanced_coverage_driven_ppo_rerun_passed": True,
                "family_balanced_input_audit_passed": True,
                "checkpoint_path": str(checkpoint),
                "checkpoint_metadata_path": str(metadata),
                "runs_new_ppo_update": True,
                **metrics,
                **self._closed(),
            },
        )
        self._write_json(
            self.gap_root / "family-balanced-algorithm-gap-closure-summary.json",
            {
                "status": "passed",
                "reason_codes": [],
                "family_balanced_algorithm_gap_closure_passed": True,
                "low_observation_gap_closed": True,
                "family_balance_audit_passed": True,
                "low_observation_count": 108,
                "family_balance_ratio": 0.467532467532,
                **self._closed(),
            },
        )

    def _closed(self) -> dict:
        return {
            "checkpoint_publication_approved": False,
            "default_policy_replacement_approved": False,
            "real_executor_connection_approved": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "modifies_network_or_action_space": False,
            "modifies_default_astar": False,
            "relaxes_guard": False,
            "ackermann_feasible_trajectory_claimed": False,
        }

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
