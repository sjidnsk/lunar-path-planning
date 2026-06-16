import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class FamilyBalancedFormalPerformanceClaimReleaseDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts = str(self.repo_root / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="family-balanced-formal-claim-"))
        self.shadow_root = self.temp_dir / "shadow"
        self.rerun_root = self.temp_dir / "rerun"
        self.gap_root = self.temp_dir / "gap"
        self.formal_root = self.temp_dir / "formal"
        self.replay_root = self.temp_dir / "replay"
        self.selected_root = self.temp_dir / "selected"
        self.output_root = self.temp_dir / "output"
        for path in (
            self.shadow_root,
            self.rerun_root,
            self.gap_root,
            self.formal_root,
            self.replay_root,
            self.selected_root,
        ):
            path.mkdir(parents=True)
        self._write_docs()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_approves_scoped_family_balanced_claim_with_low_observation_limitation(self) -> None:
        from scripts.run_family_balanced_formal_performance_claim_release_decision import (
            run_family_balanced_formal_performance_claim_release_decision,
        )

        self._write_inputs()

        summary = run_family_balanced_formal_performance_claim_release_decision(
            family_balanced_shadow_root=self.shadow_root,
            family_balanced_rerun_root=self.rerun_root,
            family_balanced_gap_root=self.gap_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.temp_dir,
        )

        self.assertEqual(summary["schema_version"], "family-balanced-formal-performance-claim-release-decision-summary/v1")
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(
            summary["decision_verdict"],
            "approved_for_family_balanced_scoped_offline_performance_claim",
        )
        self.assertTrue(summary["family_balanced_formal_performance_claim_release_decision_passed"])
        self.assertTrue(summary["family_balanced_scoped_offline_performance_claim_approved"])
        self.assertEqual(
            summary["allowed_claim_scope"],
            "scoped_offline_family_balanced_guarded_shadow_canary_only",
        )
        self.assertGreater(summary["coverage_return_improvement"], 0.0)
        self.assertGreater(summary["cumulative_coverage_rate_delta_improvement"], 0.0)
        self.assertGreater(summary["valuable_area_covered_improvement"], 0.0)
        self.assertFalse(summary["coverage_efficiency_regression"])
        self.assertLess(summary["fallback_rate"], 0.5)
        self.assertEqual(summary["fallback_gain_contamination_count"], 0)
        self.assertEqual(summary["controlled_regression_count"], 0)
        self.assertTrue(summary["family_generalization_audit_passed"])
        self.assertTrue(summary["low_observation_shadow_passed"])
        self.assertFalse(summary["low_observation_coverage_return_outperforms_teacher"])
        self.assertTrue(summary["low_observation_limitation_acknowledged"])
        self.assertEqual(summary["next_required_change"], "family_balanced_scoped_claim_publication_evidence_freeze")
        self.assertFalse(summary["checkpoint_publication_approved"])
        self.assertFalse(summary["default_policy_replacement_approved"])
        self.assertFalse(summary["real_executor_connection_approved"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])

        for field in (
            "evidence_ledger",
            "metric_consistency_audit",
            "claim_scope_audit",
            "low_observation_limitation_audit",
            "release_boundary_audit",
            "provenance_audit",
            "decision_matrix",
            "scoped_claim_statement",
            "rejection_report",
            "report",
        ):
            self.assertTrue(Path(summary[field]).is_file(), field)

        claim = Path(summary["scoped_claim_statement"]).read_text(encoding="utf-8")
        self.assertIn("Allowed claim", claim)
        self.assertIn("Prohibited claims", claim)
        self.assertIn("low-observation coverage return", claim)
        self.assertIn("does not outperform teacher", claim)

    def test_shadow_canary_failure_blocks_decision(self) -> None:
        from scripts.run_family_balanced_formal_performance_claim_release_decision import (
            run_family_balanced_formal_performance_claim_release_decision,
        )

        self._write_inputs()
        self._patch_shadow_summary({"status": "failed", "reason_codes": ["fallback_dominates"]})

        summary = run_family_balanced_formal_performance_claim_release_decision(
            family_balanced_shadow_root=self.shadow_root,
            family_balanced_rerun_root=self.rerun_root,
            family_balanced_gap_root=self.gap_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.temp_dir,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("family_balanced_shadow_canary_not_passed", summary["reason_codes"])
        self.assertFalse(summary["family_balanced_scoped_offline_performance_claim_approved"])

    def test_rejects_overbroad_claim_statement(self) -> None:
        from scripts.run_family_balanced_formal_performance_claim_release_decision import (
            run_family_balanced_formal_performance_claim_release_decision,
        )

        self._write_inputs()

        summary = run_family_balanced_formal_performance_claim_release_decision(
            family_balanced_shadow_root=self.shadow_root,
            family_balanced_rerun_root=self.rerun_root,
            family_balanced_gap_root=self.gap_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.temp_dir,
            claim_statement_override=(
                "## Allowed claim\n"
                "The model is approved for real-world deployment, Stage 17 authorization, "
                "default policy replacement, and low-observation coverage return outperforms teacher.\n"
                "## Prohibited claims\n"
            ),
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("claim_scope_overbroad", summary["reason_codes"])
        self.assertIn("low_observation_overclaimed", summary["reason_codes"])

    def test_metric_inconsistency_blocks_decision(self) -> None:
        from scripts.run_family_balanced_formal_performance_claim_release_decision import (
            run_family_balanced_formal_performance_claim_release_decision,
        )

        self._write_inputs()
        self._patch_shadow_summary({"computed_shadow_candidate_metrics": {"coverage_return": 1.0}})

        summary = run_family_balanced_formal_performance_claim_release_decision(
            family_balanced_shadow_root=self.shadow_root,
            family_balanced_rerun_root=self.rerun_root,
            family_balanced_gap_root=self.gap_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.temp_dir,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("coverage_metric_inconsistent", summary["reason_codes"])

    def test_release_boundary_violation_blocks_decision(self) -> None:
        from scripts.run_family_balanced_formal_performance_claim_release_decision import (
            run_family_balanced_formal_performance_claim_release_decision,
        )

        self._write_inputs()
        self._patch_shadow_summary({"publishes_checkpoint": True})

        summary = run_family_balanced_formal_performance_claim_release_decision(
            family_balanced_shadow_root=self.shadow_root,
            family_balanced_rerun_root=self.rerun_root,
            family_balanced_gap_root=self.gap_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.temp_dir,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("release_boundary_violation", summary["reason_codes"])
        self.assertFalse(summary["checkpoint_publication_approved"])

    def test_shell_docs_and_spec_contract_are_declared(self) -> None:
        shell_path = self.repo_root / "scripts" / "run_family_balanced_formal_performance_claim_release_decision.sh"
        spec_path = (
            self.repo_root
            / "docs"
            / "superpowers"
            / "specs"
            / "2026-06-16-family-balanced-formal-performance-claim-release-decision.md"
        )
        self.assertTrue(shell_path.is_file())
        self.assertTrue(spec_path.is_file())
        spec_text = spec_path.read_text(encoding="utf-8")
        self.assertIn("family-balanced-formal-performance-claim-release-decision-summary.json", spec_text)
        self.assertIn("README.md", spec_text)
        self.assertIn("docs/算法设计与系统架构报告.md", spec_text)
        self.assertIn("低观测", spec_text)
        self.assertIn("不发布 checkpoint", spec_text)

    def _write_inputs(self) -> None:
        candidate_metrics = self._metrics(
            actor="post_improvement_ppo",
            coverage_return=86.178989747502,
            cumulative=97.813998506806,
            valuable=76.683307708366,
            gain_per_path=0.011,
            gain_per_risk=0.42,
            gain_per_energy=0.031,
            fallback_rate=0.294444444444,
        )
        baseline_metrics = self._metrics(
            actor="teacher",
            coverage_return=30.0,
            cumulative=40.0,
            valuable=30.0,
            gain_per_path=0.005,
            gain_per_risk=0.2,
            gain_per_energy=0.015,
            fallback_rate=0.0,
        )
        shadow_summary = {
            "schema_version": "family-balanced-shadow-canary-preflight-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "preflight_verdict": "eligible_for_family_balanced_formal_performance_claim_release_decision",
            "family_balanced_shadow_canary_preflight_passed": True,
            "family_balanced_formal_performance_claim_release_decision_approved": True,
            "next_required_change": "family_balanced_formal_performance_claim_release_decision",
            "long_horizon_shadow_passed": True,
            "horizons": [10, 20, 30],
            "coverage_return_improvement": 56.178989747502,
            "cumulative_coverage_rate_delta_improvement": 57.813998506806,
            "valuable_area_covered_improvement": 46.683307708366,
            "coverage_efficiency_regression": False,
            "fallback_rate": 0.294444444444,
            "controlled_regression_count": 0,
            "fallback_gain_contamination_count": 0,
            "family_generalization_audit_passed": True,
            "low_observation_shadow_passed": True,
            "kill_switch_audit_passed": True,
            "rollback_audit_passed": True,
            "telemetry_audit_passed": True,
            "shadow_policy_takes_control": False,
            "experimental_control_activation_count": 0,
            "candidate_metrics": candidate_metrics,
            "baseline_metrics": baseline_metrics,
            "computed_shadow_candidate_metrics": dict(candidate_metrics),
            "computed_shadow_teacher_metrics": dict(baseline_metrics),
            "summary": str(self.shadow_root / "family-balanced-shadow-canary-preflight-summary.json"),
            "long_horizon_shadow_validation": str(self.shadow_root / "family-balanced-shadow-canary-long-horizon-validation.json"),
            "family_generalization_audit": str(self.shadow_root / "family-balanced-shadow-canary-family-generalization-audit.json"),
            "coverage_efficiency_audit": str(self.shadow_root / "family-balanced-shadow-canary-coverage-efficiency-audit.json"),
            "guard_fallback_audit": str(self.shadow_root / "family-balanced-shadow-canary-guard-fallback-audit.json"),
            "kill_switch_audit": str(self.shadow_root / "family-balanced-shadow-canary-kill-switch-audit.json"),
            "rollback_audit": str(self.shadow_root / "family-balanced-shadow-canary-rollback-audit.json"),
            "telemetry_audit": str(self.shadow_root / "family-balanced-shadow-canary-telemetry-audit.json"),
            "lineage_audit": str(self.shadow_root / "family-balanced-shadow-canary-lineage-audit.json"),
            "release_boundary_audit": str(self.shadow_root / "family-balanced-shadow-canary-release-boundary-audit.json"),
            "checkpoint_publication_approved": False,
            "default_policy_replacement_approved": False,
            "real_executor_connection_approved": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "performance_claimed": False,
        }
        self._write_json(self.shadow_root / "family-balanced-shadow-canary-preflight-summary.json", shadow_summary)
        self._write_json(
            self.shadow_root / "family-balanced-shadow-canary-long-horizon-validation.json",
            {
                "schema_version": "family-balanced-shadow-canary-audit/v1",
                "horizons": [10, 20, 30],
                "long_horizon_shadow_passed": True,
                "rollups": [{"horizon": 10, "coverage_return_improvement": 56.0}],
            },
        )
        self._write_json(
            self.shadow_root / "family-balanced-shadow-canary-family-generalization-audit.json",
            {
                "schema_version": "family-balanced-shadow-canary-audit/v1",
                "family_generalization_audit_passed": True,
                "low_observation_shadow_passed": True,
                "family_rollups": {
                    "low_observation_count": {
                        "candidate_coverage_return": 0.42,
                        "baseline_coverage_return": 4.32,
                        "coverage_return_improvement": -3.9,
                        "candidate_valuable_area_covered": 5.37,
                        "baseline_valuable_area_covered": 2.67,
                        "fallback_gain": 0.0,
                        "controlled_regression_count": 0,
                        "family_shadow_passed": True,
                    },
                    "mixed_risk": {"coverage_return_improvement": 31.0, "family_shadow_passed": True},
                    "rim_or_steep_slope": {"coverage_return_improvement": 12.0, "family_shadow_passed": True},
                    "smooth_high_confidence": {"coverage_return_improvement": 16.0, "family_shadow_passed": True},
                },
            },
        )
        self._write_json(
            self.shadow_root / "family-balanced-shadow-canary-coverage-efficiency-audit.json",
            {"coverage_efficiency_audit_passed": True, "coverage_efficiency_regression": False},
        )
        self._write_json(
            self.shadow_root / "family-balanced-shadow-canary-guard-fallback-audit.json",
            {
                "guard_fallback_audit_passed": True,
                "fallback_rate": 0.294444444444,
                "fallback_gain_contamination_count": 0,
                "controlled_regression_count": 0,
            },
        )
        self._write_json(
            self.shadow_root / "family-balanced-shadow-canary-kill-switch-audit.json",
            {
                "kill_switch_audit_passed": True,
                "shadow_policy_takes_control": False,
                "experimental_control_activation_count": 0,
                "real_executor_connected": False,
            },
        )
        self._write_json(
            self.shadow_root / "family-balanced-shadow-canary-rollback-audit.json",
            {
                "rollback_audit_passed": True,
                "default_policy_unchanged": True,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        self._write_json(
            self.shadow_root / "family-balanced-shadow-canary-telemetry-audit.json",
            {
                "telemetry_audit_passed": True,
                "records_coverage_gain": True,
                "records_cost_risk_energy": True,
                "records_fallback_activation_rollback": True,
            },
        )
        self._write_json(
            self.shadow_root / "family-balanced-shadow-canary-lineage-audit.json",
            {
                "lineage_audit_passed": True,
                "checks": {
                    "rerun_passed": True,
                    "gap_passed": True,
                    "formal_training_passed": True,
                    "post_training_replay_passed": True,
                    "selected_candidate_passed": True,
                },
            },
        )
        self._write_json(
            self.shadow_root / "family-balanced-shadow-canary-release-boundary-audit.json",
            {
                "release_boundary_audit_passed": True,
                "checkpoint_publication_approved": False,
                "default_policy_replacement_approved": False,
                "real_executor_connection_approved": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        self._write_json(
            self.rerun_root / "family-balanced-coverage-driven-ppo-rerun-summary.json",
            {
                "status": "passed",
                "reason_codes": [],
                "rerun_verdict": "eligible_for_family_balanced_shadow_canary_preflight",
                "coverage_return_improvement": 56.178989747502,
                "coverage_efficiency_regression": False,
                "fallback_rate": 0.294444444444,
                "controlled_regression_count": 0,
                "fallback_gain_contamination_count": 0,
                "checkpoint_publication_approved": False,
                "default_policy_replacement_approved": False,
                "real_executor_connection_approved": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        self._write_json(
            self.gap_root / "family-balanced-algorithm-gap-closure-summary.json",
            {
                "status": "passed",
                "reason_codes": [],
                "family_balanced_algorithm_gap_closure_passed": True,
                "low_observation_count": 108,
                "checkpoint_publication_approved": False,
                "default_policy_replacement_approved": False,
                "real_executor_connection_approved": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        self._write_json(
            self.formal_root / "formal-ppo-training-run-summary.json",
            {"status": "passed", "reason_codes": [], "performance_claimed": False},
        )
        self._write_json(
            self.replay_root / "formal-ppo-post-training-stability-replay-summary.json",
            {"status": "passed", "reason_codes": [], "performance_claimed": False},
        )
        self._write_json(
            self.selected_root / "selected-formal-ppo-candidate-promotion-preflight-summary.json",
            {
                "status": "passed",
                "reason_codes": [],
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "performance_claimed": False,
            },
        )

    def _metrics(
        self,
        *,
        actor: str,
        coverage_return: float,
        cumulative: float,
        valuable: float,
        gain_per_path: float,
        gain_per_risk: float,
        gain_per_energy: float,
        fallback_rate: float,
    ) -> dict:
        return {
            "actor": actor,
            "coverage_return": coverage_return,
            "cumulative_coverage_rate_delta": cumulative,
            "valuable_area_covered": valuable,
            "coverage_gain_per_path_cost": gain_per_path,
            "coverage_gain_per_risk": gain_per_risk,
            "coverage_gain_per_energy": gain_per_energy,
            "fallback_rate": fallback_rate,
            "controlled_regression_count": 0,
        }

    def _write_docs(self) -> None:
        self._write_text(
            self.temp_dir / "README.md",
            "Family-Balanced Formal Performance Claim / Release Decision v1\n"
            "outputs/path_feedback_batch_family_balanced_formal_performance_claim_release_decision_v1/\n"
            "family_balanced_scoped_claim_publication_evidence_freeze\n"
            "low_observation_limitation_acknowledged=true\n"
            "checkpoint_publication_approved=false\n"
            "default_policy_replacement_approved=false\n"
            "real_executor_connection_approved=false\n",
        )
        self._write_text(
            self.temp_dir / "docs" / "算法设计与系统架构报告.md",
            "Family-Balanced Formal Performance Claim / Release Decision v1\n"
            "低观测 family 限制必须写入声明\n"
            "不发布 checkpoint、不替换 default policy、不连接真实执行器\n",
        )
        self._write_text(
            self.temp_dir
            / "docs"
            / "superpowers"
            / "specs"
            / "2026-06-16-family-balanced-formal-performance-claim-release-decision.md",
            "Family-Balanced Formal Performance Claim / Release Decision v1\n"
            "family-balanced-formal-performance-claim-release-decision-summary.json\n"
            "README.md\n"
            "docs/算法设计与系统架构报告.md\n"
            "低观测\n"
            "不发布 checkpoint\n",
        )

    def _patch_shadow_summary(self, updates: dict) -> None:
        path = self.shadow_root / "family-balanced-shadow-canary-preflight-summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(updates)
        self._write_json(path, payload)

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _write_text(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
