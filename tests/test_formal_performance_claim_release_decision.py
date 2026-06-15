import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class FormalPerformanceClaimReleaseDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts = str(self.repo_root / "scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="formal-performance-claim-"))
        self.stage6_root = self.temp_dir / "stage6"
        self.cost_root = self.temp_dir / "cost"
        self.formal_root = self.temp_dir / "formal"
        self.replay_root = self.temp_dir / "replay"
        self.selected_root = self.temp_dir / "selected"
        self.output_root = self.temp_dir / "output"
        for path in (self.stage6_root, self.cost_root, self.formal_root, self.replay_root, self.selected_root):
            path.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_approves_scoped_offline_claim_but_not_release_actions(self) -> None:
        from scripts.run_formal_performance_claim_release_decision import (
            run_formal_performance_claim_release_decision,
        )

        self._write_inputs()

        summary = run_formal_performance_claim_release_decision(
            stage6_root=self.stage6_root,
            cost_efficiency_root=self.cost_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(
            summary["schema_version"],
            "formal-performance-claim-release-decision-summary/v1",
        )
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["decision_verdict"], "approved_for_scoped_offline_performance_claim")
        self.assertTrue(summary["scoped_offline_performance_claim_approved"])
        self.assertFalse(summary["checkpoint_publication_approved"])
        self.assertFalse(summary["default_policy_replacement_approved"])
        self.assertFalse(summary["real_executor_connection_approved"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertGreater(summary["coverage_return_improvement"], 0.0)
        self.assertGreater(summary["cumulative_coverage_rate_delta_improvement"], 0.0)
        self.assertGreater(summary["valuable_area_covered_improvement"], 0.0)
        self.assertFalse(summary["coverage_efficiency_regression"])
        self.assertEqual(summary["controlled_regression_count"], 0)
        self.assertEqual(summary["fallback_gain_contamination_count"], 0)
        self.assertLess(summary["fallback_rate"], 0.5)
        self.assertTrue(summary["stage6_long_horizon_shadow_passed"])
        self.assertTrue(summary["kill_switch_audit_passed"])
        self.assertTrue(summary["rollback_audit_passed"])
        self.assertTrue(summary["telemetry_audit_passed"])

        for field in (
            "evidence_ledger",
            "metric_consistency_audit",
            "claim_scope_audit",
            "release_boundary_audit",
            "provenance_audit",
            "decision_matrix",
            "scoped_claim_statement",
            "rejection_report",
            "report",
        ):
            self.assertTrue(Path(summary[field]).is_file(), field)

        statement = Path(summary["scoped_claim_statement"]).read_text(encoding="utf-8")
        self.assertIn("Allowed claim", statement)
        self.assertIn("Prohibited claims", statement)
        self.assertIn("offline guarded shadow/canary", statement)
        self.assertIn("not approved", statement)

    def test_stage6_failure_blocks_decision(self) -> None:
        from scripts.run_formal_performance_claim_release_decision import (
            run_formal_performance_claim_release_decision,
        )

        self._write_inputs()
        self._patch_stage6_summary({"status": "failed", "reason_codes": ["long_horizon_coverage_regression"]})

        summary = run_formal_performance_claim_release_decision(
            stage6_root=self.stage6_root,
            cost_efficiency_root=self.cost_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("stage6_not_passed", summary["reason_codes"])
        self.assertFalse(summary["scoped_offline_performance_claim_approved"])

    def test_rejects_overbroad_claim_statement(self) -> None:
        from scripts.run_formal_performance_claim_release_decision import (
            run_formal_performance_claim_release_decision,
        )

        self._write_inputs()

        summary = run_formal_performance_claim_release_decision(
            stage6_root=self.stage6_root,
            cost_efficiency_root=self.cost_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
            claim_statement_override=(
                "Allowed claim: the model is approved for real executor deployment "
                "and default policy replacement with Ackermann-feasible trajectory."
            ),
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("claim_scope_overbroad", summary["reason_codes"])
        self.assertFalse(summary["claim_scope_audit_passed"])

    def test_metric_inconsistency_blocks_decision(self) -> None:
        from scripts.run_formal_performance_claim_release_decision import (
            run_formal_performance_claim_release_decision,
        )

        self._write_inputs()
        self._patch_stage6_summary({"computed_shadow_candidate_metrics": {"coverage_return": 10.0}})

        summary = run_formal_performance_claim_release_decision(
            stage6_root=self.stage6_root,
            cost_efficiency_root=self.cost_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("coverage_metric_inconsistent", summary["reason_codes"])

    def test_fallback_dominates_blocks_decision(self) -> None:
        from scripts.run_formal_performance_claim_release_decision import (
            run_formal_performance_claim_release_decision,
        )

        self._write_inputs()
        self._patch_stage6_summary({"fallback_rate": 0.75})
        self._write_json(
            self.stage6_root / "shadow-canary-guard-fallback-audit.json",
            {
                "schema_version": "shadow-canary-release-performance-audit/v1",
                "guard_fallback_audit_passed": False,
                "fallback_rate": 0.75,
                "fallback_gain_contamination_count": 0,
                "controlled_regression_count": 0,
            },
        )

        summary = run_formal_performance_claim_release_decision(
            stage6_root=self.stage6_root,
            cost_efficiency_root=self.cost_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("fallback_dominates", summary["reason_codes"])

    def test_release_boundary_violation_blocks_publication_actions(self) -> None:
        from scripts.run_formal_performance_claim_release_decision import (
            run_formal_performance_claim_release_decision,
        )

        self._write_inputs()
        self._patch_stage6_summary({"publishes_checkpoint": True})

        summary = run_formal_performance_claim_release_decision(
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

    def test_telemetry_missing_coverage_gain_blocks_decision(self) -> None:
        from scripts.run_formal_performance_claim_release_decision import (
            run_formal_performance_claim_release_decision,
        )

        self._write_inputs()
        self._write_json(
            self.stage6_root / "shadow-canary-telemetry-audit.json",
            {
                "schema_version": "shadow-canary-release-performance-audit/v1",
                "telemetry_audit_passed": False,
                "records_coverage_gain": False,
                "missing_telemetry_row_count": 1,
            },
        )

        summary = run_formal_performance_claim_release_decision(
            stage6_root=self.stage6_root,
            cost_efficiency_root=self.cost_root,
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("telemetry_missing_coverage_gain", summary["reason_codes"])

    def test_shell_docs_and_spec_contract_are_declared(self) -> None:
        shell_path = self.repo_root / "scripts" / "run_formal_performance_claim_release_decision.sh"
        spec_path = (
            self.repo_root
            / "docs"
            / "superpowers"
            / "specs"
            / "2026-06-15-formal-performance-claim-release-decision.md"
        )
        self.assertTrue(shell_path.is_file())
        self.assertTrue(spec_path.is_file())
        spec_text = spec_path.read_text(encoding="utf-8")
        self.assertIn("formal-performance-claim-release-decision-summary.json", spec_text)
        self.assertIn("README.md", spec_text)
        self.assertIn("docs/算法设计与系统架构报告.md", spec_text)
        self.assertIn("不发布 checkpoint", spec_text)
        self.assertIn("不连接真实执行器", spec_text)

    def _write_inputs(self) -> None:
        candidate_metrics = self._metrics(
            actor="post_improvement_ppo",
            coverage_return=79.4,
            cumulative=83.6,
            valuable=44.0,
            gain_per_path=0.0037,
            gain_per_risk=0.63,
            gain_per_energy=0.058,
            fallback_rate=0.22,
        )
        baseline_metrics = self._metrics(
            actor="teacher",
            coverage_return=24.4,
            cumulative=26.7,
            valuable=18.0,
            gain_per_path=0.0011,
            gain_per_risk=0.15,
            gain_per_energy=0.009,
            fallback_rate=0.0,
        )
        stage6_summary = {
            "schema_version": "shadow-canary-release-performance-validation-preflight-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "next_required_change": "formal_performance_claim_release_decision",
            "long_horizon_shadow_passed": True,
            "coverage_return_improvement": 55.0,
            "cumulative_coverage_rate_delta_improvement": 56.9,
            "valuable_area_covered_improvement": 26.0,
            "coverage_efficiency_regression": False,
            "fallback_rate": 0.22,
            "safe_better_training_family_count": 4,
            "controlled_regression_count": 0,
            "fallback_gain_contamination_count": 0,
            "shadow_policy_takes_control": False,
            "experimental_control_activation_count": 0,
            "kill_switch_audit_passed": True,
            "rollback_audit_passed": True,
            "telemetry_audit_passed": True,
            "eligible_for_formal_performance_claim_review": True,
            "candidate_metrics": candidate_metrics,
            "baseline_metrics": baseline_metrics,
            "computed_shadow_candidate_metrics": dict(candidate_metrics),
            "computed_shadow_teacher_metrics": dict(baseline_metrics),
            "summary": str(self.stage6_root / "shadow-canary-release-performance-validation-preflight-summary.json"),
            "long_horizon_shadow_validation": str(self.stage6_root / "shadow-canary-long-horizon-validation.json"),
            "coverage_efficiency_audit": str(self.stage6_root / "shadow-canary-coverage-efficiency-audit.json"),
            "guard_fallback_audit": str(self.stage6_root / "shadow-canary-guard-fallback-audit.json"),
            "kill_switch_audit": str(self.stage6_root / "shadow-canary-kill-switch-audit.json"),
            "rollback_audit": str(self.stage6_root / "shadow-canary-rollback-audit.json"),
            "telemetry_audit": str(self.stage6_root / "shadow-canary-telemetry-audit.json"),
            "eligibility_ledger": str(self.stage6_root / "shadow-canary-eligibility-ledger.json"),
            "cost_efficiency_root": str(self.cost_root),
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "performance_claimed": False,
        }
        self._write_json(self.stage6_root / "shadow-canary-release-performance-validation-preflight-summary.json", stage6_summary)
        self._write_json(
            self.stage6_root / "shadow-canary-long-horizon-validation.json",
            {
                "schema_version": "shadow-canary-release-performance-long-horizon-validation/v1",
                "horizons": [10, 20, 30],
                "long_horizon_shadow_passed": True,
                "rollups": [
                    {
                        "group_type": "all",
                        "horizon": 10,
                        "coverage_return_improvement": 55.0,
                        "cumulative_coverage_rate_delta_improvement": 56.9,
                        "valuable_area_covered_improvement": 26.0,
                    }
                ],
            },
        )
        self._write_json(
            self.stage6_root / "shadow-canary-coverage-efficiency-audit.json",
            {
                "schema_version": "shadow-canary-release-performance-audit/v1",
                "coverage_efficiency_audit_passed": True,
                "coverage_efficiency_regression": False,
            },
        )
        self._write_json(
            self.stage6_root / "shadow-canary-guard-fallback-audit.json",
            {
                "schema_version": "shadow-canary-release-performance-audit/v1",
                "guard_fallback_audit_passed": True,
                "fallback_rate": 0.22,
                "fallback_gain_contamination_count": 0,
                "controlled_regression_count": 0,
                "source_fallback_available": True,
            },
        )
        self._write_json(
            self.stage6_root / "shadow-canary-kill-switch-audit.json",
            {
                "schema_version": "shadow-canary-release-performance-audit/v1",
                "kill_switch_audit_passed": True,
                "shadow_policy_takes_control": False,
                "experimental_control_activation_count": 0,
                "default_policy_authoritative": True,
                "real_executor_connected": False,
            },
        )
        self._write_json(
            self.stage6_root / "shadow-canary-rollback-audit.json",
            {
                "schema_version": "shadow-canary-release-performance-audit/v1",
                "rollback_audit_passed": True,
                "default_policy_authoritative": True,
                "default_policy_unchanged": True,
                "source_fallback_can_restore_control": True,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
            },
        )
        self._write_json(
            self.stage6_root / "shadow-canary-telemetry-audit.json",
            {
                "schema_version": "shadow-canary-release-performance-audit/v1",
                "telemetry_audit_passed": True,
                "records_coverage_gain": True,
                "records_cost_risk_energy": True,
                "records_fallback_activation_rollback": True,
                "missing_telemetry_row_count": 0,
            },
        )
        self._write_json(
            self.stage6_root / "shadow-canary-eligibility-ledger.json",
            {
                "schema_version": "shadow-canary-release-performance-eligibility-ledger/v1",
                "eligible_for_formal_performance_claim_review": True,
                "reason_codes": [],
            },
        )
        self._write_json(
            self.cost_root / "cost-efficiency-aware-coverage-reward-candidate-filter-summary.json",
            {
                "schema_version": "cost-efficiency-aware-coverage-reward-candidate-filter-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "coverage_return_improvement": 55.0,
                "cumulative_coverage_rate_delta_improvement": 56.9,
                "valuable_area_covered_improvement": 26.0,
                "coverage_efficiency_regression": False,
                "fallback_rate": 0.22,
                "controlled_regression_count": 0,
                "fallback_gain_contamination_count": 0,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "performance_claimed": False,
            },
        )
        self._write_jsonl(self.cost_root / "refined-coverage-driven-ppo-performance-metric-table.jsonl", [baseline_metrics, candidate_metrics])
        self._write_json(
            self.cost_root / "coverage-driven-ppo-replay-audit.json",
            {"schema_version": "coverage-driven-ppo-replay-audit/v1", "reason_codes": [], "rows": []},
        )
        self._write_json(
            self.formal_root / "formal-ppo-training-run-summary.json",
            {"schema_version": "formal-ppo-training-run-summary/v1", "status": "passed", "reason_codes": [], "controlled_regression_count": 0, "performance_claimed": False},
        )
        self._write_json(
            self.replay_root / "formal-ppo-post-training-stability-replay-summary.json",
            {"schema_version": "formal-ppo-post-training-stability-replay-summary/v1", "status": "passed", "reason_codes": [], "controlled_regression_count": 0, "performance_claimed": False},
        )
        self._write_json(
            self.selected_root / "selected-formal-ppo-candidate-promotion-preflight-summary.json",
            {"schema_version": "selected-formal-ppo-candidate-promotion-preflight-summary/v1", "status": "passed", "reason_codes": [], "controlled_regression_count": 0, "publishes_checkpoint": False, "replaces_default_policy": False, "performance_claimed": False},
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
            "schema_version": "coverage-driven-ppo-performance-metric-row/v1",
            "actor": actor,
            "row_count": 621,
            "coverage_return": coverage_return,
            "cumulative_coverage_rate_delta": cumulative,
            "valuable_area_covered": valuable,
            "coverage_gain_per_path_cost": gain_per_path,
            "coverage_gain_per_risk": gain_per_risk,
            "coverage_gain_per_energy": gain_per_energy,
            "fallback_rate": fallback_rate,
            "controlled_regression_count": 0,
        }

    def _patch_stage6_summary(self, updates: dict) -> None:
        path = self.stage6_root / "shadow-canary-release-performance-validation-preflight-summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload.update(updates)
        self._write_json(path, payload)

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
