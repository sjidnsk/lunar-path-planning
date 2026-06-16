import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class NetworkArchitectureUpgradeReadinessReviewTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        for path in (self.repo_root / "scripts", self.repo_root / "model-explorer" / "src"):
            path_text = str(path)
            if path_text not in sys.path:
                sys.path.insert(0, path_text)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="network-upgrade-readiness-"))
        self.source_root = self.temp_dir / "multi-map"
        self.output_root = self.temp_dir / "output"
        self.config_path = self.temp_dir / "config.json"
        self.source_root.mkdir(parents=True, exist_ok=True)
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_current_multi_map_evidence_defers_network_upgrade(self) -> None:
        from scripts.run_network_architecture_upgrade_readiness_review import (
            run_network_architecture_upgrade_readiness_review,
        )

        self._write_evidence()
        summary = run_network_architecture_upgrade_readiness_review(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["source_multi_map_status"], "passed")
        self.assertFalse(summary["network_upgrade_recommended"])
        self.assertEqual(
            summary["network_upgrade_readiness_decision"],
            "defer_network_upgrade_global_99_synthetic_goal_met",
        )
        self.assertEqual(summary["next_required_change"], "global_99_release_governance_preflight")
        self.assertEqual(summary["policy_guard_fallback_rate"], 3 / 95)
        self.assertIn("mlp_v1", summary["candidate_architecture_inventory"])
        self.assertIn("mlp_missing_v1", summary["candidate_architecture_inventory"])
        self.assertIn("candidate_attention_v1", summary["candidate_architecture_inventory"])
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["modifies_network"])
        self.assertFalse(summary["modifies_action_space"])
        self.assertFalse(summary["modifies_default_astar"])
        self.assertFalse(summary["uses_path_planner"])
        self.assertFalse(summary["uses_npz_or_sidecar"])

        for filename in (
            "network-architecture-upgrade-readiness-summary.json",
            "network-architecture-upgrade-evidence-audit.json",
            "network-architecture-upgrade-bottleneck-attribution.json",
            "network-architecture-upgrade-recommendation-report.md",
            "network-architecture-upgrade-readiness-manifest.json",
            "network-architecture-upgrade-rejection-report.json",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_failed_source_multi_map_blocks_review(self) -> None:
        from scripts.run_network_architecture_upgrade_readiness_review import (
            run_network_architecture_upgrade_readiness_review,
        )

        self._write_evidence(source_status="failed", reason_codes=["required_scenario_failed"])
        summary = run_network_architecture_upgrade_readiness_review(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("source_multi_map_not_passed", summary["reason_codes"])
        self.assertFalse(summary["network_upgrade_recommended"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_multi_map_generalization")

    def test_incomplete_required_evidence_requests_multi_map_fix_not_network_upgrade(self) -> None:
        from scripts.run_network_architecture_upgrade_readiness_review import (
            run_network_architecture_upgrade_readiness_review,
        )

        self._write_evidence(
            failed_required_scenario_count=1,
            aggregate_achieved_coverage_rate=0.988,
            min_scenario_achieved_coverage_rate=0.982,
        )
        summary = run_network_architecture_upgrade_readiness_review(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertFalse(summary["network_upgrade_recommended"])
        self.assertIn("multi_map_coverage_evidence_incomplete", summary["network_upgrade_blockers"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_multi_map_generalization")

    def test_policy_regression_recommends_network_architecture_upgrade(self) -> None:
        from scripts.run_network_architecture_upgrade_readiness_review import (
            run_network_architecture_upgrade_readiness_review,
        )

        self._write_evidence(policy_worse_than_baseline_count=2, controlled_regression_count=1)
        summary = run_network_architecture_upgrade_readiness_review(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertTrue(summary["network_upgrade_recommended"])
        self.assertEqual(summary["next_required_change"], "network_architecture_upgrade_v1")
        self.assertEqual(summary["bottleneck_attribution"]["classification"], "policy_regression_bottleneck")

    def test_high_guard_fallback_rate_recommends_network_architecture_upgrade(self) -> None:
        from scripts.run_network_architecture_upgrade_readiness_review import (
            run_network_architecture_upgrade_readiness_review,
        )

        self._write_evidence(policy_guided_decision_count=20, policy_guard_fallback_count=5)
        summary = run_network_architecture_upgrade_readiness_review(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertTrue(summary["network_upgrade_recommended"])
        self.assertGreater(summary["policy_guard_fallback_rate"], 0.10)
        self.assertEqual(summary["next_required_change"], "network_architecture_upgrade_v1")

    def test_no_policy_contrast_requests_more_contrast_evidence(self) -> None:
        from scripts.run_network_architecture_upgrade_readiness_review import (
            run_network_architecture_upgrade_readiness_review,
        )

        self._write_evidence(baseline_agreement_rate=0.98, policy_better_than_baseline_count=0)
        summary = run_network_architecture_upgrade_readiness_review(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertFalse(summary["network_upgrade_recommended"])
        self.assertIn("policy_contrast_insufficient", summary["network_upgrade_blockers"])
        self.assertEqual(summary["next_required_change"], "expand_global_99_policy_contrast_evidence")

    def test_missing_multi_map_artifact_fails_with_boundary_closed(self) -> None:
        from scripts.run_network_architecture_upgrade_readiness_review import (
            run_network_architecture_upgrade_readiness_review,
        )

        summary = run_network_architecture_upgrade_readiness_review(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_multi_map_evidence", summary["reason_codes"])
        self.assertFalse(summary["network_upgrade_recommended"])
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["modifies_network"])

    def _write_config(self) -> None:
        self.config_path.write_text(
            json.dumps(
                {
                    "schema_version": "network-architecture-upgrade-readiness-review-config/v1",
                    "source_multi_map_root": str(self.source_root),
                    "source_multi_map_summary": "global-99-multi-map-generalization-summary.json",
                    "source_family_summary": "global-99-multi-map-family-summary.json",
                    "source_policy_vs_baseline_audit": "global-99-multi-map-policy-vs-baseline-audit.json",
                    "source_scenario_results": "global-99-multi-map-scenario-results.jsonl",
                    "target_coverage_rate": 0.99,
                    "max_policy_guard_fallback_rate": 0.10,
                    "preferred_policy_guard_fallback_rate": 0.05,
                    "low_policy_contrast_baseline_agreement_rate": 0.95,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_evidence(
        self,
        *,
        source_status: str = "passed",
        reason_codes: list[str] | None = None,
        required_scenario_count: int = 21,
        failed_required_scenario_count: int = 0,
        aggregate_achieved_coverage_rate: float = 0.9963513964901461,
        min_scenario_achieved_coverage_rate: float = 0.9925,
        policy_guidance_applied: bool = True,
        policy_scored_candidate_count: int = 1520,
        policy_guided_decision_count: int = 95,
        policy_guard_fallback_count: int = 3,
        baseline_agreement_rate: float = 0.8210526315789474,
        policy_better_than_baseline_count: int = 14,
        policy_worse_than_baseline_count: int = 0,
        controlled_regression_count: int = 0,
    ) -> None:
        summary = {
            "schema_version": "global-99-multi-map-generalization-summary/v1",
            "status": source_status,
            "reason_codes": reason_codes or [],
            "target_coverage_rate": 0.99,
            "scenario_count": 24,
            "passed_scenario_count": 21,
            "failed_scenario_count": 3,
            "required_scenario_count": required_scenario_count,
            "failed_required_scenario_count": failed_required_scenario_count,
            "aggregate_achieved_coverage_rate": aggregate_achieved_coverage_rate,
            "min_scenario_achieved_coverage_rate": min_scenario_achieved_coverage_rate,
            "policy_guidance_applied": policy_guidance_applied,
            "policy_scored_candidate_count": policy_scored_candidate_count,
            "policy_guided_decision_count": policy_guided_decision_count,
            "policy_guard_fallback_count": policy_guard_fallback_count,
            "baseline_agreement_rate": baseline_agreement_rate,
            "policy_better_than_baseline_count": policy_better_than_baseline_count,
            "policy_worse_than_baseline_count": policy_worse_than_baseline_count,
            "controlled_regression_count": controlled_regression_count,
            "infeasible_reason_codes": ["coverage_target_not_met", "insufficient_budget"],
        }
        policy_audit = {
            "schema_version": "global-99-multi-map-policy-vs-baseline-audit/v1",
            "policy_guidance_applied": policy_guidance_applied,
            "policy_scored_candidate_count": policy_scored_candidate_count,
            "policy_guided_decision_count": policy_guided_decision_count,
            "policy_guard_fallback_count": policy_guard_fallback_count,
            "baseline_agreement_rate": baseline_agreement_rate,
            "policy_better_than_baseline_count": policy_better_than_baseline_count,
            "policy_worse_than_baseline_count": policy_worse_than_baseline_count,
            "controlled_regression_count": controlled_regression_count,
        }
        family_summary = {
            "schema_version": "global-99-multi-map-family-summary/v1",
            "family_count": 8,
            "passed_family_count": 8,
            "failed_family_count": 0,
            "families": {
                "open_field": {
                    "scenario_count": 3,
                    "required_scenario_count": 3,
                    "failed_required_scenario_count": 0,
                    "min_required_coverage_rate": 0.9925,
                    "family_passed": True,
                }
            },
        }
        scenario_rows = [
            {
                "schema_version": "global-99-multi-map-scenario-result/v1",
                "scenario_id": "open_field-01",
                "family_id": "open_field",
                "status": "passed",
                "generalization_required": True,
                "expected_infeasible": False,
                "achieved_coverage_rate": min_scenario_achieved_coverage_rate,
                "coverage_target_met": min_scenario_achieved_coverage_rate >= 0.99,
                "reason_codes": [],
                "infeasible_reason_codes": [],
            }
        ]

        self._write_json(self.source_root / "global-99-multi-map-generalization-summary.json", summary)
        self._write_json(self.source_root / "global-99-multi-map-policy-vs-baseline-audit.json", policy_audit)
        self._write_json(self.source_root / "global-99-multi-map-family-summary.json", family_summary)
        (self.source_root / "global-99-multi-map-scenario-results.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in scenario_rows),
            encoding="utf-8",
        )

    def _write_json(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
