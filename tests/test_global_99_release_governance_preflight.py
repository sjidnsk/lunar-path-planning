import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class Global99ReleaseGovernancePreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="global-99-release-governance-"))
        self.network_root = self.temp_dir / "network"
        self.multi_map_root = self.temp_dir / "multi-map"
        self.policy_root = self.temp_dir / "policy-guided"
        self.output_root = self.temp_dir / "output"
        self.config_path = self.temp_dir / "config.json"
        for root in (self.network_root, self.multi_map_root, self.policy_root):
            root.mkdir(parents=True, exist_ok=True)
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_current_evidence_passes_and_writes_required_artifacts(self) -> None:
        from scripts.run_global_99_release_governance_preflight import (
            run_global_99_release_governance_preflight,
        )

        self._write_sources()
        summary = run_global_99_release_governance_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertTrue(summary["release_governance_preflight_passed"])
        self.assertEqual(summary["release_governance_verdict"], "eligible_for_global_99_shadow_canary_preflight")
        self.assertEqual(summary["next_required_change"], "global_99_shadow_canary_preflight")
        self.assertTrue(summary["evidence_lineage_audit_passed"])
        self.assertTrue(summary["scope_audit_passed"])
        self.assertTrue(summary["release_boundary_audit_passed"])
        self.assertTrue(summary["kill_switch_audit_passed"])
        self.assertTrue(summary["rollback_audit_passed"])
        self.assertTrue(summary["telemetry_audit_passed"])
        self.assertEqual(summary["source_network_readiness_status"], "passed")
        self.assertEqual(summary["source_multi_map_status"], "passed")
        self.assertEqual(summary["source_policy_guided_status"], "passed")
        self.assertFalse(summary["network_upgrade_recommended"])
        self.assertEqual(summary["policy_guard_fallback_rate"], 3 / 95)
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["modifies_network"])
        self.assertFalse(summary["modifies_action_space"])
        self.assertFalse(summary["modifies_default_astar"])
        self.assertFalse(summary["uses_path_planner"])
        self.assertFalse(summary["uses_npz_or_sidecar"])
        self.assertFalse(summary["real_world_release_approved"])
        self.assertFalse(summary["real_world_performance_claimed"])
        self.assertFalse(summary["default_policy_replacement_approved"])
        self.assertFalse(summary["real_executor_connection_approved"])

        for filename in (
            "global-99-release-governance-preflight-summary.json",
            "global-99-release-governance-manifest.json",
            "global-99-release-evidence-lineage-audit.json",
            "global-99-release-scope-audit.json",
            "global-99-release-boundary-audit.json",
            "global-99-release-kill-switch-audit.json",
            "global-99-release-rollback-audit.json",
            "global-99-release-telemetry-audit.json",
            "global-99-release-rejection-report.json",
            "global-99-release-governance-preflight-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_missing_network_readiness_fails(self) -> None:
        from scripts.run_global_99_release_governance_preflight import (
            run_global_99_release_governance_preflight,
        )

        self._write_sources(write_network=False)
        summary = run_global_99_release_governance_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_network_readiness_summary", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_network_architecture_upgrade_readiness_review")

    def test_failed_network_readiness_blocks_release_governance(self) -> None:
        from scripts.run_global_99_release_governance_preflight import (
            run_global_99_release_governance_preflight,
        )

        self._write_sources(network_status="failed")
        summary = run_global_99_release_governance_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("network_readiness_not_passed", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_network_architecture_upgrade_readiness_review")

    def test_network_upgrade_recommended_routes_to_network_upgrade(self) -> None:
        from scripts.run_global_99_release_governance_preflight import (
            run_global_99_release_governance_preflight,
        )

        self._write_sources(network_upgrade_recommended=True)
        summary = run_global_99_release_governance_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("network_upgrade_recommended", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "network_architecture_upgrade_v1")

    def test_multi_map_required_failure_routes_to_multi_map_fix(self) -> None:
        from scripts.run_global_99_release_governance_preflight import (
            run_global_99_release_governance_preflight,
        )

        self._write_sources(failed_required_scenario_count=1, aggregate_coverage=0.98, min_coverage=0.97)
        summary = run_global_99_release_governance_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("multi_map_required_scenarios_not_passed", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_multi_map_generalization")

    def test_policy_regression_routes_to_policy_guided_fix(self) -> None:
        from scripts.run_global_99_release_governance_preflight import (
            run_global_99_release_governance_preflight,
        )

        self._write_sources(policy_worse=1, controlled_regression=1)
        summary = run_global_99_release_governance_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("policy_regression_detected", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_policy_guided_global_coverage")

    def test_release_boundary_violation_blocks_release(self) -> None:
        from scripts.run_global_99_release_governance_preflight import (
            run_global_99_release_governance_preflight,
        )

        self._write_sources(boundary_overrides={"publishes_checkpoint": True})
        summary = run_global_99_release_governance_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("release_boundary_violation", summary["reason_codes"])
        self.assertFalse(summary["release_boundary_audit_passed"])
        self.assertEqual(summary["next_required_change"], "resolve_global_99_release_boundary_rejections")

    def _write_config(self) -> None:
        self.config_path.write_text(
            json.dumps(
                {
                    "schema_version": "global-99-release-governance-preflight-config/v1",
                    "source_network_readiness_root": str(self.network_root),
                    "source_multi_map_root": str(self.multi_map_root),
                    "source_policy_guided_root": str(self.policy_root),
                    "target_coverage_rate": 0.99,
                    "max_policy_guard_fallback_rate": 0.05,
                    "require_network_upgrade_recommended_false": True,
                    "require_required_scenarios_all_passed": True,
                    "require_policy_read_only": True,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_sources(
        self,
        *,
        write_network: bool = True,
        network_status: str = "passed",
        network_upgrade_recommended: bool = False,
        failed_required_scenario_count: int = 0,
        aggregate_coverage: float = 0.9963513964901461,
        min_coverage: float = 0.9925,
        policy_worse: int = 0,
        controlled_regression: int = 0,
        boundary_overrides: dict | None = None,
    ) -> None:
        boundary = {
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "runs_new_ppo_update": False,
            "modifies_network": False,
            "modifies_action_space": False,
            "modifies_default_astar": False,
            "uses_path_planner": False,
            "uses_npz_or_sidecar": False,
            "real_world_release_approved": False,
            "real_world_performance_claimed": False,
            "default_policy_replacement_approved": False,
            "real_executor_connection_approved": False,
        }
        if boundary_overrides:
            boundary.update(boundary_overrides)
        if write_network:
            self._write_json(
                self.network_root / "network-architecture-upgrade-readiness-summary.json",
                {
                    "schema_version": "network-architecture-upgrade-readiness-summary/v1",
                    "status": network_status,
                    "reason_codes": [] if network_status == "passed" else ["source_multi_map_not_passed"],
                    "source_multi_map_status": "passed",
                    "target_coverage_rate": 0.99,
                    "required_scenario_count": 21,
                    "failed_required_scenario_count": failed_required_scenario_count,
                    "aggregate_achieved_coverage_rate": aggregate_coverage,
                    "min_scenario_achieved_coverage_rate": min_coverage,
                    "policy_guidance_applied": True,
                    "policy_scored_candidate_count": 1520,
                    "policy_guided_decision_count": 95,
                    "policy_guard_fallback_count": 3,
                    "policy_guard_fallback_rate": 3 / 95,
                    "baseline_agreement_rate": 0.8210526315789474,
                    "policy_better_than_baseline_count": 14,
                    "policy_worse_than_baseline_count": policy_worse,
                    "controlled_regression_count": controlled_regression,
                    "network_upgrade_recommended": network_upgrade_recommended,
                    "network_upgrade_readiness_decision": "defer_network_upgrade_global_99_synthetic_goal_met",
                    "next_required_change": "global_99_release_governance_preflight",
                    **boundary,
                },
            )
        self._write_json(
            self.multi_map_root / "global-99-multi-map-generalization-summary.json",
            {
                "schema_version": "global-99-multi-map-generalization-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "target_coverage_rate": 0.99,
                "required_scenario_count": 21,
                "failed_required_scenario_count": failed_required_scenario_count,
                "aggregate_achieved_coverage_rate": aggregate_coverage,
                "min_scenario_achieved_coverage_rate": min_coverage,
                "coverage_target_met_all_required_scenarios": failed_required_scenario_count == 0,
                "policy_guidance_applied": True,
                "policy_scored_candidate_count": 1520,
                "policy_guided_decision_count": 95,
                "policy_guard_fallback_count": 3,
                "baseline_agreement_rate": 0.8210526315789474,
                "policy_better_than_baseline_count": 14,
                "policy_worse_than_baseline_count": policy_worse,
                "controlled_regression_count": controlled_regression,
                "next_required_change": "network_architecture_upgrade_readiness_review",
                **boundary,
            },
        )
        self._write_json(
            self.policy_root / "policy-guided-global-coverage-summary.json",
            {
                "schema_version": "policy-guided-global-coverage-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "target_coverage_rate": 0.99,
                "achieved_coverage_rate": 0.9931,
                "coverage_target_met": True,
                "policy_loaded": True,
                "policy_guidance_applied": True,
                "policy_read_only": True,
                "policy_scored_candidate_count": 96,
                "policy_guided_decision_count": 6,
                "policy_guard_fallback_count": 0,
                "policy_worse_than_baseline_count": policy_worse,
                "controlled_regression_count": controlled_regression,
                "next_required_change": "global_99_multi_map_generalization",
                **boundary,
            },
        )

    def _write_json(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
