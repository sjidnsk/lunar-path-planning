import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class Global99RealMapReleaseGovernancePreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="global-99-real-map-release-governance-"))
        self.shadow_replay_root = self.temp_dir / "real-map-shadow-replay"
        self.preflight_root = self.temp_dir / "real-map-preflight"
        self.domain_gap_root = self.temp_dir / "domain-gap"
        self.output_root = self.temp_dir / "output"
        self.config_path = self.temp_dir / "config.json"
        for root in (self.shadow_replay_root, self.preflight_root, self.domain_gap_root):
            root.mkdir(parents=True, exist_ok=True)
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_current_evidence_passes_and_writes_required_artifacts(self) -> None:
        from scripts.run_global_99_real_map_release_governance_preflight import (
            run_global_99_real_map_release_governance_preflight,
        )

        self._write_sources()
        summary = run_global_99_real_map_release_governance_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertTrue(summary["real_map_release_governance_preflight_passed"])
        self.assertEqual(
            summary["real_map_release_governance_verdict"],
            "eligible_for_global_99_real_map_shadow_canary_preflight",
        )
        self.assertEqual(summary["next_required_change"], "global_99_real_map_shadow_canary_preflight")
        self.assertEqual(summary["source_real_map_shadow_replay_status"], "passed")
        self.assertEqual(
            summary["source_real_map_shadow_replay_next_required_change"],
            "global_99_real_map_release_governance_preflight",
        )
        self.assertEqual(summary["source_real_map_preflight_status"], "passed")
        self.assertEqual(summary["source_real_map_preflight_next_required_change"], "global_99_real_map_shadow_replay")
        self.assertEqual(summary["source_domain_gap_status"], "passed")
        self.assertEqual(summary["domain_gap_verdict"], "acceptable_for_next_pilot")
        self.assertTrue(summary["evidence_lineage_audit_passed"])
        self.assertTrue(summary["shadow_replay_audit_passed"])
        self.assertTrue(summary["real_map_preflight_audit_passed"])
        self.assertTrue(summary["domain_gap_audit_passed"])
        self.assertTrue(summary["path_feedback_audit_passed"])
        self.assertTrue(summary["scope_audit_passed"])
        self.assertTrue(summary["release_boundary_audit_passed"])
        self.assertTrue(summary["kill_switch_audit_passed"])
        self.assertTrue(summary["rollback_audit_passed"])
        self.assertTrue(summary["telemetry_audit_passed"])
        self.assertTrue(summary["source_match_audit_passed"])
        self.assertEqual(summary["scenario_mismatch_count"], 0)
        self.assertEqual(summary["max_replay_coverage_delta"], 0.0)
        self.assertEqual(summary["max_replay_path_cost_delta_m"], 0.0)
        self.assertEqual(summary["slice_count"], 12)
        self.assertEqual(summary["roi_group_count"], 4)
        self.assertEqual(summary["context_id_missing_count"], 0)
        self.assertEqual(summary["fallback_or_open_grid_count"], 0)
        self.assertEqual(summary["policy_guard_fallback_rate"], 3 / 95)
        self.assertTrue(summary["uses_lola_quasi_real_roi"])
        self.assertTrue(summary["uses_path_feedback_sidecar"])
        self.assertTrue(summary["audits_offline_path_feedback_replay"])
        self.assertTrue(summary["audits_offline_path_planner_route_replay"])
        self.assertEqual(summary["audited_path_planner_use_scope"], "offline_path_feedback_replay_only")
        self.assertFalse(summary["uses_path_planner"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["modifies_network"])
        self.assertFalse(summary["modifies_action_space"])
        self.assertFalse(summary["modifies_default_astar"])
        self.assertFalse(summary["real_world_release_approved"])
        self.assertFalse(summary["real_world_performance_claimed"])

        for filename in (
            "global-99-real-map-release-governance-preflight-summary.json",
            "global-99-real-map-release-governance-manifest.json",
            "global-99-real-map-release-evidence-lineage-audit.json",
            "global-99-real-map-release-shadow-replay-audit.json",
            "global-99-real-map-release-real-map-preflight-audit.json",
            "global-99-real-map-release-domain-gap-audit.json",
            "global-99-real-map-release-path-feedback-audit.json",
            "global-99-real-map-release-scope-audit.json",
            "global-99-real-map-release-boundary-audit.json",
            "global-99-real-map-release-kill-switch-audit.json",
            "global-99-real-map-release-rollback-audit.json",
            "global-99-real-map-release-telemetry-audit.json",
            "global-99-real-map-release-rejection-report.json",
            "global-99-real-map-release-governance-preflight-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_missing_shadow_replay_routes_to_shadow_replay_fix(self) -> None:
        from scripts.run_global_99_real_map_release_governance_preflight import (
            run_global_99_real_map_release_governance_preflight,
        )

        self._write_sources(write_shadow_replay=False)
        summary = run_global_99_real_map_release_governance_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_real_map_shadow_replay_summary", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_real_map_shadow_replay")

    def test_real_map_preflight_failure_routes_to_preflight_fix(self) -> None:
        from scripts.run_global_99_real_map_release_governance_preflight import (
            run_global_99_real_map_release_governance_preflight,
        )

        self._write_sources(preflight_status="failed")
        summary = run_global_99_real_map_release_governance_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("real_map_preflight_not_passed", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_real_map_preflight")

    def test_domain_gap_failure_routes_to_domain_gap_fix(self) -> None:
        from scripts.run_global_99_real_map_release_governance_preflight import (
            run_global_99_real_map_release_governance_preflight,
        )

        self._write_sources(domain_gap_status="failed", domain_gap_verdict="needs_more_evidence")
        summary = run_global_99_real_map_release_governance_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("domain_gap_not_passed", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_quasi_real_map_domain_gap_evidence")

    def test_source_match_mismatch_routes_to_determinism_fix(self) -> None:
        from scripts.run_global_99_real_map_release_governance_preflight import (
            run_global_99_real_map_release_governance_preflight,
        )

        self._write_sources(source_match_passed=False, scenario_mismatch_count=1, coverage_delta=0.01)
        summary = run_global_99_real_map_release_governance_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("real_map_shadow_replay_source_match_failed", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_real_map_shadow_replay_determinism")

    def test_context_identity_failure_routes_to_context_fix(self) -> None:
        from scripts.run_global_99_real_map_release_governance_preflight import (
            run_global_99_real_map_release_governance_preflight,
        )

        self._write_sources(context_id_missing_count=1, legacy_identity_fallback_count=1)
        summary = run_global_99_real_map_release_governance_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("real_map_context_id_missing", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_real_map_context_identity")

    def test_open_grid_or_path_feedback_regression_routes_to_contract_fix(self) -> None:
        from scripts.run_global_99_real_map_release_governance_preflight import (
            run_global_99_real_map_release_governance_preflight,
        )

        self._write_sources(open_grid_fallback_used=True, safety_regression_count=1)
        summary = run_global_99_real_map_release_governance_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("real_map_open_grid_fallback_detected", summary["reason_codes"])
        self.assertIn("real_map_path_feedback_regression", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_real_map_path_feedback_contract")

    def test_fallback_rate_above_threshold_routes_to_guard_fix(self) -> None:
        from scripts.run_global_99_real_map_release_governance_preflight import (
            run_global_99_real_map_release_governance_preflight,
        )

        self._write_sources(policy_guard_fallback_rate=0.2)
        summary = run_global_99_real_map_release_governance_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("policy_guard_fallback_rate_too_high", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_shadow_canary_guard_fallback")

    def test_boundary_violation_blocks_release_governance(self) -> None:
        from scripts.run_global_99_real_map_release_governance_preflight import (
            run_global_99_real_map_release_governance_preflight,
        )

        self._write_sources(boundary_overrides={"starts_online_canary": True})
        summary = run_global_99_real_map_release_governance_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("real_map_release_boundary_violation", summary["reason_codes"])
        self.assertFalse(summary["release_boundary_audit_passed"])
        self.assertEqual(
            summary["next_required_change"],
            "resolve_global_99_real_map_release_governance_boundary_rejections",
        )

    def _write_config(self) -> None:
        self.config_path.write_text(
            json.dumps(
                {
                    "schema_version": "global-99-real-map-release-governance-preflight-config/v1",
                    "source_real_map_shadow_replay_root": str(self.shadow_replay_root),
                    "source_real_map_preflight_root": str(self.preflight_root),
                    "source_quasi_real_domain_gap_root": str(self.domain_gap_root),
                    "target_coverage_rate": 0.99,
                    "max_policy_guard_fallback_rate": 0.05,
                    "source_match_coverage_tolerance": 1e-12,
                    "source_match_path_cost_tolerance_m": 1e-9,
                    "require_real_map_shadow_replay_passed": True,
                    "require_real_map_preflight_passed": True,
                    "require_domain_gap_acceptable": True,
                    "require_source_match_audit_passed": True,
                    "require_context_ids": True,
                    "require_no_open_grid_fallback": True,
                    "require_release_boundaries_closed": True,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_sources(
        self,
        *,
        write_shadow_replay: bool = True,
        shadow_status: str = "passed",
        shadow_next: str = "global_99_real_map_release_governance_preflight",
        preflight_status: str = "passed",
        preflight_next: str = "global_99_real_map_shadow_replay",
        domain_gap_status: str = "passed",
        domain_gap_verdict: str = "acceptable_for_next_pilot",
        source_match_passed: bool = True,
        scenario_mismatch_count: int = 0,
        coverage_delta: float = 0.0,
        path_cost_delta: float = 0.0,
        slice_count: int = 12,
        roi_group_count: int = 4,
        context_id_missing_count: int = 0,
        legacy_identity_fallback_count: int = 0,
        open_grid_fallback_used: bool = False,
        fallback_or_open_grid_count: int = 0,
        safety_regression_count: int = 0,
        policy_guard_fallback_rate: float = 3 / 95,
        boundary_overrides: dict | None = None,
    ) -> None:
        boundary = {
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
            "runs_new_ppo_update": False,
            "modifies_network": False,
            "modifies_action_space": False,
            "modifies_default_astar": False,
            "real_world_release_approved": False,
            "real_world_performance_claimed": False,
        }
        if boundary_overrides:
            boundary.update(boundary_overrides)

        if write_shadow_replay:
            shadow = {
                "schema_version": "global-99-real-map-shadow-replay-summary/v1",
                "status": shadow_status,
                "reason_codes": [] if shadow_status == "passed" else ["shadow_failed"],
                "real_map_shadow_replay_passed": shadow_status == "passed",
                "real_map_shadow_replay_verdict": "eligible_for_global_99_real_map_release_governance_preflight",
                "next_required_change": shadow_next,
                "source_match_audit_passed": source_match_passed,
                "scenario_mismatch_count": scenario_mismatch_count,
                "max_replay_coverage_delta": coverage_delta,
                "max_replay_path_cost_delta_m": path_cost_delta,
                "slice_count": slice_count,
                "roi_group_count": roi_group_count,
                "context_id_missing_count": context_id_missing_count,
                "legacy_identity_fallback_count": legacy_identity_fallback_count,
                "open_grid_fallback_used": open_grid_fallback_used,
                "fallback_or_open_grid_count": fallback_or_open_grid_count,
                "safety_regression_count": safety_regression_count,
                "contract_violation_count": 0,
                "path_cost_regression_count": 0,
                "risk_regression_count": 0,
                "source_selection_regression_count": 0,
                "path_planner_use_scope": "offline_path_feedback_replay_only",
                **boundary,
            }
            (self.shadow_replay_root / "global-99-real-map-shadow-replay-summary.json").write_text(
                json.dumps(shadow, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        preflight = {
            "schema_version": "global-99-real-map-preflight-summary/v1",
            "status": preflight_status,
            "reason_codes": [] if preflight_status == "passed" else ["preflight_failed"],
            "real_map_preflight_passed": preflight_status == "passed",
            "real_map_preflight_verdict": "eligible_for_global_99_real_map_shadow_replay",
            "next_required_change": preflight_next,
            "source_domain_gap_status": domain_gap_status,
            "domain_gap_verdict": domain_gap_verdict,
            "shadow_replay_source_match_audit_passed": True,
            "shadow_replay_policy_guard_fallback_rate": policy_guard_fallback_rate,
            "slice_count": slice_count,
            "roi_group_count": roi_group_count,
            "context_id_missing_count": context_id_missing_count,
            "legacy_identity_fallback_count": legacy_identity_fallback_count,
            "fallback_or_open_grid_count": fallback_or_open_grid_count,
            "safety_regression_count": safety_regression_count,
            "contract_violation_count": 0,
            "path_cost_regression_count": 0,
            "risk_regression_count": 0,
            "source_selection_regression_count": 0,
            **boundary,
        }
        (self.preflight_root / "global-99-real-map-preflight-summary.json").write_text(
            json.dumps(preflight, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        domain = {
            "schema_version": "quasi-real-map-domain-gap-summary/v1",
            "status": domain_gap_status,
            "reason_codes": [] if domain_gap_status == "passed" else ["domain_gap_failed"],
            "domain_gap_verdict": domain_gap_verdict,
            "slice_count": slice_count,
            "roi_group_count": roi_group_count,
            "context_id_missing_count": context_id_missing_count,
            "legacy_identity_fallback_count": legacy_identity_fallback_count,
            "fallback_or_open_grid_count": fallback_or_open_grid_count,
            "open_grid_fallback_count": int(open_grid_fallback_used),
            "safety_regression_count": safety_regression_count,
            "contract_violation_count": 0,
            "path_cost_regression_count": 0,
            "risk_regression_count": 0,
            "source_selection_regression_count": 0,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "runs_ppo_update": False,
            "performance_claimed": False,
        }
        (self.domain_gap_root / "quasi-real-map-domain-gap-summary.json").write_text(
            json.dumps(domain, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        path_feedback = {
            "schema_version": "path-feedback-summary/v1",
            "scenario_count": 2,
            "open_grid_fallback_used": open_grid_fallback_used,
            "path_planning_failure_count": 0,
            "replan_count": 0,
        }
        (self.domain_gap_root / "quasi-real-map-path-feedback-summary.json").write_text(
            json.dumps(path_feedback, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
