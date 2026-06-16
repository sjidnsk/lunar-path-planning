import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class Global99RealMapPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="global-99-real-map-preflight-"))
        self.shadow_replay_root = self.temp_dir / "shadow-replay"
        self.domain_gap_root = self.temp_dir / "domain-gap"
        self.path_feedback_root = self.temp_dir / "path-feedback-validation"
        self.output_root = self.temp_dir / "output"
        self.config_path = self.temp_dir / "config.json"
        for root in (self.shadow_replay_root, self.domain_gap_root, self.path_feedback_root):
            root.mkdir(parents=True, exist_ok=True)
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_current_evidence_passes_and_writes_required_artifacts(self) -> None:
        from scripts.run_global_99_real_map_preflight import run_global_99_real_map_preflight

        self._write_sources()
        summary = run_global_99_real_map_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertTrue(summary["real_map_preflight_passed"])
        self.assertEqual(summary["real_map_preflight_verdict"], "eligible_for_global_99_real_map_shadow_replay")
        self.assertEqual(summary["next_required_change"], "global_99_real_map_shadow_replay")
        self.assertTrue(summary["evidence_lineage_audit_passed"])
        self.assertTrue(summary["domain_gap_audit_passed"])
        self.assertTrue(summary["path_feedback_audit_passed"])
        self.assertTrue(summary["scope_audit_passed"])
        self.assertTrue(summary["boundary_audit_passed"])
        self.assertTrue(summary["kill_switch_audit_passed"])
        self.assertTrue(summary["rollback_audit_passed"])
        self.assertTrue(summary["telemetry_audit_passed"])
        self.assertEqual(summary["source_shadow_canary_replay_status"], "passed")
        self.assertEqual(summary["source_shadow_canary_replay_next_required_change"], "global_99_real_map_preflight")
        self.assertEqual(summary["source_domain_gap_status"], "passed")
        self.assertEqual(summary["target_coverage_rate"], 0.99)
        self.assertEqual(summary["domain_gap_verdict"], "acceptable_for_next_pilot")
        self.assertEqual(summary["slice_count"], 12)
        self.assertEqual(summary["roi_group_count"], 4)
        self.assertEqual(summary["context_id_missing_count"], 0)
        self.assertEqual(summary["legacy_identity_fallback_count"], 0)
        self.assertEqual(summary["fallback_or_open_grid_count"], 0)
        self.assertEqual(summary["shadow_replay_policy_guard_fallback_rate"], 3 / 95)
        self.assertTrue(summary["real_map_preflight_only"])
        self.assertTrue(summary["uses_lola_quasi_real_roi"])
        self.assertTrue(summary["uses_path_feedback_sidecar"])
        self.assertFalse(summary["real_world_release_approved"])
        self.assertFalse(summary["real_world_performance_claimed"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])
        self.assertEqual(summary["canary_traffic_fraction"], 0.0)
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["modifies_network"])
        self.assertFalse(summary["modifies_action_space"])
        self.assertFalse(summary["modifies_default_astar"])
        self.assertFalse(summary["uses_path_planner"])

        for filename in (
            "global-99-real-map-preflight-summary.json",
            "global-99-real-map-preflight-manifest.json",
            "global-99-real-map-evidence-lineage-audit.json",
            "global-99-real-map-domain-gap-audit.json",
            "global-99-real-map-path-feedback-audit.json",
            "global-99-real-map-scope-audit.json",
            "global-99-real-map-boundary-audit.json",
            "global-99-real-map-kill-switch-audit.json",
            "global-99-real-map-rollback-audit.json",
            "global-99-real-map-telemetry-audit.json",
            "global-99-real-map-rejection-report.json",
            "global-99-real-map-preflight-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_missing_shadow_replay_routes_to_shadow_replay_fix(self) -> None:
        from scripts.run_global_99_real_map_preflight import run_global_99_real_map_preflight

        self._write_sources(write_shadow_replay=False)
        summary = run_global_99_real_map_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_shadow_canary_replay_summary", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_shadow_canary_replay")

    def test_domain_gap_failure_routes_to_domain_gap_fix(self) -> None:
        from scripts.run_global_99_real_map_preflight import run_global_99_real_map_preflight

        self._write_sources(domain_gap_status="failed", domain_gap_verdict="needs_more_evidence")
        summary = run_global_99_real_map_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("domain_gap_not_passed", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_quasi_real_map_domain_gap_evidence")

    def test_slice_or_roi_shortfall_routes_to_real_map_coverage_expansion(self) -> None:
        from scripts.run_global_99_real_map_preflight import run_global_99_real_map_preflight

        self._write_sources(slice_count=11, roi_group_count=3)
        summary = run_global_99_real_map_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("real_map_slice_count_below_minimum", summary["reason_codes"])
        self.assertIn("real_map_roi_group_count_below_minimum", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "expand_real_map_roi_coverage")

    def test_context_identity_failure_routes_to_context_identity_fix(self) -> None:
        from scripts.run_global_99_real_map_preflight import run_global_99_real_map_preflight

        self._write_sources(context_id_missing_count=1, legacy_identity_fallback_count=1)
        summary = run_global_99_real_map_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("real_map_context_id_missing", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_real_map_context_identity")

    def test_open_grid_or_path_feedback_regression_routes_to_contract_fix(self) -> None:
        from scripts.run_global_99_real_map_preflight import run_global_99_real_map_preflight

        self._write_sources(fallback_or_open_grid_count=1, safety_regression_count=1, open_grid_fallback_used=True)
        summary = run_global_99_real_map_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("real_map_open_grid_fallback_detected", summary["reason_codes"])
        self.assertIn("real_map_path_feedback_regression", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_real_map_path_feedback_contract")

    def test_shadow_replay_fallback_rate_above_threshold_routes_to_guard_fix(self) -> None:
        from scripts.run_global_99_real_map_preflight import run_global_99_real_map_preflight

        self._write_sources(policy_guard_fallback_rate=0.2)
        summary = run_global_99_real_map_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("shadow_replay_policy_guard_fallback_rate_too_high", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_shadow_canary_guard_fallback")

    def test_boundary_violation_blocks_real_map_preflight(self) -> None:
        from scripts.run_global_99_real_map_preflight import run_global_99_real_map_preflight

        self._write_sources(boundary_overrides={"connects_real_executor": True})
        summary = run_global_99_real_map_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("global_99_real_map_boundary_violation", summary["reason_codes"])
        self.assertFalse(summary["boundary_audit_passed"])
        self.assertEqual(summary["next_required_change"], "resolve_global_99_real_map_preflight_boundary_rejections")

    def test_boundary_normalization_outputs_explicit_values_when_upstream_has_null(self) -> None:
        from scripts.run_global_99_real_map_preflight import run_global_99_real_map_preflight

        self._write_sources(
            boundary_overrides={
                "real_world_release_approved": None,
                "connects_real_executor": None,
                "uses_npz_or_sidecar": None,
            }
        )
        summary = run_global_99_real_map_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertIs(summary["real_world_release_approved"], False)
        self.assertIs(summary["connects_real_executor"], False)
        self.assertIs(summary["uses_lola_quasi_real_roi"], True)
        self.assertIs(summary["uses_path_feedback_sidecar"], True)
        self.assertIs(summary["uses_path_planner"], False)

    def _write_config(self, *, canary_traffic_fraction: float = 0.0) -> None:
        self.config_path.write_text(
            json.dumps(
                {
                    "schema_version": "global-99-real-map-preflight-config/v1",
                    "source_shadow_canary_replay_root": str(self.shadow_replay_root),
                    "source_quasi_real_domain_gap_root": str(self.domain_gap_root),
                    "source_path_feedback_validation_root": str(self.path_feedback_root),
                    "target_coverage_rate": 0.99,
                    "min_real_map_slice_count": 12,
                    "min_real_map_roi_group_count": 4,
                    "max_policy_guard_fallback_rate": 0.05,
                    "require_domain_gap_acceptable": True,
                    "require_no_open_grid_fallback": True,
                    "require_context_ids": True,
                    "real_map_preflight_only": True,
                    "canary_traffic_fraction": canary_traffic_fraction,
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
        domain_gap_status: str = "passed",
        domain_gap_verdict: str = "acceptable_for_next_pilot",
        slice_count: int = 12,
        roi_group_count: int = 4,
        context_id_missing_count: int = 0,
        legacy_identity_fallback_count: int = 0,
        fallback_or_open_grid_count: int = 0,
        safety_regression_count: int = 0,
        policy_guard_fallback_rate: float = 3 / 95,
        open_grid_fallback_used: bool = False,
        boundary_overrides: dict | None = None,
    ) -> None:
        boundary = self._boundary(boundary_overrides)
        if write_shadow_replay:
            self._write_json(
                self.shadow_replay_root / "global-99-shadow-canary-replay-summary.json",
                {
                    "schema_version": "global-99-shadow-canary-replay-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "next_required_change": "global_99_real_map_preflight",
                    "shadow_replay_passed": True,
                    "offline_canary_replay_passed": True,
                    "source_match_audit_passed": True,
                    "policy_guard_fallback_rate": policy_guard_fallback_rate,
                    **boundary,
                },
            )
        self._write_json(
            self.domain_gap_root / "quasi-real-map-domain-gap-summary.json",
            {
                "schema_version": "quasi-real-map-domain-gap-summary/v1",
                "status": domain_gap_status,
                "reason_codes": [] if domain_gap_status == "passed" else ["domain_gap_failed"],
                "domain_gap_verdict": domain_gap_verdict,
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
                "policy_shadow_only": True,
                "runs_ppo_update": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
                **boundary,
            },
        )
        self._write_json(
            self.domain_gap_root / "quasi-real-map-path-feedback-summary.json",
            {
                "schema_version": "path-feedback-summary/v1",
                "scenario_count": slice_count,
                "selection_changed_count": 10,
                "open_grid_fallback_used": open_grid_fallback_used,
                "open_grid_fallback_used_gate": {
                    "status": "failed" if open_grid_fallback_used else "passed",
                    "expected": False,
                    "actual": open_grid_fallback_used,
                    "reason_codes": ["open_grid_fallback_used" if open_grid_fallback_used else "open_grid_fallback_not_used"],
                },
            },
        )
        self._write_json(
            self.domain_gap_root / "quasi-real-map-path-feedback-bridge-summary.json",
            {
                "schema_version": "quasi-real-map-path-feedback-bridge-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "context_id_missing_count": context_id_missing_count,
                "legacy_identity_fallback_count": legacy_identity_fallback_count,
            },
        )
        self._write_json(
            self.path_feedback_root / "path-feedback-summary.json",
            {
                "schema_version": "path-feedback-validation-summary/v1",
                "status": "passed",
                "open_grid_fallback_used": open_grid_fallback_used,
            },
        )

    def _boundary(self, overrides: dict | None = None) -> dict:
        boundary = {
            "real_world_release_approved": False,
            "real_world_performance_claimed": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
            "runs_new_ppo_update": False,
            "modifies_network": False,
            "modifies_action_space": False,
            "modifies_default_astar": False,
            "uses_path_planner": False,
            "uses_npz_or_sidecar": False,
        }
        if overrides:
            boundary.update(overrides)
        return boundary

    def _write_json(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
