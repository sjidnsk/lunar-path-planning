import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class Global99RealMapShadowCanaryReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="global-99-real-map-shadow-canary-replay-"))
        self.preflight_root = self.temp_dir / "shadow-canary-preflight"
        self.release_root = self.temp_dir / "release-governance"
        self.shadow_replay_root = self.temp_dir / "real-map-shadow-replay"
        self.output_root = self.temp_dir / "output"
        self.config_path = self.temp_dir / "config.json"
        for root in (self.preflight_root, self.release_root, self.shadow_replay_root):
            root.mkdir(parents=True, exist_ok=True)
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_current_evidence_passes_and_writes_required_artifacts(self) -> None:
        from scripts.run_global_99_real_map_shadow_canary_replay import run_global_99_real_map_shadow_canary_replay

        self._write_sources()
        summary = run_global_99_real_map_shadow_canary_replay(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertTrue(summary["real_map_shadow_canary_replay_passed"])
        self.assertEqual(
            summary["real_map_shadow_canary_replay_verdict"],
            "eligible_for_global_99_real_map_evidence_refresh_drift_audit",
        )
        self.assertEqual(summary["next_required_change"], "global_99_real_map_evidence_refresh_drift_audit")
        self.assertEqual(summary["source_real_map_shadow_canary_preflight_status"], "passed")
        self.assertEqual(
            summary["source_real_map_shadow_canary_preflight_next_required_change"],
            "global_99_real_map_shadow_canary_replay",
        )
        self.assertEqual(summary["source_real_map_release_governance_status"], "passed")
        self.assertEqual(
            summary["source_real_map_release_governance_next_required_change"],
            "global_99_real_map_shadow_canary_preflight",
        )
        self.assertEqual(summary["source_real_map_shadow_replay_status"], "passed")
        self.assertEqual(
            summary["source_real_map_shadow_replay_next_required_change"],
            "global_99_real_map_release_governance_preflight",
        )
        self.assertTrue(summary["source_match_audit_passed"])
        self.assertEqual(summary["scenario_mismatch_count"], 0)
        self.assertEqual(summary["max_replay_coverage_delta"], 0.0)
        self.assertEqual(summary["max_replay_path_cost_delta_m"], 0.0)
        self.assertFalse(summary["open_grid_fallback_used"])
        self.assertEqual(summary["policy_guard_fallback_rate"], 3 / 95)
        self.assertTrue(summary["fallback_audit_passed"])
        self.assertTrue(summary["boundary_audit_passed"])
        self.assertTrue(summary["kill_switch_audit_passed"])
        self.assertTrue(summary["rollback_audit_passed"])
        self.assertTrue(summary["telemetry_audit_passed"])
        self.assertFalse(summary["starts_online_canary"])
        self.assertEqual(summary["canary_traffic_fraction"], 0.0)
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["modifies_network"])
        self.assertFalse(summary["modifies_action_space"])
        self.assertFalse(summary["modifies_default_astar"])
        self.assertFalse(summary["real_world_release_approved"])
        self.assertFalse(summary["real_world_performance_claimed"])

        for filename in (
            "global-99-real-map-shadow-canary-replay-summary.json",
            "global-99-real-map-shadow-canary-replay-manifest.json",
            "global-99-real-map-shadow-canary-replay-source-match-audit.json",
            "global-99-real-map-shadow-canary-replay-fallback-audit.json",
            "global-99-real-map-shadow-canary-replay-boundary-audit.json",
            "global-99-real-map-shadow-canary-replay-kill-switch-audit.json",
            "global-99-real-map-shadow-canary-replay-rollback-audit.json",
            "global-99-real-map-shadow-canary-replay-telemetry-audit.json",
            "global-99-real-map-shadow-canary-replay-rejection-report.json",
            "global-99-real-map-shadow-canary-replay-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_missing_preflight_routes_to_preflight_fix(self) -> None:
        from scripts.run_global_99_real_map_shadow_canary_replay import run_global_99_real_map_shadow_canary_replay

        self._write_sources(write_preflight=False)
        summary = run_global_99_real_map_shadow_canary_replay(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_real_map_shadow_canary_preflight_summary", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_real_map_shadow_canary_preflight")

    def test_source_match_failure_routes_to_determinism_fix(self) -> None:
        from scripts.run_global_99_real_map_shadow_canary_replay import run_global_99_real_map_shadow_canary_replay

        self._write_sources(source_match_passed=False, scenario_mismatch_count=1, coverage_delta=0.01)
        summary = run_global_99_real_map_shadow_canary_replay(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("real_map_shadow_canary_replay_source_match_failed", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_real_map_shadow_canary_replay_determinism")

    def test_open_grid_or_fallback_routes_to_contract_fix(self) -> None:
        from scripts.run_global_99_real_map_shadow_canary_replay import run_global_99_real_map_shadow_canary_replay

        self._write_sources(open_grid_fallback_used=True)
        summary = run_global_99_real_map_shadow_canary_replay(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("real_map_shadow_canary_open_grid_fallback_detected", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_real_map_path_feedback_contract")

    def test_fallback_rate_above_threshold_routes_to_guard_fix(self) -> None:
        from scripts.run_global_99_real_map_shadow_canary_replay import run_global_99_real_map_shadow_canary_replay

        self._write_sources(policy_guard_fallback_rate=0.2)
        summary = run_global_99_real_map_shadow_canary_replay(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("policy_guard_fallback_rate_too_high", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_real_map_shadow_canary_guard_fallback")

    def test_boundary_violation_blocks_replay(self) -> None:
        from scripts.run_global_99_real_map_shadow_canary_replay import run_global_99_real_map_shadow_canary_replay

        self._write_sources(boundary_overrides={"connects_real_executor": True})
        summary = run_global_99_real_map_shadow_canary_replay(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("real_map_shadow_canary_replay_boundary_violation", summary["reason_codes"])
        self.assertEqual(
            summary["next_required_change"],
            "resolve_global_99_real_map_shadow_canary_replay_boundary_rejections",
        )

    def _write_config(self) -> None:
        self.config_path.write_text(
            json.dumps(
                {
                    "schema_version": "global-99-real-map-shadow-canary-replay-config/v1",
                    "source_real_map_shadow_canary_preflight_root": str(self.preflight_root),
                    "source_real_map_release_governance_root": str(self.release_root),
                    "source_real_map_shadow_replay_root": str(self.shadow_replay_root),
                    "max_policy_guard_fallback_rate": 0.05,
                    "source_match_coverage_tolerance": 1e-12,
                    "source_match_path_cost_tolerance_m": 1e-9,
                    "shadow_replay_mode": True,
                    "offline_canary_replay_mode": True,
                    "canary_traffic_fraction": 0.0,
                    "require_preflight_passed": True,
                    "require_source_match_audit_passed": True,
                    "require_no_open_grid_fallback": True,
                    "require_boundary_closed": True,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_sources(
        self,
        *,
        write_preflight: bool = True,
        preflight_status: str = "passed",
        preflight_next: str = "global_99_real_map_shadow_canary_replay",
        source_match_passed: bool = True,
        scenario_mismatch_count: int = 0,
        coverage_delta: float = 0.0,
        path_cost_delta: float = 0.0,
        open_grid_fallback_used: bool = False,
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

        if write_preflight:
            preflight = {
                "schema_version": "global-99-real-map-shadow-canary-preflight-summary/v1",
                "status": preflight_status,
                "reason_codes": [] if preflight_status == "passed" else ["preflight_failed"],
                "real_map_shadow_canary_preflight_verdict": "eligible_for_global_99_real_map_shadow_canary_replay",
                "next_required_change": preflight_next,
                "source_match_audit_passed": source_match_passed,
                "scenario_mismatch_count": scenario_mismatch_count,
                "open_grid_fallback_used": open_grid_fallback_used,
                "policy_guard_fallback_rate": policy_guard_fallback_rate,
                "boundary_audit_passed": not bool(boundary_overrides),
                **boundary,
            }
            (self.preflight_root / "global-99-real-map-shadow-canary-preflight-summary.json").write_text(
                json.dumps(preflight, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        release = {
            "schema_version": "global-99-real-map-release-governance-preflight-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "next_required_change": "global_99_real_map_shadow_canary_preflight",
            "source_match_audit_passed": source_match_passed,
            "scenario_mismatch_count": scenario_mismatch_count,
            "open_grid_fallback_used": open_grid_fallback_used,
            "policy_guard_fallback_rate": policy_guard_fallback_rate,
            **boundary,
        }
        (self.release_root / "global-99-real-map-release-governance-preflight-summary.json").write_text(
            json.dumps(release, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        shadow = {
            "schema_version": "global-99-real-map-shadow-replay-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "next_required_change": "global_99_real_map_release_governance_preflight",
            "source_match_audit_passed": source_match_passed,
            "scenario_mismatch_count": scenario_mismatch_count,
            "max_replay_coverage_delta": coverage_delta,
            "max_replay_path_cost_delta_m": path_cost_delta,
            "open_grid_fallback_used": open_grid_fallback_used,
            **boundary,
        }
        (self.shadow_replay_root / "global-99-real-map-shadow-replay-summary.json").write_text(
            json.dumps(shadow, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
