import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class Global99RealMapShadowCanaryPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="global-99-real-map-shadow-canary-preflight-"))
        self.release_root = self.temp_dir / "release-governance"
        self.shadow_replay_root = self.temp_dir / "real-map-shadow-replay"
        self.output_root = self.temp_dir / "output"
        self.config_path = self.temp_dir / "config.json"
        for root in (self.release_root, self.shadow_replay_root):
            root.mkdir(parents=True, exist_ok=True)
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_current_evidence_passes_and_writes_required_artifacts(self) -> None:
        from scripts.run_global_99_real_map_shadow_canary_preflight import (
            run_global_99_real_map_shadow_canary_preflight,
        )

        self._write_sources()
        summary = run_global_99_real_map_shadow_canary_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertTrue(summary["real_map_shadow_canary_preflight_passed"])
        self.assertEqual(
            summary["real_map_shadow_canary_preflight_verdict"],
            "eligible_for_global_99_real_map_shadow_canary_replay",
        )
        self.assertEqual(summary["next_required_change"], "global_99_real_map_shadow_canary_replay")
        self.assertEqual(summary["source_real_map_release_governance_status"], "passed")
        self.assertEqual(
            summary["source_real_map_release_governance_next_required_change"],
            "global_99_real_map_shadow_canary_preflight",
        )
        self.assertTrue(summary["shadow_replay_eligibility_audit_passed"])
        self.assertTrue(summary["canary_eligibility_audit_passed"])
        self.assertTrue(summary["boundary_audit_passed"])
        self.assertTrue(summary["kill_switch_audit_passed"])
        self.assertTrue(summary["rollback_audit_passed"])
        self.assertTrue(summary["telemetry_audit_passed"])
        self.assertTrue(summary["source_match_audit_passed"])
        self.assertEqual(summary["scenario_mismatch_count"], 0)
        self.assertFalse(summary["open_grid_fallback_used"])
        self.assertEqual(summary["policy_guard_fallback_rate"], 3 / 95)
        self.assertTrue(summary["shadow_mode"])
        self.assertTrue(summary["canary_mode"])
        self.assertEqual(summary["canary_traffic_fraction"], 0.0)
        self.assertFalse(summary["starts_online_canary"])
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
            "global-99-real-map-shadow-canary-preflight-summary.json",
            "global-99-real-map-shadow-canary-manifest.json",
            "global-99-real-map-shadow-replay-eligibility-audit.json",
            "global-99-real-map-canary-eligibility-audit.json",
            "global-99-real-map-shadow-canary-boundary-audit.json",
            "global-99-real-map-shadow-canary-kill-switch-audit.json",
            "global-99-real-map-shadow-canary-rollback-audit.json",
            "global-99-real-map-shadow-canary-telemetry-audit.json",
            "global-99-real-map-shadow-canary-rejection-report.json",
            "global-99-real-map-shadow-canary-preflight-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_missing_release_governance_routes_to_release_fix(self) -> None:
        from scripts.run_global_99_real_map_shadow_canary_preflight import (
            run_global_99_real_map_shadow_canary_preflight,
        )

        self._write_sources(write_release=False)
        summary = run_global_99_real_map_shadow_canary_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_real_map_release_governance_summary", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_real_map_release_governance_preflight")

    def test_release_governance_failure_routes_to_release_fix(self) -> None:
        from scripts.run_global_99_real_map_shadow_canary_preflight import (
            run_global_99_real_map_shadow_canary_preflight,
        )

        self._write_sources(release_status="failed")
        summary = run_global_99_real_map_shadow_canary_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("real_map_release_governance_not_passed", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_real_map_release_governance_preflight")

    def test_source_match_failure_routes_to_shadow_replay_determinism_fix(self) -> None:
        from scripts.run_global_99_real_map_shadow_canary_preflight import (
            run_global_99_real_map_shadow_canary_preflight,
        )

        self._write_sources(source_match_passed=False, scenario_mismatch_count=1)
        summary = run_global_99_real_map_shadow_canary_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("real_map_shadow_replay_source_match_failed", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_real_map_shadow_replay_determinism")

    def test_fallback_rate_above_threshold_routes_to_guard_fix(self) -> None:
        from scripts.run_global_99_real_map_shadow_canary_preflight import (
            run_global_99_real_map_shadow_canary_preflight,
        )

        self._write_sources(policy_guard_fallback_rate=0.2)
        summary = run_global_99_real_map_shadow_canary_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("policy_guard_fallback_rate_too_high", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_real_map_shadow_canary_guard_fallback")

    def test_boundary_violation_blocks_preflight(self) -> None:
        from scripts.run_global_99_real_map_shadow_canary_preflight import (
            run_global_99_real_map_shadow_canary_preflight,
        )

        self._write_sources(boundary_overrides={"starts_online_canary": True})
        summary = run_global_99_real_map_shadow_canary_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("real_map_shadow_canary_boundary_violation", summary["reason_codes"])
        self.assertEqual(
            summary["next_required_change"],
            "resolve_global_99_real_map_shadow_canary_boundary_rejections",
        )

    def _write_config(self) -> None:
        self.config_path.write_text(
            json.dumps(
                {
                    "schema_version": "global-99-real-map-shadow-canary-preflight-config/v1",
                    "source_real_map_release_governance_root": str(self.release_root),
                    "source_real_map_shadow_replay_root": str(self.shadow_replay_root),
                    "max_policy_guard_fallback_rate": 0.05,
                    "shadow_mode": True,
                    "canary_mode": True,
                    "canary_traffic_fraction": 0.0,
                    "require_default_policy_authoritative": True,
                    "require_release_governance_passed": True,
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
        write_release: bool = True,
        release_status: str = "passed",
        release_next: str = "global_99_real_map_shadow_canary_preflight",
        source_match_passed: bool = True,
        scenario_mismatch_count: int = 0,
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

        if write_release:
            release = {
                "schema_version": "global-99-real-map-release-governance-preflight-summary/v1",
                "status": release_status,
                "reason_codes": [] if release_status == "passed" else ["release_failed"],
                "real_map_release_governance_preflight_passed": release_status == "passed",
                "real_map_release_governance_verdict": "eligible_for_global_99_real_map_shadow_canary_preflight",
                "next_required_change": release_next,
                "source_match_audit_passed": source_match_passed,
                "scenario_mismatch_count": scenario_mismatch_count,
                "open_grid_fallback_used": open_grid_fallback_used,
                "policy_guard_fallback_rate": policy_guard_fallback_rate,
                "release_boundary_audit_passed": not bool(boundary_overrides),
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
            "open_grid_fallback_used": open_grid_fallback_used,
            **boundary,
        }
        (self.shadow_replay_root / "global-99-real-map-shadow-replay-summary.json").write_text(
            json.dumps(shadow, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
