import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class Global99ShadowCanaryPreflightTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="global-99-shadow-canary-"))
        self.release_root = self.temp_dir / "release-governance"
        self.multi_map_root = self.temp_dir / "multi-map"
        self.policy_root = self.temp_dir / "policy-guided"
        self.output_root = self.temp_dir / "output"
        self.config_path = self.temp_dir / "config.json"
        for root in (self.release_root, self.multi_map_root, self.policy_root):
            root.mkdir(parents=True, exist_ok=True)
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_current_release_governance_evidence_passes_and_writes_artifacts(self) -> None:
        from scripts.run_global_99_shadow_canary_preflight import (
            run_global_99_shadow_canary_preflight,
        )

        self._write_sources()
        summary = run_global_99_shadow_canary_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertTrue(summary["shadow_preflight_passed"])
        self.assertTrue(summary["canary_preflight_passed"])
        self.assertEqual(summary["shadow_canary_preflight_verdict"], "eligible_for_global_99_shadow_canary_replay")
        self.assertEqual(summary["next_required_change"], "global_99_shadow_canary_replay")
        self.assertTrue(summary["shadow_replay_audit_passed"])
        self.assertTrue(summary["canary_eligibility_audit_passed"])
        self.assertTrue(summary["boundary_audit_passed"])
        self.assertTrue(summary["kill_switch_audit_passed"])
        self.assertTrue(summary["rollback_audit_passed"])
        self.assertTrue(summary["telemetry_audit_passed"])
        self.assertEqual(summary["source_release_governance_status"], "passed")
        self.assertEqual(summary["release_governance_verdict"], "eligible_for_global_99_shadow_canary_preflight")
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
        self.assertFalse(summary["starts_online_canary"])
        self.assertEqual(summary["canary_traffic_fraction"], 0.0)

        for filename in (
            "global-99-shadow-canary-preflight-summary.json",
            "global-99-shadow-canary-manifest.json",
            "global-99-shadow-replay-audit.json",
            "global-99-canary-eligibility-audit.json",
            "global-99-shadow-canary-boundary-audit.json",
            "global-99-shadow-canary-kill-switch-audit.json",
            "global-99-shadow-canary-rollback-audit.json",
            "global-99-shadow-canary-telemetry-audit.json",
            "global-99-shadow-canary-rejection-report.json",
            "global-99-shadow-canary-preflight-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_missing_release_governance_fails(self) -> None:
        from scripts.run_global_99_shadow_canary_preflight import (
            run_global_99_shadow_canary_preflight,
        )

        self._write_sources(write_release=False)
        summary = run_global_99_shadow_canary_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_release_governance_summary", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_release_governance_preflight")

    def test_failed_release_governance_fails(self) -> None:
        from scripts.run_global_99_shadow_canary_preflight import (
            run_global_99_shadow_canary_preflight,
        )

        self._write_sources(release_status="failed")
        summary = run_global_99_shadow_canary_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("release_governance_not_passed", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_release_governance_preflight")

    def test_multi_map_coverage_failure_routes_to_multi_map_fix(self) -> None:
        from scripts.run_global_99_shadow_canary_preflight import (
            run_global_99_shadow_canary_preflight,
        )

        self._write_sources(failed_required_scenario_count=1, aggregate_coverage=0.98, min_coverage=0.97)
        summary = run_global_99_shadow_canary_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("multi_map_required_scenarios_not_passed", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_multi_map_generalization")

    def test_policy_regression_routes_to_policy_guided_fix(self) -> None:
        from scripts.run_global_99_shadow_canary_preflight import (
            run_global_99_shadow_canary_preflight,
        )

        self._write_sources(policy_worse=1, controlled_regression=1)
        summary = run_global_99_shadow_canary_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("policy_regression_detected", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_policy_guided_global_coverage")

    def test_fallback_rate_above_threshold_routes_to_guard_fallback_fix(self) -> None:
        from scripts.run_global_99_shadow_canary_preflight import (
            run_global_99_shadow_canary_preflight,
        )

        self._write_sources(policy_guard_fallback_rate=0.20)
        summary = run_global_99_shadow_canary_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("policy_guard_fallback_rate_too_high", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_shadow_canary_guard_fallback")

    def test_shadow_canary_boundary_violation_blocks_preflight(self) -> None:
        from scripts.run_global_99_shadow_canary_preflight import (
            run_global_99_shadow_canary_preflight,
        )

        self._write_sources(boundary_overrides={"starts_online_canary": True})
        summary = run_global_99_shadow_canary_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("shadow_canary_boundary_violation", summary["reason_codes"])
        self.assertFalse(summary["boundary_audit_passed"])
        self.assertEqual(summary["next_required_change"], "resolve_global_99_shadow_canary_boundary_rejections")

    def test_nonzero_canary_traffic_fraction_blocks_preflight(self) -> None:
        from scripts.run_global_99_shadow_canary_preflight import (
            run_global_99_shadow_canary_preflight,
        )

        self._write_config(canary_traffic_fraction=0.01)
        self._write_sources()
        summary = run_global_99_shadow_canary_preflight(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("shadow_canary_boundary_violation", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "resolve_global_99_shadow_canary_boundary_rejections")

    def _write_config(self, *, canary_traffic_fraction: float = 0.0) -> None:
        self.config_path.write_text(
            json.dumps(
                {
                    "schema_version": "global-99-shadow-canary-preflight-config/v1",
                    "source_release_governance_root": str(self.release_root),
                    "source_multi_map_root": str(self.multi_map_root),
                    "source_policy_guided_root": str(self.policy_root),
                    "target_coverage_rate": 0.99,
                    "max_policy_guard_fallback_rate": 0.05,
                    "shadow_mode": True,
                    "canary_mode": True,
                    "canary_traffic_fraction": canary_traffic_fraction,
                    "require_default_policy_authoritative": True,
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
        write_release: bool = True,
        release_status: str = "passed",
        failed_required_scenario_count: int = 0,
        aggregate_coverage: float = 0.9963513964901461,
        min_coverage: float = 0.9925,
        policy_guard_fallback_rate: float = 3 / 95,
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
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
        if boundary_overrides:
            boundary.update(boundary_overrides)
        if write_release:
            self._write_json(
                self.release_root / "global-99-release-governance-preflight-summary.json",
                {
                    "schema_version": "global-99-release-governance-preflight-summary/v1",
                    "status": release_status,
                    "reason_codes": [] if release_status == "passed" else ["release_boundary_violation"],
                    "source_network_readiness_status": "passed",
                    "source_multi_map_status": "passed",
                    "source_policy_guided_status": "passed",
                    "target_coverage_rate": 0.99,
                    "aggregate_achieved_coverage_rate": aggregate_coverage,
                    "min_scenario_achieved_coverage_rate": min_coverage,
                    "required_scenario_count": 21,
                    "failed_required_scenario_count": failed_required_scenario_count,
                    "policy_guidance_applied": True,
                    "policy_guard_fallback_rate": policy_guard_fallback_rate,
                    "policy_better_than_baseline_count": 14,
                    "policy_worse_than_baseline_count": policy_worse,
                    "controlled_regression_count": controlled_regression,
                    "release_governance_preflight_passed": release_status == "passed",
                    "release_governance_verdict": "eligible_for_global_99_shadow_canary_preflight"
                    if release_status == "passed"
                    else "resolve_global_99_release_governance_preflight_rejections",
                    "next_required_change": "global_99_shadow_canary_preflight"
                    if release_status == "passed"
                    else "resolve_global_99_release_governance_preflight_rejections",
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
                "policy_guard_fallback_count": int(round(policy_guard_fallback_rate * 95)),
                "policy_guard_fallback_rate": policy_guard_fallback_rate,
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
