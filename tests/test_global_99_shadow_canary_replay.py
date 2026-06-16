import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class Global99ShadowCanaryReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="global-99-shadow-replay-"))
        self.preflight_root = self.temp_dir / "preflight"
        self.multi_map_root = self.temp_dir / "multi-map"
        self.output_root = self.temp_dir / "output"
        self.config_path = self.temp_dir / "config.json"
        for root in (self.preflight_root, self.multi_map_root):
            root.mkdir(parents=True, exist_ok=True)
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_current_evidence_replay_passes_and_writes_required_artifacts(self) -> None:
        from scripts.run_global_99_shadow_canary_replay import run_global_99_shadow_canary_replay

        self._write_sources()
        with patch(
            "scripts.run_global_99_shadow_canary_replay._run_replay_matrix",
            return_value=self._replay_payload(),
        ):
            summary = run_global_99_shadow_canary_replay(
                config_path=self.config_path,
                output_root=self.output_root,
                repo_root=self.repo_root,
            )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertTrue(summary["shadow_replay_passed"])
        self.assertTrue(summary["offline_canary_replay_passed"])
        self.assertTrue(summary["source_match_audit_passed"])
        self.assertEqual(summary["next_required_change"], "global_99_real_map_preflight")
        self.assertEqual(summary["source_shadow_canary_preflight_status"], "passed")
        self.assertEqual(
            summary["source_shadow_canary_preflight_verdict"],
            "eligible_for_global_99_shadow_canary_replay",
        )
        self.assertEqual(summary["scenario_count"], 2)
        self.assertEqual(summary["required_scenario_count"], 1)
        self.assertEqual(summary["failed_required_scenario_count"], 0)
        self.assertEqual(summary["policy_guard_fallback_rate"], 1 / 40)
        self.assertEqual(summary["max_replay_coverage_delta"], 0.0)
        self.assertEqual(summary["max_replay_path_cost_delta_m"], 0.0)
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])
        self.assertEqual(summary["canary_traffic_fraction"], 0.0)

        for filename in (
            "global-99-shadow-canary-replay-summary.json",
            "global-99-shadow-canary-replay-manifest.json",
            "global-99-shadow-canary-replay-scenario-results.jsonl",
            "global-99-shadow-canary-replay-family-summary.json",
            "global-99-shadow-canary-replay-source-match-audit.json",
            "global-99-shadow-canary-replay-policy-vs-baseline-audit.json",
            "global-99-shadow-canary-replay-boundary-audit.json",
            "global-99-shadow-canary-replay-kill-switch-audit.json",
            "global-99-shadow-canary-replay-rollback-audit.json",
            "global-99-shadow-canary-replay-telemetry-audit.json",
            "global-99-shadow-canary-replay-rejection-report.json",
            "global-99-shadow-canary-replay-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_missing_preflight_routes_to_preflight_fix(self) -> None:
        from scripts.run_global_99_shadow_canary_replay import run_global_99_shadow_canary_replay

        self._write_sources(write_preflight=False)
        with patch(
            "scripts.run_global_99_shadow_canary_replay._run_replay_matrix",
            return_value=self._replay_payload(),
        ):
            summary = run_global_99_shadow_canary_replay(
                config_path=self.config_path,
                output_root=self.output_root,
                repo_root=self.repo_root,
            )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_shadow_canary_preflight_summary", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_shadow_canary_preflight")

    def test_missing_multi_map_evidence_routes_to_multi_map_fix(self) -> None:
        from scripts.run_global_99_shadow_canary_replay import run_global_99_shadow_canary_replay

        self._write_sources(write_scenario_results=False)
        with patch(
            "scripts.run_global_99_shadow_canary_replay._run_replay_matrix",
            return_value=self._replay_payload(),
        ):
            summary = run_global_99_shadow_canary_replay(
                config_path=self.config_path,
                output_root=self.output_root,
                repo_root=self.repo_root,
            )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_multi_map_scenario_results", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_multi_map_generalization")

    def test_source_match_mismatch_routes_to_determinism_fix(self) -> None:
        from scripts.run_global_99_shadow_canary_replay import run_global_99_shadow_canary_replay

        self._write_sources(source_coverage=0.98)
        with patch(
            "scripts.run_global_99_shadow_canary_replay._run_replay_matrix",
            return_value=self._replay_payload(),
        ):
            summary = run_global_99_shadow_canary_replay(
                config_path=self.config_path,
                output_root=self.output_root,
                repo_root=self.repo_root,
            )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("shadow_replay_source_mismatch", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_shadow_canary_replay_determinism")

    def test_required_scenario_failure_routes_to_replay_fix(self) -> None:
        from scripts.run_global_99_shadow_canary_replay import run_global_99_shadow_canary_replay

        self._write_sources()
        replay_payload = self._replay_payload(required_coverage=0.98, required_status="failed")
        with patch(
            "scripts.run_global_99_shadow_canary_replay._run_replay_matrix",
            return_value=replay_payload,
        ):
            summary = run_global_99_shadow_canary_replay(
                config_path=self.config_path,
                output_root=self.output_root,
                repo_root=self.repo_root,
            )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("shadow_replay_required_scenarios_not_passed", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_shadow_canary_replay")

    def test_policy_regression_routes_to_policy_fix(self) -> None:
        from scripts.run_global_99_shadow_canary_replay import run_global_99_shadow_canary_replay

        self._write_sources()
        replay_payload = self._replay_payload(policy_worse=1, controlled_regression=1)
        with patch(
            "scripts.run_global_99_shadow_canary_replay._run_replay_matrix",
            return_value=replay_payload,
        ):
            summary = run_global_99_shadow_canary_replay(
                config_path=self.config_path,
                output_root=self.output_root,
                repo_root=self.repo_root,
            )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("policy_regression_detected", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_policy_guided_global_coverage")

    def test_fallback_rate_above_threshold_routes_to_guard_fix(self) -> None:
        from scripts.run_global_99_shadow_canary_replay import run_global_99_shadow_canary_replay

        self._write_sources()
        replay_payload = self._replay_payload(policy_guard_fallback_count=2, policy_guided_decision_count=4)
        with patch(
            "scripts.run_global_99_shadow_canary_replay._run_replay_matrix",
            return_value=replay_payload,
        ):
            summary = run_global_99_shadow_canary_replay(
                config_path=self.config_path,
                output_root=self.output_root,
                repo_root=self.repo_root,
            )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("policy_guard_fallback_rate_too_high", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_shadow_canary_guard_fallback")

    def test_boundary_violation_blocks_replay(self) -> None:
        from scripts.run_global_99_shadow_canary_replay import run_global_99_shadow_canary_replay

        self._write_sources(boundary_overrides={"starts_online_canary": True})
        with patch(
            "scripts.run_global_99_shadow_canary_replay._run_replay_matrix",
            return_value=self._replay_payload(),
        ):
            summary = run_global_99_shadow_canary_replay(
                config_path=self.config_path,
                output_root=self.output_root,
                repo_root=self.repo_root,
            )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("shadow_canary_replay_boundary_violation", summary["reason_codes"])
        self.assertFalse(summary["boundary_audit_passed"])
        self.assertEqual(summary["next_required_change"], "resolve_global_99_shadow_canary_replay_boundary_rejections")

    def _write_config(self, *, canary_traffic_fraction: float = 0.0) -> None:
        self.config_path.write_text(
            json.dumps(
                {
                    "schema_version": "global-99-shadow-canary-replay-config/v1",
                    "source_shadow_canary_preflight_root": str(self.preflight_root),
                    "source_multi_map_root": str(self.multi_map_root),
                    "source_policy_guided_config": "configs/policy_guided_global_coverage_v1.json",
                    "source_multi_map_config": "configs/global_99_multi_map_generalization_v1.json",
                    "target_coverage_rate": 0.99,
                    "max_policy_guard_fallback_rate": 0.05,
                    "replay_coverage_tolerance": 1e-12,
                    "replay_path_cost_tolerance_m": 1e-9,
                    "shadow_replay_mode": True,
                    "offline_canary_replay_mode": True,
                    "canary_traffic_fraction": canary_traffic_fraction,
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
        write_preflight: bool = True,
        write_scenario_results: bool = True,
        source_coverage: float = 0.995,
        boundary_overrides: dict | None = None,
    ) -> None:
        boundary = self._boundary(boundary_overrides)
        if write_preflight:
            self._write_json(
                self.preflight_root / "global-99-shadow-canary-preflight-summary.json",
                {
                    "schema_version": "global-99-shadow-canary-preflight-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "shadow_canary_preflight_verdict": "eligible_for_global_99_shadow_canary_replay",
                    "next_required_change": "global_99_shadow_canary_replay",
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
                "scenario_count": 2,
                "required_scenario_count": 1,
                "failed_required_scenario_count": 0,
                "aggregate_achieved_coverage_rate": source_coverage,
                "min_scenario_achieved_coverage_rate": source_coverage,
                "policy_guidance_applied": True,
                "policy_read_only": True,
                "policy_worse_than_baseline_count": 0,
                "controlled_regression_count": 0,
                "next_required_change": "network_architecture_upgrade_readiness_review",
                **boundary,
            },
        )
        self._write_json(
            self.multi_map_root / "global-99-multi-map-family-summary.json",
            {
                "schema_version": "global-99-multi-map-family-summary/v1",
                "family_count": 2,
                "passed_family_count": 2,
                "failed_family_count": 0,
                "families": {},
            },
        )
        self._write_json(
            self.multi_map_root / "global-99-multi-map-policy-vs-baseline-audit.json",
            {
                "schema_version": "global-99-multi-map-policy-vs-baseline-audit/v1",
                "policy_loaded": True,
                "policy_guidance_applied": True,
                "policy_scored_candidate_count": 8,
                "policy_guided_decision_count": 40,
                "policy_guard_fallback_count": 1,
                "policy_better_than_baseline_count": 1,
                "policy_worse_than_baseline_count": 0,
                "controlled_regression_count": 0,
            },
        )
        if write_scenario_results:
            self._write_jsonl(
                self.multi_map_root / "global-99-multi-map-scenario-results.jsonl",
                [
                    self._scenario("required-01", source_coverage, True, "passed", 1000.0),
                    self._scenario("budget-01", 0.5, False, "failed", 100.0, expected_infeasible=True),
                ],
            )

    def _replay_payload(
        self,
        *,
        required_coverage: float = 0.995,
        required_status: str = "passed",
        policy_guard_fallback_count: int = 1,
        policy_guided_decision_count: int = 40,
        policy_worse: int = 0,
        controlled_regression: int = 0,
    ) -> dict:
        scenario_results = [
            self._scenario("required-01", required_coverage, True, required_status, 1000.0),
            self._scenario("budget-01", 0.5, False, "failed", 100.0, expected_infeasible=True),
        ]
        return {
            "scenario_results": scenario_results,
            "family_summary": {
                "schema_version": "global-99-shadow-canary-replay-family-summary/v1",
                "family_count": 2,
                "passed_family_count": 2,
                "failed_family_count": 0,
                "families": {},
            },
            "policy_audit": {
                "schema_version": "global-99-shadow-canary-replay-policy-vs-baseline-audit/v1",
                "policy_loaded": True,
                "policy_guidance_applied": True,
                "policy_scored_candidate_count": 8,
                "policy_guided_decision_count": policy_guided_decision_count,
                "policy_guard_fallback_count": policy_guard_fallback_count,
                "policy_better_than_baseline_count": 1,
                "policy_worse_than_baseline_count": policy_worse,
                "controlled_regression_count": controlled_regression,
            },
        }

    def _scenario(
        self,
        scenario_id: str,
        coverage: float,
        required: bool,
        status: str,
        path_cost: float,
        *,
        expected_infeasible: bool = False,
    ) -> dict:
        reachable = 1000
        return {
            "schema_version": "global-99-shadow-canary-replay-scenario-result/v1",
            "scenario_id": scenario_id,
            "family_id": "open_field" if required else "budget_limited",
            "status": status,
            "reason_codes": [] if status == "passed" else ["coverage_target_not_met"],
            "infeasible_reason_codes": [] if required else ["insufficient_budget"],
            "generalization_required": required,
            "expected_infeasible": expected_infeasible,
            "target_coverage_rate": 0.99,
            "achieved_coverage_rate": coverage,
            "coverage_target_met": coverage >= 0.99,
            "reachable_safe_cell_count": reachable,
            "covered_reachable_safe_cell_count": int(reachable * coverage),
            "policy_scored_candidate_count": 4,
            "policy_guided_decision_count": 2,
            "policy_guard_fallback_count": 0,
            "baseline_agreement_count": 1,
            "baseline_agreement_denominator": 2,
            "controlled_regression_count": 0,
            "planned_path_cost_m": path_cost,
            "path_budget_m": 5000.0,
            "path_budget_exhausted": False,
        }

    def _boundary(self, overrides: dict | None = None) -> dict:
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
        if overrides:
            boundary.update(overrides)
        return boundary

    def _write_json(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
