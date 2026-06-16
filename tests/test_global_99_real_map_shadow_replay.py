import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class Global99RealMapShadowReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="global-99-real-map-shadow-replay-"))
        self.preflight_root = self.temp_dir / "preflight"
        self.domain_gap_root = self.temp_dir / "domain-gap"
        self.output_root = self.temp_dir / "output"
        self.config_path = self.temp_dir / "config.json"
        self.sidecar_root = self.domain_gap_root / "path_planner_sidecars"
        for root in (self.preflight_root, self.domain_gap_root, self.sidecar_root):
            root.mkdir(parents=True, exist_ok=True)
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_current_evidence_passes_and_writes_required_artifacts(self) -> None:
        from scripts.run_global_99_real_map_shadow_replay import run_global_99_real_map_shadow_replay

        self._write_sources()
        with patch(
            "scripts.run_global_99_real_map_shadow_replay._run_path_feedback_replay",
            side_effect=self._write_matching_replay_summary,
        ):
            summary = run_global_99_real_map_shadow_replay(
                config_path=self.config_path,
                output_root=self.output_root,
                repo_root=self.repo_root,
            )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertTrue(summary["real_map_shadow_replay_passed"])
        self.assertEqual(
            summary["real_map_shadow_replay_verdict"],
            "eligible_for_global_99_real_map_release_governance_preflight",
        )
        self.assertEqual(summary["next_required_change"], "global_99_real_map_release_governance_preflight")
        self.assertEqual(summary["source_real_map_preflight_status"], "passed")
        self.assertEqual(summary["source_real_map_preflight_next_required_change"], "global_99_real_map_shadow_replay")
        self.assertEqual(summary["source_domain_gap_status"], "passed")
        self.assertEqual(summary["source_path_feedback_scenario_count"], 2)
        self.assertEqual(summary["replay_path_feedback_scenario_count"], 2)
        self.assertEqual(summary["slice_count"], 2)
        self.assertEqual(summary["roi_group_count"], 2)
        self.assertEqual(summary["context_id_missing_count"], 0)
        self.assertEqual(summary["legacy_identity_fallback_count"], 0)
        self.assertTrue(summary["source_match_audit_passed"])
        self.assertEqual(summary["scenario_match_count"], 2)
        self.assertEqual(summary["scenario_mismatch_count"], 0)
        self.assertEqual(summary["max_replay_coverage_delta"], 0.0)
        self.assertEqual(summary["max_replay_path_cost_delta_m"], 0.0)
        self.assertEqual(summary["selection_changed_mismatch_count"], 0)
        self.assertEqual(summary["selected_cell_mismatch_count"], 0)
        self.assertFalse(summary["open_grid_fallback_used"])
        self.assertTrue(summary["uses_lola_quasi_real_roi"])
        self.assertTrue(summary["uses_path_feedback_sidecar"])
        self.assertTrue(summary["uses_offline_path_feedback_replay"])
        self.assertTrue(summary["uses_path_planner"])
        self.assertEqual(summary["path_planner_use_scope"], "offline_path_feedback_replay_only")
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])
        self.assertEqual(summary["canary_traffic_fraction"], 0.0)
        self.assertFalse(summary["replaces_default_policy"])

        for filename in (
            "global-99-real-map-shadow-replay-summary.json",
            "global-99-real-map-shadow-replay-manifest.json",
            "global-99-real-map-shadow-replay-path-feedback-manifest.json",
            "global-99-real-map-shadow-replay-path-feedback-summary.json",
            "global-99-real-map-shadow-replay-scenario-results.jsonl",
            "global-99-real-map-shadow-replay-source-match-audit.json",
            "global-99-real-map-shadow-replay-context-audit.json",
            "global-99-real-map-shadow-replay-boundary-audit.json",
            "global-99-real-map-shadow-replay-kill-switch-audit.json",
            "global-99-real-map-shadow-replay-rollback-audit.json",
            "global-99-real-map-shadow-replay-telemetry-audit.json",
            "global-99-real-map-shadow-replay-rejection-report.json",
            "global-99-real-map-shadow-replay-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_replay_manifest_outputs_are_isolated_from_source_artifacts(self) -> None:
        from scripts.run_global_99_real_map_shadow_replay import run_global_99_real_map_shadow_replay

        self._write_sources()
        with patch(
            "scripts.run_global_99_real_map_shadow_replay._run_path_feedback_replay",
            side_effect=self._write_matching_replay_summary,
        ):
            summary = run_global_99_real_map_shadow_replay(
                config_path=self.config_path,
                output_root=self.output_root,
                repo_root=self.repo_root,
            )

        replay_manifest = json.loads(Path(summary["replay_path_feedback_manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(
            Path(replay_manifest["outputs"]["summary"]),
            self.output_root / "global-99-real-map-shadow-replay-path-feedback-summary.json",
        )
        self.assertEqual(
            Path(replay_manifest["outputs"]["report"]),
            self.output_root / "global-99-real-map-shadow-replay-path-feedback-report.md",
        )
        self.assertNotEqual(
            Path(replay_manifest["outputs"]["summary"]),
            self.domain_gap_root / "quasi-real-map-path-feedback-summary.json",
        )

    def test_missing_preflight_routes_to_preflight_fix(self) -> None:
        from scripts.run_global_99_real_map_shadow_replay import run_global_99_real_map_shadow_replay

        self._write_sources(write_preflight=False)
        summary = run_global_99_real_map_shadow_replay(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_real_map_preflight_summary", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_real_map_preflight")

    def test_missing_source_manifest_routes_to_domain_gap_fix(self) -> None:
        from scripts.run_global_99_real_map_shadow_replay import run_global_99_real_map_shadow_replay

        self._write_sources(write_source_manifest=False)
        summary = run_global_99_real_map_shadow_replay(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_quasi_real_path_feedback_manifest", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_quasi_real_map_domain_gap_evidence")

    def test_replay_command_failure_routes_to_shadow_replay_fix(self) -> None:
        from scripts.run_global_99_real_map_shadow_replay import run_global_99_real_map_shadow_replay

        self._write_sources()
        with patch(
            "scripts.run_global_99_real_map_shadow_replay._run_path_feedback_replay",
            return_value={
                "status": "failed",
                "reason_codes": ["path_feedback_replay_run_failed"],
                "validate_exit_code": 0,
                "run_exit_code": 1,
            },
        ):
            summary = run_global_99_real_map_shadow_replay(
                config_path=self.config_path,
                output_root=self.output_root,
                repo_root=self.repo_root,
            )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("path_feedback_replay_run_failed", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_real_map_shadow_replay")

    def test_source_match_mismatch_routes_to_determinism_fix(self) -> None:
        from scripts.run_global_99_real_map_shadow_replay import run_global_99_real_map_shadow_replay

        self._write_sources()
        with patch(
            "scripts.run_global_99_real_map_shadow_replay._run_path_feedback_replay",
            side_effect=lambda manifest_path, repo_root: self._write_matching_replay_summary(
                manifest_path,
                repo_root,
                coverage_delta=0.01,
                path_delta=0.1,
                selected_cell=[9, 9],
            ),
        ):
            summary = run_global_99_real_map_shadow_replay(
                config_path=self.config_path,
                output_root=self.output_root,
                repo_root=self.repo_root,
            )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("real_map_shadow_replay_source_mismatch", summary["reason_codes"])
        self.assertEqual(summary["selected_cell_mismatch_count"], 1)
        self.assertEqual(summary["next_required_change"], "fix_global_99_real_map_shadow_replay_determinism")

    def test_context_identity_failure_routes_to_context_fix(self) -> None:
        from scripts.run_global_99_real_map_shadow_replay import run_global_99_real_map_shadow_replay

        self._write_sources(context_id_missing=True, legacy_identity_fallback=True)
        summary = run_global_99_real_map_shadow_replay(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("real_map_context_id_missing", summary["reason_codes"])
        self.assertIn("real_map_legacy_identity_fallback_used", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_real_map_context_identity")

    def test_open_grid_or_path_feedback_regression_routes_to_contract_fix(self) -> None:
        from scripts.run_global_99_real_map_shadow_replay import run_global_99_real_map_shadow_replay

        self._write_sources(open_grid=True, safety_regression_count=1)
        summary = run_global_99_real_map_shadow_replay(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("real_map_open_grid_fallback_detected", summary["reason_codes"])
        self.assertIn("real_map_path_feedback_regression", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_real_map_path_feedback_contract")

    def test_boundary_violation_blocks_shadow_replay(self) -> None:
        from scripts.run_global_99_real_map_shadow_replay import run_global_99_real_map_shadow_replay

        self._write_sources(boundary_overrides={"connects_real_executor": True})
        summary = run_global_99_real_map_shadow_replay(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("global_99_real_map_shadow_replay_boundary_violation", summary["reason_codes"])
        self.assertEqual(
            summary["next_required_change"],
            "resolve_global_99_real_map_shadow_replay_boundary_rejections",
        )

    def _write_config(self) -> None:
        self.config_path.write_text(
            json.dumps(
                {
                    "schema_version": "global-99-real-map-shadow-replay-config/v1",
                    "source_real_map_preflight_root": str(self.preflight_root),
                    "source_quasi_real_domain_gap_root": str(self.domain_gap_root),
                    "target_coverage_rate": 0.99,
                    "source_match_coverage_tolerance": 1e-12,
                    "source_match_path_cost_tolerance_m": 1e-9,
                    "require_preflight_passed": True,
                    "require_context_ids": True,
                    "require_no_open_grid_fallback": True,
                    "offline_path_feedback_replay": True,
                    "allow_offline_path_planner_route_replay": True,
                    "shadow_replay_mode": True,
                    "canary_traffic_fraction": 0.0,
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
        write_source_manifest: bool = True,
        context_id_missing: bool = False,
        legacy_identity_fallback: bool = False,
        open_grid: bool = False,
        safety_regression_count: int = 0,
        boundary_overrides: dict | None = None,
    ) -> None:
        boundary = self._boundary(boundary_overrides)
        if write_preflight:
            self._write_json(
                self.preflight_root / "global-99-real-map-preflight-summary.json",
                {
                    "schema_version": "global-99-real-map-preflight-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "next_required_change": "global_99_real_map_shadow_replay",
                    "real_map_preflight_verdict": "eligible_for_global_99_real_map_shadow_replay",
                    "source_domain_gap_status": "passed",
                    **boundary,
                },
            )
        self._write_json(
            self.domain_gap_root / "quasi-real-map-domain-gap-summary.json",
            {
                "schema_version": "quasi-real-map-domain-gap-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "domain_gap_verdict": "acceptable_for_next_pilot",
                "slice_count": 2,
                "roi_group_count": 2,
                "context_id_missing_count": 1 if context_id_missing else 0,
                "legacy_identity_fallback_count": 1 if legacy_identity_fallback else 0,
                "fallback_or_open_grid_count": 1 if open_grid else 0,
                "safety_regression_count": safety_regression_count,
                "contract_violation_count": 0,
                "path_cost_regression_count": 0,
                "risk_regression_count": 0,
                "source_selection_regression_count": 0,
                **boundary,
            },
        )
        self._write_jsonl(
            self.domain_gap_root / "quasi-real-map-slices.jsonl",
            [
                self._slice("scenario-a", "smooth", None if context_id_missing else "ctx-a", legacy_identity_fallback),
                self._slice("scenario-b", "mixed", "ctx-b", False),
            ],
        )
        self._write_source_path_feedback_summary(open_grid=open_grid, safety_regression_count=safety_regression_count)
        if write_source_manifest:
            self._write_source_manifest()

    def _write_source_manifest(self) -> None:
        self._write_json(
            self.domain_gap_root / "quasi-real-map-path-feedback-manifest.json",
            {
                "schema_version": "path-feedback-manifest/v1",
                "scenario_set": "quasi_real_map_domain_gap",
                "diagnostic_profile": "execution",
                "acceptance_gate": "quasi-real-map-domain-gap",
                "top_k": 3,
                "planner": {
                    "backend": "path_planner_route",
                    "path_planner_root": str(self.repo_root / "path-planner"),
                    "python_executable": sys.executable,
                    "extra_args": ["--planning-backend", "channel_aware_astar"],
                },
                "scenarios": [
                    self._manifest_scenario("scenario-a", "smooth"),
                    self._manifest_scenario("scenario-b", "mixed"),
                ],
                "outputs": {
                    "summary": str(self.domain_gap_root / "quasi-real-map-path-feedback-summary.json"),
                    "report": str(self.domain_gap_root / "quasi-real-map-path-feedback-report.md"),
                },
            },
        )

    def _write_source_path_feedback_summary(self, *, open_grid: bool = False, safety_regression_count: int = 0) -> None:
        self._write_json(
            self.domain_gap_root / "quasi-real-map-path-feedback-summary.json",
            {
                "schema_version": "path-feedback-summary/v1",
                "scenario_count": 2,
                "open_grid_fallback_used": open_grid,
                "tracking_safety_violation_count": safety_regression_count,
                "trajectory_optimization_fallback_count": 0,
                "region_graph_disconnected_count": 0,
                "path_cost_regression_count": 0,
                "risk_regression_count": 0,
                "source_selection_regression_count": 0,
                "scenarios": [
                    self._scenario("scenario-a", "smooth", [1, 2], 12.5, 0.25, True, open_grid, safety_regression_count),
                    self._scenario("scenario-b", "mixed", [2, 3], 14.0, 0.5, False, False, 0),
                ],
            },
        )

    def _write_matching_replay_summary(
        self,
        manifest_path: Path,
        repo_root: Path,
        *,
        coverage_delta: float = 0.0,
        path_delta: float = 0.0,
        selected_cell: list[int] | None = None,
    ) -> dict:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        summary_path = Path(manifest["outputs"]["summary"])
        source = json.loads((self.domain_gap_root / "quasi-real-map-path-feedback-summary.json").read_text(encoding="utf-8"))
        replay = dict(source)
        replay["scenarios"] = [dict(row) for row in source["scenarios"]]
        if coverage_delta or path_delta or selected_cell is not None:
            replay["scenarios"][0]["coverage_rate_delta"] += coverage_delta
            replay["scenarios"][0]["selected_path_cost_after_feedback"] += path_delta
            if selected_cell is not None:
                replay["scenarios"][0]["selected_cell_after_path_feedback"] = selected_cell
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(replay, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        Path(manifest["outputs"]["report"]).write_text("# replay report\n", encoding="utf-8")
        return {"status": "passed", "reason_codes": [], "validate_exit_code": 0, "run_exit_code": 0}

    def _manifest_scenario(self, scenario_id: str, group: str) -> dict:
        contract = self.sidecar_root / f"{scenario_id}.contract.json"
        sidecar = self.sidecar_root / f"{scenario_id}.path-planner-sidecar.json"
        self._write_json(contract, {"schema_version": "model-explorer-contract/v1", "metadata": {"scenario_id": scenario_id}})
        self._write_json(sidecar, {"schema_version": "path-planner-sidecar/v1", "metadata": {"scenario_id": scenario_id}})
        return {
            "scenario_id": scenario_id,
            "scenario_group": group,
            "scenario_seed": 1,
            "scenario_variant_id": f"{scenario_id}-variant",
            "contract": str(contract),
            "sidecar": str(sidecar),
            "current_cell": [0, 0],
        }

    def _slice(self, scenario_id: str, group: str, context_id: str | None, legacy_fallback: bool) -> dict:
        return {
            "schema_version": "quasi-real-map-slice/v1",
            "scenario_id": scenario_id,
            "scenario_group": group,
            "roi_name": group,
            "context_id": context_id,
            "legacy_identity_fallback_used": legacy_fallback,
        }

    def _scenario(
        self,
        scenario_id: str,
        group: str,
        selected_cell: list[int],
        path_cost: float,
        coverage_delta: float,
        selection_changed: bool,
        open_grid: bool,
        safety_regression_count: int,
    ) -> dict:
        return {
            "scenario_id": scenario_id,
            "scenario_group": group,
            "selected_cell_after_path_feedback": selected_cell,
            "selection_changed_by_path_feedback": selection_changed,
            "selected_path_cost_after_feedback": path_cost,
            "coverage_rate_delta": coverage_delta,
            "open_grid_fallback_used": open_grid,
            "tracking_safety_violation_count": safety_regression_count,
            "contract_violation_count": 0,
            "path_cost_regression_count": 0,
            "risk_regression_count": 0,
            "source_selection_regression_count": 0,
        }

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
        }
        if overrides:
            boundary.update(overrides)
        return boundary

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
