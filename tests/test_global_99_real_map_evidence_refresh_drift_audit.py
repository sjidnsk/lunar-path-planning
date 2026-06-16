import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class Global99RealMapEvidenceRefreshDriftAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="global-99-real-map-drift-audit-"))
        self.shadow_canary_root = self.temp_dir / "shadow-canary-replay"
        self.domain_gap_root = self.temp_dir / "domain-gap"
        self.shadow_replay_root = self.temp_dir / "shadow-replay"
        self.output_root = self.temp_dir / "output"
        self.config_path = self.temp_dir / "config.json"
        for root in (self.shadow_canary_root, self.domain_gap_root, self.shadow_replay_root):
            root.mkdir(parents=True, exist_ok=True)
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_current_evidence_passes_and_writes_required_artifacts(self) -> None:
        from scripts.run_global_99_real_map_evidence_refresh_drift_audit import run_global_99_real_map_evidence_refresh_drift_audit

        self._write_sources()
        summary = run_global_99_real_map_evidence_refresh_drift_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertTrue(summary["evidence_refresh_drift_audit_passed"])
        self.assertEqual(summary["evidence_refresh_drift_verdict"], "eligible_for_global_99_real_map_multi_roi_generalization")
        self.assertEqual(summary["next_required_change"], "global_99_real_map_multi_roi_generalization")
        self.assertEqual(summary["source_real_map_shadow_canary_replay_status"], "passed")
        self.assertEqual(summary["source_real_map_shadow_canary_replay_next_required_change"], "global_99_real_map_evidence_refresh_drift_audit")
        self.assertEqual(summary["source_domain_gap_status"], "passed")
        self.assertEqual(summary["domain_gap_verdict"], "acceptable_for_next_pilot")
        self.assertEqual(summary["slice_count"], 12)
        self.assertEqual(summary["roi_group_count"], 4)
        self.assertEqual(summary["manifest_scenario_count"], 12)
        self.assertEqual(summary["scenario_id_mismatch_count"], 0)
        self.assertEqual(summary["missing_contract_count"], 0)
        self.assertEqual(summary["missing_sidecar_count"], 0)
        self.assertEqual(summary["context_id_missing_count"], 0)
        self.assertEqual(summary["legacy_identity_fallback_count"], 0)
        self.assertEqual(summary["fallback_or_open_grid_count"], 0)
        self.assertTrue(summary["lineage_audit_passed"])
        self.assertTrue(summary["fingerprint_audit_passed"])
        self.assertTrue(summary["manifest_sidecar_audit_passed"])
        self.assertTrue(summary["context_audit_passed"])
        self.assertTrue(summary["source_match_drift_audit_passed"])
        self.assertTrue(summary["boundary_audit_passed"])
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
            "global-99-real-map-evidence-refresh-drift-audit-summary.json",
            "global-99-real-map-evidence-refresh-drift-audit-manifest.json",
            "global-99-real-map-evidence-lineage-audit.json",
            "global-99-real-map-evidence-fingerprint-audit.json",
            "global-99-real-map-manifest-sidecar-audit.json",
            "global-99-real-map-context-audit.json",
            "global-99-real-map-source-match-drift-audit.json",
            "global-99-real-map-evidence-refresh-drift-boundary-audit.json",
            "global-99-real-map-evidence-refresh-drift-kill-switch-audit.json",
            "global-99-real-map-evidence-refresh-drift-rollback-audit.json",
            "global-99-real-map-evidence-refresh-drift-telemetry-audit.json",
            "global-99-real-map-evidence-refresh-drift-rejection-report.json",
            "global-99-real-map-evidence-refresh-drift-audit-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_missing_shadow_canary_replay_routes_to_shadow_replay_fix(self) -> None:
        from scripts.run_global_99_real_map_evidence_refresh_drift_audit import run_global_99_real_map_evidence_refresh_drift_audit

        self._write_sources(write_shadow_canary=False)
        summary = run_global_99_real_map_evidence_refresh_drift_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_real_map_shadow_canary_replay_summary", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_real_map_shadow_canary_replay")

    def test_domain_gap_failure_routes_to_domain_gap_fix(self) -> None:
        from scripts.run_global_99_real_map_evidence_refresh_drift_audit import run_global_99_real_map_evidence_refresh_drift_audit

        self._write_sources(domain_gap_status="failed", domain_gap_verdict="blocked")
        summary = run_global_99_real_map_evidence_refresh_drift_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("domain_gap_evidence_not_acceptable", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_quasi_real_map_domain_gap_evidence")

    def test_manifest_slice_drift_routes_to_drift_fix(self) -> None:
        from scripts.run_global_99_real_map_evidence_refresh_drift_audit import run_global_99_real_map_evidence_refresh_drift_audit

        self._write_sources(manifest_scenario_count=11)
        summary = run_global_99_real_map_evidence_refresh_drift_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("manifest_slice_scenario_id_drift", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_real_map_evidence_drift")

    def test_context_identity_failure_routes_to_context_fix(self) -> None:
        from scripts.run_global_99_real_map_evidence_refresh_drift_audit import run_global_99_real_map_evidence_refresh_drift_audit

        self._write_sources(missing_context=True)
        summary = run_global_99_real_map_evidence_refresh_drift_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("real_map_context_identity_incomplete", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_real_map_context_identity")

    def test_missing_sidecar_or_open_grid_routes_to_path_feedback_fix(self) -> None:
        from scripts.run_global_99_real_map_evidence_refresh_drift_audit import run_global_99_real_map_evidence_refresh_drift_audit

        self._write_sources(missing_sidecar=True, fallback_or_open_grid_count=1)
        summary = run_global_99_real_map_evidence_refresh_drift_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("real_map_path_feedback_contract_incomplete", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_real_map_path_feedback_contract")

    def test_boundary_violation_blocks_drift_audit(self) -> None:
        from scripts.run_global_99_real_map_evidence_refresh_drift_audit import run_global_99_real_map_evidence_refresh_drift_audit

        self._write_sources(boundary_overrides={"connects_real_executor": True})
        summary = run_global_99_real_map_evidence_refresh_drift_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("real_map_evidence_refresh_boundary_violation", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "resolve_global_99_real_map_evidence_refresh_boundary_rejections")

    def _write_config(self) -> None:
        self.config_path.write_text(
            json.dumps(
                {
                    "schema_version": "global-99-real-map-evidence-refresh-drift-audit-config/v1",
                    "source_real_map_shadow_canary_replay_root": str(self.shadow_canary_root),
                    "source_quasi_real_domain_gap_root": str(self.domain_gap_root),
                    "source_real_map_shadow_replay_root": str(self.shadow_replay_root),
                    "min_slice_count": 12,
                    "min_roi_group_count": 4,
                    "source_match_coverage_tolerance": 1e-12,
                    "source_match_path_cost_tolerance_m": 1e-9,
                    "max_policy_guard_fallback_rate": 0.05,
                    "require_shadow_canary_replay_passed": True,
                    "require_domain_gap_acceptable": True,
                    "require_context_ids": True,
                    "require_no_legacy_identity_fallback": True,
                    "require_no_open_grid_fallback": True,
                    "require_source_match_audit_passed": True,
                    "require_all_contract_and_sidecar_paths": True,
                    "evidence_refresh_only": True,
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
        write_shadow_canary: bool = True,
        domain_gap_status: str = "passed",
        domain_gap_verdict: str = "acceptable_for_next_pilot",
        manifest_scenario_count: int = 12,
        missing_context: bool = False,
        missing_sidecar: bool = False,
        fallback_or_open_grid_count: int = 0,
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
            "uses_path_planner": False,
            "real_world_release_approved": False,
            "real_world_performance_claimed": False,
        }
        if boundary_overrides:
            boundary.update(boundary_overrides)

        if write_shadow_canary:
            self._write_json(
                self.shadow_canary_root / "global-99-real-map-shadow-canary-replay-summary.json",
                {
                    "schema_version": "global-99-real-map-shadow-canary-replay-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "next_required_change": "global_99_real_map_evidence_refresh_drift_audit",
                    "source_match_audit_passed": True,
                    "scenario_mismatch_count": 0,
                    "max_replay_coverage_delta": 0.0,
                    "max_replay_path_cost_delta_m": 0.0,
                    "open_grid_fallback_used": False,
                    "policy_guard_fallback_rate": 3 / 95,
                    **boundary,
                },
            )

        self._write_json(
            self.shadow_replay_root / "global-99-real-map-shadow-replay-summary.json",
            {
                "schema_version": "global-99-real-map-shadow-replay-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "next_required_change": "global_99_real_map_release_governance_preflight",
                "source_match_audit_passed": True,
                "scenario_mismatch_count": 0,
                "max_replay_coverage_delta": 0.0,
                "max_replay_path_cost_delta_m": 0.0,
                **boundary,
                "uses_path_planner": True,
                "path_planner_use_scope": "offline_path_feedback_replay_only",
            },
        )

        sidecar_root = self.domain_gap_root / "path_planner_sidecars"
        sidecar_root.mkdir(parents=True, exist_ok=True)
        slices = []
        manifest_scenarios = []
        for index in range(12):
            scenario_id = f"slice_{index:03d}"
            contract = sidecar_root / f"{scenario_id}.contract.json"
            sidecar = sidecar_root / f"{scenario_id}.path-planner-sidecar.json"
            contract.write_text(json.dumps({"scenario_id": scenario_id}), encoding="utf-8")
            if not (missing_sidecar and index == 0):
                sidecar.write_text(json.dumps({"scenario_id": scenario_id}), encoding="utf-8")
            slice_row = {
                "schema_version": "quasi-real-map-slice/v1",
                "scenario_id": scenario_id,
                "scenario_group": f"group_{index % 4}",
                "roi_name": f"group_{index % 4}",
                "context_id": "" if missing_context and index == 0 else f"context-{index}",
                "legacy_identity_fallback_used": bool(missing_context and index == 1),
                "contract": str(contract),
                "sidecar": str(sidecar),
            }
            slices.append(slice_row)
            if index < manifest_scenario_count:
                manifest_scenarios.append({"scenario_id": scenario_id, "contract": str(contract), "sidecar": str(sidecar)})
        (self.domain_gap_root / "quasi-real-map-slices.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in slices) + "\n",
            encoding="utf-8",
        )
        self._write_json(
            self.domain_gap_root / "quasi-real-map-path-feedback-manifest.json",
            {"schema_version": "path-feedback-manifest/v1", "scenarios": manifest_scenarios},
        )
        self._write_json(
            self.domain_gap_root / "quasi-real-map-domain-gap-summary.json",
            {
                "schema_version": "quasi-real-map-domain-gap-summary/v1",
                "status": domain_gap_status,
                "reason_codes": [] if domain_gap_status == "passed" else ["domain_gap_failed"],
                "domain_gap_verdict": domain_gap_verdict,
                "slice_count": 12,
                "roi_group_count": 4,
                "context_id_missing_count": 1 if missing_context else 0,
                "legacy_identity_fallback_count": 1 if missing_context else 0,
                "fallback_or_open_grid_count": fallback_or_open_grid_count,
                "safety_regression_count": 0,
                "contract_violation_count": 0,
                "path_cost_regression_count": 0,
                "risk_regression_count": 0,
                "source_selection_regression_count": 0,
                **boundary,
            },
        )
        self._write_json(
            self.domain_gap_root / "quasi-real-map-path-feedback-summary.json",
            {
                "schema_version": "path-feedback-summary/v1",
                "scenario_count": 12,
                "open_grid_fallback_used": bool(fallback_or_open_grid_count),
                **boundary,
            },
        )
        self._write_json(
            self.domain_gap_root / "quasi-real-map-path-feedback-bridge-summary.json",
            {
                "schema_version": "quasi-real-map-path-feedback-bridge-summary/v1",
                "status": "passed",
                "slice_count": 12,
                "roi_group_count": 4,
                "context_id_missing_count": 1 if missing_context else 0,
                "legacy_identity_fallback_count": 1 if missing_context else 0,
            },
        )

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
