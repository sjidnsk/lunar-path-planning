import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class Global99RealMapMultiRoiGeneralizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="global-99-real-map-multi-roi-"))
        self.drift_root = self.temp_dir / "drift"
        self.domain_gap_root = self.temp_dir / "domain-gap"
        self.output_root = self.temp_dir / "output"
        self.config_path = self.temp_dir / "config.json"
        self.drift_root.mkdir(parents=True)
        self.domain_gap_root.mkdir(parents=True)
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_current_evidence_passes_and_writes_required_artifacts(self) -> None:
        from scripts.run_global_99_real_map_multi_roi_generalization import run_global_99_real_map_multi_roi_generalization

        self._write_sources()
        summary = run_global_99_real_map_multi_roi_generalization(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertTrue(summary["real_map_multi_roi_generalization_passed"])
        self.assertEqual(summary["real_map_multi_roi_generalization_verdict"], "eligible_for_default_policy_candidate_authorization_preflight")
        self.assertEqual(summary["next_required_change"], "default_policy_candidate_authorization_preflight")
        self.assertEqual(summary["source_evidence_refresh_drift_status"], "passed")
        self.assertEqual(summary["source_evidence_refresh_drift_next_required_change"], "global_99_real_map_multi_roi_generalization")
        self.assertEqual(summary["source_domain_gap_status"], "passed")
        self.assertEqual(summary["domain_gap_verdict"], "acceptable_for_next_pilot")
        self.assertEqual(summary["slice_count"], 12)
        self.assertEqual(summary["roi_group_count"], 4)
        self.assertEqual(summary["required_roi_group_count"], 4)
        self.assertEqual(summary["passed_roi_group_count"], 4)
        self.assertEqual(summary["failed_roi_group_count"], 0)
        self.assertEqual(summary["required_scenario_count"], 12)
        self.assertEqual(summary["passed_required_scenario_count"], 12)
        self.assertEqual(summary["failed_required_scenario_count"], 0)
        self.assertTrue(summary["split_coverage_complete"])
        self.assertEqual(summary["context_id_missing_count"], 0)
        self.assertEqual(summary["legacy_identity_fallback_count"], 0)
        self.assertEqual(summary["missing_contract_count"], 0)
        self.assertEqual(summary["missing_sidecar_count"], 0)
        self.assertEqual(summary["fallback_or_open_grid_count"], 0)
        self.assertTrue(summary["roi_generalization_audit_passed"])
        self.assertTrue(summary["scenario_matrix_audit_passed"])
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

        for filename in (
            "global-99-real-map-multi-roi-generalization-summary.json",
            "global-99-real-map-multi-roi-generalization-manifest.json",
            "global-99-real-map-multi-roi-scenario-results.jsonl",
            "global-99-real-map-multi-roi-family-summary.json",
            "global-99-real-map-multi-roi-lineage-audit.json",
            "global-99-real-map-multi-roi-scenario-matrix-audit.json",
            "global-99-real-map-multi-roi-boundary-audit.json",
            "global-99-real-map-multi-roi-kill-switch-audit.json",
            "global-99-real-map-multi-roi-rollback-audit.json",
            "global-99-real-map-multi-roi-telemetry-audit.json",
            "global-99-real-map-multi-roi-rejection-report.json",
            "global-99-real-map-multi-roi-generalization-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_missing_or_failed_drift_routes_to_drift_fix(self) -> None:
        from scripts.run_global_99_real_map_multi_roi_generalization import run_global_99_real_map_multi_roi_generalization

        self._write_sources(write_drift=False)
        summary = run_global_99_real_map_multi_roi_generalization(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_evidence_refresh_drift_summary", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_real_map_evidence_refresh_drift_audit")

    def test_domain_gap_failure_routes_to_domain_gap_fix(self) -> None:
        from scripts.run_global_99_real_map_multi_roi_generalization import run_global_99_real_map_multi_roi_generalization

        self._write_sources(domain_gap_status="failed", domain_gap_verdict="blocked")
        summary = run_global_99_real_map_multi_roi_generalization(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("domain_gap_evidence_not_acceptable", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_quasi_real_map_domain_gap_evidence")

    def test_insufficient_roi_or_split_coverage_routes_to_expand_roi(self) -> None:
        from scripts.run_global_99_real_map_multi_roi_generalization import run_global_99_real_map_multi_roi_generalization

        self._write_sources(roi_group_count=3)
        summary = run_global_99_real_map_multi_roi_generalization(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("real_map_multi_roi_coverage_insufficient", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "expand_real_map_roi_coverage")

    def test_context_identity_failure_routes_to_context_fix(self) -> None:
        from scripts.run_global_99_real_map_multi_roi_generalization import run_global_99_real_map_multi_roi_generalization

        self._write_sources(missing_context=True)
        summary = run_global_99_real_map_multi_roi_generalization(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("real_map_context_identity_incomplete", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_real_map_context_identity")

    def test_missing_sidecar_or_open_grid_routes_to_path_feedback_fix(self) -> None:
        from scripts.run_global_99_real_map_multi_roi_generalization import run_global_99_real_map_multi_roi_generalization

        self._write_sources(missing_sidecar=True, fallback_or_open_grid_count=1)
        summary = run_global_99_real_map_multi_roi_generalization(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("real_map_path_feedback_contract_incomplete", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_real_map_path_feedback_contract")

    def test_boundary_violation_blocks_multi_roi(self) -> None:
        from scripts.run_global_99_real_map_multi_roi_generalization import run_global_99_real_map_multi_roi_generalization

        self._write_sources(boundary_overrides={"replaces_default_policy": True})
        summary = run_global_99_real_map_multi_roi_generalization(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("real_map_multi_roi_boundary_violation", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "resolve_global_99_real_map_multi_roi_boundary_rejections")

    def _write_config(self) -> None:
        self.config_path.write_text(
            json.dumps(
                {
                    "schema_version": "global-99-real-map-multi-roi-generalization-config/v1",
                    "source_evidence_refresh_drift_root": str(self.drift_root),
                    "source_quasi_real_domain_gap_root": str(self.domain_gap_root),
                    "min_slice_count": 12,
                    "min_roi_group_count": 4,
                    "min_slices_per_roi_group": 3,
                    "required_splits": ["train", "validation", "test"],
                    "require_evidence_refresh_drift_passed": True,
                    "require_domain_gap_acceptable": True,
                    "require_context_ids": True,
                    "require_no_legacy_identity_fallback": True,
                    "require_no_open_grid_fallback": True,
                    "require_all_contract_and_sidecar_paths": True,
                    "multi_roi_generalization_only": True,
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
        write_drift: bool = True,
        domain_gap_status: str = "passed",
        domain_gap_verdict: str = "acceptable_for_next_pilot",
        roi_group_count: int = 4,
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
        if write_drift:
            self._write_json(
                self.drift_root / "global-99-real-map-evidence-refresh-drift-audit-summary.json",
                {
                    "schema_version": "global-99-real-map-evidence-refresh-drift-audit-summary/v1",
                    "status": "passed",
                    "reason_codes": [],
                    "next_required_change": "global_99_real_map_multi_roi_generalization",
                    "slice_count": 12,
                    "roi_group_count": roi_group_count,
                    "context_id_missing_count": 1 if missing_context else 0,
                    "legacy_identity_fallback_count": 1 if missing_context else 0,
                    "fallback_or_open_grid_count": fallback_or_open_grid_count,
                    **boundary,
                },
            )

        sidecar_root = self.domain_gap_root / "path_planner_sidecars"
        sidecar_root.mkdir(parents=True, exist_ok=True)
        rows = []
        splits = ["train", "validation", "test"]
        for group_index in range(roi_group_count):
            for split_index, split in enumerate(splits):
                scenario_id = f"roi_{group_index}_{split}"
                contract = sidecar_root / f"{scenario_id}.contract.json"
                sidecar = sidecar_root / f"{scenario_id}.path-planner-sidecar.json"
                contract.write_text(json.dumps({"scenario_id": scenario_id}), encoding="utf-8")
                if not (missing_sidecar and group_index == 0 and split_index == 0):
                    sidecar.write_text(json.dumps({"scenario_id": scenario_id}), encoding="utf-8")
                rows.append(
                    {
                        "schema_version": "quasi-real-map-slice/v1",
                        "scenario_id": scenario_id,
                        "scenario_group": f"roi_{group_index}",
                        "roi_name": f"roi_{group_index}",
                        "split": split,
                        "context_id": "" if missing_context and group_index == 0 and split_index == 0 else f"context-{scenario_id}",
                        "legacy_identity_fallback_used": bool(missing_context and group_index == 0 and split_index == 1),
                        "contract": str(contract),
                        "sidecar": str(sidecar),
                    }
                )
        (self.domain_gap_root / "quasi-real-map-slices.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
        self._write_json(
            self.domain_gap_root / "quasi-real-map-domain-gap-summary.json",
            {
                "schema_version": "quasi-real-map-domain-gap-summary/v1",
                "status": domain_gap_status,
                "reason_codes": [] if domain_gap_status == "passed" else ["domain_gap_failed"],
                "domain_gap_verdict": domain_gap_verdict,
                "slice_count": len(rows),
                "roi_group_count": roi_group_count,
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

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
