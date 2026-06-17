import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class XunceHighFidelityRealMapRoiExpansionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="xunce-hifi-roi-expansion-"))
        self.release_root = self.temp_dir / "release"
        self.domain_gap_root = self.temp_dir / "domain-gap"
        self.output_root = self.temp_dir / "output"
        self.config_path = self.temp_dir / "config.json"
        self._write_release_summary()
        self._write_domain_gap_sources()
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_default_fixture_expands_to_24_slices_and_8_roi_groups(self) -> None:
        from scripts.run_xunce_high_fidelity_real_map_roi_expansion import run_xunce_high_fidelity_real_map_roi_expansion

        summary = run_xunce_high_fidelity_real_map_roi_expansion(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["slice_count"], 24)
        self.assertEqual(summary["roi_group_count"], 8)
        self.assertEqual(summary["context_id_missing_count"], 0)
        self.assertEqual(summary["legacy_identity_fallback_count"], 0)
        self.assertEqual(summary["fallback_or_open_grid_count"], 0)
        self.assertEqual(summary["domain_gap_verdict"], "acceptable_for_high_fidelity_comparison")
        self.assertEqual(summary["next_required_change"], "xunce_high_fidelity_real_map_policy_comparison")
        self.assertTrue(summary["split_coverage_complete"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["modifies_network"])

        for filename in (
            "xunce-high-fidelity-real-map-roi-expansion-summary.json",
            "xunce-high-fidelity-real-map-roi-expansion-manifest.json",
            "xunce-high-fidelity-real-map-slices.jsonl",
            "xunce-high-fidelity-real-map-roi-groups.json",
            "xunce-high-fidelity-domain-gap-audit.json",
            "xunce-high-fidelity-path-feedback-audit.json",
            "xunce-high-fidelity-boundary-audit.json",
            "xunce-high-fidelity-rejection-report.json",
            "xunce-high-fidelity-real-map-roi-expansion-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_runtime_source_overrides_allow_external_evidence_bundle(self) -> None:
        from scripts.run_xunce_high_fidelity_real_map_roi_expansion import run_xunce_high_fidelity_real_map_roi_expansion

        bad_config = {
            "schema_version": "xunce-high-fidelity-real-map-roi-expansion-config/v1",
            "source_xunce_release_governance_root": "outputs/missing-release",
            "source_quasi_real_domain_gap_root": "outputs/missing-domain-gap",
            "target_slice_count": 24,
            "target_roi_group_count": 8,
            "required_splits": ["train", "validation", "test"],
            "run_bridge": False,
            "run_path_feedback": False,
            "canary_traffic_fraction": 0.0,
        }
        self.config_path.write_text(json.dumps(bad_config, ensure_ascii=False, indent=2), encoding="utf-8")

        summary = run_xunce_high_fidelity_real_map_roi_expansion(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
            config_overrides={
                "source_xunce_release_governance_root": str(self.release_root),
                "source_quasi_real_domain_gap_root": str(self.domain_gap_root),
            },
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["next_required_change"], "xunce_high_fidelity_real_map_policy_comparison")

    def test_context_identity_failure_routes_to_context_fix(self) -> None:
        from scripts.run_xunce_high_fidelity_real_map_roi_expansion import run_xunce_high_fidelity_real_map_roi_expansion

        self._write_domain_gap_sources(missing_context=True)
        summary = run_xunce_high_fidelity_real_map_roi_expansion(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("real_map_context_identity_incomplete", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_real_map_context_identity")

    def test_open_grid_or_missing_sidecar_routes_to_path_feedback_fix(self) -> None:
        from scripts.run_xunce_high_fidelity_real_map_roi_expansion import run_xunce_high_fidelity_real_map_roi_expansion

        self._write_domain_gap_sources(missing_sidecar=True, fallback_or_open_grid_count=1)
        summary = run_xunce_high_fidelity_real_map_roi_expansion(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("real_map_path_feedback_contract_incomplete", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_real_map_path_feedback_contract")

    def test_roi_shortage_routes_to_expansion_fix(self) -> None:
        from scripts.run_xunce_high_fidelity_real_map_roi_expansion import run_xunce_high_fidelity_real_map_roi_expansion

        self._write_domain_gap_sources(roi_group_count=2)
        summary = run_xunce_high_fidelity_real_map_roi_expansion(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("high_fidelity_real_map_roi_coverage_insufficient", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "expand_high_fidelity_real_map_roi_coverage")

    def test_release_boundary_violation_blocks_expansion(self) -> None:
        from scripts.run_xunce_high_fidelity_real_map_roi_expansion import run_xunce_high_fidelity_real_map_roi_expansion

        self._write_release_summary({"replaces_default_policy": True})
        summary = run_xunce_high_fidelity_real_map_roi_expansion(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("xunce_high_fidelity_roi_boundary_violation", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "resolve_xunce_high_fidelity_real_map_boundary_rejections")

    def _write_config(self) -> None:
        payload = {
            "schema_version": "xunce-high-fidelity-real-map-roi-expansion-config/v1",
            "source_xunce_release_governance_root": str(self.release_root),
            "source_quasi_real_domain_gap_root": str(self.domain_gap_root),
            "target_slice_count": 24,
            "target_roi_group_count": 8,
            "required_splits": ["train", "validation", "test"],
            "new_roi_groups": [
                "shadowed_transition",
                "crater_rim_fragmented",
                "low_sun_roughness",
                "mixed_passability_edge",
            ],
            "run_bridge": False,
            "run_path_feedback": False,
            "canary_traffic_fraction": 0.0,
        }
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_release_summary(self, updates: dict | None = None) -> None:
        payload = {
            "schema_version": "xunce-release-governance-gate-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "next_required_change": "xunce_research_track_complete",
            "release_governance_gate_passed": True,
            "xunce_research_chain_complete": True,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "runs_new_ppo_update": False,
            "modifies_network": False,
            "modifies_action_space": False,
            "modifies_default_astar": False,
            "real_world_release_approved": False,
            "real_world_performance_claimed": False,
        }
        if updates:
            payload.update(updates)
        self.release_root.mkdir(parents=True, exist_ok=True)
        (self.release_root / "xunce-release-governance-gate-summary.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_domain_gap_sources(
        self,
        *,
        roi_group_count: int = 4,
        missing_context: bool = False,
        missing_sidecar: bool = False,
        fallback_or_open_grid_count: int = 0,
    ) -> None:
        self.domain_gap_root.mkdir(parents=True, exist_ok=True)
        sidecar_root = self.domain_gap_root / "path_planner_sidecars"
        sidecar_root.mkdir(parents=True, exist_ok=True)
        rows = []
        scenarios = []
        groups = [f"roi_{index}" for index in range(roi_group_count)]
        splits = ["train", "validation", "test"]
        for group_index, group in enumerate(groups):
            for split_index, split in enumerate(splits):
                scenario_id = f"lola_qreal_{group}_{split}_{group_index}{split_index}"
                contract = sidecar_root / f"{scenario_id}.contract.json"
                sidecar = sidecar_root / f"{scenario_id}.path-planner-sidecar.json"
                contract.write_text(json.dumps({"scenario_id": scenario_id}), encoding="utf-8")
                if not (missing_sidecar and group_index == 0 and split_index == 0):
                    sidecar.write_text(json.dumps({"scenario_id": scenario_id}), encoding="utf-8")
                context_id = "" if missing_context and group_index == 0 and split_index == 0 else f"context-{scenario_id}"
                rows.append(
                    {
                        "schema_version": "quasi-real-map-slice/v1",
                        "scenario_id": scenario_id,
                        "scenario_group": group,
                        "roi_name": group,
                        "split": split,
                        "context_id": context_id,
                        "legacy_identity_fallback_used": False,
                        "contract": str(contract),
                        "sidecar": str(sidecar),
                        "map_source": {"roi": {"x": 100 * group_index, "y": 10 * split_index, "width": 32, "height": 32}},
                        "passable_ratio": 1.0,
                    }
                )
                scenarios.append(
                    {
                        "scenario_id": scenario_id,
                        "scenario_group": group,
                        "selected_cell_after_path_feedback": [1, 1],
                        "selected_cell_before_path_feedback": [2, 2],
                        "selected_path_cost_after_feedback": 10.0 + group_index,
                        "selected_path_cost_before_feedback": 12.0 + group_index,
                        "coverage_rate_delta": 0.1,
                        "open_grid_fallback_used": False,
                        "tracking_safety_violation_count": 0,
                        "path_feedback": {
                            "candidates": [
                                {"cell": [1, 1], "reachable": True, "path_cost": 10.0 + group_index, "risk": 0.1},
                                {"cell": [2, 2], "reachable": True, "path_cost": 12.0 + group_index, "risk": 0.2},
                            ]
                        },
                    }
                )
        (self.domain_gap_root / "quasi-real-map-slices.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
            encoding="utf-8",
        )
        summary = {
            "schema_version": "quasi-real-map-domain-gap-summary/v1",
            "status": "passed",
            "reason_codes": [],
            "domain_gap_verdict": "acceptable_for_next_pilot",
            "slice_count": len(rows),
            "roi_group_count": roi_group_count,
            "context_id_missing_count": 1 if missing_context else 0,
            "legacy_identity_fallback_count": 0,
            "fallback_or_open_grid_count": fallback_or_open_grid_count,
            "open_grid_fallback_count": fallback_or_open_grid_count,
            "safety_regression_count": 0,
            "contract_violation_count": 0,
            "path_cost_regression_count": 0,
            "risk_regression_count": 0,
            "source_selection_regression_count": 0,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
        }
        (self.domain_gap_root / "quasi-real-map-domain-gap-summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        path_feedback_summary = {
            "schema_version": "path-feedback-summary/v1",
            "scenario_count": len(scenarios),
            "candidate_count": len(scenarios) * 2,
            "reachable_count": len(scenarios) * 2,
            "open_grid_fallback_used": False,
            "open_grid_fallback_used_count": fallback_or_open_grid_count,
            "tracking_safety_violation_count": 0,
            "scenarios": scenarios,
        }
        (self.domain_gap_root / "quasi-real-map-path-feedback-summary.json").write_text(
            json.dumps(path_feedback_summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
