import copy
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class XunceCandidateLevelCoverageOpportunityMaterializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="xunce-coverage-materialization-"))
        self.source_root = self.temp_dir / "stage18a"
        self.coverage_root = self.temp_dir / "stage18c"
        self.output_root = self.temp_dir / "stage18e"
        self.config_path = self.temp_dir / "config.json"
        self._write_stage18a_evidence()
        self._write_stage18c_evidence()
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_materializes_candidate_coverage_fields_and_enriched_root(self) -> None:
        from scripts.run_xunce_candidate_level_coverage_opportunity_materialization import (
            run_xunce_candidate_level_coverage_opportunity_materialization,
        )

        original_payload = json.loads((self.source_root / "xunce-high-fidelity-path-feedback-audit.json").read_text(encoding="utf-8"))

        summary = run_xunce_candidate_level_coverage_opportunity_materialization(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["next_required_change"], "run_oracle_separability_benchmark")
        self.assertGreater(summary["candidate_coverage_spread_range"], 0.005)
        self.assertGreaterEqual(summary["roi_group_with_nonzero_spread_count"], 3)
        self.assertGreater(summary["useful_disagreement_opportunity_count"], 0)
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])
        self.assertEqual(summary["canary_traffic_fraction"], 0.0)

        enriched = json.loads((self.output_root / "xunce-high-fidelity-path-feedback-audit.json").read_text(encoding="utf-8"))
        candidate = enriched["scenarios"][0]["path_feedback"]["candidates"][0]
        expected_fields = {
            "endpoint_coverage_delta",
            "path_line_coverage_delta",
            "expected_coverage_rate_delta",
            "expected_new_coverage_cell_count",
            "roi_weighted_coverage_delta",
            "revisit_penalty",
            "coverage_gain_per_path_cost",
            "coverage_gain_per_risk",
            "coverage_opportunity_rank",
            "coverage_opportunity_source",
        }
        self.assertTrue(expected_fields.issubset(candidate))
        self.assertEqual(candidate["coverage_opportunity_source"], "geometric_counterfactual_from_stage18a_candidate/v1")
        self.assertTrue((self.output_root / "xunce-high-fidelity-real-map-roi-expansion-summary.json").is_file())
        self.assertTrue((self.output_root / "xunce-high-fidelity-real-map-slices.jsonl").is_file())
        self.assertTrue((self.output_root / "xunce-candidate-coverage-overlay.jsonl").is_file())
        self.assertTrue((self.output_root / "xunce-candidate-coverage-spread-by-scenario.jsonl").is_file())
        self.assertTrue((self.output_root / "xunce-candidate-coverage-roi-summary.json").is_file())
        self.assertTrue((self.output_root / "xunce-candidate-level-path-feedback-audit.json").is_file())
        self.assertTrue((self.output_root / "xunce-candidate-level-coverage-opportunity-manifest.json").is_file())
        self.assertTrue((self.output_root / "xunce-candidate-level-coverage-opportunity-report.md").is_file())

        unchanged_payload = json.loads((self.source_root / "xunce-high-fidelity-path-feedback-audit.json").read_text(encoding="utf-8"))
        self.assertEqual(unchanged_payload, original_payload)

    def test_low_spread_routes_to_repair_materialization(self) -> None:
        from scripts.run_xunce_candidate_level_coverage_opportunity_materialization import (
            run_xunce_candidate_level_coverage_opportunity_materialization,
        )

        self._write_stage18a_evidence(flat_candidates=True)

        summary = run_xunce_candidate_level_coverage_opportunity_materialization(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("candidate_coverage_spread_insufficient", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "repair_candidate_level_coverage_materialization")

    def test_roi_spread_insufficient_routes_to_roi_weighting_repair(self) -> None:
        from scripts.run_xunce_candidate_level_coverage_opportunity_materialization import (
            run_xunce_candidate_level_coverage_opportunity_materialization,
        )

        self._write_stage18a_evidence(roi_group_count=1)

        summary = run_xunce_candidate_level_coverage_opportunity_materialization(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("roi_spread_insufficient", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "expand_roi_or_refine_roi_weighting")

    def test_missing_candidate_geometry_fails_without_faking_coverage(self) -> None:
        from scripts.run_xunce_candidate_level_coverage_opportunity_materialization import (
            run_xunce_candidate_level_coverage_opportunity_materialization,
        )

        payload = json.loads((self.source_root / "xunce-high-fidelity-path-feedback-audit.json").read_text(encoding="utf-8"))
        del payload["scenarios"][0]["path_feedback"]["candidates"][0]["cell"]
        (self.source_root / "xunce-high-fidelity-path-feedback-audit.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        summary = run_xunce_candidate_level_coverage_opportunity_materialization(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("candidate_geometry_insufficient", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "repair_candidate_materialization_inputs")

    def _write_config(self) -> None:
        payload = {
            "schema_version": "xunce-candidate-level-coverage-opportunity-materialization-config/v1",
            "source_roi_expansion_root": str(self.source_root),
            "source_coverage_comparison_root": str(self.coverage_root),
            "required_scenario_count": 12,
            "rollout_steps": 4,
            "coverage_radius_cells": 1,
            "coverage_denominator_cells": 1000,
            "min_candidate_coverage_spread": 0.005,
            "min_roi_group_with_nonzero_spread_count": 3,
            "canary_traffic_fraction": 0.0,
        }
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_stage18a_evidence(self, *, flat_candidates: bool = False, roi_group_count: int = 4) -> None:
        self.source_root.mkdir(parents=True, exist_ok=True)
        scenarios = []
        slices = []
        for index in range(12):
            scenario_id = f"scenario_{index:03d}"
            roi_group = f"roi_{index % roi_group_count}"
            start_cell = [index * 30, 0]
            if flat_candidates:
                cells = [[index * 30 + 1, 1], [index * 30 + 1, 1], [index * 30 + 1, 1]]
            else:
                cells = [[index * 30 + 1, 1], [index * 30 + 4, 2], [index * 30 + 9, 3]]
            candidates = []
            for action_index, cell in enumerate(cells):
                candidates.append(
                    {
                        "action_index": action_index,
                        "cell": cell,
                        "reachable": True,
                        "path_cost": 5.0 + action_index,
                        "risk": 0.1 + action_index * 0.01,
                        "energy_cost": 2.0 + action_index,
                        "value": 0.2 + action_index * 0.1,
                    }
                )
            scenarios.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_group": roi_group,
                    "roi_group": roi_group,
                    "start_cell": start_cell,
                    "open_grid_fallback_used": False,
                    "tracking_safety_violation_count": 0,
                    "path_feedback": {"candidates": candidates},
                }
            )
            slices.append(
                {
                    "schema_version": "quasi-real-map-slice/v1",
                    "scenario_id": scenario_id,
                    "scenario_group": roi_group,
                    "roi_name": roi_group,
                    "split": "train",
                    "context_id": f"context-{scenario_id}",
                    "contract": str(self.source_root / f"{scenario_id}.contract.json"),
                    "sidecar": str(self.source_root / f"{scenario_id}.sidecar.json"),
                }
            )
        self._write_json(
            self.source_root / "xunce-high-fidelity-real-map-roi-expansion-summary.json",
            {
                "schema_version": "xunce-high-fidelity-real-map-roi-expansion-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "slice_count": len(slices),
                "roi_group_count": roi_group_count,
                "canary_traffic_fraction": 0.0,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "starts_online_canary": False,
            },
        )
        self._write_jsonl(self.source_root / "xunce-high-fidelity-real-map-slices.jsonl", slices)
        self._write_json(
            self.source_root / "xunce-high-fidelity-path-feedback-audit.json",
            {
                "schema_version": "xunce-high-fidelity-path-feedback-audit/v1",
                "scenario_count": len(scenarios),
                "candidate_count": sum(len(row["path_feedback"]["candidates"]) for row in scenarios),
                "reachable_count": sum(len(row["path_feedback"]["candidates"]) for row in scenarios),
                "fallback_or_open_grid_count": 0,
                "open_grid_fallback_used": False,
                "scenarios": scenarios,
            },
        )

    def _write_stage18c_evidence(self) -> None:
        self.coverage_root.mkdir(parents=True, exist_ok=True)
        self._write_json(
            self.coverage_root / "xunce-exploration-coverage-comparison-summary.json",
            {
                "schema_version": "xunce-exploration-coverage-comparison-summary/v1",
                "status": "passed",
                "scenario_count": 12,
                "rollout_steps": 4,
                "proxy_selection_used": False,
                "true_model_inference_executed": True,
            },
        )

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
