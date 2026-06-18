import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class XunceCostEfficientCoverageOpportunityRefinementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="xunce-cost-efficient-refinement-"))
        self.materialized_root = self.temp_dir / "stage18e"
        self.oracle_root = self.temp_dir / "stage18f"
        self.output_root = self.temp_dir / "stage18f1"
        self.config_path = self.temp_dir / "config.json"
        self._write_materialized_root()
        self._write_oracle_root()
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_refines_candidates_and_preserves_stage18e_root(self) -> None:
        from scripts.run_xunce_cost_efficient_coverage_opportunity_refinement import (
            run_xunce_cost_efficient_coverage_opportunity_refinement,
        )

        original_payload = json.loads((self.materialized_root / "xunce-high-fidelity-path-feedback-audit.json").read_text(encoding="utf-8"))

        summary = run_xunce_cost_efficient_coverage_opportunity_refinement(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertTrue(summary["cost_efficient_refinement_passed"])
        self.assertEqual(summary["next_required_change"], "run_oracle_separability_benchmark")
        self.assertGreater(summary["safe_efficient_opportunity_count"], 0)
        self.assertGreaterEqual(summary["roi_group_with_safe_efficient_opportunity_count"], 3)
        self.assertGreater(summary["cost_efficient_coverage_spread_range"], 0.005)
        self.assertEqual(summary["canary_traffic_fraction"], 0.0)
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])

        refined = json.loads((self.output_root / "xunce-high-fidelity-path-feedback-audit.json").read_text(encoding="utf-8"))
        candidates = refined["scenarios"][0]["path_feedback"]["candidates"]
        safe_candidate = candidates[1]
        inefficient_candidate = candidates[2]
        expected_fields = {
            "cost_efficiency_adjusted_coverage_delta",
            "risk_adjusted_coverage_delta",
            "budget_adjusted_coverage_delta",
            "coverage_cost_pareto_rank",
            "dominance_status",
            "safe_efficient_opportunity",
            "cost_efficiency_guard_passed",
            "risk_efficiency_guard_passed",
            "efficiency_refinement_source",
        }
        self.assertTrue(expected_fields.issubset(safe_candidate))
        self.assertTrue(safe_candidate["safe_efficient_opportunity"])
        self.assertFalse(inefficient_candidate["safe_efficient_opportunity"])
        self.assertGreater(inefficient_candidate["roi_weighted_coverage_delta"], safe_candidate["roi_weighted_coverage_delta"])
        self.assertGreater(inefficient_candidate["path_cost"], safe_candidate["path_cost"])

        expected_files = (
            "xunce-cost-efficient-coverage-opportunity-summary.json",
            "xunce-cost-efficient-candidate-overlay.jsonl",
            "xunce-cost-efficient-spread-by-scenario.jsonl",
            "xunce-cost-efficient-roi-summary.json",
            "xunce-cost-efficient-decision-audit.json",
            "xunce-cost-efficient-coverage-opportunity-manifest.json",
            "xunce-cost-efficient-coverage-opportunity-report.md",
            "xunce-high-fidelity-real-map-roi-expansion-summary.json",
            "xunce-high-fidelity-real-map-slices.jsonl",
            "xunce-high-fidelity-path-feedback-audit.json",
        )
        for filename in expected_files:
            self.assertTrue((self.output_root / filename).is_file(), filename)

        unchanged_payload = json.loads((self.materialized_root / "xunce-high-fidelity-path-feedback-audit.json").read_text(encoding="utf-8"))
        self.assertEqual(unchanged_payload, original_payload)

    def test_no_safe_efficient_opportunities_routes_to_roi_or_map_complexity(self) -> None:
        from scripts.run_xunce_cost_efficient_coverage_opportunity_refinement import (
            run_xunce_cost_efficient_coverage_opportunity_refinement,
        )

        self._write_materialized_root(no_safe_candidates=True)

        summary = run_xunce_cost_efficient_coverage_opportunity_refinement(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("safe_efficient_opportunity_insufficient", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "expand_roi_or_map_complexity")

    def test_missing_incumbent_selection_records_fallback_source(self) -> None:
        from scripts.run_xunce_cost_efficient_coverage_opportunity_refinement import (
            run_xunce_cost_efficient_coverage_opportunity_refinement,
        )

        self._write_materialized_root(omit_incumbent_selection=True)

        summary = run_xunce_cost_efficient_coverage_opportunity_refinement(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertIn("fallback_action_index_0", summary["incumbent_selection_sources"])
        refined = json.loads((self.output_root / "xunce-high-fidelity-path-feedback-audit.json").read_text(encoding="utf-8"))
        self.assertEqual(refined["scenarios"][0]["incumbent_selection_source"], "fallback_action_index_0")

    def _write_config(self) -> None:
        payload = {
            "schema_version": "xunce-cost-efficient-coverage-opportunity-refinement-config/v1",
            "source_materialized_coverage_root": str(self.materialized_root),
            "source_oracle_separability_root": str(self.oracle_root),
            "required_scenario_count": 6,
            "min_cost_efficient_spread": 0.005,
            "min_safe_efficient_opportunity_count": 1,
            "min_roi_group_with_safe_efficient_opportunity": 3,
            "path_budget_m": 5000.0,
            "epsilon": 1.0e-12,
            "canary_traffic_fraction": 0.0,
        }
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_materialized_root(self, *, no_safe_candidates: bool = False, omit_incumbent_selection: bool = False) -> None:
        self.materialized_root.mkdir(parents=True, exist_ok=True)
        scenarios = []
        slices = []
        for index in range(6):
            scenario_id = f"scenario_{index:03d}"
            roi_group = f"roi_{index % 3}"
            if no_safe_candidates:
                coverage = [0.02, 0.06, 0.14]
                costs = [10.0, 18.0, 30.0]
                risks = [0.10, 0.20, 0.30]
            else:
                coverage = [0.02, 0.06, 0.14]
                costs = [10.0, 8.0, 30.0]
                risks = [0.10, 0.09, 0.30]
            candidates = []
            for action_index, value in enumerate(coverage):
                candidates.append(
                    {
                        "action_index": action_index,
                        "cell": [index * 20 + action_index + 1, action_index],
                        "reachable": True,
                        "path_cost": costs[action_index],
                        "risk": risks[action_index],
                        "endpoint_coverage_delta": value / 2.0,
                        "path_line_coverage_delta": value,
                        "expected_coverage_rate_delta": value,
                        "expected_new_coverage_cell_count": int(value * 1000),
                        "roi_weighted_coverage_delta": value,
                        "revisit_penalty": 0.0,
                        "coverage_gain_per_path_cost": value / costs[action_index],
                        "coverage_gain_per_risk": value / risks[action_index],
                        "coverage_opportunity_rank": 3 - action_index,
                        "coverage_opportunity_source": "geometric_counterfactual_from_stage18a_candidate/v1",
                    }
                )
            scenario = {
                "scenario_id": scenario_id,
                "roi_group": roi_group,
                "xunce_selected_action_index": 0,
                "path_feedback": {"candidates": candidates},
                "open_grid_fallback_used": False,
            }
            if not omit_incumbent_selection:
                scenario["incumbent_selected_action_index"] = 0
            scenarios.append(scenario)
            slices.append(
                {
                    "scenario_id": scenario_id,
                    "roi_name": roi_group,
                    "split": "train",
                    "context_id": f"context-{scenario_id}",
                    "contract": str(self.materialized_root / f"{scenario_id}.contract.json"),
                    "sidecar": str(self.materialized_root / f"{scenario_id}.sidecar.json"),
                }
            )
        self._write_json(
            self.materialized_root / "xunce-candidate-level-coverage-opportunity-summary.json",
            {
                "schema_version": "xunce-candidate-level-coverage-opportunity-summary/v1",
                "status": "passed",
                "next_required_change": "run_oracle_separability_benchmark",
                "candidate_coverage_spread_range": 0.12,
                "roi_group_with_nonzero_spread_count": 3,
                "useful_disagreement_opportunity_count": 6,
                "canary_traffic_fraction": 0.0,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "starts_online_canary": False,
            },
        )
        self._write_json(
            self.materialized_root / "xunce-high-fidelity-real-map-roi-expansion-summary.json",
            {
                "schema_version": "xunce-high-fidelity-real-map-roi-expansion-summary/v1",
                "status": "passed",
                "slice_count": len(slices),
                "roi_group_count": 3,
                "canary_traffic_fraction": 0.0,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "starts_online_canary": False,
            },
        )
        self._write_jsonl(self.materialized_root / "xunce-high-fidelity-real-map-slices.jsonl", slices)
        self._write_json(
            self.materialized_root / "xunce-high-fidelity-path-feedback-audit.json",
            {
                "schema_version": "xunce-high-fidelity-path-feedback-audit/v1",
                "scenario_count": len(scenarios),
                "candidate_count": len(scenarios) * 3,
                "reachable_count": len(scenarios) * 3,
                "fallback_or_open_grid_count": 0,
                "open_grid_fallback_used": False,
                "scenarios": scenarios,
            },
        )

    def _write_oracle_root(self) -> None:
        self.oracle_root.mkdir(parents=True, exist_ok=True)
        self._write_json(
            self.oracle_root / "xunce-oracle-separability-summary.json",
            {
                "schema_version": "xunce-oracle-separability-summary/v1",
                "status": "passed",
                "oracle_separable": False,
                "next_required_change": "expand_roi_or_map_complexity",
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
