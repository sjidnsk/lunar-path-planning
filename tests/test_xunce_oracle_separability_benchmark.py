import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class XunceOracleSeparabilityBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="xunce-oracle-separability-"))
        self.materialized_root = self.temp_dir / "stage18e"
        self.output_root = self.temp_dir / "stage18f"
        self.config_path = self.temp_dir / "config.json"
        self.xunce_checkpoint = self.temp_dir / "xunce.pt"
        self.incumbent_checkpoint = self.temp_dir / "incumbent.pt"
        self.xunce_checkpoint.write_bytes(b"xunce")
        self.incumbent_checkpoint.write_bytes(b"incumbent")
        self._write_materialized_root()
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_oracle_separable_but_xunce_not_better_still_routes_to_model_comparison(self) -> None:
        from scripts.run_xunce_oracle_separability_benchmark import run_xunce_oracle_separability_benchmark

        summary = run_xunce_oracle_separability_benchmark(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertTrue(summary["oracle_separable"])
        self.assertFalse(summary["xunce_coverage_advantage_established"])
        self.assertTrue(summary["comparison_allowed"])
        self.assertEqual(summary["next_required_change"], "run_stage18c_v2_model_comparison")
        self.assertIn("xunce_coverage_advantage_not_established", summary["diagnostic_reason_codes"])
        self.assertGreater(summary["greedy_oracle_coverage_return_delta_vs_incumbent"], 0.0)
        self.assertGreater(summary["cost_aware_oracle_coverage_return_delta_vs_incumbent"], 0.0)
        self.assertGreaterEqual(summary["oracle_better_scenario_fraction"], 0.60)
        self.assertGreaterEqual(summary["oracle_better_roi_group_count"], 3)
        self.assertEqual(summary["oracle_mask_violation_count"], 0)
        self.assertEqual(summary["open_grid_fallback_count"], 0)
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])
        self.assertTrue((self.output_root / "xunce-oracle-separability-summary.json").is_file())
        self.assertTrue((self.output_root / "xunce-oracle-separability-scenarios.jsonl").is_file())
        self.assertTrue((self.output_root / "xunce-oracle-separability-roi-summary.json").is_file())
        self.assertTrue((self.output_root / "xunce-oracle-separability-decision-audit.json").is_file())
        self.assertTrue((self.output_root / "xunce-oracle-separability-manifest.json").is_file())
        self.assertTrue((self.output_root / "xunce-oracle-separability-report.md").is_file())
        self.assertTrue((self.output_root / "stage18c_v2").is_dir())

    def test_oracle_not_separable_is_diagnostic_and_allows_stage18c(self) -> None:
        from scripts.run_xunce_oracle_separability_benchmark import run_xunce_oracle_separability_benchmark

        self._write_materialized_root(flat=True)

        summary = run_xunce_oracle_separability_benchmark(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertFalse(summary["oracle_separable"])
        self.assertTrue(summary["comparison_allowed"])
        self.assertEqual(summary["next_required_change"], "run_stage18c_v2_model_comparison")
        self.assertIn("oracle_not_separable", summary["diagnostic_reason_codes"])
        self.assertEqual(summary["diagnostic_recommended_change"], "refine_candidate_generation_or_roi_complexity")

    def test_oracles_do_not_select_masked_or_unreachable_candidate(self) -> None:
        from scripts.run_xunce_oracle_separability_benchmark import run_xunce_oracle_separability_benchmark

        self._write_materialized_root(mask_best_candidate=True)

        summary = run_xunce_oracle_separability_benchmark(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["oracle_mask_violation_count"], 0)
        scenario_rows = self._read_jsonl(self.output_root / "xunce-oracle-separability-scenarios.jsonl")
        self.assertTrue(scenario_rows)
        self.assertTrue(all(row["greedy_oracle_selected_action_index"] != 2 for row in scenario_rows))
        self.assertTrue(all(row["cost_aware_oracle_selected_action_index"] != 2 for row in scenario_rows))

    def test_cost_aware_oracle_efficiency_regression_blocks_separability_gate(self) -> None:
        from scripts.run_xunce_oracle_separability_benchmark import run_xunce_oracle_separability_benchmark

        self._write_materialized_root(high_cost_oracle=True)

        summary = run_xunce_oracle_separability_benchmark(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertGreater(summary["cost_aware_oracle_coverage_return_delta_vs_incumbent"], 0.0)
        self.assertGreater(summary["cost_aware_oracle_efficiency_regression_count"], 0)
        self.assertFalse(summary["oracle_separable"])
        self.assertEqual(summary["next_required_change"], "run_stage18c_v2_model_comparison")
        self.assertIn("oracle_not_separable", summary["diagnostic_reason_codes"])

    def test_refined_safe_efficient_candidates_are_reported_but_do_not_filter_oracle_pool(self) -> None:
        from scripts.run_xunce_oracle_separability_benchmark import run_xunce_oracle_separability_benchmark

        self._write_refined_root()

        summary = run_xunce_oracle_separability_benchmark(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertTrue(summary["source_cost_efficient_refinement_detected"])
        self.assertGreater(summary["safe_efficient_opportunity_count"], 0)
        self.assertTrue(summary["oracle_separable"])
        self.assertEqual(summary["cost_aware_oracle_efficiency_regression_count"], 0)
        self.assertEqual(summary["next_required_change"], "run_stage18c_v2_model_comparison")
        scenario_rows = self._read_jsonl(self.output_root / "xunce-oracle-separability-scenarios.jsonl")
        self.assertTrue(scenario_rows)
        self.assertTrue(all(row["greedy_oracle_selected_action_index"] == 2 for row in scenario_rows))
        self.assertTrue(all(row["cost_aware_oracle_selected_action_index"] in (1, 2) for row in scenario_rows))

    def test_safe_efficient_candidate_alias_does_not_filter_cost_aware_oracle(self) -> None:
        from scripts.run_xunce_oracle_separability_benchmark import run_xunce_oracle_separability_benchmark

        self._write_refined_root(candidate_alias_only=True)

        summary = run_xunce_oracle_separability_benchmark(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertTrue(summary["source_cost_efficient_refinement_detected"])
        self.assertGreater(summary["safe_efficient_opportunity_count"], 0)
        self.assertTrue(summary["oracle_separable"])
        self.assertEqual(summary["cost_aware_oracle_efficiency_regression_count"], 0)
        scenario_rows = self._read_jsonl(self.output_root / "xunce-oracle-separability-scenarios.jsonl")
        self.assertTrue(scenario_rows)
        self.assertTrue(all(row["greedy_oracle_selected_action_index"] == 2 for row in scenario_rows))
        self.assertTrue(all(row["cost_aware_oracle_selected_action_index"] in (1, 2) for row in scenario_rows))

    def test_refined_root_without_safe_efficient_opportunities_still_allows_oracle_benchmark(self) -> None:
        from scripts.run_xunce_oracle_separability_benchmark import run_xunce_oracle_separability_benchmark

        self._write_refined_root(no_safe=True)

        summary = run_xunce_oracle_separability_benchmark(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertTrue(summary["oracle_separable"])
        self.assertEqual(summary["safe_efficient_opportunity_count"], 0)
        self.assertIn("safe_efficient_opportunity_insufficient", summary["diagnostic_reason_codes"])
        self.assertEqual(summary["next_required_change"], "run_stage18c_v2_model_comparison")

    def _write_config(self) -> None:
        payload = {
            "schema_version": "xunce-oracle-separability-benchmark-config/v1",
            "source_materialized_coverage_root": str(self.materialized_root),
            "xunce_candidate_checkpoint": str(self.xunce_checkpoint),
            "incumbent_policy_checkpoint": str(self.incumbent_checkpoint),
            "scenario_count": 12,
            "rollout_steps": 4,
            "oracle_better_scenario_fraction_threshold": 0.60,
            "oracle_better_roi_group_count_threshold": 3,
            "candidate_refresh_mode": "dynamic_from_coverage_memory",
            "coverage_metric_mode": "path_line_plus_endpoint",
            "include_oracle_baselines": True,
            "include_roi_weighted_coverage": True,
            "canary_traffic_fraction": 0.0,
        }
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_materialized_root(
        self,
        *,
        flat: bool = False,
        mask_best_candidate: bool = False,
        high_cost_oracle: bool = False,
    ) -> None:
        self.materialized_root.mkdir(parents=True, exist_ok=True)
        scenarios = []
        slices = []
        for index in range(12):
            scenario_id = f"scenario_{index:03d}"
            roi_group = f"roi_{index % 4}"
            if flat:
                coverage = [0.03, 0.03, 0.03]
                costs = [5.0, 6.0, 7.0]
            elif high_cost_oracle:
                coverage = [0.02, 0.06, 0.14]
                costs = [5.0, 30.0, 20.0]
            else:
                coverage = [0.02, 0.05, 0.12]
                costs = [10.0, 7.0, 5.0]
            candidates = []
            for action_index, value in enumerate(coverage):
                reachable = not (mask_best_candidate and action_index == 2)
                candidates.append(
                    {
                        "action_index": action_index,
                        "cell": [index * 20 + action_index + 1, action_index],
                        "reachable": reachable,
                        "path_cost": costs[action_index],
                        "risk": 0.1 + action_index * 0.01,
                        "endpoint_coverage_delta": value / 2.0,
                        "path_line_coverage_delta": value,
                        "expected_coverage_rate_delta": value,
                        "expected_new_coverage_cell_count": int(value * 1000),
                        "roi_weighted_coverage_delta": value,
                        "revisit_penalty": 0.0,
                        "coverage_gain_per_path_cost": value / costs[action_index],
                        "coverage_gain_per_risk": value / (0.1 + action_index * 0.01),
                        "coverage_opportunity_rank": 3 - action_index,
                        "coverage_opportunity_source": "geometric_counterfactual_from_stage18a_candidate/v1",
                    }
                )
            scenarios.append(
                {
                    "scenario_id": scenario_id,
                    "roi_group": roi_group,
                    "incumbent_selected_action_index": 0,
                    "xunce_selected_action_index": 0,
                    "path_feedback": {"candidates": candidates},
                    "open_grid_fallback_used": False,
                }
            )
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
                "candidate_coverage_spread_range": 0.1 if not flat else 0.0,
                "roi_group_with_nonzero_spread_count": 4 if not flat else 0,
                "useful_disagreement_opportunity_count": 12 if not flat else 0,
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
                "roi_group_count": 4,
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
                "reachable_count": sum(
                    1
                    for scenario in scenarios
                    for candidate in scenario["path_feedback"]["candidates"]
                    if candidate["reachable"]
                ),
                "fallback_or_open_grid_count": 0,
                "open_grid_fallback_used": False,
                "scenarios": scenarios,
            },
        )

    def _write_refined_root(self, *, no_safe: bool = False, candidate_alias_only: bool = False) -> None:
        self.materialized_root.mkdir(parents=True, exist_ok=True)
        scenarios = []
        slices = []
        for index in range(12):
            scenario_id = f"scenario_{index:03d}"
            roi_group = f"roi_{index % 4}"
            coverage = [0.02, 0.08, 0.14]
            costs = [10.0, 8.0, 5.0]
            risks = [0.10, 0.09, 0.40]
            candidates = []
            for action_index, value in enumerate(coverage):
                safe = bool(action_index == 1 and not no_safe)
                candidate = {
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
                        "cost_efficiency_adjusted_coverage_delta": value / (costs[action_index] * (1.0 + risks[action_index])),
                        "risk_adjusted_coverage_delta": value / (1.0 + risks[action_index]),
                        "budget_adjusted_coverage_delta": value * (1.0 - costs[action_index] / 5000.0),
                        "coverage_cost_pareto_rank": 1,
                        "dominance_status": "pareto_frontier",
                        "safe_efficient_opportunity": safe,
                        "cost_efficiency_guard_passed": safe,
                        "risk_efficiency_guard_passed": safe,
                        "efficiency_refinement_source": "cost_efficient_counterfactual_from_stage18e_candidate/v1",
                }
                if candidate_alias_only:
                    candidate["safe_efficient_candidate"] = candidate.pop("safe_efficient_opportunity")
                candidates.append(candidate)
            scenarios.append(
                {
                    "scenario_id": scenario_id,
                    "roi_group": roi_group,
                    "incumbent_selected_action_index": 0,
                    "xunce_selected_action_index": 0,
                    "path_feedback": {"candidates": candidates},
                    "open_grid_fallback_used": False,
                }
            )
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
            self.materialized_root / "xunce-cost-efficient-coverage-opportunity-summary.json",
            {
                "schema_version": "xunce-cost-efficient-coverage-opportunity-summary/v1",
                "status": "passed" if not no_safe else "failed",
                "cost_efficient_refinement_passed": not no_safe,
                "safe_efficient_opportunity_count": 0 if no_safe else 12,
                "roi_group_with_safe_efficient_opportunity_count": 0 if no_safe else 4,
                "next_required_change": "run_oracle_separability_benchmark" if not no_safe else "expand_roi_or_map_complexity",
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
                "roi_group_count": 4,
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

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
