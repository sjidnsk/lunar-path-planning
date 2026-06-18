import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class XunceRiskCoverageCostQuantizationAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="xunce-risk-coverage-cost-"))
        self.bound_root = self.temp_dir / "stage18g0"
        self.output_root = self.temp_dir / "stage18h0"
        self.config_path = self.temp_dir / "config.json"
        self._write_bound_root()
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_quantizes_24_scenarios_and_72_candidates(self) -> None:
        from scripts.run_xunce_risk_coverage_cost_quantization_audit import (
            run_xunce_risk_coverage_cost_quantization_audit,
        )

        summary = run_xunce_risk_coverage_cost_quantization_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["scenario_count"], 24)
        self.assertEqual(summary["candidate_count"], 72)
        self.assertFalse(summary["metric_coupling_detected"])
        self.assertEqual(summary["finite_metric_issue_count"], 0)
        self.assertEqual(summary["missing_atomic_metric_count"], 0)
        self.assertEqual(summary["fallback_action_index_0_count"], 0)
        self.assertEqual(summary["open_grid_fallback_count"], 0)
        self.assertGreater(summary["safe_efficient_candidate_count"], 0)
        self.assertGreaterEqual(summary["roi_group_with_safe_efficient_candidate_count"], 3)
        self.assertEqual(summary["next_required_change"], "rerun_oracle_separability_with_quantized_root")
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])

        rows = self._read_jsonl(self.output_root / "xunce-risk-coverage-cost-candidates.jsonl")
        self.assertEqual(len(rows), 72)
        candidate = rows[1]
        coverage = candidate["coverage_vector"]
        self.assertEqual(coverage["coverage_source"], "geometric_counterfactual_from_stage18a_candidate/v1")
        self.assertEqual(coverage["coverage_cell_set_kind"], "path_line_plus_endpoint_union")
        self.assertEqual(coverage["coverage_dedupe_scope"], "scenario_step_new_cells")
        self.assertEqual(coverage["endpoint_new_cell_count"], 30.0)
        self.assertEqual(coverage["path_line_new_cell_count"], 50.0)
        self.assertEqual(coverage["total_new_cell_count"], 60.0)
        self.assertNotEqual(
            coverage["total_new_cell_count"],
            coverage["endpoint_new_cell_count"] + coverage["path_line_new_cell_count"],
        )
        self.assertEqual(set(candidate["cost_vector"]), {"path_cost", "budget_used_ratio", "planning_latency_ms", "cost_source"})
        self.assertNotIn("max_curvature", candidate["cost_vector"])
        self.assertNotIn("ackermann_feasibility", candidate["cost_vector"])
        self.assertTrue(candidate["safe_efficient_candidate"])
        self.assertTrue(candidate["safe_efficient_opportunity"])
        self.assertEqual(candidate["safe_efficient_opportunity"], candidate["safe_efficient_candidate"])
        generated = json.loads((self.output_root / "xunce-high-fidelity-path-feedback-audit.json").read_text(encoding="utf-8"))
        generated_candidate = generated["scenarios"][0]["path_feedback"]["candidates"][1]
        self.assertTrue(generated_candidate["safe_efficient_candidate"])
        self.assertTrue(generated_candidate["safe_efficient_opportunity"])
        self.assertEqual(generated_candidate["safe_efficient_opportunity"], generated_candidate["safe_efficient_candidate"])

        expected_files = (
            "xunce-risk-coverage-cost-quantization-summary.json",
            "xunce-risk-coverage-cost-candidates.jsonl",
            "xunce-risk-coverage-cost-scenarios.jsonl",
            "xunce-risk-coverage-cost-roi-summary.json",
            "xunce-risk-coverage-cost-pareto-audit.json",
            "xunce-risk-coverage-cost-decision-audit.json",
            "xunce-risk-coverage-cost-report.md",
            "xunce-high-fidelity-real-map-roi-expansion-summary.json",
            "xunce-high-fidelity-real-map-slices.jsonl",
            "xunce-high-fidelity-path-feedback-audit.json",
            "xunce-candidate-level-coverage-opportunity-summary.json",
        )
        for filename in expected_files:
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_missing_true_incumbent_binding_routes_to_binding_stage(self) -> None:
        from scripts.run_xunce_risk_coverage_cost_quantization_audit import (
            run_xunce_risk_coverage_cost_quantization_audit,
        )

        self._write_json(
            self.bound_root / "xunce-true-incumbent-selection-binding-summary.json",
            {"schema_version": "xunce-true-incumbent-selection-binding-summary/v1", "status": "failed", "true_incumbent_selection_bound": False},
        )

        summary = run_xunce_risk_coverage_cost_quantization_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_true_incumbent_binding", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "run_true_incumbent_selection_binding")

    def test_unvalidated_positive_proposal_cannot_be_safe(self) -> None:
        from scripts.run_xunce_risk_coverage_cost_quantization_audit import (
            run_xunce_risk_coverage_cost_quantization_audit,
        )

        self._write_bound_root(unvalidated_positive=True)

        summary = run_xunce_risk_coverage_cost_quantization_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["safe_efficient_candidate_count"], 0)
        self.assertIn("path_feedback_validation_missing", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "repair_path_feedback_candidate_validation")

        rows = self._read_jsonl(self.output_root / "xunce-risk-coverage-cost-candidates.jsonl")
        positive = [row for row in rows if row["action_index"] == 1]
        self.assertTrue(positive)
        self.assertTrue(all(not row["safe_efficient_candidate"] for row in positive))
        self.assertTrue(all(not row["safe_efficient_opportunity"] for row in positive))
        self.assertTrue(all(row["failure_primary_axis"] == "validation" for row in positive))

    def test_risk_regression_is_reported_as_diagnostic_not_blocking(self) -> None:
        from scripts.run_xunce_risk_coverage_cost_quantization_audit import (
            run_xunce_risk_coverage_cost_quantization_audit,
        )

        self._write_bound_root(risk_regression_only=True)

        summary = run_xunce_risk_coverage_cost_quantization_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertTrue(summary["comparison_allowed"])
        self.assertEqual(summary["blocking_reason_codes"], [])
        self.assertEqual(summary["safe_efficient_candidate_count"], 0)
        self.assertGreater(summary["coverage_positive_but_risk_regressive_count"], 0)
        self.assertEqual(summary["risk_regression_primary_axis_counts"], {"path_risk_peak": 48})
        self.assertIn("risk_guard_too_strict_or_miscalibrated", summary["diagnostic_reason_codes"])
        self.assertEqual(summary["diagnostic_recommended_change"], "calibrate_risk_margin_or_risk_attribution")
        self.assertEqual(summary["next_required_change"], "rerun_oracle_separability_with_quantized_root")

        rows = self._read_jsonl(self.output_root / "xunce-risk-coverage-cost-candidates.jsonl")
        risk_failed = [row for row in rows if row["action_index"] in (1, 2)]
        self.assertTrue(all(row["failure_primary_axis"] == "risk" for row in risk_failed))

    def _write_config(self) -> None:
        payload = {
            "schema_version": "xunce-risk-coverage-cost-quantization-audit-config/v1",
            "source_bound_coverage_root": str(self.bound_root),
            "required_scenario_count": 24,
            "coverage_denominator_cells": 1000,
            "coverage_margin": 0.0,
            "risk_margin": 0.01,
            "cost_margin": 0.0,
            "budget_margin": 0.0,
            "path_budget_m": 5000.0,
            "min_roi_group_with_safe_efficient_candidate": 3,
            "canary_traffic_fraction": 0.0,
        }
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_bound_root(self, *, unvalidated_positive: bool = False, risk_regression_only: bool = False) -> None:
        self.bound_root.mkdir(parents=True, exist_ok=True)
        scenarios = []
        slices = []
        for index in range(24):
            scenario_id = f"scenario_{index:03d}"
            roi_group = f"roi_{index % 8}"
            risks = [0.10, 0.09, 0.30]
            costs = [10.0, 8.0, 30.0]
            if risk_regression_only:
                risks = [0.10, 0.20, 0.30]
                costs = [10.0, 8.0, 9.0]
            candidates = [
                self._candidate(index, 0, endpoint=10, path_line=20, total=20, coverage=0.02, path_cost=costs[0], risk=risks[0]),
                self._candidate(index, 1, endpoint=30, path_line=50, total=60, coverage=0.06, path_cost=costs[1], risk=risks[1], unvalidated=unvalidated_positive),
                self._candidate(index, 2, endpoint=60, path_line=120, total=140, coverage=0.14, path_cost=costs[2], risk=risks[2], unvalidated=unvalidated_positive),
            ]
            scenarios.append(
                {
                    "scenario_id": scenario_id,
                    "roi_group": roi_group,
                    "incumbent_selected_action_index": 0,
                    "incumbent_selection_source": "true_checkpoint_inference",
                    "xunce_selected_action_index": 1,
                    "xunce_selection_source": "true_checkpoint_inference",
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
                    "contract": "contract.json",
                    "sidecar": "sidecar.json",
                }
            )
        self._write_json(
            self.bound_root / "xunce-true-incumbent-selection-binding-summary.json",
            {
                "schema_version": "xunce-true-incumbent-selection-binding-summary/v1",
                "status": "passed",
                "true_incumbent_selection_bound": True,
                "fallback_action_index_0_count": 0,
                "canary_traffic_fraction": 0.0,
            },
        )
        self._write_json(
            self.bound_root / "xunce-candidate-level-coverage-opportunity-summary.json",
            {
                "schema_version": "xunce-candidate-level-coverage-opportunity-summary/v1",
                "status": "passed",
                "candidate_coverage_spread_range": 0.12,
                "canary_traffic_fraction": 0.0,
            },
        )
        self._write_json(
            self.bound_root / "xunce-high-fidelity-real-map-roi-expansion-summary.json",
            {
                "schema_version": "xunce-high-fidelity-real-map-roi-expansion-summary/v1",
                "status": "passed",
                "slice_count": len(slices),
                "roi_group_count": 8,
                "canary_traffic_fraction": 0.0,
            },
        )
        self._write_jsonl(self.bound_root / "xunce-high-fidelity-real-map-slices.jsonl", slices)
        self._write_json(
            self.bound_root / "xunce-high-fidelity-path-feedback-audit.json",
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
    def _candidate(
        scenario_index: int,
        action_index: int,
        *,
        endpoint: int,
        path_line: int,
        total: int,
        coverage: float,
        path_cost: float,
        risk: float,
        unvalidated: bool = False,
    ) -> dict:
        row = {
            "action_index": action_index,
            "cell": [scenario_index * 20 + action_index, action_index],
            "reachable": True,
            "path_cost": path_cost,
            "risk": risk,
            "endpoint_coverage_delta": endpoint / 1000.0,
            "path_line_coverage_delta": path_line / 1000.0,
            "expected_new_coverage_cell_count": total,
            "expected_coverage_rate_delta": total / 1000.0,
            "roi_weighted_coverage_delta": coverage,
            "revisit_penalty": 0.0,
            "coverage_overlap_count": 0,
            "coverage_overlap_ratio": 0.0,
        }
        if unvalidated and action_index in (1, 2):
            row["proposal_validated_by_path_feedback"] = False
        return row

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
