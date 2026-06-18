import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class XunceRiskAwareFrontierNbvCandidateRepairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="xunce-frontier-nbv-repair-"))
        self.stage18a_root = self.temp_dir / "stage18a"
        self.binding_root = self.temp_dir / "stage18g0"
        self.quantization_root = self.temp_dir / "stage18h0"
        self.output_root = self.temp_dir / "stage18i2"
        self.config_path = self.temp_dir / "config.json"
        self._write_stage18a_root()
        self._write_true_binding_root()
        self._write_quantization_root()
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_generates_true_frontier_and_roi_boundary_candidates_without_action0_fallback(self) -> None:
        from scripts.run_xunce_risk_aware_frontier_nbv_candidate_repair import (
            run_xunce_risk_aware_frontier_nbv_candidate_repair,
        )

        source_payload = json.loads((self.stage18a_root / "xunce-high-fidelity-path-feedback-audit.json").read_text(encoding="utf-8"))

        summary = run_xunce_risk_aware_frontier_nbv_candidate_repair(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["next_required_change"], "rerun_true_model_inference_and_binding")
        self.assertEqual(summary["scenario_count"], 24)
        self.assertGreaterEqual(summary["validated_candidate_count"], 72)
        self.assertGreater(summary["frontier_boundary_candidate_count"], 0)
        self.assertGreater(summary["roi_undercovered_boundary_candidate_count"], 0)
        self.assertGreater(summary["safe_efficient_candidate_count"], 0)
        self.assertGreaterEqual(summary["roi_group_with_safe_efficient_candidate_count"], 3)
        self.assertEqual(summary["proposal_unvalidated_positive_count"], 0)
        self.assertEqual(summary["path_feedback_validation_missing_count"], 0)
        self.assertEqual(summary["open_grid_fallback_count"], 0)
        self.assertEqual(summary["fallback_action_index_0_count"], 0)
        self.assertFalse(summary["metric_coupling_detected"])
        self.assertEqual(summary["canary_traffic_fraction"], 0.0)
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])

        expected_files = (
            "xunce-risk-aware-frontier-nbv-candidate-repair-summary.json",
            "xunce-risk-aware-frontier-nbv-proposals.jsonl",
            "xunce-risk-aware-frontier-nbv-validated-candidates.jsonl",
            "xunce-risk-aware-frontier-nbv-rejection-audit.json",
            "xunce-risk-aware-frontier-nbv-report.md",
            "xunce-high-fidelity-real-map-roi-expansion-summary.json",
            "xunce-high-fidelity-real-map-slices.jsonl",
            "xunce-high-fidelity-path-feedback-audit.json",
            "xunce-candidate-level-coverage-opportunity-summary.json",
        )
        for filename in expected_files:
            self.assertTrue((self.output_root / filename).is_file(), filename)

        generated = json.loads((self.output_root / "xunce-high-fidelity-path-feedback-audit.json").read_text(encoding="utf-8"))
        first_scenario = generated["scenarios"][0]
        self.assertNotIn("incumbent_selected_action_index", first_scenario)
        self.assertNotIn("xunce_selected_action_index", first_scenario)
        candidates = first_scenario["path_feedback"]["candidates"]
        self.assertGreaterEqual(len(candidates), 3)
        self.assertEqual([row["action_index"] for row in candidates], list(range(len(candidates))))
        sources = {row["frontier_candidate_source"] for row in candidates}
        self.assertIn("incumbent_neighborhood", sources)
        self.assertIn("frontier_boundary", sources)
        self.assertIn("roi_undercovered_boundary", sources)
        self.assertNotEqual(sources, {"incumbent_neighborhood"})

        for row in candidates:
            self.assertEqual(row["true_incumbent_selected_action_index"], 2)
            self.assertNotEqual(row["incumbent_selection_source"], "fallback_action_index_0")
            self.assertEqual(row["candidate_generation_source"], "risk_aware_frontier_nbv_candidate_repair/v1")
            self.assertEqual(row["coverage_source"], "geometric_counterfactual_from_risk_aware_frontier_nbv_repair/v1")
            self.assertEqual(row["coverage_opportunity_source"], row["coverage_source"])
            self.assertEqual(row["coverage_cell_set_kind"], "path_line_plus_endpoint_union")
            self.assertEqual(row["coverage_dedupe_scope"], "scenario_step_new_cells")
            self.assertTrue(row["proposal_validated_by_path_feedback"])
            self.assertFalse(row["proposal_only"])
            self.assertEqual(row["safe_efficient_opportunity"], row["safe_efficient_candidate"])

        safe = [row for row in candidates if row["safe_efficient_candidate"]]
        self.assertTrue(safe)
        self.assertTrue(all(row["frontier_candidate_source"] == "frontier_boundary" for row in safe))

        unchanged_payload = json.loads((self.stage18a_root / "xunce-high-fidelity-path-feedback-audit.json").read_text(encoding="utf-8"))
        self.assertEqual(unchanged_payload, source_payload)

    def test_unvalidated_positive_proposal_fails_and_never_becomes_safe(self) -> None:
        from scripts.run_xunce_risk_aware_frontier_nbv_candidate_repair import (
            run_xunce_risk_aware_frontier_nbv_candidate_repair,
        )

        self._write_stage18a_root(unvalidated_positive=True)
        self._write_true_binding_root(unvalidated_positive=True)

        summary = run_xunce_risk_aware_frontier_nbv_candidate_repair(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertGreater(summary["proposal_unvalidated_positive_count"], 0)
        self.assertGreater(summary["path_feedback_validation_missing_count"], 0)
        self.assertIn("path_feedback_validation_missing", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "repair_path_feedback_candidate_validation")

        proposals = self._read_jsonl(self.output_root / "xunce-risk-aware-frontier-nbv-proposals.jsonl")
        unvalidated = [row for row in proposals if row.get("proposal_only") and row.get("expected_new_coverage_cell_count", 0) > 0]
        self.assertTrue(unvalidated)
        self.assertTrue(all(not row.get("safe_efficient_candidate", False) for row in unvalidated))
        self.assertTrue(all(not row.get("safe_efficient_opportunity", False) for row in unvalidated))

    def test_missing_true_incumbent_binding_routes_to_binding_stage(self) -> None:
        from scripts.run_xunce_risk_aware_frontier_nbv_candidate_repair import (
            run_xunce_risk_aware_frontier_nbv_candidate_repair,
        )

        self._write_config(source_true_incumbent_binding_root=self.temp_dir / "missing-binding")

        summary = run_xunce_risk_aware_frontier_nbv_candidate_repair(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_true_incumbent_binding", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "run_true_incumbent_selection_binding")

    def test_frontier_and_roi_boundary_missing_is_diagnostic_and_allows_recomparison(self) -> None:
        from scripts.run_xunce_risk_aware_frontier_nbv_candidate_repair import (
            run_xunce_risk_aware_frontier_nbv_candidate_repair,
        )

        self._write_stage18a_root(flat_neighbors=True)
        self._write_true_binding_root(flat_neighbors=True)

        summary = run_xunce_risk_aware_frontier_nbv_candidate_repair(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertTrue(summary["comparison_allowed"])
        self.assertEqual(summary["next_required_change"], "rerun_true_model_inference_and_binding")
        self.assertEqual(summary["blocking_reason_codes"], [])
        self.assertIn("frontier_boundary_candidate_missing", summary["diagnostic_reason_codes"])
        self.assertEqual(summary["diagnostic_recommended_change"], "repair_frontier_nbv_sampling")

    def _write_config(
        self,
        *,
        source_roi_expansion_root: Path | None = None,
        source_true_incumbent_binding_root: Path | None = None,
        source_quantization_root: Path | None = None,
    ) -> None:
        payload = {
            "schema_version": "xunce-risk-aware-frontier-nbv-candidate-repair-config/v1",
            "source_roi_expansion_root": str(source_roi_expansion_root or self.stage18a_root),
            "source_true_incumbent_binding_root": str(source_true_incumbent_binding_root or self.binding_root),
            "source_quantization_root": str(source_quantization_root or self.quantization_root),
            "required_scenario_count": 24,
            "max_candidates_per_scenario": 6,
            "coverage_denominator_cells": 1000,
            "risk_margin": 0.01,
            "cost_margin": 0.0,
            "path_budget_m": 5000.0,
            "allow_open_grid_fallback": False,
            "canary_traffic_fraction": 0.0,
        }
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_stage18a_root(self, *, unvalidated_positive: bool = False, flat_neighbors: bool = False) -> None:
        self.stage18a_root.mkdir(parents=True, exist_ok=True)
        scenarios = []
        slices = []
        for index in range(24):
            scenario_id = f"scenario_{index:03d}"
            roi_group = f"roi_{index % 8}"
            scenarios.append(
                {
                    "scenario_id": scenario_id,
                    "roi_group": roi_group,
                    "scenario_group": roi_group,
                    "start_cell": [index * 40, 0],
                    "open_grid_fallback_used": False,
                    "path_feedback": {"candidates": self._candidates(index, unvalidated_positive=unvalidated_positive, flat_neighbors=flat_neighbors)},
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
                    "contract": str(self.stage18a_root / f"{scenario_id}.contract.json"),
                    "sidecar": str(self.stage18a_root / f"{scenario_id}.sidecar.json"),
                }
            )
        self._write_json(
            self.stage18a_root / "xunce-high-fidelity-real-map-roi-expansion-summary.json",
            {
                "schema_version": "xunce-high-fidelity-real-map-roi-expansion-summary/v1",
                "status": "passed",
                "slice_count": len(slices),
                "roi_group_count": 8,
                "canary_traffic_fraction": 0.0,
            },
        )
        self._write_jsonl(self.stage18a_root / "xunce-high-fidelity-real-map-slices.jsonl", slices)
        self._write_json(
            self.stage18a_root / "xunce-high-fidelity-path-feedback-audit.json",
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

    def _write_true_binding_root(self, *, unvalidated_positive: bool = False, flat_neighbors: bool = False) -> None:
        self.binding_root.mkdir(parents=True, exist_ok=True)
        scenarios = []
        for index in range(24):
            scenario_id = f"scenario_{index:03d}"
            roi_group = f"roi_{index % 8}"
            scenarios.append(
                {
                    "scenario_id": scenario_id,
                    "roi_group": roi_group,
                    "start_cell": [index * 40, 0],
                    "incumbent_selected_action_index": 2,
                    "incumbent_selection_source": "true_checkpoint_inference",
                    "xunce_selected_action_index": 1,
                    "xunce_selection_source": "true_checkpoint_inference",
                    "path_feedback": {"candidates": self._candidates(index, unvalidated_positive=unvalidated_positive, flat_neighbors=flat_neighbors)},
                    "open_grid_fallback_used": False,
                }
            )
        self._write_json(
            self.binding_root / "xunce-true-incumbent-selection-binding-summary.json",
            {
                "schema_version": "xunce-true-incumbent-selection-binding-summary/v1",
                "status": "passed",
                "true_incumbent_selection_bound": True,
                "incumbent_selected_action_index_missing_count": 0,
                "fallback_action_index_0_count": 0,
                "candidate_cell_mismatch_count": 0,
                "true_model_inference_executed": True,
                "proxy_selection_used": False,
                "canary_traffic_fraction": 0.0,
            },
        )
        self._write_json(
            self.binding_root / "xunce-high-fidelity-path-feedback-audit.json",
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

    def _write_quantization_root(self) -> None:
        self.quantization_root.mkdir(parents=True, exist_ok=True)
        self._write_json(
            self.quantization_root / "xunce-risk-coverage-cost-quantization-summary.json",
            {
                "schema_version": "xunce-risk-coverage-cost-quantization-summary/v1",
                "status": "failed",
                "next_required_change": "repair_risk_aware_candidate_generation",
                "safe_efficient_candidate_count": 0,
                "canary_traffic_fraction": 0.0,
            },
        )

    def _candidates(self, scenario_index: int, *, unvalidated_positive: bool = False, flat_neighbors: bool = False) -> list[dict]:
        base = scenario_index * 40
        cells = [[base + 1, 0], [base + 8, 1], [base + 2, 0], [base + 22, 5], [base + 14, 4]]
        if flat_neighbors:
            cells = [[base + 1, 0], [base + 2, 0], [base + 3, 0], [base + 4, 0], [base + 5, 0]]
        costs = [7.0, 8.0, 10.0, 18.0, 14.0]
        risks = [0.06, 0.07, 0.08, 0.24, 0.06]
        rows = []
        for action_index, cell in enumerate(cells):
            row = {
                "action_index": action_index,
                "cell": cell,
                "reachable": True,
                "path_cost": costs[action_index],
                "risk": risks[action_index],
                "open_grid_fallback_used": False,
                "proposal_validated_by_path_feedback": True,
            }
            if unvalidated_positive and action_index == 1:
                row["proposal_validated_by_path_feedback"] = False
            rows.append(row)
        return rows

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
