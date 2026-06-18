import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class XunceRiskConstrainedFrontierNbvCandidateGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="xunce-frontier-nbv-"))
        self.source_root = self.temp_dir / "stage18a"
        self.quantization_root = self.temp_dir / "stage18h0"
        self.output_root = self.temp_dir / "stage18i"
        self.config_path = self.temp_dir / "config.json"
        self._write_stage18a_evidence()
        self._write_quantization_summary()
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_generates_frontier_nbv_candidates_and_compatible_root(self) -> None:
        from scripts.run_xunce_risk_constrained_frontier_nbv_candidate_generation import (
            run_xunce_risk_constrained_frontier_nbv_candidate_generation,
        )

        original_payload = json.loads((self.source_root / "xunce-high-fidelity-path-feedback-audit.json").read_text(encoding="utf-8"))

        summary = run_xunce_risk_constrained_frontier_nbv_candidate_generation(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["next_required_change"], "rerun_true_model_inference_and_binding")
        self.assertEqual(summary["scenario_count"], 24)
        self.assertGreaterEqual(summary["validated_candidate_count"], 72)
        self.assertGreater(summary["candidate_coverage_spread_range"], 0.005)
        self.assertGreaterEqual(summary["roi_group_with_nonzero_spread_count"], 3)
        self.assertGreater(summary["safe_efficient_candidate_count"], 0)
        self.assertGreaterEqual(summary["roi_group_with_safe_efficient_candidate_count"], 3)
        self.assertEqual(summary["proposal_unvalidated_positive_count"], 0)
        self.assertEqual(summary["path_feedback_validation_missing_count"], 0)
        self.assertEqual(summary["open_grid_fallback_count"], 0)
        self.assertEqual(summary["fallback_action_index_0_count"], 0)
        self.assertFalse(summary["metric_coupling_detected"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])
        self.assertEqual(summary["canary_traffic_fraction"], 0.0)

        expected_files = (
            "xunce-risk-constrained-frontier-nbv-candidate-generation-summary.json",
            "xunce-frontier-nbv-proposals.jsonl",
            "xunce-frontier-nbv-validated-candidates.jsonl",
            "xunce-frontier-nbv-rejection-audit.json",
            "xunce-frontier-nbv-roi-summary.json",
            "xunce-frontier-nbv-pareto-audit.json",
            "xunce-frontier-nbv-manifest.json",
            "xunce-frontier-nbv-report.md",
            "xunce-high-fidelity-real-map-roi-expansion-summary.json",
            "xunce-high-fidelity-real-map-slices.jsonl",
            "xunce-high-fidelity-path-feedback-audit.json",
            "xunce-candidate-level-coverage-opportunity-summary.json",
        )
        for filename in expected_files:
            self.assertTrue((self.output_root / filename).is_file(), filename)

        compatible_summary = json.loads((self.output_root / "xunce-high-fidelity-real-map-roi-expansion-summary.json").read_text(encoding="utf-8"))
        self.assertEqual(compatible_summary["status"], "passed")
        self.assertEqual(compatible_summary["next_required_change"], "xunce_high_fidelity_real_map_policy_comparison")
        self.assertEqual(compatible_summary["frontier_nbv_candidate_generation_status"], "passed")

        generated = json.loads((self.output_root / "xunce-high-fidelity-path-feedback-audit.json").read_text(encoding="utf-8"))
        first_scenario = generated["scenarios"][0]
        self.assertNotIn("incumbent_selected_action_index", first_scenario)
        self.assertNotIn("xunce_selected_action_index", first_scenario)
        candidates = first_scenario["path_feedback"]["candidates"]
        self.assertGreaterEqual(len(candidates), 3)
        self.assertLessEqual(len(candidates), 6)
        self.assertEqual([row["action_index"] for row in candidates], list(range(len(candidates))))
        safe = [row for row in candidates if row["safe_efficient_candidate"]]
        self.assertTrue(safe)
        self.assertTrue(all(row["safe_efficient_opportunity"] == row["safe_efficient_candidate"] for row in candidates))
        candidate = candidates[0]
        self.assertEqual(candidate["coverage_opportunity_source"], "geometric_counterfactual_from_frontier_nbv_candidate/v1")
        self.assertEqual(candidate["coverage_cell_set_kind"], "path_line_plus_endpoint_union")
        self.assertEqual(candidate["coverage_dedupe_scope"], "scenario_step_new_cells")
        self.assertIn(candidate["frontier_candidate_source"], {"frontier_boundary", "roi_undercovered_boundary", "incumbent_neighborhood"})
        self.assertIn("endpoint_coverage_delta", candidate)
        self.assertIn("path_line_coverage_delta", candidate)
        self.assertIn("roi_weighted_coverage_delta", candidate)
        self.assertIn("revisit_penalty", candidate)
        self.assertIn("risk_guard_passed", candidate)
        self.assertIn("cost_guard_passed", candidate)
        self.assertTrue(candidate["proposal_validated_by_path_feedback"])

        unchanged_payload = json.loads((self.source_root / "xunce-high-fidelity-path-feedback-audit.json").read_text(encoding="utf-8"))
        self.assertEqual(unchanged_payload, original_payload)

    def test_high_coverage_risk_regression_is_not_safe(self) -> None:
        from scripts.run_xunce_risk_constrained_frontier_nbv_candidate_generation import (
            run_xunce_risk_constrained_frontier_nbv_candidate_generation,
        )

        summary = run_xunce_risk_constrained_frontier_nbv_candidate_generation(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        rows = self._read_jsonl(self.output_root / "xunce-frontier-nbv-validated-candidates.jsonl")
        risky_high_coverage = [row for row in rows if row["frontier_candidate_source"] == "roi_undercovered_boundary"]
        self.assertTrue(risky_high_coverage)
        self.assertTrue(any(row["coverage_guard_passed"] and not row["risk_guard_passed"] for row in risky_high_coverage))
        self.assertTrue(all(not row["safe_efficient_candidate"] for row in risky_high_coverage if not row["risk_guard_passed"]))

    def test_unvalidated_positive_proposal_is_rejected(self) -> None:
        from scripts.run_xunce_risk_constrained_frontier_nbv_candidate_generation import (
            run_xunce_risk_constrained_frontier_nbv_candidate_generation,
        )

        self._write_stage18a_evidence(unvalidated_positive=True)

        summary = run_xunce_risk_constrained_frontier_nbv_candidate_generation(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertGreater(summary["proposal_unvalidated_positive_count"], 0)
        self.assertGreater(summary["path_feedback_validation_missing_count"], 0)
        self.assertIn("path_feedback_validation_missing", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "repair_path_feedback_candidate_validation")

        proposals = self._read_jsonl(self.output_root / "xunce-frontier-nbv-proposals.jsonl")
        unvalidated = [row for row in proposals if row.get("proposal_only") and row.get("expected_new_coverage_cell_count", 0) > 0]
        self.assertTrue(unvalidated)
        self.assertTrue(all(not row.get("safe_efficient_candidate", False) for row in unvalidated))
        self.assertTrue(all(not row.get("safe_efficient_opportunity", False) for row in unvalidated))

    def test_missing_path_feedback_metrics_cannot_become_safe(self) -> None:
        from scripts.run_xunce_risk_constrained_frontier_nbv_candidate_generation import (
            run_xunce_risk_constrained_frontier_nbv_candidate_generation,
        )

        self._write_stage18a_evidence(missing_path_feedback_metrics=True)

        summary = run_xunce_risk_constrained_frontier_nbv_candidate_generation(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertGreater(summary["path_feedback_validation_missing_count"], 0)
        self.assertEqual(summary["safe_efficient_candidate_count"], 0)
        self.assertIn("path_feedback_validation_missing", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "repair_path_feedback_candidate_validation")

        proposals = self._read_jsonl(self.output_root / "xunce-frontier-nbv-proposals.jsonl")
        self.assertTrue(proposals)
        self.assertTrue(all(not row.get("safe_efficient_candidate", False) for row in proposals))
        self.assertTrue(all(not row.get("safe_efficient_opportunity", False) for row in proposals))

    def test_open_grid_candidate_is_counted_as_open_grid_rejection(self) -> None:
        from scripts.run_xunce_risk_constrained_frontier_nbv_candidate_generation import (
            run_xunce_risk_constrained_frontier_nbv_candidate_generation,
        )

        self._write_stage18a_evidence(open_grid_candidate=True)

        summary = run_xunce_risk_constrained_frontier_nbv_candidate_generation(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertGreater(summary["open_grid_fallback_count"], 0)
        self.assertIn("open_grid_fallback_candidate", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "repair_path_feedback_candidate_validation")

        audit = json.loads((self.output_root / "xunce-frontier-nbv-rejection-audit.json").read_text(encoding="utf-8"))
        self.assertGreater(audit["rejection_reason_counts"]["open_grid_fallback_candidate"], 0)

    def test_low_spread_routes_to_sampling_repair(self) -> None:
        from scripts.run_xunce_risk_constrained_frontier_nbv_candidate_generation import (
            run_xunce_risk_constrained_frontier_nbv_candidate_generation,
        )

        self._write_stage18a_evidence(flat_candidates=True)

        summary = run_xunce_risk_constrained_frontier_nbv_candidate_generation(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("proposal_coverage_spread_insufficient", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "repair_frontier_nbv_sampling")

    def test_missing_roi_expansion_routes_to_stage18a(self) -> None:
        from scripts.run_xunce_risk_constrained_frontier_nbv_candidate_generation import (
            run_xunce_risk_constrained_frontier_nbv_candidate_generation,
        )

        missing_root = self.temp_dir / "missing-stage18a"
        self._write_config(source_roi_expansion_root=missing_root)

        summary = run_xunce_risk_constrained_frontier_nbv_candidate_generation(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_stage18a_roi_expansion", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "run_xunce_high_fidelity_real_map_roi_expansion")

    def test_missing_quantization_summary_routes_to_quantization_audit(self) -> None:
        from scripts.run_xunce_risk_constrained_frontier_nbv_candidate_generation import (
            run_xunce_risk_constrained_frontier_nbv_candidate_generation,
        )

        self._write_config(source_quantization_root=self.temp_dir / "missing-stage18h0")

        summary = run_xunce_risk_constrained_frontier_nbv_candidate_generation(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_stage18h0_quantization", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "run_xunce_risk_coverage_cost_quantization_audit")

    def _write_config(self, *, source_roi_expansion_root: Path | None = None, source_quantization_root: Path | None = None) -> None:
        payload = {
            "schema_version": "xunce-risk-constrained-frontier-nbv-candidate-generation-config/v1",
            "source_roi_expansion_root": str(source_roi_expansion_root or self.source_root),
            "source_quantization_root": str(source_quantization_root or self.quantization_root),
            "required_scenario_count": 24,
            "max_candidates_per_scenario": 6,
            "frontier_radius_cells": [2, 4, 6],
            "frontier_direction_count": 8,
            "nbv_candidate_pool_limit": 24,
            "coverage_denominator_cells": 1000,
            "risk_margin": 0.01,
            "cost_margin": 0.0,
            "path_budget_m": 5000.0,
            "allow_open_grid_fallback": False,
            "canary_traffic_fraction": 0.0,
        }
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_stage18a_evidence(
        self,
        *,
        flat_candidates: bool = False,
        unvalidated_positive: bool = False,
        missing_path_feedback_metrics: bool = False,
        open_grid_candidate: bool = False,
    ) -> None:
        self.source_root.mkdir(parents=True, exist_ok=True)
        scenarios = []
        slices = []
        for index in range(24):
            scenario_id = f"scenario_{index:03d}"
            roi_group = f"roi_{index % 8}"
            start_cell = [index * 30, 0]
            if flat_candidates:
                cells = [[index * 30 + 1, 0], [index * 30 + 1, 0], [index * 30 + 1, 0]]
                endpoint_values = [10, 10, 10]
                path_values = [20, 20, 20]
                coverage_values = [0.02, 0.02, 0.02]
            else:
                cells = [[index * 30 + 1, 0], [index * 30 + 3, 1], [index * 30 + 8, 2], [index * 30 + 12, 3]]
                endpoint_values = [10, 35, 70, 20]
                path_values = [20, 55, 130, 40]
                coverage_values = [0.02, 0.07, 0.16, 0.04]
            risks = [0.10, 0.09, 0.25, 0.11]
            costs = [10.0, 8.0, 30.0, 9.5]
            candidates = []
            for action_index, cell in enumerate(cells):
                row = {
                    "action_index": action_index,
                    "cell": cell,
                    "reachable": True,
                    "path_cost": costs[action_index],
                    "risk": risks[action_index],
                    "endpoint_coverage_delta": endpoint_values[action_index] / 1000.0,
                    "path_line_coverage_delta": path_values[action_index] / 1000.0,
                    "expected_new_coverage_cell_count": max(endpoint_values[action_index], path_values[action_index]),
                    "expected_coverage_rate_delta": max(endpoint_values[action_index], path_values[action_index]) / 1000.0,
                    "roi_weighted_coverage_delta": coverage_values[action_index],
                    "revisit_penalty": 0.0,
                    "coverage_overlap_count": 0,
                    "coverage_overlap_ratio": 0.0,
                }
                if unvalidated_positive and action_index == 1:
                    row["proposal_validated_by_path_feedback"] = False
                if missing_path_feedback_metrics:
                    row.pop("path_cost", None)
                    row.pop("risk", None)
                if open_grid_candidate and action_index == 1:
                    row["open_grid_fallback_used"] = True
                candidates.append(row)
            scenarios.append(
                {
                    "scenario_id": scenario_id,
                    "roi_group": roi_group,
                    "scenario_group": roi_group,
                    "start_cell": start_cell,
                    "open_grid_fallback_used": False,
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
                "slice_count": len(slices),
                "roi_group_count": 8,
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

    def _write_quantization_summary(self) -> None:
        self.quantization_root.mkdir(parents=True, exist_ok=True)
        self._write_json(
            self.quantization_root / "xunce-risk-coverage-cost-quantization-summary.json",
            {
                "schema_version": "xunce-risk-coverage-cost-quantization-summary/v1",
                "status": "failed",
                "next_required_change": "repair_risk_aware_candidate_generation",
                "safe_efficient_candidate_count": 0,
                "candidate_count": 72,
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
