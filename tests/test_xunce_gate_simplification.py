from __future__ import annotations

import sys
import unittest
from pathlib import Path


class XunceGateSimplificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)

    def test_stage18i2_diagnostics_do_not_block_model_recomparison(self) -> None:
        from scripts.run_xunce_risk_aware_frontier_nbv_candidate_repair import _decision

        decision = _decision(
            {"required_scenario_count": 24},
            {"reason_codes": []},
            {
                "reason_codes": [],
                "candidate_cell_mismatch_count": 0,
                "path_feedback_validation_missing_count": 0,
                "open_grid_fallback_count": 0,
                "validated_candidate_count": 72,
                "candidate_count": 72,
                "frontier_boundary_candidate_count": 0,
                "roi_undercovered_boundary_candidate_count": 12,
                "candidate_coverage_spread_range": 0.001,
                "roi_group_with_nonzero_spread_count": 1,
                "safe_efficient_candidate_count": 0,
                "roi_group_with_safe_efficient_candidate_count": 0,
                "coverage_positive_but_risk_regressive_count": 4,
                "coverage_positive_but_cost_regressive_count": 8,
            },
        )

        self.assertEqual(decision["status"], "passed")
        self.assertTrue(decision["evidence_authenticity_gate_passed"])
        self.assertTrue(decision["candidate_validity_gate_passed"])
        self.assertTrue(decision["comparison_allowed"])
        self.assertEqual(decision["blocking_reason_codes"], [])
        self.assertEqual(decision["next_required_change"], "rerun_true_model_inference_and_binding")
        self.assertIn("frontier_boundary_candidate_missing", decision["diagnostic_reason_codes"])
        self.assertIn("safe_efficient_candidate_missing", decision["diagnostic_reason_codes"])
        self.assertEqual(decision["diagnostic_recommended_change"], "repair_frontier_nbv_sampling")

    def test_stage18h0_cost_and_risk_regression_are_diagnostics_only(self) -> None:
        from scripts.run_xunce_risk_coverage_cost_quantization_audit import _decision

        decision = _decision(
            {"min_roi_group_with_safe_efficient_candidate": 3},
            {"binding_summary": {"true_incumbent_selection_bound": True}},
            {
                "reason_codes": [],
                "fallback_action_index_0_count": 0,
                "path_feedback_validation_missing_count": 0,
                "metric_coupling_detected": False,
                "normalization_instability_count": 0,
                "candidate_count": 72,
                "valid_candidate_count": 72,
                "safe_efficient_candidate_count": 0,
                "roi_group_with_safe_efficient_candidate_count": 0,
                "coverage_positive_candidate_count": 24,
                "coverage_positive_but_risk_regressive_count": 8,
                "coverage_positive_but_cost_regressive_count": 24,
            },
        )

        self.assertEqual(decision["status"], "passed")
        self.assertTrue(decision["evidence_authenticity_gate_passed"])
        self.assertTrue(decision["candidate_validity_gate_passed"])
        self.assertTrue(decision["comparison_allowed"])
        self.assertEqual(decision["blocking_reason_codes"], [])
        self.assertEqual(decision["next_required_change"], "rerun_oracle_separability_with_quantized_root")
        self.assertIn("safe_efficient_candidate_missing", decision["diagnostic_reason_codes"])
        self.assertIn("coverage_positive_but_cost_regressive", decision["diagnostic_reason_codes"])
        self.assertEqual(decision["diagnostic_recommended_change"], "repair_risk_aware_candidate_generation")

    def test_stage18f_cost_aware_oracle_uses_valid_candidates_not_safe_only(self) -> None:
        from scripts.run_xunce_oracle_separability_benchmark import _decision, _oracle_index

        candidates = [
            {
                "reachable": True,
                "proposal_only": False,
                "proposal_validated_by_path_feedback": True,
                "open_grid_fallback_used": False,
                "safe_efficient_candidate": False,
                "safe_efficient_opportunity": False,
                "roi_weighted_coverage_delta": 0.2,
                "path_cost": 10.0,
                "risk": 0.1,
            },
            {
                "reachable": True,
                "proposal_only": False,
                "proposal_validated_by_path_feedback": True,
                "open_grid_fallback_used": False,
                "safe_efficient_candidate": True,
                "safe_efficient_opportunity": True,
                "roi_weighted_coverage_delta": 0.01,
                "path_cost": 10.0,
                "risk": 0.1,
            },
        ]

        self.assertEqual(_oracle_index(candidates, mode="cost_aware"), 0)

        decision = _decision(
            {
                "reason_codes": [],
                "source_cost_efficient_refinement_detected": True,
                "safe_efficient_opportunity_available": False,
                "xunce_coverage_advantage_established": False,
                "xunce_efficiency_regression_count": 0,
                "oracle_separable": False,
            }
        )
        self.assertEqual(decision["status"], "passed")
        self.assertTrue(decision["comparison_allowed"])
        self.assertEqual(decision["blocking_reason_codes"], [])
        self.assertEqual(decision["next_required_change"], "run_stage18c_v2_model_comparison")
        self.assertIn("oracle_not_separable", decision["diagnostic_reason_codes"])

    def test_stage18c_successful_comparison_routes_to_metric_review_even_without_advantage(self) -> None:
        from scripts.run_xunce_high_fidelity_exploration_coverage_comparison import _decision

        decision = _decision(
            source={"read_reason_codes": []},
            boundary={"reason_codes": []},
            source_match={"reason_codes": []},
            model_inference={
                "reason_codes": [],
                "true_model_inference_executed": True,
                "proxy_selection_used": False,
                "model_inference_mask_violation_count": 0,
            },
            comparison={
                "xunce_coverage_return_delta_vs_incumbent": 0.0,
                "xunce_coverage_curve_auc_delta_vs_incumbent": 0.0,
                "xunce_safety_regression_count": 0,
                "xunce_efficiency_regression_count": 3,
                "open_grid_fallback_count": 0,
                "coverage_gain_per_risk_delta_vs_incumbent": -0.1,
                "coverage_gain_per_path_cost_delta_vs_incumbent": -0.1,
                "incumbent_oracle_regret": 1.0,
                "xunce_oracle_regret": 2.0,
            },
        )

        self.assertEqual(decision["status"], "passed")
        self.assertTrue(decision["comparison_allowed"])
        self.assertEqual(decision["blocking_reason_codes"], [])
        self.assertEqual(decision["next_required_change"], "review_xunce_incumbent_comparison_metrics")
        self.assertIn("xunce_coverage_advantage_not_established", decision["diagnostic_reason_codes"])

    def test_stage18i_closure_does_not_fail_on_diagnostic_metrics(self) -> None:
        from scripts.run_xunce_stage18i_evidence_closure_audit import _decision

        decision = _decision(
            {
                "reason_codes": [],
                "stage18i_candidate_generation_passed": True,
                "true_model_inference_executed": True,
                "true_incumbent_selection_bound": True,
                "safe_efficient_candidate_count": 0,
                "roi_group_with_safe_efficient_candidate_count": 0,
                "oracle_separable": False,
                "xunce_coverage_advantage_established": False,
            }
        )

        self.assertEqual(decision["status"], "passed")
        self.assertTrue(decision["evidence_authenticity_gate_passed"])
        self.assertTrue(decision["candidate_validity_gate_passed"])
        self.assertTrue(decision["comparison_allowed"])
        self.assertEqual(decision["blocking_reason_codes"], [])
        self.assertEqual(decision["next_required_change"], "review_xunce_incumbent_comparison_metrics")
        self.assertIn("safe_efficient_candidate_missing", decision["diagnostic_reason_codes"])
        self.assertIn("oracle_not_separable", decision["diagnostic_reason_codes"])


if __name__ == "__main__":
    unittest.main()
