import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class XunceStage18iEvidenceClosureAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="xunce-stage18i-closure-"))
        self.stage18i_root = self.temp_dir / "stage18i"
        self.comparison_root = self.temp_dir / "stage18b"
        self.binding_root = self.temp_dir / "stage18g0"
        self.quantization_root = self.temp_dir / "stage18h0"
        self.oracle_root = self.temp_dir / "stage18f"
        self.coverage_root = self.temp_dir / "stage18c"
        self.output_root = self.temp_dir / "closure"
        self.config_path = self.temp_dir / "config.json"
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_closure_audit_summarizes_complete_after_stage18i_evidence(self) -> None:
        from scripts.run_xunce_stage18i_evidence_closure_audit import run_xunce_stage18i_evidence_closure_audit

        self._write_complete_evidence()

        summary = run_xunce_stage18i_evidence_closure_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertTrue(summary["stage18i_candidate_generation_passed"])
        self.assertTrue(summary["true_model_inference_executed"])
        self.assertTrue(summary["true_incumbent_selection_bound"])
        self.assertEqual(summary["safe_efficient_candidate_count"], 18)
        self.assertEqual(summary["roi_group_with_safe_efficient_candidate_count"], 6)
        self.assertTrue(summary["oracle_separable"])
        self.assertEqual(summary["cost_aware_oracle_efficiency_regression_count"], 0)
        self.assertFalse(summary["xunce_coverage_advantage_established"])
        self.assertTrue(summary["comparison_allowed"])
        self.assertEqual(summary["next_required_change"], "review_xunce_incumbent_comparison_metrics")
        self.assertIn("xunce_coverage_advantage_not_established", summary["diagnostic_reason_codes"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])
        self.assertEqual(summary["canary_traffic_fraction"], 0.0)
        self.assertTrue((self.output_root / "xunce-stage18i-evidence-closure-summary.json").is_file())
        self.assertTrue((self.output_root / "xunce-stage18i-evidence-closure-report.md").is_file())

    def test_missing_artifacts_fail_with_explicit_route(self) -> None:
        from scripts.run_xunce_stage18i_evidence_closure_audit import run_xunce_stage18i_evidence_closure_audit

        summary = run_xunce_stage18i_evidence_closure_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertFalse(summary["stage18i_candidate_generation_passed"])
        self.assertIn("missing_stage18i_candidate_generation", summary["reason_codes"])
        self.assertIn("missing_after_stage18i_model_inference", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "run_xunce_risk_constrained_frontier_nbv_candidate_generation")

    def test_oracle_separable_but_xunce_not_better_routes_to_metric_review(self) -> None:
        from scripts.run_xunce_stage18i_evidence_closure_audit import run_xunce_stage18i_evidence_closure_audit

        self._write_complete_evidence(oracle_separable=True, xunce_advantage=False)

        summary = run_xunce_stage18i_evidence_closure_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["next_required_change"], "review_xunce_incumbent_comparison_metrics")
        self.assertIn("xunce_coverage_advantage_not_established", summary["diagnostic_reason_codes"])

    def test_oracle_not_separable_is_diagnostic_and_routes_to_metric_review(self) -> None:
        from scripts.run_xunce_stage18i_evidence_closure_audit import run_xunce_stage18i_evidence_closure_audit

        self._write_complete_evidence(oracle_separable=False, xunce_advantage=False)

        summary = run_xunce_stage18i_evidence_closure_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["next_required_change"], "review_xunce_incumbent_comparison_metrics")
        self.assertIn("oracle_not_separable", summary["diagnostic_reason_codes"])

    def test_closure_accepts_stage18i2_candidate_repair_summary(self) -> None:
        from scripts.run_xunce_stage18i_evidence_closure_audit import run_xunce_stage18i_evidence_closure_audit

        self._write_complete_evidence(stage18i2=True, oracle_separable=False, xunce_advantage=False)

        summary = run_xunce_stage18i_evidence_closure_audit(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertTrue(summary["stage18i_candidate_generation_passed"])
        self.assertEqual(summary["next_required_change"], "review_xunce_incumbent_comparison_metrics")

    def _write_config(self) -> None:
        payload = {
            "schema_version": "xunce-stage18i-evidence-closure-audit-config/v1",
            "stage18i_candidate_generation_root": str(self.stage18i_root),
            "after_stage18i_model_inference_root": str(self.comparison_root),
            "after_stage18i_true_incumbent_binding_root": str(self.binding_root),
            "after_stage18i_quantization_root": str(self.quantization_root),
            "after_stage18i_oracle_separability_root": str(self.oracle_root),
            "after_stage18i_coverage_comparison_root": str(self.coverage_root),
            "canary_traffic_fraction": 0.0,
        }
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_complete_evidence(self, *, oracle_separable: bool = True, xunce_advantage: bool = False, stage18i2: bool = False) -> None:
        stage18i_filename = (
            "xunce-risk-aware-frontier-nbv-candidate-repair-summary.json"
            if stage18i2
            else "xunce-risk-constrained-frontier-nbv-candidate-generation-summary.json"
        )
        self._write_json(
            self.stage18i_root / stage18i_filename,
            {
                "schema_version": "xunce-risk-aware-frontier-nbv-candidate-repair-summary/v1"
                if stage18i2
                else "xunce-risk-constrained-frontier-nbv-candidate-generation-summary/v1",
                "status": "passed",
                "safe_efficient_candidate_count": 18,
                "roi_group_with_safe_efficient_candidate_count": 6,
                "next_required_change": "rerun_true_model_inference_and_binding",
                "canary_traffic_fraction": 0.0,
            },
        )
        self._write_json(
            self.comparison_root / "xunce-high-fidelity-real-map-comparison-summary.json",
            {
                "schema_version": "xunce-high-fidelity-real-map-comparison-summary/v1",
                "status": "passed",
                "true_model_inference_executed": True,
                "proxy_selection_used": False,
                "xunce_checkpoint_loaded": True,
                "incumbent_checkpoint_loaded": True,
                "canary_traffic_fraction": 0.0,
            },
        )
        self._write_json(
            self.binding_root / "xunce-true-incumbent-selection-binding-summary.json",
            {
                "schema_version": "xunce-true-incumbent-selection-binding-summary/v1",
                "status": "passed",
                "true_incumbent_selection_bound": True,
                "fallback_action_index_0_count": 0,
                "candidate_cell_mismatch_count": 0,
                "canary_traffic_fraction": 0.0,
            },
        )
        self._write_json(
            self.quantization_root / "xunce-risk-coverage-cost-quantization-summary.json",
            {
                "schema_version": "xunce-risk-coverage-cost-quantization-summary/v1",
                "status": "passed",
                "safe_efficient_candidate_count": 18,
                "roi_group_with_safe_efficient_candidate_count": 6,
                "fallback_action_index_0_count": 0,
                "open_grid_fallback_count": 0,
                "next_required_change": "rerun_oracle_separability_with_quantized_root",
                "canary_traffic_fraction": 0.0,
            },
        )
        self._write_json(
            self.oracle_root / "xunce-oracle-separability-summary.json",
            {
                "schema_version": "xunce-oracle-separability-summary/v1",
                "status": "passed",
                "oracle_separable": oracle_separable,
                "cost_aware_oracle_efficiency_regression_count": 0,
                "safe_efficient_opportunity_count": 18,
                "next_required_change": "run_stage18c_v2_with_refined_cost_efficient_root" if oracle_separable else "expand_roi_or_map_complexity",
                "canary_traffic_fraction": 0.0,
            },
        )
        self._write_json(
            self.coverage_root / "xunce-exploration-coverage-comparison-summary.json",
            {
                "schema_version": "xunce-exploration-coverage-comparison-summary/v1",
                "status": "passed",
                "xunce_coverage_advantage_established": xunce_advantage,
                "true_model_inference_executed": True,
                "proxy_selection_used": False,
                "model_inference_mask_violation_count": 0,
                "open_grid_fallback_count": 0,
                "next_required_change": "xunce_training_or_adapter_iteration_required",
                "canary_traffic_fraction": 0.0,
            },
        )

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
