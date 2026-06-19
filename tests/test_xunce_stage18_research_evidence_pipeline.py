import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class XunceStage18ResearchEvidencePipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="xunce-stage18-pipeline-"))
        self.roi_root = self.temp_dir / "stage18_1"
        self.candidate_root = self.temp_dir / "stage18_2"
        self.model_root = self.temp_dir / "stage18_3_model"
        self.binding_root = self.temp_dir / "stage18_3_binding"
        self.quant_root = self.temp_dir / "stage18_4_quant"
        self.oracle_root = self.temp_dir / "stage18_4_oracle"
        self.coverage_root = self.temp_dir / "stage18_4_coverage"
        self.output_root = self.temp_dir / "pipeline"
        self.config_path = self.temp_dir / "config.json"
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_plan_only_summarizes_complete_stage18_pipeline(self) -> None:
        from scripts.xunce_stage18_pipeline import build_stage18_pipeline_summary, write_stage18_pipeline_artifacts

        self._write_complete_evidence()

        summary = build_stage18_pipeline_summary(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
            plan_only=True,
        )
        write_stage18_pipeline_artifacts(self.output_root, summary)

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["evidence_status"], "passed")
        self.assertEqual(summary["candidate_validity_status"], "passed")
        self.assertEqual(summary["comparison_verdict"], "xunce_advantage_not_established")
        self.assertEqual(summary["overall_conclusion"], "evidence_valid_but_xunce_advantage_not_established")
        self.assertIn("oracle_not_separable", summary["diagnostic_reason_codes"])
        self.assertEqual(summary["release_readiness"], "not_authorized")
        self.assertEqual(summary["training_readiness"], "not_authorized")
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])
        self.assertTrue((self.output_root / "xunce-stage18-pipeline-summary.json").is_file())
        self.assertTrue((self.output_root / "xunce-stage18-resolved-roots.json").is_file())
        self.assertTrue((self.output_root / "xunce-stage18-module-results.jsonl").is_file())
        self.assertTrue((self.output_root / "xunce-stage18-pipeline-report.md").is_file())

    def test_missing_artifacts_are_partial_with_explicit_routes(self) -> None:
        from scripts.xunce_stage18_pipeline import build_stage18_pipeline_summary

        summary = build_stage18_pipeline_summary(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
            plan_only=True,
        )

        self.assertEqual(summary["status"], "partial")
        self.assertEqual(summary["evidence_status"], "partial")
        self.assertEqual(summary["comparison_verdict"], "inconclusive")
        self.assertIn("missing_stage18_1_roi_expansion", summary["missing_reason_codes"])
        self.assertIn("missing_stage18_4_coverage_comparison", summary["missing_reason_codes"])
        self.assertEqual(summary["next_required_change"], "refresh_stage18_research_evidence_pipeline")

    def test_mixed_candidate_roots_block_pipeline(self) -> None:
        from scripts.xunce_stage18_pipeline import build_stage18_pipeline_summary

        self._write_complete_evidence(binding_candidate_root=self.temp_dir / "other_candidate_root")

        summary = build_stage18_pipeline_summary(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
            plan_only=True,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["evidence_status"], "blocked")
        self.assertIn("stale_or_mixed_stage18_roots", summary["blocking_reason_codes"])
        self.assertEqual(summary["next_required_change"], "rerun_stage18_downstream_evidence_for_candidate_root")

    def test_authenticity_failure_remains_hard_blocker(self) -> None:
        from scripts.xunce_stage18_pipeline import build_stage18_pipeline_summary

        self._write_complete_evidence(true_model_inference=False)

        summary = build_stage18_pipeline_summary(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
            plan_only=True,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("true_model_inference_not_executed", summary["blocking_reason_codes"])
        self.assertEqual(summary["comparison_verdict"], "inconclusive")

    def test_runner_writes_summary_and_accepts_plan_only(self) -> None:
        self._write_complete_evidence()

        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_xunce_stage18_research_evidence_pipeline.py",
                "--config",
                str(self.config_path),
                "--output-root",
                str(self.output_root),
                "--repo-root",
                str(self.repo_root),
                "--plan-only",
            ],
            cwd=self.repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("evidence_valid_but_xunce_advantage_not_established", completed.stdout)
        self.assertTrue((self.output_root / "xunce-stage18-pipeline-summary.json").is_file())

    def test_stage_runner_direct_passthrough_supports_plan_only(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_stage.py",
                "--stage",
                "xunce-stage18-research-evidence-pipeline",
                "--plan-only",
                "--dry-run",
            ],
            cwd=self.repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("run_xunce_stage18_research_evidence_pipeline.py", completed.stdout)
        self.assertIn("--plan-only", completed.stdout)
        self.assertNotIn("bash", completed.stdout.lower())
        self.assertNotIn("python3", completed.stdout.lower())
        self.assertNotIn("/home/kai", completed.stdout)

    def _write_config(self) -> None:
        payload = {
            "schema_version": "xunce-stage18-research-evidence-pipeline-config/v1",
            "stage18_1_roi_expansion_root": str(self.roi_root),
            "stage18_2_candidate_root": str(self.candidate_root),
            "stage18_3_model_inference_root": str(self.model_root),
            "stage18_3_true_incumbent_binding_root": str(self.binding_root),
            "stage18_4_quantization_root": str(self.quant_root),
            "stage18_4_oracle_root": str(self.oracle_root),
            "stage18_4_coverage_comparison_root": str(self.coverage_root),
            "canary_traffic_fraction": 0.0,
        }
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_complete_evidence(
        self,
        *,
        binding_candidate_root: Path | None = None,
        true_model_inference: bool = True,
    ) -> None:
        self._write_json(
            self.roi_root / "xunce-high-fidelity-real-map-roi-expansion-summary.json",
            {"schema_version": "xunce-high-fidelity-real-map-roi-expansion-summary/v1", "status": "passed"},
        )
        self._write_json(
            self.candidate_root / "xunce-true-frontier-nbv-candidate-source-summary.json",
            {
                "schema_version": "xunce-true-frontier-nbv-candidate-source-summary/v1",
                "status": "passed",
                "candidate_count": 144,
                "valid_candidate_count": 144,
                "candidate_validation_mode": "in_process_evaluate_candidate_paths",
                "safe_efficient_candidate_count": 120,
                "canary_traffic_fraction": 0.0,
            },
        )
        self._write_json(
            self.model_root / "xunce-high-fidelity-real-map-comparison-summary.json",
            {
                "schema_version": "xunce-high-fidelity-real-map-comparison-summary/v1",
                "status": "passed",
                "true_model_inference_executed": true_model_inference,
                "proxy_selection_used": False,
                "xunce_checkpoint_loaded": True,
                "incumbent_checkpoint_loaded": True,
                "xunce_candidate_advantage_established": False,
                "xunce_better_than_incumbent_count": 0,
                "xunce_worse_than_incumbent_count": 24,
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
                "source_materialized_coverage_root": str((binding_candidate_root or self.candidate_root).resolve()),
                "source_model_inference_root": str(self.model_root.resolve()),
                "canary_traffic_fraction": 0.0,
            },
        )
        self._write_json(
            self.quant_root / "xunce-risk-coverage-cost-quantization-summary.json",
            {
                "schema_version": "xunce-risk-coverage-cost-quantization-summary/v1",
                "status": "passed",
                "candidate_validity_gate_passed": True,
                "source_bound_coverage_root": str(self.binding_root.resolve()),
                "candidate_count": 144,
                "valid_candidate_count": 144,
                "safe_efficient_candidate_count": 0,
                "canary_traffic_fraction": 0.0,
            },
        )
        self._write_json(
            self.oracle_root / "xunce-oracle-separability-summary.json",
            {
                "schema_version": "xunce-oracle-separability-summary/v1",
                "status": "passed",
                "oracle_separable": False,
                "source_materialized_coverage_root": str(self.quant_root.resolve()),
                "greedy_oracle_coverage_return_delta_vs_incumbent": 0.007,
                "cost_aware_oracle_coverage_return_delta_vs_incumbent": 0.0,
                "canary_traffic_fraction": 0.0,
            },
        )
        self._write_json(
            self.coverage_root / "xunce-exploration-coverage-comparison-summary.json",
            {
                "schema_version": "xunce-exploration-coverage-comparison-summary/v1",
                "status": "passed",
                "source_roi_expansion_root": str(self.quant_root.resolve()),
                "xunce_coverage_advantage_established": False,
                "xunce_coverage_return_delta_vs_incumbent": 0.07,
                "xunce_new_covered_cell_delta_vs_incumbent": 70.0,
                "coverage_gain_per_path_cost_delta_vs_incumbent": 0.000001,
                "xunce_efficiency_regression_count": 24,
                "policy_disagreement_count": 240,
                "useful_disagreement_count": 0,
                "canary_traffic_fraction": 0.0,
            },
        )

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
