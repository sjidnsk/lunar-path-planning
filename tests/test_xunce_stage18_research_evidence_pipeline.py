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
        self.assertIn("comparison_metric_summary", summary)
        self.assertIn("coverage_delta_distribution", summary)
        self.assertIn("cost_delta_distribution", summary)
        self.assertIn("risk_delta_distribution", summary)
        self.assertIn("scenario_win_loss_summary", summary)
        self.assertIn("utility_profile_summary", summary)
        self.assertIn("legacy_label_summary", summary)
        self.assertEqual(summary["comparison_metric_summary"]["coverage_delta_cells_mean"], 70.0)
        self.assertEqual(summary["cost_delta_distribution"]["mean"], 5.0)
        self.assertEqual(summary["risk_delta_distribution"]["mean"], 0.2)
        self.assertEqual(summary["coverage_rollout_comparison_summary"]["candidate_refresh_mode"], "dynamic_frontier_nbv_in_process")
        self.assertTrue(summary["coverage_rollout_comparison_summary"]["dynamic_candidate_generation_executed"])
        self.assertEqual(summary["coverage_rollout_comparison_summary"]["dynamic_candidate_validation_mode"], "in_process_path_planner_astar_batch")
        self.assertEqual(summary["coverage_rollout_comparison_summary"]["dynamic_validation_success_count"], 144)
        self.assertEqual(summary["coverage_rollout_comparison_summary"]["in_process_batch_astar_validation_count"], 144)
        self.assertEqual(summary["coverage_rollout_comparison_summary"]["path_planner_route_adapter_success_count"], 0)
        self.assertEqual(summary["coverage_rollout_comparison_summary"]["path_planner_route_adapter_failure_count"], 0)
        self.assertEqual(summary["coverage_rollout_comparison_summary"]["sidecar_grid_astar_screening_count"], 0)
        self.assertFalse(summary["coverage_rollout_comparison_summary"]["dynamic_validation_full_adapter_evidence_passed"])
        self.assertFalse(summary["coverage_rollout_comparison_summary"]["adapter_audit_passed"])
        self.assertIn("dynamic_batch_astar_screening_not_full_adapter_evidence", summary["diagnostic_reason_codes"])
        self.assertIn("closed_loop_dynamic_rollout_summary", summary["coverage_rollout_comparison_summary"])
        self.assertIn("same_candidate_set_policy_selection_summary", summary["coverage_rollout_comparison_summary"])
        self.assertNotIn("xunce_coverage_advantage_established", summary["coverage_rollout_comparison_summary"])
        self.assertIn("xunce_coverage_advantage_established", summary["legacy_label_summary"])
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

    def test_static_coverage_summary_is_partial_for_dynamic_mainline(self) -> None:
        from scripts.xunce_stage18_pipeline import build_stage18_pipeline_summary

        self._write_complete_evidence(dynamic_rollout=False)

        summary = build_stage18_pipeline_summary(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
            plan_only=True,
        )

        self.assertEqual(summary["status"], "partial")
        self.assertEqual(summary["evidence_status"], "partial")
        self.assertIn("missing_dynamic_frontier_nbv_rollout_comparison", summary["missing_reason_codes"])
        self.assertEqual(summary["next_required_change"], "run_dynamic_frontier_nbv_rollout_comparison")

    def test_sidecar_screening_is_diagnostic_not_full_adapter_blocker(self) -> None:
        from scripts.xunce_stage18_pipeline import build_stage18_pipeline_summary

        self._write_complete_evidence(dynamic_rollout=True)
        coverage_path = self.coverage_root / "xunce-exploration-coverage-comparison-summary.json"
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        coverage.update(
            {
                "path_planner_route_adapter_success_count": 0,
                "path_planner_route_adapter_failure_count": 144,
                "sidecar_grid_astar_screening_count": 144,
                "planner_validation_backend_counts": {"sidecar_grid_astar_screening": 144},
                "dynamic_validation_full_adapter_evidence_passed": False,
            }
        )
        self._write_json(coverage_path, coverage)

        summary = build_stage18_pipeline_summary(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
            plan_only=True,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertIn("sidecar_screening_not_full_adapter_evidence", summary["diagnostic_reason_codes"])
        self.assertEqual(summary["coverage_rollout_comparison_summary"]["sidecar_grid_astar_screening_count"], 144)
        self.assertFalse(summary["coverage_rollout_comparison_summary"]["dynamic_validation_full_adapter_evidence_passed"])

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

    def test_candidate_generation_exhaustion_is_diagnostic_not_blocker(self) -> None:
        from scripts.xunce_stage18_pipeline import build_stage18_pipeline_summary

        self._write_complete_evidence()
        coverage_summary_path = self.coverage_root / "xunce-exploration-coverage-comparison-summary.json"
        coverage_summary = json.loads(coverage_summary_path.read_text(encoding="utf-8"))
        coverage_summary["candidate_generation_exhausted_count"] = 24
        coverage_summary["model_inference_failure_count"] = 0
        coverage_summary_path.write_text(json.dumps(coverage_summary, ensure_ascii=False, indent=2), encoding="utf-8")

        summary = build_stage18_pipeline_summary(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
            plan_only=True,
        )

        self.assertNotIn("candidate_generation_exhausted", summary["blocking_reason_codes"])
        self.assertIn("candidate_generation_exhausted", summary["diagnostic_reason_codes"])
        self.assertEqual(summary["coverage_rollout_comparison_summary"]["candidate_generation_exhausted_count"], 24)

    def test_model_inference_failure_count_is_hard_blocker(self) -> None:
        from scripts.xunce_stage18_pipeline import build_stage18_pipeline_summary

        self._write_complete_evidence()
        coverage_summary_path = self.coverage_root / "xunce-exploration-coverage-comparison-summary.json"
        coverage_summary = json.loads(coverage_summary_path.read_text(encoding="utf-8"))
        coverage_summary["model_inference_failure_count"] = 2
        coverage_summary_path.write_text(json.dumps(coverage_summary, ensure_ascii=False, indent=2), encoding="utf-8")

        summary = build_stage18_pipeline_summary(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
            plan_only=True,
        )

        self.assertIn("model_inference_failure", summary["blocking_reason_codes"])

    def test_legacy_coverage_summary_missing_new_episode_fields_does_not_crash(self) -> None:
        from scripts.xunce_stage18_pipeline import build_stage18_pipeline_summary

        self._write_complete_evidence()

        summary = build_stage18_pipeline_summary(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
            plan_only=True,
        )

        self.assertIn("candidate_generation_exhausted_count", summary["coverage_rollout_comparison_summary"])
        self.assertEqual(summary["coverage_rollout_comparison_summary"]["candidate_generation_exhausted_count"], 0)

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

    def test_command_plan_uses_dynamic_frontier_nbv_mainline(self) -> None:
        from scripts.xunce_stage18_pipeline import build_stage18_pipeline_summary

        self._write_complete_evidence()
        summary = build_stage18_pipeline_summary(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
            plan_only=True,
        )
        coverage_commands = [
            row
            for row in summary["stage_command_plan"]
            if row["stage"] == "xunce-high-fidelity-exploration-coverage-comparison"
        ]
        self.assertEqual(len(coverage_commands), 1)
        self.assertIn("dynamic_frontier_nbv_in_process", coverage_commands[0]["display"])
        self.assertIn("in_process_path_planner_astar_batch", coverage_commands[0]["display"])

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
        dynamic_rollout: bool = True,
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
                "candidate_validation_mode": "in_process_path_planner_astar_batch",
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
                "candidate_refresh_mode": "dynamic_frontier_nbv_in_process" if dynamic_rollout else "static_from_source",
                "dynamic_candidate_generation_executed": bool(dynamic_rollout),
                "dynamic_candidate_generation_source": "dynamic_frontier_nbv_in_process/v1" if dynamic_rollout else "",
                "dynamic_candidate_validation_mode": "in_process_path_planner_astar_batch",
                "dynamic_validation_work_root": str((self.temp_dir / "_xunce_dynamic_validation_work").resolve()) if dynamic_rollout else "",
                "dynamic_validation_work_root_path_length": len(str((self.temp_dir / "_xunce_dynamic_validation_work").resolve())) if dynamic_rollout else 0,
                "dynamic_validation_max_path_length": 180,
                "dynamic_proposal_count": 240 if dynamic_rollout else 0,
                "dynamic_validation_attempt_count": 240 if dynamic_rollout else 0,
                "dynamic_validation_success_count": 144 if dynamic_rollout else 0,
                "dynamic_validation_failure_count": 96 if dynamic_rollout else 0,
                "dynamic_validation_cache_hit_count": 12 if dynamic_rollout else 0,
                "dynamic_contract_sidecar_missing_count": 0,
                "dynamic_path_length_preflight_failure_count": 0,
                "in_process_batch_astar_validation_count": 144 if dynamic_rollout else 0,
                "path_planner_route_adapter_success_count": 0,
                "path_planner_route_adapter_failure_count": 0,
                "path_planner_route_adapter_audit_sample_count": 0,
                "adapter_batch_astar_mismatch_count": 0,
                "adapter_audit_passed": False,
                "sidecar_grid_astar_screening_count": 0,
                "sidecar_grid_astar_diagnostic_count": 0,
                "adapter_error_type_counts": {},
                "adapter_error_message_samples": [],
                "planner_validation_backend_counts": {"in_process_path_planner_astar_batch": 144} if dynamic_rollout else {},
                "validation_evidence_kind_counts": {"in_process_astar_screening": 144} if dynamic_rollout else {},
                "dynamic_validation_full_adapter_evidence_passed": False,
                "dynamic_validation_source_root": str(self.quant_root.resolve()) if dynamic_rollout else "",
                "dynamic_candidate_generation_missing_count": 0,
                "state_conditioned_candidate_generation": bool(dynamic_rollout),
                "candidate_set_hash_mismatch_count": 0,
                "paired_decision_audit_row_count": 48 if dynamic_rollout else 0,
                "candidate_generation_effect_scope": "dynamic_generator_plus_policy_closed_loop" if dynamic_rollout else "static_candidate_set",
                "model_selection_evidence_scope": "same_state_same_candidate_set_paired_decision_audit" if dynamic_rollout else "static_candidate_set",
                "closed_loop_dynamic_rollout_summary": {"enabled": bool(dynamic_rollout), "scenario_count": 24},
                "same_candidate_set_policy_selection_summary": {"paired_decision_audit_row_count": 48 if dynamic_rollout else 0},
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
        self._write_json(
            self.coverage_root / "xunce-exploration-coverage-comparison-aggregate.json",
            {
                "schema_version": "xunce-exploration-coverage-comparison-aggregate/v1",
                "scenario_count": 24,
                "xunce_coverage_win_count": 8,
                "xunce_coverage_tie_count": 4,
                "xunce_coverage_loss_count": 12,
                "xunce_coverage_win_rate": 8 / 24,
                "coverage_delta_cells_mean": 70.0,
                "coverage_delta_cells_median": 60.0,
                "coverage_delta_cells_iqr": 25.0,
                "coverage_delta_cells_min": -10.0,
                "coverage_delta_cells_max": 120.0,
                "path_cost_delta_m_mean": 5.0,
                "path_cost_delta_median": 4.0,
                "path_cost_delta_iqr": 2.0,
                "risk_delta_mean": 0.2,
                "risk_delta_median": 0.1,
                "risk_delta_iqr": 0.05,
                "coverage_per_100m_delta_mean": 1.5,
                "coverage_per_100m_delta_median": 1.0,
                "coverage_per_100m_delta_iqr": 0.4,
                "greedy_oracle_coverage_regret_delta_mean": -0.1,
                "cost_aware_oracle_utility_regret_delta_mean": 0.3,
                "utility_profile_summary": {
                    "coverage_first": {"xunce_win_count": 8, "incumbent_win_count": 12, "tie_count": 4},
                    "cost_aware": {"xunce_win_count": 6, "incumbent_win_count": 14, "tie_count": 4},
                    "risk_aware": {"xunce_win_count": 5, "incumbent_win_count": 15, "tie_count": 4},
                },
            },
        )

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
