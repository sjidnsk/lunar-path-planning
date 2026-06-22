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
        self.assertEqual(summary["next_required_change"], "review_xunce_incumbent_comparison_metrics")
        self.assertIsNone(summary["stage18_5_attribution_summary"])
        self.assertIsNone(summary["stage18_5_guard_verdict"])
        self.assertIsNone(summary["stage18_5_primary_next_required_change"])
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

    def test_pipeline_consumes_stage18_5_attribution_when_available(self) -> None:
        from scripts.xunce_stage18_pipeline import build_stage18_pipeline_summary

        self._write_complete_evidence()
        stage18_5_root = self.temp_dir / "stage18_5"
        self._write_stage18_5_attribution_summary(
            stage18_5_root,
            status="failed",
            guard_passed=False,
            primary_route="refine_coverage_reward_and_cost_guard",
        )
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        config["stage18_5_attribution_root"] = str(stage18_5_root)
        self.config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

        summary = build_stage18_pipeline_summary(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
            plan_only=True,
        )

        self.assertEqual(summary["stage18_5_guard_verdict"], "failed")
        self.assertEqual(summary["stage18_5_primary_next_required_change"], "refine_coverage_reward_and_cost_guard")
        self.assertEqual(summary["next_required_change"], "refine_coverage_reward_and_cost_guard")
        self.assertEqual(summary["stage18_5_attribution_summary"]["status"], "failed")
        self.assertFalse(summary["stage18_5_attribution_summary"]["stage19_authorized"])
        self.assertEqual(summary["release_readiness"], "not_authorized")
        self.assertEqual(summary["training_readiness"], "not_authorized")

    def test_valid_stage18_5_route_overrides_dynamic_rollout_refresh_route(self) -> None:
        from scripts.xunce_stage18_pipeline import build_stage18_pipeline_summary

        self._write_complete_evidence(dynamic_rollout=False)
        stage18_5_root = self.temp_dir / "stage18_5_dynamic_override"
        self._write_stage18_5_attribution_summary(
            stage18_5_root,
            status="failed",
            guard_passed=False,
            primary_route="refine_coverage_reward_and_cost_guard",
        )
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        config["stage18_5_attribution_root"] = str(stage18_5_root)
        self.config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

        summary = build_stage18_pipeline_summary(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
            plan_only=True,
        )

        self.assertIn("missing_dynamic_frontier_nbv_rollout_comparison", summary["missing_reason_codes"])
        self.assertEqual(summary["stage18_5_primary_next_required_change"], "refine_coverage_reward_and_cost_guard")
        self.assertEqual(summary["next_required_change"], "refine_coverage_reward_and_cost_guard")

    def test_stale_stage18_5_attribution_summary_does_not_override_pipeline_route(self) -> None:
        from scripts.xunce_stage18_pipeline import build_stage18_pipeline_summary

        self._write_complete_evidence()
        stage18_5_root = self.temp_dir / "stage18_5_stale"
        self._write_stage18_5_attribution_summary(
            stage18_5_root,
            status="failed",
            guard_passed=False,
            primary_route="refine_coverage_reward_and_cost_guard",
            coverage_comparison_root=self.temp_dir / "other_coverage_root",
        )
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        config["stage18_5_attribution_root"] = str(stage18_5_root)
        self.config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

        summary = build_stage18_pipeline_summary(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
            plan_only=True,
        )

        self.assertIn("stale_stage18_5_attribution_root", summary["blocking_reason_codes"])
        self.assertIsNone(summary["stage18_5_attribution_summary"])
        self.assertIsNone(summary["stage18_5_primary_next_required_change"])
        self.assertEqual(summary["next_required_change"], "refresh_stage18_research_evidence_pipeline")

    def test_malformed_stage18_5_attribution_summary_does_not_override_pipeline_route(self) -> None:
        from scripts.xunce_stage18_pipeline import build_stage18_pipeline_summary

        for case_name, mutation in {
            "bad_routing_schema": lambda payload: payload["next_stage_routing"].update({"schema_version": "bad/v1"}),
            "arbitrary_route": lambda payload: payload["next_stage_routing"].update({"primary_route": "publish_checkpoint_now"}),
            "missing_stage19_authorized": lambda payload: payload["next_stage_routing"].pop("stage19_authorized"),
            "bad_guard_schema": lambda payload: payload["guard_evaluation"].update({"schema_version": "bad/v1"}),
            "boundary_violation": lambda payload: payload.update({"publishes_checkpoint": True}),
        }.items():
            with self.subTest(case_name=case_name):
                self.tearDown()
                self.setUp()
                self._write_complete_evidence()
                stage18_5_root = self.temp_dir / f"stage18_5_{case_name}"
                self._write_stage18_5_attribution_summary(
                    stage18_5_root,
                    status="failed",
                    guard_passed=False,
                    primary_route="refine_coverage_reward_and_cost_guard",
                )
                path = stage18_5_root / "xunce-stage18-5-evidence-attribution-summary.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                mutation(payload)
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                config = json.loads(self.config_path.read_text(encoding="utf-8"))
                config["stage18_5_attribution_root"] = str(stage18_5_root)
                self.config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

                summary = build_stage18_pipeline_summary(
                    config_path=self.config_path,
                    output_root=self.output_root,
                    repo_root=self.repo_root,
                    plan_only=True,
                )

                self.assertIsNone(summary["stage18_5_primary_next_required_change"])
                expected_route = (
                    "resolve_stage18_research_evidence_boundary_rejections"
                    if case_name == "boundary_violation"
                    else "refresh_stage18_research_evidence_pipeline"
                )
                self.assertEqual(summary["next_required_change"], expected_route)
                self.assertTrue(
                    {
                        "invalid_stage18_5_attribution_summary",
                        "invalid_stage18_5_guard_summary_schema",
                        "invalid_stage18_5_routing_summary_schema",
                        "boundary_violation",
                    }
                    & set(summary["blocking_reason_codes"])
                )

    def test_stage18_5_cli_override_is_accepted(self) -> None:
        self._write_complete_evidence()
        stage18_5_root = self.temp_dir / "stage18_5_cli"
        self._write_stage18_5_attribution_summary(
            stage18_5_root,
            status="passed",
            guard_passed=True,
            primary_route="prepare_stage19_evaluator_critic_preflight",
        )

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
                "--stage18-5-attribution-root",
                str(stage18_5_root),
                "--plan-only",
            ],
            cwd=self.repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["next_required_change"], "prepare_stage19_evaluator_critic_preflight")

    def test_pipeline_consumes_valid_stage18_6_guard_refinement_root(self) -> None:
        from scripts.xunce_stage18_pipeline import build_stage18_pipeline_summary

        self._write_complete_evidence()
        stage18_5_root = self.temp_dir / "stage18_5"
        stage18_6_root = self.temp_dir / "stage18_6"
        self._write_stage18_5_attribution_summary(
            stage18_5_root,
            status="passed",
            guard_passed=False,
            primary_route="refine_coverage_reward_and_cost_guard",
        )
        self._write_stage18_6_guard_refinement_summary(
            stage18_6_root,
            stage18_5_root=stage18_5_root,
            primary_route="rerun_stage18_4e_with_candidate_metric_audit",
        )
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        config["stage18_5_attribution_root"] = str(stage18_5_root)
        config["stage18_6_guard_refinement_root"] = str(stage18_6_root)
        self.config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

        summary = build_stage18_pipeline_summary(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
            plan_only=True,
        )

        self.assertEqual(summary["stage18_6_guard_refinement_verdict"], "failed")
        self.assertEqual(summary["stage18_6_primary_next_required_change"], "rerun_stage18_4e_with_candidate_metric_audit")
        self.assertEqual(summary["next_required_change"], "rerun_stage18_4e_with_candidate_metric_audit")
        self.assertEqual(summary["stage18_6_candidate_metric_readiness"]["full_candidate_metric_replay_available"], False)
        self.assertFalse(summary["stage18_6_guard_refinement_summary"]["stage19_authorized"])

    def test_stale_stage18_6_guard_refinement_root_does_not_override_stage18_5_route(self) -> None:
        from scripts.xunce_stage18_pipeline import build_stage18_pipeline_summary

        self._write_complete_evidence()
        stage18_5_root = self.temp_dir / "stage18_5"
        stage18_6_root = self.temp_dir / "stage18_6_stale"
        self._write_stage18_5_attribution_summary(
            stage18_5_root,
            status="passed",
            guard_passed=False,
            primary_route="refine_coverage_reward_and_cost_guard",
        )
        self._write_stage18_6_guard_refinement_summary(
            stage18_6_root,
            stage18_5_root=stage18_5_root,
            coverage_comparison_root=self.temp_dir / "other_coverage",
            primary_route="prepare_stage19_evaluator_critic_preflight",
        )
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        config["stage18_5_attribution_root"] = str(stage18_5_root)
        config["stage18_6_guard_refinement_root"] = str(stage18_6_root)
        self.config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

        summary = build_stage18_pipeline_summary(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
            plan_only=True,
        )

        self.assertIn("stale_stage18_6_guard_refinement_root", summary["blocking_reason_codes"])
        self.assertIsNone(summary["stage18_6_primary_next_required_change"])
        self.assertEqual(summary["next_required_change"], "refine_coverage_reward_and_cost_guard")

    def test_stage18_6_summary_missing_lineage_paths_does_not_override_stage18_5_route(self) -> None:
        from scripts.xunce_stage18_pipeline import build_stage18_pipeline_summary

        self._write_complete_evidence()
        stage18_5_root = self.temp_dir / "stage18_5"
        stage18_6_root = self.temp_dir / "stage18_6_missing_lineage_paths"
        self._write_stage18_5_attribution_summary(
            stage18_5_root,
            status="passed",
            guard_passed=False,
            primary_route="refine_coverage_reward_and_cost_guard",
        )
        self._write_stage18_6_guard_refinement_summary(
            stage18_6_root,
            stage18_5_root=stage18_5_root,
            primary_route="prepare_stage19_evaluator_critic_preflight",
            guard_refinement_passed=True,
            same_candidate_set_guard_clean_advantage_established=True,
        )
        summary_path = stage18_6_root / "xunce-stage18-6-guard-refinement-summary.json"
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload.pop("coverage_comparison_root")
        payload.pop("stage18_5_attribution_root")
        summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        config["stage18_5_attribution_root"] = str(stage18_5_root)
        config["stage18_6_guard_refinement_root"] = str(stage18_6_root)
        self.config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

        summary = build_stage18_pipeline_summary(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
            plan_only=True,
        )

        self.assertIn("stale_stage18_6_guard_refinement_root", summary["blocking_reason_codes"])
        self.assertIsNone(summary["stage18_6_primary_next_required_change"])
        self.assertEqual(summary["next_required_change"], "refine_coverage_reward_and_cost_guard")

    def test_malformed_stage18_6_route_does_not_override_stage18_5_route(self) -> None:
        from scripts.xunce_stage18_pipeline import build_stage18_pipeline_summary

        self._write_complete_evidence()
        stage18_5_root = self.temp_dir / "stage18_5"
        stage18_6_root = self.temp_dir / "stage18_6_bad_route"
        self._write_stage18_5_attribution_summary(
            stage18_5_root,
            status="passed",
            guard_passed=False,
            primary_route="refine_coverage_reward_and_cost_guard",
        )
        self._write_stage18_6_guard_refinement_summary(
            stage18_6_root,
            stage18_5_root=stage18_5_root,
            primary_route="coverage_driven_ppo_improvement_run",
        )
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        config["stage18_5_attribution_root"] = str(stage18_5_root)
        config["stage18_6_guard_refinement_root"] = str(stage18_6_root)
        self.config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

        summary = build_stage18_pipeline_summary(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
            plan_only=True,
        )

        self.assertIn("invalid_stage18_6_guard_refinement_summary", summary["blocking_reason_codes"])
        self.assertIsNone(summary["stage18_6_primary_next_required_change"])
        self.assertEqual(summary["next_required_change"], "refine_coverage_reward_and_cost_guard")

    def test_stage18_6_profile_mismatch_does_not_override_stage18_5_route(self) -> None:
        from scripts.xunce_stage18_pipeline import build_stage18_pipeline_summary

        self._write_complete_evidence()
        stage18_5_root = self.temp_dir / "stage18_5"
        stage18_6_root = self.temp_dir / "stage18_6_bad_profile"
        self._write_stage18_5_attribution_summary(
            stage18_5_root,
            status="passed",
            guard_passed=False,
            primary_route="refine_coverage_reward_and_cost_guard",
            profile_hash="fixture-profile-hash",
        )
        self._write_stage18_6_guard_refinement_summary(
            stage18_6_root,
            stage18_5_root=stage18_5_root,
            primary_route="rerun_stage18_4e_with_candidate_metric_audit",
            profile_hash="different-profile-hash",
        )
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        config["stage18_5_attribution_root"] = str(stage18_5_root)
        config["stage18_6_guard_refinement_root"] = str(stage18_6_root)
        self.config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

        summary = build_stage18_pipeline_summary(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
            plan_only=True,
        )

        self.assertIn("invalid_stage18_6_guard_refinement_profile_lineage", summary["blocking_reason_codes"])
        self.assertIsNone(summary["stage18_6_primary_next_required_change"])
        self.assertEqual(summary["next_required_change"], "refine_coverage_reward_and_cost_guard")

    def test_inconsistent_stage18_6_preflight_route_does_not_override_stage18_5_route(self) -> None:
        from scripts.xunce_stage18_pipeline import build_stage18_pipeline_summary

        self._write_complete_evidence()
        stage18_5_root = self.temp_dir / "stage18_5"
        stage18_6_root = self.temp_dir / "stage18_6_inconsistent_preflight"
        self._write_stage18_5_attribution_summary(
            stage18_5_root,
            status="passed",
            guard_passed=False,
            primary_route="refine_coverage_reward_and_cost_guard",
        )
        self._write_stage18_6_guard_refinement_summary(
            stage18_6_root,
            stage18_5_root=stage18_5_root,
            primary_route="prepare_stage19_evaluator_critic_preflight",
            guard_refinement_passed=False,
            full_candidate_metric_replay_available=False,
            same_candidate_set_guard_clean_advantage_established=False,
        )
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        config["stage18_5_attribution_root"] = str(stage18_5_root)
        config["stage18_6_guard_refinement_root"] = str(stage18_6_root)
        self.config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

        summary = build_stage18_pipeline_summary(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
            plan_only=True,
        )

        self.assertIn("invalid_stage18_6_guard_refinement_preflight_semantics", summary["blocking_reason_codes"])
        self.assertIsNone(summary["stage18_6_primary_next_required_change"])
        self.assertEqual(summary["next_required_change"], "refine_coverage_reward_and_cost_guard")

    def test_stage18_6_cli_override_is_accepted(self) -> None:
        self._write_complete_evidence()
        stage18_5_root = self.temp_dir / "stage18_5_cli"
        stage18_6_root = self.temp_dir / "stage18_6_cli"
        self._write_stage18_5_attribution_summary(
            stage18_5_root,
            status="passed",
            guard_passed=False,
            primary_route="refine_coverage_reward_and_cost_guard",
        )
        self._write_stage18_6_guard_refinement_summary(
            stage18_6_root,
            stage18_5_root=stage18_5_root,
            primary_route="rerun_stage18_4e_with_candidate_metric_audit",
        )

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
                "--stage18-5-attribution-root",
                str(stage18_5_root),
                "--stage18-6-guard-refinement-root",
                str(stage18_6_root),
                "--plan-only",
            ],
            cwd=self.repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["next_required_change"], "rerun_stage18_4e_with_candidate_metric_audit")

    def test_pipeline_consumes_valid_stage18_7_candidate_count_scaling_root(self) -> None:
        from scripts.xunce_stage18_pipeline import build_stage18_pipeline_summary

        self._write_complete_evidence()
        stage18_5_root = self.temp_dir / "stage18_5"
        stage18_6_root = self.temp_dir / "stage18_6"
        stage18_7_root = self.temp_dir / "stage18_7"
        self._write_stage18_5_attribution_summary(
            stage18_5_root,
            status="passed",
            guard_passed=False,
            primary_route="refine_coverage_reward_and_cost_guard",
        )
        self._write_stage18_6_guard_refinement_summary(
            stage18_6_root,
            stage18_5_root=stage18_5_root,
            primary_route="refine_coverage_reward_and_cost_guard",
        )
        self._write_stage18_7_candidate_count_scaling_summary(
            stage18_7_root,
            primary_route="expand_candidate_generation_roi_complexity",
            stage18_6_guard_refinement_passed_count=0,
            same_candidate_set_guard_clean_advantage_established_count=0,
        )
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        config["stage18_5_attribution_root"] = str(stage18_5_root)
        config["stage18_6_guard_refinement_root"] = str(stage18_6_root)
        config["stage18_7_candidate_count_scaling_root"] = str(stage18_7_root)
        self.config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

        summary = build_stage18_pipeline_summary(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
            plan_only=True,
        )

        self.assertEqual(summary["stage18_7_primary_next_required_change"], "expand_candidate_generation_roi_complexity")
        self.assertEqual(summary["next_required_change"], "expand_candidate_generation_roi_complexity")
        self.assertFalse(summary["stage18_7_candidate_count_scaling_summary"]["stage19_authorized"])

    def test_malformed_stage18_7_preflight_route_does_not_override_stage18_6_route(self) -> None:
        from scripts.xunce_stage18_pipeline import build_stage18_pipeline_summary

        self._write_complete_evidence()
        stage18_5_root = self.temp_dir / "stage18_5"
        stage18_6_root = self.temp_dir / "stage18_6"
        stage18_7_root = self.temp_dir / "stage18_7_bad_preflight"
        self._write_stage18_5_attribution_summary(
            stage18_5_root,
            status="passed",
            guard_passed=False,
            primary_route="refine_coverage_reward_and_cost_guard",
        )
        self._write_stage18_6_guard_refinement_summary(
            stage18_6_root,
            stage18_5_root=stage18_5_root,
            primary_route="refine_coverage_reward_and_cost_guard",
        )
        self._write_stage18_7_candidate_count_scaling_summary(
            stage18_7_root,
            primary_route="prepare_stage19_evaluator_critic_preflight",
            stage18_6_guard_refinement_passed_count=0,
            same_candidate_set_guard_clean_advantage_established_count=0,
        )
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        config["stage18_5_attribution_root"] = str(stage18_5_root)
        config["stage18_6_guard_refinement_root"] = str(stage18_6_root)
        config["stage18_7_candidate_count_scaling_root"] = str(stage18_7_root)
        self.config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

        summary = build_stage18_pipeline_summary(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
            plan_only=True,
        )

        self.assertIn("invalid_stage18_7_candidate_count_scaling_preflight_semantics", summary["blocking_reason_codes"])
        self.assertIsNone(summary["stage18_7_primary_next_required_change"])
        self.assertEqual(summary["next_required_change"], "refine_coverage_reward_and_cost_guard")

    def test_forged_stage18_7_preflight_rows_do_not_override_stage18_6_route(self) -> None:
        from scripts.xunce_stage18_pipeline import build_stage18_pipeline_summary

        self._write_complete_evidence()
        stage18_5_root = self.temp_dir / "stage18_5"
        stage18_6_root = self.temp_dir / "stage18_6"
        stage18_7_root = self.temp_dir / "stage18_7_forged"
        self._write_stage18_5_attribution_summary(
            stage18_5_root,
            status="passed",
            guard_passed=False,
            primary_route="refine_coverage_reward_and_cost_guard",
        )
        self._write_stage18_6_guard_refinement_summary(
            stage18_6_root,
            stage18_5_root=stage18_5_root,
            primary_route="refine_coverage_reward_and_cost_guard",
        )
        self._write_stage18_7_candidate_count_scaling_summary(
            stage18_7_root,
            primary_route="prepare_stage19_evaluator_critic_preflight",
            stage18_6_guard_refinement_passed_count=1,
            same_candidate_set_guard_clean_advantage_established_count=1,
            candidate_count_results=[],
        )
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        config["stage18_5_attribution_root"] = str(stage18_5_root)
        config["stage18_6_guard_refinement_root"] = str(stage18_6_root)
        config["stage18_7_candidate_count_scaling_root"] = str(stage18_7_root)
        self.config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

        summary = build_stage18_pipeline_summary(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
            plan_only=True,
        )

        self.assertIn("invalid_stage18_7_candidate_count_scaling_results", summary["blocking_reason_codes"])
        self.assertIsNone(summary["stage18_7_primary_next_required_change"])
        self.assertEqual(summary["next_required_change"], "refine_coverage_reward_and_cost_guard")

    def test_stage18_7_without_stage18_6_root_does_not_override_stage18_5_route(self) -> None:
        from scripts.xunce_stage18_pipeline import build_stage18_pipeline_summary

        self._write_complete_evidence()
        stage18_5_root = self.temp_dir / "stage18_5"
        stage18_7_root = self.temp_dir / "stage18_7_no_stage18_6"
        self._write_stage18_5_attribution_summary(
            stage18_5_root,
            status="passed",
            guard_passed=False,
            primary_route="refine_coverage_reward_and_cost_guard",
        )
        self._write_stage18_7_candidate_count_scaling_summary(
            stage18_7_root,
            primary_route="expand_candidate_generation_roi_complexity",
        )
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        config["stage18_5_attribution_root"] = str(stage18_5_root)
        config["stage18_7_candidate_count_scaling_root"] = str(stage18_7_root)
        self.config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

        summary = build_stage18_pipeline_summary(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
            plan_only=True,
        )

        self.assertIn("missing_stage18_6_guard_refinement_for_stage18_7", summary["blocking_reason_codes"])
        self.assertIsNone(summary["stage18_7_primary_next_required_change"])
        self.assertEqual(summary["next_required_change"], "refine_coverage_reward_and_cost_guard")

    def test_stage18_7_cli_override_is_accepted(self) -> None:
        self._write_complete_evidence()
        stage18_5_root = self.temp_dir / "stage18_5_cli"
        stage18_6_root = self.temp_dir / "stage18_6_cli"
        stage18_7_root = self.temp_dir / "stage18_7_cli"
        self._write_stage18_5_attribution_summary(
            stage18_5_root,
            status="passed",
            guard_passed=False,
            primary_route="refine_coverage_reward_and_cost_guard",
        )
        self._write_stage18_6_guard_refinement_summary(
            stage18_6_root,
            stage18_5_root=stage18_5_root,
            primary_route="refine_coverage_reward_and_cost_guard",
        )
        self._write_stage18_7_candidate_count_scaling_summary(
            stage18_7_root,
            primary_route="run_missing_candidate_count_sweeps_with_metric_audit",
        )

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
                "--stage18-5-attribution-root",
                str(stage18_5_root),
                "--stage18-6-guard-refinement-root",
                str(stage18_6_root),
                "--stage18-7-candidate-count-scaling-root",
                str(stage18_7_root),
                "--plan-only",
            ],
            cwd=self.repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["next_required_change"], "run_missing_candidate_count_sweeps_with_metric_audit")

    def test_pipeline_prefers_valid_stage18_9_trajectory_risk_reward_root(self) -> None:
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self._write_complete_evidence()
        stage18_9_root = self.temp_dir / "stage18_9"
        self._write_stage18_9_trajectory_risk_reward_summary(
            stage18_9_root,
            primary_route="prepare_stage19_evaluator_critic_preflight",
            trajectory_guard_passed=True,
        )
        config["stage18_9_trajectory_risk_reward_root"] = str(stage18_9_root)
        self.config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

        from scripts.xunce_stage18_pipeline import build_stage18_pipeline_summary

        summary = build_stage18_pipeline_summary(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["next_required_change"], "prepare_stage19_evaluator_critic_preflight")
        self.assertEqual(summary["stage18_9_trajectory_risk_reward_verdict"], "passed")
        self.assertFalse(summary["stage18_9_trajectory_risk_reward_summary"]["stage19_authorized"])

    def test_malformed_stage18_9_preflight_route_does_not_override_stage18_7_route(self) -> None:
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self._write_complete_evidence()
        stage18_9_root = self.temp_dir / "stage18_9_bad_preflight"
        self._write_stage18_9_trajectory_risk_reward_summary(
            stage18_9_root,
            primary_route="prepare_stage19_evaluator_critic_preflight",
            trajectory_guard_passed=False,
        )
        config["stage18_9_trajectory_risk_reward_root"] = str(stage18_9_root)
        self.config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

        from scripts.xunce_stage18_pipeline import build_stage18_pipeline_summary

        summary = build_stage18_pipeline_summary(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertIn("invalid_stage18_9_trajectory_risk_reward_preflight_semantics", summary["blocking_reason_codes"])
        self.assertIsNone(summary["stage18_9_primary_next_required_change"])

    def test_stage18_9_v2_profile_is_rejected_by_pipeline(self) -> None:
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self._write_complete_evidence()
        stage18_9_root = self.temp_dir / "stage18_9_v2_profile"
        self._write_stage18_9_trajectory_risk_reward_summary(
            stage18_9_root,
            primary_route="calibrate_soft_risk_exposure_weight",
            trajectory_guard_passed=False,
            profile_id="xunce-coverage-cost-risk-budget-v2",
            profile_version="v2",
            profile_hash="fixture-profile-hash-v2",
        )
        config["stage18_9_trajectory_risk_reward_root"] = str(stage18_9_root)
        self.config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

        from scripts.xunce_stage18_pipeline import build_stage18_pipeline_summary

        summary = build_stage18_pipeline_summary(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertIn("invalid_stage18_9_trajectory_risk_reward_profile_lineage", summary["blocking_reason_codes"])
        self.assertIsNone(summary["stage18_9_primary_next_required_change"])

    def test_stage18_9_cli_override_is_accepted(self) -> None:
        self._write_complete_evidence()
        stage18_9_root = self.temp_dir / "stage18_9_cli"
        self._write_stage18_9_trajectory_risk_reward_summary(
            stage18_9_root,
            primary_route="calibrate_soft_risk_exposure_weight",
            trajectory_guard_passed=False,
        )

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
                "--stage18-9-trajectory-risk-reward-root",
                str(stage18_9_root),
            ],
            cwd=self.repo_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        payload = json.loads(completed.stdout)
        self.assertEqual(payload["next_required_change"], "calibrate_soft_risk_exposure_weight")

    def test_pipeline_prefers_valid_stage19_evaluator_critic_preflight_root(self) -> None:
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self._write_complete_evidence()
        stage18_11_root = self.temp_dir / "stage18_11"
        stage19_root = self.temp_dir / "stage19"
        self._write_stage18_11_path_cost_weight_calibration_summary(stage18_11_root)
        self._write_stage19_evaluator_critic_preflight_summary(stage19_root, stage18_11_root=stage18_11_root)
        config["stage18_11_path_cost_weight_calibration_root"] = str(stage18_11_root)
        config["stage19_evaluator_critic_preflight_root"] = str(stage19_root)
        self.config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

        from scripts.xunce_stage18_pipeline import build_stage18_pipeline_summary

        summary = build_stage18_pipeline_summary(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["next_required_change"], "stage20_reward_rerank_oracle_preference_dataset_preparation")
        self.assertEqual(summary["stage18_11_primary_next_required_change"], "prepare_stage19_evaluator_critic_preflight")
        self.assertEqual(summary["stage19_primary_next_required_change"], "stage20_reward_rerank_oracle_preference_dataset_preparation")
        self.assertEqual(summary["stage19_primary_target_candidate_count"], 36)
        self.assertEqual(summary["stage19_primary_target_path_cost_weight"], 0.1)
        self.assertTrue(summary["stage19_oracle_target_feasible"])
        self.assertFalse(summary["stage19_xunce_checkpoint_advantage_established"])
        self.assertFalse(summary["stage19_training_authorized"])

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

    def _write_stage18_5_attribution_summary(
        self,
        root: Path,
        *,
        status: str,
        guard_passed: bool,
        primary_route: str,
        coverage_comparison_root: Path | None = None,
        profile_hash: str = "fixture-profile-hash",
        profile_id: str = "xunce-coverage-cost-risk-budget-v2",
        profile_version: str = "v2",
    ) -> None:
        self._write_json(
            root / "xunce-stage18-5-evidence-attribution-summary.json",
            {
                "schema_version": "xunce-stage18-5-evidence-attribution-summary/v1",
                "status": status,
                "coverage_comparison_root": str((coverage_comparison_root or self.coverage_root).resolve()),
                "profile_id": profile_id,
                "profile_version": profile_version,
                "profile_hash": profile_hash,
                "evidence_authenticity_gate_passed": True,
                "candidate_validity_gate_passed": True,
                "next_required_change": primary_route,
                "guard_evaluation": {
                    "schema_version": "xunce-stage18-5-guard-evaluation/v1",
                    "passed": guard_passed,
                    "failed_guards": [] if guard_passed else ["path_cost_regression"],
                    "thresholds": {
                        "profile_id": profile_id,
                        "profile_version": profile_version,
                        "profile_hash": profile_hash,
                    },
                },
                "next_stage_routing": {
                    "schema_version": "xunce-stage18-5-next-stage-routing/v1",
                    "primary_route": primary_route,
                    "stage19_authorized": False,
                },
                "stage19_readiness": {
                    "readiness": "ready_for_stage19_preflight_human_review_only" if guard_passed else "not_authorized",
                    "authorized": False,
                },
                "canary_traffic_fraction": 0.0,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "starts_online_canary": False,
                "runs_new_ppo_update": False,
                "real_world_release_approved": False,
                "real_world_performance_claimed": False,
                "default_policy_replacement_approved": False,
            },
        )

    def _write_stage18_6_guard_refinement_summary(
        self,
        root: Path,
        *,
        stage18_5_root: Path,
        primary_route: str,
        coverage_comparison_root: Path | None = None,
        profile_hash: str = "fixture-profile-hash",
        profile_id: str = "xunce-coverage-cost-risk-budget-v2",
        profile_version: str = "v2",
        guard_refinement_passed: bool | None = None,
        full_candidate_metric_replay_available: bool = False,
        same_candidate_set_guard_clean_advantage_established: bool = False,
    ) -> None:
        passed = primary_route == "prepare_stage19_evaluator_critic_preflight" if guard_refinement_passed is None else guard_refinement_passed
        self._write_json(
            root / "xunce-stage18-6-guard-refinement-summary.json",
            {
                "schema_version": "xunce-stage18-6-guard-refinement-summary/v1",
                "status": "passed",
                "coverage_comparison_root": str((coverage_comparison_root or self.coverage_root).resolve()),
                "stage18_5_attribution_root": str(stage18_5_root.resolve()),
                "profile_id": profile_id,
                "profile_version": profile_version,
                "profile_hash": profile_hash,
                "guard_refinement_passed": passed,
                "counterfactual_reselection_claimed": False,
                "candidate_metric_readiness": {
                    "schema_version": "xunce-stage18-6-candidate-metric-readiness/v1",
                    "full_candidate_metric_replay_available": full_candidate_metric_replay_available,
                    "counterfactual_reselection_claim_allowed": full_candidate_metric_replay_available,
                    "reason_codes": [] if full_candidate_metric_replay_available else ["missing_candidate_metric_audit"],
                },
                "paired_decision_summary": {
                    "same_candidate_set_guard_clean_advantage_established": same_candidate_set_guard_clean_advantage_established,
                },
                "next_stage_routing": {
                    "schema_version": "xunce-stage18-6-next-stage-routing/v1",
                    "primary_route": primary_route,
                    "stage19_authorized": False,
                },
                "stage19_readiness": {
                    "readiness": "not_authorized",
                    "authorized": False,
                },
                "canary_traffic_fraction": 0.0,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "starts_online_canary": False,
                "runs_new_ppo_update": False,
                "real_world_release_approved": False,
                "real_world_performance_claimed": False,
                "default_policy_replacement_approved": False,
            },
        )

    def _write_stage18_7_candidate_count_scaling_summary(
        self,
        root: Path,
        *,
        primary_route: str,
        profile_hash: str = "fixture-profile-hash",
        profile_id: str = "xunce-coverage-cost-risk-budget-v2",
        profile_version: str = "v2",
        stage18_6_guard_refinement_passed_count: int = 0,
        same_candidate_set_guard_clean_advantage_established_count: int = 0,
        stage19_authorized: bool = False,
        candidate_count_results: list[dict] | None = None,
    ) -> None:
        if candidate_count_results is None:
            candidate_count_results = [
                {
                    "schema_version": "xunce-stage18-7-candidate-count-scaling-result/v1",
                    "candidate_count": count,
                    "proposal_pool_limit": pool,
                    "expected_proposal_pool_limit": pool,
                    "profile_hash": profile_hash,
                    "candidate_metric_replay_available": True,
                    "guard_refinement_passed": stage18_6_guard_refinement_passed_count > 0,
                    "stage18_6_next_required_change": "prepare_stage19_evaluator_critic_preflight"
                    if primary_route == "prepare_stage19_evaluator_critic_preflight"
                    else "refine_coverage_reward_and_cost_guard",
                    "stage19_authorized": False,
                    "guard_clean_candidate_available_rate": 0.25,
                    "xunce_selected_guard_clean_rate": 1.0
                    if stage18_6_guard_refinement_passed_count > 0
                    else 0.0,
                    "incumbent_selected_guard_clean_rate": 0.0,
                    "same_candidate_set_guard_clean_advantage_established": same_candidate_set_guard_clean_advantage_established_count > 0,
                    "boundary_flags_all_false": True,
                    "sweep_complete": True,
                }
                for count, pool in ((6, 48), (12, 96), (24, 192), (36, 288))
            ]
        self._write_json(
            root / "xunce-stage18-7-candidate-count-scaling-summary.json",
            {
                "schema_version": "xunce-stage18-7-candidate-count-scaling-summary/v1",
                "status": "passed",
                "profile_id": profile_id,
                "profile_version": profile_version,
                "profile_hash": profile_hash,
                "sweep_complete_count": 4,
                "stage18_6_guard_refinement_passed_count": stage18_6_guard_refinement_passed_count,
                "same_candidate_set_guard_clean_advantage_established_count": same_candidate_set_guard_clean_advantage_established_count,
                "best_guard_clean_candidate_available_rate": 0.25,
                "best_xunce_selected_guard_clean_rate": 0.0,
                "best_incumbent_selected_guard_clean_rate": 1.0,
                "candidate_count_results": candidate_count_results,
                "next_stage_routing": {
                    "schema_version": "xunce-stage18-7-next-stage-routing/v1",
                    "primary_route": primary_route,
                    "stage19_authorized": stage19_authorized,
                },
                "stage19_readiness": {
                    "readiness": "ready_for_stage19_preflight_human_review_only"
                    if primary_route == "prepare_stage19_evaluator_critic_preflight"
                    else "not_authorized",
                    "authorized": stage19_authorized,
                },
                "canary_traffic_fraction": 0.0,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "starts_online_canary": False,
                "runs_new_ppo_update": False,
                "real_world_release_approved": False,
                "real_world_performance_claimed": False,
                "default_policy_replacement_approved": False,
            },
        )

    def _write_stage18_9_trajectory_risk_reward_summary(
        self,
        root: Path,
        *,
        primary_route: str,
        trajectory_guard_passed: bool,
        coverage_comparison_root: Path | None = None,
        profile_hash: str = "fixture-profile-hash-v3",
        profile_id: str = "xunce-coverage-cost-risk-boundary-v3",
        profile_version: str = "v3",
    ) -> None:
        self._write_json(
            root / "xunce-stage18-9-trajectory-risk-reward-summary.json",
            {
                "schema_version": "xunce-stage18-9-trajectory-risk-reward-summary/v1",
                "status": "passed" if trajectory_guard_passed else "failed",
                "coverage_comparison_root": str((coverage_comparison_root or self.coverage_root).resolve()),
                "profile_id": profile_id,
                "profile_version": profile_version,
                "profile_hash": profile_hash,
                "trajectory_guard_passed": trajectory_guard_passed,
                "path_risk_boundary_summary": {
                    "path_risk_boundary_passed": trajectory_guard_passed,
                    "hard_risk_violation_count": 0 if trajectory_guard_passed else 1,
                },
                "trajectory_guard_summary": {
                    "coverage_advantage_established": trajectory_guard_passed,
                    "path_cost_budget_passed": trajectory_guard_passed,
                    "coverage_efficiency_passed": trajectory_guard_passed,
                    "soft_risk_exposure_passed": trajectory_guard_passed,
                },
                "candidate_diagnostics": {"diagnostic_only": True},
                "next_stage_routing": {
                    "schema_version": "xunce-stage18-9-next-stage-routing/v1",
                    "primary_route": primary_route,
                    "stage19_authorized": False,
                },
                "stage19_readiness": {
                    "schema_version": "xunce-stage18-9-stage19-readiness/v1",
                    "readiness": "ready_for_stage19_preflight_human_review_only"
                    if primary_route == "prepare_stage19_evaluator_critic_preflight"
                    else "not_authorized",
                    "authorized": False,
                    "trajectory_guard_passed": trajectory_guard_passed,
                },
                "canary_traffic_fraction": 0.0,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "starts_online_canary": False,
                "runs_new_ppo_update": False,
                "real_world_release_approved": False,
                "real_world_performance_claimed": False,
                "default_policy_replacement_approved": False,
            },
        )

    def _write_stage18_11_path_cost_weight_calibration_summary(self, root: Path) -> None:
        self._write_json(
            root / "xunce-stage18-11-path-cost-weight-calibration-summary.json",
            {
                "schema_version": "xunce-stage18-11-path-cost-weight-calibration-summary/v1",
                "status": "passed",
                "profile_id": "xunce-coverage-cost-risk-boundary-v3",
                "profile_version": "v3",
                "profile_hash": "fixture-profile-hash-v3",
                "target_final_coverage_rate": 0.99,
                "best_diagnostic_rollout_candidate_count": 36,
                "best_diagnostic_rollout_path_cost_weight": 0.1,
                "best_diagnostic_final_coverage_rate_mean": 1.0,
                "best_diagnostic_final_coverage_rate_max": 1.0,
                "diagnostic_rollout_summary": {
                    "complete_diagnostic_rollout_count": 4,
                    "best_final_coverage_rate_mean": 1.0,
                    "best_hard_risk_violation_count": 0.0,
                },
                "next_required_change": "prepare_stage19_evaluator_critic_preflight",
                "next_stage_routing": {
                    "schema_version": "xunce-stage18-11-next-stage-routing/v1",
                    "primary_route": "prepare_stage19_evaluator_critic_preflight",
                    "stage19_authorized": False,
                },
                "stage19_authorized": False,
                "stage19_readiness": {
                    "schema_version": "xunce-stage18-11-stage19-readiness/v1",
                    "readiness": "ready_for_stage19_preflight_human_review_only",
                    "authorized": False,
                },
                "canary_traffic_fraction": 0.0,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "starts_online_canary": False,
                "runs_new_ppo_update": False,
                "real_world_release_approved": False,
                "real_world_performance_claimed": False,
                "default_policy_replacement_approved": False,
            },
        )

    def _write_stage19_evaluator_critic_preflight_summary(self, root: Path, *, stage18_11_root: Path) -> None:
        self._write_json(
            root / "xunce-stage19-evaluator-critic-preflight-summary.json",
            {
                "schema_version": "xunce-stage19-evaluator-critic-preflight-summary/v1",
                "status": "passed",
                "stage18_11_path_cost_weight_calibration_root": str(stage18_11_root.resolve()),
                "profile_id": "xunce-coverage-cost-risk-boundary-v3",
                "profile_version": "v3",
                "profile_hash": "fixture-profile-hash-v3",
                "oracle_target_feasible": True,
                "primary_target_selected": True,
                "selected_candidate_count": 36,
                "selected_path_cost_weight": 0.1,
                "xunce_checkpoint_advantage_established": False,
                "training_or_release_authorized": False,
                "stage20_authorized": False,
                "practical_target_selection": {
                    "schema_version": "xunce-stage19-practical-target-selection/v1",
                    "primary_target_feasible": True,
                    "primary_budget_passed": True,
                    "selected_candidate_count": 36,
                    "selected_path_cost_weight": 0.1,
                },
                "critic_target_readiness": {
                    "schema_version": "xunce-stage19-critic-target-readiness/v1",
                    "critic_target_ready": True,
                    "preference_pair_count": 4,
                },
                "next_required_change": "stage20_reward_rerank_oracle_preference_dataset_preparation",
                "next_stage_routing": {
                    "schema_version": "xunce-stage19-next-stage-routing/v1",
                    "primary_route": "stage20_reward_rerank_oracle_preference_dataset_preparation",
                    "stage20_authorized": False,
                },
                "stage20_readiness": {
                    "schema_version": "xunce-stage19-stage20-readiness/v1",
                    "readiness": "ready_for_stage20_preference_dataset_preparation_human_review_only",
                    "authorized": False,
                },
                "canary_traffic_fraction": 0.0,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "starts_online_canary": False,
                "runs_new_ppo_update": False,
                "real_world_release_approved": False,
                "real_world_performance_claimed": False,
            },
        )

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
