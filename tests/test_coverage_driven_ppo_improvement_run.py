import json
import math
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class CoverageDrivenPpoImprovementRunTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        for path in (self.repo_root / "scripts", self.repo_root / "model-explorer" / "src"):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)
        from model_explorer.policy.canonical_reward import load_canonical_reward_profile

        self.reward_profile = load_canonical_reward_profile(
            self.repo_root / "configs" / "xunce_canonical_reward_guard_profile_v2.json"
        )
        self.temp_dir = Path(tempfile.mkdtemp(prefix="coverage-driven-ppo-"))
        self.formal_root = self.temp_dir / "formal-training"
        self.replay_root = self.temp_dir / "post-training-replay"
        self.selected_root = self.temp_dir / "selected-candidate"
        self.base_candidate_root = self.temp_dir / "base-candidate"
        self.signal_root = self.temp_dir / "coverage-signal"
        self.performance_root = self.temp_dir / "coverage-performance"
        self.reward_root = self.temp_dir / "reward-refinement"
        self.reward_source_root = self.temp_dir / "reward-source"
        self.output_root = self.temp_dir / "coverage-driven-output"
        for path in (
            self.formal_root,
            self.replay_root,
            self.selected_root,
            self.base_candidate_root,
            self.signal_root,
            self.performance_root,
            self.reward_root,
            self.reward_source_root,
        ):
            path.mkdir(parents=True)
        self.observation = self._observation()
        self._write_base_candidate()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_runs_coverage_aware_ppo_update_and_passes_when_post_update_coverage_beats_frozen_baselines(self) -> None:
        from scripts.run_coverage_driven_ppo_improvement_run import (
            run_coverage_driven_ppo_improvement_run,
        )

        rows = [
            self._reward_row("ctx-selected-0", step_index=0, coverage_delta=0.20, valuable=0.12),
            self._reward_row("ctx-selected-1", step_index=1, coverage_delta=0.18, valuable=0.10),
        ]
        self._write_inputs(rows, baseline_coverage_return=0.10, baseline_valuable=0.05)

        summary = run_coverage_driven_ppo_improvement_run(
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            coverage_signal_root=self.signal_root,
            coverage_performance_root=self.performance_root,
            reward_refinement_root=self.reward_root,
            reward_source_root=self.reward_source_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["coverage_driven_ppo_improvement_status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertTrue(summary["runs_new_ppo_update"])
        self.assertTrue(summary["experimental_checkpoint"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["performance_claimed"])
        self.assertFalse(summary["formal_release_claimed"])
        self.assertGreater(summary["coverage_return_improvement"], 0.0)
        self.assertGreater(summary["valuable_area_covered_improvement"], 0.0)
        self.assertEqual(summary["controlled_regression_count"], 0)
        self.assertEqual(summary["fallback_rate"], 0.0)
        self.assertEqual(summary["optimizer_train_transition_count"], 2)
        self.assertEqual(summary["coverage_aware_reward_transition_count"], 2)
        self.assertLessEqual(summary["old_log_prob_max_abs_error"], 1.0e-4)
        self.assertLessEqual(summary["old_value_max_abs_error"], 1.0e-4)
        self.assertTrue(Path(summary["checkpoint_path"]).is_file())

        batch_rows = self._read_jsonl(self.output_root / "coverage-aware-ppo-batch" / "ppo-rollout-transitions.jsonl")
        self.assertEqual(len(batch_rows), 2)
        self.assertTrue(all(math.isfinite(row["log_prob"]) for row in batch_rows))
        self.assertTrue(all(math.isfinite(row["value"]) for row in batch_rows))
        self.assertEqual(batch_rows[0]["reward"], rows[0]["coverage_aware_reward"])
        self.assertIn("coverage_component", batch_rows[0]["reward_components"])
        self.assertEqual(batch_rows[0]["profile_hash"], self.reward_profile.profile_hash)
        self.assertEqual(batch_rows[0]["info"]["controlled_action_index"], 0)
        self.assertEqual(batch_rows[0]["info"]["source_reward_audit_index"], 0)

        for filename in (
            "coverage-driven-ppo-improvement-run-summary.json",
            "coverage-driven-ppo-update-summary.json",
            "coverage-driven-ppo-training-curves.json",
            "coverage-driven-ppo-diagnostics.json",
            "coverage-driven-experimental-policy-candidate.pt",
            "coverage-driven-experimental-policy-candidate-metadata.json",
            "coverage-driven-ppo-performance-metric-table.jsonl",
            "coverage-driven-ppo-replay-audit.json",
            "coverage-driven-ppo-rejection-report.json",
            "coverage-driven-ppo-improvement-run-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_fails_explicitly_when_coverage_reward_update_is_teacher_equivalent(self) -> None:
        from scripts.run_coverage_driven_ppo_improvement_run import (
            run_coverage_driven_ppo_improvement_run,
        )

        rows = [
            self._reward_row("ctx-selected-0", step_index=0, coverage_delta=0.10, valuable=0.04),
            self._reward_row("ctx-selected-1", step_index=1, coverage_delta=0.08, valuable=0.03),
        ]
        self._write_inputs(rows, baseline_coverage_return=0.1792, baseline_valuable=0.07)

        summary = run_coverage_driven_ppo_improvement_run(
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            coverage_signal_root=self.signal_root,
            coverage_performance_root=self.performance_root,
            reward_refinement_root=self.reward_root,
            reward_source_root=self.reward_source_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["coverage_driven_ppo_improvement_status"], "failed")
        self.assertIn("no_coverage_return_improvement", summary["reason_codes"])
        self.assertIn("valuable_coverage_not_improved", summary["reason_codes"])
        self.assertTrue(summary["runs_new_ppo_update"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["performance_claimed"])

    def test_rejects_coverage_reward_rows_without_profile_hash(self) -> None:
        from scripts.run_coverage_driven_ppo_improvement_run import (
            run_coverage_driven_ppo_improvement_run,
        )

        rows = [self._reward_row("ctx-selected-0", step_index=0, coverage_delta=0.20, valuable=0.12)]
        rows[0].pop("profile_hash")
        self._write_inputs(rows, baseline_coverage_return=0.10, baseline_valuable=0.05)

        summary = run_coverage_driven_ppo_improvement_run(
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            coverage_signal_root=self.signal_root,
            coverage_performance_root=self.performance_root,
            reward_refinement_root=self.reward_root,
            reward_source_root=self.reward_source_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("canonical_reward_profile_hash_missing", summary["reason_codes"])
        self.assertFalse(summary["runs_new_ppo_update"])

    def test_blocks_update_when_reward_source_contract_is_not_passed(self) -> None:
        from scripts.run_coverage_driven_ppo_improvement_run import (
            run_coverage_driven_ppo_improvement_run,
        )

        rows = [self._reward_row("ctx-selected-0", step_index=0, coverage_delta=0.20, valuable=0.12)]
        self._write_inputs(
            rows,
            baseline_coverage_return=0.10,
            baseline_valuable=0.05,
            reward_refinement_status="failed",
            source_missing_component_count=1,
        )

        summary = run_coverage_driven_ppo_improvement_run(
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            coverage_signal_root=self.signal_root,
            coverage_performance_root=self.performance_root,
            reward_refinement_root=self.reward_root,
            reward_source_root=self.reward_source_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("coverage_reward_source_not_passed", summary["reason_codes"])
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertEqual(summary["optimizer_train_transition_count"], 0)
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["performance_claimed"])

    def test_v3_profile_rejects_incomplete_stage18_9_readiness_before_ppo_update(self) -> None:
        from model_explorer.policy.canonical_reward import load_canonical_reward_profile
        from scripts.run_coverage_driven_ppo_improvement_run import (
            run_coverage_driven_ppo_improvement_run,
        )

        self.reward_profile = load_canonical_reward_profile(
            self.repo_root / "configs" / "xunce_canonical_reward_guard_profile_v3.json"
        )
        rows = [
            self._reward_row("ctx-selected-0", step_index=0, coverage_delta=0.20, valuable=0.12),
            self._reward_row("ctx-selected-1", step_index=1, coverage_delta=0.18, valuable=0.10),
        ]
        self._write_inputs(rows, baseline_coverage_return=0.10, baseline_valuable=0.05)
        stage18_9_root = self.temp_dir / "stage18-9-incomplete"
        self._write_stage18_9_summary(
            stage18_9_root,
            include_readiness=False,
            include_guard_details=False,
        )

        summary = run_coverage_driven_ppo_improvement_run(
            formal_training_root=self.formal_root,
            post_training_replay_root=self.replay_root,
            selected_candidate_root=self.selected_root,
            coverage_signal_root=self.signal_root,
            coverage_performance_root=self.performance_root,
            reward_refinement_root=self.reward_root,
            reward_source_root=self.reward_source_root,
            output_root=self.output_root,
            repo_root=self.repo_root,
            stage18_9_trajectory_risk_reward_root=stage18_9_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("stage18_9_readiness_not_passed", summary["reason_codes"])
        self.assertFalse(summary["runs_new_ppo_update"])

    def _write_inputs(
        self,
        rows: list[dict],
        *,
        baseline_coverage_return: float,
        baseline_valuable: float,
        reward_refinement_status: str = "passed",
        source_missing_component_count: int = 0,
    ) -> None:
        self._write_json(
            self.formal_root / "formal-ppo-training-run-summary.json",
            {
                "schema_version": "guarded-formal-ppo-training-run-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "seed_count": 5,
                "optimizer_train_transition_count": 684,
                "teacher_agreement_rate": 1.0,
                "controlled_regression_count": 0,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
            },
        )
        self._write_jsonl(
            self.formal_root / "formal-ppo-training-run-seed-summaries.jsonl",
            [{"seed": seed, "status": "passed", "controlled_regression_count": 0} for seed in range(5)],
        )
        self._write_json(
            self.replay_root / "formal-ppo-post-training-stability-replay-summary.json",
            {
                "schema_version": "guarded-formal-ppo-post-training-stability-replay-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "teacher_agreement_rate": 1.0,
                "controlled_regression_count": 0,
                "runs_new_ppo_update": False,
            },
        )
        self._write_json(
            self.selected_root / "selected-formal-ppo-candidate-promotion-preflight-summary.json",
            {
                "schema_version": "selected-formal-ppo-candidate-promotion-preflight-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "selected_seed": 0,
                "selected_budget": "epochs1_lr3e-6",
                "selected_candidate_root": str(self.base_candidate_root),
                "checkpoint_path": str(self.base_candidate_root / "experimental-hybrid-policy-candidate.pt"),
                "checkpoint_metadata_path": str(self.base_candidate_root / "experimental-hybrid-policy-candidate-metadata.json"),
                "multihorizon_steps": str(self.selected_root / "multihorizon-shadow-rollout-steps.jsonl"),
                "teacher_agreement_rate": 1.0,
                "controlled_regression_count": 0,
                "runs_new_ppo_update": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
            },
        )
        self._write_jsonl(
            self.selected_root / "multihorizon-shadow-rollout-steps.jsonl",
            [self._shadow_step(row) for row in rows],
        )
        self._write_json(
            self.signal_root / "exploration-coverage-signal-audit-summary.json",
            {
                "schema_version": "exploration-coverage-signal-audit-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "coverage_signal_status": "passed",
                "actual_coverage_gain_source": "path_feedback",
                "nonzero_actual_coverage_delta_count": len(rows),
                "expected_actual_coverage_confusion_count": 0,
                "fallback_coverage_gain_claimed_as_policy_gain_count": 0,
                "controlled_regression_count": 0,
                "coverage_delta_audit": str(self.signal_root / "coverage-delta-audit.jsonl"),
            },
        )
        self._write_jsonl(self.signal_root / "coverage-delta-audit.jsonl", rows)

        selected_metric = self._metric(
            "selected_ppo_candidate",
            coverage_return=baseline_coverage_return,
            cumulative=baseline_coverage_return,
            valuable=baseline_valuable,
        )
        teacher_metric = self._metric(
            "teacher",
            coverage_return=baseline_coverage_return,
            cumulative=baseline_coverage_return,
            valuable=baseline_valuable,
        )
        source_metric = self._metric(
            "source_default",
            coverage_return=max(baseline_coverage_return - 0.01, 0.0),
            cumulative=max(baseline_coverage_return - 0.01, 0.0),
            valuable=max(baseline_valuable - 0.01, 0.0),
        )
        self._write_json(
            self.performance_root / "exploration-coverage-performance-evaluation-summary.json",
            {
                "schema_version": "exploration-coverage-performance-evaluation-summary/v1",
                "status": "failed",
                "reason_codes": ["coverage_performance_not_improved", "valuable_coverage_not_improved"],
                "coverage_performance_status": "failed",
                "selected_actor": "selected_ppo_candidate",
                "best_baseline_actor": "teacher",
                "selected_metrics": selected_metric,
                "best_baseline_metrics": teacher_metric,
                "coverage_return_improvement": 0.0,
                "valuable_area_covered_improvement": 0.0,
                "controlled_regression_count": 0,
                "metric_table": str(self.performance_root / "coverage-performance-metric-table.jsonl"),
                "comparison_audit": str(self.performance_root / "coverage-performance-comparison-audit.json"),
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
            },
        )
        self._write_jsonl(
            self.performance_root / "coverage-performance-metric-table.jsonl",
            [selected_metric, teacher_metric, source_metric],
        )
        self._write_json(
            self.performance_root / "coverage-performance-comparison-audit.json",
            {
                "schema_version": "coverage-performance-comparison-audit/v1",
                "selected_actor": "selected_ppo_candidate",
                "best_baseline_actor": "teacher",
                "baseline_actors": ["teacher", "source_default"],
                "selected_teacher_equivalent": True,
            },
        )
        self._write_json(
            self.reward_root / "coverage-aware-reward-refinement-summary.json",
            {
                "schema_version": "coverage-aware-reward-refinement-summary/v1",
                "status": reward_refinement_status,
                "reason_codes": [] if reward_refinement_status == "passed" else ["reward_component_source_missing"],
                "reward_refinement_status": reward_refinement_status,
                "next_required_change": "coverage_driven_ppo_improvement_run",
                "reward_component_audit": str(self.reward_root / "reward-component-audit.jsonl"),
                "profile_id": self.reward_profile.profile_id,
                "profile_version": self.reward_profile.profile_version,
                "profile_hash": self.reward_profile.profile_hash,
                "source_field_missing_component_count": source_missing_component_count,
                "component_source_overlay_row_count": len(rows),
                "controlled_regression_count": 0,
                "expected_actual_coverage_confusion_count": 0,
                "fallback_coverage_gain_claimed_as_policy_gain_count": 0,
                "runs_new_ppo_update": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
                "formal_release_claimed": False,
            },
        )
        self._write_jsonl(self.reward_root / "reward-component-audit.jsonl", rows)
        self._write_json(
            self.reward_source_root / "connect-reward-component-source-fields-summary.json",
            {
                "schema_version": "connect-reward-component-source-fields-summary/v1",
                "status": reward_refinement_status,
                "reason_codes": [] if reward_refinement_status == "passed" else ["reward_component_source_missing"],
                "reward_component_source_field_status": reward_refinement_status,
                "audited_row_count": len(rows),
                "connected_row_count": 0 if source_missing_component_count else len(rows),
                "controlled_regression_count": 0,
                "expected_actual_coverage_confusion_count": 0,
                "fallback_coverage_gain_claimed_as_policy_gain_count": 0,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
            },
        )

    def _write_stage18_9_summary(
        self,
        root: Path,
        *,
        include_readiness: bool = True,
        include_guard_details: bool = True,
    ) -> None:
        payload = {
            "schema_version": "xunce-stage18-9-trajectory-risk-reward-summary/v1",
            "status": "passed",
            "profile_id": self.reward_profile.profile_id,
            "profile_version": self.reward_profile.profile_version,
            "profile_hash": self.reward_profile.profile_hash,
            "trajectory_guard_passed": True,
            "stage19_authorized": False,
            "next_stage_routing": {
                "schema_version": "xunce-stage18-9-next-stage-routing/v1",
                "primary_route": "prepare_stage19_evaluator_critic_preflight",
                "stage19_authorized": False,
            },
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
        if include_readiness:
            payload["stage19_readiness"] = {
                "schema_version": "xunce-stage18-9-stage19-readiness/v1",
                "readiness": "ready_for_stage19_preflight_human_review_only",
                "authorized": False,
                "trajectory_guard_passed": True,
            }
        if include_guard_details:
            payload["path_risk_boundary_summary"] = {
                "path_risk_boundary_passed": True,
                "hard_risk_violation_count": 0,
            }
            payload["trajectory_guard_summary"] = {
                "coverage_advantage_established": True,
                "path_cost_budget_passed": True,
                "coverage_efficiency_passed": True,
                "soft_risk_exposure_passed": True,
            }
        self._write_json(root / "xunce-stage18-9-trajectory-risk-reward-summary.json", payload)

    def _observation(self):
        from model_explorer.policy.features import PolicyObservation

        return PolicyObservation(
            candidate_feature_names=("value", "risk", "path_cost", "energy_cost"),
            candidate_features=((1.0, 0.1, 1.0, 0.1),),
            global_feature_names=("step_index",),
            global_features=(0.0,),
            action_mask=(True,),
            candidate_cells=((1, 2),),
            candidate_missing_feature_names=((),),
            candidate_missing_indicator_names=(),
            candidate_missing_indicators=((),),
        )

    def _write_base_candidate(self) -> None:
        import torch
        from model_explorer.policy.architectures import build_policy_network

        torch.manual_seed(13)
        network = build_policy_network(None, observation=self.observation, hidden_size=8)
        checkpoint = {
            "schema_version": "controlled-hybrid-policy-candidate-checkpoint/v1",
            "experimental": True,
            "architecture": network.architecture_name,
            "model_state_dict": network.state_dict(),
            "training": {"hidden_size": 8, "seed": 13},
            "git_provenance": {"current_matches_sources": True},
        }
        import torch as torch_module

        torch_module.save(checkpoint, self.base_candidate_root / "experimental-hybrid-policy-candidate.pt")
        self._write_json(
            self.base_candidate_root / "experimental-hybrid-policy-candidate-metadata.json",
            {
                "schema_version": "controlled-hybrid-policy-candidate-checkpoint-metadata/v1",
                "experimental": True,
                "checkpoint_path": "experimental-hybrid-policy-candidate.pt",
                "architecture": network.architecture_name,
                "hidden_size": 8,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
                "git_provenance": {"current_matches_sources": True},
            },
        )
        self._write_json(
            self.base_candidate_root / "raw-policy-generalization-candidate-summary.json",
            {
                "schema_version": "raw-policy-generalization-candidate-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "checkpoint_path": "experimental-hybrid-policy-candidate.pt",
                "checkpoint_metadata_path": "experimental-hybrid-policy-candidate-metadata.json",
                "experimental_checkpoint": True,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "performance_claimed": False,
            },
        )

    def _policy_log_prob_and_value(self) -> tuple[float, float]:
        import torch
        from model_explorer.policy.architectures import build_policy_network
        from model_explorer.policy.torch_policy import observation_to_tensors

        checkpoint = torch.load(
            self.base_candidate_root / "experimental-hybrid-policy-candidate.pt",
            map_location="cpu",
            weights_only=False,
        )
        network = build_policy_network(None, observation=self.observation, hidden_size=8)
        network.load_state_dict(checkpoint["model_state_dict"])
        with torch.no_grad():
            output = network(**observation_to_tensors(self.observation))
            distribution = torch.distributions.Categorical(logits=output.masked_logits)
            log_prob = float(distribution.log_prob(torch.tensor([0])).item())
            value = float(output.value[0].item())
        return log_prob, value

    def _reward_row(self, context_id: str, *, step_index: int, coverage_delta: float, valuable: float) -> dict:
        from model_explorer.policy.canonical_reward import compute_canonical_reward_components

        metrics = {
            "coverage_gain_rate": coverage_delta,
            "valuable_coverage": valuable,
            "roi_coverage": valuable,
            "information_gain": coverage_delta,
            "path_cost_m": 1.0,
            "risk_proxy": 0.1,
            "soft_risk_exposure": 0.1,
            "path_allowed_by_risk": True,
            "hard_risk_violation_count": 0,
            "fallback_used": False,
            "failure": False,
        }
        component_result = compute_canonical_reward_components(
            metrics,
            self.reward_profile,
        )
        reward_components = component_result.components
        if self.reward_profile.profile_version == "v3":
            source_fields = {
                "coverage_component": "coverage_rate_delta",
                "roi_coverage_component": "path_feedback.coverage_rate_delta*path_feedback.candidates.utility",
                "information_component": "path_feedback.coverage_rate_delta",
                "path_cost_component": "observation.candidate_features.path_cost",
                "soft_risk_component": "path_feedback.candidates.soft_risk_exposure",
                "fallback_component": "controlled_choice_source",
                "failure_component": "failure_reason",
            }
            source_values = {
                "coverage_component": coverage_delta,
                "roi_coverage_component": valuable,
                "information_component": coverage_delta,
                "path_cost_component": 1.0,
                "soft_risk_component": 0.1,
                "fallback_component": 0.0,
                "failure_component": 0.0,
            }
        else:
            source_fields = {
                "coverage_component": "coverage_rate_delta",
                "valuable_coverage_component": "path_feedback.coverage_rate_delta*path_feedback.candidates.utility",
                "information_component": "path_feedback.coverage_rate_delta",
                "path_cost_component": "observation.candidate_features.path_cost",
                "risk_component": "path_feedback.candidates.risk",
                "fallback_component": "controlled_choice_source",
                "failure_component": "failure_reason",
            }
            source_values = {
                "coverage_component": coverage_delta,
                "valuable_coverage_component": valuable,
                "information_component": coverage_delta,
                "path_cost_component": 1.0,
                "risk_component": 0.1,
                "fallback_component": 0.0,
                "failure_component": 0.0,
            }
        return {
            "schema_version": "coverage-aware-reward-component-audit-row/v1",
            "reward_audit_index": step_index,
            "actor": "selected_ppo_candidate",
            "claimed_actor": "selected_ppo_candidate",
            "episode_id": "selected-episode",
            "source_episode_id": "selected-episode",
            "step_index": step_index,
            "source_step_index": step_index,
            "context_id": context_id,
            "scenario_id": f"scenario-{step_index}",
            "scenario_family": "family-a" if step_index == 0 else "family-b",
            "split": "train",
            "controlled_choice_source": "policy",
            "controlled_choice_detail": "policy_teacher_aligned",
            "controlled_action_index": 0,
            "teacher_action_index": 0,
            "actual_coverage_gain_source": "path_feedback",
            "coverage_rate_delta": coverage_delta,
            "final_coverage_rate": coverage_delta,
            "cumulative_coverage_rate_delta": coverage_delta,
            "expected_coverage_rate_delta": 0.0,
            "fallback_like": False,
            "fallback_policy_gain_contamination": False,
            "expected_actual_coverage_confusion": False,
            "controlled_regression_reason_codes": [],
            "reward_components": reward_components,
            "coverage_aware_reward": component_result.reward,
            "profile_id": self.reward_profile.profile_id,
            "profile_version": self.reward_profile.profile_version,
            "profile_hash": self.reward_profile.profile_hash,
            "source_fields": source_fields,
            "source_values": source_values,
            "path_cost": 1.0,
            "risk": 0.1,
            "soft_risk_exposure": 0.1,
            "energy_cost": 0.1,
            "new_area_covered": coverage_delta,
            "valuable_area_covered": valuable,
            "information_gain": coverage_delta,
        }

    def _shadow_step(self, row: dict) -> dict:
        log_prob, value = self._policy_log_prob_and_value()
        return {
            "schema_version": "selected-formal-ppo-candidate-multihorizon-shadow-step/v1",
            "episode_id": row["source_episode_id"],
            "shadow_episode_id": row["episode_id"],
            "step_index": row["source_step_index"],
            "shadow_step_index": row["step_index"],
            "context_id": row["context_id"],
            "scenario_id": row["scenario_id"],
            "scenario_family": row["scenario_family"],
            "split": "train",
            "controlled_choice_source": "policy",
            "controlled_choice_detail": "policy_teacher_aligned",
            "controlled_action_index": 0,
            "teacher_action_index": 0,
            "raw_policy_action_index": 0,
            "policy_takes_control": True,
            "ppo_trainable": True,
            "log_prob": log_prob,
            "value": value,
            "reward": 1.0,
            "reward_components": {"teacher_following_bonus": 1.0},
            "path_cost_delta": 1.0,
            "risk_delta": 0.1,
            "energy_cost_delta": 0.1,
            "controlled_regression_reason_codes": [],
            "gate_reason_codes": [],
            "observation": {
                "candidate_feature_names": list(self.observation.candidate_feature_names),
                "candidate_features": [list(item) for item in self.observation.candidate_features],
                "global_feature_names": list(self.observation.global_feature_names),
                "global_features": list(self.observation.global_features),
                "action_mask": list(self.observation.action_mask),
                "candidate_cells": [list(item) for item in self.observation.candidate_cells],
                "candidate_missing_feature_names": [
                    list(item) for item in self.observation.candidate_missing_feature_names
                ],
                "candidate_missing_indicator_names": list(self.observation.candidate_missing_indicator_names),
                "candidate_missing_indicators": [
                    list(item) for item in self.observation.candidate_missing_indicators
                ],
            },
        }

    def _metric(
        self,
        actor: str,
        *,
        coverage_return: float,
        cumulative: float,
        valuable: float,
    ) -> dict:
        return {
            "schema_version": "coverage-performance-metric-row/v1",
            "actor": actor,
            "row_count": 2,
            "episode_count": 1,
            "coverage_return": round(coverage_return, 12),
            "cumulative_coverage_rate_delta": round(cumulative, 12),
            "final_coverage_rate": round(cumulative, 12),
            "new_area_covered": round(cumulative, 12),
            "valuable_area_covered": round(valuable, 12),
            "information_gain": round(cumulative, 12),
            "path_cost": 2.0,
            "risk": 0.2,
            "energy_cost": 0.2,
            "coverage_gain_per_path_cost": round(cumulative / 2.0, 12),
            "coverage_gain_per_risk": round(cumulative / 0.2, 12),
            "coverage_gain_per_energy": round(cumulative / 0.2, 12),
            "accepted_policy_activation_rate": 1.0 if actor == "selected_ppo_candidate" else 0.0,
            "fallback_rate": 0.0,
            "teacher_agreement_rate": 1.0,
            "controlled_regression_count": 0,
            "fallback_coverage_gain": 0.0,
            "comparator_basis": [],
        }

    def _write_json(self, path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _write_jsonl(self, path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )

    def _read_jsonl(self, path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
