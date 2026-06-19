import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not available")
class XunceHighFidelityExplorationCoverageComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_path = str(self.repo_root / "scripts")
        if scripts_path not in sys.path:
            sys.path.insert(0, scripts_path)
        model_explorer_src = str(self.repo_root / "model-explorer" / "src")
        if model_explorer_src not in sys.path:
            sys.path.insert(0, model_explorer_src)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="xunce-coverage-comparison-"))
        self.expansion_root = self.temp_dir / "expansion"
        self.output_root = self.temp_dir / "coverage-comparison"
        self.config_path = self.temp_dir / "config.json"
        self.xunce_checkpoint = self.temp_dir / "xunce.pt"
        self.incumbent_checkpoint = self.temp_dir / "incumbent.pt"
        self._write_xunce_checkpoint()
        self._write_incumbent_checkpoint()
        self._write_expansion_evidence()
        self._write_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_runs_true_checkpoint_coverage_rollout_without_proxy_selection(self) -> None:
        from scripts.run_xunce_high_fidelity_exploration_coverage_comparison import (
            run_xunce_high_fidelity_exploration_coverage_comparison,
        )

        summary = run_xunce_high_fidelity_exploration_coverage_comparison(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["scenario_count"], 24)
        self.assertEqual(summary["rollout_steps"], 10)
        self.assertTrue(summary["true_model_inference_executed"])
        self.assertFalse(summary["proxy_selection_used"])
        self.assertTrue(summary["xunce_checkpoint_loaded"])
        self.assertTrue(summary["incumbent_checkpoint_loaded"])
        self.assertEqual(summary["model_inference_mask_violation_count"], 0)
        self.assertEqual(summary["open_grid_fallback_count"], 0)
        self.assertFalse(summary["xunce_coverage_advantage_established"])
        self.assertTrue(summary["comparison_allowed"])
        self.assertEqual(summary["next_required_change"], "review_xunce_incumbent_comparison_metrics")
        self.assertIn("xunce_coverage_advantage_not_established", summary["diagnostic_reason_codes"])
        self.assertIn("xunce_coverage_return_delta_vs_incumbent", summary)
        self.assertIn("xunce_coverage_curve_auc_delta_vs_incumbent", summary)
        self.assertIn("coverage_gain_per_path_cost_delta_vs_incumbent", summary)
        self.assertIn("policy_disagreement_count", summary)
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])
        self.assertFalse(summary["runs_new_ppo_update"])

        expected_files = (
            "xunce-exploration-coverage-comparison-summary.json",
            "xunce-exploration-coverage-episodes.jsonl",
            "xunce-exploration-coverage-steps.jsonl",
            "xunce-exploration-coverage-model-inference.jsonl",
            "xunce-exploration-coverage-roi-breakdown.json",
            "xunce-exploration-coverage-decision-audit.json",
            "xunce-exploration-coverage-comparison-report.md",
        )
        for filename in expected_files:
            self.assertTrue((self.output_root / filename).is_file(), filename)

        step_rows = self._read_jsonl(self.output_root / "xunce-exploration-coverage-steps.jsonl")
        self.assertEqual(len(step_rows), 24 * 10 * 2)
        self.assertTrue(all(row["true_model_inference_executed"] for row in step_rows))
        self.assertTrue(all("coverage_rate_delta" in row for row in step_rows))
        self.assertTrue(all("coverage_gain_per_path_cost" in row for row in step_rows))

        episode_rows = self._read_jsonl(self.output_root / "xunce-exploration-coverage-episodes.jsonl")
        self.assertEqual(len(episode_rows), 24 * 2)
        first_episode = episode_rows[0]
        self.assertIn("coverage_return", first_episode)
        self.assertIn("coverage_curve_auc", first_episode)
        self.assertIn("revisit_rate", first_episode)
        self.assertIn("min_roi_group_coverage_rate", first_episode)

    def test_masked_unreachable_candidate_is_not_selected(self) -> None:
        from scripts.run_xunce_high_fidelity_exploration_coverage_comparison import (
            run_xunce_high_fidelity_exploration_coverage_comparison,
        )

        self._write_expansion_evidence(mask_first_candidate=True)
        summary = run_xunce_high_fidelity_exploration_coverage_comparison(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["model_inference_mask_violation_count"], 0)
        step_rows = self._read_jsonl(self.output_root / "xunce-exploration-coverage-steps.jsonl")
        self.assertTrue(step_rows)
        self.assertTrue(all(row["selected_action_index"] != 0 for row in step_rows))

    def test_revisits_do_not_increase_new_coverage(self) -> None:
        from scripts.run_xunce_high_fidelity_exploration_coverage_comparison import (
            run_xunce_high_fidelity_exploration_coverage_comparison,
        )

        self._write_expansion_evidence(repeated_cells=True)
        summary = run_xunce_high_fidelity_exploration_coverage_comparison(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        episode_rows = self._read_jsonl(self.output_root / "xunce-exploration-coverage-episodes.jsonl")
        self.assertTrue(any(row["revisit_rate"] > 0.0 for row in episode_rows))
        self.assertTrue(any(row["new_covered_cell_count"] < row["executed_step_count"] for row in episode_rows))

    def test_missing_xunce_checkpoint_routes_to_sandbox_preflight_fix(self) -> None:
        from scripts.run_xunce_high_fidelity_exploration_coverage_comparison import (
            run_xunce_high_fidelity_exploration_coverage_comparison,
        )

        self.xunce_checkpoint.unlink()
        summary = run_xunce_high_fidelity_exploration_coverage_comparison(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("missing_xunce_candidate_checkpoint", summary["reason_codes"])
        self.assertEqual(summary["next_required_change"], "fix_xunce_sandbox_candidate_preflight")

    def test_coverage_gain_with_efficiency_regression_does_not_authorize(self) -> None:
        from scripts import run_xunce_high_fidelity_exploration_coverage_comparison as module

        original_run_policy = module._run_policy_episode

        def fake_episode(*, policy_name, **kwargs):
            row = original_run_policy(policy_name=policy_name, **kwargs)
            if policy_name == "xunce":
                row["coverage_return"] += 1.0
                row["coverage_curve_auc"] += 1.0
                row["cumulative_coverage_rate_delta"] += 1.0
                row["path_cost"] += 10_000.0
                row["risk"] += 10_000.0
                row["coverage_gain_per_path_cost"] = 0.0
                row["coverage_gain_per_risk"] = 0.0
            return row

        module._run_policy_episode = fake_episode
        try:
            summary = module.run_xunce_high_fidelity_exploration_coverage_comparison(
                config_path=self.config_path,
                output_root=self.output_root,
                repo_root=self.repo_root,
            )
        finally:
            module._run_policy_episode = original_run_policy

        self.assertEqual(summary["status"], "passed")
        self.assertFalse(summary["xunce_coverage_advantage_established"])
        self.assertGreater(summary["xunce_efficiency_regression_count"], 0)
        self.assertEqual(summary["next_required_change"], "review_xunce_incumbent_comparison_metrics")
        self.assertIn("xunce_coverage_advantage_with_efficiency_regression", summary["diagnostic_reason_codes"])
        self.assertEqual(summary["diagnostic_recommended_change"], "refine_coverage_reward_and_cost_guard")

    def test_dynamic_v2_legacy_mode_does_not_offset_candidates_and_writes_oracle_artifacts(self) -> None:
        from scripts.run_xunce_high_fidelity_exploration_coverage_comparison import (
            run_xunce_high_fidelity_exploration_coverage_comparison,
        )

        self._write_expansion_evidence(dynamic_validated_candidates=True)
        self._update_config(
            candidate_refresh_mode="dynamic_from_coverage_memory",
            coverage_metric_mode="path_line_plus_endpoint",
            include_oracle_baselines=True,
            include_roi_weighted_coverage=True,
        )
        summary = run_xunce_high_fidelity_exploration_coverage_comparison(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["candidate_refresh_mode"], "dynamic_validated_only")
        self.assertIn("dynamic_from_coverage_memory_legacy_mode", summary["diagnostic_reason_codes"])
        self.assertEqual(summary["coverage_metric_mode"], "path_line_plus_endpoint")
        self.assertTrue(summary["include_oracle_baselines"])
        self.assertTrue(summary["true_model_inference_executed"])
        self.assertIn("xunce_oracle_regret", summary)
        self.assertIn("incumbent_oracle_regret", summary)
        self.assertIn("evaluation_task_discriminative", summary)
        self.assertTrue((self.output_root / "xunce-exploration-coverage-comparison-v2-summary.json").is_file())
        self.assertTrue((self.output_root / "xunce-exploration-coverage-v2-steps.jsonl").is_file())
        self.assertTrue((self.output_root / "xunce-exploration-coverage-v2-episodes.jsonl").is_file())

        v2_steps = self._read_jsonl(self.output_root / "xunce-exploration-coverage-v2-steps.jsonl")
        xunce_rows = [row for row in v2_steps if row["policy"] == "xunce"]
        self.assertTrue(xunce_rows)
        xunce_step_0 = next(row for row in xunce_rows if row["step_index"] == 0)
        xunce_step_1 = next(row for row in xunce_rows if row["step_index"] == 1)
        self.assertEqual(xunce_step_0["candidate_cells"], [[1, 1], [2, 2], [3, 3]])
        self.assertEqual(xunce_step_1["candidate_cells"], [[101, 51], [102, 52], [103, 53]])
        self.assertFalse(xunce_step_1["dynamic_candidate_validation_missing"])
        self.assertTrue(all(row["policy_inference_kind"] == "true_checkpoint_inference" for row in xunce_rows))
        self.assertTrue(all(row["oracle_rollout_executed"] is False for row in xunce_rows))
        self.assertTrue(any(row["policy"] == "greedy_coverage_oracle" for row in v2_steps))
        self.assertTrue(any(row["policy"] == "cost_aware_coverage_oracle" for row in v2_steps))
        oracle_rows = [row for row in v2_steps if row["policy"].endswith("_oracle")]
        self.assertTrue(oracle_rows)
        self.assertTrue(all(row["executed"] for row in oracle_rows))
        self.assertTrue(all(row["true_model_inference_executed"] is False for row in oracle_rows))
        self.assertTrue(all(row["oracle_rollout_executed"] is True for row in oracle_rows))
        self.assertTrue(all(row["policy_inference_kind"] == "oracle_offline_policy" for row in oracle_rows))

    def test_dynamic_validated_only_uses_validated_dynamic_candidates(self) -> None:
        from scripts.run_xunce_high_fidelity_exploration_coverage_comparison import (
            run_xunce_high_fidelity_exploration_coverage_comparison,
        )

        self._write_expansion_evidence(dynamic_validated_candidates=True)
        self._update_config(
            candidate_refresh_mode="dynamic_validated_only",
            coverage_metric_mode="path_line_plus_endpoint",
            include_oracle_baselines=True,
        )
        summary = run_xunce_high_fidelity_exploration_coverage_comparison(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["candidate_refresh_mode"], "dynamic_validated_only")
        v2_steps = self._read_jsonl(self.output_root / "xunce-exploration-coverage-v2-steps.jsonl")
        xunce_step_1 = next(row for row in v2_steps if row["policy"] == "xunce" and row["step_index"] == 1)
        self.assertEqual(xunce_step_1["candidate_cells"], [[101, 51], [102, 52], [103, 53]])
        self.assertFalse(xunce_step_1["dynamic_candidate_validation_missing"])
        self.assertNotIn("dynamic_candidate_validation_missing", xunce_step_1["reason_codes"])

    def test_dynamic_validated_only_records_diagnostic_when_validation_missing(self) -> None:
        from scripts.run_xunce_high_fidelity_exploration_coverage_comparison import (
            run_xunce_high_fidelity_exploration_coverage_comparison,
        )

        self._write_expansion_evidence(dynamic_validated_candidates="missing_step")
        self._update_config(
            candidate_refresh_mode="dynamic_validated_only",
            coverage_metric_mode="path_line_plus_endpoint",
        )
        summary = run_xunce_high_fidelity_exploration_coverage_comparison(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertIn("dynamic_candidate_validation_missing", summary["diagnostic_reason_codes"])
        self.assertTrue(summary["candidate_validity_gate_passed"])
        self.assertGreater(summary["dynamic_candidate_validation_missing_count"], 0)
        v2_steps = self._read_jsonl(self.output_root / "xunce-exploration-coverage-v2-steps.jsonl")
        xunce_step_1 = next(row for row in v2_steps if row["policy"] == "xunce" and row["step_index"] == 1)
        self.assertEqual(xunce_step_1["candidate_cells"], [[1, 1], [2, 2], [3, 3]])
        self.assertTrue(xunce_step_1["dynamic_candidate_validation_missing"])
        self.assertIn("dynamic_candidate_validation_missing", xunce_step_1["reason_codes"])

    def test_dynamic_v2_oracles_do_not_select_masked_candidate(self) -> None:
        from scripts.run_xunce_high_fidelity_exploration_coverage_comparison import (
            run_xunce_high_fidelity_exploration_coverage_comparison,
        )

        self._write_expansion_evidence(mask_first_candidate=True, dynamic_validated_candidates=True)
        self._update_config(
            candidate_refresh_mode="dynamic_from_coverage_memory",
            coverage_metric_mode="path_line_plus_endpoint",
            include_oracle_baselines=True,
        )
        summary = run_xunce_high_fidelity_exploration_coverage_comparison(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["model_inference_mask_violation_count"], 0)
        v2_steps = self._read_jsonl(self.output_root / "xunce-exploration-coverage-v2-steps.jsonl")
        oracle_rows = [row for row in v2_steps if row["policy"].endswith("_oracle")]
        self.assertTrue(oracle_rows)
        self.assertTrue(all(row["selected_action_index"] != 0 for row in oracle_rows))

    def test_dynamic_v2_requires_oracle_regret_efficiency_and_safety_for_authorization(self) -> None:
        from scripts import run_xunce_high_fidelity_exploration_coverage_comparison as module

        self._write_expansion_evidence(dynamic_validated_candidates=True)
        self._update_config(
            candidate_refresh_mode="dynamic_from_coverage_memory",
            coverage_metric_mode="path_line_plus_endpoint",
            include_oracle_baselines=True,
        )
        original_run_policy = module._run_policy_episode

        def fake_episode(*, policy_name, **kwargs):
            row = original_run_policy(policy_name=policy_name, **kwargs)
            if policy_name == "xunce":
                row["coverage_return"] = 5.0
                row["coverage_curve_auc"] = 5.0
                row["cumulative_coverage_rate_delta"] = 5.0
                row["new_covered_cell_count"] = 500
                row["coverage_gain_per_path_cost"] = 1.0
                row["coverage_gain_per_risk"] = 1.0
                row["path_cost"] = 5.0
                row["risk"] = 0.1
            elif policy_name == "incumbent":
                row["coverage_return"] = 2.0
                row["coverage_curve_auc"] = 2.0
                row["cumulative_coverage_rate_delta"] = 2.0
                row["new_covered_cell_count"] = 200
                row["coverage_gain_per_path_cost"] = 0.5
                row["coverage_gain_per_risk"] = 0.5
                row["path_cost"] = 4.0
                row["risk"] = 0.1
            elif policy_name == "greedy_coverage_oracle":
                row["coverage_return"] = 6.0
                row["coverage_curve_auc"] = 6.0
            return row

        module._run_policy_episode = fake_episode
        try:
            summary = module.run_xunce_high_fidelity_exploration_coverage_comparison(
                config_path=self.config_path,
                output_root=self.output_root,
                repo_root=self.repo_root,
            )
        finally:
            module._run_policy_episode = original_run_policy

        self.assertFalse(summary["xunce_coverage_advantage_established"])
        self.assertGreater(summary["xunce_efficiency_regression_count"], 0)
        self.assertEqual(summary["next_required_change"], "review_xunce_incumbent_comparison_metrics")
        self.assertIn("xunce_coverage_advantage_with_efficiency_regression", summary["diagnostic_reason_codes"])
        self.assertEqual(summary["diagnostic_recommended_change"], "refine_coverage_reward_and_cost_guard")

    def _write_xunce_checkpoint(self) -> None:
        import torch

        from scripts.xunce_full_network_common import XunceFullNetworkV1

        config = self._xunce_model_config()
        model = XunceFullNetworkV1(
            candidate_feature_count=config["candidate_feature_count"],
            edge_feature_count=config["edge_feature_count"],
            memory_feature_count=config["memory_feature_count"],
            context_feature_count=config["context_feature_count"],
            missing_indicator_count=config["missing_indicator_count"],
            hidden_dim=config["hidden_dim"],
            message_passing_layers=config["message_passing_layers"],
            dropout=0.0,
        )
        for parameter in model.parameters():
            torch.nn.init.constant_(parameter, 0.0)
        torch.save(
            {
                "schema_version": "xunce-controlled-training-candidate-checkpoint/v1",
                "model_state_dict": model.state_dict(),
                "metadata": {
                    "schema_version": "xunce-controlled-training-checkpoint-metadata/v1",
                    "architecture": "xunce_full_network_v1",
                    **config,
                    "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                    "publishes_checkpoint": False,
                    "default_policy_replacement_approved": False,
                },
            },
            self.xunce_checkpoint,
        )

    def _write_incumbent_checkpoint(self) -> None:
        import torch

        from model_explorer.policy.features import (
            CANDIDATE_FEATURE_NAMES,
            GLOBAL_FEATURE_NAMES,
            MISSING_INDICATOR_NAMES,
        )
        from model_explorer.policy.torch_policy import MaskedCandidatePolicyNetwork

        network = MaskedCandidatePolicyNetwork(
            candidate_feature_count=len(CANDIDATE_FEATURE_NAMES),
            global_feature_count=len(GLOBAL_FEATURE_NAMES),
            hidden_size=16,
            missing_indicator_count=len(MISSING_INDICATOR_NAMES),
        )
        for parameter in network.parameters():
            torch.nn.init.constant_(parameter, 0.0)
        torch.save(
            {
                "state_dict": network.state_dict(),
                "hidden_size": 16,
                "candidate_feature_names": tuple(CANDIDATE_FEATURE_NAMES),
                "global_feature_names": tuple(GLOBAL_FEATURE_NAMES),
                "candidate_missing_indicator_names": tuple(MISSING_INDICATOR_NAMES),
                "metadata": {
                    "format": "model-explorer-masked-policy",
                    "version": 2,
                    "format_version": "model-explorer-masked-policy/v2",
                    "architecture": "mlp_v1",
                    "architecture_config": {"hidden_dim": 16, "dropout": 0.0},
                    "candidate_feature_names": tuple(CANDIDATE_FEATURE_NAMES),
                    "global_feature_names": tuple(GLOBAL_FEATURE_NAMES),
                    "candidate_missing_indicator_names": tuple(MISSING_INDICATOR_NAMES),
                    "action_count": 3,
                },
            },
            self.incumbent_checkpoint,
        )

    @staticmethod
    def _xunce_model_config() -> dict[str, int]:
        return {
            "seed": 18,
            "candidate_feature_count": 8,
            "edge_feature_count": 5,
            "memory_feature_count": 6,
            "context_feature_count": 7,
            "missing_indicator_count": 3,
            "hidden_dim": 32,
            "message_passing_layers": 2,
            "candidate_count": 3,
        }

    def _write_config(self) -> None:
        payload = {
            "schema_version": "xunce-high-fidelity-exploration-coverage-comparison-config/v1",
            "source_roi_expansion_root": str(self.expansion_root),
            "xunce_candidate_checkpoint": str(self.xunce_checkpoint),
            "incumbent_policy_checkpoint": str(self.incumbent_checkpoint),
            "required_scenario_count": 24,
            "rollout_steps": 10,
            "coverage_radius_cells": 1,
            "coverage_radius_sensitivity": [3],
            "planning_backend": "channel_aware_astar",
            "allow_open_grid_fallback": False,
            "require_context_ids": True,
            "require_contract_and_sidecar_paths": True,
            "max_xunce_parameter_count": 10_000_000,
            "max_median_inference_latency_ms": 5.0,
            "max_latency_ratio_vs_incumbent": 10.0,
            "canary_traffic_fraction": 0.0,
        }
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _update_config(self, **updates) -> None:
        payload = json.loads(self.config_path.read_text(encoding="utf-8"))
        payload.update(updates)
        self.config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_expansion_evidence(
        self,
        *,
        repeated_cells: bool = False,
        mask_first_candidate: bool = False,
        dynamic_validated_candidates: bool | str = False,
    ) -> None:
        self.expansion_root.mkdir(parents=True, exist_ok=True)
        scenarios = []
        slices = []
        splits = ["train", "validation", "test"]
        for index in range(24):
            group = f"roi_{index // 3}"
            split = splits[index % 3]
            scenario_id = f"scenario_{index:03d}"
            base_x = index * 20
            cells = [[base_x + 1, 1], [base_x + 2, 2], [base_x + 3, 3]]
            if repeated_cells:
                cells = [[1, 1], [1, 1], [1, 1]]
            candidates = []
            for action_index, cell in enumerate(cells):
                reachable = not (mask_first_candidate and action_index == 0)
                candidates.append(
                    {
                        "action_index": action_index,
                        "cell": cell,
                        "reachable": reachable,
                        "path_cost": 5.0 + action_index,
                        "risk": 0.1 + action_index * 0.01,
                        "energy_cost": 2.0 + action_index,
                        "expected_coverage_rate_delta": 0.02 + action_index * 0.005,
                        "expected_new_coverage_area": 1.0 + action_index,
                        "information_gain": 0.1 + action_index * 0.1,
                        "value": 0.2 + action_index * 0.1,
                    }
                )
                if dynamic_validated_candidates:
                    candidates[-1]["dynamic_validated_candidates"] = [
                        {
                            "step_index": 0,
                            "cell": cell,
                            "reachable": reachable,
                            "path_cost": 5.0 + action_index,
                            "risk": 0.1 + action_index * 0.01,
                            "open_grid_fallback_used": False,
                            "proposal_validated_by_path_feedback": True,
                        }
                    ]
                    if dynamic_validated_candidates != "missing_step":
                        candidates[-1]["dynamic_validated_candidates"].append(
                            {
                                "step_index": 1,
                                "cell": [base_x + 101 + action_index, 51 + action_index],
                                "reachable": reachable,
                                "path_cost": 15.0 + action_index,
                                "risk": 0.2 + action_index * 0.01,
                                "open_grid_fallback_used": False,
                                "proposal_validated_by_path_feedback": True,
                            }
                        )
            scenarios.append(
                {
                    "scenario_id": scenario_id,
                    "scenario_group": group,
                    "roi_group": group,
                    "selected_cell_after_path_feedback": cells[0],
                    "selected_cell_before_path_feedback": cells[1],
                    "selected_path_cost_after_feedback": 5.0,
                    "selected_path_cost_before_feedback": 6.0,
                    "coverage_rate": 0.0,
                    "coverage_rate_delta": 0.02,
                    "open_grid_fallback_used": False,
                    "tracking_safety_violation_count": 0,
                    "path_feedback": {"candidates": candidates},
                }
            )
            slices.append(
                {
                    "schema_version": "quasi-real-map-slice/v1",
                    "scenario_id": scenario_id,
                    "scenario_group": group,
                    "roi_name": group,
                    "split": split,
                    "context_id": f"context-{scenario_id}",
                    "legacy_identity_fallback_used": False,
                    "contract": str(self.expansion_root / f"{scenario_id}.contract.json"),
                    "sidecar": str(self.expansion_root / f"{scenario_id}.sidecar.json"),
                }
            )
        (self.expansion_root / "xunce-high-fidelity-real-map-slices.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in slices) + "\n",
            encoding="utf-8",
        )
        self._write_json(
            self.expansion_root / "xunce-high-fidelity-real-map-roi-expansion-summary.json",
            {
                "schema_version": "xunce-high-fidelity-real-map-roi-expansion-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "next_required_change": "xunce_high_fidelity_real_map_policy_comparison",
                "slice_count": 24,
                "roi_group_count": 8,
                "context_id_missing_count": 0,
                "legacy_identity_fallback_count": 0,
                "fallback_or_open_grid_count": 0,
                "domain_gap_verdict": "acceptable_for_high_fidelity_comparison",
                "canary_traffic_fraction": 0.0,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "starts_online_canary": False,
                "runs_new_ppo_update": False,
                "modifies_network": False,
                "modifies_action_space": False,
                "modifies_default_astar": False,
                "real_world_release_approved": False,
                "real_world_performance_claimed": False,
            },
        )
        self._write_json(
            self.expansion_root / "xunce-high-fidelity-path-feedback-audit.json",
            {
                "schema_version": "xunce-high-fidelity-path-feedback-audit/v1",
                "scenario_count": 24,
                "candidate_count": 72,
                "reachable_count": 48 if mask_first_candidate else 72,
                "fallback_or_open_grid_count": 0,
                "open_grid_fallback_used": False,
                "scenarios": scenarios,
            },
        )

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    unittest.main()
