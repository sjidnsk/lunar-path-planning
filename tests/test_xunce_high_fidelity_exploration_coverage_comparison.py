import importlib.util
import json
import math
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
            "xunce-exploration-coverage-comparison-pairs.jsonl",
            "xunce-exploration-coverage-comparison-aggregate.json",
            "xunce-exploration-coverage-model-inference.jsonl",
            "xunce-exploration-coverage-roi-breakdown.json",
            "xunce-exploration-coverage-decision-audit.json",
            "xunce-exploration-coverage-comparison-report.md",
        )
        for filename in expected_files:
            self.assertTrue((self.output_root / filename).is_file(), filename)

        inference_rows = self._read_jsonl(self.output_root / "xunce-exploration-coverage-model-inference.jsonl")
        self.assertTrue(inference_rows)
        first_inference = inference_rows[0]
        for field in (
            "scenario_id",
            "step_index",
            "current_cell",
            "covered_cells_hash",
            "candidate_set_hash",
            "selected_action_index",
            "selected_rank",
            "action_probs",
            "logits",
            "masked_logits",
            "value",
            "finite_outputs",
            "latency_ms",
        ):
            self.assertIn(field, first_inference)
        self.assertEqual(first_inference["current_cell"], first_inference["current_cell_before"])
        self.assertEqual(first_inference["selected_rank"], first_inference["detail"]["selected_rank"])
        self.assertEqual(first_inference["action_probs"], first_inference["detail"]["action_probs"])
        self.assertEqual(first_inference["logits"], first_inference["detail"]["logits"])
        self.assertEqual(first_inference["masked_logits"], first_inference["detail"]["masked_logits"])
        self.assertEqual(first_inference["value"], first_inference["detail"]["value"])
        self.assertEqual(first_inference["finite_outputs"], first_inference["detail"]["finite_outputs"])
        self.assertEqual(first_inference["latency_ms"], first_inference["detail"]["latency_ms"])

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
        self.assertIn("total_new_cell_count", first_episode)
        self.assertIn("path_cost_total_m", first_episode)
        self.assertIn("risk_total", first_episode)
        self.assertIn("risk_cost_weighted_total", first_episode)
        self.assertIn("soft_risk_exposure_total", first_episode)
        self.assertIn("path_risk_peak_max", first_episode)
        self.assertIn("hard_risk_violation_count", first_episode)
        self.assertIn("risk_boundary_violation_steps", first_episode)
        self.assertIn("risk_source", first_episode)
        self.assertIn("roi_weighted_coverage_source", first_episode)
        self.assertIn("coverage_per_100m", first_episode)
        self.assertIn("risk_per_100m", first_episode)
        self.assertIn("revisit_rate", first_episode)
        self.assertIn("min_roi_group_coverage_rate", first_episode)

        pair_rows = self._read_jsonl(self.output_root / "xunce-exploration-coverage-comparison-pairs.jsonl")
        self.assertEqual(len(pair_rows), 24)
        first_pair = pair_rows[0]
        for field in (
            "coverage_delta_cells",
            "roi_weighted_coverage_delta",
            "path_cost_delta_m",
            "risk_delta",
            "risk_cost_weighted_delta",
            "soft_risk_exposure_delta",
            "hard_risk_violation_delta",
            "hard_risk_violation_count",
            "path_risk_peak_delta",
            "coverage_per_100m_delta",
            "risk_per_100m_delta",
            "greedy_oracle_coverage_regret_xunce",
            "greedy_oracle_coverage_regret_incumbent",
            "cost_aware_oracle_utility_regret_xunce",
            "cost_aware_oracle_utility_regret_incumbent",
            "path_cost_budget_exceeded",
            "risk_budget_exceeded",
            "risk_cost_weighted_budget_exceeded",
            "coverage_efficiency_regression",
            "coverage_gain_per_path_cost_delta_audit_only",
            "undefined_metric_reason_codes",
        ):
            self.assertIn(field, first_pair)
        self.assertIsInstance(first_pair["undefined_metric_reason_codes"], list)

        aggregate = self._read_json(self.output_root / "xunce-exploration-coverage-comparison-aggregate.json")
        self.assertEqual(aggregate["scenario_count"], 24)
        self.assertEqual(aggregate["profile_id"], "xunce-coverage-cost-risk-budget-v2")
        self.assertEqual(aggregate["profile_version"], "v2")
        self.assertTrue(aggregate["profile_hash"])
        self.assertIn("coverage_delta_cells_mean", aggregate)
        self.assertIn("coverage_delta_cells_median", aggregate)
        self.assertIn("coverage_delta_cells_iqr", aggregate)
        self.assertIn("path_cost_delta_m_mean", aggregate)
        self.assertIn("risk_delta_mean", aggregate)
        self.assertIn("coverage_per_100m_delta_median", aggregate)
        self.assertIn("greedy_oracle_coverage_regret_delta_mean", aggregate)
        self.assertIn("cost_aware_oracle_utility_regret_delta_mean", aggregate)
        self.assertEqual(summary["xunce_path_cost_delta_vs_incumbent"], aggregate["path_cost_delta_m_mean"])
        self.assertEqual(summary["xunce_risk_delta_vs_incumbent"], aggregate["risk_delta_mean"])
        self.assertIn("comparison_utility_profiles", summary)
        self.assertEqual(summary["profile_id"], "xunce-coverage-cost-risk-budget-v2")
        self.assertEqual(summary["profile_version"], "v2")
        self.assertTrue(summary["profile_hash"])
        self.assertEqual(summary["canonical_guard_thresholds"]["max_path_cost_delta_m"], 20.0)
        self.assertEqual(summary["canonical_guard_thresholds"]["max_risk_delta"], 0.5)
        self.assertTrue(summary["comparison_utility_profiles_diagnostic_only"])
        self.assertEqual(summary["stage19_readiness_source"], "canonical_guard_not_utility_profiles")
        manifest = self._read_json(self.output_root / "xunce-exploration-coverage-comparison-manifest.json")
        self.assertEqual(manifest["profile_id"], summary["profile_id"])
        self.assertEqual(manifest["profile_version"], summary["profile_version"])
        self.assertEqual(manifest["profile_hash"], summary["profile_hash"])

    def test_continuous_theta_eval_uses_reachable_proposal_when_mu_is_unreachable(self) -> None:
        import scripts.run_xunce_high_fidelity_exploration_coverage_comparison as hf

        candidates = [
            {"cell": [0, 0], "hybrid_astar_reachable": False},
            {"cell": [1, 0], "hybrid_astar_reachable": False},
        ]
        detail = {
            "logits": [0.0, 2.0],
            "masked_logits": [0.0, 2.0],
            "theta_mu_rad": [math.radians(11.0), math.radians(17.0)],
            "action_probs": [0.119, 0.881],
            "selected_action_index": 1,
            "selected_probability": 0.881,
            "selected_rank": 1,
        }

        original = hf._enrich_candidates_with_hybrid_astar_path_cost

        def fake_enrich(probes, **kwargs):
            for probe in probes:
                theta = float(probe.get("candidate_theta_deg"))
                cell = probe.get("cell")
                reachable = cell == [1, 0] and abs(theta - 0.0) < 1.0e-6
                probe["path_cost_source"] = "hybrid_astar_pose_path/v1"
                probe["hybrid_astar_trajectory_kind"] = "hybrid_astar_pose_path"
                probe["hybrid_astar_reachable"] = reachable
                probe["hybrid_astar_failure_reason"] = None if reachable else "test_unreachable"
                probe["hybrid_astar_path_cost"] = 3.0 if reachable else None
                probe["hybrid_astar_pose_path_hash"] = "reachable-hash" if reachable else None
                probe["point_grid_path_cost_fallback_used"] = False
                probe["default_astar_replaced"] = False
                probe["hybrid_astar_ackermann_feasible_claimed"] = False
                probe["platform_contract_hash"] = "platform-hash"
                probe["max_traversable_slope_deg"] = 30.0
                if reachable:
                    probe["path_cost"] = 3.0
                    probe["reachable"] = True

        hf._enrich_candidates_with_hybrid_astar_path_cost = fake_enrich
        try:
            hf._apply_continuous_theta_hybrid_reachable_eval_policy(
                detail,
                candidates=candidates,
                action_mask=[True, True],
                current_cell=(0, 0),
                current_theta_deg=0.0,
                candidate_set_hash_value="candidate-set",
                step_index=0,
                scenario_id="scenario",
                config={"theta_step_deg": 45.0, "hybrid_astar_pose_path_cost_enabled": True},
                slice_row={},
                repo_root=self.repo_root,
            )
        finally:
            hf._enrich_candidates_with_hybrid_astar_path_cost = original

        self.assertEqual(detail["selected_action_index"], 1)
        self.assertEqual(detail["continuous_theta_eval_policy"], "hybrid_astar_reachable_theta_proposal_argmax/v1")
        self.assertAlmostEqual(float(detail["selected_theta_deg"]), 0.0)
        self.assertTrue(candidates[1]["hybrid_astar_reachable"])
        self.assertEqual(candidates[1]["hybrid_astar_pose_path_hash"], "reachable-hash")

    def test_hybrid_astar_reachability_overrides_legacy_candidate_reachability(self) -> None:
        import scripts.run_xunce_high_fidelity_exploration_coverage_comparison as hf

        candidate = {"cell": [1, 0], "reachable": False, "path_cost": None}
        hf._apply_hybrid_astar_candidate_fields(
            candidate,
            row={
                "hybrid_astar_reachable": True,
                "hybrid_astar_path_cost": 12.5,
                "hybrid_astar_pose_path_hash": "hybrid-hash",
                "hybrid_astar_trajectory_kind": "hybrid_astar_pose_path",
                "hybrid_astar_failure_reason": None,
                "legacy_grid_astar_path_cost": None,
                "hybrid_vs_grid_path_cost_delta": None,
            },
            current_pose=[0.5, 0.5, 0.0],
            platform_contract_hash="platform-hash",
            max_traversable_slope_deg=30.0,
        )

        self.assertTrue(candidate["hybrid_astar_reachable"])
        self.assertTrue(candidate["reachable"])
        self.assertEqual(candidate["path_cost"], 12.5)
        self.assertEqual(candidate["path_cost_source"], "hybrid_astar_pose_path/v1")

    def test_hybrid_astar_path_cost_enabled_enriches_model_inference_rows(self) -> None:
        from scripts.run_xunce_high_fidelity_exploration_coverage_comparison import (
            run_xunce_high_fidelity_exploration_coverage_comparison,
        )

        sidecar_path = self.expansion_root / "scenario_000.sidecar.json"
        sidecar = self._read_json(sidecar_path)
        sidecar["synthetic_los_blocker_cells"] = [[2, 0], [3, 0]]
        sidecar["synthetic_hard_obstacle_cells"] = [[2, 0]]
        sidecar["synthetic_terrain_hash"] = "synthetic-hash"
        sidecar["synthetic_source_kind"] = "synthetic_terrain_obstacle_proxy/v1"
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
        self._update_config(
            required_scenario_count=1,
            rollout_steps=1,
            theta_aware_candidate_viewpoints_enabled=True,
            theta_bin_count=8,
            theta_step_deg=45,
            sensor_fov_deg=90.0,
            sensor_range_cells=2,
            hybrid_astar_pose_path_cost_enabled=True,
            path_cost_source="hybrid_astar_pose_path/v1",
            initial_theta_deg=0.0,
            hybrid_astar_theta_bin_count=72,
            hybrid_astar_goal_theta_tolerance_deg=5.0,
            hybrid_astar_max_iterations=100000,
            hybrid_astar_primitive_duration_s=1.0,
            hybrid_astar_integration_dt_s=0.25,
            hybrid_astar_max_speed_mps=1.0,
            hybrid_astar_max_angular_speed_degps=45.0,
            hybrid_astar_rotation_cost_weight=0.2,
            hybrid_astar_reverse_penalty_weight=0.5,
            hybrid_astar_turn_penalty_weight=0.05,
            platform_contract_hash="platform-hash",
            max_traversable_slope_deg=30.0,
            synthetic_credit_feature_exposure_enabled=True,
            synthetic_terrain_contract_enabled=True,
            synthetic_terrain_hash="synthetic-hash",
            synthetic_source_kind="synthetic_terrain_obstacle_proxy/v1",
        )

        summary = run_xunce_high_fidelity_exploration_coverage_comparison(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        inference_rows = self._read_jsonl(self.output_root / "xunce-exploration-coverage-model-inference.jsonl")
        self.assertTrue(inference_rows)
        first = inference_rows[0]
        self.assertEqual(first["path_cost_source"], "hybrid_astar_pose_path/v1")
        self.assertEqual(first["hybrid_astar_trajectory_kind"], "hybrid_astar_pose_path")
        self.assertIsNotNone(first["hybrid_astar_path_cost"])
        self.assertIsNotNone(first["path_cost"])
        self.assertAlmostEqual(float(first["path_cost"]), float(first["hybrid_astar_path_cost"]))
        self.assertTrue(first["reachable"])
        self.assertTrue(first["hybrid_astar_reachable"])
        self.assertTrue(first["hybrid_astar_pose_path_hash"])
        self.assertIsNotNone(first["legacy_grid_astar_path_cost"])
        self.assertIsNotNone(first["hybrid_vs_grid_path_cost_delta"])
        self.assertFalse(first["default_astar_replaced"])
        self.assertFalse(first["hybrid_astar_ackermann_feasible_claimed"])
        self.assertEqual(first["platform_contract_hash"], "platform-hash")
        self.assertEqual(
            first["xunce_batch_feature_semantic_map"]["feature_contract_id"],
            "synthetic_credit_candidate_features/v1",
        )
        self.assertTrue(first["synthetic_credit_feature_rows"])
        episodes = self._read_jsonl(self.output_root / "xunce-exploration-coverage-episodes.jsonl")
        self.assertTrue(episodes)
        xunce_episode = next(row for row in episodes if row["policy"] == "xunce")
        self.assertGreater(float(xunce_episode["path_cost_total_m"]), 0.0)
        self.assertEqual(xunce_episode["unreachable_selected_count"], 0)

    def test_main_coverable_denominator_excludes_hard_obstacles_but_keeps_los_only_blockers(self) -> None:
        import scripts.run_xunce_high_fidelity_exploration_coverage_comparison as coverage

        sidecar_path = self.temp_dir / "semantic.sidecar.json"
        sidecar_path.write_text(
            json.dumps(
                {
                    "passable_mask": [[True, True, True, True] for _ in range(4)],
                    "physical_obstacle_cells": [[0, 0]],
                    "slope_blocked_cells": [[1, 0]],
                    "blocked_cells": [[2, 0]],
                    "synthetic_hard_obstacle_cells": [[3, 0]],
                    "synthetic_los_blocker_cells": [[0, 1]],
                    "synthetic_high_risk_cells": [[1, 1]],
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        context = coverage.resolve_coverage_denominator(
            {},
            {"sidecar": str(sidecar_path)},
            {
                "coverage_denominator_mode": "main_coverable_cells",
                "coverage_denominator_cells": 999,
                "derive_slope_blocked_cells_from_sidecar_dem": False,
                "max_traversable_slope_deg": 30.0,
            },
            self.repo_root,
        )
        self.assertEqual(context["coverage_denominator_source"], "main_coverable_cells/v1")
        self.assertEqual(context["coverage_denominator_cells"], 12)
        self.assertEqual(context["passable_denominator_cells"], 16)
        self.assertEqual(context["hard_obstacle_cell_count"], 4)
        self.assertEqual(context["synthetic_los_only_blocker_cell_count"], 1)
        self.assertEqual(context["synthetic_los_only_blocker_main_coverable_count"], 1)

        covered_cells = {(0, 1), (1, 1), (3, 3), (3, 0)}
        self.assertEqual(coverage._coverage_count_for_denominator(covered_cells, context), 3)
        fields = coverage._coverage_denominator_episode_fields(covered_cells, context)
        self.assertEqual(fields["main_coverable_denominator_cells"], 12)
        self.assertEqual(fields["main_covered_cell_count"], 3)
        self.assertAlmostEqual(fields["main_coverage_rate"], 3 / 12)
        self.assertAlmostEqual(fields["raw_roi_coverage_rate"], 4 / 16)
        self.assertAlmostEqual(fields["passable_coverage_rate"], 4 / 16)
        self.assertEqual(fields["hazard_observed_cell_count"], 2)
        self.assertAlmostEqual(fields["hazard_observation_rate"], 2 / 4)
        self.assertTrue(fields["coverable_cell_semantics_hash"])
        json.dumps(coverage._public_coverage_denominator_context(context), sort_keys=True)

    def test_roi_valid_cells_denominator_behavior_remains_unchanged(self) -> None:
        import scripts.run_xunce_high_fidelity_exploration_coverage_comparison as coverage

        sidecar_path = self.temp_dir / "roi-valid.sidecar.json"
        sidecar_path.write_text(
            json.dumps({"passable_mask": [[True, False], [True, True]]}, sort_keys=True),
            encoding="utf-8",
        )
        context = coverage.resolve_coverage_denominator(
            {},
            {"sidecar": str(sidecar_path)},
            {"coverage_denominator_mode": "roi_valid_cells", "coverage_denominator_cells": 999},
            self.repo_root,
        )
        self.assertEqual(context["coverage_denominator_source"], "sidecar_passable_mask_valid_cells/v1")
        self.assertEqual(context["coverage_denominator_cells"], 3)

    def test_main_coverable_episode_keeps_raw_new_count_separate_from_main_new_count(self) -> None:
        from scripts.run_xunce_high_fidelity_exploration_coverage_comparison import (
            run_xunce_high_fidelity_exploration_coverage_comparison,
        )

        sidecar_path = self.expansion_root / "scenario_000.sidecar.json"
        sidecar = self._read_json(sidecar_path)
        sidecar["passable_mask"] = [[True, True, True, True] for _ in range(4)]
        sidecar["synthetic_hard_obstacle_cells"] = [[1, 0]]
        sidecar["synthetic_los_blocker_cells"] = [[2, 0]]
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
        self._update_config(
            required_scenario_count=1,
            rollout_steps=1,
            coverage_denominator_mode="main_coverable_cells",
            coverage_denominator_cells=999,
            candidate_refresh_mode="static_from_source",
            emit_candidate_metric_audit=False,
        )

        summary = run_xunce_high_fidelity_exploration_coverage_comparison(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        episodes = self._read_jsonl(self.output_root / "xunce-exploration-coverage-episodes.jsonl")
        self.assertTrue(episodes)
        first = episodes[0]
        self.assertEqual(first["coverage_denominator_source"], "main_coverable_cells/v1")
        self.assertIn("main_covered_cell_count", first)
        self.assertGreaterEqual(first["raw_new_covered_cell_count"], first["new_covered_cell_count"])

    def test_synthetic_credit_feature_exposure_overrides_xunce_batch_features(self) -> None:
        import torch
        import scripts.run_xunce_high_fidelity_exploration_coverage_comparison as hf

        sidecar = self.temp_dir / "synthetic-sidecar.json"
        sidecar.write_text(
            json.dumps(
                {
                    "synthetic_los_blocker_cells": [[2, 0], [3, 0]],
                    "synthetic_hard_obstacle_cells": [[2, 0]],
                    "synthetic_terrain_hash": "synthetic-hash",
                    "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
                }
            ),
            encoding="utf-8",
        )
        adapter = {"xunce_batch": {"candidate_features": torch.zeros((1, 2, 8), dtype=torch.float32)}}
        candidates = [
            {
                "cell": [1, 0],
                "candidate_theta_deg": 0.0,
                "path_cost": 1.0,
                "risk": 0.1,
                "hybrid_astar_path_cost": 2.0,
                "hybrid_astar_reachable": True,
            },
            {
                "cell": [0, 1],
                "candidate_theta_deg": 180.0,
                "path_cost": 1.5,
                "risk": 0.2,
                "hybrid_astar_path_cost": 4.0,
                "hybrid_astar_reachable": True,
            },
        ]

        metadata = hf._apply_synthetic_credit_feature_exposure_to_adapter(
            adapter,
            candidates=candidates,
            current_cell=(0, 0),
            current_theta_deg=0.0,
            covered_cells=set(),
            candidate_set_hash_value="candidate-hash",
            config={
                "synthetic_credit_feature_exposure_enabled": True,
                "coverage_radius_cells": 1,
                "coverage_metric_mode": "endpoint_footprint",
                "sensor_range_cells": 3,
                "sensor_fov_deg": 90.0,
                "hybrid_astar_pose_path_cost_enabled": True,
            },
            slice_row={"sidecar": str(sidecar)},
            obstacle_source_linkage=None,
            repo_root=self.repo_root,
        )

        features = adapter["xunce_batch"]["candidate_features"]
        self.assertEqual(features.shape, (1, 2, 8))
        self.assertEqual(
            metadata["xunce_batch_feature_semantic_map"]["feature_contract_id"],
            "synthetic_credit_candidate_features/v1",
        )
        self.assertEqual(
            metadata["xunce_batch_feature_semantic_map"]["synthetic_pressure_source"],
            "sidecar_candidate_footprint_intersection/v1",
        )
        self.assertGreater(float(features[0, 0, 5]), float(features[0, 1, 5]))
        self.assertGreater(float(features[0, 0, 6]), float(features[0, 1, 6]))

    def test_strict_v3_profile_loads_without_v2_only_risk_guard_keys(self) -> None:
        from scripts import run_xunce_high_fidelity_exploration_coverage_comparison as module

        self._update_config(
            canonical_reward_profile=str(self.repo_root / "configs" / "xunce_canonical_reward_guard_profile_v3.json"),
            include_roi_weighted_coverage=True,
        )

        config = module._load_config(self.config_path, self.repo_root)

        self.assertEqual(config["profile_id"], "xunce-coverage-cost-risk-boundary-v3")
        self.assertEqual(config["profile_version"], "v3")
        self.assertTrue(config["profile_hash"])
        thresholds = config["canonical_guard_thresholds"]
        self.assertIsNone(thresholds["max_risk_delta"])
        self.assertFalse(thresholds["risk_delta_hard_gate_enabled"])
        self.assertTrue(thresholds["candidate_level_risk_delta_guard_is_diagnostic_only"])
        self.assertEqual(thresholds["max_soft_risk_exposure_delta"], 25.0)

    def test_canonical_reward_rerank_oracle_is_diagnostic_only(self) -> None:
        from scripts.run_xunce_high_fidelity_exploration_coverage_comparison import (
            run_xunce_high_fidelity_exploration_coverage_comparison,
        )

        rerank_profile = self.repo_root / "configs" / "xunce_canonical_reward_guard_profile_v3_path_cost_w080.json"
        self._update_config(
            canonical_reward_profile=str(self.repo_root / "configs" / "xunce_canonical_reward_guard_profile_v3.json"),
            canonical_reward_rerank_profile=str(rerank_profile),
            include_canonical_reward_rerank_oracle=True,
            required_scenario_count=2,
            rollout_steps=1,
            include_roi_weighted_coverage=True,
        )

        summary = run_xunce_high_fidelity_exploration_coverage_comparison(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertTrue(summary["include_canonical_reward_rerank_oracle"])
        self.assertEqual(summary["canonical_reward_rerank_profile_id"], "xunce-coverage-cost-risk-boundary-v3-path-cost-w080")
        steps = self._read_jsonl(self.output_root / "xunce-exploration-coverage-steps.jsonl")
        policies = {row["policy"] for row in steps}
        self.assertEqual(policies, {"xunce", "incumbent", "canonical_reward_rerank_oracle"})
        xunce_rows = [row for row in steps if row["policy"] == "xunce"]
        incumbent_rows = [row for row in steps if row["policy"] == "incumbent"]
        oracle_rows = [row for row in steps if row["policy"] == "canonical_reward_rerank_oracle"]
        self.assertTrue(all(row["true_model_inference_executed"] for row in xunce_rows + incumbent_rows))
        self.assertTrue(all(row["oracle_rollout_executed"] is False for row in xunce_rows + incumbent_rows))
        self.assertTrue(all(row["true_model_inference_executed"] is False for row in oracle_rows))
        self.assertTrue(all(row["oracle_rollout_executed"] is True for row in oracle_rows))
        self.assertTrue(all(row["selected_reward_components"] for row in oracle_rows))
        self.assertTrue(all(row["selected_reward_profile_id"] == "xunce-coverage-cost-risk-boundary-v3-path-cost-w080" for row in oracle_rows))
        paired_rows = self._read_jsonl(self.output_root / "xunce-exploration-coverage-paired-decision-audit.jsonl")
        self.assertTrue(paired_rows)
        self.assertFalse(any(row["executing_policy"] == "canonical_reward_rerank_oracle" for row in paired_rows))

    def test_on_policy_oracle_teacher_labels_do_not_change_xunce_rollout(self) -> None:
        from scripts.run_xunce_high_fidelity_exploration_coverage_comparison import (
            run_xunce_high_fidelity_exploration_coverage_comparison,
        )

        teacher_profile = self.repo_root / "configs" / "xunce_canonical_reward_guard_profile_v3_path_cost_w010.json"
        self._update_config(
            canonical_reward_profile=str(self.repo_root / "configs" / "xunce_canonical_reward_guard_profile_v3.json"),
            on_policy_oracle_teacher_profile=str(teacher_profile),
            emit_on_policy_oracle_teacher_labels=True,
            on_policy_oracle_teacher_baseline_policy="xunce",
            required_scenario_count=2,
            rollout_steps=1,
            include_roi_weighted_coverage=True,
            emit_candidate_metric_audit=True,
        )

        summary = run_xunce_high_fidelity_exploration_coverage_comparison(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        audit_path = self.output_root / "xunce-exploration-coverage-on-policy-oracle-teacher-labels.jsonl"
        self.assertEqual(summary["on_policy_oracle_teacher_label_audit"], str(audit_path))
        self.assertGreater(summary["on_policy_oracle_teacher_label_row_count"], 0)
        self.assertEqual(summary["on_policy_oracle_teacher_profile_id"], "xunce-coverage-cost-risk-boundary-v3-path-cost-w010")
        labels = self._read_jsonl(audit_path)
        steps = self._read_jsonl(self.output_root / "xunce-exploration-coverage-steps.jsonl")
        xunce_steps = [row for row in steps if row["policy"] == "xunce"]
        self.assertEqual(len(labels), len(xunce_steps))
        for label, step in zip(labels, xunce_steps):
            self.assertEqual(label["schema_version"], "xunce-on-policy-oracle-teacher-label/v1")
            self.assertEqual(label["baseline_policy"], "xunce")
            self.assertEqual(label["teacher_policy"], "canonical_reward_rerank_oracle")
            self.assertTrue(label["same_candidate_set"])
            self.assertEqual(label["candidate_set_hash"], step["candidate_set_hash"])
            self.assertEqual(label["covered_cells_hash"], step["covered_cells_hash"])
            self.assertEqual(label["xunce_action_index"], step["selected_action_index"])
            self.assertEqual(label["teacher_profile_hash"], summary["on_policy_oracle_teacher_profile_hash"])
            self.assertEqual(label["training_signal_type"], "teacher_imitation_label")
        self.assertFalse(any(row["policy"] == "canonical_reward_rerank_oracle" for row in steps))

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

    def test_dynamic_frontier_nbv_in_process_generates_and_validates_step_candidates(self) -> None:
        from scripts.run_xunce_high_fidelity_exploration_coverage_comparison import (
            run_xunce_high_fidelity_exploration_coverage_comparison,
        )

        self._write_expansion_evidence(metadata_only_roi_indices={1})
        self._update_config(
            required_scenario_count=2,
            rollout_steps=2,
            candidate_refresh_mode="dynamic_frontier_nbv_in_process",
            coverage_metric_mode="path_line_plus_endpoint",
            include_oracle_baselines=True,
            include_roi_weighted_coverage=True,
            dynamic_frontier_radius_cells=[1, 2],
            dynamic_proposal_pool_limit_per_step=8,
            dynamic_max_candidates_per_step=3,
            dynamic_validation_work_root=str(self.temp_dir / "_xunce_dynamic_validation_work"),
            dynamic_validation_max_path_length=1000,
            dynamic_sidecar_fallback_mode="diagnostic_only",
        )

        summary = run_xunce_high_fidelity_exploration_coverage_comparison(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["candidate_refresh_mode"], "dynamic_frontier_nbv_in_process")
        self.assertTrue(summary["dynamic_candidate_generation_executed"])
        self.assertEqual(summary["dynamic_candidate_validation_mode"], "in_process_path_planner_astar_batch")
        self.assertGreater(summary["dynamic_proposal_count"], 0)
        self.assertGreater(summary["dynamic_validation_attempt_count"], 0)
        self.assertGreater(summary["dynamic_validation_success_count"], 0)
        self.assertEqual(summary["dynamic_candidate_generation_missing_count"], 0)
        self.assertEqual(summary["dynamic_contract_sidecar_missing_count"], 0)
        self.assertEqual(summary["dynamic_path_length_preflight_failure_count"], 0)
        self.assertGreater(summary["in_process_batch_astar_validation_count"], 0)
        self.assertEqual(summary["path_planner_route_adapter_success_count"], 0)
        self.assertEqual(summary["path_planner_route_adapter_failure_count"], 0)
        self.assertEqual(summary["path_planner_route_adapter_audit_sample_count"], 0)
        self.assertFalse(summary["adapter_audit_passed"])
        self.assertEqual(summary["sidecar_grid_astar_screening_count"], 0)
        self.assertFalse(summary["dynamic_validation_full_adapter_evidence_passed"])
        self.assertIn("in_process_path_planner_astar_batch", summary["planner_validation_backend_counts"])
        self.assertEqual(summary["validation_evidence_kind_counts"]["in_process_astar_screening"], summary["dynamic_validation_success_count"])
        self.assertGreater(summary["paired_decision_audit_row_count"], 0)
        self.assertIn("closed_loop_dynamic_rollout_summary", summary)
        self.assertIn("same_candidate_set_policy_selection_summary", summary)
        self.assertEqual(summary["candidate_generation_effect_scope"], "dynamic_generator_plus_policy_closed_loop")
        self.assertEqual(summary["model_selection_evidence_scope"], "same_state_same_candidate_set_paired_decision_audit")
        self.assertTrue((self.output_root / "xunce-exploration-coverage-dynamic-proposals.jsonl").is_file())
        self.assertTrue((self.output_root / "xunce-exploration-coverage-dynamic-validation-results.jsonl").is_file())
        self.assertTrue((self.output_root / "xunce-exploration-coverage-dynamic-validation-audit.json").is_file())
        self.assertTrue((self.output_root / "xunce-exploration-coverage-paired-decision-audit.jsonl").is_file())

        steps = self._read_jsonl(self.output_root / "xunce-exploration-coverage-v2-steps.jsonl")
        xunce_steps = [row for row in steps if row["policy"] == "xunce"]
        self.assertTrue(xunce_steps)
        self.assertTrue(all(row["candidate_generation_source"] == "dynamic_frontier_nbv_in_process/v1" for row in xunce_steps))
        self.assertTrue(all(row["state_conditioned_candidate_generation"] is True for row in xunce_steps))
        self.assertTrue(all(row["dynamic_proposal_count"] > 0 for row in xunce_steps))
        self.assertTrue(all(row["dynamic_validated_candidate_count"] > 0 for row in xunce_steps))
        self.assertTrue(all(row["candidate_set_hash"] for row in xunce_steps))
        self.assertNotEqual(xunce_steps[0]["candidate_set_hash"], xunce_steps[1]["candidate_set_hash"])
        oracle_rows = [row for row in steps if row["policy"].endswith("_oracle")]
        self.assertTrue(oracle_rows)
        self.assertTrue(all(row["oracle_rollout_executed"] is True for row in oracle_rows))
        self.assertTrue(all(row["true_model_inference_executed"] is False for row in oracle_rows))

        validations = self._read_jsonl(self.output_root / "xunce-exploration-coverage-dynamic-validation-results.jsonl")
        formal = [row for row in validations if row.get("proposal_only") is False]
        self.assertTrue(formal)
        self.assertTrue(all(row["proposal_validated_by_path_feedback"] is True for row in formal))
        self.assertTrue(all(row["coverage_validated_by_path_feedback"] is False for row in formal))
        self.assertTrue(all(row["policy"] in {"xunce", "incumbent", "greedy_coverage_oracle", "cost_aware_coverage_oracle"} for row in validations))
        self.assertTrue(all(row["current_cell_before"] for row in validations))
        self.assertTrue(all(row["covered_cells_hash"] for row in validations))
        self.assertTrue(all(row["candidate_set_hash"] for row in validations))
        self.assertTrue(all(row["planner_validation_backend"] for row in formal))
        self.assertTrue(all(row["planner_validation_backend"] == "in_process_path_planner_astar_batch" for row in formal))
        self.assertTrue(all(row["validation_evidence_kind"] == "in_process_astar_screening" for row in formal))
        self.assertIn("dynamic_planner_validation_backend_counts", summary)
        paired = self._read_jsonl(self.output_root / "xunce-exploration-coverage-paired-decision-audit.jsonl")
        self.assertTrue(paired)
        self.assertTrue(all(row["same_state_same_candidate_set"] is True for row in paired))
        self.assertTrue(all("xunce_selected_expected_new_coverage_cell_count" in row for row in paired))
        self.assertTrue(all("incumbent_selected_expected_new_coverage_cell_count" in row for row in paired))
        expected_roi_by_scenario = {"scenario_000": "roi_0", "scenario_001": "metadata_roi_1"}
        self._assert_artifact_roi_groups(
            "xunce-exploration-coverage-dynamic-proposals.jsonl",
            expected_roi_by_scenario,
        )
        self._assert_artifact_roi_groups(
            "xunce-exploration-coverage-dynamic-validation-results.jsonl",
            expected_roi_by_scenario,
        )
        self._assert_artifact_roi_groups(
            "xunce-exploration-coverage-steps.jsonl",
            expected_roi_by_scenario,
        )
        self._assert_artifact_roi_groups(
            "xunce-exploration-coverage-v2-steps.jsonl",
            expected_roi_by_scenario,
        )
        self._assert_artifact_roi_groups(
            "xunce-exploration-coverage-episodes.jsonl",
            expected_roi_by_scenario,
        )
        self._assert_artifact_roi_groups(
            "xunce-exploration-coverage-comparison-pairs.jsonl",
            expected_roi_by_scenario,
        )
        self._assert_artifact_roi_groups(
            "xunce-exploration-coverage-paired-decision-audit.jsonl",
            expected_roi_by_scenario,
        )
        self._assert_artifact_roi_groups(
            "xunce-exploration-coverage-model-inference.jsonl",
            expected_roi_by_scenario,
        )
        roi_breakdown = self._read_json(self.output_root / "xunce-exploration-coverage-roi-breakdown.json")
        roi_groups = {row["roi_group"] for row in roi_breakdown["families"]}
        self.assertIn("roi_0", roi_groups)
        self.assertIn("metadata_roi_1", roi_groups)
        self.assertNotIn("unknown", roi_groups)

    def test_candidate_metric_audit_is_emitted_when_enabled(self) -> None:
        from scripts.run_xunce_high_fidelity_exploration_coverage_comparison import (
            run_xunce_high_fidelity_exploration_coverage_comparison,
        )

        self._update_config(
            required_scenario_count=2,
            rollout_steps=1,
            candidate_refresh_mode="dynamic_frontier_nbv_in_process",
            coverage_metric_mode="path_line_plus_endpoint",
            include_roi_weighted_coverage=True,
            dynamic_proposal_pool_limit_per_step=8,
            dynamic_max_candidates_per_step=3,
            dynamic_validation_work_root=str(self.temp_dir / "_xunce_dynamic_validation_work"),
            dynamic_validation_max_path_length=1000,
            emit_candidate_metric_audit=True,
            obstacle_occlusion_enabled=True,
        )

        summary = run_xunce_high_fidelity_exploration_coverage_comparison(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        audit_path = self.output_root / "xunce-exploration-coverage-candidate-metric-audit.jsonl"
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["candidate_metric_audit"], str(audit_path))
        self.assertGreater(summary["candidate_metric_audit_row_count"], 0)
        self.assertEqual(summary["normalized_config"]["dynamic_max_candidates_per_step"], 3)
        self.assertEqual(summary["normalized_config"]["dynamic_proposal_pool_limit_per_step"], 8)
        self.assertTrue(audit_path.is_file())
        rows = self._read_jsonl(audit_path)
        required = {
            "schema_version",
            "scenario_id",
            "split",
            "roi_group",
            "policy",
            "executing_policy",
            "step_index",
            "current_cell",
            "covered_cells_hash",
            "candidate_set_id",
            "candidate_set_hash",
            "candidate_index",
            "candidate_cell",
            "action_mask_valid",
            "expected_new_coverage_cell_count",
            "roi_weighted_coverage_delta",
            "path_cost",
            "risk",
            "risk_proxy",
            "risk_cost_weighted",
            "path_allowed_by_risk",
            "hard_risk_flags",
            "soft_risk_flags",
            "soft_risk_exposure",
            "path_risk_exposure",
            "path_risk_peak",
            "high_risk_distance_m",
            "recovery_margin_min",
            "risk_semantics_source",
            "risk_proxy_is_physical_risk",
            "risk_source",
            "risk_proxy_source",
            "coverage_gain_per_path_cost",
            "profile_id",
            "profile_version",
            "profile_hash",
            "obstacle_occlusion_enabled",
            "obstacle_source",
            "obstacle_source_missing",
        }
        self.assertTrue(rows)
        self.assertTrue(all(required <= row.keys() for row in rows))
        self.assertTrue(all(row["obstacle_occlusion_enabled"] is True for row in rows))
        self.assertTrue(all(row["obstacle_source_missing"] is True for row in rows))
        self.assertTrue(all(row["profile_hash"] == summary["profile_hash"] for row in rows))
        paired_rows = self._read_jsonl(self.output_root / "xunce-exploration-coverage-paired-decision-audit.jsonl")
        paired_keys = {
            (row["scenario_id"], row["step_index"], row["candidate_set_hash"], row["covered_cells_hash"])
            for row in paired_rows
        }
        candidate_keys = {
            (row["scenario_id"], row["step_index"], row["candidate_set_hash"], row["covered_cells_hash"])
            for row in rows
        }
        self.assertTrue(paired_keys)
        self.assertTrue(paired_keys <= candidate_keys)
        manifest = self._read_json(self.output_root / "xunce-exploration-coverage-comparison-manifest.json")
        self.assertEqual(manifest["candidate_metric_audit_row_count"], len(rows))
        self.assertEqual(manifest["normalized_config"]["dynamic_max_candidates_per_step"], 3)

    def test_obstacle_source_artifact_links_candidate_audit_rows(self) -> None:
        from scripts.run_xunce_high_fidelity_exploration_coverage_comparison import (
            run_xunce_high_fidelity_exploration_coverage_comparison,
        )

        self._write_expansion_evidence(blocked_rectangles=[[1, 0, 1, 2]])
        self._update_config(
            required_scenario_count=1,
            rollout_steps=1,
            theta_aware_candidate_viewpoints_enabled=True,
            theta_bin_count=8,
            theta_step_deg=45,
            sensor_fov_deg=90.0,
            sensor_range_cells=3,
            emit_candidate_metric_audit=True,
            emit_obstacle_source_audit=True,
            obstacle_occlusion_enabled=True,
        )

        summary = run_xunce_high_fidelity_exploration_coverage_comparison(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        source_path = self.output_root / "xunce-exploration-coverage-obstacle-sources.json"
        audit_path = self.output_root / "xunce-exploration-coverage-candidate-metric-audit.jsonl"
        self.assertEqual(summary["status"], "passed")
        self.assertTrue(source_path.is_file())
        sources = self._read_json(source_path)
        self.assertEqual(sources["source_count"], 1)
        self.assertEqual(sources["sources"][0]["obstacle_source_kind"], "blocked_as_obstacle_proxy")
        self.assertTrue(sources["sources"][0]["obstacle_source_hash"])
        rows = self._read_jsonl(audit_path)
        self.assertTrue(rows)
        self.assertTrue(all(row["obstacle_source_missing"] is False for row in rows))
        self.assertTrue(all(row["obstacle_source_id"] == sources["sources"][0]["obstacle_source_id"] for row in rows))
        self.assertTrue(all(row["obstacle_source_hash"] == sources["sources"][0]["obstacle_source_hash"] for row in rows))
        self.assertTrue(all(row["obstacle_source_kind"] == "blocked_as_obstacle_proxy" for row in rows))
        manifest = self._read_json(self.output_root / "xunce-exploration-coverage-comparison-manifest.json")
        self.assertEqual(manifest["obstacle_source_audit"], str(source_path))
        self.assertEqual(manifest["obstacle_source_count"], 1)

    def test_sidecar_passable_mask_false_materializes_blocked_proxy_source(self) -> None:
        from scripts.run_xunce_high_fidelity_exploration_coverage_comparison import (
            run_xunce_high_fidelity_exploration_coverage_comparison,
        )

        self._write_expansion_evidence(blocked_mask_cells={(1, 0)})
        self._update_config(
            required_scenario_count=1,
            rollout_steps=1,
            theta_aware_candidate_viewpoints_enabled=True,
            sensor_fov_deg=90.0,
            sensor_range_cells=3,
            emit_candidate_metric_audit=True,
            emit_obstacle_source_audit=True,
            obstacle_occlusion_enabled=True,
        )

        summary = run_xunce_high_fidelity_exploration_coverage_comparison(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        sources = self._read_json(self.output_root / "xunce-exploration-coverage-obstacle-sources.json")
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(sources["source_count"], 1)
        self.assertEqual(sources["sources"][0]["obstacle_source_payload"], "sidecar")
        self.assertEqual(sources["sources"][0]["obstacle_source_field"], "passable_mask_false")
        self.assertEqual(sources["sources"][0]["obstacle_source_kind"], "blocked_as_obstacle_proxy")
        self.assertIn([1, 0], sources["sources"][0]["obstacle_cells"])
        rows = self._read_jsonl(self.output_root / "xunce-exploration-coverage-candidate-metric-audit.jsonl")
        self.assertTrue(all(row["obstacle_source_missing"] is False for row in rows))

    def test_sidecar_blocked_cells_materializes_blocked_proxy_source(self) -> None:
        from scripts.run_xunce_high_fidelity_exploration_coverage_comparison import (
            run_xunce_high_fidelity_exploration_coverage_comparison,
        )

        self._write_expansion_evidence()
        sidecar_path = self.expansion_root / "scenario_000.sidecar.json"
        sidecar = self._read_json(sidecar_path)
        sidecar["blocked_cells"] = [[2, 0]]
        sidecar["blocked_source_kind"] = "passable_mask_false"
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
        self._update_config(
            required_scenario_count=1,
            rollout_steps=1,
            theta_aware_candidate_viewpoints_enabled=True,
            sensor_fov_deg=90.0,
            sensor_range_cells=3,
            emit_candidate_metric_audit=True,
            emit_obstacle_source_audit=True,
            obstacle_occlusion_enabled=True,
        )

        summary = run_xunce_high_fidelity_exploration_coverage_comparison(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        sources = self._read_json(self.output_root / "xunce-exploration-coverage-obstacle-sources.json")
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(sources["source_count"], 1)
        self.assertEqual(sources["sources"][0]["obstacle_source_payload"], "sidecar")
        self.assertEqual(sources["sources"][0]["obstacle_source_field"], "blocked_cells")
        self.assertEqual(sources["sources"][0]["obstacle_source_kind"], "blocked_as_obstacle_proxy")
        self.assertIn([2, 0], sources["sources"][0]["obstacle_cells"])

    def test_sidecar_slope_blocked_cells_materializes_slope_proxy_source(self) -> None:
        from scripts.run_xunce_high_fidelity_exploration_coverage_comparison import (
            run_xunce_high_fidelity_exploration_coverage_comparison,
        )

        self._write_expansion_evidence()
        sidecar_path = self.expansion_root / "scenario_000.sidecar.json"
        sidecar = self._read_json(sidecar_path)
        sidecar["slope_blocked_cells"] = [[2, 0]]
        sidecar["slope_blocked_source_kind"] = "slope_gt_max_traversable_deg"
        sidecar["max_traversable_slope_deg"] = 20.0
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
        self._update_config(
            required_scenario_count=1,
            rollout_steps=1,
            theta_aware_candidate_viewpoints_enabled=True,
            sensor_fov_deg=90.0,
            sensor_range_cells=3,
            emit_candidate_metric_audit=True,
            emit_obstacle_source_audit=True,
            obstacle_occlusion_enabled=True,
        )

        summary = run_xunce_high_fidelity_exploration_coverage_comparison(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        sources = self._read_json(self.output_root / "xunce-exploration-coverage-obstacle-sources.json")
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(sources["source_count"], 1)
        self.assertEqual(sources["sources"][0]["obstacle_source_payload"], "sidecar")
        self.assertEqual(sources["sources"][0]["obstacle_source_field"], "slope_blocked_cells")
        self.assertEqual(sources["sources"][0]["obstacle_source_kind"], "slope_blocked_as_obstacle_proxy")
        self.assertTrue(sources["sources"][0]["obstacle_source_is_proxy"])
        self.assertIn([2, 0], sources["sources"][0]["obstacle_cells"])
        rows = self._read_jsonl(self.output_root / "xunce-exploration-coverage-candidate-metric-audit.jsonl")
        self.assertTrue(all(row["obstacle_source_kind"] == "slope_blocked_as_obstacle_proxy" for row in rows))

    def test_sidecar_slope_and_synthetic_cells_materialize_effective_synthetic_los_source(self) -> None:
        from scripts.run_xunce_high_fidelity_exploration_coverage_comparison import (
            run_xunce_high_fidelity_exploration_coverage_comparison,
        )

        self._write_expansion_evidence()
        sidecar_path = self.expansion_root / "scenario_000.sidecar.json"
        sidecar = self._read_json(sidecar_path)
        sidecar["slope_blocked_cells"] = [[2, 0]]
        sidecar["synthetic_los_blocker_cells"] = [[3, 0]]
        sidecar["synthetic_hard_obstacle_cells"] = [[4, 0]]
        sidecar["synthetic_terrain_hash"] = "synthetic-hash"
        sidecar["synthetic_source_kind"] = "synthetic_terrain_obstacle_proxy/v1"
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
        self._update_config(
            required_scenario_count=1,
            rollout_steps=1,
            theta_aware_candidate_viewpoints_enabled=True,
            sensor_fov_deg=90.0,
            sensor_range_cells=5,
            emit_candidate_metric_audit=True,
            emit_obstacle_source_audit=True,
            obstacle_occlusion_enabled=True,
            synthetic_terrain_contract_enabled=True,
            synthetic_terrain_hash="synthetic-hash",
            synthetic_source_kind="synthetic_terrain_obstacle_proxy/v1",
        )

        summary = run_xunce_high_fidelity_exploration_coverage_comparison(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        sources = self._read_json(self.output_root / "xunce-exploration-coverage-obstacle-sources.json")
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(sources["source_count"], 1)
        source = sources["sources"][0]
        self.assertEqual(source["obstacle_source_payload"], "effective")
        self.assertEqual(source["obstacle_source_field"], "effective_los_blocker_cells")
        self.assertEqual(source["obstacle_source_kind"], "synthetic_terrain_obstacle_proxy/v1")
        self.assertIn([2, 0], source["obstacle_cells"])
        self.assertIn([3, 0], source["obstacle_cells"])
        self.assertNotIn([4, 0], source["obstacle_cells"])
        self.assertIn("slope_blocked_cells", source["obstacle_source_components"])
        self.assertIn("synthetic_los_blocker_cells", source["obstacle_source_components"])
        rows = self._read_jsonl(self.output_root / "xunce-exploration-coverage-candidate-metric-audit.jsonl")
        self.assertTrue(rows)
        self.assertTrue(all(row["obstacle_source_kind"] == "synthetic_terrain_obstacle_proxy/v1" for row in rows))
        self.assertTrue(all(row["obstacle_cell_count"] >= 2 for row in rows))

    def test_obstacle_source_precedence_prefers_physical_then_slope_then_blocked(self) -> None:
        from scripts.run_xunce_high_fidelity_exploration_coverage_comparison import (
            run_xunce_high_fidelity_exploration_coverage_comparison,
        )

        self._write_expansion_evidence()
        scenario_path = self.expansion_root / "xunce-high-fidelity-path-feedback-audit.json"
        audit = self._read_json(scenario_path)
        audit["scenarios"][0]["obstacle_cells"] = [[1, 0]]
        scenario_path.write_text(json.dumps(audit), encoding="utf-8")
        sidecar_path = self.expansion_root / "scenario_000.sidecar.json"
        sidecar = self._read_json(sidecar_path)
        sidecar["slope_blocked_cells"] = [[2, 0]]
        sidecar["blocked_cells"] = [[3, 0]]
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
        self._update_config(
            required_scenario_count=1,
            rollout_steps=1,
            theta_aware_candidate_viewpoints_enabled=True,
            sensor_fov_deg=90.0,
            sensor_range_cells=3,
            emit_candidate_metric_audit=True,
            emit_obstacle_source_audit=True,
            obstacle_occlusion_enabled=True,
        )

        summary = run_xunce_high_fidelity_exploration_coverage_comparison(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        sources = self._read_json(self.output_root / "xunce-exploration-coverage-obstacle-sources.json")
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(sources["sources"][0]["obstacle_source_kind"], "physical_obstacle_cells")
        self.assertEqual(sources["sources"][0]["obstacle_source_field"], "obstacle_cells")
        self.assertIn([1, 0], sources["sources"][0]["obstacle_cells"])

    def test_sidecar_dem_can_derive_slope_blocked_proxy_source_for_legacy_sidecars(self) -> None:
        from scripts.run_xunce_high_fidelity_exploration_coverage_comparison import (
            run_xunce_high_fidelity_exploration_coverage_comparison,
        )

        self._write_expansion_evidence()
        sidecar_path = self.expansion_root / "scenario_000.sidecar.json"
        sidecar = self._read_json(sidecar_path)
        sidecar["terrain_layers"] = {"dem": [[0.0, 10.0], [0.0, 0.0]]}
        sidecar["blocked_cells"] = [[1, 1]]
        sidecar["metadata"] = {"map_source": {"resolution_m": 20.0}}
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
        self._update_config(
            required_scenario_count=1,
            rollout_steps=1,
            theta_aware_candidate_viewpoints_enabled=True,
            sensor_fov_deg=90.0,
            sensor_range_cells=3,
            emit_candidate_metric_audit=True,
            emit_obstacle_source_audit=True,
            obstacle_occlusion_enabled=True,
            derive_slope_blocked_cells_from_sidecar_dem=True,
            max_traversable_slope_deg=20.0,
        )

        summary = run_xunce_high_fidelity_exploration_coverage_comparison(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        sources = self._read_json(self.output_root / "xunce-exploration-coverage-obstacle-sources.json")
        self.assertEqual(summary["status"], "passed")
        self.assertEqual(sources["source_count"], 1)
        self.assertEqual(sources["sources"][0]["obstacle_source_payload"], "sidecar")
        self.assertEqual(sources["sources"][0]["obstacle_source_field"], "sidecar_dem_slope_gt_max_traversable_deg")
        self.assertEqual(sources["sources"][0]["obstacle_source_kind"], "slope_blocked_as_obstacle_proxy")
        self.assertTrue(sources["sources"][0]["obstacle_cells"])

    def test_dynamic_frontier_nbv_adapter_sample_audit_compares_batch_astar_rows(self) -> None:
        from scripts import run_xunce_high_fidelity_exploration_coverage_comparison as module

        original_validate = module.validate_candidate_cells

        def fake_adapter_validate(**kwargs):
            rows = []
            for proposal in kwargs["proposal_rows"]:
                row = dict(proposal)
                row.update(
                    {
                        "proposal_only": False,
                        "proposal_validated_by_path_feedback": True,
                        "path_feedback_validation_source": "in_process_evaluate_candidate_paths/v1",
                        "planner_validation_backend": "path_planner_route_adapter",
                        "validation_evidence_kind": "full_path_planner_adapter",
                        "path_planner_adapter_audit_status": "passed",
                        "planner_reachable": True,
                        "reachable": True,
                        "open_grid_fallback_used": False,
                        "failure_reason": None,
                        "replan_required": False,
                        "path_cost": float(proposal["path_cost"]),
                        "path_length": float(proposal["path_length"]),
                        "risk": float(proposal["risk"]),
                        "risk_source": "planner_route_result",
                    }
                )
                rows.append(row)
            return rows

        module.validate_candidate_cells = fake_adapter_validate
        try:
            self._write_expansion_evidence()
            self._update_config(
                required_scenario_count=1,
                rollout_steps=1,
                candidate_refresh_mode="dynamic_frontier_nbv_in_process",
                coverage_metric_mode="path_line_plus_endpoint",
                include_oracle_baselines=False,
                dynamic_frontier_radius_cells=[1],
                dynamic_proposal_pool_limit_per_step=4,
                dynamic_max_candidates_per_step=3,
                dynamic_validation_work_root=str(self.temp_dir / "_xunce_dynamic_validation_work"),
                dynamic_validation_max_path_length=1000,
                dynamic_adapter_audit_enabled=True,
                dynamic_adapter_audit_max_routes=2,
                dynamic_adapter_audit_min_per_frontier_source=1,
            )

            summary = module.run_xunce_high_fidelity_exploration_coverage_comparison(
                config_path=self.config_path,
                output_root=self.output_root,
                repo_root=self.repo_root,
            )
        finally:
            module.validate_candidate_cells = original_validate

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["dynamic_candidate_validation_mode"], "in_process_path_planner_astar_batch")
        self.assertGreater(summary["in_process_batch_astar_validation_count"], 0)
        self.assertGreater(summary["path_planner_route_adapter_audit_sample_count"], 0)
        self.assertEqual(summary["adapter_batch_astar_mismatch_count"], 0)
        self.assertTrue(summary["adapter_audit_passed"])
        self.assertFalse(summary["dynamic_validation_full_adapter_evidence_passed"])

    def test_dynamic_candidate_exhaustion_cleanly_terminates_without_model_failure(self) -> None:
        from scripts import run_xunce_high_fidelity_exploration_coverage_comparison as module

        original_dynamic_builder = module.build_dynamic_frontier_nbv_candidates

        def exhausted_dynamic_candidates(**kwargs):
            scenario_id = str(kwargs["scenario"].get("scenario_id", "scenario_000"))
            proposal = {
                "scenario_id": scenario_id,
                "roi_group": "roi_0",
                "proposal_id": f"{scenario_id}:exhausted",
                "cell": [99, 99],
                "proposal_only": True,
                "proposal_validated_by_path_feedback": False,
                "reachable": False,
                "planner_reachable": False,
                "open_grid_fallback_used": False,
                "frontier_candidate_source": "frontier_boundary",
                "candidate_generation_source": "dynamic_frontier_nbv_in_process/v1",
                "coverage_source": "geometric_counterfactual_from_dynamic_frontier_nbv/v1",
                "coverage_validated_by_path_feedback": False,
                "coverage_validation_source": "offline_geometric_counterfactual_not_path_feedback",
                "failure_reason": "proposal_unreachable",
            }
            return [], [proposal], [dict(proposal)]

        module.build_dynamic_frontier_nbv_candidates = exhausted_dynamic_candidates
        try:
            self._update_config(
                required_scenario_count=1,
                rollout_steps=2,
                candidate_refresh_mode="dynamic_frontier_nbv_in_process",
                coverage_metric_mode="path_line_plus_endpoint",
                include_oracle_baselines=False,
                dynamic_validation_work_root=str(self.temp_dir / "_xunce_dynamic_validation_work"),
                dynamic_validation_max_path_length=1000,
            )
            summary = module.run_xunce_high_fidelity_exploration_coverage_comparison(
                config_path=self.config_path,
                output_root=self.output_root,
                repo_root=self.repo_root,
            )
        finally:
            module.build_dynamic_frontier_nbv_candidates = original_dynamic_builder

        self.assertEqual(summary["candidate_generation_exhausted_count"], 2)
        self.assertEqual(summary["dynamic_candidate_generation_missing_count"], 0)
        self.assertEqual(summary["model_inference_mask_violation_count"], 0)
        self.assertEqual(summary["model_inference_failure_count"], 0)
        self.assertEqual(summary["path_planning_failure_count"], 0)
        self.assertNotIn("true_model_inference_not_executed", summary["reason_codes"])
        self.assertNotIn("model_inference_non_finite_output", summary["reason_codes"])
        self.assertNotIn("model_inference_mask_violation", summary["reason_codes"])
        self.assertIn("candidate_generation_exhausted", summary["diagnostic_reason_codes"])

        steps = self._read_jsonl(self.output_root / "xunce-exploration-coverage-steps.jsonl")
        self.assertEqual(len(steps), 2)
        for row in steps:
            self.assertFalse(row["executed"])
            self.assertIsNone(row["selected_action_index"])
            self.assertIsNone(row["selected_cell"])
            self.assertEqual(row["terminal_reason"], "candidate_generation_exhausted")
            self.assertTrue(row["candidate_generation_exhausted"])
            self.assertFalse(row["model_inference_failure"])
            self.assertFalse(row["model_inference_mask_violation"])
            self.assertFalse(row["true_model_inference_executed"])
            self.assertIn("candidate_generation_exhausted", row["reason_codes"])
            self.assertIn("no_valid_dynamic_candidates", row["reason_codes"])

        episodes = self._read_jsonl(self.output_root / "xunce-exploration-coverage-episodes.jsonl")
        self.assertEqual(len(episodes), 2)
        for episode in episodes:
            self.assertEqual(episode["episode_termination_reason"], "candidate_generation_exhausted")
            self.assertTrue(episode["candidate_generation_exhausted"])
            self.assertEqual(episode["candidate_generation_exhausted_step"], 0)
            self.assertEqual(episode["candidate_generation_exhausted_count"], 1)

    def test_model_inference_failure_is_separate_from_candidate_exhaustion(self) -> None:
        from scripts import run_xunce_high_fidelity_exploration_coverage_comparison as module

        original_score = module._score_xunce_model

        def failing_score(*args, **kwargs):
            raise RuntimeError("synthetic scorer failure")

        module._score_xunce_model = failing_score
        try:
            self._update_config(required_scenario_count=1, rollout_steps=1)
            summary = module.run_xunce_high_fidelity_exploration_coverage_comparison(
                config_path=self.config_path,
                output_root=self.output_root,
                repo_root=self.repo_root,
            )
        finally:
            module._score_xunce_model = original_score

        self.assertEqual(summary["status"], "failed")
        self.assertGreater(summary["model_inference_failure_count"], 0)
        self.assertEqual(summary["candidate_generation_exhausted_count"], 0)
        self.assertIn("model_inference_failure", summary["reason_codes"])

    def test_coverage_rate_reports_raw_capped_and_saturation_fields(self) -> None:
        from scripts.run_xunce_high_fidelity_exploration_coverage_comparison import (
            run_xunce_high_fidelity_exploration_coverage_comparison,
        )

        self._update_config(
            required_scenario_count=1,
            rollout_steps=1,
            coverage_denominator_mode="fixed_config_cells",
            coverage_denominator_cells=1,
        )
        summary = run_xunce_high_fidelity_exploration_coverage_comparison(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertGreater(summary["coverage_rate_saturation_episode_count"], 0)
        self.assertGreater(summary["max_final_coverage_rate_raw"], 1.0)
        self.assertGreater(summary["max_coverage_rate_saturation_excess"], 0.0)

        episodes = self._read_jsonl(self.output_root / "xunce-exploration-coverage-episodes.jsonl")
        self.assertTrue(episodes)
        for episode in episodes:
            self.assertGreater(episode["coverage_rate_raw"], 1.0)
            self.assertEqual(episode["coverage_rate_capped"], 1.0)
            self.assertTrue(episode["coverage_saturation_exceeded"])
            self.assertGreater(episode["coverage_saturation_excess"], 0.0)
            self.assertIn("coverage_denominator_mode", episode)

        pairs = self._read_jsonl(self.output_root / "xunce-exploration-coverage-comparison-pairs.jsonl")
        self.assertEqual(len(pairs), 1)
        self.assertIn("xunce_final_coverage_rate_capped", pairs[0])
        self.assertIn("incumbent_final_coverage_rate_capped", pairs[0])
        self.assertIn("coverage_rate_delta_capped", pairs[0])
        self.assertIn("coverage_delta_cells", pairs[0])

        aggregate = self._read_json(self.output_root / "xunce-exploration-coverage-comparison-aggregate.json")
        self.assertEqual(aggregate["coverage_delta_cells_mean"], pairs[0]["coverage_delta_cells"])
        self.assertGreater(aggregate["coverage_rate_saturation_episode_count"], 0)

    def test_roi_valid_cells_denominator_reads_sidecar_passable_mask(self) -> None:
        from scripts.run_xunce_high_fidelity_exploration_coverage_comparison import (
            run_xunce_high_fidelity_exploration_coverage_comparison,
        )

        self._update_config(
            required_scenario_count=1,
            rollout_steps=1,
            coverage_denominator_mode="roi_valid_cells",
        )
        summary = run_xunce_high_fidelity_exploration_coverage_comparison(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        episodes = self._read_jsonl(self.output_root / "xunce-exploration-coverage-episodes.jsonl")
        self.assertTrue(episodes)
        for episode in episodes:
            self.assertEqual(episode["coverage_denominator_mode"], "roi_valid_cells")
            self.assertEqual(episode["coverage_denominator_source"], "sidecar_passable_mask_valid_cells/v1")
            self.assertEqual(episode["coverage_denominator_cells"], 128 * 128)

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

        self.assertTrue(summary["xunce_coverage_advantage_established"])
        self.assertEqual(summary["xunce_efficiency_regression_count"], 0)
        self.assertEqual(summary["next_required_change"], "review_xunce_incumbent_comparison_metrics")
        self.assertNotIn("xunce_coverage_advantage_with_efficiency_regression", summary["diagnostic_reason_codes"])

    def test_pairwise_metrics_record_undefined_ratio_reasons(self) -> None:
        from scripts import run_xunce_high_fidelity_exploration_coverage_comparison as module

        original_run_policy = module._run_policy_episode

        def fake_episode(*, policy_name, **kwargs):
            row = original_run_policy(policy_name=policy_name, **kwargs)
            row["coverage_return"] = 1.0
            row["coverage_curve_auc"] = 1.0
            row["new_covered_cell_count"] = 10
            row["total_new_cell_count"] = 10
            row["valuable_area_covered"] = 10.0
            row["roi_weighted_coverage_total"] = 10.0
            row["path_cost"] = 0.0
            row["path_cost_total_m"] = 0.0
            row["risk"] = 0.0
            row["risk_total"] = 0.0
            row["risk_cost_weighted_total"] = 0.0
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
        pair_rows = self._read_jsonl(self.output_root / "xunce-exploration-coverage-comparison-pairs.jsonl")
        self.assertTrue(pair_rows)
        self.assertTrue(
            all("no_positive_incremental_coverage" in row["undefined_metric_reason_codes"] for row in pair_rows)
        )
        self.assertTrue(all("near_zero_denominator" in row["undefined_metric_reason_codes"] for row in pair_rows))
        self.assertIsNone(pair_rows[0]["incremental_cost_per_extra_cell"])
        self.assertIsNone(pair_rows[0]["xunce_coverage_per_100m"])

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
        metadata_only_roi_indices: set[int] | None = None,
        blocked_rectangles: list[list[int]] | None = None,
        blocked_mask_cells: set[tuple[int, int]] | None = None,
    ) -> None:
        self.expansion_root.mkdir(parents=True, exist_ok=True)
        metadata_only_roi_indices = metadata_only_roi_indices or set()
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
            scenario_row = {
                    "scenario_id": scenario_id,
                    "scenario_group": group,
                    "roi_group": group,
                    "start_cell": [0, 0],
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
            if blocked_rectangles is not None:
                scenario_row["blocked_rectangles"] = blocked_rectangles
            slice_row = {
                    "schema_version": "quasi-real-map-slice/v1",
                    "scenario_id": scenario_id,
                    "scenario_group": group,
                    "roi_name": group,
                    "split": split,
                    "context_id": f"context-{scenario_id}",
                    "legacy_identity_fallback_used": False,
                    "contract": str(self.expansion_root / f"{scenario_id}.contract.json"),
                    "sidecar": str(self.expansion_root / f"{scenario_id}.sidecar.json"),
                    "start_cell": [0, 0],
                    "map_source": {"roi": {"width": 128, "height": 128}},
                }
            if index in metadata_only_roi_indices:
                scenario_row.pop("scenario_group", None)
                scenario_row.pop("roi_group", None)
                slice_row.pop("scenario_group", None)
                slice_row.pop("roi_name", None)
                slice_row["metadata"] = {"roi_group": f"metadata_roi_{index}"}
            scenarios.append(scenario_row)
            slices.append(slice_row)
            self._write_tiny_contract_and_sidecar(
                self.expansion_root / f"{scenario_id}.contract.json",
                self.expansion_root / f"{scenario_id}.sidecar.json",
                width=128,
                height=128,
                blocked_mask_cells=blocked_mask_cells or set(),
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
    def _write_tiny_contract_and_sidecar(
        contract_path: Path,
        sidecar_path: Path,
        *,
        width: int,
        height: int,
        blocked_mask_cells: set[tuple[int, int]] | None = None,
    ) -> None:
        blocked_mask_cells = blocked_mask_cells or set()
        contract = {
            "schema_version": "model-explorer-contract/v1",
            "grid": {
                "width": width,
                "height": height,
                "resolution": 1.0,
                "frame_id": "test",
                "origin": [0.0, 0.0],
                "layers": ["cost"],
            },
            "constraints": {"violation_count": 0, "passable_ratio": 1.0, "reason_counts": {}},
            "top_goals": [{"cell": [1, 0], "utility": 1.0, "reachable": True}],
            "top_sequences": [],
            "observation_update": {"coverage_rate_delta": 0.0},
        }
        sidecar = {
            "schema_version": "path-planner-sidecar/v1",
            "cost": [[1.0 for _ in range(width)] for _ in range(height)],
            "passable_mask": [
                [False if (x, y) in blocked_mask_cells else True for x in range(width)]
                for y in range(height)
            ],
            "metadata": {"fixture": "coverage-comparison"},
        }
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    @staticmethod
    def _read_json(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def _assert_artifact_roi_groups(self, filename: str, expected_by_scenario: dict[str, str]) -> None:
        rows = self._read_jsonl(self.output_root / filename)
        self.assertTrue(rows, filename)
        for row in rows:
            scenario_id = str(row.get("scenario_id"))
            if scenario_id not in expected_by_scenario:
                continue
            self.assertIn("roi_group", row, f"{filename}:{scenario_id}")
            self.assertEqual(row["roi_group"], expected_by_scenario[scenario_id], f"{filename}:{scenario_id}")
            self.assertNotEqual(row["roi_group"], "unknown", f"{filename}:{scenario_id}")


if __name__ == "__main__":
    unittest.main()
