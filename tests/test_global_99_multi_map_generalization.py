import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class Global99MultiMapGeneralizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        for path in (self.repo_root / "scripts", self.repo_root / "model-explorer" / "src"):
            path_text = str(path)
            if path_text not in sys.path:
                sys.path.insert(0, path_text)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="global-99-multi-map-"))
        self.source_config_path = self.temp_dir / "global-99-source.json"
        self.frontier_config_path = self.temp_dir / "frontier-config.json"
        self.memory_config_path = self.temp_dir / "memory-config.json"
        self.policy_config_path = self.temp_dir / "policy-guided-config.json"
        self.multi_map_config_path = self.temp_dir / "multi-map-config.json"
        self.policy_root = self.temp_dir / "policy"
        self.output_root = self.temp_dir / "output"
        self.policy_root.mkdir(parents=True, exist_ok=True)
        self._write_policy_candidate(summary_status="passed")
        self._write_policy_guided_sources()
        self._write_multi_map_config()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_default_matrix_passes_and_writes_required_artifacts(self) -> None:
        from scripts.run_global_99_multi_map_generalization import run_global_99_multi_map_generalization

        summary = run_global_99_multi_map_generalization(
            config_path=self.multi_map_config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["scenario_count"], 24)
        self.assertEqual(summary["family_count"], 8)
        self.assertEqual(summary["passed_family_count"], 8)
        self.assertEqual(summary["failed_family_count"], 0)
        self.assertTrue(summary["coverage_target_met_all_required_scenarios"])
        self.assertGreaterEqual(summary["aggregate_achieved_coverage_rate"], 0.99)
        self.assertGreaterEqual(summary["min_scenario_achieved_coverage_rate"], 0.99)
        self.assertTrue(summary["policy_loaded"])
        self.assertTrue(summary["policy_guidance_applied"])
        self.assertGreater(summary["policy_scored_candidate_count"], 0)
        self.assertGreater(summary["policy_guided_decision_count"], 0)
        self.assertEqual(summary["next_required_change"], "network_architecture_upgrade_readiness_review")
        self.assertTrue(summary["uses_policy_guidance"])
        self.assertTrue(summary["uses_checkpoint_inference"])
        self.assertTrue(summary["uses_ppo_policy"])
        self.assertTrue(summary["policy_read_only"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["modifies_network"])
        self.assertFalse(summary["modifies_action_space"])
        self.assertFalse(summary["modifies_default_astar"])
        self.assertFalse(summary["uses_path_planner"])
        self.assertFalse(summary["uses_npz_or_sidecar"])

        for filename in (
            "global-99-multi-map-generalization-summary.json",
            "global-99-multi-map-generalization-manifest.json",
            "global-99-multi-map-scenario-results.jsonl",
            "global-99-multi-map-family-summary.json",
            "global-99-multi-map-policy-vs-baseline-audit.json",
            "global-99-multi-map-budget-audit.json",
            "global-99-multi-map-rejection-report.json",
            "global-99-multi-map-generalization-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_family_summary_tracks_required_and_expected_infeasible_scenarios(self) -> None:
        from scripts.run_global_99_multi_map_generalization import run_global_99_multi_map_generalization

        summary = run_global_99_multi_map_generalization(
            config_path=self.multi_map_config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )
        family_summary = self._read_json(self.output_root / "global-99-multi-map-family-summary.json")

        self.assertIn("budget_limited", family_summary["families"])
        self.assertEqual(family_summary["families"]["open_field"]["scenario_count"], 3)
        self.assertEqual(family_summary["families"]["open_field"]["failed_required_scenario_count"], 0)
        self.assertEqual(family_summary["families"]["budget_limited"]["required_scenario_count"], 0)
        self.assertGreater(family_summary["families"]["budget_limited"]["failed_scenario_count"], 0)
        self.assertIn("insufficient_budget", summary["infeasible_reason_codes"])
        self.assertIn("coverage_target_not_met", summary["infeasible_reason_codes"])

    def test_policy_vs_baseline_audit_records_comparison_counters(self) -> None:
        from scripts.run_global_99_multi_map_generalization import run_global_99_multi_map_generalization

        summary = run_global_99_multi_map_generalization(
            config_path=self.multi_map_config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )
        audit = self._read_json(self.output_root / "global-99-multi-map-policy-vs-baseline-audit.json")

        self.assertEqual(audit["policy_scored_candidate_count"], summary["policy_scored_candidate_count"])
        self.assertIn("baseline_agreement_rate", audit)
        self.assertIn("policy_better_than_baseline_count", audit)
        self.assertIn("policy_worse_than_baseline_count", audit)
        self.assertIn("controlled_regression_count", audit)

    def test_invalid_policy_candidate_summary_fails(self) -> None:
        from scripts.run_global_99_multi_map_generalization import run_global_99_multi_map_generalization

        self._write_policy_candidate(summary_status="failed")
        summary = run_global_99_multi_map_generalization(
            config_path=self.multi_map_config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("policy_guidance_unavailable", summary["reason_codes"])
        self.assertFalse(summary["policy_loaded"])
        self.assertFalse(summary["policy_guidance_applied"])
        self.assertEqual(summary["next_required_change"], "fix_global_99_multi_map_generalization")

    def _write_policy_guided_sources(self) -> None:
        self.source_config_path.write_text(
            json.dumps(
                {
                    "schema_version": "global-99-coverage-benchmark-config/v1",
                    "target_coverage_rate": 0.99,
                    "path_budget_m": 5000.0,
                    "scenarios": [
                        {
                            "scenario_id": "source-placeholder",
                            "grid": {"width": 20, "height": 20, "resolution_m": 1.0},
                            "start_cell": [0, 0],
                            "roi": {"kind": "rect", "x0": 0, "y0": 0, "x1": 20, "y1": 20},
                            "blocked_rectangles": [],
                            "unsafe_rectangles": [],
                            "coverage_events": [],
                        }
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self.frontier_config_path.write_text(
            json.dumps(
                {
                    "schema_version": "frontier-coverage-planner-baseline-config/v1",
                    "source_global_99_config": str(self.source_config_path),
                    "target_coverage_rate": 0.99,
                    "path_budget_m": 5000.0,
                    "coverage_radius_cells": 30,
                    "frontier_step_limit": 256,
                    "revisit_penalty_weight": 1.0,
                    "new_coverage_weight": 4.0,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self.memory_config_path.write_text(
            json.dumps(
                {
                    "schema_version": "coverage-memory-replanning-loop-config/v1",
                    "source_frontier_baseline_config": str(self.frontier_config_path),
                    "source_global_99_config": str(self.source_config_path),
                    "target_coverage_rate": 0.99,
                    "path_budget_m": 5000.0,
                    "coverage_radius_cells": 30,
                    "replanning_cycle_limit": 64,
                    "segment_step_limit": 64,
                    "memory_snapshot_interval": 1,
                    "resume_from_memory_snapshot": None,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        self.policy_config_path.write_text(
            json.dumps(
                {
                    "schema_version": "policy-guided-global-coverage-config/v1",
                    "source_global_99_config": str(self.source_config_path),
                    "source_frontier_baseline_config": str(self.frontier_config_path),
                    "source_coverage_memory_config": str(self.memory_config_path),
                    "target_coverage_rate": 0.99,
                    "path_budget_m": 5000.0,
                    "coverage_radius_cells": 30,
                    "replanning_cycle_limit": 64,
                    "segment_step_limit": 64,
                    "memory_snapshot_interval": 1,
                    "max_policy_candidates": 16,
                    "policy_logit_weight": 0.25,
                    "policy_candidate_root": str(self.policy_root),
                    "policy_checkpoint": "experimental-hybrid-policy-candidate.pt",
                    "policy_checkpoint_metadata": "experimental-hybrid-policy-candidate-metadata.json",
                    "policy_candidate_summary": "raw-policy-generalization-candidate-summary.json",
                    "allow_policy_candidate_git_provenance_mismatch": True,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_multi_map_config(self) -> None:
        self.multi_map_config_path.write_text(
            json.dumps(
                {
                    "schema_version": "global-99-multi-map-generalization-config/v1",
                    "source_policy_guided_config": str(self.policy_config_path),
                    "target_coverage_rate": 0.99,
                    "path_budget_m": 5000.0,
                    "coverage_radius_cells": 30,
                    "scenario_matrix_seed": 99,
                    "scenario_families": [
                        "open_field",
                        "corridor",
                        "rooms",
                        "blocked_roi",
                        "unsafe_patch",
                        "narrow_passage",
                        "cells_roi",
                        "budget_limited",
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_policy_candidate(self, *, summary_status: str) -> None:
        import torch
        from model_explorer.policy.architectures import build_policy_network
        from model_explorer.policy.features import (
            CANDIDATE_FEATURE_NAMES,
            GLOBAL_FEATURE_NAMES,
            MISSING_INDICATOR_NAMES,
            PolicyObservation,
        )

        observation = PolicyObservation(
            candidate_feature_names=CANDIDATE_FEATURE_NAMES,
            candidate_features=(
                tuple([0.0] * len(CANDIDATE_FEATURE_NAMES)),
                tuple([1.0] * len(CANDIDATE_FEATURE_NAMES)),
            ),
            global_feature_names=GLOBAL_FEATURE_NAMES,
            global_features=tuple([0.0] * len(GLOBAL_FEATURE_NAMES)),
            action_mask=(True, True),
            candidate_cells=((0, 0), (1, 0)),
            candidate_missing_indicator_names=MISSING_INDICATOR_NAMES,
            candidate_missing_indicators=(
                tuple([0.0] * len(MISSING_INDICATOR_NAMES)),
                tuple([0.0] * len(MISSING_INDICATOR_NAMES)),
            ),
        )
        network = build_policy_network(None, observation=observation, hidden_size=8)
        with torch.no_grad():
            for parameter in network.parameters():
                parameter.zero_()
            network.candidate_encoder[0].weight[0, 0] = 1.0
            network.candidate_encoder[1].weight.fill_(1.0)
            network.candidate_encoder[4].weight[0, 0] = 1.0
            network.policy_head[0].weight[0, 0] = 1.0
            network.policy_head[-1].weight[0, 0] = 1.0
        torch.save(
            {
                "schema_version": "controlled-hybrid-policy-candidate-checkpoint/v1",
                "experimental": True,
                "architecture": network.architecture_name,
                "training": {"hidden_size": 8, "seed": 99},
                "model_state_dict": network.state_dict(),
            },
            self.policy_root / "experimental-hybrid-policy-candidate.pt",
        )
        (self.policy_root / "experimental-hybrid-policy-candidate-metadata.json").write_text(
            json.dumps(
                {
                    "schema_version": "controlled-hybrid-policy-candidate-checkpoint-metadata/v1",
                    "checkpoint_path": "experimental-hybrid-policy-candidate.pt",
                    "architecture": network.architecture_name,
                    "hidden_size": 8,
                    "experimental": True,
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (self.policy_root / "raw-policy-generalization-candidate-summary.json").write_text(
            json.dumps(
                {
                    "schema_version": "raw-policy-generalization-candidate-summary/v1",
                    "status": summary_status,
                    "reason_codes": [] if summary_status == "passed" else ["synthetic_failure"],
                    "experimental_checkpoint": True,
                    "checkpoint_path": str(self.policy_root / "experimental-hybrid-policy-candidate.pt"),
                    "checkpoint_metadata_path": str(
                        self.policy_root / "experimental-hybrid-policy-candidate-metadata.json"
                    ),
                    "publishes_checkpoint": False,
                    "replaces_default_policy": False,
                    "performance_claimed": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _read_json(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8"))
