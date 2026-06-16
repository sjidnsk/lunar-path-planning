import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class PolicyGuidedGlobalCoverageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        for path in (self.repo_root / "scripts", self.repo_root / "model-explorer" / "src"):
            path_text = str(path)
            if path_text not in sys.path:
                sys.path.insert(0, path_text)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="policy-guided-global-coverage-"))
        self.source_config_path = self.temp_dir / "global-99-source.json"
        self.frontier_config_path = self.temp_dir / "frontier-config.json"
        self.memory_config_path = self.temp_dir / "memory-config.json"
        self.config_path = self.temp_dir / "policy-guided-config.json"
        self.policy_root = self.temp_dir / "policy"
        self.output_root = self.temp_dir / "output"
        self.policy_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_default_1km_fixture_uses_read_only_policy_guidance_and_reaches_target(self) -> None:
        from scripts.run_policy_guided_global_coverage import run_policy_guided_global_coverage

        self._write_source_config([self._default_1km_scenario()])
        self._write_frontier_config(path_budget_m=5000.0, coverage_radius_cells=30)
        self._write_memory_config(path_budget_m=5000.0, coverage_radius_cells=30)
        self._write_policy_candidate(state_key="model_state_dict", prefer_high_x=True)
        self._write_policy_config(path_budget_m=5000.0, coverage_radius_cells=30)

        summary = run_policy_guided_global_coverage(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertGreaterEqual(summary["achieved_coverage_rate"], 0.99)
        self.assertTrue(summary["coverage_target_met"])
        self.assertTrue(summary["policy_loaded"])
        self.assertTrue(summary["policy_guidance_applied"])
        self.assertGreater(summary["policy_scored_candidate_count"], 0)
        self.assertGreater(summary["policy_guided_decision_count"], 0)
        self.assertEqual(summary["next_required_change"], "global_99_multi_map_generalization")
        self.assertTrue(summary["uses_policy_guidance"])
        self.assertTrue(summary["uses_checkpoint_inference"])
        self.assertTrue(summary["uses_ppo_policy"])
        self.assertTrue(summary["policy_read_only"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["modifies_network"])
        self.assertFalse(summary["modifies_action_space"])
        self.assertFalse(summary["modifies_default_astar"])
        self.assertFalse(summary["uses_path_planner"])
        self.assertFalse(summary["uses_npz_or_sidecar"])

        for filename in (
            "policy-guided-global-coverage-summary.json",
            "policy-guided-global-coverage-manifest.json",
            "policy-guided-global-coverage-decisions.jsonl",
            "policy-guided-global-coverage-ledger.jsonl",
            "policy-guided-global-coverage-memory-snapshots.jsonl",
            "policy-guided-global-coverage-policy-score-audit.json",
            "policy-guided-global-coverage-guard-audit.json",
            "policy-guided-global-coverage-budget-audit.json",
            "policy-guided-global-coverage-rejection-report.json",
            "policy-guided-global-coverage-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

    def test_state_dict_checkpoint_format_is_supported(self) -> None:
        from scripts.run_policy_guided_global_coverage import run_policy_guided_global_coverage

        self._write_source_config([self._default_1km_scenario()])
        self._write_frontier_config(path_budget_m=5000.0, coverage_radius_cells=30)
        self._write_memory_config(path_budget_m=5000.0, coverage_radius_cells=30)
        self._write_policy_candidate(state_key="state_dict", prefer_high_x=True)
        self._write_policy_config(path_budget_m=5000.0, coverage_radius_cells=30)

        summary = run_policy_guided_global_coverage(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertTrue(summary["policy_loaded"])
        self.assertEqual(summary["policy_checkpoint_state_key"], "state_dict")

    def test_guard_falls_back_when_policy_choice_has_no_coverage_progress(self) -> None:
        from scripts.run_policy_guided_global_coverage import run_policy_guided_global_coverage

        self._write_source_config(
            [
                {
                    "scenario_id": "policy-guard-budget-fallback",
                    "grid": {"width": 7, "height": 7, "resolution_m": 1.0},
                    "start_cell": [3, 3],
                    "roi": {"kind": "rect", "x0": 0, "y0": 0, "x1": 4, "y1": 7},
                    "blocked_rectangles": [],
                    "unsafe_rectangles": [],
                    "coverage_events": [],
                }
            ],
            target_coverage_rate=0.90,
        )
        self._write_frontier_config(path_budget_m=40.0, coverage_radius_cells=1)
        self._write_memory_config(path_budget_m=40.0, coverage_radius_cells=1)
        self._write_policy_candidate(state_key="model_state_dict", prefer_high_x=True)
        self._write_policy_config(
            path_budget_m=40.0,
            coverage_radius_cells=1,
            target_coverage_rate=0.90,
            policy_logit_weight=500.0,
        )

        summary = run_policy_guided_global_coverage(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertGreater(summary["policy_guard_fallback_count"], 0)
        self.assertGreaterEqual(summary["achieved_coverage_rate"], 0.90)
        guard_audit = self._read_json(self.output_root / "policy-guided-global-coverage-guard-audit.json")
        self.assertGreater(guard_audit["guard_reason_code_counts"].get("policy_choice_no_coverage_progress", 0), 0)

    def test_invalid_policy_candidate_summary_fails_without_running_guidance(self) -> None:
        from scripts.run_policy_guided_global_coverage import run_policy_guided_global_coverage

        self._write_source_config([self._default_1km_scenario()])
        self._write_frontier_config(path_budget_m=5000.0, coverage_radius_cells=30)
        self._write_memory_config(path_budget_m=5000.0, coverage_radius_cells=30)
        self._write_policy_candidate(state_key="model_state_dict", prefer_high_x=True, summary_status="failed")
        self._write_policy_config(path_budget_m=5000.0, coverage_radius_cells=30)

        summary = run_policy_guided_global_coverage(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("policy_guidance_unavailable", summary["reason_codes"])
        self.assertFalse(summary["policy_loaded"])
        self.assertFalse(summary["policy_guidance_applied"])
        self.assertEqual(summary["next_required_change"], "fix_policy_guided_global_coverage")

    def test_cells_roi_counts_only_explicit_cells(self) -> None:
        from scripts.run_policy_guided_global_coverage import run_policy_guided_global_coverage

        self._write_source_config(
            [
                {
                    "scenario_id": "policy-cells-roi",
                    "grid": {"width": 5, "height": 5, "resolution_m": 1.0},
                    "start_cell": [0, 0],
                    "roi": {"kind": "cells", "cells": [[0, 0], [2, 2], [4, 4], [4, 0]]},
                    "blocked_rectangles": [],
                    "unsafe_rectangles": [],
                    "coverage_events": [],
                }
            ]
        )
        self._write_frontier_config(path_budget_m=40.0, coverage_radius_cells=1)
        self._write_memory_config(path_budget_m=40.0, coverage_radius_cells=1)
        self._write_policy_candidate(state_key="model_state_dict", prefer_high_x=True)
        self._write_policy_config(path_budget_m=40.0, coverage_radius_cells=1)

        summary = run_policy_guided_global_coverage(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reachable_safe_cell_count"], 4)
        self.assertEqual(summary["covered_reachable_safe_cell_count"], 4)
        self.assertEqual(summary["achieved_coverage_rate"], 1.0)

    def test_unsafe_and_unreachable_cells_are_excluded_and_reported(self) -> None:
        from scripts.run_policy_guided_global_coverage import run_policy_guided_global_coverage

        self._write_source_config(
            [
                {
                    "scenario_id": "policy-unsafe-unreachable",
                    "grid": {"width": 5, "height": 5, "resolution_m": 1.0},
                    "start_cell": [0, 2],
                    "roi": {"kind": "rect", "x0": 0, "y0": 0, "x1": 5, "y1": 5},
                    "blocked_rectangles": [{"x0": 2, "y0": 0, "x1": 3, "y1": 5}],
                    "unsafe_rectangles": [{"x0": 1, "y0": 0, "x1": 2, "y1": 1}],
                    "coverage_events": [],
                }
            ]
        )
        self._write_frontier_config(path_budget_m=20.0, coverage_radius_cells=1)
        self._write_memory_config(path_budget_m=20.0, coverage_radius_cells=1)
        self._write_policy_candidate(state_key="model_state_dict", prefer_high_x=True)
        self._write_policy_config(path_budget_m=20.0, coverage_radius_cells=1)

        summary = run_policy_guided_global_coverage(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertIn("unsafe_roi_cells", summary["infeasible_reason_codes"])
        self.assertIn("unreachable_roi_cells", summary["infeasible_reason_codes"])
        self.assertGreater(summary["policy_scored_candidate_count"], 0)
        self.assertEqual(summary["reachable_safe_cell_count"], 9)
        self.assertEqual(summary["covered_reachable_safe_cell_count"], 9)

    def test_budget_failure_reports_blockers(self) -> None:
        from scripts.run_policy_guided_global_coverage import run_policy_guided_global_coverage

        self._write_source_config(
            [
                {
                    "scenario_id": "policy-budget-fail",
                    "grid": {"width": 10, "height": 10, "resolution_m": 1.0},
                    "start_cell": [0, 0],
                    "roi": {"kind": "rect", "x0": 0, "y0": 0, "x1": 10, "y1": 10},
                    "blocked_rectangles": [],
                    "unsafe_rectangles": [],
                    "coverage_events": [],
                }
            ]
        )
        self._write_frontier_config(path_budget_m=1.0, coverage_radius_cells=1)
        self._write_memory_config(path_budget_m=1.0, coverage_radius_cells=1)
        self._write_policy_candidate(state_key="model_state_dict", prefer_high_x=True)
        self._write_policy_config(path_budget_m=1.0, coverage_radius_cells=1)

        summary = run_policy_guided_global_coverage(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("coverage_target_not_met", summary["reason_codes"])
        self.assertIn("insufficient_budget", summary["reason_codes"])
        self.assertTrue(summary["path_budget_exhausted"])
        self.assertEqual(summary["next_required_change"], "fix_policy_guided_global_coverage")

    def _default_1km_scenario(self) -> dict:
        return {
            "scenario_id": "global_1km_contract_fixture",
            "grid": {
                "width": 100,
                "height": 100,
                "resolution_m": 10.0,
                "origin": [0.0, 0.0],
            },
            "start_cell": [0, 0],
            "roi": {"kind": "rect", "x0": 0, "y0": 0, "x1": 100, "y1": 100},
            "blocked_rectangles": [],
            "unsafe_rectangles": [],
            "coverage_events": [],
        }

    def _write_source_config(self, scenarios: list[dict], *, target_coverage_rate: float = 0.99) -> None:
        self.source_config_path.write_text(
            json.dumps(
                {
                    "schema_version": "global-99-coverage-benchmark-config/v1",
                    "target_coverage_rate": target_coverage_rate,
                    "path_budget_m": 5000.0,
                    "scenarios": scenarios,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_frontier_config(self, *, path_budget_m: float, coverage_radius_cells: int) -> None:
        self.frontier_config_path.write_text(
            json.dumps(
                {
                    "schema_version": "frontier-coverage-planner-baseline-config/v1",
                    "source_global_99_config": str(self.source_config_path),
                    "target_coverage_rate": 0.99,
                    "path_budget_m": path_budget_m,
                    "coverage_radius_cells": coverage_radius_cells,
                    "frontier_step_limit": 256,
                    "revisit_penalty_weight": 1.0,
                    "new_coverage_weight": 4.0,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_memory_config(self, *, path_budget_m: float, coverage_radius_cells: int) -> None:
        self.memory_config_path.write_text(
            json.dumps(
                {
                    "schema_version": "coverage-memory-replanning-loop-config/v1",
                    "source_frontier_baseline_config": str(self.frontier_config_path),
                    "source_global_99_config": str(self.source_config_path),
                    "target_coverage_rate": 0.99,
                    "path_budget_m": path_budget_m,
                    "coverage_radius_cells": coverage_radius_cells,
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

    def _write_policy_config(
        self,
        *,
        path_budget_m: float,
        coverage_radius_cells: int,
        target_coverage_rate: float = 0.99,
        policy_logit_weight: float = 0.25,
    ) -> None:
        self.config_path.write_text(
            json.dumps(
                {
                    "schema_version": "policy-guided-global-coverage-config/v1",
                    "source_global_99_config": str(self.source_config_path),
                    "source_frontier_baseline_config": str(self.frontier_config_path),
                    "source_coverage_memory_config": str(self.memory_config_path),
                    "target_coverage_rate": target_coverage_rate,
                    "path_budget_m": path_budget_m,
                    "coverage_radius_cells": coverage_radius_cells,
                    "replanning_cycle_limit": 64,
                    "segment_step_limit": 64,
                    "memory_snapshot_interval": 1,
                    "max_policy_candidates": 16,
                    "policy_logit_weight": policy_logit_weight,
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

    def _write_policy_candidate(
        self,
        *,
        state_key: str,
        prefer_high_x: bool,
        summary_status: str = "passed",
    ) -> None:
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
            direction = 1.0 if prefer_high_x else -1.0
            network.candidate_encoder[0].weight[0, 0] = direction
            network.candidate_encoder[0].weight[1, 0] = -direction
            network.candidate_encoder[1].weight.fill_(1.0)
            network.candidate_encoder[4].weight[0, 0] = 1.0
            network.policy_head[0].weight[0, 0] = 1.0
            network.policy_head[-1].weight[0, 0] = 1.0
        checkpoint = {
            "schema_version": "controlled-hybrid-policy-candidate-checkpoint/v1",
            "experimental": True,
            "architecture": network.architecture_name,
            "training": {"hidden_size": 8, "seed": 7},
            state_key: network.state_dict(),
        }
        if state_key == "state_dict":
            checkpoint["hidden_size"] = 8
        torch.save(checkpoint, self.policy_root / "experimental-hybrid-policy-candidate.pt")
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
