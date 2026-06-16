import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class FrontierCoveragePlannerBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_dir = str(self.repo_root / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="frontier-coverage-baseline-"))
        self.source_config_path = self.temp_dir / "global-99-source.json"
        self.config_path = self.temp_dir / "frontier-baseline-config.json"
        self.output_root = self.temp_dir / "output"

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_default_1km_fixture_generates_frontier_plan_that_reaches_target(self) -> None:
        from scripts.run_frontier_coverage_planner_baseline import (
            run_frontier_coverage_planner_baseline,
        )

        self._write_source_config(
            [
                {
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
                    "coverage_events": [
                        {
                            "event_id": "ignored-source-event",
                            "path_cost_m": 1.0,
                            "covered_rectangles": [{"x0": 0, "y0": 0, "x1": 1, "y1": 1}],
                        }
                    ],
                }
            ]
        )
        self._write_baseline_config(
            coverage_radius_cells=30,
            frontier_step_limit=256,
            path_budget_m=5000.0,
        )

        summary = run_frontier_coverage_planner_baseline(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertGreaterEqual(summary["achieved_coverage_rate"], 0.99)
        self.assertTrue(summary["coverage_target_met"])
        self.assertEqual(summary["reachable_safe_cell_count"], 10_000)
        self.assertGreaterEqual(summary["covered_reachable_safe_cell_count"], 9_900)
        self.assertGreater(summary["generated_coverage_event_count"], 0)
        self.assertLessEqual(summary["planned_path_cost_m"], summary["path_budget_m"])
        self.assertFalse(summary["path_budget_exhausted"])
        self.assertTrue(summary["frontier_plan_complete"])
        self.assertEqual(summary["next_required_change"], "coverage_memory_replanning_loop")
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["modifies_network"])
        self.assertFalse(summary["modifies_action_space"])
        self.assertFalse(summary["modifies_default_astar"])

        for filename in (
            "frontier-coverage-planner-baseline-summary.json",
            "frontier-coverage-planner-baseline-manifest.json",
            "frontier-coverage-plan.jsonl",
            "frontier-coverage-ledger.jsonl",
            "frontier-coverage-budget-audit.json",
            "frontier-coverage-rejection-report.json",
            "frontier-coverage-planner-baseline-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

        plan_rows = self._read_jsonl(self.output_root / "frontier-coverage-plan.jsonl")
        ledger_rows = self._read_jsonl(self.output_root / "frontier-coverage-ledger.jsonl")
        self.assertEqual(len(plan_rows), summary["generated_coverage_event_count"])
        self.assertEqual(len(ledger_rows), summary["generated_coverage_event_count"])
        self.assertTrue(all(row["choice_source"] == "frontier_baseline" for row in plan_rows))
        self.assertTrue(all(row["counted"] for row in ledger_rows))

    def test_cells_roi_counts_only_explicit_cells(self) -> None:
        from scripts.run_frontier_coverage_planner_baseline import (
            run_frontier_coverage_planner_baseline,
        )

        self._write_source_config(
            [
                {
                    "scenario_id": "cells-roi-frontier",
                    "grid": {"width": 5, "height": 5, "resolution_m": 1.0},
                    "start_cell": [0, 0],
                    "roi": {"kind": "cells", "cells": [[0, 0], [2, 2], [4, 4], [4, 0]]},
                    "blocked_rectangles": [],
                    "unsafe_rectangles": [],
                    "coverage_events": [],
                }
            ]
        )
        self._write_baseline_config(coverage_radius_cells=1, frontier_step_limit=32, path_budget_m=40.0)

        summary = run_frontier_coverage_planner_baseline(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reachable_safe_cell_count"], 4)
        self.assertEqual(summary["covered_reachable_safe_cell_count"], 4)
        self.assertEqual(summary["achieved_coverage_rate"], 1.0)

    def test_unsafe_and_unreachable_cells_are_excluded_and_reported(self) -> None:
        from scripts.run_frontier_coverage_planner_baseline import (
            run_frontier_coverage_planner_baseline,
        )

        self._write_source_config(
            [
                {
                    "scenario_id": "unsafe-unreachable-frontier",
                    "grid": {"width": 5, "height": 3, "resolution_m": 1.0},
                    "start_cell": [0, 1],
                    "roi": {"kind": "rect", "x0": 0, "y0": 0, "x1": 5, "y1": 3},
                    "blocked_rectangles": [{"x0": 2, "y0": 0, "x1": 3, "y1": 3}],
                    "unsafe_rectangles": [{"x0": 1, "y0": 0, "x1": 2, "y1": 1}],
                    "coverage_events": [],
                }
            ]
        )
        self._write_baseline_config(coverage_radius_cells=2, frontier_step_limit=16, path_budget_m=20.0)

        summary = run_frontier_coverage_planner_baseline(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertIn("unsafe_roi_cells", summary["infeasible_reason_codes"])
        self.assertIn("unreachable_roi_cells", summary["infeasible_reason_codes"])
        self.assertEqual(summary["reachable_safe_cell_count"], 5)
        self.assertEqual(summary["covered_reachable_safe_cell_count"], 5)

    def test_budget_failure_stops_and_reports_blockers(self) -> None:
        from scripts.run_frontier_coverage_planner_baseline import (
            run_frontier_coverage_planner_baseline,
        )

        self._write_source_config(
            [
                {
                    "scenario_id": "frontier-budget-fail",
                    "grid": {"width": 10, "height": 10, "resolution_m": 1.0},
                    "start_cell": [0, 0],
                    "roi": {"kind": "rect", "x0": 0, "y0": 0, "x1": 10, "y1": 10},
                    "blocked_rectangles": [],
                    "unsafe_rectangles": [],
                    "coverage_events": [],
                }
            ]
        )
        self._write_baseline_config(coverage_radius_cells=1, frontier_step_limit=64, path_budget_m=1.0)

        summary = run_frontier_coverage_planner_baseline(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("coverage_target_not_met", summary["reason_codes"])
        self.assertIn("insufficient_budget", summary["reason_codes"])
        self.assertTrue(summary["path_budget_exhausted"])
        self.assertFalse(summary["coverage_target_met"])
        self.assertEqual(summary["next_required_change"], "fix_frontier_coverage_planner_baseline")

    def test_revisit_penalty_increases_candidate_score_for_repeated_path_cells(self) -> None:
        from scripts.run_frontier_coverage_planner_baseline import score_frontier_candidate

        low_revisit = score_frontier_candidate(
            path_cost_m=12.0,
            revisited_path_cell_count=1,
            new_covered_cell_count=20,
            revisit_penalty_weight=3.0,
            new_coverage_weight=1.0,
        )
        high_revisit = score_frontier_candidate(
            path_cost_m=12.0,
            revisited_path_cell_count=5,
            new_covered_cell_count=20,
            revisit_penalty_weight=3.0,
            new_coverage_weight=1.0,
        )

        self.assertLess(low_revisit, high_revisit)

    def _write_source_config(self, scenarios: list[dict]) -> None:
        self.source_config_path.write_text(
            json.dumps(
                {
                    "schema_version": "global-99-coverage-benchmark-config/v1",
                    "target_coverage_rate": 0.99,
                    "path_budget_m": 5000.0,
                    "scenarios": scenarios,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _write_baseline_config(
        self,
        *,
        coverage_radius_cells: int,
        frontier_step_limit: int,
        path_budget_m: float,
        revisit_penalty_weight: float = 1.0,
        new_coverage_weight: float = 4.0,
    ) -> None:
        self.config_path.write_text(
            json.dumps(
                {
                    "schema_version": "frontier-coverage-planner-baseline-config/v1",
                    "source_global_99_config": str(self.source_config_path),
                    "target_coverage_rate": 0.99,
                    "path_budget_m": path_budget_m,
                    "coverage_radius_cells": coverage_radius_cells,
                    "frontier_step_limit": frontier_step_limit,
                    "revisit_penalty_weight": revisit_penalty_weight,
                    "new_coverage_weight": new_coverage_weight,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
