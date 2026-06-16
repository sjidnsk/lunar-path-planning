import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class Global99CoverageBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_dir = str(self.repo_root / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="global-99-benchmark-"))
        self.config_path = self.temp_dir / "global-99-config.json"
        self.output_root = self.temp_dir / "output"

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_passes_1km_fixture_with_9900_of_10000_reachable_safe_cells(self) -> None:
        from scripts.run_global_99_coverage_benchmark import (
            run_global_99_coverage_benchmark,
        )

        self._write_config(
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
                            "event_id": "global-pass-001",
                            "path_cost_m": 1000.0,
                            "covered_rectangles": [
                                {"x0": 0, "y0": 0, "x1": 100, "y1": 99}
                            ],
                        }
                    ],
                }
            ]
        )

        summary = run_global_99_coverage_benchmark(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["target_coverage_rate"], 0.99)
        self.assertEqual(summary["reachable_safe_cell_count"], 10_000)
        self.assertEqual(summary["covered_reachable_safe_cell_count"], 9_900)
        self.assertEqual(summary["achieved_coverage_rate"], 0.99)
        self.assertTrue(summary["coverage_target_met"])
        self.assertTrue(summary["coverage_denominator_valid"])
        self.assertTrue(summary["coverage_ledger_complete"])
        self.assertFalse(summary["path_budget_exhausted"])
        self.assertEqual(summary["next_required_change"], "frontier_coverage_planner_baseline")
        self.assertFalse(summary["default_policy_replacement_approved"])
        self.assertFalse(summary["real_executor_connection_approved"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["modifies_network"])
        self.assertFalse(summary["modifies_action_space"])
        self.assertFalse(summary["modifies_default_astar"])
        self.assertFalse(summary["ackermann_feasible_trajectory_claimed"])

        for filename in (
            "global-99-coverage-benchmark-summary.json",
            "global-99-coverage-benchmark-manifest.json",
            "global-99-coverage-ledger.jsonl",
            "global-99-coverage-denominator-audit.json",
            "global-99-coverage-rejection-report.json",
            "global-99-coverage-benchmark-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

        ledger_rows = self._read_jsonl(self.output_root / "global-99-coverage-ledger.jsonl")
        self.assertEqual(len(ledger_rows), 1)
        self.assertEqual(ledger_rows[0]["scenario_id"], "global_1km_contract_fixture")
        self.assertEqual(ledger_rows[0]["event_id"], "global-pass-001")
        self.assertEqual(ledger_rows[0]["new_covered_reachable_safe_cell_count"], 9_900)

    def test_cells_roi_uses_only_explicit_roi_cells_as_denominator(self) -> None:
        from scripts.run_global_99_coverage_benchmark import (
            run_global_99_coverage_benchmark,
        )

        self._write_config(
            [
                {
                    "scenario_id": "arbitrary-cells-roi",
                    "grid": {"width": 5, "height": 5, "resolution_m": 1.0},
                    "start_cell": [0, 0],
                    "roi": {
                        "kind": "cells",
                        "cells": [[0, 0], [2, 2], [4, 4], [4, 0]],
                    },
                    "blocked_rectangles": [],
                    "unsafe_rectangles": [],
                    "coverage_events": [
                        {
                            "event_id": "cells-pass",
                            "path_cost_m": 4.0,
                            "covered_cells": [[0, 0], [2, 2], [4, 4], [4, 0]],
                        }
                    ],
                }
            ]
        )

        summary = run_global_99_coverage_benchmark(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reachable_safe_cell_count"], 4)
        self.assertEqual(summary["covered_reachable_safe_cell_count"], 4)
        self.assertEqual(summary["achieved_coverage_rate"], 1.0)

    def test_unsafe_and_unreachable_cells_are_excluded_and_reported_as_context(self) -> None:
        from scripts.run_global_99_coverage_benchmark import (
            run_global_99_coverage_benchmark,
        )

        self._write_config(
            [
                {
                    "scenario_id": "unsafe-unreachable-context",
                    "grid": {"width": 5, "height": 3, "resolution_m": 1.0},
                    "start_cell": [0, 1],
                    "roi": {"kind": "rect", "x0": 0, "y0": 0, "x1": 5, "y1": 3},
                    "blocked_rectangles": [{"x0": 2, "y0": 0, "x1": 3, "y1": 3}],
                    "unsafe_rectangles": [{"x0": 1, "y0": 0, "x1": 2, "y1": 1}],
                    "coverage_events": [
                        {
                            "event_id": "context-pass",
                            "path_cost_m": 5.0,
                            "covered_rectangles": [{"x0": 0, "y0": 0, "x1": 2, "y1": 3}],
                        }
                    ],
                }
            ]
        )

        summary = run_global_99_coverage_benchmark(
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

    def test_fails_when_budget_excludes_required_coverage_event(self) -> None:
        from scripts.run_global_99_coverage_benchmark import (
            run_global_99_coverage_benchmark,
        )

        self._write_config(
            [
                {
                    "scenario_id": "budget-fail",
                    "grid": {"width": 10, "height": 10, "resolution_m": 1.0},
                    "start_cell": [0, 0],
                    "roi": {"kind": "rect", "x0": 0, "y0": 0, "x1": 10, "y1": 10},
                    "blocked_rectangles": [],
                    "unsafe_rectangles": [],
                    "coverage_events": [
                        {
                            "event_id": "within-budget",
                            "path_cost_m": 60.0,
                            "covered_rectangles": [{"x0": 0, "y0": 0, "x1": 10, "y1": 5}],
                        },
                        {
                            "event_id": "over-budget",
                            "path_cost_m": 60.0,
                            "covered_rectangles": [{"x0": 0, "y0": 5, "x1": 10, "y1": 10}],
                        },
                    ],
                }
            ],
            path_budget_m=100.0,
        )

        summary = run_global_99_coverage_benchmark(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("insufficient_budget", summary["reason_codes"])
        self.assertIn("coverage_target_not_met", summary["reason_codes"])
        self.assertTrue(summary["path_budget_exhausted"])
        self.assertEqual(summary["covered_reachable_safe_cell_count"], 50)
        self.assertEqual(summary["achieved_coverage_rate"], 0.5)
        self.assertEqual(summary["next_required_change"], "fix_global_99_coverage_benchmark_contract")

        ledger_rows = self._read_jsonl(self.output_root / "global-99-coverage-ledger.jsonl")
        self.assertEqual([row["counted"] for row in ledger_rows], [True, False])

    def test_fails_when_roi_has_no_reachable_safe_denominator(self) -> None:
        from scripts.run_global_99_coverage_benchmark import (
            run_global_99_coverage_benchmark,
        )

        self._write_config(
            [
                {
                    "scenario_id": "invalid-denominator",
                    "grid": {"width": 3, "height": 3, "resolution_m": 1.0},
                    "start_cell": [0, 0],
                    "roi": {"kind": "rect", "x0": 1, "y0": 1, "x1": 2, "y1": 2},
                    "blocked_rectangles": [],
                    "unsafe_rectangles": [{"x0": 1, "y0": 1, "x1": 2, "y1": 2}],
                    "coverage_events": [],
                }
            ]
        )

        summary = run_global_99_coverage_benchmark(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("coverage_denominator_invalid", summary["reason_codes"])
        self.assertFalse(summary["coverage_denominator_valid"])
        self.assertEqual(summary["reachable_safe_cell_count"], 0)
        self.assertEqual(summary["achieved_coverage_rate"], 0.0)

    def _write_config(
        self,
        scenarios: list[dict],
        *,
        target_coverage_rate: float = 0.99,
        path_budget_m: float = 5000.0,
    ) -> None:
        self.config_path.write_text(
            json.dumps(
                {
                    "schema_version": "global-99-coverage-benchmark-config/v1",
                    "target_coverage_rate": target_coverage_rate,
                    "path_budget_m": path_budget_m,
                    "scenarios": scenarios,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
