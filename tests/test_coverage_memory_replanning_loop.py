import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


class CoverageMemoryReplanningLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        scripts_dir = str(self.repo_root / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="coverage-memory-loop-"))
        self.source_config_path = self.temp_dir / "global-99-source.json"
        self.frontier_config_path = self.temp_dir / "frontier-baseline-config.json"
        self.config_path = self.temp_dir / "coverage-memory-config.json"
        self.output_root = self.temp_dir / "output"

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_default_1km_fixture_replans_with_memory_until_target(self) -> None:
        from scripts.run_coverage_memory_replanning_loop import (
            run_coverage_memory_replanning_loop,
        )

        self._write_source_config([self._default_1km_scenario()])
        self._write_frontier_config(path_budget_m=5000.0, coverage_radius_cells=30)
        self._write_memory_config(path_budget_m=5000.0, coverage_radius_cells=30)

        summary = run_coverage_memory_replanning_loop(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertGreaterEqual(summary["achieved_coverage_rate"], 0.99)
        self.assertTrue(summary["coverage_target_met"])
        self.assertEqual(summary["next_required_change"], "policy_guided_global_coverage")
        self.assertTrue(summary["coverage_memory_complete"])
        self.assertGreater(summary["replanning_cycle_count"], 0)
        self.assertGreater(summary["memory_snapshot_count"], 0)
        self.assertTrue(summary["memory_resume_supported"])
        self.assertTrue(summary["memory_resume_verified"])
        self.assertFalse(summary["path_budget_exhausted"])
        self.assertFalse(summary["publishes_checkpoint"])
        self.assertFalse(summary["replaces_default_policy"])
        self.assertFalse(summary["connects_real_executor"])
        self.assertFalse(summary["starts_online_canary"])
        self.assertFalse(summary["runs_new_ppo_update"])
        self.assertFalse(summary["modifies_network"])
        self.assertFalse(summary["modifies_action_space"])
        self.assertFalse(summary["modifies_default_astar"])
        self.assertFalse(summary["uses_ppo_policy"])
        self.assertFalse(summary["uses_path_planner"])
        self.assertFalse(summary["uses_npz_or_sidecar"])

        for filename in (
            "coverage-memory-replanning-loop-summary.json",
            "coverage-memory-replanning-loop-manifest.json",
            "coverage-memory-replanning-trace.jsonl",
            "coverage-memory-snapshots.jsonl",
            "coverage-memory-ledger.jsonl",
            "coverage-memory-budget-audit.json",
            "coverage-memory-rejection-report.json",
            "coverage-memory-replanning-loop-report.md",
        ):
            self.assertTrue((self.output_root / filename).is_file(), filename)

        snapshots = self._read_jsonl(self.output_root / "coverage-memory-snapshots.jsonl")
        covered_counts = [row["covered_reachable_safe_cell_count"] for row in snapshots]
        self.assertEqual(covered_counts, sorted(covered_counts))

    def test_resume_from_snapshot_preserves_deterministic_result(self) -> None:
        from scripts.run_coverage_memory_replanning_loop import (
            run_coverage_memory_replanning_loop,
        )

        self._write_source_config([self._default_1km_scenario()])
        self._write_frontier_config(path_budget_m=5000.0, coverage_radius_cells=30)
        self._write_memory_config(path_budget_m=5000.0, coverage_radius_cells=30)
        uninterrupted = run_coverage_memory_replanning_loop(
            config_path=self.config_path,
            output_root=self.output_root / "uninterrupted",
            repo_root=self.repo_root,
        )
        snapshots = self._read_jsonl(
            self.output_root / "uninterrupted" / "coverage-memory-snapshots.jsonl"
        )
        resume_snapshot = self.output_root / "resume-snapshot.json"
        resume_snapshot.write_text(json.dumps(snapshots[1], ensure_ascii=False), encoding="utf-8")
        self._write_memory_config(
            path_budget_m=5000.0,
            coverage_radius_cells=30,
            resume_from_memory_snapshot=str(resume_snapshot),
        )

        resumed = run_coverage_memory_replanning_loop(
            config_path=self.config_path,
            output_root=self.output_root / "resumed",
            repo_root=self.repo_root,
        )

        self.assertEqual(resumed["status"], "passed")
        self.assertTrue(resumed["memory_resume_verified"])
        self.assertEqual(
            resumed["covered_reachable_safe_cell_count"],
            uninterrupted["covered_reachable_safe_cell_count"],
        )
        self.assertEqual(resumed["achieved_coverage_rate"], uninterrupted["achieved_coverage_rate"])

    def test_cells_roi_counts_only_explicit_cells(self) -> None:
        from scripts.run_coverage_memory_replanning_loop import (
            run_coverage_memory_replanning_loop,
        )

        self._write_source_config(
            [
                {
                    "scenario_id": "cells-roi-memory",
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

        summary = run_coverage_memory_replanning_loop(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reachable_safe_cell_count"], 4)
        self.assertEqual(summary["covered_reachable_safe_cell_count"], 4)
        self.assertEqual(summary["achieved_coverage_rate"], 1.0)

    def test_unsafe_and_unreachable_cells_are_excluded_and_reported(self) -> None:
        from scripts.run_coverage_memory_replanning_loop import (
            run_coverage_memory_replanning_loop,
        )

        self._write_source_config(
            [
                {
                    "scenario_id": "unsafe-unreachable-memory",
                    "grid": {"width": 5, "height": 3, "resolution_m": 1.0},
                    "start_cell": [0, 1],
                    "roi": {"kind": "rect", "x0": 0, "y0": 0, "x1": 5, "y1": 3},
                    "blocked_rectangles": [{"x0": 2, "y0": 0, "x1": 3, "y1": 3}],
                    "unsafe_rectangles": [{"x0": 1, "y0": 0, "x1": 2, "y1": 1}],
                    "coverage_events": [],
                }
            ]
        )
        self._write_frontier_config(path_budget_m=20.0, coverage_radius_cells=2)
        self._write_memory_config(path_budget_m=20.0, coverage_radius_cells=2)

        summary = run_coverage_memory_replanning_loop(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertIn("unsafe_roi_cells", summary["infeasible_reason_codes"])
        self.assertIn("unreachable_roi_cells", summary["infeasible_reason_codes"])
        self.assertEqual(summary["reachable_safe_cell_count"], 5)
        self.assertEqual(summary["covered_reachable_safe_cell_count"], 5)

    def test_budget_failure_stops_and_reports_blockers(self) -> None:
        from scripts.run_coverage_memory_replanning_loop import (
            run_coverage_memory_replanning_loop,
        )

        self._write_source_config(
            [
                {
                    "scenario_id": "memory-budget-fail",
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

        summary = run_coverage_memory_replanning_loop(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("coverage_target_not_met", summary["reason_codes"])
        self.assertIn("insufficient_budget", summary["reason_codes"])
        self.assertTrue(summary["path_budget_exhausted"])
        self.assertFalse(summary["coverage_target_met"])
        self.assertEqual(summary["next_required_change"], "fix_coverage_memory_replanning_loop")

    def test_cycle_limit_failure_reports_replanning_exhaustion(self) -> None:
        from scripts.run_coverage_memory_replanning_loop import (
            run_coverage_memory_replanning_loop,
        )

        self._write_source_config([self._default_1km_scenario()])
        self._write_frontier_config(path_budget_m=5000.0, coverage_radius_cells=10)
        self._write_memory_config(
            path_budget_m=5000.0,
            coverage_radius_cells=10,
            replanning_cycle_limit=1,
        )

        summary = run_coverage_memory_replanning_loop(
            config_path=self.config_path,
            output_root=self.output_root,
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "failed")
        self.assertIn("replanning_cycle_limit_exhausted", summary["reason_codes"])
        self.assertIn("coverage_target_not_met", summary["reason_codes"])

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

    def _write_frontier_config(
        self,
        *,
        path_budget_m: float,
        coverage_radius_cells: int,
        frontier_step_limit: int = 256,
        revisit_penalty_weight: float = 1.0,
        new_coverage_weight: float = 4.0,
    ) -> None:
        self.frontier_config_path.write_text(
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

    def _write_memory_config(
        self,
        *,
        path_budget_m: float,
        coverage_radius_cells: int,
        replanning_cycle_limit: int = 64,
        segment_step_limit: int = 64,
        resume_from_memory_snapshot: str | None = None,
    ) -> None:
        self.config_path.write_text(
            json.dumps(
                {
                    "schema_version": "coverage-memory-replanning-loop-config/v1",
                    "source_frontier_baseline_config": str(self.frontier_config_path),
                    "source_global_99_config": str(self.source_config_path),
                    "target_coverage_rate": 0.99,
                    "path_budget_m": path_budget_m,
                    "coverage_radius_cells": coverage_radius_cells,
                    "replanning_cycle_limit": replanning_cycle_limit,
                    "segment_step_limit": segment_step_limit,
                    "memory_snapshot_interval": 1,
                    "resume_from_memory_snapshot": resume_from_memory_snapshot,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
