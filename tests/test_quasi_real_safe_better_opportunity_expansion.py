from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


class QuasiRealSafeBetterOpportunityExpansionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[1]
        for path in (self.repo_root / "scripts", self.repo_root / "model-explorer" / "src"):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="qreal-safe-better-expansion-"))

    def test_expansion_matrix_generates_start_cell_variants_without_split_leakage(self) -> None:
        from scripts.run_quasi_real_safe_better_opportunity_expansion import (
            run_quasi_real_safe_better_opportunity_expansion,
        )

        source_matrix = self._selection_matrix()
        output_root = self.temp_dir / "expansion"
        summary = run_quasi_real_safe_better_opportunity_expansion(
            matrix_manifest_path=source_matrix,
            output_root=output_root,
            config=self._config(),
            repo_root=self.repo_root,
        )

        self.assertEqual(summary["status"], "passed")
        self.assertEqual(summary["reason_codes"], [])
        self.assertEqual(summary["roi_group_count"], 4)
        self.assertGreaterEqual(summary["candidate_context_count"], 48)
        self.assertEqual(summary["start_cell_missing_count"], 0)
        self.assertEqual(summary["context_id_missing_count"], 0)
        self.assertEqual(summary["context_id_overlap_count"], 0)
        self.assertEqual(summary["scenario_id_overlap_count"], 0)
        manifest_path = Path(summary["expansion_matrix_manifest"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema_version"], "model-explorer-quasi-real-evaluation/v1")
        self.assertEqual(manifest["candidate_count"], 6)
        self.assertGreaterEqual(len(manifest["rois"]), 48)
        self.assertTrue(all("start_cell" in roi for roi in manifest["rois"]))
        self.assertEqual({roi["name"] for roi in manifest["rois"]}, self._roi_groups())

    def test_readiness_accepts_passed_expansion_summary(self) -> None:
        from scripts.run_policy_training_readiness_review import (
            _quasi_real_safe_better_opportunity_expansion_readiness,
        )

        readiness = _quasi_real_safe_better_opportunity_expansion_readiness(
            {
                "schema_version": "quasi-real-safe-better-opportunity-expansion-summary/v1",
                "status": "passed",
                "reason_codes": [],
                "candidate_context_count": 48,
                "roi_group_count": 4,
                "start_cell_missing_count": 0,
                "context_id_missing_count": 0,
                "context_id_overlap_count": 0,
                "scenario_id_overlap_count": 0,
                "safe_alternative_context_count": 8,
                "safe_better_opportunity_context_count": 4,
                "roi_group_with_safe_better_opportunity_count": 2,
                "git_provenance": {"current_matches_sources": True},
            }
        )

        self.assertTrue(readiness["completed"])
        self.assertEqual(readiness["training_blockers"], [])

    def _selection_matrix(self) -> Path:
        matrix = self.temp_dir / "selection-matrix.json"
        matrix.write_text(
            json.dumps(
                {
                    "schema_version": "model-explorer-quasi-real-evaluation/v1",
                    "name": "fixture-qreal-selection",
                    "run_id": "fixture-selection",
                    "dataset_manifest": "lunar_south_pole_lro_lola_gdr_875s_20m.json",
                    "output_root": "../processed/fixture-qreal-selection",
                    "candidate_count": 8,
                    "episode_count": 1,
                    "seed": 100,
                    "rois": [
                        {
                            "name": name,
                            "split": split,
                            "roi_x": x,
                            "roi_y": y,
                            "roi_width": 32,
                            "roi_height": 32,
                            "seed": 100 + index,
                        }
                        for index, (name, split, x, y) in enumerate(
                            [
                                ("smooth_high_confidence", "train", 3700, 3700),
                                ("smooth_high_confidence", "validation", 3740, 3700),
                                ("smooth_high_confidence", "test", 3700, 3740),
                                ("rim_or_steep_slope", "train", 3900, 3900),
                                ("rim_or_steep_slope", "validation", 3940, 3900),
                                ("rim_or_steep_slope", "test", 3900, 3940),
                                ("low_observation_count", "train", 20, 20),
                                ("low_observation_count", "validation", 60, 20),
                                ("low_observation_count", "test", 20, 60),
                                ("mixed_risk", "train", 3600, 3900),
                                ("mixed_risk", "validation", 3640, 3900),
                                ("mixed_risk", "test", 3600, 3940),
                            ]
                        )
                    ],
                }
            ),
            encoding="utf-8",
        )
        return matrix

    def _config(self) -> dict[str, object]:
        return {
            "schema_version": "quasi-real-safe-better-opportunity-expansion-config/v1",
            "output_files": {
                "matrix_manifest": str(self.temp_dir / "lunar_south_pole_lro_lola_safe_better_opportunity_matrix_v1.json"),
                "summary": "quasi-real-safe-better-opportunity-expansion-summary.json",
                "report": "quasi-real-safe-better-opportunity-expansion-report.md",
            },
            "candidate_count": 6,
            "roi_offsets": [[0, 0], [8, 0], [0, 8], [8, 8]],
            "start_cells": [[0, 0], [31, 31], [16, 16], [8, 24]],
            "validation": {
                "min_candidate_context_count": 48,
                "min_roi_group_count": 4,
                "max_start_cell_missing_count": 0,
                "max_context_id_missing_count": 0,
                "max_context_id_overlap_count": 0,
                "max_scenario_id_overlap_count": 0,
            },
        }

    def _roi_groups(self) -> set[str]:
        return {"smooth_high_confidence", "rim_or_steep_slope", "low_observation_count", "mixed_risk"}


if __name__ == "__main__":
    unittest.main()
