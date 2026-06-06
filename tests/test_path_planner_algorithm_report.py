import unittest
from pathlib import Path


class PathPlannerAlgorithmReportTests(unittest.TestCase):
    def test_report_tracks_gcs_cli_batch_evidence_stage(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        content = (repo_root / "docs" / "算法设计与系统架构报告.md").read_text(encoding="utf-8")

        self.assertIn("GCS Direction-Cone CLI Scenario Batch Evidence v1", content)
        self.assertIn("Portal/Cost Calibration 与 Scenario Matrix", content)
        self.assertIn("route JSON", content)
        self.assertIn("不宣称 Ackermann-feasible trajectory", content)
        self.assertIn("`a_gcs_ws-2.0.1-direction-cone-curvature` 实验目录", content)
        self.assertNotIn("下一阶段应继续围绕 `GCS Direction-Cone Portal and Cost Calibration v1` 推进", content)


if __name__ == "__main__":
    unittest.main()
