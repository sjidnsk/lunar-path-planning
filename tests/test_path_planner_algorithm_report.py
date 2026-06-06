import unittest
from pathlib import Path


class PathPlannerAlgorithmReportTests(unittest.TestCase):
    def test_report_tracks_gcs_control_point_direction_cone_stage(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        content = (repo_root / "docs" / "算法设计与系统架构报告.md").read_text(encoding="utf-8")

        self.assertIn("GCS Control-Point Direction-Cone Prototype v1", content)
        self.assertIn("pydrake_control_point_direction_cone_program", content)
        self.assertIn("control_point_derivative_proxy", content)
        self.assertIn("derivative_constraint_count", content)
        self.assertIn("control_point_region_containment_count", content)
        self.assertIn("--gcs-control-point-candidate", content)
        self.assertIn("GCS Motion-Feasibility Batch Gate v1", content)
        self.assertIn("gcs_motion_feasibility_cli_batch/v1", content)
        self.assertIn("GCS Direction-Cone CLI Scenario Batch Evidence v1", content)
        self.assertIn("Direction-Cone CLI Scenario Batch Evidence", content)
        self.assertIn("route JSON", content)
        self.assertIn("heading violation", content)
        self.assertIn("turning-radius", content)
        self.assertIn("不宣称 Ackermann-feasible trajectory", content)
        self.assertIn("`a_gcs_ws-2.0.1-direction-cone-curvature` 实验目录", content)
        self.assertNotIn("下一阶段应继续围绕 `GCS Direction-Cone Portal and Cost Calibration v1` 推进", content)


if __name__ == "__main__":
    unittest.main()
