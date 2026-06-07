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
        self.assertIn("Platform-Aware Target/Anchor Contract v1", content)
        self.assertIn("platform_goal_admissibility", content)
        self.assertIn("policy_target_cell", content)
        self.assertIn("execution_goal_cell", content)
        self.assertIn("not_positive_evidence", content)
        self.assertIn("platform_inflated_goal_blocked", content)
        self.assertIn("platform_goal_trainable_anchor_projection_count", content)
        self.assertIn("platform_goal_nontrainable_blocked_target_count", content)
        self.assertIn("Channel-Aware Route Execution Alignment v1", content)
        self.assertIn("default_route_replacement_verified=false", content)
        self.assertIn("current_git_provenance_mismatch", content)
        self.assertIn("Anchor-Projection Evidence Contract v1", content)
        self.assertIn("不宣称 Ackermann-feasible trajectory", content)
        self.assertIn("`a_gcs_ws-2.0.1-direction-cone-curvature` 实验目录", content)
        self.assertNotIn("后处理 corridor 仍先基于 `baseline_result` 构建", content)
        self.assertNotIn("最新 evidence 的 git provenance 仍锚定在父仓库 commit `3ffe8bb`", content)
        self.assertNotIn("下一阶段应继续围绕 `GCS Direction-Cone Portal and Cost Calibration v1` 推进", content)


if __name__ == "__main__":
    unittest.main()
