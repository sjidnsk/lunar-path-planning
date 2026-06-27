import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_agents_md_is_short_current_boundary_document() -> None:
    text = _read("AGENTS.md")
    lines = text.splitlines()

    assert len(lines) <= 180
    assert text.count("## Stage") <= 5
    assert "docs/xunce-stage-documentation-index.md" in text

    for required in [
        "coverage_source=endpoint_theta_slope_obstacle_los/v1",
        "path_cost_source=hybrid_astar_pose_path/v1",
        "synthetic_source_kind=synthetic_terrain_obstacle_proxy/v1",
        "action_space_type=hybrid_discrete_xy_continuous_theta/v1",
        "max_traversable_slope_deg=30.0",
        "不发布 checkpoint",
        "不替换 default policy",
        "不连接 executor",
        "不启动 canary",
        "禁止继续把每个阶段完整说明追加到 AGENTS.md",
    ]:
        assert required in text


def test_documentation_index_defines_file_boundaries_and_current_route() -> None:
    text = _read("docs/xunce-stage-documentation-index.md")

    for required in [
        "AGENTS.md",
        "README.md",
        "docs/superpowers/plans",
        "outputs/.../report.md",
        "docs/算法设计与系统架构报告.md",
        "docs/superpowers/specs",
        "configs/stage_registry.json",
        "Stage26.0 -> Stage26.5",
        "repair_stage26_synthetic_exploration_credit_assignment",
    ]:
        assert required in text


def test_readme_has_documentation_map_and_current_status() -> None:
    text = _read("README.md")

    assert "## 中文说明" in text
    assert "## English" in text
    assert "月球无人平台自主探索与路径规划系统级仓库" in text
    assert "System-level repository for lunar rover autonomous exploration and path planning" in text
    assert "## Documentation Map" in text
    assert "## 文档地图" in text
    assert "docs/xunce-stage-documentation-index.md" in text
    assert "Stage26.5" in text
    assert "repair_stage26_synthetic_exploration_credit_assignment" in text
    assert text.count("Stage24.0") <= 2
    assert text.count("Stage26.0") <= 3


def test_stage_registry_keeps_stage26_5_and_registers_stage26_5b() -> None:
    registry = json.loads(_read("configs/stage_registry.json"))
    stages = registry["stages"]

    assert "xunce-stage26-5-synthetic-discrete-margin-crossing-calibration" in stages
    assert "xunce-stage26-5b-documentation-boundary-consolidation" in stages
    assert (
        stages["xunce-stage26-5b-documentation-boundary-consolidation"]["script"]
        == "scripts/run_xunce_stage26_5b_documentation_boundary_consolidation.py"
    )


def test_stage26_5b_runner_passes_for_consolidated_docs(tmp_path: Path) -> None:
    from scripts.run_xunce_stage26_5b_documentation_boundary_consolidation import (
        run_xunce_stage26_5b_documentation_boundary_consolidation,
    )

    summary = run_xunce_stage26_5b_documentation_boundary_consolidation(
        config_path=REPO_ROOT / "configs/xunce_stage26_5b_documentation_boundary_consolidation_v1.json",
        output_root=tmp_path,
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "continue_stage26_6_synthetic_exploration_credit_assignment"
    assert summary["agents_md_line_count"] <= 180
    assert summary["agents_stage_section_count"] <= 5
    assert summary["readme_has_documentation_map"] is True
