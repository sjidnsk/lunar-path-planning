from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VISUAL_WORKBENCH_ROOT = REPO_ROOT / "visual-workbench"

sys.path.insert(0, str(VISUAL_WORKBENCH_ROOT))


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_visual_workbench_structure_and_documented_boundaries() -> None:
    assert (VISUAL_WORKBENCH_ROOT / "pyproject.toml").is_file()
    assert (VISUAL_WORKBENCH_ROOT / "visual_workbench" / "api.py").is_file()
    assert (VISUAL_WORKBENCH_ROOT / "web" / "package.json").is_file()

    docs = "\n".join(
        [
            _read(REPO_ROOT / "README.md"),
            _read(REPO_ROOT / "docs" / "算法设计与系统架构报告.md"),
            _read(REPO_ROOT / "docs" / "superpowers" / "specs" / "2026-06-17-visual-workbench-design.md"),
            _read(VISUAL_WORKBENCH_ROOT / "README.md"),
        ]
    )

    for required in (
        "visual-workbench",
        "fourth-subproject",
        "React + FastAPI",
        "dry-run",
        "validate",
        "No PPO",
        "network/action space/default A*",
        "Ackermann-feasible",
        "IRIS/GCS/path-planner",
        "mission-first",
        "环境测绘",
        "目标捕获",
        "路线制导",
        "可达确认",
        "风险复核",
        "任务简报",
        "progressive evidence",
        "Evidence Trace",
        "Map Replay",
        "Validate",
        "dry-run 或 validate",
        "完整 run 不会",
    ):
        assert required in docs


def test_visual_workbench_schema_and_api_contract() -> None:
    from visual_workbench.api import create_app
    from visual_workbench.artifacts import SUPPORTED_SCHEMA_VERSIONS

    assert {
        "model-explorer-contract/v1",
        "path-planner-sidecar/v1",
        "path-planner-route/v1",
        "path-feedback-manifest/v1",
        "path-feedback-summary/v1",
        "model-explorer-experiment/v1",
    } <= SUPPORTED_SCHEMA_VERSIONS

    routes = {route.path for route in create_app().routes}
    assert {
        "/api/health",
        "/api/project/status",
        "/api/artifacts",
        "/api/artifacts/{artifact_id}",
        "/api/artifacts/{artifact_id}/raw",
        "/api/commands/dry-run",
        "/api/commands/validate",
    } <= routes
