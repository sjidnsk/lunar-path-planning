from pathlib import Path
import inspect
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audit_route_retirement_preflight import build_manifest, classify_path, load_policy, validate_graph


def test_protected_rules_win_over_retirement_patterns() -> None:
    policy = load_policy(Path("configs/route_retirement_policy_v1.json"))
    assert classify_path("scripts/xunce_artifact_io.py", policy).classification == "protected"
    assert classify_path("scripts/run_xunce_mid_dual_g3_closed_loop.py", policy).classification == "protected"
    assert classify_path("path-planner/src/path_planner/search/astar.py", policy).classification == "protected"
    assert classify_path("dev-platform-constraints/src/dev_platform_constraints/core/contracts.py", policy).classification == "protected"


def test_retired_and_manual_review_boundaries_are_distinct() -> None:
    policy = load_policy(Path("configs/route_retirement_policy_v1.json"))
    assert classify_path("scripts/run_xunce_stage26_3_synthetic_terrain_post_update_trajectory_eval_smoke.py", policy).classification == "retire_candidate"
    assert classify_path("scripts/run_path_feedback_validation.py", policy).classification == "retire_candidate"
    assert classify_path("model-explorer", policy).classification == "manual_review"
    assert classify_path("visual-workbench", policy).classification == "manual_review"
    assert classify_path("configs/stage_registry.json", policy).classification == "manual_review"


def test_validates_the_frozen_knowledge_graph() -> None:
    summary = validate_graph(Path("D:/codex/project/lunar-path-planning/.ua/knowledge-graph.json"))
    assert (summary.analyzed_files, summary.nodes, summary.edges, summary.layers) == (252, 1480, 2710, 7)


def test_audit_module_has_no_delete_capability() -> None:
    import audit_route_retirement_preflight as module

    source = inspect.getsource(module).lower()
    for forbidden in ("unlink(", "rmdir(", "remove-item", "rm -rf", "git rm", "shutil.rmtree"):
        assert forbidden not in source
