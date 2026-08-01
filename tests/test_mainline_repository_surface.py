"""Repository-surface contract for the retained parent mainline."""

import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RETAINED_SCRIPTS = {
    "benchmark_ppo_stage6_sensor_hotpath.py", "bootstrap_env.py", "bootstrap_ubuntu_conda.sh", "bootstrap_windows_conda.ps1",
    "create_ppo_stage6_planning_child_source_repair_continuation.py", "create_ppo_stage6_planning_child_source_repair.py", "create_ppo_stage6_planning_warm_start.py", "create_ppo_stage6_source_repair_amendment.py",
    "finalize_ppo_stage6_r3.py", "finalize_xunce_mid_dual_double_gate.py", "freeze_xunce_mid_dual_scenarios.py", "platform_command.py",
    "plot_xunce_mid_dual_separate_figures.py", "prepare_xunce_mid_dual_scenario_sources.py", "prewarm_ppo_stage6_coverage_cache.py", "render_xunce_mid_dual_figures.py", "render_xunce_mid_dual_g1_animation.py",
    "run_platform_smoke.py", "run_platform_validation_matrix.py", "run_ppo_foundation_preflight.py", "run_ppo_highres_frontier_stage2.py", "run_ppo_highres_frontier_stage3.py", "run_ppo_highres_frontier_stage4.py", "run_ppo_stage1_smoke_env.py", "run_ppo_stage5_baselines.py", "run_ppo_stage6_r3_evaluation_pipeline.py", "run_ppo_stage6_standard.py",
    "run_xunce_mid_dual_aggregate.py", "run_xunce_mid_dual_g1_coverage.py", "run_xunce_mid_dual_g1_independent_audit.py", "run_xunce_mid_dual_g1_repair.py", "run_xunce_mid_dual_g2_planning_time.py", "run_xunce_mid_dual_g3_closed_loop.py",
    "xunce_artifact_io.py", "xunce_artifact_paths.py", "xunce_mid_dual_artifacts.py", "xunce_mid_dual_contracts.py", "xunce_mid_dual_g1_existing_run_assessment.py", "xunce_mid_dual_g2_inputs.py",
}


def _active_tracked_paths() -> set[str]:
    tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=REPO_ROOT).decode().split("\0"))
    deleted = set(subprocess.check_output(["git", "ls-files", "--deleted", "-z"], cwd=REPO_ROOT).decode().split("\0"))
    return tracked - deleted - {""}


def test_config_and_script_surface_match_the_mainline_allowlist() -> None:
    active = _active_tracked_paths()
    configs = {path for path in active if path.startswith("configs/")}
    expected_configs = {
        path for path in configs if path.startswith("configs/ppo_highres_frontier_") or path.startswith("configs/xunce_mid_dual_")
    } | {
        "configs/mainline_routes_v1.json",
        "configs/platforms/v3/README.md",
        "configs/platforms/v3/wheeled_skid_steer_v3_example_v1.json",
        "configs/platforms/v3/legged_body_v3_example_v1.json",
        "configs/platforms/v3/hopper_ballistic_v3_example_v1.json",
    }
    assert configs == expected_configs
    assert {path.removeprefix("scripts/") for path in active if path.startswith("scripts/")} == RETAINED_SCRIPTS
    assert "configs/stage_registry.json" not in active
    assert "scripts/run_stage.py" not in active


def test_mainline_routes_are_machine_readable_and_resolve_to_retained_paths() -> None:
    active = _active_tracked_paths()
    routes = json.loads((REPO_ROOT / "configs/mainline_routes_v1.json").read_text(encoding="utf-8"))["routes"]
    assert {"stage6", "g1", "g2", "g3", "default_grid_astar", "multiplatform_v3", "platform_constraints"} == set(routes)
    assert routes["default_grid_astar"]["mode"] == "default"
    assert routes["multiplatform_v3"]["mode"] == "opt_in"
    for route in routes.values():
        for field in ("entrypoints", "configs", "tests", "docs"):
            for path in route.get(field, []):
                assert path in active
        submodule = route.get("submodule")
        if submodule:
            assert submodule in active


def test_default_astar_and_platform_slope_contracts_remain_explicit() -> None:
    adapter = (REPO_ROOT / "src/lunar_exploration_ppo/integrations/path_planner_adapter.py").read_text(encoding="utf-8")
    stage6 = (REPO_ROOT / "src/lunar_exploration_ppo/configs/stage6.py").read_text(encoding="utf-8")
    assert "AStarPlanner" in adapter
    assert "max_traversable_slope_deg" in stage6 and "30.0" in stage6


def test_retained_ppo_test_surface_excludes_retired_r1_review_route() -> None:
    active = _active_tracked_paths()
    retired = "tests/ppo_highres_frontier/test_stage1_smoke_env_r1.py"
    stage6_contract = "tests/ppo_highres_frontier/test_stage6_standard_config.py"

    assert retired not in active
    assert not (REPO_ROOT / retired).exists()
    assert stage6_contract in active
