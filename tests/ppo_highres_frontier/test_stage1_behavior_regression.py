import ast
import math
import tomllib
from collections import deque
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE1_CONFIG = REPO_ROOT / "configs" / "ppo_highres_frontier_smoke_v1.json"


def test_policy_callback_receives_only_observed_policy_observation(tmp_path: Path) -> None:
    from lunar_exploration_ppo.configs.stage1 import load_stage1_config
    from lunar_exploration_ppo.env.env import LunarExplorationEnv
    from lunar_exploration_ppo.policy.observation import PolicyObservation
    from lunar_exploration_ppo.workflows.stage1 import run_episode, select_rule_action

    env = LunarExplorationEnv(load_stage1_config(STAGE1_CONFIG))
    received: list[PolicyObservation] = []

    def observed_only_policy(observation: PolicyObservation):
        received.append(observation)
        assert not hasattr(observation, "truth")
        assert not hasattr(observation, "coverable_mask")
        return select_rule_action(observation)

    episode = run_episode(env, episode_index=0, policy=observed_only_policy)
    assert episode.transition_count == len(received) > 0


def test_hidden_truth_official_masks_and_policy_arrays_are_readonly_and_env_isolated() -> None:
    from lunar_exploration_ppo.configs.stage1 import load_stage1_config
    from lunar_exploration_ppo.env.env import LunarExplorationEnv

    config = load_stage1_config(STAGE1_CONFIG)
    first = LunarExplorationEnv(config)
    second = LunarExplorationEnv(config)
    observation = first.reset()

    assert not hasattr(first, "scenario")
    assert not hasattr(first, "coverage_masks")
    truth_layers = (
        first._scenario.truth.height,
        first._scenario.truth.hard_obstacle,
        first._scenario.truth.slope_deg,
        first._scenario.truth.traversability,
    )
    official_masks = (
        first._coverage_masks.safe_free_mask,
        first._coverage_masks.reachable_safe_mask,
        first._coverage_masks.coverable_mask,
    )
    assert all(not array.flags.writeable for array in (*truth_layers, *official_masks))
    assert all(not array.flags.writeable for array in observation.array_fields())
    assert not first.current_action_set.frontier_features.flags.writeable
    assert not first.current_action_set.candidate_mask.flags.writeable
    assert not any(
        __import__("numpy").shares_memory(left, right)
        for left, right in zip(
            official_masks,
            (
                second._coverage_masks.safe_free_mask,
                second._coverage_masks.reachable_safe_mask,
                second._coverage_masks.coverable_mask,
            ),
            strict=True,
        )
    )
    with pytest.raises(ValueError):
        first._coverage_masks.coverable_mask[0, 0] = True
    with pytest.raises(ValueError):
        observation.candidate_mask[0] = False


def _open_truth(width: int = 128, height: int = 128):
    from lunar_exploration_ppo.env.scenario import TruthMap
    from lunar_exploration_ppo.utils.geometry import GridGeometry

    geometry = GridGeometry(width, height, 0.5)
    return TruthMap(
        geometry=geometry,
        height=np.zeros(geometry.shape, dtype=float),
        hard_obstacle=np.zeros(geometry.shape, dtype=bool),
        slope_deg=np.zeros(geometry.shape, dtype=float),
        traversability=np.ones(geometry.shape, dtype=float),
        provenance={
            "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
            "physical_obstacle_cells_written": False,
        },
    )


@pytest.mark.parametrize("heading_deg", [0.0, 8.0, 38.0, 45.0])
def test_sensor_never_reveals_cell_center_beyond_exact_20m(heading_deg: float) -> None:
    from lunar_exploration_ppo.env.map_state import ObservedMapState
    from lunar_exploration_ppo.env.sensor_model import SensorPose, SensorUpdater
    from lunar_exploration_ppo.utils.geometry import CellXY

    truth = _open_truth()
    state = ObservedMapState.empty(truth.geometry)
    origin = truth.geometry.cell_to_world_center(CellXY(48, 48))
    delta = SensorUpdater(range_m=20.0, fov_deg=0.0).reveal(
        truth,
        state,
        (SensorPose(origin, math.radians(heading_deg), "endpoint_theta"),),
    )

    distances = [
        math.hypot(
            truth.geometry.cell_to_world_center(cell).x - origin.x,
            truth.geometry.cell_to_world_center(cell).y - origin.y,
        )
        for cell in delta.visible_cells
    ]
    assert max(distances) <= 20.0
    if heading_deg == 0.0:
        assert CellXY(88, 48) in delta.visible_cells
        assert CellXY(89, 48) not in delta.visible_cells


def test_sensor_range_uses_original_subcell_pose_and_stops_cleanly_at_map_edge() -> None:
    from lunar_exploration_ppo.env.map_state import ObservedMapState
    from lunar_exploration_ppo.env.sensor_model import SensorPose, SensorUpdater
    from lunar_exploration_ppo.utils.geometry import CellXY, WorldXY

    truth = _open_truth()
    updater = SensorUpdater(range_m=20.0, fov_deg=0.0)
    subcell_origin = WorldXY(20.01, 32.25)
    subcell = updater.reveal(
        truth,
        ObservedMapState.empty(truth.geometry),
        (SensorPose(subcell_origin, 0.0, "path_tangent"),),
    )
    assert CellXY(80, 64) not in subcell.visible_cells
    assert all(
        math.hypot(
            truth.geometry.cell_to_world_center(cell).x - subcell_origin.x,
            truth.geometry.cell_to_world_center(cell).y - subcell_origin.y,
        )
        <= 20.0
        for cell in subcell.visible_cells
    )

    edge_origin = truth.geometry.cell_to_world_center(CellXY(126, 126))
    edge = updater.reveal(
        truth,
        ObservedMapState.empty(truth.geometry),
        (SensorPose(edge_origin, math.pi / 4.0, "endpoint_theta"),),
    )
    assert edge.visible_cells == (CellXY(126, 126), CellXY(127, 127))


def _independent_coverable_oracle(
    truth,
    start,
    *,
    sensor_range_m: float,
    min_clearance_m: float,
    max_slope_deg: float = 30.0,
    traversability_threshold: float = 0.5,
):
    height, width = truth.geometry.shape
    resolution = truth.geometry.resolution_m
    free = (
        np.isfinite(truth.height)
        & np.isfinite(truth.slope_deg)
        & np.isfinite(truth.traversability)
        & ~truth.hard_obstacle
        & (truth.slope_deg <= max_slope_deg)
        & (truth.traversability >= traversability_threshold)
    )
    unsafe_y, unsafe_x = np.nonzero(~free)
    safe = np.zeros_like(free)
    for y in range(height):
        for x in range(width):
            if not free[y, x]:
                continue
            boundary_clearance = min(x + 0.5, width - x - 0.5, y + 0.5, height - y - 0.5) * resolution
            blocker_clearance = min(
                (math.hypot(x - bx, y - by) * resolution for by, bx in zip(unsafe_y, unsafe_x, strict=True)),
                default=math.inf,
            )
            safe[y, x] = min(boundary_clearance, blocker_clearance) >= min_clearance_m

    reachable = np.zeros_like(safe)
    if 0 <= start.x < width and 0 <= start.y < height and safe[start.y, start.x]:
        reachable[start.y, start.x] = True
        queue = deque([(start.x, start.y)])
        for_current = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1))
        while queue:
            x, y = queue.popleft()
            for dx, dy in for_current:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < width and 0 <= ny < height) or reachable[ny, nx] or not safe[ny, nx]:
                    continue
                if dx and dy and (not safe[y, nx] or not safe[ny, x]):
                    continue
                reachable[ny, nx] = True
                queue.append((nx, ny))

    def independent_los(source_x: int, source_y: int, target_x: int, target_y: int) -> bool:
        steps = max(abs(target_x - source_x), abs(target_y - source_y)) * 32 + 1
        seen: set[tuple[int, int]] = set()
        for index in range(1, steps + 1):
            fraction = index / steps
            world_x = (source_x + 0.5 + fraction * (target_x - source_x)) * resolution
            world_y = (source_y + 0.5 + fraction * (target_y - source_y)) * resolution
            cell_x = min(width - 1, max(0, int(world_x / resolution)))
            cell_y = min(height - 1, max(0, int(world_y / resolution)))
            if (cell_x, cell_y) in seen:
                continue
            seen.add((cell_x, cell_y))
            if (cell_x, cell_y) == (target_x, target_y):
                return True
            if truth.hard_obstacle[cell_y, cell_x] or truth.slope_deg[cell_y, cell_x] > max_slope_deg:
                return False
        return True

    coverable = np.zeros_like(free)
    reachable_cells = [(x, y) for y, x in np.argwhere(reachable)]
    for target_y, target_x in np.argwhere(free):
        for source_x, source_y in reachable_cells:
            distance = math.hypot(target_x - source_x, target_y - source_y) * resolution
            if distance <= sensor_range_m and independent_los(source_x, source_y, int(target_x), int(target_y)):
                coverable[target_y, target_x] = True
                break
    return safe, reachable, coverable


def _manual_truth(
    obstacle: np.ndarray,
    *,
    slope: np.ndarray | None = None,
    traversability: np.ndarray | None = None,
):
    from lunar_exploration_ppo.env.scenario import TruthMap
    from lunar_exploration_ppo.utils.geometry import GridGeometry

    geometry = GridGeometry(obstacle.shape[1], obstacle.shape[0], 1.0)
    return TruthMap(
        geometry=geometry,
        height=np.zeros(obstacle.shape, dtype=float),
        hard_obstacle=np.asarray(obstacle, dtype=bool),
        slope_deg=np.zeros(obstacle.shape, dtype=float) if slope is None else slope,
        traversability=np.ones(obstacle.shape, dtype=float) if traversability is None else traversability,
        provenance={
            "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
            "physical_obstacle_cells_written": False,
        },
    )


def test_exact_coverable_matches_independent_bruteforce_oracle_per_cell() -> None:
    from lunar_exploration_ppo.env.coverage import compute_coverage_masks
    from lunar_exploration_ppo.utils.geometry import CellXY

    corner_obstacles = np.zeros((5, 5), dtype=bool)
    corner_obstacles[1, 2] = True
    corner_obstacles[2, 1] = True
    wall_obstacles = np.zeros((7, 7), dtype=bool)
    wall_obstacles[:, 3] = True
    mixed_obstacles = np.zeros((7, 7), dtype=bool)
    mixed_slope = np.zeros((7, 7), dtype=float)
    mixed_slope[3, 4] = 31.0
    mixed_traversability = np.ones((7, 7), dtype=float)
    mixed_traversability[2, 4] = 0.49
    fixtures = (
        (_manual_truth(corner_obstacles), CellXY(1, 1), 3.0),
        (_manual_truth(wall_obstacles), CellXY(1, 3), 5.0),
        (
            _manual_truth(
                mixed_obstacles,
                slope=mixed_slope,
                traversability=mixed_traversability,
            ),
            CellXY(2, 3),
            3.0,
        ),
    )
    for truth, start, sensor_range in fixtures:
        expected_safe, expected_reachable, expected_coverable = _independent_coverable_oracle(
            truth,
            start,
            sensor_range_m=sensor_range,
            min_clearance_m=0.5215874761,
        )
        actual = compute_coverage_masks(
            truth,
            start,
            sensor_range_m=sensor_range,
            min_clearance_m=0.5215874761,
            max_slope_deg=30.0,
            traversability_threshold=0.5,
        )
        assert np.array_equal(actual.safe_free_mask, expected_safe)
        assert np.array_equal(actual.reachable_safe_mask, expected_reachable)
        assert np.array_equal(actual.coverable_mask, expected_coverable)

    open_truth = _manual_truth(np.zeros((7, 7), dtype=bool))
    open_masks = compute_coverage_masks(
        open_truth,
        CellXY(3, 3),
        sensor_range_m=2.0,
        min_clearance_m=0.5215874761,
        max_slope_deg=30.0,
        traversability_threshold=0.5,
    )
    assert open_masks.coverable_mask[3, 1]
    assert open_masks.coverable_mask[3, 5]


def test_exact_coverable_zero_denominator_fails_with_stable_reason() -> None:
    from lunar_exploration_ppo.env.coverage import compute_coverage_masks
    from lunar_exploration_ppo.utils.geometry import CellXY

    truth = _manual_truth(np.ones((5, 5), dtype=bool))
    with pytest.raises(ValueError, match="coverable denominator is zero"):
        compute_coverage_masks(
            truth,
            CellXY(2, 2),
            sensor_range_m=20.0,
            min_clearance_m=0.5215874761,
            max_slope_deg=30.0,
            traversability_threshold=0.5,
        )


def _independent_planning_opportunity_count(state, pose, sensor_range_m: float) -> tuple[int, int]:
    safe = state.planning_safe_mask
    height, width = safe.shape
    component = np.zeros_like(safe)
    if safe[pose.cell.y, pose.cell.x]:
        component[pose.cell.y, pose.cell.x] = True
        queue = deque([(pose.cell.x, pose.cell.y)])
        directions = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1))
        while queue:
            x, y = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < width and 0 <= ny < height) or component[ny, nx] or not safe[ny, nx]:
                    continue
                if dx and dy and (not safe[y, nx] or not safe[ny, x]):
                    continue
                component[ny, nx] = True
                queue.append((nx, ny))

    unknown = ~state.observed_mask
    resolution = state.geometry.resolution_m
    radius_cells = sensor_range_m / resolution

    def clear_observed_los(source_x: int, source_y: int, target_x: int, target_y: int) -> bool:
        steps = max(abs(target_x - source_x), abs(target_y - source_y)) * 16 + 1
        seen: set[tuple[int, int]] = set()
        for index in range(1, steps + 1):
            fraction = index / steps
            cell_x = min(width - 1, max(0, int(source_x + 0.5 + fraction * (target_x - source_x))))
            cell_y = min(height - 1, max(0, int(source_y + 0.5 + fraction * (target_y - source_y))))
            if (cell_x, cell_y) in seen:
                continue
            seen.add((cell_x, cell_y))
            if unknown[cell_y, cell_x]:
                return True
            if state.obstacle[cell_y, cell_x] or state.slope_deg[cell_y, cell_x] > 30.0:
                return False
        return False

    opportunities = 0
    for source_y, source_x in np.argwhere(component):
        if source_x == pose.cell.x and source_y == pose.cell.y:
            continue
        min_x = max(0, math.floor(source_x - radius_cells))
        max_x = min(width - 1, math.ceil(source_x + radius_cells))
        min_y = max(0, math.floor(source_y - radius_cells))
        max_y = min(height - 1, math.ceil(source_y + radius_cells))
        local_y, local_x = np.nonzero(unknown[min_y : max_y + 1, min_x : max_x + 1])
        has_opportunity = False
        for target_y, target_x in zip(local_y + min_y, local_x + min_x, strict=True):
            if math.hypot(target_x - source_x, target_y - source_y) > radius_cells:
                continue
            if clear_observed_los(int(source_x), int(source_y), int(target_x), int(target_y)):
                has_opportunity = True
                break
        opportunities += int(has_opportunity)
    return opportunities, int(np.count_nonzero(component))


def test_production_no_candidate_terminal_has_zero_independent_observed_opportunities() -> None:
    from lunar_exploration_ppo.configs.stage1 import load_stage1_config
    from lunar_exploration_ppo.env.env import LunarExplorationEnv
    from lunar_exploration_ppo.workflows.stage1 import select_rule_action

    env = LunarExplorationEnv(load_stage1_config(STAGE1_CONFIG))
    observation = env.reset()
    while not env.is_done:
        result = env.step(select_rule_action(observation))
        observation = result.observation
    if env.terminal_reason == "no_candidate_done":
        opportunity_count, component_size = _independent_planning_opportunity_count(
            env.observed_state,
            env.pose,
            env.config.sensor_range_m,
        )
        assert component_size > 0
        assert opportunity_count == 0
    else:
        assert env.terminal_reason in {
            "success_done",
            "failure_done",
            "stagnation_done",
        }


def test_real_frontier_generator_empty_with_observed_opportunity_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.configs.stage1 import load_stage1_config
    from lunar_exploration_ppo.env.env import LunarExplorationEnv
    from lunar_exploration_ppo.env.frontier import FrontierActionSet, FrontierGenerator

    config = load_stage1_config(STAGE1_CONFIG)
    generator = FrontierGenerator(top_m=config.frontier_top_m)
    env = LunarExplorationEnv(config, frontier_generator=generator)

    def forced_empty(observed_state, prior, pose):
        return FrontierActionSet(
            cells=(),
            frontier_features=np.zeros((config.frontier_top_m, 22), dtype=np.float32),
            candidate_mask=np.zeros((config.frontier_top_m,), dtype=bool),
        )

    monkeypatch.setattr(generator, "extract", forced_empty)
    with pytest.raises(RuntimeError, match="production frontier is empty.*opportunities remain"):
        env.reset()


def test_terminal_safety_transition_does_not_require_a_next_frontier_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.configs.stage1 import load_stage1_config
    from lunar_exploration_ppo.env.env import LunarExplorationEnv
    from lunar_exploration_ppo.env.frontier import FrontierActionSet, FrontierGenerator

    class InjectedSevereExecutionSafetyChecker:
        def check(self, safe_mask, path_cells):
            assert safe_mask.ndim == 2
            assert path_cells
            return "collision_detected"

    config = load_stage1_config(STAGE1_CONFIG)
    generator = FrontierGenerator(top_m=config.frontier_top_m)
    env = LunarExplorationEnv(
        config,
        frontier_generator=generator,
        execution_safety_checker=InjectedSevereExecutionSafetyChecker(),
    )
    observation = env.reset()
    action = env.select_rule_action(observation)

    def forced_empty(observed_state, prior, pose):
        return FrontierActionSet(
            cells=(),
            frontier_features=np.zeros((config.frontier_top_m, 22), dtype=np.float32),
            candidate_mask=np.zeros((config.frontier_top_m,), dtype=bool),
        )

    monkeypatch.setattr(generator, "extract", forced_empty)
    result = env.step(action)

    assert result.done and result.terminal
    assert result.reason == "safety_done"
    assert result.bootstrap_value == 0.0
    assert result.trainable is True
    assert result.diagnostics.frontier.candidate_count == 0
    assert result.diagnostics.frontier.oracle_opportunity_count > 0
    assert env.needs_policy is False


def test_real_frontier_generator_zero_opportunity_is_legal_no_candidate_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.configs.stage1 import load_stage1_config
    from lunar_exploration_ppo.env.env import LunarExplorationEnv
    from lunar_exploration_ppo.env.frontier import FrontierGenerator
    from lunar_exploration_ppo.workflows.stage1 import select_rule_action

    config = load_stage1_config(STAGE1_CONFIG)
    generator = FrontierGenerator(top_m=config.frontier_top_m)
    production_extract = generator.extract
    call_count = 0

    def exhaust_unknown_before_second_extract(observed_state, prior, pose):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            observed_state.observed_mask[...] = True
        return production_extract(observed_state, prior, pose)

    monkeypatch.setattr(generator, "extract", exhaust_unknown_before_second_extract)
    env = LunarExplorationEnv(config, frontier_generator=generator)
    observation = env.reset()
    assert env.current_action_set.candidate_count > 0

    result = env.step(select_rule_action(observation))

    assert result.done is True
    assert result.reason == "no_candidate_done"
    assert result.terminal is True
    assert result.bootstrap_value == 0.0
    assert result.diagnostics.frontier.candidate_count == 0
    assert result.diagnostics.frontier.unknown_count == 0
    assert result.diagnostics.frontier.oracle_opportunity_count == 0


def test_observed_frontier_opportunity_audit_distinguishes_remaining_gain_from_exhaustion() -> None:
    from lunar_exploration_ppo.configs.stage1 import load_stage1_config
    from lunar_exploration_ppo.env.env import LunarExplorationEnv
    from lunar_exploration_ppo.env.frontier_oracle import audit_frontier_opportunities

    env = LunarExplorationEnv(load_stage1_config(STAGE1_CONFIG))
    env.reset()
    initial = audit_frontier_opportunities(
        env.observed_state,
        env.pose,
        planning_safe_mask=env.observed_state.planning_safe_mask,
        sensor_range_m=env.config.sensor_range_m,
    )
    assert initial.component_size > 0
    assert initial.unknown_count > 0
    assert initial.opportunity_count > 0

    env.observed_state.observed_mask[...] = True
    exhausted = audit_frontier_opportunities(
        env.observed_state,
        env.pose,
        planning_safe_mask=env.observed_state.planning_safe_mask,
        sensor_range_m=env.config.sensor_range_m,
    )
    assert exhausted.unknown_count == 0
    assert exhausted.opportunity_count == 0


def test_no_candidate_diagnostics_record_independent_oracle_and_remaining_counts() -> None:
    from lunar_exploration_ppo.configs.stage1 import load_stage1_config
    from lunar_exploration_ppo.env.env import LunarExplorationEnv
    from lunar_exploration_ppo.env.frontier import FrontierActionSet

    class EmptyGenerator:
        def extract(self, observed_state, prior, pose):
            return FrontierActionSet(
                cells=(),
                frontier_features=np.zeros((512, 22), dtype=np.float32),
                candidate_mask=np.zeros((512,), dtype=bool),
            )

    env = LunarExplorationEnv(
        load_stage1_config(STAGE1_CONFIG),
        frontier_generator=EmptyGenerator(),
    )
    observation = env.reset()
    diagnostics = env.last_reset_diagnostics.frontier
    assert not np.any(observation.candidate_mask)
    assert diagnostics.candidate_count == 0
    assert diagnostics.component_size > 0
    assert diagnostics.unknown_count > 0
    assert diagnostics.remaining_unobserved_coverable_count > 0
    assert diagnostics.oracle_opportunity_count > 0


def test_packaging_declares_planner_and_ast_confines_direct_external_imports() -> None:
    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = set(metadata["project"]["dependencies"])
    assert "numpy>=1.26,<2.3" in dependencies
    assert "path-planner==0.1.0" in dependencies

    direct_importers: list[str] = []
    source_root = REPO_ROOT / "src" / "lunar_exploration_ppo"
    for source_file in source_root.rglob("*.py"):
        tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(alias.name == "path_planner" or alias.name.startswith("path_planner.") for alias in node.names):
                direct_importers.append(source_file.relative_to(source_root).as_posix())
            if isinstance(node, ast.ImportFrom) and node.module and (
                node.module == "path_planner" or node.module.startswith("path_planner.")
            ):
                direct_importers.append(source_file.relative_to(source_root).as_posix())
    assert sorted(set(direct_importers)) == [
        "integrations/path_planner_adapter.py",
        "workflows/stage6_planning_child_source_repair.py",
    ]
    integrations_init = (source_root / "integrations" / "__init__.py").read_text(encoding="utf-8")
    assert "import_module" not in integrations_init
    assert '".path_" + "planner_adapter"' not in integrations_init
