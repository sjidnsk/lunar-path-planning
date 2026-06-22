import sys
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    scripts_path = str(repo_root / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    return repo_root


def test_failure_unreachable_and_open_grid_fallback_are_hard_risk() -> None:
    _repo_root()
    from scripts.xunce_path_risk_semantics import classify_path_risk

    cases = [
        {"reachable": False, "failure_reason": "unreachable"},
        {"success": False, "failure_reason": "goal_blocked"},
        {"reachable": True, "open_grid_fallback_used": True},
    ]

    for route in cases:
        semantics = classify_path_risk(route)

        assert semantics["path_allowed_by_risk"] is False
        assert semantics["hard_risk_flags"]
        assert semantics["risk_proxy_is_physical_risk"] is False


def test_peak_above_soft_threshold_below_hard_limit_is_soft_only() -> None:
    _repo_root()
    from scripts.xunce_path_risk_semantics import classify_path_risk

    semantics = classify_path_risk(
        {"reachable": True},
        sidecar_cost=np.array([[1.0, 9.0, 2.0]], dtype=float),
        path_cells=[(0, 0), (1, 0), (2, 0)],
        config={"soft_high_risk_threshold": 8.0, "hard_path_risk_peak_limit": 10.0},
    )

    assert semantics["path_allowed_by_risk"] is True
    assert semantics["hard_risk_flags"] == []
    assert "path_risk_peak_above_soft_threshold" in semantics["soft_risk_flags"]
    assert semantics["path_risk_peak"] == 9.0


def test_exposure_is_reproducible_from_path_cost_proxy_values() -> None:
    _repo_root()
    from scripts.xunce_path_risk_semantics import classify_path_risk

    route = {"reachable": True}
    sidecar_cost = np.array([[1.0, 5.0], [7.0, 11.0]], dtype=float)
    path_cells = [(0, 0), (1, 0), (0, 1), (1, 1)]

    first = classify_path_risk(route, sidecar_cost=sidecar_cost, path_cells=path_cells)
    second = classify_path_risk(route, sidecar_cost=sidecar_cost, path_cells=list(reversed(path_cells)))

    assert first["path_risk_exposure"] == 24.0
    assert second["path_risk_exposure"] == first["path_risk_exposure"]
    assert first["path_risk_peak"] == second["path_risk_peak"] == 11.0


def test_path_allowed_by_risk_false_rejects_formal_candidate() -> None:
    _repo_root()
    from scripts.xunce_path_planner_astar_batch import _row_from_route_payload

    row = _row_from_route_payload(
        {"proposal_id": "too-risky"},
        {
            "reachable": True,
            "path_cost": 4.0,
            "path_length": 3.0,
            "risk": 11.0,
            "path_allowed_by_risk": False,
            "hard_risk_flags": ["path_risk_peak_above_hard_limit"],
            "soft_risk_flags": ["path_risk_peak_above_soft_threshold"],
            "path_risk_peak": 11.0,
            "path_risk_exposure": 11.0,
            "high_risk_distance_m": 1.0,
            "recovery_margin_min": -1.0,
            "risk_semantics_source": "path_cost_proxy_risk_semantics/v1",
            "risk_proxy_is_physical_risk": False,
        },
        route_cache_hit=False,
    )

    assert row["proposal_only"] is True
    assert row["validation_evidence_kind"] == "validation_failure"
    assert row["replan_required"] is True
    assert "path_risk_peak_above_hard_limit" in row["validation_diagnostic_flags"]
