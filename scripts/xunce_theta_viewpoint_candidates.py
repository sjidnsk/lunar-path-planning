from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from typing import Any

try:  # pragma: no cover - exercised by script execution
    from xunce_theta_sensor_coverage import theta_bins, theta_coverage_hash, visible_cells_for_viewpoint
except ModuleNotFoundError:  # pragma: no cover
    from scripts.xunce_theta_sensor_coverage import theta_bins, theta_coverage_hash, visible_cells_for_viewpoint


THETA_VIEWPOINT_CONTRACT_VERSION = "xunce-theta-aware-candidate-viewpoint/v1"
THETA_SENSOR_MODEL_ID = "theta-fov-90-range-radius/v1"


def theta_candidate_config(config: dict[str, Any]) -> dict[str, Any]:
    radius = config.get("sensor_range_cells")
    if radius is None:
        radius = config.get("coverage_radius_cells", 1)
    theta_count = _positive_int(config.get("theta_bin_count", 8), "theta_bin_count")
    theta_step = _positive_int(config.get("theta_step_deg", 45), "theta_step_deg")
    return {
        "theta_bin_count": theta_count,
        "theta_step_deg": theta_step,
        "theta_values": theta_bins(theta_bin_count=theta_count, theta_step_deg=theta_step),
        "sensor_model_id": str(config.get("sensor_model_id") or THETA_SENSOR_MODEL_ID),
        "sensor_range_cells": _nonnegative_int(radius, "sensor_range_cells"),
        "sensor_fov_deg": _positive_float(config.get("sensor_fov_deg", 90.0), "sensor_fov_deg"),
        "coverage_denominator_cells": _positive_float(
            config.get("coverage_denominator_cells", 1.0),
            "coverage_denominator_cells",
        ),
    }


def theta_viewpoints_enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("theta_aware_candidate_viewpoints_enabled", False))


def expand_theta_aware_candidates(
    candidates: list[dict[str, Any]],
    *,
    current_cell: Sequence[int] | None,
    covered_cells: set[tuple[int, int]],
    config: dict[str, Any],
    base_candidate_set_hash: str | None = None,
) -> dict[str, Any]:
    theta_config = theta_candidate_config(config)
    expanded: list[dict[str, Any]] = []
    raw_viewpoint_count = 0
    zero_new_visible_drop_count = 0
    for base_index, candidate in enumerate(candidates):
        cell = _cell_tuple(candidate.get("cell") or candidate.get("candidate_cell"))
        if cell is None:
            continue
        base_action_index = _int_or_none(candidate.get("action_index"))
        for theta in theta_config["theta_values"]:
            raw_viewpoint_count += 1
            visible = visible_cells_for_viewpoint(
                cell,
                theta_deg=float(theta),
                sensor_range_cells=int(theta_config["sensor_range_cells"]),
                sensor_fov_deg=float(theta_config["sensor_fov_deg"]),
            )
            new_count = len(visible - covered_cells)
            if new_count <= 0:
                zero_new_visible_drop_count += 1
                continue
            row = dict(candidate)
            row.setdefault("base_expected_new_coverage_cell_count", candidate.get("expected_new_coverage_cell_count"))
            row.setdefault("base_expected_coverage_rate_delta", candidate.get("expected_coverage_rate_delta"))
            row["schema_version"] = THETA_VIEWPOINT_CONTRACT_VERSION
            row["cell"] = [int(cell[0]), int(cell[1])]
            row["candidate_cell"] = [int(cell[0]), int(cell[1])]
            row["candidate_theta_deg"] = int(theta)
            row["candidate_viewpoint"] = [int(cell[0]), int(cell[1]), int(theta)]
            row["base_candidate_index"] = int(base_action_index if base_action_index is not None else base_index)
            row["viewpoint_index"] = len(expanded)
            row["base_candidate_set_hash"] = base_candidate_set_hash
            row["sensor_model_id"] = theta_config["sensor_model_id"]
            row["sensor_fov_deg"] = float(theta_config["sensor_fov_deg"])
            row["sensor_range_cells"] = int(theta_config["sensor_range_cells"])
            row["theta_visible_cell_count"] = int(len(visible))
            row["theta_new_visible_cell_count"] = int(new_count)
            row["theta_coverage_hash"] = theta_coverage_hash(visible)
            row["theta_feature_available"] = True
            row["sin_theta"] = math.sin(math.radians(float(theta)))
            row["cos_theta"] = math.cos(math.radians(float(theta)))
            row["theta_deg_norm"] = float(theta) / 360.0
            row["expected_new_coverage_cell_count"] = float(new_count)
            row["expected_new_coverage_area"] = float(new_count)
            row["expected_coverage_rate_delta"] = float(new_count) / float(theta_config["coverage_denominator_cells"])
            row["information_gain"] = row["expected_coverage_rate_delta"]
            row["value"] = float(new_count) * _float_default(row.get("roi_weight"), 1.0)
            row["roi_weighted_coverage_delta"] = row["expected_coverage_rate_delta"] * _float_default(row.get("roi_weight"), 1.0)
            row["theta_coverage_gain_per_path_cost"] = _safe_ratio(float(new_count), row.get("path_cost"))
            row["coverage_source"] = "theta_aware_sensor_footprint/v1"
            row["coverage_cell_set_kind"] = "theta_sensor_endpoint_footprint"
            row["action_index"] = len(expanded)
            expanded.append(row)
    return {
        "candidates": expanded,
        "base_candidate_count": len(candidates),
        "viewpoint_candidate_count": len(expanded),
        "raw_viewpoint_candidate_count": raw_viewpoint_count,
        "zero_new_visible_viewpoint_drop_count": zero_new_visible_drop_count,
        "theta_value_count": len(theta_config["theta_values"]),
        "theta_values": theta_config["theta_values"],
        "candidate_set_hash": theta_viewpoint_candidate_set_hash(expanded),
        "sensor_model_id": theta_config["sensor_model_id"],
        "sensor_fov_deg": theta_config["sensor_fov_deg"],
        "sensor_range_cells": theta_config["sensor_range_cells"],
        "current_cell": list(_cell_tuple(current_cell) or (0, 0)),
    }


def theta_viewpoint_candidate_set_hash(candidates: Iterable[dict[str, Any]]) -> str:
    payload = []
    for candidate in candidates:
        payload.append(
            {
                "candidate_viewpoint": candidate.get("candidate_viewpoint"),
                "base_candidate_index": candidate.get("base_candidate_index"),
                "path_cost": _finite_float(candidate.get("path_cost")),
                "risk": _finite_float(candidate.get("risk")),
                "reachable": candidate.get("reachable"),
                "theta_new_visible_cell_count": _finite_float(candidate.get("theta_new_visible_cell_count")),
                "theta_coverage_hash": candidate.get("theta_coverage_hash"),
                "sensor_model_id": candidate.get("sensor_model_id"),
                "sensor_range_cells": candidate.get("sensor_range_cells"),
                "sensor_fov_deg": candidate.get("sensor_fov_deg"),
            }
        )
    return _hash_payload(payload)


def candidate_observation_cells(candidates: Iterable[dict[str, Any]]) -> list[list[int] | None]:
    cells: list[list[int] | None] = []
    for candidate in candidates:
        viewpoint = candidate.get("candidate_viewpoint")
        if isinstance(viewpoint, Sequence) and not isinstance(viewpoint, (str, bytes)) and len(viewpoint) >= 3:
            cells.append([int(viewpoint[0]), int(viewpoint[1]), int(viewpoint[2])])
            continue
        cell = _cell_tuple(candidate.get("cell") or candidate.get("candidate_cell"))
        cells.append([int(cell[0]), int(cell[1])] if cell is not None else None)
    return cells


def theta_metadata(candidates: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(candidates)
    has_theta = any(row.get("candidate_theta_deg") is not None or row.get("candidate_viewpoint") for row in rows)
    if not has_theta:
        return {"theta_feature_available": False}
    return {
        "candidate_theta_deg": [_int_or_none(row.get("candidate_theta_deg")) for row in rows],
        "candidate_viewpoints": [row.get("candidate_viewpoint") for row in rows],
        "base_candidate_indices": [_int_or_none(row.get("base_candidate_index")) for row in rows],
        "viewpoint_indices": [_int_or_none(row.get("viewpoint_index")) for row in rows],
        "theta_visible_cell_counts": [_int_or_none(row.get("theta_visible_cell_count")) for row in rows],
        "theta_new_visible_cell_counts": [_int_or_none(row.get("theta_new_visible_cell_count")) for row in rows],
        "theta_coverage_hashes": [row.get("theta_coverage_hash") for row in rows],
        "theta_coverage_gain_per_path_costs": [_finite_float(row.get("theta_coverage_gain_per_path_cost")) for row in rows],
        "theta_feature_available": all(row.get("theta_feature_available") is True for row in rows) if rows else False,
        "sensor_model_id": rows[0].get("sensor_model_id") if rows else None,
        "sensor_fov_deg": rows[0].get("sensor_fov_deg") if rows else None,
        "sensor_range_cells": rows[0].get("sensor_range_cells") if rows else None,
        "coverage_source": rows[0].get("coverage_source") if rows else None,
    }


def row_has_theta_viewpoint_contract(row: dict[str, Any]) -> bool:
    info = row.get("info") if isinstance(row.get("info"), dict) else {}
    observation = row.get("observation") if isinstance(row.get("observation"), dict) else {}
    for source in (row, info, observation):
        if _present_theta_value(source.get("candidate_viewpoint")):
            return True
        if _present_theta_value(source.get("candidate_viewpoints")):
            return True
        if _present_theta_value(source.get("candidate_theta_deg")):
            return True
    for item in (info.get("candidate_cells") or observation.get("candidate_cells") or row.get("candidate_cells") or []):
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) >= 3:
            return True
    return False


def _present_theta_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_present_theta_value(item) for item in value)
    return True


def _cell_tuple(value: Any) -> tuple[int, int] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        return None
    try:
        return (int(value[0]), int(value[1]))
    except (TypeError, ValueError):
        return None


def _safe_ratio(numerator: float, denominator: Any) -> float | None:
    denom = _finite_float(denominator)
    if denom is None or abs(denom) <= 1.0e-12:
        return None
    return float(numerator) / denom


def _float_default(value: Any, default: float) -> float:
    number = _finite_float(value)
    return default if number is None else number


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _positive_int(value: Any, name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _nonnegative_int(value: Any, name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _positive_float(value: Any, name: str) -> float:
    number = _finite_float(value)
    if number is None or number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
