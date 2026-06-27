from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import torch


SYNTHETIC_CREDIT_FEATURE_NAMES = (
    "relative_distance_norm",
    "hybrid_reachable",
    "obstacle_aware_new_visible_norm",
    "obstacle_aware_gain_per_hybrid_cost_norm",
    "hybrid_astar_path_cost_norm",
    "synthetic_los_blocker_pressure_norm",
    "synthetic_hard_obstacle_pressure_norm",
    "risk_or_clearance_proxy_norm",
)

BEHAVIOR_POLICY_ID = "synthetic_credit_mixture_policy/v1"
SYNTHETIC_CREDIT_SCORE_V1 = "coverage_proxy_v1"
SYNTHETIC_CREDIT_SCORE_V2 = "path_efficiency_v2"


def build_synthetic_credit_feature_rows(
    *,
    candidate_count: int,
    relative_distances: Sequence[Any] | None = None,
    hybrid_reachable_flags: Sequence[Any] | None = None,
    obstacle_aware_new_visible_cell_counts: Sequence[Any] | None = None,
    obstacle_aware_gain_per_hybrid_costs: Sequence[Any] | None = None,
    hybrid_astar_path_costs: Sequence[Any] | None = None,
    synthetic_los_blocker_candidate_counts: Sequence[Any] | None = None,
    synthetic_hard_obstacle_candidate_counts: Sequence[Any] | None = None,
    risk_values: Sequence[Any] | None = None,
) -> list[list[float]]:
    count = max(0, int(candidate_count))
    distances = _norm_by_max(_values(relative_distances, count))
    new_visible = _norm_by_max(_values(obstacle_aware_new_visible_cell_counts, count))
    gains = _norm_by_max(_values(obstacle_aware_gain_per_hybrid_costs, count))
    costs = _norm_by_max(_values(hybrid_astar_path_costs, count))
    los = _norm_by_max(_values(synthetic_los_blocker_candidate_counts, count))
    hard = _norm_by_max(_values(synthetic_hard_obstacle_candidate_counts, count))
    risk = _norm_by_max(_values(risk_values, count))
    reachable = [1.0 if bool(flag) else 0.0 for flag in _items(hybrid_reachable_flags, count, default=False)]
    return [
        [
            distances[index],
            reachable[index],
            new_visible[index],
            gains[index],
            costs[index],
            los[index],
            hard[index],
            risk[index],
        ]
        for index in range(count)
    ]


def select_synthetic_credit_target(
    feature_rows: Sequence[Sequence[float]],
    *,
    action_mask: Sequence[Any],
    sampling_mask: Sequence[Any],
    hard_risk_clean_mask: Sequence[Any],
    hybrid_reachable_flags: Sequence[Any] | None = None,
    score_version: str = SYNTHETIC_CREDIT_SCORE_V1,
    path_cost_weight: float = 0.05,
    risk_weight: float = 0.10,
    path_efficiency_max_cost_norm: float = 0.70,
) -> dict[str, Any]:
    count = len(feature_rows)
    if hybrid_reachable_flags is None or len(hybrid_reachable_flags) != count:
        return {
            "synthetic_credit_target_index": None,
            "synthetic_credit_target_score": None,
            "synthetic_credit_score_version": score_version,
            "path_efficiency_filter_relaxed": False,
            "selected_target_hybrid_cost_norm": None,
            "selected_target_gain_per_cost_norm": None,
            "synthetic_credit_target_reason": "missing_or_mismatched_hybrid_reachable_flags",
        }
    reachable = _items(hybrid_reachable_flags, count, default=False)
    allowed_indices = [
        index
        for index in range(count)
        if (
            _bool_at(action_mask, index)
            and _bool_at(sampling_mask, index)
            and _bool_at(hard_risk_clean_mask, index)
            and bool(reachable[index])
        )
    ]
    if not allowed_indices:
        return {
            "synthetic_credit_target_index": None,
            "synthetic_credit_target_score": None,
            "synthetic_credit_score_version": score_version,
            "path_efficiency_filter_relaxed": False,
            "selected_target_hybrid_cost_norm": None,
            "selected_target_gain_per_cost_norm": None,
            "synthetic_credit_target_reason": "no_mask_allowed_synthetic_credit_target",
        }
    normalized_version = str(score_version or SYNTHETIC_CREDIT_SCORE_V1)
    if normalized_version == SYNTHETIC_CREDIT_SCORE_V2:
        max_cost = float(path_efficiency_max_cost_norm)
        if not math.isfinite(max_cost) or max_cost < 0.0:
            raise ValueError("path_efficiency_max_cost_norm_must_be_finite_nonnegative")
        strict_indices = [index for index in allowed_indices if _feature(feature_rows[index], 4) <= max_cost]
        relaxed = not strict_indices
        candidate_indices = strict_indices if strict_indices else allowed_indices
        best_index, best_score = _best_target_index(
            feature_rows,
            candidate_indices,
            score_fn=_path_efficiency_score,
        )
        reason = "best_path_efficiency_score_relaxed_filter" if relaxed else "best_path_efficiency_score"
    else:
        normalized_version = SYNTHETIC_CREDIT_SCORE_V1
        relaxed = False
        best_index, best_score = _best_target_index(
            feature_rows,
            allowed_indices,
            score_fn=lambda row: _coverage_proxy_score(row, path_cost_weight=path_cost_weight, risk_weight=risk_weight),
        )
        reason = "best_synthetic_credit_score"
    if best_index is None:
        return {
            "synthetic_credit_target_index": None,
            "synthetic_credit_target_score": None,
            "synthetic_credit_score_version": normalized_version,
            "path_efficiency_filter_relaxed": relaxed,
            "selected_target_hybrid_cost_norm": None,
            "selected_target_gain_per_cost_norm": None,
            "synthetic_credit_target_reason": "no_mask_allowed_synthetic_credit_target",
        }
    return {
        "synthetic_credit_target_index": int(best_index),
        "synthetic_credit_target_score": float(best_score),
        "synthetic_credit_score_version": normalized_version,
        "path_efficiency_filter_relaxed": bool(relaxed),
        "selected_target_hybrid_cost_norm": _feature(feature_rows[best_index], 4),
        "selected_target_gain_per_cost_norm": _feature(feature_rows[best_index], 3),
        "synthetic_credit_target_reason": reason,
    }


def _best_target_index(
    feature_rows: Sequence[Sequence[float]],
    candidate_indices: Sequence[int],
    *,
    score_fn: Any,
) -> tuple[int | None, float | None]:
    best_index: int | None = None
    best_score = -math.inf
    for index in candidate_indices:
        score = float(score_fn(feature_rows[index]))
        if score > best_score or (score == best_score and (best_index is None or index < best_index)):
            best_index = int(index)
            best_score = score
    return best_index, (None if best_index is None else float(best_score))


def _coverage_proxy_score(row: Sequence[float], *, path_cost_weight: float, risk_weight: float) -> float:
    gain = _feature(row, 3)
    path_cost = _feature(row, 4)
    blocker_pressure = max(_feature(row, 5), _feature(row, 6), _feature(row, 7))
    return gain - float(path_cost_weight) * path_cost - float(risk_weight) * blocker_pressure


def _path_efficiency_score(row: Sequence[float]) -> float:
    return (
        1.00 * _feature(row, 3)
        + 0.30 * _feature(row, 2)
        - 0.75 * _feature(row, 4)
        - 0.35 * _feature(row, 5)
        - 0.35 * _feature(row, 6)
        - 0.25 * _feature(row, 7)
    )


def synthetic_credit_behavior_logprob(
    *,
    policy_probs: Sequence[Any],
    action_index: int,
    target_index: int | None,
    mixture_probability: float,
    theta_log_prob: float | None = None,
) -> dict[str, Any]:
    p = float(mixture_probability)
    if not (0.0 <= p <= 1.0) or not math.isfinite(p):
        raise ValueError("synthetic_credit_mixture_probability_must_be_finite_0_1")
    action = int(action_index)
    probs = [max(0.0, _finite(value, 0.0)) for value in policy_probs]
    if action < 0 or action >= len(probs):
        raise ValueError("synthetic_credit_action_index_out_of_range")
    policy_prob = max(probs[action], 1.0e-12)
    target = None if target_index is None else int(target_index)
    behavior_prob = (1.0 - p) * policy_prob + (p if target == action else 0.0)
    behavior_prob = max(behavior_prob, 1.0e-12)
    theta = 0.0 if theta_log_prob is None else float(theta_log_prob)
    old_policy_point = math.log(policy_prob)
    old_behavior_point = math.log(behavior_prob)
    return {
        "behavior_policy_id": BEHAVIOR_POLICY_ID,
        "synthetic_credit_mixture_probability": p,
        "synthetic_credit_target_index": target,
        "synthetic_credit_target_selected": target == action,
        "old_policy_point_log_prob": old_policy_point,
        "old_policy_log_prob": old_policy_point + theta,
        "old_behavior_point_log_prob": old_behavior_point,
        "old_behavior_log_prob": old_behavior_point + theta,
        "old_log_prob": old_behavior_point + theta,
    }


def apply_feature_rows_to_xunce_batch(
    xunce_batch: dict[str, torch.Tensor],
    feature_rows: Sequence[Sequence[float]],
) -> dict[str, torch.Tensor]:
    if "candidate_features" not in xunce_batch:
        raise ValueError("xunce_batch_missing_candidate_features")
    features = xunce_batch["candidate_features"].clone()
    if features.ndim != 3:
        raise ValueError("candidate_features_must_have_batch_candidate_feature_dims")
    width = int(features.shape[2])
    if width < len(SYNTHETIC_CREDIT_FEATURE_NAMES):
        raise ValueError("candidate_feature_width_too_small_for_synthetic_credit")
    count = min(int(features.shape[1]), len(feature_rows))
    for index in range(count):
        values = [float(value) for value in feature_rows[index][: len(SYNTHETIC_CREDIT_FEATURE_NAMES)]]
        features[0, index, : len(values)] = torch.tensor(values, dtype=features.dtype, device=features.device)
    updated = dict(xunce_batch)
    updated["candidate_features"] = features
    return updated


def feature_semantic_map() -> dict[str, Any]:
    return {
        "schema_version": "xunce-synthetic-credit-feature-semantic-map/v1",
        "feature_names": list(SYNTHETIC_CREDIT_FEATURE_NAMES),
        "network_candidate_feature_count": len(SYNTHETIC_CREDIT_FEATURE_NAMES),
        "network_structure_changed": False,
    }


def candidate_pressure_values(values: Sequence[Any] | None, count: int) -> list[float | None]:
    """Return explicit candidate-level synthetic pressure values.

    Deliberately do not expand a map-level total to every candidate: that creates
    a constant feature and falsely suggests candidate-level visibility.
    """
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and len(values) == count:
        return [_finite(value, None) for value in values]
    return [0.0 for _ in range(count)]


def _values(values: Sequence[Any] | None, count: int) -> list[float | None]:
    result: list[float | None] = []
    for item in _items(values, count, default=None):
        result.append(None if item is None else _finite(item, None))
    return result


def _items(values: Sequence[Any] | None, count: int, *, default: Any) -> list[Any]:
    source = list(values) if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) else []
    return [(source[index] if index < len(source) else default) for index in range(count)]


def _norm_by_max(values: Iterable[float | None]) -> list[float]:
    raw = [None if value is None else max(0.0, float(value)) for value in values]
    max_value = max((value for value in raw if value is not None), default=0.0)
    if max_value <= 0.0:
        return [0.0 for _ in raw]
    return [0.0 if value is None else max(0.0, min(1.0, value / max_value)) for value in raw]


def _finite(value: Any, default: float | None) -> float | None:
    if isinstance(value, bool) or value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _bool_at(values: Sequence[Any], index: int) -> bool:
    return index < len(values) and bool(values[index])


def _feature(row: Sequence[float], index: int) -> float:
    return float(row[index]) if index < len(row) else 0.0
