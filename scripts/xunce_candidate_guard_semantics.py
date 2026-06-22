from __future__ import annotations

import math
from typing import Any


TOLERANCE = 1e-9
CANDIDATE_GUARD_MODE = "paired_baseline_delta_vs_incumbent_selected"


def paired_decision_key(row: dict[str, Any]) -> tuple[str, int, str, str]:
    step_index = _int_or_none(row.get("step_index"))
    return (
        str(row.get("scenario_id") or ""),
        step_index if step_index is not None else -1,
        str(row.get("candidate_set_hash") or ""),
        str(row.get("covered_cells_hash") or ""),
    )


def selected_baseline_candidate(
    paired_row: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    *,
    policy: str = "incumbent",
) -> dict[str, Any] | None:
    selected_index = _int_or_none(paired_row.get(f"{policy}_selected_action_index"))
    if selected_index is None:
        return None
    for row in candidate_rows:
        if _int_or_none(row.get("candidate_index")) == selected_index:
            return row
    return None


def candidate_guard_observation(
    candidate_row: dict[str, Any],
    paired_row: dict[str, Any],
    *,
    baseline_candidate_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    action_valid = candidate_row.get("action_mask_valid") is True
    if not action_valid:
        reason_codes.append("candidate_action_mask_invalid")

    candidate_coverage = _finite(candidate_row.get("expected_new_coverage_cell_count"))
    candidate_cost = _finite(candidate_row.get("path_cost"))
    candidate_risk = risk_proxy_value(candidate_row)
    candidate_risk_cost = _finite(candidate_row.get("risk_cost_weighted"))
    if candidate_risk_cost is None and candidate_cost is not None and candidate_risk is not None:
        candidate_risk_cost = candidate_cost * candidate_risk

    baseline_coverage = _baseline_value(
        paired_row,
        baseline_candidate_row,
        paired_field="incumbent_selected_expected_new_coverage_cell_count",
        candidate_field="expected_new_coverage_cell_count",
    )
    baseline_cost = _baseline_value(
        paired_row,
        baseline_candidate_row,
        paired_field="incumbent_selected_path_cost",
        candidate_field="path_cost",
    )
    baseline_risk = _baseline_value(
        paired_row,
        baseline_candidate_row,
        paired_field="incumbent_selected_risk",
        candidate_field="risk",
        candidate_alias="risk_proxy",
    )
    baseline_risk_cost = _finite(paired_row.get("incumbent_selected_risk_cost_weighted"))
    if baseline_risk_cost is None and baseline_candidate_row is not None:
        baseline_risk_cost = _finite(baseline_candidate_row.get("risk_cost_weighted"))
    if baseline_risk_cost is None and baseline_cost is not None and baseline_risk is not None:
        baseline_risk_cost = baseline_cost * baseline_risk

    _missing_metric_reasons(
        reason_codes,
        candidate_coverage=candidate_coverage,
        candidate_cost=candidate_cost,
        candidate_risk=candidate_risk,
        candidate_risk_cost=candidate_risk_cost,
        baseline_coverage=baseline_coverage,
        baseline_cost=baseline_cost,
        baseline_risk=baseline_risk,
        baseline_risk_cost=baseline_risk_cost,
    )

    candidate_per_100m = _per_100m(candidate_coverage, candidate_cost)
    baseline_per_100m = _per_100m(baseline_coverage, baseline_cost)
    if candidate_per_100m is None:
        reason_codes.append("undefined_candidate_coverage_per_100m")
    if baseline_per_100m is None:
        reason_codes.append("undefined_baseline_coverage_per_100m")

    observation = {
        "candidate_guard_mode": CANDIDATE_GUARD_MODE,
        "action_mask_valid": action_valid,
        "comparable": False,
        "reason_codes": sorted(set(reason_codes)),
        "candidate_index": _int_or_none(candidate_row.get("candidate_index")),
        "baseline_policy": "incumbent",
        "baseline_candidate_index": _int_or_none(paired_row.get("incumbent_selected_action_index")),
        "absolute_coverage_cells": candidate_coverage,
        "absolute_path_cost": candidate_cost,
        "absolute_risk_proxy": candidate_risk,
        "absolute_risk_source": str(
            candidate_row.get("risk_proxy_source") or candidate_row.get("risk_source") or "unknown"
        ),
        "absolute_risk_cost_weighted": candidate_risk_cost,
        "baseline_coverage_cells": baseline_coverage,
        "baseline_path_cost": baseline_cost,
        "baseline_risk_proxy": baseline_risk,
        "baseline_risk_cost_weighted": baseline_risk_cost,
        "coverage_delta_cells": None,
        "path_cost_delta_m": None,
        "risk_delta": None,
        "risk_cost_weighted_delta": None,
        "coverage_per_100m_delta": None,
    }
    if reason_codes:
        return observation

    observation.update(
        {
            "comparable": True,
            "coverage_delta_cells": candidate_coverage - baseline_coverage,
            "path_cost_delta_m": candidate_cost - baseline_cost,
            "risk_delta": candidate_risk - baseline_risk,
            "risk_cost_weighted_delta": candidate_risk_cost - baseline_risk_cost,
            "coverage_per_100m_delta": candidate_per_100m - baseline_per_100m,
        }
    )
    return observation


def candidate_guard_failure_reasons(observation: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    if observation.get("comparable") is not True:
        return list(observation.get("reason_codes") or ["candidate_metric_not_comparable"])
    reasons = []
    if _finite(observation.get("coverage_delta_cells")) < float(thresholds["min_coverage_delta_cells"]) - TOLERANCE:
        reasons.append("coverage_delta_below_minimum")
    if _finite(observation.get("path_cost_delta_m")) > float(thresholds["max_path_cost_delta_m"]) + TOLERANCE:
        reasons.append("path_cost_budget_exceeded")
    if (
        thresholds.get("risk_delta_hard_gate_enabled", True)
        and thresholds.get("max_risk_delta") is not None
        and _finite(observation.get("risk_delta")) > float(thresholds["max_risk_delta"]) + TOLERANCE
    ):
        reasons.append("risk_budget_exceeded")
    soft_risk_budget = thresholds.get("max_soft_risk_exposure_delta", thresholds.get("max_risk_cost_weighted_delta"))
    if (
        soft_risk_budget is not None
        and _finite(observation.get("risk_cost_weighted_delta")) > float(soft_risk_budget) + TOLERANCE
    ):
        reasons.append(
            "soft_risk_exposure_budget_exceeded"
            if thresholds.get("candidate_level_risk_delta_guard_is_diagnostic_only")
            else "risk_cost_weighted_budget_exceeded"
        )
    if _finite(observation.get("coverage_per_100m_delta")) < float(thresholds["min_coverage_per_100m_delta"]) - TOLERANCE:
        reasons.append("coverage_per_100m_regression")
    return reasons


def candidate_guard_clean(observation: dict[str, Any], thresholds: dict[str, Any]) -> bool:
    return candidate_guard_failure_reasons(observation, thresholds) == []


def risk_proxy_value(row: dict[str, Any]) -> float | None:
    value = _finite(row.get("risk_proxy"))
    if value is not None:
        return value
    return _finite(row.get("risk"))


def _baseline_value(
    paired_row: dict[str, Any],
    baseline_candidate_row: dict[str, Any] | None,
    *,
    paired_field: str,
    candidate_field: str,
    candidate_alias: str | None = None,
) -> float | None:
    value = _finite(paired_row.get(paired_field))
    if value is not None or baseline_candidate_row is None:
        return value
    if candidate_alias is not None:
        value = _finite(baseline_candidate_row.get(candidate_alias))
        if value is not None:
            return value
    return _finite(baseline_candidate_row.get(candidate_field))


def _missing_metric_reasons(reason_codes: list[str], **metrics: float | None) -> None:
    for name, value in metrics.items():
        if value is None:
            reason_codes.append(f"missing_candidate_guard_metric:{name}")


def _per_100m(coverage: float | None, path_cost: float | None) -> float | None:
    if coverage is None or path_cost is None or abs(path_cost) <= TOLERANCE:
        return None
    return coverage * 100.0 / path_cost


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
