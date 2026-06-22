from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator


@dataclass(frozen=True)
class Stage18GuardThresholds:
    profile_id: str
    profile_version: str
    profile_hash: str
    min_coverage_delta_cells: float
    max_path_cost_delta_m: float
    min_coverage_per_100m_delta: float
    max_risk_delta: float | None
    max_risk_cost_weighted_delta: float | None
    max_soft_risk_exposure_delta: float
    max_hard_risk_violation_count: float
    risk_delta_hard_gate_enabled: bool
    candidate_level_risk_delta_guard_is_diagnostic_only: bool
    coverage_gain_per_path_cost_delta_mode: str = "audit_only"

    def __getitem__(self, key: str) -> Any:
        return self.to_artifact_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_artifact_dict())

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_artifact_dict().get(key, default)

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "profile_version": self.profile_version,
            "profile_hash": self.profile_hash,
            "min_coverage_delta_cells": self.min_coverage_delta_cells,
            "max_path_cost_delta_m": self.max_path_cost_delta_m,
            "max_acceptable_path_cost_delta_m": self.max_path_cost_delta_m,
            "min_coverage_per_100m_delta": self.min_coverage_per_100m_delta,
            "max_risk_delta": self.max_risk_delta,
            "max_acceptable_risk_delta": self.max_risk_delta,
            "max_risk_cost_weighted_delta": self.max_risk_cost_weighted_delta,
            "max_acceptable_risk_cost_weighted_delta": self.max_risk_cost_weighted_delta,
            "max_soft_risk_exposure_delta": self.max_soft_risk_exposure_delta,
            "max_hard_risk_violation_count": self.max_hard_risk_violation_count,
            "risk_delta_hard_gate_enabled": self.risk_delta_hard_gate_enabled,
            "candidate_level_risk_delta_guard_is_diagnostic_only": self.candidate_level_risk_delta_guard_is_diagnostic_only,
            "coverage_gain_per_path_cost_delta_mode": self.coverage_gain_per_path_cost_delta_mode,
        }


def stage18_guard_thresholds(profile: Any) -> Stage18GuardThresholds:
    version = str(getattr(profile, "profile_version", ""))
    guards = getattr(profile, "guards", {})
    if version == "v3":
        soft_risk_budget = _required_float(guards, "max_soft_risk_exposure_delta")
        return Stage18GuardThresholds(
            profile_id=str(profile.profile_id),
            profile_version=version,
            profile_hash=str(profile.profile_hash),
            min_coverage_delta_cells=_required_float(guards, "min_coverage_delta_cells"),
            max_path_cost_delta_m=_required_float(guards, "max_acceptable_path_cost_delta_m"),
            min_coverage_per_100m_delta=_required_float(guards, "min_coverage_per_100m_delta"),
            max_risk_delta=None,
            max_risk_cost_weighted_delta=soft_risk_budget,
            max_soft_risk_exposure_delta=soft_risk_budget,
            max_hard_risk_violation_count=_required_float(guards, "max_hard_risk_violation_count"),
            risk_delta_hard_gate_enabled=False,
            candidate_level_risk_delta_guard_is_diagnostic_only=True,
        )
    if version == "v2":
        risk_cost_budget = _required_float(guards, "max_acceptable_risk_cost_weighted_delta")
        return Stage18GuardThresholds(
            profile_id=str(profile.profile_id),
            profile_version=version,
            profile_hash=str(profile.profile_hash),
            min_coverage_delta_cells=_required_float(guards, "min_coverage_delta_cells"),
            max_path_cost_delta_m=_required_float(guards, "max_acceptable_path_cost_delta_m"),
            min_coverage_per_100m_delta=_required_float(guards, "min_coverage_per_100m_delta"),
            max_risk_delta=_required_float(guards, "max_acceptable_risk_delta"),
            max_risk_cost_weighted_delta=risk_cost_budget,
            max_soft_risk_exposure_delta=risk_cost_budget,
            max_hard_risk_violation_count=0.0,
            risk_delta_hard_gate_enabled=True,
            candidate_level_risk_delta_guard_is_diagnostic_only=False,
            coverage_gain_per_path_cost_delta_mode=str(
                guards.get("coverage_gain_per_path_cost_delta_mode", "audit_only")
            ),
        )
    raise ValueError(f"unsupported canonical reward profile version for Stage18 thresholds: {version!r}")


def _required_float(payload: dict[str, Any], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{key} must be finite")
    parsed = float(value)
    if parsed != parsed or parsed in {float("inf"), float("-inf")}:
        raise ValueError(f"{key} must be finite")
    return parsed
