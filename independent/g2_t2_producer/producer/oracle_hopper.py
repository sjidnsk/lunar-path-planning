from __future__ import annotations

import math
import re
from decimal import Decimal, InvalidOperation
from typing import Any


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_FIELDS = (
    "schema_version",
    "parameter_set_id",
    "body_envelope_radius_m",
    "launch_reference_height_m",
    "arc_clearance_margin_m",
    "landing_footprint_radius_m",
    "stop_condition",
    "energy_model",
    "evidence_class",
    "simulation_proxy",
    "physical_capability_claimed",
    "formal_evidence_eligible",
    "status",
)


def _positive_decimal(record: dict[str, Any], field: str) -> Decimal:
    try:
        value = Decimal(str(record[field]))
    except (InvalidOperation, KeyError) as error:
        raise ValueError(f"invalid {field}") from error
    if not value.is_finite() or value <= 0:
        raise ValueError(f"invalid {field}")
    return value


def validate_hopper_parameter_record(record: dict[str, Any]) -> None:
    for field in _REQUIRED_FIELDS:
        if field not in record:
            raise ValueError(f"missing {field}")
    for field in (
        "body_envelope_radius_m",
        "launch_reference_height_m",
        "arc_clearance_margin_m",
        "landing_footprint_radius_m",
    ):
        _positive_decimal(record, field)
    if record["evidence_class"] != "candidate_engineering_proxy":
        raise ValueError("evidence_class must be candidate_engineering_proxy")
    if record["simulation_proxy"] is not True:
        raise ValueError("simulation_proxy must be true")
    if record["physical_capability_claimed"] is not False:
        raise ValueError("physical_capability_claimed must be false")
    if record["formal_evidence_eligible"] is not False:
        raise ValueError("formal_evidence_eligible must be false")
    if record["status"] != "pending_external_evidence":
        raise ValueError("status must be pending_external_evidence")
    stop = record["stop_condition"]
    energy = record["energy_model"]
    if stop.get("model_id") != "touchdown-speed-upper-bound/v1":
        raise ValueError("unsupported stop_condition model_id")
    if energy.get("model_id") != "quadratic-normalized-speed/v1":
        raise ValueError("unsupported energy_model model_id")
    _positive_decimal(stop, "max_touchdown_speed_m_s")
    _positive_decimal(energy, "reference_speed_m_s")
    _positive_decimal(energy, "max_energy_decimal")
    for container, name in ((stop, "stop_condition"), (energy, "energy_model")):
        digest = str(container.get("evaluator_source_sha256", ""))
        if not _SHA256_RE.fullmatch(digest):
            raise ValueError(f"invalid {name} evaluator_source_sha256")


def gaussian_square_mass_ppm(*, halfwidth_mm: int, sigma_mm: int) -> int:
    if halfwidth_mm <= 0 or sigma_mm <= 0:
        raise ValueError("halfwidth_mm and sigma_mm must be positive")
    one_axis = math.erf(halfwidth_mm / (sigma_mm * math.sqrt(2.0)))
    return round(one_axis * one_axis * 1_000_000)


def ballistic_witness(case: dict[str, Any], parameter_record: dict[str, Any]) -> dict[str, int]:
    action = case["action"]
    speed_m_s = int(action["speed_mm_s"]) / 1000.0
    elevation_rad = math.radians(int(action["elevation_mdeg"]) / 1000.0)
    gravity_m_s2 = 1.62
    launch_height_m = float(Decimal(str(parameter_record["launch_reference_height_m"])))
    vertical = speed_m_s * math.sin(elevation_rad)
    horizontal = speed_m_s * math.cos(elevation_rad)
    flight_time_s = (
        vertical + math.sqrt(vertical * vertical + 2.0 * gravity_m_s2 * launch_height_m)
    ) / gravity_m_s2
    range_m = horizontal * flight_time_s
    apex_m = launch_height_m + vertical * vertical / (2.0 * gravity_m_s2)
    return {
        "apex_height_mm": round(apex_m * 1000.0),
        "flight_time_us": round(flight_time_s * 1_000_000.0),
        "range_mm": round(range_m * 1000.0),
    }


def _decision(
    safe: bool,
    reason: str,
    *,
    witness: dict[str, Any],
    slacks: dict[str, int],
) -> dict[str, Any]:
    return {
        "numeric_witness": witness,
        "oracle_reason_code": reason,
        "oracle_safe": safe,
        "safety_slacks": slacks,
    }


def evaluate_hopper(
    case: dict[str, Any],
    parameter_record: dict[str, Any],
) -> dict[str, Any]:
    try:
        validate_hopper_parameter_record(parameter_record)
    except ValueError as error:
        return _decision(
            False,
            "G2I_H_PARAMETER_RECORD_UNAPPROVED",
            witness={"parameter_error": str(error)},
            slacks={},
        )
    if case.get("numeric_state") != "decided":
        return _decision(
            False,
            "G2I_NUMERIC_UNDECIDED",
            witness={"numeric_state": str(case.get("numeric_state"))},
            slacks={},
        )
    action = case["action"]
    terrain = case["terrain"]
    action_indices = (
        int(action["speed_index"]),
        int(action["elevation_index"]),
        int(action["azimuth_index"]),
    )
    if not (
        0 <= action_indices[0] < 4
        and 0 <= action_indices[1] < 3
        and 0 <= action_indices[2] < 16
    ):
        return _decision(
            False,
            "G2I_H_MODEL_DOMAIN",
            witness={"action_indices": list(action_indices)},
            slacks={},
        )
    touchdown_speed = int(case["touchdown_speed_mm_s"])
    if touchdown_speed < 0:
        return _decision(
            False,
            "G2I_H_MODEL_DOMAIN",
            witness={"touchdown_speed_mm_s": touchdown_speed},
            slacks={},
        )
    stop_limit = round(
        float(Decimal(str(parameter_record["stop_condition"]["max_touchdown_speed_m_s"])))
        * 1000.0
    )
    reference_speed = Decimal(
        str(parameter_record["energy_model"]["reference_speed_m_s"])
    )
    energy = (Decimal(touchdown_speed) / Decimal(1000) / reference_speed) ** 2
    max_energy = Decimal(str(parameter_record["energy_model"]["max_energy_decimal"]))
    landing_mass = gaussian_square_mass_ppm(
        halfwidth_mm=int(terrain["landing_halfwidth_mm"]),
        sigma_mm=int(terrain["landing_sigma_mm"]),
    )
    slacks = {
        "arc_clearance_mm": int(terrain["arc_clearance_slack_mm"]),
        "landing_footprint_mm": int(terrain["landing_footprint_clearance_mm"]),
        "landing_height_mm": int(terrain.get("landing_height_tolerance_mm", 50))
        - abs(int(terrain["landing_height_error_mm"])),
        "landing_mass_ppm": landing_mass - 990000,
        "landing_slope_cdeg": 1500 - int(terrain["landing_slope_cdeg"]),
        "launch_clearance_mm": int(terrain["launch_clearance_mm"]),
        "stop_speed_mm_s": stop_limit - touchdown_speed,
    }
    witness: dict[str, Any] = {
        **ballistic_witness(case, parameter_record),
        "action_indices": list(action_indices),
        "energy_decimal": format(energy, "f"),
        "landing_mass_ppm": landing_mass,
        "touchdown_speed_mm_s": touchdown_speed,
    }
    if energy < 0:
        return _decision(
            False, "G2I_H_MODEL_DOMAIN", witness=witness, slacks=slacks
        )
    if slacks["launch_clearance_mm"] <= 0:
        return _decision(
            False, "G2I_H_LAUNCH_FOOTPRINT", witness=witness, slacks=slacks
        )
    if not bool(terrain["arc_inside_map"]):
        return _decision(False, "G2I_H_ARC_OUTSIDE", witness=witness, slacks=slacks)
    if not bool(terrain["arc_known"]):
        return _decision(False, "G2I_H_ARC_UNKNOWN", witness=witness, slacks=slacks)
    if slacks["arc_clearance_mm"] <= 0:
        return _decision(
            False, "G2I_H_ARC_CLEARANCE", witness=witness, slacks=slacks
        )
    if slacks["landing_mass_ppm"] < 0:
        return _decision(
            False, "G2I_H_LANDING_MASS", witness=witness, slacks=slacks
        )
    if slacks["landing_footprint_mm"] <= 0:
        return _decision(
            False, "G2I_H_LANDING_FOOTPRINT", witness=witness, slacks=slacks
        )
    if slacks["landing_slope_cdeg"] < 0:
        return _decision(
            False, "G2I_H_LANDING_SLOPE", witness=witness, slacks=slacks
        )
    if slacks["landing_height_mm"] < 0:
        return _decision(
            False, "G2I_H_LANDING_HEIGHT", witness=witness, slacks=slacks
        )
    if slacks["stop_speed_mm_s"] < 0:
        return _decision(False, "G2I_H_STOP", witness=witness, slacks=slacks)
    if energy > max_energy:
        return _decision(
            False, "G2I_H_MODEL_DOMAIN", witness=witness, slacks=slacks
        )
    return _decision(True, "G2I_H_SAFE", witness=witness, slacks=slacks)
