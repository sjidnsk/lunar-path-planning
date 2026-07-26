from __future__ import annotations

import math
from typing import Any


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


def evaluate_legged(case: dict[str, Any]) -> dict[str, Any]:
    if case.get("numeric_state") != "decided":
        return _decision(
            False,
            "G2I_NUMERIC_UNDECIDED",
            witness={"numeric_state": str(case.get("numeric_state"))},
            slacks={},
        )
    target = case["target"]
    source_xyz = tuple(int(value) for value in case["source_foot_mm"])
    target_xyz = tuple(int(value) for value in case["target_foot_mm"])
    dx = target_xyz[0] - source_xyz[0]
    dy = target_xyz[1] - source_xyz[1]
    dz = target_xyz[2] - source_xyz[2]
    step_length_mm = round(math.sqrt(dx * dx + dy * dy + dz * dz))
    step_height_mm = abs(dz)
    foothold_slope = int(target["foothold_slope_cdeg"])
    body_slope = int(target["body_slope_cdeg"])
    support_margin = int(target["support_margin_mm"])
    body_clearance = int(target["body_sweep_clearance_mm"])
    slacks = {
        "body_clearance_mm": body_clearance,
        "body_slope_cdeg": 3000 - body_slope,
        "foothold_slope_cdeg": 2500 - foothold_slope,
        "step_height_mm": 250 - step_height_mm,
        "step_length_mm": 500 - step_length_mm,
        "support_margin_mm": support_margin - 50,
    }
    witness = {
        "body_slope_cdeg": body_slope,
        "foothold_slope_cdeg": foothold_slope,
        "moving_leg_id": int(case["moving_leg_id"]),
        "step_height_mm": step_height_mm,
        "step_length_mm": step_length_mm,
        "support_margin_mm": support_margin,
    }
    if int(case["phase"]) != int(case["required_phase"]):
        return _decision(False, "G2I_L_SEQUENCE", witness=witness, slacks=slacks)
    if not bool(target["grid_valid"]):
        return _decision(False, "G2I_L_GRID_DOMAIN", witness=witness, slacks=slacks)
    if bool(target["foothold_unknown"]):
        return _decision(
            False, "G2I_L_FOOTHOLD_UNKNOWN", witness=witness, slacks=slacks
        )
    if bool(target["foothold_obstacle"]):
        return _decision(
            False, "G2I_L_FOOTHOLD_OBSTACLE", witness=witness, slacks=slacks
        )
    if not bool(target["foothold_traversable"]) or foothold_slope > 2500:
        return _decision(
            False, "G2I_L_FOOTHOLD_SLOPE", witness=witness, slacks=slacks
        )
    if step_length_mm > 500:
        return _decision(False, "G2I_L_STEP_LENGTH", witness=witness, slacks=slacks)
    if step_height_mm > 250:
        return _decision(False, "G2I_L_STEP_HEIGHT", witness=witness, slacks=slacks)
    if support_margin < 50:
        return _decision(
            False, "G2I_L_SUPPORT_MARGIN", witness=witness, slacks=slacks
        )
    if body_clearance <= 0 or body_slope > 3000:
        return _decision(False, "G2I_L_BODY_SWEEP", witness=witness, slacks=slacks)
    return _decision(True, "G2I_L_SAFE", witness=witness, slacks=slacks)
