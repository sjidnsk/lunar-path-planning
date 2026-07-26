from __future__ import annotations

from decimal import Decimal
from typing import Any


MODEL_ID = "quadratic-normalized-speed/v1"


def evaluate_energy(
    *,
    action_speed_mm_s: int,
    reference_speed_mm_s: int,
    max_energy_decimal: str,
) -> dict[str, Any]:
    if action_speed_mm_s < 0 or reference_speed_mm_s <= 0:
        raise ValueError("energy evaluator inputs must be positive")
    energy = (
        Decimal(int(action_speed_mm_s)) / Decimal(int(reference_speed_mm_s))
    ) ** 2
    maximum = Decimal(str(max_energy_decimal))
    return {
        "accepted": energy <= maximum,
        "energy_decimal": format(energy, "f"),
        "max_energy_decimal": format(maximum, "f"),
        "model_id": MODEL_ID,
    }
