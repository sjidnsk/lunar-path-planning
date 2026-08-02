from __future__ import annotations

from typing import Any


MODEL_ID = "touchdown-speed-upper-bound/v1"


def evaluate_stop(
    *,
    action_speed_mm_s: int,
    max_touchdown_speed_mm_s: int,
) -> dict[str, Any]:
    if action_speed_mm_s < 0 or max_touchdown_speed_mm_s <= 0:
        raise ValueError("stop evaluator inputs must be positive")
    slack = int(max_touchdown_speed_mm_s) - int(action_speed_mm_s)
    return {
        "accepted": slack >= 0,
        "action_speed_mm_s": int(action_speed_mm_s),
        "model_id": MODEL_ID,
        "slack_mm_s": slack,
    }
