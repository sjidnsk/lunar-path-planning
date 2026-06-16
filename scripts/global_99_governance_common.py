"""Shared governance helpers for Global 99 audit runners."""

from __future__ import annotations

from typing import Any


GLOBAL_99_BOUNDARY_DEFAULTS: dict[str, bool] = {
    "publishes_checkpoint": False,
    "replaces_default_policy": False,
    "connects_real_executor": False,
    "starts_online_canary": False,
    "runs_new_ppo_update": False,
    "modifies_network": False,
    "modifies_action_space": False,
    "modifies_default_astar": False,
}


def global_99_boundary_defaults() -> dict[str, bool]:
    """Return a fresh copy of the standard Global 99 closed boundary fields."""

    return dict(GLOBAL_99_BOUNDARY_DEFAULTS)


def merge_global_99_boundary_defaults(summary: dict[str, Any]) -> dict[str, Any]:
    """Return a summary copy with missing standard boundary fields backfilled."""

    merged: dict[str, Any] = global_99_boundary_defaults()
    merged.update(summary)
    return merged
