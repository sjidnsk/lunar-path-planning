from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "xunce-platform-contract/v1"
DEFAULT_PLATFORM_CONTRACT = "configs/platforms/agilex_scout_mini_piper_v1.json"


def stable_contract_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def resolve_path(path: str | Path, repo_root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (repo_root / value).resolve()


def load_platform_contract(config: dict[str, Any] | None = None, *, repo_root: Path) -> dict[str, Any]:
    payload = dict(config or {})
    contract_path = payload.get("platform_contract") or payload.get("platform_contract_path") or DEFAULT_PLATFORM_CONTRACT
    path = resolve_path(str(contract_path), repo_root)
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"platform contract schema_version must be {SCHEMA_VERSION!r}")
    return contract


def platform_lineage(contract: dict[str, Any]) -> dict[str, Any]:
    terrain = contract.get("terrain_policy", {}) if isinstance(contract.get("terrain_policy"), dict) else {}
    base = contract.get("base", {}) if isinstance(contract.get("base"), dict) else {}
    max_climb = float(terrain.get("platform_max_climb_deg", base.get("platform_max_climb_deg", 30.0)))
    max_slope = float(terrain.get("max_traversable_slope_deg", max_climb))
    sensitivity = terrain.get("slope_sensitivity_thresholds_deg", [20.0, max_slope])
    return {
        "platform_contract_id": str(contract.get("platform_id") or "unknown_platform"),
        "platform_contract_hash": stable_contract_hash(contract),
        "platform_max_climb_deg": max_climb,
        "max_traversable_slope_deg": max_slope,
        "slope_sensitivity_thresholds_deg": [float(value) for value in sensitivity],
    }


def load_platform_lineage(config: dict[str, Any] | None = None, *, repo_root: Path) -> dict[str, Any]:
    return platform_lineage(load_platform_contract(config, repo_root=repo_root))


def apply_stage23_platform_defaults(config: dict[str, Any], *, repo_root: Path) -> dict[str, Any]:
    payload = dict(config)
    lineage = load_platform_lineage(payload, repo_root=repo_root)
    payload.setdefault("platform_contract", DEFAULT_PLATFORM_CONTRACT)
    payload.setdefault("platform_contract_id", lineage["platform_contract_id"])
    payload.setdefault("platform_contract_hash", lineage["platform_contract_hash"])
    payload.setdefault("platform_max_climb_deg", lineage["platform_max_climb_deg"])
    payload.setdefault("max_traversable_slope_deg", lineage["max_traversable_slope_deg"])
    payload.setdefault("slope_sensitivity_thresholds_deg", lineage["slope_sensitivity_thresholds_deg"])
    payload["platform_lineage"] = lineage
    return payload
