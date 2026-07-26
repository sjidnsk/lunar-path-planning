from __future__ import annotations

from typing import Any, Iterable, Mapping

from .canonical import canonical_json_bytes, domain_hash


_TRUTH_BLIND_FORBIDDEN = {
    "complete_l2",
    "expected_reason",
    "expected_safe",
    "oracle_reachable",
    "oracle_reason_code",
    "oracle_safe",
    "provider_safe",
    "provider_success",
    "runtime_ms",
    "target_reason",
}
_TRUTH_ROW_FORBIDDEN = {
    "cache_hit",
    "complete_l2",
    "expanded_states",
    "provider_safe",
    "provider_success",
    "runtime_ms",
    "timing_ms",
}


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key)
            yield from _walk_keys(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_keys(child)


def _reject_keys(value: Mapping[str, Any], forbidden: set[str]) -> None:
    for key in _walk_keys(value):
        normalized = key.casefold()
        if normalized in forbidden:
            raise ValueError(f"forbidden key: {key}")


def validate_truth_blind_case(row: dict[str, Any]) -> None:
    _reject_keys(row, _TRUTH_BLIND_FORBIDDEN)
    canonical_json_bytes(row)


def validate_truth_row(row: dict[str, Any]) -> None:
    _reject_keys(row, _TRUTH_ROW_FORBIDDEN)
    canonical_json_bytes(row)


def make_case_identity(platform: str, case: dict[str, Any]) -> tuple[str, str]:
    if platform not in {"wheel", "legged", "hopper"}:
        raise ValueError(f"unknown platform: {platform}")
    case_sha = domain_hash(
        "g2-case/v1",
        platform.encode("utf-8"),
        canonical_json_bytes(case),
    )
    return case_sha, f"g2i-label-{platform}-{case_sha[:20]}"


def assert_unique(values: Iterable[str], *, field_name: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate {field_name}: {value}")
        seen.add(value)
