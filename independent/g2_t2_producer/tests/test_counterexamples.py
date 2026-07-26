from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from producer.generate_cases import generate_all_cases
from producer.models import make_case_identity, validate_truth_blind_case
from producer.oracle_hopper import build_hopper_parameter_record, evaluate_hopper
from producer.oracle_legged import evaluate_legged
from producer.oracle_wheel import evaluate_wheel, wheel_sweep_points
from producer.geometry import point_in_polygon_closed


ROOT = Path(__file__).resolve().parents[1]


def _hopper_record() -> dict[str, object]:
    return build_hopper_parameter_record(ROOT)


def _all_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from _all_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_keys(child)


def test_full_truth_blind_case_generation_has_exact_quotas_and_diversity() -> None:
    spec = json.loads((ROOT / "SPECIFICATION.json").read_text(encoding="utf-8"))
    generated = generate_all_cases(spec, hopper_parameter_record=_hopper_record())
    assert {platform: len(rows) for platform, rows in generated.items()} == {
        "wheel": 3334,
        "legged": 3334,
        "hopper": 3334,
    }

    all_case_hashes: list[str] = []
    all_label_ids: list[str] = []
    evaluator = {
        "wheel": lambda case: evaluate_wheel(case),
        "legged": lambda case: evaluate_legged(case),
        "hopper": lambda case: evaluate_hopper(case, _hopper_record()),
    }
    for platform, rows in generated.items():
        assert Counter(row["stratum"] for row in rows) == Counter(
            spec["primitive_quotas"][platform]
        )
        terrain_hashes = {row["case"]["terrain_sha256"] for row in rows}
        assert len(terrain_hashes) >= 64
        decisions = []
        for row in rows:
            validate_truth_blind_case(row)
            keys = set(_all_keys(row))
            assert not {
                "expected_safe",
                "oracle_safe",
                "oracle_reachable",
                "provider_success",
                "runtime_ms",
            }.intersection(keys)
            case_sha, label_id = make_case_identity(platform, row["case"])
            all_case_hashes.append(case_sha)
            all_label_ids.append(label_id)
            decisions.append(evaluator[platform](row["case"]))
        safe_ratio = sum(bool(row["oracle_safe"]) for row in decisions) / len(decisions)
        assert 0.35 <= safe_ratio <= 0.65
        assert len({row["oracle_reason_code"] for row in decisions}) >= 5

    assert len(all_case_hashes) == len(set(all_case_hashes)) == 10002
    assert len(all_label_ids) == len(set(all_label_ids)) == 10002
    assert len({row["case"]["primitive"]["heading_bin"] for row in generated["wheel"]}) >= 72
    assert len({row["case"]["primitive"]["control_family"] for row in generated["wheel"]}) >= 7
    leg_counts = Counter(row["case"]["moving_leg_id"] for row in generated["legged"])
    assert all(leg_counts[leg_id] >= 700 for leg_id in range(4))
    action_counts = Counter(
        (
            row["case"]["action"]["speed_index"],
            row["case"]["action"]["elevation_index"],
            row["case"]["action"]["azimuth_index"],
        )
        for row in generated["hopper"]
    )
    assert len(action_counts) == 192
    assert all(count >= 2 for count in action_counts.values())


def test_curved_sweep_between_endpoint_counterexample_is_not_accepted() -> None:
    spec = json.loads((ROOT / "SPECIFICATION.json").read_text(encoding="utf-8"))
    rows = generate_all_cases(spec, hopper_parameter_record=_hopper_record())["wheel"]
    between = next(
        row
        for row in rows
        if row["stratum"] == "anti_alias"
        and evaluate_wheel(row["case"])["oracle_reason_code"]
        == "G2I_W_CLOSED_OBSTACLE_CONTACT"
    )
    sweep = wheel_sweep_points(between["case"])
    polygon = between["case"]["terrain"]["obstacle_polygons_mm"][0]
    assert point_in_polygon_closed(sweep[0], polygon) is False
    assert point_in_polygon_closed(sweep[-1], polygon) is False
    result = evaluate_wheel(between["case"])
    assert result["oracle_safe"] is False
    assert result["oracle_reason_code"] == "G2I_W_CLOSED_OBSTACLE_CONTACT"
