from __future__ import annotations

import hashlib
import json
import math
import struct
import sys
import types
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

from producer.canonical import (
    canonical_json_bytes,
    canonical_loads,
    canonical_npz_bytes,
    decode_canonical_npz,
    derive_seed,
    domain_hash,
)
from producer.models import (
    assert_unique,
    make_case_identity,
    validate_truth_blind_case,
    validate_truth_row,
)
from producer.oracle_hopper import (
    evaluate_hopper,
    gaussian_square_mass_ppm,
    validate_hopper_parameter_record,
)
from producer.oracle_legged import evaluate_legged
from producer.oracle_wheel import evaluate_wheel, wheel_endpoint_mm
from run_producer import collect_isolation_evidence


ROOT = Path(__file__).resolve().parents[1]


def _wheel_case() -> dict[str, object]:
    return {
        "numeric_state": "decided",
        "start_pose_mm_urad": [0, 0, 0],
        "primitive": {
            "angular_urad_s": 0,
            "control_family": "straight",
            "duration_us": 2_000_000,
            "endpoint_mm_urad": [1000, 0, 0],
            "heading_bin": 0,
            "speed_um_s": 500_000,
        },
        "limits": {
            "allow_reverse": True,
            "allow_turn": True,
            "endpoint_tolerance_mm": 1,
            "max_abs_angular_urad_s": 2_000_000,
            "max_abs_speed_um_s": 1_000_000,
            "max_duration_us": 5_000_000,
            "min_turn_radius_mm": 250,
        },
        "terrain": {
            "inside_map": True,
            "known_sweep": True,
            "slope_cdeg": 0,
            "sweep_clearance_mm": 20,
            "traversable": True,
        },
    }


def _legged_case() -> dict[str, object]:
    return {
        "numeric_state": "decided",
        "moving_leg_id": 0,
        "phase": 0,
        "required_phase": 0,
        "source_foot_mm": [0, 0, 0],
        "target_foot_mm": [300, 0, 0],
        "target": {
            "body_slope_cdeg": 0,
            "body_sweep_clearance_mm": 20,
            "foothold_obstacle": False,
            "foothold_slope_cdeg": 0,
            "foothold_traversable": True,
            "foothold_unknown": False,
            "grid_valid": True,
            "support_margin_mm": 75,
        },
    }


def _hopper_record() -> dict[str, object]:
    return {
        "schema_version": "g2-hopper-candidate/v1",
        "parameter_set_id": "hopper-generic-internal-proxy/v1",
        "body_envelope_radius_m": "0.375",
        "launch_reference_height_m": "0.750",
        "arc_clearance_margin_m": "0.125",
        "landing_footprint_radius_m": "0.625",
        "stop_condition": {
            "model_id": "touchdown-speed-upper-bound/v1",
            "max_touchdown_speed_m_s": "2.500",
            "evaluator_source_sha256": "1" * 64,
        },
        "energy_model": {
            "model_id": "quadratic-normalized-speed/v1",
            "reference_speed_m_s": "2.500",
            "max_energy_decimal": "1.000000",
            "evaluator_source_sha256": "2" * 64,
        },
        "evidence_class": "candidate_engineering_proxy",
        "simulation_proxy": True,
        "physical_capability_claimed": False,
        "formal_evidence_eligible": False,
        "status": "pending_external_evidence",
    }


def _hopper_case() -> dict[str, object]:
    return {
        "numeric_state": "decided",
        "action": {
            "azimuth_index": 0,
            "azimuth_mdeg": 0,
            "elevation_index": 1,
            "elevation_mdeg": 45000,
            "speed_index": 1,
            "speed_mm_s": 2000,
        },
        "terrain": {
            "arc_clearance_slack_mm": 20,
            "arc_inside_map": True,
            "arc_known": True,
            "landing_footprint_clearance_mm": 20,
            "landing_height_error_mm": 0,
            "landing_halfwidth_mm": 5000,
            "landing_slope_cdeg": 0,
            "landing_sigma_mm": 289,
            "launch_clearance_mm": 20,
        },
        "touchdown_speed_mm_s": 2000,
    }


def _manual_domain_hash(domain: str, *parts: bytes) -> str:
    payload = bytearray()
    for part in (domain.encode("utf-8"), *parts):
        payload.extend(struct.pack(">Q", len(part)))
        payload.extend(part)
    return hashlib.sha256(payload).hexdigest()


def test_specification_freezes_exact_reduced_g2_scale() -> None:
    spec = json.loads((ROOT / "SPECIFICATION.json").read_text(encoding="utf-8"))

    assert spec["root_seed"] == 20260727
    assert spec["primitive_quotas"] == {
        "wheel": {
            "anti_alias": 404,
            "compound": 400,
            "kinematic": 630,
            "knownness_boundary": 300,
            "nominal": 700,
            "obstacle": 450,
            "slope": 450,
        },
        "legged": {
            "body_sweep": 420,
            "compound": 300,
            "foothold": 480,
            "nominal": 650,
            "sequence_grid": 284,
            "step_height": 360,
            "step_length": 360,
            "support": 480,
        },
        "hopper": {
            "action_lattice": 384,
            "arc": 600,
            "landing": 550,
            "landing_probability": 300,
            "launch": 360,
            "nominal": 516,
            "stop_model": 324,
            "unknown_boundary": 300,
        },
    }
    assert {
        platform: sum(quotas.values())
        for platform, quotas in spec["primitive_quotas"].items()
    } == {"wheel": 3334, "legged": 3334, "hopper": 3334}
    assert spec["request_mix"] == {
        "kilometer": {"hard_reachable": 2, "reachable": 6, "unreachable": 2},
        "standard": {"hard_reachable": 7, "reachable": 23, "unreachable": 3},
    }
    assert spec["candidate_attestation"]["formal_evidence_eligible"] is False
    assert spec["candidate_attestation"]["technical_independence"] == "T2_candidate"


def test_domain_hash_is_length_prefixed_and_domain_separated() -> None:
    want = _manual_domain_hash("g2-test/v1", b"a", b"bc")
    assert domain_hash("g2-test/v1", b"a", b"bc") == want
    assert domain_hash("g2-test/v1", b"ab", b"c") != want
    assert domain_hash("g2-test/v2", b"a", b"bc") != want


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, -0.0])
def test_canonical_json_rejects_noncanonical_floats(bad: float) -> None:
    with pytest.raises(ValueError, match="non-canonical float"):
        canonical_json_bytes({"value": bad})


def test_canonical_json_is_sorted_utf8_and_duplicate_keys_are_rejected() -> None:
    assert canonical_json_bytes({"z": 1, "a": "月"}) == (
        '{"a":"月","z":1}'.encode("utf-8")
    )
    with pytest.raises(ValueError, match="duplicate key"):
        canonical_loads(b'{"a":1,"a":2}')


def test_canonical_npz_has_fixed_order_metadata_and_pickle_free_round_trip() -> None:
    arrays = {
        "known": np.array([[1, 0]], dtype=np.uint8),
        "height_mm": np.array([[0, 125]], dtype=np.int32),
        "confidence_ppm": np.array([[1_000_000, 500_000]], dtype=np.uint32),
        "cell_class": np.array([[0, 2]], dtype=np.uint8),
    }
    metadata = {"cell_size_mm": 500, "source_kind": "procedural_simulation_proxy/v1"}

    first = canonical_npz_bytes(arrays, metadata)
    second = canonical_npz_bytes(dict(reversed(list(arrays.items()))), metadata)
    assert first == second

    with zipfile.ZipFile(BytesIO(first), "r") as archive:
        infos = archive.infolist()
        assert [info.filename for info in infos] == [
            "height_mm.npy",
            "cell_class.npy",
            "known.npy",
            "confidence_ppm.npy",
            "metadata.json",
        ]
        assert all(info.compress_type == zipfile.ZIP_STORED for info in infos)
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in infos)
        assert all(info.external_attr == (0o100644 << 16) for info in infos)

    decoded_arrays, decoded_metadata = decode_canonical_npz(first)
    assert decoded_metadata == metadata
    for key, expected in arrays.items():
        np.testing.assert_array_equal(decoded_arrays[key], expected)


def test_seed_derivation_uses_every_domain_component() -> None:
    base = derive_seed(20260727, "wheel", "labels", "nominal", 0)
    assert base == derive_seed(20260727, "wheel", "labels", "nominal", 0)
    assert len(
        {
            base,
            derive_seed(20260727, "legged", "labels", "nominal", 0),
            derive_seed(20260727, "wheel", "requests", "nominal", 0),
            derive_seed(20260727, "wheel", "labels", "slope", 0),
            derive_seed(20260727, "wheel", "labels", "nominal", 1),
        }
    ) == 5


def test_truth_blind_rows_reject_truth_and_provider_outcome_keys_recursively() -> None:
    neutral = {
        "schema_version": "g2-case/v1",
        "platform_kind": "wheel",
        "stratum": "slope",
        "case": {"heading_index": 4, "terrain": {"slope_cdeg": 3000}},
    }
    validate_truth_blind_case(neutral)

    for key in (
        "expected_safe",
        "oracle_reachable",
        "target_reason",
        "provider_success",
        "runtime_ms",
        "complete_l2",
    ):
        contaminated = {**neutral, "nested": {key: False}}
        with pytest.raises(ValueError, match=key):
            validate_truth_blind_case(contaminated)


def test_truth_rows_reject_provider_fields_but_allow_independent_oracle_fields() -> None:
    row = {
        "oracle_safe": True,
        "oracle_reason_code": "G2I_W_SAFE",
        "numeric_witness": {"clearance_mm": 20},
    }
    validate_truth_row(row)
    with pytest.raises(ValueError, match="provider_safe"):
        validate_truth_row({**row, "evidence": {"provider_safe": True}})


def test_case_identity_is_content_addressed_and_unique_check_is_strict() -> None:
    case = {"heading_index": 3, "terrain_id": "terrain-1"}
    case_sha, label_id = make_case_identity("wheel", case)
    assert case_sha == domain_hash("g2-case/v1", b"wheel", canonical_json_bytes(case))
    assert label_id == f"g2i-label-wheel-{case_sha[:20]}"
    assert_unique(["a", "b"], field_name="label_id")
    with pytest.raises(ValueError, match="duplicate label_id"):
        assert_unique(["a", "a"], field_name="label_id")


def test_isolation_preflight_rejects_loaded_forbidden_module_without_importing_it() -> None:
    fake = types.ModuleType("path_planner.injected")
    fake.__file__ = "D:/forbidden/path_planner/injected.py"
    sys.modules[fake.__name__] = fake
    try:
        evidence = collect_isolation_evidence(
            project_root=Path("C:/absent-project-root"),
            environment={"PATH": "x", "SECRET": "must-not-leak"},
        )
    finally:
        del sys.modules[fake.__name__]

    assert evidence["passed"] is False
    assert "path_planner.injected" in evidence["forbidden_loaded_modules"]
    assert evidence["environment"] == {"PATH": "x"}


def test_wheel_closed_form_endpoint_and_inclusive_upper_limits() -> None:
    case = _wheel_case()
    assert wheel_endpoint_mm(case) == (1000, 0, 0)
    case["terrain"]["slope_cdeg"] = 3000  # type: ignore[index]
    case["primitive"]["speed_um_s"] = 1_000_000  # type: ignore[index]
    case["primitive"]["endpoint_mm_urad"] = [2000, 0, 0]  # type: ignore[index]
    result = evaluate_wheel(case)
    assert result["oracle_safe"] is True
    assert result["oracle_reason_code"] == "G2I_W_SAFE"


def test_wheel_one_epsilon_closed_contact_numeric_and_precedence() -> None:
    slope = _wheel_case()
    slope["terrain"]["slope_cdeg"] = 3001  # type: ignore[index]
    assert evaluate_wheel(slope)["oracle_reason_code"] == "G2I_W_SLOPE_LIMIT"

    contact = _wheel_case()
    contact["terrain"]["sweep_clearance_mm"] = 0  # type: ignore[index]
    assert (
        evaluate_wheel(contact)["oracle_reason_code"]
        == "G2I_W_CLOSED_OBSTACLE_CONTACT"
    )

    compound = _wheel_case()
    compound["primitive"]["speed_um_s"] = 1_000_001  # type: ignore[index]
    compound["terrain"]["sweep_clearance_mm"] = -1  # type: ignore[index]
    assert evaluate_wheel(compound)["oracle_reason_code"] == "G2I_W_SPEED_DOMAIN"

    undecided = _wheel_case()
    undecided["numeric_state"] = "undecided"
    assert evaluate_wheel(undecided)["oracle_reason_code"] == "G2I_NUMERIC_UNDECIDED"


def test_legged_inclusive_step_height_support_and_closed_body_boundaries() -> None:
    equal = _legged_case()
    equal["target_foot_mm"] = [0, 0, 250]
    equal["target"]["support_margin_mm"] = 50  # type: ignore[index]
    result = evaluate_legged(equal)
    assert result["oracle_safe"] is True

    length_equal = _legged_case()
    length_equal["target_foot_mm"] = [500, 0, 0]
    assert evaluate_legged(length_equal)["oracle_safe"] is True

    too_long = _legged_case()
    too_long["target_foot_mm"] = [501, 0, 0]
    assert evaluate_legged(too_long)["oracle_reason_code"] == "G2I_L_STEP_LENGTH"

    too_high = _legged_case()
    too_high["target_foot_mm"] = [300, 0, 251]
    assert evaluate_legged(too_high)["oracle_reason_code"] == "G2I_L_STEP_HEIGHT"

    support = _legged_case()
    support["target"]["support_margin_mm"] = 49  # type: ignore[index]
    assert evaluate_legged(support)["oracle_reason_code"] == "G2I_L_SUPPORT_MARGIN"

    contact = _legged_case()
    contact["target"]["body_sweep_clearance_mm"] = 0  # type: ignore[index]
    assert evaluate_legged(contact)["oracle_reason_code"] == "G2I_L_BODY_SWEEP"

    compound = _legged_case()
    compound["phase"] = 3
    compound["target"]["foothold_unknown"] = True  # type: ignore[index]
    assert evaluate_legged(compound)["oracle_reason_code"] == "G2I_L_SEQUENCE"


def test_hopper_complete_record_gaussian_and_inclusive_probability_boundary() -> None:
    record = _hopper_record()
    validate_hopper_parameter_record(record)
    assert gaussian_square_mass_ppm(halfwidth_mm=811, sigma_mm=289) == 990000
    assert gaussian_square_mass_ppm(halfwidth_mm=810, sigma_mm=289) == 989892

    equal = _hopper_case()
    equal["terrain"]["landing_halfwidth_mm"] = 811  # type: ignore[index]
    assert evaluate_hopper(equal, record)["oracle_safe"] is True

    below = _hopper_case()
    below["terrain"]["landing_halfwidth_mm"] = 810  # type: ignore[index]
    assert evaluate_hopper(below, record)["oracle_reason_code"] == "G2I_H_LANDING_MASS"


def test_hopper_closed_contact_stop_energy_and_parameter_fail_closed() -> None:
    record = _hopper_record()
    contact = _hopper_case()
    contact["terrain"]["arc_clearance_slack_mm"] = 0  # type: ignore[index]
    assert evaluate_hopper(contact, record)["oracle_reason_code"] == "G2I_H_ARC_CLEARANCE"

    stop_equal = _hopper_case()
    stop_equal["touchdown_speed_mm_s"] = 2500
    assert evaluate_hopper(stop_equal, record)["oracle_safe"] is True

    stop_above = _hopper_case()
    stop_above["touchdown_speed_mm_s"] = 2501
    assert evaluate_hopper(stop_above, record)["oracle_reason_code"] == "G2I_H_STOP"

    incomplete = dict(record)
    incomplete.pop("arc_clearance_margin_m")
    with pytest.raises(ValueError, match="arc_clearance_margin_m"):
        validate_hopper_parameter_record(incomplete)
