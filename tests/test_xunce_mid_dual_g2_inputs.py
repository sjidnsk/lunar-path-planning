from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import zipfile

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "xunce_mid_dual_g2_inputs.py"
CONFIG_PATH = REPO_ROOT / "configs" / "xunce_mid_dual_g2_planning_time_v1.json"
AUTHORIZATION_SHA256 = (
    "720e11ef04ad2b57283421809a077ccf1f0b35167a9f482082241398ad0214d2"
)
PLATFORMS = ("wheel", "legged", "hopper")
SHA_A = hashlib.sha256(b"a").hexdigest()
SHA_B = hashlib.sha256(b"b").hexdigest()
SHA_C = hashlib.sha256(b"c").hexdigest()
SHA_D = hashlib.sha256(b"d").hexdigest()


def _load_module():
    assert SCRIPT_PATH.is_file(), "G2 input readiness implementation is missing"
    spec = importlib.util.spec_from_file_location("xunce_mid_dual_g2_inputs_tested", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    if str(SCRIPT_PATH.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT_PATH.parent))
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _domain_hash(domain: str, *parts: bytes) -> str:
    payload = bytearray()
    for part in (domain.encode("utf-8"), *parts):
        payload.extend(struct.pack(">Q", len(part)))
        payload.extend(part)
    return hashlib.sha256(payload).hexdigest()


def _primitive_label(platform: str, index: int) -> dict[str, object]:
    terrain_sha256 = hashlib.sha256(
        f"primitive-terrain:{platform}:{index}".encode("utf-8")
    ).hexdigest()
    primitive = {
        "case_index": index,
        "terrain_sha256": terrain_sha256,
    }
    case_sha256 = _domain_hash(
        "g2-case/v1",
        platform.encode("utf-8"),
        _canonical_json_bytes(primitive),
    )
    return {
        "case_envelope_sha256": SHA_A,
        "case_sha256": case_sha256,
        "label_id": f"g2i-label-{platform}-{case_sha256[:20]}",
        "numeric_witness": {"finite": True},
        "oracle_reason_code": f"G2I_{platform[0].upper()}_OK",
        "oracle_safe": index % 2 == 0,
        "oracle_spec_sha256": SHA_B,
        "platform_kind": platform,
        "primitive_canonical": primitive,
        "primitive_sha256": _domain_hash(
            "g2-primitive/v2",
            _canonical_json_bytes(primitive),
        ),
        "producer_id": "g2-isolated-t2-producer",
        "producer_implementation_sha256": SHA_C,
        "producer_revision": "0.2.0",
        "profile_or_parameter_record_sha256": SHA_D,
        "profile_or_parameter_set_id": f"independent-{platform}/v1",
        "safety_slacks": {"finite": 1},
        "schema_version": "g2-primitive-label/v2",
        "terrain_sha256": terrain_sha256,
    }


def _primitive_labels() -> list[dict[str, object]]:
    return [
        _primitive_label(platform, index)
        for platform in PLATFORMS
        for index in range(3334)
    ]


def _small_map_optima() -> list[dict[str, object]]:
    return [
        {
            "candidate_edge_count": 10,
            "case_id": f"small-map-{platform}",
            "certificate_sha256": hashlib.sha256(
                f"certificate:{platform}".encode("utf-8")
            ).hexdigest(),
            "completeness": {
                "actual_candidate_edge_count": 10,
                "actual_node_count": 5,
                "complete": True,
                "expected_candidate_edge_count": 10,
                "expected_node_count": 5,
            },
            "map_sha256": hashlib.sha256(
                f"map:{platform}".encode("utf-8")
            ).hexdigest(),
            "platform_kind": platform,
            "schema_version": "g2-small-map-optimum/v1",
            "solver_implementation_sha256": SHA_A,
        }
        for platform in PLATFORMS
    ]


def _request_row(
    platform: str,
    scale: str,
    difficulty_class: str,
    index: int,
    *,
    terrain_sha256: str | None = None,
    terrain_geometry_sha256: str | None = None,
) -> dict[str, object]:
    raw_source_sha256 = hashlib.sha256(
        f"raw:{platform}:{scale}:{index}".encode("utf-8")
    ).hexdigest()
    certificate_sha256 = hashlib.sha256(
        f"certificate:{platform}:{scale}:{index}".encode("utf-8")
    ).hexdigest()
    terrain_sha256 = terrain_sha256 or hashlib.sha256(
        f"terrain:{platform}:{scale}:{index}".encode("utf-8")
    ).hexdigest()
    terrain_geometry_sha256 = terrain_geometry_sha256 or hashlib.sha256(
        f"geometry:{platform}:{scale}:{index}".encode("utf-8")
    ).hexdigest()
    profile_sha256 = hashlib.sha256(
        f"profile:{platform}".encode("utf-8")
    ).hexdigest()
    truth_request_sha256 = _domain_hash(
        "g2-truth-request/v2",
        raw_source_sha256.encode("ascii"),
        certificate_sha256.encode("ascii"),
        terrain_sha256.encode("ascii"),
        terrain_geometry_sha256.encode("ascii"),
        profile_sha256.encode("ascii"),
    )
    reachable = difficulty_class != "unreachable"
    return {
        "determinism_seed": index,
        "difficulty_class": difficulty_class,
        "goal": {"node_id": "n:1:1"},
        "graph_template_sha256": SHA_A,
        "objective": {
            "kind": "minimum_resource_complete_l2/v1",
            "resource_weight_milli": 1000,
        },
        "oracle_reachable": reachable,
        "platform_kind": platform,
        "producer_implementation_sha256": SHA_B,
        "profile_or_parameter_record_sha256": profile_sha256,
        "raw_source_sha256": raw_source_sha256,
        "request_id": (
            f"g2i-req-{platform}-{scale}-{truth_request_sha256[:20]}"
        ),
        "resource_budget": {
            "max_path_primitives": 128 if scale == "standard" else 512
        },
        "scale": scale,
        "schema_version": "g2-truth-request/v2",
        "selection_hash": SHA_C,
        "selection_rank": index,
        "selection_seed": 20260727,
        "source_ids": [f"raw-source-{platform}-{scale}-{index}"],
        "start": {"node_id": "n:0:0"},
        "terrain_geometry_sha256": terrain_geometry_sha256,
        "terrain_provenance": {
            "physical_obstacle_cells_written": False,
            "source_kind": (
                "procedural_simulation_proxy/v1"
                if scale == "standard"
                else "synthetic_terrain_obstacle_proxy/v1"
            ),
        },
        "terrain_sha256": terrain_sha256,
        "truth_certificate_sha256": certificate_sha256,
        "truth_request_sha256": truth_request_sha256,
    }


def _requests() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for platform in PLATFORMS:
        index = 0
        for scale, normal_count, hard_count, unreachable_count in (
            ("standard", 23, 7, 3),
            ("kilometer", 6, 2, 2),
        ):
            for difficulty_class, count in (
                ("reachable", normal_count),
                ("hard_reachable", hard_count),
                ("unreachable", unreachable_count),
            ):
                for _ in range(count):
                    rows.append(
                        _request_row(platform, scale, difficulty_class, index)
                    )
                    index += 1
    return rows


def _canonical_npy(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(
        stream,
        np.ascontiguousarray(array),
        version=(2, 0),
        allow_pickle=False,
    )
    return stream.getvalue()


def _terrain_blob() -> tuple[bytes, str]:
    arrays = {
        "height_mm": np.array([[0, 1], [2, 3]], dtype="<i4"),
        "cell_class": np.array([[0, 2], [0, 0]], dtype="u1"),
        "known": np.ones((2, 2), dtype="u1"),
        "confidence_ppm": np.full((2, 2), 1_000_000, dtype="<u4"),
    }
    metadata = {
        "geometry_schema_version": "g2-neutral-terrain/v2",
        "macro_cell_size_mm": 500,
        "provenance": {
            "physical_obstacle_cells_written": False,
            "source_kind": "procedural_simulation_proxy/v1",
        },
        "sub_20m_layer_source_kind": None,
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in (
            "height_mm",
            "cell_class",
            "known",
            "confidence_ppm",
        ):
            info = zipfile.ZipInfo(
                filename=f"{name}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.flag_bits = 0
            archive.writestr(info, _canonical_npy(arrays[name]))
        info = zipfile.ZipInfo(
            filename="metadata.json",
            date_time=(1980, 1, 1, 0, 0, 0),
        )
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        info.flag_bits = 0
        archive.writestr(info, _canonical_json_bytes(metadata))
    geometry_sha256 = _domain_hash(
        "g2-terrain-geometry/v1",
        *(
            name.encode("ascii") + b"\0" + arrays[name].tobytes(order="C")
            for name in (
                "height_mm",
                "cell_class",
                "known",
                "confidence_ppm",
            )
        ),
    )
    return output.getvalue(), geometry_sha256


def _source(identity: str, marker: str) -> dict[str, object]:
    return {
        "identity": identity,
        "source_bytes_sha256": hashlib.sha256(
            f"source:{marker}".encode("utf-8")
        ).hexdigest(),
        "implementation_sha256": hashlib.sha256(
            f"implementation:{marker}".encode("utf-8")
        ).hexdigest(),
    }


def _hopper_candidate() -> dict[str, object]:
    return {
        "arc_clearance_margin_m": "0.125",
        "body_envelope_radius_m": "0.375",
        "energy_model": {
            "evaluator_source_sha256": SHA_A,
            "model_id": "quadratic-normalized-speed/v1",
        },
        "evidence_class": "candidate_engineering_proxy",
        "formal_evidence_eligible": False,
        "landing_footprint_radius_m": "0.625",
        "launch_reference_height_m": "0.750",
        "parameter_set_id": "hopper-generic-internal-proxy/v2",
        "physical_capability_claimed": False,
        "schema_version": "g2-hopper-candidate/v2",
        "simulation_proxy": True,
        "status": "pending_external_evidence",
        "stop_condition": {
            "evaluator_source_sha256": SHA_B,
            "model_id": "touchdown-speed-upper-bound/v1",
        },
    }


def test_default_config_is_blocked_and_contains_no_formal_input_paths() -> None:
    module = _load_module()
    assert CONFIG_PATH.is_file(), "default G2 config is missing"
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    result = module.preflight(config)

    assert config["formal_inputs"] == {
        "truth_bundle": None,
        "artifact_bound_approval": None,
        "hopper_candidate_record": None,
        "execution_bundle": None,
    }
    assert config["approval_artifact"] == {
        "base": "D:/xunce/inputs/mid_dual/g2-approvals",
        "relative_contract": (
            "<input_set_id>/artifact-bound-o2-approval.json"
        ),
        "file_name": "artifact-bound-o2-approval.json",
        "publication_mode": "separate-no-candidate-mutation/v1",
    }
    assert module.approval_artifact_path(
        "g2t2-candidate-" + "a" * 24
    ).as_posix() == (
        "D:/xunce/inputs/mid_dual/g2-approvals/"
        "g2t2-candidate-aaaaaaaaaaaaaaaaaaaaaaaa/"
        "artifact-bound-o2-approval.json"
    )
    assert result["status"] == "blocked"
    assert result["formal_evidence_eligible"] is False
    assert result["formal_row_count"] == 0
    assert set(result["blockers"]) == {
        "missing_independent_g2_input_bundle",
        "missing_artifact_bound_o2_approval",
        "approve_midterm_hopper_simulation_proxy_parameter_set",
    }


def test_input_bundle_requires_3334_primitive_labels_per_platform() -> None:
    module = _load_module()
    rows = _primitive_labels()

    result = module.audit_primitive_labels(rows)

    assert result["platform_counts"] == {
        "wheel": 3334,
        "legged": 3334,
        "hopper": 3334,
    }
    with pytest.raises(
        module.G2InputContractError,
        match="primitive_label_count_mismatch",
    ):
        module.audit_primitive_labels(rows[:-1])


def test_input_bundle_requires_one_small_map_optimum_per_platform() -> None:
    module = _load_module()
    rows = _small_map_optima()

    assert module.audit_small_map_optima(rows)["platform_counts"] == {
        "wheel": 1,
        "legged": 1,
        "hopper": 1,
    }
    with pytest.raises(
        module.G2InputContractError,
        match="small_map_optimum_count_mismatch",
    ):
        module.audit_small_map_optima(rows[:-1])


def test_request_matrix_is_33_standard_plus_10_kilometer_per_platform() -> None:
    module = _load_module()
    rows = _requests()

    result = module.audit_request_matrix(rows)

    assert result["platform_scale_counts"] == {
        "wheel/standard": 33,
        "wheel/kilometer": 10,
        "legged/standard": 33,
        "legged/kilometer": 10,
        "hopper/standard": 33,
        "hopper/kilometer": 10,
    }
    rows[0] = {**rows[0], "scale": "kilometer"}
    with pytest.raises(
        module.G2InputContractError,
        match="request_matrix_count_mismatch",
    ):
        module.audit_request_matrix(rows)


def test_request_matrix_has_exact_reachable_hard_unreachable_counts() -> None:
    module = _load_module()
    rows = _requests()

    result = module.audit_request_matrix(rows)

    assert result["platform_scale_difficulty_counts"]["wheel/standard"] == {
        "reachable": 23,
        "hard_reachable": 7,
        "unreachable": 3,
    }
    target = next(
        index
        for index, row in enumerate(rows)
        if row["platform_kind"] == "wheel"
        and row["scale"] == "standard"
        and row["difficulty_class"] == "reachable"
    )
    rows[target] = {
        **rows[target],
        "difficulty_class": "hard_reachable",
    }
    with pytest.raises(
        module.G2InputContractError,
        match="request_difficulty_count_mismatch",
    ):
        module.audit_request_matrix(rows)


def test_oracle_and_provider_source_identity_must_be_independent() -> None:
    module = _load_module()
    oracle = _source("independent-finite-oracle/v1", "oracle")
    provider = _source("path-planner-v2-provider/v1", "provider")

    result = module.audit_source_separation(
        oracle_source=oracle,
        provider_source=provider,
    )

    assert result["separated"] is True
    with pytest.raises(
        module.G2InputContractError,
        match="provider_oracle_source_not_separated",
    ):
        module.audit_source_separation(
            oracle_source=oracle,
            provider_source=dict(oracle),
        )


def test_request_hash_terrain_hash_and_label_join_are_one_to_one() -> None:
    module = _load_module()
    terrain_blob, terrain_geometry_sha256 = _terrain_blob()
    terrain_sha256 = hashlib.sha256(terrain_blob).hexdigest()
    request = _request_row(
        "wheel",
        "standard",
        "reachable",
        0,
        terrain_sha256=terrain_sha256,
        terrain_geometry_sha256=terrain_geometry_sha256,
    )
    label = _primitive_label("wheel", 7)

    module.validate_primitive_label_identity(label)
    result = module.build_execution_crosswalk(
        [request],
        {terrain_sha256: terrain_blob},
    )

    assert len(result["provider_requests"]) == 1
    assert result["crosswalk"] == [
        {
            "schema_version": "xunce-mid-dual-g2-truth-provider-crosswalk/v1",
            "request_id": request["request_id"],
            "platform": "wheel",
            "scale": "standard",
            "request_class": "normal_reachable",
            "outcome_kind": "reachable",
            "truth_request_sha256": request["truth_request_sha256"],
            "truth_certificate_sha256": request["truth_certificate_sha256"],
            "terrain_sha256": terrain_sha256,
            "provider_request_sha256": result["provider_requests"][0][
                "provider_request_sha256"
            ],
        }
    ]
    provider_payload = result["provider_requests"][0]
    assert "oracle_reachable" not in provider_payload
    assert "difficulty_class" not in provider_payload
    assert "truth_certificate_sha256" not in provider_payload
    with pytest.raises(
        module.G2InputContractError,
        match="terrain_blob_sha256_mismatch",
    ):
        module.build_execution_crosswalk(
            [request],
            {terrain_sha256: terrain_blob + b"drift"},
        )


def test_hopper_test_fixture_is_rejected_as_formal_evidence() -> None:
    module = _load_module()
    replay_truth_sha256 = [
        hashlib.sha256(f"g3-truth:{index}".encode("utf-8")).hexdigest()
        for index in range(6)
    ]
    replay_provider_sha256 = [
        hashlib.sha256(f"g3-provider:{index}".encode("utf-8")).hexdigest()
        for index in range(6)
    ]
    input_binding = {
        "authorization_sha256": AUTHORIZATION_SHA256,
        "input_set_id": "g2t2-candidate-fixture",
        "manifest_core_sha256": SHA_A,
        "payload_root_sha256": SHA_B,
        "source_attestations_sha256": SHA_C,
        "producer_implementation_sha256": hashlib.sha256(
            b"producer-implementation"
        ).hexdigest(),
        "producer_source_bytes_sha256": hashlib.sha256(
            b"producer-source"
        ).hexdigest(),
        "reproduction_audit_sha256": hashlib.sha256(
            b"reproduction-audit"
        ).hexdigest(),
        "technical_isolation_audit_sha256": hashlib.sha256(
            b"technical-isolation-audit"
        ).hexdigest(),
        "technical_independence": "T2_candidate",
        "organizational_independence": "project_internal",
        "structural_audit_passed": True,
        "g3_replay_cohort_sha256": hashlib.sha256(
            b"g3-replay-cohort"
        ).hexdigest(),
        "g3_replay_truth_request_sha256": replay_truth_sha256,
        "g3_replay_provider_request_sha256": replay_provider_sha256,
    }
    provider = _source("path-planner-v2-provider/v1", "provider")
    oracle = _source("independent-finite-oracle/v1", "oracle")
    gate5b_fixture = {
        "evidence_class": "test_fixture",
        "formal_evidence_eligible": False,
        "parameter_set_id": "hopper_gate5b_algorithm_fixture/v1",
        "schema_version": "hopper-parameter-set-record/v1",
        "simulation_proxy": True,
    }

    rejected = module.resolve_hopper_formal_eligibility(
        candidate_record=gate5b_fixture,
        approval_record=None,
        input_binding=input_binding,
        provider_source=provider,
        oracle_source=oracle,
    )

    assert rejected["formal_evidence_eligible"] is False
    assert rejected["blockers"] == ["hopper_gate5b_test_fixture_not_formal"]

    candidate = _hopper_candidate()
    candidate_sha256 = hashlib.sha256(_canonical_json_bytes(candidate)).hexdigest()
    approval = {
        "schema_version": "xunce-mid-dual-g2-artifact-bound-approval/v1",
        "approval_id": "mid-dual-g2-o2-approval/v1",
        "authorization_sha256": AUTHORIZATION_SHA256,
        "formal_evidence_eligible": True,
        "input_set_id": input_binding["input_set_id"],
        "manifest_core_sha256": input_binding["manifest_core_sha256"],
        "payload_root_sha256": input_binding["payload_root_sha256"],
        "source_attestations_sha256": input_binding[
            "source_attestations_sha256"
        ],
        "producer_implementation_sha256": input_binding[
            "producer_implementation_sha256"
        ],
        "producer_source_bytes_sha256": input_binding[
            "producer_source_bytes_sha256"
        ],
        "reproduction_audit_sha256": input_binding[
            "reproduction_audit_sha256"
        ],
        "technical_isolation_audit_sha256": input_binding[
            "technical_isolation_audit_sha256"
        ],
        "hopper_candidate_record_sha256": candidate_sha256,
        "hopper_candidate_parameter_set_id": candidate["parameter_set_id"],
        "hopper_implementation_parameter_set_id": (
            "hopper_generic_internal_computational_simulation_proxy_midterm_g2g3/v1"
        ),
        "hopper_implementation_record_sha256": SHA_D,
        "regression_evidence_sha256": hashlib.sha256(b"regression").hexdigest(),
        "independent_technical_review_sha256": hashlib.sha256(
            b"review"
        ).hexdigest(),
        "provider_source": provider,
        "oracle_source": oracle,
        "g2_scope_authorized": True,
        "g3_scope_authorized": True,
        "g3_replay_cohort_sha256": input_binding[
            "g3_replay_cohort_sha256"
        ],
        "g3_replay_truth_request_sha256": replay_truth_sha256,
        "g3_replay_provider_request_sha256": replay_provider_sha256,
        "physical_capability_claimed": False,
        "hardware_certification_claimed": False,
    }
    approved = module.resolve_hopper_formal_eligibility(
        candidate_record=candidate,
        approval_record=approval,
        input_binding=input_binding,
        provider_source=provider,
        oracle_source=oracle,
    )

    assert approved["formal_evidence_eligible"] is True
    assert approved["blockers"] == []
    assert approved["candidate_record_sha256"] == candidate_sha256
    assert len(approved["approval_record_sha256"]) == 64


def test_ppo_target_is_optional_for_reduced_contract_but_required_for_gate6_mode() -> None:
    module = _load_module()
    request_hashes = tuple(
        str(row["truth_request_sha256"]) for row in _requests()
    )

    reduced = module.audit_ppo_targets(
        mode="reduced",
        request_sha256=request_hashes,
        ppo_target_rows=None,
    )

    assert reduced == {
        "mode": "reduced",
        "ppo_targets_required": False,
        "ppo_target_count": 0,
    }
    with pytest.raises(
        module.G2InputContractError,
        match="gate6_ppo_targets_missing",
    ):
        module.audit_ppo_targets(
            mode="gate6",
            request_sha256=request_hashes,
            ppo_target_rows=None,
        )
    rows = [
        {
            "schema_version": "g2-gate6-ppo-target/v1",
            "truth_request_sha256": request_sha256,
            "target_sha256": hashlib.sha256(
                f"target:{request_sha256}".encode("utf-8")
            ).hexdigest(),
        }
        for request_sha256 in request_hashes
    ]
    assert module.audit_ppo_targets(
        mode="gate6",
        request_sha256=request_hashes,
        ppo_target_rows=rows,
    )["ppo_target_count"] == 129
