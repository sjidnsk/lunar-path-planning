from __future__ import annotations

import copy
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import struct
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUTS_PATH = REPO_ROOT / "scripts" / "xunce_mid_dual_g2_inputs.py"
RUNNER_PATH = REPO_ROOT / "scripts" / "run_xunce_mid_dual_g2_planning_time.py"
CONFIG_V1_PATH = (
    REPO_ROOT / "configs" / "xunce_mid_dual_g2_planning_time_v1.json"
)
CONFIG_V2_PATH = (
    REPO_ROOT / "configs" / "xunce_mid_dual_g2_planning_time_v2.json"
)
PLATFORMS = ("wheel", "legged", "hopper")
SCALES = ("standard", "kilometer")
REACHABLE_CLASSES = ("normal_reachable", "hard_reachable")
SHA_A = hashlib.sha256(b"a").hexdigest()
SHA_B = hashlib.sha256(b"b").hexdigest()
_PROVIDER_SOURCE_CLOSURE_CACHE: dict[str, object] | None = None


def _load(path: Path, name: str):
    assert path.is_file()
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _inputs():
    return _load(INPUTS_PATH, "xunce_mid_dual_g2_inputs_r3_tested")


def _runner():
    return _load(RUNNER_PATH, "run_xunce_mid_dual_g2_r3_tested")


def _canonical(value: object) -> bytes:
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


def _canonical_jsonl(rows: list[dict[str, object]]) -> bytes:
    return b"".join(_canonical(row) + b"\n" for row in rows)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _provider_source_closure() -> dict[str, object]:
    global _PROVIDER_SOURCE_CLOSURE_CACHE
    if _PROVIDER_SOURCE_CLOSURE_CACHE is None:
        _PROVIDER_SOURCE_CLOSURE_CACHE = (
            _inputs().capture_path_planner_runtime_source_closure()
        )
    return copy.deepcopy(_PROVIDER_SOURCE_CLOSURE_CACHE)


def _word(value: float) -> str:
    return struct.pack(">d", value).hex()


def _blind_contract() -> dict[str, object]:
    core = {
        "schema_version": "xunce-mid-dual-g2-producer-schema-contract/v1",
        "provider_blind_schema_version": "producer-r3-test-fixture/v1",
        "provider_blind_exact_keys": [
            "execution_payload",
            "platform",
            "provider_request_identity_sha256",
            "request_id",
            "scale",
            "schema_version",
        ],
    }
    return {
        **core,
        "contract_sha256": _domain_hash(
            "xunce-mid-dual-g2-producer-schema-contract/v1",
            _canonical(core),
        ),
    }


def _blind_request(
    *,
    platform: str = "wheel",
    scale: str = "standard",
    index: int = 0,
) -> dict[str, object]:
    return {
        "schema_version": "producer-r3-test-fixture/v1",
        "request_id": f"r3-{platform}-{scale}-{index:03d}",
        "platform": platform,
        "scale": scale,
        "provider_request_identity_sha256": hashlib.sha256(
            f"{platform}:{scale}:{index}".encode()
        ).hexdigest(),
        "execution_payload": {"opaque_fixture": f"payload-{index}"},
    }


def _request_matrix() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for platform in PLATFORMS:
        for index in range(43):
            scale = "standard" if index < 33 else "kilometer"
            rows.append(
                _blind_request(
                    platform=platform,
                    scale=scale,
                    index=index,
                )
            )
    return rows


def _probe_selection(
    requests: list[dict[str, object]],
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for platform in PLATFORMS:
        for scale in SCALES:
            candidates = [
                row
                for row in requests
                if row["platform"] == platform and row["scale"] == scale
            ]
            for offset, request_class in enumerate(REACHABLE_CLASSES):
                selected.append(
                    {
                        "request_id": str(candidates[offset]["request_id"]),
                        "platform": platform,
                        "scale": scale,
                        "probe_class": request_class,
                    }
                )
    return selected


def _snapshot() -> dict[str, object]:
    core = {
        "schema_version": "xunce-mid-dual-g2-local-snapshot-normalized/v1",
        "width": 20,
        "height": 20,
        "resolution_m_binary64": _word(0.5),
        "origin_x_m_binary64": _word(0.0),
        "origin_y_m_binary64": _word(0.0),
        "frame_id": "provider-local-map",
        "vertical_datum_id": "producer-r3-local-datum",
        "elevation_m_binary64": [_word(0.0)] * 400,
        "slope_deg_binary64": [_word(0.0)] * 400,
        "cell_class": [0] * 400,
        "known": [1] * 400,
        "confidence_ppm": [1_000_000] * 400,
        "physical_obstacle_cells_written": False,
    }
    return {
        **core,
        "snapshot_sha256": _domain_hash(
            "xunce-mid-dual-g2-local-snapshot-normalized/v1",
            _canonical(core),
        ),
    }


def _hopper_record() -> dict[str, object]:
    core = {
        "schema_version": "hopper-parameter-set-record/v1",
        "parameter_set_id": (
            "hopper_generic_internal_computational_simulation_proxy_midterm_g2g3/v1"
        ),
        "capability_revision": (
            "simulation_proxy_generic_internal_lunar_ballistic/v3"
        ),
        "support_plane_model_id": (
            "hopper_horizontal_same_support_full_envelope_50mm/v1"
        ),
        "support_height_tolerance_m_binary64": _word(0.05),
        "relief_preservation_required": True,
    }
    return {
        **core,
        "record_sha256": _domain_hash(
            "xunce-mid-dual-g2-hopper-record-binding/v1",
            _canonical(core),
        ),
    }


def _exact_hopper_record() -> dict[str, object]:
    return {
        "arc_clearance_margin_m": "0.125",
        "body_envelope_radius_m": "0.375",
        "energy_model": {
            "evaluator_relative_path": "producer/hopper_energy_evaluator.py",
            "evaluator_source_sha256": SHA_A,
            "max_energy_decimal": "1.000000",
            "model_id": "quadratic-normalized-speed/v1",
            "reference_speed_m_s": "2.500",
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
            "evaluator_relative_path": "producer/hopper_stop_evaluator.py",
            "evaluator_source_sha256": SHA_B,
            "max_touchdown_speed_m_s": "2.500",
            "model_id": "touchdown-speed-upper-bound/v1",
        },
    }


def _exact_producer_binding(
    *,
    bundle_root: str = "D:/xunce/inputs/mid_dual/g2-r3-handoff",
    producer_implementation_sha256: str = SHA_A,
    input_contract_sha256: str = SHA_B,
) -> dict[str, object]:
    hopper_sha = hashlib.sha256(
        _canonical(_exact_hopper_record())
    ).hexdigest()
    core = {
        "schema_version": "xunce-mid-dual-g2-producer-binding/v1",
        "bundle_root": bundle_root,
        "producer_implementation_sha256": producer_implementation_sha256,
        "input_contract_sha256": input_contract_sha256,
        "hopper_parameter_record_sha256": hopper_sha,
        "provider_blind_schema_version": "g2-provider-blind-request/v1",
    }
    return {
        **core,
        "binding_sha256": _domain_hash(
            "xunce-mid-dual-g2-producer-binding/v1",
            _canonical(core),
        ),
    }


def _exact_pose(
    x_m: float,
    y_m: float,
    heading_rad: float = 0.0,
) -> dict[str, str]:
    return {
        "heading_rad_hex": heading_rad.hex(),
        "heading_rad_word_hex": _word(heading_rad),
        "x_m_hex": x_m.hex(),
        "x_m_word_hex": _word(x_m),
        "y_m_hex": y_m.hex(),
        "y_m_word_hex": _word(y_m),
    }


def _rehash_exact_blind(row: dict[str, object]) -> dict[str, object]:
    core = {
        key: value
        for key, value in row.items()
        if key not in {"provider_request_id", "provider_request_sha256"}
    }
    provider_sha = _domain_hash(
        "g2-provider-blind-request/v1",
        _canonical(core),
    )
    return {
        **core,
        "provider_request_id": (
            f"g2i-provider-{row['platform_kind']}-{row['scale']}-"
            f"{provider_sha[:20]}"
        ),
        "provider_request_sha256": provider_sha,
    }


def _exact_blind_request(
    *,
    platform: str = "wheel",
    scale: str = "standard",
    index: int = 0,
    binding: dict[str, object] | None = None,
) -> dict[str, object]:
    producer_binding = binding or _exact_producer_binding()
    snapshot_sha = hashlib.sha256(
        f"snapshot:{platform}:{scale}:{index}".encode()
    ).hexdigest()
    snapshot_payload_sha = hashlib.sha256(
        f"snapshot-payload:{platform}:{scale}:{index}".encode()
    ).hexdigest()
    start_x = 1.25
    goal_x = {
        "wheel": 1.75,
        "legged": 1.50,
        "hopper": start_x + float.fromhex("0x1.638e38e38e38fp+0"),
    }[platform]
    start_pose = _exact_pose(start_x, 2.25)
    goal_pose = _exact_pose(goal_x, 2.25)
    state_schema = {
        "wheel": "g2-wheel-node-state/v1",
        "legged": "g2-legged-node-state/v1",
        "hopper": "g2-hopper-node-state/v1",
    }[platform]
    pose_key = "body_pose" if platform == "legged" else "pose"
    node_states = {
        "n:0:0": {
            pose_key: start_pose,
            "schema_version": state_schema,
        },
        "n:1:0": {
            pose_key: goal_pose,
            "schema_version": state_schema,
        },
    }
    node_state_root = _domain_hash(
        "g2-request-node-state-root/v1",
        _canonical(node_states),
    )
    start_safety = {
        "cell_xy": [[2, 4]],
        "safe": True,
        "snapshot_sha256": snapshot_sha,
    }
    goal_safety = {
        "cell_xy": [[3, 4]],
        "safe": True,
        "snapshot_sha256": snapshot_sha,
    }
    start = {
        "endpoint_safety": start_safety,
        "node_id": "n:0:0",
        "node_state": node_states["n:0:0"],
        "pose_binary64_m_rad": start_pose,
        "pose_mm_urad": [round(start_x * 1000), 2250, 0],
    }
    goal = {
        "endpoint_safety": goal_safety,
        "node_id": "n:1:0",
        "node_state": node_states["n:1:0"],
        "pose_binary64_m_rad": goal_pose,
        "pose_mm_urad": [round(goal_x * 1000), 2250, 0],
    }
    vertical_datum = {
        "height_um": 0,
        "reference_sha256": hashlib.sha256(
            f"vertical:{platform}:{scale}:{index}".encode()
        ).hexdigest(),
        "vertical_unit": "um",
    }
    metric_start = {
        key: value for key, value in start.items() if key != "node_state"
    }
    metric_goal = {
        key: value for key, value in goal.items() if key != "node_state"
    }
    metric_problem = {
        "frame_id": "g2-local-metric-frame-mm/v1",
        "goal": metric_goal,
        "local_proxy_difficulty_mode": "nominal",
        "node_state_root_sha256": node_state_root,
        "node_states": node_states,
        "provider_local_snapshot_sha256": snapshot_sha,
        "schema_version": "g2-metric-planning-problem/v3",
        "start": metric_start,
        "vertical_datum": vertical_datum,
    }
    action_envelope = {
        "actions": [
            {
                "action_id": (
                    f"g2i-action-{platform}-{index:03d}"
                ),
            }
        ],
        "platform_kind": platform,
        "schema_version": "g2-request-action-envelope/v2",
    }
    objective = {
        "completion_norm": "L2",
        "kind": "distance-only-complete-L2/v1",
        "resource_weight_milli": 0,
    }
    profile_sha = (
        str(producer_binding["hopper_parameter_record_sha256"])
        if platform == "hopper"
        else hashlib.sha256(
            f"profile:{platform}".encode()
        ).hexdigest()
    )
    core: dict[str, object] = {
        "action_envelope": action_envelope,
        "action_envelope_sha256": _domain_hash(
            "g2-request-action-envelope/v2",
            _canonical(action_envelope),
        ),
        "execution_graph": {
            "node_state_root_sha256": node_state_root,
            "node_states": node_states,
            "nodes": ["n:0:0", "n:1:0"],
            "schema_version": "g2-provider-execution-topology/v1",
        },
        "frame_id": "g2-local-metric-frame-mm/v1",
        "goal": goal,
        "metric_problem": metric_problem,
        "metric_problem_sha256": _domain_hash(
            "g2-request-metric-problem/v3",
            _canonical(metric_problem),
        ),
        "objective": objective,
        "objective_sha256": _domain_hash(
            "g2-request-objective/v1",
            _canonical(objective),
        ),
        "platform_kind": platform,
        "producer_implementation_sha256": producer_binding[
            "producer_implementation_sha256"
        ],
        "profile_or_parameter_record_sha256": profile_sha,
        "provider_local_snapshot_payload_sha256": snapshot_payload_sha,
        "provider_local_snapshot_ref": (
            f"terrain/provider-local/{snapshot_sha}.json"
        ),
        "provider_local_snapshot_sha256": snapshot_sha,
        "resource_budget": {
            "final_target_runtime_ms": 1000,
            "max_path_primitives": 128,
            "midterm_max_runtime_ms": 2000,
        },
        "scale": scale,
        "schema_version": "g2-provider-blind-request/v1",
        "start": start,
        "terrain_geometry_sha256": hashlib.sha256(
            f"geometry:{platform}:{scale}:{index}".encode()
        ).hexdigest(),
        "terrain_sha256": hashlib.sha256(
            f"terrain:{platform}:{scale}:{index}".encode()
        ).hexdigest(),
        "vertical_datum": vertical_datum,
    }
    if platform == "hopper":
        core["hopper_parameter_record"] = _exact_hopper_record()
    return _rehash_exact_blind(core)


def _exact_sidecar(
    request: dict[str, object],
    *,
    difficulty_class: str = "reachable",
    request_hop_count: int = 1,
) -> dict[str, object]:
    reachable = difficulty_class in {"reachable", "hard_reachable"}
    truth = {
        "action_envelope_sha256": request["action_envelope_sha256"],
        "difficulty_class": difficulty_class,
        "goal": request["goal"],
        "metric_problem_sha256": request["metric_problem_sha256"],
        "objective_sha256": request["objective_sha256"],
        "oracle_reachable": reachable,
        "platform_kind": request["platform_kind"],
        "producer_implementation_sha256": request[
            "producer_implementation_sha256"
        ],
        "profile_or_parameter_record_sha256": request[
            "profile_or_parameter_record_sha256"
        ],
        "provider_local_snapshot_payload_sha256": request[
            "provider_local_snapshot_payload_sha256"
        ],
        "provider_local_snapshot_sha256": request[
            "provider_local_snapshot_sha256"
        ],
        "request_hop_count": request_hop_count,
        "request_id": (
            f"g2i-req-{request['platform_kind']}-{request['scale']}-"
            f"{request['provider_request_sha256'][:20]}"
        ),
        "resource_budget": request["resource_budget"],
        "scale": request["scale"],
        "schema_version": "g2-truth-request/v4",
        "start": request["start"],
        "terrain_geometry_sha256": request["terrain_geometry_sha256"],
        "terrain_sha256": request["terrain_sha256"],
        "truth_request_sha256": hashlib.sha256(
            (
                f"truth:{request['provider_request_sha256']}:"
                f"{difficulty_class}"
            ).encode()
        ).hexdigest(),
    }
    return {
        "provider_request_id": request["provider_request_id"],
        "provider_request_sha256": request["provider_request_sha256"],
        "schema_version": "g2-truth-request-sidecar/v1",
        "truth_request": truth,
    }


def _exact_snapshot(
    *,
    platform: str = "wheel",
) -> dict[str, object]:
    zeros = [[0] * 20 for _ in range(20)]
    ones = [[1] * 20 for _ in range(20)]
    confidence = [[1_000_000] * 20 for _ in range(20)]
    input_elevation_sha = _domain_hash(
        "g2-provider-local-elevation-um/v1",
        _canonical(zeros),
    )
    relief_sha = _domain_hash(
        "g2-provider-local-relative-relief-um/v1",
        _canonical(zeros),
    )
    witness_core = {
        "operations": [],
        "physical_obstacle_cells_written": False,
        "source_kind": "synthetic_terrain_obstacle_proxy/v1",
    }
    witness = {
        **witness_core,
        "semantic_audit_sha256": _domain_hash(
            "g2-provider-local-proxy-semantic-audit/v1",
            _canonical(witness_core),
            _canonical(zeros),
            _canonical(zeros),
        ),
    }
    translation_core = {
        "anchor_cell_xy": [0, 0],
        "anchor_node_id": "n:0:0",
        "delta_z_um": 0,
        "input_anchor_elevation_um": 0,
        "input_elevation_sha256": input_elevation_sha,
        "input_relief_sha256": relief_sha,
        "input_snapshot_sha256": SHA_A,
        "output_elevation_sha256": input_elevation_sha,
        "output_relief_sha256": relief_sha,
        "semantic_kind": "uniform-vertical-translation/v1",
    }
    translation = {
        **translation_core,
        "translation_sha256": _domain_hash(
            "g2-provider-uniform-vertical-translation/v1",
            _canonical(translation_core),
        ),
    }
    core = {
        "arrays": {
            "confidence_ppm": confidence,
            "elevation_um": zeros,
            "hard_obstacle": zeros,
            "known": ones,
            "slope_cdeg": zeros,
            "traversable": ones,
        },
        "frame_id": "g2-local-metric-frame-mm/v1",
        "origin_mm": [0, 0],
        "physical_obstacle_cells_written": False,
        "platform_kind": platform,
        "proxy_modification_witness": witness,
        "relief_preservation_sha256": relief_sha,
        "resolution_mm": 500,
        "schema_version": "g2-provider-local-terrain-snapshot/v1",
        "shape_height_width": [20, 20],
        "source_kind": "procedural_fine_height/v1",
        "vertical_translation": translation,
    }
    return {
        **core,
        "snapshot_sha256": _domain_hash(
            "g2-provider-local-terrain-snapshot/v1",
            _canonical(core),
        ),
    }


def _exact_support_binding(
    snapshot: dict[str, object],
    record: dict[str, object],
) -> dict[str, object]:
    support_core = {
        "H_ref_m_hex": (0.0).hex(),
        "H_ref_m_word_hex": _word(0.0),
        "H_ref_um": 0,
        "anchor_node_id": "n:0:0",
        "horizontal": True,
        "normal": [0, 0, 1],
        "schema_version": "g2-horizontal-support-plane/v1",
        "snapshot_sha256": snapshot["snapshot_sha256"],
    }
    support_plane = {
        **support_core,
        "support_plane_sha256": _domain_hash(
            "g2-horizontal-support-plane/v1",
            _canonical(support_core),
        ),
    }
    return {
        "parameter_record_sha256": hashlib.sha256(
            _canonical(record)
        ).hexdigest(),
        "provider_local_snapshot_sha256": snapshot["snapshot_sha256"],
        "relief_preservation_sha256": snapshot[
            "relief_preservation_sha256"
        ],
        "support_plane": support_plane,
    }


def _activation_snapshot(
    *,
    platform: str,
    unique_index: int,
) -> dict[str, object]:
    snapshot = copy.deepcopy(_exact_snapshot(platform=platform))
    translation = snapshot["vertical_translation"]
    translation["input_snapshot_sha256"] = _sha(
        f"activation-input-snapshot:{platform}:{unique_index}"
    )
    translation_core = {
        key: value
        for key, value in translation.items()
        if key != "translation_sha256"
    }
    translation["translation_sha256"] = _domain_hash(
        "g2-provider-uniform-vertical-translation/v1",
        _canonical(translation_core),
    )
    core = {
        key: value
        for key, value in snapshot.items()
        if key != "snapshot_sha256"
    }
    snapshot["snapshot_sha256"] = _domain_hash(
        "g2-provider-local-terrain-snapshot/v1",
        _canonical(core),
    )
    return snapshot


def _bind_activation_snapshot(
    request: dict[str, object],
    snapshot: dict[str, object],
    payload: bytes,
) -> dict[str, object]:
    row = copy.deepcopy(request)
    snapshot_sha = str(snapshot["snapshot_sha256"])
    payload_sha = hashlib.sha256(payload).hexdigest()
    row["provider_local_snapshot_sha256"] = snapshot_sha
    row["provider_local_snapshot_payload_sha256"] = payload_sha
    row["provider_local_snapshot_ref"] = (
        f"terrain/provider-local/{snapshot_sha}.json"
    )
    row["start"]["endpoint_safety"]["snapshot_sha256"] = snapshot_sha
    row["goal"]["endpoint_safety"]["snapshot_sha256"] = snapshot_sha
    metric = row["metric_problem"]
    metric["provider_local_snapshot_sha256"] = snapshot_sha
    metric["start"]["endpoint_safety"]["snapshot_sha256"] = snapshot_sha
    metric["goal"]["endpoint_safety"]["snapshot_sha256"] = snapshot_sha
    row["metric_problem_sha256"] = _domain_hash(
        "g2-request-metric-problem/v3",
        _canonical(metric),
    )
    return _rehash_exact_blind(row)


def _activation_fixture() -> dict[str, object]:
    binding = _exact_producer_binding(
        bundle_root="D:/xunce/inputs/mid_dual/g2-r3-final-candidate",
    )
    requests: list[dict[str, object]] = []
    sidecars: list[dict[str, object]] = []
    snapshot_payloads: dict[str, bytes] = {}
    for platform_index, platform in enumerate(PLATFORMS):
        for index in range(43):
            scale = "standard" if index < 33 else "kilometer"
            snapshot = _activation_snapshot(
                platform=platform,
                unique_index=platform_index * 43 + index,
            )
            snapshot_payload = _canonical(snapshot) + b"\n"
            request = _bind_activation_snapshot(
                _exact_blind_request(
                    platform=platform,
                    scale=scale,
                    index=index,
                    binding=binding,
                ),
                snapshot,
                snapshot_payload,
            )
            if index < 23 or 33 <= index < 39:
                difficulty = "reachable"
            elif 23 <= index < 30 or 39 <= index < 41:
                difficulty = "hard_reachable"
            else:
                difficulty = "unreachable"
            requests.append(request)
            sidecars.append(
                _exact_sidecar(
                    request,
                    difficulty_class=difficulty,
                    request_hop_count=0 if difficulty == "unreachable" else 1,
                )
            )
            snapshot_payloads[str(snapshot["snapshot_sha256"])] = (
                snapshot_payload
            )
    blind_payload = _canonical_jsonl(requests)
    sidecar_payload = _canonical_jsonl(sidecars)
    snapshot_index = [
        {
            "snapshot_sha256": snapshot_sha,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
        }
        for snapshot_sha, payload in sorted(snapshot_payloads.items())
    ]
    candidate = {
        "schema_version": "xunce-mid-dual-g2-r3-final-candidate/v1",
        "fixture_only": False,
        "formal_evidence_eligible": False,
        "formal_ineligibility_reason": "awaiting_artifact_bound_approval",
        "candidate_id": f"g2t2-candidate-{SHA_B[:24]}",
        "input_set_id": f"g2t2-candidate-{SHA_B[:24]}",
        "producer_binding_sha256": binding["binding_sha256"],
        "producer_manifest_file_sha256": _sha("producer-manifest"),
        "manifest_core_sha256": _sha("manifest-core"),
        "producer_payload_root_sha256": _sha("producer-payload-root"),
        "provider_blind_requests_file_sha256": hashlib.sha256(
            blind_payload
        ).hexdigest(),
        "truth_request_sidecar_file_sha256": hashlib.sha256(
            sidecar_payload
        ).hexdigest(),
        "source_attestations_file_sha256": _sha("source-attestations-file"),
        "source_attestation_sha256": _sha("source-attestation"),
        "provider_local_snapshot_index_root_sha256": _domain_hash(
            "g2-provider-local-snapshot-index-root/v1",
            _canonical(snapshot_index),
        ),
        "request_graph_archive_root_sha256": _sha("request-graph-root"),
        "hopper_parameter_record_sha256": binding[
            "hopper_parameter_record_sha256"
        ],
        "producer_repeatability_audit_sha256": _sha(
            "producer-repeatability-audit"
        ),
        "producer_repeatability_passed": True,
        "request_count": 129,
    }
    return {
        "binding": binding,
        "candidate": candidate,
        "requests": requests,
        "sidecars": sidecars,
        "blind_payload": blind_payload,
        "sidecar_payload": sidecar_payload,
        "snapshot_payloads": snapshot_payloads,
        "hopper_parameter_record_payload": _canonical(
            _exact_hopper_record()
        ),
        "provider_source_closure": _provider_source_closure(),
        "consumer_source_closure": (
            _inputs().capture_r3_consumer_source_closure()
        ),
    }


def _approval_for_target(
    target: dict[str, object],
) -> dict[str, object]:
    return {
        "schema_version": (
            "xunce-mid-dual-g2-artifact-bound-approval/v2"
        ),
        "approval_id": "mid-dual-g2-r3-artifact-approval/test",
        "decision": "approved",
        "formal_evidence_eligible": True,
        "blockers": [],
        "approval_target": target,
        "approval_target_sha256": _domain_hash(
            "xunce-mid-dual-g2-r3-approval-target/v1",
            _canonical(target),
        ),
    }


def _seal_activation_fixture(tmp_path: Path) -> dict[str, object]:
    inputs = _inputs()
    fixture = _activation_fixture()
    output_root = tmp_path / "execution"
    prepared = inputs.prepare_r3_execution_data(
        output_root=output_root,
        **fixture,
    )
    target = inputs.build_r3_approval_target(prepared)
    target_core = {
        key: value
        for key, value in target.items()
        if key != "approval_target_sha256"
    }
    approval = _approval_for_target(target_core)
    approval_path = tmp_path / "artifact-bound-o2-approval-v2.json"
    approval_path.write_bytes(_canonical(approval) + b"\n")
    manifest = inputs.seal_r3_execution_bundle(
        output_root=output_root,
        prepared=prepared,
        approval_path=approval_path,
    )
    config = json.loads(CONFIG_V2_PATH.read_text(encoding="utf-8"))
    config["formal_inputs"] = {
        "producer_candidate_bundle": str(
            fixture["binding"]["bundle_root"]
        ),
        "producer_manifest_sha256": prepared[
            "producer_manifest_file_sha256"
        ],
        "provider_blind_requests_sha256": prepared[
            "provider_blind_requests_file_sha256"
        ],
        "truth_request_sidecar_sha256": prepared[
            "truth_request_sidecar_file_sha256"
        ],
        "hopper_parameter_record_sha256": prepared[
            "hopper_parameter_record_sha256"
        ],
        "execution_bundle": output_root.as_posix(),
        "execution_manifest_sha256": hashlib.sha256(
            (output_root / "manifest.json").read_bytes()
        ).hexdigest(),
        "artifact_bound_approval": approval_path.as_posix(),
        "artifact_bound_approval_sha256": hashlib.sha256(
            approval_path.read_bytes()
        ).hexdigest(),
    }
    config["producer_binding"] = fixture["binding"]
    config["activation_binding"] = {
        "candidate_id": prepared["candidate_id"],
        "input_set_id": prepared["input_set_id"],
        "manifest_core_sha256": fixture["candidate"][
            "manifest_core_sha256"
        ],
        "producer_payload_root_sha256": prepared[
            "producer_payload_root_sha256"
        ],
        "producer_repeatability_audit_sha256": prepared[
            "producer_repeatability_audit_sha256"
        ],
        "provider_runtime_source_closure_sha256": prepared[
            "provider_runtime_source_closure_sha256"
        ],
        "consumer_source_closure_sha256": prepared[
            "consumer_source_closure_sha256"
        ],
        "consumer_activation_contract_sha256": prepared[
            "consumer_activation_contract_sha256"
        ],
        "resource_policy_root_sha256": prepared[
            "resource_policy_root_sha256"
        ],
        "execution_request_root_sha256": prepared[
            "execution_request_root_sha256"
        ],
        "p03_probe_selection_sha256": prepared[
            "p03_probe_selection_sha256"
        ],
        "formal_schedule_sha256": prepared["formal_schedule_sha256"],
        "execution_data_root_sha256": prepared[
            "execution_data_root_sha256"
        ],
        "approval_target_sha256": target["approval_target_sha256"],
    }
    config["default_readiness"] = {
        "status": "ready",
        "formal_evidence_eligible": True,
        "blockers": [],
    }
    config_path = tmp_path / "xunce_mid_dual_g2_planning_time_v2.json"
    config_path.write_bytes(_canonical(config) + b"\n")
    inputs.validate_r3_activation_config(config, require_resolved=True)
    return {
        **fixture,
        "prepared": prepared,
        "target": target,
        "approval": approval,
        "approval_path": approval_path,
        "manifest": manifest,
        "output_root": output_root,
        "config": config,
        "config_path": config_path,
    }


def test_r3_consumer_accepts_only_provider_blind_request_schema() -> None:
    module = _inputs()
    row = _blind_request()
    contract = _blind_contract()

    assert module.validate_r3_provider_blind_request(row, contract) == row

    extra = {**row, "unknown": 1}
    with pytest.raises(module.G2InputContractError, match="r3_blind_schema"):
        module.validate_r3_provider_blind_request(extra, contract)

    leak = copy.deepcopy(row)
    leak["execution_payload"]["oracle_reachable"] = True
    with pytest.raises(
        module.G2InputContractError,
        match="g2_provider_truth_leak_v3",
    ):
        module.validate_r3_provider_blind_request(leak, contract)


def test_r3_provider_worker_cannot_reach_truth_sidecar() -> None:
    module = _inputs()
    truth_path = "D:/xunce/inputs/mid_dual/g2-r3/truth-sidecar.jsonl"
    spec = module.build_r3_provider_worker_spec(
        _blind_request(),
        producer_schema_contract=_blind_contract(),
        terrain_payload=b"provider-local-snapshot",
        environment={"OMP_NUM_THREADS": "1"},
    )

    module.validate_r3_provider_worker_isolation(
        spec,
        truth_sidecar_path=truth_path,
    )
    assert truth_path.encode() not in pickle.dumps(spec, protocol=5)

    contaminated = copy.deepcopy(spec)
    contaminated["environment"]["G2_TRUTH_SIDECAR"] = truth_path
    with pytest.raises(
        module.G2InputContractError,
        match="g2_provider_truth_leak_v3",
    ):
        module.validate_r3_provider_worker_isolation(
            contaminated,
            truth_sidecar_path=truth_path,
        )


def test_r3_projection_uses_binary64_pose_not_node_or_macro_scale() -> None:
    module = _inputs()
    words = {
        "x_m": _word(1.25),
        "y_m": _word(2.5),
        "heading_rad": _word(0.0),
    }
    projected = module.project_r3_canonical_pose(words)

    assert projected["binary64_words"] == words
    assert projected["pose"] == {
        "x_m": 1.25,
        "y_m": 2.5,
        "heading_rad": 0.0,
    }
    with pytest.raises(
        module.G2InputContractError,
        match="r3_canonical_pose_schema",
    ):
        module.project_r3_canonical_pose(
            {**words, "node_id": "n:0:0"},
        )


def test_r3_hopper_one_ulp_witness_is_exact_not_general_tolerance() -> None:
    module = _inputs()
    start = 1.25
    modeled_range = float.fromhex("0x1.638e38e38e38fp+0")
    goal = float.fromhex("0x1.51c71c71c71c8p+1")
    observed_delta = goal - start
    witness = {
        "start_x_binary64": _word(start),
        "goal_x_binary64": _word(goal),
        "modeled_range_binary64": _word(modeled_range),
        "node_l2_binary64": _word(observed_delta),
        "ulp_distance": 1,
        "cost_milli": 1389,
    }

    assert module.validate_r3_hopper_one_ulp_witness(witness) == witness
    drifted = {**witness, "ulp_distance": 2}
    with pytest.raises(
        module.G2InputContractError,
        match="r3_hopper_ulp_witness",
    ):
        module.validate_r3_hopper_one_ulp_witness(drifted)


def test_r3_all_scales_use_materialized_local_snapshot_without_resampling() -> None:
    module = _inputs()
    source = _snapshot()

    standard = module.project_r3_local_snapshot(source, scale="standard")
    kilometer = module.project_r3_local_snapshot(source, scale="kilometer")

    assert standard["shape"] == (20, 20)
    assert standard["resolution_m"] == 0.5
    assert standard["source_snapshot_sha256"] == source["snapshot_sha256"]
    assert kilometer["shape"] == (20, 20)
    assert kilometer["resolution_m"] == 0.5
    assert kilometer["projection_sha256"] == standard["projection_sha256"]


def test_r3_hopper_missing_record_is_blocked_before_provider() -> None:
    module = _inputs()
    with pytest.raises(
        module.G2InputContractError,
        match="g2_hopper_parameter_record_missing",
    ):
        module.validate_r3_hopper_execution_binding(
            None,
            {
                "parameter_record_sha256": SHA_A,
                "support_plane_model_id": (
                    "hopper_horizontal_same_support_full_envelope_50mm/v1"
                ),
                "support_reference_height_m_binary64": _word(0.0),
                "required_cells_sha256": SHA_B,
                "relief_preservation_required": True,
            },
        )


def test_r3_hopper_support_datum_relief_and_record_exact_join() -> None:
    module = _inputs()
    record = _hopper_record()
    binding = {
        "parameter_record_sha256": record["record_sha256"],
        "support_plane_model_id": record["support_plane_model_id"],
        "support_reference_height_m_binary64": _word(0.0),
        "required_cells_sha256": SHA_B,
        "relief_preservation_required": True,
    }

    assert module.validate_r3_hopper_execution_binding(
        record,
        binding,
    )["parameter_record_sha256"] == record["record_sha256"]

    changed = {**binding, "relief_preservation_required": False}
    with pytest.raises(
        module.G2InputContractError,
        match="g2_hopper_parameter_record_mismatch",
    ):
        module.validate_r3_hopper_execution_binding(record, changed)


def test_r3_platform_specific_resource_policy_is_hash_bound() -> None:
    module = _inputs()
    policies = {
        platform: module.r3_provider_resource_policy(platform)
        for platform in PLATFORMS
    }

    assert policies["wheel"]["work_budget_basis"] == (
        "wheel_explicit_corridor_sqp/v1"
    )
    assert policies["legged"]["max_route_states"] == 5
    assert policies["hopper"]["max_route_states"] == 2
    for policy in policies.values():
        assert module.validate_r3_provider_resource_policy(policy) == policy

    changed = {**policies["hopper"], "max_route_states": 3}
    with pytest.raises(
        module.G2InputContractError,
        match="g2_provider_resource_policy",
    ):
        module.validate_r3_provider_resource_policy(changed)


def test_r3_legged_graph_hop_maps_to_four_provider_primitives() -> None:
    module = _inputs()
    legged = module.r3_provider_resource_policy("legged")
    hopper = module.r3_provider_resource_policy("hopper")

    assert legged["graph_hops"] == 1
    assert legged["provider_primitives_per_graph_hop"] == 4
    assert legged["max_route_states"] == 5
    assert hopper["graph_hops"] == 1
    assert hopper["provider_primitives_per_graph_hop"] == 1
    assert hopper["max_route_states"] == 2


def test_r3_p03_has_exactly_12_plus_12_calls() -> None:
    module = _runner()
    requests = _request_matrix()
    schedules = module.build_r3_nonformal_schedules(
        requests,
        producer_schema_contract=_blind_contract(),
        probe_selection=_probe_selection(requests),
    )

    assert len(schedules["cold_start"]) == 3
    assert len(schedules["warmup"]) == 30
    assert len(schedules["worker_one"]) == 12
    assert len(schedules["worker_four"]) == 12
    assert {
        call["request_id"] for call in schedules["worker_one"]
    } == {
        call["request_id"] for call in schedules["worker_four"]
    }


def test_r3_p03_covers_six_strata_and_two_reachable_classes() -> None:
    module = _runner()
    requests = _request_matrix()
    schedules = module.build_r3_nonformal_schedules(
        requests,
        producer_schema_contract=_blind_contract(),
        probe_selection=_probe_selection(requests),
    )
    calls = schedules["worker_one"]

    assert {
        (call["platform"], call["scale"], call["probe_class"])
        for call in calls
    } == {
        (platform, scale, probe_class)
        for platform in PLATFORMS
        for scale in SCALES
        for probe_class in REACHABLE_CLASSES
    }
    assert all(
        "probe_class" not in call["provider_worker_request"]
        for call in calls
    )


def test_r3_formal_schedule_is_exactly_129_times_five() -> None:
    module = _runner()
    schedule = module.build_r3_formal_schedule(
        "g2-r3-test-input",
        _request_matrix(),
        producer_schema_contract=_blind_contract(),
    )

    assert schedule["schema_version"] == (
        "xunce-mid-dual-g2-formal-schedule/v2"
    )
    assert len(schedule["calls"]) == 645
    assert len({call["call_id"] for call in schedule["calls"]}) == 645
    assert {
        call["repeat_index"] for call in schedule["calls"]
    } == set(range(5))


def test_r3_formal_success_and_failure_outcomes_are_exact_and_mutually_exclusive() -> None:
    module = _runner()
    success = {
        "outcome_type": "success",
        "request_id": "r3-wheel-standard-000",
        "platform_kind": "wheel",
        "route": [{"primitive_id": "p0"}],
        "validation": {
            "validator_id": "wheel-route-l2/v1",
            "level": "L2",
            "passed": True,
            "checks": ["route_complete"],
        },
    }
    failure = {
        "outcome_type": "failure",
        "request_id": "r3-wheel-standard-000",
        "platform_kind": "wheel",
        "category": "search_exhausted",
        "reason_code": "wheel_no_complete_route",
        "stage": "search",
        "checks": ["frontier_empty"],
    }

    assert module.canonicalize_r3_provider_outcome(success)[
        "provider_success"
    ] is True
    assert module.canonicalize_r3_provider_outcome(failure)[
        "provider_success"
    ] is False
    with pytest.raises(module.G2Blocked, match="g2_r3_outcome_schema"):
        module.canonicalize_r3_provider_outcome(
            {**success, "reason_code": "must-not-coexist"},
        )


def test_r3_scorer_opens_truth_only_after_provider_batch() -> None:
    module = _runner()
    opened = 0

    def load_truth() -> list[dict[str, object]]:
        nonlocal opened
        opened += 1
        return [{"request_id": "truth-row"}]

    with pytest.raises(module.G2Blocked, match="g2_r3_provider_batch_incomplete"):
        module.open_r3_truth_after_provider_batch(
            [{"call_id": "only-one"}],
            expected_call_count=645,
            truth_loader=load_truth,
        )
    assert opened == 0

    rows = [{"call_id": f"call-{index:03d}"} for index in range(645)]
    assert module.open_r3_truth_after_provider_batch(
        rows,
        expected_call_count=645,
        truth_loader=load_truth,
    ) == [{"request_id": "truth-row"}]
    assert opened == 1


def test_r3_six_strata_summary_recomputes_from_645_raw_rows() -> None:
    module = _runner()
    schedule = module.build_r3_formal_schedule(
        "g2-r3-test-input",
        _request_matrix(),
        producer_schema_contract=_blind_contract(),
    )
    truth_by_request: dict[str, dict[str, object]] = {}
    formal_rows: list[dict[str, object]] = []
    for call in schedule["calls"]:
        request_index = int(call["request_index"])
        reachable = request_index not in {30, 31, 32, 41, 42}
        truth_by_request[str(call["request_id"])] = {
            "request_id": call["request_id"],
            "platform": call["platform"],
            "scale": call["scale"],
            "oracle_reachable": reachable,
        }
        formal_rows.append(
            {
                "call_id": call["call_id"],
                "request_id": call["request_id"],
                "platform": call["platform"],
                "scale": call["scale"],
                "repeat_index": call["repeat_index"],
                "provider_success": reachable,
                "route_l2_valid": reachable,
                "elapsed_ms": 10.0,
            }
        )

    summary = module.recompute_r3_six_strata_summary(
        formal_rows,
        list(truth_by_request.values()),
    )

    assert summary["formal_call_count"] == 645
    assert summary["unique_request_count"] == 129
    assert summary["reachable_unique_request_count"] == 114
    assert summary["unreachable_unique_request_count"] == 15
    assert len(summary["strata"]) == 6
    assert sum(
        cell["reachable_expected_call_count"]
        for cell in summary["strata"].values()
    ) == 570


def test_r3_old_v1_bundle_and_approval_are_read_only_ineligible() -> None:
    module = _runner()
    old_manifest = {
        "schema_version": "xunce-mid-dual-g2-execution-bundle/v1",
        "approval_schema_version": (
            "xunce-mid-dual-g2-artifact-bound-approval/v1"
        ),
    }
    with pytest.raises(module.G2Blocked, match="g2_r3_legacy_bundle_ineligible"):
        module.validate_r3_execution_manifest(old_manifest)


def test_r3_producer_reachable_hops_fit_exact_consumer_resource_policy() -> None:
    fixture_value = os.environ.get("XUNCE_G2_PRODUCER_R3_FIXTURE_ROOT")
    if fixture_value is None:
        pytest.skip("XUNCE_G2_PRODUCER_R3_FIXTURE_ROOT is not configured")
    fixture_root = Path(fixture_value)
    request_path = fixture_root / "requests.jsonl"
    sidecar_path = fixture_root / "truth" / "request-sidecar.jsonl"
    assert request_path.is_file()
    assert sidecar_path.is_file()

    requests = [
        json.loads(line)
        for line in request_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    sidecars = [
        json.loads(line)
        for line in sidecar_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(requests) == 129
    assert len(sidecars) == 129
    requests_by_sha = {
        row["provider_request_sha256"]: row for row in requests
    }
    assert len(requests_by_sha) == 129

    module = _inputs()
    checked_counts = {"legged": 0, "hopper": 0}
    mismatch_counts: dict[str, int] = {}
    for sidecar in sidecars:
        assert set(sidecar) == {
            "provider_request_id",
            "provider_request_sha256",
            "schema_version",
            "truth_request",
        }
        assert sidecar["schema_version"] == "g2-truth-request-sidecar/v1"
        provider_request = requests_by_sha[
            sidecar["provider_request_sha256"]
        ]
        assert (
            provider_request["provider_request_id"]
            == sidecar["provider_request_id"]
        )
        truth = sidecar["truth_request"]
        platform = truth["platform_kind"]
        assert provider_request["platform_kind"] == platform
        if (
            platform not in checked_counts
            or truth["difficulty_class"]
            not in {"reachable", "hard_reachable"}
        ):
            continue

        policy = module.r3_provider_resource_policy(platform)
        hop_count = truth["request_hop_count"]
        checked_counts[platform] += 1
        required_primitives = (
            hop_count * policy["provider_primitives_per_graph_hop"]
        )
        required_route_states = required_primitives + 1
        if (
            hop_count != policy["graph_hops"]
            or required_primitives
            != policy["provider_primitives_per_graph_hop"]
            or required_route_states != policy["max_route_states"]
        ):
            mismatch_key = (
                f"{platform}/{truth['difficulty_class']}/"
                f"hops={hop_count}/required_primitives="
                f"{required_primitives}/required_route_states="
                f"{required_route_states}"
            )
            mismatch_counts[mismatch_key] = (
                mismatch_counts.get(mismatch_key, 0) + 1
            )

    assert checked_counts == {"legged": 38, "hopper": 38}
    assert mismatch_counts == {}


def test_r3_exact_blind_schema_identity_and_binding_are_parameterized() -> None:
    module = _inputs()
    binding = _exact_producer_binding()
    row = _exact_blind_request(binding=binding)

    assert module.validate_r3_producer_binding(binding) == binding
    assert module.validate_r3_provider_blind_request(row, binding) == row
    projected = module.project_r3_canonical_pose(
        row["start"]["pose_binary64_m_rad"]
    )
    assert projected["pose"] == {
        "x_m": 1.25,
        "y_m": 2.25,
        "heading_rad": 0.0,
    }

    changed = copy.deepcopy(row)
    changed["producer_implementation_sha256"] = SHA_B
    changed = _rehash_exact_blind(changed)
    with pytest.raises(
        module.G2InputContractError,
        match="r3_producer_binding_mismatch",
    ):
        module.validate_r3_provider_blind_request(changed, binding)

    other_binding = _exact_producer_binding(
        bundle_root="D:/xunce/inputs/mid_dual/g2-r3-other",
        producer_implementation_sha256=SHA_B,
        input_contract_sha256=SHA_A,
    )
    assert other_binding["binding_sha256"] != binding["binding_sha256"]


def test_r3_exact_snapshot_projection_is_scale_invariant_and_hash_bound() -> None:
    module = _inputs()
    snapshot = _exact_snapshot()
    standard = module.project_r3_local_snapshot(
        snapshot,
        scale="standard",
    )
    kilometer = module.project_r3_local_snapshot(
        snapshot,
        scale="kilometer",
    )

    assert standard["shape"] == (20, 20)
    assert standard["resolution_m"] == 0.5
    assert standard["origin_m"] == (0.0, 0.0)
    assert standard["source_snapshot_sha256"] == snapshot["snapshot_sha256"]
    assert kilometer["source_snapshot_sha256"] == snapshot["snapshot_sha256"]
    assert kilometer["projection_sha256"] == standard["projection_sha256"]
    assert standard["elevation_m"].shape == (20, 20)
    assert standard["slope_deg"].shape == (20, 20)


def test_r3_exact_hopper_record_support_plane_and_relief_are_joined() -> None:
    module = _inputs()
    record = _exact_hopper_record()
    snapshot = _exact_snapshot(platform="hopper")
    binding = _exact_support_binding(snapshot, record)

    result = module.validate_r3_hopper_execution_binding(record, binding)
    assert result["parameter_record_sha256"] == hashlib.sha256(
        _canonical(record)
    ).hexdigest()
    assert (
        result["provider_local_snapshot_sha256"]
        == snapshot["snapshot_sha256"]
    )

    changed = copy.deepcopy(binding)
    changed["support_plane"]["H_ref_um"] = 1
    changed["support_plane"] = {
        **changed["support_plane"],
        "support_plane_sha256": _domain_hash(
            "g2-horizontal-support-plane/v1",
            _canonical(
                {
                    key: value
                    for key, value in changed["support_plane"].items()
                    if key != "support_plane_sha256"
                }
            ),
        ),
    }
    with pytest.raises(
        module.G2InputContractError,
        match="G2I_BLOCKED_HOPPER_PARAMETER_RECORD_MISMATCH",
    ):
        module.validate_r3_hopper_execution_binding(record, changed)


def test_r3_parent_static_join_accepts_one_hop_and_binds_resource_identity() -> None:
    module = _inputs()
    binding = _exact_producer_binding()
    legged = _exact_blind_request(
        platform="legged",
        index=1,
        binding=binding,
    )
    hopper = _exact_blind_request(
        platform="hopper",
        index=2,
        binding=binding,
    )
    audit = module.validate_r3_parent_static_hop_resource_join(
        [legged, hopper],
        [
            _exact_sidecar(legged, difficulty_class="reachable"),
            _exact_sidecar(hopper, difficulty_class="hard_reachable"),
        ],
        producer_binding=binding,
    )

    assert audit["status"] == "passed"
    assert audit["request_count"] == 2
    assert audit["reachable_hop_counts"] == {"legged": [1], "hopper": [1]}
    assert len(audit["execution_crosswalk"]) == 2
    by_platform = {
        row["platform_kind"]: row
        for row in audit["execution_crosswalk"]
    }
    assert by_platform["legged"]["resource_policy"]["max_route_states"] == 5
    assert by_platform["hopper"]["resource_policy"]["max_route_states"] == 2
    assert all(
        set(row)
        == {
            "execution_request_sha256",
            "platform_kind",
            "provider_request_id",
            "provider_request_sha256",
            "resource_policy",
            "resource_policy_sha256",
            "scale",
        }
        for row in audit["execution_crosswalk"]
    )


@pytest.mark.parametrize(
    ("platform", "hop_count", "required_primitives", "required_states"),
    [
        ("legged", 8, 32, 33),
        ("hopper", 4, 4, 5),
    ],
)
def test_r3_parent_static_join_rejects_multi_hop_resource_mismatch(
    platform: str,
    hop_count: int,
    required_primitives: int,
    required_states: int,
) -> None:
    module = _inputs()
    binding = _exact_producer_binding()
    request = _exact_blind_request(
        platform=platform,
        binding=binding,
    )
    sidecar = _exact_sidecar(
        request,
        difficulty_class="reachable",
        request_hop_count=hop_count,
    )

    with pytest.raises(
        module.G2InputContractError,
        match=(
            "g2_r3_hop_resource_mismatch.*"
            f"required_primitives={required_primitives}.*"
            f"required_route_states={required_states}"
        ),
    ):
        module.validate_r3_parent_static_hop_resource_join(
            [request],
            [sidecar],
            producer_binding=binding,
        )


def test_r3_exact_schedule_carries_resource_bound_execution_identity() -> None:
    module = _runner()
    binding = _exact_producer_binding()
    requests = [
        _exact_blind_request(
            platform=platform,
            scale="standard" if index < 33 else "kilometer",
            index=index,
            binding=binding,
        )
        for platform in PLATFORMS
        for index in range(43)
    ]
    schedule = module.build_r3_formal_schedule(
        "g2-r3-synthetic-one-hop",
        requests,
        producer_schema_contract=binding,
    )

    assert len(schedule["calls"]) == 645
    assert (
        schedule["producer_binding_sha256"]
        == binding["binding_sha256"]
    )
    assert all(
        call["execution_request_sha256"]
        == call["provider_worker_request"]["execution_request_sha256"]
        for call in schedule["calls"]
    )
    assert all(
        call["resource_policy_sha256"]
        == call["provider_worker_request"]["resource_policy_sha256"]
        for call in schedule["calls"]
    )
    assert {
        call["provider_worker_request"]["resource_policy"][
            "max_route_states"
        ]
        for call in schedule["calls"]
        if call["platform"] == "legged"
    } == {5}
    assert {
        call["provider_worker_request"]["resource_policy"][
            "max_route_states"
        ]
        for call in schedule["calls"]
        if call["platform"] == "hopper"
    } == {2}


def test_r3_v2_config_has_exact_activation_and_phase_contract() -> None:
    module = _inputs()
    assert CONFIG_V2_PATH.is_file()
    config = json.loads(CONFIG_V2_PATH.read_text(encoding="utf-8"))

    validated = module.validate_r3_activation_config(
        config,
        require_resolved=False,
    )

    assert validated["schema_version"] == (
        "xunce-mid-dual-g2-planning-time-config/v2"
    )
    assert validated["runner_id"] == (
        "run_xunce_mid_dual_g2_planning_time/v2"
    )
    assert validated["phase_contract"]["p01"]["provider_call_count"] == 0
    assert validated["phase_contract"]["p02"]["cold_start_count"] == 3
    assert validated["phase_contract"]["p02"]["warmup_count"] == 30
    assert validated["phase_contract"]["p03"]["worker_one_count"] == 12
    assert validated["phase_contract"]["p03"]["worker_four_count"] == 12
    assert validated["phase_contract"]["p04"]["formal_call_count"] == 645
    assert validated["default_readiness"] == {
        "status": "ready",
        "formal_evidence_eligible": True,
        "blockers": [],
    }
    assert "${" not in CONFIG_V2_PATH.read_text(encoding="utf-8")


def test_r3_v2_config_rejects_unresolved_handoff_placeholder() -> None:
    module = _inputs()
    config = json.loads(CONFIG_V2_PATH.read_text(encoding="utf-8"))
    config["formal_inputs"]["producer_candidate_bundle"] = (
        "${FINAL_PRODUCER_BUNDLE_ROOT}"
    )

    with pytest.raises(
        module.G2InputContractError,
        match="g2_r3_config_placeholder_unresolved",
    ):
        module.validate_r3_activation_config(
            config,
            require_resolved=False,
        )


def test_r3_consumer_source_closure_is_computed_from_exact_noncyclic_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _inputs()

    closure = module.capture_r3_consumer_source_closure()

    assert closure["schema_version"] == (
        "xunce-mid-dual-g2-r3-consumer-source-closure/v1"
    )
    assert [row["relative_path"] for row in closure["files"]] == [
        "scripts/run_xunce_mid_dual_g2_planning_time.py",
        "scripts/xunce_artifact_io.py",
        "scripts/xunce_artifact_paths.py",
        "scripts/xunce_mid_dual_artifacts.py",
        "scripts/xunce_mid_dual_contracts.py",
        "scripts/xunce_mid_dual_g2_inputs.py",
    ]
    assert all(
        hashlib.sha256(
            (REPO_ROOT / row["relative_path"]).read_bytes()
        ).hexdigest()
        == row["sha256"]
        for row in closure["files"]
    )
    assert len(closure["consumer_source_closure_sha256"]) == 64
    assert all(
        not row["relative_path"].endswith(
            "xunce_mid_dual_g2_planning_time_v2.json"
        )
        for row in closure["files"]
    )
    for drift_path in (
        str(row["relative_path"]) for row in closure["files"]
    ):
        drifted = module.capture_r3_consumer_source_closure(
            source_reader=lambda path, drift_path=drift_path: (
                path.read_bytes() + b"\nsource-drift"
                if path.as_posix().endswith(drift_path)
                else path.read_bytes()
            )
        )
        assert (
            drifted["consumer_source_closure_sha256"]
            != closure["consumer_source_closure_sha256"]
        )

    runner = _runner()
    sealed = _seal_activation_fixture(tmp_path)
    runtime_inputs = __import__("xunce_mid_dual_g2_inputs")
    drifted_runtime = copy.deepcopy(closure)
    drifted_runtime["consumer_source_closure_sha256"] = SHA_A
    monkeypatch.setattr(
        runtime_inputs,
        "capture_r3_consumer_source_closure",
        lambda: drifted_runtime,
    )
    with pytest.raises(runner.G2Blocked, match="g2_r3_source_closure"):
        runner.read_r3_execution_bundle(
            sealed["output_root"],
            config=sealed["config"],
        )


def test_r3_cli_rejects_v1_config_before_bundle_or_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _runner()

    with pytest.raises(
        module.G2Blocked,
        match="g2_r3_legacy_config_ineligible",
    ):
        module.validate_r3_cli_config_schema(CONFIG_V1_PATH)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("v1 must stop before bundle or Provider execution")

    monkeypatch.setattr(module, "run_g2_r3", forbidden)
    output = io.StringIO()
    with redirect_stdout(output):
        code = module.main(
            [
                "--config",
                str(CONFIG_V1_PATH),
                "--input-bundle",
                "D:/xunce/inputs/mid_dual/g2/legacy-must-not-open",
                "--run-id",
                "r3-v1-terminal-blocked",
                "--mode",
                "preflight",
            ]
        )
    payload = json.loads(output.getvalue())
    assert code == 0
    assert payload["execution_status"] == "complete"
    assert payload["gate_status"] == "blocked"
    assert payload["blocking_reason"] == "g2_r3_legacy_config_ineligible"
    assert payload["formal_row_count"] == 0


def test_r3_cli_rejects_v1_bundle_and_approval_before_provider() -> None:
    runner = _runner()
    inputs = _inputs()
    old_manifest = {
        "schema_version": "xunce-mid-dual-g2-execution-bundle/v1",
        "approval_schema_version": (
            "xunce-mid-dual-g2-artifact-bound-approval/v1"
        ),
    }
    with pytest.raises(
        runner.G2Blocked,
        match="g2_r3_legacy_bundle_ineligible",
    ):
        runner.validate_r3_execution_manifest(old_manifest)
    with pytest.raises(
        inputs.G2InputContractError,
        match="g2_r3_legacy_approval_ineligible",
    ):
        inputs.validate_r3_artifact_bound_approval_v2(
            {
                "schema_version": (
                    "xunce-mid-dual-g2-artifact-bound-approval/v1"
                )
            },
            {},
        )


def test_r3_final_candidate_rejects_fixture_null_id_and_repeatability_pending() -> None:
    module = _inputs()
    fixture = _activation_fixture()
    candidate = fixture["candidate"]
    assert module.validate_r3_final_candidate_metadata(candidate) == candidate

    mutations = (
        {"fixture_only": True},
        {"candidate_id": None},
        {"producer_repeatability_passed": False},
    )
    for mutation in mutations:
        with pytest.raises(
            module.G2InputContractError,
            match="g2_r3_final_candidate_ineligible",
        ):
            module.validate_r3_final_candidate_metadata(
                {**candidate, **mutation}
            )


def test_r3_prepare_execution_data_copies_129_blind_and_sidecar_bytes_exactly(
    tmp_path: Path,
) -> None:
    module = _inputs()
    fixture = _activation_fixture()
    output_root = tmp_path / "execution"

    prepared = module.prepare_r3_execution_data(
        output_root=output_root,
        **fixture,
    )

    assert prepared["request_count"] == 129
    assert prepared["truth_sidecar_count"] == 129
    assert (
        output_root / "provider-blind-requests.jsonl"
    ).read_bytes() == fixture["blind_payload"]
    assert (
        output_root / "truth" / "request-sidecar.jsonl"
    ).read_bytes() == fixture["sidecar_payload"]
    assert not (output_root / "manifest.json").exists()


def test_r3_prepare_execution_data_binds_43_hopper_records_and_three_resource_policies(
    tmp_path: Path,
) -> None:
    module = _inputs()
    fixture = _activation_fixture()

    prepared = module.prepare_r3_execution_data(
        output_root=tmp_path / "execution",
        **fixture,
    )

    assert prepared["hopper_request_count"] == 43
    assert prepared["resource_policy_count"] == 3
    assert len(prepared["resource_policy_root_sha256"]) == 64
    assert len(prepared["execution_request_root_sha256"]) == 64
    assert len(prepared["p03_probe_selection_sha256"]) == 64
    assert len(prepared["formal_schedule_sha256"]) == 64


def test_r3_execution_data_root_is_sorted_complete_and_manifest_free(
    tmp_path: Path,
) -> None:
    module = _inputs()
    fixture = _activation_fixture()
    prepared = module.prepare_r3_execution_data(
        output_root=tmp_path / "execution",
        **fixture,
    )

    indexed = [
        row["relative_path"] for row in prepared["payload_index"]
    ]
    assert indexed == sorted(indexed)
    assert len(indexed) == len(set(indexed))
    assert "manifest.json" not in indexed
    assert all("approval" not in path for path in indexed)
    assert len(prepared["execution_data_root_sha256"]) == 64


def test_r3_approval_target_has_no_manifest_or_approval_hash_cycle(
    tmp_path: Path,
) -> None:
    module = _inputs()
    fixture = _activation_fixture()
    prepared = module.prepare_r3_execution_data(
        output_root=tmp_path / "execution",
        **fixture,
    )

    target = module.build_r3_approval_target(prepared)

    assert "manifest_sha256" not in target
    assert "approval_artifact_sha256" not in target
    assert target["execution_data_root_sha256"] == prepared[
        "execution_data_root_sha256"
    ]
    assert len(target["approval_target_sha256"]) == 64


def test_r3_approval_v2_drift_blocks_manifest_seal(
    tmp_path: Path,
) -> None:
    module = _inputs()
    fixture = _activation_fixture()
    prepared = module.prepare_r3_execution_data(
        output_root=tmp_path / "execution",
        **fixture,
    )
    target = module.build_r3_approval_target(prepared)
    approval = _approval_for_target(
        {
            key: value
            for key, value in target.items()
            if key != "approval_target_sha256"
        }
    )
    assert module.validate_r3_artifact_bound_approval_v2(
        approval,
        target,
    ) == approval

    drifted = copy.deepcopy(approval)
    drifted["approval_target"]["formal_call_count"] = 644
    with pytest.raises(
        module.G2InputContractError,
        match="g2_r3_approval_target_drift",
    ):
        module.validate_r3_artifact_bound_approval_v2(
            drifted,
            target,
        )


def test_r3_manifest_v2_is_written_last_and_revalidates_every_payload(
    tmp_path: Path,
) -> None:
    module = _inputs()
    fixture = _activation_fixture()
    output_root = tmp_path / "execution"
    prepared = module.prepare_r3_execution_data(
        output_root=output_root,
        **fixture,
    )
    target = module.build_r3_approval_target(prepared)
    approval = _approval_for_target(
        {
            key: value
            for key, value in target.items()
            if key != "approval_target_sha256"
        }
    )
    approval_path = tmp_path / "artifact-bound-o2-approval-v2.json"
    approval_path.write_bytes(_canonical(approval) + b"\n")

    manifest = module.seal_r3_execution_bundle(
        output_root=output_root,
        prepared=prepared,
        approval_path=approval_path,
    )

    assert manifest["schema_version"] == (
        "xunce-mid-dual-g2-execution-bundle/v2"
    )
    assert manifest["publication_order"] == "data-first-manifest-last"
    assert manifest["formal_evidence_eligible"] is True
    assert (output_root / "manifest.json").is_file()
    assert module.validate_r3_sealed_execution_bundle(
        output_root,
        approval_path=approval_path,
    )["manifest"] == manifest

    blind_path = output_root / "provider-blind-requests.jsonl"
    blind_path.write_bytes(blind_path.read_bytes() + b" ")
    with pytest.raises(
        module.G2InputContractError,
        match="g2_r3_execution_payload_drift",
    ):
        module.validate_r3_sealed_execution_bundle(
            output_root,
            approval_path=approval_path,
        )


def _r3_timing() -> dict[str, int]:
    return {
        "input_validation_ns": 1,
        "platform_instantiation_ns": 2,
        "search_ns": 3,
        "complete_route_validation_ns": 4,
        "result_assembly_ns": 5,
        "total_ns": 15,
    }


def _r3_success_outcome(call: dict[str, object]) -> dict[str, object]:
    return {
        "outcome_type": "success",
        "request_id": call["request_id"],
        "platform_kind": call["platform"],
        "route": [{"primitive_id": f"p-{call['request_id']}"}],
        "validation": {
            "validator_id": f"{call['platform']}-route-l2/v1",
            "level": "L2",
            "passed": True,
            "checks": ["route_complete"],
        },
    }


def _r3_fake_success_batch(
    runner,
    calls: list[dict[str, object]],
    *,
    max_workers: int,
) -> list[dict[str, object]]:
    return [
        runner.build_r3_provider_diagnostic_row(
            {**call, "_worker_count": max_workers},
            _r3_success_outcome(call),
            _r3_timing(),
        )
        for call in calls
    ]


def _assert_no_truth_derived_worker_fields(value: object) -> None:
    forbidden = {"probe_class", "truth", "difficulty_class", "oracle_reachable"}
    if isinstance(value, dict):
        assert forbidden.isdisjoint(value)
        for child in value.values():
            _assert_no_truth_derived_worker_fields(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _assert_no_truth_derived_worker_fields(child)


def _configure_r3_formal_environment(
    runner,
    sealed: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = copy.deepcopy(sealed["config"])
    policy = config["execution"]["formal_environment_gate"]
    policy["lease_path"] = (tmp_path / "formal-lease.json").as_posix()
    sealed["config_path"].write_bytes(_canonical(config) + b"\n")

    def capture(phase: str, observed_policy: dict[str, object]):
        assert observed_policy == policy
        return {
            "schema_version": (
                runner.FORMAL_ENVIRONMENT_OBSERVATION_SCHEMA_VERSION
            ),
            "phase": phase,
            "captured_utc": f"2026-07-28T00:00:0{phase == 'end'}Z",
            "host": runner.host_platform.node() or "unknown-host",
            "pid": os.getpid(),
            "power_scheme_guid": policy["allowed_power_scheme_guids"][0],
            "power_probe_sha256": SHA_A,
            "thread_variables": policy["required_thread_variables"],
            "sample_window_seconds": policy["sample_window_seconds"],
            "cpu_percent": 1.0,
            "memory_percent": 10.0,
            "memory_available_bytes": (
                policy["limits"]["memory_available_bytes_min"] + 1
            ),
            "disk_busy_percent": 1.0,
            "disk_free_bytes": policy["limits"]["disk_free_bytes_min"] + 1,
            "disk_root": "D:/",
            "competing_processes": [],
            "excluded_process_ids": [os.getpid()],
            "process_inventory_sha256": SHA_B,
        }

    monkeypatch.setattr(
        runner,
        "capture_windows_formal_environment",
        capture,
    )


def test_r3_formal_mode_executes_recoverable_exact_645_and_finalizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    sealed = _seal_activation_fixture(tmp_path)
    _configure_r3_formal_environment(
        runner,
        sealed,
        tmp_path,
        monkeypatch,
    )
    diagnostic_worker_counts: list[int] = []
    formal_tasks: list[dict[str, object]] = []

    def fake_diagnostic_batch(calls, *, max_workers):
        materialized = list(calls)
        diagnostic_worker_counts.append(max_workers)
        return _r3_fake_success_batch(
            runner,
            materialized,
            max_workers=max_workers,
        )

    def fake_formal_task(task):
        materialized = dict(task)
        formal_tasks.append(materialized)
        _assert_no_truth_derived_worker_fields(materialized)
        reachable = int(materialized["request_index"]) not in {
            30,
            31,
            32,
            41,
            42,
        }
        outcome = (
            _r3_success_outcome(materialized)
            if reachable
            else {
                "outcome_type": "failure",
                "request_id": materialized["request_id"],
                "platform_kind": materialized["platform"],
                "category": "search_exhausted",
                "reason_code": "no_complete_route",
                "stage": "search",
                "checks": ["frontier_empty"],
            }
        )
        return runner.build_r3_provider_formal_row(
            {**materialized, "_worker_count": 4},
            outcome,
            _r3_timing(),
        )

    monkeypatch.setattr(
        runner,
        "_default_r3_formal_task_executor",
        fake_formal_task,
        raising=False,
    )
    result = runner.run_g2_r3(
        config_path=sealed["config_path"],
        input_bundle=sealed["output_root"],
        run_id="r3-formal-exact-645-test",
        mode="formal",
        output_base=tmp_path / "runs",
        batch_executor=fake_diagnostic_batch,
    )

    run_root = Path(result["run_root"])
    assert result["execution_status"] == "complete"
    assert result["formal_row_count"] == 645
    assert result["formal_environment_gate_status"] == "passed"
    assert result["accepted_phase_ids"] == ["p01", "p02", "p03", "p04"]
    assert diagnostic_worker_counts == [1, 4, 1, 4]
    assert len(formal_tasks) == 645
    assert all(task["formal_sample"] is True for task in formal_tasks)
    assert len(
        runner.artifact_io.read_jsonl(run_root / "job-state.jsonl")
    ) == 645
    p04_state = next(
        row
        for row in runner.artifact_io.read_jsonl(
            run_root / "phase-state.jsonl"
        )
        if row["phase_id"] == "p04"
    )
    assert len(
        runner.artifact_io.read_jsonl(run_root / p04_state["rows_path"])
    ) == 645
    runner.MidDualRunStore.verify_manifest(run_root)


def test_r3_loader_blocks_live_provider_source_drift_before_provider_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    sealed = _seal_activation_fixture(tmp_path)
    runtime_inputs = __import__("xunce_mid_dual_g2_inputs")
    drifted = copy.deepcopy(sealed["provider_source_closure"])
    drifted["dirty_inventory"] = [{"path": "drift.py", "status": " M"}]
    monkeypatch.setattr(
        runtime_inputs,
        "capture_path_planner_runtime_source_closure",
        lambda: drifted,
    )

    with pytest.raises(runner.G2Blocked, match="g2_r3_source_closure"):
        runner.read_r3_execution_bundle(
            sealed["output_root"],
            config=sealed["config"],
        )


def test_r3_worker_prepare_and_execute_never_receive_probe_or_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    sealed = _seal_activation_fixture(tmp_path)
    bundle = runner.read_r3_execution_bundle(
        sealed["output_root"],
        config=sealed["config"],
    )
    schedules = runner.build_r3_nonformal_schedules(
        bundle["provider_blind_requests"],
        producer_schema_contract=bundle["producer_binding"],
        probe_selection=bundle["p03_probe_selection"],
    )
    parent_call = schedules["worker_one"][0]
    hydrated = runner.hydrate_r3_calls([parent_call], bundle=bundle)
    observed: list[str] = []

    def prepare(task):
        _assert_no_truth_derived_worker_fields(task)
        assert task["_worker_count"] == 1
        observed.append("prepare")
        return dict(task)

    def execute(task):
        _assert_no_truth_derived_worker_fields(task)
        observed.append("execute")
        return runner.build_r3_provider_diagnostic_row(
            task,
            _r3_success_outcome(task),
            _r3_timing(),
        )

    monkeypatch.setattr(runner, "_prepare_r3_provider_task", prepare)
    monkeypatch.setattr(runner, "execute_r3_provider_timed_call", execute)
    raw_rows = runner._default_r3_batch_executor(
        hydrated,
        max_workers=1,
    )
    assert observed == ["prepare", "execute"]
    assert "probe_class" not in raw_rows[0]

    filled = runner._validate_r3_diagnostic_batch(
        raw_rows,
        hydrated,
        worker_count=1,
        probe_classes_by_call_id={
            str(parent_call["call_id"]): str(parent_call["probe_class"])
        },
    )
    assert filled[0]["probe_class"] == parent_call["probe_class"]


def test_r3_loader_recomputes_bundle_approval_source_and_schedule_identities(
    tmp_path: Path,
) -> None:
    runner = _runner()
    sealed = _seal_activation_fixture(tmp_path)

    bundle = runner.read_r3_execution_bundle(
        sealed["output_root"],
        config=sealed["config"],
    )

    assert bundle["manifest"] == sealed["manifest"]
    assert bundle["request_count"] == 129
    assert bundle["truth_sidecar_count"] == 129
    assert len(bundle["worker_inputs"]) == 129
    assert bundle["formal_schedule"]["schedule_sha256"] == sealed[
        "prepared"
    ]["formal_schedule_sha256"]
    assert bundle["resource_policy_root_sha256"] == sealed["prepared"][
        "resource_policy_root_sha256"
    ]


def test_r3_loader_keeps_truth_sidecar_parent_only(
    tmp_path: Path,
) -> None:
    runner = _runner()
    sealed = _seal_activation_fixture(tmp_path)
    bundle = runner.read_r3_execution_bundle(
        sealed["output_root"],
        config=sealed["config"],
    )

    assert len(bundle["parent_truth_sidecars"]) == 129
    assert all(
        "truth" not in repr(worker_input).casefold()
        and "oracle" not in repr(worker_input).casefold()
        and "sidecar" not in repr(worker_input).casefold()
        and "difficulty_class" not in repr(worker_input).casefold()
        for worker_input in bundle["worker_inputs"].values()
    )


def test_r3_worker_decodes_execution_request_v3_without_macro_resampling(
    tmp_path: Path,
) -> None:
    inputs = _inputs()
    runner = _runner()
    sealed = _seal_activation_fixture(tmp_path)
    bundle = runner.read_r3_execution_bundle(
        sealed["output_root"],
        config=sealed["config"],
    )
    worker_input = next(iter(bundle["worker_inputs"].values()))

    decoded = inputs.decode_r3_provider_execution_request(
        worker_input["provider_execution_request"],
        worker_input["provider_local_snapshot_payload"],
        producer_binding=bundle["producer_binding"],
    )

    assert decoded["planning_request"].terrain_snapshot.geometry.width == 20
    assert decoded["planning_request"].terrain_snapshot.geometry.height == 20
    assert decoded[
        "planning_request"
    ].terrain_snapshot.geometry.resolution_m == 0.5
    assert decoded["planning_request"].start_state.x_m == 1.25
    assert decoded["projection_sha256"]
    assert decoded["execution_signature_sha256"]


def test_r3_worker_applies_exact_legged_and_hopper_route_state_budgets(
    tmp_path: Path,
) -> None:
    inputs = _inputs()
    runner = _runner()
    sealed = _seal_activation_fixture(tmp_path)
    bundle = runner.read_r3_execution_bundle(
        sealed["output_root"],
        config=sealed["config"],
    )
    decoded_by_platform: dict[str, dict[str, object]] = {}
    for worker_input in bundle["worker_inputs"].values():
        platform = worker_input["provider_execution_request"][
            "provider_blind_request"
        ]["platform_kind"]
        if platform not in {"legged", "hopper"} or platform in decoded_by_platform:
            continue
        decoded_by_platform[platform] = (
            inputs.decode_r3_provider_execution_request(
                worker_input["provider_execution_request"],
                worker_input["provider_local_snapshot_payload"],
                producer_binding=bundle["producer_binding"],
            )
        )

    assert decoded_by_platform["legged"][
        "planning_request"
    ].resource_budget.max_route_states == 5
    assert decoded_by_platform["hopper"][
        "planning_request"
    ].resource_budget.max_route_states == 2


def test_r3_worker_canonicalizes_full_success_or_structured_failure(
    tmp_path: Path,
) -> None:
    runner = _runner()
    sealed = _seal_activation_fixture(tmp_path)
    bundle = runner.read_r3_execution_bundle(
        sealed["output_root"],
        config=sealed["config"],
    )
    schedules = runner.build_r3_nonformal_schedules(
        bundle["provider_blind_requests"],
        producer_schema_contract=bundle["producer_binding"],
        probe_selection=bundle["p03_probe_selection"],
    )
    task = runner.hydrate_r3_calls(
        [schedules["worker_one"][0]],
        bundle=bundle,
    )[0]
    success = runner.build_r3_provider_diagnostic_row(
        {**task, "_worker_count": 1},
        _r3_success_outcome(task),
        _r3_timing(),
    )
    failure_outcome = {
        "outcome_type": "failure",
        "request_id": task["request_id"],
        "platform_kind": task["platform"],
        "category": "search_exhausted",
        "reason_code": "no_complete_route",
        "stage": "search",
        "checks": ["frontier_empty"],
    }
    failure = runner.build_r3_provider_diagnostic_row(
        {**task, "_worker_count": 1},
        failure_outcome,
        _r3_timing(),
    )

    assert success["provider_success"] is True
    assert success["route_l2_valid"] is True
    assert success["canonical_outcome"] == _r3_success_outcome(task)
    assert failure["provider_success"] is False
    assert failure["route_l2_valid"] is False
    assert failure["canonical_outcome"] == failure_outcome


def test_r3_cli_preflight_accepts_only_p01_and_calls_no_provider(
    tmp_path: Path,
) -> None:
    runner = _runner()
    sealed = _seal_activation_fixture(tmp_path)
    calls = 0

    def forbidden_batch(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("Provider batch must not run during p01")

    result = runner.run_g2_r3(
        config_path=sealed["config_path"],
        input_bundle=sealed["output_root"],
        run_id="r3-preflight-test",
        mode="preflight",
        output_base=tmp_path / "runs",
        batch_executor=forbidden_batch,
    )

    assert result["accepted_phase_ids"] == ["p01"]
    assert result["formal_row_count"] == 0
    assert result["provider_called"] is False
    assert calls == 0


def test_r3_cli_diagnostic_resumes_then_accepts_p02_before_p03(
    tmp_path: Path,
) -> None:
    runner = _runner()
    sealed = _seal_activation_fixture(tmp_path)
    worker_counts: list[int] = []

    def fake_batch(calls, *, max_workers):
        worker_counts.append(max_workers)
        return _r3_fake_success_batch(
            runner,
            list(calls),
            max_workers=max_workers,
        )

    runner.run_g2_r3(
        config_path=sealed["config_path"],
        input_bundle=sealed["output_root"],
        run_id="r3-diagnostic-test",
        mode="preflight",
        output_base=tmp_path / "runs",
        batch_executor=fake_batch,
    )
    result = runner.run_g2_r3(
        config_path=sealed["config_path"],
        input_bundle=sealed["output_root"],
        run_id="r3-diagnostic-test",
        mode="diagnostic",
        output_base=tmp_path / "runs",
        batch_executor=fake_batch,
    )

    assert result["accepted_phase_ids"] == ["p01", "p02", "p03"]
    assert result["p02_row_count"] == 33
    assert result["p03_row_count"] == 24
    assert result["formal_row_count"] == 0
    assert worker_counts == [1, 4, 1, 4]


def test_r3_p02_has_3_cold_30_warmup_and_zero_formal_rows(
    tmp_path: Path,
) -> None:
    runner = _runner()
    sealed = _seal_activation_fixture(tmp_path)

    result = runner.run_g2_r3(
        config_path=sealed["config_path"],
        input_bundle=sealed["output_root"],
        run_id="r3-p02-count-test",
        mode="diagnostic",
        output_base=tmp_path / "runs",
        batch_executor=lambda calls, *, max_workers: (
            _r3_fake_success_batch(
                runner,
                list(calls),
                max_workers=max_workers,
            )
        ),
    )

    assert result["p02_cold_start_count"] == 3
    assert result["p02_warmup_count"] == 30
    assert result["formal_row_count"] == 0


def test_r3_p03_executes_same_12_requests_at_worker_one_and_four(
    tmp_path: Path,
) -> None:
    runner = _runner()
    sealed = _seal_activation_fixture(tmp_path)
    p03_requests: dict[int, list[str]] = {1: [], 4: []}

    def fake_batch(calls, *, max_workers):
        materialized = list(calls)
        if (
            materialized
            and materialized[0]["schedule_kind"]
            in {"worker-one", "worker-four"}
        ):
            p03_requests[max_workers] = [
                str(call["request_id"]) for call in materialized
            ]
        return _r3_fake_success_batch(
            runner,
            materialized,
            max_workers=max_workers,
        )

    runner.run_g2_r3(
        config_path=sealed["config_path"],
        input_bundle=sealed["output_root"],
        run_id="r3-p03-same-test",
        mode="diagnostic",
        output_base=tmp_path / "runs",
        batch_executor=fake_batch,
    )

    assert len(p03_requests[1]) == 12
    assert p03_requests[1] == p03_requests[4]


def test_r3_resume_revalidates_bundle_approval_and_phase_prefix(
    tmp_path: Path,
) -> None:
    runner = _runner()
    sealed = _seal_activation_fixture(tmp_path)
    runner.run_g2_r3(
        config_path=sealed["config_path"],
        input_bundle=sealed["output_root"],
        run_id="r3-resume-drift-test",
        mode="preflight",
        output_base=tmp_path / "runs",
        batch_executor=lambda *_args, **_kwargs: [],
    )
    sealed["approval_path"].write_bytes(
        sealed["approval_path"].read_bytes() + b" "
    )

    with pytest.raises(
        runner.G2Blocked,
        match="g2_r3_approval_artifact",
    ):
        runner.run_g2_r3(
            config_path=sealed["config_path"],
            input_bundle=sealed["output_root"],
            run_id="r3-resume-drift-test",
            mode="diagnostic",
            output_base=tmp_path / "runs",
            batch_executor=lambda *_args, **_kwargs: [],
        )


def test_r3_p03_failure_never_opens_p04(
    tmp_path: Path,
) -> None:
    runner = _runner()
    sealed = _seal_activation_fixture(tmp_path)

    def one_failure_batch(calls, *, max_workers):
        rows = _r3_fake_success_batch(
            runner,
            list(calls),
            max_workers=max_workers,
        )
        if (
            rows
            and rows[0]["schedule_kind"] == "worker-four"
        ):
            rows[0]["provider_success"] = False
            rows[0]["route_l2_valid"] = False
        return rows

    result = runner.run_g2_r3(
        config_path=sealed["config_path"],
        input_bundle=sealed["output_root"],
        run_id="r3-p03-failure-test",
        mode="formal",
        output_base=tmp_path / "runs",
        batch_executor=one_failure_batch,
    )

    assert result["gate_status"] == "blocked"
    assert result["blocking_reason"] == "g2_r3_p03_not_all_l2"
    assert result["accepted_phase_ids"] == ["p01", "p02"]
    assert result["p04_started"] is False
    with pytest.raises(
        runner.G2Blocked,
        match="g2_r3_terminal_blocked_run_id",
    ):
        runner.run_g2_r3(
            config_path=sealed["config_path"],
            input_bundle=sealed["output_root"],
            run_id="r3-p03-failure-test",
            mode="diagnostic",
            output_base=tmp_path / "runs",
            batch_executor=lambda calls, *, max_workers: (
                _r3_fake_success_batch(
                    runner,
                    list(calls),
                    max_workers=max_workers,
                )
            ),
        )


def _r3_p03_fixture_rows(
    tmp_path: Path,
) -> tuple[
    object,
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    runner = _runner()
    sealed = _seal_activation_fixture(tmp_path)
    bundle = runner.read_r3_execution_bundle(
        sealed["output_root"],
        config=sealed["config"],
    )
    schedules = runner.build_r3_nonformal_schedules(
        bundle["provider_blind_requests"],
        producer_schema_contract=bundle["producer_binding"],
        probe_selection=bundle["p03_probe_selection"],
    )
    worker_one_calls = runner.hydrate_r3_calls(
        schedules["worker_one"],
        bundle=bundle,
    )
    worker_four_calls = runner.hydrate_r3_calls(
        schedules["worker_four"],
        bundle=bundle,
    )
    worker_one_rows = runner._validate_r3_diagnostic_batch(
        _r3_fake_success_batch(
            runner,
            worker_one_calls,
            max_workers=1,
        ),
        worker_one_calls,
        worker_count=1,
        probe_classes_by_call_id={
            call["call_id"]: call["probe_class"]
            for call in schedules["worker_one"]
        },
    )
    worker_four_rows = runner._validate_r3_diagnostic_batch(
        _r3_fake_success_batch(
            runner,
            worker_four_calls,
            max_workers=4,
        ),
        worker_four_calls,
        worker_count=4,
        probe_classes_by_call_id={
            call["call_id"]: call["probe_class"]
            for call in schedules["worker_four"]
        },
    )
    return (
        runner,
        bundle,
        worker_one_rows,
        worker_four_rows,
    )


def test_r3_p03_requires_all_24_successful_complete_l2_rows(
    tmp_path: Path,
) -> None:
    runner, _bundle, worker_one, worker_four = _r3_p03_fixture_rows(
        tmp_path
    )
    assert runner.validate_r3_p03_diagnostic_rows(
        worker_one,
        worker_four,
    )["status"] == "passed"

    for rows in (worker_one, worker_four):
        rows[0]["canonical_outcome"]["route"] = []
    with pytest.raises(
        runner.G2Blocked,
        match="g2_r3_p03_not_all_l2",
    ):
        runner.validate_r3_p03_diagnostic_rows(worker_one, worker_four)


def test_r3_p03_compares_projection_geometry_outcome_and_semantic_identity(
    tmp_path: Path,
) -> None:
    runner, _bundle, worker_one, worker_four = _r3_p03_fixture_rows(
        tmp_path
    )
    audit = runner.validate_r3_p03_diagnostic_rows(
        worker_one,
        worker_four,
    )

    assert audit["semantic_identity_fields"] == [
        "provider_request_sha256",
        "execution_request_sha256",
        "resource_policy_sha256",
        "projection_sha256",
        "binary64_pose_projection_sha256",
        "provider_local_snapshot_sha256",
        "terrain_geometry_sha256",
        "profile_or_parameter_record_sha256",
        "capability_id",
        "hopper_parameter_record_sha256",
        "execution_signature_sha256",
        "canonical_outcome",
        "provider_result_sha256",
        "semantic_digest",
    ]
    worker_four[0]["terrain_geometry_sha256"] = SHA_A
    with pytest.raises(
        runner.G2Blocked,
        match="g2_r3_p03_semantic_drift",
    ):
        runner.validate_r3_p03_diagnostic_rows(worker_one, worker_four)


def test_r3_p03_blocks_unrepresented_execution_signature(
    tmp_path: Path,
) -> None:
    runner, bundle, worker_one, worker_four = _r3_p03_fixture_rows(tmp_path)
    expected_signatures = bundle["p03_execution_signatures"]
    assert runner.validate_r3_p03_diagnostic_rows(
        worker_one,
        worker_four,
        expected_execution_signatures=expected_signatures,
    )["status"] == "passed"

    worker_one[0]["execution_signature_sha256"] = SHA_A
    worker_four[0]["execution_signature_sha256"] = SHA_A
    with pytest.raises(
        runner.G2Blocked,
        match="g2_r3_p03_signature_uncovered",
    ):
        runner.validate_r3_p03_diagnostic_rows(
            worker_one,
            worker_four,
            expected_execution_signatures=expected_signatures,
        )


def test_r3_truth_loader_runs_only_after_each_provider_batch_is_complete() -> None:
    runner = _runner()
    opened = 0

    def load_truth() -> list[dict[str, object]]:
        nonlocal opened
        opened += 1
        return [{"request_id": "truth-row"}]

    worker_one = [{"call_id": f"one-{index:02d}"} for index in range(12)]
    worker_four = [{"call_id": f"four-{index:02d}"} for index in range(11)]
    with pytest.raises(
        runner.G2Blocked,
        match="g2_r3_provider_batch_incomplete",
    ):
        runner.open_r3_truth_after_provider_batches(
            worker_one,
            worker_four,
            truth_loader=load_truth,
        )
    assert opened == 0

    worker_four.append({"call_id": "four-11"})
    assert runner.open_r3_truth_after_provider_batches(
        worker_one,
        worker_four,
        truth_loader=load_truth,
    ) == [{"request_id": "truth-row"}]
    assert opened == 1


def test_r3_nonformal_rows_never_enter_645_timing_distribution() -> None:
    runner = _runner()
    formal = [
        {
            "call_id": f"formal-{index:03d}",
            "schedule_kind": "formal",
            "formal_sample": True,
            "timing": _r3_timing(),
        }
        for index in range(645)
    ]
    nonformal = [
        {
            "call_id": f"diagnostic-{index:02d}",
            "schedule_kind": "worker-one",
            "formal_sample": False,
            "timing": _r3_timing(),
        }
        for index in range(24)
    ]

    selected = runner.select_r3_formal_timing_rows(
        [*nonformal, *formal]
    )
    assert len(selected) == 645
    assert all(row["formal_sample"] is True for row in selected)
    with pytest.raises(
        runner.G2Blocked,
        match="g2_r3_formal_timing_distribution",
    ):
        runner.select_r3_formal_timing_rows([*nonformal, *formal[:-1]])


def test_r3_timer_excludes_preload_queue_truth_io_journal_and_summary() -> None:
    runner = _runner()
    events: list[str] = []
    outside_clock = 0

    def outside(name: str):
        def operation(value=None):
            nonlocal outside_clock
            outside_clock += 100
            events.append(name)
            return value

        return operation

    def timed_provider(value):
        events.append("timer-start")
        timing = _r3_timing()
        events.append("timer-end")
        return {"value": value, "timing": timing}

    result = runner.execute_r3_timing_pure_pipeline(
        {"request": "encoded"},
        preload_request=outside("preload"),
        enqueue_request=outside("queue"),
        execute_timed_provider=timed_provider,
        open_truth=lambda: outside("truth")([{"truth": 1}]),
        journal_result=outside("journal"),
        summarize_result=outside("summary"),
    )

    assert events == [
        "preload",
        "queue",
        "timer-start",
        "timer-end",
        "truth",
        "journal",
        "summary",
    ]
    assert result["timing"] == _r3_timing()
    assert result["outside_clock_ignored"] is True
    assert outside_clock == 500
