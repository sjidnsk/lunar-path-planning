from __future__ import annotations

import io
import re
import tarfile
from pathlib import Path
from typing import Any, Mapping

from .canonical import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    canonical_loads,
    domain_hash,
    sha256_bytes,
)
from .finite_graph import build_all_optima
from .generate_cases import generate_all_cases
from .generate_requests import (
    _graph_from_raw_source,
    build_repeat_mapping,
    generate_raw_request_sources,
    select_requests,
    solve_raw_request_sources,
    terrain_npz_bytes,
)
from .lola import decode_lola_pair
from .models import make_case_identity
from .oracle_hopper import (
    evaluator_source_bytes,
    evaluate_hopper,
    validate_hopper_parameter_record,
)
from .oracle_legged import evaluate_legged
from .oracle_wheel import evaluate_wheel


_AUTHORIZATION_SHA256 = (
    "720e11ef04ad2b57283421809a077ccf1f0b35167a9f482082241398ad0214d2"
)
_FIXTURE_PARENT = Path("D:/xunce/tmp/pytest-mid-dual")
_PRODUCTION_PARENT = Path("D:/xunce/inputs/mid_dual/g2-truth")
_CANDIDATE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,63}$")


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _source_paths(source_root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in source_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source_root)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if ".pytest_cache" in relative.parts:
            continue
        paths.append(relative)
    return sorted(paths, key=lambda item: item.as_posix())


def producer_implementation_sha256(source_root: Path) -> str:
    parts = [
        relative.as_posix().encode("utf-8")
        + b"\0"
        + (source_root / relative).read_bytes()
        for relative in _source_paths(source_root)
    ]
    return domain_hash("g2-producer-implementation/v2", *parts)


def deterministic_source_tar(source_root: Path) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.GNU_FORMAT) as archive:
        for relative in _source_paths(source_root):
            data = (source_root / relative).read_bytes()
            info = tarfile.TarInfo(relative.as_posix())
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            archive.addfile(info, io.BytesIO(data))
    return output.getvalue()


def _validate_fixture_target(output_root: Path) -> tuple[Path, Path]:
    resolved = output_root.resolve()
    fixture_parent = _FIXTURE_PARENT.resolve()
    if fixture_parent != resolved and fixture_parent not in resolved.parents:
        raise ValueError(f"fixture output must be beneath {fixture_parent}")
    if not any(part.startswith("g2-producer-") for part in resolved.parts):
        raise ValueError("fixture output requires unique g2-producer-* test root")
    staging = resolved.with_name(f".{resolved.name}.staging")
    if resolved.exists() or staging.exists():
        raise FileExistsError(f"fixture output or staging already exists: {resolved}")
    return resolved, staging


def _validate_production_target(
    output_root: Path, candidate_id: str
) -> tuple[Path, Path]:
    if not _CANDIDATE_ID_RE.fullmatch(candidate_id):
        raise ValueError("invalid production candidate id")
    resolved = output_root.resolve()
    expected = (_PRODUCTION_PARENT / candidate_id).resolve()
    if resolved != expected:
        raise ValueError(f"production output must equal {expected}")
    staging = resolved.with_name(f".{candidate_id}.staging")
    if resolved.exists() or staging.exists():
        raise FileExistsError(f"production output or staging already exists: {resolved}")
    return resolved, staging


def _raw_hashes(raw_sources: Mapping[str, bytes]) -> dict[str, str]:
    for relative_name in raw_sources:
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe raw source path: {relative_name}")
    return {
        key: sha256_bytes(payload)
        for key, payload in sorted(raw_sources.items())
    }


def _validate_authorization(raw_sources: Mapping[str, bytes]) -> None:
    if "project-authorization.md" not in raw_sources:
        raise ValueError("missing raw source: project-authorization.md")
    if sha256_bytes(raw_sources["project-authorization.md"]) != _AUTHORIZATION_SHA256:
        raise ValueError("project authorization SHA-256 mismatch")


def _fixture_raw_source_contract(
    raw_sources: Mapping[str, bytes],
    lola_provenance: dict[str, Any],
) -> dict[str, str]:
    _validate_authorization(raw_sources)
    jp2_keys = sorted(key for key in raw_sources if key.lower().endswith(".jp2"))
    lbl_keys = sorted(key for key in raw_sources if key.lower().endswith(".lbl"))
    if len(jp2_keys) != 1:
        raise ValueError("fixture raw source requires exactly one LOLA JP2")
    if len(lbl_keys) != 1:
        inferred = str(Path(jp2_keys[0]).with_suffix(".LBL")).replace("\\", "/")
        raise ValueError(f"missing raw source: {inferred}")
    expected_keys = {
        "project-authorization.md",
        jp2_keys[0],
        lbl_keys[0],
    }
    if set(raw_sources) != expected_keys:
        raise ValueError("fixture raw source set has extra or duplicate inputs")
    if sha256_bytes(raw_sources[jp2_keys[0]]) != lola_provenance["jp2_sha256"]:
        raise ValueError("LOLA JP2 SHA-256 mismatch")
    if sha256_bytes(raw_sources[lbl_keys[0]]) != lola_provenance["lbl_sha256"]:
        raise ValueError("LOLA LBL SHA-256 mismatch")
    if lola_provenance.get("fixture_only") is not True:
        raise ValueError("build_fixture_bundle requires fixture-only LOLA provenance")
    return _raw_hashes(raw_sources)


def _production_raw_source_contract(
    raw_sources: Mapping[str, bytes],
    *,
    hopper_parameter_record: dict[str, Any],
    source_root: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    _validate_authorization(raw_sources)
    exact_keys = {
        "hopper/HOPPER_PARAMETER_RECORD.json",
        "hopper/hopper_energy_evaluator.py",
        "hopper/hopper_stop_evaluator.py",
        "lola/LDEM_875S_20M.JP2",
        "lola/LDEM_875S_20M_JP2.LBL",
        "project-authorization.md",
    }
    if set(raw_sources) != exact_keys:
        missing = sorted(exact_keys - set(raw_sources))
        extra = sorted(set(raw_sources) - exact_keys)
        raise ValueError(
            f"production raw source set mismatch: missing={missing}, extra={extra}"
        )
    parsed_record = canonical_loads(
        raw_sources["hopper/HOPPER_PARAMETER_RECORD.json"]
    )
    if parsed_record != hopper_parameter_record:
        raise ValueError("Hopper parameter record argument/raw bytes mismatch")
    evaluator_sources = {
        "energy_model": raw_sources["hopper/hopper_energy_evaluator.py"],
        "stop_condition": raw_sources["hopper/hopper_stop_evaluator.py"],
    }
    validate_hopper_parameter_record(
        hopper_parameter_record,
        source_root=source_root,
        evaluator_sources=evaluator_sources,
    )
    local_sources = evaluator_source_bytes(source_root)
    if evaluator_sources != local_sources:
        raise ValueError("Hopper evaluator raw bytes/source snapshot mismatch")
    specification = canonical_loads((source_root / "SPECIFICATION.json").read_bytes())
    decoded_lola = decode_lola_pair(
        raw_sources["lola/LDEM_875S_20M.JP2"],
        raw_sources["lola/LDEM_875S_20M_JP2.LBL"],
        specification,
    )
    return _raw_hashes(raw_sources), decoded_lola


def _primitive_labels(
    cases: dict[str, list[dict[str, Any]]],
    *,
    specification_sha256: str,
    implementation_sha256: str,
    profile_hashes: dict[str, str],
    hopper_parameter_record: dict[str, Any],
) -> list[dict[str, Any]]:
    evaluator = {
        "wheel": evaluate_wheel,
        "legged": evaluate_legged,
        "hopper": lambda case: evaluate_hopper(case, hopper_parameter_record),
    }
    rows: list[dict[str, Any]] = []
    case_envelope_sha = domain_hash(
        "g2-case-envelope/v2", specification_sha256.encode("ascii")
    )
    for platform in ("wheel", "legged", "hopper"):
        profile_id = (
            str(hopper_parameter_record["parameter_set_id"])
            if platform == "hopper"
            else f"g2-independent-{platform}-profile/v2"
        )
        for source_row in cases[platform]:
            case = source_row["case"]
            decision = evaluator[platform](case)
            if decision["oracle_reason_code"] == "G2I_NUMERIC_UNDECIDED":
                raise ValueError("numeric-undecided case cannot enter primitive quota")
            case_sha, label_id = make_case_identity(platform, case)
            rows.append(
                {
                    "case_envelope_sha256": case_envelope_sha,
                    "case_sha256": case_sha,
                    "label_id": label_id,
                    "numeric_witness": decision["numeric_witness"],
                    "oracle_reason_code": decision["oracle_reason_code"],
                    "oracle_safe": decision["oracle_safe"],
                    "oracle_spec_sha256": specification_sha256,
                    "platform_kind": platform,
                    "primitive_canonical": case,
                    "primitive_sha256": domain_hash(
                        "g2-primitive/v2", canonical_json_bytes(case)
                    ),
                    "producer_id": "g2-isolated-t2-producer",
                    "producer_implementation_sha256": implementation_sha256,
                    "producer_revision": "0.2.0",
                    "profile_or_parameter_record_sha256": profile_hashes[platform],
                    "profile_or_parameter_set_id": profile_id,
                    "safety_slacks": decision["safety_slacks"],
                    "schema_version": "g2-primitive-label/v2",
                    "terrain_sha256": case["terrain_sha256"],
                }
            )
    return rows


def _json_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _index_from_payloads(payloads: Mapping[str, bytes]) -> list[dict[str, Any]]:
    return [
        {
            "byte_length": len(payload),
            "relative_path": relative_path,
            "sha256": sha256_bytes(payload),
        }
        for relative_path, payload in sorted(payloads.items())
    ]


def _index_root(index: list[dict[str, Any]]) -> str:
    return domain_hash("g2-payload-root/v1", canonical_json_bytes(index))


def _bundle_payloads(
    *,
    source_root: Path,
    hopper_parameter_record: dict[str, Any],
    raw_sources: Mapping[str, bytes],
    raw_hashes: dict[str, str],
    lola_provenance: dict[str, Any],
    fixture_only: bool,
    preflight_evidence: dict[str, Any] | None,
    candidate_id: str | None,
) -> dict[str, bytes]:
    specification_bytes = (source_root / "SPECIFICATION.json").read_bytes()
    specification = canonical_loads(specification_bytes)
    specification_sha = sha256_bytes(specification_bytes)
    implementation_sha = producer_implementation_sha256(source_root)
    hopper_sha = sha256_bytes(canonical_json_bytes(hopper_parameter_record))
    profile_hashes = {
        "wheel": domain_hash("g2-independent-profile/v2", b"wheel"),
        "legged": domain_hash("g2-independent-profile/v2", b"legged"),
        "hopper": hopper_sha,
    }
    preflight_sha = (
        None
        if preflight_evidence is None
        else str(preflight_evidence["preflight_sha256"])
    )
    input_contract_sha = domain_hash(
        "g2-producer-input-contract/v2",
        specification_sha.encode("ascii"),
        implementation_sha.encode("ascii"),
        hopper_sha.encode("ascii"),
        canonical_json_bytes(raw_hashes),
        str(preflight_sha).encode("ascii"),
    )
    input_set_prefix = "g2t2-fixture" if fixture_only else "g2t2-candidate"
    input_set_id = f"{input_set_prefix}-{input_contract_sha[:24]}"

    cases = generate_all_cases(
        specification, hopper_parameter_record=hopper_parameter_record
    )
    labels = _primitive_labels(
        cases,
        specification_sha256=specification_sha,
        implementation_sha256=implementation_sha,
        profile_hashes=profile_hashes,
        hopper_parameter_record=hopper_parameter_record,
    )
    optima = build_all_optima(
        specification, hopper_parameter_record=hopper_parameter_record
    )
    raw_request_sources = generate_raw_request_sources(
        specification, lola_provenance=lola_provenance
    )
    request_pool = solve_raw_request_sources(
        raw_request_sources,
        specification=specification,
        profile_record_sha256=profile_hashes,
        hopper_parameter_record=hopper_parameter_record,
    )
    requests = select_requests(request_pool, specification)
    repeat_mapping = build_repeat_mapping(requests, input_set_id=input_set_id)

    payloads: dict[str, bytes] = {
        "primitive-labels.jsonl": canonical_jsonl_bytes(labels),
        "small-map-optima.jsonl": canonical_jsonl_bytes(
            optima[platform]["row"]
            for platform in ("wheel", "legged", "hopper")
        ),
        "raw/case-pool.jsonl": canonical_jsonl_bytes(
            row
            for platform in ("wheel", "legged", "hopper")
            for row in cases[platform]
        ),
        "raw/oracle-rows.jsonl": canonical_jsonl_bytes(labels),
        "raw/request-source-pool.jsonl": canonical_jsonl_bytes(
            raw_request_sources
        ),
        "raw/request-pool.jsonl": canonical_jsonl_bytes(
            {
                key: value
                for key, value in row.items()
                if key != "truth_certificate"
            }
            for row in request_pool
        ),
        "raw/hopper-parameter-record.json": _json_bytes(
            hopper_parameter_record
        ),
        "repeat-mapping.jsonl": canonical_jsonl_bytes(repeat_mapping),
        "source/producer-source.tar": deterministic_source_tar(source_root),
    }
    for name in ("SPECIFICATION.json", "producer-lock.json", "LICENSES.json"):
        payloads[f"source/{name}"] = (source_root / name).read_bytes()
    for relative_name, raw_payload in sorted(raw_sources.items()):
        payloads[f"raw/{relative_name}"] = raw_payload
    payloads["raw/lola-provenance.json"] = _json_bytes(
        {
            key: value
            for key, value in lola_provenance.items()
            if key != "roi_records"
        }
    )
    if lola_provenance.get("roi_records"):
        payloads["raw/lola-roi-records.jsonl"] = canonical_jsonl_bytes(
            lola_provenance["roi_records"]
        )
        payloads["raw/lola-decoded-provenance.json"] = _json_bytes(
            {
                key: value
                for key, value in lola_provenance.items()
                if key != "roi_records"
            }
        )

    request_rows: list[dict[str, Any]] = []
    graph_templates: dict[str, dict[str, Any]] = {}
    graph_shape_cache: dict[tuple[str, str], dict[str, Any]] = {}
    raw_by_sha = {
        row["raw_source_sha256"]: row for row in raw_request_sources
    }
    for request in request_pool:
        certificate = request["truth_certificate"]
        certificate_sha = request["truth_certificate_sha256"]
        payloads[f"certificates/{certificate_sha}.json"] = canonical_json_bytes(
            certificate
        )
        raw = raw_by_sha[request["raw_source_sha256"]]
        topology_sha = domain_hash(
            "g2-request-topology/v1",
            canonical_json_bytes(raw["terrain_arrays"]["cell_class"]),
        )
        graph_key = (str(raw["platform_kind"]), topology_sha)
        if graph_key not in graph_shape_cache:
            graph_shape_cache[graph_key] = _graph_from_raw_source(
                raw, hopper_parameter_record=hopper_parameter_record
            )[0]
        graph = graph_shape_cache[graph_key]
        graph_sha = domain_hash(
            "g2-request-graph-template/v1", canonical_json_bytes(graph)
        )
        if graph_sha != request["graph_template_sha256"]:
            raise ValueError("request graph template SHA-256 drift")
        graph_templates.setdefault(graph_sha, graph)
        terrain_payload = terrain_npz_bytes(
            request["terrain_provenance"],
            terrain_arrays=request["terrain_arrays"],
        )
        if sha256_bytes(terrain_payload) != request["terrain_sha256"]:
            raise ValueError("request terrain SHA-256 drift")
        payloads[f"terrain/{request['terrain_sha256']}.npz"] = terrain_payload
    for graph_sha, graph in graph_templates.items():
        payloads[f"raw/graph-templates/{graph_sha}.json"] = canonical_json_bytes(
            graph
        )
    for request in requests:
        request_rows.append(
            {
                key: value
                for key, value in request.items()
                if key not in {"truth_certificate", "terrain_arrays"}
            }
        )
    payloads["requests.jsonl"] = canonical_jsonl_bytes(request_rows)
    for platform in ("wheel", "legged", "hopper"):
        certificate = optima[platform]["certificate"]
        certificate_sha = optima[platform]["row"]["certificate_sha256"]
        payloads[f"certificates/{certificate_sha}.json"] = canonical_json_bytes(
            certificate
        )

    reject_rows = [
        {
            "artifact_kind": "primitive_label",
            "artifact_sha256": row["case_sha256"],
            "reason_code": row["oracle_reason_code"],
        }
        for row in labels
        if not bool(row["oracle_safe"])
    ] + [
        {
            "artifact_kind": "request_pool",
            "artifact_sha256": row["truth_request_sha256"],
            "reason_code": "G2I_REQUEST_UNREACHABLE",
        }
        for row in request_pool
        if not bool(row["oracle_reachable"])
    ]
    payloads["reject-ledger.jsonl"] = canonical_jsonl_bytes(reject_rows)

    for shard_index, platform in enumerate(("wheel", "legged", "hopper")):
        platform_rows = [
            row for row in labels if row["platform_kind"] == platform
        ]
        payloads[
            f"shard-manifests/shard-{shard_index:04d}.json"
        ] = _json_bytes(
            {
                "implementation_sha256": implementation_sha,
                "input_sha256": input_contract_sha,
                "preflight_sha256": preflight_sha,
                "row_count": len(platform_rows),
                "row_root_sha256": domain_hash(
                    "g2-shard-rows/v2", canonical_json_bytes(platform_rows)
                ),
                "shard_index": shard_index,
            }
        )

    from .audit_bundle import audit_source_tree

    static_audit = audit_source_tree(source_root)
    if not static_audit["passed"]:
        raise ValueError(f"source isolation audit failed: {static_audit}")
    technical_isolation = {
        **static_audit,
        "fixture_only": fixture_only,
        "preflight_sha256": preflight_sha,
        "production_process_isolation_verified": (
            not fixture_only
            and preflight_evidence is not None
            and preflight_evidence.get("passed") is True
        ),
    }
    if not fixture_only:
        if preflight_evidence is None or preflight_evidence.get("passed") is not True:
            raise ValueError("production bundle requires passing strict preflight")
        payloads["audits/preflight.json"] = _json_bytes(preflight_evidence)
    payloads["audits/technical-isolation.json"] = _json_bytes(
        technical_isolation
    )

    counts = {
        "primitive_labels": len(labels),
        "raw_request_pool": len(request_pool),
        "repeat_mapping": len(repeat_mapping),
        "requests": len(request_rows),
        "small_map_optima": len(optima),
    }
    platform_label_counts = {
        platform: sum(row["platform_kind"] == platform for row in labels)
        for platform in ("wheel", "legged", "hopper")
    }
    platform_request_counts = {
        platform: sum(row["platform_kind"] == platform for row in request_rows)
        for platform in ("wheel", "legged", "hopper")
    }
    payloads["audits/cardinality-and-join.json"] = _json_bytes(
        {
            "counts": counts,
            "label_ids_unique": len({row["label_id"] for row in labels})
            == len(labels),
            "platform_label_counts": platform_label_counts,
            "platform_request_counts": platform_request_counts,
            "raw_request_hashes_unique": len(
                {row["truth_request_sha256"] for row in request_pool}
            )
            == len(request_pool),
            "request_ids_unique": len(
                {row["request_id"] for row in request_rows}
            )
            == len(request_rows),
            "truth_request_hashes_unique": len(
                {row["truth_request_sha256"] for row in request_rows}
            )
            == len(request_rows),
        }
    )
    payloads["audits/selection.json"] = _json_bytes(
        {
            "provider_blind": True,
            "selected_request_sha256": [
                row["truth_request_sha256"] for row in request_rows
            ],
        }
    )
    payloads["audits/phase-bindings.json"] = _json_bytes(
        {
            phase: preflight_sha
            for phase in (
                "cases",
                "labels",
                "optima",
                "requests",
                "package",
                "audit",
                "reproduce",
            )
        }
    )
    payloads["audits/reproduction.json"] = _json_bytes(
        {
            "comparison_kind": "two-fresh-independent-materializations/v1",
            "fixture_only": fixture_only,
            "formal_evidence_eligible": False,
            "fresh_run_comparison_complete": True,
            "matched": True,
        }
    )
    payloads["audits/raw-to-truth.json"] = _json_bytes(
        {
            "full_case_regeneration_required": True,
            "full_optimum_regeneration_required": True,
            "full_request_regeneration_required": True,
            "input_contract_sha256": input_contract_sha,
            "preflight_sha256": preflight_sha,
        }
    )

    manifest_core = {
        "authorization_sha256": _AUTHORIZATION_SHA256,
        "candidate_id": candidate_id,
        "counts": counts,
        "fixture_only": fixture_only,
        "formal_evidence_eligible": False,
        "hopper_parameter_record_sha256": hopper_sha,
        "input_contract_sha256": input_contract_sha,
        "lola_roi_root_sha256": lola_provenance.get("roi_root_sha256"),
        "preflight_sha256": preflight_sha,
        "producer_implementation_sha256": implementation_sha,
        "profile_or_parameter_record_sha256": profile_hashes,
        "raw_source_sha256": raw_hashes,
        "schema_version": "g2-truth-manifest-core/v2",
        "specification_sha256": specification_sha,
    }
    manifest_core_sha = domain_hash(
        "g2-manifest-core/v2", canonical_json_bytes(manifest_core)
    )
    payloads["manifest.json"] = _json_bytes(
        {
            **manifest_core,
            "input_set_id": input_set_id,
            "manifest_core_sha256": manifest_core_sha,
        }
    )
    payload_index = _index_from_payloads(payloads)
    payload_root = _index_root(payload_index)
    attestation = {
        "authorization_sha256": _AUTHORIZATION_SHA256,
        "fixture_only": fixture_only,
        "formal_evidence_eligible": False,
        "input_set_id": input_set_id,
        "manifest_core_sha256": manifest_core_sha,
        "organizational_independence": "project_internal",
        "payload_root_sha256": payload_root,
        "status": "pending_o2_signature",
        "technical_independence": (
            "fixture_exercise" if fixture_only else "T2_candidate"
        ),
    }
    attestation_bytes = _json_bytes(attestation)
    payloads["source-attestations.json"] = attestation_bytes
    freeze = {
        "candidate_id": candidate_id,
        "counts": counts,
        "fixture_only": fixture_only,
        "formal_evidence_eligible": False,
        "fresh_reproduction_payload_root_sha256": payload_root,
        "input_set_id": input_set_id,
        "manifest_core_sha256": manifest_core_sha,
        "payload_index": payload_index,
        "payload_root_sha256": payload_root,
        "preflight_sha256": preflight_sha,
        "publication_order": "data-first-freeze-last",
        "schema_version": "g2-truth-freeze/v2",
        "source_attestations_sha256": sha256_bytes(attestation_bytes),
    }
    payloads["truth-freeze.json"] = _json_bytes(freeze)
    return payloads


def _compare_payload_maps(
    first: Mapping[str, bytes], second: Mapping[str, bytes]
) -> dict[str, Any]:
    paths = sorted(set(first) | set(second))
    mismatched = [
        path
        for path in paths
        if path not in first or path not in second or first[path] != second[path]
    ]
    return {
        "matched": not mismatched,
        "mismatched_paths": mismatched,
        "path_count": len(paths),
    }


def _publish_payloads(
    output_root: Path,
    staging: Path,
    payloads: Mapping[str, bytes],
) -> dict[str, Any]:
    staging.mkdir(parents=True, exist_ok=False)
    for relative_path, payload in sorted(payloads.items()):
        if relative_path == "truth-freeze.json":
            continue
        _write_bytes(staging / Path(relative_path), payload)
    _write_bytes(staging / "truth-freeze.json", payloads["truth-freeze.json"])
    staging.rename(output_root)
    return canonical_loads(payloads["truth-freeze.json"])


def _two_fresh_payloads(**kwargs: Any) -> dict[str, bytes]:
    first = _bundle_payloads(**kwargs)
    second = _bundle_payloads(**kwargs)
    comparison = _compare_payload_maps(first, second)
    if not comparison["matched"]:
        raise ValueError(f"fresh reproduction mismatch: {comparison}")
    return first


def build_fixture_bundle(
    output_root: Path,
    *,
    source_root: Path,
    hopper_parameter_record: dict[str, Any],
    raw_sources: dict[str, bytes],
    lola_provenance: dict[str, Any],
) -> dict[str, Any]:
    output_root, staging = _validate_fixture_target(output_root)
    validate_hopper_parameter_record(
        hopper_parameter_record, source_root=source_root
    )
    raw_hashes = _fixture_raw_source_contract(raw_sources, lola_provenance)
    payloads = _two_fresh_payloads(
        source_root=source_root,
        hopper_parameter_record=hopper_parameter_record,
        raw_sources=raw_sources,
        raw_hashes=raw_hashes,
        lola_provenance=lola_provenance,
        fixture_only=True,
        preflight_evidence=None,
        candidate_id=None,
    )
    freeze = _publish_payloads(output_root, staging, payloads)
    from .audit_bundle import audit_bundle

    audit = audit_bundle(output_root)
    if not audit["passed"]:
        raise ValueError(f"fixture bundle audit failed: {audit['reasons']}")
    return freeze


def build_production_candidate(
    output_root: Path,
    *,
    candidate_id: str,
    source_root: Path,
    hopper_parameter_record: dict[str, Any],
    raw_sources: dict[str, bytes],
    preflight_evidence: dict[str, Any],
) -> dict[str, Any]:
    output_root, staging = _validate_production_target(output_root, candidate_id)
    raw_hashes, lola_provenance = _production_raw_source_contract(
        raw_sources,
        hopper_parameter_record=hopper_parameter_record,
        source_root=source_root,
    )
    payloads = _two_fresh_payloads(
        source_root=source_root,
        hopper_parameter_record=hopper_parameter_record,
        raw_sources=raw_sources,
        raw_hashes=raw_hashes,
        lola_provenance=lola_provenance,
        fixture_only=False,
        preflight_evidence=preflight_evidence,
        candidate_id=candidate_id,
    )
    freeze = _publish_payloads(output_root, staging, payloads)
    from .audit_bundle import audit_bundle

    audit = audit_bundle(output_root)
    if not audit["passed"]:
        raise ValueError(f"production bundle audit failed: {audit['reasons']}")
    return freeze


def verify_contiguous_shard_prefix(
    shard_root: Path,
    *,
    expected_input_sha256: str,
    expected_implementation_sha256: str,
) -> dict[str, Any]:
    files = sorted(shard_root.glob("shard-*.json"), key=lambda path: path.name)
    manifests: list[tuple[int, dict[str, Any]]] = []
    for path in files:
        try:
            filename_index = int(path.stem.split("-")[-1])
        except ValueError as error:
            raise ValueError(f"invalid shard filename: {path.name}") from error
        manifest = canonical_loads(path.read_bytes())
        if manifest["input_sha256"] != expected_input_sha256:
            raise ValueError(f"input drift at shard {filename_index}")
        if manifest["implementation_sha256"] != expected_implementation_sha256:
            raise ValueError(f"implementation drift at shard {filename_index}")
        if int(manifest["shard_index"]) != filename_index:
            raise ValueError(
                f"shard prefix is not contiguous: index mismatch at {path.name}"
            )
        manifests.append((filename_index, manifest))
    indices = [index for index, _ in manifests]
    if indices != list(range(len(indices))):
        raise ValueError(f"shard prefix is not contiguous: {indices}")
    return {
        "accepted_indices": indices,
        "row_count": sum(int(manifest["row_count"]) for _, manifest in manifests),
    }
