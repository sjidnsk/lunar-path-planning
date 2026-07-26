from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import Any

from .canonical import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    canonical_loads,
    content_index,
    content_index_root,
    domain_hash,
    sha256_bytes,
)
from .finite_graph import build_all_optima
from .generate_cases import generate_all_cases
from .generate_requests import (
    generate_request_pool,
    select_requests,
    terrain_npz_bytes,
)
from .models import make_case_identity
from .oracle_hopper import evaluate_hopper, validate_hopper_parameter_record
from .oracle_legged import evaluate_legged
from .oracle_wheel import evaluate_wheel


_AUTHORIZATION_SHA256 = (
    "720e11ef04ad2b57283421809a077ccf1f0b35167a9f482082241398ad0214d2"
)
_FIXTURE_PARENT = Path("D:/xunce/tmp/pytest-mid-dual")


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _write_json(path: Path, value: Any) -> None:
    _write_bytes(path, canonical_json_bytes(value) + b"\n")


def _source_paths(source_root: Path) -> list[Path]:
    paths: list[Path] = []
    for path in source_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source_root)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.name == ".pytest_cache":
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
    return domain_hash("g2-producer-implementation/v1", *parts)


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


def _raw_source_contract(
    raw_sources: dict[str, bytes],
    lola_provenance: dict[str, Any],
) -> dict[str, str]:
    if "project-authorization.md" not in raw_sources:
        raise ValueError("missing raw source: project-authorization.md")
    authorization_sha = sha256_bytes(raw_sources["project-authorization.md"])
    if authorization_sha != _AUTHORIZATION_SHA256:
        raise ValueError("project authorization SHA-256 mismatch")
    jp2_keys = sorted(key for key in raw_sources if key.lower().endswith(".jp2"))
    lbl_keys = sorted(key for key in raw_sources if key.lower().endswith(".lbl"))
    if not jp2_keys:
        raise ValueError("missing raw source: lola/LDEM_FIXTURE.JP2")
    if not lbl_keys:
        inferred = str(Path(jp2_keys[0]).with_suffix(".LBL")).replace("\\", "/")
        raise ValueError(f"missing raw source: {inferred}")
    if sha256_bytes(raw_sources[jp2_keys[0]]) != lola_provenance["jp2_sha256"]:
        raise ValueError("LOLA JP2 SHA-256 mismatch")
    if sha256_bytes(raw_sources[lbl_keys[0]]) != lola_provenance["lbl_sha256"]:
        raise ValueError("LOLA LBL SHA-256 mismatch")
    if lola_provenance.get("fixture_only") is not True:
        raise ValueError("build_fixture_bundle requires fixture-only LOLA provenance")
    return {
        key: sha256_bytes(payload) for key, payload in sorted(raw_sources.items())
    }


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
        "g2-case-envelope/v1", specification_sha256.encode("ascii")
    )
    for platform in ("wheel", "legged", "hopper"):
        profile_id = (
            str(hopper_parameter_record["parameter_set_id"])
            if platform == "hopper"
            else f"g2-independent-{platform}-profile/v1"
        )
        for source_row in cases[platform]:
            case = source_row["case"]
            decision = evaluator[platform](case)
            if decision["oracle_reason_code"] == "G2I_NUMERIC_UNDECIDED":
                raise ValueError("numeric-undecided case cannot enter primitive quota")
            case_sha, label_id = make_case_identity(platform, case)
            primitive = dict(case)
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
                    "primitive_canonical": primitive,
                    "primitive_sha256": domain_hash(
                        "g2-primitive/v1", canonical_json_bytes(primitive)
                    ),
                    "producer_id": "g2-isolated-t2-producer",
                    "producer_implementation_sha256": implementation_sha256,
                    "producer_revision": "0.1.0",
                    "profile_or_parameter_record_sha256": profile_hashes[platform],
                    "profile_or_parameter_set_id": profile_id,
                    "safety_slacks": decision["safety_slacks"],
                    "schema_version": "g2-primitive-label/v1",
                    "terrain_sha256": case["terrain_sha256"],
                }
            )
    return rows

def build_fixture_bundle(
    output_root: Path,
    *,
    source_root: Path,
    hopper_parameter_record: dict[str, Any],
    raw_sources: dict[str, bytes],
    lola_provenance: dict[str, Any],
) -> dict[str, Any]:
    output_root, staging = _validate_fixture_target(output_root)
    validate_hopper_parameter_record(hopper_parameter_record)
    raw_hashes = _raw_source_contract(raw_sources, lola_provenance)
    spec_path = source_root / "SPECIFICATION.json"
    specification = canonical_loads(spec_path.read_bytes())
    specification_sha = sha256_bytes(spec_path.read_bytes())
    implementation_sha = producer_implementation_sha256(source_root)
    hopper_sha = sha256_bytes(canonical_json_bytes(hopper_parameter_record))
    profile_hashes = {
        "wheel": domain_hash("g2-independent-profile/v1", b"wheel"),
        "legged": domain_hash("g2-independent-profile/v1", b"legged"),
        "hopper": hopper_sha,
    }
    input_contract_sha = domain_hash(
        "g2-producer-input-contract/v1",
        specification_sha.encode("ascii"),
        implementation_sha.encode("ascii"),
        hopper_sha.encode("ascii"),
        canonical_json_bytes(raw_hashes),
    )

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
    request_pool = generate_request_pool(
        specification,
        profile_record_sha256=profile_hashes,
        lola_provenance=lola_provenance,
    )
    requests = select_requests(request_pool, specification)

    staging.mkdir(parents=True)
    _write_bytes(staging / "primitive-labels.jsonl", canonical_jsonl_bytes(labels))
    _write_bytes(
        staging / "small-map-optima.jsonl",
        canonical_jsonl_bytes(optima[platform]["row"] for platform in ("wheel", "legged", "hopper")),
    )
    request_rows: list[dict[str, Any]] = []
    for request in requests:
        certificate = request["truth_certificate"]
        certificate_sha = request["truth_certificate_sha256"]
        _write_bytes(
            staging / "certificates" / f"{certificate_sha}.json",
            canonical_json_bytes(certificate),
        )
        terrain_bytes = terrain_npz_bytes(request["terrain_provenance"])
        terrain_sha = sha256_bytes(terrain_bytes)
        if terrain_sha != request["terrain_sha256"]:
            raise ValueError("request terrain SHA-256 drift")
        terrain_path = staging / "terrain" / f"{terrain_sha}.npz"
        if terrain_path.exists() and terrain_path.read_bytes() != terrain_bytes:
            raise ValueError("terrain hash collision")
        if not terrain_path.exists():
            _write_bytes(terrain_path, terrain_bytes)
        request_rows.append(
            {key: value for key, value in request.items() if key != "truth_certificate"}
        )
    _write_bytes(staging / "requests.jsonl", canonical_jsonl_bytes(request_rows))
    for platform in ("wheel", "legged", "hopper"):
        certificate = optima[platform]["certificate"]
        certificate_sha = optima[platform]["row"]["certificate_sha256"]
        _write_bytes(
            staging / "certificates" / f"{certificate_sha}.json",
            canonical_json_bytes(certificate),
        )

    _write_bytes(
        staging / "raw" / "case-pool.jsonl",
        canonical_jsonl_bytes(
            row for platform in ("wheel", "legged", "hopper") for row in cases[platform]
        ),
    )
    _write_bytes(
        staging / "raw" / "oracle-rows.jsonl", canonical_jsonl_bytes(labels)
    )
    _write_bytes(
        staging / "raw" / "request-pool.jsonl", canonical_jsonl_bytes(request_pool)
    )
    _write_bytes(staging / "reject-ledger.jsonl", b"")
    for relative_name, payload in sorted(raw_sources.items()):
        relative = Path(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe raw source path: {relative_name}")
        _write_bytes(staging / "raw" / relative, payload)
    _write_json(
        staging / "raw" / "hopper-parameter-record.json", hopper_parameter_record
    )
    for name in ("SPECIFICATION.json", "producer-lock.json", "LICENSES.json"):
        _write_bytes(staging / "source" / name, (source_root / name).read_bytes())
    source_tar = deterministic_source_tar(source_root)
    _write_bytes(staging / "source" / "producer-source.tar", source_tar)

    for shard_index, platform in enumerate(("wheel", "legged", "hopper")):
        platform_rows = [row for row in labels if row["platform_kind"] == platform]
        _write_json(
            staging / "shard-manifests" / f"shard-{shard_index:04d}.json",
            {
                "implementation_sha256": implementation_sha,
                "input_sha256": input_contract_sha,
                "row_count": len(platform_rows),
                "row_root_sha256": domain_hash(
                    "g2-shard-rows/v1", canonical_json_bytes(platform_rows)
                ),
                "shard_index": shard_index,
            },
        )

    from .audit_bundle import audit_source_tree

    static_audit = audit_source_tree(source_root)
    if not static_audit["passed"]:
        raise ValueError(f"source isolation audit failed: {static_audit}")
    _write_json(
        staging / "audits" / "technical-isolation.json",
        {
            **static_audit,
            "fixture_only": True,
            "production_process_isolation_verified": False,
        },
    )
    counts = {
        "primitive_labels": len(labels),
        "requests": len(request_rows),
        "small_map_optima": len(optima),
    }
    _write_json(
        staging / "audits" / "cardinality-and-join.json",
        {
            "counts": counts,
            "label_ids_unique": len({row["label_id"] for row in labels}) == len(labels),
            "request_ids_unique": len({row["request_id"] for row in request_rows})
            == len(request_rows),
            "truth_request_hashes_unique": len(
                {row["truth_request_sha256"] for row in request_rows}
            )
            == len(request_rows),
        },
    )
    _write_json(
        staging / "audits" / "selection.json",
        {
            "provider_blind": True,
            "selected_request_sha256": [
                row["truth_request_sha256"] for row in request_rows
            ],
        },
    )
    _write_json(
        staging / "audits" / "reproduction.json",
        {
            "fixture_only": True,
            "fresh_run_comparison_complete": False,
            "formal_evidence_eligible": False,
        },
    )

    manifest_core = {
        "authorization_sha256": _AUTHORIZATION_SHA256,
        "counts": counts,
        "fixture_only": True,
        "formal_evidence_eligible": False,
        "hopper_parameter_record_sha256": hopper_sha,
        "input_contract_sha256": input_contract_sha,
        "producer_implementation_sha256": implementation_sha,
        "profile_or_parameter_record_sha256": profile_hashes,
        "raw_source_sha256": raw_hashes,
        "schema_version": "g2-truth-manifest-core/v1",
        "specification_sha256": specification_sha,
    }
    manifest_core_sha = domain_hash(
        "g2-manifest-core/v1", canonical_json_bytes(manifest_core)
    )
    input_set_id = f"g2t2-fixture-{manifest_core_sha[:24]}"
    _write_json(
        staging / "manifest.json",
        {
            **manifest_core,
            "input_set_id": input_set_id,
            "manifest_core_sha256": manifest_core_sha,
        },
    )
    payload_paths = [
        path.relative_to(staging)
        for path in staging.rglob("*")
        if path.is_file()
        and path.name not in {"source-attestations.json", "truth-freeze.json"}
    ]
    payload_index = content_index(staging, payload_paths)
    payload_root = content_index_root(payload_index)
    attestation = {
        "authorization_sha256": _AUTHORIZATION_SHA256,
        "fixture_only": True,
        "formal_evidence_eligible": False,
        "input_set_id": input_set_id,
        "manifest_core_sha256": manifest_core_sha,
        "organizational_independence": "project_internal",
        "payload_root_sha256": payload_root,
        "status": "pending_o2_signature",
        "technical_independence": "fixture_exercise",
    }
    _write_json(staging / "source-attestations.json", attestation)
    attestation_sha = sha256_bytes(
        (staging / "source-attestations.json").read_bytes()
    )
    freeze = {
        "counts": counts,
        "fixture_only": True,
        "formal_evidence_eligible": False,
        "input_set_id": input_set_id,
        "manifest_core_sha256": manifest_core_sha,
        "payload_index": payload_index,
        "payload_root_sha256": payload_root,
        "publication_order": "data-first-freeze-last",
        "schema_version": "g2-truth-freeze/v1",
        "source_attestations_sha256": attestation_sha,
    }
    _write_json(staging / "truth-freeze.json", freeze)
    staging.rename(output_root)

    from .audit_bundle import audit_bundle

    audit = audit_bundle(output_root)
    if not audit["passed"]:
        raise ValueError(f"fixture bundle audit failed: {audit['reasons']}")
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
