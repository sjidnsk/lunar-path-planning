from __future__ import annotations

import ast
import io
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    canonical_loads,
    content_index,
    content_index_root,
    domain_hash,
    sha256_bytes,
)
from .finite_graph import build_all_optima, verify_optimum_record
from .generate_cases import generate_all_cases
from .generate_requests import (
    _graph_from_raw_source,
    build_repeat_mapping,
    generate_raw_request_sources,
    select_requests,
    solve_raw_request_sources,
    terrain_npz_bytes,
)
from .models import make_case_identity
from .oracle_hopper import evaluate_hopper, validate_hopper_parameter_record
from .oracle_legged import evaluate_legged
from .oracle_wheel import evaluate_wheel


_FORBIDDEN_IMPORT_PREFIXES = (
    "path_planner",
    "lunar_exploration_ppo",
    "xunce_mid_dual_g2_inputs",
)
_EXPECTED_CARDINALITIES = {
    "primitive_labels": 10002,
    "raw_request_pool": 1056,
    "repeat_mapping": 645,
    "requests": 129,
    "small_map_optima": 3,
}


def audit_expected_cardinalities(
    counts: dict[str, int],
) -> list[str]:
    reasons: list[str] = []
    for field, expected in _EXPECTED_CARDINALITIES.items():
        actual = counts.get(field)
        if actual != expected:
            reasons.append(
                f"{field} cardinality mismatch: expected {expected}, got {actual}"
            )
    return reasons


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_bytes().splitlines(), start=1):
        if not raw_line:
            continue
        value = canonical_loads(raw_line)
        if not isinstance(value, dict):
            raise ValueError(f"{path.name}:{line_number} is not an object")
        rows.append(value)
    return rows


def _audit_python_sources(
    source_items: Iterable[tuple[str, bytes]],
) -> dict[str, Any]:
    forbidden_imports: list[str] = []
    forbidden_dynamic_calls: list[str] = []
    source_files: list[dict[str, Any]] = []
    for relative, data in sorted(source_items):
        if not relative.endswith(".py") or "__pycache__" in Path(relative).parts:
            continue
        source_files.append(
            {
                "relative_path": relative,
                "sha256": sha256_bytes(data),
            }
        )
        tree = ast.parse(data.decode("utf-8"), filename=relative)
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            for module in modules:
                if module.startswith(_FORBIDDEN_IMPORT_PREFIXES):
                    forbidden_imports.append(f"{relative}:{module}")
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in {
                    "__import__",
                    "exec",
                    "eval",
                }:
                    forbidden_dynamic_calls.append(f"{relative}:{node.func.id}")
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "importlib"
                ):
                    forbidden_dynamic_calls.append(
                        f"{relative}:importlib.{node.func.attr}"
                    )
    forbidden_imports.sort()
    forbidden_dynamic_calls.sort()
    return {
        "forbidden_dynamic_calls": forbidden_dynamic_calls,
        "forbidden_imports": forbidden_imports,
        "passed": not forbidden_imports and not forbidden_dynamic_calls,
        "source_files": source_files,
        "static_audit_schema_version": "g2-static-isolation-audit/v1",
    }


def audit_source_tree(source_root: Path) -> dict[str, Any]:
    items = [
        (path.relative_to(source_root).as_posix(), path.read_bytes())
        for path in source_root.rglob("*.py")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    return _audit_python_sources(items)


def _source_tar_items(payload: bytes) -> dict[str, bytes]:
    items: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        for member in archive.getmembers():
            path = Path(member.name)
            if (
                not member.isfile()
                or path.is_absolute()
                or ".." in path.parts
                or member.name in items
            ):
                raise ValueError(f"unsafe producer source tar member: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"unreadable producer source tar member: {member.name}")
            items[member.name] = stream.read()
    return items


def _implementation_sha_from_items(items: Mapping[str, bytes]) -> str:
    parts = [
        relative.encode("utf-8") + b"\0" + items[relative]
        for relative in sorted(items)
    ]
    return domain_hash("g2-producer-implementation/v2", *parts)


def _expected_labels(
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
    case_envelope_sha = domain_hash(
        "g2-case-envelope/v2", specification_sha256.encode("ascii")
    )
    rows: list[dict[str, Any]] = []
    for platform in ("wheel", "legged", "hopper"):
        profile_id = (
            hopper_parameter_record["parameter_set_id"]
            if platform == "hopper"
            else f"g2-independent-{platform}-profile/v2"
        )
        for source_row in cases[platform]:
            case = source_row["case"]
            decision = evaluator[platform](case)
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


def _compare_bytes(
    reasons: list[str],
    *,
    path: Path,
    expected: bytes,
    label: str,
) -> None:
    try:
        actual = path.read_bytes()
    except OSError as error:
        reasons.append(f"{label} missing: {error}")
        return
    if actual != expected:
        reasons.append(f"{label} byte mismatch")


def _strong_metadata_checks(
    *,
    freeze: dict[str, Any],
    manifest: dict[str, Any],
    attestation: dict[str, Any],
    attestation_bytes: bytes,
    reasons: list[str],
) -> None:
    for name, value in (
        ("truth-freeze", freeze),
        ("manifest", manifest),
        ("source-attestations", attestation),
    ):
        if value.get("formal_evidence_eligible") is not False:
            reasons.append(f"{name} formal_evidence_eligible must be false")
    if attestation.get("status") != "pending_o2_signature":
        reasons.append("source-attestations status must remain pending_o2_signature")
    fixture_only = bool(freeze.get("fixture_only"))
    expected_independence = "fixture_exercise" if fixture_only else "T2_candidate"
    if attestation.get("technical_independence") != expected_independence:
        reasons.append("source-attestations technical_independence mismatch")
    for field in (
        "input_set_id",
        "manifest_core_sha256",
        "payload_root_sha256",
    ):
        if attestation.get(field) != freeze.get(field):
            reasons.append(f"attestation/freeze {field} mismatch")
    if sha256_bytes(attestation_bytes) != freeze.get(
        "source_attestations_sha256"
    ):
        reasons.append("source-attestations SHA-256 drift")
    if manifest.get("input_set_id") != freeze.get("input_set_id"):
        reasons.append("manifest/freeze input_set_id mismatch")
    if manifest.get("manifest_core_sha256") != freeze.get(
        "manifest_core_sha256"
    ):
        reasons.append("manifest/freeze manifest_core_sha256 mismatch")
    if manifest.get("counts") != freeze.get("counts"):
        reasons.append("manifest/freeze counts mismatch")
    if manifest.get("preflight_sha256") != freeze.get("preflight_sha256"):
        reasons.append("manifest/freeze preflight mismatch")


def audit_bundle(bundle_root: Path) -> dict[str, Any]:
    reasons: list[str] = []
    try:
        freeze = canonical_loads((bundle_root / "truth-freeze.json").read_bytes())
        manifest = canonical_loads((bundle_root / "manifest.json").read_bytes())
        attestation_bytes = (bundle_root / "source-attestations.json").read_bytes()
        attestation = canonical_loads(attestation_bytes)
    except (OSError, ValueError, KeyError) as error:
        return {
            "counts": {},
            "passed": False,
            "reasons": [f"bundle metadata unreadable: {error}"],
        }
    _strong_metadata_checks(
        freeze=freeze,
        manifest=manifest,
        attestation=attestation,
        attestation_bytes=attestation_bytes,
        reasons=reasons,
    )

    try:
        indexed_paths = [Path(row["relative_path"]) for row in freeze["payload_index"]]
        recomputed_index = content_index(bundle_root, indexed_paths)
    except (OSError, KeyError, TypeError, ValueError) as error:
        reasons.append(f"payload index unreadable: {error}")
        recomputed_index = []
    if recomputed_index != freeze.get("payload_index"):
        reasons.append("payload index mismatch")
    if recomputed_index and content_index_root(recomputed_index) != freeze.get(
        "payload_root_sha256"
    ):
        reasons.append("payload root mismatch")
    actual_payload_paths = sorted(
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*")
        if path.is_file()
        and path.name not in {"truth-freeze.json", "source-attestations.json"}
    )
    indexed_path_names = sorted(
        row["relative_path"] for row in freeze.get("payload_index", [])
    )
    if actual_payload_paths != indexed_path_names:
        reasons.append("payload file set mismatch")

    try:
        source_tar = (bundle_root / "source" / "producer-source.tar").read_bytes()
        source_items = _source_tar_items(source_tar)
        implementation_sha = _implementation_sha_from_items(source_items)
        source_static = _audit_python_sources(source_items.items())
        specification_bytes = source_items["SPECIFICATION.json"]
        specification = canonical_loads(specification_bytes)
        specification_sha = sha256_bytes(specification_bytes)
        hopper_record = canonical_loads(
            (bundle_root / "raw" / "hopper-parameter-record.json").read_bytes()
        )
        evaluator_sources = {
            "stop_condition": source_items["producer/hopper_stop_evaluator.py"],
            "energy_model": source_items["producer/hopper_energy_evaluator.py"],
        }
        validate_hopper_parameter_record(
            hopper_record, evaluator_sources=evaluator_sources
        )
    except (OSError, KeyError, ValueError, tarfile.TarError) as error:
        reasons.append(f"trusted source snapshot invalid: {error}")
        source_items = {}
        source_static = {"passed": False}
        specification = {}
        specification_sha = ""
        implementation_sha = ""
        hopper_record = {}
    if not source_static.get("passed"):
        reasons.append("trusted source snapshot static isolation failed")
    if implementation_sha != manifest.get("producer_implementation_sha256"):
        reasons.append("producer implementation SHA-256 mismatch")
    if specification_sha != manifest.get("specification_sha256"):
        reasons.append("specification SHA-256 mismatch")
    if hopper_record and sha256_bytes(
        canonical_json_bytes(hopper_record)
    ) != manifest.get("hopper_parameter_record_sha256"):
        reasons.append("Hopper parameter record SHA-256 mismatch")

    raw_hashes = manifest.get("raw_source_sha256", {})
    for relative_name, expected_hash in sorted(raw_hashes.items()):
        raw_path = bundle_root / "raw" / Path(relative_name)
        if not raw_path.is_file():
            reasons.append(f"missing raw source: {relative_name}")
        elif sha256_bytes(raw_path.read_bytes()) != expected_hash:
            reasons.append(f"raw source hash mismatch: {relative_name}")
    actual_raw_contract_paths = {
        path.relative_to(bundle_root / "raw").as_posix()
        for path in (bundle_root / "raw").rglob("*")
        if path.is_file()
        and (
            path.name == "project-authorization.md"
            or path.suffix.lower() in {".jp2", ".lbl"}
            or path.parts[-2:-1] == ("hopper",)
        )
    }
    if actual_raw_contract_paths != set(raw_hashes):
        reasons.append("raw source exact file-set mismatch")

    try:
        lola_provenance = canonical_loads(
            (bundle_root / "raw" / "lola-provenance.json").read_bytes()
        )
        if (bundle_root / "raw" / "lola-roi-records.jsonl").is_file():
            lola_provenance["roi_records"] = _jsonl(
                bundle_root / "raw" / "lola-roi-records.jsonl"
            )
    except (OSError, ValueError) as error:
        reasons.append(f"LOLA provenance unreadable: {error}")
        lola_provenance = {}

    expected_labels: list[dict[str, Any]] = []
    expected_optima: dict[str, dict[str, Any]] = {}
    expected_raw_sources: list[dict[str, Any]] = []
    expected_pool: list[dict[str, Any]] = []
    expected_requests: list[dict[str, Any]] = []
    expected_repeat_mapping: list[dict[str, Any]] = []
    if specification and hopper_record and lola_provenance:
        hopper_sha = sha256_bytes(canonical_json_bytes(hopper_record))
        profile_hashes = {
            "wheel": domain_hash("g2-independent-profile/v2", b"wheel"),
            "legged": domain_hash("g2-independent-profile/v2", b"legged"),
            "hopper": hopper_sha,
        }
        if profile_hashes != manifest.get("profile_or_parameter_record_sha256"):
            reasons.append("profile/parameter record hash map mismatch")
        try:
            cases = generate_all_cases(
                specification, hopper_parameter_record=hopper_record
            )
            expected_labels = _expected_labels(
                cases,
                specification_sha256=specification_sha,
                implementation_sha256=implementation_sha,
                profile_hashes=profile_hashes,
                hopper_parameter_record=hopper_record,
            )
            _compare_bytes(
                reasons,
                path=bundle_root / "raw" / "case-pool.jsonl",
                expected=canonical_jsonl_bytes(
                    row
                    for platform in ("wheel", "legged", "hopper")
                    for row in cases[platform]
                ),
                label="raw case pool regeneration",
            )
            labels_bytes = canonical_jsonl_bytes(expected_labels)
            _compare_bytes(
                reasons,
                path=bundle_root / "primitive-labels.jsonl",
                expected=labels_bytes,
                label="primitive label regeneration",
            )
            _compare_bytes(
                reasons,
                path=bundle_root / "raw" / "oracle-rows.jsonl",
                expected=labels_bytes,
                label="raw oracle row regeneration",
            )

            expected_optima = build_all_optima(
                specification, hopper_parameter_record=hopper_record
            )
            _compare_bytes(
                reasons,
                path=bundle_root / "small-map-optima.jsonl",
                expected=canonical_jsonl_bytes(
                    expected_optima[platform]["row"]
                    for platform in ("wheel", "legged", "hopper")
                ),
                label="small-map optimum regeneration",
            )
            for artifact in expected_optima.values():
                certificate = artifact["certificate"]
                certificate_sha = artifact["row"]["certificate_sha256"]
                _compare_bytes(
                    reasons,
                    path=bundle_root
                    / "certificates"
                    / f"{certificate_sha}.json",
                    expected=canonical_json_bytes(certificate),
                    label=f"small-map certificate {certificate_sha}",
                )
                if not verify_optimum_record(artifact):
                    reasons.append("regenerated optimum certificate invalid")

            expected_raw_sources = generate_raw_request_sources(
                specification, lola_provenance=lola_provenance
            )
            _compare_bytes(
                reasons,
                path=bundle_root / "raw" / "request-source-pool.jsonl",
                expected=canonical_jsonl_bytes(expected_raw_sources),
                label="truth-blind request source regeneration",
            )
            expected_pool = solve_raw_request_sources(
                expected_raw_sources,
                specification=specification,
                profile_record_sha256=profile_hashes,
                hopper_parameter_record=hopper_record,
            )
            _compare_bytes(
                reasons,
                path=bundle_root / "raw" / "request-pool.jsonl",
                expected=canonical_jsonl_bytes(
                    {
                        key: value
                        for key, value in row.items()
                        if key != "truth_certificate"
                    }
                    for row in expected_pool
                ),
                label="raw request pool regeneration",
            )
            expected_requests = select_requests(expected_pool, specification)
            _compare_bytes(
                reasons,
                path=bundle_root / "requests.jsonl",
                expected=canonical_jsonl_bytes(
                    {
                        key: value
                        for key, value in row.items()
                        if key not in {"truth_certificate", "terrain_arrays"}
                    }
                    for row in expected_requests
                ),
                label="selected request regeneration",
            )
            expected_repeat_mapping = build_repeat_mapping(
                expected_requests,
                input_set_id=str(freeze["input_set_id"]),
            )
            _compare_bytes(
                reasons,
                path=bundle_root / "repeat-mapping.jsonl",
                expected=canonical_jsonl_bytes(expected_repeat_mapping),
                label="repeat mapping regeneration",
            )
        except (KeyError, OSError, TypeError, ValueError) as error:
            reasons.append(f"raw-to-truth regeneration failed: {error}")

    for request in expected_pool:
        certificate_sha = request["truth_certificate_sha256"]
        _compare_bytes(
            reasons,
            path=bundle_root / "certificates" / f"{certificate_sha}.json",
            expected=canonical_json_bytes(request["truth_certificate"]),
            label=f"request certificate {certificate_sha}",
        )
        terrain_payload = terrain_npz_bytes(
            request["terrain_provenance"],
            terrain_arrays=request["terrain_arrays"],
        )
        _compare_bytes(
            reasons,
            path=bundle_root / "terrain" / f"{request['terrain_sha256']}.npz",
            expected=terrain_payload,
            label=f"request terrain {request['terrain_sha256']}",
        )

    graph_template_payloads: dict[str, bytes] = {}
    graph_cache: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in expected_raw_sources:
        topology_sha = domain_hash(
            "g2-request-topology/v1",
            canonical_json_bytes(raw["terrain_arrays"]["cell_class"]),
        )
        key = (str(raw["platform_kind"]), topology_sha)
        if key not in graph_cache:
            graph_cache[key] = _graph_from_raw_source(
                raw, hopper_parameter_record=hopper_record
            )[0]
        graph = graph_cache[key]
        graph_sha = domain_hash(
            "g2-request-graph-template/v1", canonical_json_bytes(graph)
        )
        graph_template_payloads[graph_sha] = canonical_json_bytes(graph)
    for graph_sha, payload in graph_template_payloads.items():
        _compare_bytes(
            reasons,
            path=bundle_root
            / "raw"
            / "graph-templates"
            / f"{graph_sha}.json",
            expected=payload,
            label=f"request graph template {graph_sha}",
        )
    actual_graph_templates = {
        path.stem
        for path in (bundle_root / "raw" / "graph-templates").glob("*.json")
    }
    if actual_graph_templates != set(graph_template_payloads):
        reasons.append("request graph template file-set mismatch")

    try:
        labels = _jsonl(bundle_root / "primitive-labels.jsonl")
        optima = _jsonl(bundle_root / "small-map-optima.jsonl")
        requests = _jsonl(bundle_root / "requests.jsonl")
        raw_pool = _jsonl(bundle_root / "raw" / "request-pool.jsonl")
        repeat_mapping = _jsonl(bundle_root / "repeat-mapping.jsonl")
    except (OSError, ValueError) as error:
        reasons.append(f"primary row artifact unreadable: {error}")
        labels, optima, requests, raw_pool, repeat_mapping = [], [], [], [], []
    counts = {
        "primitive_labels": len(labels),
        "raw_request_pool": len(raw_pool),
        "repeat_mapping": len(repeat_mapping),
        "requests": len(requests),
        "small_map_optima": len(optima),
    }
    reasons.extend(audit_expected_cardinalities(counts))
    if counts != freeze.get("counts"):
        reasons.append(f"freeze count mismatch: {counts}")
    if len({row.get("label_id") for row in labels}) != len(labels):
        reasons.append("duplicate label_id")
    if len({row.get("case_sha256") for row in labels}) != len(labels):
        reasons.append("duplicate case_sha256")
    if Counter(row.get("platform_kind") for row in labels) != Counter(
        {"wheel": 3334, "legged": 3334, "hopper": 3334}
    ):
        reasons.append("primitive platform quota mismatch")
    hopper_stop = sum(
        row.get("platform_kind") == "hopper"
        and row.get("oracle_reason_code") == "G2I_H_STOP"
        for row in labels
    )
    if hopper_stop != 48:
        reasons.append(f"Hopper 3.0 m/s stop reject count mismatch: {hopper_stop}")
    if len({row.get("truth_request_sha256") for row in raw_pool}) != len(raw_pool):
        reasons.append("duplicate raw truth_request_sha256")
    if len({row.get("request_id") for row in requests}) != len(requests):
        reasons.append("duplicate request_id")
    if len({row.get("truth_request_sha256") for row in requests}) != len(requests):
        reasons.append("duplicate selected truth_request_sha256")
    if Counter(row.get("platform_kind") for row in requests) != Counter(
        {"wheel": 43, "legged": 43, "hopper": 43}
    ):
        reasons.append("selected request platform quota mismatch")
    reachable = sum(bool(row.get("oracle_reachable")) for row in requests)
    if (reachable, len(requests) - reachable) != (114, 15):
        reasons.append("selected request reachable/unreachable quota mismatch")
    mapping_keys = {
        (
            row.get("platform_kind"),
            row.get("request_id"),
            row.get("repeat_index"),
        )
        for row in repeat_mapping
    }
    if len(mapping_keys) != 645:
        reasons.append("repeat mapping unique key mismatch")
    if Counter(row.get("repeat_index") for row in repeat_mapping) != Counter(
        {index: 129 for index in range(5)}
    ):
        reasons.append("repeat mapping per-repeat quota mismatch")
    if Counter(row.get("platform_kind") for row in repeat_mapping) != Counter(
        {"wheel": 215, "legged": 215, "hopper": 215}
    ):
        reasons.append("repeat mapping per-platform quota mismatch")
    if any("provider_request_sha256" in row for row in requests):
        reasons.append("provider request hash leaked into truth source")

    isolation_path = bundle_root / "audits" / "technical-isolation.json"
    try:
        isolation = canonical_loads(isolation_path.read_bytes())
        if isolation.get("passed") is not True:
            reasons.append("technical static isolation audit failed")
        if not bool(freeze.get("fixture_only")):
            if isolation.get("production_process_isolation_verified") is not True:
                reasons.append("production process isolation was not verified")
            preflight = canonical_loads(
                (bundle_root / "audits" / "preflight.json").read_bytes()
            )
            if preflight.get("passed") is not True:
                reasons.append("production preflight did not pass")
            if preflight.get("preflight_sha256") != freeze.get(
                "preflight_sha256"
            ):
                reasons.append("production preflight hash binding mismatch")
    except (OSError, ValueError) as error:
        reasons.append(f"technical isolation audit unreadable: {error}")
    return {"counts": counts, "passed": not reasons, "reasons": reasons}


def compare_bundle_bytes(first_root: Path, second_root: Path) -> dict[str, Any]:
    first_paths = sorted(
        path.relative_to(first_root).as_posix()
        for path in first_root.rglob("*")
        if path.is_file()
    )
    second_paths = sorted(
        path.relative_to(second_root).as_posix()
        for path in second_root.rglob("*")
        if path.is_file()
    )
    mismatched = sorted(set(first_paths).symmetric_difference(second_paths))
    for relative_path in sorted(set(first_paths).intersection(second_paths)):
        if (first_root / relative_path).read_bytes() != (
            second_root / relative_path
        ).read_bytes():
            mismatched.append(relative_path)
    return {
        "matched": not mismatched,
        "mismatched_paths": sorted(set(mismatched)),
        "path_count": len(set(first_paths).union(second_paths)),
    }
