from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from .canonical import (
    canonical_json_bytes,
    canonical_loads,
    content_index,
    content_index_root,
    sha256_bytes,
)
from .finite_graph import verify_optimum_record
from .generate_requests import terrain_npz_bytes


_FORBIDDEN_IMPORT_PREFIXES = (
    "path_planner",
    "lunar_exploration_ppo",
    "xunce_mid_dual_g2_inputs",
)


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

def audit_source_tree(source_root: Path) -> dict[str, Any]:
    forbidden_imports: list[str] = []
    forbidden_dynamic_calls: list[str] = []
    source_files: list[dict[str, Any]] = []
    for path in sorted(source_root.rglob("*.py"), key=lambda item: item.as_posix()):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(source_root).as_posix()
        data = path.read_bytes()
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
    if freeze.get("formal_evidence_eligible") is not False:
        reasons.append("truth-freeze formal_evidence_eligible must be false")
    if manifest.get("formal_evidence_eligible") is not False:
        reasons.append("manifest formal_evidence_eligible must be false")
    if attestation.get("formal_evidence_eligible") is not False:
        reasons.append("source-attestations formal_evidence_eligible must be false")
    if attestation.get("status") != "pending_o2_signature":
        reasons.append("source-attestations status must remain pending_o2_signature")
    if sha256_bytes(attestation_bytes) != freeze.get("source_attestations_sha256"):
        reasons.append("source-attestations SHA-256 drift")
    if attestation.get("payload_root_sha256") != freeze.get("payload_root_sha256"):
        reasons.append("attestation payload root mismatch")
    if attestation.get("input_set_id") != freeze.get("input_set_id"):
        reasons.append("attestation input set mismatch")

    indexed_paths = [Path(row["relative_path"]) for row in freeze["payload_index"]]
    try:
        recomputed_index = content_index(bundle_root, indexed_paths)
    except OSError as error:
        reasons.append(f"payload file missing: {error}")
        recomputed_index = []
    if recomputed_index != freeze["payload_index"]:
        reasons.append("payload index mismatch")
    if recomputed_index and content_index_root(recomputed_index) != freeze.get(
        "payload_root_sha256"
    ):
        reasons.append("payload root mismatch")
    actual_payload_paths = sorted(
        (
            path.relative_to(bundle_root).as_posix()
            for path in bundle_root.rglob("*")
            if path.is_file()
            and path.name not in {"truth-freeze.json", "source-attestations.json"}
        )
    )
    if actual_payload_paths != sorted(row["relative_path"] for row in freeze["payload_index"]):
        reasons.append("payload file set mismatch")

    try:
        labels = _jsonl(bundle_root / "primitive-labels.jsonl")
        optima = _jsonl(bundle_root / "small-map-optima.jsonl")
        requests = _jsonl(bundle_root / "requests.jsonl")
    except (OSError, ValueError) as error:
        reasons.append(f"primary row artifact unreadable: {error}")
        labels, optima, requests = [], [], []
    counts = {
        "primitive_labels": len(labels),
        "requests": len(requests),
        "small_map_optima": len(optima),
    }
    if counts != freeze.get("counts"):
        reasons.append(f"count mismatch: {counts}")
    if len({row.get("label_id") for row in labels}) != len(labels):
        reasons.append("duplicate label_id")
    if len({row.get("case_sha256") for row in labels}) != len(labels):
        reasons.append("duplicate case_sha256")
    if len({row.get("request_id") for row in requests}) != len(requests):
        reasons.append("duplicate request_id")
    if len({row.get("truth_request_sha256") for row in requests}) != len(requests):
        reasons.append("duplicate truth_request_sha256")
    if any(row.get("oracle_reason_code") == "G2I_NUMERIC_UNDECIDED" for row in labels):
        reasons.append("numeric-undecided label entered quota")
    if any("provider_request_sha256" in row for row in requests):
        reasons.append("provider request hash leaked into truth source")

    for request in requests:
        certificate_sha = request.get("truth_certificate_sha256")
        certificate_path = bundle_root / "certificates" / f"{certificate_sha}.json"
        if not certificate_path.is_file():
            reasons.append(f"dangling request certificate: {certificate_sha}")
        elif sha256_bytes(certificate_path.read_bytes()) != certificate_sha:
            reasons.append(f"request certificate hash mismatch: {certificate_sha}")
        terrain_sha = request.get("terrain_sha256")
        terrain_path = bundle_root / "terrain" / f"{terrain_sha}.npz"
        if not terrain_path.is_file():
            reasons.append(f"dangling request terrain: {terrain_sha}")
        else:
            terrain_bytes = terrain_path.read_bytes()
            if sha256_bytes(terrain_bytes) != terrain_sha:
                reasons.append(f"terrain filename/hash mismatch: {terrain_sha}")
            try:
                if terrain_npz_bytes(request["terrain_provenance"]) != terrain_bytes:
                    reasons.append(f"terrain reproduction mismatch: {terrain_sha}")
            except (ValueError, KeyError) as error:
                reasons.append(f"terrain provenance invalid: {terrain_sha}: {error}")

    for row in optima:
        certificate_path = (
            bundle_root / "certificates" / f"{row.get('certificate_sha256')}.json"
        )
        if not certificate_path.is_file():
            reasons.append(f"dangling optimum certificate: {row.get('case_id')}")
            continue
        try:
            certificate = canonical_loads(certificate_path.read_bytes())
            if not verify_optimum_record({"certificate": certificate, "row": row}):
                reasons.append(f"invalid optimum certificate: {row.get('case_id')}")
        except (ValueError, KeyError) as error:
            reasons.append(f"invalid optimum artifact: {row.get('case_id')}: {error}")

    raw_hashes = manifest.get("raw_source_sha256", {})
    for relative_name, expected_hash in raw_hashes.items():
        raw_path = bundle_root / "raw" / Path(relative_name)
        if not raw_path.is_file():
            reasons.append(f"missing raw source: {relative_name}")
        elif sha256_bytes(raw_path.read_bytes()) != expected_hash:
            reasons.append(f"raw source hash mismatch: {relative_name}")
    isolation_path = bundle_root / "audits" / "technical-isolation.json"
    try:
        isolation = canonical_loads(isolation_path.read_bytes())
        if isolation.get("passed") is not True:
            reasons.append("technical static isolation audit failed")
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
