from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from typing import Mapping


_FORBIDDEN_IMPORT_TOKENS = (
    "path_planner",
    "lunar_exploration_ppo",
    "xunce_mid_dual_g2_inputs",
)
_ENVIRONMENT_ALLOWLIST = (
    "PATH",
    "PYTHONHASHSEED",
    "PYTHONPATH",
    "PYTHONDONTWRITEBYTECODE",
    "TEMP",
    "TMP",
)


def phase_names() -> tuple[str, ...]:
    return (
        "preflight",
        "cases",
        "labels",
        "optima",
        "requests",
        "package",
        "audit",
        "reproduce",
        "all",
    )


def collect_isolation_evidence(
    *,
    project_root: Path,
    environment: Mapping[str, str],
) -> dict[str, object]:
    forbidden_loaded: list[str] = []
    module_origins: list[dict[str, str]] = []
    for name, module in sorted(sys.modules.items()):
        origin = str(getattr(module, "__file__", "") or "")
        lowered = f"{name}|{origin}".casefold().replace("\\", "/")
        if any(token in lowered for token in _FORBIDDEN_IMPORT_TOKENS):
            forbidden_loaded.append(name)
        if origin:
            module_origins.append({"module": name, "origin": origin})

    resolved_project = project_root.resolve()
    contaminated_sys_path: list[str] = []
    for entry in sys.path:
        if not entry:
            continue
        try:
            resolved_entry = Path(entry).resolve()
        except OSError:
            continue
        if resolved_entry == resolved_project or resolved_project in resolved_entry.parents:
            contaminated_sys_path.append(str(resolved_entry))

    evidence = {
        "command": list(sys.argv),
        "cwd": str(Path.cwd().resolve()),
        "environment": {
            key: environment[key]
            for key in _ENVIRONMENT_ALLOWLIST
            if key in environment
        },
        "forbidden_loaded_modules": forbidden_loaded,
        "module_origins": module_origins,
        "passed": not forbidden_loaded and not contaminated_sys_path,
        "platform": platform.platform(),
        "project_path_entries": contaminated_sys_path,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "sys_path": list(sys.path),
    }
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Standalone deterministic reduced-G2 truth producer"
    )
    parser.add_argument("phase", choices=phase_names())
    parser.add_argument("--project-root", default=os.environ.get("G2_PROJECT_ROOT", "__unset__"))
    parser.add_argument("--json-output")
    parser.add_argument("--source-root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--output-root")
    parser.add_argument("--bundle-root")
    parser.add_argument("--first-root")
    parser.add_argument("--second-root")
    parser.add_argument("--hopper-record")
    parser.add_argument("--lola-provenance")
    parser.add_argument("--raw-source", action="append", default=[])
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args(argv)

    def emit(value: dict[str, object]) -> None:
        from producer.canonical import canonical_json_bytes

        payload = canonical_json_bytes(value) + b"\n"
        if args.json_output:
            target = Path(args.json_output)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        else:
            sys.stdout.buffer.write(payload)

    if args.phase == "preflight":
        evidence = collect_isolation_evidence(
            project_root=Path(args.project_root),
            environment=os.environ,
        )
        emit(evidence)
        return 0 if evidence["passed"] else 2

    if args.phase == "audit":
        if not args.bundle_root:
            parser.error("audit requires --bundle-root")
        from producer.audit_bundle import audit_bundle

        result = audit_bundle(Path(args.bundle_root))
        emit(result)
        return 0 if result["passed"] else 2

    if args.phase == "reproduce":
        if not args.first_root or not args.second_root:
            parser.error("reproduce requires --first-root and --second-root")
        from producer.audit_bundle import compare_bundle_bytes

        result = compare_bundle_bytes(Path(args.first_root), Path(args.second_root))
        emit(result)
        return 0 if result["matched"] else 2

    if not args.fixture:
        parser.error(
            "generation phases require --fixture in this implementation handoff; "
            "production candidate publication is intentionally disabled"
        )
    if not args.hopper_record:
        parser.error("generation phases require --hopper-record")
    source_root = Path(args.source_root)
    hopper_record = json.loads(Path(args.hopper_record).read_text(encoding="utf-8"))
    specification = json.loads(
        (source_root / "SPECIFICATION.json").read_text(encoding="utf-8")
    )

    if args.phase in {"package", "all"}:
        if not args.output_root or not args.lola_provenance:
            parser.error(
                "package/all require --output-root and --lola-provenance"
            )
        raw_sources: dict[str, bytes] = {}
        for raw_argument in args.raw_source:
            if "=" not in raw_argument:
                parser.error("--raw-source must be RELATIVE_NAME=PATH")
            relative_name, source_path = raw_argument.split("=", 1)
            raw_sources[relative_name] = Path(source_path).read_bytes()
        lola = json.loads(Path(args.lola_provenance).read_text(encoding="utf-8"))
        from producer.package_bundle import build_fixture_bundle

        result = build_fixture_bundle(
            Path(args.output_root),
            source_root=source_root,
            hopper_parameter_record=hopper_record,
            raw_sources=raw_sources,
            lola_provenance=lola,
        )
        emit(result)
        return 0

    if not args.output_root:
        parser.error(f"{args.phase} requires --output-root")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    from producer.canonical import (
        canonical_json_bytes,
        canonical_jsonl_bytes,
        domain_hash,
        sha256_bytes,
    )
    from producer.generate_cases import generate_all_cases

    cases = generate_all_cases(
        specification, hopper_parameter_record=hopper_record
    )
    if args.phase == "cases":
        rows = [
            row
            for platform_kind in ("wheel", "legged", "hopper")
            for row in cases[platform_kind]
        ]
        (output_root / "case-pool.jsonl").write_bytes(canonical_jsonl_bytes(rows))
        emit({"phase": "cases", "row_count": len(rows)})
        return 0
    profile_hashes = {
        "wheel": domain_hash("g2-independent-profile/v1", b"wheel"),
        "legged": domain_hash("g2-independent-profile/v1", b"legged"),
        "hopper": sha256_bytes(canonical_json_bytes(hopper_record)),
    }
    if args.phase == "labels":
        from producer.package_bundle import _primitive_labels, producer_implementation_sha256

        labels = _primitive_labels(
            cases,
            specification_sha256=sha256_bytes(
                (source_root / "SPECIFICATION.json").read_bytes()
            ),
            implementation_sha256=producer_implementation_sha256(source_root),
            profile_hashes=profile_hashes,
            hopper_parameter_record=hopper_record,
        )
        (output_root / "primitive-labels.jsonl").write_bytes(
            canonical_jsonl_bytes(labels)
        )
        emit({"phase": "labels", "row_count": len(labels)})
        return 0
    if args.phase == "optima":
        from producer.finite_graph import build_all_optima

        artifacts = build_all_optima(
            specification, hopper_parameter_record=hopper_record
        )
        rows = [artifacts[name]["row"] for name in ("wheel", "legged", "hopper")]
        (output_root / "small-map-optima.jsonl").write_bytes(
            canonical_jsonl_bytes(rows)
        )
        emit({"phase": "optima", "row_count": len(rows)})
        return 0
    if args.phase == "requests":
        if not args.lola_provenance:
            parser.error("requests requires --lola-provenance")
        from producer.generate_requests import generate_request_pool, select_requests

        lola = json.loads(Path(args.lola_provenance).read_text(encoding="utf-8"))
        pool = generate_request_pool(
            specification,
            profile_record_sha256=profile_hashes,
            lola_provenance=lola,
        )
        selected = select_requests(pool, specification)
        (output_root / "request-pool.jsonl").write_bytes(
            canonical_jsonl_bytes(pool)
        )
        (output_root / "requests.jsonl").write_bytes(
            canonical_jsonl_bytes(selected)
        )
        emit({"phase": "requests", "pool_count": len(pool), "selected_count": len(selected)})
        return 0
    parser.error(f"unsupported phase: {args.phase}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
