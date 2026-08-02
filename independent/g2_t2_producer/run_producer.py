from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable, Mapping


_FORBIDDEN_IMPORT_TOKENS = (
    "path_planner",
    "lunar_exploration_ppo",
    "xunce_mid_dual_g2_inputs",
)
_FORBIDDEN_PATH_SEGMENTS = frozenset(
    {
        "lunar-path-planning",
        "lunar_exploration_ppo",
        "path-planner",
        "path_planner",
        "xunce_mid_dual_g2_inputs",
    }
)
_ENVIRONMENT_ALLOWLIST = (
    "PATH",
    "PYTHONHASHSEED",
    "PYTHONNOUSERSITE",
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


def validate_provider_execution_rows(
    rows: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    from producer.models import validate_provider_blind_request

    materialized = list(rows)
    for row in materialized:
        validate_provider_blind_request(row)
    return materialized


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


def _is_relative_to(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _is_reparse_point(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = int(getattr(path.stat(), "st_file_attributes", 0))
    except OSError:
        return False
    return bool(attributes & 0x400)


def _module_name_and_origin(
    name: str, module_or_origin: object
) -> tuple[str, str]:
    if isinstance(module_or_origin, str):
        return name, module_or_origin
    if isinstance(module_or_origin, ModuleType):
        return name, str(getattr(module_or_origin, "__file__", "") or "")
    return name, str(getattr(module_or_origin, "__file__", "") or "")


def _normalized_path_segments(value: object) -> tuple[str, ...]:
    normalized = str(value).casefold().replace("\\", "/")
    strip_characters = " \t\r\n'\"`()[]{};,"
    return tuple(
        segment.strip(strip_characters)
        for segment in normalized.split("/")
        if segment.strip(strip_characters)
    )


def _contains_forbidden_path_segment(value: object) -> bool:
    return bool(
        _FORBIDDEN_PATH_SEGMENTS.intersection(
            _normalized_path_segments(value)
        )
    )


def _is_forbidden_module_name(value: object) -> bool:
    if not isinstance(value, str):
        return False
    normalized = value.casefold()
    return any(
        normalized == token or normalized.startswith(f"{token}.")
        for token in _FORBIDDEN_IMPORT_TOKENS
    )


def _contains_forbidden_pth_import(value: str) -> bool:
    if _contains_forbidden_path_segment(value):
        return True
    identifiers = {
        token
        for token in (
            value.casefold()
            .replace(".", " ")
            .replace(";", " ")
            .replace("(", " ")
            .replace(")", " ")
            .replace(",", " ")
            .split()
        )
    }
    return bool(set(_FORBIDDEN_IMPORT_TOKENS).intersection(identifiers))


def collect_production_isolation_evidence(
    *,
    source_root: Path,
    project_root: Path,
    path_planner_root: Path,
    input_root: Path,
    output_root: Path,
    environment: Mapping[str, str],
    sys_path_entries: Iterable[str] | None = None,
    loaded_modules: Mapping[str, object] | None = None,
    require_d_drive: bool = True,
) -> dict[str, object]:
    named_roots = {
        "source_root": Path(source_root),
        "project_root": Path(project_root),
        "path_planner_root": Path(path_planner_root),
        "input_root": Path(input_root),
        "output_root": Path(output_root),
    }
    reasons: list[str] = []
    if (
        require_d_drive
        and environment.get("PYTHONNOUSERSITE") != "1"
    ):
        reasons.append("PYTHONNOUSERSITE must equal '1'")
    resolved: dict[str, Path] = {}
    for name, raw_path in named_roots.items():
        if not raw_path.is_absolute():
            reasons.append(f"{name} must be absolute")
        resolved_path = raw_path.resolve()
        resolved[name] = resolved_path
        if require_d_drive and resolved_path.drive.casefold() != "d:":
            reasons.append(f"{name} must be on D drive")
        if name != "output_root" and not resolved_path.exists():
            reasons.append(f"{name} missing: {resolved_path}")
        if name == "output_root" and not resolved_path.parent.exists():
            reasons.append(f"output_root parent missing: {resolved_path.parent}")
        if resolved_path.exists() and _is_reparse_point(resolved_path):
            reasons.append(f"{name} is a reparse point")

    source = resolved["source_root"]
    project = resolved["project_root"]
    planner = resolved["path_planner_root"]
    inputs = resolved["input_root"]
    output = resolved["output_root"]
    for name, path in (
        ("source_root", source),
        ("input_root", inputs),
        ("output_root", output),
    ):
        if _is_relative_to(path, project) or _is_relative_to(path, planner):
            reasons.append(f"{name} overlaps project/provider root")
    if _is_relative_to(output, inputs) is False:
        reasons.append("output_root must be beneath input_root")

    entries = list(sys.path if sys_path_entries is None else sys_path_entries)
    contaminated_paths: list[str] = []
    pth_injections: list[dict[str, str]] = []
    for raw_entry in entries:
        if not raw_entry:
            continue
        entry_text = str(raw_entry)
        normalized = entry_text.casefold().replace("\\", "/")
        try:
            entry = Path(entry_text).resolve()
        except OSError:
            reasons.append(f"unresolvable sys.path entry: {entry_text}")
            continue
        if (
            _is_relative_to(entry, project)
            or _is_relative_to(entry, planner)
            or _contains_forbidden_path_segment(normalized)
        ):
            contaminated_paths.append(str(entry))
        if entry.is_dir():
            for pth_path in sorted(entry.glob("*.pth"), key=lambda path: path.name):
                try:
                    lines = pth_path.read_text(
                        encoding="utf-8", errors="strict"
                    ).splitlines()
                except (OSError, UnicodeError) as error:
                    reasons.append(f"unreadable .pth file: {pth_path}: {error}")
                    continue
                for line in lines:
                    candidate_text = line.strip()
                    if not candidate_text or candidate_text.startswith("#"):
                        continue
                    candidate_normalized = candidate_text.casefold().replace(
                        "\\", "/"
                    )
                    if candidate_text.startswith("import "):
                        if _contains_forbidden_pth_import(
                            candidate_normalized
                        ):
                            pth_injections.append(
                                {
                                    "line": candidate_text,
                                    "path": str(pth_path.resolve()),
                                }
                            )
                        continue
                    candidate = Path(candidate_text)
                    if not candidate.is_absolute():
                        candidate = pth_path.parent / candidate
                    candidate = candidate.resolve()
                    if (
                        _is_relative_to(candidate, project)
                        or _is_relative_to(candidate, planner)
                        or _contains_forbidden_path_segment(
                            candidate_normalized
                        )
                        or _contains_forbidden_path_segment(candidate)
                    ):
                        pth_injections.append(
                            {
                                "line": candidate_text,
                                "path": str(pth_path.resolve()),
                            }
                        )
    contaminated_paths = sorted(set(contaminated_paths))
    if contaminated_paths:
        reasons.append("project/provider path visible in sys.path")
    if pth_injections:
        reasons.append("project/provider .pth injection visible")

    modules = sys.modules if loaded_modules is None else loaded_modules
    forbidden_loaded: list[str] = []
    module_origins: list[dict[str, str]] = []
    for name, module_or_origin in sorted(modules.items()):
        module_name, origin_text = _module_name_and_origin(name, module_or_origin)
        if origin_text:
            module_origins.append(
                {"module": module_name, "origin": origin_text}
            )
        origin_path = Path(origin_text).resolve() if origin_text else None
        origin_is_independent = (
            origin_path is not None
            and _is_relative_to(origin_path, source)
        )
        if (
            _is_forbidden_module_name(module_name)
            or (
                origin_path is not None
                and (
                    _is_relative_to(origin_path, project)
                    or _is_relative_to(origin_path, planner)
                    or (
                        not origin_is_independent
                        and _contains_forbidden_path_segment(origin_path)
                    )
                )
            )
        ):
            forbidden_loaded.append(module_name)
    if forbidden_loaded:
        reasons.append("project/provider module origin visible")

    evidence: dict[str, object] = {
        "command": list(sys.argv),
        "cwd": str(Path.cwd().resolve()),
        "environment": {
            key: environment[key]
            for key in _ENVIRONMENT_ALLOWLIST
            if key in environment
        },
        "forbidden_loaded_modules": sorted(set(forbidden_loaded)),
        "module_origins": module_origins,
        "passed": not reasons,
        "path_planner_root": str(planner),
        "platform": platform.platform(),
        "project_path_entries": contaminated_paths,
        "project_root": str(project),
        "pth_injections": pth_injections,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "reasons": reasons,
        "roots": {name: str(path) for name, path in sorted(resolved.items())},
        "schema_version": "g2-production-isolation-preflight/v1",
        "sys_path": entries,
    }
    from producer.canonical import canonical_json_bytes, domain_hash

    evidence["preflight_sha256"] = domain_hash(
        "g2-production-preflight/v1", canonical_json_bytes(evidence)
    )
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Standalone deterministic reduced-G2 truth producer"
    )
    parser.add_argument("phase", choices=phase_names())
    parser.add_argument("--project-root", default=os.environ.get("G2_PROJECT_ROOT", "__unset__"))
    parser.add_argument("--path-planner-root")
    parser.add_argument("--input-root")
    parser.add_argument("--candidate-id")
    parser.add_argument("--json-output")
    parser.add_argument("--source-root", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--output-root")
    parser.add_argument("--bundle-root")
    parser.add_argument("--first-root")
    parser.add_argument("--second-root")
    parser.add_argument("--hopper-record")
    parser.add_argument("--hopper-stop-evaluator")
    parser.add_argument("--hopper-energy-evaluator")
    parser.add_argument("--authorization")
    parser.add_argument("--lola-jp2")
    parser.add_argument("--lola-lbl")
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
        if (
            args.path_planner_root
            and args.input_root
            and args.output_root
            and args.source_root
        ):
            evidence = collect_production_isolation_evidence(
                source_root=Path(args.source_root),
                project_root=Path(args.project_root),
                path_planner_root=Path(args.path_planner_root),
                input_root=Path(args.input_root),
                output_root=Path(args.output_root),
                environment=os.environ,
            )
        else:
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
        if args.phase != "all":
            parser.error("production candidate supports only the all phase")
        required = {
            "--authorization": args.authorization,
            "--candidate-id": args.candidate_id,
            "--hopper-record": args.hopper_record,
            "--input-root": args.input_root,
            "--lola-jp2": args.lola_jp2,
            "--lola-lbl": args.lola_lbl,
            "--output-root": args.output_root,
            "--path-planner-root": args.path_planner_root,
            "--project-root": (
                None if args.project_root == "__unset__" else args.project_root
            ),
            "--source-root": args.source_root,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            parser.error(
                "production all requires explicit arguments: "
                + ", ".join(missing)
            )
        source_root = Path(args.source_root)
        stop_path = Path(
            args.hopper_stop_evaluator
            or source_root / "producer" / "hopper_stop_evaluator.py"
        )
        energy_path = Path(
            args.hopper_energy_evaluator
            or source_root / "producer" / "hopper_energy_evaluator.py"
        )
        preflight = collect_production_isolation_evidence(
            source_root=source_root,
            project_root=Path(args.project_root),
            path_planner_root=Path(args.path_planner_root),
            input_root=Path(args.input_root),
            output_root=Path(args.output_root),
            environment=os.environ,
        )
        if preflight["passed"] is not True:
            emit(preflight)
            return 2
        from producer.canonical import canonical_loads
        from producer.package_bundle import build_production_candidate

        hopper_record_bytes = Path(args.hopper_record).read_bytes()
        hopper_record = canonical_loads(hopper_record_bytes)
        raw_sources = {
            "hopper/HOPPER_PARAMETER_RECORD.json": hopper_record_bytes,
            "hopper/hopper_energy_evaluator.py": energy_path.read_bytes(),
            "hopper/hopper_stop_evaluator.py": stop_path.read_bytes(),
            "lola/LDEM_875S_20M.JP2": Path(args.lola_jp2).read_bytes(),
            "lola/LDEM_875S_20M_JP2.LBL": Path(args.lola_lbl).read_bytes(),
            "project-authorization.md": Path(args.authorization).read_bytes(),
        }
        freeze = build_production_candidate(
            Path(args.output_root),
            candidate_id=str(args.candidate_id),
            source_root=source_root,
            hopper_parameter_record=hopper_record,
            raw_sources=raw_sources,
            preflight_evidence=preflight,
        )
        emit(freeze)
        return 0
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
        "wheel": domain_hash("g2-independent-profile/v2", b"wheel"),
        "legged": domain_hash("g2-independent-profile/v2", b"legged"),
        "hopper": sha256_bytes(canonical_json_bytes(hopper_record)),
    }
    implementation_sha = None
    if args.phase in {"labels", "requests"}:
        from producer.package_bundle import producer_implementation_sha256

        implementation_sha = producer_implementation_sha256(source_root)
    if args.phase == "labels":
        from producer.package_bundle import _primitive_labels

        labels = _primitive_labels(
            cases,
            specification_sha256=sha256_bytes(
                (source_root / "SPECIFICATION.json").read_bytes()
            ),
            implementation_sha256=implementation_sha,
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
        from producer.generate_requests import (
            generate_raw_request_sources,
            provider_blind_request,
            request_graph_from_cache,
            select_requests,
            solve_raw_request_sources,
        )

        lola = json.loads(Path(args.lola_provenance).read_text(encoding="utf-8"))
        raw_request_sources = generate_raw_request_sources(
            specification,
            lola_provenance=lola,
            hopper_parameter_record=hopper_record,
        )
        pool = solve_raw_request_sources(
            raw_request_sources,
            specification=specification,
            profile_record_sha256=profile_hashes,
            producer_implementation_sha256=implementation_sha,
            hopper_parameter_record=hopper_record,
        )
        selected = select_requests(pool, specification)
        raw_by_sha = {
            row["raw_source_sha256"]: row
            for row in raw_request_sources
        }
        graph_cache: dict[
            str, tuple[dict[str, object], str, str]
        ] = {}
        provider_rows: list[dict[str, object]] = []
        sidecar_rows: list[dict[str, object]] = []
        for request in selected:
            raw = raw_by_sha[request["raw_source_sha256"]]
            graph, _, _ = request_graph_from_cache(
                raw,
                profile_record_sha256=profile_hashes[
                    request["platform_kind"]
                ],
                hopper_parameter_record=hopper_record,
                graph_cache=graph_cache,
            )
            blind = provider_blind_request(
                request,
                graph=graph,
                hopper_parameter_record=hopper_record,
            )
            provider_rows.append(blind)
            sidecar_rows.append(
                {
                    "provider_request_id": blind["provider_request_id"],
                    "provider_request_sha256": blind[
                        "provider_request_sha256"
                    ],
                    "schema_version": "g2-truth-request-sidecar/v1",
                    "truth_request": {
                        key: value
                        for key, value in request.items()
                        if key
                        not in {"truth_certificate", "terrain_arrays"}
                    },
                }
            )
        (output_root / "request-pool.jsonl").write_bytes(
            canonical_jsonl_bytes(pool)
        )
        (output_root / "requests.jsonl").write_bytes(
            canonical_jsonl_bytes(provider_rows)
        )
        truth_root = output_root / "truth"
        truth_root.mkdir()
        (truth_root / "request-sidecar.jsonl").write_bytes(
            canonical_jsonl_bytes(sidecar_rows)
        )
        emit({"phase": "requests", "pool_count": len(pool), "selected_count": len(selected)})
        return 0
    parser.error(f"unsupported phase: {args.phase}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
