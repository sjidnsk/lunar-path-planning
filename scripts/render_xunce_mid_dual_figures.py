"""Render publication-formal figures from a completed midterm aggregate root.

The renderer is intentionally downstream-only.  It verifies the aggregate
manifest and all three referenced source manifests before creating an output
directory, then independently recomputes every displayed value from immutable
raw rows.  A blocked or malformed evidence root never produces figure files.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import csv
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import importlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import random
import re
import statistics
import subprocess
import sys
import textwrap
from typing import Any, Callable, Mapping, Sequence
import uuid
import xml.etree.ElementTree as ET

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, __version__ as PILLOW_VERSION  # noqa: E402

import xunce_artifact_io as artifact_io  # noqa: E402
from xunce_mid_dual_artifacts import MidDualRunStore  # noqa: E402


# Editable text and a portable sans-serif fallback are part of the frozen style.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Arial",
    "Helvetica",
    "DejaVu Sans",
    "Liberation Sans",
]
plt.rcParams["svg.fonttype"] = "none"


FIGURE_CONFIG_SCHEMA_VERSION = "xunce-mid-dual-figures-config/v1"
FIGURE_RENDERER_ID = "render_xunce_mid_dual_figures/v1"
FIGURE_MANIFEST_SCHEMA_VERSION = "xunce-mid-dual-figure-manifest/v1"
SOURCE_DATA_SCHEMA_VERSION = "xunce-mid-dual-figure-source-data/v1"
CODE_LINEAGE_SCHEMA_VERSION = "xunce-mid-dual-figure-code-lineage/v1"
SCALE_PROFILE = "midterm_reduced_w8x3_update80/v1"
REDUCED_QUALIFIER = (
    "midterm reduced scale (8×3, update80); not full-scale acceptance"
)
AGGREGATE_SCHEMA_VERSION = "xunce-mid-dual-independent-aggregate/v1"
AGGREGATE_CONFIG_SCHEMA_VERSION = "xunce-mid-dual-aggregate-config/v1"
MANIFEST_SCHEMA_VERSION = "mid-dual-manifest/v1"
FORMAL_AGGREGATE_BASE = "D:/xunce/out/mid_dual/aggregate"
FORMAL_FIGURE_BASE = "D:/xunce/out/mid_dual/figures"
GATES = ("g1", "g2", "g3")
PLATFORMS = ("wheel", "legged", "hopper")
SCALES = ("standard", "kilometer")
REQUEST_CLASSES = ("normal_reachable", "hard_reachable", "unreachable")
TIMING_COMPONENT_FIELDS = (
    "input_validation_ns",
    "platform_instantiation_ns",
    "search_ns",
    "complete_route_validation_ns",
    "result_assembly_ns",
)
FIGURE_FAMILIES = (
    "fig01_dual_gate_overview",
    "fig02_g1_coverage",
    "fig03_g2_timing",
    "fig04_g3_crosscheck",
)
EXPORT_FORMATS = ("svg", "pdf", "png", "tiff")
SOURCE_DATA_FORMATS = ("json", "csv")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_G2_CLASS_COUNTS = {
    ("standard", "normal_reachable"): 23,
    ("standard", "hard_reachable"): 7,
    ("standard", "unreachable"): 3,
    ("kilometer", "normal_reachable"): 6,
    ("kilometer", "hard_reachable"): 2,
    ("kilometer", "unreachable"): 2,
}


class FigureBlocked(ValueError):
    """Stable fail-closed error raised before any figure artifact is written."""


@dataclass(frozen=True)
class VerifiedFigureInput:
    """Immutable projection of manifest-bound evidence used by all figures."""

    aggregate_run_id: str
    aggregate_manifest_sha256: str
    status: str
    run_ids: Mapping[str, str]
    source_manifest_sha256: Mapping[str, str]
    input_hashes: Mapping[str, str]
    aggregate_summary: Mapping[str, Any]
    gates: Mapping[str, Mapping[str, Any]]
    rows: Mapping[str, tuple[dict[str, Any], ...]]


@dataclass(frozen=True)
class CodeLineageSnapshot:
    """Byte-bound project source closure used by one figure attempt."""

    repository_root: Path
    payload: Mapping[str, Any]
    source_bytes: Mapping[str, bytes]
    payload_sha256: str
    renderer_sha256: str


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_json_text(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def _require_exact_keys(
    value: object,
    expected: set[str],
    reason: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise FigureBlocked(reason)
    return value


def _require_finite(value: object, reason: str) -> float:
    if type(value) not in (int, float):
        raise FigureBlocked(reason)
    result = float(value)
    if not math.isfinite(result):
        raise FigureBlocked(reason)
    return result


def _require_nonnegative_int(value: object, reason: str) -> int:
    if type(value) is not int or value < 0:
        raise FigureBlocked(reason)
    return value


def _parse_json_object(payload: bytes, reason: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise FigureBlocked(reason) from exc
    if not isinstance(value, dict):
        raise FigureBlocked(reason)
    return value


def _parse_jsonl_rows(payload: bytes, reason: str) -> list[dict[str, Any]]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise FigureBlocked(reason) from exc
    if not text or not text.endswith("\n"):
        raise FigureBlocked(reason)
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line:
            raise FigureBlocked(reason)
        try:
            value = json.loads(line, parse_constant=reject_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise FigureBlocked(reason) from exc
        if not isinstance(value, dict):
            raise FigureBlocked(reason)
        rows.append(value)
    return rows


def _safe_manifest_path(value: object, reason: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise FigureBlocked(reason)
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or any(part in ("", ".", "..") for part in posix.parts)
    ):
        raise FigureBlocked(reason)
    return value


def _snapshot_completed_root(
    root: Path,
    label: str,
    required_paths: set[str],
) -> tuple[dict[str, Any], dict[str, bytes], str, dict[str, str]]:
    """Verify once, then bind every later read to an immutable byte snapshot."""

    try:
        if MidDualRunStore.verify_manifest(root) is not True:
            raise FigureBlocked(f"{label}_manifest_invalid")
    except FigureBlocked:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise FigureBlocked(
            f"{label}_manifest_invalid:{type(exc).__name__}"
        ) from exc
    manifest_path = root / "manifest.json"
    try:
        manifest_bytes = artifact_io.read_bytes(manifest_path)
    except OSError as exc:
        raise FigureBlocked(f"{label}_manifest_snapshot_missing") from exc
    manifest = _parse_json_object(
        manifest_bytes,
        f"{label}_manifest_snapshot_invalid",
    )
    _require_exact_keys(
        manifest,
        {
            "schema_version",
            "config_sha256",
            "formal_evidence_eligible",
            "artifacts",
        },
        f"{label}_manifest_schema_invalid",
    )
    if (
        manifest["schema_version"] != MANIFEST_SCHEMA_VERSION
        or manifest["formal_evidence_eligible"] is not True
        or not _is_sha256(manifest["config_sha256"])
    ):
        raise FigureBlocked(f"{label}_root_not_completed_eligible")
    entries = manifest["artifacts"]
    if not isinstance(entries, list):
        raise FigureBlocked(f"{label}_manifest_entries_invalid")
    snapshot: dict[str, bytes] = {}
    digests: dict[str, str] = {}
    for entry in entries:
        entry = _require_exact_keys(
            entry,
            {"path", "sha256"},
            f"{label}_manifest_entry_invalid",
        )
        relative = _safe_manifest_path(
            entry["path"],
            f"{label}_manifest_entry_invalid",
        )
        digest = entry["sha256"]
        if not _is_sha256(digest) or relative in snapshot:
            raise FigureBlocked(f"{label}_manifest_entry_invalid")
        try:
            payload = artifact_io.read_bytes(root / relative)
        except OSError as exc:
            raise FigureBlocked(
                f"{label}_immutable_bytes_missing:{relative}"
            ) from exc
        if _sha256_bytes(payload) != digest:
            raise FigureBlocked(
                f"{label}_immutable_bytes_drift:{relative}"
            )
        snapshot[relative] = payload
        digests[relative] = digest
    if not required_paths.issubset(snapshot):
        raise FigureBlocked(f"{label}_manifest_artifact_set_incomplete")
    return manifest, snapshot, _sha256_bytes(manifest_bytes), digests


def _load_figure_config(
    config_path: str | Path,
) -> tuple[dict[str, Any], str]:
    try:
        payload = artifact_io.read_bytes(config_path)
    except OSError as exc:
        raise FigureBlocked("figure_config_missing") from exc
    config = _parse_json_object(payload, "figure_config_invalid_json")
    expected = {
        "schema_version",
        "renderer_id",
        "backend",
        "scale_profile",
        "output_base",
        "required_figure_families",
        "export_formats",
        "source_data_formats",
        "raster_dpi",
        "bootstrap",
        "thresholds",
        "sample_contract",
        "figure_sizes_inches",
        "style",
        "rendering_behavior",
        "canvas_contract",
        "runtime_source_contract",
        "visible_content_scope",
        "claims",
    }
    _require_exact_keys(config, expected, "figure_config_schema_invalid")
    if (
        config["schema_version"] != FIGURE_CONFIG_SCHEMA_VERSION
        or config["renderer_id"] != FIGURE_RENDERER_ID
        or config["backend"] != "python-matplotlib"
        or config["scale_profile"] != SCALE_PROFILE
        or config["output_base"] != FORMAL_FIGURE_BASE
        or config["required_figure_families"] != list(FIGURE_FAMILIES)
        or config["export_formats"] != list(EXPORT_FORMATS)
        or config["source_data_formats"] != list(SOURCE_DATA_FORMATS)
        or config["raster_dpi"] != 600
    ):
        raise FigureBlocked("figure_config_contract_drift")
    bootstrap = _require_exact_keys(
        config["bootstrap"],
        {"seed", "resamples", "confidence", "method"},
        "figure_bootstrap_config_invalid",
    )
    if (
        type(bootstrap["seed"]) is not int
        or type(bootstrap["resamples"]) is not int
        or bootstrap["resamples"] < 2000
        or bootstrap["confidence"] != 0.95
        or bootstrap["method"]
        != "deterministic-percentile-bootstrap-mean-nearest-rank/v1"
    ):
        raise FigureBlocked("figure_bootstrap_config_invalid")
    thresholds = _require_exact_keys(
        config["thresholds"],
        {
            "coverage_midterm",
            "coverage_final",
            "timing_midterm_ms",
            "timing_final_ms",
            "paired_g1_delta_min",
        },
        "figure_threshold_config_invalid",
    )
    if thresholds != {
        "coverage_midterm": 0.8,
        "coverage_final": 0.99,
        "timing_midterm_ms": 2000.0,
        "timing_final_ms": 1000.0,
        "paired_g1_delta_min": -0.01,
    }:
        raise FigureBlocked("figure_threshold_config_invalid")
    samples = _require_exact_keys(
        config["sample_contract"],
        {
            "g1_test_q24",
            "g1_unseen24",
            "g2_platform_count",
            "g2_requests_per_platform",
            "g2_repeats",
            "g2_formal_calls",
            "g3_wheel_episodes",
            "g3_legged_replays",
            "g3_hopper_replays",
        },
        "figure_sample_config_invalid",
    )
    if samples != {
        "g1_test_q24": 24,
        "g1_unseen24": 24,
        "g2_platform_count": 3,
        "g2_requests_per_platform": 43,
        "g2_repeats": 5,
        "g2_formal_calls": 645,
        "g3_wheel_episodes": 10,
        "g3_legged_replays": 3,
        "g3_hopper_replays": 3,
    }:
        raise FigureBlocked("figure_sample_config_invalid")
    sizes = _require_exact_keys(
        config["figure_sizes_inches"],
        set(FIGURE_FAMILIES),
        "figure_size_config_invalid",
    )
    for family in FIGURE_FAMILIES:
        value = sizes[family]
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(type(item) not in (int, float) or item <= 0 for item in value)
        ):
            raise FigureBlocked("figure_size_config_invalid")
    style = _require_exact_keys(
        config["style"],
        {
            "palette_id",
            "colors",
            "font_fallback",
            "base_font_pt",
            "panel_label_pt",
            "axes_linewidth",
            "grid_alpha",
            "white_background",
        },
        "figure_style_config_invalid",
    )
    colors = _require_exact_keys(
        style["colors"],
        {
            "orange",
            "sky_blue",
            "bluish_green",
            "yellow",
            "blue",
            "vermillion",
            "reddish_purple",
            "black",
            "neutral",
            "light_neutral",
        },
        "figure_palette_invalid",
    )
    if (
        style["palette_id"] != "okabe-ito/v1"
        or not all(
            isinstance(value, str)
            and re.fullmatch(r"#[0-9A-F]{6}", value) is not None
            for value in colors.values()
        )
        or style["font_fallback"]
        != ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans"]
        or style["white_background"] is not True
    ):
        raise FigureBlocked("figure_style_config_invalid")
    if (
        any(
            type(style[field]) not in (int, float)
            or not math.isfinite(float(style[field]))
            or float(style[field]) <= 0.0
            for field in (
                "base_font_pt",
                "panel_label_pt",
                "axes_linewidth",
            )
        )
        or type(style["grid_alpha"]) not in (int, float)
        or not 0.0 <= float(style["grid_alpha"]) <= 1.0
    ):
        raise FigureBlocked("figure_style_config_invalid")
    behavior = _require_exact_keys(
        config["rendering_behavior"],
        {"passed", "failed", "blocked"},
        "figure_rendering_behavior_invalid",
    )
    if behavior != {
        "passed": "render",
        "failed": "render",
        "blocked": "reject_without_output",
    }:
        raise FigureBlocked("figure_rendering_behavior_invalid")
    canvas = _require_exact_keys(
        config["canvas_contract"],
        {
            "canonical_bbox",
            "dimension_tolerance_mm",
            "raster_dpi_tolerance",
            "width_mm",
        },
        "figure_canvas_contract_invalid",
    )
    if canvas != {
        "canonical_bbox": "fixed_canvas",
        "dimension_tolerance_mm": 0.2,
        "raster_dpi_tolerance": 1.0,
        "width_mm": 182.88,
    }:
        raise FigureBlocked("figure_canvas_contract_invalid")
    runtime_source = _require_exact_keys(
        config["runtime_source_contract"],
        {
            "schema_version",
            "required_paths",
            "transitive_project_imports_required",
            "snapshot_dirty_untracked",
        },
        "figure_runtime_source_contract_invalid",
    )
    if runtime_source != {
        "schema_version": (
            "xunce-mid-dual-figure-runtime-source-contract/v1"
        ),
        "required_paths": [
            "scripts/render_xunce_mid_dual_figures.py",
            "configs/xunce_mid_dual_figures_v1.json",
            "scripts/run_xunce_mid_dual_aggregate.py",
            "scripts/xunce_mid_dual_artifacts.py",
            "scripts/xunce_artifact_io.py",
            "scripts/xunce_artifact_paths.py",
        ],
        "transitive_project_imports_required": True,
        "snapshot_dirty_untracked": True,
    }:
        raise FigureBlocked("figure_runtime_source_contract_invalid")
    if config["visible_content_scope"] != "result_only/v1":
        raise FigureBlocked("figure_visible_content_scope_invalid")
    claims = _require_exact_keys(
        config["claims"],
        {
            "physical_capability_certified",
            "hardware_certified",
            "legged_kind",
            "hopper_kind",
        },
        "figure_claim_config_invalid",
    )
    if claims != {
        "physical_capability_certified": False,
        "hardware_certified": False,
        "legged_kind": "simulation_proxy",
        "hopper_kind": "simulation_proxy",
    }:
        raise FigureBlocked("figure_claim_config_invalid")
    return config, _sha256_bytes(payload)


def _git_output(
    repository_root: Path,
    arguments: Sequence[str],
    reason: str,
) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
        )
    except (OSError, subprocess.CalledProcessError, UnicodeError) as exc:
        raise FigureBlocked(reason) from exc
    return completed.stdout.strip()


def _path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _project_search_roots(repository_root: Path) -> tuple[Path, ...]:
    candidates = (
        repository_root / "scripts",
        repository_root / "src",
        repository_root / "path-planner" / "src",
        repository_root,
    )
    return tuple(
        candidate.resolve()
        for candidate in candidates
        if candidate.is_dir()
    )


def _module_paths(
    module_name: str,
    *,
    current_path: Path,
    level: int,
    search_roots: Sequence[Path],
) -> tuple[Path, ...]:
    if not isinstance(module_name, str):
        return ()
    current_root = next(
        (
            root
            for root in search_roots
            if _path_within(current_path, root)
        ),
        None,
    )
    parts = [part for part in module_name.split(".") if part]
    if level:
        if current_root is None:
            raise FigureBlocked("runtime_source_relative_import_unresolved")
        package_parts = list(
            current_path.parent.resolve().relative_to(current_root).parts
        )
        parent_hops = level - 1
        if parent_hops > len(package_parts):
            raise FigureBlocked("runtime_source_relative_import_unresolved")
        if parent_hops:
            package_parts = package_parts[:-parent_hops]
        parts = [*package_parts, *parts]
        candidate_roots = (current_root,)
    else:
        candidate_roots = tuple(search_roots)
    if not parts:
        return ()
    resolved: list[Path] = []
    for root in candidate_roots:
        module_file = root.joinpath(*parts).with_suffix(".py")
        package_file = root.joinpath(*parts, "__init__.py")
        selected = (
            module_file
            if module_file.is_file()
            else package_file
            if package_file.is_file()
            else None
        )
        if selected is None:
            continue
        for index in range(1, len(parts)):
            init_path = root.joinpath(*parts[:index], "__init__.py")
            if init_path.is_file() and init_path.resolve() not in resolved:
                resolved.append(init_path.resolve())
        if selected.resolve() not in resolved:
            resolved.append(selected.resolve())
        break
    return tuple(resolved)


def _local_import_paths(
    path: Path,
    payload: bytes,
    search_roots: Sequence[Path],
) -> tuple[Path, ...]:
    try:
        tree = ast.parse(payload.decode("utf-8-sig"), filename=str(path))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise FigureBlocked(
            f"runtime_source_parse_invalid:{path.name}"
        ) from exc
    discovered: set[Path] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                discovered.update(
                    _module_paths(
                        alias.name,
                        current_path=path,
                        level=0,
                        search_roots=search_roots,
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            discovered.update(
                _module_paths(
                    base,
                    current_path=path,
                    level=node.level,
                    search_roots=search_roots,
                )
            )
            for alias in node.names:
                if alias.name == "*":
                    continue
                child = f"{base}.{alias.name}" if base else alias.name
                discovered.update(
                    _module_paths(
                        child,
                        current_path=path,
                        level=node.level,
                        search_roots=search_roots,
                    )
                )
        elif isinstance(node, ast.Call) and node.args:
            function_name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else node.func.id
                if isinstance(node.func, ast.Name)
                else ""
            )
            if function_name not in {"import_module", "_import_local_script"}:
                continue
            module_value = node.args[0]
            if isinstance(module_value, ast.Constant) and isinstance(
                module_value.value,
                str,
            ):
                discovered.update(
                    _module_paths(
                        module_value.value,
                        current_path=path,
                        level=0,
                        search_roots=search_roots,
                    )
                )
    return tuple(sorted(discovered, key=lambda item: item.as_posix()))


def _git_identity_for_file(
    path: Path,
    *,
    repository_root: Path,
    repository_cache: dict[Path, str],
) -> tuple[str, str, str]:
    file_repository_text = _git_output(
        path.parent,
        ["rev-parse", "--show-toplevel"],
        "runtime_source_git_root_unavailable",
    )
    file_repository = Path(file_repository_text).resolve()
    if not _path_within(file_repository, repository_root):
        raise FigureBlocked("runtime_source_git_root_escape")
    head = repository_cache.get(file_repository)
    if head is None:
        head = _git_output(
            file_repository,
            ["rev-parse", "HEAD"],
            "runtime_source_git_head_unavailable",
        )
        if re.fullmatch(r"[0-9a-f]{40}", head) is None:
            raise FigureBlocked("runtime_source_git_head_invalid")
        repository_cache[file_repository] = head
    relative = path.resolve().relative_to(file_repository).as_posix()
    status_text = _git_output(
        file_repository,
        ["status", "--porcelain=v1", "--untracked-files=all", "--", relative],
        "runtime_source_git_status_unavailable",
    )
    if not status_text:
        status = "clean"
    else:
        status_code = status_text[:2]
        status = "untracked" if status_code == "??" else f"dirty:{status_code}"
    repository_relative = (
        "."
        if file_repository == repository_root
        else file_repository.relative_to(repository_root).as_posix()
    )
    return repository_relative, head, status


def _capture_code_lineage(
    *,
    repository_root: str | Path,
    entry_paths: Sequence[str | Path],
    required_relative_paths: set[str],
) -> CodeLineageSnapshot:
    root = Path(repository_root).resolve()
    if not root.is_absolute() or not root.is_dir() or not entry_paths:
        raise FigureBlocked("runtime_source_repository_invalid")
    root_from_git = Path(
        _git_output(
            root,
            ["rev-parse", "--show-toplevel"],
            "runtime_source_git_root_unavailable",
        )
    ).resolve()
    if root_from_git != root:
        raise FigureBlocked("runtime_source_repository_mismatch")
    root_commit = _git_output(
        root,
        ["rev-parse", "HEAD"],
        "runtime_source_git_head_unavailable",
    )
    if re.fullmatch(r"[0-9a-f]{40}", root_commit) is None:
        raise FigureBlocked("runtime_source_git_head_invalid")
    search_roots = _project_search_roots(root)
    queue: list[Path] = []
    for entry in entry_paths:
        candidate = Path(entry).resolve()
        if not _path_within(candidate, root) or not candidate.is_file():
            raise FigureBlocked("runtime_source_entry_missing")
        queue.append(candidate)
    source_bytes_by_path: dict[Path, bytes] = {}
    while queue:
        path = queue.pop(0)
        if path in source_bytes_by_path:
            continue
        try:
            payload = artifact_io.read_bytes(path)
        except OSError as exc:
            raise FigureBlocked("runtime_source_entry_missing") from exc
        source_bytes_by_path[path] = payload
        if path.suffix.lower() == ".py":
            for imported in _local_import_paths(
                path,
                payload,
                search_roots,
            ):
                if imported not in source_bytes_by_path:
                    queue.append(imported)
    source_bytes = {
        path.relative_to(root).as_posix(): payload
        for path, payload in source_bytes_by_path.items()
    }
    missing = required_relative_paths - set(source_bytes)
    if missing:
        raise FigureBlocked(
            "runtime_source_closure_incomplete:"
            + ",".join(sorted(missing))
        )
    first_entry = Path(entry_paths[0]).resolve().relative_to(root).as_posix()
    repository_cache: dict[Path, str] = {root: root_commit}
    file_rows: list[dict[str, Any]] = []
    for relative in sorted(source_bytes):
        path = root / relative
        repository_relative, file_head, status = _git_identity_for_file(
            path,
            repository_root=root,
            repository_cache=repository_cache,
        )
        snapshot_path = (
            f"code_snapshot/{relative}"
            if status != "clean"
            else None
        )
        payload = source_bytes[relative]
        file_rows.append(
            {
                "path": relative,
                "byte_count": len(payload),
                "sha256": _sha256_bytes(payload),
                "git_repository": repository_relative,
                "git_head": file_head,
                "git_status": status,
                "snapshot_path": snapshot_path,
            }
        )
    payload = {
        "schema_version": CODE_LINEAGE_SCHEMA_VERSION,
        "root_commit": root_commit,
        "renderer_path": first_entry,
        "renderer_sha256": _sha256_bytes(source_bytes[first_entry]),
        "required_paths": sorted(required_relative_paths),
        "transitive_project_imports_complete": True,
        "files": file_rows,
    }
    payload_sha256 = _sha256_bytes(_canonical_json_bytes(payload))
    return CodeLineageSnapshot(
        repository_root=root,
        payload=payload,
        source_bytes=source_bytes,
        payload_sha256=payload_sha256,
        renderer_sha256=str(payload["renderer_sha256"]),
    )


def _verify_code_lineage(snapshot: CodeLineageSnapshot) -> None:
    if (
        snapshot.payload.get("schema_version")
        != CODE_LINEAGE_SCHEMA_VERSION
        or snapshot.payload_sha256
        != _sha256_bytes(_canonical_json_bytes(snapshot.payload))
        or snapshot.renderer_sha256
        != snapshot.payload.get("renderer_sha256")
    ):
        raise FigureBlocked("code_lineage_payload_invalid")
    rows = snapshot.payload.get("files")
    if not isinstance(rows, list):
        raise FigureBlocked("code_lineage_payload_invalid")
    expected_paths = set(snapshot.source_bytes)
    if {
        row.get("path")
        for row in rows
        if isinstance(row, Mapping)
    } != expected_paths:
        raise FigureBlocked("code_lineage_payload_invalid")
    for relative, expected in snapshot.source_bytes.items():
        path = snapshot.repository_root / relative
        try:
            current = artifact_io.read_bytes(path)
        except OSError as exc:
            raise FigureBlocked(
                f"code_lineage_missing:{relative}"
            ) from exc
        if current != expected:
            raise FigureBlocked(f"code_lineage_drift:{relative}")


def _capture_runtime_code_lineage(
    config_path: str | Path,
    aggregate_module: Any,
) -> CodeLineageSnapshot:
    renderer_path = Path(__file__).resolve()
    repository_root = Path(
        _git_output(
            renderer_path.parent,
            ["rev-parse", "--show-toplevel"],
            "runtime_source_git_root_unavailable",
        )
    ).resolve()
    config, _ = _load_figure_config(config_path)
    required = set(config["runtime_source_contract"]["required_paths"])
    aggregate_path_value = getattr(aggregate_module, "__file__", None)
    artifacts_module = sys.modules.get(MidDualRunStore.__module__)
    artifacts_path_value = getattr(artifacts_module, "__file__", None)
    if not isinstance(aggregate_path_value, str) or not isinstance(
        artifacts_path_value,
        str,
    ):
        raise FigureBlocked("runtime_source_module_path_missing")
    entry_paths = (
        renderer_path,
        Path(config_path).resolve(),
        Path(aggregate_path_value).resolve(),
        Path(artifacts_path_value).resolve(),
        Path(artifact_io.__file__).resolve(),
        repository_root / "scripts" / "xunce_artifact_paths.py",
    )
    snapshot = _capture_code_lineage(
        repository_root=repository_root,
        entry_paths=entry_paths,
        required_relative_paths=required,
    )
    _verify_code_lineage(snapshot)
    return snapshot


def _validate_aggregate_config(
    config: Mapping[str, Any],
    manifest_config_sha256: str,
    aggregate_root: Path,
    *,
    enforce_formal_paths: bool,
) -> None:
    expected = {
        "actual_sample_counts",
        "output_root",
        "reference_sample_counts",
        "required_phase_ids",
        "scale_profile",
        "schema_version",
        "source_contracts",
        "run_id",
        "g1_root",
        "g2_root",
        "g3_root",
        "config_sha256",
    }
    _require_exact_keys(config, expected, "aggregate_config_schema_invalid")
    config_without_sha = dict(config)
    config_sha256 = config_without_sha.pop("config_sha256")
    source_contracts = config["source_contracts"]
    if (
        config["schema_version"] != AGGREGATE_CONFIG_SCHEMA_VERSION
        or config["scale_profile"] != SCALE_PROFILE
        or config["required_phase_ids"] != ["p01"]
        or config_sha256 != manifest_config_sha256
        or config_sha256 != _sha256_bytes(
            _canonical_json_bytes(config_without_sha)
        )
        or not isinstance(source_contracts, Mapping)
        or set(source_contracts) != set(GATES)
        or not isinstance(config["run_id"], str)
        or RUN_ID_PATTERN.fullmatch(config["run_id"]) is None
    ):
        raise FigureBlocked("aggregate_config_contract_drift")
    if any(
        not isinstance(source_contracts[gate_id], Mapping)
        or "output_base" not in source_contracts[gate_id]
        for gate_id in GATES
    ):
        raise FigureBlocked("aggregate_source_contract_invalid")
    if config["actual_sample_counts"] != {
        "g1_episodes_per_split": 24,
        "g1_split_count": 2,
        "g2_formal_calls": 645,
        "g3_interface_replays": 6,
        "g3_wheel_episodes": 10,
    }:
        raise FigureBlocked("aggregate_config_sample_contract_drift")
    if config["reference_sample_counts"] != {
        "g1_episodes_per_split": 64,
        "g1_split_count": 2,
        "g2_formal_calls": 1950,
        "g3_interface_replays": 18,
        "g3_wheel_episodes": 30,
    }:
        raise FigureBlocked("aggregate_config_reference_contract_drift")
    if enforce_formal_paths:
        if config["output_root"] != FORMAL_AGGREGATE_BASE:
            raise FigureBlocked("aggregate_output_base_invalid")
        expected_root = Path(config["output_root"]) / config["run_id"]
        if aggregate_root.resolve() != expected_root.resolve():
            raise FigureBlocked("aggregate_root_identity_mismatch")
        for gate_id in GATES:
            source_root = Path(str(config[f"{gate_id}_root"]))
            source_base = Path(
                str(source_contracts[gate_id]["output_base"])
            )
            if (
                not source_root.is_absolute()
                or source_root.parent.resolve() != source_base.resolve()
            ):
                raise FigureBlocked(f"{gate_id}_formal_root_invalid")


def _load_aggregate_module() -> Any:
    try:
        return importlib.import_module("run_xunce_mid_dual_aggregate")
    except (ImportError, RuntimeError) as exc:
        raise FigureBlocked("aggregate_validator_import_failed") from exc


@contextmanager
def _bind_aggregate_source_snapshots(
    aggregate: Any,
    bindings: Mapping[
        str,
        tuple[Path, Mapping[str, Any], Mapping[str, bytes], str],
    ],
) -> Any:
    snapshot_function = getattr(
        aggregate,
        "_snapshot_verified_manifest",
        None,
    )
    blocked_type = getattr(aggregate, "AggregateBlocked", FigureBlocked)
    if not callable(snapshot_function):
        raise FigureBlocked("aggregate_snapshot_interface_missing")

    def frozen_snapshot(
        source_root: str | Path,
        gate_id: str,
    ) -> tuple[dict[str, Any], dict[str, bytes], str]:
        binding = bindings.get(gate_id)
        if binding is None:
            raise blocked_type(f"{gate_id}_snapshot_binding_missing")
        expected_root, manifest, snapshot, manifest_sha256 = binding
        if Path(source_root).resolve() != expected_root.resolve():
            raise blocked_type(f"{gate_id}_snapshot_root_mismatch")
        return (
            dict(manifest),
            {key: bytes(value) for key, value in snapshot.items()},
            manifest_sha256,
        )

    setattr(aggregate, "_snapshot_verified_manifest", frozen_snapshot)
    try:
        yield aggregate
    finally:
        setattr(
            aggregate,
            "_snapshot_verified_manifest",
            snapshot_function,
        )


def _recompute_from_verified_sources(
    aggregate: Any,
    sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    g1_source = sources["g1"]
    if g1_source.get("source_kind") == "g1_native":
        g1 = dict(g1_source["metric_projection"])
    else:
        g1 = aggregate.recompute_g1(
            g1_source["rows"],
            g1_source["config"],
            g1_source["input_audit"],
        )
    g2_source = sources["g2"]
    g2 = aggregate.recompute_g2(
        g2_source["rows"],
        g2_source["config"],
        g2_source["input_audit"],
    )
    g3_source = sources["g3"]
    g3 = aggregate.recompute_g3(
        g3_source["rows"],
        g3_source["config"],
        g3_source["input_audit"],
        g1_rows=g1_source["rows"],
        g2_rows=g2_source["rows"],
        g1_manifest_sha256=g1_source["manifest_sha256"],
        g2_manifest_sha256=g2_source["manifest_sha256"],
        g1_config=g1_source["config"],
        g1_input_audit=g1_source["input_audit"],
        g1_native_summary=g1_source.get(
            "native_summary",
            g1_source["stored_summary"],
        ),
        g1_trace_row_count=g1_source["trace_row_count"],
        g2_config=g2_source["config"],
        g2_input_audit=g2_source["input_audit"],
        upstream_lineage=g3_source["upstream_lineage"],
        upstream_lineage_audit_sha256=g3_source[
            "upstream_lineage_audit_sha256"
        ],
    )
    gates = {"g1": g1, "g2": g2, "g3": g3}
    sample_counts = {
        "g1_total_episodes": int(g1["sample_count"]),
        "g1_test_q24": int(g1["splits"]["test_q24"]["sample_count"]),
        "g1_unseen24": int(g1["splits"]["unseen24"]["sample_count"]),
        "g2_formal_calls": int(g2["formal_call_count"]),
        "g3_wheel_episodes": int(g3["wheel_episode_count"]),
        "g3_interface_replays": int(g3["interface_replay_count"]),
    }
    midterm = aggregate.midterm_reduced_truth_table(
        g1["g1_coverage_80_passed"],
        g2["g2_all_platforms_2s_passed"],
        g3["g3_midterm_crosscheck_passed"],
    )
    final = aggregate.final_threshold_reduced_truth_table(
        g1["g1_coverage_99_passed"],
        g2["g2_all_platforms_1s_passed"],
        g3["g3_final_crosscheck_passed"],
    )
    return {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "scale_profile": SCALE_PROFILE,
        "status": "passed" if midterm else "failed",
        "formal_evidence_eligible": True,
        "midterm_reduced_gate_passed": midterm,
        "final_threshold_reduced_gate_passed": final,
        "blockers": [],
        "gates": gates,
        "sample_counts": sample_counts,
        "source_manifest_sha256": {
            gate_id: sources[gate_id]["manifest_sha256"]
            for gate_id in GATES
        },
    }


def load_verified_figure_input(
    aggregate_root: str | Path,
    *,
    enforce_formal_paths: bool = True,
    aggregate_module: Any | None = None,
) -> VerifiedFigureInput:
    """Load and independently recompute a completed aggregate evidence root."""

    root = Path(aggregate_root)
    if not root.is_absolute():
        raise FigureBlocked("aggregate_root_not_absolute")
    required_aggregate_paths = {
        "config.json",
        "results.jsonl",
        "summary.json",
        "routing.json",
        "phase-state.jsonl",
        "report.md",
        "lineage_audit.json",
        "source_roots_audit.json",
    }
    (
        aggregate_manifest,
        aggregate_snapshot,
        aggregate_manifest_sha256,
        aggregate_artifact_hashes,
    ) = _snapshot_completed_root(
        root,
        "aggregate",
        required_aggregate_paths,
    )
    aggregate_config = _parse_json_object(
        aggregate_snapshot["config.json"],
        "aggregate_config_invalid",
    )
    _validate_aggregate_config(
        aggregate_config,
        aggregate_manifest["config_sha256"],
        root,
        enforce_formal_paths=enforce_formal_paths,
    )
    stored_summary = _parse_json_object(
        aggregate_snapshot["summary.json"],
        "aggregate_summary_invalid",
    )
    if (
        stored_summary.get("schema_version") != AGGREGATE_SCHEMA_VERSION
        or stored_summary.get("scale_profile") != SCALE_PROFILE
        or stored_summary.get("formal_evidence_eligible") is not True
        or stored_summary.get("status") not in {"passed", "failed"}
        or stored_summary.get("blockers") != []
    ):
        raise FigureBlocked("aggregate_summary_not_renderable")
    source_roots_audit = _parse_json_object(
        aggregate_snapshot["source_roots_audit.json"],
        "aggregate_source_roots_audit_invalid",
    )
    _require_exact_keys(
        source_roots_audit,
        {
            "g1_root",
            "g2_root",
            "g3_root",
            "source_manifest_sha256",
        },
        "aggregate_source_roots_audit_invalid",
    )
    expected_roots = {
        gate_id: str(aggregate_config[f"{gate_id}_root"])
        for gate_id in GATES
    }
    if any(
        Path(str(source_roots_audit[f"{gate_id}_root"])).resolve()
        != Path(expected_roots[gate_id]).resolve()
        for gate_id in GATES
    ):
        raise FigureBlocked("aggregate_source_root_binding_mismatch")
    expected_source_hashes = stored_summary.get("source_manifest_sha256")
    if (
        not isinstance(expected_source_hashes, Mapping)
        or set(expected_source_hashes) != set(GATES)
        or source_roots_audit["source_manifest_sha256"]
        != expected_source_hashes
        or any(
            not _is_sha256(expected_source_hashes[gate_id])
            for gate_id in GATES
        )
    ):
        raise FigureBlocked("aggregate_source_manifest_binding_mismatch")

    source_snapshots: dict[str, dict[str, bytes]] = {}
    source_manifests: dict[str, dict[str, Any]] = {}
    source_manifest_hashes: dict[str, str] = {}
    source_rows_raw: dict[str, list[dict[str, Any]]] = {}
    source_artifact_hashes: dict[str, dict[str, str]] = {}
    for gate_id in GATES:
        source_root = Path(expected_roots[gate_id])
        if not source_root.is_absolute():
            raise FigureBlocked(f"{gate_id}_root_not_absolute")
        source_manifest, snapshot, manifest_sha256, artifact_hashes = (
            _snapshot_completed_root(
                source_root,
                gate_id,
                {
                    "config.json",
                    "results.jsonl",
                    "summary.json",
                    "phase-state.jsonl",
                    "report.md",
                    "lineage_audit.json",
                },
            )
        )
        if manifest_sha256 != expected_source_hashes[gate_id]:
            raise FigureBlocked(
                f"{gate_id}_source_manifest_hash_mismatch"
            )
        source_snapshots[gate_id] = snapshot
        source_manifests[gate_id] = source_manifest
        source_manifest_hashes[gate_id] = manifest_sha256
        source_artifact_hashes[gate_id] = artifact_hashes
        source_rows_raw[gate_id] = _parse_jsonl_rows(
            snapshot["results.jsonl"],
            f"{gate_id}_results_jsonl_invalid",
        )

    aggregate = (
        aggregate_module
        if aggregate_module is not None
        else _load_aggregate_module()
    )
    sources: dict[str, dict[str, Any]] = {}
    try:
        bindings = {
            gate_id: (
                Path(expected_roots[gate_id]),
                source_manifests[gate_id],
                source_snapshots[gate_id],
                source_manifest_hashes[gate_id],
            )
            for gate_id in GATES
        }
        with _bind_aggregate_source_snapshots(aggregate, bindings):
            for gate_id in GATES:
                sources[gate_id] = aggregate._load_verified_source(  # noqa: SLF001
                    Path(expected_roots[gate_id]),
                    gate_id,
                    aggregate_config["source_contracts"][gate_id],
                )
    except Exception as exc:
        if isinstance(exc, getattr(aggregate, "AggregateBlocked", ())):
            reason = str(exc)
        else:
            reason = f"malformed_evidence:{type(exc).__name__}"
        raise FigureBlocked(f"source_revalidation_failed:{reason}") from exc

    for gate_id in GATES:
        if sources[gate_id]["manifest_sha256"] != expected_source_hashes[gate_id]:
            raise FigureBlocked(f"{gate_id}_validator_manifest_drift")
        if sources[gate_id]["config"] != _parse_json_object(
            source_snapshots[gate_id]["config.json"],
            f"{gate_id}_config_snapshot_invalid",
        ):
            raise FigureBlocked(f"{gate_id}_validator_config_snapshot_mismatch")
        if sources[gate_id]["stored_summary"] != _parse_json_object(
            source_snapshots[gate_id]["summary.json"],
            f"{gate_id}_summary_snapshot_invalid",
        ):
            raise FigureBlocked(f"{gate_id}_validator_summary_snapshot_mismatch")
        raw_rows = source_rows_raw[gate_id]
        if gate_id == "g1":
            coverage_rows = [
                row
                for row in raw_rows
                if row.get("row_kind") == "coverage_episode"
            ]
            trace_count = sum(
                row.get("row_kind") in {"decision", "planner_call"}
                for row in raw_rows
            )
            if (
                coverage_rows != sources[gate_id]["rows"]
                or trace_count != sources[gate_id]["trace_row_count"]
                or len(coverage_rows) + trace_count != len(raw_rows)
            ):
                raise FigureBlocked("g1_raw_row_snapshot_mismatch")
        elif raw_rows != sources[gate_id]["rows"]:
            raise FigureBlocked(f"{gate_id}_raw_row_snapshot_mismatch")

    try:
        recomputed = _recompute_from_verified_sources(aggregate, sources)
    except Exception as exc:
        if isinstance(exc, getattr(aggregate, "AggregateBlocked", ())):
            reason = str(exc)
        else:
            reason = f"malformed_evidence:{type(exc).__name__}"
        raise FigureBlocked(f"figure_metric_recompute_failed:{reason}") from exc
    if stored_summary != recomputed:
        raise FigureBlocked("aggregate_summary_recompute_mismatch")

    aggregate_rows = _parse_jsonl_rows(
        aggregate_snapshot["results.jsonl"],
        "aggregate_results_jsonl_invalid",
    )
    expected_aggregate_rows = [
        {
            "row_kind": "independent_gate_recalculation",
            "gate_id": gate_id,
            "recalculated": recomputed["gates"][gate_id],
        }
        for gate_id in GATES
    ]
    if aggregate_rows != expected_aggregate_rows:
        raise FigureBlocked("aggregate_results_recompute_mismatch")
    routing = _parse_json_object(
        aggregate_snapshot["routing.json"],
        "aggregate_routing_invalid",
    )
    expected_routing = {
        "schema_version": "xunce-mid-dual-aggregate-routing/v1",
        "scale_profile": SCALE_PROFILE,
        "status": recomputed["status"],
        "midterm_reduced_gate_passed": recomputed[
            "midterm_reduced_gate_passed"
        ],
        "final_threshold_reduced_gate_passed": recomputed[
            "final_threshold_reduced_gate_passed"
        ],
        "blockers": [],
        "formal_evidence_eligible": True,
    }
    if routing != expected_routing:
        raise FigureBlocked("aggregate_routing_recompute_mismatch")

    input_hashes = {
        "aggregate/manifest.json": aggregate_manifest_sha256,
        **{
            f"aggregate/{path}": digest
            for path, digest in aggregate_artifact_hashes.items()
        },
    }
    for gate_id in GATES:
        input_hashes[f"{gate_id}/manifest.json"] = expected_source_hashes[
            gate_id
        ]
        for path in ("config.json", "results.jsonl", "summary.json"):
            input_hashes[f"{gate_id}/{path}"] = source_artifact_hashes[
                gate_id
            ][path]
    run_ids = {
        "aggregate": str(aggregate_config["run_id"]),
        **{
            gate_id: str(sources[gate_id]["run_id"])
            for gate_id in GATES
        },
    }
    return VerifiedFigureInput(
        aggregate_run_id=run_ids["aggregate"],
        aggregate_manifest_sha256=aggregate_manifest_sha256,
        status=str(recomputed["status"]),
        run_ids=run_ids,
        source_manifest_sha256=dict(expected_source_hashes),
        input_hashes=input_hashes,
        aggregate_summary=recomputed,
        gates=recomputed["gates"],
        rows={
            gate_id: tuple(dict(row) for row in sources[gate_id]["rows"])
            for gate_id in GATES
        },
    )


def _nearest_rank(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise FigureBlocked("statistics_partition_empty")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _bootstrap_endpoint_ranks(
    *,
    resamples: int,
    confidence: float,
) -> tuple[int, int]:
    if (
        type(resamples) is not int
        or resamples <= 0
        or type(confidence) not in (int, float)
        or confidence != 0.95
    ):
        raise FigureBlocked("bootstrap_contract_invalid")
    confidence_fraction = Fraction(str(confidence))
    lower_quantile = (Fraction(1, 1) - confidence_fraction) / 2
    upper_quantile = Fraction(1, 1) - lower_quantile

    def nearest_rank(quantile: Fraction) -> int:
        numerator = quantile.numerator * resamples
        rank = (numerator + quantile.denominator - 1) // quantile.denominator
        return min(resamples, max(1, rank))

    return nearest_rank(lower_quantile), nearest_rank(upper_quantile)


def _coverage_statistics(values: Sequence[float]) -> dict[str, Any]:
    if not values or any(
        not math.isfinite(value) or value < 0.0 or value > 1.0
        for value in values
    ):
        raise FigureBlocked("coverage_values_invalid")
    mean_value = statistics.mean(values)
    count80 = sum(value >= 0.80 for value in values)
    count99 = sum(value >= 0.99 for value in values)
    midterm = mean_value >= 0.80 and count80 >= 23
    final = mean_value >= 0.99 and count99 >= 23
    return {
        "status": "passed" if midterm else "failed",
        "sample_count": len(values),
        "mean": mean_value,
        "p50": _nearest_rank(values, 0.50),
        "p95": _nearest_rank(values, 0.95),
        "p99": _nearest_rank(values, 0.99),
        "sample_stddev": statistics.stdev(values)
        if len(values) > 1
        else 0.0,
        "min": min(values),
        "max": max(values),
        "coverage_80_count": count80,
        "coverage_99_count": count99,
        "midterm_reduced_passed": midterm,
        "final_threshold_reduced_passed": final,
    }


def _timing_statistics(values: Sequence[float]) -> dict[str, Any]:
    if not values or any(
        not math.isfinite(value) or value < 0.0 for value in values
    ):
        raise FigureBlocked("timing_values_invalid")
    mean_value = statistics.mean(values)
    p95 = _nearest_rank(values, 0.95)
    maximum = max(values)
    midterm = (
        mean_value <= 2000.0
        and p95 <= 2000.0
        and maximum <= 2000.0
    )
    final = (
        mean_value <= 1000.0
        and p95 <= 1000.0
        and sum(value <= 1000.0 for value in values) / len(values) >= 0.95
        and maximum <= 2000.0
    )
    return {
        "sample_count": len(values),
        "mean_ms": mean_value,
        "p50_ms": _nearest_rank(values, 0.50),
        "p95_ms": p95,
        "p99_ms": _nearest_rank(values, 0.99),
        "sample_stddev_ms": statistics.stdev(values)
        if len(values) > 1
        else 0.0,
        "min_ms": min(values),
        "max_ms": maximum,
        "at_or_below_1000_count": sum(
            value <= 1000.0 for value in values
        ),
        "over_2000_count": sum(value > 2000.0 for value in values),
        "midterm_reduced_passed": midterm,
        "final_threshold_reduced_passed": final,
    }


def _same_number(left: object, right: object, reason: str) -> None:
    left_value = _require_finite(left, reason)
    right_value = _require_finite(right, reason)
    if not math.isclose(
        left_value,
        right_value,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise FigureBlocked(reason)


def deterministic_bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int,
    resamples: int,
    confidence: float,
    label: str,
) -> dict[str, Any]:
    """Return a deterministic nearest-rank percentile CI for the mean."""

    materialized = tuple(
        _require_finite(value, "bootstrap_value_invalid")
        for value in values
    )
    if (
        not materialized
        or type(seed) is not int
        or type(resamples) is not int
        or resamples <= 0
        or confidence != 0.95
        or not isinstance(label, str)
        or not label
    ):
        raise FigureBlocked("bootstrap_contract_invalid")
    derived_seed = int.from_bytes(
        hashlib.sha256(f"{seed}:{label}".encode("utf-8")).digest()[:8],
        "big",
    )
    generator = random.Random(derived_seed)
    means = sorted(
        statistics.mean(
            generator.choice(materialized)
            for _ in range(len(materialized))
        )
        for _ in range(resamples)
    )
    low_rank, high_rank = _bootstrap_endpoint_ranks(
        resamples=resamples,
        confidence=confidence,
    )
    return {
        "method": "deterministic-percentile-bootstrap-mean-nearest-rank/v1",
        "seed": seed,
        "derived_seed": derived_seed,
        "resamples": resamples,
        "confidence": confidence,
        "mean": statistics.mean(materialized),
        "ci_low_rank": low_rank,
        "ci_high_rank": high_rank,
        "ci_low": means[low_rank - 1],
        "ci_high": means[high_rank - 1],
    }


def _derive_figure_data(
    verified: VerifiedFigureInput,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize exact figure rows and compare every statistic to aggregate."""

    bootstrap = config["bootstrap"]
    g1_rows: list[dict[str, Any]] = []
    for row in verified.rows["g1"]:
        if row.get("row_kind") != "coverage_episode":
            raise FigureBlocked("g1_noncoverage_row_exposed")
        split = row.get("split")
        if split not in {"test_q24", "unseen24"}:
            raise FigureBlocked("g1_split_invalid")
        denominator = _require_nonnegative_int(
            row.get("denominator_cell_count"),
            "g1_denominator_invalid",
        )
        final_count = _require_nonnegative_int(
            row.get("final_covered_cell_count"),
            "g1_final_covered_count_invalid",
        )
        if denominator <= 0 or final_count > denominator:
            raise FigureBlocked("g1_denominator_invalid")
        recomputed_coverage = final_count / denominator
        stored_coverage = _require_finite(
            row.get("coverage"),
            "g1_coverage_invalid",
        )
        if (
            stored_coverage < 0.0
            or stored_coverage > 1.0
            or not math.isclose(
                recomputed_coverage,
                stored_coverage,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
        ):
            raise FigureBlocked("g1_coverage_recompute_mismatch")
        g1_rows.append(
            {
                "split": split,
                "episode_id": str(row["episode_id"]),
                "episode_index": _require_nonnegative_int(
                    row.get("episode_index"),
                    "g1_episode_index_invalid",
                ),
                "scenario_id": str(row["scenario_id"]),
                "lane_id": str(row["lane_id"]),
                "denominator_cell_count": denominator,
                "final_covered_cell_count": final_count,
                "coverage": recomputed_coverage,
            }
        )
    if (
        len(g1_rows) != 48
        or len({row["episode_id"] for row in g1_rows}) != 48
        or len({row["scenario_id"] for row in g1_rows}) != 48
    ):
        raise FigureBlocked("g1_exact_denominator_invalid")
    g1_splits: dict[str, dict[str, Any]] = {}
    for split in ("test_q24", "unseen24"):
        selected = sorted(
            (row for row in g1_rows if row["split"] == split),
            key=lambda row: row["episode_index"],
        )
        if (
            len(selected) != 24
            or [row["episode_index"] for row in selected]
            != list(range(24))
        ):
            raise FigureBlocked("g1_exact_denominator_invalid")
        values = [row["coverage"] for row in selected]
        statistics_payload = _coverage_statistics(values)
        if statistics_payload != verified.gates["g1"]["splits"][split]:
            raise FigureBlocked("g1_display_statistic_mismatch")
        g1_splits[split] = {
            "rows": selected,
            "statistics": statistics_payload,
            "bootstrap_ci": deterministic_bootstrap_mean_ci(
                values,
                seed=bootstrap["seed"],
                resamples=bootstrap["resamples"],
                confidence=bootstrap["confidence"],
                label=f"g1:{split}",
            ),
        }

    g2_rows: list[dict[str, Any]] = []
    request_repeats: dict[tuple[str, str], set[int]] = {}
    class_counts: dict[tuple[str, str, str], int] = {}
    for row in verified.rows["g2"]:
        if row.get("row_kind") != "planning_call":
            raise FigureBlocked("g2_row_kind_invalid")
        platform_name = str(row.get("platform"))
        scale = str(row.get("scale"))
        request_class = str(row.get("request_class"))
        outcome_kind = str(row.get("outcome_kind"))
        if (
            platform_name not in PLATFORMS
            or scale not in SCALES
            or request_class not in REQUEST_CLASSES
            or outcome_kind
            != (
                "unreachable"
                if request_class == "unreachable"
                else "reachable"
            )
            or row.get("formal_sample") is not True
        ):
            raise FigureBlocked("g2_partition_invalid")
        component_ns = [
            _require_nonnegative_int(
                row.get(field),
                "g2_timing_component_invalid",
            )
            for field in TIMING_COMPONENT_FIELDS
        ]
        total_ns = _require_nonnegative_int(
            row.get("total_ns"),
            "g2_total_ns_invalid",
        )
        elapsed_ms = _require_finite(
            row.get("elapsed_ms"),
            "g2_elapsed_ms_invalid",
        )
        recomputed_ms = sum(component_ns) / 1_000_000.0
        if (
            total_ns != sum(component_ns)
            or not math.isclose(
                elapsed_ms,
                recomputed_ms,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise FigureBlocked("g2_timing_recompute_mismatch")
        repeat_index = _require_nonnegative_int(
            row.get("repeat_index"),
            "g2_repeat_index_invalid",
        )
        if repeat_index >= 5:
            raise FigureBlocked("g2_repeat_index_invalid")
        request_id = str(row.get("request_id"))
        request_key = (platform_name, request_id)
        request_repeats.setdefault(request_key, set()).add(repeat_index)
        class_key = (platform_name, scale, request_class)
        class_counts[class_key] = class_counts.get(class_key, 0) + 1
        g2_rows.append(
            {
                "platform": platform_name,
                "scale": scale,
                "request_class": request_class,
                "outcome_kind": outcome_kind,
                "request_id": request_id,
                "request_sha256": str(row["request_sha256"]),
                "call_id": str(row["call_id"]),
                "repeat_index": repeat_index,
                **{
                    field: int(row[field])
                    for field in TIMING_COMPONENT_FIELDS
                },
                "total_ns": total_ns,
                "elapsed_ms": recomputed_ms,
            }
        )
    if (
        len(g2_rows) != 645
        or len({row["call_id"] for row in g2_rows}) != 645
        or len(request_repeats) != 129
        or any(repeats != set(range(5)) for repeats in request_repeats.values())
    ):
        raise FigureBlocked("g2_exact_denominator_invalid")
    for platform_name in PLATFORMS:
        for (scale, request_class), request_count in (
            EXPECTED_G2_CLASS_COUNTS.items()
        ):
            if (
                class_counts.get((platform_name, scale, request_class))
                != request_count * 5
            ):
                raise FigureBlocked("g2_partition_denominator_invalid")
    g2_groups: dict[str, dict[str, Any]] = {}
    for platform_name in PLATFORMS:
        for scale in SCALES:
            key = f"{platform_name}/{scale}"
            selected = sorted(
                (
                    row
                    for row in g2_rows
                    if row["platform"] == platform_name
                    and row["scale"] == scale
                ),
                key=lambda row: (
                    REQUEST_CLASSES.index(row["request_class"]),
                    row["request_id"],
                    row["repeat_index"],
                ),
            )
            expected_n = (33 if scale == "standard" else 10) * 5
            if len(selected) != expected_n:
                raise FigureBlocked("g2_partition_denominator_invalid")
            stats = _timing_statistics(
                [row["elapsed_ms"] for row in selected]
            )
            if stats != verified.gates["g2"][
                "timing_by_platform_scale"
            ][key]:
                raise FigureBlocked("g2_display_statistic_mismatch")
            outcome_stats: dict[str, dict[str, Any]] = {}
            for outcome in ("reachable", "unreachable"):
                values = [
                    row["elapsed_ms"]
                    for row in selected
                    if row["outcome_kind"] == outcome
                ]
                outcome_stats[outcome] = _timing_statistics(values)
                if outcome_stats[outcome] != verified.gates["g2"][
                    "timing_by_platform_scale_outcome"
                ][f"{key}/{outcome}"]:
                    raise FigureBlocked("g2_outcome_statistic_mismatch")
            class_stats: dict[str, dict[str, Any]] = {}
            for request_class in REQUEST_CLASSES:
                values = [
                    row["elapsed_ms"]
                    for row in selected
                    if row["request_class"] == request_class
                ]
                class_stats[request_class] = _timing_statistics(values)
                if class_stats[request_class] != verified.gates["g2"][
                    "timing_by_platform_scale_class"
                ][f"{key}/{request_class}"]:
                    raise FigureBlocked("g2_class_statistic_mismatch")
            g2_groups[key] = {
                "rows": selected,
                "statistics": stats,
                "outcome_statistics": outcome_stats,
                "class_statistics": class_stats,
            }

    wheel_rows = [
        row
        for row in verified.rows["g3"]
        if row.get("row_kind") == "g3_wheel_step"
    ]
    interface_rows = [
        row
        for row in verified.rows["g3"]
        if row.get("row_kind") == "g3_interface_replay"
    ]
    if len(wheel_rows) + len(interface_rows) != len(verified.rows["g3"]):
        raise FigureBlocked("g3_row_kind_invalid")
    terminal_rows = sorted(
        (row for row in wheel_rows if row.get("is_terminal") is True),
        key=lambda row: (
            0 if row.get("split") == "test_q24" else 1,
            int(row.get("episode_index", -1)),
        ),
    )
    if (
        len(terminal_rows) != 10
        or len({row.get("episode_id") for row in terminal_rows}) != 10
        or {
            split: sum(row.get("split") == split for row in terminal_rows)
            for split in ("test_q24", "unseen24")
        }
        != {"test_q24": 5, "unseen24": 5}
    ):
        raise FigureBlocked("g3_wheel_denominator_invalid")
    wheel_terminal: list[dict[str, Any]] = []
    for row in terminal_rows:
        coverage = _require_finite(
            row.get("coverage"),
            "g3_coverage_invalid",
        )
        paired = _require_finite(
            row.get("paired_g1_coverage"),
            "g3_paired_coverage_invalid",
        )
        if not 0.0 <= coverage <= 1.0 or not 0.0 <= paired <= 1.0:
            raise FigureBlocked("g3_coverage_invalid")
        wheel_terminal.append(
            {
                "split": str(row["split"]),
                "episode_id": str(row["episode_id"]),
                "episode_index": int(row["episode_index"]),
                "scenario_id": str(row["scenario_id"]),
                "g1_coverage": paired,
                "wheel_coverage": coverage,
                "paired_delta": coverage - paired,
            }
        )
    if len(interface_rows) != 6:
        raise FigureBlocked("g3_interface_denominator_invalid")
    interface_projection: list[dict[str, Any]] = []
    platform_counts = {"legged": 0, "hopper": 0}
    for row in interface_rows:
        platform_name = str(row.get("platform"))
        if (
            platform_name not in platform_counts
            or row.get("formal_input_eligible") is not True
            or row.get("g2_semantic_digest")
            != row.get("replay_semantic_digest")
        ):
            raise FigureBlocked("g3_interface_completeness_invalid")
        platform_counts[platform_name] += 1
        interface_projection.append(
            {
                "platform": platform_name,
                "replay_id": str(row["replay_id"]),
                "request_id": str(row["request_id"]),
                "complete": True,
                "capability_kind": "simulation_proxy",
            }
        )
    if platform_counts != {"legged": 3, "hopper": 3}:
        raise FigureBlocked("g3_interface_denominator_invalid")
    wheel_coverages = [row["wheel_coverage"] for row in wheel_terminal]
    deltas = [row["paired_delta"] for row in wheel_terminal]
    _same_number(
        statistics.mean(wheel_coverages),
        verified.gates["g3"]["wheel_coverage_mean"],
        "g3_display_coverage_mismatch",
    )
    _same_number(
        statistics.mean(deltas),
        verified.gates["g3"]["paired_g1_coverage_delta_mean"],
        "g3_display_delta_mismatch",
    )
    delta_ci = deterministic_bootstrap_mean_ci(
        deltas,
        seed=bootstrap["seed"],
        resamples=bootstrap["resamples"],
        confidence=bootstrap["confidence"],
        label="g3:paired_delta",
    )
    return {
        "g1": {"splits": g1_splits},
        "g2": {"groups": g2_groups},
        "g3": {
            "wheel_terminal": wheel_terminal,
            "interface_replays": interface_projection,
            "platform_counts": platform_counts,
            "paired_delta_bootstrap_ci": delta_ci,
        },
    }


def _apply_publication_style(
    config: Mapping[str, Any],
    config_sha256: str,
) -> None:
    style = config["style"]
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": list(style["font_fallback"]),
            "svg.fonttype": "none",
            "svg.hashsalt": f"xunce-mid-dual-{config_sha256}",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": float(style["base_font_pt"]),
            "axes.labelsize": float(style["base_font_pt"]),
            "axes.titlesize": float(style["base_font_pt"]) + 0.5,
            "axes.linewidth": float(style["axes_linewidth"]),
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": float(style["base_font_pt"]) - 0.5,
            "ytick.labelsize": float(style["base_font_pt"]) - 0.5,
            "legend.fontsize": float(style["base_font_pt"]) - 0.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _panel_label(
    ax: Any,
    label: str,
    config: Mapping[str, Any],
    *,
    x: float = -0.10,
    y: float = 1.04,
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=float(config["style"]["panel_label_pt"]),
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def _family_status(
    family: str,
    verified: VerifiedFigureInput,
) -> str:
    if family == "fig01_dual_gate_overview":
        return verified.status
    gate_id = {
        "fig02_g1_coverage": "g1",
        "fig03_g2_timing": "g2",
        "fig04_g3_crosscheck": "g3",
    }[family]
    status = str(verified.gates[gate_id].get("status"))
    if status not in {"passed", "failed", "blocked"}:
        raise FigureBlocked(f"{family}_status_invalid")
    return status


def _family_sample_sizes(
    family: str,
    verified: VerifiedFigureInput,
) -> dict[str, int]:
    samples = verified.aggregate_summary["sample_counts"]
    if family == "fig01_dual_gate_overview":
        return {
            "g1_test_q24": int(samples["g1_test_q24"]),
            "g1_unseen24": int(samples["g1_unseen24"]),
            "g2_formal_calls": int(samples["g2_formal_calls"]),
            "g3_wheel_episodes": int(samples["g3_wheel_episodes"]),
            "g3_interface_replays": int(samples["g3_interface_replays"]),
        }
    if family == "fig02_g1_coverage":
        return {
            "test_q24": int(samples["g1_test_q24"]),
            "unseen24": int(samples["g1_unseen24"]),
        }
    if family == "fig03_g2_timing":
        return {"formal_calls": int(samples["g2_formal_calls"])}
    return {
        "wheel_episodes": int(samples["g3_wheel_episodes"]),
        "legged_replays": 3,
        "hopper_replays": 3,
    }


def _family_statistical_methods(family: str) -> list[str]:
    if family == "fig01_dual_gate_overview":
        return [
            "raw-row independent recomputation",
            "coverage mean and threshold margin",
            "nearest-rank P95 timing and threshold margin",
        ]
    if family == "fig02_g1_coverage":
        return [
            "episode-level mean, nearest-rank P50, and sample SD",
            "deterministic percentile-bootstrap 95% CI of the mean",
            "no null-hypothesis significance test",
        ]
    if family == "fig03_g2_timing":
        return [
            "all formal raw calls",
            "nearest-rank P95",
            "mean and sample SD",
            "no null-hypothesis significance test",
        ]
    return [
        "paired wheel minus G1 episode delta",
        "deterministic percentile-bootstrap 95% CI of mean paired delta",
        "exact interface replay completeness count",
        "no null-hypothesis significance test",
    ]


def _family_units(family: str) -> list[str]:
    if family == "fig01_dual_gate_overview":
        return ["coverage fraction", "milliseconds", "categorical status"]
    if family == "fig02_g1_coverage":
        return ["coverage fraction"]
    if family == "fig03_g2_timing":
        return ["milliseconds"]
    return ["coverage fraction", "paired coverage delta", "replay count"]


def _figure_metadata(
    family: str,
    verified: VerifiedFigureInput,
    config: Mapping[str, Any],
    *,
    figure_run_id: str | None = None,
    code_lineage: CodeLineageSnapshot | None = None,
) -> dict[str, Any]:
    low_rank, high_rank = _bootstrap_endpoint_ranks(
        resamples=int(config["bootstrap"]["resamples"]),
        confidence=float(config["bootstrap"]["confidence"]),
    )
    metadata = {
        "schema_version": "xunce-mid-dual-figure-metadata/v1",
        "figure_family": family,
        "status": _family_status(family, verified),
        "reduced_qualifier": REDUCED_QUALIFIER,
        "aggregate_manifest_sha256": verified.aggregate_manifest_sha256,
        "run_ids": dict(verified.run_ids),
        "sample_sizes": _family_sample_sizes(family, verified),
        "statistical_methods": _family_statistical_methods(family),
        "units": _family_units(family),
        "thresholds": dict(config["thresholds"]),
        "bootstrap": {
            **dict(config["bootstrap"]),
            "ci_low_rank": low_rank,
            "ci_high_rank": high_rank,
        },
        "claims": dict(config["claims"]),
    }
    if figure_run_id is not None:
        metadata["figure_run_id"] = figure_run_id
    if code_lineage is not None:
        metadata["code_lineage_sha256"] = _code_lineage_file_sha256(
            code_lineage
        )
        metadata["renderer_sha256"] = code_lineage.renderer_sha256
    return metadata


def _add_metadata_footer(
    fig: Any,
    metadata: Mapping[str, Any],
) -> None:
    samples = ", ".join(
        f"{key}={value}"
        for key, value in metadata["sample_sizes"].items()
    )
    methods = "; ".join(metadata["statistical_methods"])
    footer = (
        f"{metadata['reduced_qualifier']} | status={metadata['status']} | "
        f"n: {samples} | units: {', '.join(metadata['units'])}\n"
        f"Independent raw-row recomputation; Methods: {methods}\n"
        "Thresholds: coverage 0.80/0.99; planning time "
        "2000/1000 ms; paired-delta floor -0.01"
    )
    wrapped_lines: list[str] = []
    for line in footer.splitlines():
        wrapped_lines.extend(
            textwrap.wrap(
                line,
                width=125,
                break_long_words=False,
                break_on_hyphens=False,
            )
            or [""]
        )
    fig.text(
        0.01,
        0.008,
        "\n".join(wrapped_lines),
        ha="left",
        va="bottom",
        fontsize=4.3,
        color="#333333",
        linespacing=1.15,
    )


def _style_quantitative_axis(
    ax: Any,
    config: Mapping[str, Any],
    *,
    grid_axis: str = "y",
) -> None:
    ax.grid(
        True,
        axis=grid_axis,
        color=config["style"]["colors"]["light_neutral"],
        alpha=float(config["style"]["grid_alpha"]),
        linewidth=0.6,
    )
    ax.set_axisbelow(True)


def build_fig01_dual_gate_overview(
    verified: VerifiedFigureInput,
    derived: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Any:
    """Build the compact hero overview without mixing unlike units."""

    colors = config["style"]["colors"]
    size = config["figure_sizes_inches"]["fig01_dual_gate_overview"]
    fig = plt.figure(figsize=tuple(size), layout="constrained")
    fig.get_layout_engine().set(rect=(0.0, 0.12, 1.0, 0.98))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(0.95, 1.55, 0.90),
        wspace=0.48,
    )
    ax_cov = fig.add_subplot(grid[0, 0])
    ax_time = fig.add_subplot(grid[0, 1])
    ax_status = fig.add_subplot(grid[0, 2])

    coverage_labels = ["G1 Test-Q24", "G1 Unseen-24", "G3 wheel"]
    coverage_values = [
        derived["g1"]["splits"]["test_q24"]["statistics"]["mean"],
        derived["g1"]["splits"]["unseen24"]["statistics"]["mean"],
        verified.gates["g3"]["wheel_coverage_mean"],
    ]
    y = np.arange(len(coverage_labels))
    ax_cov.barh(
        y,
        coverage_values,
        color=[colors["blue"], colors["sky_blue"], colors["orange"]],
        edgecolor=colors["black"],
        linewidth=0.6,
        hatch=["", "//", ".."],
    )
    ax_cov.axvline(
        0.80,
        color=colors["vermillion"],
        linestyle="--",
        linewidth=1.0,
        label="Midterm threshold (0.80)",
    )
    ax_cov.axvline(
        0.99,
        color=colors["black"],
        linestyle=":",
        linewidth=1.0,
        label="Final threshold (0.99)",
    )
    ax_cov.set_xlim(0.0, 1.02)
    ax_cov.set_yticks(y, coverage_labels)
    ax_cov.invert_yaxis()
    ax_cov.set_xlabel("Coverage fraction")
    ax_cov.set_title(
        "Coverage margins\n(0.80 dashed; 0.99 dotted)"
    )
    _style_quantitative_axis(ax_cov, config, grid_axis="x")
    _panel_label(ax_cov, "a", config, y=1.12)

    timing_items = [
        (
            f"G2 {platform_name.capitalize()} {scale.capitalize()}",
            derived["g2"]["groups"][f"{platform_name}/{scale}"][
                "statistics"
            ]["p95_ms"],
        )
        for platform_name in PLATFORMS
        for scale in SCALES
    ]
    timing_items.extend(
        [
            ("G3 wheel", verified.gates["g3"]["wheel_timing"]["p95_ms"]),
            (
                "G3 interfaces",
                verified.gates["g3"]["interface_timing"]["p95_ms"],
            ),
        ]
    )
    timing_labels = [item[0] for item in timing_items]
    timing_values = [float(item[1]) for item in timing_items]
    y_time = np.arange(len(timing_items))
    ax_time.barh(
        y_time,
        timing_values,
        color=[
            colors["blue"],
            colors["sky_blue"],
            colors["reddish_purple"],
            colors["yellow"],
            colors["orange"],
            colors["bluish_green"],
            colors["neutral"],
            colors["light_neutral"],
        ],
        edgecolor=colors["black"],
        linewidth=0.5,
        hatch=["", "//", "..", "\\\\", "xx", "++", "--", "oo"],
    )
    ax_time.axvline(
        1000.0,
        color=colors["black"],
        linestyle=":",
        linewidth=1.0,
        label="Final threshold (1000 ms)",
    )
    ax_time.axvline(
        2000.0,
        color=colors["vermillion"],
        linestyle="--",
        linewidth=1.0,
        label="Midterm threshold (2000 ms)",
    )
    ax_time.set_xlim(0.0, max(2100.0, max(timing_values) * 1.08))
    ax_time.set_yticks(y_time, timing_labels)
    ax_time.invert_yaxis()
    ax_time.set_xlabel("Nearest-rank P95 time (ms)")
    ax_time.set_title(
        "Timing margins\n(1000 ms dotted; 2000 ms dashed)"
    )
    _style_quantitative_axis(ax_time, config, grid_axis="x")
    _panel_label(ax_time, "b", config, y=1.12)

    status_rows = ["G1", "G2", "G3", "Aggregate"]
    status_columns = ["Midterm", "Final"]
    status_values = [
        [
            "passed"
            if verified.gates["g1"]["g1_coverage_80_passed"]
            else "failed",
            "passed"
            if verified.gates["g1"]["g1_coverage_99_passed"]
            else "failed",
        ],
        [
            "passed"
            if verified.gates["g2"]["g2_all_platforms_2s_passed"]
            else "failed",
            "passed"
            if verified.gates["g2"]["g2_all_platforms_1s_passed"]
            else "failed",
        ],
        [
            "passed"
            if verified.gates["g3"]["g3_midterm_crosscheck_passed"]
            else "failed",
            "passed"
            if verified.gates["g3"]["g3_final_crosscheck_passed"]
            else "failed",
        ],
        [
            "passed"
            if verified.aggregate_summary["midterm_reduced_gate_passed"]
            else "failed",
            "passed"
            if verified.aggregate_summary[
                "final_threshold_reduced_gate_passed"
            ]
            else "failed",
        ],
    ]
    status_colors = {
        "passed": colors["bluish_green"],
        "failed": colors["vermillion"],
        "blocked": colors["neutral"],
    }
    for row_index, row_values in enumerate(status_values):
        for column_index, status in enumerate(row_values):
            rectangle = mpl.patches.Rectangle(
                (column_index - 0.46, row_index - 0.42),
                0.92,
                0.84,
                facecolor=status_colors[status],
                edgecolor=colors["black"],
                linewidth=0.6,
                hatch="" if status == "passed" else "//",
            )
            ax_status.add_patch(rectangle)
            ax_status.text(
                column_index,
                row_index,
                status.upper(),
                ha="center",
                va="center",
                fontsize=4.3,
                fontweight="bold",
                color="white" if status != "failed" else "black",
            )
    ax_status.set_xlim(-0.5, 1.5)
    ax_status.set_ylim(3.5, -0.5)
    ax_status.set_xticks(range(2), status_columns, rotation=20, ha="right")
    ax_status.set_yticks(range(4), status_rows)
    ax_status.set_title("Evidence status")
    ax_status.tick_params(length=0)
    for spine in ax_status.spines.values():
        spine.set_visible(False)
    _panel_label(ax_status, "c", config, y=1.12)

    metadata = _figure_metadata(
        "fig01_dual_gate_overview",
        verified,
        config,
    )
    _add_metadata_footer(fig, metadata)
    return fig


def build_fig02_g1_coverage(
    verified: VerifiedFigureInput,
    derived: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Any:
    """Build separate raw coverage distributions with bootstrap mean CIs."""

    colors = config["style"]["colors"]
    size = config["figure_sizes_inches"]["fig02_g1_coverage"]
    fig, axes = plt.subplots(1, 2, figsize=tuple(size), sharey=True)
    split_specs = (
        ("test_q24", "Test-Q24", colors["blue"], "o"),
        ("unseen24", "Unseen-24", colors["orange"], "s"),
    )
    for index, (split, title, color, marker) in enumerate(split_specs):
        ax = axes[index]
        payload = derived["g1"]["splits"][split]
        rows = payload["rows"]
        values = [row["coverage"] for row in rows]
        x = np.array(
            [
                ((row["episode_index"] * 37) % 101) / 100.0 - 0.5
                for row in rows
            ]
        ) * 0.30
        ax.scatter(
            x,
            values,
            s=18,
            marker=marker,
            facecolor=color if marker != "s" else "none",
            edgecolor=color,
            linewidth=0.8,
            alpha=0.82,
            label="Raw episode",
            zorder=3,
        )
        ci = payload["bootstrap_ci"]
        ax.errorbar(
            [0.0],
            [ci["mean"]],
            yerr=[
                [ci["mean"] - ci["ci_low"]],
                [ci["ci_high"] - ci["mean"]],
            ],
            fmt="D",
            color=colors["black"],
            mfc="white",
            mec=colors["black"],
            markersize=5.0,
            capsize=3.0,
            linewidth=1.2,
            label="Mean and deterministic 95% bootstrap CI",
            zorder=5,
        )
        ax.axhline(
            0.80,
            color=colors["vermillion"],
            linestyle="--",
            linewidth=1.0,
            label="Midterm threshold (0.80)",
        )
        ax.axhline(
            0.99,
            color=colors["black"],
            linestyle=":",
            linewidth=1.0,
            label="Final threshold (0.99)",
        )
        ax.set_xlim(-0.42, 0.42)
        ax.set_ylim(0.0, 1.02)
        ax.set_xticks([0.0], [f"{title}\n(n={len(values)})"])
        ax.set_title(
            f"{title}: mean={ci['mean']:.3f}, "
            "nearest-rank P50="
            f"{payload['statistics']['p50']:.3f}, "
            f"SD={payload['statistics']['sample_stddev']:.3f}"
        )
        if index == 0:
            ax.set_ylabel("Coverage fraction")
            ax.legend(loc="lower left", fontsize=5.2)
        _style_quantitative_axis(ax, config)
        _panel_label(ax, chr(ord("a") + index), config)
    metadata = _figure_metadata("fig02_g1_coverage", verified, config)
    _add_metadata_footer(fig, metadata)
    fig.tight_layout(rect=(0.0, 0.20, 1.0, 0.97))
    return fig


def _g2_jitter(row: Mapping[str, Any]) -> float:
    digest = str(row["request_sha256"])
    if not _is_sha256(digest):
        raise FigureBlocked("g2_request_sha256_invalid")
    base = int(digest[:8], 16)
    repeat = int(row["repeat_index"])
    return (((base + 97 * repeat) % 1001) / 1000.0 - 0.5) * 0.34


def build_fig03_g2_timing(
    verified: VerifiedFigureInput,
    derived: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Any:
    """Build six exact platform-by-scale panels containing every timing row."""

    colors = config["style"]["colors"]
    size = config["figure_sizes_inches"]["fig03_g2_timing"]
    fig, axes = plt.subplots(
        3,
        2,
        figsize=tuple(size),
        sharey=False,
        squeeze=False,
    )
    class_style = {
        "normal_reachable": (colors["blue"], "o", "Normal reachable"),
        "hard_reachable": (colors["orange"], "s", "Hard reachable"),
        "unreachable": (colors["reddish_purple"], "x", "Unreachable"),
    }
    all_values = [
        row["elapsed_ms"]
        for payload in derived["g2"]["groups"].values()
        for row in payload["rows"]
    ]
    global_upper = max(2100.0, max(all_values) * 1.06)
    for row_index, platform_name in enumerate(PLATFORMS):
        for column_index, scale in enumerate(SCALES):
            ax = axes[row_index][column_index]
            key = f"{platform_name}/{scale}"
            payload = derived["g2"]["groups"][key]
            rows = payload["rows"]
            for class_index, request_class in enumerate(REQUEST_CLASSES):
                color, marker, label = class_style[request_class]
                selected = [
                    row
                    for row in rows
                    if row["request_class"] == request_class
                ]
                x_values = [
                    class_index + _g2_jitter(row) for row in selected
                ]
                scatter_kwargs = {
                    "s": 9,
                    "marker": marker,
                    "color": color,
                    "alpha": 0.62,
                    "linewidth": 0.55,
                    "label": label,
                    "zorder": 3,
                }
                if marker == "s":
                    scatter_kwargs["facecolor"] = "none"
                    scatter_kwargs["edgecolor"] = color
                ax.scatter(
                    x_values,
                    [row["elapsed_ms"] for row in selected],
                    **scatter_kwargs,
                )
            statistics_payload = payload["statistics"]
            ax.axhline(
                statistics_payload["mean_ms"],
                color=colors["bluish_green"],
                linestyle="-",
                linewidth=1.0,
                label=f"Mean ({statistics_payload['mean_ms']:.1f} ms)",
                zorder=4,
            )
            ax.axhline(
                statistics_payload["p95_ms"],
                color=colors["blue"],
                linestyle="-.",
                linewidth=1.2,
                label=f"P95 ({statistics_payload['p95_ms']:.1f} ms)",
                zorder=4,
            )
            ax.axhline(
                1000.0,
                color=colors["black"],
                linestyle=":",
                linewidth=0.9,
                label="Final threshold (1000 ms)",
            )
            ax.axhline(
                2000.0,
                color=colors["vermillion"],
                linestyle="--",
                linewidth=0.9,
                label="Midterm threshold (2000 ms)",
            )
            ax.set_ylim(0.0, global_upper)
            ax.set_xticks(
                range(3),
                ["Normal\nreachable", "Hard\nreachable", "Unreachable"],
            )
            ax.set_title(
                f"{platform_name.capitalize()} — {scale.capitalize()} "
                f"(n={len(rows)}, P95={statistics_payload['p95_ms']:.1f} ms)"
            )
            ax.set_ylabel("Formal planning time (ms)")
            if row_index == 2:
                ax.set_xlabel("Request-class partition")
            _style_quantitative_axis(ax, config)
            _panel_label(
                ax,
                chr(ord("a") + row_index * 2 + column_index),
                config,
            )
            if row_index == 0 and column_index == 0:
                ax.legend(
                    loc="upper left",
                    ncol=2,
                    fontsize=4.9,
                    columnspacing=0.8,
                    handlelength=1.7,
                )
    metadata = _figure_metadata("fig03_g2_timing", verified, config)
    _add_metadata_footer(fig, metadata)
    fig.tight_layout(rect=(0.0, 0.13, 1.0, 0.98), h_pad=1.2, w_pad=1.1)
    return fig


def build_fig04_g3_crosscheck(
    verified: VerifiedFigureInput,
    derived: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Any:
    """Build paired wheel evidence and separate simulation-proxy completeness."""

    colors = config["style"]["colors"]
    size = config["figure_sizes_inches"]["fig04_g3_crosscheck"]
    fig = plt.figure(figsize=tuple(size), layout="constrained")
    fig.get_layout_engine().set(rect=(0.0, 0.16, 1.0, 0.98))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.55, 1.15, 0.90),
        wspace=0.42,
    )
    ax_pair = fig.add_subplot(grid[0, 0])
    ax_delta = fig.add_subplot(grid[0, 1])
    ax_replay = fig.add_subplot(grid[0, 2])
    rows = derived["g3"]["wheel_terminal"]
    x = np.arange(1, len(rows) + 1)
    g1_values = [row["g1_coverage"] for row in rows]
    wheel_values = [row["wheel_coverage"] for row in rows]
    ax_pair.plot(
        x,
        g1_values,
        color=colors["blue"],
        marker="o",
        markersize=4.0,
        linewidth=1.0,
        label="Paired G1 coverage",
    )
    ax_pair.plot(
        x,
        wheel_values,
        color=colors["orange"],
        marker="s",
        markerfacecolor="none",
        markersize=4.2,
        linewidth=1.0,
        linestyle="--",
        label="Wheel closed-loop coverage",
    )
    for index, (g1_value, wheel_value) in enumerate(
        zip(g1_values, wheel_values, strict=True),
        start=1,
    ):
        ax_pair.plot(
            [index, index],
            [g1_value, wheel_value],
            color=colors["neutral"],
            linewidth=0.45,
            alpha=0.65,
            zorder=1,
        )
    ax_pair.axhline(
        0.80,
        color=colors["vermillion"],
        linestyle="--",
        linewidth=0.9,
        label="Midterm threshold (0.80)",
    )
    ax_pair.axhline(
        0.99,
        color=colors["black"],
        linestyle=":",
        linewidth=0.9,
        label="Final threshold (0.99)",
    )
    ax_pair.set_xlim(0.5, 10.5)
    ax_pair.set_ylim(0.0, 1.02)
    ax_pair.set_xticks(x)
    ax_pair.set_xlabel("Paired wheel episode")
    ax_pair.set_ylabel("Coverage fraction")
    ax_pair.set_title("Ten paired G1/wheel observations")
    ax_pair.legend(loc="lower left", fontsize=5.0)
    _style_quantitative_axis(ax_pair, config)
    _panel_label(ax_pair, "a", config)

    deltas = [row["paired_delta"] for row in rows]
    delta_x = np.arange(1, len(deltas) + 1)
    positive = [value >= 0.0 for value in deltas]
    ax_delta.scatter(
        delta_x,
        deltas,
        s=22,
        marker="o",
        facecolor=[
            colors["bluish_green"] if item else colors["vermillion"]
            for item in positive
        ],
        edgecolor=colors["black"],
        linewidth=0.5,
        label="Episode delta",
        zorder=3,
    )
    ci = derived["g3"]["paired_delta_bootstrap_ci"]
    ax_delta.errorbar(
        [5.5],
        [ci["mean"]],
        yerr=[
            [ci["mean"] - ci["ci_low"]],
            [ci["ci_high"] - ci["mean"]],
        ],
        fmt="D",
        color=colors["black"],
        mfc="white",
        capsize=3.0,
        markersize=5.0,
        linewidth=1.2,
        label="Mean + deterministic 95% CI",
        zorder=5,
    )
    ax_delta.axhline(
        0.0,
        color=colors["neutral"],
        linestyle="-",
        linewidth=0.8,
        label="Zero reference",
    )
    ax_delta.axhline(
        -0.01,
        color=colors["vermillion"],
        linestyle="--",
        linewidth=1.0,
        label="Mean-delta floor (-0.01)",
    )
    extent = max(
        0.02,
        max(abs(value) for value in [*deltas, ci["ci_low"], ci["ci_high"]])
        * 1.25,
    )
    ax_delta.set_xlim(0.5, 10.5)
    ax_delta.set_ylim(-extent, extent)
    ax_delta.set_xticks(delta_x)
    ax_delta.set_xlabel("Paired wheel episode")
    ax_delta.set_ylabel("Wheel minus G1 coverage")
    ax_delta.set_title(
        f"Paired deltas (mean={ci['mean']:.4f}, n={len(deltas)})"
    )
    ax_delta.legend(loc="upper right", fontsize=4.2)
    _style_quantitative_axis(ax_delta, config)
    _panel_label(ax_delta, "b", config)

    replay_counts = derived["g3"]["platform_counts"]
    bars = ax_replay.bar(
        [0, 1],
        [replay_counts["legged"], replay_counts["hopper"]],
        color=[colors["sky_blue"], colors["reddish_purple"]],
        edgecolor=colors["black"],
        linewidth=0.7,
        hatch=["//", "xx"],
    )
    for bar, value in zip(
        bars,
        [replay_counts["legged"], replay_counts["hopper"]],
        strict=True,
    ):
        ax_replay.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.08,
            f"{value}/3 complete",
            ha="center",
            va="bottom",
            fontsize=5.5,
        )
    ax_replay.set_ylim(0.0, 3.45)
    ax_replay.set_xticks([0, 1], ["Legged", "Hopper"])
    ax_replay.set_xlabel("Simulation-proxy platform")
    ax_replay.set_ylabel("Verified replay count")
    ax_replay.set_title(
        "Interface replay completeness\n(simulation proxies only)"
    )
    _style_quantitative_axis(ax_replay, config)
    _panel_label(ax_replay, "c", config, x=0.02, y=0.93)

    metadata = _figure_metadata("fig04_g3_crosscheck", verified, config)
    _add_metadata_footer(fig, metadata)
    return fig


FIGURE_BUILDERS: Mapping[
    str,
    Callable[
        [VerifiedFigureInput, Mapping[str, Any], Mapping[str, Any]],
        Any,
    ],
] = {
    "fig01_dual_gate_overview": build_fig01_dual_gate_overview,
    "fig02_g1_coverage": build_fig02_g1_coverage,
    "fig03_g2_timing": build_fig03_g2_timing,
    "fig04_g3_crosscheck": build_fig04_g3_crosscheck,
}


def _overview_source_rows(
    verified: VerifiedFigureInput,
    derived: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    coverage_metrics = (
        (
            "g1",
            "Test-Q24",
            derived["g1"]["splits"]["test_q24"]["statistics"]["mean"],
            24,
            verified.gates["g1"]["status"],
        ),
        (
            "g1",
            "Unseen-24",
            derived["g1"]["splits"]["unseen24"]["statistics"]["mean"],
            24,
            verified.gates["g1"]["status"],
        ),
        (
            "g3",
            "wheel",
            verified.gates["g3"]["wheel_coverage_mean"],
            10,
            verified.gates["g3"]["status"],
        ),
    )
    for gate_id, group, value, sample_size, status in coverage_metrics:
        rows.append(
            {
                "panel": "coverage",
                "gate": gate_id,
                "group": group,
                "metric": "mean_coverage",
                "value": value,
                "unit": "coverage_fraction",
                "sample_size": sample_size,
                "midterm_threshold": 0.80,
                "midterm_margin": value - 0.80,
                "final_threshold": 0.99,
                "final_margin": value - 0.99,
                "evidence_status": status,
            }
        )
    for platform_name in PLATFORMS:
        for scale in SCALES:
            payload = derived["g2"]["groups"][f"{platform_name}/{scale}"]
            value = payload["statistics"]["p95_ms"]
            rows.append(
                {
                    "panel": "timing",
                    "gate": "g2",
                    "group": f"{platform_name}/{scale}",
                    "metric": "nearest_rank_p95",
                    "value": value,
                    "unit": "milliseconds",
                    "sample_size": payload["statistics"]["sample_count"],
                    "midterm_threshold": 2000.0,
                    "midterm_margin": 2000.0 - value,
                    "final_threshold": 1000.0,
                    "final_margin": 1000.0 - value,
                    "evidence_status": verified.gates["g2"]["status"],
                }
            )
    for group, timing in (
        ("wheel", verified.gates["g3"]["wheel_timing"]),
        ("interfaces", verified.gates["g3"]["interface_timing"]),
    ):
        value = timing["p95_ms"]
        rows.append(
            {
                "panel": "timing",
                "gate": "g3",
                "group": group,
                "metric": "nearest_rank_p95",
                "value": value,
                "unit": "milliseconds",
                "sample_size": timing["sample_count"],
                "midterm_threshold": 2000.0,
                "midterm_margin": 2000.0 - value,
                "final_threshold": 1000.0,
                "final_margin": 1000.0 - value,
                "evidence_status": verified.gates["g3"]["status"],
            }
        )
    status_rows = {
        "g1": (
            verified.gates["g1"]["g1_coverage_80_passed"],
            verified.gates["g1"]["g1_coverage_99_passed"],
        ),
        "g2": (
            verified.gates["g2"]["g2_all_platforms_2s_passed"],
            verified.gates["g2"]["g2_all_platforms_1s_passed"],
        ),
        "g3": (
            verified.gates["g3"]["g3_midterm_crosscheck_passed"],
            verified.gates["g3"]["g3_final_crosscheck_passed"],
        ),
        "aggregate": (
            verified.aggregate_summary["midterm_reduced_gate_passed"],
            verified.aggregate_summary["final_threshold_reduced_gate_passed"],
        ),
    }
    for gate_id, (midterm, final) in status_rows.items():
        rows.append(
            {
                "panel": "status",
                "gate": gate_id,
                "group": "midterm",
                "metric": "gate_status",
                "value": "passed" if midterm else "failed",
                "unit": "categorical_status",
                "sample_size": "",
                "midterm_threshold": "",
                "midterm_margin": "",
                "final_threshold": "",
                "final_margin": "",
                "evidence_status": "passed" if midterm else "failed",
            }
        )
        rows.append(
            {
                "panel": "status",
                "gate": gate_id,
                "group": "final",
                "metric": "gate_status",
                "value": "passed" if final else "failed",
                "unit": "categorical_status",
                "sample_size": "",
                "midterm_threshold": "",
                "midterm_margin": "",
                "final_threshold": "",
                "final_margin": "",
                "evidence_status": "passed" if final else "failed",
            }
        )
    return rows


def _g1_source_rows(derived: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in ("test_q24", "unseen24"):
        payload = derived["g1"]["splits"][split]
        ci = payload["bootstrap_ci"]
        statistics_payload = payload["statistics"]
        for row in payload["rows"]:
            rows.append(
                {
                    **row,
                    "unit": "coverage_fraction",
                    "sample_size": statistics_payload["sample_count"],
                    "split_mean": statistics_payload["mean"],
                    "split_nearest_rank_p50": statistics_payload["p50"],
                    "split_sample_sd": statistics_payload["sample_stddev"],
                    "bootstrap_ci_low": ci["ci_low"],
                    "bootstrap_ci_high": ci["ci_high"],
                    "bootstrap_ci_low_rank": ci["ci_low_rank"],
                    "bootstrap_ci_high_rank": ci["ci_high_rank"],
                    "bootstrap_seed": ci["seed"],
                    "bootstrap_derived_seed": ci["derived_seed"],
                    "bootstrap_resamples": ci["resamples"],
                    "coverage_midterm_threshold": 0.80,
                    "coverage_final_threshold": 0.99,
                }
            )
    return rows


def _g2_source_rows(derived: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for platform_name in PLATFORMS:
        for scale in SCALES:
            key = f"{platform_name}/{scale}"
            payload = derived["g2"]["groups"][key]
            statistics_payload = payload["statistics"]
            for row in payload["rows"]:
                rows.append(
                    {
                        **row,
                        "unit": "milliseconds",
                        "sample_size": statistics_payload["sample_count"],
                        "partition_mean_ms": statistics_payload["mean_ms"],
                        "partition_p95_ms": statistics_payload["p95_ms"],
                        "partition_sample_sd_ms": statistics_payload[
                            "sample_stddev_ms"
                        ],
                        "timing_final_threshold_ms": 1000.0,
                        "timing_midterm_threshold_ms": 2000.0,
                    }
                )
    return rows


def _g3_source_rows(
    derived: Mapping[str, Any],
) -> list[dict[str, Any]]:
    ci = derived["g3"]["paired_delta_bootstrap_ci"]
    rows: list[dict[str, Any]] = []
    for row in derived["g3"]["wheel_terminal"]:
        rows.append(
            {
                "record_kind": "wheel_pair",
                **row,
                "platform": "wheel",
                "capability_kind": "closed_loop_crosscheck",
                "complete": True,
                "unit": "coverage_fraction_and_delta",
                "sample_size": 10,
                "bootstrap_ci_low": ci["ci_low"],
                "bootstrap_ci_high": ci["ci_high"],
                "bootstrap_ci_low_rank": ci["ci_low_rank"],
                "bootstrap_ci_high_rank": ci["ci_high_rank"],
                "bootstrap_seed": ci["seed"],
                "bootstrap_derived_seed": ci["derived_seed"],
                "bootstrap_resamples": ci["resamples"],
                "coverage_midterm_threshold": 0.80,
                "coverage_final_threshold": 0.99,
                "paired_delta_floor": -0.01,
            }
        )
    for row in derived["g3"]["interface_replays"]:
        rows.append(
            {
                "record_kind": "interface_replay",
                **row,
                "unit": "replay_completeness",
                "sample_size": 3,
                "bootstrap_ci_low": "",
                "bootstrap_ci_high": "",
                "bootstrap_seed": "",
                "bootstrap_derived_seed": "",
                "bootstrap_resamples": "",
                "coverage_midterm_threshold": "",
                "coverage_final_threshold": "",
                "paired_delta_floor": "",
            }
        )
    return rows


def _family_source_rows(
    family: str,
    verified: VerifiedFigureInput,
    derived: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if family == "fig01_dual_gate_overview":
        return _overview_source_rows(verified, derived)
    if family == "fig02_g1_coverage":
        return _g1_source_rows(derived)
    if family == "fig03_g2_timing":
        return _g2_source_rows(derived)
    return _g3_source_rows(derived)


def _csv_text(
    rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> str:
    common = {
        "aggregate_manifest_sha256": metadata[
            "aggregate_manifest_sha256"
        ],
        "aggregate_run_id": metadata["run_ids"]["aggregate"],
        "g1_run_id": metadata["run_ids"]["g1"],
        "g2_run_id": metadata["run_ids"]["g2"],
        "g3_run_id": metadata["run_ids"]["g3"],
        "figure_run_id": metadata["figure_run_id"],
        "figure_family": metadata["figure_family"],
        "figure_status": metadata["status"],
        "reduced_qualifier": metadata["reduced_qualifier"],
        "statistical_method": "; ".join(metadata["statistical_methods"]),
        "figure_units": "; ".join(metadata["units"]),
    }
    row_fields = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in common
        }
    )
    fieldnames = [*common, *row_fields]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=fieldnames,
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    for row in rows:
        materialized = {**common, **dict(row)}
        writer.writerow(
            {
                key: (
                    json.dumps(
                        value,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    if isinstance(value, (list, dict, tuple))
                    else value
                )
                for key, value in materialized.items()
            }
        )
    return output.getvalue()


def _source_data_payload(
    family: str,
    rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SOURCE_DATA_SCHEMA_VERSION,
        "figure_family": family,
        "metadata": dict(metadata),
        "row_count": len(rows),
        "rows": [dict(row) for row in rows],
    }


def _export_figure_bytes(
    fig: Any,
    *,
    family: str,
    file_format: str,
    dpi: int,
    metadata: Mapping[str, Any],
) -> bytes:
    if file_format not in EXPORT_FORMATS:
        raise FigureBlocked("unsupported_figure_format")
    metadata_json = json.dumps(
        metadata,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    save_metadata: dict[str, Any] | None
    if file_format == "svg":
        save_metadata = {
            "Title": family,
            "Description": metadata_json,
            "Creator": FIGURE_RENDERER_ID,
            "Date": None,
        }
    elif file_format == "pdf":
        save_metadata = {
            "Title": family,
            "Subject": metadata_json,
            "Creator": FIGURE_RENDERER_ID,
            "CreationDate": None,
            "ModDate": None,
        }
    elif file_format == "png":
        save_metadata = {
            "Title": family,
            "Description": metadata_json,
            "Software": FIGURE_RENDERER_ID,
        }
    else:
        save_metadata = None
    output = io.BytesIO()
    kwargs: dict[str, Any] = {
        "format": file_format,
        "facecolor": "white",
    }
    if file_format in {"png", "tiff"}:
        kwargs["dpi"] = dpi
    if save_metadata is not None:
        kwargs["metadata"] = save_metadata
    fig.savefig(output, **kwargs)
    return output.getvalue()


def _length_inches(value: str, reason: str) -> float:
    match = re.fullmatch(
        r"\s*([0-9]+(?:\.[0-9]+)?)\s*(pt|in|mm|cm)\s*",
        value,
    )
    if match is None:
        raise FigureBlocked(reason)
    magnitude = float(match.group(1))
    return magnitude * {
        "pt": 1.0 / 72.0,
        "in": 1.0,
        "mm": 1.0 / 25.4,
        "cm": 10.0 / 25.4,
    }[match.group(2)]


def _inspect_export_bytes(
    payload: bytes,
    *,
    file_format: str,
    expected_size_inches: tuple[float, float],
    expected_dpi: int,
    dimension_tolerance_mm: float,
    dpi_tolerance: float,
) -> dict[str, Any]:
    expected_width, expected_height = expected_size_inches
    pixel_width: int | None = None
    pixel_height: int | None = None
    dpi_x: float | None = None
    dpi_y: float | None = None
    if file_format == "svg":
        try:
            root = ET.fromstring(payload)
        except ET.ParseError as exc:
            raise FigureBlocked("svg_export_invalid") from exc
        width_inches = _length_inches(
            str(root.attrib.get("width", "")),
            "svg_canvas_invalid",
        )
        height_inches = _length_inches(
            str(root.attrib.get("height", "")),
            "svg_canvas_invalid",
        )
        editable_text = any(
            str(element.tag).endswith("text")
            for element in root.iter()
        )
    elif file_format == "pdf":
        match = re.search(
            rb"/MediaBox\s*\[\s*([-+0-9.]+)\s+([-+0-9.]+)\s+"
            rb"([-+0-9.]+)\s+([-+0-9.]+)\s*\]",
            payload,
        )
        if match is None:
            raise FigureBlocked("pdf_canvas_invalid")
        x0, y0, x1, y1 = (float(item) for item in match.groups())
        width_inches = (x1 - x0) / 72.0
        height_inches = (y1 - y0) / 72.0
        editable_text = (
            b"/FontFile2" in payload or b"/CIDFontType2" in payload
        )
    elif file_format in {"png", "tiff"}:
        try:
            with Image.open(io.BytesIO(payload)) as image:
                pixel_width, pixel_height = image.size
                dpi_value = image.info.get("dpi")
        except (OSError, ValueError) as exc:
            raise FigureBlocked("raster_export_invalid") from exc
        if (
            not isinstance(dpi_value, tuple)
            or len(dpi_value) < 2
        ):
            raise FigureBlocked("raster_dpi_missing")
        dpi_x = float(dpi_value[0])
        dpi_y = float(dpi_value[1])
        if (
            not math.isclose(
                dpi_x,
                expected_dpi,
                rel_tol=0.0,
                abs_tol=dpi_tolerance,
            )
            or not math.isclose(
                dpi_y,
                expected_dpi,
                rel_tol=0.0,
                abs_tol=dpi_tolerance,
            )
            or pixel_width != round(expected_width * expected_dpi)
            or pixel_height != round(expected_height * expected_dpi)
        ):
            raise FigureBlocked("raster_dpi_or_pixels_invalid")
        width_inches = pixel_width / dpi_x
        height_inches = pixel_height / dpi_y
        editable_text = False
    else:
        raise FigureBlocked("unsupported_figure_format")
    width_mm = width_inches * 25.4
    height_mm = height_inches * 25.4
    if (
        not math.isclose(
            width_mm,
            expected_width * 25.4,
            rel_tol=0.0,
            abs_tol=dimension_tolerance_mm,
        )
        or not math.isclose(
            height_mm,
            expected_height * 25.4,
            rel_tol=0.0,
            abs_tol=dimension_tolerance_mm,
        )
    ):
        raise FigureBlocked(f"{file_format}_canvas_dimension_invalid")
    if file_format in {"svg", "pdf"} and editable_text is not True:
        raise FigureBlocked(f"{file_format}_editable_text_missing")
    return {
        "format": file_format,
        "width_mm": width_mm,
        "height_mm": height_mm,
        "pixel_width": pixel_width,
        "pixel_height": pixel_height,
        "dpi_x": dpi_x,
        "dpi_y": dpi_y,
        "editable_text": editable_text,
    }


def _backend_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "matplotlib": str(mpl.__version__),
        "numpy": str(np.__version__),
        "pillow": str(PILLOW_VERSION),
    }


def _code_lineage_document(
    snapshot: CodeLineageSnapshot,
) -> dict[str, Any]:
    return {
        **dict(snapshot.payload),
        "lineage_payload_sha256": snapshot.payload_sha256,
    }


def _code_lineage_bytes(snapshot: CodeLineageSnapshot) -> bytes:
    return _canonical_json_text(_code_lineage_document(snapshot)).encode(
        "utf-8"
    )


def _code_lineage_file_sha256(snapshot: CodeLineageSnapshot) -> str:
    return _sha256_bytes(_code_lineage_bytes(snapshot))


def _write_code_lineage_files(
    target: Path,
    snapshot: CodeLineageSnapshot,
) -> tuple[str, dict[str, str], tuple[str, ...]]:
    _verify_code_lineage(snapshot)
    lineage_bytes = _code_lineage_bytes(snapshot)
    lineage_sha256 = _sha256_bytes(lineage_bytes)
    artifact_io.write_bytes(target / "code_lineage.json", lineage_bytes)
    file_hashes = {"code_lineage.json": lineage_sha256}
    snapshot_paths: list[str] = []
    rows = snapshot.payload["files"]
    for row in rows:
        snapshot_path = row["snapshot_path"]
        if snapshot_path is None:
            continue
        relative = _safe_manifest_path(
            snapshot_path,
            "code_lineage_snapshot_path_invalid",
        )
        source_relative = str(row["path"])
        payload = snapshot.source_bytes[source_relative]
        if (
            _sha256_bytes(payload) != row["sha256"]
            or len(payload) != row["byte_count"]
        ):
            raise FigureBlocked("code_lineage_snapshot_bytes_invalid")
        artifact_io.write_bytes(target / relative, payload)
        file_hashes[relative] = _sha256_bytes(payload)
        snapshot_paths.append(relative)
    return lineage_sha256, file_hashes, tuple(sorted(snapshot_paths))


def _manifest_payload_sha256(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    stored = payload.pop("manifest_payload_sha256", None)
    if stored is None and "manifest_payload_sha256" in manifest:
        raise FigureBlocked("figure_manifest_payload_hash_invalid")
    return _sha256_bytes(_canonical_json_bytes(payload))


def _verify_figure_bundle_contents(
    root: Path,
    *,
    expected_run_id: str,
    require_path_identity: bool,
) -> dict[str, Any]:
    try:
        manifest_bytes = artifact_io.read_bytes(root / "figure_manifest.json")
        sidecar = artifact_io.read_text(
            root / "figure_manifest.sha256"
        ).strip()
    except OSError as exc:
        raise FigureBlocked("figure_manifest_missing") from exc
    manifest = _parse_json_object(
        manifest_bytes,
        "figure_manifest_invalid",
    )
    if (
        manifest.get("schema_version")
        != FIGURE_MANIFEST_SCHEMA_VERSION
        or manifest.get("figure_run_id") != expected_run_id
        or manifest.get("output_identity")
        != {
            "figure_run_id": expected_run_id,
            "relative_root": expected_run_id,
        }
    ):
        raise FigureBlocked("figure_output_identity_mismatch")
    if require_path_identity and root.name != expected_run_id:
        raise FigureBlocked("figure_output_identity_mismatch")
    manifest_payload_sha256 = manifest.get("manifest_payload_sha256")
    if (
        not _is_sha256(manifest_payload_sha256)
        or manifest_payload_sha256 != _manifest_payload_sha256(manifest)
    ):
        raise FigureBlocked("figure_manifest_payload_hash_invalid")
    if (
        not _is_sha256(sidecar)
        or sidecar != _sha256_bytes(manifest_bytes)
    ):
        raise FigureBlocked("figure_manifest_sidecar_invalid")
    file_hashes = manifest.get("file_hashes")
    if not isinstance(file_hashes, Mapping) or not file_hashes:
        raise FigureBlocked("figure_manifest_file_hashes_invalid")
    for relative, expected_sha256 in file_hashes.items():
        safe_relative = _safe_manifest_path(
            relative,
            "figure_manifest_file_path_invalid",
        )
        if not _is_sha256(expected_sha256):
            raise FigureBlocked("figure_manifest_file_hashes_invalid")
        try:
            payload = artifact_io.read_bytes(root / safe_relative)
        except OSError as exc:
            raise FigureBlocked(
                f"figure_file_missing:{safe_relative}"
            ) from exc
        if _sha256_bytes(payload) != expected_sha256:
            raise FigureBlocked(f"figure_file_hash_drift:{safe_relative}")
    actual_files = set(artifact_io.list_relative_files(root))
    expected_files = {
        *file_hashes,
        "figure_manifest.json",
        "figure_manifest.sha256",
    }
    if actual_files != expected_files:
        raise FigureBlocked("figure_bundle_file_set_invalid")
    try:
        lineage_bytes = artifact_io.read_bytes(root / "code_lineage.json")
    except OSError as exc:
        raise FigureBlocked("code_lineage_file_missing") from exc
    if _sha256_bytes(lineage_bytes) != manifest.get("code_lineage_sha256"):
        raise FigureBlocked("code_lineage_file_hash_invalid")
    lineage = _parse_json_object(
        lineage_bytes,
        "code_lineage_file_invalid",
    )
    lineage_payload_sha256 = lineage.pop("lineage_payload_sha256", None)
    if (
        not _is_sha256(lineage_payload_sha256)
        or lineage_payload_sha256
        != _sha256_bytes(_canonical_json_bytes(lineage))
    ):
        raise FigureBlocked("code_lineage_file_invalid")
    return manifest


def verify_figure_bundle(root: str | Path) -> dict[str, Any]:
    target = Path(root)
    if not target.is_absolute():
        raise FigureBlocked("figure_output_root_not_absolute")
    try:
        manifest = artifact_io.read_json(target / "figure_manifest.json")
    except (OSError, ValueError, TypeError) as exc:
        raise FigureBlocked("figure_manifest_invalid") from exc
    run_id = manifest.get("figure_run_id")
    if not isinstance(run_id, str):
        raise FigureBlocked("figure_output_identity_mismatch")
    return _verify_figure_bundle_contents(
        target,
        expected_run_id=run_id,
        require_path_identity=True,
    )


def render_figure_bundle(
    verified: VerifiedFigureInput,
    config: Mapping[str, Any],
    config_sha256: str,
    output_root: str | Path,
    *,
    figure_run_id: str,
    code_lineage: CodeLineageSnapshot,
) -> Path:
    """Render into a unique staging root and atomically publish on success."""

    if verified.status == "blocked":
        raise FigureBlocked("blocked_evidence_is_not_renderable")
    if verified.status not in {"passed", "failed"}:
        raise FigureBlocked("aggregate_status_invalid")
    figure_run_id = _validate_run_id(figure_run_id)
    target = Path(output_root)
    if not target.is_absolute() or target.name != figure_run_id:
        raise FigureBlocked("figure_output_identity_mismatch")
    if artifact_io.path_exists(target):
        raise FigureBlocked("figure_output_root_exists")
    _verify_code_lineage(code_lineage)
    derived = _derive_figure_data(verified, config)
    _apply_publication_style(config, config_sha256)
    style_sha256 = _sha256_bytes(
        _canonical_json_bytes(config["style"])
    )
    artifact_io.make_dirs(target.parent)
    staging = target.parent / (
        f".{figure_run_id}.incomplete-{uuid.uuid4().hex}"
    )
    if artifact_io.path_exists(staging):
        raise FigureBlocked("figure_staging_identity_collision")
    artifact_io.make_dirs(staging)
    incomplete_marker = staging / "INCOMPLETE.json"
    artifact_io.write_bytes(
        incomplete_marker,
        _canonical_json_text(
            {
                "schema_version": "xunce-mid-dual-figure-incomplete/v1",
                "figure_run_id": figure_run_id,
                "formal_evidence_eligible": False,
            }
        ).encode("utf-8"),
    )
    (
        code_lineage_sha256,
        all_file_hashes,
        code_snapshot_paths,
    ) = _write_code_lineage_files(staging, code_lineage)
    figures_manifest: list[dict[str, Any]] = []
    for family in FIGURE_FAMILIES:
        metadata = _figure_metadata(
            family,
            verified,
            config,
            figure_run_id=figure_run_id,
            code_lineage=code_lineage,
        )
        source_rows = _family_source_rows(family, verified, derived)
        source_payload = _source_data_payload(
            family,
            source_rows,
            metadata,
        )
        json_name = f"{family}_source_data.json"
        csv_name = f"{family}_source_data.csv"
        json_bytes = _canonical_json_text(source_payload).encode("utf-8")
        csv_bytes = _csv_text(source_rows, metadata).encode("utf-8")
        artifact_io.write_bytes(staging / json_name, json_bytes)
        artifact_io.write_bytes(staging / csv_name, csv_bytes)
        all_file_hashes[json_name] = _sha256_bytes(json_bytes)
        all_file_hashes[csv_name] = _sha256_bytes(csv_bytes)

        fig = FIGURE_BUILDERS[family](verified, derived, config)
        figure_files: list[dict[str, Any]] = []
        try:
            for file_format in EXPORT_FORMATS:
                file_name = f"{family}.{file_format}"
                payload = _export_figure_bytes(
                    fig,
                    family=family,
                    file_format=file_format,
                    dpi=int(config["raster_dpi"]),
                    metadata=metadata,
                )
                expected_size = tuple(
                    float(value)
                    for value in config["figure_sizes_inches"][family]
                )
                canvas = _inspect_export_bytes(
                    payload,
                    file_format=file_format,
                    expected_size_inches=expected_size,
                    expected_dpi=int(config["raster_dpi"]),
                    dimension_tolerance_mm=float(
                        config["canvas_contract"][
                            "dimension_tolerance_mm"
                        ]
                    ),
                    dpi_tolerance=float(
                        config["canvas_contract"][
                            "raster_dpi_tolerance"
                        ]
                    ),
                )
                artifact_io.write_bytes(staging / file_name, payload)
                digest = _sha256_bytes(payload)
                all_file_hashes[file_name] = digest
                figure_files.append(
                    {
                        "path": file_name,
                        "format": file_format,
                        "sha256": digest,
                        "dpi": (
                            int(config["raster_dpi"])
                            if file_format in {"png", "tiff"}
                            else None
                        ),
                        "editable_text": file_format in {"svg", "pdf"},
                        "canvas": canvas,
                    }
                )
        finally:
            plt.close(fig)
        figures_manifest.append(
            {
                "figure_family": family,
                "figure_run_id": figure_run_id,
                "status": metadata["status"],
                "code_lineage_sha256": code_lineage_sha256,
                "renderer_sha256": code_lineage.renderer_sha256,
                "canvas_inches": list(
                    config["figure_sizes_inches"][family]
                ),
                "sample_sizes": metadata["sample_sizes"],
                "statistical_methods": metadata["statistical_methods"],
                "units": metadata["units"],
                "source_data": [
                    {
                        "path": json_name,
                        "format": "json",
                        "sha256": all_file_hashes[json_name],
                        "row_count": len(source_rows),
                    },
                    {
                        "path": csv_name,
                        "format": "csv",
                        "sha256": all_file_hashes[csv_name],
                        "row_count": len(source_rows),
                    },
                ],
                "files": figure_files,
            }
        )
    manifest = {
        "schema_version": FIGURE_MANIFEST_SCHEMA_VERSION,
        "renderer_id": FIGURE_RENDERER_ID,
        "renderer_sha256": code_lineage.renderer_sha256,
        "code_lineage_sha256": code_lineage_sha256,
        "code_snapshot_paths": list(code_snapshot_paths),
        "figure_run_id": figure_run_id,
        "output_identity": {
            "figure_run_id": figure_run_id,
            "relative_root": figure_run_id,
        },
        "backend": {
            "name": "python-matplotlib",
            "library_versions": _backend_versions(),
        },
        "status": verified.status,
        "reduced_qualifier": REDUCED_QUALIFIER,
        "aggregate_manifest_sha256": verified.aggregate_manifest_sha256,
        "run_ids": dict(verified.run_ids),
        "source_manifest_sha256": dict(
            verified.source_manifest_sha256
        ),
        "input_hashes": dict(verified.input_hashes),
        "figure_config_sha256": config_sha256,
        "style_config_sha256": style_sha256,
        "bootstrap": {
            **dict(config["bootstrap"]),
            "ci_low_rank": _bootstrap_endpoint_ranks(
                resamples=int(config["bootstrap"]["resamples"]),
                confidence=float(config["bootstrap"]["confidence"]),
            )[0],
            "ci_high_rank": _bootstrap_endpoint_ranks(
                resamples=int(config["bootstrap"]["resamples"]),
                confidence=float(config["bootstrap"]["confidence"]),
            )[1],
        },
        "statistical_definitions": {
            "g1": (
                "episode-level mean, nearest-rank P50, sample SD, and "
                "deterministic percentile-bootstrap 95% CI of the mean"
            ),
            "g2": (
                "all formal raw calls, arithmetic mean, sample SD, and exact "
                "nearest-rank P95 within every platform-scale/outcome/class "
                "partition"
            ),
            "g3": (
                "wheel minus paired G1 episode deltas with deterministic "
                "percentile-bootstrap 95% CI of the mean; exact 3+3 "
                "simulation-proxy replay completeness"
            ),
            "hypothesis_tests": "none",
        },
        "thresholds": dict(config["thresholds"]),
        "canvas_contract": dict(config["canvas_contract"]),
        "visible_content_scope": config["visible_content_scope"],
        "claims": dict(config["claims"]),
        "image_integrity": {
            "quantitative_vector_plots_only": True,
            "raster_input_used": False,
            "smoothing_used": False,
            "row_trimming_used": False,
            "winsorization_used": False,
            "selective_visual_adjustment_used": False,
        },
        "figures": figures_manifest,
        "file_hashes": dict(sorted(all_file_hashes.items())),
    }
    manifest["manifest_payload_sha256"] = _manifest_payload_sha256(
        manifest
    )
    manifest_bytes = _canonical_json_text(manifest).encode("utf-8")
    artifact_io.write_bytes(staging / "figure_manifest.json", manifest_bytes)
    if artifact_io.read_bytes(staging / "figure_manifest.json") != manifest_bytes:
        raise FigureBlocked("figure_manifest_reread_mismatch")
    artifact_io.write_bytes(
        staging / "figure_manifest.sha256",
        (_sha256_bytes(manifest_bytes) + "\n").encode("ascii"),
    )
    os.remove(artifact_io.windows_safe_path(incomplete_marker))
    _verify_figure_bundle_contents(
        staging,
        expected_run_id=figure_run_id,
        require_path_identity=False,
    )
    _verify_code_lineage(code_lineage)
    if artifact_io.path_exists(target):
        raise FigureBlocked("figure_output_root_exists")
    try:
        os.replace(
            artifact_io.windows_safe_path(staging),
            artifact_io.windows_safe_path(target),
        )
    except OSError as exc:
        raise FigureBlocked("figure_atomic_publish_failed") from exc
    _verify_figure_bundle_contents(
        target,
        expected_run_id=figure_run_id,
        require_path_identity=True,
    )
    return target


def _validate_run_id(value: str) -> str:
    if RUN_ID_PATTERN.fullmatch(value) is None:
        raise FigureBlocked("figure_run_id_invalid")
    return value


def _validate_output_root(
    output_root: Path,
    config: Mapping[str, Any],
    run_id: str,
    *,
    enforce_formal_paths: bool,
) -> None:
    if artifact_io.path_exists(output_root):
        raise FigureBlocked("figure_output_root_exists")
    if enforce_formal_paths:
        base = Path(str(config["output_base"]))
        if (
            config["output_base"] != FORMAL_FIGURE_BASE
            or output_root.resolve() != (base / run_id).resolve()
        ):
            raise FigureBlocked("figure_output_root_invalid")


def run_figure_pipeline(
    *,
    config_path: str | Path,
    aggregate_root: str | Path,
    run_id: str,
    output_root: str | Path | None = None,
    enforce_formal_paths: bool = True,
) -> Path:
    """Validate the full input before resolving or creating any output root."""

    run_id = _validate_run_id(run_id)
    config, config_sha256 = _load_figure_config(config_path)
    target = (
        Path(output_root)
        if output_root is not None
        else Path(str(config["output_base"])) / run_id
    )
    _validate_output_root(
        target,
        config,
        run_id,
        enforce_formal_paths=enforce_formal_paths,
    )
    aggregate_module = _load_aggregate_module()
    code_lineage = _capture_runtime_code_lineage(
        config_path,
        aggregate_module,
    )
    _verify_code_lineage(code_lineage)
    verified = load_verified_figure_input(
        aggregate_root,
        enforce_formal_paths=enforce_formal_paths,
        aggregate_module=aggregate_module,
    )
    _verify_code_lineage(code_lineage)
    return render_figure_bundle(
        verified,
        config,
        config_sha256,
        target,
        figure_run_id=run_id,
        code_lineage=code_lineage,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Render publication-formal reduced midterm figures from one "
            "completed aggregate root."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--aggregate-root", required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        output_root = run_figure_pipeline(
            config_path=args.config,
            aggregate_root=args.aggregate_root,
            run_id=args.run_id,
        )
    except FigureBlocked as exc:
        print(
            json.dumps(
                {
                    "schema_version": "xunce-mid-dual-figure-cli-result/v1",
                    "status": "blocked",
                    "reason": str(exc),
                    "artifacts_written": False,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "schema_version": "xunce-mid-dual-figure-cli-result/v1",
                "status": "complete",
                "output_root": str(output_root).replace("\\", "/"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
