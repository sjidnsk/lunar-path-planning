from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from urllib.request import url2pathname
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
REPO_ROOT = SCRIPT_DIR.parent
MODEL_EXPLORER_SRC = REPO_ROOT / "model-explorer" / "src"
if MODEL_EXPLORER_SRC.is_dir() and str(MODEL_EXPLORER_SRC) not in sys.path:
    sys.path.insert(0, str(MODEL_EXPLORER_SRC))

try:
    from git_provenance import git_snapshot
    from run_quasi_real_map_path_feedback_bridge import _first_goal, _sidecar_from_roi, _slice_context_id
    from xunce_platform_contract import apply_stage23_platform_defaults
except ModuleNotFoundError:  # pragma: no cover
    from scripts.git_provenance import git_snapshot
    from scripts.run_quasi_real_map_path_feedback_bridge import _first_goal, _sidecar_from_roi, _slice_context_id
    from scripts.xunce_platform_contract import apply_stage23_platform_defaults

from model_explorer.data.geotiff import GeoTiffDecodeUnavailable, read_geotiff_window
from model_explorer.data.lola_south_pole import LolaSouthPoleRoiConfig, write_lola_south_pole_scenarios_json


CONFIG_SCHEMA_VERSION = "xunce-stage23-2a-high-resolution-terrain-data-ingestion-config/v1"
SUMMARY_SCHEMA_VERSION = "xunce-stage23-2a-summary/v1"
ROUTING_SCHEMA_VERSION = "xunce-stage23-2a-next-stage-routing/v1"
MANIFEST_SCHEMA_VERSION = "xunce-stage23-2a-manifest/v1"
STAGE_ID = "xunce-stage23-2a-high-resolution-terrain-data-ingestion"

DEFAULT_CONFIG = "configs/xunce_stage23_2a_high_resolution_terrain_data_ingestion_v1.json"
DEFAULT_OUTPUT_ROOT = (
    "D:/CodexDownloads/lunar-path-planning/stage23_high_resolution_terrain_ingestion/"
    "outputs/path_feedback_batch_xunce_stage23_2a_high_resolution_terrain_data_ingestion_v1"
)

SUMMARY_FILE = "xunce-stage23-2a-summary.json"
DOWNLOAD_MANIFEST_FILE = "xunce-stage23-2a-download-manifest.json"
DATASET_VALIDATION_FILE = "xunce-stage23-2a-dataset-validation-summary.json"
HASH_AUDIT_FILE = "xunce-stage23-2a-hash-audit.json"
ROI_OVERLAP_FILE = "xunce-stage23-2a-roi-overlap-audit.json"
STAGE23_1_SMOKE_SUMMARY_FILE = "xunce-stage23-2a-rerun-stage23-1-summary.json"
ROUTING_FILE = "xunce-stage23-2a-next-stage-routing.json"
REPORT_FILE = "xunce-stage23-2a-report.md"
MANIFEST_FILE = "xunce-stage23-2a-manifest.json"
HIGH_RES_SLICES_FILE = "xunce-high-res-terrain-slices.jsonl"

EXPANSION_SUMMARY_FILE = "xunce-high-fidelity-real-map-roi-expansion-summary.json"
EXPANSION_SLICES_FILE = "xunce-high-fidelity-real-map-slices.jsonl"
EXPANSION_PATH_FEEDBACK_AUDIT_FILE = "xunce-high-fidelity-path-feedback-audit.json"

BOUNDARY_FIELDS = (
    "stage23_2a_authorized",
    "runs_new_ppo_update",
    "publishes_checkpoint",
    "replaces_default_policy",
    "connects_real_executor",
    "starts_online_canary",
)


@dataclass(frozen=True)
class _PreparedProduct:
    product_id: str
    role: str
    path: Path
    resolution_m: float
    projection: dict[str, Any]
    sha256: str
    source_url: str


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Stage23.2A high-resolution terrain data ingestion.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args(argv)
    summary = run_xunce_stage23_2a_high_resolution_terrain_data_prepare(
        config_path=Path(args.config),
        output_root=Path(args.output_root),
        repo_root=Path(args.repo_root),
    )
    print(json.dumps({"status": summary["status"], "next_required_change": summary["next_required_change"]}, ensure_ascii=False))
    return 0 if summary["status"] in {"passed", "partial"} else 1


def run_xunce_stage23_2a_high_resolution_terrain_data_prepare(
    *,
    config_path: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_root = _resolve_path(output_root, repo_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config = _load_config(config_path, repo_root)
    boundary_reasons = _boundary_rejections(config)

    primary_manifest_path = _resolve_path(Path(str(config["primary_manifest"])), repo_root)
    primary_manifest = _read_json(primary_manifest_path)
    validation_issues = _validate_high_res_source_manifest(primary_manifest)
    lroc_report = _lroc_candidate_report(_optional_manifest(config.get("lroc_manifest"), repo_root), config.get("roi_windows", []))

    products: dict[str, _PreparedProduct] = {}
    download_rows: list[dict[str, Any]] = []
    hash_rows: list[dict[str, Any]] = []
    decode_issue: str | None = None
    if not boundary_reasons and not validation_issues:
        try:
            products, download_rows, hash_rows = _prepare_products(
                primary_manifest,
                output_root=output_root,
                raw_data_root=_resolve_path(Path(str(config["raw_data_root"])), repo_root),
                download_missing=bool(config.get("download_missing", True)),
            )
        except (OSError, ValueError, GeoTiffDecodeUnavailable) as exc:
            decode_issue = str(exc)

    high_res_root = output_root / "high_res_roi_expansion"
    roi_result: dict[str, Any] = {
        "high_res_root": str(high_res_root),
        "slice_count": 0,
        "slope_blocked_cell_count_total": 0,
        "scenario_count": 0,
        "slope_source_counts": {},
    }
    if not boundary_reasons and not validation_issues and decode_issue is None:
        try:
            roi_result = _build_high_res_roi_expansion(
                primary_manifest,
                products,
                config,
                output_root=high_res_root,
                repo_root=repo_root,
            )
        except (OSError, ValueError, GeoTiffDecodeUnavailable) as exc:
            decode_issue = str(exc)

    stage23_1_summary: dict[str, Any] = {}
    stage23_1_smoke_executed = False
    if (
        not boundary_reasons
        and not validation_issues
        and decode_issue is None
        and bool(config.get("run_stage23_1_smoke", False))
    ):
        stage23_1_smoke_executed = True
        stage23_1_summary = _run_stage23_1_smoke(config, output_root=output_root, high_res_root=high_res_root, repo_root=repo_root)

    status, route, route_reason, blockers = _route(
        boundary_reasons=boundary_reasons,
        validation_issues=validation_issues,
        decode_issue=decode_issue,
        roi_result=roi_result,
        stage23_1_smoke_executed=stage23_1_smoke_executed,
        stage23_1_summary=stage23_1_summary,
    )

    dataset_validation = {
        "schema_version": "xunce-stage23-2a-dataset-validation-summary/v1",
        "primary_manifest": str(primary_manifest_path),
        "primary_dataset_id": primary_manifest.get("dataset_id"),
        "validation_issues": validation_issues,
        "decode_issue": decode_issue,
        "geotiff_reader_backend": "pillow_basic_tiff",
        "lroc_candidate_report": lroc_report,
    }
    download_manifest = {
        "schema_version": "xunce-stage23-2a-download-manifest/v1",
        "raw_data_root": str(_resolve_path(Path(str(config["raw_data_root"])), repo_root)),
        "products": download_rows,
    }
    hash_audit = {"schema_version": "xunce-stage23-2a-hash-audit/v1", "products": hash_rows}
    roi_overlap = {
        "schema_version": "xunce-stage23-2a-roi-overlap-audit/v1",
        "roi_windows": config.get("roi_windows", []),
        "high_res_root": str(high_res_root),
        "slice_count": roi_result["slice_count"],
        "lroc_candidate_report": lroc_report,
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "stage_id": STAGE_ID,
        "status": status,
        "next_required_change": route,
        "route_reason": route_reason,
        "blocking_reason_codes": blockers,
        "primary_manifest": str(primary_manifest_path),
        "primary_dataset_id": primary_manifest.get("dataset_id"),
        "primary_resolution_m": _manifest_resolution(primary_manifest),
        "preferred_for_stage23": bool(primary_manifest.get("preferred_for_stage23", False)),
        "source_priority": primary_manifest.get("source_priority"),
        "platform_contract_id": config.get("platform_contract_id"),
        "platform_contract_hash": config.get("platform_contract_hash"),
        "platform_max_climb_deg": float(config.get("platform_max_climb_deg", 30.0)),
        "max_traversable_slope_deg": float(config.get("max_traversable_slope_deg", 30.0)),
        "slope_sensitivity_thresholds_deg": config.get("slope_sensitivity_thresholds_deg", [20.0, 30.0]),
        "high_res_root": str(high_res_root),
        "high_res_slice_count": roi_result["slice_count"],
        "scenario_count": roi_result["scenario_count"],
        "slope_blocked_cell_count_total": roi_result["slope_blocked_cell_count_total"],
        "slope_source_counts": roi_result["slope_source_counts"],
        "stage23_1_smoke_executed": stage23_1_smoke_executed,
        "stage23_1_status": stage23_1_summary.get("status"),
        "stage23_1_next_required_change": stage23_1_summary.get("next_required_change"),
        "lroc_candidate_report": lroc_report,
        "stage23_2a_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
        "git_provenance": {"current": git_snapshot(repo_root), "current_matches_sources": True},
    }
    routing = {
        "schema_version": ROUTING_SCHEMA_VERSION,
        "status": status,
        "next_required_change": route,
        "route_reason": route_reason,
        "stage23_2a_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "summary": str(output_root / SUMMARY_FILE),
        "download_manifest": str(output_root / DOWNLOAD_MANIFEST_FILE),
        "dataset_validation": str(output_root / DATASET_VALIDATION_FILE),
        "hash_audit": str(output_root / HASH_AUDIT_FILE),
        "roi_overlap_audit": str(output_root / ROI_OVERLAP_FILE),
        "high_res_roi_expansion_root": str(high_res_root),
        "high_res_slices": str(high_res_root / HIGH_RES_SLICES_FILE),
        "stage23_1_smoke_summary": str(output_root / STAGE23_1_SMOKE_SUMMARY_FILE),
        "next_stage_routing": str(output_root / ROUTING_FILE),
        "report": str(output_root / REPORT_FILE),
    }

    _write_json(output_root / SUMMARY_FILE, summary)
    _write_json(output_root / DOWNLOAD_MANIFEST_FILE, download_manifest)
    _write_json(output_root / DATASET_VALIDATION_FILE, dataset_validation)
    _write_json(output_root / HASH_AUDIT_FILE, hash_audit)
    _write_json(output_root / ROI_OVERLAP_FILE, roi_overlap)
    _write_json(output_root / STAGE23_1_SMOKE_SUMMARY_FILE, stage23_1_summary)
    _write_json(output_root / ROUTING_FILE, routing)
    _write_json(output_root / MANIFEST_FILE, manifest)
    (output_root / REPORT_FILE).write_text(_render_report(summary), encoding="utf-8")
    return summary


def _validate_high_res_source_manifest(manifest: dict[str, Any]) -> list[str]:
    issues: set[str] = set()
    if not manifest.get("dataset_id"):
        issues.add("missing_dataset_id")
    products = manifest.get("products")
    if not isinstance(products, list) or not products:
        issues.add("missing_products")
        return sorted(issues)
    for product in products:
        if not isinstance(product, dict):
            issues.add("invalid_product")
            continue
        files = product.get("files")
        file_items = files if isinstance(files, list) else []
        if not product.get("source_url") and not any(isinstance(row, dict) and row.get("url") for row in file_items):
            issues.add("missing_source_url")
        if product.get("resolution_m") is None and _manifest_resolution(manifest) is None:
            issues.add("missing_resolution_m")
        if not product.get("source_hash_policy") and not any(isinstance(row, dict) and row.get("sha256") for row in file_items):
            issues.add("missing_source_hash_policy")
    return sorted(issues)


def _lroc_candidate_report(lroc_manifest: dict[str, Any], roi_windows: list[dict[str, Any]]) -> dict[str, Any]:
    products = [row for row in lroc_manifest.get("products", []) if isinstance(row, dict)]
    dtm_products = [row for row in products if str(row.get("role", "")).lower() in {"dtm", "nac_dtm", "dem"}]
    if not dtm_products:
        return {
            "schema_version": "xunce-stage23-2a-lroc-candidate-report/v1",
            "dataset_id": lroc_manifest.get("dataset_id"),
            "roi_window_count": len(roi_windows),
            "candidate_product_count": 0,
            "selected_product_count": 0,
            "default_replacement_allowed": False,
            "reason_codes": ["lroc_nac_dtm_roi_selection_required"],
            "candidate_products": [],
        }
    selected = [row for row in dtm_products if _product_overlaps_any_roi(row, roi_windows)]
    return {
        "schema_version": "xunce-stage23-2a-lroc-candidate-report/v1",
        "dataset_id": lroc_manifest.get("dataset_id"),
        "roi_window_count": len(roi_windows),
        "candidate_product_count": len(dtm_products),
        "selected_product_count": len(selected),
        "default_replacement_allowed": bool(selected),
        "reason_codes": [] if selected else ["lroc_nac_dtm_roi_selection_required"],
        "candidate_products": selected[:20],
    }


def _prepare_products(
    manifest: dict[str, Any],
    *,
    output_root: Path,
    raw_data_root: Path,
    download_missing: bool,
) -> tuple[dict[str, _PreparedProduct], list[dict[str, Any]], list[dict[str, Any]]]:
    dataset_id = str(manifest.get("dataset_id", "high_res"))
    dataset_root = raw_data_root / dataset_id
    dataset_root.mkdir(parents=True, exist_ok=True)
    products: dict[str, _PreparedProduct] = {}
    download_rows: list[dict[str, Any]] = []
    hash_rows: list[dict[str, Any]] = []
    for product in manifest.get("products", []):
        if not isinstance(product, dict):
            continue
        role = str(product.get("role") or "")
        file_info = _first_product_file(product)
        if file_info is None:
            continue
        name = str(file_info.get("name") or Path(str(urlparse(str(file_info.get("url") or product.get("source_url"))).path)).name)
        source_url = str(file_info.get("url") or product.get("source_url"))
        target = dataset_root / name
        if not target.exists():
            if not download_missing:
                raise FileNotFoundError(f"missing high-resolution product and download_missing=false: {target}")
            _download_or_copy(source_url, target)
        sha = _sha256_file(target)
        prepared = _PreparedProduct(
            product_id=str(product.get("product_id") or role or name),
            role=role,
            path=target,
            resolution_m=float(product.get("resolution_m") or _manifest_resolution(manifest) or 1.0),
            projection=dict(product.get("projection") or manifest.get("projection") or {}),
            sha256=sha,
            source_url=source_url,
        )
        products[role] = prepared
        download_rows.append(
            {
                "product_id": prepared.product_id,
                "role": role,
                "source_url": source_url,
                "local_path": str(target),
                "bytes": target.stat().st_size,
                "sha256": sha,
                "downloaded_at": _now_iso(),
            }
        )
        hash_rows.append({"product_id": prepared.product_id, "role": role, "sha256": sha, "source_hash_policy": product.get("source_hash_policy")})
    if "dem" not in products:
        raise ValueError("primary high-resolution manifest must include a dem product")
    return products, download_rows, hash_rows


def _build_high_res_roi_expansion(
    manifest: dict[str, Any],
    products: dict[str, _PreparedProduct],
    config: dict[str, Any],
    *,
    output_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    scenario_root = output_root / "path_planner_sidecars"
    scenario_root.mkdir(parents=True, exist_ok=True)
    dataset_id = str(manifest.get("dataset_id", "high_res"))
    dem_product = products["dem"]
    slope_product = products.get("slope_map")
    slope_product_is_scalar_degrees = _product_scalar_slope_degrees(manifest, slope_product.product_id if slope_product else None)
    roi_windows = list(config.get("roi_windows", []))
    max_traversable = float(config.get("max_traversable_slope_deg", 30.0))
    platform_lineage = dict(config.get("platform_lineage") or {})
    candidate_count = int(config.get("candidate_count", 8))
    slice_records: list[dict[str, Any]] = []
    high_res_slice_records: list[dict[str, Any]] = []
    path_feedback_scenarios: list[dict[str, Any]] = []
    slope_blocked_total = 0
    slope_source_counts: dict[str, int] = {}
    for index, roi_payload in enumerate(roi_windows):
        roi = _roi_namespace(roi_payload, candidate_count=candidate_count)
        scenario_id = f"{dataset_id}_{roi.name}_{roi.split}_{index:03d}"
        dem_window = read_geotiff_window(
            dem_product.path,
            x=roi.roi_x,
            y=roi.roi_y,
            width=roi.roi_width,
            height=roi.roi_height,
            resolution_m=dem_product.resolution_m,
            projection=dem_product.projection,
            nodata_value=roi_payload.get("dem_nodata_value", config.get("dem_nodata_value")),
        )
        slope_values = None
        slope_source = "derived_from_dem"
        if slope_product is not None and slope_product_is_scalar_degrees:
            slope_window = read_geotiff_window(
                slope_product.path,
                x=roi.roi_x,
                y=roi.roi_y,
                width=roi.roi_width,
                height=roi.roi_height,
                resolution_m=slope_product.resolution_m,
                projection=slope_product.projection,
                nodata_value=roi_payload.get("slope_nodata_value", config.get("slope_nodata_value")),
            )
            slope_values = slope_window.values
            slope_source = "provided_slope_map"
        elif slope_product is not None:
            slope_source = "derived_from_dem"
        count_values = tuple(tuple(1.0 for _ in range(dem_window.width)) for _ in range(dem_window.height))
        start_cell = tuple(int(value) for value in roi_payload.get("start_cell", [0, 0])[:2])
        scenario_paths = write_lola_south_pole_scenarios_json(
            scenario_root,
            dem_window.values,
            count_values,
            dataset_id=dataset_id,
            data_class=str(manifest.get("data_class", "high_resolution_terrain")),
            region=str(manifest.get("region", "lunar_south_pole")),
            resolution=dem_product.resolution_m,
            config=LolaSouthPoleRoiConfig(
                roi_x=0,
                roi_y=0,
                roi_width=roi.roi_width,
                roi_height=roi.roi_height,
                candidate_count=roi.candidate_count,
                episode_count=1,
                seed=roi.seed,
                start_cell=start_cell,
            ),
            source_config=LolaSouthPoleRoiConfig(
                roi_x=roi.roi_x,
                roi_y=roi.roi_y,
                roi_width=roi.roi_width,
                roi_height=roi.roi_height,
                candidate_count=roi.candidate_count,
                episode_count=1,
                seed=roi.seed,
                start_cell=start_cell,
            ),
            metadata_extra={
                "scenario_id": scenario_id,
                "scenario_group": roi.name,
                "scenario_seed": roi.seed,
                "scenario_variant_id": f"{scenario_id}-seed-{roi.seed}-start-{start_cell[0]}-{start_cell[1]}",
                "start_cell": [start_cell[0], start_cell[1]],
                "roi_name": roi.name,
                "split": roi.split,
                "map_source": {
                    "kind": "stage23_high_resolution_terrain_roi",
                    "dataset_id": dataset_id,
                    "source_priority": manifest.get("source_priority"),
                    "resolution_m": dem_product.resolution_m,
                    "slope_source": slope_source,
                },
            },
        )
        generated_scenario_path = scenario_paths[0]
        scenario_payload = json.loads(generated_scenario_path.read_text(encoding="utf-8"))
        contract = scenario_payload["snapshots"][0]
        contract["metadata"] = dict(scenario_payload.get("metadata", {}))
        file_stem = _artifact_file_stem(scenario_root, scenario_id=scenario_id, index=index)
        contract_path = scenario_root / f"{file_stem}.contract.json"
        _write_json(contract_path, contract)
        generated_scenario_path.unlink(missing_ok=True)
        sidecar = _sidecar_from_roi(
            dem_window.values,
            count_values,
            contract=contract,
            scenario_id=scenario_id,
            data_manifest=manifest,
            roi=roi,
            resolution=dem_product.resolution_m,
            max_traversable_slope_deg=max_traversable,
            slope_deg_values=slope_values,
            slope_source=slope_source,
            platform_contract_lineage=platform_lineage,
        )
        sidecar["metadata"]["map_source"]["kind"] = "stage23_high_resolution_terrain_roi"
        sidecar["metadata"]["map_source"]["dataset_id"] = dataset_id
        sidecar["metadata"]["map_source"]["source_priority"] = manifest.get("source_priority")
        sidecar["metadata"]["map_source"]["dem_product_id"] = dem_product.product_id
        if slope_product is not None:
            sidecar["metadata"]["map_source"]["slope_product_id"] = slope_product.product_id
        sidecar_path = scenario_root / f"{file_stem}.path-planner-sidecar.json"
        _write_json(sidecar_path, sidecar)
        first_goal = _first_goal(contract)
        variant_id = f"{scenario_id}-seed-{roi.seed}-start-{start_cell[0]}-{start_cell[1]}"
        context_id = _slice_context_id(
            scenario_id=scenario_id,
            scenario_group=roi.name,
            scenario_seed=roi.seed,
            scenario_variant_id=variant_id,
            top_k=int(config.get("top_k", 3)),
            goal_cell=first_goal.get("cell") if first_goal else None,
            start_cell=start_cell,
        )
        record = {
            "schema_version": "xunce-high-fidelity-real-map-slice/v1",
            "scenario_id": scenario_id,
            "scenario_group": roi.name,
            "scenario_seed": roi.seed,
            "scenario_variant_id": variant_id,
            "dataset_id": dataset_id,
            "data_class": str(manifest.get("data_class", "high_resolution_terrain")),
            "map_source": sidecar["metadata"]["map_source"],
            "map_id": dataset_id,
            "slice_id": scenario_id,
            "roi_name": roi.name,
            "split": roi.split,
            "context_id": context_id,
            "context_id_schema_version": "policy-context-id/v1",
            "context_id_source": "stable_semantic_fields",
            "legacy_identity_fallback_used": False,
            "start_cell": [start_cell[0], start_cell[1]],
            "contract": str(contract_path),
            "sidecar": str(sidecar_path),
            "passable_ratio": sidecar["metadata"]["passable_ratio"],
            "platform_contract_id": sidecar.get("platform_contract_id"),
            "platform_contract_hash": sidecar.get("platform_contract_hash"),
            "platform_max_climb_deg": sidecar.get("platform_max_climb_deg"),
            "max_traversable_slope_deg": sidecar["max_traversable_slope_deg"],
            "slope_blocked_cells": sidecar["slope_blocked_cells"],
            "slope_blocked_cell_count": sidecar["slope_blocked_cell_count"],
            "slope_source": sidecar["slope_source"],
            "resolution_m": dem_product.resolution_m,
        }
        slice_records.append(record)
        high_res_slice_records.append(
            {
                "schema_version": "xunce-high-res-terrain-slice/v1",
                "dataset_id": dataset_id,
                "scenario_id": scenario_id,
                "source_product_id": dem_product.product_id,
                "slope_product_id": slope_product.product_id if slope_product else None,
                "resolution_m": dem_product.resolution_m,
                "slope_source": slope_source,
                "platform_contract_id": sidecar.get("platform_contract_id"),
                "platform_contract_hash": sidecar.get("platform_contract_hash"),
                "platform_max_climb_deg": sidecar.get("platform_max_climb_deg"),
                "max_traversable_slope_deg": sidecar["max_traversable_slope_deg"],
                "roi": {"x": roi.roi_x, "y": roi.roi_y, "width": roi.roi_width, "height": roi.roi_height},
                "slope_blocked_cell_count": sidecar["slope_blocked_cell_count"],
            }
        )
        path_feedback_scenarios.append(
            {
                "scenario_id": scenario_id,
                "scenario_group": roi.name,
                "scenario_seed": roi.seed,
                "scenario_variant_id": variant_id,
                "contract": str(contract_path),
                "sidecar": str(sidecar_path),
                "current_cell": [start_cell[0], start_cell[1]],
                "slope_blocked_cells": sidecar["slope_blocked_cells"],
                "slope_blocked_cell_count": sidecar["slope_blocked_cell_count"],
                "slope_source": sidecar["slope_source"],
                "platform_contract_id": sidecar.get("platform_contract_id"),
                "platform_contract_hash": sidecar.get("platform_contract_hash"),
                "platform_max_climb_deg": sidecar.get("platform_max_climb_deg"),
                "max_traversable_slope_deg": sidecar["max_traversable_slope_deg"],
            }
        )
        slope_blocked_total += int(sidecar["slope_blocked_cell_count"])
        slope_source_counts[slope_source] = slope_source_counts.get(slope_source, 0) + 1

    _write_jsonl(output_root / EXPANSION_SLICES_FILE, slice_records)
    _write_jsonl(output_root / HIGH_RES_SLICES_FILE, high_res_slice_records)
    expansion_summary = {
        "schema_version": "xunce-high-fidelity-real-map-roi-expansion-summary/v1",
        "status": "passed" if slice_records else "failed",
        "dataset_id": dataset_id,
        "data_class": str(manifest.get("data_class", "high_resolution_terrain")),
        "slice_count": len(slice_records),
        "source_priority": manifest.get("source_priority"),
        "preferred_for_stage23": bool(manifest.get("preferred_for_stage23", False)),
        "resolution_m": dem_product.resolution_m,
        "platform_contract_id": platform_lineage.get("platform_contract_id"),
        "platform_contract_hash": platform_lineage.get("platform_contract_hash"),
        "platform_max_climb_deg": platform_lineage.get("platform_max_climb_deg"),
        "max_traversable_slope_deg": max_traversable,
        "slope_blocked_cell_count_total": slope_blocked_total,
        "slope_source_counts": slope_source_counts,
    }
    path_feedback = {
        "schema_version": "xunce-high-fidelity-path-feedback-audit/v1",
        "scenario_count": len(path_feedback_scenarios),
        "scenarios": path_feedback_scenarios,
    }
    _write_json(output_root / EXPANSION_SUMMARY_FILE, expansion_summary)
    _write_json(output_root / EXPANSION_PATH_FEEDBACK_AUDIT_FILE, path_feedback)
    return {
        "high_res_root": str(output_root),
        "slice_count": len(slice_records),
        "scenario_count": len(path_feedback_scenarios),
        "slope_blocked_cell_count_total": slope_blocked_total,
        "slope_source_counts": slope_source_counts,
    }


def _run_stage23_1_smoke(
    config: dict[str, Any],
    *,
    output_root: Path,
    high_res_root: Path,
    repo_root: Path,
) -> dict[str, Any]:
    try:
        from run_xunce_stage23_1_slope_derived_obstacle_source_for_endpoint_theta_los import (
            run_xunce_stage23_1_slope_derived_obstacle_source_for_endpoint_theta_los,
        )
    except ModuleNotFoundError:  # pragma: no cover
        from scripts.run_xunce_stage23_1_slope_derived_obstacle_source_for_endpoint_theta_los import (
            run_xunce_stage23_1_slope_derived_obstacle_source_for_endpoint_theta_los,
        )

    stage23_0a_config = _read_json(_resolve_path(Path(str(config["stage23_0a_config"])), repo_root))
    stage23_0a_config.update(
        {
            "source_roi_expansion_root": str(high_res_root),
            "run_high_fidelity_smoke": True,
            "platform_contract": config.get("platform_contract"),
            "platform_contract_id": config.get("platform_contract_id"),
            "platform_contract_hash": config.get("platform_contract_hash"),
            "platform_max_climb_deg": float(config.get("platform_max_climb_deg", 30.0)),
            "max_traversable_slope_deg": float(config.get("max_traversable_slope_deg", 30.0)),
            "slope_sensitivity_thresholds_deg": config.get("slope_sensitivity_thresholds_deg", [20.0, 30.0]),
            "required_scenario_count": int(config.get("stage23_1_required_scenario_count", 1)),
            "rollout_steps": int(config.get("stage23_1_rollout_steps", 2)),
            "dynamic_max_candidates_per_step": int(config.get("dynamic_max_candidates_per_step", 8)),
            "dynamic_proposal_pool_limit_per_step": int(config.get("dynamic_proposal_pool_limit_per_step", 32)),
            "stage23_0a_authorized": False,
            "runs_new_ppo_update": False,
            "publishes_checkpoint": False,
            "replaces_default_policy": False,
            "connects_real_executor": False,
            "starts_online_canary": False,
            "canary_traffic_fraction": 0.0,
        }
    )
    stage23_0a_config_path = output_root / "xunce-stage23-2a-stage23-0a-config.json"
    _write_json(stage23_0a_config_path, stage23_0a_config)
    stage23_1_config = {
        "schema_version": "xunce-stage23-1-slope-derived-obstacle-source-for-endpoint-theta-los-config/v1",
        "run_stage23_0a": True,
        "stage23_0a_config": str(stage23_0a_config_path),
        "stage23_0a_output_subdir": "s23_1_a",
        "platform_contract": config.get("platform_contract"),
        "platform_contract_id": config.get("platform_contract_id"),
        "platform_contract_hash": config.get("platform_contract_hash"),
        "platform_max_climb_deg": float(config.get("platform_max_climb_deg", 30.0)),
        "max_traversable_slope_deg": float(config.get("max_traversable_slope_deg", 30.0)),
        "slope_sensitivity_thresholds_deg": config.get("slope_sensitivity_thresholds_deg", [20.0, 30.0]),
        "stage23_1_authorized": False,
        "runs_new_ppo_update": False,
        "publishes_checkpoint": False,
        "replaces_default_policy": False,
        "connects_real_executor": False,
        "starts_online_canary": False,
        "canary_traffic_fraction": 0.0,
    }
    stage23_1_config_path = output_root / "xunce-stage23-2a-stage23-1-config.json"
    _write_json(stage23_1_config_path, stage23_1_config)
    return run_xunce_stage23_1_slope_derived_obstacle_source_for_endpoint_theta_los(
        config_path=stage23_1_config_path,
        output_root=output_root / "s23_1",
        repo_root=repo_root,
    )


def _route(
    *,
    boundary_reasons: list[str],
    validation_issues: list[str],
    decode_issue: str | None,
    roi_result: dict[str, Any],
    stage23_1_smoke_executed: bool,
    stage23_1_summary: dict[str, Any],
) -> tuple[str, str, str, list[str]]:
    if boundary_reasons:
        return "failed", "resolve_stage23_2a_boundary_rejections", "boundary_rejection", sorted(boundary_reasons)
    if validation_issues:
        return "failed", "repair_stage23_2a_high_res_manifest", "manifest_validation_failed", list(validation_issues)
    if decode_issue:
        return "failed", "install_or_provide_geotiff_reader_dependency", "geotiff_decode_or_download_failed", [decode_issue]
    if int(roi_result.get("slice_count", 0)) <= 0:
        return "failed", "rerun_stage23_2a_required_inputs", "no_high_res_roi_slices", ["no_high_res_roi_slices"]
    if stage23_1_smoke_executed:
        if stage23_1_summary.get("status") != "passed":
            return "failed", "repair_stage23_2a_stage23_1_high_res_smoke", "stage23_1_high_res_smoke_failed", [
                str(stage23_1_summary.get("next_required_change") or "stage23_1_failed")
            ]
        return "passed", str(stage23_1_summary.get("next_required_change") or "run_stage23_2b_slope_obstacle_aware_theta_reward_contract"), "stage23_1_high_res_smoke_passed", []
    return "passed", "run_stage23_2b_slope_obstacle_aware_theta_reward_contract", "high_res_ingestion_ready_without_stage23_1_smoke", []


def _load_config(config_path: Path, repo_root: Path) -> dict[str, Any]:
    payload = _read_json(_resolve_path(config_path, repo_root))
    if payload.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA_VERSION!r}")
    payload.setdefault("primary_manifest", "model-explorer/data/manifests/lunar_south_pole_usgs_lro_dem_slope_4m.json")
    payload.setdefault("lroc_manifest", "model-explorer/data/manifests/lunar_lroc_nac_dtm_roi_2m_5m.json")
    payload.setdefault("raw_data_root", "D:/CodexDownloads/lunar-path-planning/data/raw/high_resolution_lunar_terrain")
    payload.setdefault("download_missing", True)
    payload.setdefault("run_stage23_1_smoke", False)
    payload.setdefault("stage23_0a_config", "configs/xunce_stage23_0a_materialize_obstacle_sources_for_theta_los_v1.json")
    payload.setdefault("candidate_count", 8)
    payload.setdefault("top_k", 3)
    if not isinstance(payload.get("roi_windows"), list) or not payload["roi_windows"]:
        payload["roi_windows"] = [{"roi_name": "usgs4m_smoke", "split": "train", "x": 0, "y": 0, "width": 32, "height": 32}]
    return apply_stage23_platform_defaults(payload, repo_root=repo_root)


def _boundary_rejections(config: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for field in BOUNDARY_FIELDS:
        if bool(config.get(field, False)):
            reasons.append(f"{field}_must_be_false")
    if float(config.get("canary_traffic_fraction", 0.0) or 0.0) != 0.0:
        reasons.append("canary_traffic_fraction_must_be_zero")
    return reasons


def _optional_manifest(path_value: Any, repo_root: Path) -> dict[str, Any]:
    if not path_value:
        return {}
    path = _resolve_path(Path(str(path_value)), repo_root)
    if not path.exists():
        return {}
    return _read_json(path)


def _manifest_resolution(manifest: dict[str, Any]) -> float | None:
    projection = manifest.get("projection", {})
    if isinstance(projection, dict) and projection.get("map_scale_meters_per_pixel") is not None:
        return float(projection["map_scale_meters_per_pixel"])
    if manifest.get("resolution_m") is not None:
        return float(manifest["resolution_m"])
    return None


def _first_product_file(product: dict[str, Any]) -> dict[str, Any] | None:
    files = product.get("files")
    if isinstance(files, list):
        for row in files:
            if isinstance(row, dict) and (row.get("url") or row.get("name")):
                return row
    if product.get("source_url"):
        return {"name": Path(urlparse(str(product["source_url"])).path).name, "url": product["source_url"]}
    return None


def _download_or_copy(source_url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(str(source_url))
    if parsed.scheme == "file":
        shutil.copyfile(Path(url2pathname(parsed.path)), target)
        return
    if parsed.scheme in {"", None}:
        shutil.copyfile(Path(source_url), target)
        return
    request = urllib.request.Request(source_url, headers={"User-Agent": "Codex lunar-path-planning Stage23.2A"})
    with urllib.request.urlopen(request, timeout=120) as response:
        with target.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _product_overlaps_any_roi(product: dict[str, Any], roi_windows: list[dict[str, Any]]) -> bool:
    product_roi = product.get("roi")
    if not isinstance(product_roi, dict):
        return False
    for roi in roi_windows:
        if _rects_overlap(product_roi, roi):
            return True
    return False


def _product_scalar_slope_degrees(manifest: dict[str, Any], product_id: str | None) -> bool:
    if not product_id:
        return False
    for product in manifest.get("products", []):
        if not isinstance(product, dict) or str(product.get("product_id")) != str(product_id):
            continue
        encoding = str(product.get("value_encoding", "scalar_slope_degrees")).lower()
        return encoding in {"scalar_slope_degrees", "slope_degrees", "float_slope_degrees"}
    return False


def _rects_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    ax0, ay0 = int(a.get("x", 0)), int(a.get("y", 0))
    aw, ah = int(a.get("width", 0)), int(a.get("height", 0))
    bx0, by0 = int(b.get("x", 0)), int(b.get("y", 0))
    bw, bh = int(b.get("width", 0)), int(b.get("height", 0))
    return ax0 < bx0 + bw and ax0 + aw > bx0 and ay0 < by0 + bh and ay0 + ah > by0


def _roi_namespace(payload: dict[str, Any], *, candidate_count: int) -> Any:
    from types import SimpleNamespace

    x = int(payload.get("x", payload.get("roi_x", 0)))
    y = int(payload.get("y", payload.get("roi_y", 0)))
    width = int(payload.get("width", payload.get("roi_width", 32)))
    height = int(payload.get("height", payload.get("roi_height", 32)))
    return SimpleNamespace(
        name=str(payload.get("roi_name", payload.get("name", "high_res_roi"))),
        split=str(payload.get("split", "train")),
        seed=int(payload.get("seed", 0)),
        roi_x=x,
        roi_y=y,
        roi_width=width,
        roi_height=height,
        bounds=[x, y, x + width, y + height],
        candidate_count=int(payload.get("candidate_count", candidate_count)),
    )


def _artifact_file_stem(scenario_root: Path, *, scenario_id: str, index: int) -> str:
    candidate = scenario_root / f"{scenario_id}.path-planner-sidecar.json"
    if len(str(candidate)) < 240:
        return scenario_id
    return f"hr_{index:03d}"


def _resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else (repo_root / path).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _render_report(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage23.2A High-Resolution Terrain Data Ingestion",
            "",
            f"- status: `{summary['status']}`",
            f"- next_required_change: `{summary['next_required_change']}`",
            f"- primary_dataset_id: `{summary['primary_dataset_id']}`",
            f"- primary_resolution_m: `{summary['primary_resolution_m']}`",
            f"- platform_contract_id: `{summary.get('platform_contract_id')}`",
            f"- platform_max_climb_deg: `{summary.get('platform_max_climb_deg')}`",
            f"- max_traversable_slope_deg: `{summary.get('max_traversable_slope_deg')}`",
            f"- slope_sensitivity_thresholds_deg: `{summary.get('slope_sensitivity_thresholds_deg')}`",
            f"- high_res_slice_count: `{summary['high_res_slice_count']}`",
            f"- slope_blocked_cell_count_total: `{summary['slope_blocked_cell_count_total']}`",
            f"- stage23_1_smoke_executed: `{summary['stage23_1_smoke_executed']}`",
            "",
            "本阶段只准备高分辨率 DEM/slope source 和兼容 sidecar，不启动 PPO、不发布 checkpoint、不替换默认策略。",
            "`slope_blocked_cells` 是 slope-derived obstacle proxy，不是真实岩石、墙体或裂缝标注。",
            "",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
