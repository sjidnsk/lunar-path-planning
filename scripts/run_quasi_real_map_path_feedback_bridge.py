from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any

from git_provenance import git_snapshot as _git_snapshot


SCHEMA_VERSION = "quasi-real-map-path-feedback-bridge-summary/v1"


def run_quasi_real_map_path_feedback_bridge(
    *,
    matrix_manifest_path: str | Path,
    output_root: str | Path,
    config: dict[str, Any] | None,
    repo_root: str | Path,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    _ensure_model_explorer_path(repo)
    from model_explorer.data.evaluation_matrix import load_quasi_real_evaluation_manifest
    from model_explorer.data.lola_south_pole import (
        LolaSouthPoleRoiConfig,
        write_lola_south_pole_scenarios_json,
    )
    from model_explorer.data.manifest import load_data_manifest, validate_data_manifest
    from model_explorer.data.raster import read_raster_window

    cfg = dict(config or {})
    top_k = int(cfg.get("top_k", 3))
    max_slices = int(cfg.get("max_slices", 10_000))
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    matrix = load_quasi_real_evaluation_manifest(matrix_manifest_path)
    validation = validate_data_manifest(matrix.dataset_manifest)
    validation.require_valid()
    data_manifest = load_data_manifest(matrix.dataset_manifest)
    raw_dir = Path(validation.raw_dir)
    dem_path = raw_dir / _product_file_name(data_manifest, role="shape_map_radius")
    count_path = raw_dir / _product_file_name(data_manifest, role="observation_count")
    resolution = _manifest_resolution(data_manifest)
    scenario_root = output / "path_planner_sidecars"
    scenario_root.mkdir(parents=True, exist_ok=True)

    path_feedback_scenarios: list[dict[str, Any]] = []
    slice_records: list[dict[str, Any]] = []
    context_id_missing_count = 0
    roi_groups: list[str] = []
    slice_index = 0
    for roi in matrix.rois:
        if slice_index >= max_slices:
            break
        dem_window = read_raster_window(
            dem_path,
            x=roi.roi_x,
            y=roi.roi_y,
            width=roi.roi_width,
            height=roi.roi_height,
        )
        count_window = read_raster_window(
            count_path,
            x=roi.roi_x,
            y=roi.roi_y,
            width=roi.roi_width,
            height=roi.roi_height,
        )
        scenario_id = f"lola_qreal_{roi.name}_{roi.split}_{slice_index:03d}"
        variant_id = f"{scenario_id}-seed-{roi.seed}"
        scenario_paths = write_lola_south_pole_scenarios_json(
            scenario_root,
            dem_window.values,
            count_window.values,
            dataset_id=str(data_manifest.get("dataset_id", "unknown")),
            data_class=str(data_manifest.get("data_class", "unknown")),
            region=str(data_manifest.get("region", "unknown")),
            resolution=resolution,
            config=LolaSouthPoleRoiConfig(
                roi_x=0,
                roi_y=0,
                roi_width=roi.roi_width,
                roi_height=roi.roi_height,
                candidate_count=roi.candidate_count,
                episode_count=1,
                seed=roi.seed,
                start_cell=(0, 0),
            ),
            source_config=LolaSouthPoleRoiConfig(
                roi_x=roi.roi_x,
                roi_y=roi.roi_y,
                roi_width=roi.roi_width,
                roi_height=roi.roi_height,
                candidate_count=roi.candidate_count,
                episode_count=1,
                seed=roi.seed,
                start_cell=(0, 0),
            ),
            metadata_extra={
                "scenario_id": scenario_id,
                "scenario_group": roi.name,
                "scenario_seed": roi.seed,
                "scenario_variant_id": variant_id,
                "roi_name": roi.name,
                "split": roi.split,
                "map_source": {
                    "kind": "lola_quasi_real_roi",
                    "dataset_id": str(data_manifest.get("dataset_id", "unknown")),
                    "roi_name": roi.name,
                    "split": roi.split,
                },
            },
        )
        generated_scenario_path = scenario_paths[0]
        scenario_payload = json.loads(generated_scenario_path.read_text(encoding="utf-8"))
        contract = scenario_payload["snapshots"][0]
        contract["metadata"] = dict(scenario_payload.get("metadata", {}))
        contract_path = scenario_root / f"{scenario_id}.contract.json"
        contract_path.write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
        generated_scenario_path.unlink(missing_ok=True)

        sidecar = _sidecar_from_roi(
            dem_window.values,
            count_window.values,
            contract=contract,
            scenario_id=scenario_id,
            data_manifest=data_manifest,
            roi=roi,
            resolution=resolution,
        )
        sidecar_path = scenario_root / f"{scenario_id}.path-planner-sidecar.json"
        sidecar_path.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8")

        first_goal = _first_goal(contract)
        context_id = _slice_context_id(
            scenario_id=scenario_id,
            scenario_group=roi.name,
            scenario_seed=roi.seed,
            scenario_variant_id=variant_id,
            top_k=top_k,
            goal_cell=first_goal.get("cell") if first_goal else None,
        )
        if context_id is None:
            context_id_missing_count += 1
        record = {
            "schema_version": "quasi-real-map-slice/v1",
            "scenario_id": scenario_id,
            "scenario_group": roi.name,
            "scenario_seed": roi.seed,
            "scenario_variant_id": variant_id,
            "dataset_id": str(data_manifest.get("dataset_id", "unknown")),
            "data_class": str(data_manifest.get("data_class", "unknown")),
            "map_source": sidecar["metadata"]["map_source"],
            "map_id": str(data_manifest.get("dataset_id", "unknown")),
            "slice_id": scenario_id,
            "roi_name": roi.name,
            "split": roi.split,
            "context_id": context_id,
            "context_id_schema_version": "policy-context-id/v1",
            "context_id_source": "stable_semantic_fields",
            "legacy_identity_fallback_used": False,
            "contract": str(contract_path),
            "sidecar": str(sidecar_path),
            "passable_ratio": sidecar["metadata"]["passable_ratio"],
        }
        slice_records.append(record)
        path_feedback_scenarios.append(
            {
                "scenario_id": scenario_id,
                "scenario_group": roi.name,
                "scenario_seed": roi.seed,
                "scenario_variant_id": variant_id,
                "contract": str(contract_path),
                "sidecar": str(sidecar_path),
                "current_cell": [0, 0],
            }
        )
        if roi.name not in roi_groups:
            roi_groups.append(roi.name)
        slice_index += 1

    slices_path = output / "quasi-real-map-slices.jsonl"
    slices_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in slice_records),
        encoding="utf-8",
    )
    manifest_path = output / "quasi-real-map-path-feedback-manifest.json"
    manifest_payload = {
        "schema_version": "path-feedback-manifest/v1",
        "scenario_set": "quasi_real_map_domain_gap",
        "diagnostic_profile": "execution",
        "acceptance_gate": "quasi-real-map-domain-gap",
        "top_k": top_k,
        "planner": {
            "backend": "path_planner_route",
            "path_planner_root": str(repo / "path-planner"),
            "python_executable": sys.executable,
            "extra_args": ["--planning-backend", str(cfg.get("planning_backend", "channel_aware_astar"))],
        },
        "scenarios": path_feedback_scenarios,
        "outputs": {
            "summary": str(output / "quasi-real-map-path-feedback-summary.json"),
            "report": str(output / "quasi-real-map-path-feedback-report.md"),
        },
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    reason_codes: list[str] = []
    if context_id_missing_count:
        reason_codes.append("real_map_context_id_missing")
    if not slice_records:
        reason_codes.append("real_map_bridge_contract_invalid")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "status": "passed" if not reason_codes else "failed",
        "reason_codes": reason_codes,
        "matrix_manifest": str(Path(matrix_manifest_path).resolve()),
        "path_feedback_manifest": str(manifest_path),
        "slice_records": str(slices_path),
        "slice_count": len(slice_records),
        "roi_group_count": len(roi_groups),
        "roi_groups": roi_groups,
        "context_id_missing_count": context_id_missing_count,
        "legacy_identity_fallback_count": 0,
        "git_provenance": {"current": _git_snapshot(repo), "current_matches_sources": True},
    }
    summary_path = output / "quasi-real-map-path-feedback-bridge-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["summary_output"] = str(summary_path)
    return summary


def _sidecar_from_roi(
    dem_values: list[list[float]] | tuple[tuple[float, ...], ...],
    count_values: list[list[float]] | tuple[tuple[float, ...], ...],
    *,
    contract: dict[str, Any],
    scenario_id: str,
    data_manifest: dict[str, Any],
    roi: Any,
    resolution: float,
) -> dict[str, Any]:
    dem = [[_finite(value) for value in row] for row in dem_values]
    counts = [[_finite(value) for value in row] for row in count_values]
    height = len(dem)
    width = len(dem[0])
    max_count = max((value for row in counts for value in row), default=1.0) or 1.0
    slope = _slope_grid(dem)
    max_slope = max((value for row in slope for value in row), default=1.0) or 1.0
    risk_grid: list[list[float]] = []
    confidence_grid: list[list[float]] = []
    cost_grid: list[list[float]] = []
    passable_mask: list[list[bool]] = []
    reachable_goal_cells = {tuple(goal["cell"]) for goal in contract.get("top_goals", []) if goal.get("reachable") is True}
    for y in range(height):
        risk_row: list[float] = []
        confidence_row: list[float] = []
        cost_row: list[float] = []
        mask_row: list[bool] = []
        for x in range(width):
            confidence = min(max(counts[y][x] / max_count, 0.0), 1.0)
            slope_value = min(max(slope[y][x] / max_slope, 0.0), 1.0)
            risk = min(max(0.65 * slope_value + 0.35 * (1.0 - confidence), 0.0), 1.0)
            passable = (confidence > 0.0 and risk < 0.92) or (x, y) in reachable_goal_cells or (x, y) == (0, 0)
            risk_row.append(risk)
            confidence_row.append(confidence)
            cost_row.append(float(1.0 + risk + 0.25 * slope_value))
            mask_row.append(bool(passable))
        risk_grid.append(risk_row)
        confidence_grid.append(confidence_row)
        cost_grid.append(cost_row)
        passable_mask.append(mask_row)
    passable_count = sum(1 for row in passable_mask for value in row if value)
    total_count = max(width * height, 1)
    return {
        "schema_version": "path-planner-sidecar/v1",
        "cost": cost_grid,
        "passable_mask": passable_mask,
        "terrain_layers": {
            "risk": risk_grid,
            "confidence": confidence_grid,
            "dem": dem,
            "observation_count": counts,
        },
        "metadata": {
            "source": "quasi-real-lola-map-bridge",
            "scenario_id": scenario_id,
            "map_source": {
                "kind": "lola_quasi_real_roi",
                "dataset_id": str(data_manifest.get("dataset_id", "unknown")),
                "region": str(data_manifest.get("region", "unknown")),
                "roi_name": roi.name,
                "split": roi.split,
                "roi": roi.bounds,
                "resolution_m": float(resolution),
            },
            "platform": "yutu2",
            "blocked_count": total_count - passable_count,
            "passable_ratio": passable_count / total_count,
        },
    }


def _slope_grid(values: list[list[float]]) -> list[list[float]]:
    rows: list[list[float]] = []
    for y, row in enumerate(values):
        out: list[float] = []
        for x, value in enumerate(row):
            neighbor_values = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    yy = y + dy
                    xx = x + dx
                    if 0 <= yy < len(values) and 0 <= xx < len(values[yy]):
                        neighbor_values.append(abs(value - values[yy][xx]))
            out.append(max(neighbor_values) if neighbor_values else 0.0)
        rows.append(out)
    return rows


def _slice_context_id(
    *,
    scenario_id: str,
    scenario_group: str,
    scenario_seed: int,
    scenario_variant_id: str,
    top_k: int,
    goal_cell: Any,
) -> str | None:
    if not isinstance(goal_cell, list) or len(goal_cell) != 2:
        return None
    fields = {
        "scenario_id": scenario_id,
        "scenario_group": scenario_group,
        "scenario_seed": int(scenario_seed),
        "scenario_variant_id": scenario_variant_id,
        "diagnostic_profile": "execution",
        "planning_backend": "path_planner_route",
        "top_k": int(top_k),
        "sample_type": "quasi_real_map_slice",
        "candidate_role": "quasi_real_map_slice",
        "source_action_index": 0,
        "policy_target_cell": [int(goal_cell[0]), int(goal_cell[1])],
        "execution_goal_cell": [int(goal_cell[0]), int(goal_cell[1])],
        "target_binding_mode": "source_selected_policy_target",
    }
    encoded = json.dumps(fields, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _first_goal(contract: dict[str, Any]) -> dict[str, Any] | None:
    for goal in contract.get("top_goals", []):
        if isinstance(goal, dict) and goal.get("reachable") is True:
            return goal
    goals = contract.get("top_goals", [])
    return goals[0] if goals else None


def _product_file_name(manifest: dict[str, Any], *, role: str) -> str:
    for product in manifest.get("products", []):
        if not isinstance(product, dict) or product.get("role") != role:
            continue
        for file_info in product.get("files", []):
            if isinstance(file_info, dict) and str(file_info.get("name", "")).lower().endswith(".jp2"):
                return str(file_info["name"])
        for file_info in product.get("files", []):
            if isinstance(file_info, dict) and file_info.get("name"):
                return str(file_info["name"])
    raise ValueError(f"manifest is missing product file for role {role!r}")


def _manifest_resolution(manifest: dict[str, Any]) -> float:
    projection = manifest.get("projection", {})
    if isinstance(projection, dict) and projection.get("map_scale_meters_per_pixel") is not None:
        return float(projection["map_scale_meters_per_pixel"])
    return 1.0


def _finite(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    return numeric if isfinite(numeric) else 0.0


def _ensure_model_explorer_path(repo_root: Path) -> None:
    src = repo_root / "model-explorer" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_config(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bridge LOLA quasi-real ROI slices into path-feedback artifacts.")
    parser.add_argument(
        "--matrix-manifest",
        default="model-explorer/data/manifests/lunar_south_pole_lro_lola_selection_matrix_v1.json",
    )
    parser.add_argument("--output-root", default="outputs/path_feedback_batch_quasi_real_map_domain_gap_v1")
    parser.add_argument("--config")
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    summary = run_quasi_real_map_path_feedback_bridge(
        matrix_manifest_path=args.matrix_manifest,
        output_root=args.output_root,
        config=_load_config(args.config),
        repo_root=repo_root,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
