from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "scripts", REPO_ROOT / "model-explorer" / "src"):
    _value = str(_path)
    if _value not in sys.path:
        sys.path.insert(0, _value)


def test_geotiff_window_reader_uses_manifest_resolution_and_nodata(tmp_path: Path) -> None:
    from PIL import Image

    from model_explorer.data.geotiff import read_geotiff_window

    tif = tmp_path / "tiny.tif"
    image = Image.new("F", (4, 4))
    image.putdata([float(i) for i in range(16)])
    image.save(tif)

    window = read_geotiff_window(
        tif,
        x=1,
        y=1,
        width=2,
        height=2,
        resolution_m=4.0,
        nodata_value=10.0,
        projection={"type": "polar stereographic"},
    )

    assert window.width == 2
    assert window.height == 2
    assert window.resolution_m == 4.0
    assert window.values == ((5.0, 6.0), (9.0, 10.0))
    assert window.nodata_mask == ((False, False), (False, True))
    assert window.projection["type"] == "polar stereographic"
    assert window.reader_backend == "pillow_basic_tiff"


def test_high_res_manifest_validation_requires_source_url_hash_and_resolution(tmp_path: Path) -> None:
    from scripts.run_xunce_stage23_2a_high_resolution_terrain_data_prepare import (
        _validate_high_res_source_manifest,
    )

    manifest = {
        "dataset_id": "bad",
        "products": [
            {
                "product_id": "missing-fields",
                "role": "dem",
                "files": [{"name": "bad.tif"}],
            }
        ],
    }

    issues = _validate_high_res_source_manifest(manifest)

    assert "missing_source_url" in issues
    assert "missing_resolution_m" in issues
    assert "missing_source_hash_policy" in issues


def test_sidecar_prefers_provided_slope_map_over_dem_derived_slope() -> None:
    from scripts.run_quasi_real_map_path_feedback_bridge import _sidecar_from_roi

    sidecar = _sidecar_from_roi(
        dem_values=[
            [0.0, 100.0],
            [0.0, 0.0],
        ],
        count_values=[
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        contract={"top_goals": []},
        scenario_id="high-res-slope",
        data_manifest={"dataset_id": "fixture", "region": "unit"},
        roi=SimpleNamespace(name="roi", split="train", bounds=[0, 0, 2, 2]),
        resolution=4.0,
        max_traversable_slope_deg=20.0,
        slope_deg_values=[
            [0.0, 0.0],
            [0.0, 25.0],
        ],
        slope_source="provided_slope_map",
    )

    assert sidecar["slope_source"] == "provided_slope_map"
    assert sidecar["terrain_layers"]["slope_deg"] == [[0.0, 0.0], [0.0, 25.0]]
    assert sidecar["slope_blocked_cells"] == [[1, 1]]
    assert sidecar["slope_blocked_obstacle_source_kind"] == "slope_blocked_as_obstacle_proxy"
    assert "obstacle_cells" not in sidecar


def test_sidecar_dem_only_derives_physical_slope() -> None:
    from scripts.run_quasi_real_map_path_feedback_bridge import _sidecar_from_roi

    sidecar = _sidecar_from_roi(
        dem_values=[
            [0.0, 10.0],
            [0.0, 0.0],
        ],
        count_values=[
            [1.0, 1.0],
            [1.0, 1.0],
        ],
        contract={"top_goals": []},
        scenario_id="high-res-dem",
        data_manifest={"dataset_id": "fixture", "region": "unit"},
        roi=SimpleNamespace(name="roi", split="train", bounds=[0, 0, 2, 2]),
        resolution=20.0,
        max_traversable_slope_deg=20.0,
    )

    assert sidecar["slope_source"] == "derived_from_dem"
    assert sidecar["slope_blocked_cells"]
    assert sidecar["metadata"]["slope_source"] == "derived_from_dem"


def test_lroc_without_roi_overlap_reports_candidates_without_default_replacement() -> None:
    from scripts.run_xunce_stage23_2a_high_resolution_terrain_data_prepare import _lroc_candidate_report

    report = _lroc_candidate_report(
        {"dataset_id": "lroc", "products": []},
        roi_windows=[{"roi_name": "target", "x": 0, "y": 0, "width": 32, "height": 32}],
    )

    assert report["candidate_product_count"] == 0
    assert report["selected_product_count"] == 0
    assert report["default_replacement_allowed"] is False
    assert "lroc_nac_dtm_roi_selection_required" in report["reason_codes"]


def test_stage23_2a_runner_generates_high_res_root_from_local_fixture(tmp_path: Path) -> None:
    from PIL import Image

    from scripts.run_xunce_stage23_2a_high_resolution_terrain_data_prepare import (
        SUMMARY_FILE,
        run_xunce_stage23_2a_high_resolution_terrain_data_prepare,
    )

    raw = tmp_path / "raw"
    raw.mkdir()
    dem = raw / "dem.tif"
    slope = raw / "slope.tif"
    dem_image = Image.new("F", (8, 8))
    dem_image.putdata([float(i % 8) for i in range(64)])
    dem_image.save(dem)
    slope_image = Image.new("F", (8, 8))
    slope_image.putdata([35.0 if i % 3 == 0 else 5.0 for i in range(64)])
    slope_image.save(slope)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "dataset_id": "fixture_usgs_4m",
                "data_class": "high_resolution_terrain",
                "preferred_for_stage23": True,
                "source_priority": "primary_usgs_4m",
                "region": "lunar_south_pole",
                "projection": {"map_scale_meters_per_pixel": 4.0},
                "products": [
                    {
                        "product_id": "fixture-dem",
                        "role": "dem",
                        "resolution_m": 4.0,
                        "source_url": dem.as_uri(),
                        "source_hash_policy": "runtime_sha256",
                        "files": [{"name": "dem.tif", "url": dem.as_uri()}],
                    },
                    {
                        "product_id": "fixture-slope",
                        "role": "slope_map",
                        "resolution_m": 4.0,
                        "source_url": slope.as_uri(),
                        "source_hash_policy": "runtime_sha256",
                        "files": [{"name": "slope.tif", "url": slope.as_uri()}],
                    },
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
                {
                    "schema_version": "xunce-stage23-2a-high-resolution-terrain-data-ingestion-config/v1",
                    "primary_manifest": str(manifest),
                    "raw_data_root": str(tmp_path / "downloaded_raw"),
                    "download_missing": True,
                "run_stage23_1_smoke": False,
                "roi_windows": [
                    {"roi_name": "fixture", "split": "train", "x": 0, "y": 0, "width": 4, "height": 4}
                ],
                "stage23_2a_authorized": False,
                "runs_new_ppo_update": False,
                "publishes_checkpoint": False,
                "replaces_default_policy": False,
                "connects_real_executor": False,
                "starts_online_canary": False,
                "canary_traffic_fraction": 0.0,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    summary = run_xunce_stage23_2a_high_resolution_terrain_data_prepare(
        config_path=config,
        output_root=tmp_path / "out",
        repo_root=REPO_ROOT,
    )

    assert summary["status"] == "passed"
    assert summary["next_required_change"] == "run_stage23_2b_slope_obstacle_aware_theta_reward_contract"
    assert summary["primary_dataset_id"] == "fixture_usgs_4m"
    assert summary["platform_contract_id"] == "agilex_scout_mini_piper"
    assert summary["platform_max_climb_deg"] == 30.0
    assert summary["max_traversable_slope_deg"] == 30.0
    assert summary["high_res_slice_count"] == 1
    assert summary["slope_blocked_cell_count_total"] > 0
    assert summary["stage23_1_smoke_executed"] is False
    assert (tmp_path / "out" / SUMMARY_FILE).is_file()
    sidecar = json.loads(
        (tmp_path / "out" / "high_res_roi_expansion" / "path_planner_sidecars" / "fixture_usgs_4m_fixture_train_000.path-planner-sidecar.json").read_text(
            encoding="utf-8"
        )
    )
    assert sidecar["slope_source"] == "provided_slope_map"
    assert sidecar["slope_blocked_cells"]
    assert sidecar["platform_contract_id"] == "agilex_scout_mini_piper"
    assert sidecar["platform_max_climb_deg"] == 30.0
    assert sidecar["max_traversable_slope_deg"] == 30.0
