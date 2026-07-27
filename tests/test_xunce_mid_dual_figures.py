"""Task 12 学术图：只从完整 aggregate 与逐行原始证据生成。"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from types import ModuleType
from typing import Any, Mapping
import xml.etree.ElementTree as ET

from PIL import Image
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
CONFIG = ROOT / "configs/xunce_mid_dual_figures_v1.json"
TASK10_TEST = ROOT / "tests/test_xunce_mid_dual_aggregate.py"
sys.path.insert(0, str(SCRIPTS))

import render_xunce_mid_dual_figures as figures  # noqa: E402
import xunce_artifact_io as artifact_io  # noqa: E402
from xunce_mid_dual_artifacts import MidDualRunStore  # noqa: E402


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def figure_config() -> dict[str, Any]:
    config, _ = figures._load_figure_config(CONFIG)  # noqa: SLF001
    return config


def _timing_components(elapsed_ms: float) -> dict[str, int]:
    total_ns = int(round(elapsed_ms * 1_000_000.0))
    base, remainder = divmod(total_ns, len(figures.TIMING_COMPONENT_FIELDS))
    values = [base] * len(figures.TIMING_COMPONENT_FIELDS)
    values[0] += remainder
    return dict(zip(figures.TIMING_COMPONENT_FIELDS, values, strict=True))


def _verified_input(*, coverage_passes: bool = True) -> figures.VerifiedFigureInput:
    g1_rows: list[dict[str, Any]] = []
    for split_index, split in enumerate(("test_q24", "unseen24")):
        for episode_index in range(24):
            denominator = 10_000
            if coverage_passes:
                final_count = 9_900 + split_index * 10 + episode_index % 10
            else:
                final_count = 7_900 + split_index * 10 + episode_index % 10
            g1_rows.append(
                {
                    "row_kind": "coverage_episode",
                    "split": split,
                    "episode_id": f"{split}-episode-{episode_index:02d}",
                    "episode_index": episode_index,
                    "scenario_id": f"{split}-scenario-{episode_index:02d}",
                    "lane_id": f"lane-{episode_index % 8}",
                    "denominator_cell_count": denominator,
                    "final_covered_cell_count": final_count,
                    "coverage": final_count / denominator,
                }
            )
    g1_splits = {
        split: figures._coverage_statistics(  # noqa: SLF001
            [
                row["coverage"]
                for row in g1_rows
                if row["split"] == split
            ]
        )
        for split in ("test_q24", "unseen24")
    }
    g1_midterm = all(
        payload["midterm_reduced_passed"] for payload in g1_splits.values()
    )
    g1_final = all(
        payload["final_threshold_reduced_passed"]
        for payload in g1_splits.values()
    )
    g1_gate = {
        "status": "passed" if g1_midterm else "failed",
        "sample_count": 48,
        "g1_coverage_80_passed": g1_midterm,
        "g1_coverage_99_passed": g1_final,
        "splits": g1_splits,
    }

    g2_rows: list[dict[str, Any]] = []
    for platform_index, platform_name in enumerate(figures.PLATFORMS):
        for scale_index, scale in enumerate(figures.SCALES):
            for class_index, request_class in enumerate(
                figures.REQUEST_CLASSES
            ):
                request_count = figures.EXPECTED_G2_CLASS_COUNTS[
                    (scale, request_class)
                ]
                for request_index in range(request_count):
                    request_id = (
                        f"{platform_name}-{scale}-{request_class}-"
                        f"{request_index:02d}"
                    )
                    request_sha256 = _sha(request_id)
                    for repeat_index in range(5):
                        elapsed_ms = (
                            120.0
                            + platform_index * 30.0
                            + scale_index * 60.0
                            + class_index * 10.0
                            + repeat_index
                            + request_index / 10.0
                        )
                        components = _timing_components(elapsed_ms)
                        g2_rows.append(
                            {
                                "row_kind": "planning_call",
                                "formal_sample": True,
                                "platform": platform_name,
                                "scale": scale,
                                "request_class": request_class,
                                "outcome_kind": (
                                    "unreachable"
                                    if request_class == "unreachable"
                                    else "reachable"
                                ),
                                "request_id": request_id,
                                "request_sha256": request_sha256,
                                "call_id": (
                                    f"{request_id}-repeat-{repeat_index}"
                                ),
                                "repeat_index": repeat_index,
                                **components,
                                "total_ns": sum(components.values()),
                                "elapsed_ms": elapsed_ms,
                            }
                        )
    timing_by_group: dict[str, dict[str, Any]] = {}
    timing_by_outcome: dict[str, dict[str, Any]] = {}
    timing_by_class: dict[str, dict[str, Any]] = {}
    for platform_name in figures.PLATFORMS:
        for scale in figures.SCALES:
            key = f"{platform_name}/{scale}"
            selected = [
                row
                for row in g2_rows
                if row["platform"] == platform_name
                and row["scale"] == scale
            ]
            timing_by_group[key] = figures._timing_statistics(  # noqa: SLF001
                [row["elapsed_ms"] for row in selected]
            )
            for outcome in ("reachable", "unreachable"):
                timing_by_outcome[f"{key}/{outcome}"] = (
                    figures._timing_statistics(  # noqa: SLF001
                        [
                            row["elapsed_ms"]
                            for row in selected
                            if row["outcome_kind"] == outcome
                        ]
                    )
                )
            for request_class in figures.REQUEST_CLASSES:
                timing_by_class[f"{key}/{request_class}"] = (
                    figures._timing_statistics(  # noqa: SLF001
                        [
                            row["elapsed_ms"]
                            for row in selected
                            if row["request_class"] == request_class
                        ]
                    )
                )
    g2_gate = {
        "status": "passed",
        "formal_call_count": 645,
        "g2_all_platforms_2s_passed": True,
        "g2_all_platforms_1s_passed": True,
        "timing_by_platform_scale": timing_by_group,
        "timing_by_platform_scale_outcome": timing_by_outcome,
        "timing_by_platform_scale_class": timing_by_class,
    }

    g3_rows: list[dict[str, Any]] = []
    wheel_coverages: list[float] = []
    paired_deltas: list[float] = []
    for episode_index in range(10):
        split = "test_q24" if episode_index < 5 else "unseen24"
        paired = 0.992 + (episode_index % 4) * 0.001
        wheel = paired + (-0.002 + (episode_index % 3) * 0.001)
        wheel_coverages.append(wheel)
        paired_deltas.append(wheel - paired)
        g3_rows.append(
            {
                "row_kind": "g3_wheel_step",
                "formal_sample": True,
                "is_terminal": True,
                "split": split,
                "episode_id": f"wheel-episode-{episode_index:02d}",
                "episode_index": episode_index if episode_index < 5 else episode_index - 5,
                "scenario_id": f"{split}-scenario-{episode_index % 5:02d}",
                "coverage": wheel,
                "paired_g1_coverage": paired,
            }
        )
    for platform_name in ("legged", "hopper"):
        for replay_index in range(3):
            semantic = _sha(f"{platform_name}-semantic-{replay_index}")
            g3_rows.append(
                {
                    "row_kind": "g3_interface_replay",
                    "formal_sample": True,
                    "platform": platform_name,
                    "replay_id": f"{platform_name}-replay-{replay_index}",
                    "request_id": f"{platform_name}-request-{replay_index}",
                    "formal_input_eligible": True,
                    "g2_semantic_digest": semantic,
                    "replay_semantic_digest": semantic,
                }
            )
    wheel_timing = figures._timing_statistics(  # noqa: SLF001
        [100.0 + index for index in range(10)]
    )
    interface_timing = figures._timing_statistics(  # noqa: SLF001
        [120.0 + index for index in range(6)]
    )
    g3_gate = {
        "status": "passed",
        "wheel_episode_count": 10,
        "interface_replay_count": 6,
        "g3_midterm_crosscheck_passed": True,
        "g3_final_crosscheck_passed": True,
        "wheel_coverage_mean": sum(wheel_coverages) / len(wheel_coverages),
        "paired_g1_coverage_delta_mean": sum(paired_deltas)
        / len(paired_deltas),
        "wheel_timing": wheel_timing,
        "interface_timing": interface_timing,
    }
    aggregate_midterm = (
        g1_gate["g1_coverage_80_passed"]
        and g2_gate["g2_all_platforms_2s_passed"]
        and g3_gate["g3_midterm_crosscheck_passed"]
    )
    aggregate_final = (
        g1_gate["g1_coverage_99_passed"]
        and g2_gate["g2_all_platforms_1s_passed"]
        and g3_gate["g3_final_crosscheck_passed"]
    )
    status = "passed" if aggregate_midterm else "failed"
    source_hashes = {gate_id: _sha(f"{gate_id}-manifest") for gate_id in figures.GATES}
    summary = {
        "schema_version": figures.AGGREGATE_SCHEMA_VERSION,
        "scale_profile": figures.SCALE_PROFILE,
        "status": status,
        "formal_evidence_eligible": True,
        "midterm_reduced_gate_passed": aggregate_midterm,
        "final_threshold_reduced_gate_passed": aggregate_final,
        "blockers": [],
        "gates": {"g1": g1_gate, "g2": g2_gate, "g3": g3_gate},
        "sample_counts": {
            "g1_total_episodes": 48,
            "g1_test_q24": 24,
            "g1_unseen24": 24,
            "g2_formal_calls": 645,
            "g3_wheel_episodes": 10,
            "g3_interface_replays": 6,
        },
        "source_manifest_sha256": source_hashes,
    }
    run_ids = {
        "aggregate": "aggregate-fixture",
        "g1": "g1-fixture",
        "g2": "g2-fixture",
        "g3": "g3-fixture",
    }
    return figures.VerifiedFigureInput(
        aggregate_run_id=run_ids["aggregate"],
        aggregate_manifest_sha256=_sha("aggregate-manifest"),
        status=status,
        run_ids=run_ids,
        source_manifest_sha256=source_hashes,
        input_hashes={
            "aggregate/manifest.json": _sha("aggregate-manifest"),
            "g1/results.jsonl": _sha("g1-results"),
            "g2/results.jsonl": _sha("g2-results"),
            "g3/results.jsonl": _sha("g3-results"),
        },
        aggregate_summary=summary,
        gates=summary["gates"],
        rows={
            "g1": tuple(g1_rows),
            "g2": tuple(g2_rows),
            "g3": tuple(g3_rows),
        },
    )


@pytest.fixture(scope="module")
def passed_input() -> figures.VerifiedFigureInput:
    return _verified_input(coverage_passes=True)


@pytest.fixture(scope="module")
def derived(
    passed_input: figures.VerifiedFigureInput,
    figure_config: Mapping[str, Any],
) -> dict[str, Any]:
    return figures._derive_figure_data(  # noqa: SLF001
        passed_input,
        figure_config,
    )


def _threshold_values(ax: Any, *, axis: str) -> set[float]:
    values: set[float] = set()
    for line in ax.lines:
        data = line.get_ydata() if axis == "y" else line.get_xdata()
        if len(data) >= 2 and all(float(item) == float(data[0]) for item in data):
            values.add(float(data[0]))
    return values


def _task10_helpers() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "task10_aggregate_fixture_helpers",
        TASK10_TEST,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _completed_aggregate_root(
    tmp_path: Path,
) -> tuple[Path, tuple[Path, Path, Path]]:
    helpers = _task10_helpers()
    roots = helpers._valid_roots(tmp_path / "sources")
    summary = helpers._aggregate(roots)
    assert summary["status"] == "passed"
    base = tmp_path / "aggregate"
    run_id = "aggregate-fixture"
    root = base / run_id
    effective_config = {
        "actual_sample_counts": {
            "g1_episodes_per_split": 24,
            "g1_split_count": 2,
            "g2_formal_calls": 645,
            "g3_interface_replays": 6,
            "g3_wheel_episodes": 10,
        },
        "output_root": str(base).replace("\\", "/"),
        "reference_sample_counts": {
            "g1_episodes_per_split": 64,
            "g1_split_count": 2,
            "g2_formal_calls": 1950,
            "g3_interface_replays": 18,
            "g3_wheel_episodes": 30,
        },
        "required_phase_ids": ["p01"],
        "scale_profile": figures.SCALE_PROFILE,
        "schema_version": figures.AGGREGATE_CONFIG_SCHEMA_VERSION,
        "source_contracts": helpers._FIXTURE_SOURCE_CONTRACTS,
        "run_id": run_id,
        "g1_root": str(roots[0]),
        "g2_root": str(roots[1]),
        "g3_root": str(roots[2]),
    }
    store = MidDualRunStore.create_new(root, effective_config)
    aggregate_rows = [
        {
            "row_kind": "independent_gate_recalculation",
            "gate_id": gate_id,
            "recalculated": summary["gates"][gate_id],
        }
        for gate_id in figures.GATES
    ]
    attempt = store.write_phase_attempt(
        "p01",
        aggregate_rows,
        {"kind": "independent_aggregate"},
    )
    store.accept_phase(
        "p01",
        attempt,
        store.phase_attempt_row_sha256("p01", attempt),
    )
    store.capture_lineage(
        [TASK10_TEST],
        "a" * 40,
        "b" * 40,
    )
    store.capture_environment(helpers._environment_probe)
    routing = {
        "schema_version": "xunce-mid-dual-aggregate-routing/v1",
        "scale_profile": figures.SCALE_PROFILE,
        "status": summary["status"],
        "midterm_reduced_gate_passed": summary[
            "midterm_reduced_gate_passed"
        ],
        "final_threshold_reduced_gate_passed": summary[
            "final_threshold_reduced_gate_passed"
        ],
        "blockers": [],
    }
    store.finalize(
        summary,
        routing,
        "fixture aggregate report\n",
        {
            "source_roots": {
                "g1_root": str(roots[0]),
                "g2_root": str(roots[1]),
                "g3_root": str(roots[2]),
                "source_manifest_sha256": summary[
                    "source_manifest_sha256"
                ],
            }
        },
    )
    assert MidDualRunStore.verify_manifest(root) is True
    return root, roots


def test_frozen_config_is_python_183mm_and_archival(
    figure_config: Mapping[str, Any],
) -> None:
    assert figure_config["backend"] == "python-matplotlib"
    assert figure_config["required_figure_families"] == list(
        figures.FIGURE_FAMILIES
    )
    assert figure_config["export_formats"] == ["svg", "pdf", "png", "tiff"]
    assert figure_config["raster_dpi"] == 600
    for width, _ in figure_config["figure_sizes_inches"].values():
        assert width * 25.4 == pytest.approx(183.0, abs=0.2)
    assert figure_config["rendering_behavior"] == {
        "passed": "render",
        "failed": "render",
        "blocked": "reject_without_output",
    }
    assert figure_config["claims"]["physical_capability_certified"] is False
    assert figure_config["claims"]["hardware_certified"] is False
    assert figure_config["canvas_contract"] == {
        "canonical_bbox": "fixed_canvas",
        "dimension_tolerance_mm": 0.2,
        "raster_dpi_tolerance": 1.0,
        "width_mm": 182.88,
    }
    assert figure_config["runtime_source_contract"] == {
        "schema_version": "xunce-mid-dual-figure-runtime-source-contract/v1",
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
    }
    assert figure_config["visible_content_scope"] == "result_only/v1"


def test_completed_aggregate_and_source_hashes_are_revalidated(
    tmp_path: Path,
) -> None:
    aggregate_root, roots = _completed_aggregate_root(tmp_path)
    verified = figures.load_verified_figure_input(
        aggregate_root,
        enforce_formal_paths=False,
    )
    assert verified.status == "passed"
    assert verified.aggregate_summary["sample_counts"] == {
        "g1_total_episodes": 48,
        "g1_test_q24": 24,
        "g1_unseen24": 24,
        "g2_formal_calls": 645,
        "g3_wheel_episodes": 10,
        "g3_interface_replays": 6,
    }
    assert set(verified.source_manifest_sha256) == {"g1", "g2", "g3"}

    g2_results = roots[1] / "results.jsonl"
    artifact_io.write_bytes(
        g2_results,
        artifact_io.read_bytes(g2_results) + b" ",
    )
    with pytest.raises(figures.FigureBlocked, match="g2_manifest_invalid"):
        figures.load_verified_figure_input(
            aggregate_root,
            enforce_formal_paths=False,
        )


def test_jsonl_and_nonfinite_evidence_fail_closed_without_row_dropping() -> None:
    with pytest.raises(figures.FigureBlocked):
        figures._parse_jsonl_rows(  # noqa: SLF001
            b'{"value": 1}\n\n{"value": 2}\n',
            "malformed",
        )
    with pytest.raises(figures.FigureBlocked):
        figures._parse_jsonl_rows(  # noqa: SLF001
            b'{"value": NaN}\n',
            "malformed",
        )
    with pytest.raises(figures.FigureBlocked):
        figures._parse_json_object(  # noqa: SLF001
            b'{"value": Infinity}',
            "malformed",
        )


def test_exact_denominators_partitions_and_deterministic_bootstrap(
    passed_input: figures.VerifiedFigureInput,
    derived: Mapping[str, Any],
    figure_config: Mapping[str, Any],
) -> None:
    assert {
        split: len(derived["g1"]["splits"][split]["rows"])
        for split in ("test_q24", "unseen24")
    } == {"test_q24": 24, "unseen24": 24}
    assert sum(
        len(payload["rows"])
        for payload in derived["g2"]["groups"].values()
    ) == 645
    assert {
        key: len(payload["rows"])
        for key, payload in derived["g2"]["groups"].items()
    } == {
        "wheel/standard": 165,
        "wheel/kilometer": 50,
        "legged/standard": 165,
        "legged/kilometer": 50,
        "hopper/standard": 165,
        "hopper/kilometer": 50,
    }
    assert len(derived["g3"]["wheel_terminal"]) == 10
    assert derived["g3"]["platform_counts"] == {"legged": 3, "hopper": 3}

    values = [0.90, 0.95, 1.00]
    first = figures.deterministic_bootstrap_mean_ci(
        values,
        seed=20260727,
        resamples=2000,
        confidence=0.95,
        label="fixture",
    )
    second = figures.deterministic_bootstrap_mean_ci(
        values,
        seed=20260727,
        resamples=2000,
        confidence=0.95,
        label="fixture",
    )
    assert first == second
    assert first["ci_low"] <= first["mean"] <= first["ci_high"]

    truncated_rows = {
        **passed_input.rows,
        "g2": passed_input.rows["g2"][:-1],
    }
    truncated = replace(passed_input, rows=truncated_rows)
    with pytest.raises(
        figures.FigureBlocked,
        match="g2_exact_denominator_invalid",
    ):
        figures._derive_figure_data(  # noqa: SLF001
            truncated,
            figure_config,
        )


def test_required_panels_thresholds_labels_and_all_raw_rows(
    passed_input: figures.VerifiedFigureInput,
    derived: Mapping[str, Any],
    figure_config: Mapping[str, Any],
) -> None:
    fig01 = figures.build_fig01_dual_gate_overview(
        passed_input,
        derived,
        figure_config,
    )
    fig02 = figures.build_fig02_g1_coverage(
        passed_input,
        derived,
        figure_config,
    )
    fig03 = figures.build_fig03_g2_timing(
        passed_input,
        derived,
        figure_config,
    )
    fig04 = figures.build_fig04_g3_crosscheck(
        passed_input,
        derived,
        figure_config,
    )
    try:
        for fig in (fig01, fig02, fig03, fig04):
            footer = "\n".join(text.get_text() for text in fig.texts)
            assert passed_input.aggregate_manifest_sha256 not in footer
            assert "status=passed" in footer
            assert "midterm reduced scale" in footer
            assert "independent raw-row recomputation" in footer.lower()
            assert not any(
                forbidden in footer.lower()
                for forbidden in (
                    "prestart",
                    "python",
                    "pydantic",
                    "rasterio",
                    "dll",
                    "cuda",
                    "driver",
                    "worker",
                )
            )

        assert len(fig01.axes) == 3
        assert {0.80, 0.99}.issubset(
            _threshold_values(fig01.axes[0], axis="x")
        )
        assert {1000.0, 2000.0}.issubset(
            _threshold_values(fig01.axes[1], axis="x")
        )

        assert len(fig02.axes) == 2
        for ax in fig02.axes:
            assert {0.80, 0.99}.issubset(
                _threshold_values(ax, axis="y")
            )
            assert ax.get_ylabel() in {"", "Coverage fraction"}
            assert "nearest-rank P50=" in ax.get_title()

        assert len(fig03.axes) == 6
        raw_count = 0
        for ax in fig03.axes:
            assert {1000.0, 2000.0}.issubset(
                _threshold_values(ax, axis="y")
            )
            raw_count += sum(
                len(collection.get_offsets())
                for collection in ax.collections[:3]
            )
            assert ax.get_ylabel() == "Formal planning time (ms)"
        assert raw_count == 645

        assert len(fig04.axes) == 3
        assert {0.80, 0.99}.issubset(
            _threshold_values(fig04.axes[0], axis="y")
        )
        assert -0.01 in _threshold_values(fig04.axes[1], axis="y")
        assert len(fig04.axes[2].patches) == 2
        proxy_label = (
            f"{fig04.axes[2].get_title()} "
            f"{fig04.axes[2].get_xlabel()}"
        ).lower().replace("-", " ")
        assert "simulation proxy" in proxy_label
    finally:
        figures.plt.close(fig01)
        figures.plt.close(fig02)
        figures.plt.close(fig03)
        figures.plt.close(fig04)


def test_real_export_formats_and_600_dpi_rasters(
    figure_config: Mapping[str, Any],
) -> None:
    config_sha256 = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    figures._apply_publication_style(  # noqa: SLF001
        figure_config,
        config_sha256,
    )
    fig, ax = figures.plt.subplots(figsize=(0.8, 0.6))
    ax.plot([0, 1], [0, 1], marker="o")
    ax.set_xlabel("Editable label")
    metadata = {
        "figure_family": "fixture",
        "status": "passed",
        "aggregate_manifest_sha256": _sha("aggregate"),
    }
    try:
        payloads: dict[str, bytes] = {}
        for file_format in figures.EXPORT_FORMATS:
            first = figures._export_figure_bytes(  # noqa: SLF001
                fig,
                family="fixture",
                file_format=file_format,
                dpi=600,
                metadata=metadata,
            )
            second = figures._export_figure_bytes(  # noqa: SLF001
                fig,
                family="fixture",
                file_format=file_format,
                dpi=600,
                metadata=metadata,
            )
            assert first == second
            payloads[file_format] = first
    finally:
        figures.plt.close(fig)
    assert payloads["pdf"].startswith(b"%PDF")
    assert b"<svg" in payloads["svg"]
    assert b"<text" in payloads["svg"]
    for file_format in ("png", "tiff"):
        with Image.open(io.BytesIO(payloads[file_format])) as image:
            dpi = image.info.get("dpi")
            assert dpi is not None
            assert dpi[0] == pytest.approx(600.0, abs=1.0)
            assert dpi[1] == pytest.approx(600.0, abs=1.0)
    assert figures.mpl.rcParams["svg.fonttype"] == "none"
    assert figures.mpl.rcParams["pdf.fonttype"] == 42


def _fake_export(
    fig: Any,
    *,
    family: str,
    file_format: str,
    dpi: int,
    metadata: Mapping[str, Any],
) -> bytes:
    del fig
    return json.dumps(
        {
            "family": family,
            "format": file_format,
            "dpi": dpi,
            "metadata": metadata,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _fake_export_measurement(
    payload: bytes,
    *,
    file_format: str,
    expected_size_inches: tuple[float, float],
    expected_dpi: int,
    **_: Any,
) -> dict[str, Any]:
    del payload
    width_inches, height_inches = expected_size_inches
    return {
        "format": file_format,
        "width_mm": width_inches * 25.4,
        "height_mm": height_inches * 25.4,
        "pixel_width": (
            round(width_inches * expected_dpi)
            if file_format in {"png", "tiff"}
            else None
        ),
        "pixel_height": (
            round(height_inches * expected_dpi)
            if file_format in {"png", "tiff"}
            else None
        ),
        "dpi_x": (
            float(expected_dpi)
            if file_format in {"png", "tiff"}
            else None
        ),
        "dpi_y": (
            float(expected_dpi)
            if file_format in {"png", "tiff"}
            else None
        ),
        "editable_text": file_format in {"svg", "pdf"},
    }


def _runtime_code_lineage() -> figures.CodeLineageSnapshot:
    return figures._capture_runtime_code_lineage(  # noqa: SLF001
        CONFIG,
        figures._load_aggregate_module(),  # noqa: SLF001
    )


def _length_to_mm(value: str) -> float:
    match = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*(pt|in|mm|cm)\s*", value)
    assert match is not None
    magnitude = float(match.group(1))
    return magnitude * {
        "pt": 25.4 / 72.0,
        "in": 25.4,
        "mm": 1.0,
        "cm": 10.0,
    }[match.group(2)]


def _measure_export_independently(
    payload: bytes,
    file_format: str,
) -> dict[str, Any]:
    if file_format == "svg":
        root = ET.fromstring(payload)
        return {
            "width_mm": _length_to_mm(root.attrib["width"]),
            "height_mm": _length_to_mm(root.attrib["height"]),
            "editable_text": b"<text" in payload,
        }
    if file_format == "pdf":
        match = re.search(
            rb"/MediaBox\s*\[\s*([-+0-9.]+)\s+([-+0-9.]+)\s+"
            rb"([-+0-9.]+)\s+([-+0-9.]+)\s*\]",
            payload,
        )
        assert match is not None
        x0, y0, x1, y1 = (float(item) for item in match.groups())
        return {
            "width_mm": (x1 - x0) * 25.4 / 72.0,
            "height_mm": (y1 - y0) * 25.4 / 72.0,
            "editable_text": (
                b"/FontFile2" in payload or b"/CIDFontType2" in payload
            ),
        }
    with Image.open(io.BytesIO(payload)) as image:
        dpi = image.info.get("dpi")
        assert dpi is not None
        return {
            "width_mm": image.width / float(dpi[0]) * 25.4,
            "height_mm": image.height / float(dpi[1]) * 25.4,
            "pixel_width": image.width,
            "pixel_height": image.height,
            "dpi_x": float(dpi[0]),
            "dpi_y": float(dpi[1]),
        }


def test_manifest_source_hashes_filenames_and_reproducibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    passed_input: figures.VerifiedFigureInput,
    figure_config: Mapping[str, Any],
) -> None:
    monkeypatch.setattr(figures, "_export_figure_bytes", _fake_export)
    monkeypatch.setattr(
        figures,
        "_inspect_export_bytes",
        _fake_export_measurement,
        raising=False,
    )
    config_sha256 = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    lineage = _runtime_code_lineage()
    figure_run_id = "figure-determinism-fixture"
    first = figures.render_figure_bundle(
        passed_input,
        figure_config,
        config_sha256,
        tmp_path / "first" / figure_run_id,
        figure_run_id=figure_run_id,
        code_lineage=lineage,
    )
    second = figures.render_figure_bundle(
        passed_input,
        figure_config,
        config_sha256,
        tmp_path / "second" / figure_run_id,
        figure_run_id=figure_run_id,
        code_lineage=lineage,
    )
    first_manifest = artifact_io.read_bytes(first / "figure_manifest.json")
    second_manifest = artifact_io.read_bytes(second / "figure_manifest.json")
    assert first_manifest == second_manifest
    manifest = json.loads(first_manifest)
    assert manifest["status"] == "passed"
    assert manifest["aggregate_manifest_sha256"] == (
        passed_input.aggregate_manifest_sha256
    )
    assert manifest["figure_run_id"] == figure_run_id
    assert manifest["code_lineage_sha256"] == manifest["file_hashes"][
        "code_lineage.json"
    ]
    assert manifest["manifest_payload_sha256"] == (
        figures._manifest_payload_sha256(manifest)  # noqa: SLF001
    )
    assert manifest["bootstrap"]["ci_low_rank"] == 250
    assert manifest["bootstrap"]["ci_high_rank"] == 9750
    assert manifest["bootstrap"]["seed"] == 20260727
    assert len(manifest["figures"]) == 4
    expected_names = {
        "code_lineage.json",
        "figure_manifest.json",
        "figure_manifest.sha256",
    }
    for family in figures.FIGURE_FAMILIES:
        expected_names.update(
            {
                f"{family}.svg",
                f"{family}.pdf",
                f"{family}.png",
                f"{family}.tiff",
                f"{family}_source_data.json",
                f"{family}_source_data.csv",
            }
        )
    assert {
        path.name
        for path in first.iterdir()
        if path.is_file()
    } == expected_names
    for relative, digest in manifest["file_hashes"].items():
        payload = artifact_io.read_bytes(first / relative)
        assert hashlib.sha256(payload).hexdigest() == digest
        assert payload == artifact_io.read_bytes(second / relative)
    for family in figures.FIGURE_FAMILIES:
        source = json.loads(
            artifact_io.read_text(first / f"{family}_source_data.json")
        )
        assert source["metadata"]["status"] in {"passed", "failed"}
        assert source["metadata"]["reduced_qualifier"] == (
            figures.REDUCED_QUALIFIER
        )
        assert source["metadata"]["aggregate_manifest_sha256"] == (
            passed_input.aggregate_manifest_sha256
        )
        assert source["metadata"]["figure_run_id"] == figure_run_id
        assert source["metadata"]["code_lineage_sha256"] == manifest[
            "code_lineage_sha256"
        ]
        assert source["metadata"]["bootstrap"]["ci_low_rank"] == 250
        assert source["metadata"]["bootstrap"]["ci_high_rank"] == 9750
        csv_header = artifact_io.read_text(
            first / f"{family}_source_data.csv"
        ).splitlines()[0]
        assert "aggregate_manifest_sha256" in csv_header
        assert "figure_status" in csv_header
        assert "figure_run_id" in csv_header
        assert "statistical_method" in csv_header
    for figure_row in manifest["figures"]:
        for exported in figure_row["files"]:
            assert exported["canvas"]["width_mm"] == pytest.approx(
                182.88,
                abs=0.2,
            )


def test_blocked_rejects_without_output_and_failed_remains_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    figure_config: Mapping[str, Any],
) -> None:
    monkeypatch.setattr(figures, "_export_figure_bytes", _fake_export)
    monkeypatch.setattr(
        figures,
        "_inspect_export_bytes",
        _fake_export_measurement,
        raising=False,
    )
    lineage = _runtime_code_lineage()
    passed = _verified_input(coverage_passes=True)
    blocked = replace(passed, status="blocked")
    blocked_root = tmp_path / "blocked"
    with pytest.raises(
        figures.FigureBlocked,
        match="blocked_evidence_is_not_renderable",
    ):
        figures.render_figure_bundle(
            blocked,
            figure_config,
            _sha("config"),
            blocked_root,
            figure_run_id="blocked",
            code_lineage=lineage,
        )
    assert not blocked_root.exists()

    failed = _verified_input(coverage_passes=False)
    failed_root = figures.render_figure_bundle(
        failed,
        figure_config,
        _sha("config"),
        tmp_path / "failed" / "failed",
        figure_run_id="failed",
        code_lineage=lineage,
    )
    manifest = artifact_io.read_json(failed_root / "figure_manifest.json")
    assert manifest["status"] == "failed"
    statuses = {
        row["figure_family"]: row["status"] for row in manifest["figures"]
    }
    assert statuses["fig01_dual_gate_overview"] == "failed"
    assert statuses["fig02_g1_coverage"] == "failed"
    assert statuses["fig03_g2_timing"] == "passed"
    assert statuses["fig04_g3_crosscheck"] == "passed"


def test_pipeline_creates_no_output_before_input_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "must-not-exist"

    def reject_input(
        aggregate_root: str | Path,
        *,
        enforce_formal_paths: bool,
        aggregate_module: Any,
    ) -> figures.VerifiedFigureInput:
        del aggregate_root, enforce_formal_paths, aggregate_module
        raise figures.FigureBlocked("fixture_input_invalid")

    monkeypatch.setattr(
        figures,
        "load_verified_figure_input",
        reject_input,
    )
    with pytest.raises(figures.FigureBlocked, match="fixture_input_invalid"):
        figures.run_figure_pipeline(
            config_path=CONFIG,
            aggregate_root=tmp_path / "incomplete-aggregate",
            run_id="figure-fixture",
            output_root=output_root,
            enforce_formal_paths=False,
        )
    assert not output_root.exists()


def test_exact_bootstrap_ranks_and_nearest_rank_p50_are_unambiguous(
    passed_input: figures.VerifiedFigureInput,
    derived: Mapping[str, Any],
    figure_config: Mapping[str, Any],
) -> None:
    assert figures._bootstrap_endpoint_ranks(  # noqa: SLF001
        resamples=10_000,
        confidence=0.95,
    ) == (250, 9750)
    ordered = [float(value) for value in range(1, 25)]
    assert figures._nearest_rank(ordered, 0.50) == 12.0  # noqa: SLF001
    assert statistics.median(ordered) == 12.5

    interval = figures.deterministic_bootstrap_mean_ci(
        [1.0],
        seed=20260727,
        resamples=10_000,
        confidence=0.95,
        label="rank-contract",
    )
    assert interval["ci_low_rank"] == 250
    assert interval["ci_high_rank"] == 9750

    methods = figures._family_statistical_methods(  # noqa: SLF001
        "fig02_g1_coverage"
    )
    assert any("nearest-rank P50" in method for method in methods)
    assert all("median" not in method.lower() for method in methods)
    source_rows = figures._g1_source_rows(derived)  # noqa: SLF001
    assert source_rows
    assert "split_nearest_rank_p50" in source_rows[0]
    assert "split_median" not in source_rows[0]

    figure = figures.build_fig02_g1_coverage(
        passed_input,
        derived,
        figure_config,
    )
    try:
        assert all(
            "nearest-rank P50=" in axis.get_title()
            for axis in figure.axes
        )
    finally:
        figures.plt.close(figure)


def test_runtime_source_closure_detects_renderer_validator_and_helper_drift(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "lineage-repository"
    scripts = repository / "scripts"
    artifact_io.make_dirs(scripts)
    renderer = scripts / "renderer.py"
    aggregate = scripts / "aggregate.py"
    helper = scripts / "helper.py"
    config = repository / "config.json"
    artifact_io.write_bytes(renderer, b"import aggregate\nVALUE = aggregate.VALUE\n")
    artifact_io.write_bytes(aggregate, b"import helper\nVALUE = helper.VALUE\n")
    artifact_io.write_bytes(helper, b"VALUE = 1\n")
    artifact_io.write_bytes(config, b"{}\n")
    subprocess.run(
        ["git", "init"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "add", "scripts/aggregate.py", "scripts/helper.py", "config.json"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Task12",
            "-c",
            "user.email=task12@example.invalid",
            "commit",
            "-m",
            "lineage fixture",
        ],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    artifact_io.write_bytes(helper, b"VALUE = 1\n# dirty helper\n")

    snapshot = figures._capture_code_lineage(  # noqa: SLF001
        repository_root=repository,
        entry_paths=(renderer, config),
        required_relative_paths={
            "scripts/renderer.py",
            "scripts/aggregate.py",
            "scripts/helper.py",
            "config.json",
        },
    )
    assert set(snapshot.source_bytes) == {
        "config.json",
        "scripts/aggregate.py",
        "scripts/helper.py",
        "scripts/renderer.py",
    }
    snapshotted = {
        row["path"]
        for row in snapshot.payload["files"]
        if row["snapshot_path"] is not None
    }
    assert snapshotted == {
        "scripts/helper.py",
        "scripts/renderer.py",
    }
    figures._verify_code_lineage(snapshot)  # noqa: SLF001

    for path in (renderer, aggregate, helper):
        original = artifact_io.read_bytes(path)
        artifact_io.write_bytes(path, original + b"# drift\n")
        with pytest.raises(
            figures.FigureBlocked,
            match=f"code_lineage_drift:{re.escape(path.relative_to(repository).as_posix())}",
        ):
            figures._verify_code_lineage(snapshot)  # noqa: SLF001
        artifact_io.write_bytes(path, original)


@pytest.mark.parametrize(
    "mutation_kind",
    ("config", "summary", "input_audit", "lineage"),
)
def test_verified_loader_uses_only_captured_source_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation_kind: str,
) -> None:
    aggregate_root, roots = _completed_aggregate_root(tmp_path)
    source_config = artifact_io.read_json(roots[0] / "config.json")
    if mutation_kind == "config":
        mutation_path = roots[0] / "config.json"
    elif mutation_kind == "summary":
        mutation_path = roots[0] / "summary.json"
    elif mutation_kind == "input_audit":
        binding = source_config.get("evidence_binding")
        assert isinstance(binding, Mapping)
        mutation_path = roots[0] / str(binding["input_audit_path"])
    else:
        binding = source_config.get("evidence_binding")
        assert isinstance(binding, Mapping)
        mutation_path = roots[0] / str(binding["lineage_audit_path"])
    original_snapshot = figures._snapshot_completed_root  # noqa: SLF001
    mutated = False

    def snapshot_then_mutate(
        root: Path,
        label: str,
        required_paths: set[str],
    ) -> tuple[dict[str, Any], dict[str, bytes], str, dict[str, str]]:
        nonlocal mutated
        result = original_snapshot(root, label, required_paths)
        if label == "g1" and not mutated:
            mutated = True
            artifact_io.write_bytes(mutation_path, b"{}\n")
        return result

    monkeypatch.setattr(
        figures,
        "_snapshot_completed_root",
        snapshot_then_mutate,
    )
    verified = figures.load_verified_figure_input(
        aggregate_root,
        enforce_formal_paths=False,
    )
    assert mutated is True
    assert verified.status == "passed"


def test_transactional_write_failure_leaves_final_root_retryable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    passed_input: figures.VerifiedFigureInput,
    figure_config: Mapping[str, Any],
) -> None:
    lineage = _runtime_code_lineage()
    run_id = "figure-write-retry"
    target = tmp_path / "publication" / run_id
    monkeypatch.setattr(figures, "_export_figure_bytes", _fake_export)
    monkeypatch.setattr(
        figures,
        "_inspect_export_bytes",
        _fake_export_measurement,
    )
    original_write = artifact_io.write_bytes

    def fail_family_source(path: str | Path, payload: bytes) -> None:
        if Path(path).name == "fig02_g1_coverage_source_data.csv":
            raise OSError("injected source-data write failure")
        original_write(path, payload)

    monkeypatch.setattr(artifact_io, "write_bytes", fail_family_source)
    with pytest.raises(OSError, match="injected source-data write"):
        figures.render_figure_bundle(
            passed_input,
            figure_config,
            hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
            target,
            figure_run_id=run_id,
            code_lineage=lineage,
        )
    assert not target.exists()

    monkeypatch.setattr(artifact_io, "write_bytes", original_write)
    assert figures.render_figure_bundle(
        passed_input,
        figure_config,
        hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        target,
        figure_run_id=run_id,
        code_lineage=lineage,
    ) == target


def test_transactional_publish_run_identity_and_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    passed_input: figures.VerifiedFigureInput,
    figure_config: Mapping[str, Any],
) -> None:
    lineage = _runtime_code_lineage()
    run_id = "figure-transaction-fixture"
    target = tmp_path / "publication" / run_id
    export_count = 0

    def fail_second_family(
        fig: Any,
        *,
        family: str,
        file_format: str,
        dpi: int,
        metadata: Mapping[str, Any],
    ) -> bytes:
        nonlocal export_count
        export_count += 1
        if family == figures.FIGURE_FAMILIES[1]:
            raise OSError("injected second-family export failure")
        return _fake_export(
            fig,
            family=family,
            file_format=file_format,
            dpi=dpi,
            metadata=metadata,
        )

    monkeypatch.setattr(figures, "_export_figure_bytes", fail_second_family)
    monkeypatch.setattr(
        figures,
        "_inspect_export_bytes",
        _fake_export_measurement,
        raising=False,
    )
    with pytest.raises(OSError, match="injected second-family"):
        figures.render_figure_bundle(
            passed_input,
            figure_config,
            hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
            target,
            figure_run_id=run_id,
            code_lineage=lineage,
        )
    assert export_count > len(figures.EXPORT_FORMATS)
    assert not target.exists()

    monkeypatch.setattr(figures, "_export_figure_bytes", _fake_export)
    published = figures.render_figure_bundle(
        passed_input,
        figure_config,
        hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        target,
        figure_run_id=run_id,
        code_lineage=lineage,
    )
    assert published == target
    manifest = artifact_io.read_json(target / "figure_manifest.json")
    assert manifest["figure_run_id"] == run_id
    assert manifest["output_identity"] == {
        "figure_run_id": run_id,
        "relative_root": run_id,
    }
    assert manifest["manifest_payload_sha256"] == (
        figures._manifest_payload_sha256(manifest)  # noqa: SLF001
    )
    detached = artifact_io.read_text(
        target / "figure_manifest.sha256"
    ).strip()
    assert detached == hashlib.sha256(
        artifact_io.read_bytes(target / "figure_manifest.json")
    ).hexdigest()
    figures.verify_figure_bundle(target)
    for family in figures.FIGURE_FAMILIES:
        source = artifact_io.read_json(
            target / f"{family}_source_data.json"
        )
        assert source["metadata"]["figure_run_id"] == run_id
        header = artifact_io.read_text(
            target / f"{family}_source_data.csv"
        ).splitlines()[0]
        assert "figure_run_id" in header

    renamed = target.with_name("renamed-bundle")
    os.replace(target, renamed)
    with pytest.raises(
        figures.FigureBlocked,
        match="figure_output_identity_mismatch",
    ):
        figures.verify_figure_bundle(renamed)


def test_all_real_builders_preserve_fixed_canvas_across_formats(
    passed_input: figures.VerifiedFigureInput,
    derived: Mapping[str, Any],
    figure_config: Mapping[str, Any],
) -> None:
    config_sha256 = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    figures._apply_publication_style(figure_config, config_sha256)  # noqa: SLF001
    for family in figures.FIGURE_FAMILIES:
        figure = figures.FIGURE_BUILDERS[family](
            passed_input,
            derived,
            figure_config,
        )
        try:
            expected_width, expected_height = figure_config[
                "figure_sizes_inches"
            ][family]
            metadata = {
                "figure_family": family,
                "status": "passed",
                "aggregate_manifest_sha256": _sha("aggregate"),
            }
            for file_format in figures.EXPORT_FORMATS:
                payload = figures._export_figure_bytes(  # noqa: SLF001
                    figure,
                    family=family,
                    file_format=file_format,
                    dpi=600,
                    metadata=metadata,
                )
                measured = _measure_export_independently(
                    payload,
                    file_format,
                )
                assert measured["width_mm"] == pytest.approx(
                    expected_width * 25.4,
                    abs=0.2,
                )
                assert measured["height_mm"] == pytest.approx(
                    expected_height * 25.4,
                    abs=0.2,
                )
                if file_format in {"png", "tiff"}:
                    assert measured["pixel_width"] == round(
                        expected_width * 600
                    )
                    assert measured["pixel_height"] == round(
                        expected_height * 600
                    )
                    assert measured["dpi_x"] == pytest.approx(600, abs=1)
                    assert measured["dpi_y"] == pytest.approx(600, abs=1)
                else:
                    assert measured["editable_text"] is True
        finally:
            figures.plt.close(figure)
