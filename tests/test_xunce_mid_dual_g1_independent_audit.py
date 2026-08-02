from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    REPO_ROOT / "scripts" / "run_xunce_mid_dual_g1_independent_audit.py"
)
REPAIR_INDICES = (3, 10, 23)


def _load_module():
    assert MODULE_PATH.is_file(), "independent G1 audit entrypoint not implemented"
    spec = importlib.util.spec_from_file_location(
        "run_xunce_mid_dual_g1_independent_audit",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_line(value: object) -> bytes:
    return _canonical_bytes(value) + b"\n"


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _parent_episode(index: int) -> dict[str, object]:
    safety_count = 1 if index == 5 else 0
    repair_parent_finals = {3: 2, 10: 3, 23: 1}
    final_count = repair_parent_finals.get(index, 98)
    return {
        "checkpoint_sha256": "1" * 64,
        "code_sha256": "2" * 64,
        "config_sha256": "3" * 64,
        "coverage": final_count / 100,
        "denominator_cell_count": 100,
        "denominator_sha256": f"{index + 100:064x}",
        "episode_id": f"unseen24-episode-{index:02d}",
        "episode_index": index,
        "final_covered_cell_count": final_count,
        "lane_id": f"lane-{index % 8}",
        "policy_state_sha256": "4" * 64,
        "row_kind": "coverage_episode",
        "safety_violation_count": safety_count,
        "scenario_id": f"unseen/scenario-{index:04d}/standard-proxy/v1",
        "schema_version": "xunce-mid-dual-g1-coverage-episode/v1",
        "termination_reason": "success_done",
    }


def _repair_episode(index: int) -> dict[str, object]:
    final_by_index = {3: 80, 10: 90, 23: 100}
    safety_count = 2 if index == 10 else 0
    final_count = final_by_index[index]
    return {
        "checkpoint_sha256": "1" * 64,
        "coverage": final_count / 100,
        "denominator_cell_count": 100,
        "denominator_sha256": f"{index + 100:064x}",
        "episode_id": f"unseen24-episode-{index:02d}",
        "episode_index": index,
        "final_covered_cell_count": final_count,
        "lane_id": f"lane-{index % 8}",
        "mixed_code_incremental_repair": True,
        "policy_state_sha256": "4" * 64,
        "repair_code_sha256": "5" * 64,
        "row_kind": "repair_coverage_episode",
        "safety_violation_count": safety_count,
        "scenario_id": f"unseen/scenario-{index:04d}/standard-proxy/v1",
        "schema_version": "xunce-mid-dual-g1-repair-episode/v1",
        "termination_reason": "success_done",
    }


def _test_episode(index: int) -> dict[str, object]:
    final_count = 10 if index == 20 else 95
    scenario_id = (
        "test/scenario-0094/standard-proxy/v1"
        if index == 20
        else f"test/scenario-{index:04d}/standard-proxy/v1"
    )
    return {
        "checkpoint_sha256": "1" * 64,
        "code_sha256": "2" * 64,
        "config_sha256": "3" * 64,
        "coverage": final_count / 100,
        "denominator_cell_count": 100,
        "denominator_sha256": f"{index + 200:064x}",
        "episode_id": f"test_q24-episode-{index:02d}",
        "episode_index": index,
        "final_covered_cell_count": final_count,
        "lane_id": f"lane-{index % 8}",
        "phase_id": "p03",
        "phase_name": "test_q24",
        "policy_state_sha256": "4" * 64,
        "row_kind": "coverage_episode",
        "safety_violation_count": 0,
        "scenario_id": scenario_id,
        "schema_version": "xunce-mid-dual-g1-coverage-episode/v1",
        "split": "test_q24",
        "termination_reason": "success_done",
    }


def _write_fixture(root: Path) -> dict[str, Path]:
    root.mkdir()
    parent_path = root / "parent-results.jsonl"
    repair_path = root / "repair-results.jsonl"
    merged_path = root / "merged-results.jsonl"
    manifest_path = root / "manifest.json"
    test_path = root / "test-results.jsonl"
    phase_attempts_path = root / "phase-attempts.jsonl"

    parent_rows: list[dict[str, object]] = []
    parent_sources: dict[int, tuple[int, bytes, dict[str, object]]] = {}
    for index in range(24):
        episode = _parent_episode(index)
        raw_line = _canonical_line(episode)
        parent_sources[index] = (len(parent_rows) + 1, raw_line, episode)
        parent_rows.append(episode)
        parent_rows.append(
            {
                "episode_index": index,
                "row_kind": "decision",
                "schema_version": "fixture-decision/v1",
            }
        )
    parent_payload = b"".join(_canonical_line(row) for row in parent_rows)
    parent_path.write_bytes(parent_payload)

    repair_rows: list[dict[str, object]] = []
    repair_sources: dict[int, tuple[int, bytes, dict[str, object]]] = {}
    for index in REPAIR_INDICES:
        episode = _repair_episode(index)
        raw_line = _canonical_line(episode)
        repair_sources[index] = (len(repair_rows) + 1, raw_line, episode)
        repair_rows.append(episode)
        repair_rows.append(
            {
                "episode_index": index,
                "row_kind": "repair_decision",
                "schema_version": "fixture-repair-decision/v1",
            }
        )
    repair_payload = b"".join(_canonical_line(row) for row in repair_rows)
    repair_path.write_bytes(repair_payload)

    test_rows: list[dict[str, object]] = []
    for index in range(24):
        test_rows.append(_test_episode(index))
        test_rows.append(
            {
                "episode_index": index,
                "row_kind": "decision",
                "schema_version": "fixture-test-decision/v1",
            }
        )
    test_payload = b"".join(_canonical_line(row) for row in test_rows)
    test_path.write_bytes(test_payload)

    parent_sha = _sha256(parent_payload)
    repair_sha = _sha256(repair_payload)
    test_sha = _sha256(test_payload)
    phase_attempts_path.write_bytes(
        _canonical_line(
            {
                "attempt_id": "a01",
                "phase_id": "p03",
                "row_sha256": test_sha,
                "rows_path": "phases/p03/a01/results.jsonl",
                "status": "accepted",
            }
        )
        + _canonical_line(
            {
                "attempt_id": "a01",
                "phase_id": "p04",
                "row_sha256": parent_sha,
                "rows_path": "phases/p04/a01/results.jsonl",
                "status": "accepted",
            }
        )
    )
    merged_rows: list[dict[str, object]] = []
    for index in range(24):
        parent_line_number, parent_line, parent_episode = parent_sources[index]
        if index in REPAIR_INDICES:
            _, source_line, episode = repair_sources[index]
            origin = "repair_rerun"
            lineage = {
                "parent_replaced_line_bytes_sha256": _sha256(parent_line),
                "parent_replaced_line_number": parent_line_number,
                "repair_code_sha256": "5" * 64,
                "repair_result_line_sha256": _sha256(source_line),
                "repair_results_sha256": repair_sha,
            }
        else:
            source_line = parent_line
            episode = parent_episode
            origin = "parent_read_only"
            lineage = {
                "parent_code_sha256": "2" * 64,
                "parent_line_bytes_sha256": _sha256(parent_line),
                "parent_line_number": parent_line_number,
                "parent_line_size_bytes": len(parent_line),
                "parent_results_sha256": parent_sha,
            }
        merged_rows.append(
            {
                "episode": episode,
                "episode_id": episode["episode_id"],
                "episode_index": index,
                "homogeneous_code_execution": False,
                "lane_id": episode["lane_id"],
                "lineage": lineage,
                "mixed_code_incremental_repair": True,
                "result_origin": origin,
                "result_sha256": _sha256(_canonical_bytes(episode)),
                "row_kind": "mixed_repair_episode",
                "scenario_id": episode["scenario_id"],
                "schema_version": (
                    "xunce-mid-dual-g1-mixed-repair-row/v1"
                ),
                "source_line_sha256": _sha256(source_line),
            }
        )
    merged_payload = b"".join(_canonical_line(row) for row in merged_rows)
    merged_path.write_bytes(merged_payload)
    manifest_path.write_text(
        json.dumps(
            {
                "accepted_episode_indices": list(REPAIR_INDICES),
                "coverage_80_count": 24,
                "coverage_99_count": 1,
                "homogeneous_code_execution": False,
                "mean": 0.97,
                "merged_results_sha256": _sha256(merged_payload),
                "mixed_code_incremental_repair": True,
                "parent_results_sha256": parent_sha,
                "repair_results_sha256": repair_sha,
                "schema_version": "xunce-mid-dual-g1-repair-manifest/v1",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "manifest": manifest_path,
        "merged": merged_path,
        "parent": parent_path,
        "phase_attempts": phase_attempts_path,
        "repair": repair_path,
        "test": test_path,
    }


def _audit(module, paths: dict[str, Path]) -> dict[str, object]:
    return module.audit_offline_evidence(
        merged_results_path=paths["merged"],
        parent_results_path=paths["parent"],
        repair_results_path=paths["repair"],
        repair_manifest_path=paths["manifest"],
        test_results_path=paths["test"],
        parent_phase_attempts_path=paths["phase_attempts"],
    )


def test_recomputes_statistics_and_safety_from_raw_integer_evidence(
    tmp_path: Path,
) -> None:
    module = _load_module()
    syntax_tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    local_imports = {
        alias.name
        for node in ast.walk(syntax_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
        if (
            alias.name.startswith("run_xunce_")
            or alias.name.startswith("xunce_")
        )
    }
    assert local_imports == {"xunce_artifact_io"}
    paths = _write_fixture(tmp_path / "fixture")

    audit = _audit(module, paths)

    assert audit["status"] == "passed"
    assert audit["sample_count"] == 24
    assert audit["lineage_counts"] == {
        "parent_read_only": 21,
        "repair_rerun": 3,
    }
    assert audit["coverage_80_count"] == 24
    assert audit["coverage_99_count"] == 1
    assert audit["mean"] == pytest.approx(0.97)
    assert audit["bootstrap_ci"] == {
        "confidence_level": 0.95,
        "lower": pytest.approx(0.9508333333333333),
        "resamples": 2_000,
        "seed": 20260726,
        "unit": "episode",
        "upper": pytest.approx(0.9816666666666667),
    }
    assert audit["safety"] == {
        "clean_episode_count": 22,
        "total_violation_count": 3,
        "violation_episode_count": 2,
        "violations_by_origin": {
            "parent_read_only": 1,
            "repair_rerun": 2,
        },
    }
    assert audit["repair_pairs"] == [
        {
            "after_coverage": 0.8,
            "before_coverage": 0.02,
            "episode_index": 3,
            "scenario_id": "unseen/scenario-0003/standard-proxy/v1",
        },
        {
            "after_coverage": 0.9,
            "before_coverage": 0.03,
            "episode_index": 10,
            "scenario_id": "unseen/scenario-0010/standard-proxy/v1",
        },
        {
            "after_coverage": 1.0,
            "before_coverage": 0.01,
            "episode_index": 23,
            "scenario_id": "unseen/scenario-0023/standard-proxy/v1",
        },
    ]
    assert len(audit["episodes"]) == 24
    assert all(
        row["coverage"]
        == row["final_covered_cell_count"] / row["denominator_cell_count"]
        for row in audit["episodes"]
    )
    assert audit["test_q24"]["coverage_80_count"] == 23
    assert audit["test_q24"]["mean"] == pytest.approx(
        (23 * 0.95 + 0.10) / 24
    )
    assert audit["test_q24"]["safety"]["total_violation_count"] == 0
    assert audit["test_q24"]["below_80_episodes"] == [
        {
            "coverage": 0.1,
            "episode_index": 20,
            "scenario_id": "test/scenario-0094/standard-proxy/v1",
        }
    ]
    assert audit["overall"] == {
        "scene_80_count": 47,
        "scene_count": 48,
        "split_mean_80_count": 2,
        "split_mean_count": 2,
    }


def test_rejects_raw_hash_drift_and_self_consistent_mixed_write(
    tmp_path: Path,
) -> None:
    module = _load_module()
    drift_paths = _write_fixture(tmp_path / "drift")
    drift_paths["parent"].write_bytes(
        drift_paths["parent"].read_bytes() + b" "
    )
    with pytest.raises(
        module.G1IndependentAuditError,
        match="parent_results_sha256_drift",
    ):
        _audit(module, drift_paths)

    test_drift_paths = _write_fixture(tmp_path / "test-drift")
    test_drift_paths["test"].write_bytes(
        test_drift_paths["test"].read_bytes() + b" "
    )
    with pytest.raises(
        module.G1IndependentAuditError,
        match="test_results_sha256_drift",
    ):
        _audit(module, test_drift_paths)

    mixed_paths = _write_fixture(tmp_path / "mixed")
    rows = [
        json.loads(line)
        for line in mixed_paths["merged"].read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    rows[0]["episode"]["coverage"] = 0.97
    rows[0]["episode"]["final_covered_cell_count"] = 97
    rows[0]["result_sha256"] = _sha256(
        _canonical_bytes(rows[0]["episode"])
    )
    forged_payload = b"".join(_canonical_line(row) for row in rows)
    mixed_paths["merged"].write_bytes(forged_payload)
    manifest = json.loads(
        mixed_paths["manifest"].read_text(encoding="utf-8")
    )
    manifest["merged_results_sha256"] = _sha256(forged_payload)
    mixed_paths["manifest"].write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(
        module.G1IndependentAuditError,
        match="mixed_origin_or_lineage",
    ):
        _audit(module, mixed_paths)


def test_rejects_integer_coverage_mismatch_before_aggregation(
    tmp_path: Path,
) -> None:
    module = _load_module()
    paths = _write_fixture(tmp_path / "integer-mismatch")
    rows = [
        json.loads(line)
        for line in paths["merged"].read_text(encoding="utf-8").splitlines()
    ]
    rows[7]["episode"]["coverage"] = 0.5
    rows[7]["result_sha256"] = _sha256(
        _canonical_bytes(rows[7]["episode"])
    )
    forged_payload = b"".join(_canonical_line(row) for row in rows)
    paths["merged"].write_bytes(forged_payload)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    manifest["merged_results_sha256"] = _sha256(forged_payload)
    paths["manifest"].write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(
        module.G1IndependentAuditError,
        match="integer_coverage_mismatch",
    ):
        _audit(module, paths)


def test_writes_new_audit_bundle_and_publication_exports_once(
    tmp_path: Path,
) -> None:
    module = _load_module()
    paths = _write_fixture(tmp_path / "deliverable-input")
    audit = _audit(module, paths)
    paired_layout = module._repair_display_layout(audit["repair_pairs"])
    assert [
        paired_layout[index]["x_offset"] for index in REPAIR_INDICES
    ] == [-0.035, 0.0, 0.035]
    assert len(
        {paired_layout[index]["marker"] for index in REPAIR_INDICES}
    ) == 3
    sorted_label_y = sorted(
        paired_layout[index]["label_y"] for index in REPAIR_INDICES
    )
    assert all(
        right - left >= 4.0
        for left, right in zip(
            sorted_label_y,
            sorted_label_y[1:],
        )
    )
    assert module._hero_repair_label_offset(3) != (
        module._hero_repair_label_offset(23)
    )
    output_dir = tmp_path / "independent-audit"

    manifest = module.write_deliverables(
        audit=audit,
        output_dir=output_dir,
        source_paths=paths,
    )

    expected = {
        "audit.json",
        "caption.md",
        "episodes.csv",
        "figure.pdf",
        "figure.png",
        "figure.svg",
        "manifest.json",
        "report.md",
        "source-data.json",
    }
    assert {path.name for path in output_dir.iterdir()} == expected
    assert set(manifest["artifacts"]) == expected - {"manifest.json"}
    with Image.open(output_dir / "figure.png") as image:
        dpi = image.info["dpi"]
        assert dpi[0] >= 299 and dpi[1] >= 299
        assert image.width >= 1_500
    svg = (output_dir / "figure.svg").read_text(encoding="utf-8")
    assert "<text" in svg
    assert "80% formal threshold" in svg
    assert "99% diagnostic reference" in svg
    assert "Test-Q24 original: 23/24 scenes" in svg
    assert "Unseen-24 repaired: 24/24 scenes" in svg
    assert "x-offsets only separate overlapping traces" in svg
    report = (output_dir / "report.md").read_text(encoding="utf-8")
    assert "24/24" in report
    assert "21 parent + 3 repair" in report
    assert "99% diagnostic reference" in report
    assert "Test-Q24：23/24" in report
    assert "47/48" in report
    assert "并非 48/48 个场景均过线" in report
    caption = (output_dir / "caption.md").read_text(encoding="utf-8")
    assert "n=24 episodes" in caption
    assert "2,000 episode-level bootstrap resamples" in caption
    assert "mixed-code incremental lineage" in caption
    source_data = json.loads(
        (output_dir / "source-data.json").read_text(encoding="utf-8")
    )
    assert source_data["thresholds"] == {
        "diagnostic_reference": 0.99,
        "formal_threshold": 0.8,
    }
    assert len(source_data["episodes"]) == 24
    assert len(source_data["test_q24_episodes"]) == 24
    assert len(source_data["repair_pairs"]) == 3
    assert manifest["audit_sha256"] == _sha256(
        (output_dir / "audit.json").read_bytes()
    )

    with pytest.raises(
        module.G1IndependentAuditError,
        match="output_directory_exists",
    ):
        module.write_deliverables(
            audit=copy.deepcopy(audit),
            output_dir=output_dir,
            source_paths=paths,
        )
