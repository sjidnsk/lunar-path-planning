from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from producer.audit_bundle import compare_bundle_bytes
from producer.canonical import canonical_json_bytes, domain_hash, sha256_bytes
from producer.package_bundle import (
    build_fixture_bundle,
    verify_contiguous_shard_prefix,
)
from producer.oracle_hopper import build_hopper_parameter_record
from run_producer import main, phase_names


ROOT = Path(__file__).resolve().parents[1]


def test_hash_seed_does_not_change_canonical_fixture_bytes(tmp_path: Path) -> None:
    snippet = """
from producer.canonical import canonical_json_bytes, derive_seed
payload = {
    "ids": sorted({"zeta", "alpha", "mu"}),
    "seed": derive_seed(20260727, "wheel", "labels", "nominal", 0),
}
import sys
sys.stdout.buffer.write(canonical_json_bytes(payload))
"""
    outputs: list[bytes] = []
    for hash_seed in ("1", "987654"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = hash_seed
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONPATH"] = str(ROOT)
        outputs.append(
            subprocess.run(
                [sys.executable, "-c", snippet],
                cwd=tmp_path,
                env=env,
                check=True,
                capture_output=True,
            ).stdout
        )
    assert outputs[0] == outputs[1]


def _hopper_record() -> dict[str, object]:
    return build_hopper_parameter_record(ROOT)


def _fixture_inputs() -> tuple[dict[str, bytes], dict[str, object]]:
    authorization = (
        ROOT.parents[1]
        / ".superpowers"
        / "sdd"
        / "2026-07-26-midterm-dual-gate-experiment"
        / "project-authorization.md"
    ).read_bytes()
    jp2 = b"repeatability-fixture-lola-jp2"
    label = b"repeatability-fixture-lola-label"
    return {
        "lola/LDEM_FIXTURE.JP2": jp2,
        "lola/LDEM_FIXTURE.LBL": label,
        "project-authorization.md": authorization,
    }, {
        "fixture_only": True,
        "jp2_sha256": sha256_bytes(jp2),
        "lbl_sha256": sha256_bytes(label),
        "macro_source_kind": "derived_lola_20m_macro_interpolation",
        "micro_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "physical_obstacle_cells_written": False,
    }


def test_two_fresh_fixture_roots_are_byte_identical(tmp_path: Path) -> None:
    raw, provenance = _fixture_inputs()
    first = tmp_path / "fresh-a"
    second = tmp_path / "fresh-b"
    freeze_a = build_fixture_bundle(
        first,
        source_root=ROOT,
        hopper_parameter_record=_hopper_record(),
        raw_sources=raw,
        lola_provenance=provenance,
    )
    freeze_b = build_fixture_bundle(
        second,
        source_root=ROOT,
        hopper_parameter_record=_hopper_record(),
        raw_sources=raw,
        lola_provenance=provenance,
    )
    assert freeze_a["payload_root_sha256"] == freeze_b["payload_root_sha256"]
    comparison = compare_bundle_bytes(first, second)
    assert comparison["matched"] is True
    assert comparison["mismatched_paths"] == []


def test_resume_requires_contiguous_hash_valid_prefix_and_identical_inputs(
    tmp_path: Path,
) -> None:
    shard_root = tmp_path / "shards"
    shard_root.mkdir()
    input_hash = "a" * 64
    implementation_hash = "b" * 64
    for index in (0, 1):
        rows = [{"index": index, "value": f"row-{index}"}]
        manifest = {
            "implementation_sha256": implementation_hash,
            "input_sha256": input_hash,
            "row_count": 1,
            "row_root_sha256": domain_hash(
                "g2-shard-rows/v1", canonical_json_bytes(rows)
            ),
            "shard_index": index,
        }
        (shard_root / f"shard-{index:04d}.json").write_bytes(
            canonical_json_bytes(manifest) + b"\n"
        )
    result = verify_contiguous_shard_prefix(
        shard_root,
        expected_input_sha256=input_hash,
        expected_implementation_sha256=implementation_hash,
    )
    assert result["accepted_indices"] == [0, 1]

    (shard_root / "shard-0003.json").write_bytes(
        (shard_root / "shard-0001.json").read_bytes()
    )
    with pytest.raises(ValueError, match="contiguous"):
        verify_contiguous_shard_prefix(
            shard_root,
            expected_input_sha256=input_hash,
            expected_implementation_sha256=implementation_hash,
        )
    with pytest.raises(ValueError, match="input drift"):
        verify_contiguous_shard_prefix(
            shard_root,
            expected_input_sha256="c" * 64,
            expected_implementation_sha256=implementation_hash,
        )


def test_cli_exposes_all_frozen_phases_without_formal_phase() -> None:
    assert phase_names() == (
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
    assert "formal" not in phase_names()


def test_cli_preflight_writes_canonical_evidence_without_secret_environment(
    tmp_path: Path,
) -> None:
    output = tmp_path / "preflight.json"
    exit_code = main(
        [
            "preflight",
            "--project-root",
            "C:/definitely-absent-g2-project-root",
            "--json-output",
            str(output),
        ]
    )
    assert exit_code == 0
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["passed"] is True
    assert "SECRET" not in evidence["environment"]
