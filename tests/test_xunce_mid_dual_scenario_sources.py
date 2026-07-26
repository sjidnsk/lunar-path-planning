from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "prepare_xunce_mid_dual_scenario_sources.py"
CONFIG_PATH = REPO_ROOT / "configs" / "xunce_mid_dual_scenario_sources_v1.json"
AUTHORIZATION_SHA256 = (
    "720e11ef04ad2b57283421809a077ccf1f0b35167a9f482082241398ad0214d2"
)


def _load_module():
    assert SCRIPT_PATH.is_file(), "scenario source preparer is missing"
    spec = importlib.util.spec_from_file_location("mid_dual_scenario_sources", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    if str(SCRIPT_PATH.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT_PATH.parent))
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _descriptor(scenario_id: str, split: str, scenario_hash: str) -> dict[str, object]:
    return {
        "scenario_id": scenario_id,
        "scenario_hash": scenario_hash,
        "source_split": split,
        "source_pool_sha256": "0" * 64,
        "slope_p90_deg": 12.5,
        "hard_obstacle_fraction": 0.125,
        "start_pose_bin": [0, 1, 2],
        "initial_observed_coverable_fraction": 0.0625,
        "initial_valid_frontier_count": 8,
        "coverable_cell_count": 1024,
        "parent_roi": "r00001-c00002",
        "density_profile": "medium",
        "start_to_farthest_candidate_distance_bin": 0,
    }


def test_config_pins_policy_blind_full_source_contract() -> None:
    assert CONFIG_PATH.is_file(), "scenario source config is missing"
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert value == {
        "schema_version": "mid-dual-scenario-source-materialization/v1",
        "catalog_split_counts": {"test": 150, "unseen": 64, "validation": 150},
        "excluded_splits": ["train"],
        "coverage_denominator_source": "reachable_observable_free_highres_cells/v1",
        "coverage_denominator_algorithm": "exact_reachable_safe_pose_range_los/v1",
        "distance_bin_algorithm": "within_split_stable_rank_tertiles/v1",
        "policy_blind_approval_id": "mid-dual-policy-blind-source-approval-20260727/v1",
        "project_authorization_sha256": AUTHORIZATION_SHA256,
        "publication_mode": "data_files_then_completion_manifest/v1",
    }


def test_pool_hash_binding_is_order_stable_and_split_isolated() -> None:
    module = _load_module()
    test_a = _descriptor("test-a", "test", "1" * 64)
    test_b = _descriptor("test-b", "test", "2" * 64)
    unseen = _descriptor("unseen-a", "unseen", "3" * 64)
    validation = _descriptor("validation-a", "validation", "4" * 64)

    forward = module.bind_source_pool_hashes([test_b, validation, test_a, unseen])
    reverse = module.bind_source_pool_hashes([unseen, test_a, validation, test_b])

    assert forward == reverse
    by_id = {row["scenario_id"]: row for row in forward}
    assert by_id["test-a"]["source_pool_sha256"] == (
        "10274d3332c9f5e441eb0e52594a08864ba5ac453d1d3c715a06ae8c988b0c18"
    )
    assert by_id["test-b"]["source_pool_sha256"] == by_id["test-a"]["source_pool_sha256"]
    assert by_id["unseen-a"]["source_pool_sha256"] == (
        "799f084a79a17b451cff4fdfb35bb8cc77a51fd630f7152021cc3d4053f79d64"
    )
    assert by_id["validation-a"]["source_pool_sha256"] == (
        "978cab39ec0e7c7853d9733019320051b119e04405009152dc2d309384329dc9"
    )


def test_distance_bins_are_stable_rank_tertiles_with_id_ties() -> None:
    module = _load_module()
    rows = [
        {"scenario_id": "b", "source_split": "test", "max_reset_candidate_distance_m": 5.0},
        {"scenario_id": "a", "source_split": "test", "max_reset_candidate_distance_m": 5.0},
        {"scenario_id": "c", "source_split": "test", "max_reset_candidate_distance_m": 9.0},
        {"scenario_id": "u", "source_split": "unseen", "max_reset_candidate_distance_m": 99.0},
    ]
    assert module.assign_distance_bins(rows) == {
        "a": 0,
        "b": 1,
        "c": 2,
        "u": 0,
    }
    assert module.assign_distance_bins(list(reversed(rows))) == {
        "a": 0,
        "b": 1,
        "c": 2,
        "u": 0,
    }


def test_coverable_mask_npz_is_canonical_and_byte_stable() -> None:
    module = _load_module()
    mask = np.asarray([[True, False], [True, True]], dtype=np.bool_)
    first = module.serialize_reconstruction_mask(mask)
    second = module.serialize_reconstruction_mask(mask.copy())
    assert first == second
    assert hashlib.sha256(first).hexdigest() == (
        "a0b7cd10dcf856475ee1e143fd37d986b282499e9173dab7c05ab5f646ec09db"
    )
    with zipfile.ZipFile(io.BytesIO(first), "r") as archive:
        assert archive.namelist() == ["coverable_mask.npy"]
        info = archive.getinfo("coverable_mask.npy")
        assert info.compress_type == zipfile.ZIP_STORED
        assert info.date_time == (1980, 1, 1, 0, 0, 0)
        with archive.open(info, "r") as handle:
            loaded = np.load(handle, allow_pickle=False)
    assert loaded.dtype == np.dtype(bool)
    assert np.array_equal(loaded, mask)


def test_standard_catalog_artifact_uses_the_catalogs_authoritative_encoding() -> None:
    module = _load_module()
    payload = {
        "schema_version": "standard_unseen_scenario_catalog/v1",
        "split_policy": "fixed_seed_disjoint_split/v1",
        "sources": {"dem": {"sha256": "1" * 64}},
        "records": [{"scenario_id": "test/a"}],
    }
    authoritative = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")

    class Catalog:
        sha256 = hashlib.sha256(authoritative).hexdigest()

        def to_dict(self):
            return {"catalog_sha256": self.sha256, **payload}

    result = module._catalog_artifact_bytes(Catalog())
    assert result == authoritative
    assert hashlib.sha256(result).hexdigest() == Catalog.sha256


def test_source_manifest_binds_artifacts_authorization_and_forbidden_inputs() -> None:
    module = _load_module()
    refs = {
        name: {
            "artifact_id": f"{name}/v1",
            "path": f"D:/xunce/inputs/mid_dual/source/{name}.json",
            "sha256": character * 64,
        }
        for name, character in {
            "descriptor_catalog": "1",
            "standard_catalog": "2",
            "standard_source": "3",
            "static_truth_cache": "4",
            "reset_state": "5",
        }.items()
    }
    value = module.build_source_manifest(
        artifact_refs=refs,
        descriptor_generator_path="C:/repo/scripts/prepare_xunce_mid_dual_scenario_sources.py",
        descriptor_generator_sha256="6" * 64,
        coverage_manifest_path="D:/xunce/cache/run/coverage-cache-manifest.json",
        coverage_manifest_sha256="7" * 64,
        coverage_catalog_sha256="2" * 64,
        approval_path="D:/xunce/inputs/mid_dual/source/policy-blind-approval.json",
        approval_sha256="8" * 64,
        source_pool_hashes={"test": "9" * 64, "unseen": "a" * 64, "validation": "b" * 64},
    )
    assert value["schema_version"] == "mid-dual-scenario-source-manifest/v1"
    assert value["policy_blind_attestation"] == {
        "attestation_id": "mid-dual-policy-blind-source-attestation/v1",
        "approval_id": "mid-dual-policy-blind-source-approval-20260727/v1",
        "approval_artifact_path": "D:/xunce/inputs/mid_dual/source/policy-blind-approval.json",
        "approval_artifact_sha256": "8" * 64,
        "forbidden_inputs_used": {
            "policy": False,
            "checkpoint": False,
            "reward": False,
            "runtime": False,
            "result": False,
        },
    }
    assert value["stage6_coverage_manifest"]["catalog_sha256"] == "2" * 64


def test_project_authorization_must_exist_and_match_exact_bytes(tmp_path: Path) -> None:
    module = _load_module()
    path = tmp_path / "authorization.md"
    path.write_bytes(b"not the approved authorization")
    with pytest.raises(ValueError, match="authorization SHA-256 drifted"):
        module.validate_project_authorization(path)
    with pytest.raises(ValueError, match="authorization is unreadable"):
        module.validate_project_authorization(tmp_path / "missing.md")


def test_publish_writes_completion_last_and_rejects_completed_root(
    tmp_path: Path,
) -> None:
    module = _load_module()
    data = {
        "standard-catalog.json": b'{"catalog":"exact"}',
        "source-manifest.json": b'{"source":"exact"}',
    }
    masks = {"masks/" + "a" * 64 + ".npz": b"mask-bytes"}
    completion = module.build_completion_manifest(data, masks)
    root = module.publish_source_bundle(
        data_files=data,
        mask_files=masks,
        completion_manifest=completion,
        output_root=tmp_path,
    )
    assert (root / "manifest.json").read_bytes() == completion
    assert (root / "standard-catalog.json").read_bytes() == data["standard-catalog.json"]
    assert (root / ("masks/" + "a" * 64 + ".npz")).read_bytes() == b"mask-bytes"
    with pytest.raises(FileExistsError, match="complete scenario source root"):
        module.publish_source_bundle(
            data_files=data,
            mask_files=masks,
            completion_manifest=completion,
            output_root=tmp_path,
        )


def test_cli_without_execute_refuses_before_any_artifact_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_module()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("no-execute path touched an artifact")

    monkeypatch.setattr(module.artifact_io, "read_bytes", forbidden)
    monkeypatch.setattr(module.artifact_io, "make_dirs", forbidden)
    with pytest.raises(SystemExit, match="without --execute"):
        module.main([])
