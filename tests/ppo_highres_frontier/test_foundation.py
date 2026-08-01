from __future__ import annotations

import builtins
import ast
import copy
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
import tomllib
import warnings
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

try:
    from lunar_exploration_ppo.configs.schema import (
        FOUNDATION_WORKTREE_ROOT,
        FoundationConfig,
        load_foundation_config,
        resolve_device,
    )
    from lunar_exploration_ppo.utils.artifact_io import ArtifactPathError, ArtifactStore
    from lunar_exploration_ppo.workflows.gates import GateBindings, GateError, GateRecord
except ImportError:
    FOUNDATION_WORKTREE_ROOT = ""
    FoundationConfig = load_foundation_config = resolve_device = None  # type: ignore[assignment]
    ArtifactPathError = ArtifactStore = None  # type: ignore[assignment,misc]
    GateBindings = GateError = GateRecord = None  # type: ignore[assignment,misc]

try:
    from lunar_exploration_ppo.workflows.foundation import (
        FORBIDDEN_ARTIFACT_NAMES,
        FoundationPreflightError,
        record_foundation_independent_review,
        run_foundation_preflight,
    )
except ImportError:
    FORBIDDEN_ARTIFACT_NAMES = FoundationPreflightError = record_foundation_independent_review = run_foundation_preflight = None  # type: ignore[assignment]

try:
    from lunar_exploration_ppo.workflows.gates import (
        AUTHORIZED_NEXT_STAGE,
        NO_CHECKPOINT_SENTINEL,
        GateBindingVerifier,
        FoundationGateContext,
    )
except ImportError:
    AUTHORIZED_NEXT_STAGE = NO_CHECKPOINT_SENTINEL = None  # type: ignore[assignment]
    FoundationGateContext = GateBindingVerifier = None  # type: ignore[assignment,misc]


REPO_ROOT = Path(__file__).resolve().parents[2]
FOUNDATION_CONFIG = REPO_ROOT / "configs" / "ppo_highres_frontier_foundation_v1.json"


def _valid_config_dict() -> dict[str, object]:
    return {
        "schema_version": "ppo_highres_frontier_foundation/v1",
        "goal_id": "ppo-highres-frontier-map-exploration",
        "stage_id": "foundation",
        "repository_identity": {
            "source": "foundation_linked_worktree_identity/v1",
            "worktree_root": "C:/Users/77634/.codex/worktrees/ca49/lunar-path-planning",
            "branch": "codex/ppo-highres-frontier-map-exploration",
            "git_dir": "D:/codex/project/lunar-path-planning/.git/worktrees/lunar-path-planning1",
            "git_common_dir": "D:/codex/project/lunar-path-planning/.git",
        },
        "synthetic_source_kind": "synthetic_terrain_obstacle_proxy/v1",
        "physical_obstacle_cells_written": False,
        "value_prior_source": "constant_neutral/v1",
        "coordinate_convention": "world_xy_grid_col_row_cell_center/v1",
        "theta_convention": "radians_world_x_ccw_normalized_minus_pi_to_pi/v1",
        "grid_resolution_m": 0.5,
        "world_origin_xy_m": [0.0, 0.0],
        "cell_00_world_center_xy_m": [0.25, 0.25],
        "vehicle_radius_m": 0.4215874761,
        "safety_margin_m": 0.10,
        "min_clearance_m": 0.5215874761,
        "traversability_threshold": 0.50,
        "max_traversable_slope_deg": 30.0,
        "sensor_model_id": "path-tangent-plus-endpoint-theta-fov-90-range-20m-los/v1",
        "initial_observation": {
            "source": "reset_start_pose_los_scan/v1",
            "range_m": 20.0,
            "fov_deg": 90.0,
            "counts_reward": False,
            "counts_step": False,
        },
        "verified_data_sources": {
            "policy_access": "metadata_only_no_policy_read/v1",
            "dem": {
                "path": "D:/CodexDownloads/lunar-path-planning/data/raw/high_resolution_lunar_terrain/lunar_south_pole_usgs_lro_dem_slope_4m/MOON_LRO_NAC_DEM_89S210E_4mp.tif",
                "size_bytes": 310140246,
                "sha256": "7c431d32d977cd6b66572a0fae874ec5eae91be16aa21c4149ac897c677add22",
            },
            "slope": {
                "path": "D:/CodexDownloads/lunar-path-planning/data/raw/high_resolution_lunar_terrain/lunar_south_pole_usgs_lro_dem_slope_4m/MOON_LRO_NAC_Slope_89S210E_4mp.tif",
                "size_bytes": 70131711,
                "sha256": "df741123daec2d935b12481a579ce4443ce6c2e8916b13906ecd7801961112dd",
            },
        },
        "device": "cpu",
        "output_root": "D:/xunce/out/ppo_frontier",
    }


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _bindings(**updates: str) -> GateBindings:
    values = {
        "goal_hash": _digest("goal"),
        "stage_hash": _digest("stage"),
        "git_tree_hash": _digest("git-tree"),
        "config_hash": _digest("config"),
        "data_hash": _digest("data"),
        "environment_hash": _digest("environment"),
        "checkpoint_hash": _digest("checkpoint-none"),
        "review_hash": _digest("review"),
        "manifest_hash": _digest("manifest"),
        "authorized_next_stage": "ppo_highres_frontier_stage1_smoke_environment/v1",
    }
    values.update(updates)
    return GateBindings(**values)


def test_foundation_package_api_is_available() -> None:
    assert FoundationConfig is not None
    assert ArtifactStore is not None
    assert GateRecord is not None


def test_review_fix_package_apis_are_available() -> None:
    assert FoundationGateContext is not None
    assert GateBindingVerifier is not None
    assert run_foundation_preflight is not None
    assert record_foundation_independent_review is not None


def test_root_package_metadata_discovers_only_new_src_namespace() -> None:
    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["requires-python"] == ">=3.12,<3.13"
    discovery = metadata["tool"]["setuptools"]["packages"]["find"]
    assert discovery["where"] == ["src"]
    assert discovery["include"] == ["lunar_exploration_ppo*"]


def test_namespace_is_importable_and_isolated_from_legacy_modules() -> None:
    import lunar_exploration_ppo

    assert lunar_exploration_ppo.__version__
    source_root = REPO_ROOT / "src" / "lunar_exploration_ppo"
    forbidden = ("model_explorer", "scripts.xunce_", "sys.path")
    direct_planner_importers: list[Path] = []
    for source_file in source_root.rglob("*.py"):
        source = source_file.read_text(encoding="utf-8")
        assert not any(token in source for token in forbidden), source_file
        tree = ast.parse(source, filename=str(source_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(
                alias.name == "path_planner" or alias.name.startswith("path_planner.")
                for alias in node.names
            ):
                direct_planner_importers.append(source_file)
            if isinstance(node, ast.ImportFrom) and node.module and (
                node.module == "path_planner" or node.module.startswith("path_planner.")
            ):
                direct_planner_importers.append(source_file)
    assert {path.name for path in direct_planner_importers} == {
        "path_planner_adapter.py",
        "stage6_planning_child_source_repair.py",
    }


def test_foundation_config_freezes_coordinate_safety_proxy_and_scan_contracts() -> None:
    config = FoundationConfig.model_validate(_valid_config_dict())

    assert config.synthetic_source_kind == "synthetic_terrain_obstacle_proxy/v1"
    assert config.physical_obstacle_cells_written is False
    assert config.value_prior_source == "constant_neutral/v1"
    assert config.coordinate_convention == "world_xy_grid_col_row_cell_center/v1"
    assert config.theta_convention == "radians_world_x_ccw_normalized_minus_pi_to_pi/v1"
    assert config.cell_xy_to_array_indices(7, 11) == (11, 7)
    assert config.cell_center_world_xy(0, 0) == pytest.approx((0.25, 0.25))
    assert config.min_clearance_m == pytest.approx(config.vehicle_radius_m + config.safety_margin_m)
    assert config.traversability_threshold == 0.50
    assert config.max_traversable_slope_deg == 30.0
    assert config.sensor_model_id == "path-tangent-plus-endpoint-theta-fov-90-range-20m-los/v1"
    assert config.initial_observation.range_m == 20.0
    assert config.initial_observation.fov_deg == 90.0
    assert config.initial_observation.counts_reward is False
    assert config.initial_observation.counts_step is False
    assert config.repository_identity.model_dump() == {
        "source": "foundation_linked_worktree_identity/v1",
        "worktree_root": "C:/Users/77634/.codex/worktrees/ca49/lunar-path-planning",
        "branch": "codex/ppo-highres-frontier-map-exploration",
        "git_dir": "D:/codex/project/lunar-path-planning/.git/worktrees/lunar-path-planning1",
        "git_common_dir": "D:/codex/project/lunar-path-planning/.git",
    }


@pytest.mark.parametrize("field", ["worktree_root", "branch", "git_dir", "git_common_dir"])
def test_foundation_config_rejects_linked_worktree_identity_drift(field: str) -> None:
    payload = _valid_config_dict()
    identity = payload["repository_identity"]
    assert isinstance(identity, dict)
    identity[field] = "alternate"

    with pytest.raises(ValidationError):
        FoundationConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("synthetic_source_kind", "physical_obstacle_cells/v1"),
        ("physical_obstacle_cells_written", True),
        ("coordinate_convention", "array_xy/v1"),
        ("max_traversable_slope_deg", 31.0),
        ("output_root", "C:/tmp/ppo_frontier"),
    ],
)
def test_foundation_config_rejects_fixed_contract_drift(field: str, value: object) -> None:
    payload = _valid_config_dict()
    payload[field] = value

    with pytest.raises(ValidationError):
        FoundationConfig.model_validate(payload)


def test_foundation_config_fails_closed_on_unknown_fields_and_clearance_mismatch() -> None:
    unknown = _valid_config_dict() | {"future_override": True}
    mismatch = _valid_config_dict() | {"min_clearance_m": math.nextafter(0.5215874761, math.inf)}

    with pytest.raises(ValidationError):
        FoundationConfig.model_validate(unknown)
    with pytest.raises(ValidationError):
        FoundationConfig.model_validate(mismatch)


def test_repository_foundation_config_round_trips_exactly() -> None:
    config = load_foundation_config(FOUNDATION_CONFIG)

    assert config == FoundationConfig.model_validate_json(FOUNDATION_CONFIG.read_text(encoding="utf-8"))
    assert config.output_root == "D:/xunce/out/ppo_frontier"


def test_verified_data_source_evidence_is_frozen_without_requiring_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.configs import schema as schema_module

    def reject_file_probe(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("schema validation must not probe source files")

    monkeypatch.setattr(schema_module, "Path", reject_file_probe)
    config = FoundationConfig.model_validate(_valid_config_dict())

    assert config.verified_data_sources.policy_access == "metadata_only_no_policy_read/v1"
    assert config.verified_data_sources.dem.path.startswith("D:/")
    assert config.verified_data_sources.dem.path.endswith("MOON_LRO_NAC_DEM_89S210E_4mp.tif")
    assert config.verified_data_sources.dem.size_bytes == 310140246
    assert config.verified_data_sources.dem.sha256 == "7c431d32d977cd6b66572a0fae874ec5eae91be16aa21c4149ac897c677add22"
    assert config.verified_data_sources.slope.path.startswith("D:/")
    assert config.verified_data_sources.slope.path.endswith("MOON_LRO_NAC_Slope_89S210E_4mp.tif")
    assert config.verified_data_sources.slope.size_bytes == 70131711
    assert config.verified_data_sources.slope.sha256 == "df741123daec2d935b12481a579ce4443ce6c2e8916b13906ecd7801961112dd"


def test_verified_data_source_authority_cannot_be_rebound_through_module_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.configs import schema as schema_module

    assert not hasattr(schema_module, "DEM_SOURCE")
    assert not hasattr(schema_module, "SLOPE_SOURCE")
    monkeypatch.setattr(schema_module, "DEM_PATH", "D:/tampered/dem.tif")
    payload = _valid_config_dict()
    sources = payload["verified_data_sources"]
    assert isinstance(sources, dict)
    dem = sources["dem"]
    assert isinstance(dem, dict)
    dem["path"] = schema_module.DEM_PATH

    with pytest.raises(ValidationError):
        FoundationConfig.model_validate(payload)


@pytest.mark.parametrize("source_name", ["dem", "slope"])
def test_verified_data_source_evidence_rejects_hash_size_or_path_drift(source_name: str) -> None:
    for field, value in (
        ("path", "C:/unexpected/source.tif"),
        ("size_bytes", 1),
        ("sha256", "0" * 64),
    ):
        payload = _valid_config_dict()
        sources = payload["verified_data_sources"]
        assert isinstance(sources, dict)
        source = sources[source_name]
        assert isinstance(source, dict)
        source[field] = value
        with pytest.raises(ValidationError):
            FoundationConfig.model_validate(payload)


def test_cuda_request_never_silently_falls_back_to_cpu() -> None:
    assert resolve_device("cpu", cuda_available=False) == "cpu"
    assert resolve_device("cuda", cuda_available=True) == "cuda"
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda", cuda_available=False)


def test_artifact_store_warns_over_180_and_fails_at_240_full_path_chars(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    warning_name = "w" * (181 - len(str(tmp_path)) - 1)
    failure_name = "f" * (240 - len(str(tmp_path)) - 1)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert len(str(store.resolve(warning_name))) == 181
    assert any("180" in str(item.message) for item in caught)
    with pytest.raises(ArtifactPathError, match="240"):
        store.resolve(failure_name)


def test_artifact_store_atomic_writes_append_jsonl_and_builds_sha256_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import lunar_exploration_ppo.utils.artifact_io as artifact_module

    store = ArtifactStore(tmp_path)
    real_replace = artifact_module.durable_replace
    real_fsync = os.fsync
    real_open = builtins.open
    replacements: list[tuple[Path, Path]] = []
    fsync_calls: list[int] = []
    jsonl_modes: list[str] = []

    def checked_replace(
        source: str | os.PathLike[str],
        destination: str | os.PathLike[str],
        *,
        expected_source_identity: tuple[int, int, int, int, int, int],
        replace_existing: bool = True,
    ) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        assert source_path.parent == destination_path.parent
        assert source_path.is_file()
        assert len(expected_source_identity) == 6
        assert replace_existing is True
        replacements.append((source_path, destination_path))
        real_replace(
            source,
            destination,
            expected_source_identity=expected_source_identity,
            replace_existing=replace_existing,
        )

    def checked_fsync(file_descriptor: int) -> None:
        fsync_calls.append(file_descriptor)
        real_fsync(file_descriptor)

    def checked_open(file: object, mode: str = "r", *args: object, **kwargs: object) -> object:
        if Path(file).name == "metrics.jsonl":  # type: ignore[arg-type]
            jsonl_modes.append(mode)
        return real_open(file, mode, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(artifact_module, "durable_replace", checked_replace)
    monkeypatch.setattr(os, "fsync", checked_fsync)
    monkeypatch.setattr(builtins, "open", checked_open)
    store.write_json("config.json", {"b": 2, "a": 1})
    store.write_bytes("payload.bin", b"payload")
    store.write_checkpoint("checkpoint.ckpt", b"checkpoint")
    store.append_jsonl("metrics.jsonl", {"step": 1})
    store.append_jsonl("metrics.jsonl", {"step": 2})

    assert len(replacements) == 3
    assert len(fsync_calls) == 5
    assert jsonl_modes == ["ab", "ab"]
    assert json.loads((tmp_path / "config.json").read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert (tmp_path / "payload.bin").read_bytes() == b"payload"
    assert (tmp_path / "checkpoint.ckpt").read_bytes() == b"checkpoint"
    assert [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()] == [
        {"step": 1},
        {"step": 2},
    ]
    assert not list(tmp_path.glob("*.tmp*"))

    manifest = store.build_manifest(["config.json", "payload.bin", "checkpoint.ckpt", "metrics.jsonl"])
    entries = {entry["path"]: entry for entry in manifest["artifacts"]}
    for relative_path, entry in entries.items():
        payload = (tmp_path / relative_path).read_bytes()
        assert entry["size_bytes"] == len(payload)
        assert entry["sha256"] == hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize(
    "nonfinite",
    (float("nan"), float("inf"), float("-inf")),
)
def test_artifact_store_rejects_nested_nonfinite_json_without_side_effects(
    tmp_path: Path,
    nonfinite: float,
) -> None:
    store = ArtifactStore(tmp_path)
    value = {"outer": [{"nonfinite": nonfinite}]}
    existing_jsonl = tmp_path / "existing.jsonl"
    existing_payload = b'{"stable":true}\n'
    existing_jsonl.write_bytes(existing_payload)

    with pytest.raises(ValueError, match="JSON compliant|Out of range"):
        ArtifactStore.canonical_json_bytes(value)
    with pytest.raises(ValueError, match="JSON compliant|Out of range"):
        store.write_json("atomic.json", value)
    with pytest.raises(ValueError, match="JSON compliant|Out of range"):
        store.write_json_exclusive("exclusive.json", value)
    with pytest.raises(ValueError, match="JSON compliant|Out of range"):
        store.append_jsonl("nested/new.jsonl", value)
    with pytest.raises(ValueError, match="JSON compliant|Out of range"):
        store.append_jsonl("existing.jsonl", value)

    assert existing_jsonl.read_bytes() == existing_payload
    assert not (tmp_path / "atomic.json").exists()
    assert not (tmp_path / "exclusive.json").exists()
    assert not (tmp_path / "nested").exists()
    assert not [path for path in tmp_path.iterdir() if ".tmp" in path.name]


def test_artifact_store_exclusive_json_publishes_canonical_complete_bytes(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    value = {"unicode": "月面", "b": 2, "a": 1}

    written = store.write_json_exclusive("gate.json", value)

    assert written.resolve() == (tmp_path / "gate.json").resolve()
    assert written.read_bytes() == ArtifactStore.canonical_json_bytes(value)
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".gate.json.")]


def test_artifact_store_exclusive_bytes_rejects_preexisting_target_without_mutation(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path)
    target = tmp_path / "gate.json"
    foreign = b'{"actor":"foreign-preexisting"}\n'
    target.write_bytes(foreign)

    with pytest.raises(FileExistsError):
        store.write_bytes_exclusive("gate.json", b'{"actor":"ours"}\n')

    assert target.read_bytes() == foreign
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".gate.json.")]


def test_artifact_store_exclusive_publish_race_preserves_foreign_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lunar_exploration_ppo.utils.path_security as path_security

    store = ArtifactStore(tmp_path)
    target = tmp_path / "gate.json"
    ours = b'{"actor":"ours"}\n'
    foreign = b'{"actor":"foreign-racer"}\n'
    race_injections = 0

    def inject_competing_destination(
        event: str,
        source: Path,
        destination: Path | None,
    ) -> None:
        nonlocal race_injections
        if event != "after_source_handle_bound" or destination != target:
            return
        race_injections += 1
        source_path = Path(source)
        destination_path = Path(destination)
        assert source_path.parent == destination_path.parent
        assert source_path.name.startswith(".gate.json.")
        assert not destination_path.exists()
        destination_path.write_bytes(foreign)

    def reject_replace(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("exclusive publish must never fall back to os.replace")

    monkeypatch.setattr(path_security, "_namespace_event", inject_competing_destination)
    monkeypatch.setattr(os, "replace", reject_replace)

    with pytest.raises(FileExistsError):
        store.write_bytes_exclusive("gate.json", ours)

    assert race_injections == 1
    assert target.read_bytes() == foreign
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".gate.json.")]


def test_artifact_store_exclusive_namespace_unsupported_fails_closed_without_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lunar_exploration_ppo.utils.artifact_io as artifact_module

    store = ArtifactStore(tmp_path)
    namespace_error = OSError("identity-bound namespace primitive unavailable")

    def reject_publish(*_args: object, **_kwargs: object) -> None:
        raise namespace_error

    def reject_replace(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("exclusive publish must never fall back to os.replace")

    monkeypatch.setattr(
        artifact_module,
        "durable_publish_exclusive",
        reject_publish,
    )
    monkeypatch.setattr(os, "replace", reject_replace)

    with pytest.raises(OSError) as captured:
        store.write_json_exclusive("gate.json", {"actor": "ours"})

    assert captured.value is namespace_error
    assert not (tmp_path / "gate.json").exists()
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".gate.json.")]


@pytest.mark.skipif(os.name != "nt", reason="Windows long-path prefix contract")
def test_artifact_store_executes_real_windows_long_path_io(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    padding_length = 181 - len(str(tmp_path / "config.json")) - 1
    long_directory = "l" * padding_length

    with pytest.warns(RuntimeWarning, match="180"):
        json_path = store.write_json(Path(long_directory) / "config.json", {"ok": True})
    with pytest.warns(RuntimeWarning, match="180"):
        checkpoint_path = store.write_checkpoint(Path(long_directory) / "checkpoint.ckpt", b"checkpoint")
    with pytest.warns(RuntimeWarning, match="180"):
        jsonl_path = store.append_jsonl(Path(long_directory) / "metrics.jsonl", {"step": 1})

    assert str(store._native_path(json_path)).startswith("\\\\?\\")
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"ok": True}
    assert checkpoint_path.read_bytes() == b"checkpoint"
    assert [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()] == [{"step": 1}]
    unc_native = str(store._native_path(Path(r"\\server\share\folder\artifact.json")))
    assert unc_native.startswith("\\\\?\\UNC\\")


def _mock_authoritative_git(
    monkeypatch: pytest.MonkeyPatch,
    *,
    top_level: Path = REPO_ROOT,
    status: str = "",
    tree_oid: list[str] | None = None,
    physical_calls: list[Path] | None = None,
    branch: str = "codex/ppo-highres-frontier-map-exploration",
    git_dir: str = "D:/codex/project/lunar-path-planning/.git/worktrees/lunar-path-planning1",
    git_common_dir: str = "D:/codex/project/lunar-path-planning/.git",
) -> tuple[list[str], list[list[str]]]:
    from lunar_exploration_ppo.workflows import gates as gates_module

    mutable_tree_oid = tree_oid or ["1" * 40]
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if "status" in command:
            return subprocess.CompletedProcess(command, 0, stdout=status, stderr="")
        if "--show-toplevel" in command:
            return subprocess.CompletedProcess(command, 0, stdout=str(top_level.resolve()) + "\n", stderr="")
        if "--git-dir" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=git_dir + "\n",
                stderr="",
            )
        if "--git-common-dir" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=git_common_dir + "\n",
                stderr="",
            )
        if "branch" in command:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=branch + "\n",
                stderr="",
            )
        if "HEAD^{tree}" in command:
            return subprocess.CompletedProcess(command, 0, stdout=mutable_tree_oid[0] + "\n", stderr="")
        raise AssertionError(f"unexpected git command: {command}")

    monkeypatch.setattr(gates_module.subprocess, "run", fake_run)
    normalized_repo_root = os.path.normcase(os.path.normpath(str(REPO_ROOT.resolve())))
    normalized_contract_root = os.path.normcase(
        os.path.normpath(str(FOUNDATION_WORKTREE_ROOT))
    )
    if normalized_repo_root != normalized_contract_root:
        def portable_verify_repository(
            cls: object,
            repo_root: str | Path,
        ) -> tuple[Path, str]:
            if Path(repo_root).resolve() != REPO_ROOT.resolve():
                raise GateError("Foundation repository root does not match the frozen linked worktree")
            return REPO_ROOT.resolve(), mutable_tree_oid[0]

        monkeypatch.setattr(
            gates_module.GateBindingVerifier,
            "_verify_foundation_repository",
            classmethod(portable_verify_repository),
        )

    config = load_foundation_config(FOUNDATION_CONFIG)
    expected_physical = {
        Path(config.verified_data_sources.dem.path).name: (
            config.verified_data_sources.dem.size_bytes,
            config.verified_data_sources.dem.sha256,
        ),
        Path(config.verified_data_sources.slope.path).name: (
            config.verified_data_sources.slope.size_bytes,
            config.verified_data_sources.slope.sha256,
        ),
    }

    def fake_stream_file_size_sha256(path: Path) -> tuple[int, str]:
        if physical_calls is not None:
            physical_calls.append(path)
        try:
            return expected_physical[path.name]
        except KeyError as exc:
            raise AssertionError(f"unexpected physical source: {path}") from exc

    monkeypatch.setattr(
        gates_module.GateBindingVerifier,
        "_stream_file_size_sha256",
        staticmethod(fake_stream_file_size_sha256),
        raising=False,
    )
    return mutable_tree_oid, calls


def _make_foundation_gate_context(tmp_path: Path, *, run_id: str = "gate-run-001") -> object:
    base_output_root = tmp_path / "gate-output"
    result = run_foundation_preflight(
        config_path=FOUNDATION_CONFIG,
        base_output_root=base_output_root,
        run_id=run_id,
    )
    environment_root = tmp_path / "environment-lock"
    environment_root.mkdir()
    lock_file = environment_root / "lock.txt"
    lock_file.write_bytes(b"environment-lock/v1")
    environment_manifest = environment_root / "environment-lock-manifest.json"
    environment_manifest.write_bytes(
        ArtifactStore.canonical_json_bytes(
            {
                "schema_version": "ppo_foundation_environment_lock_manifest/v1",
                "environment_prefix": "D:/conda_envs/lunar-explorer",
                "files": [
                    {
                        "file": "lock.txt",
                        "size_bytes": len(lock_file.read_bytes()),
                        "sha256": hashlib.sha256(lock_file.read_bytes()).hexdigest(),
                    }
                ],
            }
        )
    )
    stage_root = result.stage_root
    record_foundation_independent_review(
        stage_root=stage_root,
        environment_manifest=environment_manifest,
        review_source_hash=_digest("review-source"),
        review_package_hash=_digest("review-package"),
        spec_verdict="approved",
        quality_verdict="approved",
        critical_count=0,
        important_count=0,
        minor_count=0,
    )
    return FoundationGateContext(
        repo_root="C:/Users/77634/.codex/worktrees/ca49/lunar-path-planning",
        stage_root=stage_root,
        environment_manifest=environment_manifest,
        run_id=run_id,
    )


def _make_machine_review_fixture(
    tmp_path: Path,
    *,
    run_id: str,
    config_path: Path = FOUNDATION_CONFIG,
) -> tuple[Path, Path]:
    result = run_foundation_preflight(
        config_path=config_path,
        base_output_root=tmp_path / "review-output",
        run_id=run_id,
    )
    environment_root = tmp_path / "review-environment"
    environment_root.mkdir()
    lock_file = environment_root / "lock.txt"
    lock_file.write_bytes(b"environment-lock/v1")
    environment_manifest = environment_root / "environment-lock-manifest.json"
    environment_manifest.write_bytes(
        ArtifactStore.canonical_json_bytes(
            {
                "schema_version": "ppo_foundation_environment_lock_manifest/v1",
                "environment_prefix": "D:/conda_envs/lunar-explorer",
                "files": [
                    {
                        "file": "lock.txt",
                        "size_bytes": len(lock_file.read_bytes()),
                        "sha256": hashlib.sha256(lock_file.read_bytes()).hexdigest(),
                    }
                ],
            }
        )
    )
    return result.stage_root, environment_manifest


def _sync_review_manifest_hash(context: object) -> None:
    review_path = context.stage_root / "review.json"
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["manifest_hash"] = hashlib.sha256((context.stage_root / "manifest.json").read_bytes()).hexdigest()
    review_path.write_bytes(ArtifactStore.canonical_json_bytes(review))


def test_foundation_gate_context_exposes_only_minimal_locator_fields(tmp_path: Path) -> None:
    context = _make_foundation_gate_context(tmp_path)

    assert set(FoundationGateContext.model_fields) == {
        "repo_root",
        "stage_root",
        "environment_manifest",
        "run_id",
    }
    with pytest.raises(ValidationError):
        FoundationGateContext.model_validate(
            context.model_dump()
            | {
                "goal_identity": "alternate-goal",
                "config_path": tmp_path / "stale-config.json",
            }
        )


def test_foundation_gate_context_rejects_alternate_clean_repo_locator(tmp_path: Path) -> None:
    stage_root = tmp_path / "alternate-repo-001" / "s0"
    stage_root.mkdir(parents=True)
    environment_manifest = tmp_path / "environment-lock-manifest.json"
    environment_manifest.write_text("{}", encoding="utf-8")
    alternate_repo = tmp_path / "independent-clean-repo"
    for marker in (
        "pyproject.toml",
        "docs/superpowers/specs/2026-07-09-ppo-highres-frontier-map-exploration-design.md",
        "docs/superpowers/plans/2026-07-10-ppo-highres-frontier-map-exploration.md",
        "configs/ppo_highres_frontier_foundation_v1.json",
    ):
        marker_path = alternate_repo / marker
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text("copied marker", encoding="utf-8")
    with pytest.raises(ValidationError):
        FoundationGateContext(
            repo_root=str(alternate_repo),
            stage_root=stage_root,
            environment_manifest=environment_manifest,
            run_id="alternate-repo-001",
        )


def test_gate_machine_passed_derives_authoritative_foundation_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _make_foundation_gate_context(tmp_path)
    _, calls = _mock_authoritative_git(monkeypatch)
    record = GateRecord.machine_passed(context)

    assert record.state == "machine_passed"
    assert record.history == ("machine_passed",)
    assert record.context == context
    assert record.bindings.git_tree_hash == "1" * 40
    assert record.bindings.authorized_next_stage == "ppo_highres_frontier_stage1_smoke_environment/v1"
    assert record.bindings.checkpoint_hash == hashlib.sha256(NO_CHECKPOINT_SENTINEL.encode("utf-8")).hexdigest()
    assert record.to_dict()["bindings"]["authorized_next_stage"] == AUTHORIZED_NEXT_STAGE
    assert any("--show-toplevel" in command for command in calls)
    assert any("HEAD^{tree}" in command for command in calls)


def test_gate_accepts_cuda_runtime_override_through_review_and_verified_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.workflows import foundation as foundation_module

    canonical_payload = json.loads(FOUNDATION_CONFIG.read_text(encoding="utf-8"))
    assert canonical_payload["device"] == "cpu"
    runtime_config_path = tmp_path / "cuda-runtime-config.json"
    runtime_config_path.write_bytes(
        ArtifactStore.canonical_json_bytes(canonical_payload | {"device": "cuda"})
    )
    monkeypatch.setattr(
        foundation_module,
        "resolve_device",
        lambda requested: resolve_device(requested, cuda_available=True),
    )
    run_id = "cuda-runtime-gate-001"
    stage_root, environment_manifest = _make_machine_review_fixture(
        tmp_path,
        run_id=run_id,
        config_path=runtime_config_path,
    )
    record_foundation_independent_review(
        stage_root=stage_root,
        environment_manifest=environment_manifest,
        review_source_hash=_digest("cuda-review-source"),
        review_package_hash=_digest("cuda-review-package"),
        spec_verdict="approved",
        quality_verdict="approved",
        critical_count=0,
        important_count=0,
        minor_count=0,
    )
    context = FoundationGateContext(
        repo_root="C:/Users/77634/.codex/worktrees/ca49/lunar-path-planning",
        stage_root=stage_root,
        environment_manifest=environment_manifest,
        run_id=run_id,
    )
    _mock_authoritative_git(monkeypatch)

    record = GateRecord.machine_passed(context)
    loaded = GateRecord.load_verified(record.to_dict())

    stage_config = load_foundation_config(stage_root / "config.json")
    summary = json.loads((stage_root / "summary.json").read_text(encoding="utf-8"))
    assert stage_config.device == summary["device"] == "cuda"
    assert loaded.to_dict() == record.to_dict()


def test_gate_rejects_summary_device_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _make_foundation_gate_context(tmp_path)
    _mock_authoritative_git(monkeypatch)
    summary_path = context.stage_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["device"] = "cuda"
    summary_path.write_bytes(ArtifactStore.canonical_json_bytes(summary))

    with pytest.raises(GateError, match="summary device"):
        GateRecord.machine_passed(context)


def test_gate_streams_both_frozen_physical_sources_before_building_data_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _make_foundation_gate_context(tmp_path)
    physical_calls: list[Path] = []
    _mock_authoritative_git(monkeypatch, physical_calls=physical_calls)
    record = GateRecord.machine_passed(context)
    config = load_foundation_config(context.stage_root / "config.json")
    verified_records = {
        "schema_version": "foundation_verified_physical_data/v1",
        "sources": [
            config.verified_data_sources.dem.model_dump(mode="json"),
            config.verified_data_sources.slope.model_dump(mode="json"),
        ],
    }

    assert [path.name for path in physical_calls] == [
        Path(config.verified_data_sources.dem.path).name,
        Path(config.verified_data_sources.slope.path).name,
    ]
    assert record.bindings.data_hash == hashlib.sha256(
        ArtifactStore.canonical_json_bytes(verified_records)
    ).hexdigest()


@pytest.mark.parametrize("failure_kind", ["missing", "size", "sha256"])
def test_gate_fails_closed_on_physical_data_missing_size_or_hash_drift(
    failure_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.workflows import gates as gates_module

    context = _make_foundation_gate_context(tmp_path)
    _mock_authoritative_git(monkeypatch)
    config = load_foundation_config(context.stage_root / "config.json")
    expected_size = config.verified_data_sources.dem.size_bytes
    expected_hash = config.verified_data_sources.dem.sha256

    def failing_stream(path: Path) -> tuple[int, str]:
        if path.name != Path(config.verified_data_sources.dem.path).name:
            return (
                config.verified_data_sources.slope.size_bytes,
                config.verified_data_sources.slope.sha256,
            )
        if failure_kind == "missing":
            raise GateError("physical data source is missing")
        if failure_kind == "size":
            return expected_size + 1, expected_hash
        return expected_size, "0" * 64

    monkeypatch.setattr(
        gates_module.GateBindingVerifier,
        "_stream_file_size_sha256",
        staticmethod(failing_stream),
        raising=False,
    )
    with pytest.raises(GateError, match="physical data"):
        GateRecord.machine_passed(context)


_FROZEN_DATA_CONFIG = load_foundation_config(FOUNDATION_CONFIG)
_FROZEN_DATA_PATHS = (
    Path(_FROZEN_DATA_CONFIG.verified_data_sources.dem.path),
    Path(_FROZEN_DATA_CONFIG.verified_data_sources.slope.path),
)


@pytest.mark.skipif(
    not hasattr(GateBindingVerifier, "_stream_file_size_sha256")
    or not all(path.is_file() for path in _FROZEN_DATA_PATHS),
    reason="frozen DEM/Slope files are not available on this host",
)
def test_frozen_physical_data_stream_hash_integration_when_files_exist() -> None:
    evidence = (
        _FROZEN_DATA_CONFIG.verified_data_sources.dem,
        _FROZEN_DATA_CONFIG.verified_data_sources.slope,
    )
    for source, expected in zip(_FROZEN_DATA_PATHS, evidence, strict=True):
        size_bytes, sha256 = GateBindingVerifier._stream_file_size_sha256(source)
        assert size_bytes == expected.size_bytes
        assert sha256 == expected.sha256


def test_physical_data_stream_helper_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(GateError, match="physical data source is missing"):
        GateBindingVerifier._stream_file_size_sha256(tmp_path / "missing-dem.tif")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("goal_id", "alternate-goal"),
        ("stage_id", "alternate-stage"),
    ],
)
def test_gate_rejects_wrong_foundation_identity_in_canonical_config(
    field: str,
    value: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _make_foundation_gate_context(tmp_path)
    _mock_authoritative_git(monkeypatch)
    payload = json.loads((context.stage_root / "config.json").read_text(encoding="utf-8"))
    payload[field] = value
    (context.stage_root / "config.json").write_bytes(ArtifactStore.canonical_json_bytes(payload))

    with pytest.raises(GateError, match="config"):
        GateRecord.machine_passed(context)


def test_gate_rejects_submodule_repo_and_stale_alternate_path_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _make_foundation_gate_context(tmp_path)
    _mock_authoritative_git(monkeypatch, top_level=REPO_ROOT / "path-planner")
    submodule_context = context.model_copy(update={"repo_root": REPO_ROOT / "path-planner"})
    with pytest.raises(GateError, match="Foundation repository"):
        GateRecord.machine_passed(submodule_context)

    with pytest.raises(ValidationError):
        FoundationGateContext.model_validate(
            context.model_dump() | {"manifest_path": tmp_path / "stale-manifest.json"}
        )


def test_gate_rejects_repo_top_level_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _make_foundation_gate_context(tmp_path)
    _mock_authoritative_git(monkeypatch, top_level=REPO_ROOT / "path-planner")

    with pytest.raises(GateError, match="top-level"):
        GateRecord.machine_passed(context)


@pytest.mark.parametrize(
    ("git_override", "error"),
    [
        ({"branch": "alternate-branch"}, "branch"),
        ({"git_dir": "D:/alternate/worktree"}, "Git dir"),
        ({"git_common_dir": "D:/alternate/common"}, "common dir"),
    ],
)
def test_gate_rejects_frozen_branch_git_dir_or_common_dir_mismatch(
    git_override: dict[str, str],
    error: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _make_foundation_gate_context(tmp_path)
    _mock_authoritative_git(monkeypatch, **git_override)

    with pytest.raises(GateError, match=error):
        GateRecord.machine_passed(context)


def test_gate_rejects_stage_run_mismatch_and_cross_hash_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _make_foundation_gate_context(tmp_path)
    _mock_authoritative_git(monkeypatch)
    summary_path = context.stage_root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["run_id"] = "different-run"
    summary_path.write_bytes(ArtifactStore.canonical_json_bytes(summary))
    with pytest.raises(GateError, match="run_id"):
        GateRecord.machine_passed(context)

    fresh_context = _make_foundation_gate_context(tmp_path / "fresh", run_id="gate-run-002")
    _mock_authoritative_git(monkeypatch)
    lock_file = fresh_context.environment_manifest.parent / "lock.txt"
    lock_file.write_bytes(b"drifted-environment")
    with pytest.raises(GateError, match="environment"):
        GateRecord.machine_passed(fresh_context)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra"])
def test_gate_manifest_requires_exact_six_unique_machine_artifacts(
    mutation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _make_foundation_gate_context(tmp_path)
    _mock_authoritative_git(monkeypatch)
    manifest_path = context.stage_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest["artifacts"]
    if mutation == "missing":
        manifest["artifacts"] = [entry for entry in artifacts if entry["path"] != "summary.json"]
    elif mutation == "duplicate":
        manifest["artifacts"].append(dict(next(entry for entry in artifacts if entry["path"] == "config.json")))
    else:
        extra_path = context.stage_root / "extra.txt"
        extra_path.write_bytes(b"extra artifact")
        manifest["artifacts"].append(
            {
                "path": "extra.txt",
                "size_bytes": len(extra_path.read_bytes()),
                "sha256": hashlib.sha256(extra_path.read_bytes()).hexdigest(),
            }
        )
    manifest_path.write_bytes(ArtifactStore.canonical_json_bytes(manifest))
    _sync_review_manifest_hash(context)

    with pytest.raises(GateError, match="exactly six"):
        GateRecord.machine_passed(context)


@pytest.mark.parametrize(
    "binding_name",
    [
        "goal_hash",
        "stage_hash",
        "git_tree_hash",
        "config_hash",
        "data_hash",
        "environment_hash",
        "checkpoint_hash",
        "review_hash",
        "manifest_hash",
        "authorized_next_stage",
    ],
)
def test_gate_bindings_fail_closed_when_any_field_is_missing(binding_name: str) -> None:
    payload = _bindings().model_dump()
    payload.pop(binding_name)
    with pytest.raises(ValidationError):
        GateBindings.model_validate(payload)


def test_gate_transition_history_is_a_complete_legal_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _make_foundation_gate_context(tmp_path)
    _mock_authoritative_git(monkeypatch)
    record = GateRecord.machine_passed(context)

    with pytest.raises(GateError):
        record.transition("approved")
    for state in (
        "awaiting_independent_review",
        "awaiting_human_approval",
        "approved",
        "next_stage",
    ):
        record = record.transition(state)
        assert record.state == state
        assert record.history[-1] == state
    assert record.history == (
        "machine_passed",
        "awaiting_independent_review",
        "awaiting_human_approval",
        "approved",
        "next_stage",
    )


def test_gate_record_is_private_immutable_and_not_pydantic() -> None:
    assert not issubclass(GateRecord, BaseModel)
    for forbidden_api in ("model_validate", "model_validate_json", "model_construct", "model_copy"):
        assert not hasattr(GateRecord, forbidden_api)
    with pytest.raises(TypeError):
        GateRecord()


def test_gate_record_verified_loader_replays_round_trip_and_rejects_forgery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _make_foundation_gate_context(tmp_path)
    _mock_authoritative_git(monkeypatch)
    record = GateRecord.machine_passed(context)
    record = record.transition("awaiting_independent_review")
    record = record.transition("awaiting_human_approval")
    serialized = record.to_dict()

    loaded = GateRecord.load_verified(serialized)
    assert loaded.to_dict() == serialized
    assert copy.copy(loaded) is loaded
    assert copy.deepcopy(loaded) is loaded

    forged = json.loads(json.dumps(serialized))
    forged["state"] = "next_stage"
    forged["history"].extend(
        [
            {
                "state": "approved",
                "previous_record_hash": forged["history"][-1]["record_hash"],
                "record_hash": forged["history"][-1]["record_hash"],
            },
            {
                "state": "next_stage",
                "previous_record_hash": forged["history"][-1]["record_hash"],
                "record_hash": forged["history"][-1]["record_hash"],
            },
        ]
    )
    with pytest.raises(GateError, match="history|replay|hash"):
        GateRecord.load_verified(forged)

    drifted = json.loads(json.dumps(serialized))
    drifted["bindings"]["config_hash"] = "0" * 64
    with pytest.raises(GateError, match="binding"):
        GateRecord.load_verified(drifted)


def test_gate_record_direct_advanced_construction_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _make_foundation_gate_context(tmp_path)
    _mock_authoritative_git(monkeypatch)
    initial = GateRecord.machine_passed(context)

    with pytest.raises(TypeError):
        GateRecord(
            state="next_stage",
            history=(
                "machine_passed",
                "awaiting_independent_review",
                "awaiting_human_approval",
                "approved",
                "next_stage",
            ),
            bindings=initial.bindings,
            context=context,
        )


def test_gate_transition_recomputes_canonical_sources_and_rejects_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _make_foundation_gate_context(tmp_path)
    tree_oid, _ = _mock_authoritative_git(monkeypatch)
    record = GateRecord.machine_passed(context)
    tree_oid[0] = "2" * 40

    with pytest.raises(GateError, match="git_tree_hash"):
        record.transition("awaiting_independent_review")


def test_independent_review_transition_persists_awaiting_human_approval(tmp_path: Path) -> None:
    stage_root, environment_manifest = _make_machine_review_fixture(
        tmp_path,
        run_id="review-transition-001",
    )
    source_hash = _digest("review-source")
    package_hash = _digest("review-package")

    record_foundation_independent_review(
        stage_root=stage_root,
        environment_manifest=environment_manifest,
        review_source_hash=source_hash,
        review_package_hash=package_hash,
        spec_verdict="approved",
        quality_verdict="approved",
        critical_count=0,
        important_count=0,
        minor_count=2,
    )

    summary = json.loads((stage_root / "summary.json").read_text(encoding="utf-8"))
    routing = json.loads((stage_root / "routing.json").read_text(encoding="utf-8"))
    review = json.loads((stage_root / "review.json").read_text(encoding="utf-8"))
    phases = [json.loads(line) for line in (stage_root / "phase-state.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest_bytes = (stage_root / "manifest.json").read_bytes()
    assert summary["state"] == "awaiting_human_approval"
    assert routing["route"] == "awaiting_human_approval"
    assert [event["state"] for event in phases] == [
        "machine_passed",
        "awaiting_independent_review",
        "awaiting_human_approval",
    ]
    assert review["spec_verdict"] == review["quality_verdict"] == "approved"
    assert review["issue_counts"] == {"critical": 0, "important": 0, "minor": 2}
    assert review["review_source_hash"] == source_hash
    assert review["review_package_hash"] == package_hash
    assert review["config_hash"] == summary["config_hash"]
    assert review["manifest_hash"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert review["environment_manifest_hash"] == hashlib.sha256(environment_manifest.read_bytes()).hexdigest()
    assert not (stage_root / "approval.json").exists()
    assert not (stage_root / "gate.json").exists()

    manifest = json.loads(manifest_bytes.decode("utf-8"))
    names = [entry["path"] for entry in manifest["artifacts"]]
    assert len(names) == len(set(names)) == 6
    with pytest.raises(FoundationPreflightError, match="already|state"):
        record_foundation_independent_review(
            stage_root=stage_root,
            environment_manifest=environment_manifest,
            review_source_hash=source_hash,
            review_package_hash=package_hash,
            spec_verdict="approved",
            quality_verdict="approved",
            critical_count=0,
            important_count=0,
            minor_count=2,
        )


def test_independent_review_and_gate_accept_utf8_bom_environment_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage_root, environment_manifest = _make_machine_review_fixture(
        tmp_path,
        run_id="review-bom-001",
    )
    environment_manifest.write_bytes(b"\xef\xbb\xbf" + environment_manifest.read_bytes())

    record_foundation_independent_review(
        stage_root=stage_root,
        environment_manifest=environment_manifest,
        review_source_hash=_digest("review-source"),
        review_package_hash=_digest("review-package"),
        spec_verdict="approved",
        quality_verdict="approved",
        critical_count=0,
        important_count=0,
        minor_count=0,
    )
    context = FoundationGateContext(
        repo_root="C:/Users/77634/.codex/worktrees/ca49/lunar-path-planning",
        stage_root=stage_root,
        environment_manifest=environment_manifest,
        run_id="review-bom-001",
    )
    _mock_authoritative_git(monkeypatch)

    record = GateRecord.machine_passed(context)

    assert record.state == "machine_passed"
    review = json.loads((stage_root / "review.json").read_text(encoding="utf-8"))
    assert review["environment_manifest_hash"] == hashlib.sha256(environment_manifest.read_bytes()).hexdigest()


def test_independent_review_rejects_resolved_environment_manifest_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage_root, environment_manifest = _make_machine_review_fixture(
        tmp_path,
        run_id="review-environment-escape-001",
    )
    source_path = environment_manifest.parent / "lock.txt"
    outside_path = tmp_path / "outside-lock.txt"
    outside_path.write_bytes(source_path.read_bytes())
    real_resolve = Path.resolve
    outside_resolved = real_resolve(outside_path)

    def escaped_resolve(self: Path, *args: object, **kwargs: object) -> Path:
        if self == source_path:
            return outside_resolved
        return real_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", escaped_resolve)
    with pytest.raises(FoundationPreflightError, match="escapes environment manifest root"):
        record_foundation_independent_review(
            stage_root=stage_root,
            environment_manifest=environment_manifest,
            review_source_hash=_digest("review-source"),
            review_package_hash=_digest("review-package"),
            spec_verdict="approved",
            quality_verdict="approved",
            critical_count=0,
            important_count=0,
            minor_count=0,
        )
    assert not (stage_root / "review.json").exists()


def test_independent_review_rejects_resolved_machine_artifact_escape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage_root, environment_manifest = _make_machine_review_fixture(
        tmp_path,
        run_id="review-artifact-escape-001",
    )
    source_path = stage_root / "report.md"
    outside_path = tmp_path / "outside-report.md"
    outside_path.write_bytes(source_path.read_bytes())
    real_resolve = Path.resolve
    outside_resolved = real_resolve(outside_path)

    def escaped_resolve(self: Path, *args: object, **kwargs: object) -> Path:
        if self == source_path:
            return outside_resolved
        return real_resolve(self, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", escaped_resolve)
    with pytest.raises(FoundationPreflightError, match="escapes stage root"):
        record_foundation_independent_review(
            stage_root=stage_root,
            environment_manifest=environment_manifest,
            review_source_hash=_digest("review-source"),
            review_package_hash=_digest("review-package"),
            spec_verdict="approved",
            quality_verdict="approved",
            critical_count=0,
            important_count=0,
            minor_count=0,
        )
    assert not (stage_root / "review.json").exists()


def test_independent_review_reports_missing_environment_source_stably(tmp_path: Path) -> None:
    stage_root, environment_manifest = _make_machine_review_fixture(
        tmp_path,
        run_id="review-environment-missing-001",
    )
    (environment_manifest.parent / "lock.txt").unlink()

    with pytest.raises(FoundationPreflightError, match="environment manifest file is missing"):
        record_foundation_independent_review(
            stage_root=stage_root,
            environment_manifest=environment_manifest,
            review_source_hash=_digest("review-source"),
            review_package_hash=_digest("review-package"),
            spec_verdict="approved",
            quality_verdict="approved",
            critical_count=0,
            important_count=0,
            minor_count=0,
        )
    assert not (stage_root / "review.json").exists()


@pytest.mark.parametrize(
    "updates",
    [
        {"spec_verdict": "changes_required"},
        {"quality_verdict": "changes_required"},
        {"critical_count": 1},
        {"important_count": 1},
    ],
)
def test_independent_review_transition_rejects_non_approved_or_blocking_findings(
    updates: dict[str, object],
    tmp_path: Path,
) -> None:
    stage_root, environment_manifest = _make_machine_review_fixture(
        tmp_path,
        run_id="review-reject-001",
    )
    arguments: dict[str, object] = {
        "stage_root": stage_root,
        "environment_manifest": environment_manifest,
        "review_source_hash": _digest("review-source"),
        "review_package_hash": _digest("review-package"),
        "spec_verdict": "approved",
        "quality_verdict": "approved",
        "critical_count": 0,
        "important_count": 0,
        "minor_count": 0,
    }
    arguments.update(updates)

    with pytest.raises(FoundationPreflightError, match="review"):
        record_foundation_independent_review(**arguments)
    assert not (stage_root / "review.json").exists()
    summary = json.loads((stage_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["state"] == "machine_passed"


def test_independent_review_mid_write_failure_is_fail_closed_and_not_repeatable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stage_root, environment_manifest = _make_machine_review_fixture(
        tmp_path,
        run_id="review-mid-failure-001",
    )
    real_write_json = ArtifactStore.write_json

    def fail_routing_write(self: ArtifactStore, relative_path: str | Path, value: object) -> Path:
        if str(relative_path) == "routing.json":
            raise OSError("simulated routing write failure")
        return real_write_json(self, relative_path, value)

    monkeypatch.setattr(ArtifactStore, "write_json", fail_routing_write)
    arguments = {
        "stage_root": stage_root,
        "environment_manifest": environment_manifest,
        "review_source_hash": _digest("review-source"),
        "review_package_hash": _digest("review-package"),
        "spec_verdict": "approved",
        "quality_verdict": "approved",
        "critical_count": 0,
        "important_count": 0,
        "minor_count": 0,
    }
    with pytest.raises(OSError, match="simulated"):
        record_foundation_independent_review(**arguments)
    assert not (stage_root / "review.json").exists()

    monkeypatch.setattr(ArtifactStore, "write_json", real_write_json)
    with pytest.raises(FoundationPreflightError, match="state"):
        record_foundation_independent_review(**arguments)


def test_foundation_workflow_writes_unique_run_stage_and_hashes_actual_config_bytes(tmp_path: Path) -> None:
    base_output_root = tmp_path / "foundation-output"
    result = run_foundation_preflight(
        config_path=FOUNDATION_CONFIG,
        base_output_root=base_output_root,
        run_id="foundation-run-001",
    )
    output_root = base_output_root / "foundation-run-001" / "s0"

    assert result.stage_root == output_root.resolve()
    assert {path.name for path in output_root.iterdir()} == {
        "config.json",
        "summary.json",
        "routing.json",
        "manifest.json",
        "report.md",
        "metrics.jsonl",
        "phase-state.jsonl",
    }
    config_bytes = (output_root / "config.json").read_bytes()
    written_config = json.loads(config_bytes.decode("utf-8"))
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    routing = json.loads((output_root / "routing.json").read_text(encoding="utf-8"))
    metrics = [json.loads(line) for line in (output_root / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    phases = [json.loads(line) for line in (output_root / "phase-state.jsonl").read_text(encoding="utf-8").splitlines()]
    assert summary["config_hash"] == hashlib.sha256(config_bytes).hexdigest()
    assert written_config["run_id"] == "foundation-run-001"
    assert summary["config_hash_schema"] == "sha256_file_bytes/v1"
    assert summary["run_id"] == "foundation-run-001"
    assert routing["run_id"] == "foundation-run-001"
    assert metrics == [
        {
            "device": "cpu",
            "event": "foundation_contract_check",
            "passed": True,
            "run_id": "foundation-run-001",
        }
    ]
    assert phases == [
        {
            "route": "awaiting_independent_review",
            "run_id": "foundation-run-001",
            "state": "machine_passed",
        }
    ]

    with pytest.raises(FoundationPreflightError, match="non-empty"):
        run_foundation_preflight(
            config_path=FOUNDATION_CONFIG,
            base_output_root=base_output_root,
            run_id="foundation-run-001",
        )


def test_foundation_workflow_rejects_stale_or_unsafe_run_targets(tmp_path: Path) -> None:
    stale_stage = tmp_path / "stale-run" / "s0"
    stale_stage.mkdir(parents=True)
    (stale_stage / "stale.txt").write_text("stale", encoding="utf-8")

    with pytest.raises(FoundationPreflightError, match="non-empty"):
        run_foundation_preflight(
            config_path=FOUNDATION_CONFIG,
            base_output_root=tmp_path,
            run_id="stale-run",
        )
    sibling_run_root = tmp_path / "sibling-run"
    (sibling_run_root / "s1").mkdir(parents=True)
    with pytest.raises(FoundationPreflightError, match="run root.*non-empty"):
        run_foundation_preflight(
            config_path=FOUNDATION_CONFIG,
            base_output_root=tmp_path,
            run_id="sibling-run",
        )
    for run_id in ("", ".", "..", "nested/run", "nested\\run"):
        with pytest.raises(FoundationPreflightError, match="run_id"):
            run_foundation_preflight(
                config_path=FOUNDATION_CONFIG,
                base_output_root=tmp_path,
                run_id=run_id,
            )
    assert FORBIDDEN_ARTIFACT_NAMES == frozenset({"approval.json", "gate.json"})


@pytest.mark.parametrize(
    "run_id",
    ["CON", "con.txt", "PRN", "AUX", "NUL", "COM1", "com9.log", "LPT1", "LPT9.txt", "run.", "run "],
)
def test_foundation_workflow_rejects_windows_reserved_run_ids(run_id: str, tmp_path: Path) -> None:
    with pytest.raises(FoundationPreflightError, match="run_id"):
        run_foundation_preflight(
            config_path=FOUNDATION_CONFIG,
            base_output_root=tmp_path,
            run_id=run_id,
        )


@pytest.mark.parametrize("forbidden_name", ["approval.json", "gate.json"])
def test_foundation_workflow_rejects_simulated_forbidden_artifacts_without_creating_them(
    forbidden_name: str,
) -> None:
    from lunar_exploration_ppo.workflows import foundation as foundation_module

    class SimulatedRunRoot:
        def exists(self) -> bool:
            return True

        def iterdir(self) -> tuple[Path, ...]:
            return (Path(forbidden_name),)

        def __str__(self) -> str:
            return "simulated-stage-root"

    with pytest.raises(FoundationPreflightError, match="forbidden"):
        prepare = getattr(foundation_module, "_prepare_unique_run_root", None)
        assert callable(prepare)
        prepare(SimulatedRunRoot())


def test_foundation_cli_is_thin_and_writes_only_machine_stage_artifacts(tmp_path: Path) -> None:
    base_output_root = tmp_path / "foundation-cli-output"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "run_ppo_foundation_preflight.py"),
            "--config",
            str(FOUNDATION_CONFIG),
            "--base-output-root",
            str(base_output_root),
            "--run-id",
            "cli-run-001",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    output_root = base_output_root / "cli-run-001" / "s0"
    assert {path.name for path in output_root.iterdir()} == {
        "config.json",
        "summary.json",
        "routing.json",
        "manifest.json",
        "report.md",
        "metrics.jsonl",
        "phase-state.jsonl",
    }
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    routing = json.loads((output_root / "routing.json").read_text(encoding="utf-8"))
    assert summary["state"] == "machine_passed"
    assert summary["run_id"] == "cli-run-001"
    assert routing["route"] == "awaiting_independent_review"
    assert not (output_root / "approval.json").exists()
    assert not (output_root / "gate.json").exists()

    runner_source = (REPO_ROOT / "scripts" / "run_ppo_foundation_preflight.py").read_text(encoding="utf-8")
    assert "run_foundation_preflight" in runner_source
    for forbidden_core in ("ArtifactStore", "hashlib", "MACHINE_ARTIFACTS", "phase-state.jsonl"):
        assert forbidden_core not in runner_source


def test_bootstrap_ci_and_ignore_contracts_cover_root_package_without_cuda_upgrade() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/bootstrap_env.py",
            "--dry-run",
            "--platform",
            "windows",
            "--install-editable",
            "--skip-submodules",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert str(REPO_ROOT) in output
    assert "lunar_exploration_ppo import ok" in output
    assert "pytorch-cuda" not in output.lower()

    workflow = (REPO_ROOT / ".github" / "workflows" / "platform-compatibility.yml").read_text(encoding="utf-8")
    assert 'python-version: "3.12"' in workflow
    assert "windows-latest" in workflow and "ubuntu-latest" in workflow
    assert "python -m build" in workflow
    assert "pip install" in workflow and ".whl" in workflow
    assert "test_foundation.py" in workflow
    assert "device cpu" in workflow or 'resolve_device("cpu"' in workflow

    ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("/.superpowers/", "/build/", "/dist/", "*.egg-info/", "*.ckpt", "*.pt", "/artifacts/"):
        assert pattern in ignore
