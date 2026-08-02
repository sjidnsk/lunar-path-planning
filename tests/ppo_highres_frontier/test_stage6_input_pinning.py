"""Stage 6 正式输入的 Windows 强 pin 合同。"""

from __future__ import annotations

import importlib
import importlib.util
import json
import multiprocessing
import os
import subprocess
from pathlib import Path

import pytest


MODULE_NAME = "lunar_exploration_ppo.utils.stage6_input_pinning"


def _module():
    return importlib.import_module(MODULE_NAME)


def _spawn_read(path: str, connection) -> None:
    try:
        connection.send(("ok", Path(path).read_bytes()))
    except BaseException as exc:  # pragma: no cover - parent asserts payload
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def _make_directory_junction(link: Path, target: Path) -> None:
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        pytest.skip(
            "Windows junction creation is unavailable: "
            f"{completed.stdout.strip()} {completed.stderr.strip()}"
        )


def test_stage6_input_pinning_module_import_is_side_effect_free() -> None:
    spec = importlib.util.find_spec(MODULE_NAME)

    assert spec is not None
    module = _module()
    assert hasattr(module, "Stage6InputPinError")
    assert hasattr(module, "Stage6InputPin")
    assert hasattr(module, "acquire_stage6_input_pin")


def test_pin_cannot_be_directly_constructed_or_forged() -> None:
    module = _module()

    with pytest.raises((TypeError, module.Stage6InputPinError), match="issuer|acquisition"):
        module.Stage6InputPin(resources=[], guards=[])

    forged = object.__new__(module.Stage6InputPin)
    with pytest.raises(module.Stage6InputPinError, match="forged|issued|active"):
        forged.require_current("forged pin")
    with pytest.raises(module.Stage6InputPinError, match="forged|issued|active"):
        _ = forged.records
    with pytest.raises(module.Stage6InputPinError, match="forged|issued|active"):
        _ = forged.paths


@pytest.mark.skipif(os.name != "nt", reason="Windows strong pin contract")
def test_closed_pin_is_inactive_and_hides_records_and_paths(tmp_path: Path) -> None:
    module = _module()
    source = tmp_path / "source.bin"
    source.write_bytes(b"reviewed")
    pin = module.acquire_stage6_input_pin((("source", source),))
    assert pin.records

    pin.close()

    with pytest.raises(module.Stage6InputPinError, match="closed|inactive"):
        pin.require_current("closed pin")
    with pytest.raises(module.Stage6InputPinError, match="closed|inactive"):
        _ = pin.records
    with pytest.raises(module.Stage6InputPinError, match="closed|inactive"):
        _ = pin.paths
    with pytest.raises(module.Stage6InputPinError, match="closed|inactive"):
        pin.read_bytes(source, label="closed pinned read")


def test_public_acquisition_fails_closed_off_windows_without_touching_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    source = tmp_path / "source.bin"
    source.write_bytes(b"reviewed")
    before = source.read_bytes()
    monkeypatch.setattr(module.os, "name", "posix")

    with pytest.raises(
        module.Stage6InputPinError,
        match="Windows.*only|formal.*Windows|unsupported",
    ):
        module.acquire_stage6_input_pin((("source", source),))

    assert source.read_bytes() == before


@pytest.mark.skipif(os.name != "nt", reason="Windows strong pin contract")
def test_windows_pin_blocks_mutation_allows_python_and_spawn_reads_then_releases(
    tmp_path: Path,
) -> None:
    module = _module()
    parent = tmp_path / "inputs"
    parent.mkdir()
    source = parent / "source.bin"
    replacement = parent / "replacement.bin"
    source.write_bytes(b"reviewed-source")
    replacement.write_bytes(b"foreign-replacement")
    moved_parent = tmp_path / "inputs-moved"

    with module.acquire_stage6_input_pin(
        (
            ("canonical-config", source),
            ("source:duplicate-alias", source),
        )
    ) as pin:
        assert pin.paths == (source.resolve(),)
        assert len(pin.records) == 1
        assert pin.records[0].labels == (
            "canonical-config",
            "source:duplicate-alias",
        )
        assert pin.records[0].size_bytes == len(b"reviewed-source")
        assert pin.read_bytes(source, label="descriptor-backed read") == (
            b"reviewed-source"
        )
        assert source.read_bytes() == b"reviewed-source"
        pin.require_current("Windows lock held")

        child_connection, parent_connection = multiprocessing.get_context(
            "spawn"
        ).Pipe(duplex=False)
        process = multiprocessing.get_context("spawn").Process(
            target=_spawn_read,
            args=(str(source), parent_connection),
        )
        process.start()
        parent_connection.close()
        status, payload = child_connection.recv()
        process.join(timeout=30.0)
        child_connection.close()
        assert process.exitcode == 0
        assert (status, payload) == ("ok", b"reviewed-source")

        with pytest.raises(OSError):
            with source.open("r+b") as stream:
                stream.write(b"tamper")
        with pytest.raises(OSError):
            os.replace(replacement, source)
        with pytest.raises(OSError):
            os.remove(source)
        with pytest.raises(OSError):
            os.rename(parent, moved_parent)
        assert source.read_bytes() == b"reviewed-source"
        assert replacement.read_bytes() == b"foreign-replacement"
        with pytest.raises(TypeError):
            json.dumps(pin)

    with source.open("r+b") as stream:
        stream.seek(0, os.SEEK_END)
        stream.write(b"-released")
    os.rename(parent, moved_parent)
    assert (moved_parent / "source.bin").read_bytes() == (
        b"reviewed-source-released"
    )


@pytest.mark.skipif(os.name != "nt", reason="Windows strong pin contract")
def test_windows_pin_rejects_a_writer_that_already_has_the_input_open(
    tmp_path: Path,
) -> None:
    module = _module()
    source = tmp_path / "source.bin"
    source.write_bytes(b"reviewed")

    with source.open("r+b"):
        with pytest.raises(module.Stage6InputPinError, match="acquisition|open"):
            module.acquire_stage6_input_pin((("source", source),))

    with module.acquire_stage6_input_pin((("source", source),)) as pin:
        pin.require_current("writer released")


@pytest.mark.skipif(os.name != "nt", reason="Windows strong pin contract")
def test_windows_pin_rejects_initial_hardlink_and_reparse_inputs(
    tmp_path: Path,
) -> None:
    module = _module()
    hardlink_parent = tmp_path / "hardlink"
    hardlink_parent.mkdir()
    source = hardlink_parent / "source.bin"
    alias = hardlink_parent / "alias.bin"
    source.write_bytes(b"reviewed")
    os.link(source, alias)

    with pytest.raises(module.Stage6InputPinError, match="hard link|link count"):
        module.acquire_stage6_input_pin((("hardlink", source),))

    target_parent = tmp_path / "junction-target"
    target_parent.mkdir()
    (target_parent / "source.bin").write_bytes(b"reviewed")
    linked_parent = tmp_path / "junction-input"
    _make_directory_junction(linked_parent, target_parent)

    with pytest.raises(module.Stage6InputPinError, match="reparse|link"):
        module.acquire_stage6_input_pin(
            (("reparse", linked_parent / "source.bin"),)
        )


@pytest.mark.skipif(os.name != "nt", reason="Windows strong pin contract")
def test_partial_acquisition_failure_closes_all_open_descriptors_and_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    first_parent = tmp_path / "first"
    second_parent = tmp_path / "second"
    first_parent.mkdir()
    second_parent.mkdir()
    first = first_parent / "first.bin"
    second = second_parent / "second.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    real_open = module.DurableParentGuard.open_file

    def fail_second(guard, path, flags, mode=0o600):
        if Path(path) == second.resolve():
            raise OSError("injected second input open failure")
        return real_open(guard, path, flags, mode)

    monkeypatch.setattr(module.DurableParentGuard, "open_file", fail_second)

    with pytest.raises(
        module.Stage6InputPinError,
        match="acquisition failed|open failure",
    ):
        module.acquire_stage6_input_pin(
            (("first", first), ("second", second))
        )

    with first.open("r+b") as stream:
        stream.write(b"F")
    moved = tmp_path / "first-moved"
    os.rename(first_parent, moved)
    assert (moved / "first.bin").read_bytes() == b"First"


@pytest.mark.skipif(os.name != "nt", reason="Windows strong pin contract")
def test_hash_failure_after_leaf_open_closes_that_descriptor_and_parent_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    parent = tmp_path / "hash-failure"
    parent.mkdir()
    source = parent / "source.bin"
    source.write_bytes(b"reviewed")
    monkeypatch.setattr(
        module,
        "_descriptor_sha256",
        lambda descriptor: (_ for _ in ()).throw(OSError("injected hash failure")),
    )

    with pytest.raises(module.Stage6InputPinError, match="hash failure|acquisition"):
        module.acquire_stage6_input_pin((("source", source),))

    with source.open("r+b") as stream:
        stream.write(b"R")
    moved = tmp_path / "hash-failure-moved"
    os.rename(parent, moved)
    assert (moved / "source.bin").read_bytes() == b"Reviewed"


@pytest.mark.skipif(os.name != "nt", reason="Windows strong pin contract")
def test_context_preserves_primary_error_and_closes_every_resource_on_close_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    pin = module.acquire_stage6_input_pin((("first", first), ("second", second)))
    real_close = module.os.close
    close_calls = {"count": 0}

    def close_then_fail_once(descriptor: int) -> None:
        real_close(descriptor)
        close_calls["count"] += 1
        if close_calls["count"] == 1:
            raise OSError("injected close failure after close")

    monkeypatch.setattr(module.os, "close", close_then_fail_once)
    with pytest.raises(RuntimeError, match="primary execution failure") as captured:
        with pin:
            raise RuntimeError("primary execution failure")

    assert any(
        "suppressed Stage 6 input pin close failure" in note
        for note in getattr(captured.value, "__notes__", ())
    )
    assert close_calls["count"] == 2
    with first.open("r+b") as stream:
        stream.write(b"F")
    with second.open("r+b") as stream:
        stream.write(b"S")


@pytest.mark.skipif(os.name != "nt", reason="Windows strong pin contract")
def test_real_dem_remains_readable_by_python_and_rasterio_while_pinned() -> None:
    module = _module()
    rasterio = pytest.importorskip("rasterio")
    from rasterio.windows import Window

    from lunar_exploration_ppo.configs.schema import DEM_PATH

    dem = Path(DEM_PATH)
    if not dem.is_file():
        pytest.skip("frozen Standard DEM is unavailable on this machine")

    with module.acquire_stage6_input_pin((("standard-dem", dem),)) as pin:
        with dem.open("rb") as stream:
            assert stream.read(4)
        with rasterio.open(dem) as dataset:
            sample = dataset.read(1, window=Window(0, 0, 2, 2))
        assert sample.shape == (2, 2)
        pin.require_current("Rasterio read complete", rehash=True)
