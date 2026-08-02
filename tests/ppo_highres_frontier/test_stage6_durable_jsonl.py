from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest


def _canonical_row(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _durable_file_identity(path: Path) -> tuple[int, int, int, int, int, int]:
    metadata = path.lstat()
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(stat.S_IFMT(metadata.st_mode)),
        int(metadata.st_size),
        int(getattr(metadata, "st_file_attributes", 0)),
        int(metadata.st_nlink),
    )


class _InjectedFault(OSError):
    pass


def _fault_at(expected_phase: str):
    def inject(phase: str) -> None:
        if phase == expected_phase:
            raise _InjectedFault(f"injected fault: {phase}")

    return inject


def _make_directory_reparse(link: Path, target: Path) -> None:
    if os.name != "nt":
        os.symlink(target, link, target_is_directory=True)
        return
    completed = subprocess.run(
        ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        pytest.fail(
            "Windows junction creation failed: "
            f"{completed.stdout} {completed.stderr}"
        )


def _subprocess_env() -> dict[str, str]:
    environment = os.environ.copy()
    source_root = Path(__file__).resolve().parents[2] / "src"
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{source_root}{os.pathsep}{existing}" if existing else str(source_root)
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


@pytest.mark.parametrize("operation", ("replace", "exclusive", "unlink"))
def test_identity_bound_namespace_operation_rejects_replaced_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    import lunar_exploration_ppo.utils.path_security as path_security

    source = tmp_path / f"{operation}-source.bin"
    displaced = tmp_path / f"{operation}-owned.bin"
    destination = tmp_path / f"{operation}-destination.bin"
    owned = b"owned-source-bytes"
    foreign = b"foreign-source-bytes"
    source.write_bytes(owned)
    expected_identity = _durable_file_identity(source)
    injected = False

    def replace_before_handle(
        event: str,
        event_source: Path,
        _event_destination: Path | None,
    ) -> None:
        nonlocal injected
        if event == "before_source_handle_open":
            injected = True
            os.replace(event_source, displaced)
            event_source.write_bytes(foreign)

    monkeypatch.setattr(
        path_security,
        "_namespace_event",
        replace_before_handle,
        raising=False,
    )

    with pytest.raises(path_security.PathSecurityError, match="identity|changed"):
        if operation == "replace":
            path_security.durable_replace(
                source,
                destination,
                expected_source_identity=expected_identity,
                replace_existing=True,
            )
        elif operation == "exclusive":
            path_security.durable_publish_exclusive(
                source,
                destination,
                expected_source_identity=expected_identity,
            )
        else:
            path_security.durable_unlink(
                source,
                expected_identity=expected_identity,
            )

    assert injected is True
    assert source.read_bytes() == foreign
    assert displaced.read_bytes() == owned
    assert not destination.exists()


def test_exclusive_publish_preserves_destination_created_at_publication_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lunar_exploration_ppo.utils.path_security as path_security

    source = tmp_path / "exclusive-owned.tmp"
    destination = tmp_path / "exclusive-target.json"
    owned = b"owned-publication-bytes"
    foreign = b"foreign-destination-bytes"
    source.write_bytes(owned)
    expected_identity = _durable_file_identity(source)
    injected = False

    def create_competing_destination(
        event: str,
        _event_source: Path,
        event_destination: Path | None,
    ) -> None:
        nonlocal injected
        if event == "after_source_handle_bound":
            injected = True
            assert event_destination == destination
            destination.write_bytes(foreign)

    monkeypatch.setattr(
        path_security,
        "_namespace_event",
        create_competing_destination,
        raising=False,
    )

    with pytest.raises(FileExistsError):
        path_security.durable_publish_exclusive(
            source,
            destination,
            expected_source_identity=expected_identity,
        )

    assert injected is True
    assert source.read_bytes() == owned
    assert destination.read_bytes() == foreign


@pytest.mark.skipif(os.name != "nt", reason="Windows handle-bound namespace contract")
def test_windows_source_handle_blocks_leaf_replacement_until_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lunar_exploration_ppo.utils.path_security as path_security

    source = tmp_path / "handle-bound-source.tmp"
    destination = tmp_path / "handle-bound-destination.bin"
    displaced = tmp_path / "handle-bound-displaced.tmp"
    owned = b"handle-bound-owned-bytes"
    source.write_bytes(owned)
    expected_identity = _durable_file_identity(source)
    replacement_blocked = False

    def attempt_replacement(
        event: str,
        event_source: Path,
        _event_destination: Path | None,
    ) -> None:
        nonlocal replacement_blocked
        if event == "after_source_handle_bound":
            try:
                os.replace(event_source, displaced)
            except OSError:
                replacement_blocked = True

    monkeypatch.setattr(
        path_security,
        "_namespace_event",
        attempt_replacement,
        raising=False,
    )

    path_security.durable_publish_exclusive(
        source,
        destination,
        expected_source_identity=expected_identity,
    )

    assert replacement_blocked is True
    assert not source.exists()
    assert not displaced.exists()
    assert destination.read_bytes() == owned


@pytest.mark.skipif(os.name != "nt", reason="Windows hard-link fail-closed contract")
def test_windows_source_handle_rejects_hardlink_added_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lunar_exploration_ppo.utils.path_security as path_security

    source = tmp_path / "hardlink-bound-source.tmp"
    destination = tmp_path / "hardlink-bound-destination.bin"
    alias = tmp_path / "hardlink-racing-alias.bin"
    owned = b"hardlink-bound-owned-bytes"
    source.write_bytes(owned)
    expected_identity = _durable_file_identity(source)

    def attempt_hardlink(
        event: str,
        event_source: Path,
        _event_destination: Path | None,
    ) -> None:
        if event == "after_source_handle_bound":
            os.link(event_source, alias)

    monkeypatch.setattr(
        path_security,
        "_namespace_event",
        attempt_hardlink,
        raising=False,
    )

    with pytest.raises(path_security.PathSecurityError, match="hard link count"):
        path_security.durable_publish_exclusive(
            source,
            destination,
            expected_source_identity=expected_identity,
        )

    assert source.read_bytes() == owned
    assert alias.read_bytes() == owned
    assert source.stat().st_nlink == 2
    assert not destination.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX fail-closed namespace contract")
def test_posix_identity_bound_leaf_namespace_mutation_fails_before_side_effect(
    tmp_path: Path,
) -> None:
    import lunar_exploration_ppo.utils.path_security as path_security

    source = tmp_path / "posix-source.tmp"
    destination = tmp_path / "posix-destination.bin"
    owned = b"posix-owned-bytes"
    source.write_bytes(owned)

    with pytest.raises(path_security.PathSecurityError, match="handle-bound|unsupported"):
        path_security.durable_publish_exclusive(
            source,
            destination,
            expected_source_identity=_durable_file_identity(source),
        )

    assert source.read_bytes() == owned
    assert not destination.exists()


def test_append_keeps_true_canonical_jsonl_and_file_identity(tmp_path: Path) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import DurableJsonl

    path = tmp_path / "metrics.jsonl"
    journal = DurableJsonl(path)
    first = {"sequence": 1, "message": "\u6708\u7403"}
    second = {"sequence": 2, "nested": {"ok": True}}

    assert journal.recover() is False
    assert journal.append(first) == path
    identity = (path.stat().st_dev, path.stat().st_ino)
    first_bytes = path.read_bytes()

    assert journal.append(second) == path

    assert path.read_bytes() == first_bytes + _canonical_row(second)
    assert first_bytes == _canonical_row(first)
    assert (path.stat().st_dev, path.stat().st_ino) == identity
    assert not Path(f"{path}.pending").exists()


def test_committed_snapshot_rejects_canonical_target_replacement_after_recovery(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import (
        DurableJsonl,
        DurableJsonlError,
    )

    path = tmp_path / "snapshot-replaced.jsonl"
    displaced = tmp_path / "snapshot-displaced.jsonl"
    trusted = _canonical_row({"sequence": 1})
    attacker = _canonical_row({"sequence": 2})
    path.write_bytes(trusted)
    injected = False

    def replace_after_recovery(phase: str) -> None:
        nonlocal injected
        if phase == "after_recovery_before_snapshot":
            injected = True
            os.replace(path, displaced)
            path.write_bytes(attacker)

    journal = DurableJsonl(path, fault_injector=replace_after_recovery)
    with pytest.raises(DurableJsonlError, match="identity|drift|changed"):
        journal.recover_and_snapshot()

    assert injected is True
    assert displaced.read_bytes() == trusted
    assert path.read_bytes() == attacker


def test_committed_snapshot_binds_before_internal_recovery_returns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import (
        DurableJsonl,
        DurableJsonlError,
    )

    path = tmp_path / "snapshot-recovery-return-window.jsonl"
    displaced = tmp_path / "snapshot-recovery-return-displaced.jsonl"
    trusted = _canonical_row({"sequence": 1})
    attacker = _canonical_row({"sequence": 2})
    path.write_bytes(trusted)
    original_recovery = DurableJsonl._recover_with_parent_guard
    injected = False

    def recover_then_replace(self: DurableJsonl, *, parent_guard):
        nonlocal injected
        result = original_recovery(self, parent_guard=parent_guard)
        injected = True
        os.replace(path, displaced)
        path.write_bytes(attacker)
        return result

    monkeypatch.setattr(
        DurableJsonl,
        "_recover_with_parent_guard",
        recover_then_replace,
    )

    with pytest.raises(DurableJsonlError, match="content|version|identity|drift|changed"):
        DurableJsonl(path).recover_and_snapshot()

    assert injected is True
    assert displaced.read_bytes() == trusted
    assert path.read_bytes() == attacker


@pytest.mark.parametrize("attack", ("hardlink", "reparse", "parent"))
def test_committed_snapshot_fails_closed_for_link_and_parent_replacement(
    tmp_path: Path,
    attack: str,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import (
        DurableJsonl,
        DurableJsonlError,
    )

    run_dir = tmp_path / f"snapshot-{attack}"
    run_dir.mkdir()
    path = run_dir / "ledger.jsonl"
    path.write_bytes(_canonical_row({"sequence": 1}))
    displaced = tmp_path / f"snapshot-{attack}-displaced"
    external = tmp_path / f"snapshot-{attack}-external.jsonl"

    def replace_after_recovery(phase: str) -> None:
        if phase != "after_recovery_before_snapshot":
            return
        if attack == "hardlink":
            os.link(path, external)
        elif attack == "reparse":
            os.replace(path, external)
            reparse_target = tmp_path / "snapshot-reparse-target"
            reparse_target.mkdir()
            _make_directory_reparse(path, reparse_target)
        else:
            try:
                os.replace(run_dir, displaced)
                run_dir.mkdir()
                os.link(displaced / path.name, path)
            except OSError as exc:
                raise DurableJsonlError("parent replacement was blocked") from exc

    with pytest.raises(DurableJsonlError, match="link|reparse|parent|identity|drift"):
        DurableJsonl(path, fault_injector=replace_after_recovery).recover_and_snapshot()


def test_committed_snapshot_preserves_canonical_prefix_and_exact_appended_row(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import DurableJsonl

    path = tmp_path / "snapshot-exact.jsonl"
    journal = DurableJsonl(path)
    first = {"sequence": 1}
    second = {"sequence": 2, "payload": "exact"}

    journal.append(first)
    journal.append(second)

    assert journal.recover_and_snapshot() == _canonical_row(first) + _canonical_row(second)


def test_committed_snapshot_rejects_same_inode_same_size_rewrite(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import (
        DurableJsonl,
        DurableJsonlError,
    )

    path = tmp_path / "snapshot-same-inode.jsonl"
    trusted = _canonical_row({"sequence": 1})
    attacker = _canonical_row({"sequence": 2})
    assert len(attacker) == len(trusted)
    path.write_bytes(trusted)
    object_identity = (path.stat().st_dev, path.stat().st_ino)

    def rewrite_after_recovery(phase: str) -> None:
        if phase != "after_recovery_before_snapshot":
            return
        with path.open("r+b") as stream:
            stream.write(attacker)
            stream.flush()
            os.fsync(stream.fileno())

    with pytest.raises(DurableJsonlError, match="content|version|identity|drift|changed"):
        DurableJsonl(
            path,
            fault_injector=rewrite_after_recovery,
        ).recover_and_snapshot()

    assert (path.stat().st_dev, path.stat().st_ino) == object_identity
    assert path.read_bytes() == attacker


def test_committed_snapshot_returns_empty_for_continuously_missing_target(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import DurableJsonl

    path = tmp_path / "snapshot-missing.jsonl"
    journal = DurableJsonl(path)

    assert journal.recover_and_snapshot() == b""
    assert not path.exists()
    assert not Path(f"{path}.pending").exists()
    assert not Path(f"{path}.pending.tmp").exists()


@pytest.mark.parametrize("fault_phase", ("after_pending", "before_pending_replace"))
def test_committed_snapshot_recovers_pending_state_before_returning(
    tmp_path: Path,
    fault_phase: str,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import (
        DurableJsonl,
        DurableJsonlError,
    )

    path = tmp_path / f"snapshot-pending-{fault_phase}.jsonl"
    value = {"sequence": 1, "phase": fault_phase}
    failing = DurableJsonl(path, fault_injector=_fault_at(fault_phase))
    if fault_phase == "after_pending":
        with pytest.raises(_InjectedFault, match=fault_phase):
            failing.append(value)
    else:
        with pytest.raises(DurableJsonlError, match="pending.*identity"):
            failing.append(value)

    assert os.path.lexists(f"{path}.pending") or os.path.lexists(
        f"{path}.pending.tmp"
    )
    assert DurableJsonl(path).recover_and_snapshot() == _canonical_row(value)
    assert not os.path.lexists(f"{path}.pending")
    assert not os.path.lexists(f"{path}.pending.tmp")


def test_committed_snapshot_rejects_pending_created_after_recovery_binding(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import (
        DurableJsonl,
        DurableJsonlError,
    )

    path = tmp_path / "snapshot-new-pending.jsonl"
    path.write_bytes(_canonical_row({"sequence": 1}))
    pending = Path(f"{path}.pending")

    def create_pending(phase: str) -> None:
        if phase == "after_recovery_before_snapshot":
            pending.write_bytes(b"not-a-committed-pending-record\n")

    with pytest.raises(DurableJsonlError, match="pending|changed|drift"):
        DurableJsonl(path, fault_injector=create_pending).recover_and_snapshot()


@pytest.mark.parametrize(
    "payload",
    (
        b'{"z":2, "a":1}\n',
        b'{"a":1}',
    ),
)
def test_committed_snapshot_requires_complete_canonical_jsonl_boundary(
    tmp_path: Path,
    payload: bytes,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import (
        DurableJsonl,
        DurableJsonlError,
    )

    path = tmp_path / "snapshot-invalid-boundary.jsonl"
    path.write_bytes(payload)

    with pytest.raises(DurableJsonlError, match="canonical|newline|JSONL|tail"):
        DurableJsonl(path).recover_and_snapshot()


def test_after_pending_binds_prefix_and_row_then_recovers_once(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import DurableJsonl

    path = tmp_path / "job-state.jsonl"
    stable = DurableJsonl(path)
    stable.append({"sequence": 1})
    prefix = path.read_bytes()
    value = {"sequence": 2, "status": "prepared"}
    row = _canonical_row(value)
    failing = DurableJsonl(path, fault_injector=_fault_at("after_pending"))

    with pytest.raises(_InjectedFault, match="after_pending"):
        failing.append(value)

    assert path.read_bytes() == prefix
    pending_path = Path(f"{path}.pending")
    pending_bytes = pending_path.read_bytes()
    assert pending_bytes.endswith(b"\n")
    assert pending_bytes.count(b"\n") == 1
    pending = json.loads(pending_bytes)
    assert pending == {
        "prior_sha256": hashlib.sha256(prefix).hexdigest(),
        "prior_size": len(prefix),
        "row_base64": base64.b64encode(row).decode("ascii"),
        "row_sha256": hashlib.sha256(row).hexdigest(),
        "schema_version": "durable_jsonl_pending/v1",
        "target_path": os.path.normcase(os.path.abspath(path)),
    }

    recovering = DurableJsonl(path)
    assert recovering.recover() is True
    assert path.read_bytes() == prefix + row
    assert not pending_path.exists()
    assert recovering.recover() is False
    assert path.read_bytes() == prefix + row


def test_mid_append_leaves_real_torn_tail_and_recovery_replaces_only_tail(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import DurableJsonl

    path = tmp_path / "phase-state.jsonl"
    stable = DurableJsonl(path)
    stable.append({"sequence": 1})
    prefix = path.read_bytes()
    value = {"sequence": 2, "payload": "x" * 31}
    row = _canonical_row(value)
    failing = DurableJsonl(path, fault_injector=_fault_at("mid_append"))

    with pytest.raises(_InjectedFault, match="mid_append"):
        failing.append(value)

    torn_tail = path.read_bytes()[len(prefix) :]
    assert torn_tail
    assert torn_tail != row
    assert row.startswith(torn_tail)
    assert Path(f"{path}.pending").is_file()

    recovering = DurableJsonl(path)
    assert recovering.recover() is True
    assert path.read_bytes() == prefix + row
    assert recovering.recover() is False
    assert path.read_bytes() == prefix + row


@pytest.mark.parametrize(
    "fault_phase",
    (
        "after_append_write",
        "after_append_flush",
        "after_append_fsync",
        "before_pending_remove",
    ),
)
def test_full_row_with_pending_commits_exactly_once(
    tmp_path: Path,
    fault_phase: str,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import DurableJsonl

    path = tmp_path / f"receipt-{fault_phase}.jsonl"
    value = {"phase": fault_phase, "sequence": 1}
    row = _canonical_row(value)
    failing = DurableJsonl(path, fault_injector=_fault_at(fault_phase))

    with pytest.raises(_InjectedFault, match=fault_phase):
        failing.append(value)

    assert path.read_bytes() == row
    pending_path = Path(f"{path}.pending")
    assert pending_path.is_file()

    recovering = DurableJsonl(path)
    assert recovering.recover() is True
    assert path.read_bytes() == row
    assert not pending_path.exists()
    assert recovering.recover() is False
    assert path.read_bytes() == row


@pytest.mark.parametrize("fault_phase", ("after_append_write", "after_append_flush"))
def test_recover_fsyncs_existing_full_row_before_committing_pending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault_phase: str,
) -> None:
    import lunar_exploration_ppo.utils.durable_jsonl as durable_module

    path = tmp_path / f"needs-recovery-fsync-{fault_phase}.jsonl"
    value = {"phase": fault_phase, "sequence": 1}
    row = _canonical_row(value)
    failing = durable_module.DurableJsonl(
        path,
        fault_injector=_fault_at(fault_phase),
    )
    with pytest.raises(_InjectedFault):
        failing.append(value)
    identity = (path.stat().st_dev, path.stat().st_ino)
    real_fsync = durable_module.os.fsync
    target_fsyncs = 0

    def record_fsync(descriptor: int) -> None:
        nonlocal target_fsyncs
        metadata = os.fstat(descriptor)
        if stat.S_ISREG(metadata.st_mode) and (
            metadata.st_dev,
            metadata.st_ino,
        ) == identity:
            target_fsyncs += 1
        real_fsync(descriptor)

    monkeypatch.setattr(durable_module.os, "fsync", record_fsync)

    assert durable_module.DurableJsonl(path).recover() is True

    assert target_fsyncs >= 1
    assert path.read_bytes() == row
    assert (path.stat().st_dev, path.stat().st_ino) == identity
    assert not Path(f"{path}.pending").exists()


def test_recover_fails_closed_when_historical_prefix_drifted(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import (
        DurableJsonl,
        DurableJsonlError,
    )

    path = tmp_path / "resource.jsonl"
    stable = DurableJsonl(path)
    stable.append({"sequence": 1})
    failing = DurableJsonl(path, fault_injector=_fault_at("after_pending"))
    with pytest.raises(_InjectedFault):
        failing.append({"sequence": 2})

    original = path.read_bytes()
    path.write_bytes(bytes([original[0] ^ 0x01]) + original[1:])
    drifted = path.read_bytes()
    pending_path = Path(f"{path}.pending")
    pending_before = pending_path.read_bytes()

    with pytest.raises(DurableJsonlError, match="prefix"):
        DurableJsonl(path).recover()

    assert path.read_bytes() == drifted
    assert pending_path.read_bytes() == pending_before


def test_recover_fails_closed_on_unknown_torn_tail(tmp_path: Path) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import (
        DurableJsonl,
        DurableJsonlError,
    )

    path = tmp_path / "math.jsonl"
    stable = DurableJsonl(path)
    stable.append({"sequence": 1})
    prefix = path.read_bytes()
    failing = DurableJsonl(path, fault_injector=_fault_at("mid_append"))
    with pytest.raises(_InjectedFault):
        failing.append({"sequence": 2, "payload": "x" * 17})

    torn = path.read_bytes()[len(prefix) :]
    unknown = bytes([torn[0] ^ 0x01]) + torn[1:]
    path.write_bytes(prefix + unknown)
    before = path.read_bytes()

    with pytest.raises(DurableJsonlError, match="tail"):
        DurableJsonl(path).recover()

    assert path.read_bytes() == before
    assert Path(f"{path}.pending").is_file()


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows writable leaf sharing prevents the injected external write",
)
def test_recover_revalidates_torn_tail_immediately_before_repair(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import (
        DurableJsonl,
        DurableJsonlError,
    )

    path = tmp_path / "tail-repair-race.jsonl"
    stable = DurableJsonl(path)
    stable.append({"sequence": 1})
    prefix = path.read_bytes()
    value = {"sequence": 2, "payload": "x" * 17}
    failing = DurableJsonl(path, fault_injector=_fault_at("mid_append"))
    with pytest.raises(_InjectedFault):
        failing.append(value)
    torn = path.read_bytes()[len(prefix) :]
    unknown = bytes([torn[0] ^ 0x01]) + torn[1:]
    injected = False

    def replace_torn_tail_before_repair(phase: str) -> None:
        nonlocal injected
        if phase == "before_torn_tail_repair":
            injected = True
            with path.open("r+b") as stream:
                stream.seek(len(prefix))
                stream.write(unknown)
                stream.truncate(len(prefix) + len(unknown))
                stream.flush()
                os.fsync(stream.fileno())

    with pytest.raises(DurableJsonlError, match="tail|prefix|changed|drift"):
        DurableJsonl(
            path,
            fault_injector=replace_torn_tail_before_repair,
        ).recover()

    assert injected is True
    assert path.read_bytes() == prefix + unknown
    assert Path(f"{path}.pending").is_file()


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows parent handles prevent the rename before identity drift",
)
def test_recover_rejects_parent_identity_drift_before_commit(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import (
        DurableJsonl,
        DurableJsonlError,
    )

    run_dir = tmp_path / "recover-run"
    displaced_dir = tmp_path / "recover-run-displaced"
    run_dir.mkdir()
    path = run_dir / "recover-parent-race.jsonl"
    stable = DurableJsonl(path)
    stable.append({"sequence": 1})
    failing = DurableJsonl(path, fault_injector=_fault_at("mid_append"))
    with pytest.raises(_InjectedFault):
        failing.append({"sequence": 2, "payload": "x" * 17})
    torn = path.read_bytes()
    injected = False

    def replace_parent_but_keep_leaf_identity(phase: str) -> None:
        nonlocal injected
        if phase == "before_recovery_commit":
            injected = True
            os.replace(run_dir, displaced_dir)
            run_dir.mkdir()
            os.link(displaced_dir / path.name, path)

    with pytest.raises(DurableJsonlError, match="parent|identity|drift|changed"):
        DurableJsonl(
            path,
            fault_injector=replace_parent_but_keep_leaf_identity,
        ).recover()

    assert injected is True
    assert path.read_bytes() == torn
    assert (displaced_dir / path.name).read_bytes() == torn
    assert Path(f"{displaced_dir / path.name}.pending").is_file()


def test_recover_fails_closed_on_duplicate_expected_tail(tmp_path: Path) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import (
        DurableJsonl,
        DurableJsonlError,
    )

    path = tmp_path / "checkpoint.jsonl"
    value = {"sequence": 1, "status": "complete"}
    row = _canonical_row(value)
    failing = DurableJsonl(path, fault_injector=_fault_at("after_append_fsync"))
    with pytest.raises(_InjectedFault):
        failing.append(value)
    with path.open("ab") as stream:
        stream.write(row)
        stream.flush()
        os.fsync(stream.fileno())
    duplicate = path.read_bytes()

    with pytest.raises(DurableJsonlError, match="tail"):
        DurableJsonl(path).recover()

    assert path.read_bytes() == duplicate
    assert Path(f"{path}.pending").is_file()


@pytest.mark.parametrize(
    "case",
    (
        "schema",
        "target",
        "row_hash",
        "extra_field",
        "missing_field",
        "invalid_base64",
        "invalid_prior_size",
        "noncanonical_row",
    ),
)
def test_recover_rejects_invalid_pending_records(
    tmp_path: Path,
    case: str,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import (
        DurableJsonl,
        DurableJsonlError,
    )

    path = tmp_path / f"invalid-{case}.jsonl"
    failing = DurableJsonl(path, fault_injector=_fault_at("after_pending"))
    with pytest.raises(_InjectedFault):
        failing.append({"a": 1, "z": 2})
    pending_path = Path(f"{path}.pending")
    pending = json.loads(pending_path.read_bytes())
    if case == "schema":
        pending["schema_version"] = "durable_jsonl_pending/v0"
    elif case == "target":
        pending["target_path"] = os.path.normcase(
            os.path.abspath(tmp_path / "other.jsonl")
        )
    elif case == "row_hash":
        pending["row_sha256"] = "0" * 64
    elif case == "extra_field":
        pending["unexpected"] = True
    elif case == "missing_field":
        del pending["row_sha256"]
    elif case == "invalid_base64":
        pending["row_base64"] = "***"
    elif case == "invalid_prior_size":
        pending["prior_size"] = True
    elif case == "noncanonical_row":
        noncanonical = b'{"z":2, "a":1}\n'
        pending["row_base64"] = base64.b64encode(noncanonical).decode("ascii")
        pending["row_sha256"] = hashlib.sha256(noncanonical).hexdigest()
    pending_path.write_bytes(_canonical_row(pending))
    pending_before = pending_path.read_bytes()

    with pytest.raises(DurableJsonlError, match="pending"):
        DurableJsonl(path).recover()

    assert not path.exists()
    assert pending_path.read_bytes() == pending_before


def test_recover_rejects_malformed_pending_record(tmp_path: Path) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import (
        DurableJsonl,
        DurableJsonlError,
    )

    path = tmp_path / "malformed.jsonl"
    pending_path = Path(f"{path}.pending")
    pending_path.write_bytes(b"{not-json\n")

    with pytest.raises(DurableJsonlError, match="pending"):
        DurableJsonl(path).recover()

    assert pending_path.read_bytes() == b"{not-json\n"
    assert not path.exists()


def test_unbound_torn_canonical_fails_closed_for_recover_and_append(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import (
        DurableJsonl,
        DurableJsonlError,
    )

    path = tmp_path / "unbound-torn.jsonl"
    torn = b'{"sequence":1'
    path.write_bytes(torn)
    journal = DurableJsonl(path)

    with pytest.raises(DurableJsonlError, match="newline|tail"):
        journal.recover()
    with pytest.raises(DurableJsonlError, match="newline|tail"):
        journal.append({"sequence": 2})

    assert path.read_bytes() == torn
    assert not Path(f"{path}.pending").exists()


@pytest.mark.parametrize("operation", ("recover", "append"))
def test_invalid_complete_jsonl_history_fails_closed(
    tmp_path: Path,
    operation: str,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import (
        DurableJsonl,
        DurableJsonlError,
    )

    path = tmp_path / f"invalid-history-{operation}.jsonl"
    invalid = b"not-json\n"
    path.write_bytes(invalid)
    journal = DurableJsonl(path)

    with pytest.raises(DurableJsonlError, match="JSONL|JSON"):
        if operation == "recover":
            journal.recover()
        else:
            journal.append({"sequence": 2})

    assert path.read_bytes() == invalid
    assert not Path(f"{path}.pending").exists()


def test_valid_but_noncanonical_jsonl_history_fails_closed(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import (
        DurableJsonl,
        DurableJsonlError,
    )

    path = tmp_path / "noncanonical-history.jsonl"
    noncanonical = b'{"z":2, "a":1}\n'
    path.write_bytes(noncanonical)

    with pytest.raises(DurableJsonlError, match="canonical|JSONL"):
        DurableJsonl(path).append({"sequence": 2})

    assert path.read_bytes() == noncanonical
    assert not Path(f"{path}.pending").exists()


def _pending_payload(path: Path, value: object) -> bytes:
    row = _canonical_row(value)
    return _canonical_row(
        {
            "prior_sha256": hashlib.sha256(b"").hexdigest(),
            "prior_size": 0,
            "row_base64": base64.b64encode(row).decode("ascii"),
            "row_sha256": hashlib.sha256(row).hexdigest(),
            "schema_version": "durable_jsonl_pending/v1",
            "target_path": os.path.normcase(os.path.abspath(path)),
        }
    )


def test_recover_publishes_fsynced_pending_temp_then_commits_once(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import DurableJsonl

    path = tmp_path / "pending-temp-valid.jsonl"
    value = {"sequence": 1, "status": "prepared"}
    pending_temp = Path(f"{path}.pending.tmp")
    pending_temp.write_bytes(_pending_payload(path, value))

    journal = DurableJsonl(path)
    assert journal.recover() is True

    assert path.read_bytes() == _canonical_row(value)
    assert not pending_temp.exists()
    assert not Path(f"{path}.pending").exists()
    assert journal.recover() is False


def test_recover_discards_unpublished_torn_pending_temp_only(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import DurableJsonl

    path = tmp_path / "pending-temp-torn.jsonl"
    pending_temp = Path(f"{path}.pending.tmp")
    pending_temp.write_bytes(b'{"prior_sha256":')
    journal = DurableJsonl(path)

    assert journal.recover() is False
    assert not path.exists()
    assert not pending_temp.exists()

    journal.append({"sequence": 1})
    assert path.read_bytes() == _canonical_row({"sequence": 1})


def test_append_rejects_pending_temp_replacement_before_publish(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import (
        DurableJsonl,
        DurableJsonlError,
    )

    path = tmp_path / "pending-replace-race.jsonl"
    original = _canonical_row({"sequence": 1})
    path.write_bytes(original)
    pending = Path(f"{path}.pending")
    pending_temp = Path(f"{pending}.tmp")
    displaced_temp = tmp_path / "pending-replace-race.displaced"
    attacker = b'{"attacker":true}\n'
    injected = False

    def replace_pending_temp(phase: str) -> None:
        nonlocal injected
        if phase == "before_pending_replace":
            injected = True
            os.replace(pending_temp, displaced_temp)
            pending_temp.write_bytes(attacker)

    with pytest.raises(DurableJsonlError, match="pending|identity|changed|drift"):
        DurableJsonl(path, fault_injector=replace_pending_temp).append(
            {"sequence": 2}
        )

    assert injected is True
    assert path.read_bytes() == original
    assert not pending.exists()
    assert pending_temp.read_bytes() == attacker
    assert displaced_temp.is_file()


@pytest.mark.parametrize("operation", ("append", "recover"))
def test_durable_jsonl_rejects_reparse_path_components(
    tmp_path: Path,
    operation: str,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import (
        DurableJsonl,
        DurableJsonlError,
    )

    outside = tmp_path / f"outside-{operation}"
    outside.mkdir()
    canonical = outside / "metrics.jsonl"
    canonical.write_bytes(_canonical_row({"stable": True}))
    linked = tmp_path / f"linked-{operation}"
    _make_directory_reparse(linked, outside)
    journal = DurableJsonl(linked / "metrics.jsonl")

    with pytest.raises(DurableJsonlError, match="link|reparse"):
        if operation == "append":
            journal.append({"escaped": True})
        else:
            journal.recover()

    assert canonical.read_bytes() == _canonical_row({"stable": True})
    assert not Path(f"{canonical}.pending").exists()


def test_append_rejects_target_leaf_replacement_after_pending(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import (
        DurableJsonl,
        DurableJsonlError,
    )

    path = tmp_path / "leaf-race.jsonl"
    displaced = tmp_path / "leaf-race.displaced"
    original = _canonical_row({"sequence": 1})
    replacement = _canonical_row({"replacement": True})
    path.write_bytes(original)

    def replace_leaf(phase: str) -> None:
        if phase == "after_pending":
            os.replace(path, displaced)
            path.write_bytes(replacement)

    with pytest.raises(DurableJsonlError, match="identity|drift|changed"):
        DurableJsonl(path, fault_injector=replace_leaf).append({"sequence": 2})

    assert path.read_bytes() == replacement
    assert displaced.read_bytes() == original
    assert Path(f"{path}.pending").is_file()


@pytest.mark.skipif(
    os.name == "nt",
    reason="Windows parent handles prevent the rename before identity drift",
)
def test_append_rejects_parent_identity_drift_with_same_leaf_inode(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import (
        DurableJsonl,
        DurableJsonlError,
    )

    run_dir = tmp_path / "run"
    displaced_dir = tmp_path / "run-displaced"
    run_dir.mkdir()
    path = run_dir / "parent-race.jsonl"
    original = _canonical_row({"sequence": 1})
    path.write_bytes(original)

    def replace_parent_but_keep_leaf_identity(phase: str) -> None:
        if phase == "after_pending":
            os.replace(run_dir, displaced_dir)
            run_dir.mkdir()
            os.link(displaced_dir / path.name, path)

    with pytest.raises(DurableJsonlError, match="parent|identity|drift|changed"):
        DurableJsonl(
            path,
            fault_injector=replace_parent_but_keep_leaf_identity,
        ).append({"sequence": 2})

    assert path.read_bytes() == original
    assert (displaced_dir / path.name).read_bytes() == original
    assert Path(f"{displaced_dir / path.name}.pending").is_file()


def test_append_rejects_hard_link_without_mutating_external_inode(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import (
        DurableJsonl,
        DurableJsonlError,
    )

    trusted = tmp_path / "trusted-append"
    trusted.mkdir()
    outside = tmp_path / "outside-append.jsonl"
    target = trusted / "metrics.jsonl"
    original = _canonical_row({"sequence": 1})
    outside.write_bytes(original)
    os.link(outside, target)
    before = outside.read_bytes()

    with pytest.raises(DurableJsonlError, match="hard link|link count|identity"):
        DurableJsonl(target).append({"sequence": 2})

    assert outside.read_bytes() == before
    assert target.read_bytes() == before
    assert not Path(f"{target}.pending").exists()


def test_torn_tail_repair_rejects_hard_link_without_mutating_external_inode(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import (
        DurableJsonl,
        DurableJsonlError,
    )

    trusted = tmp_path / "trusted-repair"
    trusted.mkdir()
    target = trusted / "metrics.jsonl"
    failing = DurableJsonl(target, fault_injector=_fault_at("mid_append"))
    with pytest.raises(_InjectedFault, match="mid_append"):
        failing.append({"sequence": 1})
    outside = tmp_path / "outside-repair.jsonl"
    os.link(target, outside)
    before = outside.read_bytes()

    with pytest.raises(DurableJsonlError, match="hard link|link count|identity"):
        DurableJsonl(target).recover()

    assert outside.read_bytes() == before
    assert target.read_bytes() == before
    assert Path(f"{target}.pending").is_file()


def test_run_lease_rejects_hard_link_without_mutating_external_inode(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import RunLease, RunLeaseError

    trusted = tmp_path / "trusted-lease"
    trusted.mkdir()
    outside = tmp_path / "outside.lease"
    lease_path = trusted / "run.lease"
    original = b"external-lease-bytes\n"
    outside.write_bytes(original)
    os.link(outside, lease_path)

    with pytest.raises(RunLeaseError, match="hard link|link count|identity"):
        RunLease(lease_path).acquire()

    assert outside.read_bytes() == original
    assert lease_path.read_bytes() == original


@pytest.mark.skipif(os.name == "nt", reason="POSIX dir-fd anchoring contract")
def test_posix_append_anchors_leaf_open_before_parent_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lunar_exploration_ppo.utils.durable_jsonl as durable_module

    run_dir = tmp_path / "posix-anchor-run"
    displaced = tmp_path / "posix-anchor-displaced"
    run_dir.mkdir()
    target = run_dir / "metrics.jsonl"
    real_open = durable_module.os.open
    swapped = False

    def swap_parent_before_target_open(
        path: str | os.PathLike[str],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if (
            not swapped
            and Path(path).name == target.name
            and flags & os.O_WRONLY
            and flags & os.O_APPEND
        ):
            os.replace(run_dir, displaced)
            run_dir.mkdir()
            swapped = True
        return real_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(durable_module.os, "open", swap_parent_before_target_open)

    with pytest.raises(
        durable_module.DurableJsonlError,
        match="parent|identity|drift|changed",
    ):
        durable_module.DurableJsonl(target).append({"sequence": 1})

    assert swapped is True
    assert not target.exists()
    assert (displaced / target.name).is_file()
    assert Path(f"{displaced / target.name}.pending").is_file()


@pytest.mark.skipif(os.name != "nt", reason="Windows directory sharing contract")
def test_windows_parent_guard_blocks_real_ancestor_rename_during_append(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import DurableJsonl

    guarded_root = tmp_path / "windows-guarded-root"
    run_dir = guarded_root / "run"
    displaced = tmp_path / "windows-guarded-root-displaced"
    run_dir.mkdir(parents=True)
    target = run_dir / "metrics.jsonl"
    blocked: list[OSError] = []

    def try_real_parent_rename(phase: str) -> None:
        if phase != "after_pending":
            return
        try:
            os.replace(guarded_root, displaced)
        except OSError as exc:
            blocked.append(exc)

    DurableJsonl(target, fault_injector=try_real_parent_rename).append(
        {"sequence": 1}
    )

    assert blocked
    assert getattr(blocked[0], "winerror", None) in {5, 32}
    assert guarded_root.is_dir()
    assert not displaced.exists()
    assert target.read_bytes() == _canonical_row({"sequence": 1})


def test_run_lease_rejects_second_handle_then_releases_without_unlink(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import RunLease, RunLeaseError

    path = tmp_path / "run.lease"
    first = RunLease(path)
    second = RunLease(path)

    with first:
        assert path.is_file()
        with pytest.raises(RunLeaseError, match="already held|lease"):
            with second:
                pass

    assert path.is_file()
    with RunLease(path):
        assert path.is_file()
    assert path.is_file()


def test_run_lease_require_current_rejects_never_acquired_and_released_handles(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import RunLease, RunLeaseError

    lease = RunLease(tmp_path / "current.lease")
    with pytest.raises(RunLeaseError, match="current|held|acquired"):
        lease.require_current()

    with lease:
        lease.require_current()

    with pytest.raises(RunLeaseError, match="current|held|acquired|released"):
        lease.require_current()


def test_run_lease_require_current_rejects_descriptor_path_identity_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lunar_exploration_ppo.utils.durable_jsonl as durable_module

    lease = durable_module.RunLease(tmp_path / "identity.lease")
    with lease:
        lease.require_current()

        def reject_identity(*args: object, **kwargs: object) -> object:
            del args, kwargs
            raise durable_module._PathSafetyError("injected identity drift")

        monkeypatch.setattr(
            durable_module,
            "_require_descriptor_path_identity",
            reject_identity,
        )
        with pytest.raises(durable_module.RunLeaseError, match="identity|current"):
            lease.require_current()


def test_run_lease_can_acquire_preexisting_stale_file(tmp_path: Path) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import RunLease

    path = tmp_path / "stale.lease"
    stale = b"stale-file-remains\n"
    path.write_bytes(stale)

    with RunLease(path):
        assert path.is_file()

    assert path.read_bytes() == stale


def test_run_lease_rejects_competing_process_without_blocking(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import RunLease

    path = tmp_path / "process.lease"
    contender = """
import sys
from lunar_exploration_ppo.utils.durable_jsonl import RunLease, RunLeaseError
try:
    with RunLease(sys.argv[1]):
        raise SystemExit(41)
except RunLeaseError:
    raise SystemExit(0)
"""
    with RunLease(path):
        completed = subprocess.run(
            [sys.executable, "-c", contender, str(path)],
            capture_output=True,
            check=False,
            env=_subprocess_env(),
            text=True,
            timeout=5,
        )

    assert completed.returncode == 0, completed.stderr


def test_run_lease_kernel_guard_is_path_keyed_and_released(
    tmp_path: Path,
) -> None:
    import lunar_exploration_ppo.utils.durable_jsonl as durable_module

    path = tmp_path / "kernel-guard.lease"
    first = durable_module._acquire_run_lease_kernel_guard(path)
    try:
        with pytest.raises(durable_module.RunLeaseError, match="held|guard|lease"):
            durable_module._acquire_run_lease_kernel_guard(path)
    finally:
        first.close()

    replacement = durable_module._acquire_run_lease_kernel_guard(path)
    replacement.close()


def test_run_lease_retains_kernel_guard_when_release_must_be_retried(
    tmp_path: Path,
) -> None:
    import lunar_exploration_ppo.utils.durable_jsonl as durable_module

    class _RetryableGuard:
        def __init__(self) -> None:
            self.calls = 0

        def close(self) -> None:
            self.calls += 1
            if self.calls == 1:
                raise durable_module.RunLeaseError("injected guard close failure")

    lease = durable_module.RunLease(tmp_path / "retry-release.lease")
    guard = _RetryableGuard()
    lease._kernel_guard = guard

    with pytest.raises(durable_module.RunLeaseError, match="release"):
        lease.release()

    assert lease._kernel_guard is guard
    lease.release()
    assert guard.calls == 2
    assert lease._kernel_guard is None


def test_run_lease_retains_guard_when_failed_acquire_cleanup_must_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import lunar_exploration_ppo.utils.durable_jsonl as durable_module

    class _RetryableGuard:
        def __init__(self) -> None:
            self.calls = 0

        def close(self) -> None:
            self.calls += 1
            if self.calls == 1:
                raise durable_module.RunLeaseError("injected guard close failure")

    guard = _RetryableGuard()
    monkeypatch.setattr(
        durable_module,
        "_acquire_run_lease_kernel_guard",
        lambda _path: guard,
    )
    lease = durable_module.RunLease(tmp_path / "failed-acquire.lease")
    safety_checks = 0

    def fail_after_guard_acquisition() -> None:
        nonlocal safety_checks
        safety_checks += 1
        if safety_checks == 2:
            raise durable_module.RunLeaseError("injected path identity race")

    lease._require_safe_path = fail_after_guard_acquisition

    with pytest.raises(durable_module.RunLeaseError, match="release|close"):
        lease.acquire()

    assert lease._descriptor is None
    assert lease._kernel_guard is guard
    lease.release()
    assert guard.calls == 2
    assert lease._kernel_guard is None


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux abstract AF_UNIX guard closes the inode replacement gap",
)
def test_run_lease_rejects_linux_rename_recreate_split_brain(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import RunLease, RunLeaseError

    path = tmp_path / "rename-recreate.lease"
    displaced = tmp_path / "displaced.lease"
    first = RunLease(path)
    second = RunLease(path)
    first.acquire()
    try:
        os.replace(path, displaced)
        path.write_bytes(b"replacement inode\n")
        with pytest.raises(RunLeaseError, match="held|guard|lease"):
            second.acquire()
    finally:
        second.release()
        first.release()


def test_run_lease_is_released_by_process_exit(tmp_path: Path) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import RunLease

    path = tmp_path / "process-exit.lease"
    owner = """
import os
import sys
from lunar_exploration_ppo.utils.durable_jsonl import RunLease
lease = RunLease(sys.argv[1])
lease.acquire()
os._exit(0)
"""
    completed = subprocess.run(
        [sys.executable, "-c", owner, str(path)],
        capture_output=True,
        check=False,
        env=_subprocess_env(),
        text=True,
        timeout=5,
    )

    assert completed.returncode == 0, completed.stderr
    assert path.is_file()
    with RunLease(path):
        assert path.is_file()


def test_run_lease_rejects_reparse_path_component(tmp_path: Path) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import RunLease, RunLeaseError

    outside = tmp_path / "outside-lease"
    outside.mkdir()
    linked = tmp_path / "linked-lease"
    _make_directory_reparse(linked, outside)

    with pytest.raises(RunLeaseError, match="link|reparse"):
        with RunLease(linked / "run.lease"):
            pass

    assert not (outside / "run.lease").exists()
