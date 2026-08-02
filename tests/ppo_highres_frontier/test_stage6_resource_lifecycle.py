from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest


PRETERMINAL_ACCEPTANCE = {
    "sha256": "a" * 64,
    "size_bytes": 321,
}


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


def _resource(
    *,
    root_pid: int,
    sample_count: int,
    rss_bytes: int,
    passed: bool = True,
) -> dict[str, object]:
    return {
        "d_free_bytes": 200 * 1024**3,
        "rss_bytes": rss_bytes,
        "peak_vram_bytes": 2048,
        "rss_source": "process_tree_lifecycle_peak_current_sum/v1",
        "rss_root_pid": root_pid,
        "rss_sample_count": sample_count,
        "rss_latest_process_count": 1,
        "rss_peak_process_count": 1,
        "warnings": [],
        "hard_stops": [] if passed else ["injected hard stop"],
        "passed": passed,
    }


def _append_attempt(
    path: Path,
    segment: dict[str, object],
    *,
    transaction_key: str,
    first_sample_count: int,
    pre_peak: int,
    post_peak: int,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    from lunar_exploration_ppo.utils.durable_jsonl import DurableJsonl
    from lunar_exploration_ppo.utils.resource_lifecycle import (
        bind_resource_row_to_segment,
    )

    root_pid = int(segment["root_pid"])
    identity = {
        "kind": "update",
        "transaction_key": transaction_key,
        "seed": 20260716,
        "update": int(transaction_key.rsplit(":", 1)[-1]),
    }
    pre_resource = _resource(
        root_pid=root_pid,
        sample_count=first_sample_count,
        rss_bytes=pre_peak,
    )
    post_resource = _resource(
        root_pid=root_pid,
        sample_count=first_sample_count + 1,
        rss_bytes=post_peak,
    )
    pre = bind_resource_row_to_segment(
        {
            "schema_version": "stage6_resource_attempt/v1",
            **identity,
            "attempt": 1,
            "phase": "pre",
            "accepted": False,
            "resource": pre_resource,
        },
        segment,
    )
    post = bind_resource_row_to_segment(
        {
            "schema_version": "stage6_resource_attempt/v1",
            **identity,
            "attempt": 1,
            "phase": "post",
            "accepted": False,
            "resource": post_resource,
        },
        segment,
    )
    accepted = bind_resource_row_to_segment(
        {
            "schema_version": "stage6_resource_acceptance/v2",
            **identity,
            "attempt": 1,
            "phase": "accepted",
            "accepted": True,
            "pre": pre_resource,
            "post": post_resource,
            "checkpoint": {
                "transaction_key": transaction_key,
                "seed": 20260716,
                "update": identity["update"],
            },
        },
        segment,
    )
    journal = DurableJsonl(path)
    journal.append(pre)
    journal.append(post)
    journal.append(accepted)
    return pre, post, accepted


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _append_bound_terminal(
    path: Path,
    *,
    terminal_resource: dict[str, object],
) -> dict[str, object]:
    from lunar_exploration_ppo.utils.resource_lifecycle import (
        append_bound_resource_lifecycle_terminal,
    )

    return append_bound_resource_lifecycle_terminal(
        path,
        terminal_resource=terminal_resource,
        preterminal_acceptance=PRETERMINAL_ACCEPTANCE,
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


_SUBPROCESS_PROGRAM = textwrap.dedent(
    r"""
    import json
    import os
    import sys
    from pathlib import Path

    from lunar_exploration_ppo.utils.durable_jsonl import DurableJsonl
    from lunar_exploration_ppo.utils.resource_lifecycle import (
        append_bound_resource_lifecycle_terminal,
        append_resource_segment_start,
        bind_resource_row_to_segment,
        validate_resource_lifecycle_ledger,
    )

    path = Path(sys.argv[1])
    mode = sys.argv[2]

    def resource(sample_count, rss_bytes):
        return {
            "d_free_bytes": 200 * 1024**3,
            "rss_bytes": rss_bytes,
            "peak_vram_bytes": 2048,
            "rss_source": "process_tree_lifecycle_peak_current_sum/v1",
            "rss_root_pid": os.getpid(),
            "rss_sample_count": sample_count,
            "rss_latest_process_count": 1,
            "rss_peak_process_count": 1,
            "warnings": [],
            "hard_stops": [],
            "passed": True,
        }

    if mode == "validate":
        result = validate_resource_lifecycle_ledger(path, require_terminal=True)
        print(json.dumps(result, sort_keys=True))
        raise SystemExit(0)

    first_peak, pre_peak, post_peak = (
        (100, 300, 700) if mode == "first" else (50, 200, 300)
    )
    segment = append_resource_segment_start(
        path,
        first_sample=resource(1, first_peak),
    )
    transaction_key = f"update:{segment['segment_index']}"
    identity = {
        "schema_version": "stage6_resource_attempt/v1",
        "kind": "update",
        "transaction_key": transaction_key,
        "seed": 20260716,
        "update": segment["segment_index"],
        "attempt": 1,
        "accepted": False,
    }
    pre_resource = resource(2, pre_peak)
    post_resource = resource(3, post_peak)
    pre = bind_resource_row_to_segment(
        {**identity, "phase": "pre", "resource": pre_resource},
        segment,
    )
    post = bind_resource_row_to_segment(
        {**identity, "phase": "post", "resource": post_resource},
        segment,
    )
    accepted = bind_resource_row_to_segment(
        {
            "schema_version": "stage6_resource_acceptance/v2",
            "kind": "update",
            "transaction_key": transaction_key,
            "seed": 20260716,
            "update": segment["segment_index"],
            "attempt": 1,
            "phase": "accepted",
            "accepted": True,
            "pre": pre_resource,
            "post": post_resource,
            "checkpoint": {
                "transaction_key": transaction_key,
                "seed": 20260716,
                "update": segment["segment_index"],
            },
        },
        segment,
    )
    journal = DurableJsonl(path)
    journal.append(pre)
    journal.append(post)
    journal.append(accepted)
    if mode == "terminal":
        append_bound_resource_lifecycle_terminal(
            path,
            terminal_resource=resource(4, 350),
            preterminal_acceptance={"sha256": "a" * 64, "size_bytes": 321},
        )
    print(
        json.dumps(
            {
                "pid": os.getpid(),
                "segment_id": segment["segment_id"],
                "segment_index": segment["segment_index"],
            },
            sort_keys=True,
        )
    )
    """
)


def _run_python(
    path: Path,
    mode: str,
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(
        [sys.executable, "-c", _SUBPROCESS_PROGRAM, str(path), mode],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_subprocess_env(),
        shell=False,
    )
    try:
        stdout, stderr = process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=5)
        pytest.fail(
            f"resource lifecycle subprocess timed out: {mode}\n{stdout}\n{stderr}"
        )
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
    completed = subprocess.CompletedProcess(
        process.args,
        int(process.returncode),
        stdout,
        stderr,
    )
    if check and completed.returncode != 0:
        pytest.fail(
            "resource lifecycle subprocess failed: "
            f"{mode}\nstdout={stdout}\nstderr={stderr}"
        )
    return completed


def test_segment_start_binds_full_prefix_and_local_counter_can_reset(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.resource_lifecycle import (
        append_resource_segment_start,
        validate_resource_lifecycle_ledger,
    )

    path = tmp_path / "resource_audit.jsonl"
    first = append_resource_segment_start(
        path,
        first_sample=_resource(root_pid=os.getpid(), sample_count=1, rss_bytes=100),
    )
    assert first["schema_version"] == "stage6_resource_lifecycle_segment/v1"
    assert first["segment_index"] == 1
    assert first["root_pid"] == os.getpid()
    assert first["prior_resource_log"] == {
        "sha256": hashlib.sha256(b"").hexdigest(),
        "size_bytes": 0,
    }
    _append_attempt(
        path,
        first,
        transaction_key="update:1",
        first_sample_count=2,
        pre_peak=200,
        post_peak=500,
    )
    prior = path.read_bytes()

    second = append_resource_segment_start(
        path,
        first_sample=_resource(root_pid=os.getpid(), sample_count=1, rss_bytes=50),
    )

    assert second["segment_index"] == 2
    assert second["segment_id"] != first["segment_id"]
    assert second["prior_resource_log"] == {
        "sha256": hashlib.sha256(prior).hexdigest(),
        "size_bytes": len(prior),
    }
    assert path.read_bytes() == prior + _canonical_row(second)
    validated = validate_resource_lifecycle_ledger(path, require_terminal=False)
    assert validated["segment_count"] == 2
    assert validated["segment_chain"][0]["local_final_sample_count"] == 3
    assert validated["segment_chain"][1]["local_first_sample_count"] == 1
    assert validated["run_rss_lifecycle_peak_bytes"] == 500


def test_terminal_base_stays_receiptless_and_bound_builder_adds_only_receipt(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.resource_lifecycle import (
        append_resource_segment_start,
        build_bound_terminal_resource_evidence,
        build_terminal_resource_evidence,
        validate_resource_lifecycle_rows,
    )

    path = tmp_path / "resource_audit.jsonl"
    append_resource_segment_start(
        path,
        first_sample=_resource(root_pid=os.getpid(), sample_count=1, rss_bytes=100),
    )
    prefix_rows = _rows(path)
    terminal_resource = _resource(
        root_pid=os.getpid(), sample_count=2, rss_bytes=150
    )

    base = build_terminal_resource_evidence(
        prefix_rows,
        terminal_resource=terminal_resource,
    )
    bound = build_bound_terminal_resource_evidence(
        prefix_rows,
        terminal_resource=terminal_resource,
        preterminal_acceptance=PRETERMINAL_ACCEPTANCE,
    )

    assert "preterminal_acceptance" not in base
    assert bound == {
        **base,
        "preterminal_acceptance": PRETERMINAL_ACCEPTANCE,
    }
    validated = validate_resource_lifecycle_rows(
        [*prefix_rows, bound],
        require_terminal=True,
    )
    assert validated["terminal_evidence_present"] is True


@pytest.mark.parametrize(
    "schema_version",
    (
        "stage6_terminal_resource_evidence/v2",
        "stage6_terminal_resource_evidence/v3",
        "stage6_terminal_resource_evidence/v4",
    ),
)
def test_validator_rejects_old_or_receiptless_terminal(
    tmp_path: Path,
    schema_version: str,
) -> None:
    from lunar_exploration_ppo.utils.resource_lifecycle import (
        ResourceLifecycleError,
        append_resource_segment_start,
        build_terminal_resource_evidence,
        validate_resource_lifecycle_rows,
    )

    path = tmp_path / "resource_audit.jsonl"
    append_resource_segment_start(
        path,
        first_sample=_resource(root_pid=os.getpid(), sample_count=1, rss_bytes=100),
    )
    rows = _rows(path)
    terminal = build_terminal_resource_evidence(
        rows,
        terminal_resource=_resource(
            root_pid=os.getpid(), sample_count=2, rss_bytes=150
        ),
    )
    terminal["schema_version"] = schema_version

    with pytest.raises(ResourceLifecycleError, match="receipt|preterminal|terminal v4"):
        validate_resource_lifecycle_rows([*rows, terminal], require_terminal=True)


@pytest.mark.parametrize(
    "receipt",
    (
        {"sha256": "a" * 64},
        {"sha256": "a" * 64, "size_bytes": 1, "extra": True},
        {"sha256": "A" * 64, "size_bytes": 1},
        {"sha256": "a" * 64, "size_bytes": True},
        {"sha256": "a" * 64, "size_bytes": -1},
    ),
)
def test_bound_terminal_builder_rejects_non_exact_receipt_identity(
    tmp_path: Path,
    receipt: dict[str, object],
) -> None:
    from lunar_exploration_ppo.utils.resource_lifecycle import (
        ResourceLifecycleError,
        append_resource_segment_start,
        build_bound_terminal_resource_evidence,
    )

    path = tmp_path / "resource_audit.jsonl"
    append_resource_segment_start(
        path,
        first_sample=_resource(root_pid=os.getpid(), sample_count=1, rss_bytes=100),
    )

    with pytest.raises(ResourceLifecycleError, match="preterminal|receipt|binding"):
        build_bound_terminal_resource_evidence(
            _rows(path),
            terminal_resource=_resource(
                root_pid=os.getpid(), sample_count=2, rss_bytes=150
            ),
            preterminal_acceptance=receipt,
        )


def test_terminal_binds_ordered_chain_and_run_peak_is_max_not_sum(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.resource_lifecycle import (
        append_resource_segment_start,
        validate_resource_lifecycle_ledger,
    )

    path = tmp_path / "resource_audit.jsonl"
    first = append_resource_segment_start(
        path,
        first_sample=_resource(root_pid=os.getpid(), sample_count=1, rss_bytes=100),
    )
    _append_attempt(
        path,
        first,
        transaction_key="update:1",
        first_sample_count=2,
        pre_peak=200,
        post_peak=500,
    )
    second = append_resource_segment_start(
        path,
        first_sample=_resource(root_pid=os.getpid(), sample_count=1, rss_bytes=50),
    )
    _append_attempt(
        path,
        second,
        transaction_key="update:2",
        first_sample_count=2,
        pre_peak=200,
        post_peak=300,
    )
    prior = path.read_bytes()

    terminal = _append_bound_terminal(
        path,
        terminal_resource=_resource(
            root_pid=os.getpid(),
            sample_count=4,
            rss_bytes=350,
        ),
    )

    assert terminal["schema_version"] == "stage6_terminal_resource_evidence/v4"
    assert terminal["preterminal_acceptance"] == PRETERMINAL_ACCEPTANCE
    assert terminal["segment_id"] == second["segment_id"]
    assert terminal["segment_index"] == 2
    assert terminal["prior_resource_audit"] == {
        "sha256": hashlib.sha256(prior).hexdigest(),
        "size_bytes": len(prior),
    }
    assert terminal["run_rss_lifecycle_peak_bytes"] == 500
    assert terminal["segment_chain"] == [
        {
            "segment_id": first["segment_id"],
            "segment_index": 1,
            "root_pid": os.getpid(),
            "local_first_sample_count": 1,
            "local_final_sample_count": 3,
            "local_peak_rss_bytes": 500,
            "start_row_identity": {
                "offset_bytes": 0,
                "sha256": hashlib.sha256(_canonical_row(first)).hexdigest(),
                "size_bytes": len(_canonical_row(first)),
            },
        },
        {
            "segment_id": second["segment_id"],
            "segment_index": 2,
            "root_pid": os.getpid(),
            "local_first_sample_count": 1,
            "local_final_sample_count": 4,
            "local_peak_rss_bytes": 350,
            "start_row_identity": {
                "offset_bytes": int(second["prior_resource_log"]["size_bytes"]),
                "sha256": hashlib.sha256(_canonical_row(second)).hexdigest(),
                "size_bytes": len(_canonical_row(second)),
            },
        },
    ]
    validated = validate_resource_lifecycle_ledger(path, require_terminal=True)
    assert validated["terminal_evidence_present"] is True
    assert validated["segment_chain"] == terminal["segment_chain"]
    assert validated["run_rss_lifecycle_peak_bytes"] == 500


def test_resume_can_append_delayed_acceptance_bound_to_prior_segment(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.durable_jsonl import DurableJsonl
    from lunar_exploration_ppo.utils.resource_lifecycle import (
        append_resource_segment_start,
        bind_resource_row_to_segment,
        validate_resource_lifecycle_ledger,
    )

    path = tmp_path / "resource_audit.jsonl"
    first = append_resource_segment_start(
        path,
        first_sample=_resource(root_pid=os.getpid(), sample_count=1, rss_bytes=100),
    )
    identity = {
        "kind": "update",
        "transaction_key": "update:1",
        "seed": 20260716,
        "update": 1,
        "attempt": 1,
    }
    pre_resource = _resource(
        root_pid=os.getpid(), sample_count=2, rss_bytes=200
    )
    post_resource = _resource(
        root_pid=os.getpid(), sample_count=3, rss_bytes=300
    )
    pre = bind_resource_row_to_segment(
        {
            "schema_version": "stage6_resource_attempt/v1",
            **identity,
            "phase": "pre",
            "accepted": False,
            "resource": pre_resource,
        },
        first,
    )
    post = bind_resource_row_to_segment(
        {
            "schema_version": "stage6_resource_attempt/v1",
            **identity,
            "phase": "post",
            "accepted": False,
            "resource": post_resource,
        },
        first,
    )
    journal = DurableJsonl(path)
    journal.append(pre)
    journal.append(post)

    second = append_resource_segment_start(
        path,
        first_sample=_resource(root_pid=os.getpid(), sample_count=1, rss_bytes=50),
    )
    delayed_acceptance = bind_resource_row_to_segment(
        {
            "schema_version": "stage6_resource_acceptance/v2",
            **identity,
            "phase": "accepted",
            "accepted": True,
            "pre": pre_resource,
            "post": post_resource,
            "checkpoint": {
                "transaction_key": "update:1",
                "seed": 20260716,
                "update": 1,
            },
        },
        first,
    )
    journal.append(delayed_acceptance)
    _append_bound_terminal(
        path,
        terminal_resource=_resource(
            root_pid=os.getpid(), sample_count=2, rss_bytes=75
        ),
    )

    validated = validate_resource_lifecycle_ledger(path, require_terminal=True)
    assert validated["segment_chain"][0]["local_final_sample_count"] == 3
    assert validated["segment_chain"][1]["local_final_sample_count"] == 2
    assert validated["run_rss_lifecycle_peak_bytes"] == 300
    assert delayed_acceptance["segment_id"] == first["segment_id"]
    assert delayed_acceptance["segment_id"] != second["segment_id"]


@pytest.mark.parametrize(
    "mutation",
    (
        "old_segment",
        "prior_binding",
        "chain",
        "run_peak",
        "segment_id",
        "root_pid",
        "sample_rollback",
    ),
)
def test_validator_rejects_segment_and_terminal_tampering(
    tmp_path: Path,
    mutation: str,
) -> None:
    from lunar_exploration_ppo.utils.resource_lifecycle import (
        ResourceLifecycleError,
        append_resource_segment_start,
        validate_resource_lifecycle_rows,
    )

    path = tmp_path / "resource_audit.jsonl"
    first = append_resource_segment_start(
        path,
        first_sample=_resource(root_pid=os.getpid(), sample_count=1, rss_bytes=100),
    )
    _append_attempt(
        path,
        first,
        transaction_key="update:1",
        first_sample_count=2,
        pre_peak=200,
        post_peak=500,
    )
    second = append_resource_segment_start(
        path,
        first_sample=_resource(root_pid=os.getpid(), sample_count=1, rss_bytes=50),
    )
    _append_attempt(
        path,
        second,
        transaction_key="update:2",
        first_sample_count=2,
        pre_peak=200,
        post_peak=300,
    )
    _append_bound_terminal(
        path,
        terminal_resource=_resource(
            root_pid=os.getpid(), sample_count=4, rss_bytes=350
        ),
    )
    rows = copy.deepcopy(_rows(path))

    if mutation == "old_segment":
        rows[0]["first_sample"]["d_free_bytes"] += 1
    elif mutation == "prior_binding":
        rows[4]["prior_resource_log"]["sha256"] = "0" * 64
    elif mutation == "chain":
        rows[-1]["segment_chain"][0]["local_peak_rss_bytes"] += 1
    elif mutation == "run_peak":
        rows[-1]["run_rss_lifecycle_peak_bytes"] += 1
    elif mutation == "segment_id":
        rows[4]["segment_id"] = rows[0]["segment_id"]
    elif mutation == "root_pid":
        rows[4]["root_pid"] += 1
    elif mutation == "sample_rollback":
        rows[6]["resource"]["rss_sample_count"] = 1
    else:
        raise AssertionError(mutation)

    with pytest.raises(ResourceLifecycleError):
        validate_resource_lifecycle_rows(rows, require_terminal=True)


@pytest.mark.parametrize("kind", ("incomplete_tail", "noncanonical", "binding"))
def test_append_segment_fails_closed_for_invalid_existing_ledger(
    tmp_path: Path,
    kind: str,
) -> None:
    from lunar_exploration_ppo.utils.resource_lifecycle import (
        ResourceLifecycleError,
        append_resource_segment_start,
    )

    path = tmp_path / "resource_audit.jsonl"
    first = append_resource_segment_start(
        path,
        first_sample=_resource(root_pid=os.getpid(), sample_count=1, rss_bytes=100),
    )
    if kind == "incomplete_tail":
        path.write_bytes(path.read_bytes() + b'{"phase":"pre"')
    elif kind == "noncanonical":
        row = {"phase": "pre", "segment_id": first["segment_id"]}
        path.write_bytes(path.read_bytes() + (json.dumps(row, indent=2) + "\n").encode())
    elif kind == "binding":
        second = append_resource_segment_start(
            path,
            first_sample=_resource(
                root_pid=os.getpid(), sample_count=1, rss_bytes=50
            ),
        )
        rows = _rows(path)
        rows[-1]["prior_resource_log"]["size_bytes"] = 0
        path.write_bytes(b"".join(_canonical_row(row) for row in rows))
        assert rows[-1]["segment_id"] == second["segment_id"]
    else:
        raise AssertionError(kind)

    with pytest.raises(ResourceLifecycleError):
        append_resource_segment_start(
            path,
            first_sample=_resource(
                root_pid=os.getpid(), sample_count=1, rss_bytes=25
            ),
        )


def test_resource_lifecycle_securely_reads_and_exactly_verifies_appends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.utils.resource_lifecycle import (
        append_resource_segment_start,
        validate_resource_lifecycle_ledger,
    )

    path = tmp_path / "resource_audit.jsonl"
    first = append_resource_segment_start(
        path,
        first_sample=_resource(root_pid=os.getpid(), sample_count=1, rss_bytes=100),
    )
    original_read_bytes = Path.read_bytes

    def reject_target_read(self: Path) -> bytes:
        if self == path:
            raise AssertionError("resource lifecycle used naked Path.read_bytes")
        return original_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", reject_target_read)
    second = append_resource_segment_start(
        path,
        first_sample=_resource(root_pid=os.getpid(), sample_count=1, rss_bytes=50),
    )

    assert second["segment_index"] == 2
    assert first["segment_index"] == 1
    assert validate_resource_lifecycle_ledger(
        path,
        require_terminal=False,
    )["segment_count"] == 2


def test_terminal_requires_last_segment_progress_and_passed_hard_gate(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.resource_lifecycle import (
        ResourceLifecycleError,
        append_resource_segment_start,
    )

    path = tmp_path / "resource_audit.jsonl"
    segment = append_resource_segment_start(
        path,
        first_sample=_resource(root_pid=os.getpid(), sample_count=1, rss_bytes=100),
    )
    _append_attempt(
        path,
        segment,
        transaction_key="update:1",
        first_sample_count=2,
        pre_peak=200,
        post_peak=300,
    )

    with pytest.raises(ResourceLifecycleError, match="advance"):
        _append_bound_terminal(
            path,
            terminal_resource=_resource(
                root_pid=os.getpid(), sample_count=3, rss_bytes=350
            ),
        )
    with pytest.raises(ResourceLifecycleError, match="hard gate"):
        _append_bound_terminal(
            path,
            terminal_resource=_resource(
                root_pid=os.getpid(),
                sample_count=4,
                rss_bytes=350,
                passed=False,
            ),
        )


def test_terminal_is_final_and_rejects_every_later_row(tmp_path: Path) -> None:
    from lunar_exploration_ppo.utils.resource_lifecycle import (
        ResourceLifecycleError,
        append_resource_segment_start,
        bind_resource_row_to_segment,
        validate_resource_lifecycle_rows,
    )

    path = tmp_path / "resource_audit.jsonl"
    segment = append_resource_segment_start(
        path,
        first_sample=_resource(root_pid=os.getpid(), sample_count=1, rss_bytes=100),
    )
    _append_bound_terminal(
        path,
        terminal_resource=_resource(
            root_pid=os.getpid(), sample_count=2, rss_bytes=150
        ),
    )

    with pytest.raises(ResourceLifecycleError, match="terminal"):
        append_resource_segment_start(
            path,
            first_sample=_resource(
                root_pid=os.getpid(), sample_count=1, rss_bytes=50
            ),
        )
    later = bind_resource_row_to_segment(
        {
            "schema_version": "stage6_resource_attempt/v1",
            "kind": "update",
            "transaction_key": "update:2",
            "seed": 20260716,
            "update": 2,
            "attempt": 1,
            "phase": "pre",
            "accepted": False,
            "resource": _resource(
                root_pid=os.getpid(), sample_count=3, rss_bytes=175
            ),
        },
        segment,
    )
    with pytest.raises(ResourceLifecycleError, match="terminal"):
        validate_resource_lifecycle_rows([*_rows(path), later], require_terminal=True)


def test_two_real_processes_resume_with_new_pid_and_reset_sample_counter(
    tmp_path: Path,
) -> None:
    from lunar_exploration_ppo.utils.resource_lifecycle import (
        validate_resource_lifecycle_ledger,
    )

    path = tmp_path / "resource_audit.jsonl"
    first = json.loads(_run_python(path, "first").stdout)
    second = json.loads(_run_python(path, "terminal").stdout)

    assert first["pid"] != second["pid"]
    assert first["segment_id"] != second["segment_id"]
    assert (first["segment_index"], second["segment_index"]) == (1, 2)
    validated = validate_resource_lifecycle_ledger(path, require_terminal=True)
    assert [item["root_pid"] for item in validated["segment_chain"]] == [
        first["pid"],
        second["pid"],
    ]
    assert [
        item["local_first_sample_count"] for item in validated["segment_chain"]
    ] == [1, 1]
    assert [
        item["local_final_sample_count"] for item in validated["segment_chain"]
    ] == [3, 4]
    assert validated["run_rss_lifecycle_peak_bytes"] == 700


def test_cross_process_validation_rejects_tampered_prior_segment(
    tmp_path: Path,
) -> None:
    path = tmp_path / "resource_audit.jsonl"
    _run_python(path, "first")
    _run_python(path, "terminal")
    rows = _rows(path)
    rows[0]["first_sample"]["d_free_bytes"] += 1
    path.write_bytes(b"".join(_canonical_row(row) for row in rows))

    completed = _run_python(path, "validate", check=False)

    assert completed.returncode != 0
    assert "ResourceLifecycleError" in completed.stderr
