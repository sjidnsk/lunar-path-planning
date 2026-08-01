from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import json
import os
import subprocess
import sys
import textwrap
import uuid
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_NAME = "lunar_exploration_ppo.workflows.stage6_terminal_recovery"
MODULE_PATH = (
    ROOT
    / "src"
    / "lunar_exploration_ppo"
    / "workflows"
    / "stage6_terminal_recovery.py"
)
RECEIPT_NAME = "preterminal_acceptance.json"
TERMINAL_ARTIFACT_NAMES = (
    "summary.json",
    "routing.json",
    "standard_training_report.md",
    "standard_eval_report.md",
    "report.md",
)
PRETERMINAL_STATES = (
    "preflight",
    "global_best_frozen",
    "final_test_running",
    "final_unseen_running",
    "baselines_running",
)
COMPLETE_STATES = (*PRETERMINAL_STATES, "machine_passed", "awaiting_independent_review")
ABSENT_HASH = "3057252298292ae9a47b0879812581277456d1e9d9c0995178967da7c929236d"
FIRST_SEGMENT_ID = "123456789abc4def8abc123456789abc"
SECOND_SEGMENT_ID = "abcdef0123454abc9def0123456789ab"
THIRD_SEGMENT_ID = "fedcba9876544abc8def0123456789ab"
FOURTH_SEGMENT_ID = "0fedcba987654abc8def0123456789ab"
CAPABILITY_RUN_ID = "s6-standard-single-r1-20260717T010203Z"
IMMUTABLE_BINDING_FIELDS = frozenset(
    {
        "config_sha256",
        "source_set_sha256",
        "prospective_tree_sha256",
        "data_sha256",
        "environment_identity",
        "environment_sha256",
        "stage5_gate_sha256",
        "formal_run_id",
        "changed_path_set_sha256",
        "review_authorization_record_sha256",
        "authorization_file_sha256",
        "review_identity_sha256",
        "reviewed_prospective_git_tree",
        "frozen_diff_sha256",
        "spec_review_sha256",
        "quality_review_sha256",
    }
)


class InjectedCrash(RuntimeError):
    pass


def _module():
    try:
        return importlib.import_module(MODULE_NAME)
    except ModuleNotFoundError as exc:
        pytest.fail(f"terminal recovery module is missing: {exc}")


def _task1_planning_child_recovery_capability(
    tmp_path: Path,
    *,
    capability_sha256: str = "d" * 64,
):
    from lunar_exploration_ppo.workflows.stage6_planning_child_recovery import (
        PlanningChildLineageEpoch,
        PlanningChildRecoveryCapability,
        PlanningChildResumeCursor,
    )

    return PlanningChildRecoveryCapability(
        formal_run_id="s6-standard-single-r1-20260724T000124Z",
        seed=20260716,
        stage_root=tmp_path / "s6",
        parent_artifact_sha256="a" * 64,
        continuation_artifact_sha256="b" * 64,
        input_snapshot_sha256="c" * 64,
        capability_sha256=capability_sha256,
        resume_cursor=PlanningChildResumeCursor(
            last_accepted_update=85,
            next_update=86,
            next_attempt=1,
            next_transaction_key="0011:20260716:update:086",
            next_resource_segment_index=5,
            latest_complete_checkpoint_update=85,
            pending_pre_attempt=None,
        ),
        lineage_epochs=(
            PlanningChildLineageEpoch(75, 84, "1" * 64, "2" * 64),
            PlanningChildLineageEpoch(85, 85, "a" * 64, "3" * 64),
            PlanningChildLineageEpoch(86, None, "b" * 64, "4" * 64),
        ),
        protected_checkpoint_updates=(84, 85),
        input_pin_requests=(),
        acceptance_binding={
            "schema_version": "task1-terminal-acceptance"
        },
    )


def test_planning_child_terminal_recovery_uses_capability_acceptance_binding(
    tmp_path: Path,
) -> None:
    module = _module()
    capability = _task1_planning_child_recovery_capability(tmp_path)

    semantic = module._bind_planning_child_semantic_result(
        {"passed": True},
        capability=capability,
    )

    assert semantic["planning_child_recovery"][
        "capability_sha256"
    ] == capability.capability_sha256
    assert semantic["planning_child_recovery"][
        "acceptance_binding"
    ] == capability.acceptance_binding
    assert module._semantic_planning_child_capability(semantic) is capability


def test_machine_and_terminal_reject_different_recovery_capability_digest(
    tmp_path: Path,
) -> None:
    module = _module()
    machine_capability = _task1_planning_child_recovery_capability(
        tmp_path,
        capability_sha256="d" * 64,
    )
    terminal_capability = _task1_planning_child_recovery_capability(
        tmp_path,
        capability_sha256="e" * 64,
    )
    semantic = module._bind_planning_child_semantic_result(
        {"passed": True},
        capability=machine_capability,
    )

    with pytest.raises(
        module.TerminalRecoveryError,
        match="capability digest",
    ):
        module._require_planning_child_recovery_binding(
            semantic,
            terminal_capability,
        )


def test_terminal_commit_does_not_reopen_planning_child_journals() -> None:
    module_tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in module_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "recover_stage6_terminal_commit"
    )
    source = ast.get_source_segment(
        MODULE_PATH.read_text(encoding="utf-8"),
        function,
    )
    assert source is not None

    assert "planning_child_source_repair_context" not in source
    assert "_issue_planning_child_terminal_capability" not in source
    assert "_PlanningChildTerminalCapability" not in source


def test_planning_child_terminal_resource_prefix_uses_capability_binding() -> None:
    source = inspect.getsource(
        _module()._validate_semantically_complete_resource_rows
    )

    assert "0010:20260716:update:085" not in source
    assert 'get("update") != 85' not in source
    assert "journal_prefixes" in source


def test_formal_recovery_mutation_surfaces_require_execution_capability() -> None:
    module = _module()

    for name in (
        "write_stage6_preterminal_acceptance",
        "append_stage6_recovery_resource_segment",
        "append_stage6_recovery_resource_terminal",
        "recover_stage6_terminal_commit",
    ):
        function = getattr(module, name)
        parameters = inspect.signature(function).parameters
        assert "execution_capability" in parameters, name
        assert (
            parameters["execution_capability"].kind
            is inspect.Parameter.KEYWORD_ONLY
        )
        assert not hasattr(function, "__wrapped__"), name


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _canonical_row(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _identity(payload: bytes) -> dict[str, object]:
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _record_hash(event: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(dict(event))).hexdigest()


def _phase_row(
    state: str,
    previous_record_hash: str,
    bindings: Mapping[str, object],
) -> dict[str, object]:
    event = {
        "state": state,
        "previous_record_hash": previous_record_hash,
        "bindings": dict(bindings),
    }
    return {**event, "record_hash": _record_hash(event)}


def _resource(
    *,
    root_pid: int,
    sample_count: int,
    rss_bytes: int,
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
        "hard_stops": [],
        "passed": True,
    }


def _immutable_bindings() -> dict[str, object]:
    environment = {"implementation": "cpython", "python": "3.12", "system": "test"}
    reviewed_tree = "0123456789abcdef0123456789abcdef01234567"
    return {
        "config_sha256": "1" * 64,
        "source_set_sha256": "2" * 64,
        "prospective_tree_sha256": hashlib.sha256(
            reviewed_tree.encode("ascii")
        ).hexdigest(),
        "data_sha256": "4" * 64,
        "environment_identity": environment,
        "environment_sha256": hashlib.sha256(_canonical_json(environment)).hexdigest(),
        "stage5_gate_sha256": "5" * 64,
        "formal_run_id": "s6-standard-single-r1-20260716T123456Z",
        "changed_path_set_sha256": "a" * 64,
        "review_authorization_record_sha256": "b" * 64,
        "authorization_file_sha256": "c" * 64,
        "review_identity_sha256": "d" * 64,
        "reviewed_prospective_git_tree": reviewed_tree,
        "frozen_diff_sha256": "e" * 64,
        "spec_review_sha256": "f" * 64,
        "quality_review_sha256": "0" * 64,
    }


def _global_checkpoint_identity() -> dict[str, object]:
    return {
        "schema_version": "stage6_global_best/v1",
        "record": {
            "seed": 20260716,
            "update": 100,
            "success_rate_under_fixed_step_budget": 0.75,
            "mean_final_coverage": 0.625,
            "checkpoint_ref": "checkpoints/seed-20260716/update-00000100",
        },
        "transaction_key": "seed:20260716:update:100",
        "checkpoint_sha256": "6" * 64,
        "complete_marker_sha256": "7" * 64,
        "policy_state_sha256": "8" * 64,
    }


def _semantic_result(
    evidence_binding: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": "stage6_machine_acceptance_verification/v2",
        "passed": True,
        "state": "awaiting_independent_review",
        "final_evaluation_count": 10,
        "final_episode_count": 640,
        "checkpoint_receipt_count": 100,
        "global_best_policy_state_sha256": "8" * 64,
        "evidence_binding": dict(evidence_binding),
    }


def _terminal_artifacts() -> dict[str, bytes]:
    return {
        "summary.json": _canonical_json(
            {
                "schema_version": "stage6_summary/v1",
                "machine_passed": True,
                "state": "awaiting_independent_review",
            }
        ),
        "routing.json": _canonical_json(
            {
                "schema_version": "stage6_routing/v1",
                "machine_passed": True,
                "route": "awaiting_independent_review",
            }
        ),
        "standard_training_report.md": "# 标准训练报告\n\n固定字节。\n".encode("utf-8"),
        "standard_eval_report.md": "# 标准评估报告\n\n固定字节。\n".encode("utf-8"),
        "report.md": "# Stage 6\n\n固定字节。\n".encode("utf-8"),
    }


def _write_preterminal_stage(
    stage: Path,
    *,
    formal_run_id: str | None = None,
) -> dict[str, object]:
    stage.mkdir(parents=True)
    immutable = _immutable_bindings()
    if formal_run_id is not None:
        immutable["formal_run_id"] = formal_run_id
    global_checkpoint = _global_checkpoint_identity()
    phase_rows: list[dict[str, object]] = []
    previous = ABSENT_HASH
    for index, state in enumerate(PRETERMINAL_STATES):
        bindings = {
            **immutable,
            "checkpoint_sha256": (
                "9" * 64 if index == 0 else global_checkpoint["checkpoint_sha256"]
            ),
        }
        row = _phase_row(state, previous, bindings)
        phase_rows.append(row)
        previous = str(row["record_hash"])
    (stage / "phase-state.jsonl").write_bytes(
        b"".join(_canonical_row(row) for row in phase_rows)
    )

    first_sample = _resource(root_pid=os.getpid(), sample_count=1, rss_bytes=100)
    segment = {
        "schema_version": "stage6_resource_lifecycle_segment/v1",
        "kind": "resource_lifecycle_segment",
        "phase": "segment_start",
        "segment_id": FIRST_SEGMENT_ID,
        "segment_index": 1,
        "root_pid": os.getpid(),
        "first_sample": first_sample,
        "prior_resource_log": _identity(b""),
    }
    (stage / "resource_audit.jsonl").write_bytes(_canonical_row(segment))
    (stage / "config.json").write_bytes(
        _canonical_json({"schema_version": "stage6_config/v1", "seed": 20260716})
    )
    (stage / "global-best.json").write_bytes(_canonical_json(global_checkpoint))
    (stage / "metrics.jsonl").write_bytes(
        _canonical_row({"schema_version": "stage6_metric/v1", "value": 1})
    )
    module = _module()
    with module._capture_preterminal_evidence_handle(stage) as evidence:
        semantic = _semantic_result(evidence.binding)
    return {
        "stage": stage,
        "immutable": immutable,
        "global_checkpoint": global_checkpoint,
        "semantic": semantic,
        "artifacts": _terminal_artifacts(),
    }


@pytest.fixture
def recovery_authority_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from test_stage6_workflow import _active_execution_capability

    with ExitStack() as stack:
        counter = 0

        def activate(label: str = "run") -> dict[str, object]:
            nonlocal counter
            counter += 1
            run_root = tmp_path / f"{counter:02d}-{label}" / CAPABILITY_RUN_ID
            return stack.enter_context(
                _active_execution_capability(
                    tmp_path,
                    monkeypatch,
                    run_root=run_root,
                )
            )

        yield activate


@pytest.fixture
def preterminal(recovery_authority_factory) -> dict[str, object]:
    active = recovery_authority_factory("preterminal")
    value = _write_preterminal_stage(Path(active["run_root"]) / "s6")
    value["execution_capability"] = active["capability"]
    value["active_authority"] = active
    return value


def _write_receipt(preterminal: Mapping[str, object]) -> dict[str, object]:
    module = _module()
    return module.write_stage6_preterminal_acceptance(
        stage_root=preterminal["stage"],
        semantic_result=preterminal["semantic"],
        immutable_bindings=preterminal["immutable"],
        global_checkpoint_identity=preterminal["global_checkpoint"],
        terminal_artifacts=preterminal["artifacts"],
        execution_capability=preterminal["execution_capability"],
    )


def _refresh_semantic_binding(preterminal: dict[str, object]) -> None:
    module = _module()
    stage = Path(preterminal["stage"])
    with module._capture_preterminal_evidence_handle(stage) as evidence:
        preterminal["semantic"] = _semantic_result(evidence.binding)


def _install_forged_planning_child_preterminal_profile(
    preterminal: dict[str, object],
) -> None:
    stage = Path(preterminal["stage"])
    resource_path = stage / "resource_audit.jsonl"
    segment = json.loads(
        resource_path.read_text(encoding="utf-8").splitlines()[0]
    )
    u85_pre = {
        "schema_version": "stage6_resource_attempt/v1",
        "kind": "update",
        "phase": "pre",
        "accepted": False,
        "seed": 20260716,
        "update": 85,
        "transaction_key": "0010:20260716:update:085",
        "attempt": 1,
        "segment_id": segment["segment_id"],
        "segment_index": segment["segment_index"],
        "resource": _resource(
            root_pid=int(segment["root_pid"]),
            sample_count=2,
            rss_bytes=101,
        ),
    }
    resource_path.write_bytes(
        resource_path.read_bytes() + _canonical_row(u85_pre)
    )
    (stage / "lineage_audit.json").write_bytes(
        _canonical_json(
            {
                "schema_version": "stage6_lineage_audit/v3",
                "planning_warm_start": {},
            }
        )
    )
    (stage / "planning-child-source-repair.json").write_bytes(
        _canonical_json(
            {
                "schema_version": "forged_filename_profile/v1",
                "canonical_sha256": "0" * 64,
            }
        )
    )
    _refresh_semantic_binding(preterminal)


def _append_terminal(
    stage: Path,
    receipt_identity: Mapping[str, object],
    *,
    include_receipt: bool = True,
) -> dict[str, object]:
    from lunar_exploration_ppo.utils.resource_lifecycle import (
        build_bound_terminal_resource_evidence,
    )

    resource_path = stage / "resource_audit.jsonl"
    prior = resource_path.read_bytes()
    rows = [json.loads(line) for line in prior.decode("utf-8").splitlines()]
    segment = rows[-1]
    terminal_resource = _resource(
        root_pid=int(segment["root_pid"]),
        sample_count=2,
        rss_bytes=200,
    )
    terminal = build_bound_terminal_resource_evidence(
        rows,
        terminal_resource=terminal_resource,
        preterminal_acceptance=receipt_identity,
    )
    if not include_receipt:
        del terminal["preterminal_acceptance"]
    resource_path.write_bytes(prior + _canonical_row(terminal))
    return terminal


def _append_second_resource_segment(stage: Path) -> None:
    path = stage / "resource_audit.jsonl"
    prior = path.read_bytes()
    second = {
        "schema_version": "stage6_resource_lifecycle_segment/v1",
        "kind": "resource_lifecycle_segment",
        "phase": "segment_start",
        "segment_id": SECOND_SEGMENT_ID,
        "segment_index": 2,
        "root_pid": os.getpid(),
        "first_sample": _resource(
            root_pid=os.getpid(), sample_count=1, rss_bytes=50
        ),
        "prior_resource_log": _identity(prior),
    }
    path.write_bytes(prior + _canonical_row(second))


def _manifest_committer(stage_root: Path) -> dict[str, object]:
    stage = Path(stage_root)
    manifest_path = stage / "manifest.json"
    entries = []
    for path in sorted(
        (item for item in stage.rglob("*") if item.is_file() and item != manifest_path),
        key=lambda item: item.relative_to(stage).as_posix(),
    ):
        payload = path.read_bytes()
        entries.append(
            {
                "path": path.relative_to(stage).as_posix(),
                **_identity(payload),
            }
        )
    payload = _canonical_json(
        {"schema_version": "stage6_test_terminal_manifest/v1", "artifacts": entries}
    )
    if manifest_path.exists():
        if manifest_path.read_bytes() != payload:
            raise ValueError("manifest drifted")
    else:
        with manifest_path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    return _identity(payload)


def _prepare_terminal(preterminal: Mapping[str, object]) -> dict[str, object]:
    receipt_identity = _write_receipt(preterminal)
    _append_terminal(Path(preterminal["stage"]), receipt_identity)
    return receipt_identity


def _phase_states(stage: Path) -> tuple[str, ...]:
    return tuple(
        json.loads(line)["state"]
        for line in (stage / "phase-state.jsonl").read_text(encoding="utf-8").splitlines()
    )


def _snapshot_files(stage: Path) -> dict[str, bytes]:
    return {
        path.relative_to(stage).as_posix(): path.read_bytes()
        for path in sorted(stage.rglob("*"))
        if path.is_file()
    }


def _append_raw_recovery_segment(
    stage: Path,
    *,
    root_pid: int,
    segment_id: str,
    sample_count: int = 1,
    rss_bytes: int = 50,
    prior_resource_log: Mapping[str, object] | None = None,
) -> dict[str, object]:
    path = stage / "resource_audit.jsonl"
    prior = path.read_bytes()
    rows = [json.loads(line) for line in prior.decode("utf-8").splitlines()]
    segment = {
        "schema_version": "stage6_resource_lifecycle_segment/v1",
        "kind": "resource_lifecycle_segment",
        "phase": "segment_start",
        "segment_id": segment_id,
        "segment_index": 1
        + sum(row.get("phase") == "segment_start" for row in rows),
        "root_pid": root_pid,
        "first_sample": _resource(
            root_pid=root_pid,
            sample_count=sample_count,
            rss_bytes=rss_bytes,
        ),
        "prior_resource_log": dict(prior_resource_log or _identity(prior)),
    }
    path.write_bytes(prior + _canonical_row(segment))
    return segment


def _append_raw_resource_row(stage: Path, row: Mapping[str, object]) -> None:
    path = stage / "resource_audit.jsonl"
    path.write_bytes(path.read_bytes() + _canonical_row(dict(row)))


def _append_attempt_suffix(stage: Path, *, through_phase: str) -> None:
    root_pid = os.getpid() + 10_000
    segment = _append_raw_recovery_segment(
        stage,
        root_pid=root_pid,
        segment_id=SECOND_SEGMENT_ID,
    )
    common = {
        "schema_version": "stage6_resource_attempt/v1",
        "kind": "update",
        "transaction_key": "recovery-forbidden:1",
        "seed": 20260716,
        "update": 101,
        "attempt": 1,
        "accepted": False,
        "segment_id": segment["segment_id"],
        "segment_index": segment["segment_index"],
    }
    pre_resource = _resource(root_pid=root_pid, sample_count=2, rss_bytes=75)
    post_resource = _resource(root_pid=root_pid, sample_count=3, rss_bytes=90)
    _append_raw_resource_row(
        stage,
        {**common, "phase": "pre", "resource": pre_resource},
    )
    if through_phase == "pre":
        return
    _append_raw_resource_row(
        stage,
        {**common, "phase": "post", "resource": post_resource},
    )
    if through_phase == "post":
        return
    _append_raw_resource_row(
        stage,
        {
            **common,
            "schema_version": "stage6_resource_acceptance/v2",
            "phase": "accepted",
            "accepted": True,
            "pre": pre_resource,
            "post": post_resource,
            "checkpoint": {
                "transaction_key": common["transaction_key"],
                "seed": common["seed"],
                "update": common["update"],
            },
        },
    )


def _subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    source_root = str(ROOT / "src")
    environment["PYTHONPATH"] = source_root + (
        os.pathsep + environment["PYTHONPATH"]
        if environment.get("PYTHONPATH")
        else ""
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _run_recovery_resource_subprocess(
    stage: Path,
    *,
    mode: str,
    first_peak: int,
    terminal_peak: int = 0,
) -> dict[str, object]:
    script = textwrap.dedent(
        r'''
        import builtins
        import json
        import os
        import sys
        from pathlib import Path

        forbidden = (
            "torch",
            "lunar_exploration_ppo.policy",
            "lunar_exploration_ppo.ppo.checkpoint",
            "lunar_exploration_ppo.ppo.collector",
            "lunar_exploration_ppo.ppo.trainer",
            "lunar_exploration_ppo.ppo.standard_training",
            "lunar_exploration_ppo.env.standard_training",
            "lunar_exploration_ppo.eval.standard",
        )
        real_import = builtins.__import__

        def import_spy(name, globals=None, locals=None, fromlist=(), level=0):
            if any(name == item or name.startswith(item + ".") for item in forbidden):
                raise AssertionError(f"forbidden recovery import: {name}")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = import_spy
        from lunar_exploration_ppo.workflows.stage6_terminal_recovery import (
            append_stage6_recovery_resource_segment,
            append_stage6_recovery_resource_terminal,
        )

        stage = Path(sys.argv[1])
        mode = sys.argv[2]
        first_peak = int(sys.argv[3])
        terminal_peak = int(sys.argv[4])

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

        segment = None
        terminal = None
        error = None
        try:
            if mode == "segment":
                segment = append_stage6_recovery_resource_segment(
                    stage_root=stage,
                    first_sample=resource(1, first_peak),
                )
            elif mode == "terminal":
                terminal = append_stage6_recovery_resource_terminal(
                    stage_root=stage,
                    terminal_resource=resource(2, terminal_peak),
                )
            else:
                raise AssertionError(f"unknown mode: {mode}")
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
        print(json.dumps({
            "pid": os.getpid(),
            "segment": segment,
            "terminal": terminal,
            "error": error,
        }, sort_keys=True))
        '''
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(stage),
            mode,
            str(first_peak),
            str(terminal_peak),
        ],
        cwd=ROOT,
        env=_subprocess_environment(),
        shell=False,
        timeout=30,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_module_and_public_api_are_present_without_forbidden_imports() -> None:
    assert MODULE_PATH.is_file(), f"missing production module: {MODULE_PATH}"
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    forbidden_fragments = (
        "torch",
        ".policy",
        ".checkpoint",
        ".collector",
        ".trainer",
        ".standard_training",
        ".eval.standard",
    )
    assert not any(
        fragment in module_name
        for module_name in imported
        for fragment in forbidden_fragments
    )

    module = _module()
    for name in (
        "write_stage6_preterminal_acceptance",
        "detect_stage6_terminal_recovery",
        "append_stage6_recovery_resource_segment",
        "append_stage6_recovery_resource_terminal",
        "recover_stage6_terminal_commit",
    ):
        assert callable(getattr(module, name, None))
        parameters = inspect.signature(getattr(module, name)).parameters
        assert not ({"force", "skip", "fake", "semantic_callback"} & set(parameters))


def test_evidence_handle_closes_and_hides_bound_bytes_and_membership(
    preterminal: Mapping[str, object],
) -> None:
    module = _module()
    stage = Path(preterminal["stage"])
    handle = module._capture_preterminal_evidence_handle(stage)
    with handle as evidence:
        assert evidence.read_bytes("config.json", label="bound config")
        assert evidence.directory_snapshot()
        assert evidence.binding["schema_version"] == (
            "stage6_preterminal_evidence_binding/v1"
        )

    assert handle.closed is True
    for operation in (
        lambda: handle.read_bytes("config.json", label="closed bound config"),
        handle.directory_snapshot,
        lambda: handle.require_current("closed evidence"),
        lambda: handle.binding,
    ):
        with pytest.raises(module.TerminalRecoveryError, match="closed"):
            operation()


def test_evidence_handle_preserves_primary_exception_when_pin_close_fails(
    preterminal: Mapping[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lunar_exploration_ppo.utils.stage6_input_pinning import (
        Stage6InputPin,
        Stage6InputPinError,
    )

    module = _module()
    handle = module._capture_preterminal_evidence_handle(Path(preterminal["stage"]))
    original_close = Stage6InputPin.close

    def close_then_fail(pin: Stage6InputPin) -> None:
        original_close(pin)
        raise Stage6InputPinError("injected evidence pin close failure")

    monkeypatch.setattr(Stage6InputPin, "close", close_then_fail)
    with pytest.raises(RuntimeError, match="primary verifier failure") as raised:
        with handle:
            raise RuntimeError("primary verifier failure")
    assert handle.closed is True
    assert any(
        "suppressed Stage 6 evidence close failure" in note
        for note in getattr(raised.value, "__notes__", ())
    )


def test_preterminal_evidence_handle_rejects_reserved_membership_after_capture(
    preterminal: Mapping[str, object],
) -> None:
    module = _module()
    stage = Path(preterminal["stage"])
    reserved = stage / "manifest.json"
    with module._capture_preterminal_evidence_handle(stage) as evidence:
        reserved.write_bytes(b"{}")
        try:
            with pytest.raises(
                module.TerminalRecoveryError,
                match="reserved artifact appeared",
            ):
                evidence.require_current("pending membership revalidation")
        finally:
            reserved.unlink()


def test_receipt_accepts_exact_current_stage6_immutable_bindings(
    preterminal: dict[str, object],
) -> None:
    stage = Path(preterminal["stage"])
    assert set(preterminal["immutable"]) == IMMUTABLE_BINDING_FIELDS

    identity = _write_receipt(preterminal)

    receipt = json.loads((stage / RECEIPT_NAME).read_text(encoding="utf-8"))
    assert receipt["immutable_bindings"] == preterminal["immutable"]
    assert identity == _identity((stage / RECEIPT_NAME).read_bytes())


def test_receipt_rejects_legacy_seven_field_immutable_bindings(
    preterminal: dict[str, object],
) -> None:
    module = _module()
    legacy_fields = (
        "config_sha256",
        "source_set_sha256",
        "prospective_tree_sha256",
        "data_sha256",
        "environment_identity",
        "environment_sha256",
        "stage5_gate_sha256",
    )
    immutable = preterminal["immutable"]
    assert isinstance(immutable, Mapping)
    legacy = {field: immutable[field] for field in legacy_fields}

    with pytest.raises(module.TerminalRecoveryError, match="immutable"):
        module.write_stage6_preterminal_acceptance(
            stage_root=preterminal["stage"],
            semantic_result=preterminal["semantic"],
            immutable_bindings=legacy,
            global_checkpoint_identity=preterminal["global_checkpoint"],
            terminal_artifacts=preterminal["artifacts"],
            execution_capability=preterminal["execution_capability"],
        )
    assert not (Path(preterminal["stage"]) / RECEIPT_NAME).exists()


@pytest.mark.parametrize(
    ("field", "bad_value"),
    (
        ("config_sha256", "A" * 64),
        ("formal_run_id", "s6-standard-single-r1-20260229T123456Z"),
        ("formal_run_id", "s6-standard-single-r1-20260716T123456+00:00"),
        ("reviewed_prospective_git_tree", "A" * 40),
        ("reviewed_prospective_git_tree", "a" * 39),
        ("prospective_tree_sha256", "3" * 64),
        ("environment_sha256", "4" * 64),
    ),
)
def test_receipt_rejects_invalid_current_immutable_binding_values(
    preterminal: dict[str, object],
    field: str,
    bad_value: str,
) -> None:
    module = _module()
    immutable = dict(preterminal["immutable"])
    immutable[field] = bad_value

    with pytest.raises(module.TerminalRecoveryError, match="immutable"):
        module.write_stage6_preterminal_acceptance(
            stage_root=preterminal["stage"],
            semantic_result=preterminal["semantic"],
            immutable_bindings=immutable,
            global_checkpoint_identity=preterminal["global_checkpoint"],
            terminal_artifacts=preterminal["artifacts"],
            execution_capability=preterminal["execution_capability"],
        )
    assert not (Path(preterminal["stage"]) / RECEIPT_NAME).exists()


def test_preterminal_receipt_is_written_only_after_exact_passed_semantics(
    preterminal: dict[str, object],
) -> None:
    module = _module()
    stage = Path(preterminal["stage"])
    failed = dict(preterminal["semantic"])
    failed["passed"] = False
    with pytest.raises(module.TerminalRecoveryError, match="semantic"):
        module.write_stage6_preterminal_acceptance(
            stage_root=stage,
            semantic_result=failed,
            immutable_bindings=preterminal["immutable"],
            global_checkpoint_identity=preterminal["global_checkpoint"],
            terminal_artifacts=preterminal["artifacts"],
            execution_capability=preterminal["execution_capability"],
        )
    assert not (stage / RECEIPT_NAME).exists()

    unknown = dict(preterminal["semantic"])
    unknown["unknown"] = True
    with pytest.raises(module.TerminalRecoveryError, match="semantic"):
        module.write_stage6_preterminal_acceptance(
            stage_root=stage,
            semantic_result=unknown,
            immutable_bindings=preterminal["immutable"],
            global_checkpoint_identity=preterminal["global_checkpoint"],
            terminal_artifacts=preterminal["artifacts"],
            execution_capability=preterminal["execution_capability"],
        )
    assert not (stage / RECEIPT_NAME).exists()

    (stage / "summary.json").write_bytes(preterminal["artifacts"]["summary.json"])
    with pytest.raises(module.TerminalRecoveryError, match="preterminal|artifact"):
        module.write_stage6_preterminal_acceptance(
            stage_root=stage,
            semantic_result=preterminal["semantic"],
            immutable_bindings=preterminal["immutable"],
            global_checkpoint_identity=preterminal["global_checkpoint"],
            terminal_artifacts=preterminal["artifacts"],
            execution_capability=preterminal["execution_capability"],
        )
    (stage / "summary.json").unlink()
    assert not (stage / RECEIPT_NAME).exists()

    identity = _write_receipt(preterminal)
    receipt_payload = (stage / RECEIPT_NAME).read_bytes()
    assert identity == _identity(receipt_payload)
    receipt = json.loads(receipt_payload.decode("utf-8"))
    assert _canonical_json(receipt) == receipt_payload
    assert set(receipt) == {
        "schema_version",
        "semantic_verification",
        "preterminal_evidence_binding",
        "immutable_bindings",
        "global_checkpoint_identity",
        "terminal_artifacts",
        "preterminal_artifact_graph",
        "preterminal_directory_graph",
        "resource_audit_prefix",
        "phase_state_prefix",
    }
    assert receipt["schema_version"] == "stage6_preterminal_acceptance/v2"
    assert receipt["semantic_verification"] == preterminal["semantic"]
    assert receipt["preterminal_evidence_binding"] == preterminal["semantic"][
        "evidence_binding"
    ]
    assert receipt["immutable_bindings"] == preterminal["immutable"]
    assert receipt["global_checkpoint_identity"] == preterminal["global_checkpoint"]
    assert [row["path"] for row in receipt["terminal_artifacts"]] == list(
        TERMINAL_ARTIFACT_NAMES
    )
    for row in receipt["terminal_artifacts"]:
        payload = preterminal["artifacts"][row["path"]]
        assert set(row) == {"path", "utf8", "sha256", "size_bytes"}
        assert row["utf8"].encode("utf-8") == payload
        assert {key: row[key] for key in ("sha256", "size_bytes")} == _identity(
            payload
        )
    assert _write_receipt(preterminal) == identity
    assert (stage / RECEIPT_NAME).read_bytes() == receipt_payload


def test_planning_child_receipt_writer_rejects_filename_only_unvalidated_profile(
    recovery_authority_factory,
) -> None:
    active = recovery_authority_factory("forged-planning-child-profile")
    preterminal = _write_preterminal_stage(
        Path(active["run_root"]) / "s6",
        formal_run_id=CAPABILITY_RUN_ID,
    )
    preterminal["execution_capability"] = active["capability"]
    _install_forged_planning_child_preterminal_profile(preterminal)

    with pytest.raises(
        _module().TerminalRecoveryError,
        match="planning child|source-repair|capability|artifact",
    ):
        _write_receipt(preterminal)

    assert not (
        Path(preterminal["stage"]) / RECEIPT_NAME
    ).exists()


def test_old_incomplete_v1_receipt_fails_closed(
    preterminal: dict[str, object],
) -> None:
    module = _module()
    stage = Path(preterminal["stage"])
    _write_receipt(preterminal)
    receipt_path = stage / RECEIPT_NAME
    legacy = json.loads(receipt_path.read_text(encoding="utf-8"))
    legacy["schema_version"] = "stage6_preterminal_acceptance/v1"
    del legacy["preterminal_evidence_binding"]
    legacy["semantic_verification"]["schema_version"] = (
        "stage6_machine_acceptance_verification/v1"
    )
    del legacy["semantic_verification"]["evidence_binding"]
    receipt_path.write_bytes(_canonical_json(legacy))

    detection = module.detect_stage6_terminal_recovery(stage_root=stage)

    assert detection["status"] == "invalid"
    assert "schema" in str(detection["reason"]).lower()


def test_receipt_binds_full_preterminal_graph_and_both_mutable_prefixes(
    preterminal: dict[str, object],
) -> None:
    stage = Path(preterminal["stage"])
    _write_receipt(preterminal)
    receipt = json.loads((stage / RECEIPT_NAME).read_text(encoding="utf-8"))
    graph = receipt["preterminal_artifact_graph"]
    assert [row["path"] for row in graph] == sorted(
        (
            "config.json",
            "global-best.json",
            "metrics.jsonl",
            "phase-state.jsonl",
            "resource_audit.jsonl",
        )
    )
    for row in graph:
        payload = (stage / row["path"]).read_bytes()
        assert set(row) == {"path", "sha256", "size_bytes"}
        assert {key: row[key] for key in ("sha256", "size_bytes")} == _identity(
            payload
        )
    assert receipt["resource_audit_prefix"] == _identity(
        (stage / "resource_audit.jsonl").read_bytes()
    )
    assert receipt["phase_state_prefix"] == _identity(
        (stage / "phase-state.jsonl").read_bytes()
    )


@pytest.mark.parametrize(
    "bad_artifacts",
    (
        {"summary.json": b"{}\n"},
        {**_terminal_artifacts(), "unknown.txt": b"unknown"},
        {**_terminal_artifacts(), "summary.json": b'{"not":"canonical"}\n'},
        {**_terminal_artifacts(), "report.md": b"\xff"},
    ),
)
def test_receipt_rejects_unknown_missing_or_non_utf8_terminal_artifacts(
    preterminal: dict[str, object],
    bad_artifacts: dict[str, bytes],
) -> None:
    module = _module()
    with pytest.raises(module.TerminalRecoveryError, match="artifact|UTF-8|canonical"):
        module.write_stage6_preterminal_acceptance(
            stage_root=preterminal["stage"],
            semantic_result=preterminal["semantic"],
            immutable_bindings=preterminal["immutable"],
            global_checkpoint_identity=preterminal["global_checkpoint"],
            terminal_artifacts=bad_artifacts,
            execution_capability=preterminal["execution_capability"],
        )


def test_receipt_rejects_hardlink_and_reparse_graph_members(
    preterminal: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    stage = Path(preterminal["stage"])
    alias = tmp_path / "config-alias.json"
    os.link(stage / "config.json", alias)
    with pytest.raises(module.TerminalRecoveryError, match="hard link|plain|safe"):
        _write_receipt(preterminal)
    alias.unlink()

    original = module.is_link_or_reparse
    monkeypatch.setattr(
        module,
        "is_link_or_reparse",
        lambda path: Path(path).name == "config.json" or original(path),
    )
    with pytest.raises(module.TerminalRecoveryError, match="link|reparse|safe"):
        _write_receipt(preterminal)


def test_recovery_refuses_to_commit_before_terminal_binding(
    preterminal: dict[str, object],
) -> None:
    module = _module()
    stage = Path(preterminal["stage"])
    receipt_identity = _write_receipt(preterminal)
    detection = module.detect_stage6_terminal_recovery(stage_root=stage)
    assert detection == {
        "status": "valid_preterminal_recovery",
        "reason": "",
        "receipt_identity": receipt_identity,
        "phase_states": PRETERMINAL_STATES,
    }
    before = _snapshot_files(stage)
    with pytest.raises(module.TerminalRecoveryError, match="terminal"):
        module.recover_stage6_terminal_commit(
            stage_root=stage,
            manifest_committer=_manifest_committer,
            execution_capability=preterminal["execution_capability"],
        )
    assert _snapshot_files(stage) == before
    assert _phase_states(stage) == PRETERMINAL_STATES
    assert all(not (stage / name).exists() for name in TERMINAL_ARTIFACT_NAMES)
    assert not (stage / "manifest.json").exists()


def test_receipt_rejects_semantically_incomplete_resource_prefix(
    preterminal: dict[str, object],
) -> None:
    module = _module()
    stage = Path(preterminal["stage"])
    segment = json.loads(
        (stage / "resource_audit.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    _append_raw_resource_row(
        stage,
        {
            "schema_version": "stage6_resource_attempt/v1",
            "kind": "update",
            "transaction_key": "incomplete:1",
            "seed": 20260716,
            "update": 101,
            "attempt": 1,
            "phase": "pre",
            "accepted": False,
            "resource": _resource(
                root_pid=int(segment["root_pid"]),
                sample_count=2,
                rss_bytes=150,
            ),
            "segment_id": segment["segment_id"],
            "segment_index": segment["segment_index"],
        },
    )

    with pytest.raises(module.TerminalRecoveryError, match="resource|semantic"):
        _write_receipt(preterminal)
    assert not (stage / RECEIPT_NAME).exists()


def test_recovery_segment_append_validates_graph_before_durable_write(
    preterminal: dict[str, object],
) -> None:
    module = _module()
    stage = Path(preterminal["stage"])
    _write_receipt(preterminal)
    resource_path = stage / "resource_audit.jsonl"
    before = resource_path.read_bytes()
    (stage / "unknown-empty-directory").mkdir()

    with pytest.raises(module.TerminalRecoveryError, match="directory|graph"):
        module.append_stage6_recovery_resource_segment(
            stage_root=stage,
            first_sample=_resource(
                root_pid=os.getpid(),
                sample_count=1,
                rss_bytes=50,
            ),
            execution_capability=preterminal["execution_capability"],
        )
    assert resource_path.read_bytes() == before


def test_same_active_capability_transitions_recovery_suffix_to_bound_terminal(
    preterminal: dict[str, object],
) -> None:
    from lunar_exploration_ppo.utils.resource_lifecycle import (
        validate_resource_lifecycle_ledger,
    )

    module = _module()
    stage = Path(preterminal["stage"])
    resource_path = stage / "resource_audit.jsonl"
    initial = json.loads(resource_path.read_text(encoding="utf-8"))
    initial["root_pid"] = os.getpid() + 10_000
    initial["first_sample"]["rss_root_pid"] = initial["root_pid"]
    resource_path.write_bytes(_canonical_row(initial))
    _refresh_semantic_binding(preterminal)
    receipt_identity = _write_receipt(preterminal)

    statuses = [
        module.detect_stage6_terminal_recovery(stage_root=stage)["status"]
    ]
    segment = module.append_stage6_recovery_resource_segment(
        stage_root=stage,
        first_sample=_resource(
            root_pid=os.getpid(),
            sample_count=1,
            rss_bytes=500,
        ),
        execution_capability=preterminal["execution_capability"],
    )
    statuses.append(
        module.detect_stage6_terminal_recovery(stage_root=stage)["status"]
    )
    terminal = module.append_stage6_recovery_resource_terminal(
        stage_root=stage,
        terminal_resource=_resource(
            root_pid=os.getpid(),
            sample_count=2,
            rss_bytes=500,
        ),
        execution_capability=preterminal["execution_capability"],
    )
    statuses.append(
        module.detect_stage6_terminal_recovery(stage_root=stage)["status"]
    )

    assert statuses == [
        "valid_preterminal_recovery",
        "valid_preterminal_recovery",
        "valid_terminal_recovery",
    ]

    path = stage / "resource_audit.jsonl"
    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    rows = [json.loads(line) for line in lines]
    segment_rows = [row for row in rows if row.get("phase") == "segment_start"]
    assert [row["segment_index"] for row in segment_rows] == [1, 2]
    assert [row["root_pid"] for row in segment_rows] == [
        initial["root_pid"],
        os.getpid(),
    ]
    assert [row["first_sample"]["rss_sample_count"] for row in segment_rows] == [
        1,
        1,
    ]
    segment_ids = [str(row["segment_id"]) for row in segment_rows]
    assert len(set(segment_ids)) == 2
    assert all(uuid.UUID(hex=value).version == 4 for value in segment_ids)

    offset = 0
    for row, line in zip(rows, lines, strict=True):
        if row.get("phase") == "segment_start":
            assert row["prior_resource_log"] == _identity(payload[:offset])
        offset += len(line)

    terminal = rows[-1]
    full_prefix = b"".join(lines[:-1])
    assert terminal["prior_resource_audit"] == _identity(full_prefix)
    assert terminal["preterminal_acceptance"] == receipt_identity
    assert terminal["segment_id"] == segment["segment_id"]
    assert terminal["run_rss_lifecycle_peak_bytes"] == 500
    validation = validate_resource_lifecycle_ledger(path, require_terminal=True)
    assert validation["segment_count"] == 2
    assert validation["run_rss_lifecycle_peak_bytes"] == 500


def test_fresh_process_cannot_use_process_local_recovery_capability(
    preterminal: dict[str, object],
) -> None:
    stage = Path(preterminal["stage"])
    _write_receipt(preterminal)
    before = _snapshot_files(stage)

    for mode in ("segment", "terminal"):
        evidence = _run_recovery_resource_subprocess(
            stage,
            mode=mode,
            first_peak=500,
            terminal_peak=350,
        )
        assert evidence["pid"] != os.getpid()
        assert evidence["segment"] is None
        assert evidence["terminal"] is None
        assert evidence["error"]["type"] == "TerminalRecoveryError"
        assert "capability" in evidence["error"]["message"]
        assert _snapshot_files(stage) == before


@pytest.mark.parametrize(
    "tamper",
    (
        "pre",
        "post",
        "accepted",
        "unknown",
        "duplicate_pid",
        "duplicate_segment",
        "bad_prior_hash",
        "bad_counter_reset",
    ),
)
def test_receipt_only_suffix_rejects_non_segment_and_segment_chain_tampering(
    preterminal: dict[str, object],
    tamper: str,
) -> None:
    module = _module()
    stage = Path(preterminal["stage"])
    _write_receipt(preterminal)
    synthetic_pid = os.getpid() + 10_000

    if tamper in {"pre", "post", "accepted"}:
        _append_attempt_suffix(stage, through_phase=tamper)
    elif tamper == "unknown":
        _append_raw_resource_row(
            stage,
            {"schema_version": "unknown/v1", "phase": "unknown"},
        )
    elif tamper == "duplicate_pid":
        _append_raw_recovery_segment(
            stage,
            root_pid=os.getpid(),
            segment_id=SECOND_SEGMENT_ID,
        )
    elif tamper == "duplicate_segment":
        _append_raw_recovery_segment(
            stage,
            root_pid=synthetic_pid,
            segment_id=SECOND_SEGMENT_ID,
        )
        _append_raw_recovery_segment(
            stage,
            root_pid=synthetic_pid + 1,
            segment_id=SECOND_SEGMENT_ID,
        )
    elif tamper == "bad_prior_hash":
        _append_raw_recovery_segment(
            stage,
            root_pid=synthetic_pid,
            segment_id=SECOND_SEGMENT_ID,
            prior_resource_log={"sha256": "0" * 64, "size_bytes": 0},
        )
    elif tamper == "bad_counter_reset":
        _append_raw_recovery_segment(
            stage,
            root_pid=synthetic_pid,
            segment_id=SECOND_SEGMENT_ID,
            sample_count=2,
        )
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(tamper)

    detection = module.detect_stage6_terminal_recovery(stage_root=stage)
    assert detection["status"] == "invalid"
    assert detection["status"] != "no_terminal"


def test_terminal_replay_rejects_run_peak_forged_across_recovery_suffix(
    preterminal: dict[str, object],
) -> None:
    module = _module()
    stage = Path(preterminal["stage"])
    receipt_identity = _write_receipt(preterminal)
    _append_raw_recovery_segment(
        stage,
        root_pid=os.getpid() + 10_000,
        segment_id=SECOND_SEGMENT_ID,
        rss_bytes=500,
    )
    _append_raw_recovery_segment(
        stage,
        root_pid=os.getpid() + 20_000,
        segment_id=THIRD_SEGMENT_ID,
        rss_bytes=50,
    )
    _append_terminal(stage, receipt_identity)
    path = stage / "resource_audit.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[-1]["run_rss_lifecycle_peak_bytes"] == 500
    rows[-1]["run_rss_lifecycle_peak_bytes"] = 499
    path.write_bytes(b"".join(_canonical_row(row) for row in rows))

    detection = module.detect_stage6_terminal_recovery(stage_root=stage)
    assert detection["status"] == "invalid"
    assert "terminal" in detection["reason"] or "resource" in detection["reason"]


def test_detector_rejects_legacy_receipt_and_missing_receipt_bound_resource(
    preterminal: dict[str, object],
    recovery_authority_factory,
) -> None:
    module = _module()
    stage = Path(preterminal["stage"])
    _write_receipt(preterminal)
    receipt_path = stage / RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    legacy_fields = {
        "config_sha256",
        "source_set_sha256",
        "prospective_tree_sha256",
        "data_sha256",
        "environment_identity",
        "environment_sha256",
        "stage5_gate_sha256",
    }
    receipt["immutable_bindings"] = {
        key: value
        for key, value in receipt["immutable_bindings"].items()
        if key in legacy_fields
    }
    receipt_path.write_bytes(_canonical_json(receipt))

    legacy = module.detect_stage6_terminal_recovery(stage_root=stage)
    assert legacy["status"] == "invalid"
    assert "immutable" in legacy["reason"]

    clean_authority = recovery_authority_factory("missing-resource")
    clean = _write_preterminal_stage(Path(clean_authority["run_root"]) / "s6")
    clean["execution_capability"] = clean_authority["capability"]
    _write_receipt(clean)
    missing_stage = Path(clean["stage"])
    (missing_stage / "resource_audit.jsonl").unlink()
    missing = module.detect_stage6_terminal_recovery(stage_root=missing_stage)
    assert missing["status"] == "invalid"
    assert missing["status"] != "no_terminal"


def test_detector_rejects_old_v2_terminal_even_when_receipt_bound(
    preterminal: dict[str, object],
) -> None:
    module = _module()
    stage = Path(preterminal["stage"])
    receipt_identity = _write_receipt(preterminal)
    _append_terminal(stage, receipt_identity)
    path = stage / "resource_audit.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[-1]["schema_version"] = "stage6_terminal_resource_evidence/v2"
    path.write_bytes(b"".join(_canonical_row(row) for row in rows))

    detection = module.detect_stage6_terminal_recovery(stage_root=stage)
    assert detection["status"] == "invalid"
    assert "terminal" in detection["reason"]


def test_current_i2_v3_without_preterminal_receipt_binding_is_invalid(
    preterminal: dict[str, object],
) -> None:
    module = _module()
    stage = Path(preterminal["stage"])
    receipt_identity = _write_receipt(preterminal)
    _append_terminal(stage, receipt_identity, include_receipt=False)
    detection = module.detect_stage6_terminal_recovery(stage_root=stage)
    assert detection["status"] == "invalid"
    assert "preterminal_acceptance" in detection["reason"]


def test_detector_is_read_only_and_accepts_only_strict_terminal_prefix(
    preterminal: dict[str, object],
) -> None:
    module = _module()
    stage = Path(preterminal["stage"])
    receipt_identity = _prepare_terminal(preterminal)
    before = _snapshot_files(stage)
    detection = module.detect_stage6_terminal_recovery(stage_root=stage)
    assert detection == {
        "status": "valid_terminal_recovery",
        "reason": "",
        "receipt_identity": receipt_identity,
        "phase_states": PRETERMINAL_STATES,
    }
    assert _snapshot_files(stage) == before


@pytest.mark.parametrize(
    "forgery",
    (
        "segment_id",
        "start_row_identity",
        "first_sample_count",
        "prior_segment_peak_and_run_peak",
    ),
)
def test_detector_recomputes_terminal_chain_and_run_peak_from_receipt_prefix(
    preterminal: dict[str, object],
    forgery: str,
) -> None:
    module = _module()
    stage = Path(preterminal["stage"])
    if forgery == "prior_segment_peak_and_run_peak":
        _append_second_resource_segment(stage)
    _refresh_semantic_binding(preterminal)
    receipt_identity = _write_receipt(preterminal)
    _append_terminal(stage, receipt_identity)
    path = stage / "resource_audit.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    terminal = rows[-1]
    if forgery == "segment_id":
        terminal["segment_id"] = "c" * 32
        terminal["segment_chain"][-1]["segment_id"] = "c" * 32
    elif forgery == "start_row_identity":
        terminal["segment_chain"][-1]["start_row_identity"]["sha256"] = "f" * 64
    elif forgery == "first_sample_count":
        terminal["segment_chain"][-1]["local_first_sample_count"] = 2
    elif forgery == "prior_segment_peak_and_run_peak":
        terminal["segment_chain"][0]["local_peak_rss_bytes"] = 250
        terminal["run_rss_lifecycle_peak_bytes"] = 250
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(forgery)
    path.write_bytes(b"".join(_canonical_row(row) for row in rows))

    detection = module.detect_stage6_terminal_recovery(stage_root=stage)
    assert detection["status"] == "invalid"
    assert any(
        token in detection["reason"]
        for token in ("resource lifecycle", "segment", "terminal")
    )


def test_detector_rejects_unknown_empty_directory_inserted_after_receipt(
    preterminal: dict[str, object],
) -> None:
    module = _module()
    stage = Path(preterminal["stage"])
    _prepare_terminal(preterminal)
    (stage / "unknown-empty-directory").mkdir()

    detection = module.detect_stage6_terminal_recovery(stage_root=stage)
    assert detection["status"] == "invalid"
    assert "directory" in detection["reason"]
    with pytest.raises(module.TerminalRecoveryError, match="directory|invalid"):
        module.recover_stage6_terminal_commit(
            stage_root=stage,
            manifest_committer=_manifest_committer,
            execution_capability=preterminal["execution_capability"],
        )


@pytest.mark.parametrize(
    "crash_event",
    tuple(f"after_artifact:{name}" for name in TERMINAL_ARTIFACT_NAMES)
    + (
        "after_phase:machine_passed",
        "after_phase:awaiting_independent_review",
        "after_manifest",
    ),
)
def test_each_file_phase_and_manifest_crash_point_converges(
    preterminal: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    crash_event: str,
) -> None:
    module = _module()
    stage = Path(preterminal["stage"])
    receipt_identity = _prepare_terminal(preterminal)
    crashed = False

    def inject(event: str) -> None:
        nonlocal crashed
        if event == crash_event and not crashed:
            crashed = True
            raise InjectedCrash(event)

    monkeypatch.setattr(module, "_terminal_recovery_event", inject)
    with pytest.raises(InjectedCrash, match=crash_event):
        module.recover_stage6_terminal_commit(
            stage_root=stage,
            manifest_committer=_manifest_committer,
            execution_capability=preterminal["execution_capability"],
        )

    monkeypatch.setattr(module, "_terminal_recovery_event", lambda event: None)
    result = module.recover_stage6_terminal_commit(
        stage_root=stage,
        manifest_committer=_manifest_committer,
        execution_capability=preterminal["execution_capability"],
    )
    assert result["receipt_identity"] == receipt_identity
    assert result["phase_states"] == COMPLETE_STATES
    assert result["manifest_identity"] == _identity(
        (stage / "manifest.json").read_bytes()
    )
    for name, payload in preterminal["artifacts"].items():
        assert (stage / name).read_bytes() == payload
    assert _phase_states(stage) == COMPLETE_STATES
    receipt = json.loads(
        (stage / RECEIPT_NAME).read_text(encoding="utf-8")
    )
    phase_rows = [
        json.loads(line)
        for line in (stage / "phase-state.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    for row in phase_rows[-2:]:
        assert set(row["bindings"]) == {
            *IMMUTABLE_BINDING_FIELDS,
            "checkpoint_sha256",
            "preterminal_acceptance_sha256",
            "preterminal_evidence_graph_sha256",
        }
        assert row["bindings"]["preterminal_acceptance_sha256"] == (
            receipt_identity["sha256"]
        )
        assert row["bindings"]["preterminal_evidence_graph_sha256"] == (
            receipt["preterminal_evidence_binding"]["graph_sha256"]
        )


def test_manifest_callback_crash_after_write_converges(
    preterminal: dict[str, object],
) -> None:
    module = _module()
    stage = Path(preterminal["stage"])
    _prepare_terminal(preterminal)
    crashed = False

    def crashing_committer(stage_root: Path) -> dict[str, object]:
        nonlocal crashed
        identity = _manifest_committer(stage_root)
        if not crashed:
            crashed = True
            raise InjectedCrash("manifest callback after write")
        return identity

    with pytest.raises(InjectedCrash, match="manifest callback"):
        module.recover_stage6_terminal_commit(
            stage_root=stage,
            manifest_committer=crashing_committer,
            execution_capability=preterminal["execution_capability"],
        )
    result = module.recover_stage6_terminal_commit(
        stage_root=stage,
        manifest_committer=_manifest_committer,
        execution_capability=preterminal["execution_capability"],
    )
    assert result["phase_states"] == COMPLETE_STATES
    assert (stage / "manifest.json").is_file()


def test_success_phase_rejects_hash_valid_wrong_evidence_binding(
    preterminal: dict[str, object],
) -> None:
    module = _module()
    stage = Path(preterminal["stage"])
    _prepare_terminal(preterminal)
    module.recover_stage6_terminal_commit(
        stage_root=stage,
        manifest_committer=_manifest_committer,
        execution_capability=preterminal["execution_capability"],
    )
    (stage / "manifest.json").unlink()
    phase_path = stage / "phase-state.jsonl"
    rows = [
        json.loads(line)
        for line in phase_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[-2]["bindings"]["preterminal_evidence_graph_sha256"] = "f" * 64
    for index in range(len(rows) - 2, len(rows)):
        rows[index]["previous_record_hash"] = rows[index - 1]["record_hash"]
        event = {
            "state": rows[index]["state"],
            "previous_record_hash": rows[index]["previous_record_hash"],
            "bindings": rows[index]["bindings"],
        }
        rows[index]["record_hash"] = hashlib.sha256(
            _canonical_json(event)
        ).hexdigest()
    phase_path.write_bytes(b"".join(_canonical_row(row) for row in rows))

    detection = module.detect_stage6_terminal_recovery(stage_root=stage)

    assert detection["status"] == "invalid"
    assert "phase" in str(detection["reason"]).lower()


@pytest.mark.parametrize(
    "drift",
    (
        "receipt_unknown_field",
        "terminal_receipt_identity",
        "receipt_resource_prefix",
        "terminal_full_prefix",
        "resource_prefix",
        "row_after_terminal",
        "phase_prefix",
        "original_artifact",
        "terminal_artifact",
        "unknown_file",
        "hardlink_graph_member",
        "phase_over_advanced",
    ),
)
def test_detector_and_recovery_reject_receipt_terminal_phase_and_artifact_drift(
    preterminal: dict[str, object],
    tmp_path: Path,
    drift: str,
) -> None:
    module = _module()
    stage = Path(preterminal["stage"])
    _prepare_terminal(preterminal)

    if drift == "receipt_unknown_field":
        path = stage / RECEIPT_NAME
        value = json.loads(path.read_text(encoding="utf-8"))
        value["unknown"] = True
        path.write_bytes(_canonical_json(value))
    elif drift == "terminal_receipt_identity":
        path = stage / "resource_audit.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[-1]["preterminal_acceptance"]["sha256"] = "f" * 64
        path.write_bytes(b"".join(_canonical_row(row) for row in rows))
    elif drift == "receipt_resource_prefix":
        path = stage / RECEIPT_NAME
        value = json.loads(path.read_text(encoding="utf-8"))
        value["resource_audit_prefix"]["sha256"] = "f" * 64
        path.write_bytes(_canonical_json(value))
    elif drift == "terminal_full_prefix":
        path = stage / "resource_audit.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[-1]["prior_resource_audit"]["size_bytes"] -= 1
        path.write_bytes(b"".join(_canonical_row(row) for row in rows))
    elif drift == "resource_prefix":
        path = stage / "resource_audit.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["root_pid"] += 1
        path.write_bytes(b"".join(_canonical_row(row) for row in rows))
    elif drift == "row_after_terminal":
        path = stage / "resource_audit.jsonl"
        path.write_bytes(path.read_bytes() + _canonical_row({"phase": "post-terminal"}))
    elif drift == "phase_prefix":
        path = stage / "phase-state.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rows[0]["state"] = "drifted"
        path.write_bytes(b"".join(_canonical_row(row) for row in rows))
    elif drift == "original_artifact":
        (stage / "config.json").write_bytes(b"{}\n")
    elif drift == "terminal_artifact":
        (stage / "summary.json").write_bytes(b"{}\n")
    elif drift == "unknown_file":
        (stage / "unknown.bin").write_bytes(b"unknown")
    elif drift == "hardlink_graph_member":
        source = stage / "config.json"
        alias = tmp_path / "config-hardlink.json"
        os.link(source, alias)
    elif drift == "phase_over_advanced":
        phase_path = stage / "phase-state.jsonl"
        rows = [json.loads(line) for line in phase_path.read_text(encoding="utf-8").splitlines()]
        previous = rows[-1]["record_hash"]
        bindings = {
            **preterminal["immutable"],
            "checkpoint_sha256": preterminal["global_checkpoint"]["checkpoint_sha256"],
        }
        rows.extend(
            [
                _phase_row("machine_passed", previous, bindings),
            ]
        )
        previous = rows[-1]["record_hash"]
        rows.append(_phase_row("awaiting_independent_review", previous, bindings))
        previous = rows[-1]["record_hash"]
        rows.append(_phase_row("unexpected", previous, bindings))
        phase_path.write_bytes(b"".join(_canonical_row(row) for row in rows))
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(drift)

    detection = module.detect_stage6_terminal_recovery(stage_root=stage)
    assert detection["status"] == "invalid"
    with pytest.raises(module.TerminalRecoveryError, match="invalid|drift|prefix|artifact|receipt|terminal"):
        module.recover_stage6_terminal_commit(
            stage_root=stage,
            manifest_committer=_manifest_committer,
            execution_capability=preterminal["execution_capability"],
        )


def test_existing_manifest_requires_pure_verifier_and_manifest_drift_is_rejected(
    preterminal: dict[str, object],
) -> None:
    module = _module()
    stage = Path(preterminal["stage"])
    _prepare_terminal(preterminal)
    module.recover_stage6_terminal_commit(
        stage_root=stage,
        manifest_committer=_manifest_committer,
        execution_capability=preterminal["execution_capability"],
    )
    without_verifier = module.detect_stage6_terminal_recovery(stage_root=stage)
    assert without_verifier["status"] == "invalid"
    assert "manifest" in without_verifier["reason"]

    valid = module.detect_stage6_terminal_recovery(
        stage_root=stage,
        manifest_verifier=_manifest_committer,
    )
    assert valid["status"] == "valid_terminal_recovery"
    (stage / "manifest.json").write_bytes(b"{}\n")
    invalid = module.detect_stage6_terminal_recovery(
        stage_root=stage,
        manifest_verifier=_manifest_committer,
    )
    assert invalid["status"] == "invalid"


def test_repeated_recovery_is_byte_identical_and_phase_idempotent(
    preterminal: dict[str, object],
) -> None:
    module = _module()
    stage = Path(preterminal["stage"])
    _prepare_terminal(preterminal)
    first = module.recover_stage6_terminal_commit(
        stage_root=stage,
        manifest_committer=_manifest_committer,
        execution_capability=preterminal["execution_capability"],
    )
    first_bytes = _snapshot_files(stage)
    second = module.recover_stage6_terminal_commit(
        stage_root=stage,
        manifest_committer=_manifest_committer,
        execution_capability=preterminal["execution_capability"],
    )
    third = module.recover_stage6_terminal_commit(
        stage_root=stage,
        manifest_committer=_manifest_committer,
        execution_capability=preterminal["execution_capability"],
    )
    assert second == first == third
    assert _snapshot_files(stage) == first_bytes
    assert _phase_states(stage) == COMPLETE_STATES


def test_fresh_python_pid_recovery_without_capability_fails_before_workload_or_mutation(
    preterminal: dict[str, object],
) -> None:
    stage = Path(preterminal["stage"])
    _prepare_terminal(preterminal)
    before = _snapshot_files(stage)
    script = textwrap.dedent(
        r'''
        import builtins
        import hashlib
        import json
        import os
        import sys
        import types
        from pathlib import Path

        counts = {
            "torch_import": 0,
            "policy": 0,
            "AdamW": 0,
            "CheckpointManager": 0,
            "SpawnVectorEnv": 0,
            "manifest_committer": 0,
        }

        def constructor(name):
            class Spy:
                def __init__(self, *args, **kwargs):
                    counts[name] += 1
                    raise AssertionError(f"forbidden constructor: {name}")
            return Spy

        torch = types.ModuleType("torch")
        optim = types.ModuleType("torch.optim")
        optim.AdamW = constructor("AdamW")
        torch.optim = optim
        sys.modules["torch"] = torch
        sys.modules["torch.optim"] = optim

        checkpoint = types.ModuleType("lunar_exploration_ppo.ppo.checkpoint")
        checkpoint.CheckpointManager = constructor("CheckpointManager")
        sys.modules[checkpoint.__name__] = checkpoint
        environment = types.ModuleType("lunar_exploration_ppo.env.standard_training")
        environment.SpawnVectorEnv = constructor("SpawnVectorEnv")
        sys.modules[environment.__name__] = environment
        policy = types.ModuleType("lunar_exploration_ppo.policy.cross_attention")
        policy.CrossAttentionFrontierPolicy = constructor("policy")
        sys.modules[policy.__name__] = policy

        real_import = builtins.__import__
        forbidden_imports = {
            "torch",
            "lunar_exploration_ppo.ppo.checkpoint",
            "lunar_exploration_ppo.ppo.standard_training",
            "lunar_exploration_ppo.ppo.collector",
            "lunar_exploration_ppo.ppo.trainer",
            "lunar_exploration_ppo.env.standard_training",
            "lunar_exploration_ppo.policy.cross_attention",
        }

        def import_spy(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torch" or name.startswith("torch."):
                counts["torch_import"] += 1
                raise AssertionError(f"forbidden import: {name}")
            if name in forbidden_imports:
                raise AssertionError(f"forbidden import: {name}")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = import_spy
        from lunar_exploration_ppo.workflows.stage6_terminal_recovery import (
            recover_stage6_terminal_commit,
        )

        def canonical_json(value):
            return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")

        def identity(payload):
            return {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}

        def manifest_committer(stage_root):
            counts["manifest_committer"] += 1
            stage = Path(stage_root)
            manifest_path = stage / "manifest.json"
            entries = []
            for path in sorted(
                (item for item in stage.rglob("*") if item.is_file() and item != manifest_path),
                key=lambda item: item.relative_to(stage).as_posix(),
            ):
                payload = path.read_bytes()
                entries.append({"path": path.relative_to(stage).as_posix(), **identity(payload)})
            payload = canonical_json({"schema_version": "stage6_test_terminal_manifest/v1", "artifacts": entries})
            if manifest_path.exists():
                if manifest_path.read_bytes() != payload:
                    raise RuntimeError("manifest drifted")
            else:
                with manifest_path.open("xb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            return identity(payload)

        result = None
        error = None
        try:
            result = recover_stage6_terminal_commit(
                stage_root=Path(sys.argv[1]),
                manifest_committer=manifest_committer,
            )
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)}
        print(json.dumps({
            "pid": os.getpid(),
            "counts": counts,
            "result": result,
            "error": error,
        }, sort_keys=True))
        '''
    )
    env = os.environ.copy()
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", script, str(stage)],
        cwd=ROOT,
        env=env,
        shell=False,
        timeout=30,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert completed.returncode == 0, completed.stderr
    evidence = json.loads(completed.stdout.strip().splitlines()[-1])
    assert evidence["pid"] != os.getpid()
    assert evidence["counts"] == {
        "torch_import": 0,
        "policy": 0,
        "AdamW": 0,
        "CheckpointManager": 0,
        "SpawnVectorEnv": 0,
        "manifest_committer": 0,
    }
    assert evidence["result"] is None
    assert evidence["error"]["type"] == "TerminalRecoveryError"
    assert "capability" in evidence["error"]["message"]
    assert _snapshot_files(stage) == before
    assert _phase_states(stage) == PRETERMINAL_STATES
