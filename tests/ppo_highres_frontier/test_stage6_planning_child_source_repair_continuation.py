from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import inspect
import json
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from lunar_exploration_ppo.utils.durable_jsonl import RunLease
from test_stage6_planning_child_source_repair import (
    _build,
    _fixture,
)
from test_stage6_workflow import (
    _append_task5_resource_pre,
    _append_task5_resource_segment,
    _build_task5_planning_child_machine_fixture,
    _prepare_task5_accepted_u85_restart_tail,
    _task5_review_record,
)


def _continuation_module():
    return importlib.import_module(
        "lunar_exploration_ppo.workflows."
        "stage6_planning_child_source_repair_continuation"
    )


def _continuation_cli_module() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "create_ppo_stage6_planning_child_source_repair_continuation.py"
    )
    spec = importlib.util.spec_from_file_location(
        "stage6_planning_child_source_repair_continuation_f4_cli",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _recovery_anchor(fixture):
    from lunar_exploration_ppo.workflows.stage6_planning_child_recovery import (
        inspect_planning_child_recovery_anchor,
    )

    return inspect_planning_child_recovery_anchor(
        stage_root=fixture.stage,
        parent_artifact_path=fixture.child_path,
    )


def _publish_continuation(
    artifact: dict[str, object],
    *,
    fixture,
    identity: Mapping[str, object],
    review: Mapping[str, object],
    immutable: Mapping[str, object],
    output_path: Path,
) -> Path:
    module = _continuation_module()
    authorization_path = Path(
        str(artifact["current"]["authorization"]["path"])
    )
    authorization, stage5 = _r5_exact_publish_owners(
        fixture=fixture,
        review=review,
        authorization_path=authorization_path,
    )
    from lunar_exploration_ppo.workflows.stage6 import (
        FrozenStage5AuthorityHandle,
    )
    from lunar_exploration_ppo.workflows.stage6_review_authorization import (
        Stage6ReviewAuthorizationHandle,
    )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            Stage6ReviewAuthorizationHandle,
            "require_current",
            lambda self, _label: None,
        )
        patch.setattr(
            FrozenStage5AuthorityHandle,
            "require_current",
            lambda self, _label="Stage 5 authority": None,
        )
        with RunLease(fixture.stage.parent / ".stage6.lease") as run_lease:
            capability = (
                module.issue_planning_child_source_repair_continuation_publish_capability(
                    artifact,
                    stage_root=fixture.stage,
                    anchor=_recovery_anchor(fixture),
                    current_execution_identity=identity,
                    current_verified_review_authorization=review,
                    current_immutable_bindings=immutable,
                    authorization_currentness=authorization,
                    stage5_authority_currentness=stage5,
                    effective_config_bytes=(
                        fixture.stage / "config.json"
                    ).read_bytes(),
                    run_lease=run_lease,
                )
            )
            return module.publish_planning_child_source_repair_continuation_artifact(
                capability,
                stage_root=fixture.stage,
                output_path=output_path,
                run_lease=run_lease,
            )


def _current_successor(
    fixture,
) -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
    Path,
]:
    identity = copy.deepcopy(fixture.current_identity)
    prospective_tree = "8" * 40
    identity.update(
        {
            "prospective_git_tree": prospective_tree,
            "prospective_tree_sha256": hashlib.sha256(
                prospective_tree.encode("ascii")
            ).hexdigest(),
            "changed_path_set_sha256": "8" * 64,
            "source_set_sha256": "9" * 64,
        }
    )
    source_identity = identity.get("source_identity")
    assert isinstance(source_identity, dict)
    source_identity["source_set_sha256"] = identity["source_set_sha256"]
    authorization_path = (
        fixture.child_path.parent.parent
        / "review"
        / "successor"
        / "launch-authorization.json"
    )
    authorization_payload = fixture.module.ArtifactStore.canonical_json_bytes(
        {
            "authorized": True,
            "formal_run_id": fixture.stage.parent.name,
            "fixture": "planning-child-successor",
        }
    )
    authorization_path.parent.mkdir(parents=True)
    authorization_path.write_bytes(authorization_payload)
    review = _task5_review_record(
        fixture.module,
        execution_identity=identity,
        formal_run_id=fixture.stage.parent.name,
        authorization_payload=authorization_payload,
    )
    immutable = fixture.module._stage6_current_immutable_bindings(
        execution_identity=identity,
        verified_review_authorization=review,
        stage5_authority=fixture.authority,
    )
    return identity, review, immutable, authorization_path


def _continuation_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    retained_checkpoint_updates: tuple[int, ...] = (84,),
):
    fixture = _build_task5_planning_child_machine_fixture(
        tmp_path,
        monkeypatch,
        retained_checkpoint_updates=retained_checkpoint_updates,
    )
    tail = _prepare_task5_accepted_u85_restart_tail(fixture)
    fixture.continuation_path.unlink()
    identity, review, immutable, authorization_path = _current_successor(
        fixture
    )
    evidence_root = authorization_path.parent
    report_path = evidence_root / "r9-r4-implementer-report.md"
    report_path.write_text("# R9 R4\n\nGREEN\n", encoding="utf-8")
    spec_path = evidence_root / "r9-r4-spec-rereview.md"
    quality_path = evidence_root / "r9-r4-quality-rereview.md"
    spec_path.write_text("# R4 spec rereview\n\nC0/I1\n", encoding="utf-8")
    quality_path.write_text(
        "# R4 quality rereview\n\nC0/I1\n",
        encoding="utf-8",
    )
    review = copy.deepcopy(review)
    review["spec_review_sha256"] = hashlib.sha256(
        spec_path.read_bytes()
    ).hexdigest()
    review["quality_review_sha256"] = hashlib.sha256(
        quality_path.read_bytes()
    ).hexdigest()
    immutable = fixture.module._stage6_current_immutable_bindings(
        execution_identity=identity,
        verified_review_authorization=review,
        stage5_authority=fixture.authority,
    )
    artifact = _continuation_module().build_planning_child_source_repair_continuation_artifact(
        stage_root=fixture.stage,
        parent_path=fixture.child_path,
        anchor=_recovery_anchor(fixture),
        current_execution_identity=identity,
        current_verified_review_authorization=review,
        current_immutable_bindings=immutable,
        current_review_authorization_path=authorization_path,
        implementation_report_path=report_path,
        spec_review_path=spec_path,
        quality_review_path=quality_path,
        created_at_utc="2026-07-26T00:00:00Z",
    )
    return (
        fixture,
        tail,
        identity,
        review,
        immutable,
        authorization_path,
        artifact,
    )


def _f4_preview_publish_capability(
    *,
    fixture,
    identity: Mapping[str, object],
    review: Mapping[str, object],
    immutable: Mapping[str, object],
    authorization_path: Path,
    run_lease: RunLease,
    monkeypatch: pytest.MonkeyPatch,
) -> Mapping[str, object]:
    cli = _continuation_cli_module()
    from lunar_exploration_ppo.workflows.stage6 import (
        FrozenStage5AuthorityHandle,
    )
    from lunar_exploration_ppo.workflows.stage6_review_authorization import (
        Stage6ReviewAuthorizationHandle,
    )

    evidence_root = authorization_path.parent
    source_member = evidence_root / "execution-source-member.py"
    source_member.write_bytes(b"source-member-v1\n")
    stage5_gate = evidence_root / "stage5-gate.json"
    stage5_gate.write_bytes(b'{"gate":"v1"}\n')
    effective_config_bytes = (fixture.stage / "config.json").read_bytes()
    authorization_handle = Stage6ReviewAuthorizationHandle(
        _authorization_path=authorization_path.resolve(),
        _repo_root=cli.DEFAULT_REPO_ROOT.resolve(),
        _config_path=cli.DEFAULT_CONFIG_PATH.resolve(),
        _effective_config_bytes=effective_config_bytes,
        _formal_run_id=fixture.stage.parent.name,
        _record_payload=fixture.module.ArtifactStore.canonical_json_bytes(
            review
        ),
        _members=(),
    )
    stage5_authority_handle = FrozenStage5AuthorityHandle(
        identity=fixture.authority.identity,
        snapshots=(),
        stage5_root=evidence_root,
        repo_root=cli.DEFAULT_REPO_ROOT.resolve(),
    )
    authorization_snapshots = tuple(
        (path, path.read_bytes())
        for path in (fixture.stage / "config.json", source_member)
    )
    stage5_snapshots = ((stage5_gate, stage5_gate.read_bytes()),)

    def require_authorization_current(
        self: Stage6ReviewAuthorizationHandle,
        _label: str,
    ) -> None:
        assert self is authorization_handle
        for path, payload in authorization_snapshots:
            if path.read_bytes() != payload:
                raise RuntimeError(
                    f"currentness input changed: {path.name}"
                )

    def require_stage5_current(
        self: FrozenStage5AuthorityHandle,
        _label: str = "Stage 5 authority",
    ) -> None:
        assert self is stage5_authority_handle
        for path, payload in stage5_snapshots:
            if path.read_bytes() != payload:
                raise RuntimeError(
                    f"currentness input changed: {path.name}"
                )

    monkeypatch.setattr(
        Stage6ReviewAuthorizationHandle,
        "require_current",
        require_authorization_current,
    )
    monkeypatch.setattr(
        FrozenStage5AuthorityHandle,
        "require_current",
        require_stage5_current,
    )

    def verify_authorization(**kwargs):
        assert Path(kwargs["repo_root"]).resolve() == (
            cli.DEFAULT_REPO_ROOT.resolve()
        )
        assert Path(kwargs["config_path"]).resolve() == (
            cli.DEFAULT_CONFIG_PATH.resolve()
        )
        assert kwargs["effective_config_bytes"] == effective_config_bytes
        return authorization_handle

    monkeypatch.setattr(
        cli,
        "verify_stage6_review_launch_authorization",
        verify_authorization,
    )
    monkeypatch.setattr(
        cli,
        "stage6_execution_identity",
        lambda **_kwargs: copy.deepcopy(identity),
    )
    monkeypatch.setattr(
        cli,
        "validate_stage6_verified_review_authorization",
        lambda *_args, **_kwargs: copy.deepcopy(review),
    )
    monkeypatch.setattr(
        cli,
        "verify_frozen_stage5_authority",
        lambda **_kwargs: stage5_authority_handle,
    )
    monkeypatch.setattr(
        cli,
        "_stage6_current_immutable_bindings",
        lambda **_kwargs: copy.deepcopy(immutable),
    )
    monkeypatch.setattr(
        cli,
        "_created_at_utc",
        lambda: "2026-07-26T00:00:00Z",
    )
    return cli.create_planning_child_source_repair_continuation_preview(
        stage_root=fixture.stage,
        run_id=fixture.stage.parent.name,
        parent_path=fixture.child_path,
        review_authorization_path=authorization_path,
        implementation_report_path=(
            evidence_root / "r9-r4-implementer-report.md"
        ),
        spec_review_path=evidence_root / "r9-r4-spec-rereview.md",
        quality_review_path=evidence_root / "r9-r4-quality-rereview.md",
        run_lease=run_lease,
        stage5_gate_path=stage5_gate,
    )


def _same_size_drift(path: Path) -> None:
    payload = path.read_bytes()
    assert payload
    replacement = bytes((payload[0] ^ 1,)) + payload[1:]
    assert len(replacement) == len(payload)
    path.write_bytes(replacement)


def _r5_exact_publish_owners(
    *,
    fixture,
    review: Mapping[str, object],
    authorization_path: Path,
    unrelated: bool = False,
):
    from lunar_exploration_ppo.workflows.stage6 import (
        FrozenStage5AuthorityHandle,
    )
    from lunar_exploration_ppo.workflows.stage6_review_authorization import (
        Stage6ReviewAuthorizationHandle,
    )

    bound_review = copy.deepcopy(dict(review))
    formal_run_id = fixture.stage.parent.name
    if unrelated:
        bound_review["formal_run_id"] = (
            "s6-standard-single-r1-20260724T000125Z"
        )
        formal_run_id = str(bound_review["formal_run_id"])
    repo_root = Path(__file__).resolve().parents[2]
    config_payload = (fixture.stage / "config.json").read_bytes()
    authorization = Stage6ReviewAuthorizationHandle(
        _authorization_path=authorization_path.resolve(),
        _repo_root=repo_root,
        _config_path=(
            repo_root / "configs/ppo_highres_frontier_stage6_v1.json"
        ).resolve(),
        _effective_config_bytes=config_payload,
        _formal_run_id=formal_run_id,
        _record_payload=fixture.module.ArtifactStore.canonical_json_bytes(
            bound_review
        ),
        _members=(),
    )
    stage5_identity = copy.deepcopy(dict(fixture.authority.identity))
    if unrelated:
        stage5_identity["gate_sha256"] = "f" * 64
    stage5 = FrozenStage5AuthorityHandle(
        identity=stage5_identity,
        snapshots=(),
        stage5_root=fixture.stage,
        repo_root=Path(__file__).resolve().parents[2],
    )
    return authorization, stage5


@pytest.mark.parametrize(
    "owner_case",
    ("duck_typed", "unrelated_exact", "config_drift"),
)
def test_r5_publish_requires_exact_bound_owners_and_stage_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner_case: str,
) -> None:
    (
        fixture,
        _tail,
        identity,
        review,
        immutable,
        authorization_path,
        artifact,
    ) = _continuation_preview(tmp_path, monkeypatch)
    module = _continuation_module()
    output = (
        fixture.stage
        / module.PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_NAME
    )
    authorization, stage5 = _r5_exact_publish_owners(
        fixture=fixture,
        review=review,
        authorization_path=authorization_path,
        unrelated=owner_case == "unrelated_exact",
    )
    from lunar_exploration_ppo.workflows.stage6 import (
        FrozenStage5AuthorityHandle,
    )
    from lunar_exploration_ppo.workflows.stage6_review_authorization import (
        Stage6ReviewAuthorizationHandle,
    )

    monkeypatch.setattr(
        Stage6ReviewAuthorizationHandle,
        "require_current",
        lambda self, _label: None,
    )
    monkeypatch.setattr(
        FrozenStage5AuthorityHandle,
        "require_current",
        lambda self, _label="Stage 5 authority": None,
    )
    if owner_case == "duck_typed":
        authorization = SimpleNamespace(
            require_current=lambda _label: None
        )
        stage5 = SimpleNamespace(require_current=lambda _label: None)

    with RunLease(fixture.stage.parent / ".stage6.lease") as run_lease:
        if owner_case == "config_drift":
            capability = (
                module.issue_planning_child_source_repair_continuation_publish_capability(
                    artifact,
                    stage_root=fixture.stage,
                    anchor=_recovery_anchor(fixture),
                    current_execution_identity=identity,
                    current_verified_review_authorization=review,
                    current_immutable_bindings=immutable,
                    authorization_currentness=authorization,
                    stage5_authority_currentness=stage5,
                    effective_config_bytes=(
                        fixture.stage / "config.json"
                    ).read_bytes(),
                    run_lease=run_lease,
                )
            )
            _same_size_drift(fixture.stage / "config.json")
            with pytest.raises(
                module.Stage6PlanningChildSourceRepairContinuationError,
                match="config|current|changed|drift",
            ):
                module.publish_planning_child_source_repair_continuation_artifact(
                    capability,
                    stage_root=fixture.stage,
                    output_path=output,
                    run_lease=run_lease,
                )
        else:
            with pytest.raises(
                module.Stage6PlanningChildSourceRepairContinuationError,
                match="owner|authorization|Stage5|current|exact|drift",
            ):
                module.issue_planning_child_source_repair_continuation_publish_capability(
                    artifact,
                    stage_root=fixture.stage,
                    anchor=_recovery_anchor(fixture),
                    current_execution_identity=identity,
                    current_verified_review_authorization=review,
                    current_immutable_bindings=immutable,
                    authorization_currentness=authorization,
                    stage5_authority_currentness=stage5,
                    effective_config_bytes=(
                        fixture.stage / "config.json"
                    ).read_bytes(),
                    run_lease=run_lease,
                )

    assert not output.exists()


class _FileCurrentnessOwner:
    def __init__(
        self,
        *paths: Path,
        record: Mapping[str, object] | None = None,
        identity: Mapping[str, object] | None = None,
    ) -> None:
        self._snapshots = tuple(
            (path, path.read_bytes()) for path in paths
        )
        self._record = (
            None if record is None else copy.deepcopy(dict(record))
        )
        self.identity = (
            {} if identity is None else copy.deepcopy(dict(identity))
        )

    def canonical_record(self) -> dict[str, object]:
        assert self._record is not None
        return copy.deepcopy(self._record)

    def require_current(self, _label: str) -> None:
        for path, payload in self._snapshots:
            if path.read_bytes() != payload:
                raise RuntimeError(f"currentness input changed: {path.name}")


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _plain_json(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    return value


@pytest.mark.parametrize(
    ("drifted_name", "expected_message"),
    (
        ("spec", "spec"),
        ("quality", "quality"),
    ),
)
def test_continuation_rejects_review_evidence_not_bound_to_current_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drifted_name: str,
    expected_message: str,
) -> None:
    (
        fixture,
        _tail,
        identity,
        review,
        immutable,
        authorization_path,
        _artifact,
    ) = _continuation_preview(tmp_path, monkeypatch)
    evidence_root = authorization_path.parent
    report_path = evidence_root / "r9-r4-implementer-report.md"
    spec_path = evidence_root / "r9-r4-spec-rereview.md"
    quality_path = evidence_root / "r9-r4-quality-rereview.md"
    drifted_path = evidence_root / f"unbound-{drifted_name}-rereview.md"
    drifted_path.write_text(
        f"# Unbound {drifted_name} rereview\n\nC0/I0\n",
        encoding="utf-8",
    )
    if drifted_name == "spec":
        spec_path = drifted_path
    else:
        quality_path = drifted_path

    module = _continuation_module()
    with pytest.raises(
        module.Stage6PlanningChildSourceRepairContinuationError,
        match=expected_message,
    ):
        module.build_planning_child_source_repair_continuation_artifact(
            stage_root=fixture.stage,
            parent_path=fixture.child_path,
            anchor=_recovery_anchor(fixture),
            current_execution_identity=identity,
            current_verified_review_authorization=review,
            current_immutable_bindings=immutable,
            current_review_authorization_path=authorization_path,
            implementation_report_path=report_path,
            spec_review_path=spec_path,
            quality_review_path=quality_path,
            created_at_utc="2026-07-26T00:00:00Z",
        )


def test_continuation_review_binding_ignores_historical_r4_basename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        fixture,
        _tail,
        identity,
        review,
        _immutable,
        authorization_path,
        _artifact,
    ) = _continuation_preview(tmp_path, monkeypatch)
    evidence_root = authorization_path.parent
    report_path = evidence_root / "r9-r4-implementer-report.md"
    spec_path = (
        evidence_root
        / "stage6-planning-child-source-repair-r9-r4-spec-rereview.md"
    )
    quality_path = evidence_root / "r9-r4-quality-rereview.md"
    spec_payload = b"# Newly authorized successor spec review\n\nC0/I0\n"
    spec_path.write_bytes(spec_payload)
    review = copy.deepcopy(review)
    review["spec_review_sha256"] = hashlib.sha256(
        spec_payload
    ).hexdigest()
    immutable = fixture.module._stage6_current_immutable_bindings(
        execution_identity=identity,
        verified_review_authorization=review,
        stage5_authority=fixture.authority,
    )

    artifact = (
        _continuation_module()
        .build_planning_child_source_repair_continuation_artifact(
            stage_root=fixture.stage,
            parent_path=fixture.child_path,
            anchor=_recovery_anchor(fixture),
            current_execution_identity=identity,
            current_verified_review_authorization=review,
            current_immutable_bindings=immutable,
            current_review_authorization_path=authorization_path,
            implementation_report_path=report_path,
            spec_review_path=spec_path,
            quality_review_path=quality_path,
            created_at_utc="2026-07-26T00:00:00Z",
        )
    )

    assert artifact["review_evidence"]["spec_rereview"] == {
        "path": spec_path.resolve().as_posix(),
        "size_bytes": len(spec_payload),
        "sha256": hashlib.sha256(spec_payload).hexdigest(),
    }


def _publish_successor_chain(
    fixture,
    *,
    identity: dict[str, object],
    review: dict[str, object],
    immutable: dict[str, object],
    artifact: dict[str, object],
) -> Path:
    module = _continuation_module()
    continuation = (
        fixture.stage
        / module.PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_NAME
    )
    _publish_continuation(
        artifact,
        fixture=fixture,
        identity=identity,
        review=review,
        immutable=immutable,
        output_path=continuation,
    )
    fixture.continuation_path = continuation
    fixture.successor_identity = identity
    fixture.successor_review = review
    fixture.current_immutable = immutable
    return continuation


def test_parent_amendment_remains_fail_closed_for_a_new_current_identity(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    artifact = _build(fixture)
    new_identity = copy.deepcopy(fixture.current_execution_identity)
    new_identity["source_set_sha256"] = "9" * 64

    assert not hasattr(
        _continuation_module(),
        "load_planning_child_source_repair_chain",
    )

    with pytest.raises(
        importlib.import_module(
            "lunar_exploration_ppo.workflows."
            "stage6_planning_child_source_repair"
        ).Stage6PlanningChildSourceRepairError,
        match="current execution identity",
    ):
        importlib.import_module(
            "lunar_exploration_ppo.workflows."
            "stage6_planning_child_source_repair"
        ).validate_planning_child_source_repair_artifact(
            artifact,
            stage_root=fixture.stage_root,
            current_execution_identity=new_identity,
            current_verified_review_authorization=(
                fixture.current_verified_review_authorization
            ),
            current_immutable_bindings=fixture.current_immutable_bindings,
            require_exact_prefix=True,
    )


def test_write_once_continuation_bridges_accepted_u85_to_exact_u86(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        fixture,
        _tail,
        identity,
        review,
        immutable,
        authorization_path,
        artifact,
    ) = _continuation_preview(tmp_path, monkeypatch)
    module = _continuation_module()
    output = (
        fixture.stage
        / module.PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_NAME
    )
    frozen_paths = (
        fixture.child_path,
        fixture.stage
        / "checkpoints"
        / f"seed-{fixture.transactions[10].seed}"
        / "update-00000084"
        / "checkpoint.pt",
        fixture.stage
        / "checkpoints"
        / f"seed-{fixture.transactions[10].seed}"
        / "update-00000085"
        / "checkpoint.pt",
        fixture.stage / "checkpoints" / "index.jsonl",
        fixture.stage / "job-state.jsonl",
        fixture.stage / "resource_audit.jsonl",
        fixture.stage / "training_metrics.jsonl",
        fixture.stage / "validation_metrics.jsonl",
    )
    frozen = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in frozen_paths
    }

    assert not output.exists()
    published = _publish_continuation(
        artifact,
        fixture=fixture,
        identity=identity,
        review=review,
        immutable=immutable,
        output_path=output,
    )
    assert published == output
    validated, continuation_sha256 = (
        module.load_planning_child_source_repair_continuation_artifact(
            output,
            stage_root=fixture.stage,
            current_execution_identity=identity,
            current_verified_review_authorization=review,
            current_immutable_bindings=immutable,
        )
    )

    assert validated.parent_artifact_sha256 == hashlib.sha256(
        fixture.child_path.read_bytes()
    ).hexdigest()
    assert continuation_sha256 == hashlib.sha256(
        output.read_bytes()
    ).hexdigest()
    assert artifact["schema_version"] == (
        "stage6_planning_child_source_repair_continuation/v2"
    )
    assert set(artifact["parent"]) == {
        "path",
        "size_bytes",
        "sha256",
        "canonical_sha256",
    }
    assert set(artifact["current"]) == {
        "execution_identity_sha256",
        "authorization",
        "immutable_bindings_sha256",
    }
    assert "origin" not in artifact["parent"]
    assert "execution_identity" not in artifact["current"]
    assert validated.accepted_anchor["last_accepted_update"] == 85
    assert validated.lineage_epoch["first_update"] == 86
    assert not hasattr(validated, "effective_next_update")
    assert {
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in frozen_paths
    } == set(frozen.values())
    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == digest
        for path, digest in frozen.items()
    )

    with pytest.raises(Exception, match="already exists|exclusive|publish"):
        _publish_continuation(
            artifact,
            fixture=fixture,
            identity=identity,
            review=review,
            immutable=immutable,
            output_path=output,
        )


def test_f4_preview_returns_immutable_mapping_publish_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        fixture,
        _tail,
        identity,
        review,
        immutable,
        authorization_path,
        artifact,
    ) = _continuation_preview(tmp_path, monkeypatch)
    module = _continuation_module()
    output = (
        fixture.stage
        / module.PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_NAME
    )

    with RunLease(fixture.stage.parent / ".stage6.lease") as run_lease:
        capability = _f4_preview_publish_capability(
            fixture=fixture,
            identity=identity,
            review=review,
            immutable=immutable,
            authorization_path=authorization_path,
            run_lease=run_lease,
            monkeypatch=monkeypatch,
        )

        assert isinstance(capability, Mapping)
        assert type(capability) is not dict
        with pytest.raises(TypeError):
            capability["mode"] = "drifted"  # type: ignore[index]
        parent = capability["parent"]
        assert isinstance(parent, Mapping)
        with pytest.raises(TypeError):
            parent["sha256"] = "f" * 64  # type: ignore[index]
        assert (
            fixture.module.ArtifactStore.canonical_json_bytes(
                _plain_json(capability)
            )
            == fixture.module.ArtifactStore.canonical_json_bytes(artifact)
        )

        assert (
            module.publish_planning_child_source_repair_continuation_artifact(
                capability,
                stage_root=fixture.stage,
                output_path=output,
                run_lease=run_lease,
            )
            == output
        )

    assert output.read_bytes() == (
        fixture.module.ArtifactStore.canonical_json_bytes(artifact)
    )


def test_f4_publication_rejects_static_artifact_without_preview_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        fixture,
        _tail,
        identity,
        review,
        immutable,
        authorization_path,
        artifact,
    ) = _continuation_preview(tmp_path, monkeypatch)
    module = _continuation_module()
    output = (
        fixture.stage
        / module.PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_NAME
    )

    with RunLease(fixture.stage.parent / ".stage6.lease") as run_lease:
        with pytest.raises(
            module.Stage6PlanningChildSourceRepairContinuationError,
            match="publish capability",
        ):
            module.publish_planning_child_source_repair_continuation_artifact(
                artifact,
                stage_root=fixture.stage,
                output_path=output,
                run_lease=run_lease,
            )

    assert not output.exists()


def test_f4_publication_requires_same_run_lease_object_as_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        fixture,
        _tail,
        identity,
        review,
        immutable,
        authorization_path,
        _artifact,
    ) = _continuation_preview(tmp_path, monkeypatch)
    module = _continuation_module()
    output = (
        fixture.stage
        / module.PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_NAME
    )

    with RunLease(fixture.stage.parent / ".stage6.lease") as preview_lease:
        capability = _f4_preview_publish_capability(
            fixture=fixture,
            identity=identity,
            review=review,
            immutable=immutable,
            authorization_path=authorization_path,
            run_lease=preview_lease,
            monkeypatch=monkeypatch,
        )
    with RunLease(fixture.stage.parent / ".stage6.lease") as replacement_lease:
        with pytest.raises(
            module.Stage6PlanningChildSourceRepairContinuationError,
            match="same RunLease|publish capability",
        ):
            module.publish_planning_child_source_repair_continuation_artifact(
                capability,
                stage_root=fixture.stage,
                output_path=output,
                run_lease=replacement_lease,
            )

    assert not output.exists()


@pytest.mark.parametrize(
    "drifted_input",
    (
        "parent",
        "authorization",
        "implementation_report",
        "spec_rereview",
        "quality_rereview",
        "accepted_checkpoint",
        "journal_prefix",
        "implementation_report_identity",
    ),
)
def test_f4_publication_rechecks_every_preview_bound_input_before_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drifted_input: str,
) -> None:
    (
        fixture,
        _tail,
        identity,
        review,
        immutable,
        authorization_path,
        _artifact,
    ) = _continuation_preview(tmp_path, monkeypatch)
    module = _continuation_module()
    output = (
        fixture.stage
        / module.PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_NAME
    )

    with RunLease(fixture.stage.parent / ".stage6.lease") as run_lease:
        capability = _f4_preview_publish_capability(
            fixture=fixture,
            identity=identity,
            review=review,
            immutable=immutable,
            authorization_path=authorization_path,
            run_lease=run_lease,
            monkeypatch=monkeypatch,
        )
        if drifted_input == "parent":
            path = Path(str(capability["parent"]["path"]))
        elif drifted_input == "authorization":
            path = Path(
                str(capability["current"]["authorization"]["path"])
            )
        elif drifted_input in {
            "implementation_report",
            "spec_rereview",
            "quality_rereview",
            "implementation_report_identity",
        }:
            evidence_label = (
                "implementation_report"
                if drifted_input == "implementation_report_identity"
                else drifted_input
            )
            path = Path(
                str(
                    capability["review_evidence"][evidence_label]["path"]
                )
            )
        elif drifted_input == "accepted_checkpoint":
            path = Path(
                str(
                    capability["accepted_anchor"][
                        "latest_complete_checkpoint"
                    ]["members"]["checkpoint.pt"]["path"]
                )
            )
        else:
            path = fixture.stage / "training_metrics.jsonl"

        if drifted_input == "implementation_report_identity":
            replacement = path.with_name(f"{path.name}.replacement")
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
        else:
            _same_size_drift(path)

        with pytest.raises(
            module.Stage6PlanningChildSourceRepairContinuationError,
            match="drifted|changed|current|publish capability",
        ):
            module.publish_planning_child_source_repair_continuation_artifact(
                capability,
                stage_root=fixture.stage,
                output_path=output,
                run_lease=run_lease,
            )

    assert not output.exists()


@pytest.mark.parametrize(
    "drifted_currentness",
    (
        "effective_config",
        "execution_source_member",
        "stage5_gate",
    ),
)
def test_r4_publication_rechecks_preview_currentness_before_exclusive_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drifted_currentness: str,
) -> None:
    (
        fixture,
        _tail,
        identity,
        review,
        immutable,
        authorization_path,
        _artifact,
    ) = _continuation_preview(tmp_path, monkeypatch)
    module = _continuation_module()
    output = (
        fixture.stage
        / module.PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_NAME
    )

    with RunLease(fixture.stage.parent / ".stage6.lease") as run_lease:
        capability = _f4_preview_publish_capability(
            fixture=fixture,
            identity=identity,
            review=review,
            immutable=immutable,
            authorization_path=authorization_path,
            run_lease=run_lease,
            monkeypatch=monkeypatch,
        )
        drift_path = {
            "effective_config": fixture.stage / "config.json",
            "execution_source_member": (
                authorization_path.parent / "execution-source-member.py"
            ),
            "stage5_gate": authorization_path.parent / "stage5-gate.json",
        }[drifted_currentness]
        _same_size_drift(drift_path)

        with pytest.raises(
            module.Stage6PlanningChildSourceRepairContinuationError,
            match="current|changed|drift",
        ):
            module.publish_planning_child_source_repair_continuation_artifact(
                capability,
                stage_root=fixture.stage,
                output_path=output,
                run_lease=run_lease,
            )

    assert not output.exists()


def test_continuation_publication_rejects_missing_or_forged_run_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        fixture,
            _tail,
            identity,
            review,
            immutable,
            authorization_path,
            artifact,
        ) = _continuation_preview(tmp_path, monkeypatch)
    module = _continuation_module()
    output = (
        fixture.stage
        / module.PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_NAME
    )

    with RunLease(fixture.stage.parent / ".stage6.lease") as run_lease:
        authorization, stage5 = _r5_exact_publish_owners(
            fixture=fixture,
            review=review,
            authorization_path=authorization_path,
        )
        from lunar_exploration_ppo.workflows.stage6 import (
            FrozenStage5AuthorityHandle,
        )
        from lunar_exploration_ppo.workflows.stage6_review_authorization import (
            Stage6ReviewAuthorizationHandle,
        )
        monkeypatch.setattr(
            Stage6ReviewAuthorizationHandle,
            "require_current",
            lambda self, _label: None,
        )
        monkeypatch.setattr(
            FrozenStage5AuthorityHandle,
            "require_current",
            lambda self, _label="Stage 5 authority": None,
        )
        capability = (
            module.issue_planning_child_source_repair_continuation_publish_capability(
                artifact,
                stage_root=fixture.stage,
                anchor=_recovery_anchor(fixture),
                current_execution_identity=identity,
                current_verified_review_authorization=review,
                current_immutable_bindings=immutable,
                authorization_currentness=authorization,
                stage5_authority_currentness=stage5,
                effective_config_bytes=(
                    fixture.stage / "config.json"
                ).read_bytes(),
                run_lease=run_lease,
            )
        )
        with pytest.raises(
            module.Stage6PlanningChildSourceRepairContinuationError,
            match="RunLease",
        ):
            module.publish_planning_child_source_repair_continuation_artifact(
                capability,
                stage_root=fixture.stage,
                output_path=output,
                run_lease=object(),
            )

    assert not output.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("parent_sha", "parent"),
        ("current_identity", "current"),
        ("authorization", "authorization"),
        ("review", "spec"),
        ("v1", "schema"),
    ),
)
def test_continuation_rejects_rehashed_chain_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    (
        fixture,
        _tail,
        identity,
        review,
        immutable,
        _authorization_path,
        artifact,
    ) = _continuation_preview(tmp_path, monkeypatch)
    value = json.loads(json.dumps(artifact))
    if mutation == "parent_sha":
        value["parent"]["sha256"] = "f" * 64
    elif mutation == "current_identity":
        value["current"]["execution_identity_sha256"] = "e" * 64
    elif mutation == "authorization":
        value["current"]["authorization"]["sha256"] = "d" * 64
    elif mutation == "review":
        value["review_evidence"]["spec_rereview"]["sha256"] = "c" * 64
    else:
        value["schema_version"] = (
            "stage6_planning_child_source_repair_continuation/v1"
        )
    value.pop("canonical_sha256")
    value["canonical_sha256"] = hashlib.sha256(
        fixture.module.ArtifactStore.canonical_json_bytes(value)
    ).hexdigest()

    with pytest.raises(
        _continuation_module().Stage6PlanningChildSourceRepairContinuationError,
        match=message,
    ):
        _continuation_module().validate_planning_child_source_repair_continuation_artifact(
            value,
            stage_root=fixture.stage,
            current_execution_identity=identity,
            current_verified_review_authorization=review,
            current_immutable_bindings=immutable,
        )


def test_continuation_input_pins_parent_successor_and_source_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        fixture,
        _tail,
        identity,
        review,
        immutable,
        _authorization_path,
        artifact,
    ) = _continuation_preview(tmp_path, monkeypatch)
    module = _continuation_module()
    output = (
        fixture.stage
        / module.PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_NAME
    )
    _publish_continuation(
        artifact,
        fixture=fixture,
        identity=identity,
        review=review,
        immutable=immutable,
        output_path=output,
    )
    requests = module.planning_child_source_repair_chain_input_pin_requests(
        fixture.child_path,
        output,
    )
    paths = tuple(path for _label, path in requests)
    assert fixture.child_path in paths
    assert output in paths
    assert len(paths) == len(set(paths))

    stage6 = fixture.module
    assert (
        "src/lunar_exploration_ppo/workflows/"
        "stage6_planning_child_source_repair_continuation.py"
        in stage6.STAGE6_PRODUCTION_SOURCE_PATHS
    )
    assert (
        "scripts/create_ppo_stage6_planning_child_source_repair_continuation.py"
        in stage6.STAGE6_PRODUCTION_SOURCE_PATHS
    )
    assert (
        "tests/ppo_highres_frontier/"
        "test_stage6_planning_child_source_repair_continuation.py"
        in stage6.STAGE6_TEST_SOURCE_PATHS
    )


def test_stage6_workflow_loader_uses_composite_chain_without_loader_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        fixture,
        _tail,
        identity,
        review,
        immutable,
        _authorization_path,
        artifact,
    ) = _continuation_preview(tmp_path, monkeypatch)
    continuation = (
        fixture.stage
        / _continuation_module().PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_NAME
    )
    _publish_continuation(
        artifact,
        fixture=fixture,
        identity=identity,
        review=review,
        immutable=immutable,
        output_path=continuation,
    )
    handle = SimpleNamespace(
        canonical_record=lambda: copy.deepcopy(review)
    )

    capability = (
        fixture.module._load_planning_child_recovery_capability_for_workflow(
            path=fixture.child_path,
            continuation_path=continuation,
            run_id=fixture.stage.parent.name,
            execution_identity=identity,
            review_authorization_handle=handle,
            stage5_authority=fixture.authority,
        )
    )

    assert capability.parent_artifact_sha256 == hashlib.sha256(
        fixture.child_path.read_bytes()
    ).hexdigest()
    assert capability.continuation_artifact_sha256 == hashlib.sha256(
        continuation.read_bytes()
    ).hexdigest()
    assert capability.resume_cursor.next_update == 86
    validated = fixture.module._validate_planning_child_machine_context(
        capability=capability,
        lineage_audit=json.loads(
            (fixture.stage / "lineage_audit.json").read_text(
                encoding="utf-8"
            )
        ),
        current_execution_identity=identity,
        current_verified_review_authorization=review,
        current_immutable_bindings=immutable,
    )
    assert len(validated["journal_binding_epochs"]) == 3
    assert validated["current_immutable_bindings"] == immutable


@pytest.mark.parametrize(
    ("include_attempt1_pre", "expected_attempt"),
    (
        (False, 1),
        (True, 2),
    ),
)
def test_continuation_derives_partial_u86_production_resume_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    include_attempt1_pre: bool,
    expected_attempt: int,
) -> None:
    from contextlib import nullcontext

    from lunar_exploration_ppo.ppo import standard_training

    (
        fixture,
        _tail,
        identity,
        review,
        immutable,
        _authorization_path,
        artifact,
    ) = _continuation_preview(tmp_path, monkeypatch)
    _publish_successor_chain(
        fixture,
        identity=identity,
        review=review,
        immutable=immutable,
        artifact=artifact,
    )
    partial_segment = _append_task5_resource_segment(
        fixture,
        expected_index=5,
    )
    if include_attempt1_pre:
        _append_task5_resource_pre(
            fixture,
            segment=partial_segment,
            transaction=fixture.transactions[11],
            attempt=1,
        )
    capability = (
        fixture.module._load_planning_child_recovery_capability_for_workflow(
            path=fixture.child_path,
            continuation_path=fixture.continuation_path,
            run_id=fixture.stage.parent.name,
            execution_identity=identity,
            review_authorization_handle=SimpleNamespace(
                canonical_record=lambda: copy.deepcopy(review)
            ),
            stage5_authority=fixture.authority,
        )
    )
    partial_payload = (
        fixture.stage / "resource_audit.jsonl"
    ).read_bytes()

    startup_segment = _append_task5_resource_segment(
        fixture,
        expected_index=6,
    )
    backend = object.__new__(
        standard_training.StandardProductionBackend
    )
    backend._planning_child_recovery_capability = capability
    backend._planning_child_resume_boundary_consumed = False
    backend._resource_segment_start = startup_segment
    backend.stage_root = fixture.stage
    backend._execution_operation = lambda label: nullcontext(label)
    backend._require_planning_child_resume_boundary(
        fixture.transactions[11:]
    )

    assert capability.resume_cursor.next_update == 86
    assert capability.resume_cursor.next_attempt == expected_attempt
    assert capability.resume_cursor.next_resource_segment_index == 6
    assert capability.acceptance_binding["journal_prefixes"][
        "resource_audit.jsonl"
    ] == {
        "path": "resource_audit.jsonl",
        "size_bytes": len(partial_payload),
        "sha256": hashlib.sha256(partial_payload).hexdigest(),
        "line_count": partial_payload.count(b"\n"),
    }


def test_manifest_profile_requires_parent_and_keeps_classic_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _build_task5_planning_child_machine_fixture(
        tmp_path,
        monkeypatch,
    )
    stage6 = fixture.module
    child = stage6.STAGE6_PLANNING_CHILD_SOURCE_REPAIR_MANIFEST_ARTIFACT
    continuation = (
        stage6.STAGE6_PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_MANIFEST_ARTIFACT
    )
    with pytest.raises(
        stage6.Stage6WorkflowError,
        match="no parent",
    ):
        stage6._validate_stage6_manifest_repair_profile(
            fixture.stage,
            top_files={continuation},
        )
    stage6._validate_stage6_manifest_repair_profile(
        fixture.stage,
        top_files={child, continuation},
    )
    with pytest.raises(
        stage6.Stage6WorkflowError,
        match="mutually exclusive",
    ):
        stage6._validate_stage6_manifest_repair_profile(
            fixture.stage,
            top_files={
                child,
                continuation,
                "source-repair-amendment.json",
            },
        )


def test_stage6_production_workflow_exposes_write_once_continuation_input() -> None:
    stage6 = importlib.import_module(
        "lunar_exploration_ppo.workflows.stage6"
    )

    assert (
        "planning_child_source_repair_continuation_path"
        in inspect.signature(stage6.run_stage6_workflow).parameters
    )
    assert (
        "planning_child_source_repair_continuation_path"
        in inspect.signature(
            stage6._run_stage6_workflow_for_test
        ).parameters
    )
    assert (
        _continuation_module().PLANNING_CHILD_SOURCE_REPAIR_CONTINUATION_NAME
        == "planning-child-source-repair-continuation.json"
    )
